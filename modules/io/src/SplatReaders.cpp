// Copyright (c) 2026 jesus luque.
#include <algorithm>
#include <cstring>
#include <optional>
#include <string>


#include "athenea/core/Log.h"
#include "athenea/core/Platform.h"
#include "athenea/io/PlyHeader.h"
#include "athenea/io/Readers.h"

#ifdef ATHENEA_HAVE_SPZ
#include "load-spz.h"
#include "splat-extensions.h"
#endif

namespace athenea::io {
namespace {

constexpr size_t kHeaderProbe = 1 << 16;

std::string lowerExtension(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

}   // namespace

Result<RawSplats> readSplats(const std::filesystem::path& path) {
    const std::string ext = lowerExtension(path);
    if (ext == ".splat") {
        return readDotSplat(path);
    }
    if (ext == ".ply") {
        return readSplatPly(path);
    }
    if (ext == ".spz") {
        return readSpz(path);
    }
    if (ext == ".sog" || path.filename() == "meta.json") {
        return Error::make(ErrorCode::Unsupported,
                           "'{}': SOG is decoded on the device -- scene::loadSplatFile reads it", path.string());
    }
    return Error::make(ErrorCode::Unsupported, "'{}': no splat reader for '{}'", path.string(), ext);
}

Result<RawSplats> readSplatPly(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) {
        return std::move(file).error();
    }
    const auto bytes = file->bytes();
    const std::string_view probe(reinterpret_cast<const char*>(bytes.data()),
                                 std::min(bytes.size(), kHeaderProbe));
    auto header = parsePlyHeader(probe);
    if (!header) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': {}", path.string(),
                           header.error().message());
    }
    if (!header->binaryLittleEndian) {
        return Error::make(ErrorCode::Unsupported,
                           "'{}': PLY format '{}'; splat clouds are binary_little_endian",
                           path.string(), header->format);
    }
    const PlyElement* vertex = header->element("vertex");
    if (vertex == nullptr || vertex->count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no vertices", path.string());
    }
    if (vertex->hasList) {
        return Error::make(ErrorCode::Unsupported, "'{}': a list in the vertex element",
                           path.string());
    }
    if (vertex->count > 0xFFFFFFFFull) {
        return Error::make(ErrorCode::Unsupported, "'{}': more than 2^32 splats", path.string());
    }
    auto start = header->binaryOffsetOf("vertex");
    if (!start) {
        return std::move(start).error();
    }
    if (*start + vertex->stride * vertex->count > bytes.size()) {
        return Error::make(ErrorCode::IoFailure, "'{}': the file ends before its vertices do",
                           path.string());
    }

    static constexpr const char* kRequired[] = {"x",       "y",       "z",       "opacity",
                                                "scale_0", "scale_1", "scale_2", "rot_0",
                                                "rot_1",   "rot_2",   "rot_3",   "f_dc_0",
                                                "f_dc_1",  "f_dc_2"};
    for (const char* name : kRequired) {
        if (vertex->find(name) == nullptr) {
            return Error::make(ErrorCode::InvalidArgument,
                               "'{}': no '{}' -- this does not look like a trained splat cloud",
                               path.string(), name);
        }
    }

    // Every property becomes one float in the record, in the file's order --
    // widened on the device from the file's own bytes, whatever its types.
    const uint32_t floats = static_cast<uint32_t>(vertex->properties.size());
    const auto indexOf = [&](const char* name) -> uint32_t {
        const PlyProperty* p = vertex->find(name);
        return static_cast<uint32_t>(p - vertex->properties.data());
    };
    const auto typeOf = [](const std::string& t) -> std::optional<PackedFields::Type> {
        using T = PackedFields::Type;
        if (t == "float" || t == "float32") return T::F32;
        if (t == "double" || t == "float64") return T::F64;
        if (t == "uchar" || t == "uint8") return T::U8;
        if (t == "char" || t == "int8") return T::I8;
        if (t == "ushort" || t == "uint16") return T::U16;
        if (t == "short" || t == "int16") return T::I16;
        if (t == "uint" || t == "uint32") return T::U32;
        if (t == "int" || t == "int32") return T::I32;
        return std::nullopt;
    };

    RawSplats raw;
    raw.source = path.string();
    raw.count = static_cast<uint32_t>(vertex->count);
    SplatEncoding& enc = raw.encoding;
    enc.floatsPerRecord = floats;
    enc.x = indexOf("x");
    enc.y = indexOf("y");
    enc.z = indexOf("z");
    enc.opacity = indexOf("opacity");
    enc.scale0 = indexOf("scale_0");
    enc.scale1 = indexOf("scale_1");
    enc.scale2 = indexOf("scale_2");
    enc.rotW = indexOf("rot_0");
    enc.rotX = indexOf("rot_1");
    enc.rotY = indexOf("rot_2");
    enc.rotZ = indexOf("rot_3");
    enc.dc0 = indexOf("f_dc_0");
    enc.dc1 = indexOf("f_dc_1");
    enc.dc2 = indexOf("f_dc_2");

    // The harmonics: counted, contiguous, colour-outermost. Degree is the
    // largest whole one the count covers.
    uint32_t restCount = 0;
    for (const PlyProperty& p : vertex->properties) {
        if (p.name.rfind("f_rest_", 0) == 0) {
            ++restCount;
        }
    }
    uint32_t perColour = 0;
    for (uint32_t basis : {15u, 8u, 3u}) {
        if (restCount >= basis * 3) {
            perColour = basis;
            break;
        }
    }
    if (perColour > 0) {
        enc.restBase = indexOf("f_rest_0");
        for (uint32_t k = 0; k < perColour * 3; ++k) {
            // The file's own per-colour count decides where green starts.
            const uint32_t filePerColour = restCount / 3;
            const uint32_t colour = k / perColour;
            const uint32_t basis = k % perColour;
            const std::string name = "f_rest_" + std::to_string(colour * filePerColour + basis);
            const PlyProperty* p = vertex->find(name);
            if (p == nullptr || indexOf(name.c_str()) != enc.restBase + colour * filePerColour + basis) {
                return Error::make(ErrorCode::Unsupported,
                                   "'{}': harmonics are not contiguous in the vertex element",
                                   path.string());
            }
        }
        // restPerColour is the kept count; the kernel indexes green as
        // restBase + filePerColour + basis, so it needs the file's count too.
        enc.restPerColour = perColour;
        if (restCount / 3 != perColour) {
            return Error::make(ErrorCode::Unsupported,
                               "'{}': {} harmonic coefficients is not a whole degree",
                               path.string(), restCount);
        }
    }
    enc.restColourOuter = 1;
    enc.opacity_ = SplatEncoding::Opacity::Logit;
    enc.scale_ = SplatEncoding::Scale::Log;
    enc.colour = SplatEncoding::Colour::ShDc;
    enc.rotation = SplatEncoding::Rotation::Float;

    // The vertex block as the file has it: bytes copied, nothing read out
    // of them here.
    raw.encoding.record = SplatEncoding::Record::PackedRecords;
    raw.fields.stride = static_cast<uint32_t>(vertex->stride);
    for (const PlyProperty& p : vertex->properties) {
        const auto type = typeOf(p.type);
        if (!type) {
            return Error::make(ErrorCode::Unsupported, "'{}': property '{}' of type '{}'", path.string(), p.name,
                               p.type);
        }
        raw.fields.offsets.push_back(static_cast<uint32_t>(p.offset));
        raw.fields.types.push_back(*type);
    }
    const auto* base = reinterpret_cast<const uint8_t*>(bytes.data()) + *start;
    raw.packed.assign(base, base + vertex->stride * vertex->count);
    return raw;
}

Result<RawSplats> readDotSplat(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) {
        return std::move(file).error();
    }
    const auto bytes = file->bytes();
    constexpr size_t kRecord = 32;
    if (bytes.size() % kRecord != 0 || bytes.empty()) {
        return Error::make(ErrorCode::InvalidArgument,
                           "'{}': {} bytes is not a whole number of 32-byte .splat records",
                           path.string(), bytes.size());
    }
    // pos f32*3 | scale f32*3 | rgba u8*4 | rot u8*4 (w x y z, (v-128)/128)
    RawSplats raw;
    raw.source = path.string();
    raw.count = static_cast<uint32_t>(bytes.size() / kRecord);
    SplatEncoding& enc = raw.encoding;
    enc.floatsPerRecord = 14;
    enc.x = 0; enc.y = 1; enc.z = 2;
    enc.scale0 = 3; enc.scale1 = 4; enc.scale2 = 5;
    enc.dc0 = 6; enc.dc1 = 7; enc.dc2 = 8; enc.opacity = 9;
    enc.rotW = 10; enc.rotX = 11; enc.rotY = 12; enc.rotZ = 13;
    enc.opacity_ = SplatEncoding::Opacity::Byte;
    enc.scale_ = SplatEncoding::Scale::Linear;
    enc.colour = SplatEncoding::Colour::Byte;
    enc.rotation = SplatEncoding::Rotation::Byte;

    // The records as the file has them, widened on the device: six floats,
    // then eight bytes as values.
    enc.record = SplatEncoding::Record::PackedRecords;
    raw.fields.stride = kRecord;
    for (uint32_t k = 0; k < 14; ++k) {
        raw.fields.offsets.push_back(k < 6 ? 4 * k : 24 + (k - 6));
        raw.fields.types.push_back(k < 6 ? PackedFields::Type::F32 : PackedFields::Type::U8);
    }
    const auto* data = reinterpret_cast<const uint8_t*>(bytes.data());
    raw.packed.assign(data, data + bytes.size());
    return raw;
}

SplatEncoding spzEncoding(uint32_t shDim, uint32_t fractionalBits, bool smallestThree, bool flipYZ) {
    SplatEncoding enc;
    // Up to degree 3 kept: the engine evaluates no further, and a degree-4
    // file's fourth band is 27 floats a splat for nothing.
    enc.record = SplatEncoding::Record::SpzPacked;
    enc.floatsPerRecord = 14 + std::min(shDim, uint32_t{15}) * 3;
    enc.x = 0; enc.y = 1; enc.z = 2;
    enc.opacity = 3;
    enc.scale0 = 4; enc.scale1 = 5; enc.scale2 = 6;
    enc.dc0 = 7; enc.dc1 = 8; enc.dc2 = 9;
    enc.rotX = 10; enc.rotY = 11; enc.rotZ = 12; enc.rotW = 13;
    enc.restBase = 14;
    enc.restPerColour = std::min(shDim, uint32_t{15});
    enc.restColourOuter = 0;   // rgb per basis
    enc.opacity_ = SplatEncoding::Opacity::Byte;
    enc.scale_ = SplatEncoding::Scale::SpzByte;
    enc.colour = SplatEncoding::Colour::SpzByte;
    enc.rotation = smallestThree ? SplatEncoding::Rotation::SmallestThree : SplatEncoding::Rotation::FirstThree;
    enc.rest = SplatEncoding::Rest::Byte;
    enc.positionScale = 1.0F / static_cast<float>(1u << fractionalBits);
    enc.flipYZ = flipYZ;
    return enc;
}

Result<RawSplats> readSpz(const std::filesystem::path& path) {
    auto file = platform::MappedFile::open(path);
    if (!file) return std::move(file).error();
    const std::span<const std::byte> bytes = file->bytes();
    return readSpz(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()),
                   path.string());
}

Result<RawSplats> readSpz(std::span<const uint8_t> bytes, const std::string& source) {
#ifndef ATHENEA_HAVE_SPZ
    (void)bytes;
    return Error::make(ErrorCode::Unsupported, "'{}': this build reads no .spz (zstd was not found)", source);
#else
    spz::PackedGaussians packed = spz::loadSpzPacked(bytes.data(), bytes.size());
    if (packed.numPoints <= 0) {
        return Error::make(ErrorCode::IoFailure, "'{}': not a readable SPZ file", source);
    }
    if (packed.usesFloat16()) {
        return Error::make(ErrorCode::Unsupported, "'{}': SPZ float16 positions (a format never released)",
                           source);
    }
    // The axes the data is stored in: the coordinate-system extension's, and
    // the format's right-up-back without one. The engine reads right-down-
    // front (PLY's); the decode turns right-up-back by changing the sign of
    // y and z, and nothing else is turned.
    const spz::CoordinateSystem stored = spz::getPackedCoordinateSystem(packed.extensions);
    if (stored != spz::CoordinateSystem::RUB && stored != spz::CoordinateSystem::RDF) {
        return Error::make(ErrorCode::Unsupported,
                           "'{}': SPZ data stored in coordinate system {}; this reader reads right-up-back (4) and "
                           "right-down-front (6)",
                           source, static_cast<uint32_t>(stored));
    }
    const auto n = static_cast<size_t>(packed.numPoints);
    const uint32_t shDim = packed.shDegree >= 4 ? 24 : packed.shDegree == 3 ? 15 : packed.shDegree == 2 ? 8
                                                   : packed.shDegree == 1 ? 3 : 0;
    const bool smallestThree = packed.usesQuaternionSmallestThree;
    if (packed.positions.size() < n * 9 || packed.alphas.size() < n || packed.colors.size() < n * 3 ||
        packed.scales.size() < n * 3 || packed.rotations.size() < n * (smallestThree ? 4 : 3) ||
        packed.sh.size() < n * shDim * 3) {
        return Error::make(ErrorCode::IoFailure, "'{}': SPZ attribute streams shorter than {} points", source, n);
    }

    RawSplats raw;
    raw.source = source;
    raw.count = static_cast<uint32_t>(n);
    raw.antialiased = packed.antialiased;
    raw.encoding = spzEncoding(shDim, static_cast<uint32_t>(packed.fractionalBits), smallestThree,
                               stored == spz::CoordinateSystem::RUB);

    // The six streams one after another, as they came out of the
    // decompressor: bytes copied, nothing read out of them here.
    SpzStreams& at = raw.spz;
    at.shDim = shDim;
    at.fractionalBits = static_cast<uint32_t>(packed.fractionalBits);
    at.smallestThree = smallestThree;
    const std::vector<uint8_t>* streams[6] = {&packed.positions, &packed.alphas, &packed.colors,
                                              &packed.scales, &packed.rotations, &packed.sh};
    uint64_t* offsets[6] = {&at.positions, &at.alphas, &at.colours, &at.scales, &at.rotations, &at.sh};
    uint64_t total = 0;
    for (const auto* stream : streams) {
        total += (stream->size() + 3) / 4 * 4;
    }
    raw.packed.reserve(total);
    for (size_t k = 0; k < 6; ++k) {
        *offsets[k] = raw.packed.size();
        raw.packed.insert(raw.packed.end(), streams[k]->begin(), streams[k]->end());
        raw.packed.resize((raw.packed.size() + 3) / 4 * 4, 0);   // each stream on a word
    }
    return raw;
#endif
}

}   // namespace athenea::io
