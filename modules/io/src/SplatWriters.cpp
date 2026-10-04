// Copyright (c) 2026 jesus luque.
//
// SplatWriters.h says what these are: headers, containers and compression
// around numbers the device worked out.
#include "athenea/io/SplatWriters.h"

#include <cstring>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>
#include <zlib.h>
#ifdef ATHENEA_HAVE_ZSTD
#include <zstd.h>
#endif

namespace athenea::io {
namespace {

Result<std::ofstream> openForWriting(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "'{}': cannot be written", path.string());
    }
    return out;
}

Result<void> closed(std::ofstream& out, const std::filesystem::path& path) {
    out.close();
    if (!out) {
        return Error::make(ErrorCode::IoFailure, "'{}': writing failed", path.string());
    }
    return ok();
}

template <typename T>
void put(std::vector<uint8_t>& into, T value) {
    const size_t at = into.size();
    into.resize(at + sizeof(T));
    std::memcpy(into.data() + at, &value, sizeof(T));
}

/// gzip around `data`, as SPZ versions 1 to 3 are.
Result<std::vector<uint8_t>> gzipped(const std::vector<uint8_t>& data) {
    z_stream stream{};
    // 16 + MAX_WBITS: a gzip header and trailer, not zlib's.
    if (deflateInit2(&stream, Z_BEST_COMPRESSION, Z_DEFLATED, 16 + MAX_WBITS, 9, Z_DEFAULT_STRATEGY) != Z_OK) {
        return Error(ErrorCode::IoFailure, "SPZ: zlib would not start");
    }
    std::vector<uint8_t> out(deflateBound(&stream, static_cast<uLong>(data.size())) + 64);
    stream.next_in = const_cast<Bytef*>(data.data());
    stream.avail_in = static_cast<uInt>(data.size());
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(out.size());
    const int result = deflate(&stream, Z_FINISH);
    const size_t written = out.size() - stream.avail_out;
    deflateEnd(&stream);
    if (result != Z_STREAM_END) {
        return Error(ErrorCode::IoFailure, "SPZ: gzip failed");
    }
    out.resize(written);
    return out;
}

constexpr uint32_t kNgspMagic = 0x5053474e;
constexpr uint8_t  kFlagAntialiased = 0x1;

}   // namespace

Result<void> writePly3dgs(const std::filesystem::path& path, std::span<const std::vector<float>> chunks,
                          std::span<const std::string> comments) {
    size_t floats = 0;
    for (const std::vector<float>& chunk : chunks) {
        if (chunk.size() % 62 != 0) {
            return Error(ErrorCode::InvalidArgument, "PLY: a chunk is not whole records of 62 floats");
        }
        floats += chunk.size();
    }
    auto out = openForWriting(path);
    if (!out) return std::move(out).error();
    std::string header = "ply\nformat binary_little_endian 1.0\n";
    for (const std::string& comment : comments) {
        header += "comment " + comment + "\n";
    }
    header += "element vertex " + std::to_string(floats / 62) + "\n";
    for (const char* name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"}) {
        header += std::string("property float ") + name + "\n";
    }
    for (int k = 0; k < 45; ++k) {
        header += "property float f_rest_" + std::to_string(k) + "\n";
    }
    header += "property float opacity\n";
    for (const char* name : {"scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}) {
        header += std::string("property float ") + name + "\n";
    }
    header += "end_header\n";
    out->write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const std::vector<float>& chunk : chunks) {
        out->write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size() * 4));
    }
    return closed(*out, path);
}

bool writesSpzVersion4() noexcept {
#ifdef ATHENEA_HAVE_ZSTD
    return true;
#else
    return false;
#endif
}

Result<void> writeSpz(const std::filesystem::path& path, const SpzHeader& header,
                      const std::array<std::vector<uint8_t>, 6>& streams) {
    if (header.version < 3 || header.version > 4) {
        return Error(ErrorCode::Unsupported, "SPZ: this writer writes versions 3 and 4");
    }
    if (header.shDegree > 3) {
        return Error(ErrorCode::Unsupported, "SPZ: degree 3 harmonics at most");
    }
    static constexpr std::array<uint32_t, 4> kShDim{0, 3, 8, 15};
    const std::array<uint64_t, 6> perGaussian{9, 1, 3, 3, 4, uint64_t{kShDim[header.shDegree]} * 3};
    for (size_t s = 0; s < 6; ++s) {
        if (streams[s].size() != uint64_t{header.count} * perGaussian[s]) {
            return Error::make(ErrorCode::InvalidArgument, "SPZ: stream {} is {} bytes, not {}", s,
                               streams[s].size(), uint64_t{header.count} * perGaussian[s]);
        }
    }
    const uint8_t flags = header.antialiased ? kFlagAntialiased : 0;
    std::vector<uint8_t> file;
    if (header.version == 3) {
        // The legacy layout: a 16-byte header and the six streams, gzipped
        // whole.
        std::vector<uint8_t> body;
        put(body, kNgspMagic);
        put(body, header.version);
        put(body, header.count);
        put(body, static_cast<uint8_t>(header.shDegree));
        put(body, static_cast<uint8_t>(header.fractionalBits));
        put(body, flags);
        put(body, uint8_t{0});
        for (const std::vector<uint8_t>& stream : streams) {
            body.insert(body.end(), stream.begin(), stream.end());
        }
        auto zipped = gzipped(body);
        if (!zipped) return std::move(zipped).error();
        file = std::move(*zipped);
    } else {
#ifdef ATHENEA_HAVE_ZSTD
        // NGSP: a 32-byte header, a table of (compressed, uncompressed)
        // sizes, then each non-empty stream compressed on its own.
        std::vector<std::vector<uint8_t>> chunks;
        std::vector<uint64_t> sizes;
        for (const std::vector<uint8_t>& stream : streams) {
            if (stream.empty()) {
                continue;
            }
            std::vector<uint8_t> chunk(ZSTD_compressBound(stream.size()));
            ZSTD_CCtx* context = ZSTD_createCCtx();
            if (context == nullptr) {
                return Error(ErrorCode::OutOfMemory, "SPZ: no ZSTD context");
            }
            ZSTD_CCtx_setParameter(context, ZSTD_c_compressionLevel, 12);
            const size_t made = ZSTD_compress2(context, chunk.data(), chunk.size(), stream.data(), stream.size());
            ZSTD_freeCCtx(context);
            if (ZSTD_isError(made) != 0U) {
                return Error(ErrorCode::IoFailure, "SPZ: ZSTD failed");
            }
            chunk.resize(made);
            chunks.push_back(std::move(chunk));
            sizes.push_back(stream.size());
        }
        put(file, kNgspMagic);
        put(file, header.version);
        put(file, header.count);
        put(file, static_cast<uint8_t>(header.shDegree));
        put(file, static_cast<uint8_t>(header.fractionalBits));
        put(file, flags);
        put(file, static_cast<uint8_t>(chunks.size()));
        put(file, uint32_t{32});   // the table follows the header: no extensions
        file.resize(32, 0);
        for (size_t k = 0; k < chunks.size(); ++k) {
            put(file, static_cast<uint64_t>(chunks[k].size()));
            put(file, sizes[k]);
        }
        for (const std::vector<uint8_t>& chunk : chunks) {
            file.insert(file.end(), chunk.begin(), chunk.end());
        }
#else
        return Error(ErrorCode::Unsupported, "SPZ version 4: this build has no ZSTD; write version 3");
#endif
    }
    auto out = openForWriting(path);
    if (!out) return std::move(out).error();
    out->write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    return closed(*out, path);
}

Result<void> writeGlbGaussians(const std::filesystem::path& path, const GlbGaussians& g) {
    if (g.counts.size() != g.chunks.size() || g.degree > 3) {
        return Error(ErrorCode::InvalidArgument, "glb: one count a chunk, degree 3 at most");
    }
    const uint32_t coefficients = (g.degree + 1) * (g.degree + 1);
    // Blocks of a chunk, in the order they are laid out: width each.
    std::vector<uint32_t> widths{3, 4, 3, 1};
    for (uint32_t k = 0; k < 16; ++k) {
        widths.push_back(3);
    }
    uint64_t total = 0;
    for (size_t c = 0; c < g.chunks.size(); ++c) {
        if (g.chunks[c].size() != uint64_t{g.counts[c]} * 59) {
            return Error(ErrorCode::InvalidArgument, "glb: a chunk is not 59 floats a gaussian");
        }
        total += g.counts[c];
    }
    if (total > 0xFFFFFFFFull) {
        return Error(ErrorCode::Unsupported, "glb: more than 2^32 gaussians");
    }
    // Attributes written: position, rotation, scale, opacity, then the
    // harmonics up to the degree.
    const size_t attributes = 4 + coefficients;
    nlohmann::json bufferViews = nlohmann::json::array();
    nlohmann::json accessors = nlohmann::json::array();
    nlohmann::json attributesJson = nlohmann::json::object();
    static const char* kTypes[5] = {"", "SCALAR", "VEC2", "VEC3", "VEC4"};
    uint64_t offset = 0;
    for (size_t a = 0; a < attributes; ++a) {
        const uint64_t bytes = total * widths[a] * 4;
        bufferViews.push_back({{"buffer", 0}, {"byteOffset", offset}, {"byteLength", bytes}});
        nlohmann::json accessor = {{"bufferView", a},
                                   {"componentType", 5126},
                                   {"count", total},
                                   {"type", kTypes[widths[a]]}};
        if (a == 0) {
            accessor["min"] = {g.min[0], g.min[1], g.min[2]};
            accessor["max"] = {g.max[0], g.max[1], g.max[2]};
        }
        accessors.push_back(accessor);
        std::string name;
        if (a == 0) {
            name = "POSITION";
        } else if (a == 1) {
            name = "KHR_gaussian_splatting:ROTATION";
        } else if (a == 2) {
            name = "KHR_gaussian_splatting:SCALE";
        } else if (a == 3) {
            name = "KHR_gaussian_splatting:OPACITY";
        } else {
            const uint32_t k = static_cast<uint32_t>(a - 4);
            const uint32_t l = k == 0 ? 0 : k < 4 ? 1 : k < 9 ? 2 : 3;
            name = "KHR_gaussian_splatting:SH_DEGREE_" + std::to_string(l) + "_COEF_" + std::to_string(k - l * l);
        }
        attributesJson[name] = a;
        offset += bytes;
    }
    nlohmann::json primitive = {
        {"mode", 0},
        {"attributes", attributesJson},
        {"extensions", {{"KHR_gaussian_splatting", {{"kernel", "ellipse"}, {"colorSpace", g.colorSpace}}}}}};
    nlohmann::json doc = {{"asset", {{"version", "2.0"}, {"generator", g.generator}}},
                          {"extensionsUsed", {"KHR_gaussian_splatting"}},
                          {"scene", 0},
                          {"scenes", {{{"nodes", {0}}}}},
                          {"nodes", {{{"mesh", 0}}}},
                          {"meshes", {{{"primitives", {primitive}}}}},
                          {"buffers", {{{"byteLength", offset}}}},
                          {"bufferViews", bufferViews},
                          {"accessors", accessors}};
    std::string json = doc.dump();
    while (json.size() % 4 != 0) {
        json += ' ';
    }
    const uint64_t length = 12 + 8 + json.size() + 8 + offset;
    if (length > 0xFFFFFFFFull) {
        return Error(ErrorCode::Unsupported, "glb: a file past 4 GB");
    }
    auto out = openForWriting(path);
    if (!out) return std::move(out).error();
    std::vector<uint8_t> head;
    put(head, uint32_t{0x46546C67});   // glTF
    put(head, uint32_t{2});
    put(head, static_cast<uint32_t>(length));
    put(head, static_cast<uint32_t>(json.size()));
    put(head, uint32_t{0x4E4F534A});   // JSON
    out->write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
    out->write(json.data(), static_cast<std::streamsize>(json.size()));
    head.clear();
    put(head, static_cast<uint32_t>(offset));
    put(head, uint32_t{0x004E4942});   // BIN
    out->write(reinterpret_cast<const char*>(head.data()), static_cast<std::streamsize>(head.size()));
    // Attribute by attribute, each cloud's block of it in turn.
    uint64_t blockStart = 0;
    for (size_t a = 0; a < attributes; ++a) {
        for (size_t c = 0; c < g.chunks.size(); ++c) {
            const uint64_t start = blockStart * g.counts[c];
            out->write(reinterpret_cast<const char*>(g.chunks[c].data() + start),
                       static_cast<std::streamsize>(uint64_t{g.counts[c]} * widths[a] * 4));
        }
        blockStart += widths[a];
    }
    return closed(*out, path);
}

}   // namespace athenea::io
