// Copyright (c) 2026 jesus luque.
#include "athenea/scene/GpuClouds.h"

#include <array>
#include "athenea/scene/DecodeParams.h"
#include "athenea/io/Readers.h"

#include <algorithm>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::scene {
namespace {

constexpr uint32_t kChunk = 4096;

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element,
                                 const std::string& label, const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, data);
}

}   // namespace

void setDecodeParams(rhi::ShaderCursor cursor, const io::SplatEncoding& e, uint32_t count,
                     uint32_t base, uint32_t keepPerColour, uint32_t shWords, uint32_t recordBase) {
    rhi::ShaderCursor p = cursor["params"];
    p["count"].setData(count);
    p["stride"].setData(e.floatsPerRecord);
    p["base"].setData(base);
    p["keepPerColour"].setData(keepPerColour);
    p["x"].setData(e.x);
    p["y"].setData(e.y);
    p["z"].setData(e.z);
    p["opacity"].setData(e.opacity);
    p["scale0"].setData(e.scale0);
    p["scale1"].setData(e.scale1);
    p["scale2"].setData(e.scale2);
    p["rotW"].setData(e.rotW);
    p["rotX"].setData(e.rotX);
    p["rotY"].setData(e.rotY);
    p["rotZ"].setData(e.rotZ);
    p["dc0"].setData(e.dc0);
    p["dc1"].setData(e.dc1);
    p["dc2"].setData(e.dc2);
    p["restBase"].setData(e.restBase);
    p["filePerColour"].setData(e.restPerColour);
    p["restColourOuter"].setData(e.restColourOuter);
    p["opacityMode"].setData(static_cast<uint32_t>(e.opacity_));
    p["scaleMode"].setData(static_cast<uint32_t>(e.scale_));
    p["colourMode"].setData(static_cast<uint32_t>(e.colour));
    p["rotationMode"].setData(static_cast<uint32_t>(e.rotation));
    p["restMode"].setData(static_cast<uint32_t>(e.rest));
    p["flipYZ"].setData(uint32_t{e.flipYZ ? 1u : 0u});
    p["positionScale"].setData(e.positionScale);
    p["shWords"].setData(shWords);
    p["metallic"].setData(e.metallic);
    p["roughness"].setData(e.roughness);
    p["transmission"].setData(e.transmission);
    p["cryptoObject"].setData(e.cryptoObject);
    p["transferBase"].setData(e.transferBase);
    p["transferCount"].setData(e.transferCount);
    p["transferWords"].setData((e.transferCount + 1) / 2);
    p["shadowBits"].setData(e.shadowBits);
    p["shadowWords"].setData(e.shadowWords);
    p["recordBase"].setData(recordBase);
    p["normal"].setData(e.normal);
    p["emission"].setData(e.emission);
    p["lobes"].setData(e.lobes);
}


Result<CloudLoader> CloudLoader::create(gpu::ShaderLibrary& library) {
    CloudLoader loader;
    loader.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    loader.prefix_ = std::move(*prefix);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(loader.splatValidate_, "athenea/scene/splat_validate", "splatValidate"));
    ATHENEA_TRY(make(loader.sogDecode_, "athenea/scene/sog_decode", "sogDecode"));
    ATHENEA_TRY(make(loader.splatDecode_, "athenea/scene/splat_decode", "splatDecode"));
    ATHENEA_TRY(make(loader.splatKept_, "athenea/scene/splat_kept", "splatKept"));
    ATHENEA_TRY(make(loader.splatKeptScatter_, "athenea/scene/splat_kept", "splatKeptScatter"));
    ATHENEA_TRY(make(loader.splatUnpack_, "athenea/scene/splat_unpack", "splatUnpack"));
    ATHENEA_TRY(make(loader.pointsValidate_, "athenea/scene/points_validate", "pointsValidate"));
    ATHENEA_TRY(make(loader.pointsDecode_, "athenea/scene/points_decode", "pointsDecode"));
    ATHENEA_TRY(make(loader.splatStreams_, "athenea/scene/streams", "splatStreams"));
    ATHENEA_TRY(make(loader.pointStreams_, "athenea/scene/streams", "pointStreams"));
    ATHENEA_TRY(make(loader.boundsChunks_, "athenea/scene/bounds_chunks", "boundsChunks"));
    ATHENEA_TRY(make(loader.boundsReduce_, "athenea/scene/bounds_reduce", "boundsReduce"));
    return loader;
}

Result<io::RawSplats> CloudLoader::records(const GpuSplats& splats) {
    gpu::Device& device = *device_;
    io::RawSplats raw;
    raw.source = splats.source;
    io::SplatEncoding& e = raw.encoding;
    const uint32_t keep = splats.restPerColour;
    e.floatsPerRecord = 14 + 3 * keep;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14;
    e.restPerColour = keep;
    e.restColourOuter = 0;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Log;
    e.colour = io::SplatEncoding::Colour::Linear;   // the base colour, as the shape keeps it
    raw.linear = splats.linear;                      // in the space the cloud keeps it in
    raw.count = splats.count;
    if (splats.count == 0) {
        return raw;
    }
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{splats.count} * e.floatsPerRecord * sizeof(float);
    desc.elementBytes = 4;
    desc.label = "unpack.records";
    auto out = gpu::Buffer::create(device, desc);
    if (!out) return std::move(out).error();
    gpu::CommandBatch batch(device);
    splatUnpack_.dispatch(batch, {splats.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(splats.positions.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        cursor["records"].setBinding(out->rhi());
        cursor["params"]["count"].setData(splats.count);
        cursor["params"]["keep"].setData(keep);
        cursor["params"]["shWords"].setData(splats.shWords);
        cursor["params"]["stride"].setData(e.floatsPerRecord);
    });
    ATHENEA_TRY(batch.submit(true));
    auto read = out->readAll<float>(device);
    if (!read) return std::move(read).error();
    raw.records = std::move(*read);
    return raw;
}

Result<Bounds> CloudLoader::boundsOf(const gpu::Buffer& positions, uint32_t count) {
    Bounds bounds;
    if (count == 0) {
        return bounds;
    }
    const uint32_t chunks = (count + kChunk - 1) / kChunk;
    auto extents = deviceBuffer(*device_, uint64_t{chunks} * 2, 16, "bounds.extents");
    if (!extents) return std::move(extents).error();
    auto result = deviceBuffer(*device_, 2, 16, "bounds.result");
    if (!result) return std::move(result).error();
    gpu::CommandBatch batch(*device_);
    const auto params = [&](rhi::ShaderCursor cursor) {
        cursor["params"]["count"].setData(count);
        cursor["params"]["chunkSize"].setData(kChunk);
        cursor["params"]["chunkCount"].setData(chunks);
    };
    boundsChunks_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(positions.rhi());
        cursor["extents"].setBinding(extents->rhi());
        params(cursor);
    });
    boundsReduce_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["extents"].setBinding(extents->rhi());
        cursor["result"].setBinding(result->rhi());
        params(cursor);
    });
    ATHENEA_TRY(batch.submit(true));
    float extent[8];
    ATHENEA_TRY(result->read(*device_, 0, sizeof(extent), extent));
    bounds.min = {extent[0], extent[1], extent[2]};
    bounds.max = {extent[4], extent[5], extent[6]};
    return bounds;
}

Result<GpuSplats> CloudLoader::startSplats(const std::string& source, uint32_t declared, uint32_t keep,
                                           bool withPbr, bool withCrypto, uint32_t transferCount,
                                           uint32_t shadowWords, bool withNormals, bool withEmission,
                                           bool withLobes) {
    GpuSplats splats;
    splats.source = source;
    splats.declared = declared;
    splats.restPerColour = keep;
    splats.shWords = keep == 0 ? 1 : (keep * 3 + 1) / 2;
    auto positions = deviceBuffer(*device_, declared, 16, "splats.positions");
    if (!positions) return std::move(positions).error();
    auto shape = deviceBuffer(*device_, uint64_t{declared} * 4, 4, "splats.shape");
    if (!shape) return std::move(shape).error();
    auto sh = deviceBuffer(*device_, keep == 0 ? 1 : uint64_t{declared} * splats.shWords, 4, "splats.sh");
    if (!sh) return std::move(sh).error();
    splats.positions = *positions;
    splats.shape = *shape;
    splats.sh = *sh;
    auto origin = deviceBuffer(*device_, declared, 4, "splats.origin");
    if (!origin) return std::move(origin).error();
    splats.origin = *origin;
    if (withPbr) {
        auto pbr = deviceBuffer(*device_, declared, 4, "splats.pbr");
        if (!pbr) return std::move(pbr).error();
        splats.pbr = *pbr;
    }
    if (withCrypto) {
        auto crypto = deviceBuffer(*device_, declared, 4, "splats.crypto");
        if (!crypto) return std::move(crypto).error();
        splats.crypto = *crypto;
    }
    if (transferCount > 0) {
        splats.transferCount = transferCount;
        splats.transferWords = (transferCount + 1) / 2;
        auto transfer = deviceBuffer(*device_, uint64_t{declared} * splats.transferWords, 4, "splats.transfer");
        if (!transfer) return std::move(transfer).error();
        splats.transfer = *transfer;
    }
    if (shadowWords > 0) {
        splats.shadowWords = shadowWords;
        auto bits = deviceBuffer(*device_, uint64_t{declared} * shadowWords, 4, "splats.shadowBits");
        if (!bits) return std::move(bits).error();
        splats.shadowBits = *bits;
    }
    if (withNormals) {
        auto normals = deviceBuffer(*device_, declared, 4, "splats.normals");
        if (!normals) return std::move(normals).error();
        splats.normals = *normals;
    }
    if (withEmission) {
        auto emission = deviceBuffer(*device_, declared, 4, "splats.emission");
        if (!emission) return std::move(emission).error();
        splats.emission = *emission;
    }
    if (withLobes) {
        auto lobes = deviceBuffer(*device_, uint64_t{declared} * 3, 4, "splats.lobes");
        if (!lobes) return std::move(lobes).error();
        splats.lobes = *lobes;
    }
    return splats;
}

Result<uint32_t> CloudLoader::decodeSlice(const gpu::Buffer& raw, const io::SplatEncoding& e, uint32_t n,
                                          uint32_t written, uint32_t keep, GpuSplats& splats,
                                          uint32_t recordBase) {
    auto valid = deviceBuffer(*device_, n, 4, "splats.valid");
    if (!valid) return std::move(valid).error();
    auto dest = deviceBuffer(*device_, n, 4, "splats.dest");
    if (!dest) return std::move(dest).error();
    auto total = deviceBuffer(*device_, 1, 4, "splats.total");
    if (!total) return std::move(total).error();

    gpu::CommandBatch batch(*device_);
    splatValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        setDecodeParams(cursor, e, n, written, keep, splats.shWords, recordBase);
    });
    ATHENEA_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
    // The decode does not need the total, so it is queued now: `base` is
    // this slice's start, known from the slices before it.
    splatDecode_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        cursor["dest"].setBinding(dest->rhi());
        cursor["positions"].setBinding(splats.positions.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        // Bound whether or not there is one: the offsets in the parameters
        // are what say whether anything is written.
        cursor["pbr"].setBinding(splats.hasPbr() ? splats.pbr.rhi() : splats.shape.rhi());
        cursor["crypto"].setBinding(splats.hasCrypto() ? splats.crypto.rhi() : splats.shape.rhi());
        cursor["transfer"].setBinding(splats.hasTransfer() ? splats.transfer.rhi() : splats.shape.rhi());
        cursor["shadowBits"].setBinding(splats.hasShadowBits() ? splats.shadowBits.rhi() : splats.shape.rhi());
        cursor["origin"].setBinding(splats.origin.rhi());
        cursor["normals"].setBinding(splats.hasNormals() ? splats.normals.rhi() : splats.shape.rhi());
        // A cloud allocated without normals reads none, whatever the
        // encoding says: the buffer bound in their place is the shape.
        cursor["emission"].setBinding(splats.hasEmission() ? splats.emission.rhi() : splats.shape.rhi());
        cursor["lobes"].setBinding(splats.hasLobes() ? splats.lobes.rhi() : splats.shape.rhi());
        io::SplatEncoding decoded = e;
        if (!splats.hasNormals()) {
            decoded.normal = io::SplatEncoding::kNoField;
        }
        if (!splats.hasEmission()) {
            decoded.emission = io::SplatEncoding::kNoField;
        }
        if (!splats.hasLobes()) {
            decoded.lobes = io::SplatEncoding::kNoField;
        }
        setDecodeParams(cursor, decoded, n, written, keep, splats.shWords, recordBase);
    });
    ATHENEA_TRY(batch.submit(true));
    uint32_t kept = 0;
    ATHENEA_TRY(total->read(*device_, 0, sizeof(kept), &kept));
    return kept;
}

Result<gpu::Buffer> CloudLoader::keptOnly(const GpuSplats& splats, const gpu::Buffer& byRecord, uint32_t words,
                                          const char* label) {
    if (splats.count == splats.declared || words == 0) {
        return byRecord;
    }
    auto packed = deviceBuffer(*device_, uint64_t{splats.count} * words, 4, label);
    if (!packed) return std::move(packed).error();
    gpu::CommandBatch batch(*device_);
    splatKept_.dispatch(batch, {splats.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["origin"].setBinding(splats.origin.rhi());
        cursor["byRecord"].setBinding(byRecord.rhi());
        cursor["bySplat"].setBinding(packed->rhi());
        cursor["fromSplat"].setBinding(byRecord.rhi());
        cursor["toRecord"].setBinding(packed->rhi());
        cursor["params"]["count"].setData(splats.count);
        cursor["params"]["words"].setData(words);
    });
    ATHENEA_TRY(batch.submit(true));
    return *packed;
}

Result<gpu::Buffer> CloudLoader::toRecords(const GpuSplats& splats, const gpu::Buffer& bySplat, uint32_t words,
                                           uint32_t fill, const char* label) {
    if (splats.count == splats.declared || words == 0) {
        return bySplat;
    }
    const std::vector<uint32_t> filled(size_t{splats.declared} * words, fill);
    auto spread = gpu::Buffer::fromSpan<uint32_t>(*device_, filled, label);
    if (!spread) return std::move(spread).error();
    gpu::CommandBatch batch(*device_);
    splatKeptScatter_.dispatch(batch, {splats.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["origin"].setBinding(splats.origin.rhi());
        cursor["byRecord"].setBinding(bySplat.rhi());
        cursor["bySplat"].setBinding(spread->rhi());
        cursor["fromSplat"].setBinding(bySplat.rhi());
        cursor["toRecord"].setBinding(spread->rhi());
        cursor["params"]["count"].setData(splats.count);
        cursor["params"]["words"].setData(words);
    });
    ATHENEA_TRY(batch.submit(true));
    return *spread;
}

Result<void> CloudLoader::finishSplats(GpuSplats& splats, uint32_t written) {
    splats.count = written;
    if (splats.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat survived validation", splats.source);
    }
    auto bounds = boundsOf(splats.positions, splats.count);
    if (!bounds) return std::move(bounds).error();
    splats.bounds = *bounds;
    log::info("{}: {} splats of {} (degree {})", splats.source, splats.count, splats.declared, splats.degree());
    return ok();
}

Result<GpuSplats> CloudLoader::upload(const io::RawSplats& raw, uint32_t maxDegree) {
    const io::SplatEncoding& e = raw.encoding;
    if (raw.count == 0 || e.floatsPerRecord == 0 ||
        raw.records.size() < size_t{raw.count} * e.floatsPerRecord) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat records", raw.source);
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(maxDegree, 3u)]);
    const bool pbr = e.metallic != io::SplatEncoding::kNoField ||
                     e.roughness != io::SplatEncoding::kNoField ||
                     e.transmission != io::SplatEncoding::kNoField;
    auto splats = startSplats(raw.source, raw.count, keep, pbr,
                              e.cryptoObject != io::SplatEncoding::kNoField, e.transferCount,
                              e.shadowBits != io::SplatEncoding::kNoField ? e.shadowWords : 0u,
                              e.normal != io::SplatEncoding::kNoField,
                              e.emission != io::SplatEncoding::kNoField,
                              e.lobes != io::SplatEncoding::kNoField);
    if (!splats) return std::move(splats).error();
    splats->linear = raw.linear || e.colour == io::SplatEncoding::Colour::LinearLight;

    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = deviceBuffer(*device_, uint64_t{n} * e.floatsPerRecord, 4, "splats.raw",
                                      raw.records.data() + size_t{first} * e.floatsPerRecord);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto kept = decodeSlice(*rawBuffer, e, n, written, keep, *splats, first);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    ATHENEA_TRY(finishSplats(*splats, written));
    return splats;
}

Result<GpuSplats> CloudLoader::upload(const gpu::Buffer& records, uint32_t count, const io::SplatEncoding& e,
                                      const std::string& source, uint32_t maxDegree) {
    const uint64_t recordBytes = uint64_t{e.floatsPerRecord} * 4;
    if (count == 0 || e.floatsPerRecord == 0 || !records.valid() || records.bytes() < uint64_t{count} * recordBytes) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no splat records on the device", source);
    }
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = std::min(e.restPerColour, kPerDegree[std::min(maxDegree, 3u)]);
    const bool pbr = e.metallic != io::SplatEncoding::kNoField ||
                     e.roughness != io::SplatEncoding::kNoField ||
                     e.transmission != io::SplatEncoding::kNoField;
    auto splats = startSplats(source, count, keep, pbr, e.cryptoObject != io::SplatEncoding::kNoField,
                              e.transferCount,
                              e.shadowBits != io::SplatEncoding::kNoField ? e.shadowWords : 0u,
                              e.normal != io::SplatEncoding::kNoField,
                              e.emission != io::SplatEncoding::kNoField,
                              e.lobes != io::SplatEncoding::kNoField);
    if (!splats) return std::move(splats).error();
    // Slices as the host upload takes them; a slice of a larger buffer is
    // copied out of it on the device, and one that is the whole is the
    // buffer itself.
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < count; first += perSlice) {
        const uint32_t n = std::min(perSlice, count - first);
        gpu::Buffer slice = records;
        if (n != count) {
            auto part = deviceBuffer(*device_, uint64_t{n} * e.floatsPerRecord, 4, "splats.raw");
            if (!part) return std::move(part).error();
            gpu::CommandBatch batch(*device_);
            batch.encoder()->copyBuffer(part->rhi(), 0, records.rhi(), uint64_t{first} * recordBytes,
                                        uint64_t{n} * recordBytes);
            batch.markDirty();
            ATHENEA_TRY(batch.submit(true));
            slice = std::move(*part);
        }
        auto kept = decodeSlice(slice, e, n, written, keep, *splats, first);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    ATHENEA_TRY(finishSplats(*splats, written));
    return splats;
}

struct CloudLoader::SogOnDevice {
    const io::RawSog*  sog = nullptr;
    uint32_t           keep = 0;
    io::SplatEncoding  encoding;
    gpu::Buffer        meansL, meansU, quats, scales, sh0, labels, centroids, codebooks;
};

Result<CloudLoader::SogOnDevice> CloudLoader::sogOnDevice(const io::RawSog& sog, uint32_t maxDegree) {
    SogOnDevice on;
    on.sog = &sog;
    on.keep = std::min(sog.shCoefficients, std::array<uint32_t, 4>{0, 3, 8, 15}[std::min(maxDegree, 3u)]);
    io::SplatEncoding& e = on.encoding;
    e.floatsPerRecord = 14 + on.keep * 3;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14;
    e.restPerColour = on.keep;
    e.restColourOuter = 0;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Log;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;

    const auto image = [&](const io::SogImage& from, const char* label, gpu::Buffer& into) -> Result<void> {
        static const uint32_t kPlaceholder = 0;
        const bool none = from.empty();
        auto made = deviceBuffer(*device_, none ? 1 : from.texels.size(), 4, label,
                                 none ? &kPlaceholder : from.texels.data());
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(image(sog.meansL, "sog.meansL", on.meansL));
    ATHENEA_TRY(image(sog.meansU, "sog.meansU", on.meansU));
    ATHENEA_TRY(image(sog.quats, "sog.quats", on.quats));
    ATHENEA_TRY(image(sog.scales, "sog.scales", on.scales));
    ATHENEA_TRY(image(sog.sh0, "sog.sh0", on.sh0));
    ATHENEA_TRY(image(on.keep > 0 ? sog.shLabels : io::SogImage{}, "sog.shLabels", on.labels));
    ATHENEA_TRY(image(on.keep > 0 ? sog.shCentroids : io::SogImage{}, "sog.shCentroids", on.centroids));
    std::vector<float> books(768, 0.0F);
    const auto place = [&](const std::vector<float>& book, size_t at) {
        std::copy_n(book.begin(), std::min<size_t>(book.size(), 256), books.begin() + static_cast<long>(at));
    };
    place(sog.scalesBook, 0);
    place(sog.sh0Book, 256);
    place(sog.shNBook, 512);
    auto codebooks = deviceBuffer(*device_, books.size(), 4, "sog.codebooks", books.data());
    if (!codebooks) return std::move(codebooks).error();
    on.codebooks = std::move(*codebooks);
    return on;
}

Result<void> CloudLoader::sogSlice(const SogOnDevice& on, uint32_t first, uint32_t n, const gpu::Buffer& into) {
    const io::RawSog& sog = *on.sog;
    gpu::CommandBatch batch(*device_);
    sogDecode_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["meansL"].setBinding(on.meansL.rhi());
        cursor["meansU"].setBinding(on.meansU.rhi());
        cursor["quats"].setBinding(on.quats.rhi());
        cursor["scales"].setBinding(on.scales.rhi());
        cursor["sh0"].setBinding(on.sh0.rhi());
        cursor["shLabels"].setBinding(on.labels.rhi());
        cursor["shCentroids"].setBinding(on.centroids.rhi());
        cursor["codebooks"].setBinding(on.codebooks.rhi());
        cursor["records"].setBinding(into.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["count"].setData(n);
        p["first"].setData(first);
        p["stride"].setData(on.encoding.floatsPerRecord);
        p["keep"].setData(on.keep);
        p["version"].setData(sog.version);
        p["coefficients"].setData(sog.shCoefficients);
        p["centroidsWidth"].setData(sog.shCentroids.width);
        static constexpr const char* kMeans[6] = {"meansMinX", "meansMinY", "meansMinZ",
                                                  "meansMaxX", "meansMaxY", "meansMaxZ"};
        static constexpr const char* kScales[6] = {"scalesMinX", "scalesMinY", "scalesMinZ",
                                                   "scalesMaxX", "scalesMaxY", "scalesMaxZ"};
        static constexpr const char* kSh0[8] = {"sh0MinR", "sh0MinG", "sh0MinB", "sh0MinA",
                                                "sh0MaxR", "sh0MaxG", "sh0MaxB", "sh0MaxA"};
        for (size_t k = 0; k < 3; ++k) {
            p[kMeans[k]].setData(sog.meansMin[k]);
            p[kMeans[k + 3]].setData(sog.meansMax[k]);
            p[kScales[k]].setData(sog.scalesMin[k]);
            p[kScales[k + 3]].setData(sog.scalesMax[k]);
        }
        for (size_t k = 0; k < 4; ++k) {
            p[kSh0[k]].setData(sog.sh0Min[k]);
            p[kSh0[k + 4]].setData(sog.sh0Max[k]);
        }
        p["shNMin"].setData(sog.shNMin);
        p["shNMax"].setData(sog.shNMax);
    });
    return batch.submit(true);
}

Result<GpuSplats> CloudLoader::upload(const io::RawSog& sog, uint32_t maxDegree) {
    auto on = sogOnDevice(sog, maxDegree);
    if (!on) return std::move(on).error();
    auto splats = startSplats(sog.source, sog.count, on->keep);
    if (!splats) return std::move(splats).error();
    const uint64_t recordBytes = uint64_t{on->encoding.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    uint32_t written = 0;
    for (uint32_t first = 0; first < sog.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, sog.count - first);
        auto records = deviceBuffer(*device_, uint64_t{n} * on->encoding.floatsPerRecord, 4, "sog.records");
        if (!records) return std::move(records).error();
        ATHENEA_TRY(sogSlice(*on, first, n, *records));
        auto kept = decodeSlice(*records, on->encoding, n, written, on->keep, *splats, first);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    ATHENEA_TRY(finishSplats(*splats, written));
    return splats;
}

Result<io::RawSplats> CloudLoader::records(const io::RawSog& sog, uint32_t maxDegree) {
    auto on = sogOnDevice(sog, maxDegree);
    if (!on) return std::move(on).error();
    io::RawSplats raw;
    raw.source = sog.source;
    raw.count = sog.count;
    raw.encoding = on->encoding;
    raw.records.resize(size_t{sog.count} * on->encoding.floatsPerRecord);
    const uint64_t recordBytes = uint64_t{on->encoding.floatsPerRecord} * 4;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / recordBytes));
    for (uint32_t first = 0; first < sog.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, sog.count - first);
        auto buffer = deviceBuffer(*device_, uint64_t{n} * on->encoding.floatsPerRecord, 4, "sog.records");
        if (!buffer) return std::move(buffer).error();
        ATHENEA_TRY(sogSlice(*on, first, n, *buffer));
        ATHENEA_TRY(buffer->read(*device_, 0, uint64_t{n} * recordBytes,
                             raw.records.data() + size_t{first} * on->encoding.floatsPerRecord));
    }
    return raw;
}

Result<uint32_t> CloudLoader::decodePoints(const gpu::Buffer& raw, uint32_t n, uint32_t first,
                                           uint32_t colourKind, float detail, uint32_t written,
                                           GpuPoints& points) {
    auto valid = deviceBuffer(*device_, n, 4, "points.valid");
    if (!valid) return std::move(valid).error();
    auto dest = deviceBuffer(*device_, n, 4, "points.dest");
    if (!dest) return std::move(dest).error();
    auto total = deviceBuffer(*device_, 1, 4, "points.total");
    if (!total) return std::move(total).error();
    const auto params = [&](rhi::ShaderCursor cursor) {
        cursor["params"]["count"].setData(n);
        cursor["params"]["base"].setData(first);
        cursor["params"]["colourKind"].setData(colourKind);
        cursor["params"]["detail"].setData(detail);
    };
    gpu::CommandBatch batch(*device_);
    pointsValidate_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        params(cursor);
    });
    ATHENEA_TRY(prefix_.apply(batch, *valid, *dest, *total, n));
    ATHENEA_TRY(batch.submit(true));
    gpu::CommandBatch decode(*device_);
    pointsDecode_.dispatch(decode, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["raw"].setBinding(raw.rhi());
        cursor["valid"].setBinding(valid->rhi());
        cursor["dest"].setBinding(dest->rhi());
        cursor["positions"].setBinding(points.positions.rhi());
        cursor["colours"].setBinding(points.colours.rhi());
        params(cursor);
        // The decode writes at base + dest; base here is where this
        // slice's survivors start, not where its records started.
        cursor["params"]["base"].setData(written);
    });
    ATHENEA_TRY(decode.submit(true));
    uint32_t kept = 0;
    ATHENEA_TRY(total->read(*device_, 0, sizeof(kept), &kept));
    return kept;
}

Result<GpuPoints> CloudLoader::upload(const io::RawPoints& raw, float detail) {
    if (raw.count == 0 || raw.records.size() < size_t{raw.count} * 6) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no points", raw.source);
    }
    GpuPoints points;
    points.source = raw.source;
    points.declared = raw.count;
    auto positions = deviceBuffer(*device_, raw.count, 16, "points.positions");
    if (!positions) return std::move(positions).error();
    auto colours = deviceBuffer(*device_, uint64_t{raw.count} * 2, 4, "points.colours");
    if (!colours) return std::move(colours).error();
    points.positions = *positions;
    points.colours = *colours;

    const uint32_t perSlice = static_cast<uint32_t>(kSliceBytes / 24);
    uint32_t written = 0;
    for (uint32_t first = 0; first < raw.count; first += perSlice) {
        const uint32_t n = std::min(perSlice, raw.count - first);
        auto rawBuffer = deviceBuffer(*device_, uint64_t{n} * 6, 4, "points.raw",
                                      raw.records.data() + size_t{first} * 6);
        if (!rawBuffer) return std::move(rawBuffer).error();
        auto kept = decodePoints(*rawBuffer, n, first, raw.colourKind, detail, written, points);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    points.count = written;
    if (points.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no point survived", raw.source);
    }
    auto bounds = boundsOf(points.positions, points.count);
    if (!bounds) return std::move(bounds).error();
    points.bounds = *bounds;
    log::info("{}: {} points of {}", raw.source, points.count, points.declared);
    return points;
}

Result<gpu::Buffer> CloudLoader::planeBuffer(const std::vector<FloatStream>& planes, uint32_t count,
                                             const char* label) {
    // One buffer, each plane's `count` rgb values written where the one
    // before it ends: the bytes as they are, nothing read on the host.
    const uint64_t planeBytes = uint64_t{count} * 3 * (planes.front().half ? 2 : 4);
    auto made = deviceBuffer(*device_, std::max<uint64_t>((planeBytes * planes.size() + 3) / 4, 1), 4, label);
    if (!made) return std::move(made).error();
    for (size_t k = 0; k < planes.size(); ++k) {
        ATHENEA_TRY(made->write(*device_, planeBytes * k, planeBytes, planes[k].bytes.data()));
    }
    return made;
}

Result<gpu::Buffer> CloudLoader::streamBuffer(const FloatStream& stream, const char* label) {
    // Words of four bytes; an odd count of halves leaves the last word short,
    // so the bytes are written into a buffer already the whole word long.
    const uint64_t words = std::max<uint64_t>((stream.bytes.size() + 3) / 4, 1);
    auto made = deviceBuffer(*device_, words, 4, label);
    if (!made) return std::move(made).error();
    if (!stream.empty()) {
        ATHENEA_TRY(made->write(*device_, 0, stream.bytes.size(), stream.bytes.data()));
    }
    return made;
}

namespace {

constexpr uint32_t kPositions = 1, kRotations = 2, kScales = 4, kOpacities = 8, kSh = 16, kColours = 32,
                   kMetallic = 64, kRoughness = 128, kTransmission = 256, kCrypto = 512,
                   kTransferDirect = 1024, kTransferIndirect = 2048, kShadowBits = 4096,
                   kThinWalled = 8192, kNormals = 16384, kEmission = 32768,
                   kBase = 65536, kShPlanes = 131072, kSpecularWeight = 1u << 18,
                   kSpecularColour = 1u << 19, kSpecularIor = 1u << 20, kCoatWeight = 1u << 21,
                   kCoatRoughness = 1u << 22, kCoatIor = 1u << 23, kSheenColour = 1u << 24,
                   kSheenRoughness = 1u << 25, kCoatDarkening = 1u << 26, kTransferReflected = 1u << 27,
                   kTransferZonal = 1u << 28,
                   kSchlickMetal = 1u << 29;

}   // namespace

Result<GpuSplats> CloudLoader::upload(const SplatStreams& in, uint32_t maxDegree) {
    const uint32_t n = in.count;
    const auto holds = [&](const FloatStream& stream, uint64_t perElement) {
        return stream.empty() || stream.values() >= uint64_t{n} * perElement;
    };
    const bool shapes = in.coefficients == 0 || in.coefficients == 1 || in.coefficients == 4 ||
                        in.coefficients == 9 || in.coefficients == 16;
    // Blender's planes: one array a basis function, all of one kind, and as
    // many as the coefficients say. Halves are laid end to end, so a plane
    // of an odd number of them would start inside a word.
    const bool planes = !in.shPlanes.empty();
    bool planesFit = !planes || (in.coefficients == in.shPlanes.size() + 1 && in.sh.empty());
    for (const FloatStream& plane : in.shPlanes) {
        planesFit = planesFit && plane.values() >= uint64_t{n} * 3 && !plane.isDouble &&
                    plane.half == in.shPlanes.front().half && (!plane.half || (uint64_t{n} * 3) % 2 == 0);
    }
    if (n == 0 || in.positions.values() < uint64_t{n} * 3 || !holds(in.rotations, 4) || !holds(in.scales, 3) ||
        !holds(in.opacities, 1) || !holds(in.radianceBase, 4) || !shapes || !planesFit ||
        (!planes && !holds(in.sh, uint64_t{in.coefficients} * 3))) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': splat arrays of the wrong lengths", in.source);
    }
    const bool haveSh = (planes || !in.sh.empty()) && in.coefficients > 0;
    static constexpr uint32_t kPerDegree[] = {0, 3, 8, 15};
    const uint32_t keep = haveSh ? std::min(in.coefficients - 1, kPerDegree[std::min(maxDegree, 3u)]) : 0;

    const bool havePbr = !in.metallic.empty() || !in.roughness.empty() || !in.transmission.empty();
    const bool haveCrypto = !in.cryptoObject.empty() && in.cryptoObject.values() >= n;
    // Nine values a gaussian of the direct half, and twenty-seven more where
    // the indirect one is there: a cloud carries both or the direct alone.
    // Sixteen and forty-eight at degree 3 (a TX transfer's), and its
    // reflected field's forty-eight after them: the count is the layout
    // (athenea/common/transfer_layout.slang).
    const bool haveDirect = !in.transferDirect.empty() && in.transferDirect.values() >= uint64_t{n} * 9;
    const uint32_t directCount = haveDirect && in.transferDirect.values() >= uint64_t{n} * 16 ? 16u : 9u;
    const bool haveIndirect = haveDirect && !in.transferIndirect.empty() &&
                              in.transferIndirect.values() >= uint64_t{n} * directCount * 3;
    const bool haveField = haveIndirect && !in.transferReflected.empty() &&
                           in.transferReflected.values() >= uint64_t{n} * 48;
    // A ZONAL TRANSFER, ten values a gaussian, wins where both are there: it
    // turns with the gaussian, which harmonics in the world do not.
    const bool haveZonal = !in.transferZonal.empty() &&
                           in.transferZonal.values() >= uint64_t{n} * GpuSplats::kTransferZonalCount;
    const uint32_t transferCount =
        haveZonal ? GpuSplats::kTransferZonalCount
                  : (haveDirect ? directCount + (haveIndirect ? directCount * 3 : 0u) + (haveField ? 48u : 0u) : 0u);
    // Two words a gaussian of which ways out are open, only beside a transfer.
    const bool haveShadowBits = transferCount > 0 && !in.shadowBits.empty() &&
                                in.shadowBits.values() >= uint64_t{n} * 2;
    // Eight words a gaussian where the stage carries the 256 cells of a TX
    // transfer, two where it carries the 64 of the first: the count says.
    const uint32_t shadowWords = !haveShadowBits ? 0u : in.shadowBits.values() >= uint64_t{n} * 8 ? 8u : 2u;
    // The shading normal, three floats a gaussian, where the stage carries it.
    const bool haveNormals = !in.normals.empty() && in.normals.values() >= uint64_t{n} * 3;
    // The radiance it gives off, three floats a gaussian, where the stage carries it.
    const bool haveEmission = !in.emission.empty() && in.emission.values() >= uint64_t{n} * 3;
    // The layers over the base, one array each: each is read where it is
    // whole, and the cloud carries them where any one is.
    const auto whole = [n](const FloatStream& stream, uint32_t per) {
        return !stream.empty() && stream.values() >= uint64_t{n} * per;
    };
    struct LobeArray {
        const FloatStream* stream;
        uint32_t           per;
        uint32_t           bit;
        const char*        label;
    };
    const LobeArray lobeArrays[] = {
        {&in.specularWeight, 1, kSpecularWeight, "splats.stream.specularWeight"},
        {&in.specularColour, 3, kSpecularColour, "splats.stream.specularColor"},
        {&in.specularIor, 1, kSpecularIor, "splats.stream.specularIor"},
        {&in.coatWeight, 1, kCoatWeight, "splats.stream.coatWeight"},
        {&in.coatRoughness, 1, kCoatRoughness, "splats.stream.coatRoughness"},
        {&in.coatIor, 1, kCoatIor, "splats.stream.coatIor"},
        {&in.sheenColour, 3, kSheenColour, "splats.stream.sheenColor"},
        {&in.sheenRoughness, 1, kSheenRoughness, "splats.stream.sheenRoughness"},
        {&in.coatDarkening, 1, kCoatDarkening, "splats.stream.coatDarkening"}};
    const bool haveLobes = std::any_of(std::begin(lobeArrays), std::end(lobeArrays),
                                       [&](const LobeArray& a) { return whole(*a.stream, a.per); });
    io::SplatEncoding e;
    e.floatsPerRecord = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0) + transferCount +
                        shadowWords + (haveNormals ? 3 : 0) + (haveEmission ? 3 : 0) +
                        (haveLobes ? 13 : 0);
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = keep; e.restColourOuter = 0;
    if (havePbr) {
        e.metallic = 14 + keep * 3;
        e.roughness = e.metallic + 1;
        e.transmission = e.metallic + 2;
    }
    if (haveCrypto) {
        e.cryptoObject = 14 + keep * 3 + (havePbr ? 3 : 0);
    }
    if (transferCount > 0) {
        e.transferBase = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0);
        e.transferCount = transferCount;
    }
    if (haveShadowBits) {
        e.shadowBits = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0) + transferCount;
        e.shadowWords = shadowWords;
    }
    if (haveNormals) {
        e.normal = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0) + transferCount +
                   shadowWords;
    }
    if (haveEmission) {
        e.emission = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0) + transferCount +
                     shadowWords + (haveNormals ? 3 : 0);
    }
    if (haveLobes) {
        e.lobes = 14 + keep * 3 + (havePbr ? 3 : 0) + (haveCrypto ? 1 : 0) + transferCount +
                  shadowWords + (haveNormals ? 3 : 0) + (haveEmission ? 3 : 0);
    }
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;

    uint32_t present = kPositions;
    uint32_t halves = in.positions.half ? kPositions : 0;
    const auto note = [&](const FloatStream& stream, uint32_t bit) {
        present |= stream.empty() ? 0 : bit;
        halves |= (!stream.empty() && stream.half) ? bit : 0;
    };
    note(in.rotations, kRotations);
    note(in.scales, kScales);
    note(in.opacities, kOpacities);
    note(in.metallic, kMetallic);
    note(in.roughness, kRoughness);
    note(in.transmission, kTransmission);
    note(in.cryptoObject, kCrypto);
    if (haveZonal) {
        note(in.transferZonal, kTransferZonal);
    } else {
        note(in.transferDirect, kTransferDirect);
        note(in.transferIndirect, kTransferIndirect);
        if (haveField) {
            note(in.transferReflected, kTransferReflected);
        }
    }
    if (haveShadowBits) {
        note(in.shadowBits, kShadowBits);
    }
    const bool haveThin = havePbr && !in.thinWalled.empty() && in.thinWalled.values() >= n;
    if (haveThin) {
        note(in.thinWalled, kThinWalled);
    }
    const bool haveSchlick = havePbr && !in.schlickMetal.empty() && in.schlickMetal.values() >= n;
    if (haveSchlick) {
        note(in.schlickMetal, kSchlickMetal);
    }
    note(in.radianceBase, kBase);
    if (haveSh) {
        note(planes ? in.shPlanes.front() : in.sh, kSh);
        present |= planes ? kShPlanes : 0;
    }
    if (haveNormals) {
        note(in.normals, kNormals);
    }
    if (haveEmission) {
        note(in.emission, kEmission);
    }
    for (const LobeArray& a : lobeArrays) {
        if (whole(*a.stream, a.per)) {
            note(*a.stream, a.bit);
        }
    }
    auto positions = streamBuffer(in.positions, "splats.stream.positions");
    if (!positions) return std::move(positions).error();
    auto rotations = streamBuffer(in.rotations, "splats.stream.rotations");
    if (!rotations) return std::move(rotations).error();
    auto scales = streamBuffer(in.scales, "splats.stream.scales");
    if (!scales) return std::move(scales).error();
    auto opacities = streamBuffer(in.opacities, "splats.stream.opacities");
    if (!opacities) return std::move(opacities).error();
    auto sh = planes ? planeBuffer(in.shPlanes, n, "splats.stream.shPlanes")
                     : streamBuffer(haveSh ? in.sh : FloatStream{}, "splats.stream.sh");
    if (!sh) return std::move(sh).error();
    auto radianceBase = streamBuffer(in.radianceBase, "splats.stream.radianceBase");
    if (!radianceBase) return std::move(radianceBase).error();
    auto none = streamBuffer({}, "splats.stream.none");
    if (!none) return std::move(none).error();

    auto metallic = streamBuffer(in.metallic, "splats.stream.metallic");
    if (!metallic) return std::move(metallic).error();
    auto roughness = streamBuffer(in.roughness, "splats.stream.roughness");
    if (!roughness) return std::move(roughness).error();
    auto transmission = streamBuffer(in.transmission, "splats.stream.transmission");
    if (!transmission) return std::move(transmission).error();
    auto cryptoObject = streamBuffer(in.cryptoObject, "splats.stream.crypto");
    if (!cryptoObject) return std::move(cryptoObject).error();
    auto transferDirect = streamBuffer(in.transferDirect, "splats.stream.transferDirect");
    if (!transferDirect) return std::move(transferDirect).error();
    auto transferIndirect = streamBuffer(in.transferIndirect, "splats.stream.transferIndirect");
    if (!transferIndirect) return std::move(transferIndirect).error();
    auto transferReflected =
        streamBuffer(haveField ? in.transferReflected : FloatStream{}, "splats.stream.transferReflected");
    if (!transferReflected) return std::move(transferReflected).error();
    auto transferZonal = streamBuffer(haveZonal ? in.transferZonal : FloatStream{}, "splats.stream.transferZonal");
    if (!transferZonal) return std::move(transferZonal).error();
    auto thinWalled = streamBuffer(haveThin ? in.thinWalled : FloatStream{}, "splats.stream.thinWalled");
    if (!thinWalled) return std::move(thinWalled).error();
    auto schlickMetal = streamBuffer(haveSchlick ? in.schlickMetal : FloatStream{}, "splats.stream.schlickMetal");
    if (!schlickMetal) return std::move(schlickMetal).error();
    auto shadowBits = streamBuffer(haveShadowBits ? in.shadowBits : FloatStream{}, "splats.stream.shadowBits");
    if (!shadowBits) return std::move(shadowBits).error();
    auto normals = streamBuffer(haveNormals ? in.normals : FloatStream{}, "splats.stream.normals");
    if (!normals) return std::move(normals).error();
    auto emission = streamBuffer(haveEmission ? in.emission : FloatStream{}, "splats.stream.emission");
    if (!emission) return std::move(emission).error();
    std::vector<gpu::Buffer> lobeBuffers;
    for (const LobeArray& a : lobeArrays) {
        auto made = streamBuffer(whole(*a.stream, a.per) ? *a.stream : FloatStream{}, a.label);
        if (!made) return std::move(made).error();
        lobeBuffers.push_back(std::move(*made));
    }
    auto splats = startSplats(in.source, n, keep, havePbr, haveCrypto, transferCount, shadowWords, haveNormals,
                              haveEmission, haveLobes);
    if (!splats) return std::move(splats).error();
    splats->linear = in.linear;
    const uint32_t perSlice = static_cast<uint32_t>(std::max<uint64_t>(1, kSliceBytes / (uint64_t{e.floatsPerRecord} * 4)));
    uint32_t written = 0;
    for (uint32_t first = 0; first < n; first += perSlice) {
        const uint32_t count = std::min(perSlice, n - first);
        auto records = deviceBuffer(*device_, uint64_t{count} * e.floatsPerRecord, 4, "splats.raw");
        if (!records) return std::move(records).error();
        {
            gpu::CommandBatch batch(*device_);
            splatStreams_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(positions->rhi());
                cursor["rotations"].setBinding(rotations->rhi());
                cursor["scales"].setBinding(scales->rhi());
                cursor["opacities"].setBinding(opacities->rhi());
                cursor["sh"].setBinding(sh->rhi());
                cursor["colours"].setBinding(none->rhi());
                cursor["metallic"].setBinding(metallic->rhi());
                cursor["roughness"].setBinding(roughness->rhi());
                cursor["transmission"].setBinding(transmission->rhi());
                cursor["cryptoObject"].setBinding(haveCrypto ? cryptoObject->rhi() : none->rhi());
                cursor["transferDirect"].setBinding(haveDirect ? transferDirect->rhi() : none->rhi());
                cursor["transferIndirect"].setBinding(haveIndirect ? transferIndirect->rhi() : none->rhi());
                cursor["transferReflected"].setBinding(haveField ? transferReflected->rhi() : none->rhi());
                cursor["transferZonal"].setBinding(haveZonal ? transferZonal->rhi() : none->rhi());
                cursor["shadowBits"].setBinding(haveShadowBits ? shadowBits->rhi() : none->rhi());
                cursor["thinWalled"].setBinding(haveThin ? thinWalled->rhi() : none->rhi());
                cursor["schlickMetal"].setBinding(haveSchlick ? schlickMetal->rhi() : none->rhi());
                cursor["normals"].setBinding(haveNormals ? normals->rhi() : none->rhi());
                cursor["emission"].setBinding(haveEmission ? emission->rhi() : none->rhi());
                {
                    static constexpr const char* kLobeNames[] = {"specularWeight", "specularColour", "specularIor",
                                                                 "coatWeight",     "coatRoughness",  "coatIor",
                                                                 "sheenColour",    "sheenRoughness", "coatDarkening"};
                    for (size_t k = 0; k < lobeBuffers.size(); ++k) {
                        cursor[kLobeNames[k]].setBinding(whole(*lobeArrays[k].stream, lobeArrays[k].per)
                                                             ? lobeBuffers[k].rhi()
                                                             : none->rhi());
                    }
                }
                cursor["radianceBase"].setBinding(radianceBase->rhi());
                cursor["records"].setBinding(records->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["first"].setData(first);
                p["planeLength"].setData(n);
                p["stride"].setData(e.floatsPerRecord);
                p["keep"].setData(keep);
                p["coefficients"].setData(in.coefficients);
                p["present"].setData(present);
                p["halves"].setData(halves);
                p["colourMode"].setData(uint32_t{0});
                p["pbrBase"].setData(havePbr ? e.metallic : io::SplatEncoding::kNoField);
                p["cryptoBase"].setData(haveCrypto ? e.cryptoObject : io::SplatEncoding::kNoField);
                p["transferBase"].setData(transferCount > 0 ? e.transferBase : io::SplatEncoding::kNoField);
                p["transferCount"].setData(transferCount);
                p["shadowBase"].setData(haveShadowBits ? e.shadowBits : io::SplatEncoding::kNoField);
                p["shadowWords"].setData(shadowWords);
                p["normalBase"].setData(haveNormals ? e.normal : io::SplatEncoding::kNoField);
                p["emissionBase"].setData(haveEmission ? e.emission : io::SplatEncoding::kNoField);
                p["lobesBase"].setData(haveLobes ? e.lobes : io::SplatEncoding::kNoField);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto kept = decodeSlice(*records, e, count, written, keep, *splats, first);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    ATHENEA_TRY(finishSplats(*splats, written));
    // The shape operator, three floats a record, laid out a kept splat each.
    if (!in.curvature.empty() && !in.curvature.half && !in.curvature.isDouble &&
        in.curvature.values() >= uint64_t{n} * 3) {
        auto byRecord = streamBuffer(in.curvature, "splats.stream.curvature");
        if (!byRecord) return std::move(byRecord).error();
        auto kept = keptOnly(*splats, *byRecord, 3, "splats.curvature");
        if (!kept) return std::move(kept).error();
        splats->curvature = std::move(*kept);
    }
    return splats;
}

Result<GpuPoints> CloudLoader::upload(const PointStreams& in, float detail) {
    const uint32_t n = in.count;
    const uint64_t colours = in.colours.values() / 3;
    if (n == 0 || in.positions.values() < uint64_t{n} * 3 || (!in.colours.empty() && colours != 1 && colours < n)) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': point arrays of the wrong lengths", in.source);
    }
    const uint32_t colourMode = in.colours.empty() ? 0 : colours >= n && n > 1 ? 2 : 1;
    GpuPoints points;
    points.source = in.source;
    points.declared = n;
    auto positionsOut = deviceBuffer(*device_, n, 16, "points.positions");
    if (!positionsOut) return std::move(positionsOut).error();
    auto coloursOut = deviceBuffer(*device_, uint64_t{n} * 2, 4, "points.colours");
    if (!coloursOut) return std::move(coloursOut).error();
    points.positions = *positionsOut;
    points.colours = *coloursOut;

    auto positions = streamBuffer(in.positions, "points.stream.positions");
    if (!positions) return std::move(positions).error();
    auto colourStream = streamBuffer(in.colours, "points.stream.colours");
    if (!colourStream) return std::move(colourStream).error();
    auto none = streamBuffer({}, "points.stream.none");
    if (!none) return std::move(none).error();
    const uint32_t halves = (in.positions.half ? kPositions : 0) | (in.colours.half ? kColours : 0);

    const uint32_t perSlice = static_cast<uint32_t>(kSliceBytes / 24);
    uint32_t written = 0;
    for (uint32_t first = 0; first < n; first += perSlice) {
        const uint32_t count = std::min(perSlice, n - first);
        auto records = deviceBuffer(*device_, uint64_t{count} * 6, 4, "points.raw");
        if (!records) return std::move(records).error();
        {
            gpu::CommandBatch batch(*device_);
            pointStreams_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["positions"].setBinding(positions->rhi());
                cursor["colours"].setBinding(colourStream->rhi());
                for (const char* unused :
                     {"rotations", "scales", "opacities", "sh", "metallic", "roughness", "transmission", "radianceBase"}) {
                    cursor[unused].setBinding(none->rhi());
                }
                cursor["records"].setBinding(records->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["first"].setData(first);
                p["stride"].setData(uint32_t{6});
                p["keep"].setData(uint32_t{0});
                p["coefficients"].setData(uint32_t{0});
                p["present"].setData(kPositions | (colourMode != 0 ? kColours : 0));
                p["halves"].setData(halves);
                p["colourMode"].setData(colourMode);
                p["pbrBase"].setData(io::SplatEncoding::kNoField);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto kept = decodePoints(*records, count, first, colourMode != 0 ? 2u : 0u, detail, written, points);
        if (!kept) return std::move(kept).error();
        written += *kept;
    }
    points.count = written;
    if (points.count == 0) {
        return Error::make(ErrorCode::InvalidArgument, "'{}': no point survived", in.source);
    }
    auto bounds = boundsOf(points.positions, points.count);
    if (!bounds) return std::move(bounds).error();
    points.bounds = *bounds;
    log::info("{}: {} points of {}", in.source, points.count, points.declared);
    return points;
}

bool isSog(const std::filesystem::path& path) {
    return path.extension() == ".sog" || path.filename() == "meta.json";
}

Result<GpuSplats> loadSplatFile(CloudLoader& loader, const std::filesystem::path& path, uint32_t maxDegree) {
    if (isSog(path)) {
        auto sog = io::readSog(path);
        if (!sog) return std::move(sog).error();
        return loader.upload(*sog, maxDegree);
    }
    auto raw = io::readSplats(path);
    if (!raw) return std::move(raw).error();
    return loader.upload(*raw, maxDegree);
}

Result<io::RawSplats> readSplatRecords(CloudLoader& loader, const std::filesystem::path& path, uint32_t maxDegree) {
    if (isSog(path)) {
        auto sog = io::readSog(path);
        if (!sog) return std::move(sog).error();
        return loader.records(*sog, maxDegree);
    }
    return io::readSplats(path);
}

}   // namespace athenea::scene
