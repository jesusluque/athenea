// Copyright (c) 2026 jesus luque.
#include "athenea/video/Planes.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::video {
namespace {

constexpr uint32_t kRun = 1024;          // video_planes_encode.slang's
constexpr uint32_t kCompareRun = 64;     // video_planes_decode.slang's

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t bytes, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes, element);
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// `buffer` holds at least `bytes`: made again when it is smaller.
Result<void> ensure(gpu::Device& device, gpu::Buffer& buffer, uint64_t bytes, uint32_t element, const char* label) {
    if (buffer.valid() && buffer.bytes() >= bytes) {
        return ok();
    }
    auto made = deviceBuffer(device, bytes, element, label);
    if (!made) return std::move(made).error();
    buffer = std::move(*made);
    return ok();
}

void setLayout(rhi::ShaderCursor p, const PlaneLayout& l) {
    p["count"].setData(l.splats);
    p["bits"].setData(l.bits);
    p["restPerColour"].setData(l.restPerColour);
    p["attributes"].setData(l.attributes());
    p["planes"].setData(l.planes());
    p["hiInVideo"].setData(uint32_t{l.hiInVideo ? 1u : 0u});
    p["columns"].setData(l.columns);
    p["tileHeight"].setData(l.tileHeight);
    p["width"].setData(l.width);
    p["height"].setData(l.height);
    p["chunkSplats"].setData(l.chunkSplats);
    p["sampleScale"].setData(uint32_t{l.bits > 8 ? 65535u : 255u});
    p["sampleShift"].setData(uint32_t{l.bits > 8 ? 16u - l.bits : 0u});
}

}   // namespace

Result<PlaneLayout> PlaneLayout::make(uint32_t nSplats, uint32_t nBits, uint32_t nRest, bool nHi,
                                      uint32_t nChunk, uint32_t nColumns, uint32_t tileSplats) {
    if (nBits != 8 && nBits != 10) {
        return Error::make(ErrorCode::InvalidArgument, "video planes: {} bits a sample (8 or 10)", nBits);
    }
    if (nRest != 0 && nRest != 3 && nRest != 8 && nRest != 15) {
        return Error::make(ErrorCode::InvalidArgument, "video planes: {} rest coefficients a colour", nRest);
    }
    if (nChunk == 0 || nChunk % kBandSplats != 0) {
        return Error::make(ErrorCode::InvalidArgument, "video planes: chunks of {} splats, not a multiple of {}",
                           nChunk, kBandSplats);
    }
    if (nSplats == 0) {
        return Error(ErrorCode::InvalidArgument, "video planes: a frame of no splats");
    }
    PlaneLayout l;
    l.splats = nSplats;
    l.bits = nBits;
    l.restPerColour = nRest;
    l.hiInVideo = nHi;
    l.chunkSplats = nChunk;
    const uint32_t tileChunks = std::max(l.chunks(), (tileSplats + nChunk - 1) / nChunk);
    l.tileHeight = tileChunks * (nChunk / kTileWidth);
    const uint32_t planes = l.planes();
    uint32_t best = 0;
    uint64_t bestSide = std::numeric_limits<uint64_t>::max(), bestArea = bestSide;
    for (uint32_t c = 1; c <= planes; ++c) {
        if (nColumns != 0 && c != nColumns) {
            continue;
        }
        const uint64_t w = uint64_t{c} * kTileWidth;
        const uint64_t h = uint64_t{(planes + c - 1) / c} * l.tileHeight;
        if (w > kMostSide || h > kMostSide || w * h > kMostArea) {
            continue;
        }
        const uint64_t side = std::max(w, h), area = w * h;
        if (side < bestSide || (side == bestSide && area < bestArea)) {
            best = c;
            bestSide = side;
            bestArea = area;
        }
    }
    if (best == 0) {
        return Error::make(ErrorCode::InvalidArgument,
                           "video planes: {} planes of {} x {} do not fit a mosaic of at most {} a side and {} "
                           "pixels{}",
                           planes, kTileWidth, l.tileHeight, kMostSide, kMostArea,
                           nColumns != 0 ? " with the nColumns asked for" : "");
    }
    l.columns = best;
    l.width = best * kTileWidth;
    l.height = ((planes + best - 1) / best) * l.tileHeight;
    return l;
}

Result<uint32_t> PlaneLayout::sliceSplats(uint32_t nSplats, uint32_t nBits, uint32_t nRest, bool nHi,
                                          uint32_t nChunk, uint32_t nColumns) {
    if (make(nSplats, nBits, nRest, nHi, nChunk, nColumns)) {
        return nSplats;
    }
    // Chunk counts, from the most that could fit down: bookkeeping on sizes.
    for (uint32_t chunks = (nSplats + nChunk - 1) / nChunk; chunks > 0; --chunks) {
        const uint64_t n = uint64_t{chunks} * nChunk;
        if (n < nSplats && make(static_cast<uint32_t>(n), nBits, nRest, nHi, nChunk, nColumns)) {
            return static_cast<uint32_t>(n);
        }
    }
    return Error::make(ErrorCode::InvalidArgument, "video planes: not even a chunk of {} splats fits a picture",
                       nChunk);
}

Result<scene::GpuSplats> PlanesCodec::allocate(gpu::Device& device, uint32_t count, uint32_t restPerColour,
                                               const char* label) {
    scene::GpuSplats out;
    out.source = label;
    out.count = count;
    out.declared = count;
    out.restPerColour = restPerColour;
    out.shWords = restPerColour == 0 ? 1 : (restPerColour * 3 + 1) / 2;
    auto p = deviceBuffer(device, uint64_t{count} * 16, 16, label);
    if (!p) return std::move(p).error();
    auto s = deviceBuffer(device, uint64_t{count} * 16, 4, label);
    if (!s) return std::move(s).error();
    auto h = deviceBuffer(device, uint64_t{count} * out.shWords * 4, 4, label);
    if (!h) return std::move(h).error();
    out.positions = std::move(*p);
    out.shape = std::move(*s);
    out.sh = std::move(*h);
    return out;
}

Result<PlanesCodec> PlanesCodec::create(gpu::ShaderLibrary& library) {
    PlanesCodec c;
    c.device_ = &library.device();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    c.sort_ = std::move(*sort);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    const char* enc = "athenea/scene/video_planes_encode";
    const char* dec = "athenea/scene/video_planes_decode";
    ATHENEA_TRY(make(c.partial_, enc, "videoPlanesRangePartial"));
    ATHENEA_TRY(make(c.finish_, enc, "videoPlanesRangeFinish"));
    ATHENEA_TRY(make(c.widen_, enc, "videoPlanesRangeWiden"));
    ATHENEA_TRY(make(c.morton_, enc, "videoPlanesMorton"));
    ATHENEA_TRY(make(c.encode_, enc, "videoPlanesEncode"));
    ATHENEA_TRY(make(c.high_, enc, "videoPlanesHigh"));
    ATHENEA_TRY(make(c.decode_, dec, "videoPlanesDecode"));
    ATHENEA_TRY(make(c.compare_, dec, "videoPlanesCompare"));
    return c;
}

Result<void> PlanesCodec::ranges(const scene::GpuSplats& cloud, const PlaneLayout& layout, uint32_t count,
                                 gpu::Buffer& ranges) {
    gpu::Device& device = *device_;
    const uint32_t n = count;
    if (cloud.count < n) {
        return Error::make(ErrorCode::InvalidArgument, "video planes: a cloud of {} splats for a frame of {}",
                           cloud.count, n);
    }
    const uint32_t runs = (n + kRun - 1) / kRun;
    ATHENEA_TRY(ensure(device, partialBuffer_, uint64_t{runs} * layout.attributes() * 8, 4, "video.ranges.partial"));
    ATHENEA_TRY(ensure(device, ranges, layout.rangeBytes(), 4, "video.ranges"));
    gpu::CommandBatch batch(device);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["sh"].setBinding(cloud.sh.rhi());
        cursor["partial"].setBinding(partialBuffer_.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setLayout(p, layout);
        p["count"].setData(n);
        p["sourceShWords"].setData(cloud.shWords);
        p["sourceRest"].setData(cloud.restPerColour);
    };
    partial_.dispatch(batch, {runs * layout.attributes(), 1, 1}, bind);
    finish_.dispatch(batch, {layout.attributes(), 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["ranges"].setBinding(ranges.rhi());
    });
    return batch.submit(true);
}

Result<void> PlanesCodec::widen(const PlaneLayout& layout, const gpu::Buffer& frame, gpu::Buffer& into, bool first) {
    gpu::Device& device = *device_;
    ATHENEA_TRY(ensure(device, into, layout.rangeBytes(), 4, "video.ranges.group"));
    gpu::CommandBatch batch(device);
    widen_.dispatch(batch, {layout.attributes(), 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["frameRanges"].setBinding(frame.rhi());
        cursor["ranges"].setBinding(into.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setLayout(p, layout);
        p["first"].setData(uint32_t{first ? 1u : 0u});
    });
    return batch.submit(true);
}

Result<void> PlanesCodec::order(const scene::GpuSplats& cloud, uint32_t count, const gpu::Buffer& ranges,
                                gpu::Buffer& order) {
    gpu::Device& device = *device_;
    const uint32_t n = count;
    ATHENEA_TRY(ensure(device, order, uint64_t{n} * 4, 4, "video.order"));
    if (sortingFor_ < n) {
        for (gpu::Buffer* b : {&sorting_.keysLo, &sorting_.values, &sorting_.scratchKeysLo, &sorting_.scratchValues}) {
            auto made = deviceBuffer(device, uint64_t{n} * 4, 4, "video.sort");
            if (!made) return std::move(made).error();
            *b = std::move(*made);
        }
        sortingFor_ = n;
    }
    gpu::CommandBatch batch(device);
    morton_.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["ranges"].setBinding(ranges.rhi());
        cursor["keys"].setBinding(sorting_.keysLo.rhi());
        cursor["values"].setBinding(sorting_.values.rhi());
        cursor["params"]["count"].setData(n);
    });
    ATHENEA_TRY(sort_.sort(batch, sorting_, n, 30));
    batch.encoder()->copyBuffer(order.rhi(), 0, sorting_.values.rhi(), 0, uint64_t{n} * 4);
    return batch.submit(true);
}

Result<EncodedPlanes> PlanesCodec::encode(const scene::GpuSplats& cloud, const PlaneLayout& layout,
                                          const gpu::Buffer& order, uint32_t first, const gpu::Buffer& ranges) {
    gpu::Device& device = *device_;
    const uint32_t perWord = layout.bits > 8 ? 2u : 4u;
    const uint64_t lumaWords = uint64_t{layout.width} * layout.height / perWord;
    const uint64_t highEntries = uint64_t{layout.chunks()} * 3 * layout.chunkSplats;
    const uint64_t highWords = (highEntries + perWord - 1) / perWord;
    ATHENEA_TRY(ensure(device, mosaic_, lumaWords * 4, 4, "video.mosaic"));
    if (!layout.hiInVideo) {
        ATHENEA_TRY(ensure(device, highBuffer_, highWords * 4, 4, "video.high"));
    }
    {
        gpu::CommandBatch batch(device);
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud.positions.rhi());
            cursor["shape"].setBinding(cloud.shape.rhi());
            cursor["sh"].setBinding(cloud.sh.rhi());
            cursor["order"].setBinding(order.rhi());
            cursor["ranges"].setBinding(ranges.rhi());
            rhi::ShaderCursor p = cursor["params"];
            setLayout(p, layout);
            p["first"].setData(first);
            p["sourceShWords"].setData(cloud.shWords);
            p["sourceRest"].setData(cloud.restPerColour);
        };
        encode_.dispatch(batch, {static_cast<uint32_t>(lumaWords), 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["mosaic"].setBinding(mosaic_.rhi());
        });
        if (!layout.hiInVideo) {
            high_.dispatch(batch, {static_cast<uint32_t>(highWords), 1, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor);
                cursor["high"].setBinding(highBuffer_.rhi());
            });
        }
        ATHENEA_TRY(batch.submit(true));
    }
    EncodedPlanes out;
    out.luma.resize(layout.lumaBytes());
    ATHENEA_TRY(mosaic_.read(device, 0, out.luma.size(), out.luma.data()));
    out.ranges.resize(size_t{layout.attributes()} * 2);
    ATHENEA_TRY(ranges.read(device, 0, out.ranges.size() * sizeof(float), out.ranges.data()));
    if (!layout.hiInVideo) {
        out.high.resize(highEntries * layout.sampleBytes());
        ATHENEA_TRY(highBuffer_.read(device, 0, out.high.size(), out.high.data()));
    }
    return out;
}

Result<void> PlanesCodec::decode(gpu::CommandBatch& batch, const PlaneLayout& layout, rhi::ITexture* luma,
                                 rhi::IBuffer* page, uint64_t rangesOffset, uint64_t highOffset, uint32_t first,
                                 uint32_t count, scene::GpuSplats& store, uint64_t at, uint32_t keep) {
    if (luma == nullptr || page == nullptr || count == 0) {
        return Error(ErrorCode::InvalidArgument, "video planes: a decode with no frame, no side data or no splats");
    }
    if (rangesOffset % 4 != 0 || highOffset % 4 != 0 || first % layout.chunkSplats != 0 ||
        uint64_t{first} + count > layout.splats || rangesOffset > UINT32_MAX || highOffset > UINT32_MAX) {
        return Error::make(ErrorCode::InvalidArgument, "video planes: splats {} + {} of a frame of {} at offsets {} {}",
                           first, count, layout.splats, rangesOffset, highOffset);
    }
    keep = std::min(keep, store.restPerColour);
    decode_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["luma"].setBinding(luma);
        cursor["page"].setBinding(page);
        cursor["outPositions"].setBinding(store.positions.rhi());
        cursor["outShape"].setBinding(store.shape.rhi());
        cursor["outSh"].setBinding(store.sh.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setLayout(p, layout);
        p["first"].setData(first);
        p["pageCount"].setData(count);
        p["at"].setData(static_cast<uint32_t>(at));
        p["keep"].setData(keep);
        p["shWords"].setData(store.shWords);
        p["rangesOffset"].setData(static_cast<uint32_t>(rangesOffset));
        p["hiOffset"].setData(static_cast<uint32_t>(highOffset));
    });
    batch.markDirty();
    return ok();
}

Result<std::vector<AttributeError>> PlanesCodec::compare(const PlaneLayout& layout, const scene::GpuSplats& decoded,
                                                         const scene::GpuSplats& source, const gpu::Buffer& order,
                                                         uint32_t first, const gpu::Buffer& ranges) {
    gpu::Device& device = *device_;
    const uint32_t a = layout.attributes();
    ATHENEA_TRY(ensure(device, errors_, uint64_t{a} * 16, 4, "video.errors"));
    const uint32_t runs = (layout.splats + kCompareRun - 1) / kCompareRun;
    {
        gpu::CommandBatch batch(device);
        batch.encoder()->clearBuffer(errors_.rhi(), 0, uint64_t{a} * 16);
        compare_.dispatch(batch, {runs * a, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["decodedPositions"].setBinding(decoded.positions.rhi());
            cursor["decodedShape"].setBinding(decoded.shape.rhi());
            cursor["decodedSh"].setBinding(decoded.sh.rhi());
            cursor["sourcePositions"].setBinding(source.positions.rhi());
            cursor["sourceShape"].setBinding(source.shape.rhi());
            cursor["sourceSh"].setBinding(source.sh.rhi());
            cursor["order"].setBinding(order.rhi());
            cursor["frameRanges"].setBinding(ranges.rhi());
            cursor["errors"].setBinding(errors_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            setLayout(p, layout);
            p["first"].setData(first);
            p["shWords"].setData(decoded.shWords);
            p["sourceShWords"].setData(source.shWords);
            p["sourceRest"].setData(source.restPerColour);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::vector<uint32_t> words(size_t{a} * 4);
    ATHENEA_TRY(errors_.read(device, 0, words.size() * 4, words.data()));
    std::vector<AttributeError> out(a);
    for (uint32_t k = 0; k < a; ++k) {
        // Counters into the report's units: 1/16 steps, the sum's carries.
        out[k].maxSteps = words[k * 4] / 16.0;
        const double sum = static_cast<double>(words[k * 4 + 1]) + 4294967296.0 * words[k * 4 + 2];
        out[k].meanSteps = sum / 16.0 / layout.splats;
        out[k].far = words[k * 4 + 3];
    }
    return out;
}

}   // namespace athenea::video
