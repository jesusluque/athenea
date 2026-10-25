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
    p["positionBlock"].setData(l.positionBlock);
    p["positionHighBits"].setData(l.positionHighBits);
    p["highChunkBytes"].setData(static_cast<uint32_t>(l.highChunkBytes()));
    p["chroma"].setData(uint32_t{l.chroma ? 1u : 0u});
    p["chromaWeight"].setData(l.chromaWeight);
    p["positionTarget"].setData(l.positionTarget);
    p["geometry"].setData(uint32_t{l.geometry ? 1u : 0u});
    p["geometryPositionBits"].setData(l.geometryPositionBits);
    p["geometryScaleBits"].setData(l.geometryScaleBits);
}

}   // namespace

Result<PlaneLayout> PlaneLayout::make(uint32_t nSplats, uint32_t nBits, uint32_t nRest, bool nHi,
                                      uint32_t nChunk, uint32_t nColumns, uint32_t tileSplats, uint32_t mostWidth,
                                      uint32_t mostHeight, bool nChroma, bool nGeometry) {
    const uint64_t mostW = mostWidth != 0 ? std::min(mostWidth, kMostSide) : kMostSide;
    const uint64_t mostH = mostHeight != 0 ? std::min(mostHeight, kMostSide) : kMostSide;
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
    l.positionHighBits = nBits;
    l.chroma = nChroma;
    l.geometry = nGeometry;
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
        if (w > mostW || h > mostH || w * h > kMostArea) {
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
                           "video planes: {} planes of {} x {} do not fit a mosaic of at most {} x {} and {} "
                           "pixels{}",
                           planes, kTileWidth, l.tileHeight, mostW, mostH, kMostArea,
                           nColumns != 0 ? " with the nColumns asked for" : "");
    }
    l.columns = best;
    l.width = best * kTileWidth;
    l.height = ((planes + best - 1) / best) * l.tileHeight;
    return l;
}

Result<uint32_t> PlaneLayout::sliceSplats(uint32_t nSplats, uint32_t nBits, uint32_t nRest, bool nHi,
                                          uint32_t nChunk, uint32_t nColumns, uint32_t mostWidth,
                                          uint32_t mostHeight, bool nChroma, bool nGeometry) {
    if (make(nSplats, nBits, nRest, nHi, nChunk, nColumns, 0, mostWidth, mostHeight, nChroma, nGeometry)) {
        return nSplats;
    }
    // Chunk counts, from the most that could fit down: bookkeeping on sizes.
    for (uint32_t chunks = (nSplats + nChunk - 1) / nChunk; chunks > 0; --chunks) {
        const uint64_t n = uint64_t{chunks} * nChunk;
        if (n < nSplats &&
            make(static_cast<uint32_t>(n), nBits, nRest, nHi, nChunk, nColumns, 0, mostWidth, mostHeight, nChroma,
                 nGeometry)) {
            return static_cast<uint32_t>(n);
        }
    }
    return Error::make(ErrorCode::InvalidArgument, "video planes: not even a chunk of {} splats fits a picture{}",
                       nChunk, mostWidth != 0 || mostHeight != 0 ? " of the size asked for" : "");
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
    ATHENEA_TRY(make(c.blocks_, enc, "videoPlanesBlocks"));
    ATHENEA_TRY(make(c.chroma_, enc, "videoPlanesChroma"));
    ATHENEA_TRY(make(c.boxUnion_, enc, "videoPlanesBoxUnion"));
    ATHENEA_TRY(make(c.blockBits_, enc, "videoPlanesBlockBits"));
    const char* geo = "athenea/scene/video_geometry";
    ATHENEA_TRY(make(c.geometryQuantise_, geo, "videoGeometryQuantise"));
    ATHENEA_TRY(make(c.geometryHold_, geo, "videoGeometryHold"));
    ATHENEA_TRY(make(c.geometryPack_, geo, "videoGeometryPack"));
    ATHENEA_TRY(make(c.geometryApply_, geo, "videoGeometryApply"));
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
                                          const gpu::Buffer& order, uint32_t first, const gpu::Buffer& ranges,
                                          const gpu::Buffer* heldBoxes) {
    gpu::Device& device = *device_;
    const uint32_t perWord = layout.bits > 8 ? 2u : 4u;
    const uint64_t lumaWords = uint64_t{layout.width} * layout.height / perWord;
    const uint64_t sideBytes = uint64_t{layout.chunks()} * layout.highChunkBytes();
    const uint64_t sideWords = sideBytes / 4;
    // The high parts' entries (one a thread's word), when there are any.
    const bool highs = !layout.hiInVideo && layout.positionHighBits != 0;
    const uint64_t highEntries = highs ? uint64_t{layout.chunks()} * 3 * layout.chunkSplats : 0;
    const uint64_t highWords = (highEntries + perWord - 1) / perWord;
    const uint32_t blocks =
        layout.positionBlock == 0 ? 0 : layout.chunks() * (layout.chunkSplats / layout.positionBlock);
    ATHENEA_TRY(ensure(device, mosaic_, lumaWords * 4, 4, "video.mosaic"));
    if (layout.chroma && layout.bits != 8) {
        return Error(ErrorCode::InvalidArgument, "video planes: colour in chroma is 8 bits only");
    }
    const uint64_t chromaWords = layout.chromaBytes() / 4;
    if (chromaWords != 0) {
        ATHENEA_TRY(ensure(device, chromaMosaic_, chromaWords * 4, 4, "video.chroma"));
    }
    ATHENEA_TRY(ensure(device, highBuffer_, std::max<uint64_t>(sideWords, 1) * 4, 4, "video.high"));
    {
        gpu::CommandBatch batch(device);
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud.positions.rhi());
            cursor["shape"].setBinding(cloud.shape.rhi());
            cursor["sh"].setBinding(cloud.sh.rhi());
            cursor["order"].setBinding(order.rhi());
            cursor["ranges"].setBinding(ranges.rhi());
            cursor["high"].setBinding(highBuffer_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            setLayout(p, layout);
            p["first"].setData(first);
            p["sourceShWords"].setData(cloud.shWords);
            p["sourceRest"].setData(cloud.restPerColour);
        };
        // The blocks' boxes first: the mosaic and the high parts are taken
        // over them.
        // The side payload from zeros: the adaptive bits are or-ed in.
        batch.encoder()->clearBuffer(highBuffer_.rhi(), 0, std::max<uint64_t>(sideWords, 1) * 4);
        batch.markDirty();
        if (blocks != 0 && heldBoxes != nullptr) {
            // The held boxes of this slice's chunks, the frame's from chunk
            // first / chunkSplats on.
            const uint64_t chunkBytes = layout.highChunkBytes();
            const uint64_t from = uint64_t{first / layout.chunkSplats} * chunkBytes;
            for (uint32_t c = 0; c < layout.chunks(); ++c) {
                batch.encoder()->copyBuffer(highBuffer_.rhi(), c * chunkBytes, heldBoxes->rhi(), from + c * chunkBytes,
                                            layout.blockBytes());
            }
            batch.markDirty();
        } else if (blocks != 0) {
            blocks_.dispatch(batch, {blocks, 1, 1}, bind);
        }
        if (blocks != 0 && layout.positionTarget > 0.0F) {
            blockBits_.dispatch(batch, {blocks, 1, 1}, bind);
        }
        encode_.dispatch(batch, {static_cast<uint32_t>(lumaWords), 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor);
            cursor["mosaic"].setBinding(mosaic_.rhi());
        });
        if (chromaWords != 0) {
            chroma_.dispatch(batch, {static_cast<uint32_t>(chromaWords), 1, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor);
                cursor["chromaMosaic"].setBinding(chromaMosaic_.rhi());
            });
        }
        if (highWords != 0) {
            high_.dispatch(batch, {static_cast<uint32_t>(highWords), 1, 1}, bind);
        }
        ATHENEA_TRY(batch.submit(true));
    }
    EncodedPlanes out;
    out.luma.resize(layout.lumaBytes());
    ATHENEA_TRY(mosaic_.read(device, 0, out.luma.size(), out.luma.data()));
    if (chromaWords != 0) {
        out.chroma.resize(layout.chromaBytes());
        ATHENEA_TRY(chromaMosaic_.read(device, 0, out.chroma.size(), out.chroma.data()));
    }
    out.ranges.resize(size_t{layout.attributes()} * 2);
    ATHENEA_TRY(ranges.read(device, 0, out.ranges.size() * sizeof(float), out.ranges.data()));
    if (sideBytes != 0) {
        out.high.resize(sideBytes);
        ATHENEA_TRY(highBuffer_.read(device, 0, out.high.size(), out.high.data()));
    }
    return out;
}

Result<std::vector<std::byte>> PlanesCodec::geometryEncode(const scene::GpuSplats& cloud, const PlaneLayout& layout,
                                                           const gpu::Buffer& order, uint32_t first,
                                                           const gpu::Buffer& ranges, const GeometryFrame& at,
                                                           gpu::Buffer& now, const gpu::Buffer& before,
                                                           bool quantiseOnly, const gpu::Buffer* before2) {
    gpu::Device& device = *device_;
    const uint32_t n = layout.splats;
    const uint64_t planeWords = uint64_t{layout.chunks()} * layout.geometryChunkBytes() / 4;
    ATHENEA_TRY(ensure(device, geometryPlanes_, std::max<uint64_t>(planeWords, 1) * 4, 4, "video.geometry"));
    gpu::CommandBatch batch(device);
    const auto bind = [&](rhi::ShaderCursor c) {
        c["positions"].setBinding(cloud.positions.rhi());
        c["shape"].setBinding(cloud.shape.rhi());
        c["order"].setBinding(order.rhi());
        c["ranges"].setBinding(ranges.rhi());
        c["now"].setBinding(now.rhi());
        c["before"].setBinding(before.rhi());
        c["before2"].setBinding(before2 != nullptr ? before2->rhi() : before.rhi());
        c["planes"].setBinding(geometryPlanes_.rhi());
        c["page"].setBinding(now.rhi());
        c["state"].setBinding(now.rhi());
        rhi::ShaderCursor p = c["params"];
        p["count"].setData(n);
        p["first"].setData(first);
        p["chunkSplats"].setData(layout.chunkSplats);
        p["frame"].setData(at.frame);
        p["interval"].setData(at.interval);
        p["stagger"].setData(uint32_t{at.stagger ? 1u : 0u});
        p["positionBits"].setData(layout.geometryPositionBits);
        p["scaleBits"].setData(layout.geometryScaleBits);
        p["rotationBits"].setData(layout.geometryRotationBits);
        p["order"].setData(before2 != nullptr ? at.order : std::min(at.order, 1u));
        p["holdPosition"].setData(at.holdPosition);
        p["holdScale"].setData(at.holdScale);
        p["holdRotation"].setData(at.holdRotation);
    };
    if (at.holds() && at.order != 1) {
        return Error(ErrorCode::InvalidArgument, "video planes: a geometry hold wants order 1 prediction");
    }
    geometryQuantise_.dispatch(batch, {n, 1, 1}, bind);
    if (at.holds()) {
        geometryHold_.dispatch(batch, {n, 1, 1}, bind);
    }
    if (!quantiseOnly) {
        geometryPack_.dispatch(batch, {static_cast<uint32_t>(planeWords), 1, 1}, bind);
    }
    ATHENEA_TRY(batch.submit(true));
    std::vector<std::byte> out;
    if (!quantiseOnly) {
        out.resize(planeWords * 4);
        ATHENEA_TRY(geometryPlanes_.read(device, 0, out.size(), out.data()));
    }
    return out;
}

Result<void> PlanesCodec::geometryApply(gpu::CommandBatch& batch, const PlaneLayout& layout, rhi::IBuffer* page,
                                        uint64_t planesOffset, const GeometryFrame& at, uint32_t frameFirst,
                                        uint32_t pageFirst, uint32_t count, gpu::Buffer& state) {
    if (page == nullptr || planesOffset % 4 != 0 || planesOffset > UINT32_MAX || pageFirst % layout.chunkSplats != 0) {
        return Error(ErrorCode::InvalidArgument, "video planes: a geometry layer's planes off a page's chunks");
    }
    geometryApply_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["page"].setBinding(page);
        c["state"].setBinding(state.rhi());
        c["positions"].setBinding(state.rhi());
        c["shape"].setBinding(state.rhi());
        c["order"].setBinding(state.rhi());
        c["ranges"].setBinding(state.rhi());
        c["now"].setBinding(state.rhi());
        c["before"].setBinding(state.rhi());
        c["before2"].setBinding(state.rhi());
        c["planes"].setBinding(state.rhi());
        rhi::ShaderCursor p = c["params"];
        p["count"].setData(count);
        p["chunkSplats"].setData(layout.chunkSplats);
        p["frame"].setData(at.frame);
        p["interval"].setData(at.interval);
        p["stagger"].setData(uint32_t{at.stagger ? 1u : 0u});
        p["planesOffset"].setData(static_cast<uint32_t>(planesOffset));
        p["frameFirst"].setData(frameFirst);
        p["pageFirst"].setData(pageFirst);
        p["positionBits"].setData(layout.geometryPositionBits);
        p["scaleBits"].setData(layout.geometryScaleBits);
        p["rotationBits"].setData(layout.geometryRotationBits);
        p["order"].setData(at.order);
        p["state2At"].setData(static_cast<uint32_t>(state.bytes() / 8));   // words: the second half
    });
    batch.markDirty();
    return ok();
}

Result<void> PlanesCodec::holdBoxes(const scene::GpuSplats& cloud, const PlaneLayout& frame,
                                   const gpu::Buffer& order, gpu::Buffer& held, bool firstFrame) {
    gpu::Device& device = *device_;
    if (frame.positionBlock == 0) {
        return Error(ErrorCode::InvalidArgument, "video planes: boxes held over frames want --position-block");
    }
    const uint64_t sideBytes = uint64_t{frame.chunks()} * frame.highChunkBytes();
    const uint32_t blocks = frame.chunks() * (frame.chunkSplats / frame.positionBlock);
    ATHENEA_TRY(ensure(device, boxBuffer_, sideBytes, 4, "video.boxes"));
    ATHENEA_TRY(ensure(device, held, sideBytes, 4, "video.boxesHeld"));
    gpu::CommandBatch batch(device);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["sh"].setBinding(cloud.sh.rhi());
        cursor["order"].setBinding(order.rhi());
        cursor["high"].setBinding(boxBuffer_.rhi());
        cursor["heldBoxes"].setBinding(held.rhi());
        rhi::ShaderCursor p = cursor["params"];
        setLayout(p, frame);
        p["first"].setData(uint32_t{0});
        p["pad6"].setData(uint32_t{firstFrame ? 1u : 0u});
        p["sourceShWords"].setData(cloud.shWords);
        p["sourceRest"].setData(cloud.restPerColour);
    };
    blocks_.dispatch(batch, {blocks, 1, 1}, bind);
    boxUnion_.dispatch(batch, {blocks * 3, 1, 1}, bind);
    return batch.submit(true);
}

Result<void> PlanesCodec::decode(gpu::CommandBatch& batch, const PlaneLayout& layout, rhi::ITexture* luma,
                                 rhi::ITexture* chromaPlane,
                                 rhi::IBuffer* page, uint64_t rangesOffset, uint64_t highOffset, uint32_t first,
                                 uint32_t count, scene::GpuSplats& store, uint64_t at, uint32_t keep,
                                 const gpu::Buffer* geometryState, uint32_t frameFirst) {
    if (layout.geometry && (geometryState == nullptr || !geometryState->valid())) {
        return Error(ErrorCode::InvalidArgument, "video planes: a geometry layer decoded without its state");
    }
    if (luma == nullptr || page == nullptr || count == 0 || (layout.chroma && chromaPlane == nullptr)) {
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
        cursor["chromaPlane"].setBinding(chromaPlane != nullptr ? chromaPlane : luma);
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
        p["geometryAt"].setData(frameFirst);
        cursor["geometryState"].setBinding(layout.geometry ? geometryState->rhi() : page);
    });
    batch.markDirty();
    return ok();
}

Result<std::vector<AttributeError>> PlanesCodec::compare(const PlaneLayout& layout, const scene::GpuSplats& decoded,
                                                         const scene::GpuSplats& source, const gpu::Buffer& order,
                                                         uint32_t first, const gpu::Buffer& ranges,
                                                         const gpu::Buffer& page, uint64_t highOffset) {
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
            cursor["page"].setBinding(page.rhi());
            cursor["errors"].setBinding(errors_.rhi());
            rhi::ShaderCursor p = cursor["params"];
            setLayout(p, layout);
            p["first"].setData(first);
            p["hiOffset"].setData(static_cast<uint32_t>(highOffset));
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
