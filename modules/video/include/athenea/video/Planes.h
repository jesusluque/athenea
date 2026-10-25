// Copyright (c) 2026 jesus luque.
//
// A CLOUD AS VIDEO PLANES. A frame's splats, sorted by Morton code, each
// attribute in a luma tile of one mosaic, quantised over the frame's own
// range: what a hardware video codec carries (docs/decisions.md, "Video
// planes"). The layout and the quantisation are shaders/athenea/scene/
// video_planes_common.slang's; this is the host side:
//
//   PlaneLayout   how many planes, how the tiles lie, how big the mosaic is
//   PlanesCodec   the kernels: a cloud into a mosaic and its side data
//                 (encode), a decoded luma texture and that side data back
//                 into a store's splats (decode), and the two clouds measured
//                 against each other (compare)
//
// THE SIDE DATA of a frame: its ranges (two floats an attribute) and, unless
// they are planes (`hiInVideo`), the high parts of its positions -- a
// position is 2 x `bits`, the high part exact and the low one through the
// codec. They are laid chunk by chunk, so a live page carries its own.
#pragma once

#include <cstdint>
#include <vector>
#include <cstddef>

#include <slang-rhi.h>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/RadixSort.h"

namespace athenea::gpu {
class CommandBatch;
class Device;
class ShaderLibrary;
}   // namespace athenea::gpu

namespace athenea::scene {
struct GpuSplats;
}

namespace athenea::video {

struct PlaneLayout {
    static constexpr uint32_t kTileWidth = 512;
    static constexpr uint32_t kBlock = 64;
    static constexpr uint32_t kBlockSplats = kBlock * kBlock;
    /// Splats a band of 64 rows of a tile holds: the smallest chunk.
    static constexpr uint32_t kBandSplats = kBlockSplats * (kTileWidth / kBlock);
    static constexpr uint32_t kAttributesBeforeRest = 14;

    uint32_t splats = 0;
    uint32_t bits = 8;            ///< 8 or 10 a sample
    uint32_t restPerColour = 0;   ///< 0, 3, 8, 15
    bool     hiInVideo = false;
    uint32_t chunkSplats = kBandSplats;
    uint32_t columns = 0;
    uint32_t tileHeight = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    /// POSITIONS OVER BLOCKS: 0, every position over the frame's range;
    /// else a power of two (64 .. chunkSplats), each run of that many splats
    /// of the Morton order over its own box (six halves, rounded outward,
    /// at the head of each chunk's side payload). A run of Morton order is a
    /// small box, so the same bits say much more.
    uint32_t positionBlock = 0;
    /// The position's bits past `bits` (0 .. bits): the high part, exact,
    /// beside the video (a byte an axis); 0 sends no high part at all. A
    /// position is `bits + positionHighBits` bits. `bits` unless asked.
    uint32_t positionHighBits = 8;
    /// COLOUR IN CHROMA: each colour quantity (the base colour, each
    /// harmonic basis) as YCoCg -- Y a luma plane of its own, Co and Cg the
    /// Cb and Cr of that plane's tile, one a 2 x 2 block of four consecutive
    /// splats (video_planes_common.slang). 8 bits only.
    bool     chroma = false;
    /// ADAPTIVE PRECISION: > 0, each block carries its own high bits (a byte
    /// after the chunk's boxes) -- none where its box is no wider than this
    /// (metres), else enough that a low-part step is no wider than this
    /// over 255; at most `positionHighBits`, which sizes the storage.
    float    positionTarget = 0.0F;
    /// THE GEOMETRY LAYER: position, scale and rotation not in planes but
    /// exact beside the video (video_geometry.slang); the planes are the
    /// opacity and the colours. Sixteen byte planes a chunk.
    bool     geometry = false;
    static constexpr uint32_t kGeometryPlanes = 16;
    uint32_t geometryPositionBits = 16;   ///< 10-16 a position axis
    uint32_t geometryScaleBits = 10;      ///< 6-10 a log scale
    uint32_t geometryRotationBits = 10;   ///< 7-10 a rotation component (the store's word keeps 10)
    [[nodiscard]] uint64_t geometryChunkBytes() const noexcept {
        return geometry ? uint64_t{kGeometryPlanes} * chunkSplats : 0;
    }
    /// The encode's block chroma: 0 the plain mean of its splats, 1 weighed
    /// by opacity x the area of their two largest axes.
    uint32_t chromaWeight = 0;

    [[nodiscard]] uint32_t attributes() const noexcept { return kAttributesBeforeRest + 3 * restPerColour; }
    [[nodiscard]] uint32_t planes() const noexcept {
        if (geometry) {
            return 1u + (chroma ? 1u + restPerColour : 3u + 3u * restPerColour);
        }
        return (hiInVideo ? 6u : 3u) + (chroma ? 8u + 1u + restPerColour : attributes() - 3u);
    }
    /// The second plane's bytes (NV12: a (Cb, Cr) pair a 2 x 2 block): only
    /// with `chroma`.
    [[nodiscard]] uint64_t chromaBytes() const noexcept { return chroma ? uint64_t{width} * (height / 2) : 0; }
    [[nodiscard]] uint32_t chunks() const noexcept { return (splats + chunkSplats - 1) / chunkSplats; }
    [[nodiscard]] uint32_t sampleBytes() const noexcept { return bits > 8 ? 2u : 1u; }
    [[nodiscard]] uint64_t lumaBytes() const noexcept { return uint64_t{width} * height * sampleBytes(); }
    /// The side payload of one chunk: its blocks' boxes (six halves each,
    /// with `positionBlock`), then three axes of `chunkSplats` high parts
    /// (none when they are planes or `positionHighBits` is 0).
    [[nodiscard]] uint64_t blockBytes() const noexcept {
        return positionBlock == 0 ? 0 : uint64_t{chunkSplats / positionBlock} * 12;
    }
    [[nodiscard]] uint64_t bitsBytes() const noexcept {
        return positionTarget > 0.0F && positionBlock != 0 ? (uint64_t{chunkSplats / positionBlock} + 3) / 4 * 4 : 0;
    }
    [[nodiscard]] uint64_t highChunkBytes() const noexcept {
        return blockBytes() + bitsBytes() +
               (hiInVideo || positionHighBits == 0 ? 0 : uint64_t{3} * chunkSplats * sampleBytes());
    }
    [[nodiscard]] uint64_t rangeBytes() const noexcept { return uint64_t{attributes()} * 2 * sizeof(float); }

    /// THE LARGEST PICTURE the hardware encoders take: 8192 x 8192 in area
    /// (VideoToolbox refused 10240 x 9600), 16384 a side.
    static constexpr uint64_t kMostArea = uint64_t{8192} * 8192;
    static constexpr uint32_t kMostSide = 16384;

    /// A layout for `splats`: `columns` tiles a row (0: the squarest mosaic
    /// within kMostSide and kMostArea). `chunkSplats` a multiple of
    /// kBandSplats.
    /// `tileSplats`: tiles sized for that many (0: for `splats`), so every
    /// picture of a stream is the same size.
    /// `mostWidth` x `mostHeight`: a smaller picture than the encoders take,
    /// for a decoder that takes less (0: kMostSide).
    [[nodiscard]] static Result<PlaneLayout> make(uint32_t splats, uint32_t bits, uint32_t restPerColour,
                                                  bool hiInVideo, uint32_t chunkSplats, uint32_t columns,
                                                  uint32_t tileSplats = 0, uint32_t mostWidth = 0,
                                                  uint32_t mostHeight = 0, bool chroma = false,
                                                  bool geometry = false);
    /// A frame too big for one picture goes as SLICES, each a picture of its
    /// own holding a run of the frame's Morton order: the most splats one
    /// takes (a multiple of `chunkSplats`), `splats` when the frame fits.
    [[nodiscard]] static Result<uint32_t> sliceSplats(uint32_t splats, uint32_t bits, uint32_t restPerColour,
                                                      bool hiInVideo, uint32_t chunkSplats, uint32_t columns,
                                                      uint32_t mostWidth = 0, uint32_t mostHeight = 0,
                                                      bool chroma = false, bool geometry = false);
};

/// One frame, encoded: what the codec takes and what goes beside it.
struct EncodedPlanes {
    std::vector<std::byte> luma;     ///< width x height samples, top row first
    std::vector<std::byte> chroma;   ///< with `chroma`: NV12's second plane, (Cb, Cr) a 2 x 2 block; else empty
    std::vector<float>     ranges;   ///< attributes x 2
    std::vector<std::byte> high;     ///< chunks x highChunkBytes (empty when hiInVideo)
};

/// Where a picture's geometry layer stands in its sequence.
struct GeometryFrame {
    uint32_t frame = 0;      ///< the source frame, from 0
    uint32_t interval = 0;   ///< frames from one key to the next (0: every frame a key)
    bool     stagger = false;///< a key refreshes blocks of 256 in turn
    uint32_t order = 1;      ///< the prediction between keys: 1 the frame before, 2 its motion carried on
};

/// How far a decoded cloud is from its source, an attribute: steps of its
/// own quantisation (positions: of the 2 x bits one).
struct AttributeError {
    double   maxSteps = 0.0;
    double   meanSteps = 0.0;
    uint64_t far = 0;   ///< positions: splats whose high part is not the source's
};

class PlanesCodec {
public:
    [[nodiscard]] static Result<PlanesCodec> create(gpu::ShaderLibrary& library);

    /// The ranges of the frame's `count` splats into `ranges` (two floats an
    /// attribute of `layout`'s), on the device.
    [[nodiscard]] Result<void> ranges(const scene::GpuSplats& cloud, const PlaneLayout& layout, uint32_t count,
                                      gpu::Buffer& ranges);
    /// A group of pictures' ranges in `into`: `frame`'s when `first`, else
    /// their union with what `into` holds -- so a splat that stays put keeps
    /// its samples from frame to frame and the codec can predict them.
    [[nodiscard]] Result<void> widen(const PlaneLayout& layout, const gpu::Buffer& frame, gpu::Buffer& into,
                                     bool first);
    /// The Morton order of the frame's `count` splats over the position
    /// ranges of `ranges` into `order`: a permutation, Morton rank -> splat.
    [[nodiscard]] Result<void> order(const scene::GpuSplats& cloud, uint32_t count, const gpu::Buffer& ranges,
                                     gpu::Buffer& order);
    /// The mosaic and the side data of `cloud` in `order`, from Morton rank
    /// `first` (a slice), read back for the codec and the file.
    /// `heldBoxes`: the blocks' boxes to use (holdBoxes' union, laid as the
    /// whole frame's side payload) rather than this frame's own.
    [[nodiscard]] Result<EncodedPlanes> encode(const scene::GpuSplats& cloud, const PlaneLayout& layout,
                                               const gpu::Buffer& order, uint32_t first, const gpu::Buffer& ranges,
                                               const gpu::Buffer* heldBoxes = nullptr);
    /// BOXES HELD OVER FRAMES: the blocks' boxes of the whole frame
    /// (`frame`: its layout over all its splats, `positionBlock` set) in
    /// `order`, into `held` laid as the frame's side payload -- this frame's
    /// (`firstFrame`) or their union with what `held` has. A stable order
    /// keeps a block's gaussians, so the union is the room they move in, and
    /// a box that holds still keeps its steps still.
    [[nodiscard]] Result<void> holdBoxes(const scene::GpuSplats& cloud, const PlaneLayout& frame,
                                         const gpu::Buffer& order, gpu::Buffer& held, bool firstFrame);

    /// Frame splats `first` .. `first + count` into `store` from splat `at`:
    /// `luma` the decoded frame's luma plane (R8Unorm, or R16Unorm for 10
    /// bit), `page` the side data (ranges at `rangesOffset`, high parts at
    /// `highOffset`, both bytes on a 4-byte boundary). `keep` rest
    /// coefficients a colour are written, zero past the planes' own.
    /// `chromaPlane`: the decoded frame's second plane (RG8Unorm), read with
    /// `layout.chroma`; may be null otherwise.
    /// `geometryState`, `frameFirst`: with `layout.geometry`, the layer's
    /// state (geometryApply wrote it) and the picture's first splat in its
    /// frame.
    [[nodiscard]] Result<void> decode(gpu::CommandBatch& batch, const PlaneLayout& layout, rhi::ITexture* luma,
                                      rhi::ITexture* chromaPlane,
                                      rhi::IBuffer* page, uint64_t rangesOffset, uint64_t highOffset, uint32_t first,
                                      uint32_t count, scene::GpuSplats& store, uint64_t at, uint32_t keep,
                                      const gpu::Buffer* geometryState = nullptr, uint32_t frameFirst = 0);

    /// THE GEOMETRY LAYER, encode: the picture's gaussians quantised into
    /// `now` (four words a frame splat, at Morton rank `first` + i of the
    /// frame) and packed -- absolute, or against `before` -- into 16 byte
    /// planes a chunk, read back. `quantiseOnly`: `now` alone, nothing packed.
    [[nodiscard]] Result<std::vector<std::byte>> geometryEncode(const scene::GpuSplats& cloud,
                                                                const PlaneLayout& layout, const gpu::Buffer& order,
                                                                uint32_t first, const gpu::Buffer& ranges,
                                                                const GeometryFrame& at, gpu::Buffer& now,
                                                                const gpu::Buffer& before, bool quantiseOnly = false,
                                                                const gpu::Buffer* before2 = nullptr);
    /// THE GEOMETRY LAYER, decode: a page's planes (at `planesOffset` bytes
    /// in `page`) onto `state` -- splats `pageFirst` .. + `count` of the
    /// picture whose first splat in its frame is `frameFirst`. `state` holds
    /// two frames' words: this one's, and in its second half the one before
    /// (what order 2 predicts from) -- eight words a frame splat.
    [[nodiscard]] Result<void> geometryApply(gpu::CommandBatch& batch, const PlaneLayout& layout, rhi::IBuffer* page,
                                             uint64_t planesOffset, const GeometryFrame& at, uint32_t frameFirst,
                                             uint32_t pageFirst, uint32_t count, gpu::Buffer& state);

    /// A slice decoded from the planes (`decoded`, in Morton order from rank
    /// `first`) against its source through `order`, attribute by attribute.
    /// `page` and `highOffset`: the slice's side data as the decode read
    /// it, whose high parts (and blocks' boxes) the position check takes.
    [[nodiscard]] Result<std::vector<AttributeError>> compare(const PlaneLayout& layout,
                                                              const scene::GpuSplats& decoded,
                                                              const scene::GpuSplats& source,
                                                              const gpu::Buffer& order, uint32_t first,
                                                              const gpu::Buffer& ranges, const gpu::Buffer& page,
                                                              uint64_t highOffset);

    /// A store of `count` packed splats with `restPerColour` rest
    /// coefficients a colour (positions, shape, harmonics only).
    [[nodiscard]] static Result<scene::GpuSplats> allocate(gpu::Device& device, uint32_t count,
                                                           uint32_t restPerColour, const char* label);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel partial_, finish_, widen_, morton_, encode_, high_, blocks_, chroma_, boxUnion_, blockBits_;
    gpu::ComputeKernel decode_, compare_, geometryQuantise_, geometryPack_, geometryApply_;
    gpu::RadixSort     sort_;
    gpu::SortBuffers   sorting_;
    uint32_t           sortingFor_ = 0;
    gpu::Buffer        partialBuffer_, mosaic_, chromaMosaic_, highBuffer_, boxBuffer_, errors_, geometryPlanes_;
};

}   // namespace athenea::video
