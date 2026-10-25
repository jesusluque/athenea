// Copyright (c) 2026 jesus luque.
//
// Level of detail for splat clouds, built and cut on the device. The method --
// Morton-ordered cells, additive moments, a per-cell cut that covers every
// place exactly once -- is written out in shaders/athenea/lod/lod_common.slang.
//
//   LodBuilder     once per cloud: Morton sort, the levels' groups, moments
//                  from the finest merged level up, one Gaussian a group
//   CutSelector    every frame, per instance: which groups of which levels and
//                  which splats to draw, gathered into one cloud the tile
//                  rasteriser draws as it would any other
//
// Only counts cross back to the CPU: one per level while building, and per
// instance per frame one read of the drawn counts (and, streaming, of the
// chunks the view wants).
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::lod {

/// One stored level: a merged Gaussian per octree cell that holds splats.
struct LodLevel {
    uint32_t          level = 0;   ///< octree level: cells of extent / 2^level
    scene::GpuSplats  gaussians;   ///< one per group
    gpu::Buffer       cells;       ///< uint per group: the cell's code at `level`
};

/// A cloud with its levels of detail. The merged levels are always on the
/// device; the cloud's own splats come in chunks -- runs of `chunkSplats` of
/// the Morton order, the last holding the rest -- which need not be. A chunk
/// on the device sits in a slot of the store (slot k: splats from
/// k * chunkSplats); built in memory, chunk c is slot c and all are there.
/// A live stream's cloud (SplatStream.h) has no merged levels: its chunks
/// on the device are drawn as they are, and the others not at all.
struct LodCloud {
    uint32_t              count = 0;         ///< splats in the cloud
    uint32_t              chunkSplats = 0;   ///< splats per chunk
    scene::GpuSplats      splats;            ///< the store: chunks' splats in their slots
    gpu::Buffer           groups;            ///< uint per store splat: its finest-level group
    std::vector<int32_t>  slots;             ///< per chunk: its slot, or -1 when not on the device
    gpu::Buffer           resident;          ///< uint per chunk: 1 when on the device
    std::vector<LodLevel> levels;            ///< coarsest first, consecutive levels; empty only for a live stream
    gpu::Buffer           starts;            ///< uint per finest-level group: its first splat
    /// uint per store splat: which splat of the cloud it was built from --
    /// the Morton order, kept so that what the cloud carries beside its
    /// gaussians can follow them. Empty for a cloud read from a `.athc`.
    gpu::Buffer           order;
    float                 boundsLo[3] = {0, 0, 0};
    float                 extent = 0.0F;     ///< the largest edge of the bounds
    /// Chunks come and go (a StreamingPool's): the cut says which it wants.
    bool                  streamed = false;
    /// Per chunk, its splats, where chunks hold any number up to
    /// `chunkSplats` (a live stream's pages); empty where each holds
    /// `chunkSplats` but the last.
    std::vector<uint32_t> chunkCounts;
    /// Per chunk, what its page said of antialiasing (a live stream's SPZ
    /// pages): -1 nothing (the cloud's own `splats.antialiased`), 0 not, 1
    /// trained with it. Empty for every other cloud.
    std::vector<int8_t>   chunkAntialiased;

    [[nodiscard]] uint32_t chunks() const noexcept { return static_cast<uint32_t>(slots.size()); }
    [[nodiscard]] uint32_t chunkCount(uint32_t chunk) const noexcept {
        return chunk < chunkCounts.size() ? chunkCounts[chunk] : std::min(chunkSplats, count - chunk * chunkSplats);
    }
    [[nodiscard]] uint32_t mergedGaussians() const noexcept {
        uint32_t n = 0;
        for (const LodLevel& l : levels) {
            n += l.gaussians.count;
        }
        return n;
    }
};

struct LodBuildSettings {
    /// The finest stored level is the deepest whose cells hold on average at
    /// least 1 / this fraction of splats each: deeper, a merged level would
    /// be nearly the splats again.
    float maxGroupFraction = 0.5F;
    /// The coarsest stored level.
    uint32_t coarsestLevel = 1;
    /// Splats per chunk: what a stream loads and drops at once.
    uint32_t chunkSplats = uint32_t{1} << 16;
};

class LodBuilder {
public:
    [[nodiscard]] static Result<LodBuilder> create(gpu::ShaderLibrary& library);
    [[nodiscard]] Result<LodCloud> build(const scene::GpuSplats& cloud, const LodBuildSettings& settings = {});

private:
    gpu::Device*       device_ = nullptr;
    gpu::RadixSort     sort_;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel morton_, reorder_, boundaries_, groups_, leafMoments_, mergeMoments_, finalize_;
    /// What a cloud carries beside its gaussians (lod_extras.slang).
    gpu::ComputeKernel extrasReorder_, extrasMerge_;
};

struct LodInstance {
    const LodCloud*      cloud = nullptr;
    render::Mat4         objectToWorld = render::Mat4::identity();
    render::SplatEdit    edit;
    std::optional<float> threshold;   ///< its own, over the call's
};

struct CutStats {
    uint32_t splats = 0;     ///< the cloud's own splats drawn
    uint32_t merged = 0;     ///< merged Gaussians drawn
    uint32_t available = 0;  ///< splats in the cloud
    /// Streamed clouds: per chunk, 0 when this view does not want its splats,
    /// else how much -- the largest projected edge, in 1/16 pixels, of the
    /// cells wanting them -- on the device or not. Empty for a cloud that is
    /// all in memory.
    std::vector<uint32_t> needs;
};

class CutSelector {
public:
    [[nodiscard]] static Result<CutSelector> create(gpu::ShaderLibrary& library);
    CutSelector(CutSelector&&) noexcept;
    CutSelector& operator=(CutSelector&&) noexcept;
    ~CutSelector();

    /// Each instance's cut for this view, as instances the rasteriser draws.
    /// A cell is drawn merged when it projects to at most `threshold` pixels;
    /// 0 draws every splat as it is. A cloud without merged levels (a live
    /// stream's) draws its chunks on the device as they are, in chunk order. The returned clouds belong to this
    /// selector and are rewritten by the next call. `stats`, when given, holds
    /// one entry per instance, in order.
    [[nodiscard]] Result<std::vector<render::SplatInstance>> select(const render::Projection& projection,
                                                                   std::span<const LodInstance> instances,
                                                                   float threshold,
                                                                   std::vector<CutStats>* stats = nullptr);

private:
    CutSelector();
    struct Frame;
    /// The frame's cloud with room for `drawn` gaussians of `lod`'s kind.
    [[nodiscard]] Result<void> ensureCloud(const LodCloud& lod, Frame& frame, uint32_t drawn);
    /// A cloud without merged levels: its chunks on the device copied into
    /// the frame's cloud, in chunk order. Returns the splats copied.
    [[nodiscard]] Result<uint32_t> copyChunks(const LodCloud& lod, Frame& frame,
                                              std::vector<const scene::GpuSplats*>& clouds);
    gpu::Device*                        device_ = nullptr;
    gpu::PrefixSum                      prefix_;
    gpu::ComputeKernel                  cutGroups_, cutFinest_, cutSplats_, chunkNeeds_, gather_, extrasGather_;
    std::vector<std::unique_ptr<Frame>> frames_;
};

/// HOW MUCH A DECIMATION MAY LOSE, and what it did.
struct DecimateSettings {
    /// How far a gaussian's colour or opacity may be from the one that
    /// replaces it, before it counts as one that differs: an absolute
    /// difference of the base colour in the cloud's own space -- sRGB code
    /// values over 255 for a capture, linear light for a cloud that holds it
    /// (`GpuSplats::linear`), where 0.05 is a coarser step in the darks and a
    /// finer one in the brights than it is in sRGB -- and of opacity.
    float colourTolerance = 0.05F;
    /// The share of what a merge stands for -- its reach, weighted by its
    /// density -- that may differ: a few is noise, many is an edge.
    float outliers = 0.05F;
    /// Where the children are discs (a surface), how thick against its width
    /// the Gaussian that replaces them may be.
    float flatTolerance = 0.08F;
    /// How many of its own standard deviations a child's centre may stand
    /// from the Gaussian that replaces it.
    float reach = 1.5F;
};

/// WHAT THE CLOUD CARRIES BESIDE ITS GAUSSIANS, a record of its file at a
/// time, where a decimation is to keep it (`Decimator::addKey`,
/// `addMaterial`). Both may be absent.
struct DecimateCarried {
    /// uint per record: records whose keys differ are never merged -- another
    /// prim, another joint, a sheet beside a solid.
    const gpu::Buffer* keys = nullptr;
    /// float4 per record: metallic, roughness, transmission, compared as a
    /// colour is.
    const gpu::Buffer* material = nullptr;
};

/// What a decimation kept, and what each of it stands for.
struct DecimateResult {
    scene::GpuSplats cloud;
    /// uint2 per kept gaussian: the run of store splats it stands for.
    gpu::Buffer      ranges;
    /// uint per store splat: the record of the file it came from.
    gpu::Buffer      recordOf;
    /// A merge by pairs puts the store in the order of what it kept, so that
    /// each kept gaussian is a run of it: the store's positions and shape in
    /// that order, which `mergeAttribute` weighs by in place of the cloud's.
    /// Empty for the octree's cut, whose runs are the Morton order itself.
    gpu::Buffer      storePositions;
    gpu::Buffer      storeShape;
    /// A merge by pairs: float per kept gaussian, W over its area -- what it
    /// covers, not held at the 0.99 its opacity is (`athenea:splat:coverage`).
    gpu::Buffer      coverage;
    /// A merge by pairs: float per kept gaussian, its hardness p (packing's
    /// hard surfel; 2 a gaussian) -- `athenea:splat:hardness`.
    gpu::Buffer      hardness;
    /// The merge tree (`decimatePairs`' `treeOut`): uint per gaussian, its
    /// parent's index (a root its own), and float, what the merge that made it
    /// cost (a leaf 0). The leaves come first, in depth-first order, so every
    /// node stands for a run of them.
    gpu::Buffer      parent;
    gpu::Buffer      cost;
    /// uint2 per gaussian of a merge tree: its first leaf and how many leaves
    /// it stands for (a leaf: itself and one).
    gpu::Buffer      run;
};

/// FEWER GAUSSIANS BY PAIRS (`Decimator::decimatePairs`, lod_pairs.slang):
/// NanoGS's merges, the cheapest first, until the cloud is `ratio` times
/// smaller or nothing more may merge.
struct PairSettings {
    /// How many times fewer gaussians: 10 keeps a tenth.
    float ratio = 10.0F;
    /// Where above 0, no merge of the flat cloud costs more than this: the
    /// ratio is then the most it goes to, and where every pair left costs
    /// more it stops short of it -- smooth surfaces merged, details kept.
    float maxCost = 0.0F;
    /// Where above 0 (and a camera is given): no merge wider than this many
    /// pixels on its screen -- one that wide cannot hold an edge, and its
    /// tails stand out past the panel as hairs.
    float maxPixels = 0.0F;
    /// A colour difference (sRGB, 0..1, per channel as a distance) that is an
    /// edge: never merged across.
    float edge = 0.15F;
    /// Normals further apart than this are never merged, in degrees.
    float maxAngle = 60.0F;
    float colourWeight = 20.0F;
    float normalWeight = 1.0F;
    float materialWeight = 4.0F;
    float flatWeight = 4.0F;
    /// On a glossy surface (GGX alpha under 0.1), normals further apart than
    /// this times sqrt(alpha) radians -- the lobe's width -- never merge: the
    /// merge would move the highlight. Large: the ratio decides alone.
    float highlightCone = 1.0F;
    /// Transmitting gaussians (glass) merge among themselves; off, they stay
    /// as they are -- a window is a sheet drawn as a layer, and merged it
    /// shimmers.
    bool  mergeGlass = false;
    /// A merge whose plane comes out longer than this against its width (or
    /// than 1.25 times its members' already were) is refused: a chain of
    /// discs along a trim merges into a needle that smears. Off by default:
    /// at 4 the Corvette merged 10x came out worse (26.9 dB against 27.9), the
    /// ratio then taken from merges that cost more.
    float maxAspect = 1000.0F;
    /// With a tree: once nothing more may merge, the roots go on with the
    /// colour edge, the angle, the glass, the aspect and the pixel limit
    /// lifted, so a far view has coarse levels to cut to (the flat cloud is
    /// what stood before).
    bool  relaxTop = true;
    /// A merge of discs whose thinnest axis is under this fraction of its
    /// middle one is written as a surfel, a scale of exactly 0 (when `surfels`).
    float flatTolerance = 0.12F;
    bool  surfels = true;
    /// A MERGE'S OPACITY. 0: its members', area weighted, the merge widened
    /// as far as they overlapped; 1: its mass over that widened area (what
    /// it covered, kept); 2: its mass over the area its moments give, not
    /// widened. Each held under 0.99.
    uint32_t opacityRule = 0;
    /// A FLAT MERGE'S HARDNESS (packing's hard surfel): a merge written as a
    /// surfel is drawn o exp(-(|u|^p + |v|^p) / 2), p from its members'
    /// kurtosis -- an even patch a box, a peaked one the gaussian -- at most
    /// `maxHardness`. It keeps its moments (no widening) and writes p as
    /// `athenea:splat:hardness`, which a reader that does not know it ignores.
    bool  hardness = false;
    float maxHardness = 16.0F;
    /// The camera the cloud is made for: a merge it cannot see -- under half
    /// a pixel -- costs nothing of its shape. `pixelsPerUnit` 0: no camera.
    float eye[3] = {0, 0, 0};
    float pixelsPerUnit = 0.0F;
    /// At most this many passes; each merges at most half of what is left.
    uint32_t passes = 60;
    /// With a tree asked for, merging goes on past `ratio` to this many times
    /// fewer (or until nothing more may merge): the tree's roots.
    float treeRatio = 4096.0F;
};

/// How an attribute merges (`Decimator::mergeAttribute`).
enum class AttributeMerge : uint32_t {
    Mean = 0,    ///< floats, weighted as the moments weigh
    First = 1,   ///< what the key keeps apart: every record agrees
    Bits = 2,    ///< each bit set where half the weight has it
    Rig = 3,     ///< joints and their weights: summed, the heaviest kept, made whole
    Normal = 4,  ///< a direction: the mean, weighted as the moments weigh, made unit again
    Zonal = 5,   ///< transferZonal: each lobe's axis through its own frame to the world, averaged, into the kept frame
    Curvature = 6,   ///< a shape operator in the frame's first two axes, the same way
    /// As a key only (`Decimator::addKey`): an RGB of light (a lamp group's
    /// layer) folded in by its brightness in bands of four times, so that
    /// what a lamp lights is not merged with what it lights four times less
    /// -- a merge would spread one member's light over the rest at night.
    LightBand = 7,
};

struct DecimateStats {
    uint32_t splats = 0;   ///< the cloud's own splats kept
    uint32_t merged = 0;   ///< merged Gaussians standing for the rest
    uint32_t before = 0;   ///< splats in the cloud
    uint32_t surfels = 0;  ///< merged Gaussians written as surfels (by pairs)
    uint32_t passes = 0;   ///< passes a merge by pairs took
};

/// FEWER GAUSSIANS, KEPT WHERE THEY MATTER: a cloud cut once by what merging
/// loses rather than every frame by distance (lod_decimate.slang says how).
/// The levels of detail are the merges; this chooses among them, for good.
/// What comes back is a cloud of positions, shapes and harmonics: what the
/// merged levels carry, and nothing a cloud keeps beside them (materials,
/// ids, transfer, rigs).
class Decimator {
public:
    [[nodiscard]] static Result<Decimator> create(gpu::ShaderLibrary& library);
    /// `lod` must hold every chunk on the device, as a build in memory does;
    /// `origin` is the cloud's (GpuSplats::origin), which says which record
    /// each splat came from.
    [[nodiscard]] Result<DecimateResult> decimate(const LodCloud& lod, const gpu::Buffer& origin,
                                                  const DecimateSettings& settings = {},
                                                  const DecimateCarried& carried = {},
                                                  DecimateStats* stats = nullptr);
    /// THE SAME, BY PAIRS (lod_pairs.slang): a cloud `settings.ratio` times
    /// smaller, merged where merging loses least and never across what
    /// `carried` keeps apart. `lod` only for its store (the splats in Morton
    /// order) and that order; its levels are not read.
    [[nodiscard]] Result<DecimateResult> decimatePairs(const LodCloud& lod, const gpu::Buffer& origin,
                                                       const PairSettings& settings = {},
                                                       const DecimateCarried& carried = {},
                                                       DecimateStats* stats = nullptr,
                                                       DecimateResult* treeOut = nullptr);
    /// A CLOUD PRUNED BY WHAT THE FRAMES SAY (lod_prune.slang). `foldScore`
    /// adds one frame's count (`render::RenderSettings::contribution`, a uint
    /// per store splat, or two -- low, high -- where `wide`, drawn from
    /// `lod.splats`) times `scale` into `score`,
    /// a float per store splat, and clears the count; `scoreHistogram` bins
    /// the scores by log10 (kScoreBins from 1e-12 to 1e6, zero in the first)
    /// for the host to pick a threshold; `prune` is the store of `lod`
    /// without the splats scored under `threshold`, in the same order. Only
    /// what `decimatePairs` and `mergeAttribute` read is carried: the store's
    /// positions, shapes, harmonics, normals and order; its levels are not.
    static constexpr uint32_t kScoreBins = 256;
    [[nodiscard]] static float scoreOfBin(uint32_t bin) noexcept;
    [[nodiscard]] Result<void> foldScore(gpu::Buffer& frameCount, gpu::Buffer& score, uint32_t count, float scale,
                                         bool wide = false);
    [[nodiscard]] Result<std::vector<uint32_t>> scoreHistogram(const gpu::Buffer& score, uint32_t count);
    [[nodiscard]] Result<LodCloud> prune(const LodCloud& lod, const gpu::Buffer& score, float threshold);
    /// Folds an attribute of `records` records, `width` words each, into
    /// `keys` (uint a record, zeros to start): its words, or with
    /// `AttributeMerge::Rig` the joint that carries each record most
    /// (`weights` beside the joints).
    [[nodiscard]] Result<void> addKey(gpu::Buffer& keys, uint32_t records, const gpu::Buffer& values, uint32_t width,
                                      AttributeMerge how, const gpu::Buffer* weights = nullptr);
    /// One float a record into component `component` of `material`.
    [[nodiscard]] Result<void> addMaterial(gpu::Buffer& material, uint32_t records, const gpu::Buffer& values,
                                           uint32_t component);
    /// One attribute of every kept gaussian, from the records it stands for:
    /// `width` words a kept gaussian, and for a rig the weights beside.
    [[nodiscard]] Result<std::pair<gpu::Buffer, gpu::Buffer>> mergeAttribute(const LodCloud& lod,
                                                                             const DecimateResult& kept,
                                                                             const gpu::Buffer& values,
                                                                             uint32_t width, AttributeMerge how,
                                                                             const gpu::Buffer* weights = nullptr);

private:
    gpu::Device*       device_ = nullptr;
    gpu::PrefixSum     prefix_;
    gpu::ComputeKernel errorFinest_, errorLevel_, keepGroups_, keepSplats_, keepShape_, gather_;
    gpu::ComputeKernel recordOf_, gatherRanges_, attributeKey_, attributeMaterial_, mergeAttribute_;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel pairInit_, pairCodes_, pairBest_, pairHistogram_, pairMerge_, pairCompact_, pairRelabel_;
    gpu::ComputeKernel pairCount_, pairScatter_, pairGather_, pairFinalize_;
    gpu::ComputeKernel pruneFold_, pruneHistogram_, pruneKeep_, pruneScatter_;
    gpu::ComputeKernel pairRootSizes_, pairTreeRoots_, pairTreeOffsets_, pairTreeSizes_, pairTreeLayout_;
};

}   // namespace athenea::lod
