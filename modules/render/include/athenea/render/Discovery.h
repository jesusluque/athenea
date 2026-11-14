// Copyright (c) 2026 jesus luque.
//
// A SCENE FOUND A LOOK AT A TIME (shaders/athenea/neural/scene_discover.slang):
// which of a piece's gaussians each look saw first. A look is a frame from a
// camera at a time; what it gives beside its picture is the gaussians it
// saw, and the scene as it is known is those the looks so far have found.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/io/RawSplats.h"
#include "athenea/render/Camera.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

/// What one look did to one piece.
struct LookCount {
    uint32_t found = 0;   ///< gaussians it saw that no look had
    uint32_t again = 0;   ///< gaussians it saw that the scene already held
};

/// What a frame left in the scene (shaders/athenea/neural/scene_frame.slang).
struct FrameLook {
    uint32_t           surfaces = 0;   ///< the frame's pixels that hold a surface
    uint32_t           held = 0;       ///< of those, the ones the scene already had at that depth
    uint32_t           made = 0;       ///< gaussians made: one a pixel the scene did not have
    /// The gaussians as a 3DGS PLY keeps them, `kFrameRecord` floats each.
    std::vector<float> records;
};

/// scene_frame.slang's kFrameRecord: INRIA's 62 floats a gaussian.
inline constexpr uint32_t kFrameRecord = 62;

/// scene_render.slang's kRenderRecord: what a gaussian made by the engine's
/// own render keeps -- its disc, its colour, the way its surface faced and
/// what it is made of (`Discovery::renderEncoding`), and after those the
/// material's row in the frame that made it, its index, whether it is a
/// sheet, and what it lets through a crossing.
inline constexpr uint32_t kRenderRecord = 30;
inline constexpr uint32_t kRenderRecordRow = 20;
inline constexpr uint32_t kRenderRecordIndex = 21;
inline constexpr uint32_t kRenderRecordSheet = 22;
inline constexpr uint32_t kRenderRecordTint = 23;   ///< r g b: what the surface lets through one crossing
/// The width of the coarser gaussians a gaussian was laid over, 0 where over
/// nothing: what `Discovery::cut` reads as its parent's. A record of 26
/// floats (a scene shot before 13 Nov) has none.
inline constexpr uint32_t kRenderRecordParent = 26;
/// And their colour, three floats, as a base colour is kept: what the cut
/// moves a gaussian towards before its parent is drawn in its place. A record
/// of 27 floats has the width alone.
inline constexpr uint32_t kRenderRecordParentColour = 27;

/// The scene as it is known, drawn through the look's camera.
struct KnownScene {
    const gpu::Buffer* colour = nullptr;   ///< float4 a pixel
    const gpu::Buffer* depth = nullptr;    ///< view z where the scene is all but opaque
    const gpu::Buffer* first = nullptr;    ///< view z where it begins (a pane's own depth); null: `depth`
    const gpu::Buffer* size = nullptr;     ///< the scene drawn by `Discovery::sizes`; null: nothing is refined
    /// The scene's sizes and depth before the look took anything out: what the
    /// gaussians it makes are laid over, and so their parents. Null: none.
    const gpu::Buffer* beforeSize = nullptr;
    const gpu::Buffer* beforeDepth = nullptr;
    const gpu::Buffer* beforeColour = nullptr;   ///< and its colour there; null: the parent's colour is left black
};

/// How a render's look is taken (scene_render.slang).
struct RenderLookSettings {
    float    tolerance = 0.06F;     ///< a surface is held where the scene's depth is within this share of its own
    float    spread = 1.0F;         ///< a disc's standard deviation, in footprints at its own depth
    float    repaint = 0.0F;        ///< scene_frame.slang's: a held surface off the frame's colour by this share is made again
    float    refine = 0.0F;         ///< and one held by gaussians this many times as wide as the look's
    uint32_t footprint = 1;         ///< pixels a side of the cell one gaussian is made of
    float    glassOpacity = 0.05F;  ///< the least a pane covers of what is behind it, however clear
    float    turned = 0.5F;         ///< the least a disc's facing is taken as: it is stretched by no more than 1 / this
    /// A surface that lets the eye through made as a pane and what stands
    /// behind it; false, as a disc carrying its pixel like any other -- what
    /// a look at coarse cells takes, which a finer one then replaces whole.
    bool     panes = true;
    /// A level of a frame's pyramid that has one above it: a gaussian laid
    /// over nothing the scene held takes cells twice as wide for its parent.
    bool     pyramid = false;
};

/// What a render left in the scene.
struct RenderLook {
    uint32_t    surfaces = 0;   ///< cells that hold a surface
    uint32_t    held = 0;       ///< of those, the ones the scene already had
    uint32_t    panes = 0;      ///< cells whose surface lets the eye through
    uint32_t    behind = 0;     ///< gaussians made of what stands behind one
    uint32_t    made = 0;       ///< gaussians made, in all
    gpu::Buffer records;        ///< `made` records of kRenderRecord floats, on the device
};

class Discovery {
public:
    [[nodiscard]] static Result<Discovery> create(gpu::ShaderLibrary& library);

    /// A piece's record, `count` gaussians: none seen, or what a file held
    /// (a uint a gaussian, the look that first saw it, 0 for none).
    [[nodiscard]] Result<gpu::Buffer> record(uint32_t count, std::span<const uint32_t> held = {});

    /// One look's mark on one piece: `contribution` is what the blend summed
    /// a slot for the look's frame (RenderSettings::contribution, mode 1),
    /// `base` the piece's first slot in it. A gaussian that gave at least
    /// `threshold` of a pixel is seen; one no look had seen is stamped `stamp`.
    [[nodiscard]] Result<LookCount> mark(const gpu::Buffer& contribution, uint32_t base, gpu::Buffer& seen,
                                         uint32_t count, uint32_t stamp, float threshold);

    /// How many of a piece's gaussians the looks up to `upTo` found.
    [[nodiscard]] Result<uint32_t> known(const gpu::Buffer& seen, uint32_t count, uint32_t upTo);

    /// The piece as it is known up to look `upTo`: the same cloud with the
    /// gaussians no look found given no opacity. It shares every buffer of
    /// `cloud` but its positions.
    [[nodiscard]] Result<scene::GpuSplats> masked(const scene::GpuSplats& cloud, const gpu::Buffer& seen,
                                                  uint32_t upTo);

    /// GAUSSIANS MADE FROM A FRAME: a gaussian a pixel of `colour` (float4,
    /// linear, bottom row first) whose `depth` (a float, view z; 0 where
    /// nothing is) the scene does not already hold -- `knownColour` and
    /// `knownDepth` being the scene as it is known drawn through the same
    /// `projection`, or null where nothing is known yet. A pixel is held
    /// where the two depths are within `tolerance` of the frame's own; a
    /// gaussian is a disc in the pixel's surface, `spread` pixels wide.
    [[nodiscard]] Result<FrameLook> fromFrame(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                              uint32_t height, const Projection& projection,
                                              const gpu::Buffer* knownColour, const gpu::Buffer* knownDepth,
                                              float tolerance = 0.06F, float spread = 1.0F, float repaint = 0.0F,
                                              const gpu::Buffer* knownSize = nullptr, float refine = 0.0F);

    /// GAUSSIANS MADE BY THE ENGINE'S OWN RENDER (scene_render.slang): the
    /// frame's `colour` and `depth` with the path tracer's planes `aux`
    /// (technique::PathAux::planes, each width x height) -- the surface's
    /// normal, what it is made of, and what stands behind where it lets the
    /// eye through. One gaussian a cell of `footprint` pixels a side that the
    /// scene does not hold, of its surface's material; and where the surface
    /// is a pane, the pane and what is seen through it apart. The records
    /// stay on the device (`renderEncoding` says how they are laid out).
    [[nodiscard]] Result<RenderLook> fromRender(const gpu::Buffer& colour, const gpu::Buffer& depth,
                                                const gpu::Buffer& aux, uint32_t width, uint32_t height,
                                                const Projection& projection, const KnownScene& known,
                                                const RenderLookSettings& settings);

    /// `mask` (seenThrough's, width x height) struck where the render's
    /// surface lets the eye through (`aux`'s planes): nothing is taken out of
    /// the scene through glass, whose depth is not the scene's there.
    [[nodiscard]] Result<void> sparePanes(gpu::Buffer& mask, const gpu::Buffer& aux, uint32_t width, uint32_t height);

    /// A render's records (`count` of `floats` floats each, on the device)
    /// as the engine's flattened records (Flatten.h's kFlatStride floats
    /// each: what FlattenPack makes a PLY, an SPZ or a glTF of), the light
    /// scaled by `gain` on the way (scene_export.slang).
    [[nodiscard]] Result<gpu::Buffer> flatRecords(const gpu::Buffer& records, uint32_t count, uint32_t floats,
                                                  float gain = 1.0F);

    /// WHAT A PIECE HANDS ON, BY KIND (scene_level.slang). Of `count` records on
    /// the device, those `drawn` flags (a cut's: `CutPlan::flags`) and that
    /// are of `kind`, gathered in their order: a lens (solid glass), a
    /// mirror (a polished metal), the rest, or every kind.
    enum class RecordKind : uint32_t { All = 0, Matter = 1, Lens = 2, Mirror = 3 };
    struct RecordSet {
        gpu::Buffer records;
        uint32_t    count = 0;
    };
    [[nodiscard]] Result<RecordSet> recordsOf(const gpu::Buffer& records, uint32_t count, uint32_t floats,
                                              const gpu::Buffer& drawn, RecordKind kind);

    /// A projection under which a width on the screen is a width in the
    /// scene's own units: the cut asked with it (`planCut`, `spans`) is a
    /// level of detail whatever the eye -- a gaussian narrower than the
    /// threshold, in units, drawn in place of the finer ones laid over it.
    [[nodiscard]] static Projection worldWidths();

    /// How a record of `fromRender`'s is laid out, for scene::CloudLoader.
    [[nodiscard]] static io::SplatEncoding renderEncoding();

    /// WHERE A FRAME SEES THROUGH THE SCENE: a float a pixel, 1 where the scene
    /// drawn through the frame's camera (`knownColour`, `knownDepth`) has a
    /// surface nearer than the frame's own by more than `tolerance`, or
    /// something where the frame has nothing. `pixels` is how many. With
    /// `repaint`, also where the scene's surface is at the frame's depth and
    /// its colour is off the frame's by more than that share (`repainted`
    /// counts those): what is there is to be taken out and made again.
    /// `only` 1 keeps the mask to what says the world changed (seen past,
    /// nothing where something was, repainted) and 2 to what says only that
    /// the frame is finer than what holds the surface.
    [[nodiscard]] Result<gpu::Buffer> seenThrough(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                                  uint32_t height, const gpu::Buffer& knownColour,
                                                  const gpu::Buffer& knownDepth, float tolerance, uint32_t& pixels,
                                                  float repaint = 0.0F, uint32_t* repainted = nullptr,
                                                  const gpu::Buffer* knownSize = nullptr, float refine = 0.0F,
                                                  const Projection* projection = nullptr, float spread = 1.0F,
                                                  uint32_t only = 0);

    /// The gaussians of one piece that a frame saw through: `lifted` is that
    /// mask carried to the scene's gaussians by the blend (SplatLift.h's
    /// sums), `base` the piece's first slot. One most of whose weight fell on
    /// seen-through pixels is stamped `stamp` in `gone`; how many were.
    /// With `cloud` and `projection` and a `coarse` ratio, a gaussian wider
    /// than that many of the look's own discs at its depth goes on a fifth
    /// of its weight: it is what a finer look is laid over.
    /// With `most` over nothing, only where that share of its weight fell on
    /// the mask, whatever its width: the rule for `gone` being a record of
    /// what a finer look refined, where a parent part covered is left.
    /// With `mask` and `frameDepth` (the mask itself and the frame's depth,
    /// `maskWidth` x `maskHeight`, with `cloud` and `projection`), only a
    /// gaussian whose own middle falls on a set pixel is stamped, and only
    /// where the frame speaks of it: gone, where the frame sees past where
    /// the gaussian is (by `nearness` of its depth); `refining`, where the
    /// frame's surface is where the gaussian is. Its fringe over a
    /// silhouette is not the frame seeing through it, and a gaussian behind
    /// what the frame shows is not what the frame refined.
    [[nodiscard]] Result<uint32_t> markGone(const gpu::Buffer& lifted, uint32_t base, gpu::Buffer& gone,
                                            uint32_t count, uint32_t stamp, float threshold = 0.05F,
                                            const scene::GpuSplats* cloud = nullptr,
                                            const Projection* projection = nullptr, float coarse = 0.0F,
                                            float most = 0.0F, const gpu::Buffer* mask = nullptr,
                                            uint32_t maskWidth = 0, uint32_t maskHeight = 0,
                                            const gpu::Buffer* frameDepth = nullptr, float nearness = 0.02F,
                                            bool refining = false);

    /// THE PIECE CUT FOR A VIEW (scene_discover.slang's discoverCut): the same
    /// cloud with no opacity on the gaussians that are not the ones to draw --
    /// gone by look `upTo` (0: by any), or refined and too fine a level to be
    /// needed, or too coarse where finer ones came. `gone` and `refined` are
    /// the piece's records and `parent` a float a gaussian, the width of what
    /// it was laid over (null: none has, or none is). `threshold` is in
    /// pixels: a gaussian narrower than it on the screen stands for the finer
    /// ones laid over it; 0 draws the finest everywhere.
    /// `parent` is a float4 a gaussian: the colour and (w) the width of what
    /// it was laid over.
    /// With `gathered`, the cloud returned holds only the gaussians to draw
    /// -- positions, shape and harmonics, nothing else of them -- so that
    /// what is not drawn costs nothing; its count may be none. And with
    /// `morph`, a gathered gaussian is moved towards what it was laid over --
    /// its width, its colour, its opacity given up as it comes to lie over
    /// its neighbours -- across that share of the threshold above the point
    /// where its parent is drawn in its place: the step from children to
    /// parent is then between two pictures all but the same.
    [[nodiscard]] Result<scene::GpuSplats> cut(const scene::GpuSplats& cloud, const gpu::Buffer* gone,
                                               const gpu::Buffer* refined, const gpu::Buffer* parent,
                                               const Projection& projection, float threshold, uint32_t upTo = 0,
                                               bool gathered = false, float morph = 0.0F);

    /// A SCENE'S CUT AS ONE CLOUD. A cut gathered a piece at a time is as
    /// many clouds as the scene has pieces, and a rasteriser pays for each:
    /// a view from far drew a twentieth of the gaussians in no less time.
    /// `planCut` flags a piece's gaussians and counts them; `cutCloud` makes
    /// a cloud with room for every piece's; `gatherCut` copies a piece's into
    /// it from `base` on (and moves them, with `morph`).
    struct CutPlan {
        gpu::Buffer flags;      ///< a uint a gaussian: drawn
        gpu::Buffer places;     ///< its place among the piece's drawn
        uint32_t    drawn = 0;
    };
    [[nodiscard]] Result<CutPlan> planCut(const scene::GpuSplats& cloud, const gpu::Buffer* gone,
                                          const gpu::Buffer* refined, const gpu::Buffer* parent,
                                          const Projection& projection, float threshold, uint32_t upTo = 0);
    [[nodiscard]] Result<scene::GpuSplats> cutCloud(const scene::GpuSplats& like, uint32_t count);
    [[nodiscard]] Result<void> gatherCut(const scene::GpuSplats& cloud, const gpu::Buffer* parent, const CutPlan& plan,
                                         const Projection& projection, float threshold, float morph,
                                         scene::GpuSplats& into, uint32_t base);

    /// HOW MANY GAUSSIANS EVERY THRESHOLD DRAWS (discoverCutSpans): a piece's
    /// gaussians each added into `histogram` (made by `cutHistogram`, shared
    /// by a scene's pieces) where the span of thresholds it is drawn between
    /// begins and taken where it ends. `thresholdFor` then reads the finest
    /// threshold, in pixels, that draws no more than `budget` gaussians, and
    /// how many it draws -- the only thing read back, a few hundred counts.
    [[nodiscard]] Result<gpu::Buffer> cutHistogram();
    [[nodiscard]] Result<void> spans(const scene::GpuSplats& cloud, const gpu::Buffer* gone, const gpu::Buffer* refined,
                                     const gpu::Buffer* parent, const Projection& projection, uint32_t upTo,
                                     gpu::Buffer& histogram);
    [[nodiscard]] Result<float> thresholdFor(const gpu::Buffer& histogram, uint64_t budget, uint64_t* drawn = nullptr);

    /// A piece without the gaussians its record `gone` stamps: the same
    /// cloud, those given no opacity. With `upTo`, only those stamped by a
    /// look up to that one: the piece as it stood then.
    [[nodiscard]] Result<scene::GpuSplats> without(const scene::GpuSplats& cloud, const gpu::Buffer& gone,
                                                   uint32_t upTo = 0);

    /// HOW COARSE A PIECE IS, AS SOMETHING TO DRAW IT BY: a neural cloud whose
    /// colour is each gaussian's largest standard deviation. A frame drawn by
    /// it (`SplatInstance::neural`) holds, a pixel, the size of the gaussians
    /// it is made of -- `knownSize` below.
    [[nodiscard]] Result<NeuralCloud> sizes(const scene::GpuSplats& cloud);

    /// A buffer of `words` zeros: what a look's frame sums its contributions into.
    [[nodiscard]] Result<gpu::Buffer> zeros(uint32_t words);

private:
    void bind(rhi::ShaderCursor cursor) const;

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  clear_, mark_, count_, mask_, gone_, sizes_, frameValid_, framePack_, frameThrough_;
    gpu::ComputeKernel  renderValid_, renderPack_, renderSpare_, cut_, cutFlags_, gather_, cutSpans_, exportFlat_, levelFlags_, levelGather_;
    std::optional<gpu::PrefixSum> prefix_;
    gpu::Buffer         counters_, spareWords_, spareFloats_;
};

}   // namespace athenea::render
