// Copyright (c) 2026 jesus luque.
//
// The splat rasteriser: global depth sort, stable tile sort, per-pixel blend,
// no per-tile capacity. The stages are described in shaders/athenea/splat/frame.slang.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/AsyncReadback.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/render/Camera.h"
#include "athenea/render/Points.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::render {

/// A selection volume in a cloud's own space, what happens to the splats in
/// it, and a grade: openFXplayer's SplatEdit. The rule is written once, in
/// shaders/athenea/common/edit.slang; every splat renderer reads it.
struct SplatEdit {
    enum class Shape : uint32_t { Box = 0, Sphere = 1 };
    enum class Mode : uint32_t { Keep = 0, Remove = 1, Grade = 2 };
    bool                 active = false;
    Shape                shape = Shape::Box;
    Mode                 mode = Mode::Grade;
    std::array<float, 3> centre{0, 0, 0};
    /// Half extents of the box; radius of the sphere in x.
    std::array<float, 3> size{1, 1, 1};
    std::array<float, 3> tint{1, 1, 1};
    float                saturation = 1.0F;
    float                brightness = 1.0F;
    /// Multiplies the opacity of what the volume selects.
    float                opacity = 1.0F;
    /// Wherever the volume is: below this opacity a splat goes (0: off).
    float                minOpacity = 0.0F;
    /// Wherever the volume is: longer than this on its longest axis, in the
    /// cloud's units, a splat goes (0: off).
    float                maxScale = 0.0F;
    /// The volume test turned round.
    bool                 invert = false;
};

/// WHAT A FRAME SAYS A PRIM IS MADE OF, over what its gaussians carry.
///
/// A converted cloud keeps the prim each gaussian came from -- the id the
/// Cryptomatte writes -- so the id a picked pixel hands back selects every
/// gaussian that stood on one material when the mesh was walked. A row keyed
/// on that id is therefore a material override a host can address by
/// clicking, which is what `athenea view` does with it.
///
/// It is read by the shading, never baked: the cloud's `pbr` and colours stay
/// as the file wrote them and nothing on disk is touched. A negative value
/// means "leave what the gaussian carries", so a prim whose roughness came
/// out of a map is not flattened by a tint.
struct SplatOverride {
    uint32_t             id = 0;
    float                metallic = -1.0F;
    float                roughness = -1.0F;
    float                transmission = -1.0F;
    std::array<float, 3> tint{1.0F, 1.0F, 1.0F};
    /// The tint is the colour rather than a factor on it.
    bool                 replaceColour = false;
};

struct SplatInstance {
    const scene::GpuSplats* splats = nullptr;
    Mat4                    objectToWorld = Mat4::identity();
    SplatEdit               edit;
    /// AtheneaSplatLightingAPI: relight these splats from the scene's lights
    /// instead of showing the radiance they were baked with.
    bool                    relight = false;
    /// `primvars:athenea:splat:litBody`: the colours are light already -- the body
    /// of the material with this scene's light on it, which is what
    /// `athenea mesh2splat` bakes -- so relighting adds the polish and nothing
    /// else. Without it the colours are an albedo and relighting lights them.
    bool                    litBody = false;
    /// The categories this cloud belongs to, one bit each, as a mesh instance
    /// carries them: a light reaches it only where its link says so.
    uint64_t                categories = 0;
    /// WHETHER A GAUSSIAN MAY REFLECT THE CLOUD IT BELONGS TO.
    ///
    /// The polish reads the prepared sky along the mirror direction, which is
    /// right for a gaussian with nothing in front of it and wrong for one
    /// facing its own asset. With this, a ray is cast into the cloud's own
    /// tree and what it meets is what the gaussian reflects -- one ray a
    /// gaussian a frame, and only where the route can trace.
    bool                    reflectCloud = false;
    /// The per-prim overrides above, two float4 a row as
    /// `athenea/common/splat_override.slang` reads them. Null or zero rows: the
    /// cloud's own values stand.
    const gpu::Buffer*      overrides = nullptr;
    uint32_t                overrideCount = 0;
    /// THE INDEX OF REFRACTION OF WHAT THIS CLOUD TRANSMITS, 0 for none.
    /// `primvars:athenea:splat:ior`. Where it is set, the transmitted half of a
    /// gaussian's body looks along the bent direction rather than straight
    /// back, which is what makes a glass ball show the room turned.
    float                   ior = 0.0F;
    /// WHETHER THE INDIRECT HALF OF A TRANSFER IS ADDED. A cloud that carries
    /// one is drawn with it or without at the frame's word, so what the
    /// interreflection is worth can be measured on the same cloud.
    bool                    transferIndirect = true;
    /// UNDER A SHUTTER. What a skeleton moved each splat over it, in the
    /// cloud's own space, two words of halves a splat (`splat_skin.slang`
    /// writes it). Null: nothing but the transform moves. It is per frame and
    /// per instance, which is why it is here and not on `scene::GpuSplats`.
    const gpu::Buffer*      motion = nullptr;
    /// The shutter over the span that displacement was measured on: Hydra
    /// hands back the authored samples bracketing the shutter, not the
    /// shutter's own ends, and a pose a frame apart under a 180 degree
    /// shutter has to be halved.
    float                   motionScale = 1.0F;
    /// Object to view, the difference over the shutter: the camera's motion
    /// and this prim's, together. Zero: neither moves.
    std::array<float, 12>   viewStep{};
    /// `primvars:athenea:splat:catcher`: a shadow catcher (athenea mesh2splat
    /// --shadow-catcher), projected by a kernel of its own: black, as opaque
    /// as what its object took of the light reaching it.
    bool                    catcher = false;
};

/// The frame's lights, as a renderer that must not depend on the light module
/// can take them: the records buffer light::LightTable uploads, how many there
/// are, and the total power their cumulative shares are of. Splats relight
/// from these where their prim asked for it (AtheneaSplatLightingAPI).
struct SplatLights {
    const gpu::Buffer* records = nullptr;
    uint32_t           count = 0;

    /// Where a relit splat's shadow ray traces: the proxies the Gaussian ray
    /// tracer built for this frame (`GaussianRayTracer::shadowScene`, the
    /// Hardware route). Null: a relit splat takes the light whole, as it did
    /// before there was a ray to ask with.
    rhi::IAccelerationStructure* shadowTlas = nullptr;
    const gpu::Buffer*           shadowFrames = nullptr;
    const gpu::Buffer*           shadowColours = nullptr;
    const gpu::Buffer*           shadowInstanceData = nullptr;
    const gpu::Buffer*           shadowInstanceIndices = nullptr;
    /// Where a shadow ray starts, in the splat's own sigmas (a captured
    /// surface is a crowd of overlapping Gaussians), and where it gives up.
    float                        shadowOffset = 3.0F;
    float                        shadowCut = 1.0e-3F;
    /// WHAT REACHES EACH SPLAT FROM EACH LIGHT, MEASURED BEFORE THE FRAME:
    /// one float a splat a light, `visibilityLights` a splat, in instance
    /// order (technique::SplatVisibility writes them from a cloud's baked
    /// fields). Set, the rasteriser takes these instead of tracing a shadow
    /// ray, and so can the traced route, which has no ray of its own.
    const gpu::Buffer*           visibilityFactors = nullptr;
    uint32_t                     visibilityLights = 0;
    /// THE SKY, PREPARED FOR A KERNEL THAT CANNOT SAMPLE IT
    /// (`technique::Environment`). A dome's image goes through the material
    /// texture table and the splat projection sits below material in the
    /// module order, so without this a cloud under an HDRI is lit by the
    /// dome's mean colour while the mesh beside it reflects the sky. Null:
    /// that is what happens, as it did before.
    const gpu::Buffer*           envTexels = nullptr;    ///< prefiltered octahedral levels
    const gpu::Buffer*           envSh = nullptr;        ///< nine float4 a dome
    const gpu::Buffer*           envOfLight = nullptr;   ///< a light's slice, or none
    uint32_t                     envLights = 0;          ///< lights `envOfLight` describes
    uint32_t                     envBaseSide = 0;        ///< octahedral side of level 0, as it was written
    /// The sun taken out of each dome: two float4 a slice, direction and
    /// solid angle then irradiance. Null, or a solid angle of 0, and the
    /// harmonics hold the whole sky as they did before.
    const gpu::Buffer*           envSun = nullptr;
    /// WHAT THESE LIGHTS AND THIS SKY ARE, as a number that changes whenever
    /// they do (light::LightTable::revision). 0 says nothing is known, and no
    /// cloud keeps what depends on them from one frame to the next.
    uint64_t                     revision = 0;

    [[nodiscard]] bool environment() const noexcept {
        return envLights > 0 && envBaseSide > 0 && envTexels != nullptr && envSh != nullptr &&
               envOfLight != nullptr && envSun != nullptr;
    }

    [[nodiscard]] bool any() const noexcept { return records != nullptr && count > 0; }
    [[nodiscard]] bool shadows() const noexcept {
        return shadowTlas != nullptr && shadowFrames != nullptr && shadowColours != nullptr &&
               shadowInstanceData != nullptr && shadowInstanceIndices != nullptr;
    }
};

struct RenderSettings {
    uint32_t width = 1920;
    uint32_t height = 1080;
    bool     antialias = true;
    uint32_t maxShDegree = 3;
    /// Premultiplied, linear. Transparent by default.
    std::array<float, 4> background{0, 0, 0, 0};
    enum class Depth { Mean, Threshold };
    Depth    depth = Depth::Mean;
    float    depthThreshold = 0.5F;
    /// CRYPTOMATTE. Keep, beside the colour, which prims covered each pixel
    /// and by how much: `RenderTargets::crypto`, the `CryptoObject` layers of
    /// the Cryptomatte specification. Only the clouds that carry per-gaussian
    /// ids (`scene::GpuSplats::crypto`, what `athenea mesh2splat` writes) and the
    /// under layer's own ids are named; a capture has no ancestry to name and
    /// is left out rather than named wrongly.
    bool     cryptomatte = false;
    /// Wait after every stage so `FrameStats` times each one. Slower; for bench.
    bool     timeStages = false;
    /// COUNT WHAT THE FRAME DID on the device -- why each culled splat was
    /// culled, the most tiles one touched, what each cloud kept -- and copy
    /// it out without waiting: `TileRasterizer::latestCounters` hands back
    /// the newest frame the device has finished, tagged `countersTag`. Four
    /// small dispatches and a copy; off, nothing is counted.
    bool     countSplats = false;
    uint64_t countersTag = 0;
};

/// WHAT A FRAME OF SPLATS DID, as the device counted it
/// (splat_frame_counters.slang): read a frame or two after it was drawn.
struct SplatCounters {
    /// Clouds counted one by one; the rest are only in the totals.
    static constexpr uint32_t kMaxClouds = 64;
    static constexpr uint32_t kWords = 16 + 2 * kMaxClouds;
    /// Why a splat was culled, in frame.slang's numbering.
    enum Cull : uint32_t {
        Unprojected = 0,   ///< a slot no splat projection wrote
        Edit,              ///< removed by its prim's SplatEdit
        Depth,             ///< nearer than the near plane or past the far one
        Degenerate,        ///< a footprint of no area
        Faint,             ///< under 1/255 of opacity once spread
        Offscreen,         ///< its footprint lies outside the frame
        NoTile,            ///< in the frame, but no tile meets its ellipse
        Reasons,
    };
    uint64_t tag = 0;        ///< the counted frame's `RenderSettings::countersTag`
    uint32_t slots = 0;      ///< splats counted
    uint32_t visible = 0;    ///< kept: what the depth sort sorts
    uint32_t pairs = 0;      ///< (tile, splat) pairs: what the tile sort sorts
    uint32_t maxTiles = 0;   ///< the most tiles one splat touched
    std::array<uint32_t, Reasons> culled{};
    struct Cloud {
        uint32_t visible = 0;
        uint32_t pairs = 0;
    };
    std::vector<Cloud> clouds;   ///< by instance, in the order the frame was given them
};

struct FrameStats {
    uint32_t splats = 0;
    uint32_t visible = 0;
    uint32_t pairs = 0;
    double   projectMs = 0, depthSortMs = 0, countsMs = 0, emitMs = 0, tileSortMs = 0,
             blendMs = 0, totalMs = 0;
};

/// The images a frame writes: bottom row first, width * height.
struct RenderTargets {
    uint32_t    width = 0;
    uint32_t    height = 0;
    gpu::Buffer colour;   ///< float4, premultiplied, linear
    gpu::Buffer depth;    ///< float, view z
    /// THE CRYPTOMATTE, where the frame asked for one: three float4 a pixel,
    /// layer after layer (`crypto[layer * width * height + pixel]`), each
    /// (id, coverage, id, coverage) in descending coverage. The ids are
    /// `athenea::core::cryptomatteId` values, carried as the floats their bits are.
    gpu::Buffer crypto;
    /// The id of the opaque surface this layer drew, one uint a pixel: what a
    /// layer under the splats hands the blend so its coverage is named. Empty
    /// where nothing wrote ids.
    gpu::Buffer cryptoIds;

    [[nodiscard]] bool hasCrypto() const noexcept { return crypto.valid(); }
};

class TileRasterizer {
public:
    [[nodiscard]] static Result<TileRasterizer> create(gpu::ShaderLibrary& library);

    /// `points` are drawn as opaque discs in the same depth order as the
    /// splats. `under`, when given, is an opaque layer (a rasterised point
    /// layer of the same size) the splats are composited over and cut by.
    [[nodiscard]] Result<FrameStats> render(const Camera& camera,
                                            std::span<const SplatInstance> instances,
                                            const RenderSettings& settings, RenderTargets& targets,
                                            std::span<const PointInstance> points = {},
                                            const RenderTargets* under = nullptr,
                                            const SplatLights* lights = nullptr);
    /// The same, from a projection someone else computed (a Hydra host).
    [[nodiscard]] Result<FrameStats> render(const Projection& projection,
                                            std::span<const SplatInstance> instances,
                                            const RenderSettings& settings, RenderTargets& targets,
                                            std::span<const PointInstance> points = {},
                                            const RenderTargets* under = nullptr,
                                            const SplatLights* lights = nullptr);

    /// The newest frame drawn with `RenderSettings::countSplats` that the
    /// device has finished, or nothing yet. Never waits.
    [[nodiscard]] std::optional<SplatCounters> latestCounters();

    /// Gives back the buffers a frame grew into (projections, sort keys,
    /// tile pairs), which the next frame makes again; from then on they
    /// grow to what a frame needs and no further, rather than half as much
    /// again. What a device that ran out of memory is asked to do first.
    void releaseScratch();

private:
    [[nodiscard]] Result<void> countFrame(gpu::CommandBatch& batch, std::span<const SplatInstance> instances,
                                          uint32_t splatSlots, uint32_t all);
    [[nodiscard]] Result<void> reserveSplats(uint32_t count);
    [[nodiscard]] Result<void> reservePairs(uint32_t count);
    [[nodiscard]] Result<void> reserveTargets(RenderTargets& targets, uint32_t width,
                                              uint32_t height, uint32_t tiles, bool crypto);

    gpu::Device*       device_ = nullptr;
    gpu::ShaderLibrary* library_ = nullptr;
    gpu::PrefixSum     prefix_;
    gpu::RadixSort     sort_;
    gpu::ComputeKernel project_;
    /// The same with the transfer's shading compiled out, for a cloud that
    /// carries none (splat_project.slang's projectSplat says why).
    gpu::ComputeKernel projectPlain_;
    /// And for a cloud with the first transfer (no cells): the TX transfer's
    /// shading compiled out of it too.
    gpu::ComputeKernel projectFirst_;
    /// A shadow catcher's (splatProjectCatcher): black, as opaque as what
    /// its object took, no relighting.
    gpu::ComputeKernel projectCatcher_;
    /// A transfer cloud's view-independent terms, a splat each, kept while
    /// the lights, the sky, the cloud and its place stand (`txCaches_`).
    gpu::ComputeKernel viewless_;
    struct TxCache {
        gpu::Buffer buffer;
        uint64_t    key = 0;
        uint64_t    seen = 0;   ///< the last frame that drew the cloud
    };
    uint64_t                                             frameOfCaches_ = 0;
    std::unordered_map<const scene::GpuSplats*, TxCache> txCaches_;
    gpu::ComputeKernel compact_;
    gpu::ComputeKernel pointsProject_;
    gpu::ComputeKernel gather_;
    gpu::ComputeKernel emit_;
    gpu::ComputeKernel clear_;
    gpu::ComputeKernel ranges_;
    gpu::ComputeKernel blend_;
    gpu::ComputeKernel blendComposite_;
    gpu::ComputeKernel blendCrypto_;
    gpu::ComputeKernel blendCompositeCrypto_;

    bool     tight_ = false;   ///< grow to the need exactly (after releaseScratch)
    uint32_t splatCapacity_ = 0;
    uint32_t pairCapacity_ = 0;
    uint32_t tileCapacity_ = 0;
    gpu::Buffer proj_, tileRects_, tilesTouched_, visible_, depthKeys_;
    /// A slot each beside `proj_`: a TX transfer's reflection's slope across
    /// the footprint, read where the record is marked (frame.slang's kSlopeMark).
    gpu::Buffer slopes_;
    gpu::Buffer sharpPolish_;   ///< a sharp record's base polish, two words a slot (kSharpMark)
    gpu::Buffer visibleOffsets_, visibleTotal_, touchedOffsets_, touchedTotal_;
    gpu::SortBuffers depthSort_;   // keysLo = visible depth keys, values = splat index
    gpu::Buffer sortedCounts_, offsets_, totalPairs_;
    gpu::SortBuffers tileSort_;    // keysLo = pair tiles, values = pair splats
    gpu::Buffer ranges_buffer_;
    gpu::Buffer placeholderColour_, placeholderDepth_;
    /// One word and one record of nothing, for a frame with no sky prepared:
    /// a name a shader declares must be bound whether it is read or not.
    gpu::Buffer emptyEnvWords_, emptyEnvSh_;
    /// The Cryptomatte id a projected slot carries, beside `proj_` and indexed
    /// the same way, so a pair names its splat's prim with one more read.
    gpu::Buffer cryptoIds_;
    gpu::Buffer placeholderCrypto_;
    /// One record of nothing, for the frames that relight nothing: a name the
    /// shader declares has to be bound whether it is read or not.
    gpu::Buffer emptyLights_;
    gpu::Buffer emptyShadow_;      ///< bound where a frame casts no splat shadow
    gpu::Buffer emptyMotion_;      ///< bound where nothing moves under the shutter
    gpu::Buffer shadowFactors_;    ///< one float a (splat, light), where a relit cloud shadows
    gpu::ComputeKernel splatShadow_;
    bool        shadowsSupported_ = false;
    /// The panel's counters (`countSplats`), made the first time they are asked for.
    bool               countersMade_ = false;
    gpu::ComputeKernel countersClear_, counters_, countersCloud_;
    gpu::Buffer        counterWords_;
    gpu::AsyncReadback counterReadback_;
};

}   // namespace athenea::render
