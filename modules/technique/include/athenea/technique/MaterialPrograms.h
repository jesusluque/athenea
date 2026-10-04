// Copyright (c) 2026 jesus luque.
//
// The generated Slang every pass that touches a material shares: a module
// that imports each compiled material (MaterialCompiler) and dispatches on a
// material row's function. Shading calls it to build a lobe stack; the
// visibility passes call it to ask whether a sample's material cuts the
// sample away (`opacityThreshold`), which is why the module has to be one
// thing both can import rather than a kernel of its own.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/light/LightTable.h"
#include "athenea/material/MaterialCompiler.h"
#include "athenea/material/TextureStore.h"
#include "athenea/render/Camera.h"
#include "athenea/world/GpuScene.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::technique {

class SplatShadows;

/// A material row, as shaders/athenea/technique/material_lookup.slang reads it.
/// `function` 0 is the fallback (displayColor); k is the k-th module given to
/// MaterialPrograms::setModules, counted from 1.
struct MaterialRecord {
    uint32_t function = 0;
    uint32_t blob = 0;      ///< its first word in the blob
    uint32_t flags = 0;
    uint32_t pad = 0;
};

/// MaterialRecord::flags: the material's opacity cuts samples out instead of
/// blending them, so visibility itself has to evaluate it.
inline constexpr uint32_t kMaterialCutout = 1u;
/// MaterialRecord::flags: a UsdPreviewSurface in `transparent` opacity mode.
/// The path tracer keeps its specular and emission at full weight and lets
/// (1 - opacity) of the light straight through; the raster still cuts by lot.
inline constexpr uint32_t kMaterialTransparent = 2u;
/// It lets light through: the bake's open-or-not rays look at it (material_lookup.slang).
inline constexpr uint32_t kMaterialTransmits = 4u;

class MaterialPrograms {
public:
    [[nodiscard]] static Result<MaterialPrograms> create(gpu::ShaderLibrary& library);

    /// The compiled materials the module dispatches to, loaded into the
    /// library here. Modules that repeat are not loaded twice; a set that
    /// repeats regenerates nothing.
    /// `opacities`, where given, is beside `modules` one for one: the
    /// opacity alone of a module a cut-out uses, or an empty module where
    /// there is none. It makes `evaluateOpacity`, which shadow rays ask.
    [[nodiscard]] Result<void> setModules(std::span<const material::CompiledMaterial> modules,
                                          std::span<const material::CompiledMaterial> opacities = {});

    /// What to import: named after the set it dispatches to, so one module
    /// serves every pass and every frame that shows the same materials.
    [[nodiscard]] const std::string& module() const noexcept { return module_; }

    [[nodiscard]] gpu::ShaderLibrary& library() const noexcept { return *library_; }

private:
    gpu::ShaderLibrary* library_ = nullptr;
    std::string         module_;
    std::string         signature_ = "unset";
};

/// Where a frame's light groups go: `count` planes of float4, a pixel each,
/// one after another in `colour` (the mean, as the frame's colour) for the
/// material shading; the path tracer keeps them in its own accumulation
/// (PathTracer::lightGroupMeanOffset). A frame with none compiles the
/// kernels without them. At most kMaxLightGroups.
struct LightGroupTargets {
    const gpu::Buffer* colour = nullptr;
    uint32_t           count = 0;
};
constexpr uint32_t kMaxLightGroups = 8;

/// What generated material code reads: the scene a visibility sample is
/// rebuilt from, the material rows and the blob their values live in, the
/// textures, and the frame's time.
struct MaterialFrame {
    const MaterialPrograms*       programs = nullptr;
    const world::GpuScene*        scene = nullptr;
    const gpu::Buffer*            records = nullptr;
    const gpu::Buffer*            blob = nullptr;
    const material::TextureStore* textures = nullptr;
    float                         time = 0.0F;
    /// Whether the visibility passes cut a fractional opacity by a pixel's
    /// lot: the raster route does; the path tracer draws its own, a sample.
    uint32_t                      alphaDither = 1;
    /// Whether any material of the frame cuts samples away: a shadow ray then
    /// asks each surface it meets whether it is there.
    bool                          cutouts = false;
    /// The frame's lights. None: the headlight, as meshes were lit before
    /// there were any.
    const light::LightTable*      lights = nullptr;
    /// What a shadow ray traces against, when the device traces at all.
    rhi::IAccelerationStructure*  shadows = nullptr;
    /// Splat clouds a shadow ray is dimmed by: their tables packed into one
    /// buffer (technique::SplatShadows). Null, or a scene with no instances,
    /// and a cloud casts no shadow on a mesh -- which is what every frame did
    /// before. Only where the device has inline rays; the path tracer says so.
    const SplatShadows*           splatShadows = nullptr;
    /// A splat shadow ray stops once this little light is left: the cloud's
    /// far side cannot brighten what its near side has already blocked.
    float                         splatShadowCut = 0.01F;
    /// What the clouds of this frame stop, measured from each light into a
    /// map (technique::SplatShadowMap): one buffer, read with no ray at all,
    /// so a device without ray tracing shadows a floor under a cloud too.
    /// Null and nothing is dimmed, which is what a frame without clouds does.
    /// A texture because the shading kernel has no buffer slot left.
    rhi::ITextureView*            cloudShadow = nullptr;
    /// And its transmittance chain (SplatShadowMap::chainView): what a floor
    /// far from the map reads at the level of its pixel's footprint. Bound
    /// with `cloudShadow`, which says whether either is.
    rhi::ITextureView*            cloudShadowChain = nullptr;
    /// Samples per light. One is the interactive choice; a test that wants
    /// an area light's irradiance without noise asks for more.
    uint32_t                      samples = 1;
    /// One light per sample, chosen by power, instead of every light at every
    /// pixel. Exact either way; what changes is where the cost goes -- with
    /// the loop it grows with the number of lights, with the choice it does
    /// not, at the price of noise a frame has to average away.
    bool                          chooseLights = false;
    /// With `chooseLights`: choose through the light BVH -- by power over
    /// distance within the lights' cones, at the shading point -- where the
    /// table has one (any bounded light); by power alone otherwise.
    bool                          lightBvh = true;
    /// The light groups the frame writes beside its colour: a light's direct
    /// contribution, at every bounce, under its group. Emission and the
    /// background are in no group.
    LightGroupTargets             groups;
    /// The frame's volumes (world::VolumeSet's words) for the path tracer,
    /// which traces free flights through them and transmittance to the
    /// lights; null or none: no medium. The raster technique draws none.
    const gpu::Buffer*            volumes = nullptr;
    uint32_t                      volumeCount = 0;
    /// The emitting triangles as a light (EmissiveTable's buffer), for the
    /// path tracer's next event estimation; null, or no power: none. Not in a
    /// frame with volumes.
    const gpu::Buffer*            emissive = nullptr;
    float                         emissivePower = 0.0F;

    [[nodiscard]] bool valid() const noexcept {
        return programs != nullptr && scene != nullptr && records != nullptr && records->valid() && blob != nullptr &&
               blob->valid() && textures != nullptr;
    }
};

/// The scene buffers shaders/athenea/technique/surface.slang reads, by name.
void bindScene(rhi::ShaderCursor cursor, const world::GpuScene& scene);

/// Those, and material_lookup.slang's tables, textures and view to world.
/// Not the lights: only the shading kernel declares them, since the cutout
/// visibility passes evaluate a material for its opacity alone.
void bindMaterialFrame(rhi::ShaderCursor cursor, const MaterialFrame& frame, const render::Projection& projection);

}   // namespace athenea::technique
