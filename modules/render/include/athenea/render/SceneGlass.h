// Copyright (c) 2026 jesus luque.
//
// WHAT THE GLASS OF A FOUND SCENE SHOWS FROM A VIEW (scene_glass.slang).
//
// A found scene keeps, a gaussian, what its surface is made of beside the
// picture the look saw: `scene::GpuSplats::pbr` (its transmission) and
// `normals` as every cloud of the engine keeps them, and here `matter`, a
// float4 a gaussian -- what one crossing of it tints and its index (0 where
// it bends nothing: an opaque surface, a sheet). Those are the glass's own
// and no eye's. What it shows is worked out a view, by `see`: each glass
// gaussian's eye ray reflected and refracted through the tree the gaussian
// ray tracer builds over the cloud, and its colour written as what those
// rays met among the scene's own gaussians.
#pragma once

#include <cstdint>
#include <optional>

#include "athenea/core/Result.h"
#include <array>
#include <optional>
#include <span>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/Discovery.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

struct GlassSeen {
    uint32_t lit = 0;         ///< gaussians the lamps were asked about (SceneGlass::setLamps)
    double   litMs = 0.0;
    uint32_t glass = 0;       ///< gaussians of solid glass in the cloud
    uint32_t behind = 0;      ///< whose ray through the glass met a gaussian of the scene
    uint32_t mirrored = 0;    ///< whose mirror ray did
    uint32_t mirrors = 0;     ///< gaussians of polished metal, shown as mirrors
    double   treeMs = 0.0;    ///< the tree over the cloud
    double   seenMs = 0.0;    ///< the rays
};

/// A LAMP NOBODY PATH TRACED, added to a found scene a gaussian (scene_glass.slang's `sceneLit`): a sphere
/// of `radius` at `position` sending `radiance` (UsdLux's intensity times colour, not normalised), within a
/// cone about `axis` where `cone` (the cosine of its half angle) is over -1.5; or a distant light `position`
/// points towards, whose `radiance` is the irradiance it lays on a surface facing it, over pi.
struct SceneLamp {
    std::array<float, 3> position{0.0F, 0.0F, 0.0F};
    float                radius = 0.1F;
    std::array<float, 3> axis{0.0F, -1.0F, 0.0F};
    float                cone = -2.0F;
    std::array<float, 3> radiance{1.0F, 1.0F, 1.0F};
    bool                 distant = false;
};

class SceneGlass {
public:
    [[nodiscard]] static Result<SceneGlass> create(gpu::ShaderLibrary& library);

    /// Room in a cut's cloud (`Discovery::cutCloud`) for what `gather` copies:
    /// its `pbr` and `normals`, and the `matter` handed back.
    [[nodiscard]] Result<gpu::Buffer> room(scene::GpuSplats& cloud);

    /// What a piece's drawn gaussians (`plan`) are made of, copied beside
    /// them into the cut's cloud from `base` on, as `Discovery::gatherCut`
    /// copies what is drawn.
    [[nodiscard]] Result<void> gather(const scene::GpuSplats& piece, const gpu::Buffer& pieceMatter,
                                      const Discovery::CutPlan& plan, scene::GpuSplats& into, gpu::Buffer& matter,
                                      uint32_t base);

    /// The colour of every solid glass gaussian of `cloud` written as what
    /// it shows from `projection`'s eye. The cloud is in the world's space
    /// and is the scene: what the rays meet is looked for in it.
    [[nodiscard]] Result<GlassSeen> see(const Projection& projection, scene::GpuSplats& cloud,
                                        const gpu::Buffer& matter);

    /// The lamps `see` adds to what is neither glass nor a mirror before the
    /// glass is shown the room (so it shows the room lit); `only` leaves out
    /// the light the scene was found under. `matter` then holds an opaque
    /// gaussian's base colour.
    [[nodiscard]] Result<void> setLamps(std::span<const SceneLamp> lamps, bool only);

private:
    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  gather_, seen_, lit_;
    gpu::Buffer         counters_, lamps_;
    uint32_t            lampCount_ = 0;
    bool                lampsOnly_ = false;
    std::optional<GaussianRayTracer> tracer_;
};

}   // namespace athenea::render
