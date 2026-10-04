// Copyright (c) 2026 jesus luque.
//
// THE GROUND'S SHADOW AS GAUSSIANS: where the catcher stands (task PLAY-G).
//
// A converted object casts its shadow on the ground it stands on, and a frame
// of gaussians alone has no ground to receive it. The shadow catcher is a
// layer of gaussians lying on that ground, under and around the object, that
// darkens whatever is drawn behind it -- the sky's own floor, a backplate, a
// mesh -- by what the object takes from it. This is where that layer stands:
//
//   the ground   found by itself -- the largest mesh outside the object, flat
//                along the stage's up axis, whose top is where the object's
//                bottom is -- or named
//   the patch    a grid of quads on the ground's plane over the object's
//                footprint widened by `margin` times its height, which is as
//                far as a sky's shadow of it reaches in any useful amount
//
// and a stage that is the source with that quad in it, for `athenea
// mesh2splat` to convert and bake: the object and the ground stay in it as
// what the bake's rays meet, the patch is what is converted.
//
// The boxes are the meshes' authored extents (UsdGeomBBoxCache with the
// extents hint): metadata the file carries, read, not computed from points.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

#include "athenea/core/Result.h"

namespace athenea::usd {

struct ShadowCatcherOptions {
    /// The object whose shadow is caught: a prim path in the stage.
    std::string object;
    /// The ground; empty finds it.
    std::string ground;
    /// How far past the object's footprint the patch reaches, in heights of
    /// the object.
    double      margin = 1.5;
    /// The cell the patch will be converted at, world units; 0 is a hundredth
    /// of the object's height. The patch is a grid of quads about 64 cells a
    /// side each, so no triangle walks more cells than a conversion takes.
    double      cell = 0.0;
};

struct ShadowCatcherStage {
    std::filesystem::path stage;      ///< the source with the patch added
    std::string           prim;       ///< the patch's prim path in it
    std::string           ground;     ///< the ground it lies on
    double                height = 0.0;   ///< the object's, along up
    double                cell = 0.0;     ///< what the patch is to be converted at
    uint32_t              quads = 1;      ///< the grid's quads a side
    std::array<double, 3> min{};          ///< the patch's box, in the world
    std::array<double, 3> max{};
};

/// Writes `into`: the stage at `source` as a sublayer, and the patch.
[[nodiscard]] Result<ShadowCatcherStage> writeShadowCatcherStage(const std::filesystem::path& source,
                                                                 const ShadowCatcherOptions& options,
                                                                 const std::filesystem::path& into);

}   // namespace athenea::usd
