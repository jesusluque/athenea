// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat --validate DIR`: a conversion measured material by
// material against the stage path traced (docs/decisions.md, task TX, "A
// conversion measured by the engine itself").
//
// The stage is path traced once (DIR/gt.exr, kept and reused at the same
// size), rasterised once as meshes with its Cryptomatte, and then, for every
// material it binds, only that material's meshes are converted -- the rest
// of the stage stays the meshes it is -- and the frame rasterised. Each
// material is measured over its own pixels: a mask out of the mesh frame's
// matte, the frames multiplied by it, and the Measure effect over its box.
// What comes out is a table on stdout and in DIR/validate.json, and for each
// material GT, the mesh rasterised and the cloud side by side
// (DIR/<material>.png and .exr) and the cloud's frame (DIR/<material>_gs.exr).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace aofx {
class Effect;
}
namespace athenea::gpu {
class ShaderLibrary;
}
namespace athenea::gpu_host {
class Context;
}

namespace athenea::cli {

struct ValidateJob {
    std::string              stage;
    std::string              directory;
    std::string              camera;        ///< "" takes the stage's first camera
    uint32_t                 width = 960;
    uint32_t                 height = 540;
    uint32_t                 gtPaths = 512;   ///< paths a pixel the GT holds
    uint32_t                 gtBounces = 6;
    double                   time = 0.0;
    std::string              prim;            ///< only meshes at or under it are converted
    std::vector<std::string> hidden;          ///< left out of everything, as --hide
    std::vector<std::string> materials;       ///< only these (prim paths or names); empty is every one
    /// Another sky for every frame: "white" (a constant of one), or an image
    /// file; the stage's other lights off. Empty: the stage as it is.
    std::string              sky;
    /// The clouds' frames blended as a standard viewer blends them, sRGB
    /// values summed (`athenea:splatDisplayBlend`): what `athenea flatten
    /// --validate` measures a standard file under.
    bool                     displayBlend = false;
    /// One cloud stands for every measured material at once (a flattened
    /// stage): each frame of it switches off the meshes of all of them, not
    /// only the material measured.
    bool                     wholeCloud = false;
};

/// A layer over `source`: nested overs down to each prim of `off`, switched
/// off, and to each of `bodies`, given those attribute lines -- as USD
/// composes them. `tail` is written after.
[[nodiscard]] std::string overLayer(const std::string& source, const std::vector<std::string>& off,
                                    const std::map<std::string, std::string>& bodies, char upAxis,
                                    double metersPerUnit, const std::string& tail);

/// Converts `stage` with `hidden` left out into `output`, returning the
/// gaussians written: the caller's own conversion, with everything else it
/// was asked. `stage` is the job's, with the job's `hidden` switched off in a
/// layer over it, so the bake does not see them either.
using ValidateConvert = std::function<Result<uint32_t>(const std::string& stage, const std::vector<std::string>& hidden,
                                                       const std::string& output)>;

[[nodiscard]] Result<void> validateConversion(const ValidateJob& job, gpu_host::Context& context,
                                              gpu::ShaderLibrary& library, aofx::Effect& measure,
                                              const ValidateConvert& convert);

}   // namespace athenea::cli
