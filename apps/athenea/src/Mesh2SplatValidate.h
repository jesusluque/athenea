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
#include <filesystem>
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

struct ValidateRow;

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
    std::vector<std::string> skipMaterials;   ///< not these (prim paths or names): left meshes in every frame
    /// Another sky for every frame: "white" (a constant of one), or an image
    /// file; the stage's other lights off unless `skyKeepsLights`. Empty: the
    /// stage's own domes.
    std::string              sky;
    /// The domes turned about the stage's up axis, in degrees (skyLayer).
    double                   skyRotate = 0.0;
    /// The stage's other lights stay on under `sky` (flatten --hdri).
    bool                     skyKeepsLights = false;
    /// A stage drawn in place of the meshes' raster as the middle of each
    /// side by side and measured beside the cloud: what `athenea flatten`
    /// compares a file with -- the TX clouds under the same sky, athenea's
    /// own raster. Under the job's sky already. Empty: the meshes' raster.
    std::string              middleStage;
    std::string              middleLabel = "mesh raster";
    /// Where the GT is kept and looked for: the directory when empty. Several
    /// validations of one stage under one sky share one path traced frame.
    std::string              gtDirectory;
    /// Filled with each row when not null.
    std::vector<ValidateRow>* results = nullptr;
    /// The clouds' frames blended as a standard viewer blends them, sRGB
    /// values summed (`athenea:splatDisplayBlend`): what `athenea flatten
    /// --validate` measures a standard file under.
    bool                     displayBlend = false;
    /// One cloud stands for every measured material at once (a flattened
    /// stage): each frame of it switches off the meshes of all of them, not
    /// only the material measured.
    bool                     wholeCloud = false;
};

/// One measured material (or "(all)", every one of them at once, for a
/// whole cloud): the cloud's frame and the middle one, each against the GT
/// over the material's pixels.
struct ValidateRow {
    std::string material;
    uint32_t    meshes = 0;
    uint32_t    splats = 0;
    double      cover = 0.0;   ///< share of the frame
    double      relMse = 0.0, p99 = 0.0;
    double      mean[3] = {0.0, 0.0, 0.0};
    double      meanGt[3] = {0.0, 0.0, 0.0};
    double      middleRelMse = 0.0, middleP99 = 0.0;
    double      middleMean[3] = {0.0, 0.0, 0.0};
    std::string error;
};

/// WHAT A SKY CHOICE DOES TO A STAGE, as overLayer's arguments: the domes
/// given `sky` ("white", an image file, or "" for their own) and turned by
/// `rotate` degrees about the up axis; every other light off unless
/// `keepLights`. A stage with no dome gets one (in `tail`) when `addDome`,
/// and is an error otherwise. `off` is added to, not replaced.
struct SkyLayer {
    std::map<std::string, std::string> bodies;
    std::string                        tail;
    std::string                        name;   ///< for file names: "white", the image's stem, "+r<deg>", "+lit"
};
[[nodiscard]] Result<SkyLayer> skyLayer(const std::filesystem::path& source, const std::string& sky, double rotate,
                                        bool keepLights, char upAxis, double time, bool addDome,
                                        std::vector<std::string>& off);

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
