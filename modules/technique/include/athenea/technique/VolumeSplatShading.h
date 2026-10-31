// Copyright (c) 2026 jesus luque.
//
// WHAT A MEDIUM'S GAUSSIANS SEND TOWARDS THE CAMERA, BAKED A FRAME (proposal
// 105 §3). The rasteriser draws a medium by the exact optical depth of each
// pixel's ray through each gaussian (splat/splat_volume.slang) and the
// radiance written here: emission (a constant and a black body over the
// gaussian's temperature), single scattering of every light with the
// transmittance towards it read from the cloud shadow map at the gaussian's
// centre, a dome's term, and Wrenninge's octaves for what scatters more than
// once. The kernel is shaders/athenea/technique/volume_splat_shade.slang; the
// black body's table (volume_blackbody.slang) is built on the device the first
// time it is wanted.
#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/SplatVolume.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::technique {

class SplatShadowMap;

struct VolumeShadeJob {
    const scene::GpuSplats* cloud = nullptr;   ///< a medium (`hasVolume`)
    std::array<float, 12>   objectToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    render::SplatVolume     look;
    uint64_t                categories = 0;     ///< which lights reach it
    const gpu::Buffer*      lights = nullptr;   ///< light::LightTable's records
    uint32_t                lightCount = 0;
    /// The frame's cloud shadow map, with this medium among its casters.
    /// Null or not valid: nothing is in any light's way.
    const SplatShadowMap*   shadowMap = nullptr;
    std::array<float, 3>    eye{0.0F, 0.0F, 0.0F};   ///< the camera, in the world
    /// Multiple scattering's octaves (Wrenninge; how many is the look's):
    /// what each takes of the one before -- the extinction towards the
    /// light, the energy and the anisotropy.
    float                   octaveExtinction = 0.5F;
    float                   octaveEnergy = 0.5F;
    float                   octaveAnisotropy = 0.5F;
};

class VolumeSplatShading {
public:
    [[nodiscard]] static Result<VolumeSplatShading> create(gpu::ShaderLibrary& library);

    /// Queued into `batch`: `job.cloud`'s radiance, two uint a gaussian (rgb
    /// halves), into `out`, which is grown to fit.
    [[nodiscard]] Result<void> shade(gpu::CommandBatch& batch, const VolumeShadeJob& job, gpu::Buffer& out);

    static constexpr uint32_t kBlackbodyEntries = 1024;
    static constexpr float    kBlackbodyMaxKelvin = 16000.0F;

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel shade_;
    gpu::ComputeKernel blackbodyKernel_;
    gpu::Buffer        blackbody_;
    gpu::Buffer        noMap_;     ///< a word of nothing, bound where no map is
    gpu::Buffer        noLights_;  ///< a record of nothing, bound where no light is
    bool               blackbodyMade_ = false;
};

}   // namespace athenea::technique
