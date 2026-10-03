// Copyright (c) 2026 jesus luque.
//
// A cloud carried by a skeleton: the gaussians are built once in the bind
// pose, each keeps the joints that move it, and this puts them where the
// skeleton is at an instant. The counterpart of `geom::Skinner`, which does
// the same for a mesh's points -- and it takes the same inputs, in the same
// layout, because they come from the same place.
#pragma once

#include <array>
#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::scene {

struct GpuSplats;

/// What moves a cloud: the joints each gaussian is held by, the skeleton's
/// transforms at this instant, and the three matrices that take a point from
/// the cloud's own space to the skeleton's and back.
struct SplatSkinInput {
    const GpuSplats*   rest = nullptr;
    /// `(joint, weight)` float2s, `perSplat` of them a gaussian.
    const gpu::Buffer* influences = nullptr;
    uint32_t           perSplat = 4;
    /// Sixteen floats a joint, row major as `GfMatrix4f` holds them; they go
    /// over transposed, exactly as `geom::Skinner` sends a mesh's.
    const gpu::Buffer* skinningXforms = nullptr;
    /// The same joints at the shutter's other end, when a frame wants to know
    /// what moved. Null: the cloud is still and no displacement is written.
    const gpu::Buffer* skinningXformsEnd = nullptr;
    std::array<float, 16> geomBindTransform{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                            0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 16> skelLocalToWorld{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
    std::array<float, 16> primWorldToLocal{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F};
};

class SplatSkinner {
public:
    [[nodiscard]] static Result<SplatSkinner> create(gpu::ShaderLibrary& library);

    /// Writes the posed positions and shapes into buffers of the rest cloud's
    /// own size. Everything else a gaussian carries -- its opacity, its
    /// colour, its harmonics, its PBR channels -- is the bind pose's and is
    /// not touched, which is what makes a frame cost one kernel.
    /// With `motion`, and `SplatSkinInput::skinningXformsEnd` set, it also
    /// writes what the shutter moved each gaussian -- two words of halves,
    /// in the cloud's own space, which is what the rasteriser's blur takes.
    /// With `normals`, and a rest cloud that keeps shading normals
    /// (`GpuSplats::hasNormals`), those are turned by the same blend as the
    /// frame and written there, one word a gaussian.
    [[nodiscard]] Result<void> skin(gpu::CommandBatch& batch, const SplatSkinInput& input,
                                    gpu::Buffer& positions, gpu::Buffer& shape,
                                    gpu::Buffer* motion = nullptr, gpu::Buffer* normals = nullptr);

private:
    gpu::ComputeKernel kernel_;
};

}   // namespace athenea::scene
