// Copyright (c) 2026 jesus luque.
//
// A LINEAR CLOUD'S COLOURS, BROUGHT INTO THE WORKING SPACE ONCE.
//
// A cloud that holds light names the linear space it holds it in
// (GpuSplats::colourSpace; lin_rec709_scene where it does not). Where that
// is not the working space, the 3x3 between them -- built on the device from
// the function OCIO compiles (colour::ColourCompiler::matrix) -- is applied
// to its base colours, harmonics, emission and indirect transfer by
// scene/cloud_to_working.slang, once, as the cloud is decoded. A capture is
// left alone: its sRGB is brought in a splat at a time when it is evaluated.
#pragma once

#include <cstdint>
#include <optional>

#include "athenea/colour/ColourCompiler.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::scene {

class CloudColour {
public:
    [[nodiscard]] static Result<CloudColour> create(gpu::ShaderLibrary& library);

    /// The matrix `splats` needs into `colour`'s working space; none for a
    /// capture, for a cloud already in it, and for a space no config knows
    /// (warned of once by the names).
    [[nodiscard]] Result<std::optional<colour::ColourMatrix>> matrixFor(const GpuSplats& splats,
                                                                       colour::ColourCompiler& colour) const;
    /// Records the conversion of splats `first` .. `first + count` of
    /// `splats` by `matrix` into `batch`: what a store does as a chunk lands.
    void record(gpu::CommandBatch& batch, const GpuSplats& splats, const colour::ColourMatrix& matrix,
                uint32_t first, uint32_t count) const;
    /// The whole cloud, submitted and waited for; `colourSpace` then names
    /// the working space, so a second call does nothing. True where it
    /// converted.
    [[nodiscard]] Result<bool> toWorking(GpuSplats& splats, colour::ColourCompiler& colour) const;

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel kernel_;
};

}   // namespace athenea::scene
