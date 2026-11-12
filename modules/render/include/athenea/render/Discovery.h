// Copyright (c) 2026 jesus luque.
//
// A SCENE FOUND A LOOK AT A TIME (shaders/athenea/neural/scene_discover.slang):
// which of a piece's gaussians each look saw first. A look is a frame from a
// camera at a time; what it gives beside its picture is the gaussians it
// saw, and the scene as it is known is those the looks so far have found.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

/// What one look did to one piece.
struct LookCount {
    uint32_t found = 0;   ///< gaussians it saw that no look had
    uint32_t again = 0;   ///< gaussians it saw that the scene already held
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

    /// A buffer of `words` zeros: what a look's frame sums its contributions into.
    [[nodiscard]] Result<gpu::Buffer> zeros(uint32_t words);

private:
    void bind(rhi::ShaderCursor cursor) const;

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  clear_, mark_, count_, mask_;
    gpu::Buffer         counters_, spareWords_, spareFloats_;
};

}   // namespace athenea::render
