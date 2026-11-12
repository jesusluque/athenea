// Copyright (c) 2026 jesus luque.
//
// A SCENE FOUND A LOOK AT A TIME (shaders/athenea/neural/scene_discover.slang):
// which of a piece's gaussians each look saw first. A look is a frame from a
// camera at a time; what it gives beside its picture is the gaussians it
// saw, and the scene as it is known is those the looks so far have found.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/render/Camera.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

/// What one look did to one piece.
struct LookCount {
    uint32_t found = 0;   ///< gaussians it saw that no look had
    uint32_t again = 0;   ///< gaussians it saw that the scene already held
};

/// What a frame left in the scene (shaders/athenea/neural/scene_frame.slang).
struct FrameLook {
    uint32_t           surfaces = 0;   ///< the frame's pixels that hold a surface
    uint32_t           held = 0;       ///< of those, the ones the scene already had at that depth
    uint32_t           made = 0;       ///< gaussians made: one a pixel the scene did not have
    /// The gaussians as a 3DGS PLY keeps them, `kFrameRecord` floats each.
    std::vector<float> records;
};

/// scene_frame.slang's kFrameRecord: INRIA's 62 floats a gaussian.
inline constexpr uint32_t kFrameRecord = 62;

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

    /// GAUSSIANS MADE FROM A FRAME: a gaussian a pixel of `colour` (float4,
    /// linear, bottom row first) whose `depth` (a float, view z; 0 where
    /// nothing is) the scene does not already hold -- `knownColour` and
    /// `knownDepth` being the scene as it is known drawn through the same
    /// `projection`, or null where nothing is known yet. A pixel is held
    /// where the two depths are within `tolerance` of the frame's own; a
    /// gaussian is a disc in the pixel's surface, `spread` pixels wide.
    [[nodiscard]] Result<FrameLook> fromFrame(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                              uint32_t height, const Projection& projection,
                                              const gpu::Buffer* knownColour, const gpu::Buffer* knownDepth,
                                              float tolerance = 0.03F, float spread = 0.8F);

    /// WHERE A FRAME SEES THROUGH THE SCENE: a float a pixel, 1 where the scene
    /// drawn through the frame's camera (`knownColour`, `knownDepth`) has a
    /// surface nearer than the frame's own by more than `tolerance`, or
    /// something where the frame has nothing. `pixels` is how many.
    [[nodiscard]] Result<gpu::Buffer> seenThrough(const gpu::Buffer& colour, const gpu::Buffer& depth, uint32_t width,
                                                  uint32_t height, const gpu::Buffer& knownColour,
                                                  const gpu::Buffer& knownDepth, float tolerance, uint32_t& pixels);

    /// The gaussians of one piece that a frame saw through: `lifted` is that
    /// mask carried to the scene's gaussians by the blend (SplatLift.h's
    /// sums), `base` the piece's first slot. One most of whose weight fell on
    /// seen-through pixels is stamped `stamp` in `gone`; how many were.
    [[nodiscard]] Result<uint32_t> markGone(const gpu::Buffer& lifted, uint32_t base, gpu::Buffer& gone,
                                            uint32_t count, uint32_t stamp, float threshold = 0.05F);

    /// A piece without the gaussians its record `gone` stamps: the same
    /// cloud, those given no opacity.
    [[nodiscard]] Result<scene::GpuSplats> without(const scene::GpuSplats& cloud, const gpu::Buffer& gone);

    /// A buffer of `words` zeros: what a look's frame sums its contributions into.
    [[nodiscard]] Result<gpu::Buffer> zeros(uint32_t words);

private:
    void bind(rhi::ShaderCursor cursor) const;

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  clear_, mark_, count_, mask_, gone_, frameValid_, framePack_, frameThrough_;
    std::optional<gpu::PrefixSum> prefix_;
    gpu::Buffer         counters_, spareWords_, spareFloats_;
};

}   // namespace athenea::render
