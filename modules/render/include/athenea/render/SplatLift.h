// Copyright (c) 2026 jesus luque.
//
// A mask carried from a screen to the gaussians and from them to another
// screen, through the trace a frame keeps (`RenderSettings::trace`,
// shaders/athenea/splat/splat_lift.slang): what a key or a segmentation made
// on one camera's picture is on every other camera's.
#pragma once

#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/render/TileRasterizer.h"

namespace athenea::render {

/// What the gaussians of a frame's instances have been told by the masks
/// lifted so far: two uints a slot, (weight * mask, weight) summed over every
/// pixel of every lifted frame, in 1/1024 of a pixel. A slot is a gaussian's
/// place among a frame's instances (`FrameStats::slotBases`), so every frame
/// lifted into one of these, and every matte drawn from it, is of the same
/// instances in the same order.
struct SplatLift {
    gpu::Buffer sums;
    uint32_t    slots = 0;
};

/// A lift nothing has been lifted into, for a frame of `slots` slots
/// (`FrameStats::splats`).
[[nodiscard]] Result<SplatLift> makeSplatLift(gpu::Device& device, uint32_t slots);

/// Adds one frame's mask to the lift. `traced` is a frame rendered with
/// `RenderSettings::trace`; `mask` is one float a pixel of it, 0 to 1, bottom
/// row first. A gaussian holds up to four million pixels of weight before its
/// sum could wrap: some five hundred full 4K frames of one gaussian filling
/// the screen.
[[nodiscard]] Result<void> liftMask(gpu::ShaderLibrary& library, const RenderTargets& traced,
                                    const gpu::Buffer& mask, SplatLift& lift);

/// The same, from every gaussian of every pixel's walk rather than the few a
/// trace names: sets `settings` so that the frame rendered with them adds
/// `mask` (one float a pixel of that frame) to the lift as it blends. The
/// frame is of the lift's instances; nothing else of it is changed.
void liftWhileBlending(RenderSettings& settings, SplatLift& lift, const gpu::Buffer& mask);

struct MatteSettings {
    /// A gaussian is in the mask or out of it (its lifted share against
    /// `threshold`), rather than in it by its share.
    bool  hard = false;
    float threshold = 0.5F;
};

/// The mask as another frame sees it: one float a pixel of `traced`, how far
/// the gaussians that pixel is made of are in the mask. `matte` is made where
/// it is not already a buffer of that size.
[[nodiscard]] Result<void> drawMatte(gpu::ShaderLibrary& library, const RenderTargets& traced,
                                     const SplatLift& lift, const MatteSettings& settings,
                                     gpu::Buffer& matte);

}   // namespace athenea::render
