// Copyright (c) 2026 jesus luque.
//
// Where a cloud is, for a camera to frame it: its core, without the few
// splats a capture leaves far out.
#pragma once

#include "athenea/core/Result.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::render {

/// The box of the central `share` of a cloud's splats along each axis: on each
/// axis the positions are sorted on the device and the two at (1 - share) / 2
/// from either end read back. A capture's sky -- a few splats thousands of
/// units out -- framed a shot in which the scene was a dot; this leaves it
/// out. A cloud of no splats gives an empty box.
[[nodiscard]] Result<scene::Bounds> coreBounds(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud,
                                               double share = 0.96);

}   // namespace athenea::render
