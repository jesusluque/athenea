// Copyright (c) 2026 jesus luque.
//
// THE LTC TABLE (shaders/athenea/light/ltc.slang): for a GGX lobe of each
// roughness and each angle to the eye, the inverse of the linearly
// transformed cosine that stands in for it, fitted on the device the first
// time a process asks (shaders/athenea/light/ltc_fit.slang) -- a few
// hundred milliseconds once, then a 64 x 64 texture every frame reads. The
// published table's licence was not one to read plainly, so the fit is ours.
#pragma once

#include <cstdint>
#include <optional>

#include <slang-rhi.h>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"

namespace athenea::gpu {
class ShaderLibrary;
class Device;
}

namespace athenea::light {

class LtcTable {
public:
    static constexpr uint32_t kSize = 64;

    /// Fits the table; waits for the device once.
    [[nodiscard]] static Result<LtcTable> create(gpu::ShaderLibrary& library);

    /// The table, RGBA32F, x the roughness and y the angle (ltcCoordinates).
    [[nodiscard]] rhi::ITextureView* view() const noexcept { return view_.get(); }
    /// The fit's error at each texel (m11, m22, m13, error), for a test.
    [[nodiscard]] const gpu::Buffer& state() const noexcept { return state_; }

private:
    gpu::Texture                      texture_;
    rhi::ComPtr<rhi::ITextureView>    view_;
    gpu::Buffer                       state_;
};

}   // namespace athenea::light
