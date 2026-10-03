// Copyright (c) 2026 jesus luque.
//
// WHAT A PICKED PRIM'S GAUSSIANS ARE CARRYING.
//
// The Cryptomatte id a pixel hands back is also a selection: every gaussian
// with that id came from one prim. What they are made of is a question only
// the device can answer, so it is answered by a kernel and only counters come
// back (`shaders/athenea/splat/splat_id_stats.slang`).
//
// A mean is not the whole answer. A material whose roughness is a map gives
// each gaussian its own, so the range comes back beside the mean and a host
// can say so rather than pretend there is one value.
#pragma once

#include <array>
#include <cstdint>

#include "athenea/core/Result.h"

namespace athenea::gpu {
class ShaderLibrary;
}
namespace athenea::scene {
struct GpuSplats;
}

namespace athenea::render {

/// Metallic, roughness and transmission, in that order, 0 to 1.
struct SplatIdReading {
    uint32_t             count = 0;      ///< gaussians carrying the id
    bool                 hasPbr = false; ///< false: the cloud carries none, and the three below are 0
    std::array<float, 3> mean{};
    std::array<float, 3> low{};
    std::array<float, 3> high{};
};

/// Counts the gaussians of `cloud` that carry `id` and what they are made of.
/// A cloud with no Cryptomatte ids answers with a count of zero.
[[nodiscard]] Result<SplatIdReading> measureSplatId(gpu::ShaderLibrary& library,
                                                    const scene::GpuSplats& cloud, uint32_t id);

}   // namespace athenea::render
