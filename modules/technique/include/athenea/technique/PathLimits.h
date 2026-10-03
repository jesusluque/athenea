// Copyright (c) 2026 jesus luque.
//
// When a path traced by technique::PathTracer stops before its total of
// bounces. Apart from PathTracer.h so that a front end (StageRenderer, the
// viewer's options) can carry it without the tracer's headers.
#pragma once

#include <cstdint>

namespace athenea::technique {

/// A bounce limit of a kind that does not limit: the total alone does.
inline constexpr uint32_t kUnlimitedBounces = 0xFFFFFFFFU;

/// WHEN A PATH STOPS, BEYOND ITS TOTAL OF BOUNCES.
///
/// Russian roulette: past `rouletteDepth` bounces a path whose throughput has
/// fallen below one goes on with that probability and is weighed up by its
/// inverse, so the mean is unchanged and the dim paths stop paying for rays
/// (pbrt-v4's `rrBeta`, Falcor's PathTracer). 0 turns it off.
///
/// Bounce limits by kind, after Falcor's PathState (BSD-3, NVIDIA): a path
/// counts its diffuse reflections, its glossy and specular ones, its
/// transmissions (any lobe whose direction crosses the surface) and its
/// scatterings in a medium, and goes on by a lobe only while that kind is
/// under its limit. Unlimited by default: the total and the glass's free
/// crossings decide, as they always did.
struct PathLimits {
    uint32_t rouletteDepth = 3;
    uint32_t diffuse = kUnlimitedBounces;
    uint32_t glossy = kUnlimitedBounces;
    uint32_t transmission = kUnlimitedBounces;
    uint32_t volume = kUnlimitedBounces;
    [[nodiscard]] bool operator==(const PathLimits&) const = default;
};

}   // namespace athenea::technique
