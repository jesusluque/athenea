// Copyright (c) 2026 jesus luque.
//
// When a path traced by technique::PathTracer stops before its total of
// bounces. Apart from PathTracer.h so that a front end (StageRenderer, the
// viewer's options) can carry it without the tracer's headers.
#pragma once

#include <cstdint>

namespace athenea::technique {

/// WHEN A PATH STOPS, BEYOND ITS TOTAL OF BOUNCES.
///
/// Russian roulette: past `rouletteDepth` bounces a path whose throughput has
/// fallen below one goes on with that probability and is weighed up by its
/// inverse, so the mean is unchanged and the dim paths stop paying for rays
/// (pbrt-v4's `rrBeta`, Falcor's PathTracer). 0 turns it off.
struct PathLimits {
    uint32_t rouletteDepth = 3;
    [[nodiscard]] bool operator==(const PathLimits&) const = default;
};

}   // namespace athenea::technique
