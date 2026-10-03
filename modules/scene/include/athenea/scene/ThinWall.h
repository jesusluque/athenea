// Copyright (c) 2026 jesus luque.
#pragma once

#include <algorithm>
#include <cmath>

namespace athenea::scene {

/// What a thin sheet of index `ior` reflects head on: both faces' Fresnel,
/// with the light bouncing between them summed, `2R / (1 + R)`.
[[nodiscard]] inline float thinWallReflectance(float ior) {
    const float r = (ior - 1.0F) / (ior + 1.0F);
    const float R = r * r;
    return 2.0F * R / (1.0F + R);
}

/// THE OPACITY A THIN WALL'S GAUSSIANS TAKE, so the card they make stops what
/// the sheet reflects and lets the rest through -- the blend is the sheet's
/// transmission. A point of a card is under several gaussians at once: each
/// is `sigma` cells wide on a grid of cells, so their footprints sum to
/// `2 pi sigma^2`, and what passes a stack of faint ones is
/// `exp(-alpha * that)`. A faint gaussian is also drawn only out to
/// `2 ln(255 alpha)` of Mahalanobis radius squared, which keeps
/// `1 - 1/(255 alpha)` of it. So a card of opacity F wants
/// `alpha (1 - 1/(255 alpha)) 2 pi sigma^2 = -ln(1 - F)`, which is linear:
/// `alpha = -ln(1 - F) / (2 pi sigma^2) + 1/255`.
[[nodiscard]] inline float thinWallOpacity(float ior, double sigma) {
    const double overlap = 2.0 * 3.14159265358979 * sigma * sigma;
    const double wanted = std::min<double>(thinWallReflectance(ior), 0.999);
    return static_cast<float>(std::min(1.0, -std::log(1.0 - wanted) / std::max(overlap, 1e-3) + 1.0 / 255.0));
}

}   // namespace athenea::scene
