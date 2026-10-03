// Copyright (c) 2026 jesus luque.
#pragma once

#include <cstdint>

#include <pxr/base/gf/matrix4d.h>

#include "athenea/render/Camera.h"

namespace athenea::usd {

/// The engine's projection from what a Hydra host decided: USD's world-to-view
/// (row vectors, camera looking down -Z) and a GL-style projection matrix, for
/// an image of `width` x `height`. Reading the matrices rather than the camera
/// prim is what makes a free camera (usdview's) and every conform policy the
/// host applied come out the way the host drew its own viewport.
[[nodiscard]] render::Projection projectionFromHydra(const pxr::GfMatrix4d& worldToView,
                                                     const pxr::GfMatrix4d& projection,
                                                     uint32_t width, uint32_t height);

/// A column-vector engine matrix from a USD row-vector one.
[[nodiscard]] render::Mat4 fromUsd(const pxr::GfMatrix4d& m);

}   // namespace athenea::usd
