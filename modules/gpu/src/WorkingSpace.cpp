// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/WorkingSpace.h"

namespace athenea::gpu {

namespace {

WorkingSpace makeLinRec709Scene() {
    WorkingSpace s;
    // sRGB (Rec.709 primaries, D65) to XYZ: Lindbloom's matrix, the inverse
    // of the rows below. No kernel read it before the record; it is here so
    // the record is whole.
    s.toXyz0 = {0.4124564F, 0.3575761F, 0.1804375F, 0.0F};
    s.toXyz1 = {0.2126729F, 0.7151522F, 0.0721750F, 0.0F};
    s.toXyz2 = {0.0193339F, 0.1191920F, 0.9503041F, 0.0F};
    // XYZ to linear sRGB: light/lights.slang `blackbody`, digit for digit.
    s.fromXyz0 = {3.2404542F, -1.5371385F, -0.4985314F, 0.0F};
    s.fromXyz1 = {-0.9692660F, 1.8760108F, 0.0415560F, 0.0F};
    s.fromXyz2 = {0.0556434F, -0.2040259F, 1.0572252F, 0.0F};
    // sRGB decoded is the working space: the identity, which the flag below
    // says so that no kernel multiplies by it.
    s.fromSrgb0 = {1.0F, 0.0F, 0.0F, 0.0F};
    s.fromSrgb1 = {0.0F, 1.0F, 0.0F, 0.0F};
    s.fromSrgb2 = {0.0F, 0.0F, 1.0F, 0.0F};
    s.toSrgb0 = s.fromSrgb0;
    s.toSrgb1 = s.fromSrgb1;
    s.toSrgb2 = s.fromSrgb2;
    // BT.709's weights as every kernel wrote them.
    s.luminance = {0.2126F, 0.7152F, 0.0722F, 0.0F};
    s.flags = {kWorkingSpaceRec709Primaries, 0u, 0u, 0u};
    return s;
}

}   // namespace

const WorkingSpace& linRec709Scene() noexcept {
    static const WorkingSpace kSpace = makeLinRec709Scene();
    return kSpace;
}

void bindWorkingSpace(rhi::ShaderCursor root, const WorkingSpace& space) {
    rhi::ShaderCursor field = root[kWorkingSpaceUniform];
    if (field.isValid()) {
        field.setData(&space, sizeof(space));
    }
}

}   // namespace athenea::gpu
