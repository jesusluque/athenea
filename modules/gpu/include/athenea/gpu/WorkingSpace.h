// Copyright (c) 2026 jesus luque.
//
// THE WORKING SPACE AS EVERY KERNEL READS IT.
//
// What the kernels used to take for granted -- that linear light is Rec.709:
// BT.709's luminance weights where a lobe, a light or a pixel is weighed by
// its brightness, the XYZ-to-sRGB matrix of a blackbody, a capture's sRGB
// decoded straight into the working space -- they read from this record.
// Four 3x3s and a row of weights: every linear RGB space is a 3x3 from
// every other, so no kernel imports an OCIO function for it. Linear Rec.709's
// record is a table (linRec709Scene); any other's is built on the device from
// the functions OCIO compiles (colour::workingSpaceRecord).
//
// It is the Slang `WorkingSpace` of shaders/athenea/common/working_space.slang,
// field for field, and reaches a kernel as the uniform `gWorkingSpace` in the
// kernel's global constant block -- the block its other uniforms already
// fill, so the record costs no buffer (a path tracer at Metal's 31 is why).
// Every kernel type binds it by name at dispatch, from its device
// (Device::workingSpace), when the kernel's program has it; a kernel that
// imports none of the modules that read it does not.
//
// Only float4 and uint4 members: a struct of 16-byte vectors is laid out
// the same in a Metal, CUDA, Vulkan and D3D constant block, which is what
// lets the record be set as one value.
#pragma once

#include <array>
#include <cstdint>

#include <slang-rhi/shader-cursor.h>

namespace athenea::gpu {

/// flags.x: the working space's primaries are Rec.709's, so sRGB once
/// decoded is already the working space and neither `fromSrgb` nor `toSrgb`
/// is applied (an identity is not even multiplied by: the bits stay those of
/// the curve).
inline constexpr uint32_t kWorkingSpaceRec709Primaries = 1u;

struct WorkingSpace {
    /// Working RGB to CIE XYZ (D65), as the rows of a 3x3; w is 0.
    std::array<float, 4> toXyz0{}, toXyz1{}, toXyz2{};
    /// CIE XYZ (D65) to working RGB: what a blackbody's chromaticity becomes.
    std::array<float, 4> fromXyz0{}, fromXyz1{}, fromXyz2{};
    /// sRGB decoded (linear Rec.709) to working RGB: a capture's splats, a
    /// PLY's 8-bit point colours.
    std::array<float, 4> fromSrgb0{}, fromSrgb1{}, fromSrgb2{};
    /// Working RGB to linear Rec.709: what the display's own views (Standard,
    /// AgX, ACES 2.0) take, since their matrices are Rec.709's.
    std::array<float, 4> toSrgb0{}, toSrgb1{}, toSrgb2{};
    /// The weights of a luminance, in xyz; w is 0.
    std::array<float, 4> luminance{};
    /// x: kWorkingSpaceRec709Primaries. y, z, w: 0.
    std::array<uint32_t, 4> flags{};

    friend bool operator==(const WorkingSpace&, const WorkingSpace&) = default;
};
static_assert(sizeof(WorkingSpace) == 224, "WorkingSpace is fourteen 16-byte rows, as the Slang struct");

/// The record of `lin_rec709_scene`: the constants the kernels held before
/// the record existed, written down -- BT.709's weights as 0.2126, 0.7152,
/// 0.0722 (not the Y row of the matrix, 0.2126729, 0.7151522, 0.0721750,
/// which would move every pixel a lobe choice or a light's share touches),
/// the blackbody's XYZ-to-sRGB rows as they were, and their inverse for
/// toXyz. A table, not arithmetic.
[[nodiscard]] const WorkingSpace& linRec709Scene() noexcept;

/// The uniform's name in every kernel.
inline constexpr const char* kWorkingSpaceUniform = "gWorkingSpace";

/// Sets `gWorkingSpace` under `root` when the program has it; nothing when
/// it does not.
void bindWorkingSpace(rhi::ShaderCursor root, const WorkingSpace& space);

}   // namespace athenea::gpu
