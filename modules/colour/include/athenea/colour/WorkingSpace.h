// Copyright (c) 2026 jesus luque.
//
// Which working-space record (gpu/WorkingSpace.h) a name asks for. The name
// is resolved as every other colour-space name is (ColourNames), so
// `lin_rec709_scene`, `lin_rec709`, `Linear Rec.709 (sRGB)` and whatever
// else the config calls that space are one space.
//
// Only linear Rec.709 is rendered in yet (phase 2a of the colour plan): its
// record is the table of constants the kernels held, and any other space is
// refused here, so a caller says so and keeps rendering in Rec.709.
#pragma once

#include <string>

#include "athenea/colour/ColourNames.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/WorkingSpace.h"

namespace athenea::colour {

/// The record for the working space `name` names. Empty or `auto`: the
/// default, lin_rec709_scene. Unsupported for a space that is not linear
/// Rec.709 (data, sRGB-encoded, ACEScg, a name no config knows), with a
/// message that names it.
[[nodiscard]] Result<gpu::WorkingSpace> workingSpaceRecord(const ColourNames& names, const std::string& name);

}   // namespace athenea::colour
