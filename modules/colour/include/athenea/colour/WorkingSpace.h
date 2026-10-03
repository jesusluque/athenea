// Copyright (c) 2026 jesus luque.
//
// The working-space record (gpu/WorkingSpace.h) of the space a compiler
// renders into (ColourCompiler::create's `workingSpace`).
//
// Linear Rec.709 -- by any of its names -- is the table the kernels held
// (gpu::linRec709Scene), so the default draws the frames it always drew, to
// the bit. Any other space's record is built on the device from two
// functions OCIO compiles, linear Rec.709 into the working space and back,
// taken as 3x3s at the basis (colour/working_space_build.slang); the record
// and two counts come back once, because the record is a uniform every
// kernel binds and a uniform is set from the host. A space no 3x3 reaches
// from Rec.709 -- a log, a display encoding -- is refused: a working space
// is linear and scene-referred.
#pragma once

#include <string>

#include "athenea/colour/ColourCompiler.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/WorkingSpace.h"

namespace athenea::colour {

/// The record of `compiler`'s working space. InvalidArgument for one that is
/// not linear (the probes' departure over 1e-5 relative) or whose two
/// functions are not each other's inverse (1e-4 an entry).
[[nodiscard]] Result<gpu::WorkingSpace> workingSpaceRecord(ColourCompiler& compiler);

/// The compiler and record for the working space `name` names (empty or
/// `auto`: lin_rec709_scene) in the config `configUri`: what a renderer sets
/// up when a stage or a setting names one.
struct WorkingSpaceChoice {
    std::shared_ptr<ColourCompiler> compiler;
    gpu::WorkingSpace               record;
};
[[nodiscard]] Result<WorkingSpaceChoice> chooseWorkingSpace(gpu::ShaderLibrary& library, const std::string& configUri,
                                                            const std::string& name);

}   // namespace athenea::colour
