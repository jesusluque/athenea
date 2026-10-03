// Copyright (c) 2026 jesus luque.
//
// What the gaussians on screen are and what the frame did with them, as the
// engine reports it: the Gaussians panel's own record (ui/GaussianReport.h),
// so the engine fills exactly what the panel reads and nothing is copied
// between two descriptions of the same numbers.
#pragma once

#include "athenea/ui/GaussianReport.h"

namespace athenea::usd {

using GaussianStats = ui::GaussianReport;
using GaussianCloudStats = ui::GaussianCloud;

}   // namespace athenea::usd
