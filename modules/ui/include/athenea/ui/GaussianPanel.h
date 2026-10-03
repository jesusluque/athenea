// Copyright (c) 2026 jesus luque.
//
// THE GAUSSIANS PANEL: what the splats on screen are and what the frame did
// with them, described once for every front end (Controls.h).
//
// Its numbers come from a GaussianReport read every frame. What the device
// counted is a frame or two late, and the panel says which frame it is about
// rather than presenting it as the frame on screen.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "athenea/ui/Controls.h"
#include "athenea/ui/GaussianReport.h"

namespace athenea::ui {

/// How many clouds the panel has a row for; past that, the totals still count them.
inline constexpr size_t kGaussianPanelClouds = 24;

/// Where the panel reads its numbers: the front end's copy of this frame's
/// report, taken once a frame and read by every row.
using GaussianSource = std::function<const GaussianReport&()>;

/// The panel over `report`, read every time a row is drawn. `timeStages` is
/// the switch that asks the rasteriser to time each stage (which makes each
/// one wait for the device); it must outlive the panel.
[[nodiscard]] Panel gaussianPanel(GaussianSource report, bool& timeStages);

/// 1234567 as "1,234,567".
[[nodiscard]] std::string thousands(uint64_t value);
/// A size in bytes as a person reads it: "812 KB", "145.2 MB", "1.31 GB".
[[nodiscard]] std::string byteSize(uint64_t bytes);
/// One cloud, as the lines its row shows.
[[nodiscard]] std::string describeCloud(const GaussianCloud& cloud, const GaussianReport& report);

}   // namespace athenea::ui
