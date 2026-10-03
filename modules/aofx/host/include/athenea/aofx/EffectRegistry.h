// Copyright (c) 2026 jesus luque.
//
// The AOFX registry, as this engine uses it.
//
// The registry is aopenfx's (`aofx::host::EffectRegistry`): discovery, the ABI
// and build-tag gates, kernel registration, every refusal kept with its
// reason. What this adds is one constructor that hands it what this engine
// declares (Host.cpp) and one `scan` that takes this engine's context.
#pragma once

#include "aofx_host/Channels.h"
#include "aofx_host/EffectRegistry.h"

namespace athenea::gpu_host {
class Context;
}

namespace athenea::aofx_host {

using BundleReport = aofx::host::BundleReport;

class EffectRegistry : public aofx::host::EffectRegistry {
public:
    EffectRegistry();

    /// Loads everything on the search path. A context with no gpe device
    /// (Vulkan) loads nothing: an effect is a kernel, and a kernel needs a
    /// device.
    void scan(gpu_host::Context* context);
    using aofx::host::EffectRegistry::scan;
};

/// The host's own kernel that puts back the channels an effect was not applied
/// to, and the four switches every effect gets.
using aofx::host::kChannelsKernel;
using aofx::host::kChannelRedParam;
using aofx::host::kChannelGreenParam;
using aofx::host::kChannelBlueParam;
using aofx::host::kChannelAlphaParam;

/// What this engine declares to the host, made once for the process.
[[nodiscard]] const aofx::host::Capabilities& capabilities();

}   // namespace athenea::aofx_host
