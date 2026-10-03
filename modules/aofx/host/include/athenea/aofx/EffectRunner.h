// Copyright (c) 2026 jesus luque.
//
// The host side of `aofx::Gpu` for one render, as this engine uses it.
//
// The runner is aopenfx's (`aofx::host::EffectRunner`): the device verbs on
// gpe, and the media and model verbs answering "not available in this host"
// because this engine declares neither (Host.cpp). What this adds is one
// constructor that takes this engine's context.
//
// Must run on the context's GPU thread (inside gpu_host::Context::run), on a
// context whose `compute()` is not null -- a caller checks that first, as
// `athenea aofx` does.
#pragma once

#include "aofx_host/EffectRunner.h"

namespace athenea::gpu_host {
class Context;
}

namespace athenea::aofx_host {

class EffectRunner final : public aofx::host::EffectRunner {
public:
    explicit EffectRunner(gpu_host::Context& context);
};

}   // namespace athenea::aofx_host
