// Copyright (c) 2026 jesus luque.
#include "aofx_host/Capabilities.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRunner.h"
#include "athenea/core/Log.h"
#include "athenea/gpu_host/Context.h"

namespace athenea::aofx_host {
namespace {

/// The host's lines through this engine's log.
class HostLogger final : public aofx::host::Logger {
public:
    void info(const std::string& line) override { log::info("{}", line); }
    void warn(const std::string& line) override { log::warn("{}", line); }
    void error(const std::string& line) override { log::error("{}", line); }
};

}   // namespace

const aofx::host::Capabilities& capabilities() {
    // A logger and nothing else: no media stack, no inference runtime. The
    // clip and model verbs answer "not available in this host" and say why,
    // from the host's own code. Never destroyed; the process registry keeps a
    // pointer to it.
    static const aofx::host::Capabilities* made = [] {
        auto* declared = new aofx::host::Capabilities();
        declared->log = new HostLogger();
        return declared;
    }();
    return *made;
}

EffectRegistry::EffectRegistry() : aofx::host::EffectRegistry(&capabilities()) {}

void EffectRegistry::scan(gpu_host::Context* context) {
    aofx::host::EffectRegistry::scan(context != nullptr ? context->compute() : nullptr);
}

// Qualified, and it has to be: inside this class `capabilities()` finds the
// base's member before the namespace's function, and the base's reads an
// `impl_` its constructor has not made yet -- a null dereference on every
// render, not a compile error.
EffectRunner::EffectRunner(gpu_host::Context& context)
    : aofx::host::EffectRunner(*context.compute(), aofx_host::capabilities()) {}

}   // namespace athenea::aofx_host
