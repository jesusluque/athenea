// Copyright (c) 2026 jesus luque.
//
// The display's OpenColorIO view, as a client of colour::ColourCompiler: the
// config's display and view become a generated function (athenea_cs_<hash>),
// and the display kernel is a second generated module that imports it and
// the display's pieces. Every pixel is transformed on the device; the LUT
// values are computed by OCIO on the host, the one exception this route has
// (docs/decisions.md).
#include "athenea/technique/DisplayTransform.h"

#include <cstdio>

#include "athenea/colour/ColourCompiler.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::technique {

struct DisplayTransform::OcioState {
    gpu::ComputeKernel      kernel;
    colour::ColourFunction  function;
    /// The compiler the function came from, kept for the next view of the same config.
    std::shared_ptr<colour::ColourCompiler> compiler;
};

bool ocioBuilt() noexcept {
    return colour::ocioBuilt();
}

const gpu::ComputeKernel& ocioKernel(const DisplayTransform::OcioState& state) {
    return state.kernel;
}

void bindOcio(const DisplayTransform::OcioState& state, rhi::ShaderCursor cursor) {
    state.function.bind(cursor);
}

const std::string& DisplayTransform::ocioDescription() const noexcept {
    static const std::string none;
    return ocio_ != nullptr ? ocio_->function.description() : none;
}

const colour::ColourFunction* DisplayTransform::ocioFunction() const noexcept {
    return ocio_ != nullptr ? &ocio_->function : nullptr;
}

namespace {

/// The display kernel's entry around a compiled function: what it shows is
/// the scene's colour, or the background under another AOV, through `entry`.
std::string ocioDisplaySource(const std::string& module, const std::string& entry) {
    return "import athenea.technique.display;\nimport " + module +
           ";\n"
           "[shader(\"compute\")]\n"
           "[numthreads(16, 16, 1)]\n"
           "void ocioDisplayTransform(uint3 tid: SV_DispatchThreadID) {\n"
           "    uint pixel;\n"
           "    if (!displayPixel(tid, pixel)) {\n"
           "        return;\n"
           "    }\n"
           "    const float3 shown = displayingColour() ? " + entry + "(float4(displayScene(pixel), 1.0)).rgb\n"
           "                                            : displayOther(pixel, " + entry +
           "(float4(displayBackground(), 1.0)).rgb);\n"
           "    displayWrite(tid, shown);\n"
           "}\n";
}

}   // namespace

Result<void> DisplayTransform::setOcio(const OcioView& view) {
    if (!colour::ocioBuilt()) {
        return Error(ErrorCode::Unsupported, "no OpenColorIO in this build (scripts/build-ocio.sh)");
    }
    auto state = std::make_shared<OcioState>();
    if (ocio_ != nullptr && ocio_->compiler->configUri() == view.config) {
        state->compiler = ocio_->compiler;
    } else {
        auto compiler = colour::ColourCompiler::create(*library_, view.config);
        if (!compiler) return std::move(compiler).error();
        state->compiler = std::move(*compiler);
    }
    auto function = state->compiler->displayView(view.source, view.display, view.view, view.look);
    if (!function) return std::move(function).error();
    state->function = std::move(*function);

    // The display kernel: a module of its own, named by what it imports.
    const std::string source = ocioDisplaySource(state->function.module(), state->function.entry());
    char name[40];
    std::snprintf(name, sizeof(name), "athenea_ocio_%016llx",
                  static_cast<unsigned long long>(colour::fnv1a(source)));
    auto program = library_->loadSource(name, "module " + std::string(name) + ";\n" + source,
                                        {"ocioDisplayTransform"});
    if (!program) return std::move(program).error();
    auto kernel = gpu::ComputeKernel::create(*library_, name, "ocioDisplayTransform");
    if (!kernel) return std::move(kernel).error();
    state->kernel = std::move(*kernel);
    ocio_ = std::move(state);
    return ok();
}

}   // namespace athenea::technique
