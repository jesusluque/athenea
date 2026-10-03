// Copyright (c) 2026 jesus luque.
//
// OpenColorIO as a compiler. A processor -- one colour space into another, or
// a display and view -- becomes a generated Slang module exporting one
// function, `float4 atheneaCs_<hash>(float4)`, and the LUT textures that
// function samples, filled on the device from OCIO's tables. Every texel and
// pixel is transformed by a kernel that imports the module; what the host
// computes is the shader text and the LUT values (docs/decisions.md, "Colour").
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>

#include <slang-rhi.h>
#include <slang-rhi/shader-cursor.h>

#include "athenea/colour/ColourNames.h"
#include "athenea/core/Result.h"

namespace athenea::gpu {
class Device;
class ShaderLibrary;
}

namespace athenea::colour {

/// Whether this build has OpenColorIO (ATHENEA_HAVE_OCIO).
[[nodiscard]] bool ocioBuilt() noexcept;

/// One compiled colour transform: a module a kernel imports, the function it
/// exports, and what that function samples. Cheap to copy; the textures are
/// shared.
class ColourFunction {
public:
    ColourFunction() = default;
    [[nodiscard]] bool valid() const noexcept { return state_ != nullptr; }
    /// `athenea_cs_<hash>`: what a kernel's source imports.
    [[nodiscard]] const std::string& module() const noexcept;
    /// `atheneaCs_<hash>`: float4 in, float4 out.
    [[nodiscard]] const std::string& entry() const noexcept;
    /// The function's text without its module line, for a check that
    /// compiles it inline.
    [[nodiscard]] const std::string& body() const noexcept;
    /// "OCIO 2.5.2: <config> / <from> -> <to>", or "/ <display> / <view>".
    [[nodiscard]] const std::string& description() const noexcept;
    /// Binds the LUTs and the uniforms (dynamic properties, read as they
    /// stand now) under the root of a kernel that imports the module.
    void bind(rhi::ShaderCursor cursor) const;

    struct State;

private:
    friend class ColourCompiler;
    std::shared_ptr<const State> state_;
};

class ColourCompiler {
public:
    /// `configUri`: a path, or one of OCIO's built-in configs (`ocio://...`).
    /// Without OpenColorIO the config is ignored and the names resolve by table.
    [[nodiscard]] static Result<std::unique_ptr<ColourCompiler>> create(gpu::ShaderLibrary& library,
                                                                       const std::string& configUri = kStudioConfig);

    /// `src` into `dst`, both as the config (or the built-in studio config)
    /// names them. Compiled once per (config, src, dst): the module is
    /// `athenea_cs_<fnv1a(cacheID + src + dst)>`, its resources prefixed
    /// `athenea_<hash>_` so that two functions in one kernel do not clash.
    [[nodiscard]] Result<ColourFunction> function(const std::string& src, const std::string& dst);
    /// A display and view from `src` (OCIO's DisplayViewTransform, with
    /// `look` overriding the view's looks when not empty). Empty display or
    /// view: the config's defaults.
    [[nodiscard]] Result<ColourFunction> displayView(const std::string& src, const std::string& display,
                                                     const std::string& view, const std::string& look);

    [[nodiscard]] const ColourNames& names() const noexcept { return names_; }
    [[nodiscard]] const std::string& configUri() const noexcept { return uri_; }

private:
    ColourCompiler() = default;

    gpu::ShaderLibrary*                       library_ = nullptr;
    gpu::Device*                              device_ = nullptr;
    std::string                               uri_;
    std::shared_ptr<const ColourConfig>       config_;
    ColourNames                               names_;
    std::map<uint64_t, ColourFunction>        cache_;
};

/// FNV-1a, 64 bits: what generated modules are named by.
[[nodiscard]] uint64_t fnv1a(const std::string& text) noexcept;

}   // namespace athenea::colour
