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

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"

namespace athenea::gpu {
class CommandBatch;
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

/// A compiled function as a 3x3, taken on the device at the basis
/// (colour/colour_matrix.slang): four float4 rows, the matrix's three and,
/// in the fourth's x, how far the function is from linear.
struct ColourMatrix {
    gpu::Buffer rows;
};

/// Where colours lie in a buffer of words (`convertWords`): `count` of them,
/// the first at word `first`, each `stride` words after the one before;
/// three floats each, or four halves in two words (`halves`: r|g, b|a).
struct ColourWordsRun {
    uint32_t first = 0;
    uint32_t count = 0;
    uint32_t stride = 3;
    bool     halves = false;
    /// Where the colours are scattered: `count` uints, the first word of
    /// each (`first` and `stride` then unused).
    const gpu::Buffer* list = nullptr;
};

class ColourCompiler {
public:
    /// `configUri`: a path, or one of OCIO's built-in configs (`ocio://...`).
    /// Without OpenColorIO the config is ignored and the names resolve by table.
    /// `workingSpace`: the space the engine renders in, as any config names
    /// it, or one of the config's roles (`role:scene_linear`, or the role's
    /// bare name). Every function's destination and the meaning of
    /// SpaceKind::Working follow it. A data space, or a name neither this
    /// config nor the studio config knows, is InvalidArgument.
    [[nodiscard]] static Result<std::unique_ptr<ColourCompiler>> create(gpu::ShaderLibrary& library,
                                                                       const std::string& configUri = kStudioConfig,
                                                                       const std::string& workingSpace = kWorkingSpace);

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

    /// `src` into `dst` as a 3x3 (ColourMatrix), built on the device by a
    /// generated kernel that evaluates the compiled function; cached.
    [[nodiscard]] Result<ColourMatrix> matrix(const std::string& src, const std::string& dst);

    /// Converts the colours `run` places in `words` from `space` -- as a
    /// material, a light or a primvar names it; empty is lin_rec709_scene,
    /// USD's default -- into the working space, recorded into `batch`. False
    /// when nothing is to do: the space is the working space, data, or a name
    /// no config knows (warned of once). The function's LUTs and the kernel
    /// stay alive in this compiler, which must outlive the batch's submit.
    [[nodiscard]] Result<bool> convertWords(gpu::CommandBatch& batch, const gpu::Buffer& words,
                                            const std::string& space, const ColourWordsRun& run);
    /// Whether a colour authored in `space` (empty: lin_rec709_scene) is
    /// converted by `convertWords`: what a caller asks before it gathers runs.
    [[nodiscard]] bool converts(const std::string& space) const;

    [[nodiscard]] const ColourNames& names() const noexcept { return names_; }
    [[nodiscard]] gpu::ShaderLibrary& library() const noexcept { return *library_; }
    [[nodiscard]] const std::string& configUri() const noexcept { return uri_; }
    /// The working space as it was asked for (`create`'s `workingSpace`).
    [[nodiscard]] const std::string& workingAsked() const noexcept { return workingAsked_; }

private:
    ColourCompiler() = default;

    gpu::ShaderLibrary*                       library_ = nullptr;
    gpu::Device*                              device_ = nullptr;
    std::string                               uri_;
    std::shared_ptr<const ColourConfig>       config_;
    ColourNames                               names_;
    std::map<uint64_t, ColourFunction>        cache_;
    std::string                               workingAsked_;
    struct MatrixKernel {
        ColourFunction     function;
        ColourMatrix       matrix;
    };
    std::map<uint64_t, MatrixKernel>          matrices_;
    struct WordsKernel {
        ColourFunction     function;
        gpu::ComputeKernel kernel;
    };
    std::map<std::string, WordsKernel>        words_;   ///< by the source space's resolved name
    mutable std::map<std::string, bool>       converts_;   ///< `converts`, by the name as given
};

/// FNV-1a, 64 bits: what generated modules are named by.
[[nodiscard]] uint64_t fnv1a(const std::string& text) noexcept;

}   // namespace athenea::colour
