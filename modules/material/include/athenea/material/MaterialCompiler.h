// Copyright (c) 2026 jesus luque.
//
// MaterialX graphs into Slang the engine shades with. A generator derived from
// MaterialX's own Slang generator emits one function per material structure:
// every node keeps its genslang/genglsl implementation except the BSDF nodes,
// which push lobes (athenea/material/mx), and the surface node, which leaves the
// lobes and emission where the shading kernels read them. Inputs are not baked
// in: every one is read from a float blob, so materials that differ only in
// values share one compiled function.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/material/TextureStore.h"

namespace athenea::material {

/// One input of a compiled material, as it sits in the blob.
struct MaterialSlot {
    enum class Kind : uint8_t {
        Value,     ///< `words` floats: scalars, vectors, colours, matrices (row major), ints and bools as floats
        Texture,   ///< two words: texture id, sampler slot
        Primvar,   ///< one word: the scene slot of `name`
    };
    Kind               kind = Kind::Value;
    std::string        variable;   ///< in the generated code
    uint32_t           offset = 0;
    uint32_t           words = 0;
    std::vector<float> value;      ///< Value: the document's
    std::string        name;       ///< Texture: the file; Primvar: the primvar
    std::string        space;      ///< Texture: the colour space as named (TextureInfo::space)
    Wrap               wrapS = Wrap::Repeat;
    Wrap               wrapT = Wrap::Repeat;
    Filter             filter = Filter::Linear;
};

/// A medium as a Material's `volume` terminal describes it, per unit of the
/// Volume's density field.
struct VolumeCoefficients {
    std::array<float, 3> absorption{0.0F, 0.0F, 0.0F};
    std::array<float, 3> scattering{0.0F, 0.0F, 0.0F};
    float                anisotropy = 0.0F;   ///< Henyey-Greenstein g
    std::array<float, 3> emission{0.0F, 0.0F, 0.0F};   ///< `uniform_edf`'s colour: radiance of the absorbing part
};

/// What a graph's BSDF nodes become.
enum class ClosureVariant : uint8_t {
    Lobes,               ///< the engine's: lobes on a stack a path tracer samples
    GenglslReference,    ///< MaterialX's own genglsl responses for one light direction, to check the lobes against
    /// The surface's opacity and nothing else: what a shadow ray asks a
    /// cut-out at a candidate, without the BSDF a second call site of every
    /// material would cost the compiler. Same blob layout as `Lobes`.
    Opacity,
};

struct CompiledMaterial {
    std::string               module;     ///< "athenea_mat_" + a hash of the source: equal structures, one module
    std::string               function;   ///< its public entry: void function(MaterialInputs, uint blobOffset)
    ClosureVariant            variant = ClosureVariant::Lobes;
    std::string               source;
    std::vector<MaterialSlot> slots;
    uint32_t                  words = 0;
};

/// How much a transmitting material lets through, as constants: the
/// transmission weight times its colour's luminance (one where a graph drives
/// either), and the dielectric's index. MaterialCompiler::transmission.
struct Transmission {
    float tint = 1.0F;
    float ior = 1.5F;
};

class MaterialCompiler {
public:
    /// `searchPaths`: where MaterialX's libraries are (a folder holding
    /// "libraries"), then the engine's shader directories.
    [[nodiscard]] static Result<std::unique_ptr<MaterialCompiler>> create(
        const std::vector<std::filesystem::path>& materialxRoots, const std::vector<std::filesystem::path>& shaderPaths);
    /// The same with libraries already loaded (hdMtlx's HdMtlxStdLibraries(), a
    /// MaterialX::DocumentPtr) and the folders their source files are found in.
    [[nodiscard]] static Result<std::unique_ptr<MaterialCompiler>> create(
        const std::shared_ptr<void>& libraries, const std::vector<std::filesystem::path>& librarySearchPaths,
        const std::vector<std::filesystem::path>& shaderPaths);
    ~MaterialCompiler();

    /// The renderable element `element` (a material or shader node) of a
    /// MaterialX document given as XML, the standard libraries imported.
    [[nodiscard]] Result<CompiledMaterial> compileXml(const std::string& xml, const std::string& element = {},
                                                      ClosureVariant variant = ClosureVariant::Lobes);

    /// The same for a document already built (by hdMtlx): a MaterialX::DocumentPtr.
    [[nodiscard]] Result<CompiledMaterial> compileDocument(const std::shared_ptr<void>& document,
                                                           const std::string& element = {},
                                                           ClosureVariant variant = ClosureVariant::Lobes);

    /// The blob words for `material`: its values, its textures requested from
    /// `textures`, its primvars' scene slots from `slotOf`.
    [[nodiscard]] static std::vector<float> parameters(const CompiledMaterial& material, TextureStore& textures,
                                                       const std::function<uint32_t(const std::string&)>& slotOf);

    /// The standard libraries' document (MaterialX::DocumentPtr), for building documents against.
    [[nodiscard]] std::shared_ptr<void> libraries() const;

    /// Whether a document's surface cuts samples away rather than blending
    /// them: a node with a non-zero `opacityThreshold`, or one a graph drives.
    /// MaterialX resolves the threshold itself -- such a surface's opacity is
    /// 0 or 1 -- so what this decides is who has to evaluate it: with a
    /// cutout, visibility does.
    [[nodiscard]] static bool cutsOut(const std::shared_ptr<void>& document);
    /// Whether the material is a UsdPreviewSurface whose fractional opacity
    /// means transparency (specification 2.6, `opacityMode` = transparent,
    /// the default, with no threshold): the specular stays at full weight and
    /// the diffuse goes down in favour of what is behind. The path tracer
    /// draws it that way; the raster, which can only cut by lot, does not.
    [[nodiscard]] static bool transparentOpacity(const std::shared_ptr<void>& document);
    /// Whether the material lets light through at all (a transmission, or a
    /// transparent opacity): what a ray that only asks whether the way is
    /// open must look at instead of stopping (technique::kMaterialTransmits).
    [[nodiscard]] static bool transmits(const std::shared_ptr<void>& document);
    /// What it lets through, where it does (`transmits`): constants a ray
    /// reads without evaluating the material (technique's `pathThrough`).
    [[nodiscard]] static std::optional<Transmission> transmission(const std::shared_ptr<void>& document);
    /// What a Material's `volume` terminal says of the medium: MaterialX's
    /// `volume(vdf, edf)` with `anisotropic_vdf(absorption, scattering,
    /// anisotropy)` or `absorption_vdf(absorption)` for the vdf and
    /// `uniform_edf(color)` for the edf, the coefficients per unit density
    /// (the Volume's `density` field multiplies them, as every renderer has
    /// it). Nothing without a volume terminal; an input driven by a graph
    /// is taken at its declared default, since a medium has no surface to
    /// evaluate a graph at, and said once.
    [[nodiscard]] static std::optional<VolumeCoefficients> volumeCoefficients(const std::shared_ptr<void>& document);

private:
    MaterialCompiler() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::material
