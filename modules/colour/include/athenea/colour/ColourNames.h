// Copyright (c) 2026 jesus luque.
//
// The name a file, a material or a stage gives a colour space, as the config
// in use knows it. MaterialX says `srgb_texture`, UsdUVTexture `sRGB` or
// `raw`, USD's GfColorSpaceNames `lin_ap1_scene`, Blender `Non-Color`, a
// studio `Utility - Raw`: one resolution, so that every reader of a colour
// space name agrees on what it means (docs/decisions.md, "Colour").
#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>

namespace athenea::colour {

/// The space the engine renders in. Linear Rec.709 in this phase; the name is
/// the studio config's (and GfColorSpaceNames') for it.
inline constexpr const char* kWorkingSpace = "lin_rec709_scene";
/// The config used when none is named: OCIO's built-in studio config, which
/// carries MaterialX's and USD's names as aliases.
inline constexpr const char* kStudioConfig = "ocio://studio-config-latest";

enum class SpaceKind : uint8_t {
    Raw,       ///< data (normals, roughness, masks): read as it is, no function
    Working,   ///< already the working space: read as it is
    Srgb,      ///< sRGB-encoded Rec.709: 8-bit files take the hardware's sRGB view
    Other,     ///< anything else the config knows: a compiled function into the working space
    Unknown,   ///< a name no config here knows: read as the file says, with a warning
};

/// What a file holds, for a name that leaves it to the file (empty, `auto`).
enum class FileEncoding : uint8_t {
    Linear,   ///< floats, 16-bit integers, 8-bit tagged linear: the working space
    Srgb8,    ///< 8-bit colour: sRGB unless the file says otherwise
};

struct ResolvedSpace {
    SpaceKind   kind = SpaceKind::Working;
    std::string name;      ///< the config's own name for it (Working, Srgb, Other); the given one (Unknown)
    bool        builtin = false;   ///< known only to the built-in studio config, not the config in use
};

struct ColourConfig;

class ColourNames {
public:
    /// `name` as the config in use names it, or by the file for an empty one
    /// or `auto`. Data names (`raw`, `Raw`, `data`, `Non-Color`, `none`,
    /// `identity`, `Utility - Raw`, any space the config marks data) are Raw.
    /// A name the config does not know is looked for in the built-in studio
    /// config; one neither knows is Unknown and is warned of once.
    [[nodiscard]] ResolvedSpace resolve(const std::string& name, FileEncoding file) const;
    /// The working space as the config in use names it.
    [[nodiscard]] const std::string& working() const noexcept { return working_; }

private:
    friend class ColourCompiler;
    std::shared_ptr<const ColourConfig>   config_;
    std::string                           working_;
    std::string                           srgb_;
    mutable std::set<std::string>         warned_;
};

}   // namespace athenea::colour
