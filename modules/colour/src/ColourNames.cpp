// Copyright (c) 2026 jesus luque.
//
// A colour space's name, as the config in use knows it. The host does no
// arithmetic here: it looks names up.
#include "athenea/colour/ColourNames.h"

#include <array>
#include <cctype>
#include <string_view>
#include <utility>

#include "ColourConfig.h"
#include "athenea/core/Log.h"

namespace athenea::colour {

namespace {

std::string trimmed(const std::string& name) {
    size_t a = 0;
    size_t b = name.size();
    while (a < b && std::isspace(static_cast<unsigned char>(name[a])) != 0) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(name[b - 1])) != 0) --b;
    return name.substr(a, b - a);
}

/// Names that mean "these are not colours", whatever the config calls its
/// data space: MaterialX's and USD's, Blender's, the ACES 1 configs'.
bool isDataName(std::string_view name) {
    static constexpr std::array<std::string_view, 9> kData{"raw",  "Raw",       "RAW",      "data",          "Data",
                                                           "none", "Non-Color", "identity", "Utility - Raw"};
    for (const std::string_view data : kData) {
        if (name == data) return true;
    }
    return false;
}

/// Names a texture is given that are not every config's: UsdUVTexture's
/// `sourceColorSpace`, and the GfColorSpaceNames tokens USD 25 writes, each
/// as the MaterialX-compatible alias a studio config carries.
std::string_view aliasOf(std::string_view name) {
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 17> kAliases{{
        {"sRGB", "srgb_texture"},
        {"srgb", "srgb_texture"},
        {"linear", "lin_rec709"},
        {"lin_rec709_scene", "lin_rec709"},
        {"lin_ap1_scene", "lin_ap1"},
        {"lin_ap0_scene", "lin_ap0"},
        {"lin_p3d65_scene", "lin_p3d65"},
        {"lin_rec2020_scene", "lin_rec2020"},
        {"lin_adobergb_scene", "lin_adobergb"},
        {"srgb_rec709_scene", "srgb_texture"},
        {"g24_rec709_scene", "g24_rec709"},
        {"g22_rec709_scene", "g22_rec709"},
        {"g18_rec709_scene", "g18_rec709"},
        {"srgb_ap1_scene", "srgb_ap1"},
        {"g22_ap1_scene", "g22_ap1"},
        {"srgb_p3d65_scene", "srgb_p3d65"},
        {"g22_adobergb_scene", "g22_adobergb"},
    }};
    for (const auto& [from, to] : kAliases) {
        if (name == from) return to;
    }
    return name;
}

#if !ATHENEA_HAVE_OCIO
/// Without a config: the names the engine always knew.
bool isSrgbName(std::string_view name) {
    return name == "srgb_texture" || name == "sRGB - Texture" || name == "sRGB Encoded Rec.709 (sRGB)" ||
           name == "srgb_tx" || name == "Utility - sRGB - Texture";
}

bool isWorkingName(std::string_view name) {
    return name == "lin_rec709" || name == "Linear Rec.709 (sRGB)" || name == "lin_srgb" ||
           name == "Utility - Linear - sRGB" || name == "Utility - Linear - Rec.709";
}
#endif

}   // namespace

ResolvedSpace ColourNames::resolve(const std::string& given, FileEncoding file) const {
    const std::string name = trimmed(given);
    ResolvedSpace out;
    if (name.empty() || name == "auto" || name == "unknown") {
        // The file decides: what Hio says of an 8-bit image, linear otherwise.
        out.kind = file == FileEncoding::Srgb8 ? SpaceKind::Srgb : SpaceKind::Working;
        out.name = file == FileEncoding::Srgb8 ? srgb_ : working_;
        return out;
    }
    if (isDataName(name)) {
        out.kind = SpaceKind::Raw;
        return out;
    }
    const std::string alias(aliasOf(name));
#if ATHENEA_HAVE_OCIO
    // The config in use first (its aliases and roles included), then the
    // built-in studio config, whose spaces a function reaches through
    // GetProcessorFromConfigs.
    for (const bool builtin : {false, true}) {
        const OCIO::ConstConfigRcPtr& config = builtin ? config_->studio : config_->config;
        if (!config || (builtin && config_->studio == config_->config)) {
            continue;
        }
        OCIO::ConstColorSpaceRcPtr space = config->getColorSpace(name.c_str());
        if (!space) {
            space = config->getColorSpace(alias.c_str());
        }
        if (!space) {
            continue;
        }
        if (space->isData()) {
            out.kind = SpaceKind::Raw;
            return out;
        }
        out.name = space->getName();
        out.builtin = builtin;
        out.kind = !builtin && out.name == working_ ? SpaceKind::Working
                   : !builtin && out.name == srgb_  ? SpaceKind::Srgb
                                                    : SpaceKind::Other;
        return out;
    }
#else
    if (isSrgbName(alias)) {
        out.kind = SpaceKind::Srgb;
        out.name = srgb_;
        return out;
    }
    if (isWorkingName(alias)) {
        out.kind = SpaceKind::Working;
        out.name = working_;
        return out;
    }
#endif
    out.kind = SpaceKind::Unknown;
    out.name = name;
    if (warned_.insert(name).second) {
        log::warn("colour: no colour space '{}' in {}; read as the file says", name, config_->label);
    }
    return out;
}

}   // namespace athenea::colour
