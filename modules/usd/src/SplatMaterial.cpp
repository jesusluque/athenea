// Copyright (c) 2026 jesus luque.
#include "athenea/usd/SplatMaterial.h"

#include <array>

#include <pxr/base/tf/token.h>

PXR_NAMESPACE_USING_DIRECTIVE

namespace athenea::usd {
namespace {

/// Which input of each surface node is which channel. Empty: that node has
/// no such input (UsdPreviewSurface's transmission), or it is not read here
/// (OpenPBR's and Standard Surface's emission, whose colour is scaled by a
/// luminance or a weight this engine does not fold into one radiance).
struct SurfaceInputs {
    const char* nodeType;
    const char* baseColour;
    const char* metallic;
    const char* roughness;
    const char* transmission;
    const char* normal;
    const char* emission;
};

constexpr std::array<SurfaceInputs, 4> kSurfaces{{
    {"UsdPreviewSurface", "diffuseColor", "metallic", "roughness", "", "normal", "emissiveColor"},
    {"ND_UsdPreviewSurface_surfaceshader", "diffuseColor", "metallic", "roughness", "", "normal", "emissiveColor"},
    {"ND_open_pbr_surface_surfaceshader", "base_color", "base_metalness", "specular_roughness",
     "transmission_weight", "geometry_normal", ""},
    {"ND_standard_surface_surfaceshader", "base_color", "metalness", "specular_roughness", "transmission",
     "normal", ""},
}};

/// The primvar a reader node reads, or empty for a node that is not one.
std::string primvarOf(const HdMaterialNode2& node) {
    const std::string& type = node.nodeTypeId.GetString();
    const char* parameter = nullptr;
    if (type.rfind("UsdPrimvarReader_", 0) == 0) {
        parameter = "varname";
    } else if (type.rfind("ND_geompropvalue_", 0) == 0) {
        parameter = "geomprop";
    } else {
        return {};
    }
    const auto found = node.parameters.find(TfToken(parameter));
    if (found == node.parameters.end()) {
        return {};
    }
    if (found->second.IsHolding<TfToken>()) {
        return found->second.UncheckedGet<TfToken>().GetString();
    }
    if (found->second.IsHolding<std::string>()) {
        return found->second.UncheckedGet<std::string>();
    }
    return {};
}

SplatMaterialInput inputOf(const HdMaterialNetwork2& network, const HdMaterialNode2& surface, const char* name) {
    SplatMaterialInput input;
    if (name == nullptr || name[0] == '\0') {
        return input;
    }
    const TfToken key(name);
    if (const auto connected = surface.inputConnections.find(key);
        connected != surface.inputConnections.end() && !connected->second.empty()) {
        const auto upstream = network.nodes.find(connected->second.front().upstreamNode);
        input.primvar = upstream != network.nodes.end() ? primvarOf(upstream->second) : std::string();
        input.unreadable = input.primvar.empty();
        return input;
    }
    if (const auto value = surface.parameters.find(key); value != surface.parameters.end()) {
        input.constant = value->second;
    }
    return input;
}

}   // namespace

SplatMaterialChannels splatMaterialChannels(const HdMaterialNetwork2& network) {
    SplatMaterialChannels channels;
    static const TfToken kSurface("surface");
    const auto terminal = network.terminals.find(kSurface);
    if (terminal == network.terminals.end()) {
        return channels;
    }
    const auto node = network.nodes.find(terminal->second.upstreamNode);
    if (node == network.nodes.end()) {
        return channels;
    }
    const std::string& type = node->second.nodeTypeId.GetString();
    for (const SurfaceInputs& surface : kSurfaces) {
        if (type != surface.nodeType) {
            continue;
        }
        channels.surface = type;
        channels.baseColour = inputOf(network, node->second, surface.baseColour);
        channels.metallic = inputOf(network, node->second, surface.metallic);
        channels.roughness = inputOf(network, node->second, surface.roughness);
        channels.transmission = inputOf(network, node->second, surface.transmission);
        channels.normal = inputOf(network, node->second, surface.normal);
        // A normal is a direction a gaussian faces, and one for every
        // particle says nothing a gaussian's own frame does not: the input's
        // value -- UsdPreviewSurface's (0, 0, 1), a tangent-space normal -- is
        // not read, only a primvar.
        channels.normal.constant = VtValue();
        channels.emission = inputOf(network, node->second, surface.emission);
        break;
    }
    return channels;
}

}   // namespace athenea::usd
