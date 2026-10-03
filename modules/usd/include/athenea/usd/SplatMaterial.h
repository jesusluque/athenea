// Copyright (c) 2026 jesus luque.
//
// A MATERIAL BOUND TO A PARTICLEFIELD, AS THE PLACE A GAUSSIAN'S SURFACE IS
// SAID. OpenUSD's ParticleField says "if a material is bound to a
// ParticleField location, a renderer may opt to use the material description
// instead of the locally defined radiance" (usdVol's overview). This is what
// the engine reads of one: the surface node's inputs that describe what a
// relit gaussian reflects with, each either a primvar a reader node names --
// one value a particle -- or the input's own value, one for every particle.
// Which arrays those are is bookkeeping; the arrays themselves go to the
// device as they are, as every other array of the field does.
#pragma once

#include <string>

#include <pxr/base/vt/value.h>
#include <pxr/imaging/hd/material.h>

namespace athenea::usd {

/// One input of the surface: a primvar read per particle, or a value for all
/// of them. Neither: the material says nothing of it.
struct SplatMaterialInput {
    /// The primvar a reader node names (`UsdPrimvarReader_*`'s `varname`,
    /// MaterialX `geompropvalue`'s `geomprop`), without `primvars:`.
    std::string  primvar;
    /// The input's own authored value, where nothing is connected to it.
    pxr::VtValue constant;
    /// Connected, but to something that is not a primvar reader (a texture,
    /// a graph): not read here, since a gaussian has no coordinates to look
    /// a texture up with.
    bool         unreadable = false;

    [[nodiscard]] bool given() const noexcept { return !primvar.empty() || !constant.IsEmpty(); }
};

/// The inputs a relit gaussian is made of, as the bound material says them.
struct SplatMaterialChannels {
    /// The surface node's type, where it is one read here: UsdPreviewSurface,
    /// OpenPBR Surface or Standard Surface. Empty for anything else, and then
    /// every input below is empty too.
    std::string        surface;
    SplatMaterialInput baseColour;     ///< diffuseColor / base_color: in place of the DC, an albedo
    SplatMaterialInput metallic;       ///< metallic / base_metalness / metalness
    SplatMaterialInput roughness;      ///< roughness / specular_roughness
    SplatMaterialInput transmission;   ///< transmission_weight / transmission (UsdPreviewSurface has none)
    SplatMaterialInput normal;         ///< normal / geometry_normal: a primvar only, in the field's space
    SplatMaterialInput emission;       ///< emissiveColor (UsdPreviewSurface only)

    [[nodiscard]] bool any() const noexcept {
        return baseColour.given() || metallic.given() || roughness.given() || transmission.given() ||
               normal.given() || emission.given();
    }
};

/// What `network`'s surface terminal says a gaussian is made of. A network
/// whose surface is not one of the three nodes above says nothing.
[[nodiscard]] SplatMaterialChannels splatMaterialChannels(const pxr::HdMaterialNetwork2& network);

}   // namespace athenea::usd
