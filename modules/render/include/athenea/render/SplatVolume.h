// Copyright (c) 2026 jesus luque.
//
// A CLOUD OF GAUSSIANS THAT IS A MEDIUM (proposal 105): what a stage says of
// it beyond its gaussians, and what a frame hands the rasteriser to draw it.
//
// Each gaussian of such a cloud (scene::GpuSplats::volume) carries the peak of
// its extinction, an albedo and a temperature; the medium is the sum
// mu_t(x) = densityScale * sum_i sigma_i G_i(x), with G_i of peak one, as
// `athenea vdb2splat` fits it. The rasteriser draws a gaussian with the alpha
// its exact optical depth along the pixel's ray gives, 1 - exp(-tau_i), and the
// radiance a shading pass baked for it this frame (technique::VolumeSplatShading).
#pragma once

#include <array>
#include <cstdint>

namespace athenea::gpu {
class Buffer;
}

namespace athenea::render {

struct SplatVolume {
    /// Density to extinction a world unit: a Volume's `athenea:densityScale`.
    float                densityScale = 1.0F;
    /// The single-scattering albedo, times each gaussian's own (a Cd field's
    /// colour, 1 where the file had none): a Volume's `athenea:albedo`, and
    /// its default.
    std::array<float, 3> albedo{0.8F, 0.8F, 0.8F};
    /// Henyey-Greenstein g: a Volume's `athenea:anisotropy`.
    float                anisotropy = 0.0F;
    /// The radiance of the absorbing part, as a Volume's material gives it
    /// (`uniform_edf`'s colour), so the medium's source is (1 - albedo) times
    /// this per unit of extinction, as the path tracer's.
    std::array<float, 3> emission{0.0F, 0.0F, 0.0F};
    /// FIRE FROM THE TEMPERATURE: a gaussian at temperature t (the grid's own
    /// units) glows as a black body at `temperatureOffset + temperatureScale
    /// * t` kelvin, times `emissionScale` (Cycles' Principled Volume
    /// convention: the blackbody colour at luminance one times
    /// sigma_SB T^4 / pi * 1e-6). 0 turns it off.
    float                emissionScale = 0.0F;
    float                temperatureScale = 1.0F;
    float                temperatureOffset = 0.0F;
    /// MULTIPLE SCATTERING, Wrenninge's octaves: how many are summed, the
    /// first being single scattering (1: single scattering alone).
    uint32_t             octaves = 4;
    /// What a frame baked for each gaussian: two uint a splat, rgb halves
    /// (technique::VolumeSplatShading). Null: the gaussian is its emission
    /// alone, (1 - albedo) times `emission`.
    const gpu::Buffer*   radiance = nullptr;
};

}   // namespace athenea::render
