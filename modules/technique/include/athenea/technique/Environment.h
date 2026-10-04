// Copyright (c) 2026 jesus luque.
//
// A DOME'S SKY, PREPARED FOR THE KERNELS THAT CANNOT SAMPLE IT.
//
// A dome's image goes through the material texture table, and the splat
// projection lives below material in the module order: it has no table. So a
// cloud under an HDRI was lit by the dome's constant colour -- the sky's mean
// with none of its direction -- while the mesh beside it reflected the sky
// itself. The one thing missing was a form of the sky that a buffer can hold.
//
// This prepares two of them, once, when the sky changes:
//
//   the harmonics   nine coefficients of radiance a dome, degree 2, which is
//                   what a diffuse surface needs
//   the levels      the sky convolved with a GGX lobe, one level a roughness,
//                   in an octahedral map: what a reflection of that sharpness
//                   sees
//
// Both layouts are `shaders/athenea/light/environment.slang`'s, which is also the
// reader: this class owns the kernels that write them and the buffers they go
// in, and the frame binds those buffers beside the lights.
//
// A dome past `kEnvironmentDomes` is not prepared and falls back to its own
// colour, which is what every dome did before this existed.
#pragma once

#include <cstdint>
#include <span>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"

namespace athenea::light {
class LightTable;
}
namespace athenea::material {
class TextureStore;
}

namespace athenea::technique {

/// Domes prepared at once. Four is what a stage with a sky, a backdrop and a
/// pair of studio wraps needs; a fifth is drawn as its colour.
inline constexpr uint32_t kEnvironmentDomes = 4;
/// The prefiltered map: a mip chain of the sky, eight levels, each half the
/// side of the one before and one roughness further on. `kEnvironmentCoarseSide`
/// is the floor under level 0 and what a dome that is only a colour takes;
/// `kEnvironmentFloorSide` is the floor under every level. All four mirror
/// `environment.slang`, and a test holds them equal.
inline constexpr uint32_t kEnvironmentLevels = 8;
inline constexpr uint32_t kEnvironmentCoarseSide = 256;
inline constexpr uint32_t kEnvironmentFloorSide = 16;
/// How wide level 0 is allowed to be: where a 4k sky lands, and what holds a
/// slice to 45 MB.
inline constexpr uint32_t kEnvironmentWidestSide = 2048;
inline constexpr uint32_t kEnvironmentCoefficients = 16;   // degree 3 (environment.slang)
/// Texels a row of `Environment::meshView` (env_mesh.slang's kEnvMeshTexels).
inline constexpr uint32_t kEnvironmentMeshTexels = 12;
/// The light a slice was prepared for, when it was not prepared at all.
inline constexpr uint32_t kEnvironmentNone = 0xFFFFFFFFU;

class Environment {
public:
    [[nodiscard]] static Result<Environment> create(gpu::ShaderLibrary& library);

    /// Prepares `domeLights` (slice -> the index of the light in `table`),
    /// reading their images out of `textures`. Every one of them must carry
    /// an image: a dome without one is a constant, which `splat_relight`
    /// already answers in closed form and exactly. Waits for the device: this
    /// is not a frame, and what follows it reads what it wrote.
    /// `domeTextures` is each dome's texture id in `textures`, or
    /// `kEnvironmentNone` for one that is only a colour: it is what chooses
    /// how wide the mirror level is made.
    [[nodiscard]] Result<void> build(const light::LightTable& table, const material::TextureStore& textures,
                                     std::span<const uint32_t> domeLights,
                                     std::span<const uint32_t> domeTextures, uint32_t lightCount);

    /// THE SUN, TAKEN OUT OF EACH DOME: two float4 a slice, the first
    /// (direction, solid angle) and the second (irradiance, the sky's mean
    /// luminance). A solid angle of 0 says that dome has no sun and its
    /// harmonics hold the whole sky, as they did before this existed.
    ///
    /// The prefiltered map keeps its sun. A reflection of a disc is a
    /// highlight and the map at 2048 a side resolves it; what could not hold
    /// it was the nine coefficients, so the disc is taken out of those alone
    /// and comes back on the body, never on the polish. Counted once.
    [[nodiscard]] const gpu::Buffer& sun() const noexcept { return sun_; }

    /// The prefiltered octahedral levels, f16 pairs a texel.
    [[nodiscard]] const gpu::Buffer& texels() const noexcept { return texels_; }
    /// Nine float4 a dome: the sky's radiance coefficients, rgb.
    [[nodiscard]] const gpu::Buffer& sh() const noexcept { return sh_; }
    /// One uint a light: the slice prepared for it, or `kEnvironmentNone`.
    [[nodiscard]] const gpu::Buffer& domeOfLight() const noexcept { return domeOfLight_; }
    [[nodiscard]] uint32_t domes() const noexcept { return domes_; }
    /// Lights `domeOfLight` describes, which is the frame's whole light list.
    [[nodiscard]] uint32_t lightCount() const noexcept { return lights_; }
    [[nodiscard]] bool ready() const noexcept { return domes_ > 0 && texels_.valid() && sh_.valid(); }
    /// THE SAME SKY FOR A MESH'S SHADING KERNEL, which has no buffer slot
    /// left: a texture of `kEnvironmentMeshTexels` a row and a row a slice
    /// (env_mesh.slang has the layout): the harmonics, the sun, and the light
    /// the row is for. What the raster route lights a mesh with when it reads
    /// a dome prefiltered rather than sampling it. Null before a build.
    [[nodiscard]] rhi::ITextureView* meshView() const noexcept { return meshView_.get(); }
    /// The octahedral side of level 0 these slices were written at, which the
    /// frame must hand to whoever reads them.
    [[nodiscard]] uint32_t baseSide() const noexcept { return baseSide_; }

    /// Texels one dome takes, every level of its map. Mirrors the shader's
    /// `envTexelsPerDome`.
    [[nodiscard]] static uint32_t texelsPerDome(uint32_t baseSide) noexcept;

    /// THE OCTAHEDRAL SIDE THAT MATCHES A LAT-LONG SKY'S OWN TEXELS.
    ///
    /// A lat-long texel at the equator stands for 2 pi^2 / (W H) steradians
    /// and an octahedral one for 4 pi / S^2, so the two meet at
    /// S = sqrt(2 W H / pi): 2048 for a 4k sky, 1024 for a 2k one. Rounded
    /// down to a power of two and held between `kEnvironmentCoarseSide` and
    /// `kEnvironmentWidestSide`; a dome with no image takes the floor, its
    /// map having nothing to resolve.
    [[nodiscard]] static uint32_t baseSideFor(uint32_t width, uint32_t height) noexcept;

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel project_;
    gpu::ComputeKernel prefilter_;
    gpu::ComputeKernel sunKernel_;
    gpu::ComputeKernel meshPack_;
    gpu::Texture       mesh_;
    rhi::ComPtr<rhi::ITextureView> meshView_;
    gpu::Buffer        texels_;
    gpu::Buffer        sh_;
    gpu::Buffer        sun_;
    gpu::Buffer        domeOfLight_;
    uint32_t           domes_ = 0;
    uint32_t           lights_ = 0;
    uint32_t           baseSide_ = kEnvironmentCoarseSide;
};

}   // namespace athenea::technique
