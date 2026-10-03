// Copyright (c) 2026 jesus luque.
//
// WHAT A CLOUD CASTS, MEASURED FROM THE LIGHT ONCE A FRAME, WITHOUT A RAY.
//
// Transmittance through a cloud is a product over the gaussians a ray meets,
// and a product does not care about order. So the pass from the light neither
// sorts nor keeps a list: every gaussian adds its optical depth into the
// texels its footprint covers, and the texel ends up holding the whole
// cloud's. `exp(-that)` is what reaches a receiver behind the cloud, exactly;
// the Fourier terms of the same map (`coefficients` above one) are what a
// receiver inside it needs.
//
// It works on a device with no ray tracing at all, which is half the point:
// the raster route had no cloud shadow of any kind, and `--splat-shadows`
// needs inline rays that CUDA does not offer.
//
// The kernels are shaders/athenea/technique/splat_shadow_map.slang; this is the
// host that clears, frames and accumulates, once a frame, into one buffer.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Texture.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::technique {

/// A cloud that casts, as the frame has it: the gaussians at the positions
/// they were posed at, and where the cloud stands.
struct ShadowMapCaster {
    const scene::GpuSplats* cloud = nullptr;
    const gpu::Buffer*      positions = nullptr;   ///< posed, or the cloud's own
    std::array<float, 12>   objectToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    uint64_t                categories = 0;        ///< which lights it casts for
    /// The box the cloud was bound in, in its own space: what the map is
    /// sized from. A posed cloud's own `bounds` follow the pose, and a map
    /// sized from them changed its texels with every flap of a wing. Empty:
    /// the cloud's `restBounds`, or its `bounds` where it has none.
    std::optional<scene::Bounds> restBounds;
};

struct ShadowMapJob {
    std::vector<ShadowMapCaster> casters;
    const gpu::Buffer*           lights = nullptr;
    uint32_t                     lightCount = 0;      ///< at most 8: the slots a map has
    uint32_t                     resolution = 1024;   ///< texels a side, per light
    /// 1: the total optical depth through the cloud, which is exact for a
    /// receiver behind it (a ground plane). 3, 5, 7: one, two or three
    /// Fourier pairs as well, which resolve the depth a receiver stands at.
    uint32_t                     coefficients = 1;
    /// How much wider than the casters' rest sphere the map is drawn, since a
    /// skinned cloud is posed away from the box it was bound in. The extent is
    /// then rounded up to a quarter octave, so a pose that reaches a little
    /// past it changes the map only when it crosses a step.
    float                        margin = 0.35F;
    /// How far behind the nearest caster in a texel a receiver still counts
    /// as standing on it, in world units. 0: two per cent of the slab, which
    /// is the casters' rest extent and so the same length every frame.
    float                        selfBias = 0.0F;
    /// What the optical depth is multiplied by: the shadow's density, as a
    /// compositor means it. 1 is what the cloud's own opacity says.
    float                        density = 1.0F;
};

class SplatShadowMap {
public:
    [[nodiscard]] static Result<SplatShadowMap> create(gpu::ShaderLibrary& library);

    /// Queued into `batch`: the map for every light of `job`, from scratch.
    /// A job with no caster, no light or no shadow-casting light leaves
    /// `valid()` false and every lookup answering 1.
    [[nodiscard]] Result<void> build(gpu::CommandBatch& batch, const ShadowMapJob& job);

    [[nodiscard]] bool valid() const noexcept { return valid_; }
    [[nodiscard]] const gpu::Buffer& map() const noexcept { return map_; }
    /// The same map as a texture: one array layer a coefficient a light, the
    /// frames in layer zero's first row. It is how a shading kernel gets it,
    /// since that kernel is at Metal's limit of thirty-one buffers and has a
    /// texture slot to spare.
    [[nodiscard]] const gpu::Texture& texture() const noexcept { return texture_; }
    /// That texture's view, made with it and kept: what a frame binds.
    [[nodiscard]] rhi::ITextureView* textureView() const noexcept { return view_.get(); }
    /// THE TRANSMITTANCE CHAIN: exp(-total), a layer a light, with a full mip
    /// chain -- what a floor far from the map reads at its pixel's footprint
    /// (the mean of the transmittances under it). A view of every level.
    [[nodiscard]] const gpu::Texture& chain() const noexcept { return chain_; }
    [[nodiscard]] rhi::ITextureView* chainView() const noexcept { return chainView_.get(); }
    [[nodiscard]] uint32_t resolution() const noexcept { return resolution_; }
    [[nodiscard]] uint32_t coefficients() const noexcept { return coefficients_; }

    /// What the map says the mean transmittance is over its texels, counted
    /// on the device: one number a frame's cloud shadow amounts to, for a
    /// test and for `ATHENEA_SHADOW_DEBUG`.
    [[nodiscard]] Result<double> meanTransmittance(uint32_t light);

    /// THE OTHER RECEIVER: the gaussians themselves.
    ///
    /// `factors[(base + i) * lights + k]` for every gaussian of `caster` --
    /// the slot a baked visibility field or a shadow ray fills, so nothing
    /// downstream changes. Read at each gaussian's own depth, which is what
    /// the Fourier terms are for: with one coefficient every gaussian of a
    /// cloud reads the whole cloud's shadow and the cloud goes flat.
    [[nodiscard]] Result<void> factors(gpu::CommandBatch& batch, const ShadowMapCaster& caster, uint32_t base,
                                       const gpu::Buffer& into);

    /// Every factor back to 1: what a cloud with no field and no map must
    /// read. `count` is how many floats there are.
    [[nodiscard]] Result<void> clearFactors(gpu::CommandBatch& batch, const gpu::Buffer& factors, uint32_t count);

    /// The map's frame for `light`, as the device built it: how many texels a
    /// world unit is, and where the grid's origin falls. Bookkeeping a test
    /// needs to put a receiver on a texel's centre; nothing is computed.
    struct FrameInfo {
        bool                 valid = false;
        float                texelsPerUnit = 0.0F;
        std::array<float, 4> rowU{};
        std::array<float, 4> rowV{};
        std::array<float, 4> rowZ{};
        float                bias = 0.0F;   ///< in the slab's units
    };
    [[nodiscard]] Result<FrameInfo> frameInfo(uint32_t light);

    /// What the map answers at world points somebody names -- the same read a
    /// receiver does, one thread a point. For tests and for ATHENEA_SHADOW_DEBUG:
    /// what comes back is what a kernel computed, a handful of numbers.
    [[nodiscard]] Result<std::vector<float>> probe(uint32_t light, std::span<const std::array<float, 4>> points);

private:
    gpu::Device*                      device_ = nullptr;
    gpu::ShaderLibrary*               library_ = nullptr;
    std::optional<gpu::ComputeKernel> clear_;
    std::optional<gpu::ComputeKernel> box_;
    std::optional<gpu::ComputeKernel> frame_;
    std::optional<gpu::ComputeKernel> splat_;
    std::optional<gpu::ComputeKernel> mean_;
    std::optional<gpu::ComputeKernel> probe_;
    std::optional<gpu::ComputeKernel> header_;
    std::optional<gpu::ComputeKernel> resolve_;
    std::optional<gpu::ComputeKernel> chainLevel_;
    std::optional<gpu::ComputeKernel> factors_;
    std::optional<gpu::ComputeKernel> clear_factors_;
    gpu::Buffer                       map_;
    gpu::Buffer                       total_;
    gpu::Texture                      texture_;
    rhi::ComPtr<rhi::ITextureView>    view_;
    gpu::Texture                      chain_;
    rhi::ComPtr<rhi::ITextureView>    chainView_;     ///< every level, for reading
    std::vector<rhi::ComPtr<rhi::ITextureView>> chainLevels_;   ///< one a level, for writing
    uint32_t                          resolution_ = 0;
    uint32_t                          coefficients_ = 0;
    uint32_t                          lights_ = 0;
    bool                              valid_ = false;
};

}   // namespace athenea::technique
