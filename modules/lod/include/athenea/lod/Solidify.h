// Copyright (c) 2026 jesus luque.
//
// A trained cloud solidified (SSF phase 2): the gaussians that describe an
// opaque surface become a solid layer, the rest stay as the residual layer;
// and the layer drawn again as flat gaussians (Mode A). The kernels and the
// reasons are in shaders/athenea/lod/ssf_solidify.slang and ssf_splat.slang.
// Only counts cross back to the CPU.
#pragma once

#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/SolidField.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::lod {

struct SolidifySettings {
    float    opacityMin = 0.5F;       ///< a candidate is at least this opaque
    float    flatMax = 0.25F;         ///< and its thin scale under this of its middle one (0: none)
    float    coherenceMin = 0.9F;     ///< weighted mean |n_i . n_j| over its neighbours a blob keeps
    float    radiusScale = 2.0F;      ///< a blob's support over its gaussian's scale
    float    thicknessRatio = 0.5F;   ///< the anchor's c over the candidates' mean middle radius
    float    tau = 0.5F;
    float    level = 1.0F;            ///< C: the field each blob's centre is weighted to
    float    mergeFraction = 0.25F;   ///< centres nearer than this of the smaller radius merge
    uint32_t weightRounds = 3;
    uint32_t ridgeRounds = 2;
    float    thinRatio = 0.1F;        ///< Mode A: a surfel's thin axis over its smaller radius
};

struct SolidifyResult {
    SolidLayer  solid;             ///< one anchor, for now
    gpu::Buffer normals;           ///< packNormal a blob (stored, the user's decision of 31 Oct)
    gpu::Buffer solidSource;       ///< uint a blob: the gaussian it came from
    gpu::Buffer residualSource;    ///< uint a residual: the gaussian it is
    uint32_t    residuals = 0;
    uint32_t    gaussians = 0;     ///< the input's
    uint32_t    candidates = 0;    ///< opaque, flat and not glass
    uint32_t    coherent = 0;      ///< of those, whose neighbours agree: the absorbed
    /// The absorbed over the gaussians.
    [[nodiscard]] double absorbed() const noexcept {
        return gaussians > 0 ? static_cast<double>(coherent) / gaussians : 0.0;
    }
};

class Solidifier {
public:
    [[nodiscard]] static Result<Solidifier> create(gpu::ShaderLibrary& library);

    [[nodiscard]] Result<SolidifyResult> solidify(const scene::GpuSplats& cloud, const SolidifySettings& settings = {});

    /// MODE A: the blobs as flat gaussians (their harmonics and shading normals
    /// their sources'), then the residuals as they were -- one cloud the
    /// rasteriser draws as any other. What else a cloud carries (material,
    /// transfer) follows the sources on export, by `solidSource` and
    /// `residualSource`.
    [[nodiscard]] Result<scene::GpuSplats> toSplats(const SolidifyResult& result, const scene::GpuSplats& source,
                                                    const SolidifySettings& settings = {});

private:
    gpu::ShaderLibrary* library_ = nullptr;
    gpu::Device*        device_ = nullptr;
    gpu::PrefixSum      prefix_;
    SolidField          field_;
    gpu::ComputeKernel  convert_, anchor_, compact_, coherence_, weights_, ridge_, normals_, merge_, iota_, pick_;
    gpu::ComputeKernel  toSplat_, gather_;
};

}   // namespace athenea::lod
