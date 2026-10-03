// Copyright (c) 2026 jesus luque.
#include "athenea/scene/CloudColour.h"

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::scene {

Result<CloudColour> CloudColour::create(gpu::ShaderLibrary& library) {
    CloudColour made;
    made.device_ = &library.device();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/cloud_to_working", "cloudToWorking");
    if (!kernel) return std::move(kernel).error();
    made.kernel_ = std::move(*kernel);
    return made;
}

Result<std::optional<colour::ColourMatrix>> CloudColour::matrixFor(const GpuSplats& splats,
                                                                   colour::ColourCompiler& colour) const {
    if (!splats.linear || splats.count == 0) {
        return std::optional<colour::ColourMatrix>{};
    }
    const std::string space = splats.colourSpace.empty() ? std::string(colour::kDefaultColourSpace)
                                                         : splats.colourSpace;
    const colour::ResolvedSpace resolved = colour.names().resolve(space, colour::FileEncoding::Linear);
    if (resolved.kind != colour::SpaceKind::Other && resolved.kind != colour::SpaceKind::Srgb) {
        return std::optional<colour::ColourMatrix>{};   // the working space, data, or unknown
    }
    auto matrix = colour.matrix(resolved.name, colour.names().working());
    if (!matrix) return std::move(matrix).error();
    return std::optional<colour::ColourMatrix>(std::move(*matrix));
}

void CloudColour::record(gpu::CommandBatch& batch, const GpuSplats& splats, const colour::ColourMatrix& matrix,
                         uint32_t first, uint32_t count) const {
    if (count == 0) {
        return;
    }
    kernel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colourRows"].setBinding(matrix.rows.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        cursor["emissionWords"].setBinding(splats.hasEmission() ? splats.emission.rhi() : splats.shape.rhi());
        cursor["transfer"].setBinding(splats.transfer.valid() ? splats.transfer.rhi() : splats.shape.rhi());
        rhi::ShaderCursor p = cursor["params"];
        p["first"].setData(first);
        p["count"].setData(count);
        p["shWords"].setData(splats.shWords);
        p["rest"].setData(splats.restPerColour);
        p["emission"].setData(uint32_t{splats.hasEmission() ? 1u : 0u});
        p["transferWords"].setData(splats.transfer.valid() ? splats.transferWords : 0u);
        p["transferCount"].setData(splats.transferCount);
    });
}

Result<bool> CloudColour::toWorking(GpuSplats& splats, colour::ColourCompiler& colour) const {
    auto matrix = matrixFor(splats, colour);
    if (!matrix) return std::move(matrix).error();
    if (!matrix->has_value()) {
        if (splats.linear && colour.names().resolve(splats.colourSpace.empty() ? std::string(colour::kDefaultColourSpace)
                                                                                : splats.colourSpace,
                                                    colour::FileEncoding::Linear)
                                     .kind == colour::SpaceKind::Working) {
            splats.colourSpace = colour.names().working();
        }
        return false;
    }
    gpu::CommandBatch batch(*device_);
    record(batch, splats, **matrix, 0, splats.count);
    ATHENEA_TRY(batch.submit(true));
    splats.colourSpace = colour.names().working();
    return true;
}

}   // namespace athenea::scene
