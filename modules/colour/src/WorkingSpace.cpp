// Copyright (c) 2026 jesus luque.
#include "athenea/colour/WorkingSpace.h"

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::colour {

namespace {

/// How far from linear a probe may be (colour_matrix.slang), relative: a
/// matrix OCIO applies in float departs by a few ulp; a curve, by its bend.
constexpr float kLinearBound = 1e-5F;

}   // namespace

Result<gpu::WorkingSpace> workingSpaceRecord(ColourCompiler& compiler) {
    const ColourNames& names = compiler.names();
    if (names.workingIsRec709()) {
        return gpu::linRec709Scene();
    }
    auto from = compiler.matrix(names.rec709(), names.working());
    if (!from) return std::move(from).error();
    auto to = compiler.matrix(names.working(), names.rec709());
    if (!to) return std::move(to).error();
    gpu::ShaderLibrary& library = compiler.library();
    gpu::Device& device = library.device();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/colour/working_space_build", "workingSpaceBuild");
    if (!kernel) return std::move(kernel).error();
    gpu::BufferDesc desc;
    desc.bytes = sizeof(gpu::WorkingSpace);
    desc.elementBytes = sizeof(gpu::WorkingSpace);
    desc.label = "colour.workingSpace";
    auto record = gpu::Buffer::create(device, desc);
    if (!record) return std::move(record).error();
    desc.bytes = 2 * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "colour.workingSpace.counts";
    auto counts = gpu::Buffer::create(device, desc);
    if (!counts) return std::move(counts).error();
    {
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["fromRec709"].setBinding(from->rows.rhi());
            cursor["toRec709"].setBinding(to->rows.rhi());
            cursor["record"].setBinding(record->rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["rec709"].setData(&gpu::linRec709Scene(), sizeof(gpu::WorkingSpace));
            cursor["linearBound"].setData(kLinearBound);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    // The record and the counts, once: a uniform is set from the host.
    gpu::WorkingSpace out;
    uint32_t c[2] = {0, 0};
    ATHENEA_TRY(record->read(device, 0, sizeof(out), &out));
    ATHENEA_TRY(counts->read(device, 0, sizeof(c), c));
    if (c[0] != 0) {
        return Error::make(ErrorCode::InvalidArgument,
                           "colour: '{}' as the working space is not linear (no 3x3 reaches it from {}); a working "
                           "space is linear and scene-referred", compiler.workingAsked(), names.rec709());
    }
    if (c[1] != 0) {
        return Error::make(ErrorCode::InvalidArgument,
                           "colour: '{}' as the working space: {} into it and back is not the identity ({} entries "
                           "off by more than 1e-4)", compiler.workingAsked(), names.rec709(), c[1]);
    }
    return out;
}

Result<WorkingSpaceChoice> chooseWorkingSpace(gpu::ShaderLibrary& library, const std::string& configUri,
                                              const std::string& name) {
    auto compiler = ColourCompiler::create(library, configUri.empty() ? std::string(kStudioConfig) : configUri,
                                           name.empty() ? std::string(kWorkingSpace) : name);
    if (!compiler) return std::move(compiler).error();
    WorkingSpaceChoice choice;
    choice.compiler = std::shared_ptr<ColourCompiler>(std::move(*compiler));
    auto record = workingSpaceRecord(*choice.compiler);
    if (!record) return std::move(record).error();
    choice.record = *record;
    return choice;
}

}   // namespace athenea::colour
