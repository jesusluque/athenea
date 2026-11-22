// Copyright (c) 2026 jesus luque.
#include "athenea/render/SplatLift.h"

#include <algorithm>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

Error noTrace() {
    return Error(ErrorCode::InvalidArgument, "the frame keeps no trace (RenderSettings::trace)");
}

}   // namespace

Result<SplatLift> makeSplatLift(gpu::Device& device, uint32_t slots) {
    SplatLift lift;
    lift.slots = slots;
    // Zeroed on purpose: a buffer nobody wrote is not a buffer of zeros on
    // every device, and these are sums.
    const std::vector<uint32_t> zeros(uint64_t{std::max(slots, 1u)} * 2, 0u);
    gpu::BufferDesc desc;
    desc.bytes = zeros.size() * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "splat.lift";
    auto made = gpu::Buffer::create(device, desc, zeros.data());
    if (!made) return std::move(made).error();
    lift.sums = *made;
    return lift;
}

Result<void> liftMask(gpu::ShaderLibrary& library, const RenderTargets& traced, const gpu::Buffer& mask,
                      SplatLift& lift) {
    if (!traced.hasTrace()) return noTrace();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/splat/splat_lift", "splatLift");
    if (!kernel) return std::move(kernel).error();
    gpu::CommandBatch batch(library.device());
    kernel->dispatch(batch, {traced.width, traced.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["trace"].setBinding(traced.trace.rhi());
        cursor["mask"].setBinding(mask.rhi());
        cursor["lift"].setBinding(lift.sums.rhi());
        // The matte is the other entry point's; a name a shader declares is bound.
        cursor["matte"].setBinding(traced.depth.rhi());
        cursor["params"]["width"].setData(traced.width);
        cursor["params"]["height"].setData(traced.height);
        cursor["params"]["slots"].setData(lift.slots);
        cursor["params"]["hard"].setData(uint32_t{0});
        cursor["params"]["threshold"].setData(0.5F);
    });
    return batch.submit(true);
}

void liftWhileBlending(RenderSettings& settings, SplatLift& lift, const gpu::Buffer& mask) {
    settings.contribution = &lift.sums;
    settings.contributionMode = 3;
    settings.liftMask = &mask;
}

void matteWhileBlending(RenderSettings& settings, const SplatLift& lift, const MatteSettings& how) {
    settings.matteLift = &lift.sums;
    settings.matteHard = how.hard;
    settings.matteThreshold = how.threshold;
}

Result<void> drawMatte(gpu::ShaderLibrary& library, const RenderTargets& traced, const SplatLift& lift,
                       const MatteSettings& settings, gpu::Buffer& matte) {
    if (!traced.hasTrace()) return noTrace();
    const uint64_t bytes = uint64_t{traced.width} * traced.height * sizeof(float);
    if (!matte.valid() || matte.bytes() != bytes) {
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = sizeof(float);
        desc.label = "splat.matte";
        auto made = gpu::Buffer::create(library.device(), desc);
        if (!made) return std::move(made).error();
        matte = *made;
    }
    auto kernel = gpu::ComputeKernel::create(library, "athenea/splat/splat_lift", "splatMatte");
    if (!kernel) return std::move(kernel).error();
    gpu::CommandBatch batch(library.device());
    kernel->dispatch(batch, {traced.width, traced.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["trace"].setBinding(traced.trace.rhi());
        // The mask is the other entry point's.
        cursor["mask"].setBinding(traced.depth.rhi());
        cursor["lift"].setBinding(lift.sums.rhi());
        cursor["matte"].setBinding(matte.rhi());
        cursor["params"]["width"].setData(traced.width);
        cursor["params"]["height"].setData(traced.height);
        cursor["params"]["slots"].setData(lift.slots);
        cursor["params"]["hard"].setData(uint32_t{settings.hard ? 1u : 0u});
        cursor["params"]["threshold"].setData(settings.threshold);
    });
    return batch.submit(true);
}

}   // namespace athenea::render
