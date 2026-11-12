// Copyright (c) 2026 jesus luque.
#include "athenea/render/Discovery.h"

#include <algorithm>
#include <array>

#include "athenea/gpu/Device.h"

namespace athenea::render {
namespace {

Result<gpu::Buffer> words(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                          const uint32_t* from = nullptr) {
    const std::vector<uint32_t> zeros(from == nullptr ? std::max<uint64_t>(count, 4) : 0, 0u);
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 4) * sizeof(uint32_t);
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, from != nullptr ? from : zeros.data());
}

}   // namespace

Result<Discovery> Discovery::create(gpu::ShaderLibrary& library) {
    Discovery d;
    d.library_ = &library;
    const auto make = [&](gpu::ComputeKernel& kernel, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/neural/scene_discover", entry);
        if (!made) return std::move(made).error();
        kernel = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(d.clear_, "discoverClear"));
    ATHENEA_TRY(make(d.mark_, "discoverMark"));
    ATHENEA_TRY(make(d.count_, "discoverCount"));
    ATHENEA_TRY(make(d.mask_, "discoverMask"));
    auto counters = words(library.device(), 4, 4, "discover.counters");
    if (!counters) return std::move(counters).error();
    d.counters_ = *counters;
    auto spare = words(library.device(), 16, 4, "discover.spare");
    if (!spare) return std::move(spare).error();
    d.spareWords_ = *spare;
    auto floats = words(library.device(), 16, 16, "discover.spareFloats");
    if (!floats) return std::move(floats).error();
    d.spareFloats_ = *floats;
    return d;
}

/// Every name the module declares, bound; a dispatch names its own after.
void Discovery::bind(rhi::ShaderCursor cursor) const {
    cursor["contribution"].setBinding(spareWords_.rhi());
    cursor["seen"].setBinding(spareWords_.rhi());
    cursor["counters"].setBinding(counters_.rhi());
    cursor["seenRead"].setBinding(spareWords_.rhi());
    cursor["positions"].setBinding(spareFloats_.rhi());
    cursor["positionsOut"].setBinding(spareFloats_.rhi());
    cursor["cleared"].setBinding(spareWords_.rhi());
}

Result<gpu::Buffer> Discovery::record(uint32_t count, std::span<const uint32_t> held) {
    if (!held.empty() && held.size() != count) {
        return Error(ErrorCode::InvalidArgument, "a record of what was seen is not of this piece's gaussians");
    }
    return words(library_->device(), count, 4, "discover.seen", held.empty() ? nullptr : held.data());
}

Result<gpu::Buffer> Discovery::zeros(uint32_t count) {
    return words(library_->device(), count, 4, "discover.contribution");
}

Result<LookCount> Discovery::mark(const gpu::Buffer& contribution, uint32_t base, gpu::Buffer& seen, uint32_t count,
                                  uint32_t stamp, float threshold) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    mark_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["contribution"].setBinding(contribution.rhi());
        cursor["seen"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["base"].setData(base);
        cursor["params"]["stamp"].setData(stamp);
        cursor["params"]["threshold"].setData(static_cast<uint32_t>(std::max(threshold, 0.0F) * 4096.0F + 0.5F));
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    return LookCount{read[0], read[1]};
}

Result<uint32_t> Discovery::known(const gpu::Buffer& seen, uint32_t count, uint32_t upTo) {
    gpu::Device& device = library_->device();
    gpu::CommandBatch batch(device);
    clear_.dispatch(batch, {4, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["cleared"].setBinding(counters_.rhi());
        cursor["counters"].setBinding(spareWords_.rhi());
        cursor["params"]["count"].setData(uint32_t{4});
    });
    count_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["seenRead"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["upTo"].setData(upTo);
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 4> read{};
    ATHENEA_TRY(counters_.read(device, 0, sizeof(read), read.data()));
    return read[2];
}

Result<scene::GpuSplats> Discovery::masked(const scene::GpuSplats& cloud, const gpu::Buffer& seen, uint32_t upTo) {
    gpu::Device& device = library_->device();
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(cloud.count, 1) * 16;
    desc.elementBytes = 16;
    desc.label = "discover.positions";
    auto positions = gpu::Buffer::create(device, desc);
    if (!positions) return std::move(positions).error();
    gpu::CommandBatch batch(device);
    mask_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        bind(cursor);
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["positionsOut"].setBinding(positions->rhi());
        cursor["seenRead"].setBinding(seen.rhi());
        cursor["params"]["count"].setData(cloud.count);
        cursor["params"]["upTo"].setData(upTo);
    });
    ATHENEA_TRY(batch.submit(true));
    scene::GpuSplats known = cloud;
    known.positions = *positions;
    return known;
}

}   // namespace athenea::render
