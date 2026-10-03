// Copyright (c) 2026 jesus luque.
#include "athenea/scene/SplatInfluences.h"

#include <algorithm>
#include <array>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::scene {
namespace {

Result<gpu::Buffer> deviceBuffer(gpu::Device& device, uint64_t count, uint32_t element, const char* label,
                                 const void* data = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, data);
}

}   // namespace

Result<InfluencePacker> InfluencePacker::create(gpu::ShaderLibrary& library) {
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/splat_influences", "splatInfluencesPack");
    if (!kernel) return std::move(kernel).error();
    InfluencePacker made;
    made.device_ = &library.device();
    made.kernel_ = std::move(*kernel);
    return made;
}

Result<PackedInfluences> InfluencePacker::pack(const GpuSplats& splats, const InfluenceSource& source,
                                               InfluenceLayout layout, const char* label) {
    if (source.jointIndices == nullptr || source.jointWeights == nullptr || source.held == 0 ||
        source.perSplat == 0) {
        return Error(ErrorCode::InvalidArgument, "influences want their joints, their weights and a count");
    }
    if (layout == InfluenceLayout::Packed && source.joints > kPackedJoints) {
        return Error::make(ErrorCode::InvalidArgument,
                           "{} joints do not fit the packed influences' 16 bits; keep them as pairs", source.joints);
    }
    const uint32_t count = splats.count;
    const uint32_t words = layout == InfluenceLayout::Packed ? 1u : 2u;
    PackedInfluences out;
    out.layout = layout;
    out.perSplat = source.perSplat;
    auto made = deviceBuffer(*device_, uint64_t{count} * source.perSplat * words, 4 * words, label);
    if (!made) return std::move(made).error();
    out.words = std::move(*made);
    const std::array<uint32_t, 2> zero{0u, 0u};
    auto counters = deviceBuffer(*device_, zero.size(), 4, "splat.influences.counters", zero.data());
    if (!counters) return std::move(counters).error();
    if (count > 0) {
        gpu::CommandBatch batch(*device_);
        kernel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["origin"].setBinding(splats.origin.rhi());
            cursor["jointIndices"].setBinding(source.jointIndices->rhi());
            cursor["jointWeights"].setBinding(source.jointWeights->rhi());
            cursor["influences"].setBinding(out.words.rhi());
            cursor["counters"].setBinding(counters->rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["perSplat"].setData(source.perSplat);
            p["held"].setData(source.held);
            p["joints"].setData(source.joints);
            p["layout"].setData(static_cast<uint32_t>(layout));
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::array<uint32_t, 2> seen{0u, 0u};
    ATHENEA_TRY(counters->read(*device_, 0, sizeof(seen), seen.data()));
    out.jointsOutOfRange = seen[0];
    out.weightsOutOfRange = seen[1];
    return out;
}

Result<PackedInfluences> InfluencePacker::pack(const GpuSplats& splats, const int32_t* jointIndices,
                                               const float* jointWeights, uint32_t held, uint32_t perSplat,
                                               uint32_t joints, InfluenceLayout layout, const char* label) {
    if (jointIndices == nullptr || jointWeights == nullptr || held == 0) {
        return Error(ErrorCode::InvalidArgument, "influences want their joints and their weights");
    }
    auto indices = deviceBuffer(*device_, held, 4, "splat.jointIndices", jointIndices);
    if (!indices) return std::move(indices).error();
    auto weights = deviceBuffer(*device_, held, 4, "splat.jointWeights", jointWeights);
    if (!weights) return std::move(weights).error();
    InfluenceSource source;
    source.jointIndices = &*indices;
    source.jointWeights = &*weights;
    source.held = held;
    source.perSplat = perSplat;
    source.joints = joints;
    return pack(splats, source, layout, label);
}

}   // namespace athenea::scene
