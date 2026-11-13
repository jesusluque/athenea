// Copyright (c) 2026 jesus luque.
#include "athenea/render/Framing.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu/algo/RadixSort.h"

namespace athenea::render {

namespace {

Result<gpu::Buffer> words(gpu::Device& device, uint32_t count, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{std::max(count, 1u)} * 4;
    desc.elementBytes = 4;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

/// splat_core_keys.slang's key back to the float it was made from.
float fromKey(uint32_t key) {
    const uint32_t bits = (key & 0x80000000u) != 0 ? key & 0x7FFFFFFFu : ~key;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

}   // namespace

Result<scene::Bounds> coreBounds(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud, double share) {
    scene::Bounds out;
    const uint32_t n = cloud.count;
    if (n == 0) {
        return out;
    }
    gpu::Device& device = library.device();
    auto keys = gpu::ComputeKernel::create(library, "athenea/splat/splat_core_keys", "splatCoreKeys");
    if (!keys) return std::move(keys).error();
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    gpu::SortBuffers buffers;
    {
        auto a = words(device, n, "core.keys");
        if (!a) return std::move(a).error();
        auto b = words(device, n, "core.values");
        if (!b) return std::move(b).error();
        auto c = words(device, n, "core.keys.scratch");
        if (!c) return std::move(c).error();
        auto d = words(device, n, "core.values.scratch");
        if (!d) return std::move(d).error();
        buffers.keysLo = std::move(*a);
        buffers.values = std::move(*b);
        buffers.scratchKeysLo = std::move(*c);
        buffers.scratchValues = std::move(*d);
    }
    const double kept = std::clamp(share, 0.0, 1.0);
    const uint32_t lo = static_cast<uint32_t>(std::floor(double(n - 1) * (1.0 - kept) * 0.5));
    const uint32_t hi = n - 1 - lo;
    for (uint32_t axis = 0; axis < 3; ++axis) {
        gpu::CommandBatch batch(device);
        keys->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud.positions.rhi());
            cursor["keys"].setBinding(buffers.keysLo.rhi());
            cursor["values"].setBinding(buffers.values.rhi());
            cursor["params"]["count"].setData(n);
            cursor["params"]["axis"].setData(axis);
        });
        ATHENEA_TRY(sort->sort(batch, buffers, n, 32));
        ATHENEA_TRY(batch.submit(true));
        uint32_t at[2] = {0, 0};
        ATHENEA_TRY(buffers.keysLo.read(device, uint64_t{lo} * 4, 4, &at[0]));
        ATHENEA_TRY(buffers.keysLo.read(device, uint64_t{hi} * 4, 4, &at[1]));
        out.min[axis] = fromKey(at[0]);
        out.max[axis] = fromKey(at[1]);
    }
    return out;
}

}   // namespace athenea::render
