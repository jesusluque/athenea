// Copyright (c) 2026 jesus luque.
#include "athenea/render/SplatQuery.h"

#include <array>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

Result<SplatIdReading> measureSplatId(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud,
                                      uint32_t id) {
    SplatIdReading out;
    if (cloud.count == 0 || !cloud.hasCrypto()) {
        return out;
    }
    gpu::Device& device = library.device();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/splat/splat_id_stats", "splatIdStats");
    if (!kernel) return std::move(kernel).error();
    // Sums start at zero and the bounds at the far end of the byte range, so
    // that the first gaussian to arrive sets both.
    const std::array<uint32_t, 10> start{0, 0, 255, 0, 0, 255, 0, 0, 255, 0};
    gpu::BufferDesc desc;
    desc.bytes = sizeof(start);
    desc.elementBytes = 4;
    desc.label = "splat.id.stats";
    auto stats = gpu::Buffer::create(device, desc, start.data());
    if (!stats) return std::move(stats).error();

    const bool hasPbr = cloud.hasPbr();
    {
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["cloudCrypto"].setBinding(cloud.crypto.rhi());
            cursor["pbr"].setBinding(hasPbr ? cloud.pbr.rhi() : cloud.shape.rhi());
            cursor["stats"].setBinding(stats->rhi());
            cursor["params"]["count"].setData(cloud.count);
            cursor["params"]["id"].setData(id);
            cursor["params"]["hasPbr"].setData(uint32_t{hasPbr ? 1u : 0u});
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::array<uint32_t, 10> read{};
    ATHENEA_TRY(stats->read(device, 0, sizeof(read), read.data()));
    out.count = read[0];
    out.hasPbr = hasPbr && out.count > 0;
    if (!out.hasPbr) {
        return out;
    }
    for (uint32_t c = 0; c < 3; ++c) {
        const uint32_t at = 1 + c * 3;
        out.mean[c] = float(read[at]) / (255.0F * float(out.count));
        out.low[c] = float(read[at + 1]) / 255.0F;
        out.high[c] = float(read[at + 2]) / 255.0F;
    }
    return out;
}

}   // namespace athenea::render
