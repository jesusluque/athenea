// Copyright (c) 2026 jesus luque.
#include "athenea/light/LtcTable.h"

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::light {

Result<LtcTable> LtcTable::create(gpu::ShaderLibrary& library) {
    gpu::Device& device = library.device();
    LtcTable table;
    gpu::TextureDesc desc;
    desc.type = rhi::TextureType::Texture2D;
    desc.width = kSize;
    desc.height = kSize;
    desc.format = rhi::Format::RGBA32Float;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
    desc.label = "lights.ltc";
    auto texture = gpu::Texture::create(device, desc);
    if (!texture) return std::move(texture).error();
    table.texture_ = std::move(*texture);
    auto view = table.texture_.view(0);
    if (!view) return std::move(view).error();
    table.view_ = std::move(*view);
    gpu::BufferDesc state;
    state.bytes = uint64_t{kSize} * kSize * 16;
    state.elementBytes = 16;
    state.label = "lights.ltcState";
    auto made = gpu::Buffer::create(device, state);
    if (!made) return std::move(made).error();
    table.state_ = std::move(*made);
    auto fit = gpu::ComputeKernel::create(library, "athenea/light/ltc_fit", "ltcFit");
    if (!fit) return std::move(fit).error();
    // A row a dispatch, roughest first: each row starts from the one above.
    // A submit every few rows, so no command buffer runs for long.
    for (uint32_t done = 0; done < kSize;) {
        gpu::CommandBatch batch(device);
        for (uint32_t k = 0; k < 8 && done < kSize; ++k, ++done) {
            const uint32_t row = kSize - 1 - done;
            fit->dispatch(batch, {kSize, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["tableOut"].setBinding(table.view_.get());
                cursor["state"].setBinding(table.state_.rhi());
                cursor["size"].setData(kSize);
                cursor["row"].setData(row);
                cursor["samples"].setData(uint32_t{16});
            });
        }
        ATHENEA_TRY(batch.submit(true));
    }
    return table;
}

}   // namespace athenea::light
