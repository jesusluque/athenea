// Copyright (c) 2026 jesus luque.
#include "athenea/render/NeuralAppearance.h"

#include <algorithm>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {

Result<NeuralCloud> makeNeuralCloud(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud, uint32_t seed) {
    gpu::Device& device = library.device();
    NeuralCloud neural;
    neural.count = cloud.count;
    const auto floats = [&](uint64_t count, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(count, 1) * sizeof(float);
        desc.elementBytes = sizeof(float);
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    auto latent = floats(uint64_t{cloud.count} * kNeuralStride, "neural.latent");
    if (!latent) return std::move(latent).error();
    auto weights = floats(kNeuralWeights, "neural.weights");
    if (!weights) return std::move(weights).error();
    neural.latent = *latent;
    neural.weights = *weights;
    auto initLatent = gpu::ComputeKernel::create(library, "athenea/neural/neural_shade", "neuralInitLatent");
    if (!initLatent) return std::move(initLatent).error();
    auto initWeights = gpu::ComputeKernel::create(library, "athenea/neural/neural_shade", "neuralInitWeights");
    if (!initWeights) return std::move(initWeights).error();
    gpu::CommandBatch batch(device);
    // Every name the module declares is bound; what an entry point does not
    // read is handed a buffer of the right kind.
    const auto bind = [&](rhi::ShaderCursor cursor, const gpu::Buffer& made, uint32_t count) {
        cursor["positions"].setBinding(cloud.positions.rhi());
        cursor["shape"].setBinding(cloud.shape.rhi());
        cursor["visible"].setBinding(cloud.shape.rhi());
        cursor["latent"].setBinding(neural.latent.rhi());
        cursor["weights"].setBinding(neural.weights.rhi());
        cursor["made"].setBinding(made.rhi());
        cursor["proj"].setBinding(cloud.positions.rhi());
        cursor["params"]["count"].setData(count);
        cursor["params"]["base"].setData(uint32_t{0});
        cursor["params"]["linearCloud"].setData(uint32_t{cloud.linear ? 1u : 0u});
        cursor["params"]["seed"].setData(seed);
        // Glorot's range for the widest layer: sqrt(6 / (in + out)).
        cursor["params"]["spread"].setData(0.35F);
    };
    if (cloud.count > 0) {
        initLatent->dispatch(batch, {cloud.count, 1, 1},
                             [&](rhi::ShaderCursor cursor) { bind(cursor, neural.latent, cloud.count); });
    }
    initWeights->dispatch(batch, {kNeuralWeights, 1, 1},
                          [&](rhi::ShaderCursor cursor) { bind(cursor, neural.weights, kNeuralWeights); });
    ATHENEA_TRY(batch.submit(true));
    return neural;
}

}   // namespace athenea::render
