// Copyright (c) 2026 jesus luque.
#include "athenea/render/NeuralAppearance.h"

#include <algorithm>
#include <string>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"

namespace athenea::render {

Result<NeuralCloud> makeNeuralCloud(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud, uint32_t seed,
                                    bool lit, bool split) {
    gpu::Device& device = library.device();
    NeuralCloud neural;
    neural.count = cloud.count;
    neural.lit = lit;
    neural.split = split;
    const auto floats = [&](uint64_t count, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(count, 1) * sizeof(float);
        desc.elementBytes = sizeof(float);
        desc.label = label;
        return gpu::Buffer::create(device, desc);
    };
    auto latent = floats(uint64_t{cloud.count} * kNeuralStride, "neural.latent");
    if (!latent) return std::move(latent).error();
    auto weights = floats(neural.weightCount(), "neural.weights");
    if (!weights) return std::move(weights).error();
    neural.latent = *latent;
    neural.weights = *weights;
    {
        const std::vector<uint32_t> never((cloud.count + kNeuralChunk - 1) / kNeuralChunk + 1, 0u);
        gpu::BufferDesc desc;
        desc.bytes = never.size() * sizeof(uint32_t);
        desc.elementBytes = sizeof(uint32_t);
        desc.label = "neural.chunkChanged";
        auto stamps = gpu::Buffer::create(device, desc, never.data());
        if (!stamps) return std::move(stamps).error();
        neural.chunkChanged = *stamps;
    }
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
        cursor["params"]["lit"].setData(uint32_t{0});
        cursor["params"]["lightCount"].setData(uint32_t{0});
        cursor["shadowFactors"].setBinding(neural.weights.rhi());
        cursor["params"]["shadowLights"].setData(uint32_t{0});
        cursor["params"]["split"].setData(uint32_t{0});
        cursor["params"]["cached"].setData(uint32_t{0});
        cursor["mattRead"].setBinding(cloud.positions.rhi());
        cursor["mattOut"].setBinding(made.rhi());
        cursor["chunkChanged"].setBinding(cloud.shape.rhi());
        cursor["chunkDone"].setBinding(made.rhi());
    };
    if (cloud.count > 0) {
        initLatent->dispatch(batch, {cloud.count, 1, 1},
                             [&](rhi::ShaderCursor cursor) { bind(cursor, neural.latent, cloud.count); });
    }
    initWeights->dispatch(batch, {neural.weightCount(), 1, 1},
                          [&](rhi::ShaderCursor cursor) { bind(cursor, neural.weights, neural.weightCount()); });
    ATHENEA_TRY(batch.submit(true));
    return neural;
}

namespace {

Result<gpu::Buffer> made(gpu::Device& device, uint64_t bytes, uint32_t element, const char* label,
                         const void* initial = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(bytes, element);
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc, initial);
}

}   // namespace

Result<std::vector<uint32_t>> packNeuralLatents(gpu::ShaderLibrary& library, const NeuralCloud& neural) {
    gpu::Device& device = library.device();
    auto packed = made(device, uint64_t{neural.count} * kNeuralWords * 4, 4, "neural.packed");
    if (!packed) return std::move(packed).error();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/neural/neural_pack", "neuralPack");
    if (!kernel) return std::move(kernel).error();
    if (neural.count > 0) {
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {neural.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["floats"].setBinding(neural.latent.rhi());
            cursor["packed"].setBinding(packed->rhi());
            // The other entry point's.
            cursor["packedRead"].setBinding(packed->rhi());
            cursor["unpacked"].setBinding(neural.weights.rhi());
            cursor["params"]["count"].setData(neural.count);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::vector<uint32_t> words(uint64_t{neural.count} * kNeuralWords);
    if (!words.empty()) {
        ATHENEA_TRY(packed->read(device, 0, words.size() * sizeof(uint32_t), words.data()));
    }
    return words;
}

Result<void> NeuralCloud::touch(gpu::Device& device, uint32_t first, uint32_t touched) {
    if (!chunkChanged.valid() || touched == 0 || first >= count) {
        return ok();
    }
    ++stamp;
    const uint32_t from = first / kNeuralChunk;
    const uint32_t to = (std::min(first + touched, count) - 1) / kNeuralChunk;
    const std::vector<uint32_t> now(to - from + 1, stamp);
    return chunkChanged.write(device, uint64_t{from} * sizeof(uint32_t), now.size() * sizeof(uint32_t), now.data());
}

Result<std::vector<float>> readNeuralWeights(gpu::ShaderLibrary& library, const NeuralCloud& neural) {
    std::vector<float> weights(neural.weightCount());
    ATHENEA_TRY(neural.weights.read(library.device(), 0, weights.size() * sizeof(float), weights.data()));
    return weights;
}

Result<NeuralCloud> neuralCloudFrom(gpu::ShaderLibrary& library, uint32_t count,
                                    std::span<const uint32_t> latentWords, std::span<const float> weights,
                                    bool lit) {
    const bool split = weights.size() == size_t{kNeuralWeights} * 2;
    if (latentWords.size() != uint64_t{count} * kNeuralWords || (weights.size() != kNeuralWeights && !split)) {
        return Error(ErrorCode::InvalidArgument,
                     "a neural appearance of other sizes than this engine's: " + std::to_string(weights.size()) +
                         " weights where it has " + std::to_string(kNeuralWeights));
    }
    gpu::Device& device = library.device();
    NeuralCloud neural;
    neural.count = count;
    neural.lit = lit;
    neural.split = split;
    auto latent = made(device, uint64_t{count} * kNeuralStride * sizeof(float), sizeof(float), "neural.latent");
    if (!latent) return std::move(latent).error();
    auto held = made(device, weights.size_bytes(), sizeof(float), "neural.weights", weights.data());
    if (!held) return std::move(held).error();
    neural.latent = *latent;
    neural.weights = *held;
    if (count == 0) {
        return neural;
    }
    auto words = made(device, latentWords.size_bytes(), 4, "neural.packed", latentWords.data());
    if (!words) return std::move(words).error();
    auto kernel = gpu::ComputeKernel::create(library, "athenea/neural/neural_pack", "neuralUnpack");
    if (!kernel) return std::move(kernel).error();
    gpu::CommandBatch batch(device);
    kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["packedRead"].setBinding(words->rhi());
        cursor["unpacked"].setBinding(neural.latent.rhi());
        // The other entry point's.
        cursor["floats"].setBinding(neural.weights.rhi());
        cursor["packed"].setBinding(words->rhi());
        cursor["params"]["count"].setData(count);
    });
    ATHENEA_TRY(batch.submit(true));
    return neural;
}

}   // namespace athenea::render
