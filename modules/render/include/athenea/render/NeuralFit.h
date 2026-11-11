// Copyright (c) 2026 jesus luque.
//
// A neural appearance (NeuralAppearance.h) fitted to pictures on the device:
// shaders/athenea/neural/neural_fit.slang. The pictures here are the cloud's
// own, drawn by its harmonics -- a cloud's appearance distilled into its
// latents, gaussian for gaussian, its geometry as it is.
#pragma once

#include <cstdint>
#include <optional>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/render/Camera.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/TileRasterizer.h"

namespace athenea::render {

struct NeuralFitSettings {
    uint32_t width = 640;
    uint32_t height = 360;
    /// Adam's rates: the features', the base colours', the network's.
    float    rateFeatures = 5.0e-3F;
    float    rateBase = 2.0e-3F;
    float    rateWeights = 2.0e-3F;
};

class NeuralFit {
public:
    [[nodiscard]] static Result<NeuralFit> create(gpu::ShaderLibrary& library);

    /// One step from one camera: `instance` drawn by its harmonics is what
    /// `neural` drawn from the same camera should be, and `neural` is moved
    /// towards it. With `readLoss`, returns the frame's mean squared error
    /// before the step (linear light, the three channels summed) -- a wait
    /// for the device, so asked for now and then and not every step.
    [[nodiscard]] Result<std::optional<double>> step(TileRasterizer& rasterizer, const Camera& camera,
                                                      const SplatInstance& instance, NeuralCloud& neural,
                                                      const NeuralFitSettings& settings, bool readLoss = false);

    [[nodiscard]] uint32_t steps() const noexcept { return steps_; }

private:
    [[nodiscard]] Result<void> reserve(const NeuralCloud& neural, const NeuralFitSettings& settings);

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  error_, clear_, backward_, weights_, adam_;
    uint32_t            steps_ = 0;
    uint32_t            count_ = 0;
    uint32_t            pixels_ = 0;
    gpu::Buffer         colourGrad_, records_, latentGrad_, weightGrad_;
    gpu::Buffer         latentM_, latentV_, weightM_, weightV_;
    gpu::Buffer         pixelGrad_, lossRows_;
    RenderTargets       teacher_, student_, again_;
};

}   // namespace athenea::render
