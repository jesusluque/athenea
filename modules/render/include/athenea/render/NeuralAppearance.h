// Copyright (c) 2026 jesus luque.
//
// A cloud's colours from a latent a gaussian and one small network, in place
// of its harmonics (shaders/athenea/neural/neural_appearance.slang).
#pragma once

#include <array>
#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::render {

/// The sizes neural_appearance.slang has.
inline constexpr uint32_t kNeuralFeatures = 8;
inline constexpr uint32_t kNeuralStride = kNeuralFeatures + 3;
inline constexpr uint32_t kNeuralIn = 16 + kNeuralFeatures * 2 + 9;
inline constexpr uint32_t kNeuralHidden = 16;
inline constexpr uint32_t kNeuralWeights = kNeuralHidden * kNeuralIn + kNeuralHidden * kNeuralHidden + 3 * kNeuralHidden;

/// WHAT A CLOUD IS DRAWN WITH IN PLACE OF ITS HARMONICS, where an instance
/// names one (`SplatInstance::neural`): `kNeuralStride` floats a gaussian --
/// its features, then its base colour in linear light -- and the network's
/// `kNeuralWeights` floats. Floats, since this is what a fit works on; what a
/// file would carry is halves.
struct NeuralCloud {
    gpu::Buffer latent;
    gpu::Buffer weights;
    uint32_t    count = 0;
    /// LIT: the cloud answers a light rather than holding one. Its base is
    /// then what a gaussian is made of, the network says how that answers a
    /// unit of light from a direction, and an instance drawn by it names the
    /// frame's light (`SplatInstance::neuralLight`).
    bool        lit = false;
};

/// The one light a lit neural cloud is drawn under: where it comes from (a
/// unit vector towards it, in the world) and its radiance, linear rgb.
struct NeuralLight {
    std::array<float, 3> towards{0.0F, 0.0F, 1.0F};
    std::array<float, 3> radiance{1.0F, 1.0F, 1.0F};
};

/// Where a fit starts: every gaussian's base the light its own colour is,
/// its features zero, and a network whose last layer is zero -- so the cloud
/// is drawn as its harmonics' first term draws it.
[[nodiscard]] Result<NeuralCloud> makeNeuralCloud(gpu::ShaderLibrary& library, const scene::GpuSplats& cloud,
                                                  uint32_t seed = 1, bool lit = false);

}   // namespace athenea::render
