// Copyright (c) 2026 jesus luque.
//
// A NEURAL APPEARANCE ON DISK: `.athn`, beside the cloud it is of. What a
// cloud is drawn with in place of its harmonics (render/NeuralAppearance.h):
// a latent a gaussian, as halves, and one network's weights, as floats. The
// words arrive here packed by the device and leave for it as they were read;
// this frames them and nothing else.
//
//   "ATHN", version, gaussians, halves a gaussian, words a gaussian, weights, flags (bit 0: lit;
//   bit 1: split -- two networks' weights, the first never told the view)
//   then gaussians * words-a-gaussian uint32 (two halves a word, low first)
//   then the weights, float32
//
// Little endian. It names no cloud: it is of the cloud it was fitted to,
// gaussian for gaussian in that cloud's order, and a reader checks the count.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::io {

struct NeuralFile {
    uint32_t              count = 0;            ///< gaussians
    uint32_t              stride = 0;           ///< halves a gaussian
    uint32_t              wordsEach = 0;        ///< uint32 a gaussian: (stride + 1) / 2
    bool                  lit = false;
    bool                  split = false;        ///< `weights` holds two networks
    std::vector<uint32_t> latent;               ///< count * wordsEach
    std::vector<float>    weights;
};

[[nodiscard]] Result<void> writeNeural(const std::filesystem::path& path, const NeuralFile& file);
[[nodiscard]] Result<NeuralFile> readNeural(const std::filesystem::path& path);

}   // namespace athenea::io
