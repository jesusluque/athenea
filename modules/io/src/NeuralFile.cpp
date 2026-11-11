// Copyright (c) 2026 jesus luque.
#include "athenea/io/NeuralFile.h"

#include <array>
#include <cstring>
#include <fstream>

namespace athenea::io {
namespace {

constexpr std::array<char, 4> kMagic{'A', 'T', 'H', 'N'};
constexpr uint32_t kVersion = 1;

}   // namespace

Result<void> writeNeural(const std::filesystem::path& path, const NeuralFile& file) {
    if (file.latent.size() != uint64_t{file.count} * file.wordsEach) {
        return Error(ErrorCode::InvalidArgument, "a neural appearance's latents are not its count times its words");
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return Error(ErrorCode::IoFailure, "cannot write " + path.string());
    }
    const std::array<uint32_t, 6> header{kVersion,       file.count,
                                         file.stride,    file.wordsEach,
                                         static_cast<uint32_t>(file.weights.size()), file.lit ? 1u : 0u};
    out.write(kMagic.data(), kMagic.size());
    out.write(reinterpret_cast<const char*>(header.data()), sizeof(header));
    out.write(reinterpret_cast<const char*>(file.latent.data()),
              static_cast<std::streamsize>(file.latent.size() * sizeof(uint32_t)));
    out.write(reinterpret_cast<const char*>(file.weights.data()),
              static_cast<std::streamsize>(file.weights.size() * sizeof(float)));
    if (!out) {
        return Error(ErrorCode::IoFailure, "writing " + path.string() + " failed");
    }
    return ok();
}

Result<NeuralFile> readNeural(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Error(ErrorCode::NotFound, "cannot open " + path.string());
    }
    std::array<char, 4> magic{};
    std::array<uint32_t, 6> header{};
    in.read(magic.data(), magic.size());
    in.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    if (!in || magic != kMagic) {
        return Error(ErrorCode::InvalidArgument, path.string() + " is not a neural appearance (.athn)");
    }
    if (header[0] != kVersion) {
        return Error(ErrorCode::Unsupported,
                     path.string() + " is version " + std::to_string(header[0]) + " of .athn; this reads 1");
    }
    NeuralFile file;
    file.count = header[1];
    file.stride = header[2];
    file.wordsEach = header[3];
    file.lit = (header[5] & 1u) != 0;
    // Sizes a header can state are held to what a cloud can be before anything is allocated by them.
    if (file.wordsEach == 0 || file.wordsEach > 64 || file.stride > file.wordsEach * 2 || header[4] > (1u << 20)) {
        return Error(ErrorCode::InvalidArgument, path.string() + ": its header's sizes are not a neural appearance's");
    }
    file.latent.resize(uint64_t{file.count} * file.wordsEach);
    file.weights.resize(header[4]);
    in.read(reinterpret_cast<char*>(file.latent.data()),
            static_cast<std::streamsize>(file.latent.size() * sizeof(uint32_t)));
    in.read(reinterpret_cast<char*>(file.weights.data()),
            static_cast<std::streamsize>(file.weights.size() * sizeof(float)));
    if (!in) {
        return Error(ErrorCode::IoFailure, path.string() + " ends before its header says it does");
    }
    return file;
}

}   // namespace athenea::io
