// Copyright (c) 2026 jesus luque.
#pragma once

#include <filesystem>

#include "athenea/core/Result.h"
#include "athenea/io/RawSplats.h"

namespace athenea::io {

/// A Gaussian splat file by extension: .ply (3DGS), .splat, .spz.
[[nodiscard]] Result<RawSplats> readSplats(const std::filesystem::path& path);

[[nodiscard]] Result<RawSplats> readSplatPly(const std::filesystem::path& path);
[[nodiscard]] Result<RawSplats> readDotSplat(const std::filesystem::path& path);
/// Niantic SPZ, v2-v4. The CPU only decompresses and arranges the quantised
/// bytes; the GPU decode dequantises them and turns right-up-back into the
/// PLY's right-down-front.
[[nodiscard]] Result<RawSplats> readSpz(const std::filesystem::path& path);

/// A point file by name: .ply (ascii or binary), .xyz/.txt/.pts/.csv,
/// COLMAP points3D.txt / points3D.bin.
[[nodiscard]] Result<RawPoints> readPoints(const std::filesystem::path& path);

}   // namespace athenea::io
