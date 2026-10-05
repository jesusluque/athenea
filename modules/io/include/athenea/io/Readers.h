// Copyright (c) 2026 jesus luque.
#pragma once

#include <filesystem>

#include "athenea/core/Result.h"
#include "athenea/io/RawSplats.h"

namespace athenea::io {

/// A Gaussian splat file by extension: .ply (3DGS), .splat, .spz, .glb.
[[nodiscard]] Result<RawSplats> readSplats(const std::filesystem::path& path);

[[nodiscard]] Result<RawSplats> readSplatPly(const std::filesystem::path& path);
[[nodiscard]] Result<RawSplats> readDotSplat(const std::filesystem::path& path);
/// Niantic SPZ, v2-v4. The CPU only decompresses and arranges the quantised
/// bytes; the GPU decode dequantises them and turns right-up-back into the
/// PLY's right-down-front.
[[nodiscard]] Result<RawSplats> readSpz(const std::filesystem::path& path);
/// A glTF binary with one KHR_gaussian_splatting primitive whose attributes
/// are float accessors of their own buffer views -- what `athenea flatten`
/// writes, read back to be checked. Node transforms are not applied: a file
/// of another unit than the metre is placed by whoever reads it (its scale
/// is the node's, `metresPerUnit` in what flatten writes beside it).
[[nodiscard]] Result<RawSplats> readGlbGaussians(const std::filesystem::path& path);

/// A point file by name: .ply (ascii or binary), .xyz/.txt/.pts/.csv,
/// COLMAP points3D.txt / points3D.bin.
[[nodiscard]] Result<RawPoints> readPoints(const std::filesystem::path& path);

}   // namespace athenea::io
