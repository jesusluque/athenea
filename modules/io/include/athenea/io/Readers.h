// Copyright (c) 2026 jesus luque.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/io/RawSplats.h"

namespace athenea::io {

/// A Gaussian splat file by extension: .ply (3DGS), .splat, .spz.
[[nodiscard]] Result<RawSplats> readSplats(const std::filesystem::path& path);

[[nodiscard]] Result<RawSplats> readSplatPly(const std::filesystem::path& path);
[[nodiscard]] Result<RawSplats> readDotSplat(const std::filesystem::path& path);
/// Niantic SPZ, v2-v4 -- the engine's input contract for splats. The CPU only
/// decompresses (spz::loadSpzPacked) and keeps the six quantised streams as
/// they are (SplatEncoding::Record::SpzPacked); the device turns them into
/// records, dequantises them and turns the file's axes into the PLY's
/// right-down-front. The axes are the file's own (the coordinate-system
/// extension; right-up-back without one): right-up-back and right-down-front
/// are read, any other system is refused. `antialiased` is the file's flag.
[[nodiscard]] Result<RawSplats> readSpz(const std::filesystem::path& path);
/// The same from bytes in memory (a page off the network, a file already
/// read); `source` names it in messages. Any thread.
[[nodiscard]] Result<RawSplats> readSpz(std::span<const uint8_t> bytes, const std::string& source);

/// The records spz_records.slang writes from an SPZ's streams, and how the
/// decode reads them: `shDim` bases per colour in the file, its fixed point's
/// `fractionalBits`, its rotation layout, and whether its axes are
/// right-up-back (`flipYZ`). What readSpz sets; any thread.
[[nodiscard]] SplatEncoding spzEncoding(uint32_t shDim, uint32_t fractionalBits, bool smallestThree, bool flipYZ);

/// A point file by name: .ply (ascii or binary), .xyz/.txt/.pts/.csv,
/// COLMAP points3D.txt / points3D.bin.
[[nodiscard]] Result<RawPoints> readPoints(const std::filesystem::path& path);

}   // namespace athenea::io
