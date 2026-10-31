// Copyright (c) 2026 jesus luque.
//
// OpenVDB in, NanoVDB out: a grid read from a .vdb and laid out as the
// NanoVDB buffer a kernel reads through PNanoVDB. The conversion is a
// re-layout, of the same class as decompressing SPZ: NanoVDB's statistics
// (bounding boxes, extrema, means) are disabled, and nothing about the
// values is read on the host -- every bound, maximum and majorant a
// volume needs is a kernel's (world::VolumeSet). What the host reads is
// the header: the grid's name, its transform, how many leaves it has.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::io {

/// A float grid as NanoVDB lays it out, ready to upload.
struct NanoGrid {
    std::string           name;
    std::vector<uint32_t> words;        ///< the NanoVDB buffer, 32-byte aligned at word 0
    uint32_t              leafCount = 0;   ///< leaves in the tree (the header's count)
    double                voxelSize = 1.0;
    /// Index to world, a row-vector 4x4 (v * M, translation in the last
    /// row), as OpenVDB's linear map holds it.
    std::array<double, 16> indexToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    /// A level set (OpenVDB's GRID_LEVEL_SET class, a header field): its
    /// values are signed distances, negative inside, and a kernel turns them
    /// into a fog's density (world::VolumeSet). `background` is the grid's
    /// background value, which for a level set is its band's half width in
    /// the distance's units; read from the header, as the class is.
    bool  levelSet = false;
    float background = 0.0F;
};

/// A box of voxels [min, max) at one value: what a fixture is made of.
struct VdbBox {
    std::array<int32_t, 3> min{0, 0, 0};
    std::array<int32_t, 3> max{8, 8, 8};
    float                  value = 1.0F;
};

/// Whether this build reads .vdb at all (OpenVDB in the toolchain).
[[nodiscard]] bool haveOpenVdb() noexcept;

/// The names of the grids a .vdb file holds.
[[nodiscard]] Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path);

/// One float grid of a .vdb, as NanoVDB. `name` empty takes the first.
[[nodiscard]] Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string& name = {});

/// Writes a fog grid of `boxes` filled at their values (a test's input; the
/// count of voxels it holds is known by construction, not measured).
[[nodiscard]] Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string& name,
                                         double voxelSize, std::span<const VdbBox> boxes);

/// Writes a level set of the box [min, max) in index space (world = index x
/// `voxelSize`), with a band of `halfWidth` voxels each side: OpenVDB's own
/// box level set, its interior the inactive tiles a level set keeps. A
/// test's input.
[[nodiscard]] Result<void> writeVdbLevelSetBox(const std::filesystem::path& path, const std::string& name,
                                               double voxelSize, const VdbBox& box, float halfWidth);

}   // namespace athenea::io
