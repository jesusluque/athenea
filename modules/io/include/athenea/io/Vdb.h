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
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::io {

/// The value types a grid is read in. Each is laid out as NanoVDB's own
/// typed grid of that type (a re-layout: no value is converted on the
/// host), and the device reads it through nanovdb.slang's typed accessors.
/// A float grid the file stored as half (OpenVDB's `is_saved_as_half_float`,
/// Houdini's 16-bit VDBs) is a float grid once read: OpenVDB decodes it as
/// it decompresses. Bool, int64 and the integer vectors are refused.
enum class VdbType : uint32_t { Float, Double, Int32, Vec3f, Vec3d, Mask };

/// The type's name as UsdVol's fieldDataType spells it ("float", "double3"...).
[[nodiscard]] const char* vdbTypeName(VdbType type) noexcept;

/// The type a UsdVol fieldDataType token names: half and float are a float
/// grid, half3 and float3 a Vec3f one; any other token is refused.
[[nodiscard]] Result<VdbType> vdbTypeFromDataType(const std::string& dataType);

/// What a vector field's values are (UsdVol's vectorDataRoleHint; OpenVDB's
/// vector type where no hint is authored).
enum class VectorRole : uint32_t { None, Point, Normal, Vector, Color };

/// A vectorDataRoleHint token as a role ("" and "None" are None).
[[nodiscard]] Result<VectorRole> vectorRoleFromHint(const std::string& hint);

/// Which grid of a file a field reads, as UsdVol's OpenVDBAsset says it:
/// the grids named `name` (every grid where it is empty), in the file's
/// order; `index` (fieldIndex) counts among them, and `dataType`
/// (fieldDataType) must agree with the grid it picks -- without an index,
/// the first that agrees. Neither: the first.
struct VdbSelection {
    std::string name;
    int         index = -1;
    std::string dataType;
};

/// A grid as NanoVDB lays it out, ready to upload.
struct NanoGrid {
    std::string           name;
    VdbType               type = VdbType::Float;
    uint32_t              nanoType = 1;        ///< NanoVDB's GridType code (PNANOVDB_GRID_TYPE_*), the header's
    bool                  savedAsHalf = false;   ///< the file held its values as half floats
    VectorRole            vectorRole = VectorRole::None;   ///< the header's vector type, as a role
    /// A grid of any type but float: a float grid of its topology, every
    /// value 0, which world::VolumeSet fills on the device with this grid's
    /// scalar -- the medium walks float grids alone. Null for a float grid.
    std::shared_ptr<const NanoGrid> scalar;
    uint32_t              lowerCount = 0;      ///< internal nodes, the header's counts
    uint32_t              upperCount = 0;
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
    double                 value = 1.0;            ///< a scalar grid's (an int grid's, truncated; a mask's: active)
    std::array<double, 3>  vector{0.0, 0.0, 0.0};  ///< a vector grid's
};

/// Whether this build reads .vdb at all (OpenVDB in the toolchain).
[[nodiscard]] bool haveOpenVdb() noexcept;

/// Whether this build reads Field3D (.f3d) at all. No toolchain this engine
/// builds with carries the Field3D library, so none does yet.
[[nodiscard]] bool haveField3d() noexcept;

/// One field of a Field3D file (a UsdVol Field3DAsset). In a build without
/// Field3D -- every build so far -- an Unsupported error that says so.
[[nodiscard]] Result<NanoGrid> readField3dGrid(const std::filesystem::path& path, const VdbSelection& selection);

/// The names of the grids a .vdb file holds.
[[nodiscard]] Result<std::vector<std::string>> vdbGridNames(const std::filesystem::path& path);

/// One grid of a .vdb, as NanoVDB of its own type. `name` empty takes the first.
[[nodiscard]] Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const std::string& name = {});
/// The grid a field's selection picks (VdbSelection).
[[nodiscard]] Result<NanoGrid> readVdbGrid(const std::filesystem::path& path, const VdbSelection& selection);

/// Writes a fog grid of `boxes` filled at their values (a test's input; the
/// count of voxels it holds is known by construction, not measured), of
/// `type`; `saveAsHalf` stores a float grid's values as half floats.
[[nodiscard]] Result<void> writeVdbBoxes(const std::filesystem::path& path, const std::string& name,
                                         double voxelSize, std::span<const VdbBox> boxes,
                                         VdbType type = VdbType::Float, bool saveAsHalf = false);

/// One grid of a fixture file: boxes of one type, at a voxel size.
struct VdbGridSpec {
    std::string         name = "density";
    VdbType             type = VdbType::Float;
    double              voxelSize = 0.1;
    std::vector<VdbBox> boxes;
    bool                saveAsHalf = false;
    VectorRole          vectorRole = VectorRole::None;   ///< written as OpenVDB's vector type
};

/// Writes several grids to one file, in order (names may repeat: OpenVDB
/// tells them apart as "name[1]"...). A test's input.
[[nodiscard]] Result<void> writeVdbGrids(const std::filesystem::path& path, std::span<const VdbGridSpec> grids);

/// Writes a level set of the box [min, max) in index space (world = index x
/// `voxelSize`), with a band of `halfWidth` voxels each side: OpenVDB's own
/// box level set, its interior the inactive tiles a level set keeps. A
/// test's input.
[[nodiscard]] Result<void> writeVdbLevelSetBox(const std::filesystem::path& path, const std::string& name,
                                               double voxelSize, const VdbBox& box, float halfWidth);

}   // namespace athenea::io
