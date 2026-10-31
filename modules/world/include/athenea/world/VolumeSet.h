// Copyright (c) 2026 jesus luque.
//
// The frame's volumes on the device, in one buffer of words: a header, a
// record a volume instance (grid offsets, world-to-index, extinction scale,
// albedo, phase, and -- written by kernels -- its index bounds and
// majorant), the NanoVDB grids, each once however many records read it (a
// level set made a fog by a kernel as it is laid out), and each leaf's
// largest value. The path tracer reads it
// all from that one buffer (shaders/athenea/volume/medium.slang), since a
// Metal kernel binds at most 31 and the traced one stands at the limit.
// Nothing here reads a value on the host: the host places words, counts
// leaves and composes transforms.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Vdb.h"
#include "athenea/render/Camera.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::world {

/// What a field is, from the name the Volume gives it: Houdini's and the
/// renderers' usual names (density; temperature or temp; flame, heat or
/// fuel; Cd, color or colour; vel, v or velocity). Others are kept, by name.
enum class FieldRole : uint32_t { Other, Density, Temperature, Flame, Color, Velocity };

[[nodiscard]] FieldRole fieldRoleOf(std::string_view name) noexcept;

/// FNV-1a of a field's name: how a kernel names the field it reads.
[[nodiscard]] uint32_t fieldNameHash(std::string_view name) noexcept;

/// How the medium looks a density up between voxels (athenea:volumeFilter):
/// trilinear between the eight about the point; the nearest; or one of the
/// eight drawn with the trilinear weights -- the trilinear value in
/// expectation, at one lookup.
enum class VolumeFilter : uint32_t { Trilinear = 0, Nearest = 1, Stochastic = 2 };

/// A filter's name as the setting spells it; nullopt for another.
[[nodiscard]] std::optional<VolumeFilter> volumeFilterFromName(std::string_view name) noexcept;

/// One of a volume's fields: its typed grid, by name.
struct VolumeFieldInput {
    const io::NanoGrid* grid = nullptr;
    uint32_t            nameHash = 0;
    FieldRole           role = FieldRole::Other;
    io::VectorRole      vectorRole = io::VectorRole::None;
};

/// One volume: a grid, where it stands, and how it scatters. A grid of any
/// type: one not of floats is walked as its float topology (NanoGrid::scalar),
/// filled on the device with its scalar.
struct VolumeInput {
    const io::NanoGrid*  grid = nullptr;   ///< shared: volumes naming one grid lay it out once
    render::Mat4         objectToWorld = render::Mat4::identity();   ///< the prim's; the grid's index-to-world composes under it
    /// An instanced volume: its instances' transforms, 3 float4 rows each
    /// (world::InstanceChain), each composed before `objectToWorld`, and one
    /// record each, written by a kernel. Null for a volume drawn once.
    const gpu::Buffer*   instanceRows = nullptr;
    uint32_t             instanceCount = 0;
    float                densityScale = 1.0F;   ///< density to extinction, per world unit
    std::array<float, 3> albedo{1.0F, 1.0F, 1.0F};
    float                g = 0.0F;
    /// Radiance of the absorbing part: what a collision that absorbs adds,
    /// so a medium of albedo one emits nothing and one of albedo zero glows
    /// `emission` where it is opaque (a Material's `uniform_edf`).
    std::array<float, 3> emission{0.0F, 0.0F, 0.0F};
    /// The light linking categories it is in (a bit each, as a mesh
    /// instance's): a linked light lights it only where its bit is set.
    uint64_t             categories = 0;
    /// Every field it names, the density among them, at most kMaxFields: a
    /// table in its record, read by name (shaders/athenea/volume/fields.slang).
    std::vector<VolumeFieldInput> fields;
    VolumeFilter         filter = VolumeFilter::Trilinear;
};

/// Where one grid sits in the words, for a kernel that reads it.
struct VolumeGridPlacement {
    uint32_t gridWord = 0;
    uint32_t leafCount = 0;
    uint32_t leafBase = 0;   ///< word offset of its leaf maxima
};

class VolumeSet {
public:
    [[nodiscard]] static Result<VolumeSet> create(gpu::ShaderLibrary& library);

    /// Lays the volumes out and runs the bounds, leaf maximum, instance and
    /// finish kernels over them.
    [[nodiscard]] Result<void> set(gpu::CommandBatch& batch, std::span<const VolumeInput> volumes);

    /// The grids laid out, each once, in the order the volumes first name them.
    [[nodiscard]] uint32_t count() const noexcept { return static_cast<uint32_t>(placements_.size()); }
    [[nodiscard]] const VolumeGridPlacement& placement(uint32_t i) const noexcept { return placements_[i]; }
    /// The records: one a volume, one an instance of an instanced one.
    [[nodiscard]] uint32_t records() const noexcept { return records_; }
    /// The words' size on the device: what the frame's volumes hold there.
    [[nodiscard]] uint64_t deviceBytes() const noexcept { return words_.valid() ? words_.bytes() : 0; }
    /// The one buffer: medium.slang's layout.
    [[nodiscard]] const gpu::Buffer& words() const noexcept { return words_; }
    /// Six ints a grid: min xyz, max xyz (exclusive), in index space.
    [[nodiscard]] const gpu::Buffer& bounds() const noexcept { return bounds_; }
    /// A float's bits a grid: its largest leaf maximum.
    [[nodiscard]] const gpu::Buffer& majorant() const noexcept { return majorant_; }

    static constexpr uint32_t kHeaderWords = 8;
    static constexpr uint32_t kMaxFields = 8;
    static constexpr uint32_t kFieldBase = 40;    ///< the field table's first word in a record
    static constexpr uint32_t kFieldWords = 16;
    static constexpr uint32_t kRecordWords = kFieldBase + kMaxFields * kFieldWords;

private:
    gpu::Device*                     device_ = nullptr;
    gpu::ComputeKernel               boundsKernel_, leafMaxKernel_, instancesKernel_, finishKernel_, levelSetKernel_,
                                     fillKernel_;
    gpu::Buffer                      words_, bounds_, majorant_;
    std::vector<VolumeGridPlacement> placements_;
    uint32_t                         records_ = 0;
};

}   // namespace athenea::world
