// Copyright (c) 2026 jesus luque.
//
// `athenea flatten` (docs/decisions.md, task TXF): a TX cloud under a sky and
// lights, written as the degree 3 harmonics a standard Gaussian Splatting file
// holds, with no bake again. Three steps, all on the device:
//
// 1. `FlattenFit::prepare`: the fit's matrix M for the N directions and the
//    roughness floor (shaders/athenea/splat/flatten_fit.slang), once.
// 2. The frame: `TileRasterizer::render` with `RenderSettings::flatten` set
//    shades each gaussian of every relit TX cloud (and every shadow catcher)
//    from N directions exactly as it shades it for a camera, and fits what
//    that shows (splat_project.slang's `splatFlatten`), into one record a
//    gaussian (flatten_basis.slang's layout).
// 3. `FlattenPack`: those records as a format's own numbers -- PLY floats,
//    SPZ's six byte streams, glTF attribute blocks (flatten_pack.slang) --
//    which athenea/io/SplatWriters.h only frames and writes.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::render {

/// Floats a flattened gaussian (flatten_basis.slang's `kFlatStride`).
inline constexpr uint32_t kFlatStride = 64;

struct FlattenSettings {
    /// Fibonacci directions a gaussian is looked at from.
    uint32_t directions = 256;
    /// Tikhonov weight, times l^2 (l + 1)^2 for band l.
    float    lambda = 1.0e-4F;
    /// What a surface's back hemisphere weighs in the fit; a thin wall is
    /// weighed all round.
    float    backWeight = 1.0e-2F;
    /// Fit the sRGB-encoded colour a viewer shows (true), or linear light.
    bool     display = true;
    /// Linear factor on the light before it is encoded.
    float    exposure = 1.0F;
    /// Evaluate reflections no sharper than degree 3 holds.
    bool     roughnessFloor = true;
    /// The share of a reflection lobe's energy allowed above band 3.
    float    floorShare = 0.05F;
    /// World -> file: rows of a rotation, and the world's unit in the file's.
    std::array<float, 9> toFile{1, 0, 0, 0, 1, 0, 0, 0, 1};
    float    unitScale = 1.0F;
    /// Gaussians a dispatch: each looks `directions` times.
    uint32_t slice = 1u << 16;
};

/// One flattened cloud: `count` records of `kFlatStride` floats.
struct FlatCloud {
    uint32_t    instance = 0;   ///< its index in the frame's instances
    uint32_t    count = 0;
    bool        catcher = false;
    gpu::Buffer records;
};

/// What a frame is asked to flatten, and what it hands back.
struct SplatFlatten {
    FlattenSettings        settings;
    const gpu::Buffer*     basis = nullptr;   ///< FlattenFit::basis()
    const gpu::Buffer*     floor = nullptr;   ///< FlattenFit::floor()
    std::vector<FlatCloud> clouds;            ///< filled by the frame
    double                 milliseconds = 0.0;
};

/// The fit's matrix and the roughness floor, on the device.
class FlattenFit {
public:
    [[nodiscard]] static Result<FlattenFit> create(gpu::ShaderLibrary& library);
    /// Solves M for `settings` and finds the floor (once a setting).
    [[nodiscard]] Result<void> prepare(const FlattenSettings& settings);
    [[nodiscard]] const gpu::Buffer& basis() const noexcept { return basis_; }
    [[nodiscard]] const gpu::Buffer& floor() const noexcept { return floor_; }
    /// The floor found, as a roughness (read back to be said).
    [[nodiscard]] Result<float> floorRoughness();

private:
    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  gram_, solve_, floorKernel_;
    gpu::Buffer         gramBuffer_, basis_, floor_;
    uint32_t            directions_ = 0;
    float               lambda_ = -1.0F, backWeight_ = -1.0F, floorShare_ = -1.0F;
};

/// What packing loses, counted on the device over every cloud of an output.
struct FlatStats {
    std::array<float, 3> min{};
    std::array<float, 3> max{};
    uint32_t sizesFloored = 0;      ///< axes under the floor of PLY and glTF
    uint32_t spzSizesFloored = 0;   ///< axes under SPZ's
    uint32_t spzSaturated = 0;      ///< gaussians whose harmonics SPZ's [-1, 1] scaled down
    uint32_t shrunk = 0;            ///< gaussians whose higher bands positivity scaled down
    uint32_t fractionalBits = 12;   ///< SPZ's fixed point for these positions
};

/// The six streams of an SPZ body, bytes of every cloud in order.
struct SpzStreams {
    std::array<std::vector<uint8_t>, 6> streams;   ///< positions, alphas, colours, scales, rotations, harmonics
};

/// Records into a format's numbers, on the device.
class FlattenPack {
public:
    [[nodiscard]] static Result<FlattenPack> create(gpu::ShaderLibrary& library);

    /// The box and what each format loses, over all `clouds`.
    [[nodiscard]] Result<FlatStats> stats(const std::vector<FlatCloud>& clouds, float logScaleFloor);
    /// 62 floats a gaussian of one cloud (flatten_pack.slang's flattenPackPly).
    [[nodiscard]] Result<std::vector<float>> ply(const FlatCloud& cloud, float logScaleFloor);
    /// The SPZ streams of all clouds, at `fractionalBits`, harmonics kept to
    /// `sh1Bits` and `shRestBits`.
    [[nodiscard]] Result<SpzStreams> spz(const std::vector<FlatCloud>& clouds, uint32_t fractionalBits,
                                         uint32_t sh1Bits, uint32_t shRestBits);
    /// One cloud's glTF blocks: position (3), rotation x y z w (4), linear
    /// scale (3), opacity (1), then sixteen harmonics (3 each), each block
    /// `count` long.
    [[nodiscard]] Result<std::vector<float>> glb(const FlatCloud& cloud, float logScaleFloor);

private:
    gpu::ShaderLibrary* library_ = nullptr;
    gpu::ComputeKernel  bounds_, resolve_, ply_, spz_, glb_;
};

}   // namespace athenea::render
