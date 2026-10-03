// Copyright (c) 2026 jesus luque.
//
// A rig's joints and weights, on the device in the layout the skinner reads.
// The file's two arrays (UsdSkel's `jointIndices` and `jointWeights`) go up
// as they are, bytes copied; pairing them, packing them and keeping only the
// gaussians validation kept is one kernel (`scene/splat_influences.slang`).
#pragma once

#include <cstdint>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"

namespace athenea::gpu {
class Device;
class ShaderLibrary;
}

namespace athenea::scene {

struct GpuSplats;

/// How a gaussian's influences are laid out (`common/influences.slang`).
enum class InfluenceLayout : uint32_t {
    /// Two words an influence, a float2 `(joint, weight)`.
    Pairs = 0,
    /// One word an influence: joint u16 (low) | weight unorm16 (high). Half
    /// the bytes, and what a skeleton of up to `kPackedJoints` joints keeps.
    Packed = 1,
};

/// The most joints the packed layout can name.
inline constexpr uint32_t kPackedJoints = 65536;

/// Bytes an influence takes in `layout`.
[[nodiscard]] constexpr uint32_t influenceBytes(InfluenceLayout layout) noexcept {
    return layout == InfluenceLayout::Packed ? 4u : 8u;
}

/// The file's arrays, already on the device.
struct InfluenceSource {
    const gpu::Buffer* jointIndices = nullptr;   ///< int32, `held` of them
    const gpu::Buffer* jointWeights = nullptr;   ///< float, `held` of them
    /// Entries in each array: the file's records times `perSplat`, or
    /// `perSplat` alone where the binding is constant (one set for all).
    uint32_t held = 0;
    uint32_t perSplat = 4;
    /// The skeleton's joints: an index at or past it holds nothing, and is
    /// counted.
    uint32_t joints = 0;
};

struct PackedInfluences {
    gpu::Buffer     words;   ///< `perSplat` influences a kept gaussian
    InfluenceLayout layout = InfluenceLayout::Packed;
    uint32_t        perSplat = 0;
    /// Influences naming a joint the skeleton has not (held as weight 0).
    uint32_t        jointsOutOfRange = 0;
    /// Weights outside [0, 1] or not a number (packed: held clamped).
    uint32_t        weightsOutOfRange = 0;
};

class InfluencePacker {
public:
    [[nodiscard]] static Result<InfluencePacker> create(gpu::ShaderLibrary& library);

    /// The layout a skeleton of `joints` joints is kept in: packed unless its
    /// indices do not fit in 16 bits.
    [[nodiscard]] static constexpr InfluenceLayout layoutFor(uint32_t joints) noexcept {
        return joints <= kPackedJoints ? InfluenceLayout::Packed : InfluenceLayout::Pairs;
    }

    /// Pairs and packs `source` to the gaussians of `splats` (through
    /// `GpuSplats::origin`, so the records validation dropped are left out).
    /// Waits for the device, and reads back the two counters: an upload is
    /// not a frame.
    [[nodiscard]] Result<PackedInfluences> pack(const GpuSplats& splats, const InfluenceSource& source,
                                                InfluenceLayout layout, const char* label);

    /// The same, from the file's arrays in host memory: they are copied up as
    /// they are, without anything on the host touching a value.
    [[nodiscard]] Result<PackedInfluences> pack(const GpuSplats& splats, const int32_t* jointIndices,
                                                const float* jointWeights, uint32_t held, uint32_t perSplat,
                                                uint32_t joints, InfluenceLayout layout, const char* label);

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel kernel_;
};

}   // namespace athenea::scene
