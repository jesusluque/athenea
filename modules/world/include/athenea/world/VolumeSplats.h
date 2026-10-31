// Copyright (c) 2026 jesus luque.
//
// VOLUMETRIC GAUSSIANS AND THE GRIDS THEY COME FROM AND GO BACK TO (proposal
// 103). The fit itself is the Vdb2Splat effect (plugins/vdb2splat); these are
// the engine's sides of it, every one a kernel (shaders/athenea/volume/
// vdb_splats.slang): the grid's leaves laid out as the picture the effect
// reads, the way back onto a given topology (§3.4) that the M9 path tracer
// then renders unchanged, a fixture's values, two grids measured against each
// other, optical depth along random rays (closed form against ratio
// tracking), and the gaussians as a ParticleField's records in the world.
//
// The gaussians are the effect's output, four float4s each, in the grid's
// index space: (centre, sigma), (A00, A10, A11, A20), (A21, A22, temperature,
// Cd rgba8), (half extents of the box, alive) -- A the lower Cholesky factor
// of the precision, sigma the peak of the extinction in density units.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Vdb.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
}

namespace athenea::world {

/// Grids side by side in one buffer of words, as the kernels read them:
/// where each starts (32-byte aligned), `kNoGrid` where one is absent.
struct PackedGrids {
    static constexpr uint32_t kNoGrid = 0xFFFFFFFFU;
    std::vector<uint32_t> words;
    uint32_t              density = kNoGrid;
    uint32_t              temperature = kNoGrid;
    uint32_t              colour = kNoGrid;
    uint32_t              leafCount = 0;   ///< the density grid's

    /// The density grid, and the others where given (null: absent).
    [[nodiscard]] static PackedGrids of(const io::NanoGrid& density, const io::NanoGrid* temperature,
                                        const io::NanoGrid* colour);
    /// Grid `word`'s words back out, `like` giving its size and its header.
    [[nodiscard]] io::NanoGrid unpack(uint32_t word, const io::NanoGrid& like) const;
};

/// Two grids of one topology: the judged against the reference.
struct GridError {
    double sumSquares = 0.0;   ///< sum (a - b)^2 over the active voxels
    double sumReference = 0.0; ///< sum b^2
    double largest = 0.0;      ///< max |a - b|
    double relativeL2 = 0.0;   ///< sqrt(sumSquares / sumReference)
};

/// The mixture's closed-form transmittance against ratio tracking.
struct RayError {
    uint32_t rays = 0;
    double   meanDifference = 0.0;   ///< mean |T_mix - T_ratio|
    uint32_t beyond = 0;             ///< rays further apart than 5 standard errors + the tolerance
    double   largestDepth = 0.0;     ///< the largest closed-form optical depth among them
};

/// A sequence's slots, frame to frame.
struct SlotStability {
    uint32_t aliveBoth = 0;      ///< alive the frame before and now
    uint32_t aliveBefore = 0;    ///< alive the frame before
    double   meanMove = 0.0;     ///< mean |centre now - centre before| over the mean deviation, where alive in both
};

class VolumeSplats {
public:
    [[nodiscard]] static Result<VolumeSplats> create(gpu::ShaderLibrary& library);

    static constexpr uint32_t kLeafEntries = 513;

    /// The density grid's leaves into `leaves` (513 float4 a leaf), the
    /// temperature and Cd read at its voxels.
    void packLeaves(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                    const gpu::Buffer& leaves) const;

    /// The way back: `count` gaussians evaluated at the active voxels of the
    /// density grid at `densityWord` of `words`, written in place; the
    /// temperature where `temperatureWord` names a grid. Submitted in slices.
    [[nodiscard]] Result<void> splatToGrid(const gpu::Buffer& gaussians, uint32_t count, const gpu::Buffer& words,
                                           uint32_t densityWord, uint32_t leafCount,
                                           uint32_t temperatureWord = PackedGrids::kNoGrid) const;

    /// A fixture's values on the density grid's topology: `kind` 1 a smooth
    /// noise blob of `radius` voxels about `centre` (index space), 0 none;
    /// the temperature, hot at the centre, where a temperature grid is named.
    void fixture(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids, uint32_t kind,
                 std::array<float, 3> centre, float radius, uint32_t seed) const;

    /// The velocity grid (`grids.colour`, a vec3 grid) at the density's
    /// voxels, as the effect's Velocity picture: 513 entries a leaf, (vel x
    /// `scale`, active) -- `scale` turning the grid's units into voxels a frame.
    void packVelocity(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids, float scale,
                      const gpu::Buffer& out) const;

    /// A sequence's frame `frame` of the noise blob carried by a known flow:
    /// a turn about the vertical through `centre` at `omega` radians a frame
    /// and a rise of `rise` voxels a frame. The density on the density grid,
    /// and the flow's velocity (world units a frame, `voxelSize` a voxel) on
    /// the vec3 grid `grids.colour`.
    void sequenceFixture(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                         std::array<float, 3> centre, float radius, uint32_t seed, float omega, float rise,
                         float frame, float voxelSize) const;

    /// Slot for slot, `before` against `now` (records of `count` slots).
    [[nodiscard]] Result<SlotStability> slots(const gpu::Buffer& before, const gpu::Buffer& now, uint32_t count) const;

    /// Flicker: the frame-to-frame change of `judged` against `reference`'s,
    /// sqrt(sum ((b1 - b0) - (o1 - o0))^2 / sum (o1 - o0)^2) over the active
    /// voxels, all four grids of one topology.
    [[nodiscard]] Result<double> flicker(const gpu::Buffer& judged, const gpu::Buffer& reference,
                                         const gpu::Buffer& judgedBefore, const gpu::Buffer& referenceBefore,
                                         uint32_t densityWord, uint32_t leafCount) const;

    /// The temperature grid at `temperatureWord` of `judged` against
    /// `reference`'s, each voxel weighed by the reference's density at the
    /// density grid's active voxels: sqrt(sum rho (a - b)^2 / sum rho b^2).
    [[nodiscard]] Result<GridError> compareWeighted(const gpu::Buffer& judged, const gpu::Buffer& reference,
                                                    uint32_t densityWord, uint32_t leafCount,
                                                    uint32_t temperatureWord) const;

    /// A known mixture of `count` random gaussians in [margin, extent -
    /// margin) of index space, deviations in [minStd, maxStd] voxels, into
    /// `out` (records, 4 float4 a gaussian).
    void randomMixture(gpu::CommandBatch& batch, const gpu::Buffer& out, uint32_t count, std::array<float, 3> extent,
                       float margin, float minStd, float maxStd, uint32_t seed) const;

    /// `judged` against `reference`, the density grid at `densityWord` in both.
    [[nodiscard]] Result<GridError> compare(const gpu::Buffer& judged, const gpu::Buffer& reference,
                                            uint32_t densityWord, uint32_t leafCount) const;

    /// `rays` random segments through volume 0 of a VolumeSet's words: the
    /// gaussians' transmittance in closed form against `samples` ratio
    /// tracking estimates a ray. `indexToWorld` is the grid's (as VolumeSet
    /// composes it, row-vector 4x4 under the prim's identity).
    [[nodiscard]] Result<RayError> rays(const gpu::Buffer& volumeWords, const gpu::Buffer& gaussians, uint32_t count,
                                        const std::array<double, 16>& indexToWorld, uint32_t rays, uint32_t samples,
                                        uint32_t seed, float tolerance) const;

    /// The gaussians in the world: 14 floats a record (position, linear
    /// opacity, three deviations, rotation w x y z, linear Cd), then 5 a
    /// gaussian as halves, one to a word (sigma, Cd, temperature): Export's
    /// `volumeHalves`. Into `out`, 19 words a gaussian.
    void exportRecords(gpu::CommandBatch& batch, const gpu::Buffer& gaussians, uint32_t count,
                       const std::array<double, 16>& indexToWorld, float densityScale, const gpu::Buffer& out) const;

private:
    gpu::Device*       device_ = nullptr;
    gpu::ComputeKernel pack_, back_, fixture_, mixture_, compare_, reduce_, rays_, export_, slots_, flicker_;
};

}   // namespace athenea::world
