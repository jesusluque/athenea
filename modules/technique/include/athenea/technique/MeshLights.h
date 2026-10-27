// Copyright (c) 2026 jesus luque.
//
// MESH LIGHTS: the frame's emitting triangles -- a material's emission, or a
// mesh with UsdLux's MeshLightAPI -- as lights the raster and the path tracer
// sample through a tree, instead of only glowing where a ray happens to meet
// them.
//
// Built on the device into the mesh region at the front of the light words
// buffer (light::LightTable::reserveMesh); shaders/athenea/light/mesh_lights.slang
// says how the region reads and mesh_light_build.slang how it is made:
//
//   * a REBUILD decides which triangles emit and which cluster each is in. It
//     runs when the scene's meshes, its materials or which instances emit
//     change, and waits on the device a few times (the counts it sizes the
//     region by). Never inside a playing frame: a frame that only moves or
//     dims does not rebuild.
//   * a REFIT puts the same triangles where their instances stand, weighs
//     them by their groups' levels and settles the tree. No host wait: it is
//     recorded into the frame's batch.
//
// Membership is decided once a rebuild, so an emitter that deforms keeps its
// clusters and its light does not flicker.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu/Texture.h"
#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/light/LightTable.h"
#include "athenea/technique/MaterialPrograms.h"

namespace athenea::gpu {
class CommandBatch;
class ShaderLibrary;
class Device;
}

namespace athenea::technique {

/// A MeshLightAPI prim's controls, as an instance record's emitter set
/// (bits 24 to 30 of its flags) names them, in the units USD authored --
/// colour times intensity (as LightTable::recordOf multiplies them), the
/// exposure and colour temperature the shader derives -- with the sync mode
/// and whether both faces emit.
struct MeshLightSet {
    std::array<float, 3> colour{1.0F, 1.0F, 1.0F};   ///< colour times intensity
    float                exposure = 0.0F;
    float                temperature = 6500.0F;
    bool                 enableTemperature = false;
    uint32_t             mode = 0;      ///< 0 materialGlowTintsLight, 1 independent, 2 noMaterialResponse
    bool                 twoSided = false;

    bool operator==(const MeshLightSet&) const = default;
};

/// InstanceRecord::flags bits the mesh lights read (world::MeshInstance::lightFlags).
inline constexpr uint32_t kInstanceMeshLight = 4u;          ///< a MeshLightAPI prim
inline constexpr uint32_t kInstanceEmits = 0x80000000u;     ///< its material, a subset's or MeshLightAPI may emit
inline constexpr uint32_t kMaxMeshLightSets = 128;         ///< emitter sets, 0 being none
inline constexpr uint32_t kMaxEmissionGroups = 32;         ///< emission groups, 0 being none

/// What a frame's mesh lights are made from.
struct MeshLightInputs {
    /// The emitter sets, index 0 unused (an instance in no set).
    std::vector<MeshLightSet>                 sets;
    /// Each emission group's weight, linear rgb; group 0 is always one.
    std::array<std::array<float, 3>, kMaxEmissionGroups> weights{};
    /// The light group output (`lightGroup:NAME`) each emission group's light
    /// goes into: 1 + its index among the frame's, 0 none.
    std::array<uint32_t, kMaxEmissionGroups> aovGroups{};
    /// Whatever, besides the meshes and materials, says which instances emit
    /// (their emitter sets and groups): a different key rebuilds.
    uint64_t                                  emitterKey = 0;
    /// athenea:meshLights: off, nothing is clustered and the path tracer
    /// takes its emitting triangles from EmissiveTable's flat distribution;
    /// a MeshLightAPI prim still shows and casts as its controls say.
    bool                                      on = true;
    /// The least emission luminance a triangle must give off to be a light.
    float                                     threshold = 1.0e-4F;
};

class MeshLights {
public:
    [[nodiscard]] static Result<MeshLights> create(gpu::ShaderLibrary& library);

    /// BEFORE THE LIGHT TABLE IS SET: decides whether this frame rebuilds,
    /// and if it does probes the candidates and counts what emits -- the
    /// region's size, which `wordsNeeded` hands to LightTable::reserveMesh.
    [[nodiscard]] Result<void> plan(const MaterialFrame& frame, const render::Projection& projection,
                                    const MeshLightInputs& inputs);
    [[nodiscard]] uint32_t wordsNeeded() const noexcept { return wordsNeeded_; }

    /// AFTER IT IS SET: the region written, rebuilt or refitted. A rebuild
    /// submits and waits; a refit is recorded into `batch` and waits for
    /// nothing.
    [[nodiscard]] Result<void> write(gpu::CommandBatch& batch, const MaterialFrame& frame,
                                     const light::LightTable& table, const MeshLightInputs& inputs);

    /// THE MESH LIGHTS AS RELIT SPLATS READ THEM (light/mesh_light_splat.slang):
    /// the clusters' rectangles and a cut for each cell of a grid of
    /// `kSplatCells` a side over the box `lo` to `hi` -- the frame's clouds --
    /// into two textures, recorded into `batch` with no wait. Made again only
    /// when the clusters or the box changed.
    static constexpr uint32_t kSplatCells = 32;
    static constexpr uint32_t kSplatCutK = 4;
    [[nodiscard]] Result<void> writeSplats(gpu::CommandBatch& batch, const light::LightTable& table,
                                           const std::array<float, 3>& lo, const std::array<float, 3>& hi);
    [[nodiscard]] rhi::ITextureView* splatRects() const noexcept { return splatRectsView_.get(); }
    [[nodiscard]] rhi::ITextureView* splatCells() const noexcept { return splatCellsView_.get(); }

    /// How many clusters, and emitting triangles, the last rebuild made.
    [[nodiscard]] uint32_t clusters() const noexcept { return clusters_; }
    [[nodiscard]] uint32_t triangles() const noexcept { return triangles_; }
    /// How many rebuilds and refits there have been: what a test holds a
    /// moving or dimmed emitter to.
    [[nodiscard]] uint64_t rebuilds() const noexcept { return rebuilds_; }
    [[nodiscard]] uint64_t refits() const noexcept { return refits_; }
    /// How many times the header alone was uploaded again (a level, a set's controls).
    [[nodiscard]] uint64_t headerWrites() const noexcept { return headerWrites_; }
    /// Changes whenever what the clusters hold may have (a rebuild, a refit).
    [[nodiscard]] uint64_t revision() const noexcept { return rebuilds_ * 1000003ULL + refits_; }

private:
    struct Key {
        uint64_t      generation = ~uint64_t{0};
        const void*   records = nullptr;
        uint32_t      instances = ~0u;
        uint64_t      emitterKey = 0;
        std::string   module;
        bool          on = false;
        float         threshold = 0.0F;
        bool operator==(const Key&) const = default;
    };
    struct RefitKey {
        uint64_t                 positions = ~uint64_t{0};
        uint64_t                 instances = ~uint64_t{0};
        float                    time = -1.0F;
        std::array<std::array<float, 3>, kMaxEmissionGroups> weights{};
        std::array<uint32_t, kMaxEmissionGroups> aovGroups{};
        std::vector<MeshLightSet> sets;
        bool operator==(const RefitKey&) const = default;
    };
    [[nodiscard]] Result<void> rebuild(const MaterialFrame& frame, const light::LightTable& table,
                                       const MeshLightInputs& inputs);
    [[nodiscard]] Result<void> refit(gpu::CommandBatch& batch, const MaterialFrame& frame,
                                     const light::LightTable& table);
    [[nodiscard]] std::vector<uint32_t> headerWords(const MeshLightInputs& inputs) const;
    [[nodiscard]] Result<void> writeHeader(const light::LightTable& table, const MeshLightInputs& inputs);
    [[nodiscard]] Result<gpu::Buffer*> scratch(gpu::Buffer& into, uint64_t count, uint32_t element, const char* label);

    gpu::ShaderLibrary* library_ = nullptr;
    gpu::Device*        device_ = nullptr;
    std::optional<gpu::ComputeKernel> candidates_, compact_, centroids_, keys_, heads_, place_, counts_;
    std::optional<gpu::ComputeKernel> triangleRefit_, clusterRefit_, finish_, treeLeaves_, pack_, depth_;
    std::optional<gpu::ComputeKernel> hierarchy_, parents_, settle_, boundsChunks_, boundsReduce_;
    std::optional<gpu::ComputeKernel> probe_;
    std::string                       probeModule_;
    std::optional<gpu::PrefixSum>     prefix_;
    std::optional<gpu::RadixSort>     sort_;
    gpu::Buffer candidateCounts_, candidateBases_, probed_, probedFlags_, probedOffsets_, compactRadiance_;
    gpu::Buffer centroidBuffer_, extents_, boundsResult_, headBuffer_, headOffsets_, leafBoxes_, children_, leaves_;
    gpu::Buffer changed_, depthCount_, total_, probeHeader_;
    std::optional<gpu::ComputeKernel> splatRects_, splatCells_;
    gpu::Texture                      splatRectsTexture_, splatCellsTexture_;
    rhi::ComPtr<rhi::ITextureView>    splatRectsView_, splatCellsView_;
    uint64_t                          splatRevision_ = ~uint64_t{0};
    std::array<float, 6>              splatBox_{};
    gpu::SortBuffers sorting_;
    Key         key_;
    RefitKey    refitKey_;
    uint64_t    regionGeneration_ = 0;
    bool        pendingBuild_ = true;
    uint32_t    records_ = 0;
    uint32_t    candidates_count_ = 0;
    uint32_t    triangles_ = 0;
    uint32_t    clusters_ = 0;
    uint32_t    depth_count_ = 0;
    uint32_t    wordsNeeded_ = 0;
    uint64_t    rebuilds_ = 0;
    uint64_t    refits_ = 0;
    uint64_t    headerWrites_ = 0;
    // The region's layout, in words, as the last rebuild set it out.
    uint32_t    recordBase_ = 0, qualifiedBase_ = 0, slotBase_ = 0, triangleBase_ = 0, clusterBase_ = 0, nodeBase_ = 0;
};

}   // namespace athenea::technique
