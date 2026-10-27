// Copyright (c) 2026 jesus luque.
#include "athenea/technique/MeshLights.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <vector>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/world/GpuScene.h"

namespace athenea::technique {

namespace {

// shaders/athenea/light/mesh_lights.slang's header.
constexpr uint32_t kClusterCount = 0, kTriangleCount = 1, kTreeNodes = 2, kTotalPower = 3, kInstanceCount = 4,
                   kNodeBase = 5, kClusterBase = 6, kTriangleBase = 7, kRecordBase = 8, kQualifiedBase = 9,
                   kSlotBase = 10, kOn = 12, kDepth = 13, kGroupBase = 16, kSetBase = 144, kHeaderWords = 2192;
constexpr uint32_t kTriangleWords = 16, kClusterWords = 32, kNodeWords = 16;

/// Each candidate triangle's radiance at its centre, in the material's own
/// terms and the MeshLightAPI's (without its group's weight, which a refit
/// applies): one material call site, the triangle's whole texture footprint
/// so a textured screen is read at the mip that averages the triangle.
const char* kProbe = R"(
import athenea.light.lights;
import athenea.light.mesh_lights;

StructuredBuffer<float>    meshRegion;
uniform uint               meshRegionWords;
StructuredBuffer<uint>     candidateBases;
RWStructuredBuffer<float4> probed;
RWStructuredBuffer<uint>   probedFlags;
uniform uint               records;
uniform uint               candidates;
uniform float              threshold;

[shader("compute")]
[numthreads(64, 1, 1)]
void meshProbe(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= candidates) {
        return;
    }
    uint lo = 0u;
    uint hi = records;
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if (candidateBases[mid] <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    probed[g] = float4(0.0);
    probedFlags[g] = 0u;
    Surface s;
    s.instanceIndex = lo;
    s.instance = instances[lo];
    s.mesh = meshes[s.instance.mesh];
    s.triangle = g - candidateBases[lo];
    const uint t = s.mesh.firstTriangle + s.triangle;
    if ((triangleSubsets[t] & 0x80000000u) != 0u) {
        return;   // a face marked invisible gives off nothing
    }
    const MaterialRecord m = materials[materialRowOf(s)];
    const uint flags = s.instance.flags;
    if ((m.flags & kMaterialEmits) == 0u && meshInstanceSet(flags) == 0u) {
        return;
    }
    s.points = uint3(indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
    s.weights = float3(1.0 / 3.0);
    const InstanceRecord r = s.instance;
    const uint base = s.mesh.firstPoint + r.pointsOffset;
    const float3 oa = positions[base + s.points.x].xyz;
    const float3 ob = positions[base + s.points.y].xyz;
    const float3 oc = positions[base + s.points.z].xyz;
    const float3 wa = rowsApply(r.world0, r.world1, r.world2, oa, 1.0);
    const float3 wb = rowsApply(r.world0, r.world1, r.world2, ob, 1.0);
    const float3 wc = rowsApply(r.world0, r.world1, r.world2, oc, 1.0);
    const float3 crossed = cross(wb - wa, wc - wa);
    if (!(dot(crossed, crossed) > 0.0)) {
        return;
    }
    MaterialInputs inputs;
    inputs.positionObject = (oa + ob + oc) / 3.0;
    inputs.positionWorld = (wa + wb + wc) / 3.0;
    inputs.normalWorld = normalize(crossed);
    inputs.normalObject = normalize(cross(ob - oa, oc - oa));
    inputs.viewPosition = inputs.positionWorld + inputs.normalWorld;
    inputs.inside = false;
    const uint stRecord = primvarRecordOf(s.mesh, kSlotSt);
    const float2 stA = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(1.0, 0.0, 0.0), float4(0.0)).xy;
    const float2 stB = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(0.0, 1.0, 0.0), float4(0.0)).xy;
    const float2 stC = primvarAt(stRecord, s.mesh, s.triangle, s.points, float3(0.0, 0.0, 1.0), float4(0.0)).xy;
    inputs.uvDx = stB - stA;
    inputs.uvDy = stC - stA;
    float3 tangent = float3(0.0);
    const float2 d1 = stB - stA;
    const float2 d2 = stC - stA;
    const float det = d1.x * d2.y - d1.y * d2.x;
    if (stRecord != kNoPrimvar && abs(det) > 1e-12) {
        tangent = ((wb - wa) * d2.y - (wc - wa) * d1.y) / det;
    }
    tangent -= dot(tangent, inputs.normalWorld) * inputs.normalWorld;
    if (dot(tangent, tangent) < 1e-20) {
        float3 x, y;
        lightFrame(inputs.normalWorld, x, y);
        tangent = x;
    }
    inputs.tangentWorld = normalize(tangent);
    inputs.bitangentWorld = cross(inputs.normalWorld, inputs.tangentWorld);
    inputs.tangentObject = inputs.tangentWorld;
    inputs.bitangentObject = inputs.bitangentWorld;
    inputs.frame = 0.0;
    inputs.time = lookup.time;
    inputs.mesh = s.mesh;
    inputs.triangle = s.triangle;
    inputs.points = s.points;
    inputs.weights = s.weights;
    inputs.displayColor = r.colour;
    inputs.worldFromObject0 = r.world0;
    inputs.worldFromObject1 = r.world1;
    inputs.worldFromObject2 = r.world2;
    evaluateMaterial(m.function, inputs, m.blob);
    // Its group's weight is the refit's, so the probe keeps the radiance whole.
    const float3 e = meshLightEmission(meshRegion, meshRegionWords, flags & ~(31u << 3), false,
                                       gAtheneaResult.emission);
    probed[g] = float4(max(e, float3(0.0)), 0.0);
    probedFlags[g] = dot(e, float3(0.2126, 0.7152, 0.0722)) > threshold ? 1u : 0u;
}
)";

void buildParams(rhi::ShaderCursor cursor, uint32_t records, uint32_t candidates, uint32_t triangles,
                 uint32_t clusters, uint32_t nodeBase) {
    rhi::ShaderCursor p = cursor["build"];
    p["records"].setData(records);
    p["candidates"].setData(candidates);
    p["triangles"].setData(triangles);
    p["clusters"].setData(clusters);
    p["nodeBase"].setData(nodeBase);
}

void bvhParams(rhi::ShaderCursor p, uint32_t count) {
    const float zero = 0.0F;
    p["count"].setData(count);
    p["nodeBase"].setData(uint32_t{0});
    p["leafBase"].setData(uint32_t{0});
    for (const char* name : {"boundsLoX", "boundsLoY", "boundsLoZ", "boundsHiX", "boundsHiY", "boundsHiZ"}) {
        p[name].setData(zero);
    }
}

}   // namespace

Result<MeshLights> MeshLights::create(gpu::ShaderLibrary& library) {
    MeshLights lights;
    lights.library_ = &library;
    lights.device_ = &library.device();
    const auto make = [&](std::optional<gpu::ComputeKernel>& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into.emplace(std::move(*kernel));
        return ok();
    };
    const char* build = "athenea/light/mesh_light_build";
    ATHENEA_TRY(make(lights.candidates_, build, "meshCandidates"));
    ATHENEA_TRY(make(lights.compact_, build, "meshCompact"));
    ATHENEA_TRY(make(lights.centroids_, build, "meshCentroids"));
    ATHENEA_TRY(make(lights.keys_, build, "meshKeys"));
    ATHENEA_TRY(make(lights.heads_, build, "meshHeads"));
    ATHENEA_TRY(make(lights.place_, build, "meshPlace"));
    ATHENEA_TRY(make(lights.counts_, build, "meshCounts"));
    ATHENEA_TRY(make(lights.triangleRefit_, build, "meshTriangleRefit"));
    ATHENEA_TRY(make(lights.clusterRefit_, build, "meshClusterRefit"));
    ATHENEA_TRY(make(lights.finish_, build, "meshFinish"));
    ATHENEA_TRY(make(lights.treeLeaves_, build, "meshTreeLeaves"));
    ATHENEA_TRY(make(lights.pack_, build, "meshPack"));
    ATHENEA_TRY(make(lights.depth_, build, "meshTreeDepth"));
    ATHENEA_TRY(make(lights.hierarchy_, "athenea/rt/bvh_hierarchy", "bvhHierarchy"));
    ATHENEA_TRY(make(lights.parents_, "athenea/light/light_bvh_build", "lightParents"));
    ATHENEA_TRY(make(lights.settle_, "athenea/light/light_bvh_build", "lightSettle"));
    ATHENEA_TRY(make(lights.boundsChunks_, "athenea/scene/bounds_chunks", "boundsChunks"));
    ATHENEA_TRY(make(lights.boundsReduce_, "athenea/scene/bounds_reduce", "boundsReduce"));
    ATHENEA_TRY(make(lights.splatRects_, "athenea/light/mesh_light_splat", "meshSplatRects"));
    ATHENEA_TRY(make(lights.splatCells_, "athenea/light/mesh_light_splat", "meshSplatCells"));
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    lights.prefix_.emplace(std::move(*prefix));
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    lights.sort_.emplace(std::move(*sort));
    return lights;
}

Result<gpu::Buffer*> MeshLights::scratch(gpu::Buffer& into, uint64_t count, uint32_t element, const char* label) {
    const uint64_t bytes = std::max<uint64_t>(count, 1) * element;
    if (!into.valid() || into.bytes() < bytes || into.elementBytes() != element) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(bytes, into.valid() ? into.bytes() : 0);
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        into = std::move(*made);
    }
    return &into;
}

Result<void> MeshLights::plan(const MaterialFrame& frame, const render::Projection& projection,
                              const MeshLightInputs& inputs) {
    if (!frame.valid()) {
        return ok();
    }
    const world::GpuScene& scene = *frame.scene;
    Key key;
    key.generation = scene.generation();
    key.records = frame.records->rhi();
    key.instances = scene.instanceCount();
    key.emitterKey = inputs.emitterKey;
    key.module = frame.programs->module();
    key.on = inputs.on;
    key.threshold = inputs.threshold;
    if (!(key == key_)) {
        key_ = key;
        pendingBuild_ = true;
    }
    if (!pendingBuild_) {
        return ok();
    }
    records_ = scene.instanceCount();
    candidates_count_ = 0;
    triangles_ = 0;
    clusters_ = 0;
    wordsNeeded_ = kHeaderWords;
    if (!inputs.on || records_ == 0) {
        return ok();
    }
    // The probe imports the frame's materials: one generated module a set of them.
    if (probeModule_ != frame.programs->module() || !probe_.has_value()) {
        const std::string name = frame.programs->module() + "_meshlights";
        const std::string source = "import " + frame.programs->module() + ";\n" + kProbe;
        auto program = library_->loadSource(name, source, {"meshProbe"});
        if (!program) return std::move(program).error();
        auto probe = gpu::ComputeKernel::create(*library_, name, "meshProbe");
        if (!probe) return std::move(probe).error();
        probe_.emplace(std::move(*probe));
        probeModule_ = frame.programs->module();
    }
    // Which records may emit, and the triangles they bring: counted on the
    // device, the total read back to size what follows.
    ATHENEA_TRY(scratch(candidateCounts_, uint64_t{records_} + 1, 4, "meshLights.candidateCounts"));
    ATHENEA_TRY(scratch(candidateBases_, uint64_t{records_} + 1, 4, "meshLights.candidateBases"));
    ATHENEA_TRY(scratch(total_, 4, 4, "meshLights.total"));
    {
        gpu::CommandBatch batch(*device_);
        candidates_->dispatch(batch, {records_ + 1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bindScene(cursor, scene);
            cursor["candidateCounts"].setBinding(candidateCounts_.rhi());
            buildParams(cursor, records_, 0, 0, 0, 0);
        });
        ATHENEA_TRY(prefix_->apply(batch, candidateCounts_, candidateBases_, total_, records_ + 1));
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(total_.read(*device_, 0, 4, &candidates_count_));
    }
    if (candidates_count_ > 0) {
        const std::vector<uint32_t> header = headerWords(inputs);
        ATHENEA_TRY(scratch(probeHeader_, kHeaderWords, 4, "meshLights.probeHeader"));
        ATHENEA_TRY(probeHeader_.write(*device_, 0, kHeaderWords * 4, header.data()));
        ATHENEA_TRY(scratch(probed_, candidates_count_, 16, "meshLights.probed"));
        ATHENEA_TRY(scratch(probedFlags_, candidates_count_, 4, "meshLights.probedFlags"));
        ATHENEA_TRY(scratch(probedOffsets_, candidates_count_, 4, "meshLights.probedOffsets"));
        gpu::CommandBatch batch(*device_);
        probe_->dispatch(batch, {candidates_count_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            // The emitter sets' controls, as the region's header will hold
            // them: the region itself is not there before the first rebuild.
            cursor["meshRegion"].setBinding(probeHeader_.rhi());
            cursor["meshRegionWords"].setData(kHeaderWords);
            cursor["candidateBases"].setBinding(candidateBases_.rhi());
            cursor["probed"].setBinding(probed_.rhi());
            cursor["probedFlags"].setBinding(probedFlags_.rhi());
            cursor["records"].setData(records_);
            cursor["candidates"].setData(candidates_count_);
            cursor["threshold"].setData(inputs.threshold);
        });
        ATHENEA_TRY(prefix_->apply(batch, probedFlags_, probedOffsets_, total_, candidates_count_));
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(total_.read(*device_, 0, 4, &triangles_));
    }
    // The region: header, the records' bases, then per emitting triangle its
    // scene number, its slot, its slot's cluster, its record; and the
    // clusters and nodes, at most one cluster a triangle.
    const uint64_t t = triangles_;
    const uint64_t words = uint64_t{kHeaderWords} + records_ + 1 + 3 * t + kTriangleWords * t + kClusterWords * t +
                           kNodeWords * 2 * t;
    if (words > UINT32_MAX / 2) {
        return Error::make(ErrorCode::OutOfMemory, "mesh lights: {} emitting triangles is more than a region holds",
                           triangles_);
    }
    wordsNeeded_ = static_cast<uint32_t>(words);
    return ok();
}

std::vector<uint32_t> MeshLights::headerWords(const MeshLightInputs& inputs) const {
    std::vector<uint32_t> header(kHeaderWords, 0u);
    header[kClusterCount] = clusters_;
    header[kTriangleCount] = triangles_;
    header[kTreeNodes] = clusters_ > 0 ? 2 * clusters_ - 1 : 0;
    header[kInstanceCount] = triangles_ > 0 ? records_ : 0;
    header[kNodeBase] = nodeBase_;
    header[kClusterBase] = clusterBase_;
    header[kTriangleBase] = triangleBase_;
    header[kRecordBase] = recordBase_;
    header[kQualifiedBase] = qualifiedBase_;
    header[kSlotBase] = slotBase_;
    header[kOn] = inputs.on && clusters_ > 0 ? 1u : 0u;
    header[kDepth] = depth_count_;
    for (uint32_t g = 0; g < kMaxEmissionGroups; ++g) {
        for (uint32_t c = 0; c < 3; ++c) {
            header[kGroupBase + 4 * g + c] = std::bit_cast<uint32_t>(g == 0 ? 1.0F : inputs.weights[g][c]);
        }
        header[kGroupBase + 4 * g + 3] = g == 0 ? 0u : inputs.aovGroups[g];
    }
    for (uint32_t s = 1; s < std::min<size_t>(inputs.sets.size(), kMaxMeshLightSets); ++s) {
        const MeshLightSet& set = inputs.sets[s];
        for (uint32_t c = 0; c < 3; ++c) {
            header[kSetBase + 8 * s + c] = std::bit_cast<uint32_t>(set.colour[c]);
        }
        header[kSetBase + 8 * s + 3] = set.mode;
        header[kSetBase + 8 * s + 4] = (set.twoSided ? 1u : 0u) | (set.enableTemperature ? 4u : 0u);
        header[kSetBase + 8 * s + 5] = std::bit_cast<uint32_t>(set.exposure);
        header[kSetBase + 8 * s + 6] = std::bit_cast<uint32_t>(set.temperature);
    }
    return header;
}

Result<void> MeshLights::writeHeader(const light::LightTable& table, const MeshLightInputs& inputs) {
    const std::vector<uint32_t> header = headerWords(inputs);
    if (table.meshWords() < kHeaderWords) {
        return Error(ErrorCode::InternalError, "mesh lights: the light table holds no region");
    }
    // The total and the revision are the device's (meshFinish): kept as they are.
    gpu::Buffer region = table.words();
    ATHENEA_TRY(region.write(*device_, 0, kTotalPower * 4, header.data()));
    ATHENEA_TRY(region.write(*device_, kInstanceCount * 4, (11 - kInstanceCount) * 4, header.data() + kInstanceCount));
    ATHENEA_TRY(region.write(*device_, kOn * 4, (kHeaderWords - kOn) * 4, header.data() + kOn));
    return ok();
}

Result<void> MeshLights::write(gpu::CommandBatch& batch, const MaterialFrame& frame, const light::LightTable& table,
                               const MeshLightInputs& inputs) {
    if (!frame.valid() || table.meshWords() < kHeaderWords) {
        return ok();
    }
    if (table.meshGeneration() != regionGeneration_) {
        pendingBuild_ = true;   // the region was made anew: everything in it is gone
        regionGeneration_ = table.meshGeneration();
    }
    RefitKey refit;
    refit.positions = frame.scene->positionsRevision();
    refit.instances = frame.scene->instancesRevision();
    refit.time = frame.time;
    refit.weights = inputs.weights;
    refit.aovGroups = inputs.aovGroups;
    refit.sets = inputs.sets;
    if (pendingBuild_) {
        if (table.meshWords() < wordsNeeded_) {
            return Error(ErrorCode::InternalError, "mesh lights: the region is smaller than the plan asked");
        }
        ATHENEA_TRY(rebuild(frame, table, inputs));
        pendingBuild_ = false;
        refitKey_ = refit;
        return ok();
    }
    if (refit == refitKey_) {
        return ok();
    }
    // Positions, transforms, levels or the sets' controls changed: the header
    // takes the new weights and controls, and the same clusters are refitted.
    const bool header =
        refit.weights != refitKey_.weights || refit.sets != refitKey_.sets || refit.aovGroups != refitKey_.aovGroups;
    // Only the outputs moved: the header says so, and nothing is refitted.
    const bool outputsOnly = header && refit.weights == refitKey_.weights && refit.sets == refitKey_.sets &&
                             refit.positions == refitKey_.positions && refit.instances == refitKey_.instances &&
                             refit.time == refitKey_.time;
    refitKey_ = refit;
    if (header) {
        ATHENEA_TRY(writeHeader(table, inputs));
        ++headerWrites_;
    }
    if (clusters_ > 0 && !outputsOnly) {
        ATHENEA_TRY(this->refit(batch, frame, table));
        ++refits_;
    }
    return ok();
}

Result<void> MeshLights::rebuild(const MaterialFrame& frame, const light::LightTable& table,
                                 const MeshLightInputs& inputs) {
    const world::GpuScene& scene = *frame.scene;
    const uint32_t t = triangles_;
    recordBase_ = kHeaderWords;
    qualifiedBase_ = recordBase_ + records_ + 1;
    slotBase_ = qualifiedBase_ + t;
    triangleBase_ = slotBase_ + 2 * t;
    clusterBase_ = triangleBase_ + kTriangleWords * t;
    nodeBase_ = clusterBase_ + kClusterWords * t;
    clusters_ = 0;
    depth_count_ = 0;
    ++rebuilds_;
    ATHENEA_TRY(writeHeader(table, inputs));
    if (t == 0 || !inputs.on) {
        return ok();
    }
    gpu::Buffer region = table.words();
    ATHENEA_TRY(scratch(compactRadiance_, t, 16, "meshLights.compactRadiance"));
    ATHENEA_TRY(scratch(centroidBuffer_, t, 16, "meshLights.centroids"));
    ATHENEA_TRY(scratch(boundsResult_, 2, 16, "meshLights.bounds"));
    ATHENEA_TRY(scratch(headBuffer_, t, 4, "meshLights.heads"));
    ATHENEA_TRY(scratch(headOffsets_, t, 4, "meshLights.headOffsets"));
    ATHENEA_TRY(scratch(sorting_.keysLo, t, 4, "meshLights.keys"));
    ATHENEA_TRY(scratch(sorting_.values, t, 4, "meshLights.values"));
    ATHENEA_TRY(scratch(sorting_.scratchKeysLo, t, 4, "meshLights.keys2"));
    ATHENEA_TRY(scratch(sorting_.scratchValues, t, 4, "meshLights.values2"));
    constexpr uint32_t kChunk = 4096;
    const auto bounds = [&](gpu::CommandBatch& batch, uint32_t count) -> Result<void> {
        const uint32_t chunks = (count + kChunk - 1) / kChunk;
        ATHENEA_TRY(scratch(extents_, uint64_t{chunks} * 2, 16, "meshLights.extents"));
        const auto params = [&](rhi::ShaderCursor cursor) {
            cursor["params"]["count"].setData(count);
            cursor["params"]["chunkSize"].setData(kChunk);
            cursor["params"]["chunkCount"].setData(chunks);
        };
        boundsChunks_->dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(centroidBuffer_.rhi());
            cursor["extents"].setBinding(extents_.rhi());
            params(cursor);
        });
        boundsReduce_->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents_.rhi());
            cursor["result"].setBinding(boundsResult_.rhi());
            params(cursor);
        });
        return ok();
    };
    const auto bindBuild = [&](rhi::ShaderCursor cursor, uint32_t clusters) {
        bindScene(cursor, scene);
        cursor["words"].setBinding(region.rhi());
        cursor["candidateCounts"].setBinding(candidateCounts_.rhi());
        cursor["candidateBases"].setBinding(candidateBases_.rhi());
        cursor["probed"].setBinding(probed_.rhi());
        cursor["probedFlags"].setBinding(probedFlags_.rhi());
        cursor["probedOffsets"].setBinding(probedOffsets_.rhi());
        cursor["compactRadiance"].setBinding(compactRadiance_.rhi());
        cursor["centroids"].setBinding(centroidBuffer_.rhi());
        cursor["bounds"].setBinding(boundsResult_.rhi());
        cursor["keys"].setBinding(sorting_.keysLo.rhi());
        cursor["values"].setBinding(sorting_.values.rhi());
        cursor["sortedKeys"].setBinding(sorting_.keysLo.rhi());
        cursor["sortedValues"].setBinding(sorting_.values.rhi());
        cursor["heads"].setBinding(headBuffer_.rhi());
        cursor["headOffsets"].setBinding(headOffsets_.rhi());
        buildParams(cursor, records_, candidates_count_, t, clusters, nodeBase_);
    };
    // Which triangles emit, in scene order; their keys, sorted; where a key
    // changes a cluster begins.
    {
        gpu::CommandBatch batch(*device_);
        batch.encoder()->copyBuffer(region.rhi(), uint64_t{recordBase_} * 4, candidateBases_.rhi(), 0,
                                    (uint64_t{records_} + 1) * 4);
        batch.markDirty();
        compact_->dispatch(batch, {candidates_count_, 1, 1}, [&](rhi::ShaderCursor c) { bindBuild(c, 0); });
        centroids_->dispatch(batch, {t, 1, 1}, [&](rhi::ShaderCursor c) { bindBuild(c, 0); });
        ATHENEA_TRY(bounds(batch, t));
        keys_->dispatch(batch, {t, 1, 1}, [&](rhi::ShaderCursor c) { bindBuild(c, 0); });
        ATHENEA_TRY(sort_->sort(batch, sorting_, t, 21));
        heads_->dispatch(batch, {t, 1, 1}, [&](rhi::ShaderCursor c) { bindBuild(c, 0); });
        ATHENEA_TRY(prefix_->apply(batch, headBuffer_, headOffsets_, total_, t));
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(total_.read(*device_, 0, 4, &clusters_));
    }
    const uint32_t c = clusters_;
    const uint32_t internalCount = c > 1 ? c - 1 : 0;
    ATHENEA_TRY(scratch(leafBoxes_, uint64_t{c} * 2, 16, "meshLights.leafBoxes"));
    ATHENEA_TRY(scratch(children_, uint64_t{std::max(c, 2u) - 1} * 2, 4, "meshLights.children"));
    ATHENEA_TRY(scratch(leaves_, c, 4, "meshLights.leaves"));
    ATHENEA_TRY(scratch(changed_, std::max(c, 2u) - 1, 4, "meshLights.changed"));
    ATHENEA_TRY(scratch(depthCount_, 1, 4, "meshLights.depth"));
    // Each triangle at its slot, the clusters' counts, the clusters where they
    // stand (their centres key the tree), the tree's links, and its depth.
    {
        gpu::CommandBatch batch(*device_);
        place_->dispatch(batch, {t, 1, 1}, [&](rhi::ShaderCursor cur) { bindBuild(cur, c); });
        counts_->dispatch(batch, {c, 1, 1}, [&](rhi::ShaderCursor cur) { bindBuild(cur, c); });
        triangleRefit_->dispatch(batch, {t, 1, 1}, [&](rhi::ShaderCursor cur) { bindBuild(cur, c); });
        clusterRefit_->dispatch(batch, {c, 1, 1}, [&](rhi::ShaderCursor cur) { bindBuild(cur, c); });
        treeLeaves_->dispatch(batch, {c, 1, 1}, [&](rhi::ShaderCursor cur) { bindBuild(cur, c); });
        ATHENEA_TRY(sort_->sort(batch, sorting_, c, 30));
        hierarchy_->dispatch(batch, {c, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["sortedKeys"].setBinding(sorting_.keysLo.rhi());
            cursor["sortedValues"].setBinding(sorting_.values.rhi());
            cursor["children"].setBinding(children_.rhi());
            cursor["leaves"].setBinding(leaves_.rhi());
            bvhParams(cursor["params"], c);
        });
        pack_->dispatch(batch, {internalCount + c, 1, 1}, [&](rhi::ShaderCursor cur) {
            bindBuild(cur, c);
            cur["children"].setBinding(children_.rhi());
            cur["leaves"].setBinding(leaves_.rhi());
        });
        parents_->dispatch(batch, {std::max(c, 2u) - 1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["nodeValues"].setBinding(region.rhi());
            cursor["nodeBase"].setData(nodeBase_);
            bvhParams(cursor["params"], c);
        });
        const uint32_t zero = 0;
        ATHENEA_TRY(depthCount_.write(*device_, 0, 4, &zero));
        depth_->dispatch(batch, {c, 1, 1}, [&](rhi::ShaderCursor cur) {
            bindBuild(cur, c);
            cur["depth"].setBinding(depthCount_.rhi());
        });
        ATHENEA_TRY(batch.submit(true));
        ATHENEA_TRY(depthCount_.read(*device_, 0, 4, &depth_count_));
    }
    ATHENEA_TRY(writeHeader(table, inputs));
    {
        gpu::CommandBatch batch(*device_);
        ATHENEA_TRY(refit(batch, frame, table));
        ATHENEA_TRY(batch.submit(true));
    }
    log::debug("mesh lights: {} candidate triangles, {} emitting, {} clusters, tree depth {}", candidates_count_, t,
               c, depth_count_);
    return ok();
}

Result<void> MeshLights::writeSplats(gpu::CommandBatch& batch, const light::LightTable& table,
                                     const std::array<float, 3>& lo, const std::array<float, 3>& hi) {
    const std::array<float, 6> box{lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
    if (clusters_ == 0 || table.meshWords() < kHeaderWords) {
        return ok();
    }
    if (splatRevision_ == revision() && box == splatBox_ && splatRectsView_ && splatCellsView_) {
        return ok();
    }
    const auto texture = [&](gpu::Texture& into, rhi::ComPtr<rhi::ITextureView>& view, uint32_t w, uint32_t h,
                             rhi::Format format, const char* label) -> Result<void> {
        if (into.valid() && into.width() >= w && into.height() >= h && view) {
            return ok();
        }
        gpu::TextureDesc desc;
        desc.width = w;
        desc.height = h;
        desc.format = format;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = label;
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        auto v = into.view(0);
        if (!v) return std::move(v).error();
        view = std::move(*v);
        return ok();
    };
    // Four texels a cluster, 256 clusters a row; the cells x K across, the
    // other two axes down.
    ATHENEA_TRY(texture(splatRectsTexture_, splatRectsView_, 1024, (clusters_ + 255) / 256, rhi::Format::RGBA32Float,
                        "meshLights.splatRects"));
    ATHENEA_TRY(texture(splatCellsTexture_, splatCellsView_, kSplatCells * kSplatCutK, kSplatCells * kSplatCells,
                        rhi::Format::RG32Uint, "meshLights.splatCells"));
    const gpu::Buffer& region = table.words();
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["meshRegion"].setBinding(region.rhi());
        cursor["meshRegionWords"].setData(table.meshWords());
        cursor["rectsOut"].setBinding(splatRectsView_.get());
        cursor["cellsOut"].setBinding(splatCellsView_.get());
        cursor["cellSide"].setData(kSplatCells);
        cursor["cutK"].setData(kSplatCutK);
        const float l[4] = {lo[0], lo[1], lo[2], 0.0F};
        const float h[4] = {hi[0], hi[1], hi[2], 0.0F};
        cursor["gridLo"].setData(l, sizeof(l));
        cursor["gridHi"].setData(h, sizeof(h));
    };
    splatRects_->dispatch(batch, {clusters_, 1, 1}, bind);
    splatCells_->dispatch(batch, {kSplatCells, kSplatCells, kSplatCells}, bind);
    splatRevision_ = revision();
    splatBox_ = box;
    return ok();
}

Result<void> MeshLights::refit(gpu::CommandBatch& batch, const MaterialFrame& frame, const light::LightTable& table) {
    const world::GpuScene& scene = *frame.scene;
    const uint32_t t = triangles_;
    const uint32_t c = clusters_;
    if (t == 0 || c == 0) {
        return ok();
    }
    const gpu::Buffer& region = table.words();
    const auto bind = [&](rhi::ShaderCursor cursor) {
        bindScene(cursor, scene);
        cursor["words"].setBinding(region.rhi());
        buildParams(cursor, records_, candidates_count_, t, c, nodeBase_);
    };
    triangleRefit_->dispatch(batch, {t, 1, 1}, bind);
    clusterRefit_->dispatch(batch, {c, 1, 1}, bind);
    // The tree's boxes, cones and power, a level a pass: as many passes as it
    // is deep, counted once at the rebuild, so nothing waits here.
    if (c > 1) {
        for (uint32_t pass = 0; pass < depth_count_; ++pass) {
            settle_->dispatch(batch, {c - 1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["nodeValues"].setBinding(region.rhi());
                cursor["nodeBase"].setData(nodeBase_);
                cursor["changed"].setBinding(changed_.rhi());
                bvhParams(cursor["params"], c);
            });
        }
    }
    finish_->dispatch(batch, {1, 1, 1}, bind);
    return ok();
}

}   // namespace athenea::technique
