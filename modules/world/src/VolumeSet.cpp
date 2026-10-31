// Copyright (c) 2026 jesus luque.
#include "athenea/world/VolumeSet.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::world {

Result<VolumeSet> VolumeSet::create(gpu::ShaderLibrary& library) {
    VolumeSet set;
    set.device_ = &library.device();
    auto bounds = gpu::ComputeKernel::create(library, "athenea/volume/volume_prepare", "volumeBounds");
    if (!bounds) return std::move(bounds).error();
    auto leafMax = gpu::ComputeKernel::create(library, "athenea/volume/volume_prepare", "volumeLeafMax");
    if (!leafMax) return std::move(leafMax).error();
    auto instances = gpu::ComputeKernel::create(library, "athenea/volume/volume_prepare", "volumeInstances");
    if (!instances) return std::move(instances).error();
    auto finish = gpu::ComputeKernel::create(library, "athenea/volume/volume_prepare", "volumeFinish");
    if (!finish) return std::move(finish).error();
    set.boundsKernel_ = std::move(*bounds);
    set.leafMaxKernel_ = std::move(*leafMax);
    set.instancesKernel_ = std::move(*instances);
    set.finishKernel_ = std::move(*finish);
    return set;
}

namespace {

uint32_t bitsOf(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    return bits;
}

}   // namespace

Result<void> VolumeSet::set(gpu::CommandBatch& batch, std::span<const VolumeInput> volumes) {
    placements_.clear();
    // Each grid once, in the order the volumes first name it; a volume's
    // records follow one another, one an instance.
    std::vector<uint32_t> gridOf(volumes.size(), 0u);
    std::vector<const io::NanoGrid*> grids;
    std::vector<uint32_t> firstRecord(volumes.size(), 0u);
    uint32_t records = 0;
    for (size_t v = 0; v < volumes.size(); ++v) {
        const VolumeInput& in = volumes[v];
        if (in.grid == nullptr) {
            return Error(ErrorCode::InvalidArgument, "volumes: a volume with no grid");
        }
        if (in.instanceRows != nullptr && !in.instanceRows->valid()) {
            return Error(ErrorCode::InvalidArgument, "volumes: an instanced volume with no instance rows");
        }
        const auto at = std::find(grids.begin(), grids.end(), in.grid);
        gridOf[v] = static_cast<uint32_t>(at - grids.begin());
        if (at == grids.end()) {
            grids.push_back(in.grid);
        }
        firstRecord[v] = records;
        records += in.instanceRows != nullptr ? in.instanceCount : 1u;
    }
    records_ = records;
    std::vector<uint32_t> words(kHeaderWords + size_t{records} * kRecordWords, 0u);
    words[0] = records;
    words[1] = static_cast<uint32_t>(grids.size());
    // The grids, each 32-byte aligned (eight words), then the leaf maxima.
    for (const io::NanoGrid* grid : grids) {
        while (words.size() % 8 != 0) words.push_back(0u);
        VolumeGridPlacement placement;
        placement.gridWord = static_cast<uint32_t>(words.size());
        placement.leafCount = grid->leafCount;
        placements_.push_back(placement);
        words.insert(words.end(), grid->words.begin(), grid->words.end());
    }
    for (VolumeGridPlacement& placement : placements_) {
        placement.leafBase = static_cast<uint32_t>(words.size());
        words.resize(words.size() + std::max<size_t>(placement.leafCount, 1), 0u);
    }
    // The records: what the host knows. World to index is the inverse of
    // index to object (the grid's own map) then object to world (the
    // prim's). OpenVDB holds its map row-vector, translation in the last row;
    // Mat4 is column-vector, applied first rightmost -- so the map is
    // transposed as it is read, and composed on the right. Read untransposed
    // and composed on the left, a translated volume sat offset by its
    // translation in voxels rather than in world units, which a pure scale
    // (the first fixture) could not show. An instanced volume's records are
    // written alike; the instance kernel then composes each one's instance
    // under that transform, on the device.
    std::vector<std::array<float, 12>> prototypes(volumes.size());
    for (size_t v = 0; v < volumes.size(); ++v) {
        const VolumeInput& in = volumes[v];
        render::Mat4 indexToObject = render::Mat4::identity();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                indexToObject.at(r, c) = in.grid->indexToWorld[static_cast<size_t>(c * 4 + r)];
            }
        }
        const render::Mat4 worldToIndex = aofx::xform::inverseAffine(in.objectToWorld * indexToObject);
        prototypes[v] = worldToIndex.rows3x4();
        const VolumeGridPlacement& placement = placements_[gridOf[v]];
        const uint32_t count = in.instanceRows != nullptr ? in.instanceCount : 1u;
        for (uint32_t k = 0; k < count; ++k) {
            const size_t b = kHeaderWords + size_t{firstRecord[v] + k} * kRecordWords;
            words[b + 0] = placement.gridWord;
            words[b + 1] = placement.leafBase;
            words[b + 2] = placement.leafCount;
            words[b + 3] = 0;   // leaf 0's word: the finish kernel's
            for (size_t i = 0; i < 12; ++i) words[b + 4 + i] = bitsOf(prototypes[v][i]);
            words[b + 22] = bitsOf(in.densityScale);
            words[b + 23] = bitsOf(in.albedo[0]);
            words[b + 24] = bitsOf(in.albedo[1]);
            words[b + 25] = bitsOf(in.albedo[2]);
            words[b + 26] = bitsOf(in.g);
            words[b + 28] = bitsOf(in.emission[0]);
            words[b + 29] = bitsOf(in.emission[1]);
            words[b + 30] = bitsOf(in.emission[2]);
            words[b + 33] = static_cast<uint32_t>(in.categories & 0xFFFFFFFFu);
            words[b + 34] = static_cast<uint32_t>(in.categories >> 32);
        }
    }
    auto madeWords = gpu::Buffer::fromSpan<uint32_t>(*device_, words, "volumes.words");
    if (!madeWords) return std::move(madeWords).error();
    words_ = std::move(*madeWords);
    // Bounds start from the empty box, so the first leaf sets them.
    const size_t gridCount = std::max<size_t>(grids.size(), 1);
    std::vector<int32_t> bounds;
    for (size_t g = 0; g < gridCount; ++g) {
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::max());
        for (int k = 0; k < 3; ++k) bounds.push_back(std::numeric_limits<int32_t>::min());
    }
    auto madeBounds = gpu::Buffer::fromSpan<int32_t>(*device_, bounds, "volumes.bounds");
    if (!madeBounds) return std::move(madeBounds).error();
    bounds_ = std::move(*madeBounds);
    const std::vector<uint32_t> zeroBits(gridCount, 0u);
    auto madeMajorant = gpu::Buffer::fromSpan<uint32_t>(*device_, zeroBits, "volumes.majorant");
    if (!madeMajorant) return std::move(madeMajorant).error();
    majorant_ = std::move(*madeMajorant);
    const auto bindGrid = [&](rhi::ShaderCursor cursor, uint32_t g) {
        const VolumeGridPlacement& p = placements_[g];
        cursor["words"].setBinding(words_.rhi());
        cursor["wordsOut"].setBinding(words_.rhi());
        cursor["bounds"].setBinding(bounds_.rhi());
        cursor["majorant"].setBinding(majorant_.rhi());
        cursor["params"]["gridWord"].setData(p.gridWord);
        cursor["params"]["leafCount"].setData(p.leafCount);
        cursor["params"]["leafBase"].setData(p.leafBase);
        cursor["params"]["slot"].setData(g);
    };
    for (uint32_t g = 0; g < placements_.size(); ++g) {
        if (placements_[g].leafCount > 0) {
            const auto bind = [&](rhi::ShaderCursor cursor) { bindGrid(cursor, g); };
            boundsKernel_.dispatch(batch, {placements_[g].leafCount, 1, 1}, bind);
            leafMaxKernel_.dispatch(batch, {placements_[g].leafCount, 1, 1}, bind);
        }
    }
    for (size_t v = 0; v < volumes.size(); ++v) {
        const VolumeInput& in = volumes[v];
        const uint32_t count = in.instanceRows != nullptr ? in.instanceCount : 1u;
        if (count == 0) {
            continue;
        }
        const auto bind = [&](rhi::ShaderCursor cursor) {
            bindGrid(cursor, gridOf[v]);
            cursor["params"]["firstRecord"].setData(firstRecord[v]);
            cursor["params"]["recordCount"].setData(count);
            cursor["params"]["prototype0"].setData(prototypes[v].data(), sizeof(float) * 4);
            cursor["params"]["prototype1"].setData(prototypes[v].data() + 4, sizeof(float) * 4);
            cursor["params"]["prototype2"].setData(prototypes[v].data() + 8, sizeof(float) * 4);
        };
        if (in.instanceRows != nullptr) {
            instancesKernel_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor);
                cursor["instanceRows"].setBinding(in.instanceRows->rhi());
            });
        }
        finishKernel_.dispatch(batch, {count, 1, 1}, bind);
    }
    return ok();
}

}   // namespace athenea::world
