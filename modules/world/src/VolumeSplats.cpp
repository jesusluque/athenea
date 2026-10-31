// Copyright (c) 2026 jesus luque.
#include "athenea/world/VolumeSplats.h"

#include <algorithm>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::world {

namespace {

constexpr const char* kModule = "athenea/volume/vdb_splats";

/// Index to world as the kernels take it: the rows of the column-vector
/// map, from OpenVDB's row-vector one (translation in its last row).
void setRows(rhi::ShaderCursor params, const std::array<double, 16>& m) {
    for (int c = 0; c < 3; ++c) {
        const float row[4] = {static_cast<float>(m[static_cast<size_t>(0 * 4 + c)]),
                              static_cast<float>(m[static_cast<size_t>(1 * 4 + c)]),
                              static_cast<float>(m[static_cast<size_t>(2 * 4 + c)]),
                              static_cast<float>(m[static_cast<size_t>(3 * 4 + c)])};
        params[c == 0 ? "indexToWorld0" : c == 1 ? "indexToWorld1" : "indexToWorld2"].setData(row, sizeof(row));
    }
}

}   // namespace

PackedGrids PackedGrids::of(const io::NanoGrid& density, const io::NanoGrid* temperature, const io::NanoGrid* colour) {
    PackedGrids p;
    const auto place = [&p](const io::NanoGrid& g) {
        while (p.words.size() % 8 != 0) p.words.push_back(0u);
        const auto at = static_cast<uint32_t>(p.words.size());
        p.words.insert(p.words.end(), g.words.begin(), g.words.end());
        return at;
    };
    p.density = place(density);
    p.leafCount = density.leafCount;
    if (temperature != nullptr) p.temperature = place(*temperature);
    if (colour != nullptr) p.colour = place(*colour);
    while (p.words.size() % 8 != 0) p.words.push_back(0u);
    return p;
}

io::NanoGrid PackedGrids::unpack(uint32_t word, const io::NanoGrid& like) const {
    io::NanoGrid out = like;
    out.words.assign(words.begin() + word, words.begin() + word + static_cast<std::ptrdiff_t>(like.words.size()));
    return out;
}

Result<VolumeSplats> VolumeSplats::create(gpu::ShaderLibrary& library) {
    VolumeSplats v;
    v.device_ = &library.device();
    const auto make = [&](gpu::ComputeKernel& into, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, kModule, entry);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(v.pack_, "vsPackLeaves"));
    ATHENEA_TRY(make(v.back_, "vsSplatToGrid"));
    ATHENEA_TRY(make(v.fixture_, "vsFixture"));
    ATHENEA_TRY(make(v.mixture_, "vsRandomMixture"));
    ATHENEA_TRY(make(v.compare_, "vsCompare"));
    ATHENEA_TRY(make(v.reduce_, "vsReduce"));
    ATHENEA_TRY(make(v.rays_, "vsRays"));
    ATHENEA_TRY(make(v.export_, "vsExport"));
    ATHENEA_TRY(make(v.slots_, "vsSlots"));
    ATHENEA_TRY(make(v.flicker_, "vsFlicker"));
    return v;
}

void VolumeSplats::packLeaves(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                              const gpu::Buffer& leaves) const {
    pack_.dispatch(batch, {grids.leafCount * 512, 1, 1}, [&](rhi::ShaderCursor c) {
        c["words"].setBinding(words.rhi());
        c["leavesOut"].setBinding(leaves.rhi());
        c["params"]["densityWord"].setData(grids.density);
        c["params"]["leafCount"].setData(grids.leafCount);
        c["params"]["temperatureWord"].setData(grids.temperature);
        c["params"]["colourWord"].setData(grids.colour);
    });
}

Result<void> VolumeSplats::splatToGrid(const gpu::Buffer& gaussians, uint32_t count, const gpu::Buffer& words,
                                       uint32_t densityWord, uint32_t leafCount, uint32_t temperatureWord) const {
    // A slice of leaves a submission: every group walks every gaussian, and
    // a command buffer that runs for seconds is one the OS may cut short.
    const uint64_t perSlice =
        std::max<uint64_t>(64, (uint64_t{1} << 28) / std::max<uint64_t>(uint64_t{count} * 16, 1));
    for (uint32_t first = 0; first < leafCount; first += static_cast<uint32_t>(perSlice)) {
        const uint32_t slice = static_cast<uint32_t>(std::min<uint64_t>(perSlice, leafCount - first));
        gpu::CommandBatch batch(*device_);
        back_.dispatch(batch, {slice * 256, 1, 1}, [&](rhi::ShaderCursor c) {
            c["words"].setBinding(words.rhi());
            c["wordsOut"].setBinding(words.rhi());
            c["gaussians"].setBinding(gaussians.rhi());
            c["params"]["densityWord"].setData(densityWord);
            c["params"]["leafCount"].setData(leafCount);
            c["params"]["temperatureWord"].setData(temperatureWord);
            c["params"]["count"].setData(count);
            c["params"]["leafFirst"].setData(first);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    return ok();
}

void VolumeSplats::fixture(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                           uint32_t kind, std::array<float, 3> centre, float radius, uint32_t seed) const {
    fixture_.dispatch(batch, {grids.leafCount * 512, 1, 1}, [&](rhi::ShaderCursor c) {
        c["words"].setBinding(words.rhi());
        c["wordsOut"].setBinding(words.rhi());
        c["params"]["densityWord"].setData(grids.density);
        c["params"]["leafCount"].setData(grids.leafCount);
        c["params"]["temperatureWord"].setData(grids.temperature);
        c["params"]["kind"].setData(kind);
        const float at[4] = {centre[0], centre[1], centre[2], radius};
        c["params"]["centre"].setData(at, sizeof(at));
        c["params"]["seed"].setData(seed);
    });
}

void VolumeSplats::packVelocity(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                                float scale, const gpu::Buffer& out) const {
    pack_.dispatch(batch, {grids.leafCount * 512, 1, 1}, [&](rhi::ShaderCursor c) {
        c["words"].setBinding(words.rhi());
        c["leavesOut"].setBinding(out.rhi());
        c["params"]["densityWord"].setData(grids.density);
        c["params"]["leafCount"].setData(grids.leafCount);
        c["params"]["temperatureWord"].setData(PackedGrids::kNoGrid);
        c["params"]["colourWord"].setData(grids.colour);
        c["params"]["kind"].setData(3u);
        const float motion[4] = {0.0F, 0.0F, 0.0F, scale};
        c["params"]["motion"].setData(motion, sizeof(motion));
    });
}

void VolumeSplats::sequenceFixture(gpu::CommandBatch& batch, const gpu::Buffer& words, const PackedGrids& grids,
                                   std::array<float, 3> centre, float radius, uint32_t seed, float omega, float rise,
                                   float frame, float voxelSize) const {
    fixture_.dispatch(batch, {grids.leafCount * 512, 1, 1}, [&](rhi::ShaderCursor c) {
        c["words"].setBinding(words.rhi());
        c["wordsOut"].setBinding(words.rhi());
        c["params"]["densityWord"].setData(grids.density);
        c["params"]["leafCount"].setData(grids.leafCount);
        c["params"]["temperatureWord"].setData(grids.temperature);
        c["params"]["colourWord"].setData(grids.colour);
        c["params"]["kind"].setData(2u);
        const float at[4] = {centre[0], centre[1], centre[2], radius};
        c["params"]["centre"].setData(at, sizeof(at));
        c["params"]["seed"].setData(seed);
        const float motion[4] = {omega, rise, frame, voxelSize};
        c["params"]["motion"].setData(motion, sizeof(motion));
    });
}

Result<SlotStability> VolumeSplats::slots(const gpu::Buffer& before, const gpu::Buffer& now, uint32_t count) const {
    gpu::BufferDesc desc;
    desc.bytes = (uint64_t{count} * 3 + 4) * 4;
    desc.elementBytes = 4;
    desc.label = "vdb2splat.slots";
    auto sums = gpu::Buffer::create(*device_, desc);
    if (!sums) return std::move(sums).error();
    gpu::CommandBatch batch(*device_);
    const auto bind = [&](rhi::ShaderCursor c) {
        c["gaussians"].setBinding(before.rhi());
        c["gaussiansB"].setBinding(now.rhi());
        c["sums"].setBinding(sums->rhi());
        c["params"]["count"].setData(count);
        c["params"]["kind"].setData(3u);
    };
    slots_.dispatch(batch, {count, 1, 1}, bind);
    reduce_.dispatch(batch, {256, 1, 1}, bind);
    ATHENEA_TRY(batch.submit(true));
    float total[4] = {};
    ATHENEA_TRY(sums->read(*device_, uint64_t{count} * 12, sizeof(total), total));
    SlotStability s;
    s.aliveBoth = static_cast<uint32_t>(total[0] + 0.5F);
    s.aliveBefore = static_cast<uint32_t>(total[1] + 0.5F);
    s.meanMove = s.aliveBoth > 0 ? total[2] / static_cast<double>(s.aliveBoth) : 0.0;
    return s;
}

Result<double> VolumeSplats::flicker(const gpu::Buffer& judged, const gpu::Buffer& reference,
                                     const gpu::Buffer& judgedBefore, const gpu::Buffer& referenceBefore,
                                     uint32_t densityWord, uint32_t leafCount) const {
    gpu::BufferDesc desc;
    desc.bytes = (uint64_t{leafCount} * 3 + 4) * 4;
    desc.elementBytes = 4;
    desc.label = "vdb2splat.flicker";
    auto sums = gpu::Buffer::create(*device_, desc);
    if (!sums) return std::move(sums).error();
    gpu::CommandBatch batch(*device_);
    const auto bind = [&](rhi::ShaderCursor c) {
        c["words"].setBinding(judged.rhi());
        c["other"].setBinding(reference.rhi());
        c["prevJudged"].setBinding(judgedBefore.rhi());
        c["prevOther"].setBinding(referenceBefore.rhi());
        c["sums"].setBinding(sums->rhi());
        c["params"]["densityWord"].setData(densityWord);
        c["params"]["leafCount"].setData(leafCount);
        c["params"]["count"].setData(leafCount);
        c["params"]["kind"].setData(0u);
    };
    flicker_.dispatch(batch, {leafCount * 256, 1, 1}, bind);
    reduce_.dispatch(batch, {256, 1, 1}, bind);
    ATHENEA_TRY(batch.submit(true));
    float total[4] = {};
    ATHENEA_TRY(sums->read(*device_, uint64_t{leafCount} * 12, sizeof(total), total));
    return static_cast<double>(total[3]);
}

void VolumeSplats::randomMixture(gpu::CommandBatch& batch, const gpu::Buffer& out, uint32_t count,
                                 std::array<float, 3> extent, float margin, float minStd, float maxStd,
                                 uint32_t seed) const {
    mixture_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["exported"].setBinding(out.rhi());
        c["params"]["count"].setData(count);
        const float box[4] = {extent[0], extent[1], extent[2], margin};
        c["params"]["centre"].setData(box, sizeof(box));
        c["params"]["tolerance"].setData(minStd);
        c["params"]["densityScale"].setData(maxStd);
        c["params"]["seed"].setData(seed);
    });
}

Result<GridError> VolumeSplats::compare(const gpu::Buffer& judged, const gpu::Buffer& reference, uint32_t densityWord,
                                        uint32_t leafCount) const {
    return compareWeighted(judged, reference, densityWord, leafCount, PackedGrids::kNoGrid);
}

Result<GridError> VolumeSplats::compareWeighted(const gpu::Buffer& judged, const gpu::Buffer& reference,
                                                uint32_t densityWord, uint32_t leafCount,
                                                uint32_t temperatureWord) const {
    gpu::BufferDesc desc;
    desc.bytes = (uint64_t{leafCount} * 3 + 4) * 4;
    desc.elementBytes = 4;
    desc.label = "vdb2splat.compare";
    auto sums = gpu::Buffer::create(*device_, desc);
    if (!sums) return std::move(sums).error();
    gpu::CommandBatch batch(*device_);
    const auto bind = [&](rhi::ShaderCursor c) {
        c["words"].setBinding(judged.rhi());
        c["other"].setBinding(reference.rhi());
        c["sums"].setBinding(sums->rhi());
        c["params"]["densityWord"].setData(densityWord);
        c["params"]["leafCount"].setData(leafCount);
        c["params"]["count"].setData(leafCount);
        c["params"]["kind"].setData(temperatureWord == PackedGrids::kNoGrid ? 0u : 1u);
        c["params"]["temperatureWord"].setData(temperatureWord);
    };
    compare_.dispatch(batch, {leafCount * 256, 1, 1}, bind);
    reduce_.dispatch(batch, {256, 1, 1}, bind);
    ATHENEA_TRY(batch.submit(true));
    float total[4] = {};
    ATHENEA_TRY(sums->read(*device_, uint64_t{leafCount} * 12, sizeof(total), total));
    GridError e;
    e.sumSquares = total[0];
    e.sumReference = total[1];
    e.largest = total[2];
    e.relativeL2 = total[3];
    return e;
}

Result<RayError> VolumeSplats::rays(const gpu::Buffer& volumeWords, const gpu::Buffer& gaussians, uint32_t count,
                                    const std::array<double, 16>& indexToWorld, uint32_t rays, uint32_t samples,
                                    uint32_t seed, float tolerance) const {
    gpu::BufferDesc desc;
    desc.bytes = (uint64_t{rays} * 3 + 4) * 4;
    desc.elementBytes = 4;
    desc.label = "vdb2splat.rays";
    auto sums = gpu::Buffer::create(*device_, desc);
    if (!sums) return std::move(sums).error();
    // Slices of rays a submission, for the same reason as the way back.
    const uint32_t perSlice = std::max<uint32_t>(64, static_cast<uint32_t>((uint64_t{1} << 26) /
                                                                          std::max<uint64_t>(count, 1)));
    for (uint32_t first = 0; first < rays; first += perSlice) {
        const uint32_t slice = std::min(perSlice, rays - first);
        gpu::CommandBatch batch(*device_);
        rays_.dispatch(batch, {slice, 1, 1}, [&](rhi::ShaderCursor c) {
            c["words"].setBinding(volumeWords.rhi());
            c["gaussians"].setBinding(gaussians.rhi());
            c["sums"].setBinding(sums->rhi());
            c["params"]["count"].setData(count);
            c["params"]["rays"].setData(rays);
            c["params"]["leafFirst"].setData(first);
            c["params"]["samples"].setData(samples);
            c["params"]["seed"].setData(seed);
            c["params"]["tolerance"].setData(tolerance);
            setRows(c["params"], indexToWorld);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    gpu::CommandBatch batch(*device_);
    reduce_.dispatch(batch, {256, 1, 1}, [&](rhi::ShaderCursor c) {
        c["sums"].setBinding(sums->rhi());
        c["params"]["count"].setData(rays);
        c["params"]["kind"].setData(2u);
    });
    ATHENEA_TRY(batch.submit(true));
    float total[4] = {};
    ATHENEA_TRY(sums->read(*device_, uint64_t{rays} * 12, sizeof(total), total));
    RayError e;
    e.rays = rays;
    e.meanDifference = total[3];
    e.beyond = static_cast<uint32_t>(total[1] + 0.5F);
    e.largestDepth = total[2];
    return e;
}

void VolumeSplats::exportRecords(gpu::CommandBatch& batch, const gpu::Buffer& gaussians, uint32_t count,
                                 const std::array<double, 16>& indexToWorld, float densityScale,
                                 const gpu::Buffer& out) const {
    export_.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["gaussians"].setBinding(gaussians.rhi());
        c["exported"].setBinding(out.rhi());
        c["params"]["count"].setData(count);
        c["params"]["densityScale"].setData(densityScale);
        setRows(c["params"], indexToWorld);
    });
}

}   // namespace athenea::world
