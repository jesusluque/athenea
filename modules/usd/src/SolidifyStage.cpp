// Copyright (c) 2026 jesus luque.
//
// `athenea solidify`'s stage side (SSF phase 2b): a stage's ParticleField
// solidified (lod::Solidifier) and written back as Mode A -- the blobs as flat
// gaussians, then the residuals -- into a copy of the stage it came from,
// every other array the field carries (material, transfer, normals, ids)
// taken from the gaussian each output came from, on the device.
#include "athenea/usd/Export.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/lod/Solidify.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::usd {
namespace {

Result<gpu::Buffer> words(gpu::Device& device, uint64_t count, const char* label, const void* initial = nullptr) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * 4;
    desc.elementBytes = 4;
    desc.label = label;
    return gpu::Buffer::create(device, desc, count > 0 ? initial : nullptr);
}

}   // namespace

Result<SolidifyStageStats> solidifyStage(gpu::ShaderLibrary& library, const std::filesystem::path& input,
                                         const std::filesystem::path& output, const SolidifyStageOptions& o) {
    gpu::Device& device = library.device();
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return std::move(loader).error();
    std::string where;
    auto raw = readParticleFieldRecords(input, o.prim, &where);
    if (!raw) return std::move(raw).error();
    const uint32_t records = raw->count;
    auto arrays = readGaussianArrays(input, where, records);
    if (!arrays) return std::move(arrays).error();
    auto cloud = uploadParticleField(library, *loader, input, where);
    if (!cloud) return std::move(cloud).error();
    const Clock::time_point t1 = Clock::now();

    auto solidifier = lod::Solidifier::create(library);
    if (!solidifier) return std::move(solidifier).error();
    auto result = solidifier->solidify(*cloud, o.settings);
    if (!result) return std::move(result).error();
    auto splats = solidifier->toSplats(*result, *cloud, o.settings);
    if (!splats) return std::move(splats).error();
    const Clock::time_point t2 = Clock::now();

    // Which record each output came from: the blobs' sources, then the
    // residuals', through the cloud's origin (validation may have dropped
    // records).
    const uint32_t m = result->solid.blobs;
    const uint32_t total = m + result->residuals;
    auto gatherWords = gpu::ComputeKernel::create(library, "athenea/lod/ssf_splat", "ssfGatherWords");
    if (!gatherWords) return std::move(gatherWords).error();
    auto dummy = words(device, 4, "solidify.dummy");
    auto sources = words(device, total, "solidify.sources");
    auto recordOf = words(device, total, "solidify.records");
    for (auto* r : {&dummy, &sources, &recordOf}) {
        if (!*r) return std::move(*r).error();
    }
    const auto gather = [&](gpu::CommandBatch& batch, const gpu::Buffer& from, const gpu::Buffer& index,
                            const gpu::Buffer& to, uint32_t count, uint32_t width) {
        gatherWords->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
            for (const char* name : {"blobPositions", "blobShape", "blobNormals", "anchors", "gridKeys", "gridOrder",
                                     "gridInfo", "srcPositions", "srcShape", "srcSh", "srcNormals", "dstPositions",
                                     "dstShape", "dstSh", "dstNormals"}) {
                c[name].setBinding(dummy->rhi());
            }
            c["sources"].setBinding(index.rhi());
            c["srcWords"].setBinding(from.rhi());
            c["dstWords"].setBinding(to.rhi());
            c["params"]["count"].setData(count);
            c["params"]["first"].setData(0u);
            c["params"]["geometry"].setData(width);
        });
    };
    {
        gpu::CommandBatch batch(device);
        if (m > 0) {
            batch.encoder()->copyBuffer(sources->rhi(), 0, result->solidSource.rhi(), 0, uint64_t{m} * 4);
        }
        if (result->residuals > 0) {
            batch.encoder()->copyBuffer(sources->rhi(), uint64_t{m} * 4, result->residualSource.rhi(), 0,
                                        uint64_t{result->residuals} * 4);
        }
        batch.markDirty();
        if (cloud->origin.valid()) {
            gather(batch, cloud->origin, *sources, *recordOf, total, 1);
        } else {
            batch.encoder()->copyBuffer(recordOf->rhi(), 0, sources->rhi(), 0, uint64_t{total} * 4);
            batch.markDirty();
        }
        ATHENEA_TRY(batch.submit(true));
    }
    // Every array, every sample, gathered on the device.
    std::vector<GaussianArray> gathered;
    gathered.reserve(arrays->size());
    for (const GaussianArray& from : *arrays) {
        GaussianArray a = from;
        for (std::vector<uint32_t>& values : a.values) {
            auto in = words(device, values.size(), "solidify.array", values.data());
            if (!in) return std::move(in).error();
            auto out = words(device, uint64_t{total} * a.width, "solidify.array");
            if (!out) return std::move(out).error();
            gpu::CommandBatch batch(device);
            gather(batch, *in, *recordOf, *out, total, a.width);
            ATHENEA_TRY(batch.submit(true));
            auto back = out->readAll<uint32_t>(device);
            if (!back) return std::move(back).error();
            back->resize(size_t{total} * a.width);
            values = std::move(*back);
        }
        gathered.push_back(std::move(a));
    }
    auto kept = loader->records(*splats);
    if (!kept) return std::move(kept).error();
    ExportOptions options;
    options.maxDegree = o.degree;
    options.addCamera = false;
    const std::filesystem::path gaussians =
        std::filesystem::temp_directory_path() / ("athenea_solidify_" + output.stem().string() + ".usdc");
    ATHENEA_TRY(writeParticleFieldStage(library, *kept, gaussians, options));
    auto written = writeDecimatedStage(input, output, where, gaussians, gathered);
    std::filesystem::remove(gaussians);
    if (!written) return std::move(written).error();
    const Clock::time_point t3 = Clock::now();

    SolidifyStageStats stats;
    stats.gaussians = result->gaussians;
    stats.candidates = result->candidates;
    stats.absorbed = result->coherent;
    stats.blobs = m;
    stats.residuals = result->residuals;
    const auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    stats.readSeconds = seconds(t0, t1);
    stats.solidifySeconds = seconds(t1, t2);
    stats.writeSeconds = seconds(t2, t3);
    return stats;
}

}   // namespace athenea::usd
