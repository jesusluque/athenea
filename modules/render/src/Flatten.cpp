// Copyright (c) 2026 jesus luque.
//
// Flatten.h says what this is. The processor here sizes buffers, sets
// parameters, and copies what the kernels wrote out; every number of a
// gaussian is the device's.
#include "athenea/render/Flatten.h"

#include <algorithm>
#include <cstring>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::render {
namespace {

Result<gpu::Buffer> floatsBuffer(gpu::Device& device, uint64_t count, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * 4;
    desc.elementBytes = 4;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

constexpr std::array<uint32_t, 6> kSpzBytes{9, 1, 3, 3, 4, 45};

}   // namespace

Result<FlattenFit> FlattenFit::create(gpu::ShaderLibrary& library) {
    FlattenFit fit;
    fit.library_ = &library;
    auto gram = gpu::ComputeKernel::create(library, "athenea/splat/flatten_fit", "flattenGram");
    if (!gram) return std::move(gram).error();
    auto solve = gpu::ComputeKernel::create(library, "athenea/splat/flatten_fit", "flattenSolve");
    if (!solve) return std::move(solve).error();
    auto floor = gpu::ComputeKernel::create(library, "athenea/splat/flatten_fit", "flattenRoughnessFloor");
    if (!floor) return std::move(floor).error();
    fit.gram_ = std::move(*gram);
    fit.solve_ = std::move(*solve);
    fit.floorKernel_ = std::move(*floor);
    return fit;
}

Result<void> FlattenFit::prepare(const FlattenSettings& settings) {
    if (settings.directions < 16) {
        return Error(ErrorCode::InvalidArgument, "flatten: at least 16 directions fit 16 harmonics");
    }
    if (directions_ == settings.directions && lambda_ == settings.lambda && backWeight_ == settings.backWeight &&
        floorShare_ == settings.floorShare) {
        return ok();
    }
    gpu::Device& device = library_->device();
    auto gram = floatsBuffer(device, 2 * 256, "flatten.gram");
    if (!gram) return std::move(gram).error();
    auto basis = floatsBuffer(device, uint64_t{2} * settings.directions * 16, "flatten.basis");
    if (!basis) return std::move(basis).error();
    // The floor's candidate starts past the last, which is what "none
    // passed" reads as.
    const std::array<uint32_t, 1> none{64u};
    auto floor = gpu::Buffer::fromSpan<uint32_t>(device, none, "flatten.floor");
    if (!floor) return std::move(floor).error();
    const auto bind = [&](rhi::ShaderCursor c) {
        c["gram"].setBinding(gram->rhi());
        c["basisRows"].setBinding(basis->rhi());
        c["floorIndex"].setBinding(floor->rhi());
        c["params"]["directions"].setData(settings.directions);
        c["params"]["lambda"].setData(settings.lambda);
        c["params"]["backWeight"].setData(settings.backWeight);
        c["params"]["floorShare"].setData(settings.floorShare);
    };
    {
        gpu::CommandBatch batch(device);
        gram_.dispatch(batch, {2, 1, 1}, bind);
        ATHENEA_TRY(batch.submit(true));
    }
    {
        gpu::CommandBatch batch(device);
        solve_.dispatch(batch, {2 * settings.directions, 1, 1}, bind);
        floorKernel_.dispatch(batch, {64, 1, 1}, bind);
        ATHENEA_TRY(batch.submit(true));
    }
    gramBuffer_ = std::move(*gram);
    basis_ = std::move(*basis);
    floor_ = std::move(*floor);
    directions_ = settings.directions;
    lambda_ = settings.lambda;
    backWeight_ = settings.backWeight;
    floorShare_ = settings.floorShare;
    return ok();
}

Result<float> FlattenFit::floorRoughness() {
    uint32_t index = 64;
    ATHENEA_TRY(floor_.read(library_->device(), 0, 4, &index));
    // flatten_basis.slang's flatFloorCandidate: (index + 1) / 64, clamped.
    return static_cast<float>(std::min(index, 63u) + 1) / 64.0F;
}

Result<FlattenPack> FlattenPack::create(gpu::ShaderLibrary& library) {
    FlattenPack pack;
    pack.library_ = &library;
    const auto make = [&](gpu::ComputeKernel& into, const char* entry) -> Result<void> {
        auto made = gpu::ComputeKernel::create(library, "athenea/splat/flatten_pack", entry);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        return ok();
    };
    ATHENEA_TRY(make(pack.bounds_, "flattenBounds"));
    ATHENEA_TRY(make(pack.resolve_, "flattenBoundsResolve"));
    ATHENEA_TRY(make(pack.ply_, "flattenPackPly"));
    ATHENEA_TRY(make(pack.spz_, "flattenPackSpz"));
    ATHENEA_TRY(make(pack.glb_, "flattenPackGlb"));
    return pack;
}

namespace {

/// Every name flatten_pack.slang declares, bound: the ones a kernel does not
/// read to `spare`.
void bindPack(rhi::ShaderCursor c, const gpu::Buffer& records, const gpu::Buffer& floats, const gpu::Buffer& words,
              const gpu::Buffer& stats) {
    c["records"].setBinding(records.rhi());
    c["floats"].setBinding(floats.rhi());
    c["words"].setBinding(words.rhi());
    c["stats"].setBinding(stats.rhi());
}

}   // namespace

Result<FlatStats> FlattenPack::stats(const std::vector<FlatCloud>& clouds, float logScaleFloor) {
    gpu::Device& device = library_->device();
    std::array<uint32_t, 32> start{};
    for (int a = 0; a < 3; ++a) {
        start[static_cast<size_t>(a)] = 0xFFFFFFFFu;
    }
    auto stats = gpu::Buffer::fromSpan<uint32_t>(device, start, "flatten.stats");
    if (!stats) return std::move(stats).error();
    gpu::CommandBatch batch(device);
    for (const FlatCloud& cloud : clouds) {
        bounds_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor c) {
            bindPack(c, cloud.records, cloud.records, *stats, *stats);
            c["params"]["count"].setData(cloud.count);
            c["params"]["logScaleFloor"].setData(logScaleFloor);
        });
    }
    resolve_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
        const FlatCloud* any = clouds.empty() ? nullptr : &clouds.front();
        bindPack(c, any != nullptr ? any->records : *stats, *stats, *stats, *stats);
        c["params"]["count"].setData(0u);
    });
    ATHENEA_TRY(batch.submit(true));
    std::array<uint32_t, 32> got{};
    ATHENEA_TRY(stats->read(device, 0, sizeof(got), got.data()));
    FlatStats out;
    for (size_t a = 0; a < 3; ++a) {
        std::memcpy(&out.min[a], &got[16 + a], 4);
        std::memcpy(&out.max[a], &got[19 + a], 4);
    }
    out.sizesFloored = got[6];
    out.spzSaturated = got[7];
    out.shrunk = got[8];
    out.spzSizesFloored = got[9];
    out.fractionalBits = got[22];
    return out;
}

Result<std::vector<float>> FlattenPack::ply(const FlatCloud& cloud, float logScaleFloor) {
    gpu::Device& device = library_->device();
    auto floats = floatsBuffer(device, uint64_t{cloud.count} * 62, "flatten.ply");
    if (!floats) return std::move(floats).error();
    {
        gpu::CommandBatch batch(device);
        ply_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor c) {
            bindPack(c, cloud.records, *floats, *floats, *floats);
            c["params"]["count"].setData(cloud.count);
            c["params"]["base"].setData(0u);
            c["params"]["logScaleFloor"].setData(logScaleFloor);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::vector<float> out(size_t{cloud.count} * 62);
    ATHENEA_TRY(floats->read(device, 0, out.size() * 4, out.data()));
    return out;
}

Result<SpzStreams> FlattenPack::spz(const std::vector<FlatCloud>& clouds, uint32_t fractionalBits,
                                    uint32_t sh1Bits, uint32_t shRestBits) {
    if (sh1Bits < 1 || sh1Bits > 8 || shRestBits < 1 || shRestBits > 8) {
        return Error(ErrorCode::InvalidArgument, "flatten: SPZ keeps 1 to 8 bits of a harmonic");
    }
    gpu::Device& device = library_->device();
    SpzStreams out;
    for (const FlatCloud& cloud : clouds) {
        for (uint32_t stream = 0; stream < 6; ++stream) {
            const uint64_t bytes = uint64_t{cloud.count} * kSpzBytes[stream];
            const uint64_t words = (bytes + 3) / 4;
            if (words > 0xFFFFFFFFull) {
                return Error(ErrorCode::Unsupported, "flatten: an SPZ stream past 2^32 words");
            }
            gpu::BufferDesc desc;
            desc.bytes = std::max<uint64_t>(words, 1) * 4;
            desc.elementBytes = 4;
            desc.label = "flatten.spz";
            auto buffer = gpu::Buffer::create(device, desc);
            if (!buffer) return std::move(buffer).error();
            {
                gpu::CommandBatch batch(device);
                spz_.dispatch(batch, {static_cast<uint32_t>(words), 1, 1}, [&](rhi::ShaderCursor c) {
                    bindPack(c, cloud.records, *buffer, *buffer, *buffer);
                    c["params"]["count"].setData(cloud.count);
                    c["params"]["stream"].setData(stream);
                    c["params"]["words"].setData(static_cast<uint32_t>(words));
                    c["params"]["fractionalBits"].setData(fractionalBits);
                    c["params"]["sh1Bits"].setData(sh1Bits);
                    c["params"]["shRestBits"].setData(shRestBits);
                });
                ATHENEA_TRY(batch.submit(true));
            }
            std::vector<uint8_t>& into = out.streams[stream];
            const size_t at = into.size();
            into.resize(at + bytes);
            if (bytes > 0) {
                ATHENEA_TRY(buffer->read(device, 0, bytes, into.data() + at));
            }
        }
    }
    return out;
}

Result<std::vector<float>> FlattenPack::glb(const FlatCloud& cloud, float logScaleFloor) {
    gpu::Device& device = library_->device();
    auto floats = floatsBuffer(device, uint64_t{cloud.count} * 59, "flatten.glb");
    if (!floats) return std::move(floats).error();
    {
        gpu::CommandBatch batch(device);
        glb_.dispatch(batch, {cloud.count, 1, 1}, [&](rhi::ShaderCursor c) {
            bindPack(c, cloud.records, *floats, *floats, *floats);
            c["params"]["count"].setData(cloud.count);
            c["params"]["base"].setData(0u);
            c["params"]["total"].setData(cloud.count);
            c["params"]["logScaleFloor"].setData(logScaleFloor);
        });
        ATHENEA_TRY(batch.submit(true));
    }
    std::vector<float> out(size_t{cloud.count} * 59);
    ATHENEA_TRY(floats->read(device, 0, out.size() * 4, out.data()));
    return out;
}

}   // namespace athenea::render
