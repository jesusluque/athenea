// Copyright (c) 2026 jesus luque.
#include "athenea/technique/VolumeSplatShading.h"

#include <algorithm>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/technique/SplatShadowMap.h"

namespace athenea::technique {
namespace {

Result<gpu::Buffer> words(gpu::Device& device, uint64_t count, uint32_t elementBytes, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * elementBytes;
    desc.elementBytes = elementBytes;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

constexpr const char* kRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                  "w12", "w13", "w20", "w21", "w22", "w23"};

}   // namespace

Result<VolumeSplatShading> VolumeSplatShading::create(gpu::ShaderLibrary& library) {
    VolumeSplatShading s;
    s.device_ = &library.device();
    auto shade = gpu::ComputeKernel::create(library, "athenea/technique/volume_splat_shade", "volumeSplatShade");
    if (!shade) return std::move(shade).error();
    s.shade_ = std::move(*shade);
    auto table = gpu::ComputeKernel::create(library, "athenea/technique/volume_blackbody", "volumeBlackbody");
    if (!table) return std::move(table).error();
    s.blackbodyKernel_ = std::move(*table);
    auto blackbody = words(*s.device_, kBlackbodyEntries, 16, "volume.blackbody");
    if (!blackbody) return std::move(blackbody).error();
    s.blackbody_ = std::move(*blackbody);
    auto noMap = words(*s.device_, 1, 4, "volume.shade.noMap");
    if (!noMap) return std::move(noMap).error();
    s.noMap_ = std::move(*noMap);
    auto noLights = words(*s.device_, 1, 160, "volume.shade.noLights");
    if (!noLights) return std::move(noLights).error();
    s.noLights_ = std::move(*noLights);
    return s;
}

Result<void> VolumeSplatShading::shade(gpu::CommandBatch& batch, const VolumeShadeJob& job, gpu::Buffer& out) {
    const scene::GpuSplats* cloud = job.cloud;
    if (cloud == nullptr || cloud->count == 0 || !cloud->hasVolume()) {
        return ok();
    }
    if (!blackbodyMade_) {
        blackbodyKernel_.dispatch(batch, {kBlackbodyEntries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["blackbodyOut"].setBinding(blackbody_.rhi());
            cursor["blackbodyParams"]["size"].setData(kBlackbodyEntries);
            cursor["blackbodyParams"]["maxKelvin"].setData(kBlackbodyMaxKelvin);
        });
        blackbodyMade_ = true;
    }
    if (!out.valid() || out.bytes() < uint64_t{cloud->count} * 8) {
        auto made = words(*device_, uint64_t{cloud->count} * 2, 4, "volume.radiance");
        if (!made) return std::move(made).error();
        out = std::move(*made);
    }
    const bool map = job.shadowMap != nullptr && job.shadowMap->valid();
    const bool lights = job.lights != nullptr && job.lightCount > 0;
    shade_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(cloud->positions.rhi());
        cursor["shape"].setBinding(cloud->shape.rhi());
        cursor["volume"].setBinding(cloud->volume.rhi());
        cursor["lights"].setBinding(lights ? job.lights->rhi() : noLights_.rhi());
        cursor["volumeShadowMap"].setBinding(map ? job.shadowMap->map().rhi() : noMap_.rhi());
        cursor["blackbodyTable"].setBinding(blackbody_.rhi());
        cursor["radianceOut"].setBinding(out.rhi());
        rhi::ShaderCursor p = cursor["shadeParams"];
        p["count"].setData(cloud->count);
        p["lightCount"].setData(lights ? job.lightCount : 0u);
        p["hasMap"].setData(uint32_t{map ? 1u : 0u});
        p["octaves"].setData(std::max(job.look.octaves, 1u));
        p["albedo"].setData(job.look.albedo.data(), sizeof(float) * 3);
        p["anisotropy"].setData(job.look.anisotropy);
        p["emission"].setData(job.look.emission.data(), sizeof(float) * 3);
        p["emissionScale"].setData(job.look.emissionScale);
        p["temperatureScale"].setData(job.look.temperatureScale);
        p["temperatureOffset"].setData(job.look.temperatureOffset);
        p["categoriesLo"].setData(static_cast<uint32_t>(job.categories & 0xFFFFFFFFU));
        p["categoriesHi"].setData(static_cast<uint32_t>(job.categories >> 32));
        for (int k = 0; k < 12; ++k) {
            p[kRow[k]].setData(job.objectToWorld[static_cast<size_t>(k)]);
        }
        p["eye"].setData(job.eye.data(), sizeof(float) * 3);
        p["octaveExtinction"].setData(job.octaveExtinction);
        p["octaveEnergy"].setData(job.octaveEnergy);
        p["octaveAnisotropy"].setData(job.octaveAnisotropy);
        p["lutSize"].setData(kBlackbodyEntries);
        p["lutMaxKelvin"].setData(kBlackbodyMaxKelvin);
    });
    return ok();
}

}   // namespace athenea::technique
