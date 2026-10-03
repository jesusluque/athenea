// Copyright (c) 2026 jesus luque.
#include "athenea/technique/SplatShadowMap.h"

#include <algorithm>
#include <cstdlib>

#include "athenea/core/Log.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::technique {
namespace {

/// The shader's, and they must agree: twenty words a light slot, then the six
/// the casters' box is reduced into.
constexpr uint32_t kFrameWords = 20;
constexpr uint32_t kSlots = 8;
constexpr uint32_t kHeaderWords = kFrameWords * kSlots + 6;

/// The rows of `objectToWorld`, by the names the parameter block gives them.
constexpr const char* kRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                  "w12", "w13", "w20", "w21", "w22", "w23"};

}   // namespace

Result<SplatShadowMap> SplatShadowMap::create(gpu::ShaderLibrary& library) {
    SplatShadowMap map;
    map.device_ = &library.device();
    map.library_ = &library;
    const auto make = [&](std::optional<gpu::ComputeKernel>& into, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, "athenea/technique/splat_shadow_map", entry);
        if (!kernel) return std::move(kernel).error();
        into.emplace(std::move(*kernel));
        return ok();
    };
    ATHENEA_TRY(make(map.clear_, "shadowMapClear"));
    ATHENEA_TRY(make(map.box_, "shadowMapBox"));
    ATHENEA_TRY(make(map.frame_, "shadowMapFrame"));
    ATHENEA_TRY(make(map.splat_, "shadowMapSplat"));
    ATHENEA_TRY(make(map.mean_, "shadowMapMean"));
    ATHENEA_TRY(make(map.probe_, "shadowMapProbe"));
    ATHENEA_TRY(make(map.header_, "shadowMapHeader"));
    ATHENEA_TRY(make(map.resolve_, "shadowMapResolve"));
    ATHENEA_TRY(make(map.factors_, "shadowMapFactors"));
    ATHENEA_TRY(make(map.clear_factors_, "shadowFactorsClear"));
    return map;
}

Result<void> SplatShadowMap::build(gpu::CommandBatch& batch, const ShadowMapJob& job) {
    valid_ = false;
    const uint32_t lights = std::min(job.lightCount, kSlots);
    if (lights == 0 || job.lights == nullptr || job.casters.empty()) {
        return ok();
    }
    // The lights' slots, then the dome's directions in what is left.
    const uint32_t domeSlots = std::min(job.domeSlots, kSlots - lights);
    const uint32_t slots = lights + domeSlots;
    const uint32_t resolution = std::max(job.resolution, 16u);
    // One, three, five or seven: a total, and Fourier pairs after it.
    const uint32_t coefficients = std::max(job.coefficients | 1u, 1u);
    // A texel keeps its coefficients and one word more, for the depth of the
    // nearest caster in it.
    const uint32_t stride = coefficients + 1;
    const uint64_t plane = uint64_t{resolution} * resolution * stride;
    const uint64_t words = kHeaderWords + plane * slots;
    if (!map_.valid() || map_.count() < words || resolution_ != resolution || coefficients_ != coefficients) {
        gpu::BufferDesc desc;
        desc.bytes = words * 4;
        desc.elementBytes = 4;
        desc.label = "splat.shadowMap";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        map_ = std::move(*made);
    }
    // And the same map as a texture, which is how a kernel with no binding
    // slot left reads it: a layer a coefficient a light, and one more at the
    // front for the frames.
    const uint32_t layers = slots * stride + 1;
    if (!texture_.valid() || texture_.width() != resolution || texture_.desc().arrayLength != layers) {
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2DArray;
        desc.width = resolution;
        desc.height = resolution;
        desc.arrayLength = layers;
        desc.format = rhi::Format::R32Float;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = "splat.shadowMap.texture";
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        texture_ = std::move(*made);
        auto made_view = texture_.view(0);
        if (!made_view) return std::move(made_view).error();
        view_ = std::move(*made_view);
    }
    resolution_ = resolution;
    coefficients_ = coefficients;
    lights_ = lights;

    // What every dispatch of this pass is told about the map itself.
    const auto setCommon = [&](rhi::ShaderCursor cursor, uint32_t light) {
        rhi::ShaderCursor p = cursor["shadowParams"];
        p["light"].setData(light);
        p["words"].setData(static_cast<uint32_t>(words));
        p["resolution"].setData(resolution);
        p["coefficients"].setData(coefficients);
        p["margin"].setData(job.margin);
        p["density"].setData(job.density);
        p["lightCount"].setData(lights);
        p["domeSlots"].setData(domeSlots);
        p["slots"].setData(slots);
        cursor["shadowMap"].setBinding(map_.rhi());
        cursor["shadowLights"].setBinding(job.lights->rhi());
    };
    const auto setCaster = [&](rhi::ShaderCursor cursor, const ShadowMapCaster& caster) {
        rhi::ShaderCursor p = cursor["shadowParams"];
        p["count"].setData(caster.cloud->count);
        p["boxMin"].setData(caster.cloud->bounds.min);
        p["boxMax"].setData(caster.cloud->bounds.max);
        for (int k = 0; k < 12; ++k) {
            p[kRow[k]].setData(caster.objectToWorld[static_cast<size_t>(k)]);
        }
        cursor["shadowPositions"].setBinding(caster.positions != nullptr ? caster.positions->rhi()
                                                                        : caster.cloud->positions.rhi());
        cursor["shadowShape"].setBinding(caster.cloud->shape.rhi());
    };

    clear_->dispatch(batch, {static_cast<uint32_t>(words), 1, 1}, [&](rhi::ShaderCursor cursor) {
        setCommon(cursor, 0);
        setCaster(cursor, job.casters.front());
    });
    // Where the casters stand, reduced on the device: eight corners a cloud.
    for (const ShadowMapCaster& caster : job.casters) {
        if (caster.cloud == nullptr || caster.cloud->count == 0) {
            continue;
        }
        box_->dispatch(batch, {8, 1, 1}, [&](rhi::ShaderCursor cursor) {
            setCommon(cursor, 0);
            setCaster(cursor, caster);
        });
    }
    frame_->dispatch(batch, {kSlots, 1, 1}, [&](rhi::ShaderCursor cursor) {
        setCommon(cursor, 0);
        setCaster(cursor, job.casters.front());
    });
    // And then the cloud, once a light. A light with no map of its own -- a
    // dome, one that casts no shadow -- leaves every thread at the first read.
    for (uint32_t k = 0; k < slots; ++k) {
        for (const ShadowMapCaster& caster : job.casters) {
            if (caster.cloud == nullptr || caster.cloud->count == 0) {
                continue;
            }
            splat_->dispatch(batch, {caster.cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                setCommon(cursor, k);
                setCaster(cursor, caster);
            });
        }
    }
    // Out of fixed point and into the texture, once everything has been added.
    header_->dispatch(batch, {kFrameWords * kSlots, 1, 1}, [&](rhi::ShaderCursor cursor) {
        setCommon(cursor, 0);
        setCaster(cursor, job.casters.front());
        cursor["shadowOut"].setBinding(view_.get());
    });
    resolve_->dispatch(batch, {resolution, resolution, slots * stride}, [&](rhi::ShaderCursor cursor) {
        setCommon(cursor, 0);
        setCaster(cursor, job.casters.front());
        cursor["shadowOut"].setBinding(view_.get());
    });
    valid_ = true;
    return ok();
}

Result<void> SplatShadowMap::factors(gpu::CommandBatch& batch, const ShadowMapCaster& caster, uint32_t base,
                                     const gpu::Buffer& into) {
    if (!valid_ || caster.cloud == nullptr || caster.cloud->count == 0) {
        return ok();
    }
    factors_->dispatch(batch, {caster.cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        rhi::ShaderCursor p = cursor["shadowParams"];
        p["count"].setData(caster.cloud->count);
        p["base"].setData(base);
        p["resolution"].setData(resolution_);
        p["coefficients"].setData(coefficients_);
        p["lightCount"].setData(lights_);
        p["boxMin"].setData(caster.cloud->bounds.min);
        p["boxMax"].setData(caster.cloud->bounds.max);
        for (int k = 0; k < 12; ++k) {
            p[kRow[k]].setData(caster.objectToWorld[static_cast<size_t>(k)]);
        }
        cursor["shadowMap"].setBinding(map_.rhi());
        cursor["shadowPositions"].setBinding(caster.positions != nullptr ? caster.positions->rhi()
                                                                        : caster.cloud->positions.rhi());
        cursor["shadowShape"].setBinding(caster.cloud->shape.rhi());
        cursor["shadowFactors"].setBinding(into.rhi());
    });
    return ok();
}

Result<void> SplatShadowMap::clearFactors(gpu::CommandBatch& batch, const gpu::Buffer& factors, uint32_t count) {
    if (count == 0 || !factors.valid()) {
        return ok();
    }
    clear_factors_->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["shadowParams"]["words"].setData(count);
        cursor["shadowFactors"].setBinding(factors.rhi());
    });
    return ok();
}

Result<double> SplatShadowMap::meanTransmittance(uint32_t light) {
    if (!valid_ || light >= lights_) {
        return 1.0;
    }
    const uint32_t zero = 0;
    auto total = gpu::Buffer::fromSpan<uint32_t>(*device_, std::span<const uint32_t>(&zero, 1), "shadow.mean");
    if (!total) return std::move(total).error();
    {
        gpu::CommandBatch batch(*device_);
        mean_->dispatch(batch, {resolution_ * resolution_, 1, 1}, [&](rhi::ShaderCursor cursor) {
            rhi::ShaderCursor p = cursor["shadowParams"];
            p["light"].setData(light);
            p["resolution"].setData(resolution_);
            p["coefficients"].setData(coefficients_);
            cursor["shadowMap"].setBinding(map_.rhi());
            cursor["shadowTotal"].setBinding(total->rhi());
        });
        ATHENEA_TRY(batch.submit(true));
    }
    uint32_t summed = 0;
    ATHENEA_TRY(total->read(*device_, 0, sizeof(summed), &summed));
    return double(summed) / 65536.0 / (double(resolution_) * double(resolution_));
}

Result<std::vector<float>> SplatShadowMap::probe(uint32_t light, std::span<const std::array<float, 4>> points) {
    std::vector<float> answers(points.size(), 1.0F);
    if (!valid_ || light >= lights_ || points.empty()) {
        return answers;
    }
    auto asked = gpu::Buffer::fromSpan<std::array<float, 4>>(*device_, points, "shadow.probe.points");
    if (!asked) return std::move(asked).error();
    auto out = gpu::Buffer::fromSpan<float>(*device_, std::span<const float>(answers), "shadow.probe.out");
    if (!out) return std::move(out).error();
    {
        gpu::CommandBatch batch(*device_);
        probe_->dispatch(batch, {static_cast<uint32_t>(points.size()), 1, 1}, [&](rhi::ShaderCursor cursor) {
            rhi::ShaderCursor p = cursor["shadowParams"];
            p["count"].setData(static_cast<uint32_t>(points.size()));
            p["light"].setData(light);
            p["resolution"].setData(resolution_);
            p["coefficients"].setData(coefficients_);
            cursor["shadowMap"].setBinding(map_.rhi());
            cursor["shadowProbePoints"].setBinding(asked->rhi());
            cursor["shadowProbeOut"].setBinding(out->rhi());
        });
        ATHENEA_TRY(batch.submit(true));
    }
    ATHENEA_TRY(out->read(*device_, 0, answers.size() * sizeof(float), answers.data()));
    return answers;
}

}   // namespace athenea::technique
