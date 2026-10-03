// Copyright (c) 2026 jesus luque.
#include "athenea/render/TileRasterizer.h"

#include <algorithm>
#include <chrono>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "FrameParams.h"

namespace athenea::render {
namespace {

constexpr uint32_t kTile = 16;

Result<gpu::Buffer> buffer(gpu::Device& device, uint64_t count, uint32_t element,
                           const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    return gpu::Buffer::create(device, desc);
}

uint32_t grow(uint32_t have, uint32_t need) {
    uint64_t size = std::max<uint32_t>(have, 1024);
    while (size < need) {
        size = size * 3 / 2;
    }
    return static_cast<uint32_t>(std::min<uint64_t>(size, 0xFFFFFFFFull));
}

class Stopwatch {
public:
    Stopwatch(gpu::CommandBatch& batch, bool enabled)
        : batch_(batch), enabled_(enabled), start_(Clock::now()) {}
    /// Milliseconds since the last lap, waiting for the GPU first when enabled.
    Result<double> lap() {
        if (!enabled_) {
            return 0.0;
        }
        ATHENEA_TRY(batch_.submit(true));
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start_).count();
        start_ = now;
        return ms;
    }

private:
    using Clock = std::chrono::steady_clock;
    gpu::CommandBatch& batch_;
    bool               enabled_;
    Clock::time_point  start_;
};

}   // namespace

Result<TileRasterizer> TileRasterizer::create(gpu::ShaderLibrary& library) {
    TileRasterizer r;
    r.device_ = &library.device();
    auto prefix = gpu::PrefixSum::create(library);
    if (!prefix) return std::move(prefix).error();
    r.prefix_ = std::move(*prefix);
    auto sort = gpu::RadixSort::create(library);
    if (!sort) return std::move(sort).error();
    r.sort_ = std::move(*sort);
    const auto make = [&](gpu::ComputeKernel& into, const char* module, const char* entry) -> Result<void> {
        auto kernel = gpu::ComputeKernel::create(library, module, entry);
        if (!kernel) return std::move(kernel).error();
        into = std::move(*kernel);
        return ok();
    };
    ATHENEA_TRY(make(r.project_, "athenea/splat/splat_project", "splatProject"));
    ATHENEA_TRY(make(r.compact_, "athenea/splat/splat_compact", "splatCompact"));
    ATHENEA_TRY(make(r.pointsProject_, "athenea/splat/points_project", "pointsProject"));
    auto placeholderColour = buffer(*r.device_, 1, 16, "blend.noUnderColour");
    if (!placeholderColour) return std::move(placeholderColour).error();
    auto placeholderDepth = buffer(*r.device_, 1, 4, "blend.noUnderDepth");
    if (!placeholderDepth) return std::move(placeholderDepth).error();
    r.placeholderColour_ = *placeholderColour;
    r.placeholderDepth_ = *placeholderDepth;
    auto placeholderCrypto = buffer(*r.device_, 1, 4, "blend.noCrypto");
    if (!placeholderCrypto) return std::move(placeholderCrypto).error();
    r.placeholderCrypto_ = *placeholderCrypto;
    // One record of nothing, for frames that relight nothing: a name the
    // shader declares must be bound whether it is read or not.
    auto emptyLights = buffer(*r.device_, 1, 96, "splat.lights.empty");
    if (!emptyLights) return std::move(emptyLights).error();
    r.emptyLights_ = *emptyLights;
    auto emptyShadow = buffer(*r.device_, 1, 4, "splat.shadow.empty");
    if (!emptyShadow) return std::move(emptyShadow).error();
    r.emptyShadow_ = *emptyShadow;
    auto emptyEnvWords = buffer(*r.device_, 4, 4, "splat.env.empty");
    if (!emptyEnvWords) return std::move(emptyEnvWords).error();
    r.emptyEnvWords_ = *emptyEnvWords;
    auto emptyEnvSh = buffer(*r.device_, 1, 16, "splat.env.sh.empty");
    if (!emptyEnvSh) return std::move(emptyEnvSh).error();
    r.emptyEnvSh_ = *emptyEnvSh;
    // Never aliased onto another of the cloud's buffers: a stale flag would
    // read a splat's rotation as a displacement, and the picture would be
    // spectacular and hard to read.
    auto emptyMotion = buffer(*r.device_, 2, 4, "splat.motion.empty");
    if (!emptyMotion) return std::move(emptyMotion).error();
    r.emptyMotion_ = *emptyMotion;
    // A relit splat's shadow ray is traced inline, which not every device can.
    r.shadowsSupported_ = r.device_->caps().rayQuery && r.device_->caps().accelerationStructure;
    if (r.shadowsSupported_) {
        ATHENEA_TRY(make(r.splatShadow_, "athenea/splat/splat_shadow", "splatShadowFactors"));
    }
    ATHENEA_TRY(make(r.gather_, "athenea/splat/splat_gather_counts", "splatGatherCounts"));
    ATHENEA_TRY(make(r.emit_, "athenea/splat/splat_emit", "splatEmit"));
    ATHENEA_TRY(make(r.clear_, "athenea/splat/splat_tiles_clear", "splatTilesClear"));
    ATHENEA_TRY(make(r.ranges_, "athenea/splat/splat_ranges", "splatRanges"));
    ATHENEA_TRY(make(r.blend_, "athenea/splat/splat_blend", "splatBlend"));
    ATHENEA_TRY(make(r.blendComposite_, "athenea/splat/splat_blend", "splatBlendComposite"));
    ATHENEA_TRY(make(r.blendCrypto_, "athenea/splat/splat_blend", "splatBlendCrypto"));
    ATHENEA_TRY(make(r.blendCompositeCrypto_, "athenea/splat/splat_blend", "splatBlendCompositeCrypto"));
    return r;
}

Result<void> TileRasterizer::reserveSplats(uint32_t count) {
    if (count <= splatCapacity_) {
        return ok();
    }
    const uint32_t n = grow(splatCapacity_, count);
    const auto assign = [&](gpu::Buffer& into, uint64_t elements, uint32_t bytes,
                            const char* label) -> Result<void> {
        auto made = buffer(*device_, elements, bytes, label);
        if (!made) return std::move(made).error();
        into = *made;
        return ok();
    };
    ATHENEA_TRY(assign(proj_, n, 48, "splat.proj"));
    ATHENEA_TRY(assign(cryptoIds_, n, 4, "splat.cryptoIds"));
    ATHENEA_TRY(assign(tileRects_, uint64_t{n} * 4, 4, "splat.tileRects"));
    ATHENEA_TRY(assign(tilesTouched_, n, 4, "splat.tilesTouched"));
    ATHENEA_TRY(assign(visible_, n, 4, "splat.visible"));
    ATHENEA_TRY(assign(depthKeys_, n, 4, "splat.depthKeysAll"));
    ATHENEA_TRY(assign(visibleOffsets_, n, 4, "splat.visibleOffsets"));
    ATHENEA_TRY(assign(visibleTotal_, 1, 4, "splat.visibleTotal"));
    ATHENEA_TRY(assign(touchedOffsets_, n, 4, "splat.touchedOffsets"));
    ATHENEA_TRY(assign(touchedTotal_, 1, 4, "splat.touchedTotal"));
    ATHENEA_TRY(assign(depthSort_.keysLo, n, 4, "splat.depthKeys"));
    ATHENEA_TRY(assign(depthSort_.values, n, 4, "splat.order"));
    ATHENEA_TRY(assign(depthSort_.scratchKeysLo, n, 4, "splat.depthKeys2"));
    ATHENEA_TRY(assign(depthSort_.scratchValues, n, 4, "splat.order2"));
    ATHENEA_TRY(assign(sortedCounts_, n, 4, "splat.sortedCounts"));
    ATHENEA_TRY(assign(offsets_, n, 4, "splat.offsets"));
    ATHENEA_TRY(assign(totalPairs_, 1, 4, "splat.totalPairs"));
    splatCapacity_ = n;
    return ok();
}

Result<void> TileRasterizer::reservePairs(uint32_t count) {
    if (count <= pairCapacity_) {
        return ok();
    }
    const uint32_t n = grow(pairCapacity_, count);
    auto keys = buffer(*device_, n, 4, "pair.tiles");
    if (!keys) return std::move(keys).error();
    auto values = buffer(*device_, n, 4, "pair.splats");
    if (!values) return std::move(values).error();
    auto keys2 = buffer(*device_, n, 4, "pair.tiles2");
    if (!keys2) return std::move(keys2).error();
    auto values2 = buffer(*device_, n, 4, "pair.splats2");
    if (!values2) return std::move(values2).error();
    tileSort_.keysLo = *keys;
    tileSort_.values = *values;
    tileSort_.scratchKeysLo = *keys2;
    tileSort_.scratchValues = *values2;
    pairCapacity_ = n;
    return ok();
}

Result<void> TileRasterizer::reserveTargets(RenderTargets& targets, uint32_t width,
                                            uint32_t height, uint32_t tiles, bool crypto) {
    if (tiles > tileCapacity_) {
        auto made = buffer(*device_, uint64_t{tiles} * 2, 4, "tile.ranges");
        if (!made) return std::move(made).error();
        ranges_buffer_ = *made;
        tileCapacity_ = tiles;
    }
    if (targets.width != width || targets.height != height || !targets.colour.valid()) {
        auto colour = buffer(*device_, uint64_t{width} * height, 16, "target.colour");
        if (!colour) return std::move(colour).error();
        auto depth = buffer(*device_, uint64_t{width} * height, 4, "target.depth");
        if (!depth) return std::move(depth).error();
        targets.colour = *colour;
        targets.depth = *depth;
        targets.width = width;
        targets.height = height;
        targets.crypto = {};
    }
    // Three float4 a pixel, allocated only where a frame asks for a matte:
    // it is a quarter again of the frame's memory, and most frames want none.
    if (crypto && !targets.crypto.valid()) {
        auto made = buffer(*device_, uint64_t{width} * height * 3, 16, "target.crypto");
        if (!made) return std::move(made).error();
        targets.crypto = *made;
    }
    return ok();
}

Result<FrameStats> TileRasterizer::render(const Camera& camera,
                                          std::span<const SplatInstance> instances,
                                          const RenderSettings& settings, RenderTargets& targets,
                                          std::span<const PointInstance> points,
                                          const RenderTargets* under, const SplatLights* lights) {
    return render(projectionFor(camera, settings.width, settings.height), instances, settings,
                  targets, points, under, lights);
}

Result<FrameStats> TileRasterizer::render(const Projection& projection,
                                          std::span<const SplatInstance> instances,
                                          const RenderSettings& settings, RenderTargets& targets,
                                          std::span<const PointInstance> points,
                                          const RenderTargets* under, const SplatLights* lights) {
    const auto frameStart = std::chrono::steady_clock::now();
    if (settings.width == 0 || settings.height == 0) {
        return Error(ErrorCode::InvalidArgument, "a frame needs a size");
    }
    FrameCommon common{settings.width, settings.height, (settings.width + kTile - 1) / kTile,
                       (settings.height + kTile - 1) / kTile, &projection, &settings};
    const uint32_t tiles = common.tilesX * common.tilesY;

    uint64_t total = 0;
    for (const SplatInstance& instance : instances) {
        if (instance.splats != nullptr) {
            total += instance.splats->count;
        }
    }
    for (const PointInstance& instance : points) {
        if (instance.points != nullptr) {
            total += instance.points->count;
        }
    }
    if (under != nullptr && (under->width != settings.width || under->height != settings.height)) {
        return Error(ErrorCode::InvalidArgument, "the under-layer is not the frame's size");
    }
    if (total > 0xFFFFFFFFull) {
        return Error(ErrorCode::Unsupported, "more than 2^32 splats in one frame");
    }
    const uint32_t splatCount = static_cast<uint32_t>(total);
    ATHENEA_TRY(reserveSplats(std::max(splatCount, 1u)));
    ATHENEA_TRY(reserveTargets(targets, settings.width, settings.height, tiles, settings.cryptomatte));

    FrameStats stats;
    stats.splats = splatCount;

    // --- batch 1: project; how many are visible and how many pairs --------
    //
    // Both counts come from prefix sums over what the projection wrote, so one
    // readback answers both, and everything after it is sized exactly.
    gpu::CommandBatch batch(*device_);
    // A relit cloud's shadows, measured before anything is projected: one ray
    // a splat a light (splat_shadow.slang), read by the projection as a
    // factor. Its own kernel, since an inline ray is a feature a device may
    // not have and the projection must stay available everywhere.
    // Factors measured before the frame stand in for the shadow ray: a baked
    // visibility answers the same question without a trace, and answers it
    // on a device that cannot trace inline at all.
    const bool measured = lights != nullptr && lights->any() && lights->visibilityFactors != nullptr &&
                          lights->visibilityLights > 0;
    const bool splatShadows = !measured && lights != nullptr && lights->any() && lights->shadows() && shadowsSupported_ &&
                              std::any_of(instances.begin(), instances.end(),
                                          [](const SplatInstance& i) { return i.relight; });
    const uint32_t shadowLights = splatShadows ? std::min(lights->count, uint32_t{8}) : 0u;
    if (splatShadows) {
        uint64_t wanted = 0;
        for (const SplatInstance& instance : instances) {
            wanted += instance.splats != nullptr ? instance.splats->count : 0;
        }
        wanted *= shadowLights;
        if (!shadowFactors_.valid() || shadowFactors_.bytes() < wanted * sizeof(float)) {
            auto made = buffer(*device_, std::max<uint64_t>(wanted, 1), 4, "splat.shadowFactors");
            if (!made) return std::move(made).error();
            shadowFactors_ = *made;
        }
        uint32_t at = 0;
        for (const SplatInstance& instance : instances) {
            const scene::GpuSplats* cloud = instance.splats;
            if (cloud == nullptr || cloud->count == 0) {
                continue;
            }
            if (instance.relight) {
                const std::array<float, 12> rows = instance.objectToWorld.rows3x4();
                splatShadow_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                    cursor["positions"].setBinding(cloud->positions.rhi());
                    cursor["shape"].setBinding(cloud->shape.rhi());
                    cursor["lights"].setBinding(lights->records->rhi());
                    cursor["shadowFrames"].setBinding(lights->shadowFrames->rhi());
                    cursor["shadowColours"].setBinding(lights->shadowColours->rhi());
                    cursor["shadowInstanceData"].setBinding(lights->shadowInstanceData->rhi());
                    cursor["shadowInstanceIndices"].setBinding(lights->shadowInstanceIndices->rhi());
                    cursor["factors"].setBinding(shadowFactors_.rhi());
                    cursor["scene"].setBinding(lights->shadowTlas);
                    rhi::ShaderCursor p = cursor["params"];
                    p["count"].setData(cloud->count);
                    p["base"].setData(at);
                    p["lights"].setData(shadowLights);
                    p["offset"].setData(lights->shadowOffset);
                    p["cut"].setData(lights->shadowCut);
                    p["categoriesLo"].setData(static_cast<uint32_t>(instance.categories & 0xFFFFFFFFu));
                    p["categoriesHi"].setData(static_cast<uint32_t>(instance.categories >> 32));
                    static const char* kRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                                   "w12", "w13", "w20", "w21", "w22", "w23"};
                    for (int k = 0; k < 12; ++k) {
                        p[kRow[k]].setData(rows[static_cast<size_t>(k)]);
                    }
                });
            }
            at += cloud->count;
        }
    }
    Stopwatch watch(batch, settings.timeStages);
    uint32_t base = 0;
    for (const SplatInstance& instance : instances) {
        const scene::GpuSplats* cloud = instance.splats;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const Mat4 objectToView = projection.worldToView * instance.objectToWorld;
        const Vec3 eyeObject =
            aofx::xform::inverseAffine(instance.objectToWorld).point(projection.eyeWorld);
        project_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["shape"].setBinding(cloud->shape.rhi());
            cursor["sh"].setBinding(cloud->sh.rhi());
            cursor["proj"].setBinding(proj_.rhi());
            cursor["visible"].setBinding(visible_.rhi());
            cursor["tileRects"].setBinding(tileRects_.rhi());
            cursor["tilesTouched"].setBinding(tilesTouched_.rhi());
            cursor["depthKeys"].setBinding(depthKeys_.rhi());
            // Cryptomatte: this cloud's ids, where it has them. A cloud
            // without ancestry (a capture) writes zero, which the blend
            // leaves out of the matte rather than calling it something.
            const uint32_t cryptoMode =
                !settings.cryptomatte ? 0u : cloud->hasCrypto() ? 1u : 2u;
            cursor["cloudCrypto"].setBinding(cloud->hasCrypto() ? cloud->crypto.rhi()
                                                                : placeholderCrypto_.rhi());
            cursor["cryptoIds"].setBinding(cryptoIds_.rhi());
            setFrame(cursor, common);
            cursor["params"]["cryptoMode"].setData(cryptoMode);
            setObject(cursor, objectToView, eyeObject);
            cursor["params"]["count"].setData(cloud->count);
            cursor["params"]["base"].setData(base);
            cursor["params"]["restPerColour"].setData(cloud->restPerColour);
            cursor["params"]["shWords"].setData(cloud->shWords);
            setEdit(cursor["params"]["edit"], instance.edit);
            // Relighting: only where the prim asked for it and the frame has
            // lights to do it with. The buffer is bound either way, since a
            // name a shader declares must be bound whether it is read or not.
            const bool relight = instance.relight && lights != nullptr && lights->any();
            cursor["params"]["relight"].setData(uint32_t{relight ? 1u : 0u});
            cursor["params"]["lightCount"].setData(relight ? lights->count : 0u);
            cursor["params"]["categoriesLo"].setData(static_cast<uint32_t>(instance.categories & 0xFFFFFFFFu));
            cursor["params"]["categoriesHi"].setData(static_cast<uint32_t>(instance.categories >> 32));
            cursor["lights"].setBinding(relight ? lights->records->rhi() : emptyLights_.rhi());
            // The shadow ray a relit splat casts against the cloud's own
            // proxies. Bound either way; taken only where there is a
            // structure to trace and the device can trace it inline.
            const bool shadows = relight && splatShadows && instance.relight;
            const bool fromField = relight && measured && instance.relight;
            cursor["params"]["shadowRays"].setData(uint32_t{(shadows || fromField) ? 1u : 0u});
            cursor["params"]["shadowLights"].setData(fromField ? lights->visibilityLights : shadowLights);
            cursor["shadowFactors"].setBinding(fromField  ? lights->visibilityFactors->rhi()
                                               : shadows ? shadowFactors_.rhi()
                                                         : emptyShadow_.rhi());
            // The sky, where one was prepared for this frame. Bound either
            // way: a name a shader declares must be bound whether it is read
            // or not, and `envLights` of 0 is what says it is not.
            const bool sky = relight && lights != nullptr && lights->environment();
            cursor["envTexels"].setBinding(sky ? lights->envTexels->rhi() : emptyEnvWords_.rhi());
            cursor["envSh"].setBinding(sky ? lights->envSh->rhi() : emptyEnvSh_.rhi());
            cursor["envSun"].setBinding(sky ? lights->envSun->rhi() : emptyEnvSh_.rhi());
            cursor["envOfLight"].setBinding(sky ? lights->envOfLight->rhi() : emptyEnvWords_.rhi());
            cursor["params"]["envLights"].setData(sky ? lights->envLights : 0u);
            cursor["params"]["envBaseSide"].setData(sky ? lights->envBaseSide : 1u);
            // What of an environment reaches each gaussian, where the cloud
            // carries it. Bound either way, as every name a shader declares.
            const bool carried = cloud->hasTransfer();
            cursor["transfer"].setBinding(carried ? cloud->transfer.rhi() : cloud->shape.rhi());
            cursor["params"]["transferWords"].setData(carried ? cloud->transferWords : 0u);
            cursor["params"]["transferCount"].setData(carried ? cloud->transferCount : 0u);
            const bool open = carried && cloud->hasShadowBits();
            cursor["shadowBits"].setBinding(open ? cloud->shadowBits.rhi() : cloud->shape.rhi());
            cursor["params"]["shadowBits"].setData(open ? 1u : 0u);
            cursor["params"]["transferIndirect"].setData(
                uint32_t{carried && instance.transferIndirect && cloud->hasIndirect() ? 1u : 0u});
            cursor["params"]["ior"].setData(instance.ior);
            // What the splat reflects with, where its cloud carries it (a
            // conversion from a mesh does, a capture does not). Bound either
            // way, and `hasPbr` is what says whether it is read.
            cursor["params"]["hasPbr"].setData(uint32_t{cloud->hasPbr() ? 1u : 0u});
            cursor["params"]["litBody"].setData(uint32_t{instance.litBody ? 1u : 0u});
            cursor["pbr"].setBinding(cloud->hasPbr() ? cloud->pbr.rhi() : cloud->shape.rhi());
            // What a pick said this prim is made of. Bound either way, as
            // every name a shader declares; `overrideCount` of 0 is what says
            // the table is not read.
            const bool said = instance.overrides != nullptr && instance.overrideCount > 0 &&
                              cloud->hasCrypto();
            cursor["splatOverrides"].setBinding(said ? instance.overrides->rhi() : emptyEnvSh_.rhi());
            cursor["params"]["overrideCount"].setData(said ? instance.overrideCount : 0u);
            // Relighting happens in the world: the rows take this cloud there
            // and the eye is already there.
            const std::array<float, 12> toWorld = instance.objectToWorld.rows3x4();
            static const char* kWorldRow[12] = {"w00", "w01", "w02", "w03", "w10", "w11",
                                                "w12", "w13", "w20", "w21", "w22", "w23"};
            for (int k = 0; k < 12; ++k) {
                cursor["params"][kWorldRow[k]].setData(toWorld[static_cast<size_t>(k)]);
            }
            cursor["params"]["eyeWorldX"].setData(static_cast<float>(projection.eyeWorld.x));
            cursor["params"]["eyeWorldY"].setData(static_cast<float>(projection.eyeWorld.y));
            cursor["params"]["eyeWorldZ"].setData(static_cast<float>(projection.eyeWorld.z));
            // What moves during the shutter. Set for every instance, still or
            // not, so that no dispatch inherits the last one's flag.
            setMotion(cursor, instance,
                      static_cast<float>(std::max(common.width, common.height)) * 0.25F);
            cursor["motion"].setBinding(instance.motion != nullptr ? instance.motion->rhi()
                                                                   : emptyMotion_.rhi());
        });
        base += cloud->count;
    }
    for (const PointInstance& instance : points) {
        const scene::GpuPoints* cloud = instance.points;
        if (cloud == nullptr || cloud->count == 0) {
            continue;
        }
        const Mat4 objectToView = projection.worldToView * instance.objectToWorld;
        pointsProject_.dispatch(batch, {cloud->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(cloud->positions.rhi());
            cursor["colours"].setBinding(cloud->colours.rhi());
            cursor["proj"].setBinding(proj_.rhi());
            cursor["visible"].setBinding(visible_.rhi());
            cursor["tileRects"].setBinding(tileRects_.rhi());
            cursor["tilesTouched"].setBinding(tilesTouched_.rhi());
            cursor["depthKeys"].setBinding(depthKeys_.rhi());
            cursor["cryptoIds"].setBinding(cryptoIds_.rhi());
            setFrame(cursor, common);
            cursor["params"]["cryptoMode"].setData(uint32_t{settings.cryptomatte ? 2u : 0u});
            setObject(cursor, objectToView, projection.eyeWorld);
            cursor["params"]["count"].setData(cloud->count);
            cursor["params"]["base"].setData(base);
            setPointParams(cursor["points"], projection, settings.width, settings.height,
                           objectToView, instance.style, cloud->count);
        });
        base += cloud->count;
    }
    auto ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.projectMs = *ms;

    const uint32_t all = std::max(splatCount, 1u);
    ATHENEA_TRY(prefix_.apply(batch, visible_, visibleOffsets_, visibleTotal_, all));
    ATHENEA_TRY(prefix_.apply(batch, tilesTouched_, touchedOffsets_, touchedTotal_, all));
    ATHENEA_TRY(batch.submit(true));
    uint32_t visibleCount = 0;
    uint32_t pairs = 0;
    if (splatCount > 0) {
        ATHENEA_TRY(visibleTotal_.read(*device_, 0, sizeof(visibleCount), &visibleCount));
        ATHENEA_TRY(touchedTotal_.read(*device_, 0, sizeof(pairs), &pairs));
    }
    ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.countsMs = *ms;
    stats.visible = visibleCount;
    stats.pairs = pairs;
    ATHENEA_TRY(reservePairs(std::max(pairs, 1u)));

    // --- batch 2: sort the visible, emit, tile sort, ranges, blend ---------
    if (visibleCount > 0) {
        compact_.dispatch(batch, {splatCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["visible"].setBinding(visible_.rhi());
            cursor["visibleOffsets"].setBinding(visibleOffsets_.rhi());
            cursor["depthKeys"].setBinding(depthKeys_.rhi());
            cursor["sortKeys"].setBinding(depthSort_.keysLo.rhi());
            cursor["sortOrder"].setBinding(depthSort_.values.rhi());
            setFrame(cursor, common);
            cursor["params"]["count"].setData(splatCount);
        });
        if (visibleCount > 1) {
            ATHENEA_TRY(sort_.sort(batch, depthSort_, visibleCount, 24));
        }
    }
    ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.depthSortMs = *ms;

    if (pairs > 0) {
        gather_.dispatch(batch, {visibleCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["order"].setBinding(depthSort_.values.rhi());
            cursor["tilesTouched"].setBinding(tilesTouched_.rhi());
            cursor["sortedCounts"].setBinding(sortedCounts_.rhi());
            setFrame(cursor, common);
            cursor["params"]["count"].setData(visibleCount);
        });
        ATHENEA_TRY(prefix_.apply(batch, sortedCounts_, offsets_, totalPairs_, visibleCount));
        emit_.dispatch(batch, {visibleCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["order"].setBinding(depthSort_.values.rhi());
            cursor["offsets"].setBinding(offsets_.rhi());
            cursor["tilesTouched"].setBinding(tilesTouched_.rhi());
            cursor["tileRects"].setBinding(tileRects_.rhi());
            cursor["proj"].setBinding(proj_.rhi());
            cursor["pairTiles"].setBinding(tileSort_.keysLo.rhi());
            cursor["pairSplats"].setBinding(tileSort_.values.rhi());
            setFrame(cursor, common);
            cursor["params"]["count"].setData(visibleCount);
            cursor["params"]["base"].setData(pairCapacity_);
        });
    }
    ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.emitMs = *ms;

    if (pairs > 1) {
        ATHENEA_TRY(sort_.sort(batch, tileSort_, pairs, bitsFor(tiles)));
    }
    clear_.dispatch(batch, {tiles, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["ranges"].setBinding(ranges_buffer_.rhi());
        setFrame(cursor, common);
    });
    if (pairs > 0) {
        ranges_.dispatch(batch, {pairs, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["pairTiles"].setBinding(tileSort_.keysLo.rhi());
            cursor["ranges"].setBinding(ranges_buffer_.rhi());
            setFrame(cursor, common);
            cursor["params"]["count"].setData(pairs);
        });

    }
    ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.tileSortMs = *ms;

    const bool composite = under != nullptr || !points.empty();
    const bool matte = settings.cryptomatte && targets.crypto.valid();
    const bool underCrypto = matte && under != nullptr && under->cryptoIds.valid();
    gpu::ComputeKernel& blend = matte ? (composite ? blendCompositeCrypto_ : blendCrypto_)
                                      : (composite ? blendComposite_ : blend_);
    blend.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["ranges"].setBinding(ranges_buffer_.rhi());
        cursor["pairSplats"].setBinding(tileSort_.values.rhi());
        cursor["proj"].setBinding(proj_.rhi());
        cursor["underColour"].setBinding(under != nullptr ? under->colour.rhi() : placeholderColour_.rhi());
        cursor["underDepth"].setBinding(under != nullptr ? under->depth.rhi() : placeholderDepth_.rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["depth"].setBinding(targets.depth.rhi());
        cursor["cryptoIds"].setBinding(cryptoIds_.rhi());
        cursor["underCrypto"].setBinding(underCrypto ? under->cryptoIds.rhi()
                                                     : placeholderCrypto_.rhi());
        cursor["crypto"].setBinding(matte ? targets.crypto.rhi() : placeholderColour_.rhi());
        setFrame(cursor, common);
        cursor["params"]["hasUnder"].setData(uint32_t{under != nullptr ? 1u : 0u});
        cursor["params"]["hasUnderCrypto"].setData(uint32_t{underCrypto ? 1u : 0u});
    });
    ATHENEA_TRY(batch.submit(true));
    ms = watch.lap();
    if (!ms) return std::move(ms).error();
    stats.blendMs = *ms;
    stats.totalMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              frameStart)
                        .count();
    return stats;
}

}   // namespace athenea::render
