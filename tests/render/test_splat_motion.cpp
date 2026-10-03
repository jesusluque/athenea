// Copyright (c) 2026 jesus luque.
//
// Motion blur in the rasteriser: a splat smeared over the shutter against the
// truth, which is the same rasteriser run at several instants and averaged.
//
// The one thing under test is the rank-one term in splat_project.slang. The
// ground truth needs none of it, so a failure here is the term's and not the
// scaffolding's.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/RawSplats.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
using test::CloudBuilder;
using test::randomCloud;

namespace {

constexpr uint32_t kWidth = 250;
constexpr uint32_t kHeight = 190;

/// The object-space displacement a whole cloud shares, packed as the skinner
/// packs it: two words of halves a splat.
std::vector<uint32_t> constantMotion(uint32_t count, render::Vec3 step) {
    const auto half = [](float a, float b) {
        // The device's `packHalves`, on the host, for a buffer the host makes.
        const auto one = [](float v) {
            uint32_t bits = 0;
            std::memcpy(&bits, &v, 4);
            const uint32_t sign = (bits >> 16) & 0x8000u;
            int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
            uint32_t mantissa = (bits >> 13) & 0x3FFu;
            if (exponent <= 0) {
                return sign;
            }
            if (exponent >= 31) {
                return sign | 0x7C00u;
            }
            return sign | (static_cast<uint32_t>(exponent) << 10) | mantissa;
        };
        return one(a) | (one(b) << 16);
    };
    std::vector<uint32_t> words(size_t{count} * 2, 0);
    for (uint32_t k = 0; k < count; ++k) {
        words[size_t{k} * 2] = half(static_cast<float>(step.x), static_cast<float>(step.y));
        words[size_t{k} * 2 + 1] = half(static_cast<float>(step.z), 0.0F);
    }
    return words;
}

struct Harness {
    test::Gpu*             gpu;
    scene::CloudLoader     loader;
    render::TileRasterizer raster;
    gpu::ComputeKernel     accumulate;
    gpu::ComputeKernel     energy;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto accumulate = gpu::ComputeKernel::create(*gpu->library, "athenea/test/image_average", "imageAccumulate");
    auto energy = gpu::ComputeKernel::create(*gpu->library, "athenea/test/image_average", "imageEnergy");
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!accumulate) FAIL(accumulate.error().toString());
    if (!energy) FAIL(energy.error().toString());
    return std::unique_ptr<Harness>(new Harness{gpu, std::move(*loader), std::move(*raster),
                                                std::move(*accumulate), std::move(*energy)});
}

render::Mat4 translation(render::Vec3 t) {
    render::Mat4 m = render::Mat4::identity();
    m.at(0, 3) = t.x;
    m.at(1, 3) = t.y;
    m.at(2, 3) = t.z;
    return m;
}

/// Object to view, the difference over the shutter, for a cloud that only
/// translates: the linear part is zero and the translation is the step taken
/// into view space. That is what `SplatInstance::viewStep` holds.
std::array<float, 12> viewStepOf(const render::Projection& projection, render::Vec3 world) {
    const render::Vec3 v = projection.worldToView.direction(world);
    std::array<float, 12> step{};
    step[3] = static_cast<float>(v.x);
    step[7] = static_cast<float>(v.y);
    step[11] = static_cast<float>(v.z);
    return step;
}

/// The mean of the frames the cloud draws at `samples` instants across the
/// shutter: the blur, done the expensive way, with no motion machinery in it.
gpu::Buffer averaged(Harness& h, const render::Camera& camera, const render::RenderSettings& settings,
                     const scene::GpuSplats& cloud, render::Vec3 step, uint32_t samples) {
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{settings.width} * settings.height * 16;
    desc.elementBytes = 16;
    desc.label = "test.motion.mean";
    auto accum = gpu::Buffer::create(*h.gpu->device, desc);
    REQUIRE(accum);
    const std::vector<float> zeros(desc.bytes / 4, 0.0F);
    REQUIRE(accum->write(*h.gpu->device, 0, desc.bytes, zeros.data()));

    const uint32_t pixels = settings.width * settings.height;
    for (uint32_t k = 0; k < samples; ++k) {
        const double at = (static_cast<double>(k) + 0.5) / static_cast<double>(samples);
        render::SplatInstance instance;
        instance.splats = &cloud;
        instance.objectToWorld = translation({step.x * at, step.y * at, step.z * at});
        render::RenderTargets frame;
        auto stats = h.raster.render(camera, std::span<const render::SplatInstance>(&instance, 1),
                                     settings, frame);
        if (!stats) FAIL(stats.error().toString());
        gpu::CommandBatch batch(*h.gpu->device);
        h.accumulate.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["accum"].setBinding(accum->rhi());
            cursor["frame"].setBinding(frame.colour.rhi());
            cursor["total"].setBinding(accum->rhi());
            cursor["params"]["pixels"].setData(pixels);
            cursor["params"]["weight"].setData(1.0F / static_cast<float>(samples));
            cursor["params"]["scale"].setData(1.0F);
        });
        REQUIRE(batch.submit(true));
    }
    return std::move(*accum);
}

/// The image's premultiplied luminance, in fixed point, read back as one word.
uint64_t energyOf(Harness& h, const gpu::Buffer& colour, uint32_t pixels) {
    gpu::BufferDesc desc;
    desc.bytes = 4;
    desc.elementBytes = 4;
    desc.label = "test.motion.energy";
    auto total = gpu::Buffer::create(*h.gpu->device, desc);
    REQUIRE(total);
    const uint32_t zero = 0;
    REQUIRE(total->write(*h.gpu->device, 0, 4, &zero));
    gpu::CommandBatch batch(*h.gpu->device);
    h.energy.dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["accum"].setBinding(total->rhi());
        cursor["frame"].setBinding(colour.rhi());
        cursor["total"].setBinding(total->rhi());
        cursor["params"]["pixels"].setData(pixels);
        cursor["params"]["weight"].setData(0.0F);
        cursor["params"]["scale"].setData(1024.0F);
    });
    REQUIRE(batch.submit(true));
    uint32_t out = 0;
    REQUIRE(total->read(*h.gpu->device, 0, 4, &out));
    return out;
}

}   // namespace

TEST_CASE("a splat smeared over the shutter is the splat convolved with its path",
          "[render][gpu][motion]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(1500, 11, 0.02F, 0.25F, 0.3F, 0.95F);
    auto cloud = h->loader.upload(built.raw, 0);
    REQUIRE(cloud);

    render::RenderSettings settings;
    settings.width = kWidth;
    settings.height = kHeight;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    const render::Projection projection = render::projectionFor(camera, kWidth, kHeight);
    const uint32_t pixels = kWidth * kHeight;

    // The one-pass blur, as the engine will ask for it: the cloud at the
    // shutter's middle, told how far it travels over the whole of it.
    const auto blurred = [&](render::Vec3 step) {
        render::SplatInstance instance;
        instance.splats = &*cloud;
        instance.objectToWorld = translation({step.x * 0.5, step.y * 0.5, step.z * 0.5});
        instance.viewStep = viewStepOf(projection, step);
        render::RenderTargets out;
        auto stats = h->raster.render(camera, std::span<const render::SplatInstance>(&instance, 1),
                                      settings, out);
        if (!stats) FAIL(stats.error().toString());
        return out;
    };

    SECTION("a gentle blur is the mean of the frames it stands for, and closer the gentler it is") {
        // The approximation is a second-moment match, so it must improve as
        // the path shortens against the splat. Two steps, and the shorter one
        // must be nearer the truth -- that trend is the assertion, the
        // absolute numbers are only bounds on where it starts.
        uint32_t wider = 0;
        for (const double metres : {0.10, 0.03}) {
            const render::Vec3 step{metres, 0.0, 0.0};
            const gpu::Buffer truth = averaged(*h, camera, settings, *cloud, step, 48);
            const render::RenderTargets ours = blurred(step);
            auto diff = render::compareImages(*h->gpu->library, ours.colour, truth, kWidth, kHeight);
            if (!diff) FAIL(diff.error().toString());
            std::printf("  step %.2f: p99 %u, max %u, %llu pixels over 2 of %llu\n", metres,
                        diff->p99, diff->max, static_cast<unsigned long long>(diff->over2),
                        static_cast<unsigned long long>(diff->pixels));
            if (metres > 0.05) {
                wider = diff->p99;
            } else {
                CHECK(diff->p99 <= wider);
                CHECK(diff->p99 <= 3);
                CHECK(diff->max <= 24);
            }
        }
    }

    SECTION("a strong blur keeps the light, where it no longer keeps the shape") {
        // Far wider than a splat: the moment-matched gaussian is a peak where
        // the truth is a plateau, so the pixels differ by construction and
        // what the approximation still owes is the light itself.
        //
        // On a cloud sparse enough that splats rarely cover each other. Where
        // they do, the two answers part company for a reason that is not the
        // blur's: compositing is not linear in opacity, so the mean of N
        // composites is not the composite of N means, and a dense cloud reads
        // about 5% under. That is a property of blurring primitives rather
        // than images, and no rank-one term can mend it.
        CloudBuilder thin = randomCloud(80, 3, 0.05F, 0.12F, 0.2F, 0.6F);
        auto sparse = h->loader.upload(thin.raw, 0);
        REQUIRE(sparse);
        const render::Vec3 step{0.9, 0.0, 0.0};
        const gpu::Buffer truth = averaged(*h, camera, settings, *sparse, step, 128);
        render::SplatInstance moving;
        moving.splats = &*sparse;
        moving.objectToWorld = translation({step.x * 0.5, step.y * 0.5, step.z * 0.5});
        moving.viewStep = viewStepOf(projection, step);
        render::RenderTargets ours;
        auto drew = h->raster.render(camera, std::span<const render::SplatInstance>(&moving, 1),
                                     settings, ours);
        if (!drew) FAIL(drew.error().toString());
        const uint64_t mine = energyOf(*h, ours.colour, pixels);
        const uint64_t theirs = energyOf(*h, truth, pixels);
        const double ratio = static_cast<double>(mine) / static_cast<double>(theirs);
        std::printf("  strong: energy %llu against %llu, ratio %.4f\n",
                    static_cast<unsigned long long>(mine),
                    static_cast<unsigned long long>(theirs), ratio);

        // The same cloud barely moved, where the opacity the blur pays back is
        // still near one: there the compensation has nowhere to hide.
        const render::Vec3 small{0.03, 0.0, 0.0};
        const gpu::Buffer near = averaged(*h, camera, settings, *sparse, small, 48);
        render::SplatInstance gentle;
        gentle.splats = &*sparse;
        gentle.objectToWorld = translation({small.x * 0.5, small.y * 0.5, small.z * 0.5});
        gentle.viewStep = viewStepOf(projection, small);
        render::RenderTargets slight;
        auto drewSlight = h->raster.render(camera, std::span<const render::SplatInstance>(&gentle, 1),
                                           settings, slight);
        if (!drewSlight) FAIL(drewSlight.error().toString());
        const double gentleRatio = static_cast<double>(energyOf(*h, slight.colour, pixels)) /
                                   static_cast<double>(energyOf(*h, near, pixels));
        std::printf("  gentle: energy ratio %.4f\n", gentleRatio);

        // Barely moved, the light is all there: this is the assertion on the
        // square root that pays for the spreading.
        CHECK(gentleRatio > 0.99);
        CHECK(gentleRatio < 1.01);
        // Smeared far, it is not, and the reason is not the rank-one term: a
        // splat is drawn only out to `cutoffPower(alpha)`, which is a smaller
        // Mahalanobis radius the fainter it is, so a streak whose opacity the
        // blur took down to a few per cent loses its tails -- 1/(255*alpha) of
        // itself, by that rule. The blur can only lose light this way, never
        // make it, which is the upper bound below.
        CHECK(ratio > 0.88);
        CHECK(ratio < 1.01);
    }

    SECTION("the skeleton's displacement and the transform's are one term") {
        // The same translation said twice: once as the difference of two
        // object-to-view matrices, once as a displacement a splat carries.
        const render::Vec3 step{0.0, 0.05, 0.0};
        const render::RenderTargets byTransform = blurred(step);

        const std::vector<uint32_t> words = constantMotion(cloud->count, step);
        gpu::BufferDesc desc;
        desc.bytes = words.size() * 4;
        desc.elementBytes = 4;
        desc.label = "test.motion.words";
        auto motion = gpu::Buffer::create(*h->gpu->device, desc, words.data());
        REQUIRE(motion);
        render::SplatInstance instance;
        instance.splats = &*cloud;
        instance.objectToWorld = translation({step.x * 0.5, step.y * 0.5, step.z * 0.5});
        instance.motion = &*motion;
        instance.motionScale = 1.0F;
        render::RenderTargets bySplat;
        auto stats = h->raster.render(camera, std::span<const render::SplatInstance>(&instance, 1),
                                      settings, bySplat);
        if (!stats) FAIL(stats.error().toString());

        auto diff = render::compareImages(*h->gpu->library, byTransform.colour, bySplat.colour,
                                          kWidth, kHeight);
        if (!diff) FAIL(diff.error().toString());
        std::printf("  two sources: p99 %u, max %u\n", diff->p99, diff->max);
        CHECK(diff->p99 <= 1);
        CHECK(diff->max <= 2);
    }

    SECTION("a shutter of nothing draws the frame it drew before there was one") {
        render::SplatInstance still;
        still.splats = &*cloud;
        render::RenderTargets before;
        auto first = h->raster.render(camera, std::span<const render::SplatInstance>(&still, 1),
                                      settings, before);
        if (!first) FAIL(first.error().toString());

        // Every part of the machinery bound and pointed at, and a scale of
        // zero: the picture must not move by one code value.
        const std::vector<uint32_t> words = constantMotion(cloud->count, {0.5, 0.5, 0.0});
        gpu::BufferDesc desc;
        desc.bytes = words.size() * 4;
        desc.elementBytes = 4;
        desc.label = "test.motion.words";
        auto motion = gpu::Buffer::create(*h->gpu->device, desc, words.data());
        REQUIRE(motion);
        render::SplatInstance asked = still;
        asked.motion = &*motion;
        asked.motionScale = 0.0F;
        render::RenderTargets after;
        auto second = h->raster.render(camera, std::span<const render::SplatInstance>(&asked, 1),
                                       settings, after);
        if (!second) FAIL(second.error().toString());

        auto diff = render::compareImages(*h->gpu->library, before.colour, after.colour, kWidth,
                                          kHeight);
        if (!diff) FAIL(diff.error().toString());
        std::printf("  shutter zero: max %u\n", diff->max);
        CHECK(diff->max == 0);
    }
}
