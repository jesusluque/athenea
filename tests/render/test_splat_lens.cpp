// Copyright (c) 2026 jesus luque.
//
// DEPTH OF FIELD IN THE RASTERISER: the splat convolved with the disk its
// out-of-focus point spreads into, against the truth, which is the same
// rasteriser run from several points of the lens and averaged.
//
// The truth is built the way a path tracer builds it -- the eye moved across
// the diaphragm, the picture window shifted so the focus plane stays where it
// is -- and uses none of the machinery under test, so a failure here belongs
// to the rank-one term in splat_project.slang.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
using test::CloudBuilder;

namespace {

constexpr uint32_t kWidth = 250;
constexpr uint32_t kHeight = 190;

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

/// The mean of the frames the cloud draws from `samples` points of the lens,
/// spread over its disk: depth of field, done the expensive way.
///
/// Moving the eye by o shifts everything by -o in the picture, so the window
/// is moved by focal * o / focus to put the focus plane back where it was --
/// which is exactly the shear a lens is, and why what stands at `focus` stays
/// sharp however wide the diaphragm opens.
gpu::Buffer averaged(Harness& h, const render::Camera& camera, const render::RenderSettings& settings,
                     const scene::GpuSplats& cloud, double radius, double focus, uint32_t samples) {
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{settings.width} * settings.height * 16;
    desc.elementBytes = 16;
    desc.label = "test.lens.mean";
    auto accum = gpu::Buffer::create(*h.gpu->device, desc);
    REQUIRE(accum);
    const std::vector<float> zeros(desc.bytes / 4, 0.0F);
    REQUIRE(accum->write(*h.gpu->device, 0, desc.bytes, zeros.data()));

    const render::Projection pinhole = render::projectionFor(camera, settings.width, settings.height);
    const uint32_t pixels = settings.width * settings.height;
    // A spiral of points over the disk, area-uniform: the golden angle keeps
    // them from lining up with anything in the picture.
    for (uint32_t k = 0; k < samples; ++k) {
        const double t = (static_cast<double>(k) + 0.5) / static_cast<double>(samples);
        const double r = radius * std::sqrt(t);
        const double phi = 2.39996322972865332 * static_cast<double>(k);
        const double ox = r * std::cos(phi);
        const double oy = r * std::sin(phi);
        render::Camera shifted = camera;
        // The lens is in the camera's own plane: its world x and y axes.
        const render::Vec3 right{camera.cameraToWorld.at(0, 0), camera.cameraToWorld.at(1, 0),
                                 camera.cameraToWorld.at(2, 0)};
        const render::Vec3 up{camera.cameraToWorld.at(0, 1), camera.cameraToWorld.at(1, 1),
                              camera.cameraToWorld.at(2, 1)};
        shifted.cameraToWorld.at(0, 3) += right.x * ox + up.x * oy;
        shifted.cameraToWorld.at(1, 3) += right.y * ox + up.y * oy;
        shifted.cameraToWorld.at(2, 3) += right.z * ox + up.z * oy;
        // And the window, in half-widths, as projectionFor reads it.
        const double half = 0.5 * static_cast<double>(settings.width);
        // The view keeps the camera's x and y (only z is flipped), so moving
        // the eye by +o takes the picture by -focal*o/z, and the window has to
        // give back +focal*o/focus for the focus plane to stand still.
        shifted.lens.windowTranslate[0] = (pinhole.focalX * ox / focus) / half;
        shifted.lens.windowTranslate[1] = (pinhole.focalY * oy / focus) / half;
        render::SplatInstance instance;
        instance.splats = &cloud;
        render::RenderTargets frame;
        auto stats = h.raster.render(shifted, std::span<const render::SplatInstance>(&instance, 1), settings,
                                     frame);
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

uint64_t energyOf(Harness& h, const gpu::Buffer& colour, uint32_t pixels) {
    gpu::BufferDesc desc;
    desc.bytes = 4;
    desc.elementBytes = 4;
    desc.label = "test.lens.energy";
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

TEST_CASE("a splat out of focus is the splat convolved with the lens disk", "[render][gpu][lens]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // Two sheets of particles, one at the focus plane and one well behind it,
    // so a frame holds both what must stay sharp and what must spread.
    CloudBuilder built;
    for (int i = -3; i <= 3; ++i) {
        for (int j = -2; j <= 2; ++j) {
            built.add(float(i) * 0.5F, float(j) * 0.5F, 0.0F, 0.9F, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0},
                      {0.9F, 0.8F, 0.7F});
        }
    }
    auto sheet = h->loader.upload(built.raw, 0);
    REQUIRE(sheet);

    render::RenderSettings settings;
    settings.width = kWidth;
    settings.height = kHeight;
    settings.antialias = false;   // the lens pays its own energy back; keep the two apart
    const double focus = 6.0;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, focus}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    camera.lens.focusDistance = focus;
    const uint32_t pixels = kWidth * kHeight;

    const auto draw = [&](double fStop) {
        render::Camera lens = camera;
        lens.lens.fStop = fStop;
        render::SplatInstance instance;
        instance.splats = &*sheet;
        render::RenderTargets out;
        auto stats = h->raster.render(lens, std::span<const render::SplatInstance>(&instance, 1), settings, out);
        if (!stats) FAIL(stats.error().toString());
        return out;
    };

    SECTION("what stands at the focus distance is drawn as a pinhole draws it, bit for bit") {
        render::RenderTargets pin = draw(0.0);
        render::RenderTargets wide = draw(1.4);
        auto same = render::compareImages(*gpu->library, pin.colour, wide.colour, kWidth, kHeight);
        if (!same) FAIL(same.error().toString());
        std::printf("  in focus, f/1.4 against the pinhole: max %u, p99 %u\n", same->max, same->p99);
        CHECK(same->max == 0);
    }

    SECTION("behind the focus plane it is the mean of the frames the lens stands for, and the narrower the "
            "diaphragm the nearer") {
        // A disk is a plateau with a hard rim and a moment-matched gaussian is
        // a peak: the two agree in energy, centroid and second moment and not
        // in shape, so the error is a function of how wide the circle of
        // confusion is against the splat. It must therefore fall as the
        // diaphragm closes, and the one pass must beat not blurring at all
        // however far open it is.
        const double behind = 3.0;
        render::Camera far = camera;
        far.cameraToWorld.at(2, 3) = focus + behind;   // the camera backs off; the focus plane does not move
        render::SplatInstance instance;
        instance.splats = &*sheet;
        render::RenderTargets sharp;
        auto sharpStats = h->raster.render(far, std::span<const render::SplatInstance>(&instance, 1), settings,
                                           sharp);
        if (!sharpStats) FAIL(sharpStats.error().toString());
        uint32_t worse = 0;
        uint32_t lastP99 = 255;
        for (const double fStop : {2.0, 5.6, 11.0}) {
            const double radius = far.lens.focal * 0.1 / (2.0 * fStop);
            render::Camera lens = far;
            lens.lens.fStop = fStop;
            render::RenderTargets one;
            auto stats = h->raster.render(lens, std::span<const render::SplatInstance>(&instance, 1), settings,
                                          one);
            if (!stats) FAIL(stats.error().toString());
            const gpu::Buffer truth = averaged(*h, far, settings, *sheet, radius, focus, 64);
            auto against = render::compareImages(*gpu->library, one.colour, truth, kWidth, kHeight);
            auto moved = render::compareImages(*gpu->library, sharp.colour, truth, kWidth, kHeight);
            if (!against) FAIL(against.error().toString());
            if (!moved) FAIL(moved.error().toString());
            // The circle of confusion this aperture draws, in pixels: what the
            // covariance was widened by, and what the shape error scales with.
            const render::Projection p = render::projectionFor(far, kWidth, kHeight);
            const double coc = p.focalX * radius * std::abs(1.0 / (focus + behind) - 1.0 / focus);
            std::printf("  f/%-4.1f circle of confusion %5.1f px: one pass against 64 samples p99 %3u max %3u; "
                        "not blurring at all p99 %3u\n",
                        fStop, coc, against->p99, against->max, moved->p99);
            CHECK(against->p99 < moved->p99);
            if (against->p99 > lastP99) {
                ++worse;
            }
            lastP99 = against->p99;
        }
        // Monotone: every closing of the diaphragm brought it nearer.
        CHECK(worse == 0);
        // It stops improving around nine: below a couple of pixels of
        // confusion the difference is the 0.3 of dilation and the footprint's
        // cutoff, neither of which is the lens.
        CHECK(lastP99 <= 10);
    }

    SECTION("opening the lens moves light about and does not make any") {
        render::RenderTargets pin = draw(0.0);
        render::Camera far = camera;
        far.cameraToWorld.at(2, 3) = focus + 3.0;
        const auto energyAt = [&](double fStop) {
            render::Camera lens = far;
            lens.lens.fStop = fStop;
            render::SplatInstance instance;
            instance.splats = &*sheet;
            render::RenderTargets out;
            auto stats = h->raster.render(lens, std::span<const render::SplatInstance>(&instance, 1), settings,
                                          out);
            if (!stats) FAIL(stats.error().toString());
            return energyOf(*h, out.colour, pixels);
        };
        const uint64_t sharp = energyAt(0.0);
        const uint64_t narrow = energyAt(11.0);
        const uint64_t open = energyAt(2.8);
        const double narrowRatio = double(narrow) / double(std::max<uint64_t>(sharp, 1));
        const double openRatio = double(open) / double(std::max<uint64_t>(sharp, 1));
        std::printf("  energy: sharp %llu, f/11 %llu (%.4f), f/2.8 %llu (%.4f)\n", (unsigned long long)sharp,
                    (unsigned long long)narrow, narrowRatio, (unsigned long long)open, openRatio);
        // A convolution by a normalised measure keeps the integral, and the
        // square root in the projection is what pays for it. What is missing
        // at a wide aperture is the footprint's own cutoff: a splat spread
        // thin has a low peak, and everything under 1/255 of it is not drawn
        // -- the same loss the shutter's long steps showed (0.9030 there).
        CHECK(narrowRatio > 0.96);   // measured 0.9677 at f/11
        CHECK(narrowRatio < 1.02);
        CHECK(openRatio > 0.85);     // measured 0.8819 at f/2.8
        CHECK(openRatio < 1.02);
    }
}
