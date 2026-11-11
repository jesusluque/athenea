// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>

#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "athenea/core/Hash.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/SplatLift.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
using test::CloudBuilder;
using test::randomCloud;

TEST_CASE("a trace names the gaussians a pixel is made of, and loses none of its weight",
          "[render][gpu][trace]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto reference = render::ReferenceRenderer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!reference) FAIL(reference.error().toString());

    // Two clouds side by side and overlapping where they meet, so a slot is
    // not a cloud's own index and the second one's start has to be right.
    CloudBuilder left = randomCloud(1200, 31, 0.02F, 0.12F);
    CloudBuilder right = randomCloud(1200, 97, 0.02F, 0.12F);
    auto cloudA = loader->upload(left.raw, 3);
    auto cloudB = loader->upload(right.raw, 3);
    REQUIRE(cloudA);
    REQUIRE(cloudB);
    const auto sideways = [](double x) {
        render::Mat4 m = render::Mat4::identity();
        m.at(0, 3) = x;
        return m;
    };
    const std::vector<render::SplatInstance> instances{{&*cloudA, sideways(-1.8)},
                                                      {&*cloudB, sideways(1.8)}};

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.5, 6.0}, {0.0, 0.0, 0.0});

    render::RenderTargets plain;
    auto without = raster->render(camera, instances, settings, plain);
    if (!without) FAIL(without.error().toString());
    CHECK_FALSE(plain.hasTrace());

    settings.trace = true;
    render::RenderTargets traced;
    auto with = raster->render(camera, instances, settings, traced);
    if (!with) FAIL(with.error().toString());
    REQUIRE(traced.hasTrace());
    // A slot is its instance's start plus the gaussian's index in its cloud.
    REQUIRE(with->slotBases.size() == 2);
    CHECK(with->slotBases[0] == 0);
    CHECK(with->slotBases[1] == cloudA->count);

    // The trace costs the colour nothing: the walk is the same walk.
    auto diff = render::compareImages(*gpu->library, plain.colour, traced.colour, settings.width,
                                      settings.height);
    if (!diff) FAIL(diff.error().toString());
    CHECK(diff->max == 0);

    // The exact walk's trace of the same frame.
    render::RenderTargets truth;
    auto overflow = reference->render(camera, instances, settings, truth);
    if (!overflow) FAIL(overflow.error().toString());
    CHECK(*overflow == 0);
    REQUIRE(truth.hasTrace());

    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "trace.stats");
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/trace_check");
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["trace"].setBinding(traced.trace.rhi());
            cursor["truth"].setBinding(truth.trace.rhi());
            cursor["colour"].setBinding(traced.colour.rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["params"]["width"].setData(settings.width);
            cursor["params"]["height"].setData(settings.height);
            cursor["params"]["slots"].setData(cloudA->count + cloudB->count);
            // Four weights and two more kept as halves (eleven bits each)
            // against one alpha summed in floats.
            cursor["params"]["tolerance"].setData(2.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> row{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(row), row.data()));
    std::printf("  trace: %u pixels name a gaussian, %u of them leave some out; %u ranks and %u pixels off the reference\n",
                row[4], row[5], row[6], row[7]);
    CHECK(row[0] == 0);   // every rank names a slot of the frame, at a depth
    CHECK(row[1] == 0);   // every pixel's ranks descend
    CHECK(row[2] == 0);   // no pixel names a gaussian twice
    CHECK(row[3] == 0);   // weights + rest + under are the pixel's alpha
    CHECK(row[4] > 1000);
    CHECK(row[4] < settings.width * settings.height);
    // Some pixels are made of more gaussians than a list holds: the case the
    // rest is there for.
    CHECK(row[5] > 0);
    CHECK(row[6] == 0);   // the ranks are the exact walk's
    CHECK(row[7] == 0);   // and so is what was left out
}

TEST_CASE("a mask made on one camera's frame is the matte another camera's frame draws",
          "[render][gpu][trace][lift]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());

    // Two prims, each a cloud with its own Cryptomatte id, overlapping where
    // they meet: the matte knows which prim covers what from any camera, so a
    // mask of the left one lifted from one camera can be measured from another.
    const uint32_t idA = core::cryptomatteId("/World/Left");
    const uint32_t idB = core::cryptomatteId("/World/Right");
    CloudBuilder left = randomCloud(1200, 31, 0.02F, 0.12F);
    CloudBuilder right = randomCloud(1200, 97, 0.02F, 0.12F);
    test::giveCryptoId(left, idA);
    test::giveCryptoId(right, idB);
    auto cloudA = loader->upload(left.raw, 3);
    auto cloudB = loader->upload(right.raw, 3);
    REQUIRE(cloudA);
    REQUIRE(cloudB);
    const auto sideways = [](double x) {
        render::Mat4 m = render::Mat4::identity();
        m.at(0, 3) = x;
        return m;
    };
    const std::vector<render::SplatInstance> instances{{&*cloudA, sideways(-1.2)},
                                                      {&*cloudB, sideways(1.2)}};
    const uint32_t slots = cloudA->count + cloudB->count;

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    settings.cryptomatte = true;
    settings.trace = true;
    const uint64_t pixels = uint64_t{settings.width} * settings.height;

    gpu::ComputeKernel maskOf = [&] {
        auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/test/matte_check", "matteMask");
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    }();
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/matte_check");
    gpu::BufferDesc floats;
    floats.bytes = pixels * sizeof(float);
    floats.elementBytes = sizeof(float);
    floats.label = "lift.mask";
    auto mask = gpu::Buffer::create(*gpu->device, floats);
    REQUIRE(mask);
    gpu::Buffer unused = test::uintBuffer(*gpu->device, 8, "lift.unused");

    // Two lifts of the same masks: through the trace (the four gaussians a
    // pixel names) and while blending (every gaussian of its walk).
    auto named = render::makeSplatLift(*gpu->device, slots);
    if (!named) FAIL(named.error().toString());
    auto whole = render::makeSplatLift(*gpu->device, slots);
    if (!whole) FAIL(whole.error().toString());

    // A frame from `eye`, and what its matte measures against the left prim's
    // own coverage there.
    const auto frame = [&](std::array<double, 3> eye, render::RenderTargets& targets) {
        const render::Camera camera = render::Camera::lookingAt({eye[0], eye[1], eye[2]}, {0.0, 0.0, 0.0});
        auto drawn = raster->render(camera, instances, settings, targets);
        if (!drawn) FAIL(drawn.error().toString());
        REQUIRE(targets.hasTrace());
        REQUIRE(targets.hasCrypto());
    };
    const auto bind = [&](rhi::ShaderCursor cursor, const render::RenderTargets& targets) {
        cursor["crypto"].setBinding(targets.crypto.rhi());
        cursor["colour"].setBinding(targets.colour.rhi());
        cursor["params"]["width"].setData(settings.width);
        cursor["params"]["height"].setData(settings.height);
        cursor["params"]["id"].setData(idA);
        cursor["params"]["tolerance"].setData(0.25F);
    };
    const auto liftFrom = [&](std::array<double, 3> eye) {
        render::RenderTargets targets;
        frame(eye, targets);
        gpu::CommandBatch batch(*gpu->device);
        maskOf.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor, targets);
            cursor["mask"].setBinding(mask->rhi());
            cursor["matte"].setBinding(targets.depth.rhi());
            cursor["stats"].setBinding(unused.rhi());
        });
        REQUIRE(batch.submit(true));
        auto lifted = render::liftMask(*gpu->library, targets, *mask, *named);
        if (!lifted) FAIL(lifted.error().toString());
        // The same frame again, lifting as it blends.
        render::RenderSettings lifting = settings;
        render::liftWhileBlending(lifting, *whole, *mask);
        render::RenderTargets again;
        const render::Camera camera = render::Camera::lookingAt({eye[0], eye[1], eye[2]}, {0.0, 0.0, 0.0});
        auto drawn = raster->render(camera, instances, lifting, again);
        if (!drawn) FAIL(drawn.error().toString());
    };
    const auto measure = [&](std::array<double, 3> eye, const render::SplatLift& lift, bool hard) {
        render::RenderTargets targets;
        frame(eye, targets);
        gpu::Buffer matte;
        render::MatteSettings how;
        how.hard = hard;
        auto drawn = render::drawMatte(*gpu->library, targets, lift, how, matte);
        if (!drawn) FAIL(drawn.error().toString());
        gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "matte.stats");
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor, targets);
            cursor["mask"].setBinding(mask->rhi());
            cursor["matte"].setBinding(matte.rhi());
            cursor["stats"].setBinding(stats.rhi());
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 8> row{};
        REQUIRE(stats.read(*gpu->device, 0, sizeof(row), row.data()));
        return row;
    };
    const auto lifted = [&](const render::SplatLift& lift) {
        gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "lift.stats");
        gpu::ComputeKernel count = test::kernel(*gpu, "athenea/test/lift_check");
        gpu::CommandBatch batch(*gpu->device);
        count.dispatch(batch, {slots, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["lift"].setBinding(lift.sums.rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["params"]["slots"].setData(slots);
            cursor["params"]["firstOutside"].setData(cloudA->count);
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 8> row{};
        REQUIRE(stats.read(*gpu->device, 0, sizeof(row), row.data()));
        return row;
    };
    const auto report = [](const char* what, const std::array<uint32_t, 8>& row) {
        std::printf("  %s: %u covered pixels, %u off by more than 0.25, mean |matte - coverage| %.4f, "
                    "matte %.1f px against the prim's %.1f px\n",
                    what, row[0], row[1], row[0] > 0 ? double(row[2]) / 1024.0 / row[0] : 0.0,
                    double(row[4]) / 1024.0, double(row[3]) / 1024.0);
    };

    const std::array<double, 3> front{0.0, 0.5, 6.0};
    const std::array<double, 3> side{3.5, 1.5, 4.5};
    const std::array<double, 3> other{-3.0, 0.2, 5.0};

    const auto said = [](const char* what, const std::array<uint32_t, 8>& row) {
        std::printf("  %s: %u + %u gaussians seen, %u + %u told wrongly, %u unseen\n", what, row[0], row[2], row[1],
                    row[3], row[4]);
    };
    // One camera's mask, seen by that camera and by another.
    liftFrom(front);
    const auto oneNamed = lifted(*named);
    const auto oneWhole = lifted(*whole);
    said("one camera, by the trace", oneNamed);
    said("one camera, while blending", oneWhole);
    const auto same = measure(front, *whole, false);
    report("the lifting camera", same);
    const auto acrossNamed = measure(side, *named, false);
    report("another camera, by the trace", acrossNamed);
    const auto across = measure(side, *whole, false);
    report("another camera", across);
    const auto acrossHard = measure(side, *whole, true);
    report("another camera, hard", acrossHard);
    // More cameras' masks of the same prim, added to the same gaussians: a
    // ring round the two clouds, the measuring camera never among them.
    std::array<uint32_t, 8> twoWhole{};
    std::array<uint32_t, 8> acrossTwo{};
    const double ring[] = {180.0, 90.0, 270.0, 45.0, 135.0, 225.0, 315.0};
    uint32_t cameras = 1;
    for (const double degrees : ring) {
        const double a = degrees * 3.14159265358979 / 180.0;
        liftFrom({6.0 * std::sin(a), 0.5 + 0.3 * cameras, 6.0 * std::cos(a)});
        ++cameras;
        if (cameras == 2 || cameras == 4 || cameras == 8) {
            twoWhole = lifted(*whole);
            char what[64];
            std::snprintf(what, sizeof(what), "%u cameras, while blending", cameras);
            said(what, twoWhole);
            acrossTwo = measure(side, *whole, false);
            std::snprintf(what, sizeof(what), "another camera, %u lifted", cameras);
            report(what, acrossTwo);
            std::snprintf(what, sizeof(what), "another camera, %u lifted, by the trace", cameras);
            report(what, measure(side, *named, false));
            std::snprintf(what, sizeof(what), "another camera, %u lifted, hard", cameras);
            report(what, measure(side, *whole, true));
        }
    }

    // Blending tells every gaussian the frame drew; the trace only the few
    // each pixel names.
    CHECK(oneWhole[0] + oneWhole[2] > oneNamed[0] + oneNamed[2]);
    // From the camera the mask was made on, the matte is the prim's coverage:
    // a pixel in a hundred further than a quarter from it.
    REQUIRE(same[0] > 1000);
    CHECK(same[1] * 100 < same[0]);
    CHECK(double(same[2]) / 1024.0 / same[0] < 0.05);
    // One camera sees half of two clouds; a ring of eight leaves few
    // gaussians untold, and a camera that was not among them then draws the
    // matte nearly as the lifting camera did. (Where one prim stands in front
    // of the other on a lifting camera's screen its mask calls both the same:
    // what is left of the error is that, which no lift of flat masks removes.)
    CHECK(twoWhole[4] * 20 < slots);
    CHECK(twoWhole[4] * 10 < oneWhole[4]);
    REQUIRE(across[0] > 1000);
    REQUIRE(acrossTwo[0] > 1000);
    CHECK(acrossTwo[1] * 4 < across[1]);
    CHECK(acrossTwo[1] * 50 < acrossTwo[0]);
    CHECK(double(acrossTwo[2]) / 1024.0 / acrossTwo[0] < 0.08);
}
