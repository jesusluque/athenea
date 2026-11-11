// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>

#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include "athenea/render/ReferenceRenderer.h"
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
