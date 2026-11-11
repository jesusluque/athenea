// Copyright (c) 2026 jesus luque.
#include <catch2/catch_test_macros.hpp>

#include "../gpu/GpuTest.h"

#include <array>
#include <chrono>
#include <cstdio>

using namespace athenea;

namespace {

constexpr uint32_t kIn = 16;
constexpr uint32_t kWeights = (32 * 16 + 32) + (32 * 32 + 32) + (3 * 32 + 3);

gpu::Buffer floats(gpu::Device& device, uint64_t count, uint32_t channels, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = count * channels * sizeof(float);
    desc.elementBytes = channels * sizeof(float);
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    REQUIRE(made);
    return *made;
}

gpu::ComputeKernel entry(test::Gpu& gpu, const char* name) {
    auto made = gpu::ComputeKernel::create(*gpu.library, "athenea/test/neural_mlp", name);
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

}   // namespace

TEST_CASE("a small network runs and differentiates in a kernel, a pixel at a time", "[render][gpu][neural]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::ComputeKernel init = entry(*gpu, "neuralMlpInit");
    gpu::ComputeKernel forward = entry(*gpu, "neuralMlpForward");
    gpu::ComputeKernel backward = entry(*gpu, "neuralMlpBackward");
    gpu::ComputeKernel check = entry(*gpu, "neuralMlpCheck");
    gpu::ComputeKernel byHand = entry(*gpu, "neuralMlpBackwardByHand");

    gpu::Buffer weights = floats(*gpu->device, kWeights, 1, "mlp.weights");
    gpu::Buffer weightGrad = test::uintBuffer(*gpu->device, kWeights, "mlp.weightGrad");
    gpu::Buffer loss = test::uintBuffer(*gpu->device, 1, "mlp.loss");
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "mlp.stats");

    struct Frame {
        uint32_t    width, height;
        gpu::Buffer features, target, colour, featureGrad;
    };
    const auto frame = [&](uint32_t width, uint32_t height) {
        const uint64_t pixels = uint64_t{width} * height;
        return Frame{width, height, floats(*gpu->device, pixels * kIn, 1, "mlp.features"),
                     floats(*gpu->device, pixels, 4, "mlp.target"), floats(*gpu->device, pixels, 4, "mlp.colour"),
                     floats(*gpu->device, pixels * kIn, 1, "mlp.featureGrad")};
    };
    const auto bind = [&](rhi::ShaderCursor cursor, const Frame& f, const gpu::Buffer& made) {
        cursor["weights"].setBinding(weights.rhi());
        cursor["weightGrad"].setBinding(weightGrad.rhi());
        cursor["features"].setBinding(f.features.rhi());
        cursor["target"].setBinding(f.target.rhi());
        cursor["colour"].setBinding(f.colour.rhi());
        cursor["featureGrad"].setBinding(f.featureGrad.rhi());
        cursor["loss"].setBinding(loss.rhi());
        cursor["made"].setBinding(made.rhi());
        cursor["stats"].setBinding(stats.rhi());
        cursor["params"]["width"].setData(f.width);
        cursor["params"]["height"].setData(f.height);
        cursor["params"]["gradScale"].setData(65536.0F);
        cursor["params"]["step"].setData(2.0e-3F);
    };
    // Numbers a hash made: `width` of them, `height` the seed, `gradScale` their range.
    const auto fill = [&](const Frame& f, const gpu::Buffer& what, uint32_t count, uint32_t seed, float range) {
        gpu::CommandBatch batch(*gpu->device);
        init.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bind(cursor, f, what);
            cursor["params"]["width"].setData(count);
            cursor["params"]["height"].setData(seed);
            cursor["params"]["gradScale"].setData(range);
        });
        REQUIRE(batch.submit(true));
    };

    // The derivatives against central differences, on a small frame.
    uint32_t slangOff = 0;
    {
        Frame f = frame(160, 120);
        const uint32_t pixels = f.width * f.height;
        fill(f, weights, kWeights, 1, 0.3F);
        fill(f, f.features, pixels * kIn, 2, 1.0F);
        fill(f, f.target, pixels * 4, 3, 1.0F);
        gpu::CommandBatch batch(*gpu->device);
        backward.dispatch(batch, {f.width, f.height, 1}, [&](rhi::ShaderCursor cursor) { bind(cursor, f, f.colour); });
        check.dispatch(batch, {f.width, f.height, 1}, [&](rhi::ShaderCursor cursor) { bind(cursor, f, f.colour); });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 8> row{};
        REQUIRE(stats.read(*gpu->device, 0, sizeof(row), row.data()));
        std::printf("  mlp: %u pixels checked, %u derivatives off their central difference, %u not finite\n", row[0],
                    row[1], row[2]);
        std::printf("  mean |difference| %.4f, mean |derivative| %.4f, mean loss %.4f\n", row[3] / 16.0 / row[0], row[4] / 16.0 / row[0], row[5] / 16.0 / row[0]);
        std::array<uint32_t, 8> hand{};
        {
            gpu::Buffer again = test::uintBuffer(*gpu->device, 8, "mlp.stats.hand");
            gpu::CommandBatch second(*gpu->device);
            byHand.dispatch(second, {f.width, f.height, 1}, [&](rhi::ShaderCursor cursor) { bind(cursor, f, f.colour); });
            check.dispatch(second, {f.width, f.height, 1}, [&](rhi::ShaderCursor cursor) {
                bind(cursor, f, f.colour);
                cursor["stats"].setBinding(again.rhi());
            });
            REQUIRE(second.submit(true));
            hand = {};
            REQUIRE(again.read(*gpu->device, 0, sizeof(hand), hand.data()));
            std::printf("  by hand: %u pixels checked, %u derivatives off, mean |derivative| %.4f\n", hand[0], hand[1],
                        hand[4] / 16.0 / hand[0]);
        }
        // The backward pass written out is the derivative: a ReLU's kink
        // inside the step is the one place it and a difference may part.
        CHECK(hand[0] == pixels);
        CHECK(hand[1] * 20 < hand[0]);
        CHECK(row[0] == pixels);
        CHECK(row[2] == 0);
        // What Slang 2026.14.1 differentiates of this network is not its
        // derivative (docs/decisions.md): measured and printed, not held to.
        (void)slangOff;
    }

    // What a pixel of it costs: the forward pass a deferred decoder runs, and the backward one.
    for (const auto& size : {std::array<uint32_t, 2>{1920, 1080}, std::array<uint32_t, 2>{3840, 2160}}) {
        Frame f = frame(size[0], size[1]);
        const uint32_t pixels = f.width * f.height;
        fill(f, f.features, pixels * kIn, 2, 1.0F);
        fill(f, f.target, pixels * 4, 3, 1.0F);
        const auto timed = [&](gpu::ComputeKernel& kernel) {
            double best = 1.0e30;
            for (int round = 0; round < 6; ++round) {
                const auto start = std::chrono::steady_clock::now();
                gpu::CommandBatch batch(*gpu->device);
                kernel.dispatch(batch, {f.width, f.height, 1},
                                [&](rhi::ShaderCursor cursor) { bind(cursor, f, f.colour); });
                REQUIRE(batch.submit(true));
                best = std::min(best, std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - start)
                                          .count());
            }
            return best;
        };
        const double forwardMs = timed(forward);
        const double backwardMs = size[0] == 1920 ? timed(backward) : 0.0;
        const double handMs = timed(byHand);
        std::printf("  mlp 16-32-32-3 at %ux%u: forward %.2f ms, backward by hand %.2f ms, by Slang %.2f ms\n", f.width, f.height,
                    forwardMs, handMs, backwardMs);
    }
}

#include "SplatFixtures.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

TEST_CASE("a cloud drawn from its latents starts as its own first harmonic draws it", "[render][gpu][neural]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    test::CloudBuilder builder = test::randomCloud(3000, 11, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    auto neural = render::makeNeuralCloud(*gpu->library, *cloud);
    if (!neural) FAIL(neural.error().toString());
    CHECK(neural->count == cloud->count);

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Camera camera = render::Camera::lookingAt({0.5, 0.5, 6.0}, {0.0, 0.0, 0.0});

    // The cloud by its first harmonic alone, and by all of them.
    const std::vector<render::SplatInstance> plain{{&*cloud, render::Mat4::identity()}};
    render::RenderTargets flat;
    settings.maxShDegree = 0;
    auto drawnFlat = raster->render(camera, plain, settings, flat);
    if (!drawnFlat) FAIL(drawnFlat.error().toString());
    render::RenderTargets whole;
    settings.maxShDegree = 3;
    auto drawnWhole = raster->render(camera, plain, settings, whole);
    if (!drawnWhole) FAIL(drawnWhole.error().toString());

    // By its latents, as a fit starts: nothing learnt, so the base alone.
    std::vector<render::SplatInstance> learnt = plain;
    learnt[0].neural = &*neural;
    render::RenderTargets started;
    auto drawnStarted = raster->render(camera, learnt, settings, started);
    if (!drawnStarted) FAIL(drawnStarted.error().toString());

    auto same = render::compareImages(*gpu->library, started.colour, flat.colour, settings.width, settings.height);
    auto other = render::compareImages(*gpu->library, started.colour, whole.colour, settings.width, settings.height);
    if (!same) FAIL(same.error().toString());
    if (!other) FAIL(other.error().toString());
    std::printf("  neural start against degree 0: p99 %u, max %u; against degree 3: p99 %u, max %u\n", same->p99,
                same->max, other->p99, other->max);
    CHECK(same->max <= 1);
    // And it is the latents that were drawn: the harmonics' own frame differs.
    CHECK(other->max > 4);
}

#include "athenea/render/NeuralFit.h"

TEST_CASE("a cloud's harmonics are learnt by its latents, from frames of itself", "[render][gpu][neural][fit]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto fit = render::NeuralFit::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!fit) FAIL(fit.error().toString());
    test::CloudBuilder builder = test::randomCloud(3000, 11, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    auto neural = render::makeNeuralCloud(*gpu->library, *cloud);
    if (!neural) FAIL(neural.error().toString());
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};

    render::NeuralFitSettings how;
    how.width = 320;
    how.height = 240;
    how.rateFeatures = 1.0e-2F;
    how.rateWeights = 4.0e-3F;
    // Cameras on a ring round the cloud, at heights that go up and down.
    const auto cameraAt = [](uint32_t k) {
        const double a = 2.39996323 * k;   // the golden angle: no two alike
        const double h = 2.5 * std::sin(0.7 * k);
        return render::Camera::lookingAt({6.0 * std::sin(a), h, 6.0 * std::cos(a)}, {0.0, 0.0, 0.0});
    };
    double first = 0.0;
    double last = 0.0;
    const uint32_t steps = 1500;
    for (uint32_t k = 0; k < steps; ++k) {
        const bool ask = k == 0 || (k + 1) % 250 == 0;
        auto loss = fit->step(*raster, cameraAt(k), instance, *neural, how, ask);
        if (!loss) FAIL(loss.error().toString());
        if (*loss) {
            std::printf("  fit step %u: mean squared error %.6f\n", k + 1, **loss);
            if (k == 0) first = **loss;
            last = **loss;
        }
    }
    CHECK(last < first * 0.3);

    // From a camera no step used: nearer the harmonics' frame than the first harmonic's is.
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    const render::Camera held = render::Camera::lookingAt({-3.3, 1.1, -4.9}, {0.0, 0.0, 0.0});
    const std::vector<render::SplatInstance> plain{instance};
    std::vector<render::SplatInstance> learnt = plain;
    learnt[0].neural = &*neural;
    render::RenderTargets whole, flat, ours;
    REQUIRE(raster->render(held, plain, settings, whole));
    settings.maxShDegree = 0;
    REQUIRE(raster->render(held, plain, settings, flat));
    settings.maxShDegree = 3;
    REQUIRE(raster->render(held, learnt, settings, ours));
    auto before = render::compareHdr(*gpu->library, flat.colour, whole.colour, settings.width, settings.height);
    auto after = render::compareHdr(*gpu->library, ours.colour, whole.colour, settings.width, settings.height);
    REQUIRE(before);
    REQUIRE(after);
    std::printf("  held-out camera against the harmonics: first harmonic alone relMSE %.5f, the latents %.5f\n",
                before->relMse, after->relMse);
    CHECK(after->relMse < before->relMse * 0.5);
    test::dumpPpm(*gpu, "fit_harmonics", whole.colour, settings.width, settings.height);
    test::dumpPpm(*gpu, "fit_first_harmonic", flat.colour, settings.width, settings.height);
    test::dumpPpm(*gpu, "fit_latents", ours.colour, settings.width, settings.height);
}
