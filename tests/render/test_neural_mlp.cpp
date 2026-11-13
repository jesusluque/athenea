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
#include "athenea/render/NeuralWall.h"
#include "athenea/render/Discovery.h"
#include "athenea/render/SplatLift.h"

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
            std::printf("  fit step %u: relMSE %.5f\n", k + 1, **loss);
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

#include "athenea/light/LightTable.h"

TEST_CASE("a lit neural cloud learns how its gaussians answer a light, and is relit by one it never saw",
          "[render][gpu][neural][fit][relight]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto fit = render::NeuralFit::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!fit) FAIL(fit.error().toString());
    if (!table) FAIL(table.error().toString());
    test::CloudBuilder builder = test::randomCloud(3000, 11, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    auto neural = render::makeNeuralCloud(*gpu->library, *cloud, 1, true);
    if (!neural) FAIL(neural.error().toString());
    CHECK(neural->lit);
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};

    // A sun from a direction: the engine's light for the picture the cloud
    // should be, and the same light as the neural cloud is told it.
    render::SplatLights lights;
    const auto sun = [&](double yaw, double pitch) {
        light::Light lamp;
        lamp.kind = light::LightKind::Distant;
        lamp.lightToWorld = aofx::xform::rotationY(yaw) * aofx::xform::rotationX(pitch);
        lamp.intensity = 3.0F;
        lamp.shadow = false;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        lights.records = &table->records();
        lights.count = table->count();
        render::NeuralFitLight told;
        told.teacher = &lights;
        // A distant light shines down its own -z: it comes from its +z.
        const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
        told.light.towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                              static_cast<float>(towards.z)};
        told.light.radiance = {lamp.intensity, lamp.intensity, lamp.intensity};
        return told;
    };
    const auto cameraAt = [](double k) {
        const double a = 6.283185307179586 * std::fmod(0.5 + k * 0.7548776662466927, 1.0);
        const double h = 3.0 * (2.0 * std::fmod(0.5 + k * 0.5698402909980532, 1.0) - 1.0);
        return render::Camera::lookingAt({6.0 * std::sin(a), h, 6.0 * std::cos(a)}, {0.0, 0.0, 0.0});
    };
    // Lights over the sphere, by a third and a fourth independent sequence.
    const auto yawOf = [](double k) { return 360.0 * std::fmod(0.5 + k * 0.8191725134, 1.0); };
    const auto pitchOf = [](double k) { return -80.0 + 160.0 * std::fmod(0.5 + k * 0.6710436067, 1.0); };

    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    // The neural cloud under one light against the engine's relighting under another (or the same).
    const auto against = [&](double camera, double told, double shone) {
        const render::Camera held = cameraAt(camera);
        const render::NeuralFitLight a = sun(yawOf(told), pitchOf(told));
        std::vector<render::SplatInstance> learnt{instance};
        learnt[0].neural = &*neural;
        learnt[0].neuralLight = &a.light;
        render::RenderTargets ours;
        REQUIRE(raster->render(held, learnt, settings, ours));
        const render::NeuralFitLight b = sun(yawOf(shone), pitchOf(shone));
        std::vector<render::SplatInstance> relit{instance};
        relit[0].relight = true;
        render::RenderTargets truth;
        REQUIRE(raster->render(held, relit, settings, truth, {}, nullptr, b.teacher));
        auto hdr = render::compareHdr(*gpu->library, ours.colour, truth.colour, settings.width, settings.height);
        REQUIRE(hdr);
        if (told == shone) {
            char name[64];
            std::snprintf(name, sizeof(name), "relight_%g_neural", told);
            test::dumpPpm(*gpu, name, ours.colour, settings.width, settings.height);
            std::snprintf(name, sizeof(name), "relight_%g_engine", told);
            test::dumpPpm(*gpu, name, truth.colour, settings.width, settings.height);
        }
        return hdr->relMse;
    };
    // A camera and two lights no step uses.
    const double before = 0.5 * (against(1000.5, 2000.5, 2000.5) + against(1000.5, 2007.5, 2007.5));

    render::NeuralFitSettings how;
    how.width = 320;
    how.height = 240;
    how.rateFeatures = 1.0e-2F;
    how.rateWeights = 4.0e-3F;
    for (uint32_t k = 0; k < 3000; ++k) {
        const render::NeuralFitLight told = sun(yawOf(k), pitchOf(k));
        const bool ask = k == 0 || (k + 1) % 500 == 0;
        auto loss = fit->step(*raster, cameraAt(k), instance, *neural, how, ask, &told);
        if (!loss) FAIL(loss.error().toString());
        if (*loss) {
            std::printf("  lit fit step %u: relMSE %.5f\n", k + 1, **loss);
        }
    }
    const double after = 0.5 * (against(1000.5, 2000.5, 2000.5) + against(1000.5, 2007.5, 2007.5));
    // Told one light and held against the picture under the other: what it
    // would measure if it had learnt a look and not an answer to a light.
    const double crossed = 0.5 * (against(1000.5, 2000.5, 2007.5) + against(1000.5, 2007.5, 2000.5));
    std::printf("  held-out camera and lights against the engine's relighting: relMSE before %.5f, after %.5f; "
                "told the other light %.5f\n",
                before, after, crossed);
    CHECK(after < before * 0.5);
    CHECK(after < crossed * 0.5);
}

#include <filesystem>
#include <fstream>

#include "athenea/io/NeuralFile.h"
#include "athenea/io/NeuralScene.h"

TEST_CASE("a neural appearance written and read draws what it drew", "[render][gpu][neural][file]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto fit = render::NeuralFit::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!fit) FAIL(fit.error().toString());
    test::CloudBuilder builder = test::randomCloud(2000, 5, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    auto neural = render::makeNeuralCloud(*gpu->library, *cloud);
    if (!neural) FAIL(neural.error().toString());
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    // A few steps, so that the features and the network hold something.
    render::NeuralFitSettings how;
    how.width = 200;
    how.height = 150;
    for (uint32_t k = 0; k < 60; ++k) {
        const double a = 2.39996323 * k;
        auto stepped = fit->step(*raster, render::Camera::lookingAt({6.0 * std::sin(a), 1.0, 6.0 * std::cos(a)}, {0.0, 0.0, 0.0}),
                                 instance, *neural, how);
        if (!stepped) FAIL(stepped.error().toString());
    }

    io::NeuralFile file;
    file.count = neural->count;
    file.stride = render::kNeuralStride;
    file.wordsEach = render::kNeuralWords;
    auto words = render::packNeuralLatents(*gpu->library, *neural);
    auto weights = render::readNeuralWeights(*gpu->library, *neural);
    if (!words) FAIL(words.error().toString());
    if (!weights) FAIL(weights.error().toString());
    file.latent = *words;
    file.weights = *weights;
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "athenea_test_neural.athn";
    REQUIRE(io::writeNeural(path, file));
    // 24 bytes a gaussian, and the network.
    CHECK(std::filesystem::file_size(path) == 4 + 6 * 4 + uint64_t{neural->count} * 24 + render::kNeuralWeights * 4);
    auto read = io::readNeural(path);
    if (!read) FAIL(read.error().toString());
    std::filesystem::remove(path);
    CHECK(read->count == neural->count);
    CHECK_FALSE(read->lit);
    auto again = render::neuralCloudFrom(*gpu->library, read->count, read->latent, read->weights, read->lit);
    if (!again) FAIL(again.error().toString());

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Camera camera = render::Camera::lookingAt({1.0, 0.5, 6.0}, {0.0, 0.0, 0.0});
    std::vector<render::SplatInstance> first{instance};
    first[0].neural = &*neural;
    std::vector<render::SplatInstance> second{instance};
    second[0].neural = &*again;
    render::RenderTargets a, b;
    REQUIRE(raster->render(camera, first, settings, a));
    REQUIRE(raster->render(camera, second, settings, b));
    auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
    REQUIRE(diff);
    std::printf("  written and read: p99 %u, max %u\n", diff->p99, diff->max);
    // Halves for floats: a code value at most.
    CHECK(diff->max <= 1);
}

TEST_CASE("a lit neural cloud under two lights is the sum of it under each", "[render][gpu][neural][relight][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto fit = render::NeuralFit::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!fit) FAIL(fit.error().toString());
    if (!table) FAIL(table.error().toString());
    test::CloudBuilder builder = test::randomCloud(2000, 7, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    auto neural = render::makeNeuralCloud(*gpu->library, *cloud, 1, true);
    if (!neural) FAIL(neural.error().toString());
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    const render::Camera camera = render::Camera::lookingAt({1.0, 0.5, 6.0}, {0.0, 0.0, 0.0});

    // A few steps under a sun, so that the network says something of a light's direction.
    render::SplatLights lights;
    render::NeuralFitSettings how;
    how.width = 200;
    how.height = 150;
    for (uint32_t k = 0; k < 80; ++k) {
        light::Light lamp;
        lamp.kind = light::LightKind::Distant;
        lamp.lightToWorld = aofx::xform::rotationY(47.0 * k) * aofx::xform::rotationX(-60.0 + 13.0 * (k % 9));
        lamp.intensity = 3.0F;
        lamp.shadow = false;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        lights.records = &table->records();
        lights.count = table->count();
        render::NeuralFitLight told;
        told.teacher = &lights;
        const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
        told.light.towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                              static_cast<float>(towards.z)};
        told.light.radiance = {3.0F, 3.0F, 3.0F};
        const double a = 2.39996323 * k;
        auto stepped = fit->step(*raster, render::Camera::lookingAt({6.0 * std::sin(a), 1.0, 6.0 * std::cos(a)}, {0.0, 0.0, 0.0}),
                                 instance, *neural, how, false, &told);
        if (!stepped) FAIL(stepped.error().toString());
    }

    // A red light from one side, a blue one from another: each alone, and both at half.
    const render::NeuralLight red{{0.8F, 0.5F, 0.33F}, {4.0F, 0.5F, 0.2F}};
    const render::NeuralLight blue{{-0.6F, 0.2F, 0.77F}, {0.2F, 0.6F, 4.0F}};
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const auto under = [&](std::vector<render::NeuralLight> which, render::RenderTargets& into) {
        std::vector<render::SplatInstance> drawn{instance};
        drawn[0].neural = &*neural;
        drawn[0].neuralLight = which.data();
        drawn[0].neuralLightCount = static_cast<uint32_t>(which.size());
        REQUIRE(raster->render(camera, drawn, settings, into));
    };
    render::RenderTargets byRed, byBlue, byBoth;
    under({red}, byRed);
    under({blue}, byBlue);
    render::NeuralLight halfRed = red;
    render::NeuralLight halfBlue = blue;
    for (int c = 0; c < 3; ++c) {
        halfRed.radiance[static_cast<size_t>(c)] *= 0.5F;
        halfBlue.radiance[static_cast<size_t>(c)] *= 0.5F;
    }
    under({halfRed, halfBlue}, byBoth);
    // Both at half is the mean of each alone: what `compareFlicker` measures a middle frame against.
    auto mean = render::compareFlicker(*gpu->library, byRed.colour, byBoth.colour, byBlue.colour, settings.width,
                                       settings.height);
    REQUIRE(mean);
    auto apart = render::compareImages(*gpu->library, byRed.colour, byBlue.colour, settings.width, settings.height);
    REQUIRE(apart);
    std::printf("  two lights at half against the mean of each alone: max %u code values; the two alone differ by %u\n",
                mean->codes.max, apart->max);
    CHECK(mean->codes.max <= 1);
    CHECK(apart->max > 20);
}

TEST_CASE("a wall's light on a cloud adds as its picture does, and its reads are learnt from frames",
          "[render][gpu][neural][wall]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto shown = render::NeuralWall::create(*gpu->library);
    auto learnt = render::NeuralWall::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!shown) FAIL(shown.error().toString());
    if (!learnt) FAIL(learnt.error().toString());
    test::CloudBuilder builder = test::randomCloud(2000, 7, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    // The cloud gives no light of its own: lit, under no light.
    auto dark = render::makeNeuralCloud(*gpu->library, *cloud, 1, true);
    if (!dark) FAIL(dark.error().toString());
    const render::NeuralLight none;

    // A wall of 8 x 4 texels behind the cloud. The one the frames come from reads it narrow and strong
    // and stands a little aside; the one that learns starts wide and faint, where the wall is said to be.
    const uint32_t wide = 8;
    const uint32_t high = 4;
    const uint32_t texels = wide * high;
    render::WallRect rect;
    rect.centre = {0.0F, 0.0F, -3.0F};
    rect.halfU = 3.0F;
    rect.halfV = 1.5F;
    render::WallRect aside = rect;
    aside.centre = {0.5F, 0.2F, -3.0F};
    REQUIRE(shown->start(*cloud, aside, 1, wide, high, texels + 2, 1.0F, 0.6F));
    REQUIRE(learnt->start(*cloud, rect, 1, wide, high, texels + 2, -1.0F, 0.05F));
    for (render::NeuralWall* wall : {&*shown, &*learnt}) {
        for (uint32_t k = 0; k < texels; ++k) {
            REQUIRE(wall->pattern(k, 0, k));
        }
        REQUIRE(wall->pattern(texels, 5, 0));      // white
        REQUIRE(wall->pattern(texels + 1, 3, 5));  // a colour a texel
    }
    const auto before = [&](render::NeuralWall& wall) {
        render::SplatInstance instance{&*cloud, render::Mat4::identity()};
        instance.neural = &*dark;
        instance.neuralLight = &none;
        instance.neuralLightCount = 0;
        instance.wall = &wall;
        return instance;
    };
    const render::SplatInstance teacher = before(*shown);
    const render::SplatInstance student = before(*learnt);
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const uint32_t pixels = settings.width * settings.height;
    const render::Camera camera = render::Camera::lookingAt({1.0, 0.5, 6.0}, {0.0, 0.0, 0.0});

    // LIGHT ADDS. The frame under the white wall is the frames under each texel alone, summed.
    render::RenderTargets whole, one;
    gpu::Buffer summed = floats(*gpu->device, pixels, 4, "wall.summed");
    shown->show(texels);
    REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), settings, whole));
    for (uint32_t k = 0; k < texels; ++k) {
        shown->show(k);
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), settings, one));
        REQUIRE(shown->accumulate(one.colour, summed, pixels, texels, k, k == 0));
    }
    auto added = render::compareImages(*gpu->library, whole.colour, summed, settings.width, settings.height);
    REQUIRE(added);
    auto lit = render::imageStats(*gpu->library, whole.colour, settings.width, settings.height);
    REQUIRE(lit);
    std::printf("  the white wall against its texels summed: max %u code values; the frame's mean %.3f\n", added->max,
                lit->mean[0]);
    CHECK(added->max <= 1);
    CHECK(lit->mean[0] > 0.01);

    // A STILL FRAME. The camera and the cloud as they were and the wall showing another picture: the frame
    // keeps the last one's geometry and is that picture's frame drawn whole; from another camera it keeps
    // nothing, and is right.
    {
        render::RenderSettings stillSettings = settings;
        stillSettings.still = true;
        render::RenderTargets kept, drawn;
        shown->show(texels);
        auto first = raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), stillSettings, kept);
        REQUIRE(first);
        CHECK(!first->still);
        shown->show(texels + 1);
        auto second = raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), stillSettings, kept);
        REQUIRE(second);
        CHECK(second->still);
        const render::Camera other = render::Camera::lookingAt({-2.0, 1.5, 5.0}, {0.0, 0.0, 0.0});
        auto moved = raster->render(other, std::span<const render::SplatInstance>(&teacher, 1), stillSettings, drawn);
        REQUIRE(moved);
        CHECK(!moved->still);
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), settings, drawn));
        auto same = render::compareImages(*gpu->library, kept.colour, drawn.colour, settings.width, settings.height);
        REQUIRE(same);
        std::printf("  a still frame under another picture against that picture's frame drawn whole: max %u\n", same->max);
        CHECK(same->max == 0);
    }

    // THE READS ARE LEARNT. Shown single texels, held against a picture of a colour a texel.
    const auto apart = [&]() {
        shown->show(texels + 1);
        learnt->show(texels + 1);
        render::RenderTargets a, b;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&teacher, 1), settings, a));
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&student, 1), settings, b));
        auto hdr = render::compareHdr(*gpu->library, b.colour, a.colour, settings.width, settings.height);
        REQUIRE(hdr);
        return hdr->relMse;
    };
    const double start = apart();
    render::NeuralWallSettings how;
    how.width = settings.width;
    how.height = settings.height;
    render::RenderTargets pictured;
    for (uint32_t k = 0; k < 600; ++k) {
        const uint32_t texel = (k * 7) % texels;
        const double a = 2.39996323 * k;
        const render::Camera from =
            render::Camera::lookingAt({6.0 * std::sin(a), 1.0 + std::sin(a * 0.37), 6.0 * std::cos(a)}, {0.0, 0.0, 0.0});
        shown->show(texel);
        learnt->show(texel);
        REQUIRE(raster->render(from, std::span<const render::SplatInstance>(&teacher, 1), settings, pictured));
        auto stepped = learnt->step(*raster, from, student, pictured.colour, how);
        if (!stepped) FAIL(stepped.error().toString());
    }
    const double end = apart();
    std::printf("  a picture the fit never saw: relMSE %.5f before, %.5f after 600 steps\n", start, end);
    CHECK(end < start * 0.25);
}

TEST_CASE("a split neural cloud starts as one network's does, and its two networks are fitted together",
          "[render][gpu][neural][split]") {
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
    auto whole = render::makeNeuralCloud(*gpu->library, *cloud);
    auto split = render::makeNeuralCloud(*gpu->library, *cloud, 1, false, true);
    if (!whole) FAIL(whole.error().toString());
    if (!split) FAIL(split.error().toString());
    CHECK(split->weightCount() == 2 * whole->weightCount());

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Camera camera = render::Camera::lookingAt({0.5, 0.5, 6.0}, {0.0, 0.0, 0.0});
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    const auto drawnBy = [&](const render::NeuralCloud& neural, render::RenderTargets& into) {
        std::vector<render::SplatInstance> learnt{instance};
        learnt[0].neural = &neural;
        REQUIRE(raster->render(camera, learnt, settings, into));
    };
    // Nothing learnt: the view's network adds nothing, and the other is the base.
    render::RenderTargets a, b;
    drawnBy(*whole, a);
    drawnBy(*split, b);
    auto same = render::compareImages(*gpu->library, b.colour, a.colour, settings.width, settings.height);
    REQUIRE(same);
    CHECK(same->max <= 1);

    render::NeuralFitSettings how;
    how.width = settings.width;
    how.height = settings.height;
    double first = 0.0;
    double last = 0.0;
    for (uint32_t k = 0; k < 400; ++k) {
        const double at = 2.39996323 * k;
        const bool ask = k == 0 || k == 399;
        auto stepped = fit->step(*raster,
                                 render::Camera::lookingAt({6.0 * std::sin(at), 1.0, 6.0 * std::cos(at)}, {0.0, 0.0, 0.0}),
                                 instance, *split, how, ask);
        if (!stepped) FAIL(stepped.error().toString());
        if (*stepped) {
            (k == 0 ? first : last) = **stepped;
        }
    }
    std::printf("  split fit: relMSE %.5f at the first step, %.5f at the four hundredth\n", first, last);
    CHECK(last < first * 0.7);
}

TEST_CASE("a neural scene's manifest written and read is the scene, and a split appearance says it is one",
          "[render][neural][scene]") {
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "athenea_test_scene.aths";
    std::filesystem::remove_all(directory);
    io::NeuralScene scene;
    scene.name = "hall";
    scene.upAxis = "Z";
    scene.metersPerUnit = 0.01;
    scene.source = "hall.usda";
    scene.licence = "CC BY 4.0";
    io::NeuralScenePiece floor;
    floor.id = "floor1";
    floor.prim = "/root/floor1";
    floor.cloud = "pieces/floor1.usda";
    floor.gaussians = 1234;
    floor.boundsMin = {-1.0F, -2.0F, 0.0F};
    floor.boundsMax = {3.0F, 4.0F, 5.0F};
    floor.transform[3] = 7.5;
    floor.appearance = "pieces/floor1.athn";
    floor.split = true;
    floor.revision = 3;
    floor.movedAt = 1;
    floor.paintedAt = 3;
    floor.origin = "discovered";
    floor.foundBy = "door";
    floor.foundAt = 12.5;
    io::NeuralSceneFit fit;
    fit.steps = 3000;
    fit.against = "harmonics";
    fit.relMse = 0.00042;
    fit.p99 = 6.5;
    fit.max = 31;
    floor.fit = fit;
    io::NeuralScenePiece prop;
    prop.id = "prop";
    prop.cloud = "pieces/prop.ply";
    prop.gaussians = 10;
    prop.boundsMin = {-9.0F, 0.0F, 0.0F};
    prop.boundsMax = {0.0F, 1.0F, 1.0F};
    prop.isStatic = false;
    scene.pieces = {floor, prop};
    scene.lights.push_back({"sun", "distant", {0.0F, 0.6F, 0.8F}, {3.0F, 2.9F, 2.6F}});
    scene.cameras.push_back({"door", {8.0, 0.0, 1.7}, {0.0, 0.0, 2.0}, {0.0, 0.0, 1.0}, 24.0});
    scene.snapshots.push_back({12.5, "door", {"floor1"}});
    REQUIRE(io::writeNeuralScene(directory, scene));
    CHECK(std::filesystem::is_directory(directory / "pieces"));

    auto read = io::readNeuralScene(directory);
    if (!read) FAIL(read.error().toString());
    CHECK(read->name == "hall");
    CHECK(read->upAxis == "Z");
    CHECK(read->metersPerUnit == 0.01);
    CHECK(read->licence == "CC BY 4.0");
    REQUIRE(read->pieces.size() == 2);
    const io::NeuralScenePiece& a = read->pieces[0];
    CHECK(a.id == "floor1");
    CHECK(a.prim == "/root/floor1");
    CHECK(a.gaussians == 1234);
    CHECK(a.boundsMax[2] == 5.0F);
    CHECK(a.transform[3] == 7.5);
    CHECK(a.appearance == "pieces/floor1.athn");
    CHECK(a.split);
    CHECK(!a.lit);
    CHECK(a.revision == 3);
    CHECK(a.movedAt == 1);
    CHECK(a.paintedAt == 3);
    CHECK(a.origin == "discovered");
    CHECK(a.foundBy == "door");
    CHECK(a.foundAt == 12.5);
    CHECK(a.isStatic);
    REQUIRE(a.fit);
    CHECK(a.fit->steps == 3000);
    CHECK(a.fit->against == "harmonics");
    CHECK(a.fit->max == 31);
    const io::NeuralScenePiece& b = read->pieces[1];
    CHECK(b.appearance.empty());
    CHECK(!b.fit);
    CHECK(!b.isStatic);
    CHECK(b.origin == "converted");
    REQUIRE(read->snapshots.size() == 1);
    CHECK(read->snapshots[0].time == 12.5);
    CHECK(read->snapshots[0].pieces == std::vector<std::string>{"floor1"});
    // The scene's box is the box round its pieces'.
    std::array<float, 3> low{};
    std::array<float, 3> high{};
    read->bounds(low, high);
    CHECK(low[0] == -9.0F);
    CHECK(high[1] == 4.0F);
    REQUIRE(read->lights.size() == 1);
    CHECK(read->lights[0].kind == "distant");
    CHECK(read->lights[0].towards[2] == 0.8F);
    REQUIRE(read->cameras.size() == 1);
    CHECK(read->cameras[0].name == "door");
    CHECK(read->cameras[0].focal == 24.0);
    CHECK(read->piece("prop") != nullptr);
    CHECK(read->piece("none") == nullptr);

    // A file that is not one, and a version this does not read, are refused with a reason.
    {
        std::ofstream other(directory / "scene.json", std::ios::trunc);
        other << "{\"format\": \"athenea-neural-scene\", \"version\": 99}";
    }
    auto newer = io::readNeuralScene(directory);
    CHECK(!newer);

    // A split appearance: two networks' weights, and the flag that says so.
    io::NeuralFile file;
    file.count = 2;
    file.stride = render::kNeuralStride;
    file.wordsEach = render::kNeuralWords;
    file.split = true;
    file.latent.assign(size_t{2} * render::kNeuralWords, 7u);
    file.weights.assign(size_t{render::kNeuralWeights} * 2, 0.25F);
    const std::filesystem::path path = directory / "pieces" / "floor1.athn";
    REQUIRE(io::writeNeural(path, file));
    auto back = io::readNeural(path);
    if (!back) FAIL(back.error().toString());
    CHECK(back->split);
    CHECK(!back->lit);
    CHECK(back->weights.size() == size_t{render::kNeuralWeights} * 2);
    std::filesystem::remove_all(directory);
}

TEST_CASE("a look finds the gaussians its frame is made of, a second look from there finds none, and the scene "
          "drawn as it is known is that look's frame",
          "[render][gpu][neural][scene][look]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    test::CloudBuilder builder = test::randomCloud(6000, 21, 0.02F, 0.12F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    const render::Camera front = render::Camera::lookingAt({0.5, 0.5, 6.0}, {0.0, 0.0, 0.0});
    const render::Camera back = render::Camera::lookingAt({-0.5, 0.3, -6.0}, {0.0, 0.0, 0.0});
    auto seen = discovery->record(cloud->count);
    REQUIRE(seen);

    const auto look = [&](const render::Camera& camera, uint32_t stamp, render::RenderTargets& frame) {
        auto given = discovery->zeros(cloud->count);
        REQUIRE(given);
        render::RenderSettings settings;
        settings.width = 320;
        settings.height = 240;
        settings.contribution = &*given;
        settings.contributionMode = 1;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
        auto counted = discovery->mark(*given, 0, *seen, cloud->count, stamp, 0.05F);
        REQUIRE(counted);
        return *counted;
    };
    render::RenderTargets first, again, other;
    const render::LookCount one = look(front, 1, first);
    const render::LookCount two = look(front, 2, again);
    const render::LookCount three = look(back, 3, other);
    auto afterOne = discovery->known(*seen, cloud->count, 1);
    auto afterAll = discovery->known(*seen, cloud->count, 0xFFFFFFFFu);
    REQUIRE(afterOne);
    REQUIRE(afterAll);
    std::printf("  looks: %u found, then %u found and %u held, then from behind %u found and %u held; the scene holds "
                "%u then %u of %u\n",
                one.found, two.found, two.again, three.found, three.again, *afterOne, *afterAll, cloud->count);
    CHECK(one.found > 100);
    CHECK(one.again == 0);
    // The same look again finds nothing new, and what it sees is what the first found.
    CHECK(two.found == 0);
    CHECK(two.again == one.found);
    // Nothing that look stamped is the first look's any less.
    CHECK(*afterOne == one.found);
    CHECK(*afterAll == one.found + three.found);
    CHECK(*afterAll <= cloud->count);

    // The scene as the first look knew it, from where it looked: its own frame, but for the gaussians
    // that gave it less than counts as seen.
    auto known = discovery->masked(*cloud, *seen, 1);
    REQUIRE(known);
    const render::SplatInstance knownInstance{&*known, render::Mat4::identity()};
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    render::RenderTargets drawn;
    REQUIRE(raster->render(front, std::span<const render::SplatInstance>(&knownInstance, 1), settings, drawn));
    auto same = render::compareImages(*gpu->library, drawn.colour, first.colour, settings.width, settings.height);
    REQUIRE(same);
    std::printf("  the scene as the first look knew it against that look's frame: p99 %u, max %u\n", same->p99, same->max);
    CHECK(same->p99 <= 4);
}

TEST_CASE("a frame's pixels become gaussians, and a second frame from there finds the scene holds them",
          "[render][gpu][neural][scene][discover]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    test::CloudBuilder builder = test::randomCloud(6000, 21, 0.05F, 0.2F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    const render::Camera camera = render::Camera::lookingAt({0.5, 0.5, 6.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    render::RenderTargets frame;
    REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);

    // Nothing known: a gaussian a pixel that holds a surface.
    auto first = discovery->fromFrame(frame.colour, frame.depth, settings.width, settings.height, projection, nullptr,
                                      nullptr);
    if (!first) FAIL(first.error().toString());
    CHECK(first->surfaces > 1000);
    CHECK(first->held == 0);
    CHECK(first->made == first->surfaces);
    CHECK(first->records.size() == size_t{first->made} * render::kFrameRecord);

    // The same frame against itself as what is known: every surface is held, and nothing is made.
    auto again = discovery->fromFrame(frame.colour, frame.depth, settings.width, settings.height, projection,
                                      &frame.colour, &frame.depth);
    if (!again) FAIL(again.error().toString());
    std::printf("  a frame of %u surfaces: %u gaussians made; against itself, %u held and %u made\n", first->surfaces,
                first->made, again->held, again->made);
    CHECK(again->surfaces == first->surfaces);
    CHECK(again->held == again->surfaces);
    CHECK(again->made == 0);
}

TEST_CASE("what a frame sees through is taken out of the scene", "[render][gpu][neural][scene][gone]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    test::CloudBuilder builder = test::randomCloud(6000, 21, 0.05F, 0.2F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    const render::Camera camera = render::Camera::lookingAt({0.5, 0.5, 6.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    const uint32_t pixels = settings.width * settings.height;
    render::RenderTargets known;
    REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, known));

    // A frame with nothing in it: every pixel where the scene has something is seen through.
    gpu::Buffer emptyColour = floats(*gpu->device, pixels, 4, "empty.colour");
    gpu::Buffer emptyDepth = floats(*gpu->device, pixels, 1, "empty.depth");
    {
        auto zero = gpu::ComputeKernel::create(*gpu->library, "athenea/neural/scene_discover", "discoverClear");
        if (!zero) FAIL(zero.error().toString());
        gpu::CommandBatch batch(*gpu->device);
        for (const auto& [buffer, words] : {std::pair<gpu::Buffer*, uint32_t>{&emptyColour, pixels * 4},
                                            std::pair<gpu::Buffer*, uint32_t>{&emptyDepth, pixels}}) {
            zero->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["contribution"].setBinding(known.depth.rhi());
                cursor["seen"].setBinding(emptyDepth.rhi());
                cursor["counters"].setBinding(emptyDepth.rhi());
                cursor["seenRead"].setBinding(known.depth.rhi());
                cursor["lifted"].setBinding(known.depth.rhi());
                cursor["positions"].setBinding(known.colour.rhi());
                cursor["positionsOut"].setBinding(emptyColour.rhi());
                cursor["cleared"].setBinding(buffer->rhi());
                cursor["params"]["count"].setData(words);
            });
        }
        REQUIRE(batch.submit(true));
    }
    // Layer by layer: the mask carried to the gaussians by the blend, those it fell on stamped gone, and
    // the scene drawn again without them -- until the frame sees through nothing more.
    auto gone = discovery->record(cloud->count);
    REQUIRE(gone);
    scene::GpuSplats standing = *cloud;
    uint32_t firstThrough = 0;
    uint32_t taken = 0;
    uint32_t passes = 0;
    for (; passes < 12; ++passes) {
        instance.splats = &standing;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, known));
        uint32_t through = 0;
        auto mask = discovery->seenThrough(emptyColour, emptyDepth, settings.width, settings.height, known.colour,
                                           known.depth, 0.03F, through);
        if (!mask) FAIL(mask.error().toString());
        if (passes == 0) {
            firstThrough = through;
        }
        if (through == 0) {
            break;
        }
        auto lift = render::makeSplatLift(*gpu->device, cloud->count);
        REQUIRE(lift);
        render::RenderSettings carrying = settings;
        render::liftWhileBlending(carrying, *lift, *mask);
        render::RenderTargets carried;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), carrying, carried));
        auto marked = discovery->markGone(lift->sums, 0, *gone, cloud->count, passes + 1);
        REQUIRE(marked);
        if (passes == 0) {
            // What is gone is not found gone again.
            auto twice = discovery->markGone(lift->sums, 0, *gone, cloud->count, 99);
            REQUIRE(twice);
            CHECK(*twice == 0);
        }
        if (*marked == 0) {
            break;
        }
        taken += *marked;
        auto kept = discovery->without(*cloud, *gone);
        REQUIRE(kept);
        standing = std::move(*kept);
    }
    CHECK(firstThrough > 1000);
    CHECK(taken > 100);
    const uint32_t through = firstThrough;
    const uint32_t* marked = &taken;

    // Without them the frame is all but empty where it was full.
    const render::SplatInstance left{&standing, render::Mat4::identity()};
    render::RenderTargets after;
    render::RenderTargets full;
    const render::SplatInstance whole{&*cloud, render::Mat4::identity()};
    REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&whole, 1), settings, full));
    REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&left, 1), settings, after));
    auto before = render::imageStats(*gpu->library, full.colour, settings.width, settings.height);
    auto now = render::imageStats(*gpu->library, after.colour, settings.width, settings.height);
    REQUIRE(before);
    REQUIRE(now);
    std::printf("  %u pixels seen through, %u gaussians gone of %u; the frame's cover %.3f before and %.3f after\n", through,
                *marked, cloud->count, before->mean[3], now->mean[3]);
    CHECK(now->mean[3] < before->mean[3] * 0.2);
}

namespace {

/// The path tracer's planes (technique::PathAux) for a frame of `pixels`, each plane one value all over:
/// no normal (a disc then faces the eye), a matt surface of row 1, and -- with `pane` -- one that lets
/// everything through, with `behind` standing behind every pixel and half the light lost on the way.
gpu::Buffer planesOf(gpu::Device& device, uint32_t pixels, bool pane, std::array<float, 3> behind = {}) {
    std::vector<std::array<float, 4>> planes(size_t{pixels} * 8, std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F});
    const auto fill = [&](uint32_t plane, std::array<float, 4> value) {
        std::fill(planes.begin() + size_t{plane} * pixels, planes.begin() + size_t{plane + 1} * pixels, value);
    };
    fill(2, {0.0F, 1.0F, pane ? 1.0F : 0.0F, 1.0F});
    fill(3, {1.0F, 1.0F, 1.0F, 1.0F});
    if (pane) {
        fill(4, {behind[0], behind[1], behind[2], 1.0F});
        fill(5, {0.5F, 0.5F, 0.5F, 1.0F});
        fill(6, {1.0F, 1.0F, 1.0F, 1.0F});
    }
    auto made = gpu::Buffer::fromSpan<std::array<float, 4>>(device, std::span<const std::array<float, 4>>(planes),
                                                            "test.planes");
    REQUIRE(made);
    return std::move(*made);
}

}   // namespace

TEST_CASE("a render's look makes a gaussian a cell of its surface's material, as wide as the cell is at its distance",
          "[render][gpu][neural][scene][shoot]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    test::CloudBuilder builder = test::randomCloud(6000, 21, 0.05F, 0.2F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    settings.depth = render::RenderSettings::Depth::Threshold;
    settings.depthThreshold = 0.5F;
    const uint32_t pixels = settings.width * settings.height;
    const gpu::Buffer matt = planesOf(*gpu->device, pixels, false);

    // The mean size of the gaussians a frame is made of, where it is covered: the cloud drawn by its sizes.
    const auto meanSize = [&](const scene::GpuSplats& made, const render::Camera& from) {
        auto sized = discovery->sizes(made);
        REQUIRE(sized);
        render::SplatInstance bySize{&made, render::Mat4::identity()};
        bySize.neural = &*sized;
        render::RenderTargets drawn;
        REQUIRE(raster->render(from, std::span<const render::SplatInstance>(&bySize, 1), settings, drawn));
        auto stats = render::imageStats(*gpu->library, drawn.colour, settings.width, settings.height, 0, 0,
                                        settings.width, settings.height);
        REQUIRE(stats);
        return std::pair<double, double>{stats->mean[0] / std::max(stats->mean[3], 1.0e-6), stats->mean[3]};
    };
    const auto lookFrom = [&](double distance, uint32_t footprint, render::RenderLook& look, render::RenderTargets& frame) {
        const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, distance}, {0.0, 0.0, 0.0});
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
        render::RenderLookSettings how;
        how.footprint = footprint;
        auto made = discovery->fromRender(frame.colour, frame.depth, matt, settings.width, settings.height,
                                          render::projectionFor(camera, settings.width, settings.height), {}, how);
        if (!made) FAIL(made.error().toString());
        look = std::move(*made);
        return camera;
    };
    const auto uploaded = [&](const render::RenderLook& look) {
        loader->keepFaint(true);
        auto made = loader->upload(look.records, look.made, render::Discovery::renderEncoding(), "test", 0);
        loader->keepFaint(false);
        if (!made) FAIL(made.error().toString());
        REQUIRE(made->count == look.made);
        return std::move(*made);
    };

    render::RenderLook near;
    render::RenderTargets nearFrame;
    const render::Camera nearCamera = lookFrom(6.0, 1, near, nearFrame);
    CHECK(near.surfaces > 1000);
    CHECK(near.made == near.surfaces);
    CHECK(near.held == 0);
    CHECK(near.behind == 0);
    const scene::GpuSplats nearCloud = uploaded(near);
    const auto [nearSize, nearCover] = meanSize(nearCloud, nearCamera);
    auto frameStats = render::imageStats(*gpu->library, nearFrame.colour, settings.width, settings.height, 0, 0,
                                         settings.width, settings.height);
    REQUIRE(frameStats);

    // TWICE AS FAR, TWICE AS WIDE: the same thing looked at from twice the distance.
    render::RenderLook far;
    render::RenderTargets farFrame;
    const render::Camera farCamera = lookFrom(12.0, 1, far, farFrame);
    const scene::GpuSplats farCloud = uploaded(far);
    const auto [farSize, farCover] = meanSize(farCloud, farCamera);

    // A LOOK AT CELLS OF FOUR PIXELS: a sixteenth of the gaussians, four times as wide.
    render::RenderLook coarse;
    render::RenderTargets coarseFrame;
    lookFrom(6.0, 4, coarse, coarseFrame);
    const scene::GpuSplats coarseCloud = uploaded(coarse);
    const auto [coarseSize, coarseCover] = meanSize(coarseCloud, nearCamera);

    std::printf("  a look from 6: %u gaussians of mean size %.5f covering %.3f of the frame (the frame itself %.3f); from "
                "12: %u of %.5f (%.2f times); at cells of 4 pixels: %u of %.5f (%.2f times), covering %.3f\n",
                near.made, nearSize, nearCover, frameStats->mean[3], far.made, farSize, farSize / nearSize, coarse.made,
                coarseSize, coarseSize / nearSize, coarseCover);
    CHECK(nearCover > 0.8 * frameStats->mean[3]);
    CHECK(farSize / nearSize > 1.7);
    CHECK(farSize / nearSize < 2.6);   // the far look also sees the cloud's far side more nearly edge on
    CHECK(coarse.made * 8 < near.made);
    CHECK(coarseSize / nearSize > 3.0);
    CHECK(coarseSize / nearSize < 5.0);
    CHECK(coarseCover > 0.7 * frameStats->mean[3]);

    // Against itself as what is known, nothing is made.
    render::KnownScene known;
    known.colour = &nearFrame.colour;
    known.depth = &nearFrame.depth;
    auto again = discovery->fromRender(nearFrame.colour, nearFrame.depth, matt, settings.width, settings.height,
                                       render::projectionFor(nearCamera, settings.width, settings.height), known, {});
    if (!again) FAIL(again.error().toString());
    CHECK(again->held == again->surfaces);
    CHECK(again->made == 0);

    // A PANE: the same frame with every pixel letting the eye through to one place behind. Two gaussians a
    // pixel -- the pane, and what is seen through it -- and the pane's own carry no picture.
    const gpu::Buffer pane = planesOf(*gpu->device, pixels, true, {0.0F, 0.0F, -3.0F});
    auto through = discovery->fromRender(nearFrame.colour, nearFrame.depth, pane, settings.width, settings.height,
                                         render::projectionFor(nearCamera, settings.width, settings.height), {}, {});
    if (!through) FAIL(through.error().toString());
    std::printf("  the same frame as a pane: %u surfaces, %u let the eye through, %u gaussians made, %u of them behind\n",
                through->surfaces, through->panes, through->made, through->behind);
    CHECK(through->panes == through->surfaces);
    CHECK(through->behind == through->surfaces);
    CHECK(through->made == 2 * through->surfaces);
}

TEST_CASE("a piece's gaussians are written and read as they are, and a scene says when each snapshot was",
          "[render][neural][scene][shoot]") {
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / "athenea-tests" / "shoot.aths";
    std::filesystem::create_directories(directory / "pieces");
    std::vector<float> records(size_t{5} * render::kRenderRecord);
    for (size_t k = 0; k < records.size(); ++k) {
        records[k] = static_cast<float>(k);
    }
    REQUIRE(io::writeGaussians(directory / "pieces/a.athg", 5, render::kRenderRecord, records));
    auto read = io::readGaussians(directory / "pieces/a.athg");
    if (!read) FAIL(read.error().toString());
    CHECK(read->count == 5);
    CHECK(read->floats == render::kRenderRecord);
    CHECK(read->records == records);

    io::NeuralScene scene;
    scene.name = "shoot";
    io::NeuralScenePiece piece;
    piece.id = "a";
    piece.cloud = "pieces/a.athg";
    piece.gaussians = 5;
    piece.origin = "discovered";
    piece.foundIn = 2;
    piece.footprint = 4;
    scene.pieces.push_back(piece);
    scene.snapshots.push_back({0.0, "camera", {}, 4, 8, "2026-11-12T12:00:00Z"});
    scene.snapshots.push_back({0.0, "camera", {"a"}, 16, 4, "2026-11-12T12:00:05Z"});
    REQUIRE(io::writeNeuralScene(directory, scene));
    auto back = io::readNeuralScene(directory);
    if (!back) FAIL(back.error().toString());
    REQUIRE(back->pieces.size() == 1);
    CHECK(back->pieces[0].foundIn == 2);
    CHECK(back->pieces[0].footprint == 4);
    REQUIRE(back->snapshots.size() == 2);
    CHECK(back->snapshots[0].samples == 4);
    CHECK(back->snapshots[0].footprint == 8);
    CHECK(back->snapshots[1].taken == "2026-11-12T12:00:05Z");
    CHECK(back->snapshots[1].pieces == std::vector<std::string>{"a"});
}

TEST_CASE("a scene cut for a view draws a coarse look's gaussians from far and the finer ones laid over them from near",
          "[render][gpu][neural][scene][shoot][lod]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    test::CloudBuilder builder = test::randomCloud(6000, 21, 0.05F, 0.2F);
    auto cloud = loader->upload(builder.raw, 3);
    REQUIRE(cloud);
    const render::SplatInstance instance{&*cloud, render::Mat4::identity()};
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    settings.depth = render::RenderSettings::Depth::Threshold;
    settings.depthThreshold = 0.5F;
    const uint32_t pixels = settings.width * settings.height;
    const gpu::Buffer matt = planesOf(*gpu->device, pixels, false);
    // Two looks at the same thing: one from far, whose gaussians are the coarse level, and one from half as
    // far, whose gaussians are laid over those.
    const auto lookFrom = [&](double distance, render::Projection& projection) {
        const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, distance}, {0.0, 0.0, 0.0});
        render::RenderTargets frame;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
        projection = render::projectionFor(camera, settings.width, settings.height);
        auto made = discovery->fromRender(frame.colour, frame.depth, matt, settings.width, settings.height, projection,
                                          {}, {});
        if (!made) FAIL(made.error().toString());
        loader->keepFaint(true);
        auto uploaded = loader->upload(made->records, made->made, render::Discovery::renderEncoding(), "test", 0);
        loader->keepFaint(false);
        if (!uploaded) FAIL(uploaded.error().toString());
        return std::move(*uploaded);
    };
    render::Projection fromFar;
    render::Projection fromNear;
    const scene::GpuSplats coarse = lookFrom(12.0, fromFar);
    const scene::GpuSplats fine = lookFrom(6.0, fromNear);
    // The coarse ones were all refined by look 2; each fine one keeps the width of what it was laid over --
    // a coarse gaussian's, which at the cloud's middle is a pixel of the far look: 12 over its focal length.
    const std::vector<uint32_t> two(coarse.count, 2u);
    auto refined = discovery->record(coarse.count, two);
    REQUIRE(refined);
    const std::vector<std::array<float, 4>> widths(
        fine.count, std::array<float, 4>{0.5F, 0.5F, 0.5F, static_cast<float>(12.0 / fromFar.focalX)});
    auto parent = gpu::Buffer::fromSpan<std::array<float, 4>>(
        *gpu->device, std::span<const std::array<float, 4>>(widths), "test.parent");
    REQUIRE(parent);
    const auto drawn = [&](const scene::GpuSplats& of, const gpu::Buffer* refinedOf, const gpu::Buffer* parentOf,
                           const render::Projection& projection, float threshold, uint32_t upTo = 0) {
        auto cut = discovery->cut(of, nullptr, refinedOf, parentOf, projection, threshold, upTo, true);
        if (!cut) FAIL(cut.error().toString());
        return cut->count;
    };
    const uint32_t coarseFar = drawn(coarse, &*refined, nullptr, fromFar, 1.5F);
    const uint32_t fineFar = drawn(fine, nullptr, &*parent, fromFar, 1.5F);
    const uint32_t coarseNear = drawn(coarse, &*refined, nullptr, fromNear, 1.5F);
    const uint32_t fineNear = drawn(fine, nullptr, &*parent, fromNear, 1.5F);
    std::printf("  a coarse look of %u gaussians refined by a look of %u: cut at 1.5 pixels from far, %u coarse and %u "
                "fine are drawn; from near, %u coarse and %u fine\n",
                coarse.count, fine.count, coarseFar, fineFar, coarseNear, fineNear);
    CHECK(coarseFar > coarse.count * 9 / 10);    // from far the coarse level stands for the fine
    CHECK(fineFar < fine.count / 10);
    CHECK(coarseNear < coarse.count / 10);       // and from near the fine one is drawn in its place
    CHECK(fineNear > fine.count * 9 / 10);
    // With no threshold, the finest there is, everywhere; and as the scene stood before look 2 refined
    // anything, the coarse ones were the finest.
    CHECK(drawn(coarse, &*refined, nullptr, fromFar, 0.0F) == 0);
    CHECK(drawn(fine, nullptr, &*parent, fromFar, 0.0F) == fine.count);
    CHECK(drawn(coarse, &*refined, nullptr, fromNear, 1.5F, 1) == coarse.count);

    // A BUDGET: both pieces' spans in one histogram, and the threshold read off it draws no more than was
    // asked -- from far, where the coarse level is enough, a budget of the coarse count is met.
    auto histogram = discovery->cutHistogram();
    REQUIRE(histogram);
    REQUIRE(discovery->spans(coarse, nullptr, &*refined, nullptr, fromFar, 0, *histogram));
    REQUIRE(discovery->spans(fine, nullptr, nullptr, &*parent, fromFar, 0, *histogram));
    uint64_t said = 0;
    auto threshold = discovery->thresholdFor(*histogram, coarse.count + coarse.count / 10, &said);
    REQUIRE(threshold);
    const uint32_t within = drawn(coarse, &*refined, nullptr, fromFar, *threshold) +
                            drawn(fine, nullptr, &*parent, fromFar, *threshold);
    std::printf("  a budget of %u from far: a threshold of %.2f pixels, said to draw %llu, drawing %u\n",
                coarse.count + coarse.count / 10, *threshold, static_cast<unsigned long long>(said), within);
    CHECK(within <= coarse.count + coarse.count / 10 + coarse.count / 20);   // the bins are a twentieth of an octave wide
    CHECK(within > coarse.count / 2);
    // MOVED TOWARDS ITS PARENT: gathered in the band, a fine gaussian is as many as before and wider.
    auto stepped = discovery->cut(fine, nullptr, nullptr, &*parent, fromNear, 2.0F, 0, true, 0.0F);
    auto moved = discovery->cut(fine, nullptr, nullptr, &*parent, fromNear, 2.0F, 0, true, 1.0F);
    REQUIRE(stepped);
    REQUIRE(moved);
    CHECK(moved->count == stepped->count);
}

#include "athenea/render/SceneGlass.h"

TEST_CASE("a found scene's glass shows what stands behind it from where the eye is, and something else from elsewhere",
          "[render][gpu][neural][scene][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto discovery = render::Discovery::create(*gpu->library);
    auto glass = render::SceneGlass::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!discovery) FAIL(discovery.error().toString());
    if (!glass) FAIL(glass.error().toString());
    // A ROOM AND A SLAB OF GLASS IN IT, as records a render's look would have left: a wall behind the slab
    // that is red on one side and blue on the other, a grey wall behind the eye, and the slab's two faces,
    // whose stored picture is green -- what some look saw in it, and no view's here.
    const uint32_t stride = render::kRenderRecord;
    std::vector<float> records;
    std::vector<std::array<float, 4>> made;
    const auto shown = [](float light) {
        const float encoded = light <= 0.0031308F ? 12.92F * light : 1.055F * std::pow(light, 1.0F / 2.4F) - 0.055F;
        return (encoded - 0.5F) / 0.28209479F;
    };
    const auto disc = [&](float x, float y, float z, float size, std::array<float, 3> light, float facing, bool isGlass) {
        const size_t at = records.size();
        records.resize(at + stride, 0.0F);
        float* r = &records[at];
        r[0] = x; r[1] = y; r[2] = z;
        r[3] = 4.0F;                                    // opaque
        r[4] = std::log(size); r[5] = std::log(size); r[6] = std::log(size * 0.05F);
        r[7] = 1.0F;                                    // no turn: the thin axis is z
        for (int c = 0; c < 3; ++c) r[11 + c] = shown(light[c]);
        r[16] = facing;                                 // the surface's normal
        r[18] = isGlass ? 0.0F : 1.0F;
        r[19] = isGlass ? 1.0F : 0.0F;                  // what it transmits
        r[render::kRenderRecordIndex] = isGlass ? 1.5F : 1.0F;
        for (int c = 0; c < 3; ++c) r[render::kRenderRecordTint + c] = 1.0F;
        made.push_back({1.0F, 1.0F, 1.0F, isGlass ? 1.5F : 0.0F});
    };
    const std::array<float, 3> red{0.8F, 0.05F, 0.05F};
    const std::array<float, 3> blue{0.05F, 0.05F, 0.8F};
    const std::array<float, 3> grey{0.3F, 0.3F, 0.3F};
    const std::array<float, 3> green{0.05F, 0.8F, 0.05F};
    for (int j = -40; j <= 40; ++j) {
        for (int i = -40; i <= 40; ++i) {
            const float x = static_cast<float>(i) * 0.15F;
            const float y = static_cast<float>(j) * 0.15F;
            disc(x, y, -4.0F, 0.1F, x < 0.0F ? red : blue, 1.0F, false);
            disc(x, y, 9.0F, 0.1F, grey, -1.0F, false);
        }
    }
    uint32_t panes = 0;
    for (int j = -20; j <= 20; ++j) {
        for (int i = -20; i <= 20; ++i) {
            const float x = static_cast<float>(i) * 0.05F;
            const float y = static_cast<float>(j) * 0.05F;
            disc(x, y, 0.25F, 0.035F, green, 1.0F, true);
            disc(x, y, -0.25F, 0.035F, green, -1.0F, true);
            panes += 2;
        }
    }
    const uint32_t count = static_cast<uint32_t>(records.size() / stride);
    io::RawSplats raw;
    raw.source = "test";
    raw.count = count;
    raw.encoding = render::Discovery::renderEncoding();
    raw.records = std::move(records);
    loader->keepFaint(true);
    auto piece = loader->upload(raw, 0);
    loader->keepFaint(false);
    if (!piece) FAIL(piece.error().toString());
    REQUIRE(piece->count == count);
    auto matter = gpu::Buffer::fromSpan<std::array<float, 4>>(
        *gpu->device, std::span<const std::array<float, 4>>(made), "test.matter");
    REQUIRE(matter);
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    // The middle of the frame, which is the slab's: its colour over its cover.
    const auto middle = [&](const render::RenderTargets& frame) {
        auto stats = render::imageStats(*gpu->library, frame.colour, settings.width, settings.height, 150, 110, 170, 130);
        REQUIRE(stats);
        const double cover = std::max(stats->mean[3], 1.0e-6);
        return std::array<double, 3>{stats->mean[0] / cover, stats->mean[1] / cover, stats->mean[2] / cover};
    };
    // The scene cut for a view as `neural-scene render --glass` cuts it: one cloud, what each gaussian is
    // made of gathered beside it, and the glass seen from the view's eye.
    const auto seenFrom = [&](double x, render::GlassSeen& said, std::array<double, 3>* before = nullptr) {
        const render::Camera camera = render::Camera::lookingAt({x, 0.0, 5.0}, {0.0, 0.0, 0.0});
        const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
        auto plan = discovery->planCut(*piece, nullptr, nullptr, nullptr, projection, 0.0F);
        if (!plan) FAIL(plan.error().toString());
        REQUIRE(plan->drawn == count);
        auto cut = discovery->cutCloud(*piece, plan->drawn);
        if (!cut) FAIL(cut.error().toString());
        auto room = glass->room(*cut);
        if (!room) FAIL(room.error().toString());
        REQUIRE(discovery->gatherCut(*piece, nullptr, *plan, projection, 0.0F, 0.0F, *cut, 0));
        REQUIRE(glass->gather(*piece, *matter, *plan, *cut, *room, 0));
        const render::SplatInstance instance{&*cut, render::Mat4::identity()};
        if (before != nullptr) {
            render::RenderTargets frame;
            REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
            *before = middle(frame);
        }
        auto seen = glass->see(projection, *cut, *room);
        if (!seen) FAIL(seen.error().toString());
        said = *seen;
        render::RenderTargets frame;
        REQUIRE(raster->render(camera, std::span<const render::SplatInstance>(&instance, 1), settings, frame));
        return middle(frame);
    };
    render::GlassSeen fromRight;
    render::GlassSeen fromLeft;
    std::array<double, 3> picture{};
    const std::array<double, 3> right = seenFrom(2.5, fromRight, &picture);
    const std::array<double, 3> left = seenFrom(-2.5, fromLeft);
    std::printf("  a slab of %u glass gaussians in a room of %u: its picture (%.2f %.2f %.2f); seen from the right "
                "(%.2f %.2f %.2f), %u rays through it met the room and %u mirrored; from the left (%.2f %.2f %.2f); "
                "the tree %.1f ms, the rays %.1f ms\n",
                panes, count - panes, picture[0], picture[1], picture[2], right[0], right[1], right[2],
                fromRight.behind, fromRight.mirrored, left[0], left[1], left[2], fromRight.treeMs, fromRight.seenMs);
    CHECK(picture[1] > 0.5);                         // drawn as it was stored, the slab is its picture
    CHECK(fromRight.glass == panes);
    CHECK(fromRight.behind > panes * 9 / 10);        // every ray through it finds the wall behind
    CHECK(fromRight.mirrored > panes * 2 / 3);       // and most mirrored ones the wall behind the eye, as far as it reaches
    // From the right the eye's ray through the slab lands on the wall's left, which is red; from the left,
    // on its right, which is blue. Nothing of the green picture is left in either.
    CHECK(right[0] > 0.5);
    CHECK(right[2] < 0.2);
    CHECK(right[1] < 0.15);
    CHECK(left[2] > 0.5);
    CHECK(left[0] < 0.2);
    CHECK(left[1] < 0.15);
}
