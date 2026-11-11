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

#include "athenea/io/NeuralFile.h"

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
