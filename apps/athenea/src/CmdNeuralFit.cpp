// Copyright (c) 2026 jesus luque.
//
// athenea neural-fit: a cloud's harmonics learnt by a latent a gaussian and
// one network (render/NeuralFit.h), and how near the cloud drawn from them is
// to the cloud drawn by its harmonics, from cameras the fit did not use.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "aofx/Transform.h"
#include "athenea/io/Exr.h"
#include "athenea/light/LightTable.h"
#include "athenea/render/NeuralFit.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::cli {
namespace {

struct NeuralFitOptions {
    std::string         splats;
    unsigned            steps = 3000;
    std::string         size = "960x540";
    std::vector<double> target;
    std::vector<double> up{0.0, 1.0, 0.0};
    double              distance = 0.0;    ///< of every camera from the target; 0: from the cloud's size
    double              focal = 35.0;
    unsigned            degree = 3;
    unsigned            heldOut = 8;
    float               rateFeatures = 1.0e-2F;
    float               rateBase = 2.0e-3F;
    float               rateWeights = 4.0e-3F;
    bool                relight = false;   ///< a lit neural cloud, fitted to the engine's relighting under a moving sun
    float               sun = 3.0F;        ///< the sun's intensity
    std::string         output;            ///< a prefix: <prefix>_harmonics.exr, _base.exr, _latents.exr
};

int run(const NeuralFitOptions& o) {
    render::NeuralFitSettings how;
    if (std::sscanf(o.size.c_str(), "%ux%u", &how.width, &how.height) != 2 || how.width == 0 || how.height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return kExitFailure;
    }
    how.rateFeatures = o.rateFeatures;
    how.rateBase = o.rateBase;
    how.rateWeights = o.rateWeights;
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto fit = render::NeuralFit::create(library);
    if (!fit) return report(fit.error());
    auto cloud = scene::loadSplatFile(*loader, o.splats, o.degree);
    if (!cloud) return report(cloud.error());
    auto neural = render::makeNeuralCloud(library, *cloud, 1, o.relight);
    if (!neural) return report(neural.error());
    const render::SplatInstance instance{&*cloud, render::Mat4::identity(), {}};

    const auto d = [](float v) { return static_cast<double>(v); };
    const scene::Bounds& b = cloud->bounds;
    const render::Vec3 target = o.target.size() == 3
                                    ? render::Vec3{o.target[0], o.target[1], o.target[2]}
                                    : render::Vec3{(d(b.min[0]) + d(b.max[0])) * 0.5, (d(b.min[1]) + d(b.max[1])) * 0.5,
                                                   (d(b.min[2]) + d(b.max[2])) * 0.5};
    const double extent = std::max({d(b.max[0]) - d(b.min[0]), d(b.max[1]) - d(b.min[1]), d(b.max[2]) - d(b.min[2])});
    const double distance = o.distance > 0.0 ? o.distance : std::max(extent, 1.0e-3) * 1.4;
    const render::Vec3 up = o.up.size() == 3 ? render::Vec3{o.up[0], o.up[1], o.up[2]} : render::Vec3{0, 1, 0};
    // Cameras over a sphere round the target, no two alike and no part of it
    // left out: the R2 sequence (Roberts), whose two coordinates are powers
    // of the plastic number and so independent -- the golden angle for both
    // puts every camera on one turn of a spiral, which a fit then learns and
    // nothing else. Azimuth from one, the sine of the elevation from the
    // other, between a little below the equator and near the pole. `phase`
    // keeps the held-out cameras off the fit's.
    const auto cameraAt = [&](uint32_t k, double phase) {
        const double a = 6.283185307179586 * std::fmod(0.5 + (k + phase) * 0.7548776662466927, 1.0);
        const double h = -0.25 + 1.15 * std::fmod(0.5 + (k + phase) * 0.5698402909980532, 1.0);
        const double r = std::sqrt(std::max(1.0 - h * h, 0.0));
        // In a frame whose second axis is `up` when up is y; other ups are a turn of it.
        render::Vec3 offset{r * std::sin(a), h, r * std::cos(a)};
        if (std::abs(up.z) > 0.5) {
            offset = render::Vec3{r * std::sin(a), r * std::cos(a), h * (up.z > 0 ? 1.0 : -1.0)};
        }
        render::Camera camera = render::Camera::lookingAt(target + offset * distance, target, up);
        camera.lens.focal = o.focal;
        return camera;
    };

    // LIT: a sun from a direction, as the engine's relighting takes it (a
    // light table's records) and as the neural cloud is told it. A different
    // one each step, over the sphere by two more independent sequences;
    // `phase` keeps the held-out suns off the fit's as it does the cameras.
    auto table = light::LightTable::create(library);
    if (!table) return report(table.error());
    render::SplatLights lights;
    const auto sunAt = [&](uint32_t k, double phase) -> Result<render::NeuralFitLight> {
        light::Light lamp;
        lamp.kind = light::LightKind::Distant;
        lamp.lightToWorld = aofx::xform::rotationY(360.0 * std::fmod(0.5 + (k + phase) * 0.8191725134, 1.0)) *
                            aofx::xform::rotationX(-80.0 + 160.0 * std::fmod(0.5 + (k + phase) * 0.6710436067, 1.0));
        lamp.intensity = o.sun;
        lamp.shadow = false;
        ATHENEA_TRY(table->set(std::span<const light::Light>(&lamp, 1)));
        lights.records = &table->records();
        lights.count = table->count();
        render::NeuralFitLight told;
        told.teacher = &lights;
        // A distant light shines down its own -z: it comes from its +z.
        const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
        told.light.towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                              static_cast<float>(towards.z)};
        told.light.radiance = {o.sun, o.sun, o.sun};
        return told;
    };

    render::RenderSettings settings;
    settings.width = how.width;
    settings.height = how.height;
    std::vector<render::SplatInstance> byHarmonics{instance};
    std::vector<render::SplatInstance> byLatents{instance};
    byLatents[0].neural = &*neural;
    // How far the cloud by its latents is from the cloud by its harmonics, over the held-out cameras.
    // The held-out sun the instances above are drawn under, where the cloud is lit: kept here, since
    // an instance only points at its light.
    render::NeuralFitLight heldSun;
    struct Measure {
        double relMse = 0.0;
        double p99 = 0.0;
        uint32_t max = 0;
    };
    const auto measure = [&](uint32_t fewerDegrees = 3) -> Result<Measure> {
        Measure m;
        for (uint32_t k = 0; k < o.heldOut; ++k) {
            const render::Camera camera = cameraAt(k, 0.5);
            render::RenderTargets whole;
            render::RenderTargets ours;
            if (o.relight) {
                auto made = sunAt(k, 0.5);
                if (!made) return std::move(made).error();
                heldSun = *made;
                byHarmonics[0].relight = true;
                byLatents[0].neuralLight = &heldSun.light;
            }
            auto a = raster->render(camera, byHarmonics, settings, whole, {}, nullptr, o.relight ? &lights : nullptr);
            if (!a) return std::move(a).error();
            // The latents; or, asked for fewer harmonics, the cloud by those.
            render::RenderSettings other = settings;
            other.maxShDegree = fewerDegrees;
            auto c = raster->render(camera, fewerDegrees < 3 ? byHarmonics : byLatents, other, ours);
            if (!c) return std::move(c).error();
            auto hdr = render::compareHdr(library, ours.colour, whole.colour, settings.width, settings.height);
            if (!hdr) return std::move(hdr).error();
            auto codes = render::compareImages(library, ours.colour, whole.colour, settings.width, settings.height);
            if (!codes) return std::move(codes).error();
            m.relMse += hdr->relMse / o.heldOut;
            m.p99 += static_cast<double>(codes->p99) / o.heldOut;
            m.max = std::max(m.max, codes->max);
        }
        return m;
    };
    const auto write = [&](const std::string& what, std::vector<render::SplatInstance> instances, uint32_t degree,
                           uint32_t sun = 0) -> Result<void> {
        if (o.output.empty()) return ok();
        render::RenderSettings s = settings;
        s.maxShDegree = degree;
        render::RenderTargets targets;
        render::NeuralFitLight told;
        if (o.relight) {
            auto made = sunAt(sun, 0.5);
            if (!made) return std::move(made).error();
            told = *made;
            instances[0].relight = instances[0].neural == nullptr;
            instances[0].neuralLight = &told.light;
        }
        auto drawn = raster->render(cameraAt(0, 0.5), instances, s, targets, {}, nullptr,
                                    o.relight ? &lights : nullptr);
        if (!drawn) return std::move(drawn).error();
        auto colour = targets.colour.readAll<float>(**device);
        if (!colour) return std::move(colour).error();
        auto depth = targets.depth.readAll<float>(**device);
        if (!depth) return std::move(depth).error();
        const std::string path = o.output + "_" + what + ".exr";
        ATHENEA_TRY(io::writeExr(path, settings.width, settings.height, *colour, *depth));
        out("wrote %s\n", path.c_str());
        return ok();
    };

    auto before = measure();
    if (!before) return report(before.error());
    out("%u gaussians; %u numbers a gaussian in place of %u, and %u weights\n", cloud->count, render::kNeuralStride,
        3 * (cloud->restPerColour + 1), render::kNeuralWeights);
    out("before: relMSE %.5f, p99 %.1f, max %u (the cloud before it learns, against what it should be, %u held-out cameras)\n",
        before->relMse, before->p99, before->max, o.heldOut);
    // What the same cloud is with fewer harmonics: what eleven numbers a gaussian are to be held against.
    for (uint32_t degree = 1; !o.relight && degree < std::min(o.degree, 3u); ++degree) {
        auto fewer = measure(degree);
        if (!fewer) return report(fewer.error());
        out("harmonics of degree %u (%u numbers a gaussian): relMSE %.5f, p99 %.1f, max %u\n", degree,
            3 * (degree + 1) * (degree + 1), fewer->relMse, fewer->p99, fewer->max);
    }
    if (o.relight) {
        // The engine's relighting under three held-out suns: what the latents' frames are to be.
        for (uint32_t sun = 0; sun < 3; ++sun) {
            if (auto w = write("engine_sun" + std::to_string(sun), byHarmonics, 3, sun); !w) return report(w.error());
        }
    } else {
        if (auto w = write("harmonics", byHarmonics, 3); !w) return report(w.error());
        if (auto w = write("base", byHarmonics, 0); !w) return report(w.error());
    }

    const auto start = std::chrono::steady_clock::now();
    const unsigned every = std::max(o.steps / 10, 1u);
    for (unsigned k = 0; k < o.steps; ++k) {
        const bool ask = k == 0 || (k + 1) % every == 0;
        render::NeuralFitLight told;
        if (o.relight) {
            auto made = sunAt(k, 0.0);
            if (!made) return report(made.error());
            told = *made;
        }
        auto loss = fit->step(*raster, cameraAt(k, 0.0), instance, *neural, how, ask, o.relight ? &told : nullptr);
        if (!loss) return report(loss.error());
        if (*loss) {
            auto now = measure();
            if (!now) return report(now.error());
            out("step %u: relMSE %.5f; held out: relMSE %.5f, p99 %.1f, max %u\n", k + 1, **loss,
                now->relMse, now->p99, now->max);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto after = measure();
    if (!after) return report(after.error());
    out("after %u steps in %.1f s (%.1f ms a step): relMSE %.5f, p99 %.1f, max %u\n", o.steps, seconds,
        o.steps > 0 ? seconds * 1000.0 / o.steps : 0.0, after->relMse, after->p99, after->max);
    // What a frame costs by each: the best of twenty from the first held-out camera.
    const auto timed = [&](const std::vector<render::SplatInstance>& instances) -> Result<double> {
        double best = 1.0e30;
        render::RenderTargets targets;
        if (o.relight) {
            auto made = sunAt(0, 0.5);
            if (!made) return std::move(made).error();
            heldSun = *made;
        }
        for (int round = 0; round < 20; ++round) {
            auto stats = raster->render(cameraAt(0, 0.5), instances, settings, targets, {}, nullptr,
                                        o.relight ? &lights : nullptr);
            if (!stats) return std::move(stats).error();
            best = std::min(best, stats->totalMs);
        }
        return best;
    };
    auto harmonicsMs = timed(byHarmonics);
    if (!harmonicsMs) return report(harmonicsMs.error());
    auto latentsMs = timed(byLatents);
    if (!latentsMs) return report(latentsMs.error());
    out("a frame at %ux%u: %.2f ms by the harmonics, %.2f ms by the latents\n", settings.width, settings.height,
        *harmonicsMs, *latentsMs);
    if (o.relight) {
        for (uint32_t sun = 0; sun < 3; ++sun) {
            if (auto w = write("latents_sun" + std::to_string(sun), byLatents, 3, sun); !w) return report(w.error());
        }
    } else if (auto w = write("latents", byLatents, 3); !w) {
        return report(w.error());
    }
    return 0;
}

}   // namespace

void addNeuralFit(CLI::App& app) {
    auto options = std::make_shared<NeuralFitOptions>();
    NeuralFitOptions& o = *options;
    CLI::App* cmd = app.add_subcommand("neural-fit", "a cloud's harmonics learnt by a latent a gaussian and one network");
    cmd->add_option("--splats", o.splats, "a .ply / .splat / .spz / .sog file")->required();
    cmd->add_option("--steps", o.steps, "steps of the fit, a camera each");
    cmd->add_option("--size", o.size, "WIDTHxHEIGHT of the frames fitted and measured");
    cmd->add_option("--target", o.target, "what every camera looks at: x y z")->expected(3);
    cmd->add_option("--up", o.up, "up vector")->expected(3);
    cmd->add_option("--distance", o.distance, "every camera's distance from the target");
    cmd->add_option("--focal", o.focal, "focal length, mm");
    cmd->add_option("--degree", o.degree, "harmonic degree cap 0..3 of the cloud being learnt");
    cmd->add_option("--held-out", o.heldOut, "cameras the fit does not use, measured before and after");
    cmd->add_option("--rate-features", o.rateFeatures, "Adam's rate for the features");
    cmd->add_option("--rate-base", o.rateBase, "Adam's rate for the base colours");
    cmd->add_option("--rate-weights", o.rateWeights, "Adam's rate for the network");
    cmd->add_flag("--relight", o.relight, "a lit neural cloud, fitted to the engine's relighting of the cloud under a sun that moves");
    cmd->add_option("--sun", o.sun, "--relight: the sun's intensity");
    cmd->add_option("-o,--output", o.output, "a prefix: the first held-out camera's frame by the harmonics, the base and the latents");
    cmd->callback([options] {
        if (const int code = run(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace athenea::cli
