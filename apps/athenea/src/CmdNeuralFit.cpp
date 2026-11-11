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
#include "athenea/io/Exr.h"
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
    auto neural = render::makeNeuralCloud(library, *cloud);
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

    render::RenderSettings settings;
    settings.width = how.width;
    settings.height = how.height;
    std::vector<render::SplatInstance> byHarmonics{instance};
    std::vector<render::SplatInstance> byLatents{instance};
    byLatents[0].neural = &*neural;
    // How far the cloud by its latents is from the cloud by its harmonics, over the held-out cameras.
    struct Measure {
        double relMse = 0.0;
        double p99 = 0.0;
        uint32_t max = 0;
    };
    const auto measure = [&]() -> Result<Measure> {
        Measure m;
        for (uint32_t k = 0; k < o.heldOut; ++k) {
            const render::Camera camera = cameraAt(k, 0.5);
            render::RenderTargets whole;
            render::RenderTargets ours;
            auto a = raster->render(camera, byHarmonics, settings, whole);
            if (!a) return std::move(a).error();
            auto c = raster->render(camera, byLatents, settings, ours);
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
    const auto write = [&](const char* what, const std::vector<render::SplatInstance>& instances,
                           uint32_t degree) -> Result<void> {
        if (o.output.empty()) return ok();
        render::RenderSettings s = settings;
        s.maxShDegree = degree;
        render::RenderTargets targets;
        auto drawn = raster->render(cameraAt(0, 0.5), instances, s, targets);
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
    out("before: relMSE %.5f, p99 %.1f, max %u (the base colour alone against the harmonics, %u held-out cameras)\n",
        before->relMse, before->p99, before->max, o.heldOut);
    if (auto w = write("harmonics", byHarmonics, 3); !w) return report(w.error());
    if (auto w = write("base", byHarmonics, 0); !w) return report(w.error());

    const auto start = std::chrono::steady_clock::now();
    const unsigned every = std::max(o.steps / 10, 1u);
    for (unsigned k = 0; k < o.steps; ++k) {
        const bool ask = k == 0 || (k + 1) % every == 0;
        auto loss = fit->step(*raster, cameraAt(k, 0.0), instance, *neural, how, ask);
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
    if (auto w = write("latents", byLatents, 3); !w) return report(w.error());
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
    cmd->add_option("-o,--output", o.output, "a prefix: the first held-out camera's frame by the harmonics, the base and the latents");
    cmd->callback([options] {
        if (const int code = run(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace athenea::cli
