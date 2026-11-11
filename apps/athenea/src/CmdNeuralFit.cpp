// Copyright (c) 2026 jesus luque.
//
// athenea neural-fit: a cloud's harmonics learnt by a latent a gaussian and
// one network (render/NeuralFit.h), and how near the cloud drawn from them is
// to the cloud drawn by its harmonics, from cameras the fit did not use.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "aofx/Transform.h"
#include "athenea/io/Exr.h"
#include "athenea/io/NeuralFile.h"
#include "athenea/light/LightTable.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/render/NeuralFit.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/technique/SplatShadowMap.h"
#include "athenea/usd/Export.h"
#include <fstream>
#include <sstream>
#include "athenea/gpu/CommandBatch.h"

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
    float               viewSmooth = 0.0F;     ///< how hard a gaussian's colour a degree off is held to its colour
    float               rateDecay = 1.0F;      ///< what the three rates are multiplied by at the last step, falling to it step by step
    bool                relight = false;   ///< a lit neural cloud, fitted to the engine's relighting under a moving sun
    float               sun = 3.0F;        ///< the sun's intensity
    bool                shadows = false;   ///< --relight: the sun casts the cloud's shadow on itself
    std::string         reached;           ///< --shadows: the neural cloud is told what reaches each gaussian: ray | map
    unsigned            mapSide = 2048;    ///< --reached map: the shadow map's texels a side
    unsigned            mapCoefficients = 7;   ///< 1, 3, 5 or 7: the map's total and its Fourier pairs
    float               mapBias = 0.0F;    ///< how far behind a texel's nearest caster a receiver still stands on it (0: 2% of the slab)
    std::string         save;              ///< the fitted appearance, as a .athn
    std::string         pictures;          ///< a manifest of the same cloud baked under suns: what a lit fit learns from
    std::string         load;              ///< an appearance to start from (or, with --steps 0, only to measure)
    unsigned            turn = 0;          ///< --relight: frames of a sun going once round, written by the engine and by the latents
    std::string         output;            ///< a prefix: <prefix>_harmonics.exr, _base.exr, _latents.exr
};

/// FITTED TO PICTURES THE PATH TRACER MADE (`--pictures`): the same mesh baked
/// into the same gaussians under one sun after another (athenea mesh2splat,
/// its bake's bounces and shadows in each gaussian's harmonics), each cloud
/// the picture of its sun. A lit neural cloud of those gaussians is told each
/// sun and moved towards its cloud, and is then held against the clouds of
/// suns it was never shown. The manifest is a line a cloud: `train` or `held`,
/// the sun's turn about y and then about x in degrees (a distant light shining
/// down its own -z), and the USD file, beside the manifest.
int runPictures(const NeuralFitOptions& o) {
    render::NeuralFitSettings how;
    if (std::sscanf(o.size.c_str(), "%ux%u", &how.width, &how.height) != 2 || how.width == 0 || how.height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return kExitFailure;
    }
    how.rateFeatures = o.rateFeatures;
    how.rateBase = o.rateBase;
    how.rateWeights = o.rateWeights;
    how.viewSmooth = o.viewSmooth;
    if (!o.reached.empty() && o.reached != "map") {
        err("--pictures has no ray to cast: --reached map, or none\n");
        return kExitFailure;
    }
    const bool byMap = o.reached == "map";
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto fit = render::NeuralFit::create(library);
    if (!fit) return report(fit.error());
    auto table = light::LightTable::create(library);
    if (!table) return report(table.error());

    struct Picture {
        bool                              held = false;
        bool                              base = false;   ///< not a picture: the gaussians with their material alone
        double                            yaw = 0.0;
        double                            pitch = 0.0;
        std::unique_ptr<scene::GpuSplats> cloud;
    };
    std::vector<Picture> pictures;
    {
        std::ifstream in(o.pictures);
        if (!in) {
            err("cannot open %s\n", o.pictures.c_str());
            return kExitFailure;
        }
        const std::filesystem::path beside = std::filesystem::path(o.pictures).parent_path();
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream words(line);
            std::string kind;
            std::string file;
            Picture picture;
            if (!(words >> kind >> picture.yaw >> picture.pitch >> file)) {
                continue;
            }
            picture.held = kind == "held";
            picture.base = kind == "base";
            auto cloud = usd::uploadParticleField(library, *loader, beside / file);
            if (!cloud) return report(cloud.error());
            picture.cloud = std::make_unique<scene::GpuSplats>(std::move(*cloud));
            if (!picture.base && !pictures.empty() && !pictures.front().base &&
                picture.cloud->count != pictures.front().cloud->count) {
                err("%s holds %u gaussians and the first cloud %u: not the same gaussians\n", file.c_str(),
                    picture.cloud->count, pictures.front().cloud->count);
                return kExitFailure;
            }
            pictures.push_back(std::move(picture));
        }
    }
    std::vector<size_t> train;
    std::vector<size_t> held;
    // WHERE THE BASE STARTS. A lit cloud's base is what its gaussians are made
    // of; started from a picture it starts as that picture's light, shadow and
    // all, for the fit to unlearn. A `base` line names the same gaussians
    // converted with no bake (`athenea mesh2splat --no-bake`): their colour is
    // then the material's own.
    const scene::GpuSplats* material = nullptr;
    for (size_t k = 0; k < pictures.size(); ++k) {
        if (pictures[k].base) {
            material = pictures[k].cloud.get();
            continue;
        }
        (pictures[k].held ? held : train).push_back(k);
    }
    if (train.empty() || held.empty()) {
        err("%s wants clouds to learn from (train) and clouds to be held against (held)\n", o.pictures.c_str());
        return kExitFailure;
    }
    // The gaussians are the first cloud's; every cloud's are the same.
    const scene::GpuSplats& cloud = *pictures[train.front()].cloud;
    if (material != nullptr && material->count != cloud.count) {
        err("the base cloud holds %u gaussians and the pictures %u: a conversion with no bake keeps the gaussians a bake drops\n",
            material->count, cloud.count);
        return kExitFailure;
    }
    auto neural = render::makeNeuralCloud(library, material != nullptr ? *material : cloud, 1, true);
    if (!neural) return report(neural.error());
    neural->shadowed = byMap;
    const render::SplatInstance instance{&cloud, render::Mat4::identity(), {}};

    const auto d = [](float v) { return static_cast<double>(v); };
    const scene::Bounds& b = cloud.bounds;
    const render::Vec3 target = o.target.size() == 3
                                    ? render::Vec3{o.target[0], o.target[1], o.target[2]}
                                    : render::Vec3{(d(b.min[0]) + d(b.max[0])) * 0.5, (d(b.min[1]) + d(b.max[1])) * 0.5,
                                                   (d(b.min[2]) + d(b.max[2])) * 0.5};
    const double extent = std::max({d(b.max[0]) - d(b.min[0]), d(b.max[1]) - d(b.min[1]), d(b.max[2]) - d(b.min[2])});
    const double distance = o.distance > 0.0 ? o.distance : std::max(extent, 1.0e-3) * 1.4;
    const auto cameraAt = [&](uint32_t k, double phase) {
        const double a = 6.283185307179586 * std::fmod(0.5 + (k + phase) * 0.7548776662466927, 1.0);
        const double h = -0.25 + 1.15 * std::fmod(0.5 + (k + phase) * 0.5698402909980532, 1.0);
        const double r = std::sqrt(std::max(1.0 - h * h, 0.0));
        render::Camera camera = render::Camera::lookingAt(
            target + render::Vec3{r * std::sin(a), h, r * std::cos(a)} * distance, target, {0.0, 1.0, 0.0});
        camera.lens.focal = o.focal;
        return camera;
    };

    // A picture's sun, as the neural cloud is told it and as the shadow map is built from it.
    std::optional<technique::SplatShadowMap> shadowMap;
    gpu::Buffer mapFactors;
    render::SplatLights lights;
    render::SplatLights mapLights;
    if (byMap) {
        auto made = technique::SplatShadowMap::create(library);
        if (!made) return report(made.error());
        shadowMap.emplace(std::move(*made));
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(cloud.count, 1) * sizeof(float);
        desc.elementBytes = sizeof(float);
        desc.label = "neural.reached";
        auto factors = gpu::Buffer::create(**device, desc);
        if (!factors) return report(factors.error());
        mapFactors = *factors;
    }
    const auto sunOf = [&](const Picture& picture) -> Result<render::NeuralFitLight> {
        light::Light lamp;
        lamp.kind = light::LightKind::Distant;
        lamp.lightToWorld = aofx::xform::rotationY(picture.yaw) * aofx::xform::rotationX(picture.pitch);
        lamp.intensity = o.sun;
        lamp.shadow = true;
        ATHENEA_TRY(table->set(std::span<const light::Light>(&lamp, 1)));
        lights.records = &table->records();
        lights.count = table->count();
        render::NeuralFitLight told;
        told.teacher = &lights;
        if (shadowMap) {
            technique::ShadowMapJob job;
            technique::ShadowMapCaster caster;
            caster.cloud = &cloud;
            caster.positions = &cloud.positions;
            caster.categories = ~uint64_t{0};
            job.casters.push_back(caster);
            job.lights = lights.records;
            job.lightCount = 1;
            job.resolution = o.mapSide;
            job.coefficients = o.mapCoefficients;
            job.selfBias = o.mapBias;
            gpu::CommandBatch batch(**device);
            ATHENEA_TRY(shadowMap->build(batch, job));
            ATHENEA_TRY(shadowMap->factors(batch, caster, 0, mapFactors));
            ATHENEA_TRY(batch.submit(true));
            mapLights = render::SplatLights{};
            mapLights.records = lights.records;
            mapLights.count = lights.count;
            mapLights.visibilityFactors = &mapFactors;
            mapLights.visibilityLights = 1;
            told.student = &mapLights;
        }
        const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
        told.light.towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                              static_cast<float>(towards.z)};
        told.light.radiance = {o.sun, o.sun, o.sun};
        return told;
    };

    render::RenderSettings settings;
    settings.width = how.width;
    settings.height = how.height;
    struct Measure {
        double   relMse = 0.0;
        double   p99 = 0.0;
        uint32_t max = 0;
    };
    // The neural cloud told each of `which` clouds' suns, against that cloud, from the held-out cameras.
    // `write`: the first camera's frames of the first three, under that prefix.
    const auto measure = [&](const std::vector<size_t>& which, const std::string& write) -> Result<Measure> {
        Measure m;
        const double share = 1.0 / (static_cast<double>(which.size()) * o.heldOut);
        for (size_t at = 0; at < which.size(); ++at) {
            const Picture& picture = pictures[which[at]];
            auto told = sunOf(picture);
            if (!told) return std::move(told).error();
            for (uint32_t k = 0; k < o.heldOut; ++k) {
                const render::Camera camera = cameraAt(k, 0.5);
                std::vector<render::SplatInstance> baked{{picture.cloud.get(), render::Mat4::identity(), {}}};
                std::vector<render::SplatInstance> learnt{instance};
                learnt[0].neural = &*neural;
                learnt[0].neuralLight = &told->light;
                render::RenderTargets whole;
                render::RenderTargets ours;
                auto a = raster->render(camera, baked, settings, whole);
                if (!a) return std::move(a).error();
                auto c = raster->render(camera, learnt, settings, ours, {}, nullptr, byMap ? &mapLights : nullptr);
                if (!c) return std::move(c).error();
                auto hdr = render::compareHdr(library, ours.colour, whole.colour, settings.width, settings.height);
                if (!hdr) return std::move(hdr).error();
                auto codes = render::compareImages(library, ours.colour, whole.colour, settings.width, settings.height);
                if (!codes) return std::move(codes).error();
                m.relMse += hdr->relMse * share;
                m.p99 += static_cast<double>(codes->p99) * share;
                m.max = std::max(m.max, codes->max);
                if (!write.empty() && k == 0 && at < 3) {
                    for (const auto& [what, targets] :
                         {std::pair<const char*, render::RenderTargets*>{"picture", &whole},
                          std::pair<const char*, render::RenderTargets*>{"latents", &ours}}) {
                        auto colour = targets->colour.readAll<float>(**device);
                        if (!colour) return std::move(colour).error();
                        auto depth = targets->depth.readAll<float>(**device);
                        if (!depth) return std::move(depth).error();
                        const std::string path = write + "_" + what + "_sun" + std::to_string(at) + ".exr";
                        ATHENEA_TRY(io::writeExr(path, settings.width, settings.height, *colour, *depth));
                        out("wrote %s\n", path.c_str());
                    }
                }
            }
        }
        return m;
    };

    out("%u gaussians; %zu clouds to learn from and %zu held out, each the mesh baked under its sun\n", cloud.count,
        train.size(), held.size());
    auto before = measure(held, {});
    if (!before) return report(before.error());
    out("before: relMSE %.5f, p99 %.1f, max %u (held-out suns and cameras, against the baked clouds)\n",
        before->relMse, before->p99, before->max);
    const auto start = std::chrono::steady_clock::now();
    const unsigned every = std::max(o.steps / 10, 1u);
    for (unsigned k = 0; k < o.steps; ++k) {
        // The clouds in an order that visits all of them before any twice.
        const Picture& picture = pictures[train[(static_cast<size_t>(k) * 7) % train.size()]];
        auto told = sunOf(picture);
        if (!told) return report(told.error());
        const render::SplatInstance baked{picture.cloud.get(), render::Mat4::identity(), {}};
        told->picture = &baked;
        const bool ask = k == 0 || (k + 1) % every == 0;
        const float falling = std::pow(o.rateDecay, static_cast<float>(k) / static_cast<float>(std::max(o.steps, 1u)));
        how.rateFeatures = o.rateFeatures * falling;
        how.rateBase = o.rateBase * falling;
        how.rateWeights = o.rateWeights * falling;
        auto loss = fit->step(*raster, cameraAt(k, 0.0), instance, *neural, how, ask, &*told);
        if (!loss) return report(loss.error());
        if (*loss) {
            auto now = measure(held, {});
            if (!now) return report(now.error());
            out("step %u: relMSE %.5f; held out: relMSE %.5f, p99 %.1f, max %u\n", k + 1, **loss, now->relMse,
                now->p99, now->max);
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    auto after = measure(held, o.output);
    if (!after) return report(after.error());
    auto seen = measure(train, {});
    if (!seen) return report(seen.error());
    out("after %u steps in %.1f s: held-out suns relMSE %.5f, p99 %.1f, max %u; the suns it learnt from relMSE %.5f, "
        "p99 %.1f, max %u\n",
        o.steps, seconds, after->relMse, after->p99, after->max, seen->relMse, seen->p99, seen->max);
    return 0;
}

int run(const NeuralFitOptions& o) {
    if (!o.pictures.empty()) {
        return runPictures(o);
    }
    render::NeuralFitSettings how;
    if (std::sscanf(o.size.c_str(), "%ux%u", &how.width, &how.height) != 2 || how.width == 0 || how.height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return kExitFailure;
    }
    how.rateFeatures = o.rateFeatures;
    how.rateBase = o.rateBase;
    how.rateWeights = o.rateWeights;
    how.viewSmooth = o.viewSmooth;
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
    if (neural && !o.load.empty()) {
        auto file = io::readNeural(o.load);
        if (!file) return report(file.error());
        if (file->count != cloud->count || file->lit != o.relight) {
            err("%s is of a cloud of %u gaussians, %s; this one has %u and the fit is %s\n", o.load.c_str(), file->count,
                file->lit ? "lit" : "not lit", cloud->count, o.relight ? "lit" : "not lit");
            return kExitFailure;
        }
        neural = render::neuralCloudFrom(library, file->count, file->latent, file->weights, file->lit);
    }
    if (neural) {
        neural->shadowed = o.relight && o.shadows && !o.reached.empty();
    }
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
    // WITH SHADOWS: the cloud's own gaussians as what a shadow ray crosses --
    // the ray tracer's structures, built once since nothing moves -- so the
    // picture the cloud should be has the pawn's head darkening its collar.
    // The neural cloud is told nothing of it: only where the light comes from.
    // WHAT REACHES A GAUSSIAN, FROM A MAP AND NOT A RAY (`--reached map`): the
    // cloud's shadow map from the sun, built each frame with no ray at all,
    // read at each gaussian's own depth -- its seven coefficients are what
    // let a gaussian inside the cloud read less than the whole cloud's shadow.
    const bool byMap = o.relight && o.shadows && o.reached == "map";
    if (o.relight && o.shadows && !o.reached.empty() && o.reached != "ray" && o.reached != "map") {
        err("--reached wants ray or map\n");
        return kExitFailure;
    }
    std::optional<technique::SplatShadowMap> shadowMap;
    gpu::Buffer mapFactors;
    render::SplatLights mapLights;
    double mapMs = 0.0;
    if (byMap) {
        auto made = technique::SplatShadowMap::create(library);
        if (!made) return report(made.error());
        shadowMap.emplace(std::move(*made));
        gpu::BufferDesc desc;
        // Room for the three suns the fit is measured under at the end.
        desc.bytes = std::max<uint64_t>(cloud->count, 1) * 3 * sizeof(float);
        desc.elementBytes = sizeof(float);
        desc.label = "neural.reached";
        auto factors = gpu::Buffer::create(**device, desc);
        if (!factors) return report(factors.error());
        mapFactors = *factors;
    }
    std::optional<render::GaussianRayTracer> shadowTracer;
    if (o.relight && o.shadows) {
        if (!render::GaussianRayTracer::hardwareSupported(**device)) {
            err("--shadows wants a device that traces rays\n");
            return kExitFailure;
        }
        render::RayTracerSettings tracing;
        tracing.route = render::RayTracingRoute::Hardware;
        auto made = render::GaussianRayTracer::create(library, tracing);
        if (!made) return report(made.error());
        shadowTracer.emplace(std::move(*made));
    }
    const auto sunFrom = [&](double yaw, double pitch) -> Result<render::NeuralFitLight> {
        light::Light lamp;
        lamp.kind = light::LightKind::Distant;
        lamp.lightToWorld = aofx::xform::rotationY(yaw) * aofx::xform::rotationX(pitch);
        lamp.intensity = o.sun;
        lamp.shadow = o.shadows;
        ATHENEA_TRY(table->set(std::span<const light::Light>(&lamp, 1)));
        lights.records = &table->records();
        lights.count = table->count();
        if (shadowTracer) {
            const render::ShadowScene scene = shadowTracer->shadowScene();
            lights.shadowTlas = scene.tlas;
            lights.shadowFrames = scene.frames;
            lights.shadowColours = scene.colours;
            lights.shadowInstanceData = scene.instanceData;
            lights.shadowInstanceIndices = scene.instanceIndices;
        }
        render::NeuralFitLight told;
        told.teacher = &lights;
        if (shadowMap) {
            const auto start = std::chrono::steady_clock::now();
            technique::ShadowMapJob job;
            technique::ShadowMapCaster caster;
            caster.cloud = &*cloud;
            caster.positions = &cloud->positions;
            caster.categories = ~uint64_t{0};
            job.casters.push_back(caster);
            job.lights = lights.records;
            job.lightCount = 1;
            job.resolution = o.mapSide;
            job.coefficients = o.mapCoefficients;
            job.selfBias = o.mapBias;
            gpu::CommandBatch batch(**device);
            ATHENEA_TRY(shadowMap->build(batch, job));
            ATHENEA_TRY(shadowMap->factors(batch, caster, 0, mapFactors));
            ATHENEA_TRY(batch.submit(true));
            mapMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            mapLights = render::SplatLights{};
            mapLights.records = lights.records;
            mapLights.count = lights.count;
            mapLights.visibilityFactors = &mapFactors;
            mapLights.visibilityLights = 1;
            told.student = &mapLights;
        }
        // A distant light shines down its own -z: it comes from its +z.
        const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
        told.light.towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                              static_cast<float>(towards.z)};
        told.light.radiance = {o.sun, o.sun, o.sun};
        return told;
    };
    const auto sunAt = [&](uint32_t k, double phase) -> Result<render::NeuralFitLight> {
        return sunFrom(360.0 * std::fmod(0.5 + (k + phase) * 0.8191725134, 1.0),
                       -80.0 + 160.0 * std::fmod(0.5 + (k + phase) * 0.6710436067, 1.0));
    };

    // THREE SUNS AT ONCE, each its own colour: what a fit made under one white
    // sun a frame is then measured under, since its answer is linear in the
    // light. The engine's table holds the three; the neural cloud is told the
    // three, in the same order.
    std::vector<render::NeuralLight> threeSuns;
    const auto sunsAt = [&](uint32_t k) -> Result<void> {
        static constexpr float kTint[3][3] = {{1.0F, 0.35F, 0.2F}, {0.25F, 1.0F, 0.4F}, {0.25F, 0.4F, 1.0F}};
        std::array<light::Light, 3> lamps;
        threeSuns.assign(3, render::NeuralLight{});
        for (uint32_t l = 0; l < 3; ++l) {
            const double at = k * 3 + l + 0.25;
            light::Light& lamp = lamps[l];
            lamp.kind = light::LightKind::Distant;
            lamp.lightToWorld = aofx::xform::rotationY(360.0 * std::fmod(0.5 + at * 0.8191725134, 1.0)) *
                                aofx::xform::rotationX(-80.0 + 160.0 * std::fmod(0.5 + at * 0.6710436067, 1.0));
            lamp.intensity = o.sun * 0.6F;
            lamp.shadow = o.shadows;
            for (int c = 0; c < 3; ++c) {
                lamp.colour[c] = kTint[l][c];
                threeSuns[l].radiance[static_cast<size_t>(c)] = lamp.intensity * kTint[l][c];
            }
            const render::Vec3 towards = lamp.lightToWorld.direction({0.0, 0.0, 1.0});
            threeSuns[l].towards = {static_cast<float>(towards.x), static_cast<float>(towards.y),
                                    static_cast<float>(towards.z)};
        }
        ATHENEA_TRY(table->set(std::span<const light::Light>(lamps.data(), lamps.size())));
        lights.records = &table->records();
        lights.count = table->count();
        if (shadowMap) {
            technique::ShadowMapJob job;
            technique::ShadowMapCaster caster;
            caster.cloud = &*cloud;
            caster.positions = &cloud->positions;
            caster.categories = ~uint64_t{0};
            job.casters.push_back(caster);
            job.lights = lights.records;
            job.lightCount = 3;
            job.resolution = o.mapSide;
            job.coefficients = o.mapCoefficients;
            job.selfBias = o.mapBias;
            gpu::CommandBatch batch(**device);
            ATHENEA_TRY(shadowMap->build(batch, job));
            ATHENEA_TRY(shadowMap->factors(batch, caster, 0, mapFactors));
            ATHENEA_TRY(batch.submit(true));
            mapLights = render::SplatLights{};
            mapLights.records = lights.records;
            mapLights.count = lights.count;
            mapLights.visibilityFactors = &mapFactors;
            mapLights.visibilityLights = 3;
        }
        return ok();
    };
    if (shadowTracer) {
        const std::vector<render::SplatInstance> whole{instance};
        auto prepared = shadowTracer->prepare(render::projectionFor(cameraAt(0, 0.0), how.width, how.height), whole);
        if (!prepared) return report(prepared.error());
    }
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
            auto c = raster->render(camera, fewerDegrees < 3 ? byHarmonics : byLatents, other, ours, {}, nullptr,
                                    !neural->shadowed ? nullptr : byMap ? &mapLights : &lights);
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
                                    !o.relight ? nullptr : byMap && instances[0].neural != nullptr ? &mapLights : &lights);
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
        const float falling = std::pow(o.rateDecay, static_cast<float>(k) / static_cast<float>(std::max(o.steps, 1u)));
        how.rateFeatures = o.rateFeatures * falling;
        how.rateBase = o.rateBase * falling;
        how.rateWeights = o.rateWeights * falling;
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
            // Waited for, stage by stage: without it the frame's clock stops before the device has.
            render::RenderSettings waited = settings;
            waited.timeStages = true;
            auto stats = raster->render(cameraAt(0, 0.5), instances, waited, targets, {}, nullptr,
                                        !o.relight ? nullptr : byMap && instances[0].neural != nullptr ? &mapLights : &lights);
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
    // HOW MUCH EACH SHIMMERS as the camera goes by: three frames half a degree
    // apart round the target, the middle one against the mean of its
    // neighbours (`compareFlicker`). The camera moves, so there is parallax
    // in both; what is compared is the one against the other.
    if (!o.relight) {
        const auto orbit = [&](double degrees) {
            const render::Camera base = cameraAt(0, 0.5);
            const double a = degrees * 3.14159265358979 / 180.0;
            const render::Vec3 eye = base.cameraToWorld.point({0.0, 0.0, 0.0});
            const render::Vec3 d{eye.x - target.x, eye.y - target.y, eye.z - target.z};
            // Round the up axis: y, or z where that is up.
            render::Vec3 turned = std::abs(up.z) > 0.5
                                      ? render::Vec3{d.x * std::cos(a) - d.y * std::sin(a), d.x * std::sin(a) + d.y * std::cos(a), d.z}
                                      : render::Vec3{d.x * std::cos(a) + d.z * std::sin(a), d.y, -d.x * std::sin(a) + d.z * std::cos(a)};
            render::Camera camera = render::Camera::lookingAt(target + turned, target, up);
            camera.lens.focal = o.focal;
            return camera;
        };
        const auto shimmer = [&](const std::vector<render::SplatInstance>& instances) -> Result<render::FlickerDifference> {
            std::array<render::RenderTargets, 3> frames;
            for (int k = 0; k < 3; ++k) {
                auto drawn = raster->render(orbit(0.5 * (k - 1)), instances, settings, frames[static_cast<size_t>(k)]);
                if (!drawn) return std::move(drawn).error();
            }
            return render::compareFlicker(library, frames[0].colour, frames[1].colour, frames[2].colour,
                                          settings.width, settings.height);
        };
        auto byH = shimmer(byHarmonics);
        if (!byH) return report(byH.error());
        auto byL = shimmer(byLatents);
        if (!byL) return report(byL.error());
        out("three frames half a degree apart, the middle against its neighbours' mean: relMSE %.6f and p99 %u by the "
            "harmonics, %.6f and %u by the latents\n",
            byH->hdr.relMse, byH->codes.p99, byL->hdr.relMse, byL->codes.p99);
    }
    if (byMap) {
        out("and the shadow map the latents' frame reads, built and read at every gaussian: %.2f ms (waited for)\n",
            mapMs);
    }
    if (o.relight) {
        // Under three coloured suns at once, against the engine's relighting under the same three.
        Measure m;
        for (uint32_t k = 0; k < o.heldOut; ++k) {
            if (auto set = sunsAt(k); !set) return report(set.error());
            const render::Camera camera = cameraAt(k, 0.5);
            std::vector<render::SplatInstance> relit{instance};
            relit[0].relight = true;
            std::vector<render::SplatInstance> learnt{instance};
            learnt[0].neural = &*neural;
            learnt[0].neuralLight = threeSuns.data();
            learnt[0].neuralLightCount = 3;
            render::RenderTargets whole;
            render::RenderTargets ours;
            auto a = raster->render(camera, relit, settings, whole, {}, nullptr, &lights);
            if (!a) return report(a.error());
            auto c = raster->render(camera, learnt, settings, ours, {}, nullptr,
                                    !neural->shadowed ? nullptr : byMap ? &mapLights : &lights);
            if (!c) return report(c.error());
            auto hdr = render::compareHdr(library, ours.colour, whole.colour, settings.width, settings.height);
            if (!hdr) return report(hdr.error());
            auto codes = render::compareImages(library, ours.colour, whole.colour, settings.width, settings.height);
            if (!codes) return report(codes.error());
            m.relMse += hdr->relMse / o.heldOut;
            m.p99 += static_cast<double>(codes->p99) / o.heldOut;
            m.max = std::max(m.max, codes->max);
            if (k == 0 && !o.output.empty()) {
                for (const auto& [what, targets] : {std::pair<const char*, render::RenderTargets*>{"engine_three", &whole},
                                                    std::pair<const char*, render::RenderTargets*>{"latents_three", &ours}}) {
                    auto colour = targets->colour.readAll<float>(**device);
                    if (!colour) return report(colour.error());
                    auto depth = targets->depth.readAll<float>(**device);
                    if (!depth) return report(depth.error());
                    const std::string path = o.output + "_" + what + ".exr";
                    if (auto w = io::writeExr(path, settings.width, settings.height, *colour, *depth); !w) {
                        return report(w.error());
                    }
                    out("wrote %s\n", path.c_str());
                }
            }
        }
        out("under three coloured suns at once (fitted under one white sun a frame): relMSE %.5f, p99 %.1f, max %u, "
            "against %.5f, %.1f and %u under one\n",
            m.relMse, m.p99, m.max, after->relMse, after->p99, after->max);
    }
    if (o.relight && o.turn > 0 && !o.output.empty()) {
        // A sun going once round the cloud, forty degrees up, from the first held-out camera: the
        // engine's relighting and the latents', a frame each a step, for a film of the two side by side.
        for (unsigned f = 0; f < o.turn; ++f) {
            auto told = sunFrom(360.0 * f / o.turn, -40.0);
            if (!told) return report(told.error());
            const render::Camera camera = cameraAt(0, 0.5);
            std::vector<render::SplatInstance> relit{instance};
            relit[0].relight = true;
            std::vector<render::SplatInstance> learnt{instance};
            learnt[0].neural = &*neural;
            learnt[0].neuralLight = &told->light;
            render::RenderTargets whole;
            render::RenderTargets ours;
            auto a = raster->render(camera, relit, settings, whole, {}, nullptr, &lights);
            if (!a) return report(a.error());
            auto c = raster->render(camera, learnt, settings, ours, {}, nullptr,
                                    !neural->shadowed ? nullptr : byMap ? &mapLights : &lights);
            if (!c) return report(c.error());
            for (const auto& [what, targets] : {std::pair<const char*, render::RenderTargets*>{"engine", &whole},
                                                std::pair<const char*, render::RenderTargets*>{"latents", &ours}}) {
                auto colour = targets->colour.readAll<float>(**device);
                if (!colour) return report(colour.error());
                auto depth = targets->depth.readAll<float>(**device);
                if (!depth) return report(depth.error());
                char name[32];
                std::snprintf(name, sizeof(name), "_turn_%s_%03u.exr", what, f);
                if (auto w = io::writeExr(o.output + name, settings.width, settings.height, *colour, *depth); !w) {
                    return report(w.error());
                }
            }
        }
        out("wrote %u frames of a sun going round, by the engine and by the latents: %s_turn_*.exr\n", o.turn,
            o.output.c_str());
    }
    if (!o.save.empty()) {
        io::NeuralFile file;
        file.count = neural->count;
        file.stride = render::kNeuralStride;
        file.wordsEach = render::kNeuralWords;
        file.lit = neural->lit;
        if (neural->shadowed) {
            out("note: a shadowed appearance is written as a lit one; what reaches a gaussian is the frame's to give\n");
        }
        auto words = render::packNeuralLatents(library, *neural);
        if (!words) return report(words.error());
        auto weights = render::readNeuralWeights(library, *neural);
        if (!weights) return report(weights.error());
        file.latent = std::move(*words);
        file.weights = std::move(*weights);
        if (auto written = io::writeNeural(o.save, file); !written) return report(written.error());
        out("wrote %s: %u bytes a gaussian and %zu weights\n", o.save.c_str(), render::kNeuralWords * 4,
            file.weights.size());
    }
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
    cmd->add_option("--splats", o.splats, "a .ply / .splat / .spz / .sog file");
    cmd->add_option("--pictures", o.pictures, "instead of --splats: a manifest of one mesh baked into the same gaussians under several suns (USD ParticleFields); a lit fit to those");
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
    cmd->add_option("--view-smooth", o.viewSmooth, "hold each gaussian's colour seen a degree off to its colour, this many times as hard as a pixel to its picture (0: not held)");
    cmd->add_option("--rate-decay", o.rateDecay, "what the three rates fall to by the last step, as a factor (1: they stay)");
    cmd->add_flag("--relight", o.relight, "a lit neural cloud, fitted to the engine's relighting of the cloud under a sun that moves");
    cmd->add_option("--sun", o.sun, "--relight: the sun's intensity");
    cmd->add_option("--turn", o.turn, "--relight with -o: this many frames of a sun going once round, by the engine and by the latents");
    cmd->add_option("--reached", o.reached, "--shadows: the neural cloud takes what reaches each gaussian from the light: ray (the engine's shadow ray) or map (the cloud's shadow map, no ray)");
    cmd->add_option("--map-side", o.mapSide, "--reached map: the shadow map's texels a side");
    cmd->add_option("--map-coefficients", o.mapCoefficients, "--reached map: 1, 3, 5 or 7 -- the map's total optical depth and its Fourier pairs");
    cmd->add_option("--map-bias", o.mapBias, "--reached map: how far behind a texel's nearest caster a receiver still stands on it, world units (0: 2% of the slab)");
    cmd->add_flag("--shadows", o.shadows, "--relight: the sun casts the cloud's shadow on itself in what the cloud should be");
    cmd->add_option("--load", o.load, "a .athn to start from; with --steps 0, only measured");
    cmd->add_option("--save", o.save, "write the fitted appearance: a .athn, to draw the cloud by (athenea render --neural)");
    cmd->add_option("-o,--output", o.output, "a prefix: the first held-out camera's frame by the harmonics, the base and the latents");
    cmd->callback([options] {
        if (options->splats.empty() == options->pictures.empty()) {
            err("give --splats or --pictures\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        if (const int code = run(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace athenea::cli
