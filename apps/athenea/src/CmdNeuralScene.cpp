// Copyright (c) 2026 jesus luque.
//
// athenea neural-scene: a scene kept in pieces (io/NeuralScene.h).
//
//   add     a piece's gaussians into the scene, counted and bounded on the device
//   camera  a named view kept in the manifest
//   fit     each piece's harmonics learnt by its own latents and network
//   render  the whole scene from a view, by the latents or by the harmonics
//   look    one camera's look at one time: which gaussians it found
//   info    what the manifest holds
//
// A piece is fitted alone -- its cloud, its appearance, the device's memory
// one piece at a time -- from cameras that stand inside the scene and look at
// the piece, which is where a building is seen from.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/io/NeuralFile.h"
#include "athenea/io/NeuralScene.h"
#include "athenea/render/Discovery.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/NeuralFit.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {
namespace {

struct SceneOptions {
    std::string              directory;
    // add
    std::string              id;
    std::string              cloud;
    std::string              prim;
    std::string              name;
    std::string              upAxis;
    double                   metersPerUnit = 0.0;
    std::string              source;
    std::string              licence;
    std::string              origin;
    std::string              foundBy;
    double                   time = -1.0;
    // camera, render
    std::string              camera;
    std::vector<double>      eye;
    std::vector<double>      target;
    std::vector<double>      up;
    double                   focal = 0.0;
    // fit
    std::vector<std::string> pieces;
    unsigned                 steps = 3000;
    std::string              size = "960x540";
    unsigned                 heldOut = 8;
    float                    rateDecay = 0.1F;
    float                    viewSmooth = 0.0F;
    bool                     split = false;
    // render
    std::string              output;
    bool                     harmonics = false;
    bool                     compare = false;
    bool                     discovered = false;
    unsigned                 upTo = 0;
    float                    threshold = 0.05F;
};

bool sizeOf(const std::string& text, uint32_t& width, uint32_t& height) {
    if (std::sscanf(text.c_str(), "%ux%u", &width, &height) != 2 || width == 0 || height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return false;
    }
    return true;
}

/// A piece's gaussians on the device: a USD ParticleField, or a splat file.
Result<scene::GpuSplats> load(gpu::ShaderLibrary& library, scene::CloudLoader& loader,
                              const std::filesystem::path& path) {
    const std::string ext = path.extension().string();
    if (ext == ".usda" || ext == ".usdc" || ext == ".usd") {
        return usd::uploadParticleField(library, loader, path);
    }
    return scene::loadSplatFile(loader, path.string(), 3);
}

render::Vec3 upOf(const io::NeuralScene& scene) {
    return scene.upAxis == "Z" ? render::Vec3{0.0, 0.0, 1.0} : render::Vec3{0.0, 1.0, 0.0};
}

int runAdd(const SceneOptions& o) {
    const std::filesystem::path directory = o.directory;
    io::NeuralScene scene;
    if (std::filesystem::exists(directory / "scene.json")) {
        auto read = io::readNeuralScene(directory);
        if (!read) return report(read.error());
        scene = std::move(*read);
    } else {
        scene.name = directory.stem().string();
    }
    if (!o.name.empty()) scene.name = o.name;
    if (!o.upAxis.empty()) scene.upAxis = o.upAxis;
    if (o.metersPerUnit > 0.0) scene.metersPerUnit = o.metersPerUnit;
    if (!o.source.empty()) scene.source = o.source;
    if (!o.licence.empty()) scene.licence = o.licence;
    if (scene.upAxis != "Y" && scene.upAxis != "Z") {
        err("--up-axis wants Y or Z\n");
        return kExitFailure;
    }
    if (o.id.empty() || o.cloud.empty()) {
        err("a piece wants --id and --cloud\n");
        return kExitFailure;
    }
    for (const char c : o.id) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '-')) {
            err("--id is a file's name: letters, digits, _ and -\n");
            return kExitFailure;
        }
    }
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto cloud = load(library, *loader, o.cloud);
    if (!cloud) return report(cloud.error());

    // The manifest is written first so that the directory is there to copy into.
    if (auto made = io::writeNeuralScene(directory, scene); !made) return report(made.error());
    const std::filesystem::path from = o.cloud;
    const std::string kept = "pieces/" + o.id + from.extension().string();
    std::error_code failed;
    std::filesystem::copy_file(from, directory / kept, std::filesystem::copy_options::overwrite_existing, failed);
    if (failed) {
        err("cannot copy %s into the scene: %s\n", o.cloud.c_str(), failed.message().c_str());
        return kExitFailure;
    }
    io::NeuralScenePiece* piece = scene.piece(o.id);
    if (piece == nullptr) {
        scene.pieces.emplace_back();
        piece = &scene.pieces.back();
        piece->id = o.id;
        piece->revision = 1;
    } else {
        // Its gaussians are others: what was fitted to the old ones is not theirs.
        ++piece->revision;
        piece->appearance.clear();
        piece->fit.reset();
    }
    // New gaussians are a new place and a new paint both.
    piece->movedAt = piece->revision;
    piece->paintedAt = piece->revision;
    if (!o.origin.empty()) {
        if (o.origin != "converted" && o.origin != "discovered" && o.origin != "generated") {
            err("--origin wants converted, discovered or generated\n");
            return kExitFailure;
        }
        piece->origin = o.origin;
    }
    piece->foundBy = o.foundBy;
    piece->foundAt = o.time >= 0.0 ? o.time : 0.0;
    if (o.time >= 0.0) {
        // The scene's history: this look, and what it brought.
        io::NeuralSceneSnapshot* held = nullptr;
        for (io::NeuralSceneSnapshot& s : scene.snapshots) {
            if (s.time == o.time && s.camera == o.foundBy) {
                held = &s;
            }
        }
        if (held == nullptr) {
            scene.snapshots.emplace_back();
            held = &scene.snapshots.back();
            held->time = o.time;
            held->camera = o.foundBy;
        }
        if (std::find(held->pieces.begin(), held->pieces.end(), o.id) == held->pieces.end()) {
            held->pieces.push_back(o.id);
        }
    }
    piece->prim = o.prim;
    piece->cloud = kept;
    piece->gaussians = cloud->count;
    piece->boundsMin = cloud->bounds.min;
    piece->boundsMax = cloud->bounds.max;
    if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
    out("%s: piece %s, %u gaussians, revision %u (%zu pieces)\n", o.directory.c_str(), o.id.c_str(), cloud->count,
        piece->revision, scene.pieces.size());
    return 0;
}

int runCamera(const SceneOptions& o) {
    auto scene = io::readNeuralScene(o.directory);
    if (!scene) return report(scene.error());
    if (o.camera.empty() || o.eye.size() != 3 || o.target.size() != 3) {
        err("a camera wants --name, --eye x y z and --target x y z\n");
        return kExitFailure;
    }
    io::NeuralSceneCamera camera;
    camera.name = o.camera;
    camera.eye = {o.eye[0], o.eye[1], o.eye[2]};
    camera.target = {o.target[0], o.target[1], o.target[2]};
    const render::Vec3 up = o.up.size() == 3 ? render::Vec3{o.up[0], o.up[1], o.up[2]} : upOf(*scene);
    camera.up = {up.x, up.y, up.z};
    camera.focal = o.focal > 0.0 ? o.focal : 35.0;
    std::erase_if(scene->cameras, [&](const io::NeuralSceneCamera& c) { return c.name == camera.name; });
    scene->cameras.push_back(camera);
    if (auto written = io::writeNeuralScene(o.directory, *scene); !written) return report(written.error());
    out("%s: camera %s (%zu cameras)\n", o.directory.c_str(), o.camera.c_str(), scene->cameras.size());
    return 0;
}

int runInfo(const SceneOptions& o) {
    auto scene = io::readNeuralScene(o.directory);
    if (!scene) return report(scene.error());
    std::array<float, 3> min{};
    std::array<float, 3> max{};
    scene->bounds(min, max);
    uint64_t total = 0;
    for (const io::NeuralScenePiece& p : scene->pieces) {
        total += p.gaussians;
    }
    out("%s: %zu pieces, %llu gaussians, %s up, %g m a unit\n", scene->name.c_str(), scene->pieces.size(),
        static_cast<unsigned long long>(total), scene->upAxis.c_str(), scene->metersPerUnit);
    out("  bounds (%.3f %.3f %.3f) to (%.3f %.3f %.3f)\n", min[0], min[1], min[2], max[0], max[1], max[2]);
    if (!scene->source.empty()) out("  from %s\n", scene->source.c_str());
    if (!scene->licence.empty()) out("  licence: %s\n", scene->licence.c_str());
    for (const io::NeuralScenePiece& p : scene->pieces) {
        out("  piece %s: %u gaussians, %s, revision %u (moved at %u, painted at %u), %s", p.id.c_str(), p.gaussians,
            p.origin.c_str(), p.revision, p.movedAt, p.paintedAt, p.isStatic ? "static" : "moving");
        if (!p.seen.empty()) {
            out(", %u found by the looks", p.discovered);
        }
        if (p.appearance.empty()) {
            out(", by its harmonics\n");
        } else if (p.fit) {
            out(", %s%s (%u steps against its %s: relMSE %.5f, p99 %.1f, max %u)\n", p.lit ? "lit " : "",
                p.split ? "two networks" : "one network", p.fit->steps, p.fit->against.c_str(), p.fit->relMse,
                p.fit->p99, p.fit->max);
        } else {
            out(", %s\n", p.appearance.c_str());
        }
    }
    for (const io::NeuralSceneSnapshot& s : scene->snapshots) {
        out("  snapshot at %g by %s: %zu pieces\n", s.time, s.camera.empty() ? "no camera" : s.camera.c_str(),
            s.pieces.size());
    }
    for (const io::NeuralSceneCamera& c : scene->cameras) {
        out("  camera %s: from (%.3f %.3f %.3f) at (%.3f %.3f %.3f), %g mm\n", c.name.c_str(), c.eye[0], c.eye[1],
            c.eye[2], c.target[0], c.target[1], c.target[2], c.focal);
    }
    return 0;
}

/// A value in 0 .. 1 from the additive sequence of `alpha`, the `k`th.
double sequence(uint32_t k, double alpha, double phase) { return std::fmod(0.5 + (k + phase) * alpha, 1.0); }

int runFit(const SceneOptions& o) {
    render::NeuralFitSettings how;
    if (!sizeOf(o.size, how.width, how.height)) return kExitFailure;
    how.viewSmooth = o.viewSmooth;
    const render::NeuralFitSettings rates = how;
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());

    std::array<float, 3> low{};
    std::array<float, 3> high{};
    scene->bounds(low, high);
    const render::Vec3 up = upOf(*scene);
    render::RenderSettings settings;
    settings.width = how.width;
    settings.height = how.height;

    for (io::NeuralScenePiece& piece : scene->pieces) {
        if (!o.pieces.empty() && std::find(o.pieces.begin(), o.pieces.end(), piece.id) == o.pieces.end()) {
            continue;
        }
        if (piece.gaussians == 0) {
            continue;
        }
        auto cloud = load(library, *loader, directory / piece.cloud);
        if (!cloud) return report(cloud.error());
        if (cloud->count != piece.gaussians) {
            err("piece %s: its cloud holds %u gaussians and the manifest says %u\n", piece.id.c_str(), cloud->count,
                piece.gaussians);
            return kExitFailure;
        }
        auto fit = render::NeuralFit::create(library);
        if (!fit) return report(fit.error());
        auto neural = render::makeNeuralCloud(library, *cloud, 1, false, o.split);
        if (!neural) return report(neural.error());
        const render::SplatInstance instance{&*cloud, render::Mat4::identity(), {}};

        // WHERE A PIECE IS SEEN FROM: inside the scene. An eye somewhere in the scene's box (kept off its
        // walls by a tenth), looking at somewhere in the piece's own, a wide lens. Each from a sequence
        // that fills its box evenly, the held-out ones half a step off the fitted ones.
        const auto cameraAt = [&](uint32_t k, double phase) {
            const auto within = [](float a, float b, double t, double margin) {
                return static_cast<double>(a) + (static_cast<double>(b) - static_cast<double>(a)) *
                                                    (margin + (1.0 - 2.0 * margin) * t);
            };
            const render::Vec3 eye{within(low[0], high[0], sequence(k, 0.8191725133961645, phase), 0.1),
                                   within(low[1], high[1], sequence(k, 0.6710436067037893, phase), 0.1),
                                   within(low[2], high[2], sequence(k, 0.5497004779019703, phase), 0.1)};
            render::Vec3 target{
                within(piece.boundsMin[0], piece.boundsMax[0], sequence(k, 0.7548776662466927, phase + 0.25), 0.0),
                within(piece.boundsMin[1], piece.boundsMax[1], sequence(k, 0.5698402909980532, phase + 0.25), 0.0),
                within(piece.boundsMin[2], piece.boundsMax[2], sequence(k, 0.4301597090019468, phase + 0.25), 0.0)};
            const render::Vec3 d{target.x - eye.x, target.y - eye.y, target.z - eye.z};
            if (d.x * d.x + d.y * d.y + d.z * d.z < 1.0e-8) {
                target = target + render::Vec3{1.0, 1.0, 1.0};
            }
            render::Camera camera = render::Camera::lookingAt(eye, target, up);
            camera.lens.focal = 20.0;
            return camera;
        };
        struct Measure {
            double   relMse = 0.0;
            double   p99 = 0.0;
            uint32_t max = 0;
        };
        const auto measure = [&]() -> Result<Measure> {
            Measure m;
            std::vector<render::SplatInstance> byHarmonics{instance};
            std::vector<render::SplatInstance> byLatents{instance};
            byLatents[0].neural = &*neural;
            for (uint32_t k = 0; k < o.heldOut; ++k) {
                const render::Camera camera = cameraAt(k, 0.5);
                render::RenderTargets a;
                render::RenderTargets b;
                auto first = raster->render(camera, byHarmonics, settings, a);
                if (!first) return std::move(first).error();
                auto second = raster->render(camera, byLatents, settings, b);
                if (!second) return std::move(second).error();
                auto hdr = render::compareHdr(library, b.colour, a.colour, settings.width, settings.height);
                if (!hdr) return std::move(hdr).error();
                auto codes = render::compareImages(library, b.colour, a.colour, settings.width, settings.height);
                if (!codes) return std::move(codes).error();
                m.relMse += hdr->relMse / o.heldOut;
                m.p99 += static_cast<double>(codes->p99) / o.heldOut;
                m.max = std::max(m.max, codes->max);
            }
            return m;
        };
        auto before = measure();
        if (!before) return report(before.error());
        const auto start = std::chrono::steady_clock::now();
        for (unsigned k = 0; k < o.steps; ++k) {
            const float falling =
                std::pow(o.rateDecay, static_cast<float>(k) / static_cast<float>(std::max(o.steps, 1u)));
            how.rateFeatures = rates.rateFeatures * falling;
            how.rateBase = rates.rateBase * falling;
            how.rateWeights = rates.rateWeights * falling;
            auto stepped = fit->step(*raster, cameraAt(k, 0.0), instance, *neural, how, false);
            if (!stepped) return report(stepped.error());
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        auto after = measure();
        if (!after) return report(after.error());

        io::NeuralFile file;
        file.count = neural->count;
        file.stride = render::kNeuralStride;
        file.wordsEach = render::kNeuralWords;
        file.lit = false;
        file.split = neural->split;
        auto words = render::packNeuralLatents(library, *neural);
        if (!words) return report(words.error());
        auto weights = render::readNeuralWeights(library, *neural);
        if (!weights) return report(weights.error());
        file.latent = std::move(*words);
        file.weights = std::move(*weights);
        piece.appearance = "pieces/" + piece.id + ".athn";
        if (auto written = io::writeNeural(directory / piece.appearance, file); !written) return report(written.error());
        piece.lit = false;
        piece.split = neural->split;
        ++piece.revision;
        piece.paintedAt = piece.revision;   // its place is as it was
        io::NeuralSceneFit fitted;
        fitted.steps = o.steps;
        fitted.against = "harmonics";
        fitted.relMse = after->relMse;
        fitted.p99 = after->p99;
        fitted.max = after->max;
        piece.fit = fitted;
        // Written a piece at a time: a fit that stops keeps the pieces it finished.
        if (auto written = io::writeNeuralScene(directory, *scene); !written) return report(written.error());
        out("piece %s, %u gaussians: relMSE %.5f, p99 %.1f, max %u before; %.5f, %.1f, %u after %u steps in %.1f s "
            "(%.1f ms a step)\n",
            piece.id.c_str(), piece.gaussians, before->relMse, before->p99, before->max, after->relMse, after->p99,
            after->max, o.steps, seconds, o.steps > 0 ? seconds * 1000.0 / o.steps : 0.0);
    }
    return 0;
}

/// The view a command asks for: one of its own (--eye, --target) or one the manifest names.
std::optional<render::Camera> viewOf(const SceneOptions& o, const io::NeuralScene& scene) {
    render::Camera camera;
    if (o.eye.size() == 3 && o.target.size() == 3) {
        const render::Vec3 up = o.up.size() == 3 ? render::Vec3{o.up[0], o.up[1], o.up[2]} : upOf(scene);
        camera = render::Camera::lookingAt({o.eye[0], o.eye[1], o.eye[2]}, {o.target[0], o.target[1], o.target[2]}, up);
        camera.lens.focal = o.focal > 0.0 ? o.focal : 35.0;
        return camera;
    }
    for (const io::NeuralSceneCamera& c : scene.cameras) {
        if (o.camera.empty() || c.name == o.camera) {
            camera = render::Camera::lookingAt({c.eye[0], c.eye[1], c.eye[2]}, {c.target[0], c.target[1], c.target[2]},
                                               {c.up[0], c.up[1], c.up[2]});
            camera.lens.focal = o.focal > 0.0 ? o.focal : c.focal;
            return camera;
        }
    }
    err("the scene has no camera %s: --eye and --target, or neural-scene camera\n", o.camera.c_str());
    return std::nullopt;
}

/// ONE LOOK. A camera of the manifest's at a time: the frame is drawn, and every gaussian that gave it
/// something is found -- stamped with this look where no look had seen it. What the look says is how
/// much of what it saw the scene already held, which is what the next frame does not have to find again.
int runLook(const SceneOptions& o) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (!sizeOf(o.size, width, height)) return kExitFailure;
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    if (o.camera.empty() || o.time < 0.0) {
        err("a look wants --camera, one of the manifest's, and --time\n");
        return kExitFailure;
    }
    const auto camera = viewOf(o, *scene);
    if (!camera) return kExitFailure;
    // The look's place in the scene's history is what its gaussians are stamped with.
    size_t at = scene->snapshots.size();
    for (size_t k = 0; k < scene->snapshots.size(); ++k) {
        if (scene->snapshots[k].time == o.time && scene->snapshots[k].camera == o.camera) {
            at = k;
        }
    }
    if (at == scene->snapshots.size()) {
        scene->snapshots.push_back({o.time, o.camera, {}});
    }
    const uint32_t stamp = static_cast<uint32_t>(at) + 1;

    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto discovery = render::Discovery::create(library);
    if (!discovery) return report(discovery.error());

    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<std::unique_ptr<render::NeuralCloud>> looks;
    std::vector<gpu::Buffer> records;
    std::vector<render::SplatInstance> instances;
    uint32_t slots = 0;
    for (const io::NeuralScenePiece& piece : scene->pieces) {
        auto cloud = load(library, *loader, directory / piece.cloud);
        if (!cloud) return report(cloud.error());
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
        const uint32_t count = clouds.back()->count;
        std::vector<uint32_t> held;
        if (!piece.seen.empty()) {
            auto read = io::readSeen(directory / piece.seen, count);
            if (!read) return report(read.error());
            held = std::move(*read);
        }
        auto record = discovery->record(count, held);
        if (!record) return report(record.error());
        records.push_back(*record);
        render::Mat4 place;
        for (size_t k = 0; k < 16; ++k) {
            place.m[k] = piece.transform[k];
        }
        render::SplatInstance instance{clouds.back().get(), place, {}};
        looks.emplace_back();
        if (!piece.appearance.empty()) {
            auto file = io::readNeural(directory / piece.appearance);
            if (!file) return report(file.error());
            auto neural = render::neuralCloudFrom(library, file->count, file->latent, file->weights, file->lit);
            if (!neural) return report(neural.error());
            looks.back() = std::make_unique<render::NeuralCloud>(std::move(*neural));
            instance.neural = looks.back().get();
        }
        instances.push_back(instance);
        slots += count;
    }
    auto given = discovery->zeros(slots);
    if (!given) return report(given.error());
    render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    settings.contribution = &*given;
    settings.contributionMode = 1;
    render::RenderTargets frame;
    auto drawn = raster->render(*camera, instances, settings, frame);
    if (!drawn) return report(drawn.error());

    uint64_t found = 0;
    uint64_t again = 0;
    uint64_t known = 0;
    uint64_t total = 0;
    uint32_t base = 0;
    io::NeuralSceneSnapshot& snapshot = scene->snapshots[at];
    for (size_t k = 0; k < scene->pieces.size(); ++k) {
        io::NeuralScenePiece& piece = scene->pieces[k];
        const uint32_t count = clouds[k]->count;
        auto counted = discovery->mark(*given, base, records[k], count, stamp, o.threshold);
        if (!counted) return report(counted.error());
        auto holds = discovery->known(records[k], count, 0xFFFFFFFFu);
        if (!holds) return report(holds.error());
        auto seen = records[k].readAll<uint32_t>(**device);
        if (!seen) return report(seen.error());
        seen->resize(count);
        piece.seen = "pieces/" + piece.id + ".seen";
        if (auto written = io::writeSeen(directory / piece.seen, *seen); !written) return report(written.error());
        piece.discovered = *holds;
        if (counted->found > 0 && std::find(snapshot.pieces.begin(), snapshot.pieces.end(), piece.id) == snapshot.pieces.end()) {
            snapshot.pieces.push_back(piece.id);
        }
        out("  piece %s: %u found, %u the scene already held; it now holds %u of %u\n", piece.id.c_str(), counted->found,
            counted->again, *holds, count);
        found += counted->found;
        again += counted->again;
        known += *holds;
        total += count;
        base += count;
    }
    if (auto written = io::writeNeuralScene(directory, *scene); !written) return report(written.error());
    const uint64_t saw = found + again;
    out("look %u (%s at %g): saw %llu gaussians, %llu new and %llu the scene already held (%.1f%%); the scene now "
        "holds %llu of %llu (%.1f%%)\n",
        stamp, o.camera.c_str(), o.time, static_cast<unsigned long long>(saw), static_cast<unsigned long long>(found),
        static_cast<unsigned long long>(again), saw > 0 ? 100.0 * static_cast<double>(again) / static_cast<double>(saw) : 0.0,
        static_cast<unsigned long long>(known), static_cast<unsigned long long>(total),
        total > 0 ? 100.0 * static_cast<double>(known) / static_cast<double>(total) : 0.0);
    if (!o.output.empty()) {
        auto colour = frame.colour.readAll<float>(**device);
        if (!colour) return report(colour.error());
        auto depth = frame.depth.readAll<float>(**device);
        if (!depth) return report(depth.error());
        if (auto w = io::writeExr(o.output, width, height, *colour, *depth); !w) return report(w.error());
        out("wrote %s\n", o.output.c_str());
    }
    return 0;
}

int runRender(const SceneOptions& o) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (!sizeOf(o.size, width, height)) return kExitFailure;
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    const auto view = viewOf(o, *scene);
    if (!view) return kExitFailure;
    const render::Camera camera = *view;
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto discovery = render::Discovery::create(library);
    if (!discovery) return report(discovery.error());

    // Every piece on the device, and its appearance where it has one.
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<std::unique_ptr<render::NeuralCloud>> looks;
    std::vector<render::SplatInstance> byHarmonics;
    std::vector<render::SplatInstance> byLatents;
    uint32_t fitted = 0;
    for (const io::NeuralScenePiece& piece : scene->pieces) {
        auto cloud = load(library, *loader, directory / piece.cloud);
        if (!cloud) return report(cloud.error());
        if (o.discovered) {
            // THE SCENE AS IT IS KNOWN: only what the looks found, up to one of them. A piece no look
            // has been taken of is not drawn at all.
            if (piece.seen.empty()) {
                continue;
            }
            auto held = io::readSeen(directory / piece.seen, cloud->count);
            if (!held) return report(held.error());
            auto record = discovery->record(cloud->count, *held);
            if (!record) return report(record.error());
            auto known = discovery->masked(*cloud, *record, o.upTo == 0 ? 0xFFFFFFFFu : o.upTo);
            if (!known) return report(known.error());
            cloud = std::move(known);
        }
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
        render::Mat4 place;
        for (size_t k = 0; k < 16; ++k) {
            place.m[k] = piece.transform[k];
        }
        render::SplatInstance instance{clouds.back().get(), place, {}};
        byHarmonics.push_back(instance);
        looks.emplace_back();
        if (!piece.appearance.empty()) {
            auto file = io::readNeural(directory / piece.appearance);
            if (!file) return report(file.error());
            if (file->count != clouds.back()->count) {
                err("piece %s: its appearance is of %u gaussians and its cloud holds %u\n", piece.id.c_str(),
                    file->count, clouds.back()->count);
                return kExitFailure;
            }
            auto neural = render::neuralCloudFrom(library, file->count, file->latent, file->weights, file->lit);
            if (!neural) return report(neural.error());
            looks.back() = std::make_unique<render::NeuralCloud>(std::move(*neural));
            instance.neural = looks.back().get();
            ++fitted;
        }
        byLatents.push_back(instance);
    }
    render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    render::RenderTargets plain;
    render::RenderTargets learnt;
    const auto timed = [&](const std::vector<render::SplatInstance>& instances, render::RenderTargets& into)
        -> Result<double> {
        render::RenderSettings waited = settings;
        waited.timeStages = true;
        double best = 1.0e30;
        for (int round = 0; round < (o.compare ? 12 : 1); ++round) {
            auto stats = raster->render(camera, instances, waited, into);
            if (!stats) return std::move(stats).error();
            best = std::min(best, stats->totalMs);
        }
        return best;
    };
    const bool drawHarmonics = o.harmonics || o.compare;
    const bool drawLatents = !o.harmonics || o.compare;
    double harmonicsMs = 0.0;
    double latentsMs = 0.0;
    if (drawHarmonics) {
        auto ms = timed(byHarmonics, plain);
        if (!ms) return report(ms.error());
        harmonicsMs = *ms;
    }
    if (drawLatents) {
        auto ms = timed(byLatents, learnt);
        if (!ms) return report(ms.error());
        latentsMs = *ms;
    }
    uint64_t total = 0;
    for (const auto& cloud : clouds) {
        total += cloud->count;
    }
    out("%s: %zu pieces, %llu gaussians, %u drawn by their latents\n", scene->name.c_str(), clouds.size(),
        static_cast<unsigned long long>(total), fitted);
    if (o.compare) {
        auto hdr = render::compareHdr(library, learnt.colour, plain.colour, width, height);
        if (!hdr) return report(hdr.error());
        auto codes = render::compareImages(library, learnt.colour, plain.colour, width, height);
        if (!codes) return report(codes.error());
        out("the latents against the harmonics: relMSE %.5f, p99 %u, max %u; a frame at %ux%u: %.2f ms by the "
            "harmonics, %.2f ms by the latents\n",
            hdr->relMse, codes->p99, codes->max, width, height, harmonicsMs, latentsMs);
    }
    if (!o.output.empty()) {
        const auto write = [&](const render::RenderTargets& targets, const std::string& path) -> Result<void> {
            auto colour = targets.colour.readAll<float>(**device);
            if (!colour) return std::move(colour).error();
            auto depth = targets.depth.readAll<float>(**device);
            if (!depth) return std::move(depth).error();
            ATHENEA_TRY(io::writeExr(path, width, height, *colour, *depth));
            out("wrote %s\n", path.c_str());
            return ok();
        };
        const std::filesystem::path given = o.output;
        if (o.compare) {
            const std::string stem = (given.parent_path() / given.stem()).string();
            if (auto w = write(plain, stem + "_harmonics.exr"); !w) return report(w.error());
            if (auto w = write(learnt, stem + "_latents.exr"); !w) return report(w.error());
        } else if (auto w = write(o.harmonics ? plain : learnt, o.output); !w) {
            return report(w.error());
        }
    }
    return 0;
}

void run(CLI::App* cmd, const std::shared_ptr<SceneOptions>& options, int (*what)(const SceneOptions&)) {
    cmd->callback([options, what] {
        if (const int code = what(*options); code != 0) {
            throw CLI::RuntimeError(code);
        }
    });
}

}   // namespace

void addNeuralScene(CLI::App& app) {
    auto options = std::make_shared<SceneOptions>();
    SceneOptions& o = *options;
    CLI::App* cmd = app.add_subcommand("neural-scene", "a scene kept in pieces, each with its own neural appearance (.aths)");
    cmd->require_subcommand(1);

    CLI::App* add = cmd->add_subcommand("add", "a piece's gaussians into the scene");
    add->add_option("scene", o.directory, "the scene's directory (.aths); made where it is not")->required();
    add->add_option("--id", o.id, "the piece's name in the scene: letters, digits, _ and -");
    add->add_option("--cloud", o.cloud, "its gaussians: a USD ParticleField (athenea mesh2splat) or a splat file");
    add->add_option("--prim", o.prim, "where it came from in the source stage");
    add->add_option("--name", o.name, "the scene's name");
    add->add_option("--up-axis", o.upAxis, "Y or Z");
    add->add_option("--meters-per-unit", o.metersPerUnit, "the scene's unit, in metres");
    add->add_option("--source", o.source, "the stage the pieces are converted from");
    add->add_option("--licence", o.licence, "what the source asks of whoever uses it");
    add->add_option("--origin", o.origin, "where the piece came from: converted (from geometry), discovered (what a camera saw) or generated");
    add->add_option("--found-by", o.foundBy, "the camera that found it");
    add->add_option("--time", o.time, "when it was found: the piece joins the scene's snapshot of that time and camera");
    run(add, options, runAdd);

    CLI::App* camera = cmd->add_subcommand("camera", "a named view kept in the manifest");
    camera->add_option("scene", o.directory, "the scene's directory")->required();
    camera->add_option("--name", o.camera, "the view's name");
    camera->add_option("--eye", o.eye, "x y z")->expected(3);
    camera->add_option("--target", o.target, "x y z")->expected(3);
    camera->add_option("--up", o.up, "x y z")->expected(3);
    camera->add_option("--focal", o.focal, "mm");
    run(camera, options, runCamera);

    CLI::App* fit = cmd->add_subcommand("fit", "each piece's harmonics learnt by its own latents and network");
    fit->add_option("scene", o.directory, "the scene's directory")->required();
    fit->add_option("--piece", o.pieces, "only these pieces (repeatable)");
    fit->add_option("--steps", o.steps, "steps a piece, a camera each");
    fit->add_option("--size", o.size, "WIDTHxHEIGHT of the frames fitted and measured");
    fit->add_option("--held-out", o.heldOut, "cameras a piece is measured from and never fitted from");
    fit->add_option("--rate-decay", o.rateDecay, "what the rates have fallen to by the last step");
    fit->add_option("--view-smooth", o.viewSmooth, "as athenea neural-fit's");
    fit->add_flag("--split", o.split, "two networks a piece: one never told the view, one for what the view adds");
    run(fit, options, runFit);

    CLI::App* render = cmd->add_subcommand("render", "the whole scene from a view");
    render->add_option("scene", o.directory, "the scene's directory")->required();
    render->add_option("--camera", o.camera, "a view of the manifest's (default: its first)");
    render->add_option("--eye", o.eye, "a view of its own: x y z (with --target)")->expected(3);
    render->add_option("--target", o.target, "x y z")->expected(3);
    render->add_option("--up", o.up, "x y z")->expected(3);
    render->add_option("--focal", o.focal, "mm");
    render->add_option("--size", o.size, "WIDTHxHEIGHT");
    render->add_option("-o,--output", o.output, "an EXR; with --compare, <stem>_harmonics.exr and <stem>_latents.exr");
    render->add_flag("--harmonics", o.harmonics, "every piece by its harmonics, whatever appearance it has");
    render->add_flag("--discovered", o.discovered, "only the gaussians the looks have found (neural-scene look): the scene as it is known");
    render->add_option("--up-to", o.upTo, "--discovered: only the looks up to this one, counted from one (0: all)");
    render->add_flag("--compare", o.compare, "both ways: the latents against the harmonics, and a frame's time by each");
    run(render, options, runRender);

    CLI::App* look = cmd->add_subcommand("look", "one camera's look at one time: which gaussians it found, and how many the scene already held");
    look->add_option("scene", o.directory, "the scene's directory")->required();
    look->add_option("--camera", o.camera, "a view of the manifest's");
    look->add_option("--time", o.time, "when: with the camera, the look's snapshot");
    look->add_option("--size", o.size, "WIDTHxHEIGHT of the look's frame");
    look->add_option("--threshold", o.threshold, "how much of a pixel a gaussian gives the frame to count as seen");
    look->add_option("-o,--output", o.output, "the look's frame, an EXR");
    run(look, options, runLook);

    CLI::App* info = cmd->add_subcommand("info", "what the manifest holds");
    info->add_option("scene", o.directory, "the scene's directory")->required();
    run(info, options, runInfo);
}

}   // namespace athenea::cli
