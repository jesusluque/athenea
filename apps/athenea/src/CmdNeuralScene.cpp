// Copyright (c) 2026 jesus luque.
//
// athenea neural-scene: a scene kept in pieces (io/NeuralScene.h).
//
//   add     a piece's gaussians into the scene, counted and bounded on the device
//   camera  a named view kept in the manifest
//   fit     each piece's harmonics learnt by its own latents and network
//   render  the whole scene from a view, by the latents or by the harmonics
//   look    one camera's look at one time: which gaussians it found
//   discover a piece made from a frame: a gaussian a pixel the scene did not hold
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
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <sstream>
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
#include "athenea/gpu/Texture.h"
#include "athenea/io/NeuralScene.h"
#include "athenea/io/Png.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/render/Discovery.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/NeuralFit.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/SplatLift.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/technique/DisplayTransform.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

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
    std::string              frame;
    float                    tolerance = 0.06F;
    float                    spread = 1.0F;
    std::string              views;
    float                    repaint = 0.0F;
    std::string              framesDir;
    float                    refine = 1.5F;
    // render: the scene as it stood
    double                   at = -1.0;
    unsigned                 snapshot = 0;
    unsigned                 storedFrame = 0;
    float                    exposure = 0.0F;
    // shoot
    std::string              stage;
    unsigned                 samples = 64;
    unsigned                 passSamples = 1;
    unsigned                 bounces = 4;
    bool                     denoise = false;
    float                    rayMargin = 0.0F;
    unsigned                 coarsest = 8;
    float                    glassOpacity = 0.05F;
    float                    turned = 0.5F;
    std::vector<double>      times;
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
    if (ext == ".athg") {
        // What a render of the engine's own left: records as render::Discovery lays them out.
        auto held = io::readGaussians(path);
        if (!held) return std::move(held).error();
        if (held->floats != render::kRenderRecord) {
            return Error::make(ErrorCode::Unsupported, "{}: records of {} floats, and a render's are {}", path.string(),
                               held->floats, render::kRenderRecord);
        }
        io::RawSplats raw;
        raw.source = path.string();
        raw.count = held->count;
        raw.encoding = render::Discovery::renderEncoding();
        raw.records = std::move(held->records);
        loader.keepFaint(true);   // a record of what is gone counts every gaussian the file holds
        auto cloud = loader.upload(raw, 0);
        loader.keepFaint(false);
        return cloud;
    }
    return scene::loadSplatFile(loader, path.string(), 3);
}

/// A piece as it is drawn: its gaussians, less those a later look found gone.
/// With `upTo`, as it stood at that snapshot: less only what had gone by then.
Result<scene::GpuSplats> loadPiece(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                                   const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                                   uint32_t upTo = 0) {
    auto cloud = load(library, loader, directory / piece.cloud);
    if (!cloud || piece.gone.empty()) {
        return cloud;
    }
    auto held = io::readSeen(directory / piece.gone, cloud->count);
    if (!held) return std::move(held).error();
    auto record = discovery.record(cloud->count, *held);
    if (!record) return std::move(record).error();
    return discovery.without(*cloud, *record, upTo);
}

/// A FRAME OF THE SCENE WRITTEN: an EXR as it is (linear, with its depth), or -- `path` ending in .png --
/// the picture of it, through the standard view transform at `exposure` stops, encoded for an sRGB display
/// on the device. The picture is what is made of the scene afterwards; the scene is the render.
Result<void> writeFrame(gpu::ShaderLibrary& library, gpu::Device& device, const render::RenderTargets& targets,
                        uint32_t width, uint32_t height, const std::string& path, float exposure) {
    if (std::filesystem::path(path).extension() != ".png") {
        auto colour = targets.colour.readAll<float>(device);
        if (!colour) return std::move(colour).error();
        auto depth = targets.depth.readAll<float>(device);
        if (!depth) return std::move(depth).error();
        return io::writeExr(path, width, height, *colour, *depth);
    }
    auto display = technique::DisplayTransform::create(library);
    if (!display) return std::move(display).error();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA8Unorm;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopySource;
    desc.label = "scene.picture";
    auto texture = gpu::Texture::create(device, desc);
    if (!texture) return std::move(texture).error();
    technique::DisplaySource source;
    source.kind = technique::DisplaySource::Kind::Colour;
    source.buffer = &targets.colour;
    source.width = width;
    source.height = height;
    technique::DisplaySettings settings;
    settings.view = technique::ViewTransform::Standard;
    settings.display = technique::DisplayEncoding::Srgb;
    settings.exposure = exposure;
    {
        gpu::CommandBatch batch(device);
        ATHENEA_TRY(display->run(batch, source, settings, texture->rhi(), width, height));
        ATHENEA_TRY(batch.submit(true));
    }
    auto texels = texture->read(device, 0, 0);
    if (!texels) return std::move(texels).error();
    return io::writePng(path, width, height,
                        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(texels->data()), texels->size()));
}

/// Whether a box may be seen through `projection`: not when its eight corners are all behind the eye, or
/// all off the same side of the frame. Which pieces are read is the files' bookkeeping; what is drawn of
/// them the device decides.
bool boxInView(const std::array<float, 3>& min, const std::array<float, 3>& max, const render::Projection& projection,
               uint32_t width, uint32_t height) {
    int behind = 0, left = 0, right = 0, below = 0, above = 0;
    for (int corner = 0; corner < 8; ++corner) {
        const double p[3] = {(corner & 1) != 0 ? max[0] : min[0], (corner & 2) != 0 ? max[1] : min[1],
                             (corner & 4) != 0 ? max[2] : min[2]};
        double v[3];
        for (int r = 0; r < 3; ++r) {
            v[r] = projection.worldToView.at(r, 0) * p[0] + projection.worldToView.at(r, 1) * p[1] +
                   projection.worldToView.at(r, 2) * p[2] + projection.worldToView.at(r, 3);
        }
        if (v[2] <= 0.0) {
            ++behind;
            continue;
        }
        const double x = v[0] / v[2] * projection.focalX + projection.centreX;
        const double y = v[1] / v[2] * projection.focalY + projection.centreY;
        left += x < 0.0 ? 1 : 0;
        right += x > static_cast<double>(width) ? 1 : 0;
        below += y < 0.0 ? 1 : 0;
        above += y > static_cast<double>(height) ? 1 : 0;
    }
    // A corner behind the eye says nothing of which side the box is off: only a box wholly in front is
    // judged by its sides.
    return behind != 8 && (behind != 0 || (left != 8 && right != 8 && below != 8 && above != 8));
}

/// The last snapshot taken by time `at`, counted from one; 0 where none was.
uint32_t snapshotAt(const io::NeuralScene& scene, double at) {
    uint32_t last = 0;
    for (size_t k = 0; k < scene.snapshots.size(); ++k) {
        if (scene.snapshots[k].time <= at) {
            last = static_cast<uint32_t>(k) + 1;
        }
    }
    return last;
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
        if (!p.gone.empty()) {
            out(", %u gone", p.removed);
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
    for (size_t k = 0; k < scene->frames.size(); ++k) {
        const io::NeuralSceneFrame& f = scene->frames[k];
        out("  frame %zu at %g by %s: %ux%u, %u paths a pixel, snapshots %u to %u, %.1f s, taken %s\n", k + 1, f.time,
            f.camera.c_str(), f.width, f.height, f.samples, f.firstSnapshot, f.lastSnapshot, f.seconds, f.taken.c_str());
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
        auto sceneDiscovery = render::Discovery::create(library);
        if (!sceneDiscovery) return report(sceneDiscovery.error());
        auto cloud = loadPiece(library, *loader, *sceneDiscovery, directory, piece);
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
        auto cloud = loadPiece(library, *loader, *discovery, directory, piece);
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

/// A PIECE MADE FROM A FRAME. Any renderer's frame -- its colour, and its depth as an EXR's Z -- through a
/// camera of the manifest's: a gaussian a pixel whose surface the scene, drawn through that camera, does
/// not already hold. The frame's own size is the look's.
int runDiscover(const SceneOptions& all) {
    const std::filesystem::path directory = all.directory;
    io::NeuralScene scene;
    if (std::filesystem::exists(directory / "scene.json")) {
        auto read = io::readNeuralScene(directory);
        if (!read) return report(read.error());
        scene = std::move(*read);
    }
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto discovery = render::Discovery::create(library);
    if (!discovery) return report(discovery.error());
    // The scene as it is known, on the device: read once, and kept from one look to the next.
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<render::SplatInstance> instances;
    for (const io::NeuralScenePiece& piece : scene.pieces) {
        auto cloud = loadPiece(library, *loader, *discovery, directory, piece);
        if (!cloud) return report(cloud.error());
        clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
        instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
    }
    // ONE LOOK: a frame, the camera it was drawn through, and when.
    const auto oneLook = [&](const SceneOptions& o) -> int {
    const auto began = std::chrono::steady_clock::now();
    // A scene that starts with a look has nothing but what the look's command says of it.
    if (scene.name.empty()) scene.name = directory.stem().string();
    if (!o.name.empty()) scene.name = o.name;
    if (!o.upAxis.empty()) scene.upAxis = o.upAxis;
    if (o.metersPerUnit > 0.0) scene.metersPerUnit = o.metersPerUnit;
    if (!o.source.empty()) scene.source = o.source;
    if (!o.licence.empty()) scene.licence = o.licence;
    if (o.eye.size() == 3 && o.target.size() == 3 && !o.camera.empty()) {
        // A view of the look's own is kept under the name it was given.
        io::NeuralSceneCamera named;
        named.name = o.camera;
        named.eye = {o.eye[0], o.eye[1], o.eye[2]};
        named.target = {o.target[0], o.target[1], o.target[2]};
        const render::Vec3 up = o.up.size() == 3 ? render::Vec3{o.up[0], o.up[1], o.up[2]} : upOf(scene);
        named.up = {up.x, up.y, up.z};
        named.focal = o.focal > 0.0 ? o.focal : 35.0;
        std::erase_if(scene.cameras, [&](const io::NeuralSceneCamera& c) { return c.name == named.name; });
        scene.cameras.push_back(named);
    }
    if (o.frame.empty() || o.id.empty() || o.camera.empty() || o.time < 0.0) {
        err("a discovery wants --frame (an EXR with Z), --id, --camera (one of the manifest's) and --time\n");
        return kExitFailure;
    }
    if (scene.piece(o.id) != nullptr) {
        err("the scene already has a piece %s: a look's piece is its own\n", o.id.c_str());
        return kExitFailure;
    }
    const auto camera = viewOf(o, scene);
    if (!camera) return kExitFailure;
    auto channels = io::readExrChannels(o.frame);
    if (!channels) return report(channels.error());
    const uint32_t width = channels->width;
    const uint32_t height = channels->height;
    const auto channel = [&](const char* name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : channels->channels) {
            if (c.name == name) {
                return &c;
            }
        }
        return nullptr;
    };
    const io::ExrChannelData* red = channel("R");
    const io::ExrChannelData* green = channel("G");
    const io::ExrChannelData* blue = channel("B");
    const io::ExrChannelData* alpha = channel("A");
    const io::ExrChannelData* z = channel("Z");
    if (red == nullptr || green == nullptr || blue == nullptr || z == nullptr) {
        err("%s wants R, G, B and Z: a frame with its depth\n", o.frame.c_str());
        return kExitFailure;
    }
    // The channels laid side by side as the device takes a frame: nothing is computed.
    const size_t pixels = size_t{width} * height;
    std::vector<std::array<float, 4>> colour(pixels);
    std::vector<float> depth(pixels);
    for (size_t k = 0; k < pixels; ++k) {
        std::memcpy(&colour[k][0], &red->words[k], sizeof(float));
        std::memcpy(&colour[k][1], &green->words[k], sizeof(float));
        std::memcpy(&colour[k][2], &blue->words[k], sizeof(float));
        colour[k][3] = 1.0F;
        if (alpha != nullptr) {
            std::memcpy(&colour[k][3], &alpha->words[k], sizeof(float));
        }
        std::memcpy(&depth[k], &z->words[k], sizeof(float));
    }
    auto frameColour = gpu::Buffer::fromSpan<std::array<float, 4>>(
        **device, std::span<const std::array<float, 4>>(colour), "discover.frame");
    if (!frameColour) return report(frameColour.error());
    auto frameDepth = gpu::Buffer::fromSpan<float>(**device, std::span<const float>(depth), "discover.depth");
    if (!frameDepth) return report(frameDepth.error());

    render::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    // The scene's depth at a pixel is its first surface's -- where half the pixel is covered -- and not
    // the mean of everything blended there: at an edge of depth the mean lies between the two surfaces
    // and is neither, and a frame would be taken to see through what it does not.
    settings.depth = render::RenderSettings::Depth::Threshold;
    settings.depthThreshold = 0.5F;
    render::RenderTargets known;
    // And how coarse it is there: the same pieces drawn with each gaussian's size as its colour.
    std::vector<std::unique_ptr<render::NeuralCloud>> sized;
    std::vector<render::SplatInstance> bySize;
    for (size_t k = 0; k < clouds.size(); ++k) {
        auto made = discovery->sizes(*clouds[k]);
        if (!made) return report(made.error());
        sized.push_back(std::make_unique<render::NeuralCloud>(std::move(*made)));
        render::SplatInstance instance = instances[k];
        instance.neural = sized.back().get();
        bySize.push_back(instance);
    }
    render::RenderTargets knownSize;
    const auto drawKnown = [&]() -> Result<void> {
        auto drawn = raster->render(*camera, instances, settings, known);
        if (!drawn) return std::move(drawn).error();
        auto sizes = raster->render(*camera, bySize, settings, knownSize);
        if (!sizes) return std::move(sizes).error();
        return ok();
    };
    if (!instances.empty()) {
        if (auto drawn = drawKnown(); !drawn) return report(drawn.error());
    }
    const render::Projection projection = render::projectionFor(*camera, width, height);
    // WHAT THE FRAME SEES THROUGH IS TAKEN OUT FIRST. Where the scene has a surface nearer than the frame's,
    // what stood there has moved or gone: the gaussians that made it are stamped gone, and the scene drawn
    // again without them, so that what the frame shows in their place is found as new.
    size_t place = scene.snapshots.size();
    for (size_t k = 0; k < scene.snapshots.size(); ++k) {
        if (scene.snapshots[k].time == o.time && scene.snapshots[k].camera == o.camera) {
            place = k;
        }
    }
    const uint32_t stamp = static_cast<uint32_t>(place) + 1;
    if (!instances.empty()) {
        // Layer by layer: what stood in front is taken out, what it hid is then in the scene's frame, and
        // the frame may see through that too. Until it sees through nothing more, a few times at most.
        uint32_t firstThrough = 0;
        uint32_t firstRepainted = 0;
        uint32_t taken = 0;
        uint32_t passes = 0;
        std::vector<gpu::Buffer> records(scene.pieces.size());
        std::vector<uint32_t> newly(scene.pieces.size(), 0);
        for (; passes < 8; ++passes) {
            uint32_t through = 0;
            uint32_t repainted = 0;
            auto mask = discovery->seenThrough(*frameColour, *frameDepth, width, height, known.colour, known.depth,
                                               o.tolerance, through, o.repaint, &repainted, &knownSize.colour,
                                               o.refine, &projection, o.spread);
            if (!mask) return report(mask.error());
            if (passes == 0) {
                firstThrough = through;
                firstRepainted = repainted;
            }
            if (through + repainted == 0) {
                break;
            }
            uint32_t slots = 0;
            for (const auto& cloud : clouds) {
                slots += cloud->count;
            }
            auto lift = render::makeSplatLift(**device, slots);
            if (!lift) return report(lift.error());
            render::RenderSettings carrying = settings;
            render::liftWhileBlending(carrying, *lift, *mask);
            render::RenderTargets carried;
            auto lifted = raster->render(*camera, instances, carrying, carried);
            if (!lifted) return report(lifted.error());
            uint32_t base = 0;
            uint32_t now = 0;
            for (size_t k = 0; k < scene.pieces.size(); ++k) {
                const io::NeuralScenePiece& piece = scene.pieces[k];
                const uint32_t count = clouds[k]->count;
                if (!records[k].valid()) {
                    std::vector<uint32_t> held;
                    if (!piece.gone.empty()) {
                        auto read = io::readSeen(directory / piece.gone, count);
                        if (!read) return report(read.error());
                        held = std::move(*read);
                    }
                    auto gone = discovery->record(count, held);
                    if (!gone) return report(gone.error());
                    records[k] = *gone;
                }
                auto marked = discovery->markGone(lift->sums, base, records[k], count, stamp, 0.05F, clouds[k].get(), &projection,
                                                  o.refine * o.spread);
                if (!marked) return report(marked.error());
                if (*marked > 0) {
                    auto kept = discovery->without(*clouds[k], records[k]);
                    if (!kept) return report(kept.error());
                    *clouds[k] = std::move(*kept);
                    newly[k] += *marked;
                    now += *marked;
                }
                base += count;
            }
            if (now == 0) {
                break;
            }
            taken += now;
            if (auto again = drawKnown(); !again) return report(again.error());
        }
        for (size_t k = 0; k < scene.pieces.size(); ++k) {
            if (newly[k] == 0) {
                continue;
            }
            io::NeuralScenePiece& piece = scene.pieces[k];
            auto words = records[k].readAll<uint32_t>(**device);
            if (!words) return report(words.error());
            words->resize(clouds[k]->count);
            piece.gone = "pieces/" + piece.id + ".gone";
            if (auto written = io::writeSeen(directory / piece.gone, *words); !written) return report(written.error());
            piece.removed += newly[k];
            ++piece.revision;
            piece.movedAt = piece.revision;
        }
        out("the frame sees through the scene at %u pixels and finds %u to be made again (repainted, or coarser than this look); %u gaussians taken out in %u passes\n",
            firstThrough, firstRepainted, taken, passes);
    }
    auto look = discovery->fromFrame(*frameColour, *frameDepth, width, height, projection,
                                     instances.empty() ? nullptr : &known.colour,
                                     instances.empty() ? nullptr : &known.depth, o.tolerance, o.spread, o.repaint,
                                     instances.empty() ? nullptr : &knownSize.colour, o.refine);
    if (!look) return report(look.error());
    out("%s through %s at %g: %u of its %zu pixels hold a surface; the scene already held %u of them (%.1f%%); %u "
        "gaussians made\n",
        o.frame.c_str(), o.camera.c_str(), o.time, look->surfaces, pixels, look->held,
        look->surfaces > 0 ? 100.0 * look->held / look->surfaces : 0.0, look->made);
    // Written either way: a look that made nothing may still have taken gaussians out.
    if (auto made = io::writeNeuralScene(directory, scene); !made) return report(made.error());
    if (look->made == 0) {
        return 0;
    }
    const std::string kept = "pieces/" + o.id + ".ply";
    const std::vector<std::vector<float>> chunks{std::move(look->records)};
    const std::vector<std::string> comments{"athenea neural-scene discover: a gaussian a pixel of " + o.frame};
    if (auto written = io::writePly3dgs(directory / kept, chunks, comments); !written) return report(written.error());
    auto cloud = load(library, *loader, directory / kept);
    if (!cloud) return report(cloud.error());
    io::NeuralScenePiece piece;
    piece.id = o.id;
    piece.cloud = kept;
    piece.gaussians = cloud->count;
    piece.boundsMin = cloud->bounds.min;
    piece.boundsMax = cloud->bounds.max;
    piece.origin = "discovered";
    piece.foundBy = o.camera;
    piece.foundAt = o.time;
    scene.pieces.push_back(piece);
    io::NeuralSceneSnapshot* held = nullptr;
    for (io::NeuralSceneSnapshot& s : scene.snapshots) {
        if (s.time == o.time && s.camera == o.camera) {
            held = &s;
        }
    }
    if (held == nullptr) {
        scene.snapshots.push_back({o.time, o.camera, {}});
        held = &scene.snapshots.back();
    }
    held->pieces.push_back(o.id);
    if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());

    // THE LOOK GIVEN BACK: the scene with the new piece, through the look's camera, against the frame.
    clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
    instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
    render::RenderTargets again;
    auto drawn = raster->render(*camera, instances, settings, again);
    if (!drawn) return report(drawn.error());
    auto hdr = render::compareHdr(library, again.colour, *frameColour, width, height);
    if (!hdr) return report(hdr.error());
    auto codes = render::compareImages(library, again.colour, *frameColour, width, height);
    if (!codes) return report(codes.error());
    out("piece %s, %u gaussians; the scene through that camera against the frame: relMSE %.5f, p99 %u, max %u; the "
        "look took %.2f s\n",
        o.id.c_str(), piece.gaussians, hdr->relMse, codes->p99, codes->max,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
    if (!o.output.empty()) {
        auto colourBack = again.colour.readAll<float>(**device);
        if (!colourBack) return report(colourBack.error());
        auto depthBack = again.depth.readAll<float>(**device);
        if (!depthBack) return report(depthBack.error());
        if (auto w = io::writeExr(o.output, width, height, *colourBack, *depthBack); !w) return report(w.error());
        out("wrote %s\n", o.output.c_str());
    }
    return 0;
    };
    if (all.views.empty()) {
        return oneLook(all);
    }
    // SEVERAL LOOKS, the scene on the device throughout: a file of views as `athenea stage --cameras` takes
    // one, each with its frame NAME.exr, each a piece `look_NAME` at a time one after the last's.
    std::ifstream in(all.views);
    if (!in) {
        err("cannot open %s\n", all.views.c_str());
        return kExitFailure;
    }
    const std::filesystem::path beside =
        all.framesDir.empty() ? std::filesystem::path(all.views).parent_path() : std::filesystem::path(all.framesDir);
    double time = all.time >= 0.0 ? all.time : 0.0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream words(line);
        std::string name;
        double e[3];
        double t[3];
        if (!(words >> name >> e[0] >> e[1] >> e[2] >> t[0] >> t[1] >> t[2])) {
            continue;
        }
        SceneOptions one = all;
        one.frame = (beside / (name + ".exr")).string();
        one.id = "look_" + name;
        one.camera = name;
        one.eye = {e[0], e[1], e[2]};
        one.target = {t[0], t[1], t[2]};
        double u[3];
        if (words >> u[0] >> u[1] >> u[2]) {
            one.up = {u[0], u[1], u[2]};
        }
        one.time = time;
        one.output.clear();
        if (const int code = oneLook(one); code != 0) {
            return code;
        }
        time += 1.0;
    }
    return 0;
}

/// A RENDER WHOSE RESULT IS THE SCENE. A USD stage is path traced through a camera, and what the render
/// leaves is gaussians, not a picture: each as wide as its pixel is at its distance, each of its
/// surface's material -- an opaque thing a disc in its surface, a pane the pane itself and, apart, what
/// is seen through it where the bent ray ends. They are found while the render gathers: a first look at
/// cells of several pixels as soon as there are paths enough for those, and finer looks over it as the
/// paths accumulate, each taking out the coarser gaussians it replaces. Every look is a snapshot with
/// its time, and nothing leaves the disk: the scene accumulates, and `render --at` draws it as it stood.
/// A picture is what `neural-scene render` makes of the scene afterwards.
int runShoot(const SceneOptions& all) {
    uint32_t width = 0;
    uint32_t height = 0;
    if (!sizeOf(all.size, width, height)) return kExitFailure;
    const std::filesystem::path directory = all.directory;
    io::NeuralScene scene;
    if (std::filesystem::exists(directory / "scene.json")) {
        auto read = io::readNeuralScene(directory);
        if (!read) return report(read.error());
        scene = std::move(*read);
    }
    std::error_code made;
    std::filesystem::create_directories(directory / "pieces", made);
    auto renderer = usd::StageRenderer::open(all.stage);
    if (!renderer) return report(renderer.error());
    usd::StageRenderer& stage = **renderer;
    stage.setPathSamples(std::max(all.passSamples, 1u));
    stage.setPathTotal(std::max(all.samples, 1u));
    stage.setPathBounces(all.bounces);
    stage.setDenoise(all.denoise);
    if (all.rayMargin > 0.0F) stage.setRayMargin(all.rayMargin);
    // The path tracer's planes beside the picture: what each pixel's surface is made of.
    stage.requestOutputs({"material"});
    gpu::Device& device = stage.device();
    gpu::ShaderLibrary& library = stage.library();
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    auto raster = render::TileRasterizer::create(library);
    if (!raster) return report(raster.error());
    auto discovery = render::Discovery::create(library);
    if (!discovery) return report(discovery.error());
    if (scene.name.empty()) scene.name = directory.stem().string();
    if (!all.name.empty()) scene.name = all.name;
    scene.upAxis = !all.upAxis.empty() ? all.upAxis : std::string(1, stage.upAxis());
    if (all.metersPerUnit > 0.0) scene.metersPerUnit = all.metersPerUnit;
    if (scene.source.empty()) scene.source = all.source.empty() ? all.stage : all.source;
    if (!all.licence.empty()) scene.licence = all.licence;

    // THE SCENE IS ON THE DISK, AND THE DEVICE HOLDS WHAT A VIEW SEES OF IT. Each look's gaussians are
    // written as they are made, and when a view is done everything of the scene leaves the device; the
    // next view reads back only the pieces whose boxes its frustum meets. So what a shoot can hold is
    // not bounded by the device, which also holds the stage being rendered -- Sponza's textures alone are
    // two thirds of it.
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<render::SplatInstance> instances;
    std::vector<size_t> pieceOf;             // an instance's piece in the manifest
    std::vector<gpu::Buffer> gone;           // its record, once one is needed
    const auto loadInView = [&](const render::Camera& camera) -> Result<uint32_t> {
        const render::Projection projection = render::projectionFor(camera, width, height);
        uint32_t skipped = 0;
        for (size_t k = 0; k < scene.pieces.size(); ++k) {
            const io::NeuralScenePiece& piece = scene.pieces[k];
            if (piece.removed != 0 && piece.removed >= piece.gaussians) {
                continue;   // nothing of it stands
            }
            if (!boxInView(piece.boundsMin, piece.boundsMax, projection, width, height)) {
                ++skipped;
                continue;
            }
            auto cloud = loadPiece(library, *loader, *discovery, directory, piece);
            if (!cloud) return std::move(cloud).error();
            clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
            instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
            pieceOf.push_back(k);
            gone.emplace_back();
        }
        return skipped;
    };
    const auto release = [&] {
        instances.clear();
        clouds.clear();
        pieceOf.clear();
        gone.clear();
    };
    const auto clock = [] {
        const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        char text[32];
        std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
        return std::string(text);
    };

    // ONE LOOK AT ONE FOOTPRINT: the render as it stands on the device, taken into the scene.
    const auto takeLook = [&](const render::Camera& camera, const std::string& name, double time, uint32_t footprint,
                              uint32_t samples, bool last, uint32_t frameFirst) -> int {
        const auto began = std::chrono::steady_clock::now();
        auto colourOf = stage.displaySource("color");
        auto depthOf = stage.displaySource("depth");
        auto planesOf = stage.displaySource("material");
        if (!colourOf) return report(colourOf.error());
        if (!depthOf) return report(depthOf.error());
        if (!planesOf) return report(planesOf.error());
        if (colourOf->buffer == nullptr || depthOf->buffer == nullptr || planesOf->buffer == nullptr) {
            err("the render left no colour, depth or material planes: neural-scene shoot wants a path traced stage of meshes\n");
            return kExitFailure;
        }
        const gpu::Buffer& frameColour = *colourOf->buffer;
        const gpu::Buffer& frameDepth = *depthOf->buffer;
        const gpu::Buffer& planes = *planesOf->buffer;
        scene.snapshots.push_back({time, name, {}, samples, footprint, clock()});
        const uint32_t stamp = static_cast<uint32_t>(scene.snapshots.size());

        render::RenderSettings settings;
        settings.width = width;
        settings.height = height;
        settings.depth = render::RenderSettings::Depth::Threshold;
        settings.depthThreshold = 0.5F;
        // Where the scene begins: a pane's own depth, which the depth above goes past.
        render::RenderSettings beginning = settings;
        beginning.depthThreshold = 0.4F * all.glassOpacity;   // under the least a pane covers
        render::RenderTargets known;
        render::RenderTargets knownFirst;
        render::RenderTargets knownSize;
        std::vector<std::unique_ptr<render::NeuralCloud>> sized;
        const auto drawKnown = [&]() -> Result<void> {
            auto drawn = raster->render(camera, instances, settings, known);
            if (!drawn) return std::move(drawn).error();
            auto first = raster->render(camera, instances, beginning, knownFirst);
            if (!first) return std::move(first).error();
            sized.clear();
            std::vector<render::SplatInstance> bySize;
            for (size_t k = 0; k < clouds.size(); ++k) {
                auto size = discovery->sizes(*clouds[k]);
                if (!size) return std::move(size).error();
                sized.push_back(std::make_unique<render::NeuralCloud>(std::move(*size)));
                render::SplatInstance instance = instances[k];
                instance.neural = sized.back().get();
                bySize.push_back(instance);
            }
            auto sizes = raster->render(camera, bySize, settings, knownSize);
            if (!sizes) return std::move(sizes).error();
            // The clouds of sizes are as large as the scene's own and wanted for that one frame.
            sized.clear();
            return ok();
        };
        const render::Projection projection = render::projectionFor(camera, width, height);
        const float disc = all.spread * static_cast<float>(footprint);
        uint32_t taken = 0;
        uint32_t firstThrough = 0;
        uint32_t firstAgain = 0;
        if (!instances.empty()) {
            if (auto drawn = drawKnown(); !drawn) return report(drawn.error());
            // What the render sees through, and what it would make finer, is taken out first, layer by layer.
            std::vector<uint32_t> newly(instances.size(), 0);
            for (uint32_t pass = 0; pass < 8; ++pass) {
                uint32_t through = 0;
                uint32_t again = 0;
                auto mask = discovery->seenThrough(frameColour, frameDepth, width, height, known.colour, known.depth,
                                                   all.tolerance, through, all.repaint, &again, &knownSize.colour,
                                                   all.refine, &projection, disc);
                if (!mask) return report(mask.error());
                if (pass == 0) {
                    firstThrough = through;
                    firstAgain = again;
                }
                if (through + again == 0) {
                    break;
                }
                uint32_t slots = 0;
                for (const auto& cloud : clouds) {
                    slots += cloud->count;
                }
                // THE MASK IS CARRIED TWICE. Whole, to this frame's own earlier looks: the coarse discs a finer
                // look replaces, glass's pictures among them. And struck where the render's surface lets
                // the eye through, to everything older: nothing found before is taken out through glass,
                // where the scene's depth is not the surface's and the mask would reach the pane and all
                // that is seen through it (a look from a second side took the floor behind a pane out).
                uint32_t now = 0;
                for (int older = 0; older < 2; ++older) {
                if (older == 1) {
                    if (auto spared = discovery->sparePanes(*mask, planes, width, height); !spared) return report(spared.error());
                }
                auto lift = render::makeSplatLift(device, slots);
                if (!lift) return report(lift.error());
                render::RenderSettings carrying = settings;
                render::liftWhileBlending(carrying, *lift, *mask);
                render::RenderTargets carried;
                auto lifted = raster->render(camera, instances, carrying, carried);
                if (!lifted) return report(lifted.error());
                uint32_t base = 0;
                for (size_t k = 0; k < instances.size(); ++k) {
                    const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                    const uint32_t count = clouds[k]->count;
                    if ((piece.foundIn >= frameFirst) != (older == 0)) {
                        base += count;
                        continue;
                    }
                    if (!gone[k].valid()) {
                        std::vector<uint32_t> held;
                        if (!piece.gone.empty()) {
                            auto read = io::readSeen(directory / piece.gone, count);
                            if (!read) return report(read.error());
                            held = std::move(*read);
                        }
                        auto record = discovery->record(count, held);
                        if (!record) return report(record.error());
                        gone[k] = *record;
                    }
                    auto marked = discovery->markGone(lift->sums, base, gone[k], count, stamp, 0.05F, clouds[k].get(),
                                                      &projection, all.refine * disc);
                    if (!marked) return report(marked.error());
                    if (*marked > 0) {
                        auto kept = discovery->without(*clouds[k], gone[k]);
                        if (!kept) return report(kept.error());
                        *clouds[k] = std::move(*kept);
                        newly[k] += *marked;
                        now += *marked;
                    }
                    base += count;
                }
                }
                if (now == 0) {
                    break;
                }
                taken += now;
                if (auto again2 = drawKnown(); !again2) return report(again2.error());
            }
            for (size_t k = 0; k < instances.size(); ++k) {
                if (newly[k] == 0) {
                    continue;
                }
                io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                auto words = gone[k].readAll<uint32_t>(device);
                if (!words) return report(words.error());
                words->resize(clouds[k]->count);
                piece.gone = "pieces/" + piece.id + ".gone";
                if (auto written = io::writeSeen(directory / piece.gone, *words); !written) return report(written.error());
                piece.removed += newly[k];
                ++piece.revision;
                piece.movedAt = piece.revision;
            }
            // A piece nothing stands of leaves the device: a coarse look's, once the finer one is in. The
            // disk keeps it, with the stamp each of its gaussians went at.
            for (size_t k = instances.size(); k-- > 0;) {
                const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                if (piece.removed >= piece.gaussians) {
                    instances.erase(instances.begin() + static_cast<std::ptrdiff_t>(k));
                    clouds.erase(clouds.begin() + static_cast<std::ptrdiff_t>(k));
                    pieceOf.erase(pieceOf.begin() + static_cast<std::ptrdiff_t>(k));
                    gone.erase(gone.begin() + static_cast<std::ptrdiff_t>(k));
                }
            }
            sized.clear();
            known = {};
            knownFirst = {};
            knownSize = {};
            if (!instances.empty()) {
                if (auto drawn = drawKnown(); !drawn) return report(drawn.error());
            }
        }
        render::KnownScene held;
        if (!instances.empty()) {
            held.colour = &known.colour;
            held.depth = &known.depth;
            held.first = &knownFirst.depth;
            held.size = &knownSize.colour;
        }
        render::RenderLookSettings how;
        how.tolerance = all.tolerance;
        how.spread = all.spread;
        how.repaint = all.repaint;
        how.refine = all.refine;
        how.footprint = footprint;
        how.glassOpacity = all.glassOpacity;
        how.turned = all.turned;
        // Glass is made as glass by the view's last look; the coarse ones before it carry its picture.
        how.panes = last;
        auto look = discovery->fromRender(frameColour, frameDepth, planes, width, height, projection, held, how);
        if (!look) return report(look.error());
        std::string id;
        if (look->made > 0) {
            char label[96];
            std::snprintf(label, sizeof(label), "%s_t%g_f%u_s%u", name.c_str(), time, footprint, stamp);
            id = label;
            const std::string kept = "pieces/" + id + ".athg";
            auto floats = look->records.readAll<float>(device);
            if (!floats) return report(floats.error());
            if (auto written = io::writeGaussians(directory / kept, look->made, render::kRenderRecord, *floats); !written) {
                return report(written.error());
            }
            // The piece is drawn from the records the device already holds.
            loader->keepFaint(true);
            auto cloud = loader->upload(look->records, look->made, render::Discovery::renderEncoding(), kept, 0);
            loader->keepFaint(false);
            if (!cloud) return report(cloud.error());
            io::NeuralScenePiece piece;
            piece.id = id;
            piece.cloud = kept;
            piece.prim = all.stage;
            piece.gaussians = cloud->count;
            piece.boundsMin = cloud->bounds.min;
            piece.boundsMax = cloud->bounds.max;
            piece.origin = "discovered";
            piece.foundBy = name;
            piece.foundAt = time;
            piece.foundIn = stamp;
            piece.footprint = footprint;
            scene.pieces.push_back(piece);
            scene.snapshots.back().pieces.push_back(id);
            clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
            instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
            pieceOf.push_back(scene.pieces.size() - 1);
            gone.emplace_back();
        }
        if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
        out("  %s at %g, %u paths a pixel, cells of %u: %u hold a surface, the scene held %u (%.1f%%), %u let the eye "
            "through; %u gaussians made (%u of them behind glass), %u taken out (seen through at %u pixels, finer at "
            "%u); %.2f s\n",
            name.c_str(), time, samples, footprint, look->surfaces, look->held,
            look->surfaces > 0 ? 100.0 * look->held / look->surfaces : 0.0, look->panes, look->made, look->behind,
            taken, firstThrough, firstAgain,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        return 0;
    };

    // ONE VIEW AT ONE TIME: drawn a pass at a time, and looked at as it gathers.
    const auto shootView = [&](const render::Camera& camera, const std::string& name, double time) -> int {
        const auto began = std::chrono::steady_clock::now();
        auto outOfView = loadInView(camera);
        if (!outOfView) return report(outOfView.error());
        const size_t read = instances.size();
        // The footprints, coarsest first, and the paths a pixel each waits for: as many a cell as the
        // last has a pixel.
        std::vector<std::pair<uint32_t, uint32_t>> levels;
        for (uint32_t footprint = std::max(all.coarsest, 1u); footprint >= 1; footprint /= 2) {
            const uint32_t wants = std::max<uint32_t>(all.samples / (footprint * footprint), 1);
            if (levels.empty() || wants > levels.back().second || footprint == 1) {
                if (!levels.empty() && wants <= levels.back().second) {
                    levels.pop_back();   // the finer look is no later: the coarser one is not taken
                }
                levels.emplace_back(footprint, wants);
            }
            if (footprint == 1) {
                break;
            }
        }
        size_t next = 0;
        double firstAt = 0.0;
        const uint32_t firstSnapshot = static_cast<uint32_t>(scene.snapshots.size()) + 1;
        for (uint32_t pass = 0; pass < 65536 && next < levels.size(); ++pass) {
            if (auto drawn = stage.draw(camera, time, width, height, "rt"); !drawn) return report(drawn.error());
            const uint32_t has = stage.pathAccumulated();
            const bool whole = stage.pathConverged();
            // The last look waits for the frame to be whole (and denoised, where that was asked).
            while (next < levels.size() &&
                   (next + 1 == levels.size() ? whole : has >= levels[next].second)) {
                if (const int code = takeLook(camera, name, time, levels[next].first, has, next + 1 == levels.size(),
                                              firstSnapshot);
                    code != 0) {
                    return code;
                }
                if (next == 0) {
                    firstAt = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                }
                ++next;
            }
            if (whole) {
                break;
            }
        }
        uint32_t standing = 0;
        for (const io::NeuralScenePiece& piece : scene.pieces) {
            standing += piece.gaussians - std::min(piece.removed, piece.gaussians);
        }
        // The frame is stored: what it took, by which camera and when, and the snapshots that are its looks.
        io::NeuralSceneFrame frame;
        frame.time = time;
        frame.camera = name;
        frame.width = width;
        frame.height = height;
        frame.samples = stage.pathAccumulated();
        frame.firstSnapshot = firstSnapshot;
        frame.lastSnapshot = static_cast<uint32_t>(scene.snapshots.size());
        frame.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        frame.taken = clock();
        scene.frames.push_back(frame);
        if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
        release();
        out("%s at %g: frame %zu, a scene to draw after %.1f s and the whole look after %.1f s; %u gaussians stand in "
            "%zu pieces, %zu snapshots; %zu pieces were read for this view and %u left on the disk, and the device "
            "is free of the scene again\n",
            name.c_str(), time, scene.frames.size(), firstAt, frame.seconds, standing, scene.pieces.size(),
            scene.snapshots.size(), read, *outOfView);
        return 0;
    };

    std::vector<double> times = all.times;
    if (times.empty()) {
        times.push_back(all.time >= 0.0 ? all.time : 0.0);
    }
    const auto keepCamera = [&](const std::string& name, const std::array<double, 3>& eye,
                                const std::array<double, 3>& target, const render::Vec3& up, double focal) {
        io::NeuralSceneCamera named;
        named.name = name;
        named.eye = eye;
        named.target = target;
        named.up = {up.x, up.y, up.z};
        named.focal = focal;
        std::erase_if(scene.cameras, [&](const io::NeuralSceneCamera& c) { return c.name == named.name; });
        scene.cameras.push_back(named);
    };
    const double focal = all.focal > 0.0 ? all.focal : 35.0;
    for (const double time : times) {
        if (!all.views.empty()) {
            std::ifstream in(all.views);
            if (!in) {
                err("cannot open %s\n", all.views.c_str());
                return kExitFailure;
            }
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') {
                    continue;
                }
                std::istringstream words(line);
                std::string name;
                double e[3];
                double t[3];
                if (!(words >> name >> e[0] >> e[1] >> e[2] >> t[0] >> t[1] >> t[2])) {
                    continue;
                }
                render::Vec3 up = upOf(scene);
                double u[3];
                if (words >> u[0] >> u[1] >> u[2]) {
                    up = {u[0], u[1], u[2]};
                }
                render::Camera camera = render::Camera::lookingAt({e[0], e[1], e[2]}, {t[0], t[1], t[2]}, up);
                camera.lens.focal = focal;
                keepCamera(name, {e[0], e[1], e[2]}, {t[0], t[1], t[2]}, up, focal);
                if (const int code = shootView(camera, name, time); code != 0) {
                    return code;
                }
            }
            continue;
        }
        if (all.eye.size() != 3 || all.target.size() != 3) {
            err("neural-scene shoot wants a view: --eye and --target, or --views\n");
            return kExitFailure;
        }
        const render::Vec3 up = all.up.size() == 3 ? render::Vec3{all.up[0], all.up[1], all.up[2]} : upOf(scene);
        render::Camera camera = render::Camera::lookingAt({all.eye[0], all.eye[1], all.eye[2]},
                                                          {all.target[0], all.target[1], all.target[2]}, up);
        camera.lens.focal = focal;
        const std::string name = all.camera.empty() ? "camera" : all.camera;
        keepCamera(name, {all.eye[0], all.eye[1], all.eye[2]}, {all.target[0], all.target[1], all.target[2]}, up, focal);
        if (const int code = shootView(camera, name, time); code != 0) {
            return code;
        }
    }
    return 0;
}

int runRender(const SceneOptions& given) {
    SceneOptions o = given;
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    if (o.storedFrame != 0) {
        // A FRAME THE SCENE STORED, drawn again: through its camera, at its size, from the scene as it stood
        // when the frame's last look was in.
        if (o.storedFrame > scene->frames.size()) {
            err("the scene stores %zu frames\n", scene->frames.size());
            return kExitFailure;
        }
        const io::NeuralSceneFrame& frame = scene->frames[o.storedFrame - 1];
        o.camera = frame.camera;
        o.eye.clear();
        o.target.clear();
        o.size = std::to_string(frame.width) + "x" + std::to_string(frame.height);
        o.snapshot = frame.lastSnapshot;
    }
    uint32_t width = 0;
    uint32_t height = 0;
    if (!sizeOf(o.size, width, height)) return kExitFailure;
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
    // THE SCENE AS IT STOOD (--at a time, --snapshot one of the manifest's): the pieces in by then, less
    // the gaussians gone by then.
    const uint32_t asOf = o.snapshot != 0 ? o.snapshot : o.at >= 0.0 ? snapshotAt(*scene, o.at) : 0u;
    if ((o.snapshot != 0 || o.at >= 0.0) && asOf == 0) {
        err("the scene has no snapshot by time %g\n", o.at);
        return kExitFailure;
    }
    for (const io::NeuralScenePiece& piece : scene->pieces) {
        if (asOf != 0 && piece.foundIn > asOf) {
            continue;
        }
        // Nothing of it stands: every gaussian it held has gone.
        if (asOf == 0 && piece.removed != 0 && piece.removed >= piece.gaussians) {
            continue;
        }
        // One view of a found scene reads the pieces its frustum meets, and no others.
        if (o.views.empty() && piece.origin == "discovered" &&
            !boxInView(piece.boundsMin, piece.boundsMax, render::projectionFor(camera, width, height), width, height)) {
            continue;
        }
        auto cloud = loadPiece(library, *loader, *discovery, directory, piece, asOf);
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
    if (!o.views.empty()) {
        // A WALK: every view of a file through the scene loaded once, each written beside the others.
        std::ifstream in(o.views);
        if (!in) {
            err("cannot open %s\n", o.views.c_str());
            return kExitFailure;
        }
        const std::vector<render::SplatInstance>& drawnAs = o.harmonics ? byHarmonics : byLatents;
        render::RenderSettings waited = settings;
        waited.timeStages = true;
        std::string line;
        uint32_t frames = 0;
        double least = 1.0e30;
        double sum = 0.0;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream words(line);
            std::string name;
            double e[3];
            double t[3];
            if (!(words >> name >> e[0] >> e[1] >> e[2] >> t[0] >> t[1] >> t[2])) {
                continue;
            }
            render::Vec3 up = upOf(*scene);
            double u[3];
            if (words >> u[0] >> u[1] >> u[2]) {
                up = {u[0], u[1], u[2]};
            }
            render::Camera view = render::Camera::lookingAt({e[0], e[1], e[2]}, {t[0], t[1], t[2]}, up);
            view.lens.focal = o.focal > 0.0 ? o.focal : 35.0;
            render::RenderTargets frame;
            auto stats = raster->render(view, drawnAs, waited, frame);
            if (!stats) return report(stats.error());
            least = std::min(least, stats->totalMs);
            sum += stats->totalMs;
            ++frames;
            // Written only where a prefix is given: a walk that is only timed leaves nothing on the disk.
            if (!o.output.empty()) {
                // A prefix ending in .png writes pictures, <stem>_NAME.png.
                const std::filesystem::path given = o.output;
                const bool picture = given.extension() == ".png";
                const std::string to = (picture ? (given.parent_path() / given.stem()).string() : o.output) + "_" +
                                       name + (picture ? ".png" : ".exr");
                if (auto w = writeFrame(library, **device, frame, width, height, to, o.exposure); !w) {
                    return report(w.error());
                }
            }
        }
        uint64_t held = 0;
        for (const auto& cloud : clouds) {
            held += cloud->count;
        }
        out("%s: %u frames of %ux%u, %llu gaussians: %.2f ms a frame at least, %.2f ms on average\n",
            scene->name.c_str(), frames, width, height, static_cast<unsigned long long>(held), least,
            frames > 0 ? sum / frames : 0.0);
        return 0;
    }
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
    {
        // HOW MUCH OF THE VIEW THE SCENE HOLDS: the frame's cover, which is its alpha -- and the alpha of the
        // EXR written is the mask of it, for whoever renders or fills the rest.
        auto cover = render::imageStats(library, (drawLatents ? learnt : plain).colour, width, height);
        if (!cover) return report(cover.error());
        out("the scene holds %.1f%% of this view; the rest is what no piece covers\n", 100.0 * cover->mean[3]);
    }
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
            ATHENEA_TRY(writeFrame(library, **device, targets, width, height, path, o.exposure));
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
    render->add_option("--views", o.views, "a walk: a file of views, NAME EX EY EZ TX TY TZ [UX UY UZ] a line, each written as <-o>_NAME.exr from the scene loaded once");
    render->add_flag("--harmonics", o.harmonics, "every piece by its harmonics, whatever appearance it has");
    render->add_flag("--discovered", o.discovered, "only the gaussians the looks have found (neural-scene look): the scene as it is known");
    render->add_option("--up-to", o.upTo, "--discovered: only the looks up to this one, counted from one (0: all)");
    render->add_option("--at", o.at, "the scene as it stood at this time: the pieces found by then, less what had gone by then");
    render->add_option("--exposure", o.exposure, "-o a .png: stops over the scene's light");
    render->add_option("--frame", o.storedFrame, "a frame the scene stored (neural-scene shoot), counted from one: its camera, its size, the scene as it stood then");
    render->add_option("--snapshot", o.snapshot, "the same by a snapshot of the manifest's, counted from one");
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

    CLI::App* discover = cmd->add_subcommand("discover", "a piece made from a frame: a gaussian a pixel the scene did not already hold");
    discover->add_option("scene", o.directory, "the scene's directory")->required();
    discover->add_option("--views", o.views, "several looks in one run, the scene kept on the device: a file of views, NAME EX EY EZ TX TY TZ [UX UY UZ] a line, each with its frame NAME.exr and made the piece look_NAME");
    discover->add_option("--frames", o.framesDir, "--views: where the frames are (default: beside the file)");
    discover->add_option("--frame", o.frame, "any renderer's frame through that camera: an EXR with R, G, B and its depth as Z");
    discover->add_option("--id", o.id, "the new piece's name");
    discover->add_option("--camera", o.camera, "the view the frame was drawn through: one of the manifest's, or the name --eye and --target are kept under");
    discover->add_option("--eye", o.eye, "the frame's camera, where the manifest does not have it: x y z (with --target)")->expected(3);
    discover->add_option("--target", o.target, "x y z")->expected(3);
    discover->add_option("--up", o.up, "x y z")->expected(3);
    discover->add_option("--focal", o.focal, "mm, over a 24.576 mm aperture");
    discover->add_option("--name", o.name, "the scene's name");
    discover->add_option("--up-axis", o.upAxis, "Y or Z");
    discover->add_option("--meters-per-unit", o.metersPerUnit, "the scene's unit, in metres");
    discover->add_option("--source", o.source, "what the frames are renders of");
    discover->add_option("--licence", o.licence, "what the source asks of whoever uses it");
    discover->add_option("--time", o.time, "when: with the camera, the look's snapshot");
    discover->add_option("--tolerance", o.tolerance, "a pixel is already held where the scene's depth is within this share of the frame's");
    discover->add_option("--refine", o.refine, "a surface the scene holds with gaussians more than this many times as wide as this look's is taken out and made again, finer (0: held however coarse)");
    discover->add_option("--repaint", o.repaint, "a surface the scene holds whose colour is off the frame's by more than this share is taken out and made again (0: held by its depth alone)");
    discover->add_option("--spread", o.spread, "a gaussian's standard deviation, in pixels at its depth");
    discover->add_option("-o,--output", o.output, "the scene after the look through that camera, an EXR");
    run(discover, options, runDiscover);

    CLI::App* shoot = cmd->add_subcommand("shoot", "a USD stage path traced into the scene: the render's result is gaussians, found as it gathers");
    shoot->add_option("stage", o.stage, "the USD stage")->required();
    shoot->add_option("scene", o.directory, "the scene's directory (made, or added to)")->required();
    shoot->add_option("--views", o.views, "the views: a file, NAME EX EY EZ TX TY TZ [UX UY UZ] a line");
    shoot->add_option("--camera", o.camera, "the name --eye and --target are kept under");
    shoot->add_option("--eye", o.eye, "one view: x y z (with --target)")->expected(3);
    shoot->add_option("--target", o.target, "x y z")->expected(3);
    shoot->add_option("--up", o.up, "x y z")->expected(3);
    shoot->add_option("--focal", o.focal, "mm, over a 24.576 mm aperture");
    shoot->add_option("--size", o.size, "WIDTHxHEIGHT of the render");
    shoot->add_option("--time", o.time, "the stage's time code");
    shoot->add_option("--times", o.times, "several time codes, each shot through every view in turn");
    shoot->add_option("--samples", o.samples, "paths a pixel the render gathers");
    shoot->add_option("--pass-samples", o.passSamples, "paths a pixel each pass");
    shoot->add_option("--bounces", o.bounces, "bounces after the first hit");
    shoot->add_flag("--denoise", o.denoise, "denoise the whole frame before its last look");
    shoot->add_option("--ray-margin", o.rayMargin, "athenea:rayMargin");
    shoot->add_option("--coarsest", o.coarsest, "pixels a side of the first look's cells (a power of two; 1: one look, when the frame is whole)");
    shoot->add_option("--spread", o.spread, "a gaussian's standard deviation, in cells at its depth");
    shoot->add_option("--tolerance", o.tolerance, "a surface is already held where the scene's depth is within this share of the render's");
    shoot->add_option("--refine", o.refine, "a surface held with gaussians more than this many times as wide as this look's is made again, finer");
    shoot->add_option("--repaint", o.repaint, "a surface held whose colour is off the render's by more than this share is made again (0: held by its depth alone)");
    shoot->add_option("--turned", o.turned, "the least a disc's facing is taken as: it is stretched along the surface's fall by no more than 1 over this");
    shoot->add_option("--glass-opacity", o.glassOpacity, "the least a pane covers of what is behind it, however clear");
    shoot->add_option("--name", o.name, "the scene's name");
    shoot->add_option("--up-axis", o.upAxis, "Y or Z (default: the stage's)");
    shoot->add_option("--meters-per-unit", o.metersPerUnit, "the scene's unit, in metres");
    shoot->add_option("--source", o.source, "what the stage is");
    shoot->add_option("--licence", o.licence, "what the source asks of whoever uses it");
    run(shoot, options, runShoot);

    CLI::App* info = cmd->add_subcommand("info", "what the manifest holds");
    info->add_option("scene", o.directory, "the scene's directory")->required();
    run(info, options, runInfo);
}

}   // namespace athenea::cli
