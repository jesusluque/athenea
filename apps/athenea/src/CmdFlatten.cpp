// Copyright (c) 2026 jesus luque.
//
// `athenea flatten`: the TX clouds of a stage under a sky and lights, written
// as standard Gaussian Splatting files -- PLY, SPZ, glTF -- with no bake
// again (docs/decisions.md, task TXF; render/Flatten.h). The processor here
// composes the stage, frames files and keeps count; the shading, the fit and
// every number a file holds are the device's.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "Commands.h"
#include "Mesh2SplatValidate.h"
#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/io/RawSplats.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/render/Flatten.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/MeshStage.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// ln sigma no axis of a PLY or glTF goes below: Spark keeps ln sigma in
/// [-12, 9] (measured on athenea.lucab.co.uk, proposal 070 section 9), and
/// e^-12 of a metre is a few micrometres -- nothing a viewer could show.
constexpr float kLogScaleFloor = -12.0F;

/// The 62-float PLY records as the engine's own reader lays out a 3DGS PLY:
/// what a USD written from them is decoded with.
io::SplatEncoding plyEncoding() {
    io::SplatEncoding e;
    e.floatsPerRecord = 62;
    e.x = 0;
    e.y = 1;
    e.z = 2;
    e.dc0 = 6;
    e.dc1 = 7;
    e.dc2 = 8;
    e.restBase = 9;
    e.restPerColour = 15;
    e.restColourOuter = 1;
    e.opacity = 54;
    e.scale0 = 55;
    e.scale1 = 56;
    e.scale2 = 57;
    e.rotW = 58;
    e.rotX = 59;
    e.rotY = 60;
    e.rotZ = 61;
    e.opacity_ = io::SplatEncoding::Opacity::Logit;
    e.scale_ = io::SplatEncoding::Scale::Log;
    e.colour = io::SplatEncoding::Colour::ShDc;
    e.rotation = io::SplatEncoding::Rotation::Float;
    e.rest = io::SplatEncoding::Rest::Float;
    return e;
}

std::vector<std::string> splitList(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string part;
    while (std::getline(in, part, ',')) {
        if (!part.empty()) {
            out.push_back(part);
        }
    }
    return out;
}

uint64_t fileBytes(const fs::path& path) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    return error ? 0 : static_cast<uint64_t>(size);
}

}   // namespace

void addFlatten(CLI::App& app) {
    struct Options {
        std::string stage, output = "flat", formats, sky, referred = "display", prim;
        std::vector<std::string> hidden;
        uint32_t directions = 256, spzVersion = 3, sh1Bits = 8, shRestBits = 8;
        double exposure = 0.0, time = 0.0;
        float lambda = 1.0e-4F, backWeight = 1.0e-2F, floorShare = 0.05F;
        bool noFloor = false;
        std::string validate, validateStage, validateCamera, validateFormat = "spz";
        std::vector<uint32_t> validateSize;
        uint32_t validatePaths = 512, validateBounces = 6;
        std::vector<std::string> validateMaterials, paths;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "flatten", "a stage's TX clouds under a sky, as standard Gaussian Splatting files (PLY, SPZ, glTF, USD) "
                   "with no bake again");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc holding TX clouds (mesh2splat --transfer) and lights")
        ->required();
    cmd->add_option("-o,--output", o->output,
                    "the files' path: without an extension, every format of --format beside each other");
    cmd->add_option("--format", o->formats, "comma list of ply, spz, glb, usdc, usda (default ply,spz,glb)");
    cmd->add_option("--sky", o->sky,
                    "another sky: an image file, or 'white'; the stage's domes take it and its other lights go "
                    "off (a stage with no dome gets one)");
    cmd->add_option("--hide", o->hidden, "a prim left out, as if inactive (repeatable)");
    cmd->add_option("--directions", o->directions, "directions each gaussian is looked at from");
    cmd->add_option("--exposure", o->exposure, "stops on the light before it is encoded");
    cmd->add_option("--referred", o->referred, "display (fit what a viewer shows, sRGB) or scene (linear light)")
        ->check(CLI::IsMember({"display", "scene"}));
    cmd->add_flag("--no-roughness-floor", o->noFloor,
                  "evaluate reflections as sharp as the material is, not as sharp as degree 3 holds");
    cmd->add_option("--roughness-floor-share", o->floorShare,
                    "the share of a reflection's energy the floor allows above band 3");
    cmd->add_option("--fit-lambda", o->lambda, "the fit's regularisation, times l^2 (l+1)^2");
    cmd->add_option("--back-weight", o->backWeight, "what a surface's back hemisphere weighs in the fit");
    cmd->add_option("--spz-version", o->spzVersion, "3 (gzip; Spark reads it) or 4 (ZSTD)")
        ->check(CLI::IsMember({3u, 4u}));
    cmd->add_option("--spz-sh1-bits", o->sh1Bits, "bits SPZ keeps of band 1")->check(CLI::Range(1u, 8u));
    cmd->add_option("--spz-sh-rest-bits", o->shRestBits, "bits SPZ keeps of bands 2 and 3")->check(CLI::Range(1u, 8u));
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_option("--validate", o->validate,
                    "measure a written file against the meshes path traced, material by material, into this "
                    "directory");
    cmd->add_option("--validate-stage", o->validateStage, "--validate: the stage of meshes (default: the input)");
    cmd->add_option("--validate-format", o->validateFormat, "--validate: which file is measured")
        ->check(CLI::IsMember({"ply", "spz"}));
    cmd->add_option("--validate-camera", o->validateCamera, "--validate: the camera (default: the stage's first)");
    cmd->add_option("--validate-size", o->validateSize, "--validate: W H of the frames")->expected(2);
    cmd->add_option("--validate-paths", o->validatePaths, "--validate: paths a pixel the GT holds");
    cmd->add_option("--validate-bounces", o->validateBounces, "--validate: bounces of the GT's paths");
    cmd->add_option("--validate-material", o->validateMaterials, "--validate: only this material (repeatable)");
    cmd->add_option("--path", o->paths, "extra AOFX bundle directories");

    cmd->callback([o] {
        const Clock::time_point began = Clock::now();
        // WHICH FILES: the extension of -o where it has a known one, else
        // --format, else the default set.
        fs::path out(o->output);
        std::vector<std::string> formats = splitList(o->formats);
        const std::set<std::string> known{"ply", "spz", "glb", "usdc", "usda"};
        if (formats.empty()) {
            const std::string ext = out.extension().string().empty() ? "" : out.extension().string().substr(1);
            if (known.count(ext) != 0) {
                formats = {ext};
                out.replace_extension();
            } else {
                formats = {"ply", "spz", "glb"};
            }
        } else if (known.count(out.extension().string().empty() ? "" : out.extension().string().substr(1)) != 0) {
            out.replace_extension();
        }
        for (const std::string& f : formats) {
            if (known.count(f) == 0) {
                cli::err("flatten: '%s' is not a format (ply, spz, glb, usdc, usda)\n", f.c_str());
                throw CLI::RuntimeError(1);
            }
        }
        const bool display = o->referred == "display";
        if (!display && (std::count(formats.begin(), formats.end(), "spz") != 0 ||
                         std::count(formats.begin(), formats.end(), "glb") != 0)) {
            cli::err("flatten: SPZ and glTF hold colours a display shows; --referred scene writes "
                                 "ply or usd\n");
            throw CLI::RuntimeError(1);
        }
        if (o->spzVersion == 4 && !io::writesSpzVersion4()) {
            cli::err("flatten: this build has no ZSTD, so SPZ version 4 cannot be written\n");
            throw CLI::RuntimeError(1);
        }
        const fs::path source = fs::absolute(o->stage);
        auto meshStage = usd::MeshStage::open(source);
        if (!meshStage) {
            cli::fail(meshStage.error());
        }
        // The frame the content was authored in, through a wrapper layer that
        // says nothing of it (MeshStage::authoredFrame).
        const auto [upAxis, metersPerUnit] = meshStage->authoredFrame();

        // THE STAGE FLATTENED: its meshes off (a TX cloud needs none of them
        // to be shaded, and a standard file holds gaussians alone), --hide's
        // prims off, and --sky on its domes with every other light off.
        auto groups = usd::stageMaterialGroups(source, "", {}, o->time);
        if (!groups) {
            cli::fail(groups.error());
        }
        std::vector<std::string> off = o->hidden;
        for (const usd::MaterialGroup& g : *groups) {
            off.insert(off.end(), g.meshes.begin(), g.meshes.end());
        }
        std::map<std::string, std::string> bodies;
        std::string tail;
        if (!o->sky.empty()) {
            auto lights = usd::stageLights(source);
            if (!lights) {
                cli::fail(lights.error());
            }
            const bool white = o->sky == "white";
            const std::string file = white ? std::string() : fs::absolute(o->sky).string();
            const std::string body = white ? "    asset inputs:texture:file = @@\n    color3f inputs:color = (1, 1, 1)\n"
                                             "    float inputs:intensity = 1\n    float inputs:exposure = 0\n"
                                           : "    asset inputs:texture:file = @" + file + "@\n";
            for (const auto& [path, dome] : *lights) {
                if (dome) {
                    bodies[path] = body;
                } else {
                    off.push_back(path);
                }
            }
            if (bodies.empty()) {
                // A dome of its own, its image's top turned to the stage's up.
                tail = "def DomeLight \"AtheneaFlattenSky\"\n{\n" + body +
                       (upAxis == 'z' ? "    float3 xformOp:rotateXYZ = (90, 0, 0)\n"
                                        "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n"
                                      : "") +
                       "}\n";
            }
        }
        std::sort(off.begin(), off.end());
        off.erase(std::unique(off.begin(), off.end()), off.end());
        std::error_code made;
        if (out.has_parent_path()) {
            fs::create_directories(out.parent_path(), made);
        }
        const fs::path composed = fs::path(out.string() + "_flatten_stage.usda");
        {
            std::ofstream layer(composed);
            layer << overLayer(source.string(), off, bodies, upAxis, metersPerUnit, tail);
            if (!layer) {
                cli::err("flatten: cannot write %s\n", composed.string().c_str());
                throw CLI::RuntimeError(1);
            }
        }

        // THE FILES' FRAME: Y up, metres -- what three.js, Spark, SuperSplat
        // and glTF take as they are (proposal 070), whatever the stage's.
        render::FlattenSettings settings;
        settings.directions = o->directions;
        settings.lambda = o->lambda;
        settings.backWeight = o->backWeight;
        settings.display = display;
        settings.exposure = static_cast<float>(std::exp2(o->exposure));
        settings.roughnessFloor = !o->noFloor;
        settings.floorShare = o->floorShare;
        settings.unitScale = static_cast<float>(metersPerUnit);
        if (upAxis == 'z') {
            settings.toFile = {1, 0, 0, 0, 0, 1, 0, -1, 0};   // (x, y, z) -> (x, z, -y)
        }

        gpu_host::Context* context = gpu_host::installProcessContext();
        if (context == nullptr || context->compute() == nullptr) {
            cli::err("no GPU compute device (gpe has no backend here)\n");
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(context->deviceShared());

        std::vector<usd::StageRenderer::FlattenedCloud> clouds;
        render::FlatStats stats;
        float floor = 0.0F;
        double flattenMs = 0.0, packMs = 0.0;
        uint64_t total = 0;
        std::vector<std::pair<std::string, uint64_t>> written;
        Result<void> inside = ok();
        auto ran = context->run([&] {
            inside = [&]() -> Result<void> {
                auto fit = render::FlattenFit::create(library);
                if (!fit) return std::move(fit).error();
                ATHENEA_TRY(fit->prepare(settings));
                if (settings.roughnessFloor) {
                    auto f = fit->floorRoughness();
                    if (!f) return std::move(f).error();
                    floor = *f;
                }
                auto renderer = usd::StageRenderer::open(composed, context->deviceShared());
                if (!renderer) return std::move(renderer).error();
                render::SplatFlatten request;
                request.settings = settings;
                request.basis = &fit->basis();
                request.floor = &fit->floor();
                const Clock::time_point flattenStart = Clock::now();
                auto flattened = (*renderer)->flatten(request, o->time);
                if (!flattened) return std::move(flattened).error();
                flattenMs = msSince(flattenStart);
                clouds = std::move(*flattened);
                if (clouds.empty()) {
                    return Error(ErrorCode::NotFound,
                                 "flatten: the stage has no relit TX cloud under a sky (mesh2splat --transfer)");
                }
                for (const auto& c : clouds) {
                    total += c.cloud.count;
                }
                if (total > 0xFFFFFFFFull) {
                    return Error(ErrorCode::Unsupported, "flatten: more than 2^32 gaussians");
                }

                const Clock::time_point packStart = Clock::now();
                auto pack = render::FlattenPack::create(library);
                if (!pack) return std::move(pack).error();
                std::vector<render::FlatCloud> flat;
                for (const auto& c : clouds) {
                    flat.push_back(c.cloud);
                }
                auto counted = pack->stats(flat, kLogScaleFloor);
                if (!counted) return std::move(counted).error();
                stats = *counted;
                const auto has = [&](const char* f) {
                    return std::find(formats.begin(), formats.end(), f) != formats.end();
                };
                const bool usd = has("usdc") || has("usda");
                std::vector<std::vector<float>> ply;
                if (has("ply") || usd) {
                    for (const render::FlatCloud& c : flat) {
                        auto records = pack->ply(c, kLogScaleFloor);
                        if (!records) return std::move(records).error();
                        ply.push_back(std::move(*records));
                    }
                }
                std::vector<std::string> comments{
                    "athenea flatten: degree 3, " + std::string(display ? "sRGB display-referred" : "linear scene-referred") +
                        ", y up, metres"};
                if (has("ply")) {
                    const fs::path path = out.string() + ".ply";
                    ATHENEA_TRY(io::writePly3dgs(path, ply, comments));
                    written.emplace_back(path.string(), fileBytes(path));
                }
                if (usd) {
                    io::RawSplats raw;
                    raw.source = o->stage + " (flattened)";
                    raw.count = static_cast<uint32_t>(total);
                    raw.encoding = plyEncoding();
                    raw.linear = !display;
                    raw.records.reserve(total * 62);
                    for (const std::vector<float>& chunk : ply) {
                        raw.records.insert(raw.records.end(), chunk.begin(), chunk.end());
                    }
                    usd::ExportOptions options;
                    options.upAxis = 'y';
                    options.metersPerUnit = 1.0;
                    options.linear = !display;
                    for (const char* ext : {"usdc", "usda"}) {
                        if (has(ext)) {
                            const fs::path path = out.string() + "." + ext;
                            ATHENEA_TRY(usd::writeParticleFieldStage(library, raw, path, options));
                            written.emplace_back(path.string(), fileBytes(path));
                        }
                    }
                }
                ply.clear();
                if (has("spz")) {
                    auto streams = pack->spz(flat, stats.fractionalBits, o->sh1Bits, o->shRestBits);
                    if (!streams) return std::move(streams).error();
                    io::SpzHeader header;
                    header.version = o->spzVersion;
                    header.count = static_cast<uint32_t>(total);
                    header.shDegree = 3;
                    header.fractionalBits = stats.fractionalBits;
                    header.antialiased = true;
                    const fs::path path = out.string() + ".spz";
                    ATHENEA_TRY(io::writeSpz(path, header, streams->streams));
                    written.emplace_back(path.string(), fileBytes(path));
                }
                if (has("glb")) {
                    io::GlbGaussians glb;
                    for (const render::FlatCloud& c : flat) {
                        auto blocks = pack->glb(c, kLogScaleFloor);
                        if (!blocks) return std::move(blocks).error();
                        glb.counts.push_back(c.count);
                        glb.chunks.push_back(std::move(*blocks));
                    }
                    glb.min = stats.min;
                    glb.max = stats.max;
                    const fs::path path = out.string() + ".glb";
                    ATHENEA_TRY(io::writeGlbGaussians(path, glb));
                    written.emplace_back(path.string(), fileBytes(path));
                }
                packMs = msSince(packStart);
                return ok();
            }();
        });
        if (!ran) {
            cli::fail(ran.error());
        }
        if (!inside) {
            cli::fail(inside.error());
        }

        for (const auto& c : clouds) {
            cli::out("flatten: %-56s %9u gaussians%s\n", c.prim.c_str(), c.cloud.count,
                        c.cloud.catcher ? " (shadow catcher)" : "");
        }
        cli::out("flatten: %llu gaussians, %u directions each, %s, roughness floor %s\n",
                    static_cast<unsigned long long>(total), settings.directions,
                    display ? "fitted to the sRGB a viewer shows" : "fitted in linear light",
                    settings.roughnessFloor ? std::to_string(floor).c_str() : "off");
        cli::out("flatten: frame y up, metres (stage %c up, %g m a unit); box (%.4g %.4g %.4g) to (%.4g %.4g %.4g)\n",
                    upAxis, metersPerUnit, static_cast<double>(stats.min[0]), static_cast<double>(stats.min[1]),
                    static_cast<double>(stats.min[2]), static_cast<double>(stats.max[0]),
                    static_cast<double>(stats.max[1]), static_cast<double>(stats.max[2]));
        const double perGaussian = total > 0 ? 1.0 / static_cast<double>(total) : 0.0;
        cli::out("flatten: %.2f%% of gaussians had their higher bands shrunk to stay positive; %.2f%% of axes "
                    "under e^%g (PLY, glTF floor), %.2f%% under e^-10 (SPZ); %.2f%% of gaussians scaled into SPZ's "
                    "[-1, 1]; SPZ fixed point %u bits\n",
                    100.0 * stats.shrunk * perGaussian, 100.0 * stats.sizesFloored * perGaussian / 3.0,
                    static_cast<double>(kLogScaleFloor), 100.0 * stats.spzSizesFloored * perGaussian / 3.0,
                    100.0 * stats.spzSaturated * perGaussian, stats.fractionalBits);
        for (const auto& [path, bytes] : written) {
            cli::out("flatten: wrote %s, %.1f MB (%.1f bytes a gaussian)\n", path.c_str(),
                        static_cast<double>(bytes) / 1.0e6, static_cast<double>(bytes) * perGaussian);
        }
        cli::out("flatten: shading and fit %.0f ms, packing and writing %.0f ms, %.0f ms in all\n", flattenMs,
                    packMs, msSince(began));

        if (o->validate.empty()) {
            return;
        }
        // --validate: the written file read back as any reader reads it, put
        // where the stage's meshes stand, rasterised blending as a standard
        // viewer blends, and measured against the meshes path traced under
        // the same sky (Mesh2SplatValidate.h).
        const fs::path measured = out.string() + "." + o->validateFormat;
        if (std::none_of(written.begin(), written.end(),
                         [&](const auto& w) { return w.first == measured.string(); })) {
            cli::err("flatten: --validate measures %s, which --format did not write\n",
                         measured.string().c_str());
            throw CLI::RuntimeError(1);
        }
        if (metersPerUnit != 1.0) {
            cli::err("flatten: --validate places the file in a stage of metres only (this one is "
                                 "%g m a unit)\n",
                         metersPerUnit);
            throw CLI::RuntimeError(1);
        }
        aofx_host::EffectRegistry registry;
        for (const std::string& path : o->paths) {
            registry.addSearchPath(path);
        }
#ifdef ATHENEA_AOFX_BUNDLE_DIR
        registry.addSearchPath(ATHENEA_AOFX_BUNDLE_DIR);
#endif
        registry.scan(context);
        aofx::Effect* measure = registry.find("rt.sparrow.aofx.measure");
        if (measure == nullptr) {
            cli::err("no Measure bundle on the AOFX search path (try `athenea aofx list`), and "
                                 "--validate measures with it\n");
            throw CLI::RuntimeError(1);
        }
        ValidateJob job;
        job.stage = o->validateStage.empty() ? o->stage : o->validateStage;
        job.directory = o->validate;
        job.camera = o->validateCamera;
        if (o->validateSize.size() == 2) {
            job.width = o->validateSize[0];
            job.height = o->validateSize[1];
        }
        job.gtPaths = o->validatePaths;
        job.gtBounces = o->validateBounces;
        job.time = o->time;
        job.hidden = o->hidden;
        job.materials = o->validateMaterials;
        job.sky = o->sky;
        job.displayBlend = display;
        job.wholeCloud = true;
        // The file's frame back to the stage's: a turn about x, by the
        // stage's up axis, and by SPZ's own axes, which a reader turns
        // (right-up-back read as right-down-front).
        double turn = upAxis == 'z' ? 90.0 : 0.0;
        if (o->validateFormat == "spz") {
            turn += 180.0;
        }
        auto validated = validateConversion(
            job, *context, library, *measure,
            [&](const std::string&, const std::vector<std::string>&, const std::string& output) -> Result<uint32_t> {
                Result<uint32_t> placed = 0u;
                auto on = context->run([&] {
                    placed = [&]() -> Result<uint32_t> {
                        auto loader = scene::CloudLoader::create(library);
                        if (!loader) return std::move(loader).error();
                        auto raw = scene::readSplatRecords(*loader, measured, 3);
                        if (!raw) return std::move(raw).error();
                        usd::ExportOptions options;
                        options.addCamera = false;
                        options.rotateXDegrees = turn;
                        options.upAxis = upAxis;
                        options.metersPerUnit = 1.0;
                        ATHENEA_TRY(usd::writeParticleFieldStage(library, *raw, output, options));
                        return raw->count;
                    }();
                });
                if (!on) return std::move(on).error();
                return placed;
            });
        if (!validated) {
            cli::fail(validated.error());
        }
    });
}

}   // namespace athenea::cli
