// Copyright (c) 2026 jesus luque.
//
// `athenea flatten`: the TX clouds of a stage under a sky and lights, written
// as standard Gaussian Splatting files -- PLY, SPZ, glTF -- with no bake
// again (docs/decisions.md, task TXF; render/Flatten.h). The processor here
// composes the stage, frames files and keeps count; the shading, the fit and
// every number a file holds are the device's.
//
// A FROZEN VERSION IS TRACEABLE: every file says how it was made -- the
// build's commit, the source, the sky and its turn, the exposure, the fit's
// settings, the unit -- in the place its format has for it (PLY comments,
// glTF's asset extras, USD customData), and `<out>.flatten.json` beside them
// says it for every format, SPZ included, with what --validate measured.
// Nothing in any of it depends on when or how fast it ran, so the same
// inputs on the same build write the same bytes.
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

#include <nlohmann/json.hpp>

#include "Commands.h"
#include "Mesh2SplatValidate.h"
#include "aofx/Effect.h"
#include "athenea/Commit.h"
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
using Json = nlohmann::ordered_json;

double msSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

/// ln sigma no axis of a PLY or glTF goes below: Spark keeps ln sigma in
/// [-12, 9] (measured on athenea.lucab.co.uk, proposal 070 section 9).
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

/// The formats, and the file each is written to: `spz` at --spz-version as
/// <out>.spz, `spz3` and `spz4` as <out>.v3.spz and <out>.v4.spz, so one run
/// writes both the SPZ Spark reads and the one three.js reads.
const std::set<std::string> kFormats{"ply", "spz", "spz3", "spz4", "glb", "usdc", "usda"};

fs::path fileOf(const fs::path& out, const std::string& format) {
    if (format == "spz3") return out.string() + ".v3.spz";
    if (format == "spz4") return out.string() + ".v4.spz";
    return out.string() + "." + format;
}

/// The units a file may be written in, and how many of each a metre is.
struct Unit {
    const char* name;
    float       perMetre;
};
constexpr Unit kUnits[] = {{"m", 1.0F}, {"cm", 100.0F}, {"mm", 1000.0F}};

std::string textOf(const Json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

}   // namespace

void addFlatten(CLI::App& app) {
    struct Options {
        std::string stage, output = "flat", formats, sky, hdri, referred = "display", units = "auto";
        std::vector<std::string> hidden;
        uint32_t directions = 256, spzVersion = 3, sh1Bits = 8, shRestBits = 8;
        double exposure = 0.0, time = 0.0, rotate = 0.0;
        float lambda = 1.0e-4F, backWeight = 1.0e-2F, floorShare = 0.05F;
        bool noFloor = false;
        std::string validate, validateStage, validateCamera, validateFormats = "spz";
        std::vector<uint32_t> validateSize;
        uint32_t validatePaths = 512, validateBounces = 6;
        std::vector<std::string> validateMaterials, validateSkip, paths;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "flatten", "a stage's TX clouds under a sky, as standard Gaussian Splatting files (PLY, SPZ, glTF, USD) "
                   "with no bake again");
    cmd->add_option("stage", o->stage,
                    ".usd / .usda / .usdc holding TX clouds (mesh2splat --transfer) and lights, or a TX cloud's own "
                    "file")
        ->required();
    cmd->add_option("-o,--output", o->output,
                    "the files' path: without an extension, every format of --format beside each other");
    cmd->add_option("--format", o->formats,
                    "comma list of ply, spz, spz3, spz4, glb, usdc, usda (default ply,spz,glb)");
    cmd->add_option("--hdri", o->hdri,
                    "an image (.hdr, .exr) the stage's domes take; its other lights stay on (a stage with no dome "
                    "gets one)");
    cmd->add_option("--sky", o->sky,
                    "another sky: an image file, or 'white'; the stage's domes take it and its other lights go "
                    "off (a stage with no dome gets one)");
    cmd->add_option("--hdri-rotate", o->rotate, "degrees the domes turn about the stage's up axis");
    cmd->add_option("--hide", o->hidden, "a prim left out, as if inactive (repeatable)");
    cmd->add_option("--directions", o->directions, "directions each gaussian is looked at from");
    cmd->add_option("--exposure", o->exposure, "stops on the light before it is encoded");
    cmd->add_option("--referred", o->referred, "display (fit what a viewer shows, sRGB) or scene (linear light)")
        ->check(CLI::IsMember({"display", "scene"}));
    cmd->add_option("--units", o->units, "the files' unit: m, cm, mm, or auto")
        ->check(CLI::IsMember({"auto", "m", "cm", "mm"}));
    cmd->add_flag("--no-roughness-floor", o->noFloor,
                  "evaluate reflections as sharp as the material is, not as sharp as degree 3 holds");
    cmd->add_option("--roughness-floor-share", o->floorShare,
                    "the share of a reflection's energy the floor allows above band 3");
    cmd->add_option("--fit-lambda", o->lambda, "the fit's regularisation, times l^2 (l+1)^2");
    cmd->add_option("--back-weight", o->backWeight, "what a surface's back hemisphere weighs in the fit");
    cmd->add_option("--spz-version", o->spzVersion, "the version --format spz writes: 3 (gzip) or 4 (ZSTD)")
        ->check(CLI::IsMember({3u, 4u}));
    cmd->add_option("--spz-sh1-bits", o->sh1Bits, "bits SPZ keeps of band 1")->check(CLI::Range(1u, 8u));
    cmd->add_option("--spz-sh-rest-bits", o->shRestBits, "bits SPZ keeps of bands 2 and 3")->check(CLI::Range(1u, 8u));
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_option("--validate", o->validate,
                    "measure written files against the meshes path traced, material by material, into this "
                    "directory");
    cmd->add_option("--validate-stage", o->validateStage, "--validate: the stage of meshes (default: the input)");
    cmd->add_option("--validate-format", o->validateFormats,
                    "--validate: comma list of the written formats measured (ply, spz, spz3, spz4, glb)");
    cmd->add_option("--validate-camera", o->validateCamera, "--validate: the camera (default: the stage's first)");
    cmd->add_option("--validate-size", o->validateSize, "--validate: W H of the frames")->expected(2);
    cmd->add_option("--validate-paths", o->validatePaths, "--validate: paths a pixel the GT holds");
    cmd->add_option("--validate-bounces", o->validateBounces, "--validate: bounces of the GT's paths");
    cmd->add_option("--validate-material", o->validateMaterials, "--validate: only this material (repeatable)");
    cmd->add_option("--validate-skip-material", o->validateSkip,
                    "--validate: not this material, left a mesh in every frame (repeatable)");
    cmd->add_option("--path", o->paths, "extra AOFX bundle directories");

    cmd->callback([o] {
        const Clock::time_point began = Clock::now();
        // WHICH FILES: the extension of -o where it has a known one, else
        // --format, else the default set.
        fs::path out(o->output);
        const std::string outExt = out.extension().string().empty() ? "" : out.extension().string().substr(1);
        const bool outHasFormat = outExt != "spz3" && outExt != "spz4" && kFormats.count(outExt) != 0;
        std::vector<std::string> formats = splitList(o->formats);
        if (formats.empty()) {
            formats = outHasFormat ? std::vector<std::string>{outExt} : std::vector<std::string>{"ply", "spz", "glb"};
        }
        if (outHasFormat) {
            out.replace_extension();
        }
        for (const std::string& f : formats) {
            if (kFormats.count(f) == 0) {
                std::fprintf(stderr, "flatten: '%s' is not a format (ply, spz, spz3, spz4, glb, usdc, usda)\n",
                             f.c_str());
                throw CLI::RuntimeError(1);
            }
        }
        std::sort(formats.begin(), formats.end());
        formats.erase(std::unique(formats.begin(), formats.end()), formats.end());
        const auto has = [&](const std::string& f) {
            return std::find(formats.begin(), formats.end(), f) != formats.end();
        };
        const bool anySpz = has("spz") || has("spz3") || has("spz4");
        const bool display = o->referred == "display";
        if (!display && (anySpz || has("glb"))) {
            std::fprintf(stderr, "flatten: SPZ and glTF hold colours a display shows; --referred scene writes "
                                 "ply or usd\n");
            throw CLI::RuntimeError(1);
        }
        if (((has("spz") && o->spzVersion == 4) || has("spz4")) && !io::writesSpzVersion4()) {
            std::fprintf(stderr, "flatten: this build has no ZSTD, so SPZ version 4 cannot be written\n");
            throw CLI::RuntimeError(1);
        }
        if (!o->hdri.empty() && !o->sky.empty()) {
            std::fprintf(stderr, "flatten: --hdri or --sky, not both (--hdri keeps the stage's lights on, --sky "
                                 "turns them off)\n");
            throw CLI::RuntimeError(1);
        }
        const std::string sky = o->hdri.empty() ? o->sky : fs::absolute(o->hdri).string();
        const bool keepLights = o->sky.empty();
        if (!sky.empty() && sky != "white" && !fs::exists(sky)) {
            std::fprintf(stderr, "flatten: no image at %s\n", sky.c_str());
            throw CLI::RuntimeError(1);
        }
        const std::vector<std::string> validateFormats = splitList(o->validateFormats);
        if (!o->validate.empty()) {
            for (const std::string& f : validateFormats) {
                if (f != "ply" && f != "spz" && f != "spz3" && f != "spz4" && f != "glb") {
                    std::fprintf(stderr, "flatten: --validate measures ply, spz, spz3, spz4 or glb, not '%s'\n",
                                 f.c_str());
                    throw CLI::RuntimeError(1);
                }
                if (!has(f)) {
                    std::fprintf(stderr, "flatten: --validate measures %s, which --format does not write\n",
                                 f.c_str());
                    throw CLI::RuntimeError(1);
                }
            }
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
        // prims off, and the sky on its domes (skyLayer).
        auto groups = usd::stageMaterialGroups(source, "", {}, o->time);
        if (!groups) {
            cli::fail(groups.error());
        }
        std::vector<std::string> off = o->hidden;
        for (const usd::MaterialGroup& g : *groups) {
            off.insert(off.end(), g.meshes.begin(), g.meshes.end());
        }
        auto skyOver = skyLayer(source, sky, o->rotate, keepLights, upAxis, o->time, true, off);
        if (!skyOver) {
            cli::fail(skyOver.error());
        }
        std::sort(off.begin(), off.end());
        off.erase(std::unique(off.begin(), off.end()), off.end());
        std::error_code made;
        if (out.has_parent_path()) {
            fs::create_directories(out.parent_path(), made);
        }
        const fs::path composed = fs::absolute(fs::path(out.string() + "_flatten_stage.usda"));
        {
            std::ofstream layer(composed);
            layer << overLayer(source.string(), off, skyOver->bodies, upAxis, metersPerUnit, skyOver->tail);
            if (!layer) {
                std::fprintf(stderr, "flatten: cannot write %s\n", composed.string().c_str());
                throw CLI::RuntimeError(1);
            }
        }

        // THE FILES' FRAME: Y up -- what three.js, Spark, SuperSplat and glTF
        // take as they are (proposal 070), whatever the stage's -- in metres,
        // then in the unit chosen below.
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
            std::fprintf(stderr, "no GPU compute device (gpe has no backend here)\n");
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(context->deviceShared());

        std::vector<usd::StageRenderer::FlattenedCloud> clouds;
        render::FlatStats stats;
        float floor = 0.0F;
        Unit unit = kUnits[0];
        std::map<std::string, render::FlatStats> perUnit;
        double flattenMs = 0.0, packMs = 0.0;
        uint64_t total = 0;
        std::vector<std::pair<std::string, fs::path>> written;   // format, file
        Json madeBy;
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
                // THE UNIT. Spark keeps ln sigma in [-12, 9] and SPZ in
                // [-10, 5.94]: a small cloud in metres has axes under the
                // floor, a large one in millimetres axes over the ceiling.
                // `auto` takes the largest unit with no more than 1% of axes
                // under the floor (SPZ's too where SPZ is written) and no more
                // axes over a ceiling than metres have; failing that, the unit
                // that loses the fewest. Counted on the device for each.
                for (const Unit& u : kUnits) {
                    auto counted = pack->stats(flat, kLogScaleFloor, u.perMetre);
                    if (!counted) return std::move(counted).error();
                    perUnit[u.name] = *counted;
                }
                const auto under = [&](const render::FlatStats& s) {
                    return uint64_t{s.sizesFloored} + (anySpz ? s.spzSizesFloored : 0u);
                };
                const auto over = [&](const render::FlatStats& s) {
                    return uint64_t{s.sizesCeiled} + (anySpz ? s.spzSizesCeiled : 0u);
                };
                if (o->units == "auto") {
                    const uint64_t axes = total * 3;
                    bool chosen = false;
                    for (const Unit& u : kUnits) {
                        const render::FlatStats& s = perUnit[u.name];
                        if (under(s) * 100 <= axes && over(s) <= over(perUnit["m"])) {
                            unit = u;
                            chosen = true;
                            break;
                        }
                    }
                    if (!chosen) {
                        uint64_t fewest = ~uint64_t{0};
                        for (const Unit& u : kUnits) {
                            const render::FlatStats& s = perUnit[u.name];
                            if (under(s) + over(s) < fewest) {
                                fewest = under(s) + over(s);
                                unit = u;
                            }
                        }
                    }
                } else {
                    for (const Unit& u : kUnits) {
                        if (o->units == u.name) unit = u;
                    }
                }
                stats = perUnit[unit.name];
                const double metresPerFileUnit = 1.0 / static_cast<double>(unit.perMetre);

                // HOW IT WAS MADE, the same in every file and the sidecar.
                madeBy = Json::object();
                madeBy["tool"] = "athenea flatten";
                madeBy["version"] = ATHENEA_VERSION;
                madeBy["commit"] = ATHENEA_COMMIT;
                madeBy["source"] = source.string();
                madeBy["hdri"] = sky;
                madeBy["skyMode"] = sky.empty()  ? "the stage's own"
                                    : keepLights ? "hdri: the stage's domes take it, its other lights on"
                                                 : "sky: the stage's domes take it, its other lights off";
                madeBy["hdriRotateDegrees"] = o->rotate;
                madeBy["exposureStops"] = o->exposure;
                madeBy["time"] = o->time;
                madeBy["hidden"] = o->hidden;
                madeBy["referred"] = display ? "display" : "scene";
                madeBy["colour"] = display ? "sRGB display-referred; colour = 0.5 + 0.2820948 f_dc"
                                           : "linear scene-referred; colour = 0.5 + 0.2820948 f_dc";
                madeBy["shDegree"] = 3;
                madeBy["frame"] = "y up, right-handed: x right, y up, z towards the viewer";
                madeBy["units"] = unit.name;
                madeBy["metersPerUnit"] = metresPerFileUnit;
                madeBy["unitsChosenBy"] = o->units == "auto" ? "auto" : "--units";
                madeBy["directions"] = settings.directions;
                madeBy["fitLambda"] = settings.lambda;
                madeBy["backWeight"] = settings.backWeight;
                madeBy["roughnessFloor"] = settings.roughnessFloor ? floor : 0.0F;
                madeBy["roughnessFloorShare"] = settings.floorShare;
                madeBy["spzSh1Bits"] = o->sh1Bits;
                madeBy["spzShRestBits"] = o->shRestBits;
                madeBy["gaussians"] = total;

                std::vector<std::vector<float>> ply;
                const bool usd = has("usdc") || has("usda");
                if (has("ply") || usd) {
                    for (const render::FlatCloud& c : flat) {
                        auto records = pack->ply(c, kLogScaleFloor, unit.perMetre);
                        if (!records) return std::move(records).error();
                        ply.push_back(std::move(*records));
                    }
                }
                if (has("ply")) {
                    // One comment a setting, "athenea key=value", in the order above.
                    std::vector<std::string> comments;
                    for (const auto& [key, value] : madeBy.items()) {
                        comments.push_back("athenea " + key + "=" + textOf(value));
                    }
                    const fs::path path = fileOf(out, "ply");
                    ATHENEA_TRY(io::writePly3dgs(path, ply, comments));
                    written.emplace_back("ply", path);
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
                    options.metersPerUnit = metresPerFileUnit;
                    options.linear = !display;
                    options.addCamera = false;
                    for (const auto& [key, value] : madeBy.items()) {
                        options.customData[key] = textOf(value);
                    }
                    for (const char* ext : {"usdc", "usda"}) {
                        if (has(ext)) {
                            const fs::path path = fileOf(out, ext);
                            ATHENEA_TRY(usd::writeParticleFieldStage(library, raw, path, options));
                            written.emplace_back(ext, path);
                        }
                    }
                }
                ply.clear();
                if (anySpz) {
                    auto streams = pack->spz(flat, stats.fractionalBits, o->sh1Bits, o->shRestBits, unit.perMetre);
                    if (!streams) return std::move(streams).error();
                    for (const std::string f : {"spz", "spz3", "spz4"}) {
                        if (!has(f)) continue;
                        io::SpzHeader header;
                        header.version = f == "spz3" ? 3u : f == "spz4" ? 4u : o->spzVersion;
                        header.count = static_cast<uint32_t>(total);
                        header.shDegree = 3;
                        header.fractionalBits = stats.fractionalBits;
                        header.antialiased = true;
                        const fs::path path = fileOf(out, f);
                        ATHENEA_TRY(io::writeSpz(path, header, streams->streams));
                        written.emplace_back(f, path);
                    }
                }
                if (has("glb")) {
                    io::GlbGaussians glb;
                    for (const render::FlatCloud& c : flat) {
                        auto blocks = pack->glb(c, kLogScaleFloor, unit.perMetre);
                        if (!blocks) return std::move(blocks).error();
                        glb.counts.push_back(c.count);
                        glb.chunks.push_back(std::move(*blocks));
                    }
                    glb.min = stats.min;
                    glb.max = stats.max;
                    glb.colorSpace = display ? "srgb_rec709_display" : "lin_rec709_display";
                    glb.metresPerUnit = metresPerFileUnit;
                    glb.extras = madeBy.dump();
                    const fs::path path = fileOf(out, "glb");
                    ATHENEA_TRY(io::writeGlbGaussians(path, glb));
                    written.emplace_back("glb", path);
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
            std::printf("flatten: %-56s %9u gaussians%s\n", c.prim.c_str(), c.cloud.count,
                        c.cloud.catcher ? " (shadow catcher)" : "");
        }
        std::printf("flatten: %llu gaussians, %u directions each, %s, roughness floor %s\n",
                    static_cast<unsigned long long>(total), settings.directions,
                    display ? "fitted to the sRGB a viewer shows" : "fitted in linear light",
                    settings.roughnessFloor ? std::to_string(floor).c_str() : "off");
        std::printf("flatten: under %s, turned %g degrees, %+g stops\n",
                    sky.empty() ? "the stage's own sky" : sky.c_str(), o->rotate, o->exposure);
        const double perGaussian = total > 0 ? 1.0 / static_cast<double>(total) : 0.0;
        const double perAxis = perGaussian / 3.0;
        for (const Unit& u : kUnits) {
            const render::FlatStats& s = perUnit[u.name];
            std::printf("flatten: in %-2s %6.2f%% of axes under e^-12 (%.2f%% of gaussians distorted), %.4f%% over "
                        "e^9; SPZ %6.2f%% under e^-10 (%.2f%% distorted), %.4f%% over e^5.94%s\n",
                        u.name, 100.0 * s.sizesFloored * perAxis, 100.0 * s.distorted * perGaussian,
                        100.0 * s.sizesCeiled * perAxis, 100.0 * s.spzSizesFloored * perAxis,
                        100.0 * s.spzDistorted * perGaussian, 100.0 * s.spzSizesCeiled * perAxis,
                        std::string(u.name) == unit.name ? "  <- written" : "");
        }
        std::printf("flatten: frame y up, %s (stage %c up, %g m a unit); box (%.4g %.4g %.4g) to (%.4g %.4g %.4g)\n",
                    unit.name, upAxis, metersPerUnit, static_cast<double>(stats.min[0]),
                    static_cast<double>(stats.min[1]), static_cast<double>(stats.min[2]),
                    static_cast<double>(stats.max[0]), static_cast<double>(stats.max[1]),
                    static_cast<double>(stats.max[2]));
        std::printf("flatten: %.2f%% of gaussians had their higher bands shrunk to stay positive; %.2f%% scaled into "
                    "SPZ's [-1, 1]; SPZ fixed point %u bits\n",
                    100.0 * stats.shrunk * perGaussian, 100.0 * stats.spzSaturated * perGaussian,
                    stats.fractionalBits);
        for (size_t b = 0; b < 3; ++b) {
            std::printf("flatten: band %zu against Spark's clamp %.3f (p95 of |c|, at least 1): %.3f%% of "
                        "coefficients past it, the largest %.2f times it\n",
                        b + 1, static_cast<double>(stats.shClamp[b]),
                        stats.shCoefficients[b] > 0 ? 100.0 * stats.shPast[b] / stats.shCoefficients[b] : 0.0,
                        static_cast<double>(stats.shWorst[b]));
        }

        // THE SIDECAR: how it was made, what packing lost, the files.
        Json sidecar = madeBy;
        Json lost = Json::object();
        lost["higherBandsShrunk"] = stats.shrunk * perGaussian;
        lost["spzHarmonicsScaled"] = stats.spzSaturated * perGaussian;
        lost["spzFractionalBits"] = stats.fractionalBits;
        for (const Unit& u : kUnits) {
            const render::FlatStats& s = perUnit[u.name];
            lost["sizes"][u.name] = {{"axesUnderE-12", s.sizesFloored * perAxis},
                                     {"gaussiansDistortedE-12", s.distorted * perGaussian},
                                     {"axesOverE9", s.sizesCeiled * perAxis},
                                     {"spzAxesUnderE-10", s.spzSizesFloored * perAxis},
                                     {"spzGaussiansDistortedE-10", s.spzDistorted * perGaussian},
                                     {"spzAxesOverE5.94", s.spzSizesCeiled * perAxis}};
        }
        for (size_t b = 0; b < 3; ++b) {
            lost["sparkShClamp"].push_back(
                {{"band", b + 1},
                 {"clamp", stats.shClamp[b]},
                 {"pastClamp", stats.shCoefficients[b] > 0
                                   ? static_cast<double>(stats.shPast[b]) / stats.shCoefficients[b]
                                   : 0.0},
                 {"largestOverClamp", stats.shWorst[b]}});
        }
        sidecar["lost"] = lost;
        for (const auto& c : clouds) {
            sidecar["clouds"].push_back({{"prim", c.prim}, {"gaussians", c.cloud.count}, {"catcher", c.cloud.catcher}});
        }
        for (const auto& [format, path] : written) {
            const uint64_t bytes = fileBytes(path);
            sidecar["files"].push_back({{"format", format}, {"file", path.filename().string()}, {"bytes", bytes}});
            std::printf("flatten: wrote %s, %.1f MB (%.1f bytes a gaussian)\n", path.string().c_str(),
                        static_cast<double>(bytes) / 1.0e6, static_cast<double>(bytes) * perGaussian);
        }
        const fs::path sidecarPath = out.string() + ".flatten.json";
        const auto writeSidecar = [&] {
            std::ofstream file(sidecarPath);
            file << sidecar.dump(2) << "\n";
            if (!file) {
                std::fprintf(stderr, "flatten: cannot write %s\n", sidecarPath.string().c_str());
                throw CLI::RuntimeError(1);
            }
        };
        writeSidecar();
        std::printf("flatten: wrote %s\n", sidecarPath.string().c_str());
        std::printf("flatten: shading and fit %.0f ms, packing and writing %.0f ms, %.0f ms in all\n", flattenMs,
                    packMs, msSince(began));

        if (o->validate.empty()) {
            return;
        }
        // --validate: each written file read back as any reader reads it, put
        // where the stage's meshes stand, rasterised blending as a standard
        // viewer blends, and measured against the meshes path traced under
        // the same sky (Mesh2SplatValidate.h); beside it the TX clouds under
        // that sky as athenea draws them (the composed stage).
        if (metersPerUnit != 1.0) {
            std::fprintf(stderr, "flatten: --validate places the files in a stage of metres only (this one is "
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
            std::fprintf(stderr, "no Measure bundle on the AOFX search path (try `athenea aofx list`), and "
                                 "--validate measures with it\n");
            throw CLI::RuntimeError(1);
        }
        for (const std::string& format : validateFormats) {
            const fs::path measured = fileOf(out, format);
            std::vector<ValidateRow> rows;
            ValidateJob job;
            job.stage = o->validateStage.empty() ? o->stage : o->validateStage;
            job.directory = (fs::path(o->validate) / format).string();
            job.gtDirectory = o->validate;
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
            job.skipMaterials = o->validateSkip;
            job.sky = sky;
            job.skyRotate = o->rotate;
            job.skyKeepsLights = keepLights;
            job.middleStage = composed.string();
            job.middleLabel = "TX raster";
            job.displayBlend = display;
            job.wholeCloud = true;
            job.results = &rows;
            // The file's frame back to the stage's: a turn about x, by the
            // stage's up axis, and by SPZ's own axes, which the engine's
            // reader turns (right-up-back read as right-down-front); and the
            // file's unit back to metres.
            double turn = upAxis == 'z' ? 90.0 : 0.0;
            if (format.rfind("spz", 0) == 0) {
                turn += 180.0;
            }
            // Read and placed once; each material's frame is given the same
            // file (a hard link, or a copy where links are refused).
            std::error_code dirMade;
            fs::create_directories(job.directory, dirMade);
            const fs::path placed = fs::absolute(fs::path(job.directory) / "placed.usdc");
            uint32_t placedCount = 0;
            std::printf("flatten: validating %s\n", measured.string().c_str());
            auto validated = validateConversion(
                job, *context, library, *measure,
                [&](const std::string&, const std::vector<std::string>&,
                    const std::string& output) -> Result<uint32_t> {
                    if (placedCount == 0) {
                        Result<uint32_t> got = 0u;
                        auto on = context->run([&] {
                            got = [&]() -> Result<uint32_t> {
                                auto loader = scene::CloudLoader::create(library);
                                if (!loader) return std::move(loader).error();
                                auto raw = scene::readSplatRecords(*loader, measured, 3);
                                if (!raw) return std::move(raw).error();
                                usd::ExportOptions options;
                                options.addCamera = false;
                                options.rotateXDegrees = turn;
                                options.uniformScale = 1.0 / static_cast<double>(unit.perMetre);
                                options.upAxis = upAxis;
                                options.metersPerUnit = 1.0;
                                ATHENEA_TRY(usd::writeParticleFieldStage(library, *raw, placed, options));
                                return raw->count;
                            }();
                        });
                        if (!on) return std::move(on).error();
                        if (!got) return std::move(got).error();
                        placedCount = *got;
                    }
                    std::error_code linked;
                    fs::remove(output, linked);
                    linked.clear();
                    fs::create_hard_link(placed, output, linked);
                    if (linked) {
                        linked.clear();
                        fs::copy_file(placed, output, fs::copy_options::overwrite_existing, linked);
                    }
                    if (linked) {
                        return Error::make(ErrorCode::IoFailure, "flatten --validate: {}: {}", output,
                                           linked.message());
                    }
                    return placedCount;
                });
            if (!validated) {
                cli::fail(validated.error());
            }
            Json quality = Json::array();
            for (const ValidateRow& r : rows) {
                Json row = {{"material", r.material}, {"cover", r.cover}};
                if (r.error.empty()) {
                    row["relMSE"] = r.relMse;
                    row["p99"] = r.p99;
                    row["mean"] = {r.mean[0], r.mean[1], r.mean[2]};
                    row["meanPT"] = {r.meanGt[0], r.meanGt[1], r.meanGt[2]};
                    row["txRelMSE"] = r.middleRelMse;
                    row["txP99"] = r.middleP99;
                    row["txMean"] = {r.middleMean[0], r.middleMean[1], r.middleMean[2]};
                } else {
                    row["error"] = r.error;
                }
                quality.push_back(row);
            }
            sidecar["quality"][format] = {
                {"against", "the meshes path traced under the same sky, " + std::to_string(o->validatePaths) +
                                " paths a pixel, " + std::to_string(o->validateBounces) + " bounces"},
                {"camera", o->validateCamera},
                {"tx", "athenea's own raster of the TX clouds under the same sky"},
                {"materials", quality}};
            writeSidecar();
        }
    });
}

}   // namespace athenea::cli
