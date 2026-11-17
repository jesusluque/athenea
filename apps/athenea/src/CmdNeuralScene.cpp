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
#include <set>
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
#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/core/Platform.h"
#include "athenea/io/Png.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/render/Discovery.h"
#include "athenea/render/Flatten.h"
#include "athenea/render/NeuralAppearance.h"
#include "athenea/render/NeuralFit.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/SplatLift.h"
#include "athenea/render/SceneGlass.h"
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
    float                    lod = 0.0F;
    uint64_t                 budget = 0;
    unsigned                 part = 1200000;
    bool                     byFrame = false;
    std::vector<uint64_t>    levels;
    std::string              format = "spz";
    bool                     history = false;
    std::string              grow;
    float                    growExposure = 0.0F;
    float                    morph = 0.0F;
    bool                     glass = false;
    std::string              lamps;                   // render: a file of lamps added to the scene (--lamps)
    bool                     lampsOnly = false;       // render: and the light it was found under left out
    uint32_t                 lampShadowRays = 8;      // render: rays to each lamp, over its disc
    uint32_t                 lampBounceRays = 16;     // render: rays that gather what the lamps' light bounces on to
    bool                     noAntialias = false;
    float                    madeSpread = -1.0F;      // render: the width, in pixels, the scene's discs were made at
    bool                     relight = false;         // shoot: a frame takes the light of what the scene holds again
    std::string              lightGroup;              // shoot: the frames are a group of lights' own, kept as its layer
    uint32_t                 sceneCache = 0;          // shoot: cells an axis of the scene's light kept for the paths' bounces (0: none)
    bool                     lightOverBase = false;   // shoot, a pass of groups: a kept light is carried from its pixel's base colour to the gaussian's own
    bool                     groupsOnly = false;      // shoot: --light-groups as a pass over a scene found: the groups' layers and no more
    std::vector<std::string> lightGroups;             // shoot: NAME=PATH, lights whose share of each frame is kept as a group's layer
    std::vector<std::string> groups;                  // render, export: NAME=LEVEL or NAME=R,G,B, a group of lights' level
    float                    ownLight = 1.0F;         // export: what the scene's own light is multiplied by
    bool                     baseColour = false;      // export: each gaussian's base colour as its colour
    bool                     noPrims = false;         // export: the parts without which prim each gaussian was made from
    std::vector<std::string> mattes;                  // render: NAME=PRIMPATH, the matte of the gaussians made from the prims at or under a path
    bool                     noHeadlight = false;     // shoot: a stage with no light is not lit from the eye
    std::vector<std::string> lightsOn;                // shoot: the stage's lights left on (the rest switched off)
    std::vector<std::string> lightsOff;               // shoot: the stage's lights switched off
    bool                     antialias = false;       // shoot: a new scene drawn with the 2D filter's compensation
    float                    shootSpread = 0.5F;      // shoot's own default: half a cell
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
    double                   timeStep = 0.0;
    bool                     freeEachFrame = false;
    double                   coveredEnough = 0.8;
    unsigned                 pyramid = 32;
    std::vector<unsigned>    storedFrames;
};

bool sizeOf(const std::string& text, uint32_t& width, uint32_t& height) {
    if (std::sscanf(text.c_str(), "%ux%u", &width, &height) != 2 || width == 0 || height == 0) {
        err("--size wants WIDTHxHEIGHT\n");
        return false;
    }
    return true;
}

/// A piece's gaussians on the device: a USD ParticleField, or a splat file.
/// The sizes a render's record has had: 26 floats, 27 with its parent's width, 30 with its parent's colour.
bool renderRecordOf(uint32_t floats) {
    return floats == render::kRenderRecord || floats == render::kRenderRecordParent ||
           floats == render::kRenderRecordParentColour;
}

/// HOW THE SCENE AT HAND IS DRAWN (io::NeuralScene::antialiased), for every cloud this command loads or
/// makes of it: unset, as the rasteriser's settings say; `false`, without the 2D filter's compensation.
/// Set once the manifest is read; a cloud carries it from there (scene::GpuSplats::antialiased).
std::optional<bool> drawnAs;
/// The width its discs were made at (io::NeuralScene::spread), for every frame this command draws of it:
/// drawn without the compensation, a disc pays it only where it projects narrower than it was made.
float madeAs = 0.0F;
/// THE TIME A PIECE'S LIGHT IS READ AT: the last snapshot whose layers of light are laid over a piece as it
/// is loaded (io::NeuralScenePiece::lights); 0, every layer there is -- the scene as it stands.
uint32_t lightAsOf = 0;
/// THE GROUPS OF LIGHTS LAID OVER A PIECE AS IT IS LOADED (io::NeuralScene::lightGroups): each group's layer
/// times its level and tint, added to the piece's own light by a kernel of `lightTool`'s. Unset (a command
/// that sets no tool): the piece's own light alone, which is what a shoot and an export read.
render::Discovery* lightTool = nullptr;
/// A PIECE IS LOADED WITH WHICH PRIM EACH GAUSSIAN WAS MADE FROM (io::NeuralScenePiece::prims): its ids as the
/// cloud's own (scene::GpuSplats::crypto), which the blend then names each pixel's coverage by. Set by a
/// render that was asked for a matte; otherwise the file is not read.
bool withPrims = false;
std::map<std::string, std::array<float, 3>> groupGain;       // a group named on the command line
std::array<float, 3> groupDefault{1.0F, 1.0F, 1.0F};         // every other group
void drawAs(const io::NeuralScene& scene) {
    drawnAs = scene.antialiased ? std::optional<bool>{} : std::optional<bool>{false};
    madeAs = scene.spread;
}

/// THE LIGHT A PIECE'S GAUSSIANS TOOK AGAIN SINCE THEY WERE MADE, laid over its records: the layers beside
/// the piece (`<piece>.l<snapshot>`), in order up to the time asked (`lightAsOf`), each where its frame said a
/// gaussian's light anew. A copy of what a kernel left (render::Discovery::relight), field for field. The
/// last layer laid (0: none).
Result<uint32_t> layLightOver(const std::filesystem::path& path, std::vector<float>& records, uint32_t floats,
                              uint32_t count) {
    uint32_t last = 0;
    std::vector<std::pair<uint32_t, std::filesystem::path>> layers;
    const std::string stem = path.filename().string() + ".l";
    std::error_code ignored;
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), ignored)) {
        const std::string name = entry.path().filename().string();
        if (name.size() > stem.size() && name.compare(0, stem.size(), stem) == 0 &&
            name.find_first_not_of("0123456789", stem.size()) == std::string::npos) {
            const uint32_t at = static_cast<uint32_t>(std::stoul(name.substr(stem.size())));
            if (lightAsOf == 0 || at <= lightAsOf) {
                layers.emplace_back(at, entry.path());
            }
        }
    }
    std::sort(layers.begin(), layers.end());
    for (const auto& [at, file] : layers) {
        last = at;
        auto light = io::readGaussians(file);
        if (!light) return std::move(light).error();
        if (light->floats != 4 || light->count > count) {
            return Error::make(ErrorCode::Unsupported, "{}: a layer of light of {} gaussians of {} floats, over a piece of {}",
                               file.string(), light->count, light->floats, count);
        }
        // Four floats a gaussian whose light was said: which one, and its colour.
        for (uint32_t k = 0; k < light->count; ++k) {
            const float* said = &light->records[size_t{k} * 4];
            const uint32_t which = static_cast<uint32_t>(said[0]);
            if (which < count) {
                float* record = &records[size_t{which} * floats];
                record[11] = said[1];
                record[12] = said[2];
                record[13] = said[3];
            }
        }
    }
    return last;
}

/// A PIECE'S GROUPS OF LIGHTS, each as a layer a gaussian (rgb, linear; a: 1 where the group lights it) and
/// the level it is drawn at (`groupGain`, `groupDefault`): the files beside the piece
/// (`<piece>.g.<group>.<n>`), a group's laid one over another in order.
///
/// WHAT GLOWS IS IN EVERY GROUP'S LAYER: a frame path traced under one group of lights holds the light of
/// the stage's emissive surfaces as well, as the scene's own does. The group named `dark` is that light alone
/// (shot with every light off), and it is taken out once for each group drawn, at that group's level -- so
/// the scene and its groups hold what glows once. It comes last: a colour is kept at nothing or more at each
/// step. Named on the command line, it is drawn at what is said (`--group dark=0` leaves it in).
Result<std::vector<std::pair<std::vector<std::array<float, 4>>, std::array<float, 3>>>> groupLayersOf(
    const std::filesystem::path& path, uint32_t count) {
    std::vector<std::pair<std::vector<std::array<float, 4>>, std::array<float, 3>>> out;
    std::map<std::string, std::vector<std::pair<uint32_t, std::filesystem::path>>> groups;
    const std::string stem = path.filename().string() + ".g.";
    std::error_code ignored;
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path(), ignored)) {
        const std::string name = entry.path().filename().string();
        if (name.size() <= stem.size() || name.compare(0, stem.size(), stem) != 0) {
            continue;
        }
        const size_t dot = name.rfind('.');
        if (dot == std::string::npos || dot <= stem.size() ||
            name.find_first_not_of("0123456789", dot + 1) != std::string::npos || dot + 1 == name.size()) {
            continue;
        }
        groups[name.substr(stem.size(), dot - stem.size())].emplace_back(
            static_cast<uint32_t>(std::stoul(name.substr(dot + 1))), entry.path());
    }
    std::array<float, 3> drawn{0.0F, 0.0F, 0.0F};
    std::vector<std::string> order;
    for (const auto& [group, files] : groups) {
        if (group == "dark") continue;
        const auto named = groupGain.find(group);
        const std::array<float, 3> gain = named != groupGain.end() ? named->second : groupDefault;
        for (int c = 0; c < 3; ++c) drawn[c] += gain[c];
        order.push_back(group);
    }
    if (groups.count("dark") != 0) order.push_back("dark");
    for (const std::string& group : order) {
        auto& files = groups[group];
        const auto named = groupGain.find(group);
        const std::array<float, 3> gain = named != groupGain.end() ? named->second
                                          : group == "dark" ? std::array<float, 3>{-drawn[0], -drawn[1], -drawn[2]}
                                                            : groupDefault;
        if (gain[0] == 0.0F && gain[1] == 0.0F && gain[2] == 0.0F) {
            continue;
        }
        std::sort(files.begin(), files.end());
        std::vector<std::array<float, 4>> dense(count, std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F});
        for (const auto& [frame, file] : files) {
            auto light = io::readGaussians(file);
            if (!light) return std::move(light).error();
            if (light->floats != 4) {
                return Error::make(ErrorCode::Unsupported, "{}: a group's layer is four floats a gaussian", file.string());
            }
            for (uint32_t k = 0; k < light->count; ++k) {
                const float* said = &light->records[size_t{k} * 4];
                const uint32_t which = static_cast<uint32_t>(said[0]);
                if (which < count) {
                    dense[which] = {said[1], said[2], said[3], 1.0F};
                }
            }
        }
        out.emplace_back(std::move(dense), gain);
    }
    return out;
}

/// With `parent`, a render's piece also gives the width of what each gaussian was laid over (left
/// invalid where the file has none).
Result<scene::GpuSplats> load(gpu::ShaderLibrary& library, scene::CloudLoader& loader,
                              const std::filesystem::path& path, gpu::Buffer* parent = nullptr,
                              gpu::Buffer* matter = nullptr) {
    const std::string ext = path.extension().string();
    if (ext == ".usda" || ext == ".usdc" || ext == ".usd") {
        return usd::uploadParticleField(library, loader, path);
    }
    if (ext == ".athg") {
        // What a render of the engine's own left: records as render::Discovery lays them out.
        auto held = io::readGaussians(path);
        if (!held) return std::move(held).error();
        // 26 floats before a gaussian kept its parent's width, 27 since: the fields the loader reads are
        // the same.
        if (!renderRecordOf(held->floats)) {
            return Error::make(ErrorCode::Unsupported, "{}: records of {} floats, and a render's are {}", path.string(),
                               held->floats, render::kRenderRecord);
        }
        if (parent != nullptr && held->floats > render::kRenderRecordParent) {
            // What each was laid over, a float4 a gaussian -- its colour and (w) its width -- laid side by
            // side out of the records: nothing is computed. A record with the width alone leaves it black.
            const bool coloured = held->floats >= render::kRenderRecordParentColour + 3;
            std::vector<std::array<float, 4>> over(held->count);
            for (uint32_t k = 0; k < held->count; ++k) {
                const float* record = &held->records[size_t{k} * held->floats];
                over[k] = {coloured ? record[render::kRenderRecordParentColour] : 0.0F,
                           coloured ? record[render::kRenderRecordParentColour + 1] : 0.0F,
                           coloured ? record[render::kRenderRecordParentColour + 2] : 0.0F,
                           record[render::kRenderRecordParent]};
            }
            auto made = gpu::Buffer::fromSpan<std::array<float, 4>>(
                library.device(), std::span<const std::array<float, 4>>(over), "piece.parent");
            if (!made) return std::move(made).error();
            *parent = *made;
        }
        if (matter != nullptr) {
            // WHAT THE SURFACE IS MADE OF THAT THE CLOUD DOES NOT KEEP (render::SceneGlass): what one crossing
            // tints and the index, a float4 a gaussian, laid side by side out of the records. No index for a
            // sheet, which bends nothing, nor for a record from before a gaussian kept its material.
            const bool carried = held->floats >= render::kRenderRecordTint + 3;
            std::vector<std::array<float, 4>> made(held->count);
            for (uint32_t k = 0; k < held->count; ++k) {
                const float* record = &held->records[size_t{k} * held->floats];
                const bool sheet = carried && record[render::kRenderRecordSheet] > 0.5F;
                made[k] = {carried ? record[render::kRenderRecordTint] : 1.0F,
                           carried ? record[render::kRenderRecordTint + 1] : 1.0F,
                           carried ? record[render::kRenderRecordTint + 2] : 1.0F,
                           carried && !sheet ? record[render::kRenderRecordIndex] : 0.0F};
            }
            auto laid = gpu::Buffer::fromSpan<std::array<float, 4>>(
                library.device(), std::span<const std::array<float, 4>>(made), "piece.matter");
            if (!laid) return std::move(laid).error();
            *matter = *laid;
        }
        uint32_t lightOf = 0;   // the last layer laid over the records
        if (auto laid = layLightOver(path, held->records, held->floats, held->count); !laid) {
            return std::move(laid).error();
        } else {
            lightOf = *laid;
        }
        io::RawSplats raw;
        // Under another light it is another cloud to whoever keeps clouds by where they came from.
        raw.source = lightOf != 0 ? path.string() + "@light" + std::to_string(lightOf) : path.string();
        raw.count = held->count;
        raw.encoding = render::Discovery::renderEncoding();
        raw.encoding.floatsPerRecord = held->floats;
        raw.records = std::move(held->records);
        raw.antialiased = drawnAs;
        loader.keepFaint(true);   // a record of what is gone counts every gaussian the file holds
        auto cloud = loader.upload(raw, 0);
        loader.keepFaint(false);
        if (cloud && lightOf != 0) {
            // Read again under another light it may come back in the buffers the last reading left: its
            // revision says it is not what was drawn from them.
            cloud->revision += lightOf;
        }
        if (cloud && lightTool != nullptr) {
            // ITS GROUPS OF LIGHTS, added to the piece's own light on the device, each at its level.
            auto groups = groupLayersOf(path, cloud->count);
            if (!groups) return std::move(groups).error();
            for (const auto& [dense, gain] : *groups) {
                auto laid = gpu::Buffer::fromSpan<std::array<float, 4>>(
                    library.device(), std::span<const std::array<float, 4>>(dense), "piece.group");
                if (!laid) return std::move(laid).error();
                ATHENEA_TRY(lightTool->addLight(*cloud, *laid, gain));
            }
        }
        return cloud;
    }
    return scene::loadSplatFile(loader, path.string(), 3);
}

/// The ids a piece keeps beside its records, as the cloud's own: a file laid on the device, nothing computed.
/// A piece without them (a scene shot before one kept any) is left as it is, and no matte names it.
Result<void> namePrims(gpu::ShaderLibrary& library, const std::filesystem::path& directory,
                       const io::NeuralScenePiece& piece, scene::GpuSplats& cloud) {
    if (!withPrims || piece.prims.empty() || cloud.count == 0) {
        return ok();
    }
    auto ids = io::readSeen(directory / piece.prims, cloud.count);
    if (!ids) return std::move(ids).error();
    auto laid = gpu::Buffer::fromSpan<uint32_t>(library.device(), std::span<const uint32_t>(*ids), "piece.prims");
    if (!laid) return std::move(laid).error();
    cloud.crypto = *laid;
    return ok();
}

/// A piece as it is drawn: its gaussians, less those a later look found gone.
/// With `upTo`, as it stood at that snapshot: less only what had gone by then.
Result<scene::GpuSplats> loadPiece(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                                   const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                                   uint32_t upTo = 0) {
    auto cloud = load(library, loader, directory / piece.cloud);
    if (!cloud) {
        return cloud;
    }
    ATHENEA_TRY(namePrims(library, directory, piece, *cloud));
    // The finest there is: less what is gone, and less what finer looks were laid over.
    for (const std::string* file : {&piece.gone, &piece.refined}) {
        if (file->empty()) {
            continue;
        }
        auto held = io::readSeen(directory / *file, cloud->count);
        if (!held) return std::move(held).error();
        auto record = discovery.record(cloud->count, *held);
        if (!record) return std::move(record).error();
        auto less = discovery.without(*cloud, *record, upTo);
        if (!less) return less;
        cloud = std::move(less);
    }
    return cloud;
}

bool boxInView(const std::array<float, 3>& min, const std::array<float, 3>& max, const render::Projection& projection,
               uint32_t width, uint32_t height);

/// A piece as the cut reads it: its gaussians, its two records and what each gaussian was laid over.
struct PieceForCut {
    scene::GpuSplats cloud;
    gpu::Buffer      gone;
    gpu::Buffer      refined;
    gpu::Buffer      parent;
    gpu::Buffer      matter;   ///< with glass: what each is made of (render::SceneGlass)
};

Result<PieceForCut> loadForCut(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                               const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                               bool withMatter = false) {
    PieceForCut held;
    auto cloud = load(library, loader, directory / piece.cloud, &held.parent, withMatter ? &held.matter : nullptr);
    if (!cloud) return std::move(cloud).error();
    held.cloud = std::move(*cloud);
    ATHENEA_TRY(namePrims(library, directory, piece, held.cloud));
    const std::string* files[2] = {&piece.gone, &piece.refined};
    for (int k = 0; k < 2; ++k) {
        if (files[k]->empty()) {
            continue;
        }
        auto words = io::readSeen(directory / *files[k], held.cloud.count);
        if (!words) return std::move(words).error();
        auto record = discovery.record(held.cloud.count, *words);
        if (!record) return std::move(record).error();
        (k == 0 ? held.gone : held.refined) = *record;
    }
    return held;
}

/// A piece cut for a view (Discovery::cut): the gaussians to draw from there at `threshold` pixels, as the
/// scene stood at look `upTo`. Gathered: what is not drawn from here is not in the cloud handed back.
Result<scene::GpuSplats> cutPiece(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                                  const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                                  const render::Projection& projection, float threshold, uint32_t upTo,
                                  float morph = 0.0F) {
    auto held = loadForCut(library, loader, discovery, directory, piece);
    if (!held) return std::move(held).error();
    return discovery.cut(held->cloud, held->gone.valid() ? &held->gone : nullptr,
                         held->refined.valid() ? &held->refined : nullptr,
                         held->parent.valid() ? &held->parent : nullptr, projection, threshold, upTo, true, morph);
}

/// THE SCENE CUT FOR A VIEW, AS ONE CLOUD: every piece in view planned, room made for what they draw
/// together, and each gathered into it at its place. One cloud to draw, whatever the pieces: a cut of a
/// cloud a piece drew a twentieth of the gaussians in no less time than the scene whole.
Result<scene::GpuSplats> cutScene(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                                  const std::filesystem::path& directory, const io::NeuralScene& scene,
                                  const render::Projection& projection, uint32_t width, uint32_t height,
                                  float threshold, uint32_t upTo, float morph, render::SceneGlass* glass = nullptr,
                                  render::GlassSeen* seen = nullptr) {
    std::vector<PieceForCut> held;
    std::vector<render::Discovery::CutPlan> plans;
    uint32_t total = 0;
    std::array<float, 3> low{1.0e30F, 1.0e30F, 1.0e30F};
    std::array<float, 3> high{-1.0e30F, -1.0e30F, -1.0e30F};
    for (const io::NeuralScenePiece& piece : scene.pieces) {
        if ((upTo != 0 && piece.foundIn > upTo) || (piece.gaussians != 0 && piece.removed >= piece.gaussians) ||
            // With glass the scene is the room its glass shows: what is behind the eye is mirrored, and
            // what the glass bends into view may stand outside it.
            (glass == nullptr && !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height))) {
            continue;
        }
        auto loaded = loadForCut(library, loader, discovery, directory, piece, glass != nullptr);
        if (!loaded) return std::move(loaded).error();
        auto plan = discovery.planCut(loaded->cloud, loaded->gone.valid() ? &loaded->gone : nullptr,
                                      loaded->refined.valid() ? &loaded->refined : nullptr,
                                      loaded->parent.valid() ? &loaded->parent : nullptr, projection, threshold, upTo);
        if (!plan) return std::move(plan).error();
        if (plan->drawn == 0) {
            continue;
        }
        total += plan->drawn;
        for (size_t axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], piece.boundsMin[axis]);
            high[axis] = std::max(high[axis], piece.boundsMax[axis]);
        }
        held.push_back(std::move(*loaded));
        plans.push_back(std::move(*plan));
    }
    if (held.empty()) {
        return scene::GpuSplats{};
    }
    auto cloud = discovery.cutCloud(held.front().cloud, total);
    if (!cloud) return cloud;
    cloud->bounds.min = low;
    cloud->bounds.max = high;
    gpu::Buffer matter;
    if (glass != nullptr) {
        auto room = glass->room(*cloud);
        if (!room) return std::move(room).error();
        matter = *room;
    }
    gpu::Buffer prims;   // which prim each gaussian of the cut was made from, where a matte was asked for
    if (withPrims) {
        auto room = discovery.primsOfCut(total);
        if (!room) return std::move(room).error();
        prims = *room;
        cloud->crypto = prims;
    }
    uint32_t base = 0;
    for (size_t k = 0; k < held.size(); ++k) {
        ATHENEA_TRY(discovery.gatherCut(held[k].cloud, held[k].parent.valid() ? &held[k].parent : nullptr, plans[k],
                                        projection, threshold, morph, *cloud, base));
        if (withPrims && held[k].cloud.hasCrypto()) {
            ATHENEA_TRY(discovery.gatherPrims(held[k].cloud.crypto, held[k].cloud.count, plans[k], prims, base));
        }
        if (glass != nullptr) {
            ATHENEA_TRY(glass->gather(held[k].cloud, held[k].matter, plans[k], *cloud, matter, base));
        }
        base += plans[k].drawn;
    }
    if (glass != nullptr) {
        // What each glass gaussian shows from this eye, written as its colour: the picture is the view's.
        auto shown = glass->see(projection, *cloud, matter);
        if (!shown) return std::move(shown).error();
        if (seen != nullptr) {
            *seen = *shown;
        }
    }
    return cloud;
}

/// THE THRESHOLD A BUDGET ASKS FOR: every piece in view adds its gaussians' spans into one histogram, and the
/// finest threshold that draws no more than `budget` of them is read off it (Discovery::thresholdFor).
Result<float> thresholdForBudget(gpu::ShaderLibrary& library, scene::CloudLoader& loader, render::Discovery& discovery,
                                 const std::filesystem::path& directory, const io::NeuralScene& scene,
                                 const render::Projection& projection, uint32_t width, uint32_t height,
                                 uint32_t upTo, uint64_t budget, uint64_t& drawn) {
    auto histogram = discovery.cutHistogram();
    if (!histogram) return std::move(histogram).error();
    for (const io::NeuralScenePiece& piece : scene.pieces) {
        if ((upTo != 0 && piece.foundIn > upTo) || (piece.gaussians != 0 && piece.removed >= piece.gaussians) ||
            !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height)) {
            continue;
        }
        auto held = loadForCut(library, loader, discovery, directory, piece);
        if (!held) return std::move(held).error();
        ATHENEA_TRY(discovery.spans(held->cloud, held->gone.valid() ? &held->gone : nullptr,
                                    held->refined.valid() ? &held->refined : nullptr,
                                    held->parent.valid() ? &held->parent : nullptr, projection, upTo, *histogram));
    }
    return discovery.thresholdFor(*histogram, budget, &drawn);
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
        drawAs(scene);
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
    drawAs(*scene);
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
    drawAs(*scene);
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
        if (p.refinedCount != 0) {
            out(", %u refined", p.refinedCount);
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
    drawAs(*scene);
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
    settings.madeSpread = madeAs;
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
    drawAs(*scene);
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
    settings.madeSpread = madeAs;
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
        drawAs(scene);
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
    settings.madeSpread = madeAs;
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

struct LensRecords;
Result<uint32_t> standingRecords(const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                                 uint32_t asOf, uint32_t& floats, std::vector<float>& records,
                                 LensRecords* lenses = nullptr);

/// One part of a scene handed on: `count` records (of `floats` floats each) written as an SPZ with no
/// harmonics above the first, the light scaled by `gain`. Its size in bytes, and its box added to `low`, `high`.
Result<uint64_t> writeSpzPart(gpu::Device& device, render::Discovery& discovery, render::FlattenPack& pack,
                              std::span<const float> records, uint32_t count, uint32_t floats, float gain,
                              const std::filesystem::path& path, std::array<float, 3>& low, std::array<float, 3>& high) {
    auto onDevice = gpu::Buffer::fromSpan<float>(device, records, "export.records");
    if (!onDevice) return std::move(onDevice).error();
    auto flat = discovery.flatRecords(*onDevice, count, floats, gain);
    if (!flat) return std::move(flat).error();
    render::FlatCloud cloud;
    cloud.count = count;
    cloud.records = *flat;
    const std::vector<render::FlatCloud> clouds{cloud};
    auto stats = pack.stats(clouds, -30.0F);
    if (!stats) return std::move(stats).error();
    // THE FORMAT IS THE FILE'S NAME: .ply (INRIA's 3DGS layout) and .glb (glTF with KHR_gaussian_splatting)
    // are the same gaussians as the .spz, for a program that reads those. Whole floats: several times the size.
    const std::string extension = path.extension().string();
    if (extension == ".ply" || extension == ".glb") {
        for (size_t axis = 0; axis < 3; ++axis) {
            low[axis] = std::min(low[axis], stats->min[axis]);
            high[axis] = std::max(high[axis], stats->max[axis]);
        }
        if (extension == ".ply") {
            auto packed = pack.ply(cloud, -30.0F);
            if (!packed) return std::move(packed).error();
            const std::vector<std::vector<float>> chunks{std::move(*packed)};
            const std::vector<std::string> comments{"athenea neural-scene export: a scene found by rendering"};
            ATHENEA_TRY(io::writePly3dgs(path, chunks, comments));
        } else {
            auto packed = pack.glb(cloud, -30.0F);
            if (!packed) return std::move(packed).error();
            io::GlbGaussians glb;
            glb.counts = {count};
            glb.chunks.push_back(std::move(*packed));
            glb.min = stats->min;
            glb.max = stats->max;
            glb.degree = 3;
            glb.generator = "athenea neural-scene export";
            ATHENEA_TRY(io::writeGlbGaussians(path, glb));
        }
        std::error_code unread;
        return static_cast<uint64_t>(std::filesystem::file_size(path, unread));
    }
    auto streams = pack.spz(clouds, stats->fractionalBits, 8, 8);
    if (!streams) return std::move(streams).error();
    io::SpzHeader header;
    header.version = 3;
    header.count = count;
    // A gaussian a look made has one colour: no harmonics above the first are written.
    header.shDegree = 0;
    header.fractionalBits = stats->fractionalBits;
    header.antialiased = drawnAs.value_or(render::RenderSettings{}.antialias);
    streams->streams[5].clear();
    ATHENEA_TRY(io::writeSpz(path, header, streams->streams));
    for (size_t axis = 0; axis < 3; ++axis) {
        low[axis] = std::min(low[axis], stats->min[axis]);
        high[axis] = std::max(high[axis], stats->max[axis]);
    }
    std::error_code ignored;
    return static_cast<uint64_t>(std::filesystem::file_size(path, ignored));
}

/// THE GAUSSIANS OF SOLID GLASS AMONG A PART'S, set apart: their records, and what the first of them is
/// made of -- one glass a part is what a viewer is told.
struct LensRecords {
    std::vector<float>   records;
    uint32_t             count = 0;
    float                ior = 1.0F;
    std::array<float, 3> tint{1.0F, 1.0F, 1.0F};
};

/// A part of a scene handed on, as its index names it: the file, how many gaussians, and -- where the part
/// is a frame's -- which frame, its time and the camera it was rendered through. Where the part holds solid
/// glass, those gaussians are in a file of their own beside it, with the glass's index and its tint: a
/// viewer that knows how shades them by the eye, and one that does not draws the picture they keep.
struct ExportedPart {
    std::string file;
    uint32_t    gaussians = 0;
    /// WHICH PRIM EACH GAUSSIAN WAS MADE FROM, beside the file and in its order (`<file's stem>.ids`): sixteen
    /// bits a gaussian, its prim's place counted from one in the index's `prims`, 0 for none. And the same
    /// of its glass and of its mirrors. Empty where the scene names no prims.
    std::string ids;
    std::string glassIds;
    std::string mirrorIds;
    std::string          glassFile;
    uint32_t             glassGaussians = 0;
    float                glassIor = 1.0F;
    std::array<float, 3> glassTint{1.0F, 1.0F, 1.0F};
    /// And where it holds polished metal, those in a file of their own too: a mirror shows what stands
    /// before it, which is the eye's to work out as a lens's is.
    std::string          mirrorFile;
    uint32_t             mirrorGaussians = 0;
    /// THE SCENE AS IT STOOD (export --history): what the part's frame found that a later frame took out or
    /// laid finer gaussians over, in files of their own, each with the frame from which it no longer stands.
    uint32_t                  untilFrame = 0;
    std::vector<ExportedPart> history;
    /// A history's file holds every span of its part, one after another by the frame each stands until:
    /// (that frame, its gaussians, its glass's, its mirrors') a span, so what stands at a frame is the
    /// file's end from a count on, and a part's history is three files at most rather than three a span.
    std::vector<std::array<uint32_t, 4>> spans;
    uint32_t    frame = 0;
    double      time = 0.0;
    std::string camera;
    double      seconds = 0.0;   ///< what the frame took to render and take in
};

/// THE INDEX A VIEWER READS, written whole beside its parts and put in place by a rename: a file that is
/// read while it grows is never read half written. `growing` says more parts may come.
/// A level of detail of a scene handed on: the budget it was asked for, the width (in the scene's units) a
/// gaussian is drawn up to in place of the finer ones laid over it, and its parts.
struct ExportedLevel {
    uint64_t                  budget = 0;
    float                     width = 0.0F;
    std::vector<ExportedPart> parts;
};

Result<void> writeExportIndex(const std::filesystem::path& stem, const io::NeuralScene& scene,
                              const std::vector<ExportedPart>& parts, const std::array<float, 3>& low,
                              const std::array<float, 3>& high, bool growing,
                              const std::vector<ExportedLevel>* levels = nullptr,
                              const std::vector<std::pair<uint32_t, std::string>>* prims = nullptr) {
    return platform::writeAtomically(stem.string() + ".json", [&](const std::filesystem::path& partial) -> Result<void> {
        std::ofstream index(partial);
        uint64_t total = 0;
        for (const ExportedPart& part : parts) {
            total += part.gaussians + part.glassGaussians + part.mirrorGaussians;
            for (const ExportedPart& earlier : part.history) {
                total += earlier.gaussians + earlier.glassGaussians + earlier.mirrorGaussians;
            }
        }
        index << "{\n  \"name\": \"" << scene.name << "\",\n  \"upAxis\": \"" << scene.upAxis
              << "\",\n  \"metersPerUnit\": " << scene.metersPerUnit << ",\n  \"antialiased\": "
              << (scene.antialiased ? "true" : "false") << ",\n  \"spread\": " << scene.spread << ",\n  \"growing\": " << (growing ? "true" : "false")
              << ",\n  \"gaussians\": " << total << ",\n  \"min\": [" << low[0] << ", " << low[1] << ", " << low[2]
              << "],\n  \"max\": [" << high[0] << ", " << high[1] << ", " << high[2] << "],\n  \"parts\": [";
        const auto writeOne = [&](const ExportedPart& part, bool first, const char* indent) {
            index << (first ? "\n" : ",\n") << indent << "{\"file\": \"" << part.file << "\", \"gaussians\": " << part.gaussians;
            if (!part.ids.empty()) {
                index << ", \"ids\": \"" << part.ids << "\"";
            }
            if (part.frame != 0) {
                index << ", \"frame\": " << part.frame << ", \"time\": " << part.time << ", \"camera\": \"" << part.camera
                      << "\", \"seconds\": " << part.seconds;
            }
            if (!part.glassFile.empty()) {
                index << ", \"glass\": {\"file\": \"" << part.glassFile << "\", \"gaussians\": " << part.glassGaussians
                      << ", \"ior\": " << part.glassIor << ", \"tint\": [" << part.glassTint[0] << ", "
                      << part.glassTint[1] << ", " << part.glassTint[2] << "]";
                if (!part.glassIds.empty()) {
                    index << ", \"ids\": \"" << part.glassIds << "\"";
                }
                index << "}";
            }
            if (!part.mirrorFile.empty()) {
                index << ", \"mirror\": {\"file\": \"" << part.mirrorFile << "\", \"gaussians\": " << part.mirrorGaussians;
                if (!part.mirrorIds.empty()) {
                    index << ", \"ids\": \"" << part.mirrorIds << "\"";
                }
                index << "}";
            }
        };
        const auto writePart = [&](const ExportedPart& part, bool first, const char* indent) {
            writeOne(part, first, indent);
            if (!part.history.empty()) {
                index << ", \"history\": [";
                for (size_t h = 0; h < part.history.size(); ++h) {
                    const ExportedPart& earlier = part.history[h];
                    writeOne(earlier, h == 0, "      ");
                    index << ", \"until\": " << earlier.untilFrame;
                    if (!earlier.spans.empty()) {
                        index << ", \"spans\": [";
                        for (size_t k = 0; k < earlier.spans.size(); ++k) {
                            const auto& span = earlier.spans[k];
                            index << (k != 0 ? ", [" : "[") << span[0] << ", " << span[1] << ", " << span[2] << ", "
                                  << span[3] << "]";
                        }
                        index << "]";
                    }
                    index << "}";
                }
                index << "\n    ]";
            }
            index << "}";
        };
        for (size_t k = 0; k < parts.size(); ++k) {
            writePart(parts[k], k == 0, "    ");
        }
        index << "\n  ],";
        if (levels != nullptr && !levels->empty()) {
            // THE LEVELS OF DETAIL, coarser one after another: the scene whole at each, for a device that
            // cannot draw the finest. `width` is the level's threshold in the scene's units.
            index << "\n  \"levels\": [";
            for (size_t l = 0; l < levels->size(); ++l) {
                const ExportedLevel& level = (*levels)[l];
                uint64_t count = 0;
                for (const ExportedPart& part : level.parts) {
                    count += part.gaussians + part.glassGaussians + part.mirrorGaussians;
                }
                index << (l != 0 ? ",\n    " : "\n    ") << "{\"budget\": " << level.budget << ", \"width\": " << level.width
                      << ", \"gaussians\": " << count << ", \"parts\": [";
                for (size_t k = 0; k < level.parts.size(); ++k) {
                    writePart(level.parts[k], k == 0, "      ");
                }
                index << "\n    ]}";
            }
            index << "\n  ],";
        }
        if (prims != nullptr && !prims->empty()) {
            // THE PRIMS THE PARTS' IDS NAME, in ascending order of their Cryptomatte id: a gaussian's sixteen
            // bits in a part's `ids` file are a place in this list counted from one.
            index << "\n  \"prims\": [";
            for (size_t k = 0; k < prims->size(); ++k) {
                char hex[16];
                std::snprintf(hex, sizeof(hex), "%08x", (*prims)[k].first);
                index << (k != 0 ? ",\n    " : "\n    ") << "{\"path\": \"" << (*prims)[k].second << "\", \"id\": \"" << hex
                      << "\"}";
            }
            index << "\n  ],";
        }
        index << "\n  \"cameras\": [";
        for (size_t k = 0; k < scene.cameras.size(); ++k) {
            const io::NeuralSceneCamera& c = scene.cameras[k];
            index << (k != 0 ? ",\n    " : "\n    ") << "{\"name\": \"" << c.name << "\", \"eye\": [" << c.eye[0] << ", "
                  << c.eye[1] << ", " << c.eye[2] << "], \"target\": [" << c.target[0] << ", " << c.target[1] << ", "
                  << c.target[2] << "], \"focal\": " << c.focal << "}";
        }
        index << "\n  ]\n}\n";
        if (!index) {
            return Error(ErrorCode::IoFailure, "cannot write " + partial.string());
        }
        return ok();
    });
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
    // A NEW SCENE IS DRAWN WITHOUT THE 2D FILTER'S COMPENSATION, and its discs are half a cell wide. The
    // compensation pays for a third of a pixel of spread in opacity, and a disc of a pixel or less pays
    // most of what it has: at a cell wide the discs overlapped enough to close a surface all the same, and
    // the frame was the scene low-passed (24 dB through its own camera); narrower, they were sharp and thin.
    // Spread and not thinned, half a cell wide closes the surface and keeps the frame (29 dB). A scene that
    // is there already is drawn as its manifest says.
    scene.antialiased = all.antialias;
    scene.spread = all.shootSpread;
    scene.baseColour = true;   // a scene begun now: every gaussian of it keeps its surface's base colour
    if (std::filesystem::exists(directory / "scene.json")) {
        auto read = io::readNeuralScene(directory);
        if (!read) return report(read.error());
        scene = std::move(*read);
        drawAs(scene);
    }
    std::error_code made;
    std::filesystem::create_directories(directory / "pieces", made);
    auto renderer = usd::StageRenderer::open(all.stage);
    if (!renderer) return report(renderer.error());
    usd::StageRenderer& stage = **renderer;
    if (!all.lightsOn.empty() || !all.lightsOff.empty()) {
        // WHICH OF THE STAGE'S LIGHTS LIGHT IT: with --lights-on only those named, with --lights-off all but
        // those. A light is named by its prim's path, or by a path above it (a scope of lamps).
        const auto under = [](const std::string& prim, const std::vector<std::string>& named) {
            return std::any_of(named.begin(), named.end(), [&](const std::string& n) {
                return prim == n || (prim.size() > n.size() && prim.compare(0, n.size(), n) == 0 && prim[n.size()] == '/');
            });
        };
        uint32_t on = 0;
        uint32_t off = 0;
        for (const usd::StageLight& light : stage.lights()) {
            const bool lit = (all.lightsOn.empty() || under(light.prim, all.lightsOn)) && !under(light.prim, all.lightsOff);
            if (auto set = stage.setLightOn(light.prim, lit); !set) return report(set.error());
            ++(lit ? on : off);
        }
        // A stage whose lights were switched is lit by what is left on and by nothing else: with none on it
        // is dark but for what glows, not lit from the eye.
        stage.setHeadlight(false);
        out("lights: %u of the stage's on, %u off%s\n", on, off, on == 0 ? "; none is on: the stage is dark but for what glows" : "");
    }
    if (all.noHeadlight || !all.lightGroup.empty() || all.groupsOnly) {
        stage.setHeadlight(false);
    }
    stage.setPathSamples(std::max(all.passSamples, 1u));
    stage.setPathTotal(std::max(all.samples, 1u));
    stage.setPathBounces(all.bounces);
    stage.setDenoise(all.denoise);
    if (all.rayMargin > 0.0F) stage.setRayMargin(all.rayMargin);
    // THE GROUPS OF LIGHTS FOUND WITH THE SCENE (--light-groups NAME=PATH): each light at or under PATH is
    // put in the group NAME, and the path tracer gathers what each group lights in a plane of its own, from
    // the same paths as the frame -- the frame is the sum of what no group lights and the planes. The scene
    // is made from the frame less the planes (its own light: what a shoot with those lights off finds), and
    // each plane is kept as its group's layer.
    std::vector<std::string> withinNames;
    for (const std::string& said : all.lightGroups) {
        const size_t eq = said.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= said.size()) {
            err("--light-groups takes NAME=PATH, not '%s'\n", said.c_str());
            return kExitFailure;
        }
        const std::string group = said.substr(0, eq);
        const std::string path = said.substr(eq + 1);
        uint32_t named = 0;
        for (const usd::StageLight& light : stage.lights()) {
            if (light.prim == path || (light.prim.size() > path.size() && light.prim.compare(0, path.size(), path) == 0 &&
                                       light.prim[path.size()] == '/')) {
                if (auto set = stage.setLightGroup(light.prim, group); !set) return report(set.error());
                ++named;
            }
        }
        if (named == 0) {
            err("--light-groups %s: the stage has no light at or under %s\n", said.c_str(), path.c_str());
            return kExitFailure;
        }
        if (std::find(withinNames.begin(), withinNames.end(), group) == withinNames.end()) {
            withinNames.push_back(group);
        }
        out("group %s: %u of the stage's lights at %s\n", group.c_str(), named, path.c_str());
    }
    if (all.groupsOnly) {
        if (withinNames.empty()) {
            err("--groups-only keeps the layers of the groups --light-groups names: none was named\n");
            return kExitFailure;
        }
        // The named groups' lights alone light the stage: every path is one of theirs.
        uint32_t off = 0;
        for (const usd::StageLight& light : stage.lights()) {
            const bool named = std::any_of(all.lightGroups.begin(), all.lightGroups.end(), [&](const std::string& said) {
                const std::string path = said.substr(said.find('=') + 1);
                return light.prim == path || (light.prim.size() > path.size() && light.prim.compare(0, path.size(), path) == 0 &&
                                              light.prim[path.size()] == '/');
            });
            if (!named) {
                if (auto set = stage.setLightOn(light.prim, false); !set) return report(set.error());
                ++off;
            }
        }
        out("groups only: %u of the stage's other lights switched off\n", off);
    }
    // A pass over a scene found: one group by its own frames (--light-group), or several by their planes of
    // the same frames (--light-groups --groups-only).
    const bool groupPass = !all.lightGroup.empty() || all.groupsOnly;
    const std::vector<std::string> passGroups = all.groupsOnly ? withinNames : std::vector<std::string>{all.lightGroup};
    if (!withinNames.empty() && !all.lightGroup.empty()) {
        err("--light-groups finds the groups with the scene; --light-group is a pass of one group over a scene found\n");
        return kExitFailure;
    }
    // The path tracer's planes beside the picture: what each pixel's surface is made of.
    std::vector<std::string> outputs{"material"};
    for (const std::string& group : withinNames) outputs.push_back("lightGroup:" + group);
    // And which prim each pixel shows -- the engine's Cryptomatte, whose ids are the hashes of the prims'
    // paths: what a gaussian made from the pixel keeps as its own (render --matte). A pass that makes no
    // gaussian asks for none.
    const bool namesPrims = !all.groupsOnly && all.lightGroup.empty();
    if (namesPrims) outputs.push_back("CryptoObject00");
    stage.requestOutputs(outputs);
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
    // The stage's own unit, as its up axis: a stage in centimetres left a scene that said metres.
    scene.metersPerUnit = all.metersPerUnit > 0.0 ? all.metersPerUnit : stage.metersPerUnit();
    // Drawn without the 2D filter's compensation from here on, and by whoever draws the scene after: what
    // the looks ask of the scene (is this pixel held?) and what a render shows are then the same picture.
    if (all.noAntialias) scene.antialiased = false;
    drawAs(scene);
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
    std::vector<gpu::Buffer> refined;        // and its record of what finer looks were laid over
    std::vector<std::unique_ptr<render::NeuralCloud>> sizedOf;   // its cloud of sizes, made once while it is here
    const auto drop = [&](size_t k) {
        instances.erase(instances.begin() + static_cast<std::ptrdiff_t>(k));
        clouds.erase(clouds.begin() + static_cast<std::ptrdiff_t>(k));
        pieceOf.erase(pieceOf.begin() + static_cast<std::ptrdiff_t>(k));
        gone.erase(gone.begin() + static_cast<std::ptrdiff_t>(k));
        refined.erase(refined.begin() + static_cast<std::ptrdiff_t>(k));
        sizedOf.erase(sizedOf.begin() + static_cast<std::ptrdiff_t>(k));
    };
    // WHAT THE NEXT VIEW STILL SEES STAYS. A view is brought in by dropping the pieces its frustum does not
    // meet and reading the ones it meets that are not here: a camera a step on reads nothing, where reading
    // every piece back for every frame was four fifths of a frame's time by the hundredth. `read` is how
    // many pieces came off the disk, `left` how many of the scene's stayed on it.
    uint32_t piecesRead = 0;
    uint32_t piecesLeft = 0;
    const auto bringInView = [&](const render::Camera& camera) -> Result<void> {
        const render::Projection projection = render::projectionFor(camera, width, height);
        piecesRead = 0;
        piecesLeft = 0;
        for (size_t k = instances.size(); k-- > 0;) {
            const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
            if (all.freeEachFrame || !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height)) {
                drop(k);
            }
        }
        for (size_t k = 0; k < scene.pieces.size(); ++k) {
            const io::NeuralScenePiece& piece = scene.pieces[k];
            if (piece.gaussians != 0 && piece.removed + piece.refinedCount >= piece.gaussians) {
                continue;   // nothing of it is the finest there is
            }
            if (std::find(pieceOf.begin(), pieceOf.end(), k) != pieceOf.end()) {
                continue;   // here already
            }
            if (!boxInView(piece.boundsMin, piece.boundsMax, projection, width, height)) {
                ++piecesLeft;
                continue;
            }
            auto cloud = loadPiece(library, *loader, *discovery, directory, piece);
            if (!cloud) return std::move(cloud).error();
            clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
            instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
            pieceOf.push_back(k);
            gone.emplace_back();
            refined.emplace_back();
            sizedOf.emplace_back();
            ++piecesRead;
        }
        return ok();
    };
    // An instance drawn by its sizes: the cloud of them is made the first time it is asked for.
    const auto bySizeOf = [&](size_t k) -> Result<render::SplatInstance> {
        if (!sizedOf[k]) {
            auto size = discovery->sizes(*clouds[k]);
            if (!size) return std::move(size).error();
            sizedOf[k] = std::make_unique<render::NeuralCloud>(std::move(*size));
        }
        render::SplatInstance instance = instances[k];
        instance.neural = sizedOf[k].get();
        return instance;
    };
    // A FILE THAT GROWS (--grow <stem>): when a frame is whole, what it found is written as a part beside the
    // others and the index is put in place again, saying more may come -- so that a viewer shows the scene
    // being made while it is made, a frame at a time, and can play it back after. The parts already written
    // are not gone back to: what a later frame takes out of an earlier one stays in the earlier part.
    std::optional<render::FlattenPack> growPack;
    std::vector<ExportedPart> grown;
    std::array<float, 3> grownLow{1.0e30F, 1.0e30F, 1.0e30F};
    std::array<float, 3> grownHigh{-1.0e30F, -1.0e30F, -1.0e30F};
    if (!all.grow.empty()) {
        auto pack = render::FlattenPack::create(library);
        if (!pack) return report(pack.error());
        growPack.emplace(std::move(*pack));
        std::error_code ignored;
        std::filesystem::create_directories(std::filesystem::path(all.grow).parent_path(), ignored);
    }
    const auto growBy = [&](const io::NeuralSceneFrame& frame, uint32_t frameNumber, bool more) -> Result<void> {
        if (all.grow.empty()) {
            return ok();
        }
        std::vector<float> records;
        LensRecords lenses;
        uint32_t floats = 0;
        uint32_t held = 0;
        for (const io::NeuralScenePiece& piece : scene.pieces) {
            if (piece.foundIn < frame.firstSnapshot || piece.foundIn > frame.lastSnapshot ||
                (piece.gaussians != 0 && piece.removed + piece.refinedCount >= piece.gaussians)) {
                continue;
            }
            auto kept = standingRecords(directory, piece, 0, floats, records, &lenses);
            if (!kept) return std::move(kept).error();
            held += *kept;
        }
        if (held != 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "_%03zu.spz", grown.size());
            const std::filesystem::path path = all.grow + name;
            auto size = writeSpzPart(device, *discovery, *growPack, std::span<const float>(records), held, floats,
                                     std::exp2(all.growExposure), path, grownLow, grownHigh);
            if (!size) return std::move(size).error();
            ExportedPart part;
            part.file = path.filename().string();
            part.gaussians = held;
            if (lenses.count != 0) {
                std::snprintf(name, sizeof(name), "_%03zu_glass.spz", grown.size());
                const std::filesystem::path glassPath = all.grow + name;
                auto glassSize = writeSpzPart(device, *discovery, *growPack, std::span<const float>(lenses.records),
                                              lenses.count, floats, std::exp2(all.growExposure), glassPath, grownLow,
                                              grownHigh);
                if (!glassSize) return std::move(glassSize).error();
                part.glassFile = glassPath.filename().string();
                part.glassGaussians = lenses.count;
                part.glassIor = lenses.ior;
                part.glassTint = lenses.tint;
            }
            part.frame = frameNumber;
            part.time = frame.time;
            part.camera = frame.camera;
            part.seconds = frame.seconds;
            grown.push_back(part);
        }
        return writeExportIndex(all.grow, scene, grown, grownLow, grownHigh, more);
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
        // With groups of lights found with the scene (--light-groups), the scene's own light is what no group
        // holds of the frame: the plane the engine keeps beside the groups', from the same paths.
        gpu::Buffer ownFrame;
        if (!withinNames.empty()) {
            auto rest = stage.displaySource("lightGroupRest");
            if (!rest) return report(rest.error());
            if (rest->buffer == nullptr) {
                err("the render left no plane for what its light groups do not hold\n");
                return kExitFailure;
            }
            auto plane = discovery->framePlane(*rest->buffer, rest->offset, width * height);
            if (!plane) return report(plane.error());
            ownFrame = *plane;
        }
        const gpu::Buffer& frameColour = withinNames.empty() ? *colourOf->buffer : ownFrame;
        const gpu::Buffer& frameDepth = *depthOf->buffer;
        const gpu::Buffer& planes = *planesOf->buffer;
        // WHICH PRIM EACH PIXEL SHOWS: the first layer of the frame's Cryptomatte, where the engine kept one.
        const gpu::Buffer* frameIds = nullptr;
        uint32_t idsAt = 0;
        if (namesPrims) {
            auto idsOf = stage.displaySource("CryptoObject00");
            if (!idsOf) return report(idsOf.error());
            frameIds = idsOf->buffer;
            idsAt = idsOf->offset;
            for (const auto& [path, id] : stage.cryptoManifest()) {
                scene.prims[path] = id;
            }
        }
        // A piece's ids beside its records, in their order (io::NeuralScenePiece::prims).
        const auto keepPrims = [&](const gpu::Buffer& prims, uint32_t count, io::NeuralScenePiece& piece) -> Result<void> {
            if (!prims.valid()) {
                return ok();
            }
            auto ids = prims.readAll<uint32_t>(device);
            if (!ids) return std::move(ids).error();
            ids->resize(count);
            piece.prims = "pieces/" + piece.id + ".prims";
            return io::writeSeen(directory / piece.prims, *ids);
        };
        scene.snapshots.push_back({time, name, {}, samples, footprint, clock()});
        const uint32_t stamp = static_cast<uint32_t>(scene.snapshots.size());

        render::RenderSettings settings;
        settings.madeSpread = madeAs;
        settings.width = width;
        settings.height = height;
        settings.depth = render::RenderSettings::Depth::Threshold;
        settings.depthThreshold = 0.5F;
        // Where the scene begins: a pane's own depth, which the depth above goes past.
        render::RenderSettings beginning = settings;
        beginning.depthThreshold = 0.4F * all.glassOpacity;   // under the least a pane covers
        // And where it is opaque past any fringe: what stands behind a disc that overhangs an edge.
        render::RenderSettings ending = settings;
        ending.depthThreshold = 0.9F;
        render::RenderTargets known;
        render::RenderTargets knownFirst;
        render::RenderTargets knownLast;
        render::RenderTargets knownSize;
        const auto drawKnown = [&]() -> Result<void> {
            auto drawn = raster->render(camera, instances, settings, known);
            if (!drawn) return std::move(drawn).error();
            auto first = raster->render(camera, instances, beginning, knownFirst);
            if (!first) return std::move(first).error();
            auto last = raster->render(camera, instances, ending, knownLast);
            if (!last) return std::move(last).error();
            std::vector<render::SplatInstance> bySize;
            for (size_t k = 0; k < clouds.size(); ++k) {
                auto instance = bySizeOf(k);
                if (!instance) return std::move(instance).error();
                bySize.push_back(*instance);
            }
            auto sizes = raster->render(camera, bySize, settings, knownSize);
            if (!sizes) return std::move(sizes).error();
            return ok();
        };
        const render::Projection projection = render::projectionFor(camera, width, height);
        const float disc = all.shootSpread * static_cast<float>(footprint);
        uint32_t taken = 0;
        uint32_t refinedNow = 0;
        uint32_t firstThrough = 0;
        uint32_t firstAgain = 0;
        // The scene as it stood when the look began, by its sizes and its depth: what the gaussians this look
        // makes are laid over, and so their parents'.
        render::RenderTargets before;
        render::RenderTargets beforeSize;
        bool hasBefore = false;
        if (all.relight && last && !instances.empty()) {
            // THE LIGHT OF WHAT THE SCENE HOLDS, TAKEN AGAIN (--relight). Before the frame is asked what it
            // holds: each standing gaussian the frame shows at its own depth takes the frame's colour, and
            // the piece keeps that as a layer of light stamped with this look. The gaussians stay -- where
            // they are, what they are made of -- and a look that asks about colour (--repaint) then finds the
            // scene's light the frame's, and takes out only what is of another surface.
            uint32_t relit = 0;
            // One piece's light taken again and kept: `cloud` on the device, `record` what of it is gone.
            const auto takeAgain = [&](io::NeuralScenePiece& piece, scene::GpuSplats& cloud, const gpu::Buffer* record) -> Result<void> {
                uint32_t took = 0;
                auto layer = discovery->relight(cloud, frameColour, frameDepth, width, height, projection,
                                                all.tolerance, took, record);
                if (!layer) return std::move(layer).error();
                if (took == 0) {
                    return ok();
                }
                auto floats = layer->readAll<float>(device);
                if (!floats) return std::move(floats).error();
                // Kept as the gaussians the frame spoke of and no more: which one, and its colour (a piece
                // holds under 2^24 gaussians, which a float counts exactly).
                std::vector<float> said;
                said.reserve(size_t{took} * 4);
                for (uint32_t g = 0; g < cloud.count; ++g) {
                    const float* one = &(*floats)[size_t{g} * 4];
                    if (one[3] > 0.5F) {
                        said.insert(said.end(), {static_cast<float>(g), one[0], one[1], one[2]});
                    }
                }
                const std::filesystem::path file = directory / (piece.cloud + ".l" + std::to_string(stamp));
                ATHENEA_TRY(io::writeGaussians(file, static_cast<uint32_t>(said.size() / 4), 4, said));
                piece.lights.push_back(stamp);
                relit += took;
                return ok();
            };
            const auto asked = [&](const io::NeuralScenePiece& piece) {
                // Not this frame's own, which is this light already; and a render's piece.
                return piece.foundIn < frameFirst && piece.cloud.size() >= 5 &&
                       piece.cloud.compare(piece.cloud.size() - 5, 5, ".athg") == 0;
            };
            const auto goneOf = [&](const io::NeuralScenePiece& piece, uint32_t count) -> Result<gpu::Buffer> {
                std::vector<uint32_t> words;
                if (!piece.gone.empty()) {
                    auto read = io::readSeen(directory / piece.gone, count);
                    if (!read) return std::move(read).error();
                    words = std::move(*read);
                }
                return discovery->record(count, words);
            };
            for (size_t k = 0; k < clouds.size(); ++k) {
                io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                if (!asked(piece)) {
                    continue;
                }
                // What it holds that finer looks were laid over is asked too: by the piece's record of what
                // is gone, not by what the cloud draws.
                gpu::Buffer record = gone[k];
                if (!record.valid()) {
                    auto read = goneOf(piece, clouds[k]->count);
                    if (!read) return report(read.error());
                    record = *read;
                }
                if (auto done = takeAgain(piece, *clouds[k], &record); !done) return report(done.error());
            }
            // AND THE PIECES NOTHING OF WHICH IS THE FINEST: the coarse looks of earlier frames, which are not
            // here to be drawn but are what the scene is drawn with from far (`render --lod`, a level of an
            // export). Read for this, asked, and let go.
            for (size_t k = 0; k < scene.pieces.size(); ++k) {
                io::NeuralScenePiece& piece = scene.pieces[k];
                if (!asked(piece) || std::find(pieceOf.begin(), pieceOf.end(), k) != pieceOf.end() ||
                    (piece.gaussians != 0 && piece.removed >= piece.gaussians) ||
                    !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height)) {
                    continue;
                }
                auto cloud = load(library, *loader, directory / piece.cloud);
                if (!cloud) return report(cloud.error());
                auto record = goneOf(piece, cloud->count);
                if (!record) return report(record.error());
                if (auto done = takeAgain(piece, *cloud, &*record); !done) return report(done.error());
            }
            out("%s at %g: %u gaussians the scene held took the frame's light again\n", name.c_str(), time, relit);
        }
        if (!instances.empty()) {
            if (auto drawn = drawKnown(); !drawn) return report(drawn.error());
            {
                // Of what earlier frames left: this frame's gathering looks are no one's parents.
                std::vector<render::SplatInstance> earlier;
                std::vector<render::SplatInstance> bySize;
                for (size_t k = 0; k < clouds.size(); ++k) {
                    if (scene.pieces[pieceOf[k]].foundIn >= frameFirst) {
                        continue;
                    }
                    earlier.push_back(instances[k]);
                    auto instance = bySizeOf(k);
                    if (!instance) return report(instance.error());
                    bySize.push_back(*instance);
                }
                if (!earlier.empty()) {
                    auto drawnBefore = raster->render(camera, earlier, settings, before);
                    if (!drawnBefore) return report(drawnBefore.error());
                    auto sizesBefore = raster->render(camera, bySize, settings, beforeSize);
                    if (!sizesBefore) return report(sizesBefore.error());
                    hasBefore = true;
                }
            }
            // A piece's record, on the device the first time it is marked in: what a file held, or nothing.
            const auto recordOf = [&](std::vector<gpu::Buffer>& records, size_t k, const std::string& file) -> Result<void> {
                if (records[k].valid()) {
                    return ok();
                }
                const uint32_t count = clouds[k]->count;
                std::vector<uint32_t> words;
                if (!file.empty()) {
                    auto read = io::readSeen(directory / file, count);
                    if (!read) return std::move(read).error();
                    words = std::move(*read);
                }
                auto record = discovery->record(count, words);
                if (!record) return std::move(record).error();
                records[k] = *record;
                return ok();
            };
            // WHAT THE RENDER SEES THROUGH IS GONE, AND WHAT IT MAKES FINER IS REFINED. Two things a look
            // finds of what the scene held, kept apart: a surface that is no longer there or no longer that
            // colour went at a time, and is stamped gone; a surface that is there, held by coarser gaussians
            // than this look makes, is still the scene's from far -- its gaussians are stamped refined, kept,
            // and drawn where the finer ones are too fine to matter (Discovery::cut). Either way they leave
            // the scene as a look knows it, which is the finest there is.
            std::vector<uint32_t> newlyGone(instances.size(), 0);
            std::vector<uint32_t> newlyRefined(instances.size(), 0);
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
                uint32_t now = 0;
                // A mask carried to the gaussians of `judged` by the blend, and those most of whose weight it
                // fell on stamped in `records`.
                const auto carry = [&](const gpu::Buffer& carried, const std::vector<size_t>& judgedAt,
                                       std::vector<gpu::Buffer>& records, bool refining, float most) -> Result<void> {
                    std::vector<render::SplatInstance> judged;
                    uint32_t slots = 0;
                    for (const size_t k : judgedAt) {
                        judged.push_back(instances[k]);
                        slots += clouds[k]->count;
                    }
                    auto lift = render::makeSplatLift(device, slots);
                    if (!lift) return std::move(lift).error();
                    render::RenderSettings carrying = settings;
                    render::liftWhileBlending(carrying, *lift, carried);
                    render::RenderTargets drawn;
                    auto lifted = raster->render(camera, judged, carrying, drawn);
                    if (!lifted) return std::move(lifted).error();
                    uint32_t base = 0;
                    for (const size_t k : judgedAt) {
                        const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                        const uint32_t count = clouds[k]->count;
                        ATHENEA_TRY(recordOf(records, k, refining ? piece.refined : piece.gone));
                        // What is gone is what the frame sees through at the gaussian's own middle; what is
                        // refined stays judged by its weight, which is how much of it finer ones cover.
                        auto marked = discovery->markGone(lift->sums, base, records[k], count, stamp, 0.05F,
                                                          clouds[k].get(), &projection, all.refine * disc, most,
                                                          &carried, width, height, &frameDepth, all.tolerance,
                                                          refining, &frameColour, all.repaint);
                        if (!marked) return std::move(marked).error();
                        if (*marked > 0) {
                            auto kept = discovery->without(*clouds[k], records[k]);
                            if (!kept) return std::move(kept).error();
                            *clouds[k] = std::move(*kept);
                            (refining ? newlyRefined : newlyGone)[k] += *marked;
                            now += *marked;
                        }
                        base += count;
                    }
                    return ok();
                };
                std::vector<size_t> own;
                std::vector<size_t> older;
                for (size_t k = 0; k < instances.size(); ++k) {
                    (scene.pieces[pieceOf[k]].foundIn >= frameFirst ? own : older).push_back(k);
                }
                // THIS FRAME'S OWN LOOKS are judged in the scene whole, by the mask as it is, and what a finer
                // look replaces of them is refined: a frame's coarse looks are the levels above its last.
                if (!own.empty()) {
                    if (auto done = carry(*mask, own, refined, true, 0.0F); !done) return report(done.error());
                }
                // WHAT EARLIER FRAMES FOUND is judged as it stood without them (a coarse disc laid in a gap
                // covers its neighbours, under which the fine gaussians of the frame before read as coarse),
                // and never through glass, where the scene's depth is not the surface's.
                if (!older.empty()) {
                    std::vector<render::SplatInstance> judged;
                    std::vector<render::SplatInstance> bySize;
                    for (const size_t k : older) {
                        judged.push_back(instances[k]);
                        auto instance = bySizeOf(k);
                        if (!instance) return report(instance.error());
                        bySize.push_back(*instance);
                    }
                    render::RenderTargets alone;
                    render::RenderTargets aloneSize;
                    auto drawn = raster->render(camera, judged, settings, alone);
                    if (!drawn) return report(drawn.error());
                    auto drawnSize = raster->render(camera, bySize, settings, aloneSize);
                    if (!drawnSize) return report(drawnSize.error());
                    for (uint32_t reason = 1; reason <= 2; ++reason) {
                        uint32_t flagged = 0;
                        uint32_t flaggedAgain = 0;
                        auto only = discovery->seenThrough(frameColour, frameDepth, width, height, alone.colour,
                                                           alone.depth, all.tolerance, flagged, all.repaint,
                                                           &flaggedAgain, &aloneSize.colour, all.refine, &projection,
                                                           disc, reason);
                        if (!only) return report(only.error());
                        if (flagged + flaggedAgain == 0) {
                            continue;
                        }
                        if (auto spared = discovery->sparePanes(*only, planes, width, height); !spared) return report(spared.error());
                        // Gone, by the rule there was. Refined, only where nine tenths of a gaussian lie under
                        // finer ones: a parent a look covered part of stays the surface's where it did not reach.
                        if (auto done = carry(*only, older, reason == 1 ? gone : refined, reason == 2,
                                              reason == 2 ? 0.9F : 0.0F);
                            !done) {
                            return report(done.error());
                        }
                    }
                }
                if (now == 0) {
                    break;
                }
                if (auto again2 = drawKnown(); !again2) return report(again2.error());
            }
            // THE LOOKS TAKEN WHILE THE FRAME GATHERED END WITH ITS LAST. They were there for something to
            // draw before the frame was whole, made of a sixty-fourth and a sixteenth of its paths; what is
            // left of their discs goes, and the levels above the last look are made of the whole frame
            // (the pyramid, below).
            if (last) {
                for (size_t k = 0; k < instances.size(); ++k) {
                    io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                    if (piece.foundIn < frameFirst) {
                        continue;
                    }
                    if (auto has = recordOf(gone, k, piece.gone); !has) return report(has.error());
                    auto words = gone[k].readAll<uint32_t>(device);
                    if (!words) return report(words.error());
                    words->resize(clouds[k]->count);
                    for (uint32_t& word : *words) {
                        if (word == 0u) {
                            word = stamp;
                        }
                    }
                    auto record = discovery->record(clouds[k]->count, *words);
                    if (!record) return report(record.error());
                    gone[k] = *record;
                    // All of it is gone now, whatever of it had been refined on the way.
                    newlyGone[k] = piece.gaussians - std::min(piece.gaussians, piece.removed + piece.refinedCount +
                                                                                   newlyRefined[k]);
                }
            }
            for (size_t k = 0; k < instances.size(); ++k) {
                io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                if (newlyGone[k] != 0) {
                    auto words = gone[k].readAll<uint32_t>(device);
                    if (!words) return report(words.error());
                    words->resize(clouds[k]->count);
                    piece.gone = "pieces/" + piece.id + ".gone";
                    if (auto written = io::writeSeen(directory / piece.gone, *words); !written) return report(written.error());
                    piece.removed += newlyGone[k];
                    taken += newlyGone[k];
                }
                if (newlyRefined[k] != 0) {
                    auto words = refined[k].readAll<uint32_t>(device);
                    if (!words) return report(words.error());
                    words->resize(clouds[k]->count);
                    piece.refined = "pieces/" + piece.id + ".refined";
                    if (auto written = io::writeSeen(directory / piece.refined, *words); !written) return report(written.error());
                    piece.refinedCount += newlyRefined[k];
                    refinedNow += newlyRefined[k];
                }
                if (newlyGone[k] + newlyRefined[k] != 0) {
                    ++piece.revision;
                    piece.movedAt = piece.revision;
                }
            }
            // A piece nothing of which is the finest any more leaves the device: a look knows the scene by
            // its finest. The disk keeps it, each gaussian with the look it went or was refined at.
            for (size_t k = instances.size(); k-- > 0;) {
                const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                if (piece.removed + piece.refinedCount >= piece.gaussians) {
                    drop(k);
                }
            }
            known = {};
            knownFirst = {};
            knownLast = {};
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
            held.last = &knownLast.depth;
            held.size = &knownSize.colour;
        }
        // A frame's gathering looks are no parents: what a gaussian is laid over is what earlier frames
        // left, and the levels above it are the pyramid's.
        if (hasBefore) {
            held.beforeSize = &beforeSize.colour;
            held.beforeDepth = &before.depth;
            held.beforeColour = &before.colour;
        }
        render::RenderLookSettings how;
        how.tolerance = all.tolerance;
        how.spread = all.shootSpread;
        how.repaint = all.repaint;
        how.refine = all.refine;
        how.footprint = footprint;
        how.glassOpacity = all.glassOpacity;
        how.turned = all.turned;
        // Glass is made as glass by the view's last look; the coarse ones before it carry its picture.
        how.panes = last;
        how.pyramid = last && all.pyramid > 1;
        auto look = discovery->fromRender(frameColour, frameDepth, planes, width, height, projection, held, how,
                                          frameIds, idsAt);
        if (!look) return report(look.error());
        // THE FRAME'S PYRAMID. Where the whole frame found surface the scene did not hold, it also leaves
        // the levels above: cells of 2, 4, 8... pixels of the same frame, each the mean of its pixels that
        // show the one surface -- a gaussian the right width for an eye that far again, and that far again.
        // They are born refined: the last look's gaussians lie over them. A look does not read them; the cut
        // draws them from where the finer ones are under its threshold. A third as many gaussians again.
        uint32_t pyramidMade = 0;
        if (last && look->made > 0) {
            render::KnownScene under = held;
            under.size = nullptr;        // held however coarse: a level is laid only where there was nothing
            under.beforeSize = nullptr;
            under.beforeDepth = nullptr;
            for (uint32_t level = 2; level <= all.pyramid; level *= 2) {
                render::RenderLookSettings above = how;
                above.footprint = level;
                above.refine = 0.0F;
                above.repaint = 0.0F;
                above.panes = false;
                above.pyramid = level * 2 <= all.pyramid;
                auto cells = discovery->fromRender(frameColour, frameDepth, planes, width, height, projection, under, above,
                                                   frameIds, idsAt);
                if (!cells) return report(cells.error());
                if (cells->made == 0) {
                    continue;
                }
                char label[96];
                std::snprintf(label, sizeof(label), "%s_t%g_f%u_s%u", name.c_str(), time, level, stamp);
                io::NeuralScenePiece piece;
                piece.id = label;
                piece.cloud = "pieces/" + piece.id + ".athg";
                auto floats = cells->records.readAll<float>(device);
                if (!floats) return report(floats.error());
                if (auto written = io::writeGaussians(directory / piece.cloud, cells->made, render::kRenderRecord, *floats);
                    !written) {
                    return report(written.error());
                }
                loader->keepFaint(true);
                auto cloud = loader->upload(cells->records, cells->made, render::Discovery::renderEncoding(), piece.cloud, 0);
                loader->keepFaint(false);
                if (!cloud) return report(cloud.error());
                cloud->antialiased = drawnAs;
                piece.prim = all.stage;
                piece.gaussians = cloud->count;
                piece.boundsMin = cloud->bounds.min;
                piece.boundsMax = cloud->bounds.max;
                piece.origin = "discovered";
                piece.foundBy = name;
                piece.foundAt = time;
                piece.foundIn = stamp;
                piece.footprint = level;
                piece.refined = "pieces/" + piece.id + ".refined";
                piece.refinedCount = cloud->count;
                if (auto kept = keepPrims(cells->prims, cells->made, piece); !kept) return report(kept.error());
                const std::vector<uint32_t> born(cloud->count, stamp);
                if (auto written = io::writeSeen(directory / piece.refined, born); !written) return report(written.error());
                scene.pieces.push_back(piece);
                scene.snapshots.back().pieces.push_back(piece.id);
                pyramidMade += cloud->count;
            }
        }
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
            cloud->antialiased = drawnAs;
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
            if (auto named = keepPrims(look->prims, look->made, piece); !named) return report(named.error());
            scene.pieces.push_back(piece);
            scene.snapshots.back().pieces.push_back(id);
            clouds.push_back(std::make_unique<scene::GpuSplats>(std::move(*cloud)));
            instances.push_back(render::SplatInstance{clouds.back().get(), render::Mat4::identity(), {}});
            pieceOf.push_back(scene.pieces.size() - 1);
            gone.emplace_back();
            refined.emplace_back();
            sizedOf.emplace_back();
        }
        // The manifest is written when the frame is whole (shootView), not a look: by the hundredth frame it
        // is four hundred pieces long.
        if (last) {
            if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
        }
        out("  %s at %g, %u paths a pixel, cells of %u: %u hold a surface, the scene held %u (%.1f%%), %u let the eye "
            "through; %u gaussians made (%u of them behind glass), %u gone and %u refined (seen through at %u "
            "pixels, finer at %u), %u more in the levels above; %.2f s\n",
            name.c_str(), time, samples, footprint, look->surfaces, look->held,
            look->surfaces > 0 ? 100.0 * look->held / look->surfaces : 0.0, look->panes, look->made, look->behind,
            taken, refinedNow, firstThrough, firstAgain, pyramidMade,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count());
        return 0;
    };

    std::set<std::string> cachedPieces;   // --scene-cache: the pieces whose light is in the cells
    uint32_t cacheHolds = 0;              // and how many they are
    uint32_t groupFrame = 0;   // --light-group: the frames of this run, counted from one
    std::map<std::string, gpu::Buffer> groupLayers;         // and what they said, a piece: on the device, four floats a gaussian
    std::map<std::string, uint32_t> groupCounts;            // and how many gaussians each is of
    std::map<std::string, gpu::Buffer> groupGone;            // a piece's record of what is gone, read once
    std::map<std::string, gpu::Buffer> groupMatter;          // --light-over-base: a piece's own base colours, read once
    std::map<size_t, std::unique_ptr<scene::GpuSplats>> groupCoarse;   // the pieces read for a group alone, kept
    // ONE VIEW AT ONE TIME: drawn a pass at a time, and looked at as it gathers.
    const auto shootView = [&](const render::Camera& camera, const std::string& name, double time) -> int {
        const auto began = std::chrono::steady_clock::now();
        if (auto brought = bringInView(camera); !brought) return report(brought.error());
        if (groupPass) {
            const auto brought = std::chrono::steady_clock::now();
            // A GROUP OF LIGHTS' OWN FRAME (--light-group). The stage handed is lit by that group alone;
            // the frame is path traced whole and what each gaussian of the scene sends in it is kept as the
            // group's layer. Nothing of the scene is made, taken out or said anew: no look is taken.
            for (uint32_t pass = 0; pass < 65536; ++pass) {
                if (auto drawn = stage.draw(camera, time, width, height, "rt"); !drawn) return report(drawn.error());
                if (stage.pathConverged()) {
                    break;
                }
            }
            auto colourOf = stage.displaySource("color");
            auto depthOf = stage.displaySource("depth");
            if (!colourOf) return report(colourOf.error());
            if (!depthOf) return report(depthOf.error());
            if (colourOf->buffer == nullptr || depthOf->buffer == nullptr) {
                err("the render left no colour or depth\n");
                return kExitFailure;
            }
            const auto traced = std::chrono::steady_clock::now();
            const render::Projection projection = render::projectionFor(camera, width, height);
            ++groupFrame;
            uint32_t kept = 0;
            // What is read for a group: the frame itself (--light-group: the stage was lit by that group alone),
            // or the group's plane of it (--groups-only: the path tracer's share of the frame a light group).
            std::string groupName;
            const gpu::Buffer* groupColour = colourOf->buffer;
            uint32_t groupBase = 0;
            // --light-over-base: the frame's base colours, a pixel (the path tracer's plane of them).
            const gpu::Buffer* frameBase = nullptr;
            uint32_t frameBaseAt = 0;
            if (all.lightOverBase) {
                auto baseOf = stage.displaySource("transmissionColour");
                if (!baseOf) return report(baseOf.error());
                if (baseOf->buffer == nullptr) {
                    err("the render left no plane of base colours: --light-over-base wants a path traced stage of meshes\n");
                    return kExitFailure;
                }
                frameBase = baseOf->buffer;
                frameBaseAt = baseOf->offset;
            }
            const auto keepOf = [&](const io::NeuralScenePiece& piece, scene::GpuSplats& cloud) -> Result<void> {
                if (piece.cloud.size() < 5 || piece.cloud.compare(piece.cloud.size() - 5, 5, ".athg") != 0 ||
                    (piece.gaussians != 0 && piece.removed >= piece.gaussians)) {
                    return ok();
                }
                // Its record of what is gone, read once a run: nothing of the scene changes under a group.
                auto found = groupGone.find(piece.cloud);
                if (found == groupGone.end()) {
                    std::vector<uint32_t> words;
                    if (!piece.gone.empty()) {
                        auto read = io::readSeen(directory / piece.gone, cloud.count);
                        if (!read) return std::move(read).error();
                        words = std::move(*read);
                    }
                    auto made = discovery->record(cloud.count, words);
                    if (!made) return std::move(made).error();
                    found = groupGone.emplace(piece.cloud, *made).first;
                }
                // Gathered a piece over the run, on the device: of the views that show a gaussian the nearest
                // one's word stands (the fourth float is how far the eye was) -- a view from far reads, at the
                // gaussian's middle, a pixel that is several gaussians wide -- and the piece's layer is read
                // back and written once, when every view is taken.
                uint32_t took = 0;
                const std::string key = groupName + "\t" + piece.cloud;
                auto whole = groupLayers.find(key);
                const gpu::Buffer* own = nullptr;
                if (frameBase != nullptr) {
                    auto matter = groupMatter.find(piece.cloud);
                    if (matter == groupMatter.end()) {
                        gpu::Buffer read;
                        auto again = load(library, *loader, directory / piece.cloud, nullptr, &read);
                        if (!again) return std::move(again).error();
                        matter = groupMatter.emplace(piece.cloud, read).first;
                    }
                    own = &matter->second;
                }
                auto layer = discovery->relight(cloud, *groupColour, *depthOf->buffer, width, height, projection,
                                                all.tolerance, took, &found->second, true,
                                                whole != groupLayers.end() ? &whole->second : nullptr, groupBase,
                                                frameBase, frameBaseAt, own, scene.spread);
                if (!layer) return std::move(layer).error();
                if (whole == groupLayers.end()) {
                    groupLayers.emplace(key, *layer);
                    groupCounts[key] = cloud.count;
                }
                kept += took;
                return ok();
            };
            for (const std::string& passGroup : passGroups) {
                groupName = passGroup;
                if (all.groupsOnly) {
                    auto planeOf = stage.displaySource("lightGroup:" + passGroup);
                    if (!planeOf) return report(planeOf.error());
                    if (planeOf->buffer == nullptr) {
                        err("the render left no plane for the light group %s\n", passGroup.c_str());
                        return kExitFailure;
                    }
                    groupColour = planeOf->buffer;
                    groupBase = planeOf->offset;
                }
                for (size_t k = 0; k < clouds.size(); ++k) {
                    if (auto done = keepOf(scene.pieces[pieceOf[k]], *clouds[k]); !done) return report(done.error());
                }
                for (size_t k = 0; k < scene.pieces.size(); ++k) {
                    const io::NeuralScenePiece& piece = scene.pieces[k];
                    if (std::find(pieceOf.begin(), pieceOf.end(), k) != pieceOf.end() ||
                        !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height) || piece.cloud.size() < 5 ||
                        piece.cloud.compare(piece.cloud.size() - 5, 5, ".athg") != 0) {
                        continue;
                    }
                    // The pieces nothing of which is the finest are read once a run and kept: every view of a walk
                    // asks most of them again, and reading them off the disk was most of a frame's time.
                    auto held = groupCoarse.find(k);
                    if (held == groupCoarse.end()) {
                        auto cloud = load(library, *loader, directory / piece.cloud);
                        if (!cloud) return report(cloud.error());
                        held = groupCoarse.emplace(k, std::make_unique<scene::GpuSplats>(std::move(*cloud))).first;
                    }
                    if (auto done = keepOf(piece, *held->second); !done) return report(done.error());
                }
            }
            const auto ended = std::chrono::steady_clock::now();
            out("%s at %g: the light of %s kept for %u gaussians, in %.1f s (the scene's pieces brought %.1f, the frame "
                "path traced %.1f, the light kept %.1f)\n", name.c_str(), time,
                all.groupsOnly ? "the groups named" : ("group " + all.lightGroup).c_str(), kept,
                std::chrono::duration<double>(ended - began).count(), std::chrono::duration<double>(brought - began).count(),
                std::chrono::duration<double>(traced - brought).count(), std::chrono::duration<double>(ended - traced).count());
            std::fflush(stdout);   // a walk of groups is long: each view is said when it is done
            return 0;
        }
        const uint32_t read = piecesRead;
        const uint32_t left = piecesLeft;
        const size_t kept = instances.size() - read;
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
        // WHERE THE SCENE ALREADY SHOWS THE VIEW, THE COARSE LOOKS ARE NOT TAKEN. They are there so that a
        // render still gathering leaves something to draw; a view the scene covers has that already, and
        // each of them cost a look's draws of the whole scene for the few cells it filled. How much of
        // the view the scene covers is its frame's mean alpha, before a path is traced.
        double covered = 0.0;
        if (!instances.empty()) {
            render::RenderSettings plain;
            plain.madeSpread = madeAs;
            plain.width = width;
            plain.height = height;
            render::RenderTargets seen;
            auto drawn = raster->render(camera, instances, plain, seen);
            if (!drawn) return report(drawn.error());
            auto cover = render::imageStats(library, seen.colour, width, height);
            if (!cover) return report(cover.error());
            covered = cover->mean[3];
        }
        if (covered >= all.coveredEnough && levels.size() > 1) {
            levels.erase(levels.begin(), levels.end() - 1);
        }
        // WHAT THE SCENE HOLDS SO FAR IS WHAT THIS VIEW'S BOUNCES MEET (--scene-cache).
        if (all.sceneCache > 0 && cacheHolds > 0) {
            auto cells = discovery->cacheResolved();
            if (!cells) return report(cells.error());
            stage.setSceneCache(*cells, discovery->cacheSide(), discovery->cachePlanes(), discovery->cacheMin(),
                                discovery->cacheCell());
        }
        size_t next = 0;
        double firstAt = 0.0;
        // Where a view's time goes, said in its line: the scene brought and drawn for its cover, the frame's
        // passes (the paths and the denoiser), the looks taken of it, and the groups' light kept.
        const auto seconds = [](std::chrono::steady_clock::time_point from) {
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
        };
        const double beforeSeconds = seconds(began);
        double pathSeconds = 0.0;
        double lookSeconds = 0.0;
        const uint32_t firstSnapshot = static_cast<uint32_t>(scene.snapshots.size()) + 1;
        for (uint32_t pass = 0; pass < 65536 && next < levels.size(); ++pass) {
            const auto passBegan = std::chrono::steady_clock::now();
            if (auto drawn = stage.draw(camera, time, width, height, "rt"); !drawn) return report(drawn.error());
            pathSeconds += seconds(passBegan);
            const uint32_t has = stage.pathAccumulated();
            const bool whole = stage.pathConverged();
            // The last look waits for the frame to be whole (and denoised, where that was asked).
            while (next < levels.size() &&
                   (next + 1 == levels.size() ? whole : has >= levels[next].second)) {
                const auto lookBegan = std::chrono::steady_clock::now();
                if (const int code = takeLook(camera, name, time, levels[next].first, has, next + 1 == levels.size(),
                                              firstSnapshot);
                    code != 0) {
                    return code;
                }
                lookSeconds += seconds(lookBegan);
                if (next == 0) {
                    firstAt = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
                }
                ++next;
            }
            if (whole) {
                break;
            }
        }
        const auto groupsBegan = std::chrono::steady_clock::now();
        // THE GROUPS' SHARE OF THIS FRAME, KEPT (--light-groups): the frame is whole, and each group's plane of
        // it is read at every gaussian the view shows, as a pass of the group alone would read its frame --
        // of the views that show a gaussian, the nearest one's word stands.
        if (!withinNames.empty()) {
            auto depthOf = stage.displaySource("depth");
            if (!depthOf) return report(depthOf.error());
            const render::Projection projection = render::projectionFor(camera, width, height);
            for (const std::string& group : withinNames) {
                auto planeOf = stage.displaySource("lightGroup:" + group);
                if (!planeOf) return report(planeOf.error());
                if (planeOf->buffer == nullptr || depthOf->buffer == nullptr) {
                    err("the render left no plane for the light group %s\n", group.c_str());
                    return kExitFailure;
                }
                // Every gaussian still there is asked, by its piece's record of what is gone: the finest, and the
                // coarser ones finer looks were laid over, which are what a level of detail draws -- in the
                // pieces on the device and in those that hold nothing finest and are read for this alone.
                const auto keepFor = [&](const io::NeuralScenePiece& piece, scene::GpuSplats& cloud, const gpu::Buffer* known) -> int {
                    if (piece.cloud.size() < 5 || piece.cloud.compare(piece.cloud.size() - 5, 5, ".athg") != 0 || cloud.count == 0) {
                        return 0;
                    }
                    gpu::Buffer record;
                    if (known != nullptr && known->valid()) {
                        record = *known;
                    } else {
                        std::vector<uint32_t> words;
                        if (!piece.gone.empty()) {
                            auto read = io::readSeen(directory / piece.gone, cloud.count);
                            if (!read) return report(read.error());
                            words = std::move(*read);
                        }
                        auto made = discovery->record(cloud.count, words);
                        if (!made) return report(made.error());
                        record = *made;
                    }
                    const std::string key = group + "\t" + piece.cloud;
                    auto whole = groupLayers.find(key);
                    if (whole != groupLayers.end() && groupCounts[key] != cloud.count) {
                        err("%s: the piece was of %u gaussians when the group %s was first kept for it, and is of %u\n",
                            piece.cloud.c_str(), groupCounts[key], group.c_str(), cloud.count);
                        return kExitFailure;
                    }
                    uint32_t took = 0;
                    auto layer = discovery->relight(cloud, *planeOf->buffer, *depthOf->buffer, width, height, projection,
                                                    all.tolerance, took, &record, true,
                                                    whole != groupLayers.end() ? &whole->second : nullptr, planeOf->offset,
                                                    nullptr, 0, nullptr, scene.spread);
                    if (!layer) return report(layer.error());
                    if (whole == groupLayers.end()) {
                        groupLayers.emplace(key, *layer);
                        groupCounts[key] = cloud.count;
                    }
                    return 0;
                };
                for (size_t k = 0; k < clouds.size(); ++k) {
                    if (const int code = keepFor(scene.pieces[pieceOf[k]], *clouds[k], k < gone.size() ? &gone[k] : nullptr);
                        code != 0) {
                        return code;
                    }
                }
                for (size_t k = 0; k < scene.pieces.size(); ++k) {
                    const io::NeuralScenePiece& piece = scene.pieces[k];
                    if (std::find(pieceOf.begin(), pieceOf.end(), k) != pieceOf.end() ||
                        !boxInView(piece.boundsMin, piece.boundsMax, projection, width, height) || piece.cloud.size() < 5 ||
                        piece.cloud.compare(piece.cloud.size() - 5, 5, ".athg") != 0) {
                        continue;
                    }
                    auto held = groupCoarse.find(k);
                    if (held == groupCoarse.end()) {
                        auto cloud = load(library, *loader, directory / piece.cloud);
                        if (!cloud) return report(cloud.error());
                        held = groupCoarse.emplace(k, std::make_unique<scene::GpuSplats>(std::move(*cloud))).first;
                    }
                    if (const int code = keepFor(piece, *held->second, nullptr); code != 0) return code;
                }
            }
        }
        // AND WHAT THIS VIEW FOUND JOINS IT: each piece on the device not yet in the cells, its own light and
        // its groups' layers. A piece is added once, as it stands then.
        if (all.sceneCache > 0) {
            if (discovery->cacheSide() == 0) {
                auto box = stage.bounds();
                if (!box) return report(box.error());
                if (box->has_value()) {
                    const auto& b = **box;
                    const float across = std::max({static_cast<float>(b.max[0] - b.min[0]), static_cast<float>(b.max[1] - b.min[1]),
                                                   static_cast<float>(b.max[2] - b.min[2]), 1.0e-6F});
                    const float cell = across * 1.02F / static_cast<float>(all.sceneCache);
                    const std::array<float, 3> low{static_cast<float>(b.min[0]) - across * 0.01F,
                                                   static_cast<float>(b.min[1]) - across * 0.01F,
                                                   static_cast<float>(b.min[2]) - across * 0.01F};
                    if (auto begun = discovery->cacheBegin(low, cell, all.sceneCache,
                                                           1 + static_cast<uint32_t>(withinNames.size()));
                        !begun) {
                        return report(begun.error());
                    }
                    out("scene cache: %u cells an axis of %g, from (%g, %g, %g)\n", all.sceneCache, static_cast<double>(cell),
                        static_cast<double>(low[0]), static_cast<double>(low[1]), static_cast<double>(low[2]));
                }
            }
            for (size_t k = 0; k < clouds.size() && discovery->cacheSide() != 0; ++k) {
                const io::NeuralScenePiece& piece = scene.pieces[pieceOf[k]];
                if (!cachedPieces.insert(piece.cloud).second) {
                    continue;
                }
                if (auto added = discovery->cacheAdd(*clouds[k], nullptr, 0); !added) return report(added.error());
                for (size_t g = 0; g < withinNames.size(); ++g) {
                    const auto layer = groupLayers.find(withinNames[g] + "\t" + piece.cloud);
                    if (layer != groupLayers.end()) {
                        if (auto added = discovery->cacheAdd(*clouds[k], &layer->second, static_cast<uint32_t>(g) + 1); !added) {
                            return report(added.error());
                        }
                    }
                }
                ++cacheHolds;
            }
        }
        uint32_t standing = 0;
        for (const io::NeuralScenePiece& piece : scene.pieces) {
            standing += piece.gaussians - std::min(piece.removed + piece.refinedCount, piece.gaussians);
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
        const double groupSeconds = seconds(groupsBegan);
        frame.taken = clock();
        scene.frames.push_back(frame);
        const auto writeBegan = std::chrono::steady_clock::now();
        if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
        if (auto grownNow = growBy(frame, static_cast<uint32_t>(scene.frames.size()), true); !grownNow) {
            return report(grownNow.error());
        }
        const double writeSeconds = seconds(writeBegan);
        out("%s at %g: frame %zu, the scene covered %.1f%% of it before a path was traced (%zu looks taken); a scene to "
            "draw after %.1f s and the whole look after %.1f s; %u gaussians stand in %zu pieces, %zu snapshots; "
            "for this view %u pieces were read, %zu were on the device already and %u stayed on the disk "
            "(the scene brought and its cover drawn %.1f s, the frame's passes %.1f, its looks %.1f, the groups' light kept "
            "%.1f, the scene written %.1f)\n",
            name.c_str(), time, scene.frames.size(), 100.0 * covered, levels.size(), covered >= all.coveredEnough ? 0.0 : firstAt,
            frame.seconds, standing, scene.pieces.size(), scene.snapshots.size(), read, kept, left, beforeSeconds, pathSeconds,
            lookSeconds, groupSeconds, writeSeconds);
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
            // A SEQUENCE: with --time-step each view of the file is that much later than the last -- a
            // camera moving through a stage that moves, a frame a line.
            double at = time;
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
                if (const int code = shootView(camera, name, at); code != 0) {
                    return code;
                }
                at += all.timeStep;
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
    if (!all.lightGroup.empty() || !withinNames.empty()) {
        // THE GROUPS' LAYERS, WRITTEN: a file a piece, of the gaussians that send any of the group's light --
        // which one, and the light. What an earlier run kept of the same group for the piece is replaced.
        std::map<std::string, uint64_t> lit;
        for (const auto& [key, layer] : groupLayers) {
            const std::string group = key.substr(0, key.find('\t'));
            const std::string cloud = key.substr(key.find('\t') + 1);
            auto read = layer.readAll<float>(device);
            if (!read) return report(read.error());
            const size_t floats = std::min(read->size(), size_t{groupCounts[key]} * 4);
            const std::vector<float> whole(read->begin(), read->begin() + static_cast<std::ptrdiff_t>(floats));
            std::vector<float> said;
            for (size_t g = 0; g * 4 < whole.size(); ++g) {
                const float* one = &whole[g * 4];
                if (one[3] > 0.0F && (one[0] > 0.0F || one[1] > 0.0F || one[2] > 0.0F)) {
                    said.insert(said.end(), {static_cast<float>(g), one[0], one[1], one[2]});
                }
            }
            const std::filesystem::path file = directory / (cloud + ".g." + group + ".1");
            const std::string stem = file.filename().string();
            const std::string prefix = stem.substr(0, stem.size() - 1);
            std::error_code ignored;
            std::vector<std::filesystem::path> old;
            for (const auto& entry : std::filesystem::directory_iterator(file.parent_path(), ignored)) {
                const std::string name = entry.path().filename().string();
                if (name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0 &&
                    name.find_first_not_of("0123456789", prefix.size()) == std::string::npos) {
                    old.push_back(entry.path());
                }
            }
            for (const std::filesystem::path& gone : old) {
                std::filesystem::remove(gone, ignored);
            }
            if (said.empty()) {
                continue;
            }
            if (auto written = io::writeGaussians(file, static_cast<uint32_t>(said.size() / 4), 4, said); !written) {
                return report(written.error());
            }
            lit[group] += said.size() / 4;
        }
        std::vector<std::string> kept = withinNames;
        if (!all.lightGroup.empty()) kept.push_back(all.lightGroup);
        for (const std::string& group : kept) {
            if (std::find(scene.lightGroups.begin(), scene.lightGroups.end(), group) == scene.lightGroups.end()) {
                scene.lightGroups.push_back(group);
            }
            out("group %s: its light kept for %llu gaussians\n", group.c_str(), static_cast<unsigned long long>(lit[group]));
        }
        if (auto written = io::writeNeuralScene(directory, scene); !written) return report(written.error());
    }
    // The file that grew is whole: its index says no more is coming.
    if (!all.grow.empty()) {
        if (auto closed = writeExportIndex(all.grow, scene, grown, grownLow, grownHigh, false); !closed) {
            return report(closed.error());
        }
    }
    return 0;
}

/// THE STORED FRAMES MADE PICTURES, the scene read once. Frames FIRST to LAST of the manifest's, each
/// through its camera at its size from the scene as it stood at its last look, written <stem>_NNNN.<ext>.
/// A piece is read the first time a frame wants it and kept; what a frame draws of it is the piece less
/// what had gone by then, which is a kernel over what is already on the device.
/// The levels `--group` names, for the pieces this command loads from here on: `tool` adds them.
bool setGroups(const SceneOptions& o, render::Discovery& tool) {
    lightTool = &tool;
    groupGain.clear();
    groupDefault = {1.0F, 1.0F, 1.0F};
    for (const std::string& said : o.groups) {
        const size_t equals = said.find('=');
        std::array<float, 3> gain{1.0F, 1.0F, 1.0F};
        bool good = equals != std::string::npos && equals > 0;
        if (good) {
            std::string numbers = said.substr(equals + 1);
            std::replace(numbers.begin(), numbers.end(), ',', ' ');
            std::istringstream words(numbers);
            std::vector<float> n;
            for (float v = 0.0F; words >> v;) n.push_back(v);
            if (n.size() == 1) {
                gain = {n[0], n[0], n[0]};
            } else if (n.size() == 3) {
                gain = {n[0], n[1], n[2]};
            } else {
                good = false;
            }
        }
        if (!good) {
            err("--group wants NAME=LEVEL or NAME=R,G,B, not %s\n", said.c_str());
            return false;
        }
        const std::string name = said.substr(0, equals);
        if (name == "*") {
            groupDefault = gain;
        } else {
            groupGain[name] = gain;
        }
    }
    return true;
}

int runRenderStored(const SceneOptions& o, const io::NeuralScene& scene) {
    const std::filesystem::path directory = o.directory;
    const uint32_t first = o.storedFrames[0];
    const uint32_t lastFrame = std::min<uint32_t>(o.storedFrames[1], static_cast<uint32_t>(scene.frames.size()));
    if (first == 0 || first > lastFrame || o.output.empty()) {
        err("--frames wants FIRST LAST within the scene's %zu stored frames, and -o a prefix (.png or .exr)\n",
            scene.frames.size());
        return kExitFailure;
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
    if (!setGroups(o, *discovery)) return kExitFailure;
    std::optional<render::SceneGlass> glass;
    if (o.glass) {
        auto made = render::SceneGlass::create(library);
        if (!made) return report(made.error());
        glass.emplace(std::move(*made));
    }
    struct Held {
        std::unique_ptr<scene::GpuSplats> cloud;
        gpu::Buffer                       gone;      // its record, where it has one
        gpu::Buffer                       refined;   // its record of what finer looks were laid over
        gpu::Buffer                       parent;    // the width of what each was laid over
        uint32_t                          light = 0; // the last layer of light laid over it as it was read
    };
    std::vector<Held> held(scene.pieces.size());
    // The snapshot a coarse look's piece was wholly replaced by: its frame's last.
    const auto endsAt = [&](const io::NeuralScenePiece& piece) {
        for (const io::NeuralSceneFrame& f : scene.frames) {
            if (piece.foundIn >= f.firstSnapshot && piece.foundIn <= f.lastSnapshot) {
                return f.lastSnapshot;
            }
        }
        return 0xFFFFFFFFu;
    };
    const std::filesystem::path given = o.output;
    const std::string stem = (given.parent_path() / given.stem()).string();
    const std::string ext = given.extension().string().empty() ? ".exr" : given.extension().string();
    const auto began = std::chrono::steady_clock::now();
    double drawMs = 0.0;
    uint32_t read = 0;
    for (uint32_t n = first; n <= lastFrame; ++n) {
        const io::NeuralSceneFrame& frame = scene.frames[n - 1];
        SceneOptions at = o;
        at.camera = frame.camera;
        at.eye.clear();
        at.target.clear();
        const auto view = viewOf(at, scene);
        if (!view) return kExitFailure;
        const render::Projection projection = render::projectionFor(*view, frame.width, frame.height);
        std::vector<scene::GpuSplats> drawn;
        drawn.reserve(scene.pieces.size() + 1);
        std::vector<render::SplatInstance> instances;
        if (glass) {
            // With glass the frame is the scene as it stood then, cut as one cloud, its glass seen from the
            // frame's camera (read again a frame: the cut owns what it loads).
            lightAsOf = frame.lastSnapshot;
            auto cut = cutScene(library, *loader, *discovery, directory, scene, projection, frame.width, frame.height,
                                o.lod, frame.lastSnapshot, o.morph, &*glass);
            if (!cut) return report(cut.error());
            if (cut->count != 0) {
                drawn.push_back(std::move(*cut));
                instances.push_back(render::SplatInstance{&drawn.back(), render::Mat4::identity(), {}});
            }
        }
        for (size_t k = 0; !glass && k < scene.pieces.size(); ++k) {
            const io::NeuralScenePiece& piece = scene.pieces[k];
            if (piece.foundIn > frame.lastSnapshot) {
                continue;
            }
            if (o.lod <= 0.0F && piece.removed + piece.refinedCount >= piece.gaussians && piece.gaussians != 0 &&
                piece.footprint > 1 && endsAt(piece) <= frame.lastSnapshot) {
                continue;   // a coarse look its own frame refined: drawn flat, nothing of it is the finest
            }
            if (piece.origin == "discovered" &&
                !boxInView(piece.boundsMin, piece.boundsMax, projection, frame.width, frame.height)) {
                continue;
            }
            // A piece whose light was taken again is read as its light stood at this frame: again, where a
            // layer came in since the frame it was last read for.
            uint32_t lit = 0;
            for (const uint32_t layer : piece.lights) {
                if (layer <= frame.lastSnapshot) lit = layer;
            }
            if (held[k].cloud && lit != held[k].light) {
                held[k] = Held{};
            }
            if (!held[k].cloud) {
                lightAsOf = frame.lastSnapshot;
                auto cloud = load(library, *loader, directory / piece.cloud, &held[k].parent);
                if (!cloud) return report(cloud.error());
                held[k].cloud = std::make_unique<scene::GpuSplats>(std::move(*cloud));
                held[k].light = lit;
                for (int which = 0; which < 2; ++which) {
                    const std::string& file = which == 0 ? piece.gone : piece.refined;
                    if (file.empty()) {
                        continue;
                    }
                    auto words = io::readSeen(directory / file, held[k].cloud->count);
                    if (!words) return report(words.error());
                    auto record = discovery->record(held[k].cloud->count, *words);
                    if (!record) return report(record.error());
                    (which == 0 ? held[k].gone : held[k].refined) = *record;
                }
                ++read;
            }
            if (held[k].gone.valid() || held[k].refined.valid() || o.lod > 0.0F) {
                // The piece as the frame's look knew it, cut for the frame's view: a kernel over what is here.
                auto then = discovery->cut(*held[k].cloud, held[k].gone.valid() ? &held[k].gone : nullptr,
                                           held[k].refined.valid() ? &held[k].refined : nullptr,
                                           held[k].parent.valid() ? &held[k].parent : nullptr, projection, o.lod,
                                           frame.lastSnapshot, o.lod > 0.0F, o.morph);
                if (!then) return report(then.error());
                if (then->count == 0) {
                    continue;
                }
                drawn.push_back(std::move(*then));
            } else {
                drawn.push_back(*held[k].cloud);
            }
            instances.push_back(render::SplatInstance{&drawn.back(), render::Mat4::identity(), {}});
        }
        render::RenderSettings settings;
        settings.madeSpread = madeAs;
        settings.width = frame.width;
        settings.height = frame.height;
        settings.timeStages = true;
        render::RenderTargets targets;
        auto stats = raster->render(*view, instances, settings, targets);
        if (!stats) return report(stats.error());
        drawMs += stats->totalMs;
        char name[32];
        std::snprintf(name, sizeof(name), "_%04u", n);
        if (auto w = writeFrame(library, **device, targets, frame.width, frame.height, stem + name + ext, o.exposure); !w) {
            return report(w.error());
        }
    }
    const uint32_t frames = lastFrame - first + 1;
    out("%s: %u stored frames written as %s_NNNN%s in %.1f s, %.2f s a frame of which %.1f ms drawing; %u pieces read\n",
        scene.name.c_str(), frames, stem.c_str(), ext.c_str(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count(),
        std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count() / frames, drawMs / frames, read);
    return 0;
}

/// THE SCENE AS IT STOOD, FROZEN INTO A .ATHC: every gaussian standing at a snapshot, one cloud, with the
/// engine's own levels of detail built over it (lod::LodBuilder: Morton order, a merged gaussian a cell a
/// level) -- what `athenea render --lod`, a stage and a stream draw by how large a cell is on the screen.
/// It is what the scene is handed on as, not what it is kept as: a .athc has no room for when a gaussian
/// came or went, nor for what it is made of beyond its shading normal.
int runBake(const SceneOptions& o) {
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    drawAs(*scene);
    if (o.output.empty() || !lod::isAthc(o.output)) {
        err("neural-scene bake wants -o a .athc\n");
        return kExitFailure;
    }
    const uint32_t asOf = o.snapshot != 0 ? o.snapshot : o.at >= 0.0 ? snapshotAt(*scene, o.at) : 0u;
    lightAsOf = asOf;   // a piece's light as it was said up to then
    // The standing records, piece after piece: which are kept is read off each piece's stamps; nothing
    // of a record is computed.
    std::vector<float> records;
    uint32_t count = 0;
    uint32_t pieces = 0;
    uint32_t floats = render::kRenderRecord;
    for (const io::NeuralScenePiece& piece : scene->pieces) {
        if (asOf != 0 && piece.foundIn > asOf) {
            continue;
        }
        if (asOf == 0 && piece.removed != 0 && piece.removed >= piece.gaussians) {
            continue;
        }
        if (std::filesystem::path(piece.cloud).extension() != ".athg") {
            err("piece %s is not a render's (.athg): neural-scene bake freezes what neural-scene shoot found\n",
                piece.id.c_str());
            return kExitFailure;
        }
        auto held = io::readGaussians(directory / piece.cloud);
        if (!held) return report(held.error());
        if (!renderRecordOf(held->floats)) {
            err("piece %s: records of %u floats\n", piece.id.c_str(), held->floats);
            return kExitFailure;
        }
        floats = held->floats;
        std::vector<uint32_t> gone;
        if (!piece.gone.empty()) {
            auto words = io::readSeen(directory / piece.gone, held->count);
            if (!words) return report(words.error());
            gone = std::move(*words);
        }
        // The finest there is: the builder makes the levels above it itself.
        std::vector<uint32_t> refined;
        if (!piece.refined.empty()) {
            auto words = io::readSeen(directory / piece.refined, held->count);
            if (!words) return report(words.error());
            refined = std::move(*words);
        }
        uint32_t kept = 0;
        for (uint32_t k = 0; k < held->count; ++k) {
            const uint32_t went = gone.empty() ? 0u : gone[k];
            if (went != 0 && (asOf == 0 || went <= asOf)) {
                continue;
            }
            const uint32_t over = refined.empty() ? 0u : refined[k];
            if (over != 0 && (asOf == 0 || over <= asOf)) {
                continue;
            }
            const auto first = held->records.begin() + static_cast<std::ptrdiff_t>(size_t{k} * held->floats);
            records.insert(records.end(), first, first + held->floats);
            ++kept;
        }
        count += kept;
        pieces += kept != 0 ? 1 : 0;
    }
    if (count == 0) {
        err("nothing stands in the scene at that snapshot\n");
        return kExitFailure;
    }
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    io::RawSplats raw;
    raw.source = directory.string();
    raw.count = count;
    raw.encoding = render::Discovery::renderEncoding();
    raw.encoding.floatsPerRecord = floats;
    raw.records = std::move(records);
    auto splats = loader->upload(raw, 0);
    if (!splats) return report(splats.error());
    const auto began = std::chrono::steady_clock::now();
    auto builder = lod::LodBuilder::create(library);
    if (!builder) return report(builder.error());
    auto built = builder->build(*splats);
    if (!built) return report(built.error());
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    auto written = platform::writeAtomically(o.output, [&](const std::filesystem::path& partial) {
        return lod::writeAthc(library.device(), *built, partial);
    });
    if (!written) return report(written.error());
    std::error_code ignored;
    out("%s: %u gaussians standing in %u pieces, built into %zu levels of detail in %.2f s; wrote %s (%.0f MB)\n",
        scene->name.c_str(), splats->count, pieces, built->levels.size(), seconds, o.output.c_str(),
        static_cast<double>(std::filesystem::file_size(o.output, ignored)) / 1048576.0);
    return 0;
}

/// The records of a piece that are the finest standing at look `asOf` (0: now), appended to `records`: which
/// are kept is read off the piece's two stamp files. How many were.
Result<uint32_t> standingRecords(const std::filesystem::path& directory, const io::NeuralScenePiece& piece,
                                 uint32_t asOf, uint32_t& floats, std::vector<float>& records, LensRecords* lenses) {
    if (std::filesystem::path(piece.cloud).extension() != ".athg") {
        return Error::make(ErrorCode::Unsupported, "piece {} is not a render's (.athg)", piece.id);
    }
    auto read = io::readGaussians(directory / piece.cloud);
    if (!read) return std::move(read).error();
    if (!renderRecordOf(read->floats) || (floats != 0 && read->floats != floats)) {
        return Error::make(ErrorCode::Unsupported, "piece {}: records of {} floats among records of {}", piece.id,
                           read->floats, floats);
    }
    floats = read->floats;
    std::vector<uint32_t> stamps[2];
    const std::string* files[2] = {&piece.gone, &piece.refined};
    for (int k = 0; k < 2; ++k) {
        if (!files[k]->empty()) {
            auto words = io::readSeen(directory / *files[k], read->count);
            if (!words) return std::move(words).error();
            stamps[k] = std::move(*words);
        }
    }
    uint32_t kept = 0;
    for (uint32_t g = 0; g < read->count; ++g) {
        bool stands = true;
        for (int k = 0; k < 2; ++k) {
            const uint32_t stamp = stamps[k].empty() ? 0u : stamps[k][g];
            stands = stands && !(stamp != 0 && (asOf == 0 || stamp <= asOf));
        }
        if (!stands) {
            continue;
        }
        const auto first = read->records.begin() + static_cast<std::ptrdiff_t>(size_t{g} * floats);
        // A lens's gaussians are set apart where the caller asks (which records they are is read off three
        // of their fields, as a stamp is): a viewer shades them by what they are made of.
        if (lenses != nullptr && floats >= render::kRenderRecordTint + 3 && first[19] >= 0.5F &&
            first[render::kRenderRecordSheet] <= 0.5F && first[render::kRenderRecordIndex] > 1.0F) {
            if (lenses->count == 0) {
                lenses->ior = first[render::kRenderRecordIndex];
                lenses->tint = {first[render::kRenderRecordTint], first[render::kRenderRecordTint + 1],
                                first[render::kRenderRecordTint + 2]};
            }
            lenses->records.insert(lenses->records.end(), first, first + floats);
            ++lenses->count;
            continue;
        }
        records.insert(records.end(), first, first + floats);
        ++kept;
    }
    return kept;
}

/// THE SCENE HANDED ON AS STANDARD SPLAT FILES: the gaussians that are the finest standing at a snapshot,
/// as SPZ -- what a viewer on the web reads -- with an index naming the parts (<stem>.json). The parts are
/// runs of no more than --part gaussians; or, with --by-frame, a part a rendered frame, each holding what
/// that frame found and still stands -- the scene in the order it grew, for a viewer to play back.
int runExport(const SceneOptions& o) {
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    drawAs(*scene);
    if (o.output.empty()) {
        err("neural-scene export wants -o a path with no extension: <stem>_NNN.spz and <stem>.json\n");
        return kExitFailure;
    }
    const uint32_t asOf = o.snapshot != 0 ? o.snapshot : o.at >= 0.0 ? snapshotAt(*scene, o.at) : 0u;
    lightAsOf = asOf;   // a piece's light as it was said up to then
    auto device = gpu::Device::create();
    if (!device) return report(device.error());
    gpu::ShaderLibrary library(*device);
    auto discovery = render::Discovery::create(library);
    if (!discovery) return report(discovery.error());
    // The groups' levels are laid over the records handed on (takePiece), not over the clouds the cut reads.
    if (!setGroups(o, *discovery)) return kExitFailure;
    lightTool = nullptr;
    auto pack = render::FlattenPack::create(library);
    if (!pack) return report(pack.error());
    const std::filesystem::path given = o.output;
    std::error_code made;
    std::filesystem::create_directories(given.parent_path(), made);
    auto loader = scene::CloudLoader::create(library);
    if (!loader) return report(loader.error());
    // WHAT A PIECE HANDS ON AT A LEVEL, BY KIND. The piece's records go to the device once; the cut -- asked
    // in the scene's own units, `width` of them (0: the finest there is) -- says which are drawn, and the
    // device sets those apart as lenses, mirrors and the rest (Discovery::recordsOf). What comes back is
    // laid end to end with the other pieces' for a part's file: copied, not computed.
    std::vector<float> records;
    LensRecords lenses;
    LensRecords mirrors;
    uint32_t held = 0;
    uint32_t floats = 0;
    uint64_t bytes = 0;
    // WHICH PRIM EACH GAUSSIAN HANDED ON WAS MADE FROM (a piece's `.prims`, the manifest's `prims`): a place
    // in the scene's prims a gaussian, counted from one in ascending order of their ids -- the order the
    // index names them in -- carried through the same sets the records are (Discovery::primIndices), so a
    // part's ids are its file's gaussians in its file's order: its glass's and its mirrors' their own, a
    // level's, a history's. Sixteen bits a gaussian, which is why a scene of more prims than that hands none.
    std::vector<std::pair<uint32_t, std::string>> primList;
    for (const auto& [path, id] : scene->prims) {
        primList.emplace_back(id, path);
    }
    std::sort(primList.begin(), primList.end());
    if (primList.size() > 65535 && !o.noPrims) {
        out("the scene names %zu prims, more than a part's ids hold (65535): the parts are written without\n", primList.size());
    }
    const bool namesPrims = !o.noPrims && !primList.empty() && primList.size() <= 65535;
    const uint32_t named = namesPrims ? static_cast<uint32_t>(primList.size()) : 0u;
    gpu::Buffer primTable;
    if (namesPrims) {
        std::vector<uint32_t> ascending;
        for (const auto& [id, path] : primList) {
            ascending.push_back(id);
        }
        ascending.resize(std::max<size_t>(ascending.size(), 4), 0xFFFFFFFFu);
        auto table = gpu::Buffer::fromSpan<uint32_t>(**device, std::span<const uint32_t>(ascending), "export.prims");
        if (!table) return report(table.error());
        primTable = *table;
    }
    std::vector<uint32_t> recordPrims;
    std::vector<uint32_t> lensPrims;
    std::vector<uint32_t> mirrorPrims;
    uint64_t idBytes = 0;
    // A piece's ids on the device, where it keeps any (a piece found before a scene kept them names none).
    const auto primsOf = [&](const io::NeuralScenePiece& piece, uint32_t count) -> Result<gpu::Buffer> {
        if (!namesPrims || piece.prims.empty() || count == 0) {
            return gpu::Buffer{};
        }
        auto ids = io::readSeen(directory / piece.prims, count);
        if (!ids) return std::move(ids).error();
        return gpu::Buffer::fromSpan<uint32_t>(**device, std::span<const uint32_t>(*ids), "export.piece.prims");
    };
    const render::Projection units = render::Discovery::worldWidths();
    const auto gatherKinds = [&](const gpu::Buffer& onDevice, uint32_t count, const gpu::Buffer& flags,
                                 const gpu::Buffer& prims) -> Result<uint32_t> {
        uint32_t matter = 0;
        const std::tuple<render::Discovery::RecordKind, std::vector<float>*, std::vector<uint32_t>*> kinds[3] = {
            {render::Discovery::RecordKind::Matter, &records, &recordPrims},
            {render::Discovery::RecordKind::Lens, &lenses.records, &lensPrims},
            {render::Discovery::RecordKind::Mirror, &mirrors.records, &mirrorPrims}};
        for (const auto& [kind, into, primsInto] : kinds) {
            auto set = discovery->recordsOf(onDevice, count, floats, flags, kind);
            if (!set) return std::move(set).error();
            if (set->count == 0) {
                continue;
            }
            if (namesPrims) {
                if (prims.valid()) {
                    auto indices = discovery->primIndices(prims, count, *set, primTable, named);
                    if (!indices) return std::move(indices).error();
                    auto read = indices->readAll<uint32_t>(**device);
                    if (!read) return std::move(read).error();
                    read->resize(set->count);
                    primsInto->insert(primsInto->end(), read->begin(), read->end());
                } else {
                    primsInto->insert(primsInto->end(), set->count, 0u);
                }
            }
            auto back = set->records.readAll<float>(**device);
            if (!back) return std::move(back).error();
            back->resize(size_t{set->count} * floats);
            if (kind == render::Discovery::RecordKind::Lens) {
                if (lenses.count == 0 && floats >= render::kRenderRecordTint + 3) {
                    lenses.ior = (*back)[render::kRenderRecordIndex];
                    lenses.tint = {(*back)[render::kRenderRecordTint], (*back)[render::kRenderRecordTint + 1],
                                   (*back)[render::kRenderRecordTint + 2]};
                }
                lenses.count += set->count;
            } else if (kind == render::Discovery::RecordKind::Mirror) {
                mirrors.count += set->count;
            } else {
                matter = set->count;
            }
            into->insert(into->end(), back->begin(), back->end());
        }
        return matter;
    };
    // THE PIECE'S LIGHT AS IT IS HANDED ON: the layers of light it took again laid over its records, and its
    // groups of lights added at their levels, on the device (`--group`, `--own-light`, `--base-colour`). One
    // place for it: a piece handed on whole, and one handed on by what of it stood until a frame (--history),
    // are lit the same.
    const auto litOnDevice = [&](const io::NeuralScenePiece& piece, io::GaussianRecords& held) -> Result<gpu::Buffer> {
        io::GaussianRecords* read = &held;
        if (auto laid = layLightOver(directory / piece.cloud, read->records, read->floats, read->count); !laid) {
            return std::move(laid).error();
        }
        auto onDevice = gpu::Buffer::fromSpan<float>(**device, std::span<const float>(read->records), "export.piece");
        if (!onDevice) return std::move(onDevice).error();
        auto groups = groupLayersOf(directory / piece.cloud, read->count);
        if (!groups) return std::move(groups).error();
        float own = o.ownLight;
        for (const auto& [dense, gain] : *groups) {
            auto layer = gpu::Buffer::fromSpan<std::array<float, 4>>(
                **device, std::span<const std::array<float, 4>>(dense), "export.group");
            if (!layer) return std::move(layer).error();
            ATHENEA_TRY(discovery->addLightToRecords(*onDevice, read->count, read->floats, *layer, gain, own));
            own = 1.0F;
        }
        if (o.baseColour) {
            // THE BASE COLOUR IN PLACE OF THE LIGHT (--base-colour): what a viewer with a lamp of its own
            // multiplies that lamp's light by, the same gaussians in the same order as the scene's.
            ATHENEA_TRY(discovery->baseColourOfRecords(*onDevice, read->count, read->floats));
            own = 1.0F;
        }
        if (own != 1.0F) {
            // No group drawn and the scene's own light scaled: the same kernel with nothing added.
            const std::vector<std::array<float, 4>> none(read->count, std::array<float, 4>{0.0F, 0.0F, 0.0F, 0.0F});
            auto layer = gpu::Buffer::fromSpan<std::array<float, 4>>(
                **device, std::span<const std::array<float, 4>>(none), "export.group");
            if (!layer) return std::move(layer).error();
            ATHENEA_TRY(discovery->addLightToRecords(*onDevice, read->count, read->floats, *layer, {0.0F, 0.0F, 0.0F}, own));
        }
        return *onDevice;
    };
    const auto takePiece = [&](const io::NeuralScenePiece& piece, float width) -> Result<uint32_t> {
        if (std::filesystem::path(piece.cloud).extension() != ".athg") {
            return Error::make(ErrorCode::Unsupported, "piece {} is not a render's (.athg)", piece.id);
        }
        auto forCut = loadForCut(library, *loader, *discovery, directory, piece);
        if (!forCut) return std::move(forCut).error();
        auto read = io::readGaussians(directory / piece.cloud);
        if (!read) return std::move(read).error();
        if (!renderRecordOf(read->floats) || (floats != 0 && read->floats != floats)) {
            return Error::make(ErrorCode::Unsupported, "piece {}: records of {} floats among records of {}", piece.id,
                               read->floats, floats);
        }
        floats = read->floats;
        auto plan = discovery->planCut(forCut->cloud, forCut->gone.valid() ? &forCut->gone : nullptr,
                                       forCut->refined.valid() ? &forCut->refined : nullptr,
                                       forCut->parent.valid() ? &forCut->parent : nullptr, units, width, asOf);
        if (!plan) return std::move(plan).error();
        if (plan->drawn == 0) {
            return uint32_t{0};
        }
        auto lit = litOnDevice(piece, *read);
        if (!lit) return std::move(lit).error();
        gpu::Buffer* onDevice = &*lit;
        auto prims = primsOf(piece, read->count);
        if (!prims) return std::move(prims).error();
        return gatherKinds(*onDevice, read->count, plan->flags, *prims);
    };
    // A piece's gaussians by the frame from which each no longer stands (0: it stands to the end): read off
    // its two records of stamps, which are looks, and the frames' own first and last looks. Bookkeeping: no
    // gaussian is read, only when each went.
    const auto frameOfLook = [&](uint32_t look) {
        for (size_t f = 0; f < scene->frames.size(); ++f) {
            if (look <= scene->frames[f].lastSnapshot) {
                return static_cast<uint32_t>(f) + 1;
            }
        }
        return static_cast<uint32_t>(scene->frames.size());
    };
    const auto untilOf = [&](const io::NeuralScenePiece& piece, uint32_t count) -> Result<std::vector<uint32_t>> {
        std::vector<uint32_t> until(count, 0u);
        for (const std::string* file : {&piece.gone, &piece.refined}) {
            if (file->empty()) {
                continue;
            }
            auto stamps = io::readSeen(directory / *file, count);
            if (!stamps) return std::move(stamps).error();
            for (uint32_t g = 0; g < count; ++g) {
                if ((*stamps)[g] != 0) {
                    const uint32_t frame = frameOfLook((*stamps)[g]);
                    until[g] = until[g] == 0 ? frame : std::min(until[g], frame);
                }
            }
        }
        return until;
    };
    // The records of a piece that a list of flags keeps, by kind, into what is held.
    const auto takeFlagged = [&](const io::NeuralScenePiece& piece, const std::vector<uint32_t>& flags) -> Result<uint32_t> {
        auto read = io::readGaussians(directory / piece.cloud);
        if (!read) return std::move(read).error();
        if (!renderRecordOf(read->floats) || (floats != 0 && read->floats != floats)) {
            return Error::make(ErrorCode::Unsupported, "piece {}: records of {} floats among records of {}", piece.id,
                               read->floats, floats);
        }
        floats = read->floats;
        auto lit = litOnDevice(piece, *read);
        if (!lit) return std::move(lit).error();
        gpu::Buffer* onDevice = &*lit;
        auto drawn = gpu::Buffer::fromSpan<uint32_t>(**device, std::span<const uint32_t>(flags), "export.flags");
        if (!drawn) return std::move(drawn).error();
        auto prims = primsOf(piece, read->count);
        if (!prims) return std::move(prims).error();
        return gatherKinds(*onDevice, read->count, *drawn, *prims);
    };
    std::array<float, 3> low{1.0e30F, 1.0e30F, 1.0e30F};
    std::array<float, 3> high{-1.0e30F, -1.0e30F, -1.0e30F};
    const float gain = std::exp2(o.exposure);
    // What is held, written as a part of `into` under `prefix`: its file, and beside it its glass and its
    // mirrors where it has any.
    const auto flush = [&](std::vector<ExportedPart>& into, const std::string& prefix, const io::NeuralSceneFrame* frame,
                           uint32_t frameNumber) -> Result<void> {
        if (held + lenses.count + mirrors.count == 0) {
            return ok();
        }
        char number[16];
        std::snprintf(number, sizeof(number), "_%03zu", into.size());
        ExportedPart part;
        const auto write = [&](const std::vector<float>& of, uint32_t count, const char* suffix) -> Result<std::string> {
            const std::filesystem::path path = prefix + number + suffix + "." + o.format;
            auto size = writeSpzPart(**device, *discovery, *pack, std::span<const float>(of), count, floats, gain, path,
                                     low, high);
            if (!size) return std::move(size).error();
            bytes += *size;
            return path.filename().string();
        };
        // A file's ids beside it: the indices two a word on the device (Discovery::primWords), and the words
        // written as they are -- sixteen bits a gaussian, little endian -- inside gzip.
        const auto writeIds = [&](const std::vector<uint32_t>& of, uint32_t count, const char* suffix) -> Result<std::string> {
            if (!namesPrims || of.size() != count || count == 0) {
                return std::string{};
            }
            auto indices = gpu::Buffer::fromSpan<uint32_t>(**device, std::span<const uint32_t>(of), "export.ids");
            if (!indices) return std::move(indices).error();
            auto packed = discovery->primWords(*indices, count);
            if (!packed) return std::move(packed).error();
            auto words = packed->readAll<uint32_t>(**device);
            if (!words) return std::move(words).error();
            const std::filesystem::path path = prefix + number + suffix + ".ids";
            // Inside gzip: a prim's gaussians come in runs, and a file of them is a sixtieth of its bytes.
            auto size = io::writeGzipped(path, std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(words->data()),
                                                                        size_t{count} * 2));
            if (!size) return std::move(size).error();
            idBytes += *size;
            return path.filename().string();
        };
        if (held != 0) {
            auto file = write(records, held, "");
            if (!file) return std::move(file).error();
            part.file = *file;
            part.gaussians = held;
            auto ids = writeIds(recordPrims, held, "");
            if (!ids) return std::move(ids).error();
            part.ids = *ids;
        }
        if (lenses.count != 0) {
            auto file = write(lenses.records, lenses.count, "_glass");
            if (!file) return std::move(file).error();
            auto ids = writeIds(lensPrims, lenses.count, "_glass");
            if (!ids) return std::move(ids).error();
            part.glassIds = *ids;
            part.glassFile = *file;
            part.glassGaussians = lenses.count;
            part.glassIor = lenses.ior;
            part.glassTint = lenses.tint;
        }
        if (mirrors.count != 0) {
            auto file = write(mirrors.records, mirrors.count, "_mirror");
            if (!file) return std::move(file).error();
            auto ids = writeIds(mirrorPrims, mirrors.count, "_mirror");
            if (!ids) return std::move(ids).error();
            part.mirrorIds = *ids;
            part.mirrorFile = *file;
            part.mirrorGaussians = mirrors.count;
        }
        if (part.file.empty()) {
            // A part of glass or mirror alone still names a file a viewer reads first: its glass's, or its
            // mirror's, as the part itself.
            if (!part.glassFile.empty()) {
                part.file = part.glassFile;
                part.gaussians = part.glassGaussians;
                part.ids = part.glassIds;
                part.glassIds.clear();
                part.glassFile.clear();
                part.glassGaussians = 0;
            } else {
                part.file = part.mirrorFile;
                part.gaussians = part.mirrorGaussians;
                part.ids = part.mirrorIds;
                part.mirrorIds.clear();
                part.mirrorFile.clear();
                part.mirrorGaussians = 0;
            }
        }
        if (frame != nullptr) {
            part.frame = frameNumber;
            part.time = frame->time;
            part.camera = frame->camera;
            part.seconds = frame->seconds;
        }
        into.push_back(part);
        held = 0;
        records.clear();
        lenses = LensRecords{};
        mirrors = LensRecords{};
        recordPrims.clear();
        lensPrims.clear();
        mirrorPrims.clear();
        return ok();
    };
    const auto wanted = [&](const io::NeuralScenePiece& piece, bool finest) {
        if (asOf != 0 && piece.foundIn > asOf) {
            return false;
        }
        if (piece.gaussians == 0) {
            return true;
        }
        // The finest level draws nothing of a piece finer looks were laid all over; a coarser one may.
        return (finest ? piece.removed + piece.refinedCount : piece.removed) < piece.gaussians;
    };
    std::vector<ExportedPart> parts;
    if (o.byFrame) {
        if (scene->frames.empty()) {
            err("--by-frame wants a scene with stored frames (neural-scene shoot)\n");
            return kExitFailure;
        }
        for (size_t f = 0; f < scene->frames.size(); ++f) {
            const io::NeuralSceneFrame& frame = scene->frames[f];
            const uint32_t number = static_cast<uint32_t>(f) + 1;
            if (!o.history) {
                for (const io::NeuralScenePiece& piece : scene->pieces) {
                    if (!wanted(piece, true) || piece.foundIn < frame.firstSnapshot || piece.foundIn > frame.lastSnapshot) {
                        continue;
                    }
                    auto kept = takePiece(piece, 0.0F);
                    if (!kept) return report(kept.error());
                    held += *kept;
                }
                if (auto done = flush(parts, given.string(), &frame, number); !done) return report(done.error());
                continue;
            }
            // THE SCENE AS IT STOOD (--history). What this frame found is set apart by the frame from which
            // it no longer stands: what stands to the end is the part, as without the option; what a later
            // frame took out -- the thing moved, the light changed -- or laid finer gaussians over goes in
            // files of its own that say until when. What went within its own frame (the coarse looks a
            // frame takes while it gathers) never stood at a frame's end and is not written.
            std::vector<std::pair<const io::NeuralScenePiece*, std::vector<uint32_t>>> found;
            std::vector<uint32_t> ends;
            for (const io::NeuralScenePiece& piece : scene->pieces) {
                if (piece.foundIn < frame.firstSnapshot || piece.foundIn > frame.lastSnapshot || piece.gaussians == 0 ||
                    std::filesystem::path(piece.cloud).extension() != ".athg") {
                    continue;
                }
                auto until = untilOf(piece, piece.gaussians);
                if (!until) return report(until.error());
                for (const uint32_t end : *until) {
                    if (end > number && std::find(ends.begin(), ends.end(), end) == ends.end()) {
                        ends.push_back(end);
                    }
                }
                found.emplace_back(&piece, std::move(*until));
            }
            std::sort(ends.begin(), ends.end());
            const auto takeUntil = [&](uint32_t end) -> Result<void> {
                for (const auto& [piece, until] : found) {
                    std::vector<uint32_t> flags(until.size());
                    bool any = false;
                    for (size_t g = 0; g < until.size(); ++g) {
                        flags[g] = until[g] == end ? 1u : 0u;
                        any = any || flags[g] != 0;
                    }
                    if (!any) {
                        continue;
                    }
                    auto kept = takeFlagged(*piece, flags);
                    if (!kept) return std::move(kept).error();
                    held += *kept;
                }
                return ok();
            };
            if (auto taken = takeUntil(0); !taken) return report(taken.error());
            const size_t before = parts.size();
            if (auto done = flush(parts, given.string(), &frame, number); !done) return report(done.error());
            if (parts.size() == before && !ends.empty()) {
                // Nothing of this frame's stands to the end, and something of it stood for a while: the
                // part is there to carry that.
                ExportedPart empty;
                empty.frame = number;
                empty.time = frame.time;
                empty.camera = frame.camera;
                empty.seconds = frame.seconds;
                parts.push_back(empty);
            }
            // One file for all of it (and one for its glass, one for its mirrors): the spans in the order of
            // the frames they stand until, each span's counts in the index. A file a span was three thousand
            // files for a walk of sixty frames.
            char label[32];
            std::snprintf(label, sizeof(label), "_%03zu_history", parts.empty() ? size_t{0} : parts.size() - 1);
            std::vector<std::array<uint32_t, 4>> spans;
            for (const uint32_t end : ends) {
                const uint32_t had = held;
                const uint32_t hadGlass = lenses.count;
                const uint32_t hadMirror = mirrors.count;
                if (auto taken = takeUntil(end); !taken) return report(taken.error());
                if (held + lenses.count + mirrors.count != had + hadGlass + hadMirror) {
                    spans.push_back({end, held - had, lenses.count - hadGlass, mirrors.count - hadMirror});
                }
            }
            if (!spans.empty()) {
                const bool plain = held != 0;
                const bool glass = lenses.count != 0;
                std::vector<ExportedPart> one;
                if (auto done = flush(one, given.string() + label, nullptr, 0); !done) return report(done.error());
                if (!one.empty()) {
                    if (!plain) {
                        // flush names a part of glass or of mirrors alone as the part itself: its counts with it.
                        for (auto& span : spans) {
                            span[1] = glass ? span[2] : span[3];
                            span[glass ? 2 : 3] = 0;
                        }
                    }
                    one.back().untilFrame = spans.back()[0];
                    one.back().spans = std::move(spans);
                    parts.back().history.push_back(one.back());
                }
            }
        }
    } else {
        for (const io::NeuralScenePiece& piece : scene->pieces) {
            if (!wanted(piece, true)) {
                continue;
            }
            auto kept = takePiece(piece, 0.0F);
            if (!kept) return report(kept.error());
            held += *kept;
            if (held + lenses.count + mirrors.count >= o.part) {
                if (auto done = flush(parts, given.string(), nullptr, 0); !done) return report(done.error());
            }
        }
        if (auto done = flush(parts, given.string(), nullptr, 0); !done) return report(done.error());
    }
    if (parts.empty()) {
        err("nothing stands in the scene at that snapshot\n");
        return kExitFailure;
    }
    uint64_t total = 0;
    uint64_t largest = 0;
    for (const ExportedPart& part : parts) {
        total += part.gaussians + part.glassGaussians + part.mirrorGaussians;
        largest = std::max<uint64_t>(largest, part.gaussians);
    }
    // THE LEVELS OF DETAIL (--levels): the scene whole at each budget, coarser one after another, for a
    // device that cannot draw the finest. A level is the cut at a width in the scene's own units -- a
    // gaussian up to that wide drawn in place of the finer ones laid over it -- and the width is the
    // finest that draws no more than the budget, read off the histogram every piece adds its spans to.
    std::vector<ExportedLevel> levels;
    for (size_t l = 0; l < o.levels.size(); ++l) {
        const uint64_t budget = o.levels[l];
        if (budget >= total) {
            out("a level of %llu gaussians is the scene itself (%llu): not written\n",
                static_cast<unsigned long long>(budget), static_cast<unsigned long long>(total));
            continue;
        }
        auto histogram = discovery->cutHistogram();
        if (!histogram) return report(histogram.error());
        for (const io::NeuralScenePiece& piece : scene->pieces) {
            if (!wanted(piece, false)) {
                continue;
            }
            auto forCut = loadForCut(library, *loader, *discovery, directory, piece);
            if (!forCut) return report(forCut.error());
            if (auto added = discovery->spans(forCut->cloud, forCut->gone.valid() ? &forCut->gone : nullptr,
                                              forCut->refined.valid() ? &forCut->refined : nullptr,
                                              forCut->parent.valid() ? &forCut->parent : nullptr, units, asOf,
                                              *histogram);
                !added) {
                return report(added.error());
            }
        }
        uint64_t said = 0;
        auto width = discovery->thresholdFor(*histogram, budget, &said);
        if (!width) return report(width.error());
        if (said > budget + budget / 4) {
            // Nothing coarser to draw: no look left wider gaussians under these (a scene shot before a
            // gaussian kept what it was laid over, or one every surface of which was seen once).
            out("a level of %llu gaussians cannot be met: the coarsest the scene holds draws %llu; not written\n",
                static_cast<unsigned long long>(budget), static_cast<unsigned long long>(said));
            continue;
        }
        if (!levels.empty() && levels.back().width == *width) {
            // The scene's levels are a few widths, an octave apart: two budgets between the same two of
            // them ask for the same cut.
            out("a level of %llu gaussians is the level before it (a width of %.3f units draws %llu); not written again\n",
                static_cast<unsigned long long>(budget), static_cast<double>(*width), static_cast<unsigned long long>(said));
            continue;
        }
        ExportedLevel level;
        level.budget = budget;
        level.width = *width;
        char name[32];
        std::snprintf(name, sizeof(name), "_lod%zu", levels.size() + 1);
        const std::string prefix = given.string() + name;
        for (const io::NeuralScenePiece& piece : scene->pieces) {
            if (!wanted(piece, false)) {
                continue;
            }
            auto kept = takePiece(piece, *width);
            if (!kept) return report(kept.error());
            held += *kept;
            if (held + lenses.count + mirrors.count >= o.part) {
                if (auto done = flush(level.parts, prefix, nullptr, 0); !done) return report(done.error());
            }
        }
        if (auto done = flush(level.parts, prefix, nullptr, 0); !done) return report(done.error());
        uint64_t drawn = 0;
        for (const ExportedPart& part : level.parts) {
            drawn += part.gaussians + part.glassGaussians + part.mirrorGaussians;
        }
        out("level %zu: a budget of %llu, a width of %.3f units: %llu gaussians in %zu parts\n", levels.size() + 1,
            static_cast<unsigned long long>(budget), static_cast<double>(*width), static_cast<unsigned long long>(drawn),
            level.parts.size());
        levels.push_back(std::move(level));
    }
    if (auto written = writeExportIndex(given, *scene, parts, low, high, false, &levels, namesPrims ? &primList : nullptr);
        !written) {
        return report(written.error());
    }
    if (namesPrims) {
        out("prims: %zu named in the index, an id a gaussian beside each file (.ids), %.2f MB\n", primList.size(),
            static_cast<double>(idBytes) / 1048576.0);
    }
    out("%s: %llu gaussians in %zu parts (the largest %llu), %zu levels of detail, %.1f MB in all; wrote %s.json\n",
        scene->name.c_str(), static_cast<unsigned long long>(total), parts.size(),
        static_cast<unsigned long long>(largest), levels.size(), static_cast<double>(bytes) / 1048576.0,
        given.string().c_str());
    return 0;
}

int runRender(const SceneOptions& given) {
    SceneOptions o = given;
    const std::filesystem::path directory = o.directory;
    auto scene = io::readNeuralScene(directory);
    if (!scene) return report(scene.error());
    drawAs(*scene);
    if (o.storedFrames.size() == 2) {
        if (!o.mattes.empty()) {
            err("--matte is of a view drawn from the scene's pieces, not of --frames\n");
            return kExitFailure;
        }
        return runRenderStored(o, *scene);
    }
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
    if (!setGroups(o, *discovery)) return kExitFailure;
    // THE MATTES ASKED FOR (--matte NAME=PRIMPATH): each the ids of the prims at or under a path, as the scene's
    // manifest names them. A name said twice is the two paths together.
    std::vector<std::pair<std::string, std::vector<uint32_t>>> mattes;
    for (const std::string& said : o.mattes) {
        const size_t eq = said.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 >= said.size()) {
            err("--matte takes NAME=PRIMPATH, not '%s'\n", said.c_str());
            return kExitFailure;
        }
        if (scene->prims.empty()) {
            err("%s does not keep which prim each gaussian was made from (it was shot before a scene did): there "
                "is no matte to give; shoot it again\n", directory.string().c_str());
            return kExitFailure;
        }
        if (o.output.empty()) {
            err("--matte writes its mattes beside the picture: say where with -o\n");
            return kExitFailure;
        }
        const std::string name = said.substr(0, eq);
        std::string path = said.substr(eq + 1);
        while (path.size() > 1 && path.back() == '/') path.pop_back();
        std::vector<uint32_t> ids;
        for (const auto& [prim, id] : scene->prims) {
            if (path == "/" || prim == path ||
                (prim.size() > path.size() && prim.compare(0, path.size(), path) == 0 && prim[path.size()] == '/')) {
                ids.push_back(id);
            }
        }
        if (ids.empty()) {
            err("--matte %s: the scene names no prim at or under %s\n", said.c_str(), path.c_str());
            return kExitFailure;
        }
        out("matte %s: %zu of the scene's prims at %s\n", name.c_str(), ids.size(), path.c_str());
        const auto before = std::find_if(mattes.begin(), mattes.end(), [&](const auto& m) { return m.first == name; });
        if (before != mattes.end()) {
            before->second.insert(before->second.end(), ids.begin(), ids.end());
        } else {
            mattes.emplace_back(name, std::move(ids));
        }
    }
    withPrims = !mattes.empty();
    // A VIEW'S MATTES, written beside its picture (`<picture's stem>_matte_<NAME>.<its extension>`): the scene
    // drawn once more by its gaussians' own blend with the rasteriser's Cryptomatte on -- each pixel's
    // coverage by the id of the prim the gaussian was made from -- and, a matte, the coverage of its ids
    // summed (render::Discovery::matte). The value is in every channel, alpha too: what the prims' gaussians
    // gave the pixel, behind whatever stands in front.
    const auto writeMattes = [&](const render::Camera& from, const std::vector<render::SplatInstance>& instances,
                                 const render::RenderSettings& drawn, const std::string& picture) -> Result<void> {
        render::RenderSettings named = drawn;
        named.cryptomatte = true;
        render::RenderTargets frame;
        auto stats = raster->render(from, instances, named, frame);
        if (!stats) return std::move(stats).error();
        if (!frame.crypto.valid()) {
            return Error(ErrorCode::InternalError, "the frame kept no Cryptomatte");
        }
        const std::filesystem::path given = picture;
        const std::string stem = (given.parent_path() / given.stem()).string();
        for (const auto& [name, ids] : mattes) {
            auto cover = discovery->matte(frame.crypto, drawn.width, drawn.height, ids);
            if (!cover) return std::move(cover).error();
            render::RenderTargets matte = frame;
            matte.colour = *cover;
            auto covered = render::imageStats(library, matte.colour, drawn.width, drawn.height);
            if (!covered) return std::move(covered).error();
            const std::string to = stem + "_matte_" + name + given.extension().string();
            ATHENEA_TRY(writeFrame(library, **device, matte, drawn.width, drawn.height, to, 0.0F));
            out("matte %s: covers %.3f%% of the view; wrote %s\n", name.c_str(), 100.0 * covered->mean[3], to.c_str());
        }
        return ok();
    };

    // Every piece on the device, and its appearance where it has one.
    std::vector<std::unique_ptr<scene::GpuSplats>> clouds;
    std::vector<std::unique_ptr<render::NeuralCloud>> looks;
    std::vector<render::SplatInstance> byHarmonics;
    std::vector<render::SplatInstance> byLatents;
    uint32_t fitted = 0;
    // THE SCENE AS IT STOOD (--at a time, --snapshot one of the manifest's): the pieces in by then, less
    // the gaussians gone by then.
    const uint32_t asOf = o.snapshot != 0 ? o.snapshot : o.at >= 0.0 ? snapshotAt(*scene, o.at) : 0u;
    lightAsOf = asOf;   // a piece's light as it was said up to then
    if ((o.snapshot != 0 || o.at >= 0.0) && asOf == 0) {
        err("the scene has no snapshot by time %g\n", o.at);
        return kExitFailure;
    }
    if (o.budget != 0 && o.views.empty()) {
        // A BUDGET (--budget): the threshold is the one that draws no more than that many from this view.
        uint64_t drawn = 0;
        auto threshold = thresholdForBudget(library, *loader, *discovery, directory, *scene,
                                            render::projectionFor(camera, width, height), width, height, asOf, o.budget,
                                            drawn);
        if (!threshold) return report(threshold.error());
        o.lod = *threshold;
        out("a budget of %llu gaussians: a threshold of %.2f pixels, which draws %llu\n",
            static_cast<unsigned long long>(o.budget), o.lod, static_cast<unsigned long long>(drawn));
    }
    // WITH GLASS (--glass) the scene is cut whatever the threshold (none: the finest there is), as one cloud
    // that holds the glass and the room it shows, and each glass gaussian's colour is the view's.
    // WITH LAMPS (--lamps) the same cut, and before the glass is shown the room each gaussian that is neither
    // glass nor a mirror is given what the lamps send it (render::SceneGlass::setLamps).
    std::vector<render::SceneLamp> lamps;
    if (!o.lamps.empty()) {
        if (!scene->baseColour) {
            err("%s does not keep its surfaces' base colour (it was shot before a scene did): a lamp has nothing "
                "to light; shoot it again\n", directory.string().c_str());
            return kExitFailure;
        }
        const std::string kindOf = std::filesystem::path(o.lamps).extension().string();
        const bool authored = kindOf == ".usd" || kindOf == ".usda" || kindOf == ".usdc" || kindOf == ".usdz";
        if (authored) {
            // A USD FILE OF LIGHTS: its sphere lights (with their cones) and its distant ones, as authored.
            uint32_t skipped = 0;
            auto read = usd::readStageLamps(o.lamps, o.at >= 0.0 ? o.at : 0.0, skipped);
            if (!read) return report(read.error());
            for (const usd::StageLamp& from : *read) {
                render::SceneLamp lamp;
                lamp.radiance = from.radiance;
                if (from.distant) {
                    // Towards it, and what it lays on a surface facing it over pi: its radiance over the
                    // disc it shows, whose half angle is half the light's.
                    const float half = 0.5F * from.angle * 3.14159265F / 180.0F;
                    if (!(half > 0.0F)) {
                        ++skipped;
                        continue;
                    }
                    lamp.distant = true;
                    lamp.position = {static_cast<float>(-from.sends[0]), static_cast<float>(-from.sends[1]),
                                     static_cast<float>(-from.sends[2])};
                    const float share = std::sin(half) * std::sin(half);
                    lamp.radiance = {from.radiance[0] * share, from.radiance[1] * share, from.radiance[2] * share};
                } else {
                    lamp.position = {static_cast<float>(from.position[0]), static_cast<float>(from.position[1]),
                                     static_cast<float>(from.position[2])};
                    lamp.radius = from.radius;
                    if (from.cone >= 0.0F) {
                        lamp.axis = {static_cast<float>(from.sends[0]), static_cast<float>(from.sends[1]),
                                     static_cast<float>(from.sends[2])};
                        lamp.cone = std::cos(from.cone * 3.14159265F / 180.0F);
                    }
                }
                lamps.push_back(lamp);
            }
            out("lamps: %zu read from %s%s\n", lamps.size(), o.lamps.c_str(),
                skipped != 0 ? (", " + std::to_string(skipped) + " lights left out (neither sphere nor distant, or normalised)").c_str() : "");
        }
        std::ifstream file(authored ? std::string{} : o.lamps);
        if (!authored && !file) {
            err("cannot read %s\n", o.lamps.c_str());
            return kExitFailure;
        }
        std::string line;
        while (!authored && std::getline(file, line)) {
            std::istringstream words(line);
            std::string kind;
            if (!(words >> kind) || kind[0] == '#') continue;
            std::vector<float> n;
            for (float v = 0.0F; words >> v;) n.push_back(v);
            render::SceneLamp lamp;
            if (kind == "sphere" && n.size() == 8) {
                // sphere x y z radius intensity r g b
                lamp.position = {n[0], n[1], n[2]};
                lamp.radius = n[3];
                lamp.radiance = {n[4] * n[5], n[4] * n[6], n[4] * n[7]};
            } else if (kind == "spot" && n.size() == 12) {
                // spot x y z  tx ty tz  half-angle(degrees) radius intensity r g b
                lamp.position = {n[0], n[1], n[2]};
                lamp.axis = {n[3] - n[0], n[4] - n[1], n[5] - n[2]};
                lamp.cone = std::cos(n[6] * 3.14159265F / 180.0F);
                lamp.radius = n[7];
                lamp.radiance = {n[8] * n[9], n[8] * n[10], n[8] * n[11]};
            } else if (kind == "distant" && n.size() == 7) {
                // distant tx ty tz irradiance r g b: towards the light, and what it lays on a surface facing it
                lamp.position = {n[0], n[1], n[2]};
                lamp.distant = true;
                const float share = n[3] / 3.14159265F;
                lamp.radiance = {share * n[4], share * n[5], share * n[6]};
            } else {
                err("%s: a lamp is `sphere x y z radius intensity r g b`, `spot x y z tx ty tz half-angle radius "
                    "intensity r g b` or `distant tx ty tz irradiance r g b`, not: %s\n", o.lamps.c_str(), line.c_str());
                return kExitFailure;
            }
            lamps.push_back(lamp);
        }
    }
    const bool shaded = o.glass || !lamps.empty();
    const bool cutting = o.lod > 0.0F || shaded;
    std::optional<render::SceneGlass> glass;
    if (shaded) {
        auto made = render::SceneGlass::create(library);
        if (!made) return report(made.error());
        glass.emplace(std::move(*made));
        if (auto set = glass->setLamps(lamps, o.lampsOnly, o.lampShadowRays, o.lampBounceRays); !set) return report(set.error());
    }
    const auto saidOfGlass = [&](const render::GlassSeen& seen) {
        out("glass: %u gaussians and %u of polished metal; through it %u met the scene, mirrored %u did; the tree "
            "%.1f ms, the rays %.1f ms\n",
            seen.glass, seen.mirrors, seen.behind, seen.mirrored, seen.treeMs, seen.seenMs);
        if (!lamps.empty()) {
            out("lamps: %zu, asked of %u gaussians in %.1f ms; their light off the scene, %u rays met it, in %.1f ms\n",
                lamps.size(), seen.lit, seen.litMs, seen.bounced, seen.bouncedMs);
        }
    };
    std::unique_ptr<scene::GpuSplats> wholeCut;
    if (cutting && o.views.empty()) {
        // THE SCENE CUT FOR THE VIEW (--lod, --budget): each gaussian drawn where it is the level the view
        // wants, a coarse one in place of the finer ones laid over it from where those are under the
        // threshold -- gathered into one cloud.
        render::GlassSeen seen;
        auto cut = cutScene(library, *loader, *discovery, directory, *scene, render::projectionFor(camera, width, height),
                            width, height, o.lod, asOf, o.morph, glass ? &*glass : nullptr, &seen);
        if (!cut) return report(cut.error());
        if (glass) {
            saidOfGlass(seen);
        }
        if (cut->count != 0) {
            wholeCut = std::make_unique<scene::GpuSplats>(std::move(*cut));
            clouds.push_back(std::make_unique<scene::GpuSplats>(*wholeCut));
            render::SplatInstance instance{clouds.back().get(), render::Mat4::identity(), {}};
            byHarmonics.push_back(instance);
            byLatents.push_back(instance);
            looks.emplace_back();
        }
    }
    for (const io::NeuralScenePiece& piece : scene->pieces) {
        if (cutting) {
            break;   // the cut above is the scene; a walk cuts for each of its views
        }
        if (asOf != 0 && piece.foundIn > asOf) {
            continue;
        }
        // Nothing of it stands: every gaussian it held has gone. And drawn flat, nothing of it is the
        // finest: finer looks were laid over all of it.
        if (asOf == 0 && piece.gaussians != 0 &&
            (o.lod > 0.0F ? piece.removed : piece.removed + piece.refinedCount) >= piece.gaussians) {
            continue;
        }
        // One view of a found scene reads the pieces its frustum meets, and no others.
        if (o.views.empty() && piece.origin == "discovered" &&
            !boxInView(piece.boundsMin, piece.boundsMax, render::projectionFor(camera, width, height), width, height)) {
            continue;
        }
        auto cloud = loadPiece(library, *loader, *discovery, directory, piece, asOf);
        if (!cloud) return report(cloud.error());
        if (cloud->count == 0) {
            continue;   // nothing of it is drawn from here
        }
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
    settings.madeSpread = madeAs;
    if (o.noAntialias) settings.antialiasOverride = false;   // dilated and not thinned: a disc narrower than a pixel stays opaque
    if (o.madeSpread >= 0.0F) settings.madeSpread = o.madeSpread;
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
        uint64_t walkDrawn = 0;
        float walkThreshold = 0.0F;
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
            // With levels of detail the cut is the view's own: each piece cut again for where this view
            // stands (read again from the disk, which the time below does not count).
            std::vector<scene::GpuSplats> cutClouds;
            std::vector<render::SplatInstance> cutInstances;
            float lod = o.lod;
            if (o.budget != 0) {
                uint64_t drawn = 0;
                auto threshold = thresholdForBudget(library, *loader, *discovery, directory, *scene,
                                                    render::projectionFor(view, width, height), width, height, asOf,
                                                    o.budget, drawn);
                if (!threshold) return report(threshold.error());
                lod = *threshold;
                walkThreshold = lod;
            }
            if (lod > 0.0F || shaded) {
                const render::Projection projection = render::projectionFor(view, width, height);
                render::GlassSeen seen;
                auto cut = cutScene(library, *loader, *discovery, directory, *scene, projection, width, height, lod, asOf,
                                    o.morph, glass ? &*glass : nullptr, &seen);
                if (!cut) return report(cut.error());
                if (glass) {
                    saidOfGlass(seen);
                }
                walkDrawn = cut->count;
                if (cut->count != 0) {
                    cutClouds.push_back(std::move(*cut));
                    cutInstances.push_back(render::SplatInstance{&cutClouds.back(), render::Mat4::identity(), {}});
                }
            }
            auto stats = raster->render(view, lod > 0.0F || shaded ? cutInstances : drawnAs, waited, frame);
            if (!stats) return report(stats.error());
            if (lod > 0.0F || shaded) {
                // A cut's clouds are new to the rasteriser, and a first frame of anything pays for that: the
                // frame timed is the one after, as a scene drawn flat is timed from its second view on.
                stats = raster->render(view, cutInstances, waited, frame);
                if (!stats) return report(stats.error());
            }
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
                if (!mattes.empty()) {
                    if (auto w = writeMattes(view, lod > 0.0F || shaded ? cutInstances : byHarmonics, settings, to); !w) {
                        return report(w.error());
                    }
                }
            }
        }
        uint64_t held = 0;
        for (const auto& cloud : clouds) {
            held += cloud->count;
        }
        if (cutting || o.budget != 0) {
            held = walkDrawn;   // the last view's cut
        }
        if (o.budget != 0) {
            out("a budget of %llu gaussians: the last view's threshold %.2f pixels\n",
                static_cast<unsigned long long>(o.budget), walkThreshold);
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
        if (!mattes.empty()) {
            if (auto w = writeMattes(camera, byHarmonics, settings, o.output); !w) return report(w.error());
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
    render->add_option("--lod", o.lod, "levels of detail: a gaussian narrower than this many pixels on the screen is drawn in place of the finer ones laid over it (0: the finest everywhere)");
    render->add_option("--budget", o.budget, "levels of detail by a count: the finest threshold that draws no more than this many gaussians from the view, in place of --lod");
    render->add_option("--made-spread", o.madeSpread, "a scene drawn without the compensation pays it under this width: the sigma in pixels its discs were made at (shoot's --spread)");
    render->add_flag("--no-antialias", o.noAntialias, "no Mip-Splatting 2D filter compensation: a gaussian is spread a third of a pixel and keeps its opacity");
    render->add_option("--group", o.groups, "a group of lights' level (shoot --light-group): NAME=LEVEL, or NAME=R,G,B for a tint; * names every group not named. A group not named is at 1");
    render->add_option("--lamps", o.lamps, "a file of lamps added to the scene: each lights the gaussians it reaches by their surface's base colour, over the light the scene was found under (implies the cut --glass draws)");
    render->add_option("--lamp-shadow-rays", o.lampShadowRays, "--lamps: rays from a gaussian to each lamp, over the disc it shows (1: to its middle, a hard shadow)");
    render->add_option("--lamp-bounce", o.lampBounceRays, "--lamps: rays a gaussian gathers what the lamps' light bounces on to with, off the scene's own gaussians (0: the lamps' direct light alone)");
    render->add_flag("--lamps-only", o.lampsOnly, "--lamps: without the light the scene was found under");
    render->add_flag("--glass", o.glass, "solid glass shows what it bends and mirrors of the scene from each view, worked out from what the glass is made of, in place of the picture the look that found it saw");
    render->add_option("--morph", o.morph, "--lod, --budget: a gaussian is moved towards what it was laid over across this share of the threshold before that is drawn in its place (0: a step)");
    render->add_option("--frames", o.storedFrames, "the stored frames FIRST to LAST as pictures, the scene read once: -o a prefix, written <stem>_NNNN.png or .exr")->expected(2);
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
    shoot->add_option("--grow", o.grow, "a file that grows: a path with no extension, where each frame's gaussians are written as it is done (<stem>_NNN.spz) and the index a viewer follows is put in place again (<stem>.json)");
    shoot->add_option("--grow-exposure", o.growExposure, "--grow: stops on the scene's light before it is encoded");
    shoot->add_option("--pyramid", o.pyramid, "the widest cells, in pixels, of the levels a frame leaves above its last look where it found new surface (a power of two; 1: none)");
    shoot->add_option("--covered-enough", o.coveredEnough, "a view the scene already covers this much of takes only its last look, at one pixel (over 1: every view takes them all)");
    shoot->add_flag("--free-each-frame", o.freeEachFrame, "the scene leaves the device after every frame and the next reads back what it sees, rather than keeping what the next view still sees");
    shoot->add_option("--time-step", o.timeStep, "--views: each view of the file is this much later than the last (a sequence, a frame a line)");
    shoot->add_option("--samples", o.samples, "paths a pixel the render gathers");
    shoot->add_option("--pass-samples", o.passSamples, "paths a pixel each pass");
    shoot->add_option("--bounces", o.bounces, "bounces after the first hit");
    shoot->add_flag("--denoise", o.denoise, "denoise the whole frame before its last look");
    shoot->add_option("--ray-margin", o.rayMargin, "athenea:rayMargin");
    shoot->add_option("--coarsest", o.coarsest, "pixels a side of the first look's cells (a power of two; 1: one look, when the frame is whole)");
    shoot->add_option("--spread", o.shootSpread, "a gaussian's standard deviation, in cells at its depth");
    shoot->add_option("--tolerance", o.tolerance, "a surface is already held where the scene's depth is within this share of the render's");
    shoot->add_option("--refine", o.refine, "a surface held with gaussians more than this many times as wide as this look's is made again, finer");
    shoot->add_option("--repaint", o.repaint, "a surface held whose colour is off the render's by more than this share is made again (0: held by its depth alone)");
    shoot->add_flag("--no-headlight", o.noHeadlight, "a stage that authors no light is left dark but for what glows, where the path tracer lights it from the eye (and so differently for each camera)");
    shoot->add_option("--lights-on", o.lightsOn, "only these lights of the stage light it (prim paths, comma separated or repeated): every other light the stage authors is switched off -- a group's lamps for --light-group")->delimiter(',');
    shoot->add_option("--lights-off", o.lightsOff, "these lights of the stage are switched off (prim paths): the scene shot without the lamps that are groups")->delimiter(',');
    shoot->add_option("--scene-cache", o.sceneCache, "the found scene is what a path's bounce meets: the light of the gaussians found so far is kept in a cube of this many cells an axis over the stage's bounds, and a path that bounces off a surface by a broad lobe and lands in a cell that holds light takes it and ends there, unshaded. 0: every path goes its bounces")->check(CLI::Range(0, 512));
    shoot->add_option("--light-groups", o.lightGroups, "NAME=PATH (repeatable): the stage's lights at or under PATH are the group NAME, and what they put in each frame is kept as that group's layer of light, from the same paths that find the scene -- no pass of its own. The scene's own light is the frame less its groups, as a scene shot with --lights-off and a --light-group pass a group");
    shoot->add_flag("--light-over-base", o.lightOverBase, "in a pass of groups over a scene found (--groups-only, --light-group): what a pixel sends is carried from that pixel's base colour to the gaussian's own, which the scene keeps at its own detail -- so the pass's frames can be much smaller than the scene's and the surfaces' texture stays in the lamps' light");
    shoot->add_flag("--groups-only", o.groupsOnly, "with --light-groups, over a scene already found: the stage is lit by the named groups' lights alone, each view is path traced once and every group's share of it is kept as that group's layer; nothing of the scene is made, taken out or said anew. Several lamps added to a scene for the path tracing of one");
    shoot->add_option("--light-group", o.lightGroup, "the stage handed is lit by one group of lights alone: each frame is path traced and what every gaussian of the scene sends in it is kept as that group's layer of light, by this name; the scene itself is not touched (render --group sets the group's level)");
    shoot->add_flag("--relight", o.relight, "each frame first takes again the light of what the scene holds and the frame shows: those gaussians keep their place and what they are made of, and a layer of light is kept for the time (a stage whose light changes)");
    shoot->add_flag("--antialias", o.antialias, "a new scene is drawn with the Mip-Splatting 2D filter's compensation, by the looks and by whoever draws it after (kept in its manifest); without it, the default, a disc narrower than a pixel stays opaque");
    shoot->add_flag("--no-antialias", o.noAntialias, "the scene is drawn without the compensation from here on, whatever its manifest says");
    shoot->add_option("--turned", o.turned, "the least a disc's facing is taken as: it is stretched along the surface's fall by no more than 1 over this");
    shoot->add_option("--glass-opacity", o.glassOpacity, "the least a pane covers of what is behind it, however clear");
    shoot->add_option("--name", o.name, "the scene's name");
    shoot->add_option("--up-axis", o.upAxis, "Y or Z (default: the stage's)");
    shoot->add_option("--meters-per-unit", o.metersPerUnit, "the scene's unit, in metres (default: the stage's)");
    shoot->add_option("--source", o.source, "what the stage is");
    shoot->add_option("--licence", o.licence, "what the source asks of whoever uses it");
    run(shoot, options, runShoot);

    CLI::App* bake = cmd->add_subcommand("bake", "the scene as it stood, frozen into a .athc with the engine's levels of detail");
    bake->add_option("scene", o.directory, "the scene's directory")->required();
    bake->add_option("--at", o.at, "the scene as it stood at this time (default: as it stands)");
    bake->add_option("--snapshot", o.snapshot, "the same by a snapshot of the manifest's, counted from one");
    bake->add_option("-o,--output", o.output, "the .athc")->required();
    run(bake, options, runBake);

    CLI::App* exported = cmd->add_subcommand("export", "the scene as standard splat files (SPZ) in parts, for a viewer on the web");
    exported->add_option("scene", o.directory, "the scene's directory")->required();
    exported->add_option("--at", o.at, "the scene as it stood at this time (default: as it stands)");
    exported->add_option("--snapshot", o.snapshot, "the same by a snapshot of the manifest's, counted from one");
    exported->add_option("--part", o.part, "the most gaussians a file holds");
    exported->add_option("--levels", o.levels, "levels of detail beside the scene: a budget of gaussians each, the scene whole within it, written as <stem>_lodK_NNN.spz and named in the index")->delimiter(',');
    exported->add_option("--format", o.format, "the parts' format: spz (what a viewer on the web reads), ply (INRIA's 3DGS layout) or glb (glTF with KHR_gaussian_splatting)")->check(CLI::IsMember({"spz", "ply", "glb"}));
    exported->add_flag("--history", o.history, "--by-frame: also what each frame found that a later frame took out or refined, in files that say until which frame they stand: the scene as it stood at every frame, for a stage that moves");
    exported->add_flag("--by-frame", o.byFrame, "a part a rendered frame, holding what that frame found: the scene in the order it grew");
    exported->add_option("--exposure", o.exposure, "stops on the scene's light before it is encoded");
    exported->add_option("--group", o.groups, "a group of lights' level in what is handed on (shoot --light-group): NAME=LEVEL or NAME=R,G,B; * names every group not named. A group not named is at 1");
    render->add_option("--matte", o.mattes, "NAME=PRIMPATH (repeatable): beside the picture, the matte of the gaussians made from the stage's prims at or under PRIMPATH -- what they cover of each pixel, by the picture's own blend, behind whatever stands in front -- written as <picture>_matte_NAME; a scene shot before it kept its gaussians' prims has none");
    exported->add_flag("--base-colour", o.baseColour, "each gaussian is handed on with its surface's base colour as its colour, in place of its light (glass black): for a viewer that lights the scene with a lamp of its own. The same parts as the scene's own export; --exposure would scale it, and is left out");
    exported->add_flag("--no-prims", o.noPrims, "the parts without which prim each gaussian was made from: no <file>.ids beside them and no prims in the index (the same parts written again for a group of lights or for the base colours need none)");
    exported->add_option("--own-light", o.ownLight, "what the scene's own light is multiplied by: 0 hands on the groups' light alone (a group by itself, with --group)");
    exported->add_option("-o,--output", o.output, "a path with no extension: <stem>_NNN.spz and <stem>.json")->required();
    run(exported, options, runExport);

    CLI::App* info = cmd->add_subcommand("info", "what the manifest holds");
    info->add_option("scene", o.directory, "the scene's directory")->required();
    run(info, options, runInfo);
}

}   // namespace athenea::cli
