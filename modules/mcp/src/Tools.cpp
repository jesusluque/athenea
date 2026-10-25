// Copyright (c) 2026 jesus luque.
//
// Every tool the renderer offers, in one table. What a caller can do here is
// what `athenea` can do from a shell -- open a stage, look at it, set what a frame
// costs, render it, read a pixel, convert a capture -- with the device and the
// stage kept warm between calls, which a shell cannot do.
//
// A render answers with the image as well as the numbers. A model that cannot
// see what it rendered is guessing, and the display transform that makes those
// pixels is the same kernel `athenea view` shows through, so what comes back is
// what the viewer would show.
#include "Tools.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>

#include "athenea/core/Hash.h"
#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/Texture.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Png.h"
#include "athenea/io/Readers.h"
#include "athenea/render/Camera.h"
#include "athenea/technique/Denoiser.h"
#include "athenea/technique/DisplayTransform.h"
#include "athenea/usd/Export.h"

namespace athenea::mcp {
namespace {

using nlohmann::json;

json object(std::initializer_list<std::pair<const std::string, json>> properties, std::vector<std::string> required = {}) {
    json schema{{"type", "object"}, {"properties", json::object()}};
    for (const auto& [name, property] : properties) {
        schema["properties"][name] = property;
    }
    if (!required.empty()) {
        schema["required"] = required;
    }
    return schema;
}

json field(const char* type, const char* description) {
    return json{{"type", type}, {"description", description}};
}

std::string errorText(const Error& error) {
    return error.toString();
}

/// Opens the stage if a path is given and none is open, or if a different one
/// is asked for. The renderer is what costs: keeping it is the point.
Result<void> ensureStage(Session& session, const json& args) {
    const std::string path = args.value("stage", std::string{});
    if (!path.empty() && (session.stage != std::filesystem::path(path) || !session.open())) {
        auto opened = usd::StageRenderer::open(path);
        if (!opened) return std::move(opened).error();
        session.renderer = std::move(*opened);
        session.stage = path;
        return ok();
    }
    if (!session.open()) {
        return Error(ErrorCode::InvalidArgument, "no stage is open: call open_stage, or pass 'stage'");
    }
    return ok();
}

/// The settings a frame is drawn with, all optional, all remembered.
void applySettings(Session& session, const json& args) {
    usd::StageRenderer& r = *session.renderer;
    if (args.contains("lightSamples")) r.setLightSamples(args["lightSamples"].get<uint32_t>());
    if (args.contains("chooseLights")) r.setChooseLights(args["chooseLights"].get<bool>());
    if (args.contains("splatShadows")) r.setSplatShadows(args["splatShadows"].get<bool>());
    if (args.contains("splatReflections")) r.setSplatReflections(args["splatReflections"].get<bool>());
    if (args.contains("pathSamples")) r.setPathSamples(args["pathSamples"].get<uint32_t>());
    if (args.contains("pathBounces")) r.setPathBounces(args["pathBounces"].get<uint32_t>());
    if (args.contains("pathTotal")) r.setPathTotal(args["pathTotal"].get<uint32_t>());
    if (args.contains("denoise")) r.setDenoise(args["denoise"].get<bool>());
    if (args.contains("defaultLights")) {
        (void)r.setDefaultLights(args["defaultLights"].get<bool>());
    }
    if (args.contains("motionBuckets")) r.setMotionBuckets(args["motionBuckets"].get<uint32_t>());
    if (args.contains("cloudShadows")) r.setCloudShadows(args["cloudShadows"].get<bool>());
    if (args.contains("cloudShadowTexels")) r.setCloudShadowResolution(args["cloudShadowTexels"].get<uint32_t>());
    if (args.contains("cloudShadowTerms")) r.setCloudShadowTerms(args["cloudShadowTerms"].get<uint32_t>());
    if (args.contains("cloudShadowDensity")) {
        r.setCloudShadowDensity(args["cloudShadowDensity"].get<float>());
    }
    // The shutter a cloud's motion blur is integrated over: [open, close] in
    // frames, or one number for 0..that. It beats a stage camera's own, as
    // `--shutter` and the viewer's slider do.
    if (args.contains("shutter")) {
        const json& asked = args["shutter"];
        if (asked.is_array() && asked.size() == 2) {
            r.setShutter(asked[0].get<double>(), asked[1].get<double>());
        } else if (asked.is_number()) {
            r.setShutter(0.0, asked.get<double>());
        }
    }
    // Variant selections, each as USD writes one in a path.
    if (args.contains("variant")) {
        const json& asked = args["variant"];
        if (asked.is_string()) {
            (void)r.setVariantSelection(asked.get<std::string>());
        } else if (asked.is_array()) {
            for (const json& one : asked) {
                if (one.is_string()) {
                    (void)r.setVariantSelection(one.get<std::string>());
                }
            }
        }
    }
    // THE SKY, a compositor's way: one image and one turn for every dome the
    // stage has (a stage with none gets the default dome and sun first), so a
    // relightable cloud is lit by whatever room the artist tries. A dome is
    // set only where it differs, since each set is a session edit Hydra
    // syncs; empty gives back the stage's own image.
    if (args.contains("sky") || args.contains("skyRotation")) {
        if (r.domes().empty() && args.contains("sky") && !args["sky"].get<std::string>().empty()) {
            (void)r.setDefaultLights(true);
        }
        for (const usd::StageDome& dome : r.domes()) {
            if (args.contains("sky")) {
                const std::string sky = args["sky"].get<std::string>();
                if (sky != dome.texture) {
                    (void)r.setDomeTexture(dome.prim, sky);
                }
            }
            if (args.contains("skyRotation")) {
                const float degrees = args["skyRotation"].get<float>();
                if (degrees != dome.rotation) {
                    (void)r.setDomeRotation(dome.prim, degrees);
                }
            }
        }
    }
    if (args.contains("skyVisible")) r.setDomeVisible(args["skyVisible"].get<bool>());
    // A parameter of the gaussians drawn in place of their light, by name or
    // by number (StageRenderer::setSplatView); "light" or 0 puts it back.
    if (args.contains("splatView")) {
        static const char* kViews[] = {"light", "albedo", "roughness", "metallic", "transmission", "ior", "normal",
                                       "opacity", "size", "skyVisibility", "bounce", "openCells", "emission", "field",
                                       "baked", "materialRoughness"};
        const json& asked = args["splatView"];
        uint32_t view = 0;
        if (asked.is_number()) {
            view = asked.get<uint32_t>();
        } else if (asked.is_string()) {
            const std::string name = asked.get<std::string>();
            for (uint32_t k = 0; k < std::size(kViews); ++k) {
                if (name == kViews[k]) view = k;
            }
        }
        r.setSplatView(view < std::size(kViews) ? view : 0u);
    }
    if (args.contains("refine")) r.setRefineLevel(args["refine"].get<uint32_t>());
    if (args.contains("visibility")) {
        (void)r.setMeshVisibility(args["visibility"].get<std::string>());
    }
}

/// The frame as the viewer would show it: the display transform on the device,
/// read back as eight-bit rows and packed into a PNG.
/// The frame as it is shown -- the AOV asked for, through the view transform
/// -- encoded as a PNG, and written to `file` as well where one is given.
Result<std::vector<uint8_t>> preview(usd::StageRenderer& renderer, const std::string& aov, uint32_t width,
                                     uint32_t height, const std::string& view, float exposure,
                                     const std::filesystem::path& file = {}) {
    auto source = renderer.displaySource(aov);
    if (!source) return std::move(source).error();
    auto display = technique::DisplayTransform::create(renderer.library());
    if (!display) return std::move(display).error();
    gpu::TextureDesc desc;
    desc.width = width;
    desc.height = height;
    desc.format = rhi::Format::RGBA8Unorm;
    desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource |
                 rhi::TextureUsage::CopySource;
    desc.label = "mcp.preview";
    auto texture = gpu::Texture::create(renderer.device(), desc);
    if (!texture) return std::move(texture).error();
    technique::DisplaySettings settings;
    settings.view = view == "standard"  ? technique::ViewTransform::Standard
                    : view == "aces"    ? technique::ViewTransform::Aces2
                                        : technique::ViewTransform::AgX;
    settings.display = technique::DisplayEncoding::Srgb;
    settings.exposure = exposure;
    {
        gpu::CommandBatch batch(renderer.device());
        ATHENEA_TRY(display->run(batch, *source, settings, texture->rhi(), width, height));
        ATHENEA_TRY(batch.submit(true));
    }
    auto texels = texture->read(renderer.device(), 0, 0);
    if (!texels) return std::move(texels).error();
    std::span<const uint8_t> bytes(reinterpret_cast<const uint8_t*>(texels->data()), texels->size());
    if (!file.empty()) {
        ATHENEA_TRY(io::writePng(file, width, height, bytes));
    }
    return io::encodePng(width, height, bytes);
}

json imageAnswer(std::string text, const std::vector<uint8_t>& png) {
    static constexpr char kBase64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((png.size() + 2) / 3 * 4);
    for (size_t at = 0; at < png.size(); at += 3) {
        const uint32_t a = png[at];
        const uint32_t b = at + 1 < png.size() ? png[at + 1] : 0;
        const uint32_t c = at + 2 < png.size() ? png[at + 2] : 0;
        const uint32_t word = (a << 16) | (b << 8) | c;
        encoded += kBase64[(word >> 18) & 63];
        encoded += kBase64[(word >> 12) & 63];
        encoded += at + 1 < png.size() ? kBase64[(word >> 6) & 63] : '=';
        encoded += at + 2 < png.size() ? kBase64[word & 63] : '=';
    }
    return json{{"content", json::array({json{{"type", "text"}, {"text", std::move(text)}},
                                         json{{"type", "image"}, {"data", std::move(encoded)},
                                              {"mimeType", "image/png"}}})}};
}

// ---- the tools ---------------------------------------------------------------

json openStage(Session& session, const json& args) {
    const std::string path = args.value("stage", std::string{});
    if (path.empty()) {
        return textAnswer("open_stage wants 'stage': a .usd, .usda or .usdc path.", true);
    }
    auto opened = usd::StageRenderer::open(path);
    if (!opened) return textAnswer(errorText(opened.error()), true);
    session.renderer = std::move(*opened);
    session.stage = path;
    std::ostringstream out;
    out << "Opened " << path << "\n";
    const std::pair<double, double> range = session.renderer->animationRange();
    out << "up axis: " << session.renderer->upAxis() << ", time codes to " << session.renderer->endTimeCode()
        << ", animated over " << range.first << ".." << range.second << "\n";
    for (const usd::StageVariantSet& set : session.renderer->variantSets()) {
        out << "variant set " << set.prim << "{" << set.name << "} = " << set.selected << ", one of "
            << set.variants.size() << ": ";
        for (size_t k = 0; k < set.variants.size() && k < 8; ++k) {
            out << (k == 0 ? "" : ", ") << set.variants[k];
        }
        out << (set.variants.size() > 8 ? ", ... (the variants tool lists them)\n" : "\n");
    }
    const std::vector<std::string> cameras = session.renderer->cameras();
    out << "cameras: " << (cameras.empty() ? std::string("none (render frames one of its own)") : std::string());
    for (const std::string& camera : cameras) {
        out << "\n  " << camera;
    }
    return textAnswer(out.str());
}

json stageTree(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    const std::string path = args.value("path", std::string("/"));
    std::ostringstream out;
    out << path << "\n";
    for (const usd::StagePrim& child : session.renderer->children(path)) {
        out << "  " << child.name << "  " << child.type << (child.hasChildren ? "  [+]" : "")
            << (child.instance ? "  (instance)" : "") << "\n";
    }
    return textAnswer(out.str());
}

/// Every variant set the stage carries, and a selection made on one of them.
///
/// Variant sets are USD's own "pick one of these": what they mean is the
/// asset's business -- an animation, a level of detail, a shirt -- so this
/// lists them and selects one without an opinion about any of it. The
/// selection goes in the session layer, so the file is not touched.
json variantsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    usd::StageRenderer& r = *session.renderer;
    std::ostringstream out;
    if (args.contains("select")) {
        const std::string selection = args["select"].get<std::string>();
        if (auto set = r.setVariantSelection(selection); !set) {
            return textAnswer(errorText(set.error()), true);
        }
        const std::pair<double, double> range = r.animationRange();
        out << "selected " << selection << ", the stage is now animated over " << range.first << ".." << range.second
            << "\n";
    }
    const std::vector<usd::StageVariantSet> sets = r.variantSets();
    if (sets.empty()) {
        out << "no variant sets on this stage\n";
        return textAnswer(out.str());
    }
    const bool full = args.value("all", false);
    for (const usd::StageVariantSet& set : sets) {
        out << set.prim << "{" << set.name << "} = " << set.selected << "  (" << set.variants.size() << ")\n";
        for (size_t k = 0; k < set.variants.size() && (full || k < 12); ++k) {
            out << "  " << (set.variants[k] == set.selected ? "* " : "  ") << set.variants[k] << "\n";
        }
        if (!full && set.variants.size() > 12) {
            out << "  ... " << set.variants.size() - 12 << " more; pass all=true\n";
        }
    }
    out << "select one with select: \"" << sets[0].prim << "{" << sets[0].name << "=<variant>}\"\n";
    return textAnswer(out.str());
}

json deviceInfo(Session& session, const json& args) {
    (void)args;
    std::ostringstream out;
    if (!session.open()) {
        out << "no stage open, so no device yet: open_stage first, or ask about a stage.\n";
        return textAnswer(out.str());
    }
    const gpu::Caps& caps = session.renderer->device().caps();
    out << "backend        " << caps.apiName << " on " << caps.adapterName << "\n";
    out << "rasterisation  " << (caps.rasterization ? "yes" : "no") << "\n";
    out << "ray queries    " << (caps.rayQuery ? "yes (inline)" : "no") << "\n";
    out << "ray pipelines  " << (caps.rayTracing ? "yes" : "no") << "\n";
    out << "structures     " << (caps.accelerationStructure ? "yes" : "no") << "\n";
    if (caps.optixVersion != 0) {
        out << "optix          " << caps.optixVersion << "\n";
    }
    out << "half           " << (caps.half ? "yes" : "no") << "\n";
    out << "converting stores " << (caps.convertingStores ? "yes" : "no") << "\n";
    out << "denoiser       " << (technique::denoiserBuilt() ? "OIDN" : "none") << "\n";
    out << "ocio           " << (technique::ocioBuilt() ? "yes" : "no") << "\n";
    return textAnswer(out.str());
}

json renderTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    applySettings(session, args);
    usd::StageRenderer& renderer = *session.renderer;
    const uint32_t width = args.value("width", 640u);
    const uint32_t height = args.value("height", 360u);
    const double time = args.value("time", 0.0);
    const std::string technique = args.value("technique", std::string("raster"));
    const std::string camera = args.value("camera", std::string{});
    std::vector<std::string> outputs =
        args.contains("aovs") ? args["aovs"].get<std::vector<std::string>>() : std::vector<std::string>{};
    // The layers a `raw` frame carries after its colour and depth: P, made
    // from the depth, and the Cryptomatte's layers, which are outputs the
    // frame must keep.
    const std::vector<std::string> layers =
        args.contains("layers") ? args["layers"].get<std::vector<std::string>>() : std::vector<std::string>{};
    for (const std::string& layer : layers) {
        if (layer != "P" && !(layer.size() == 14 && layer.rfind("CryptoObject0", 0) == 0 && layer[13] >= '0' &&
                              layer[13] <= '2')) {
            return textAnswer("layer '" + layer + "': P or CryptoObject00..02.", true);
        }
        if (layer != "P" && std::find(outputs.begin(), outputs.end(), layer) == outputs.end()) {
            outputs.push_back(layer);
        }
    }
    if (!outputs.empty()) {
        renderer.requestOutputs(outputs);
    } else {
        renderer.requestOutputs({"primId", "instanceId"});
    }
    const auto start = std::chrono::steady_clock::now();
    Result<usd::StageImage> image = Error(ErrorCode::InternalError, "not rendered");
    std::optional<render::Camera> drawnFrom;   // the camera P is made with, where it is the tool's own
    // A camera of the caller's own beats anything on the stage: it is how you
    // look at something in particular rather than at everything at once.
    if (args.contains("xform")) {
        // A compositor's camera as it holds one: where it stands and how it
        // turns, one matrix (camera to world, row by row, looking down -z
        // with +y up, as USD's and Nuke's do), and the lens in millimetres.
        const std::vector<double> m = args["xform"].get<std::vector<double>>();
        if (m.size() != 16) {
            return textAnswer("'xform' wants sixteen numbers, the camera to world matrix row by row.", true);
        }
        render::Camera own;
        std::copy(m.begin(), m.end(), own.cameraToWorld.m.begin());
        own.lens.focal = args.value("focal", 50.0);
        own.lens.haperture = args.value("haperture", own.lens.haperture);
        own.lens.vaperture = args.value("vaperture", own.lens.haperture * height / std::max(width, 1u));
        own.lens.nearZ = args.value("near", own.lens.nearZ);
        own.lens.farZ = args.value("far", own.lens.farZ);
        image = renderer.render(own, time, width, height, technique);
        drawnFrom = own;
    } else if (args.contains("eye") && args.contains("target")) {
        const std::vector<double> eye = args["eye"].get<std::vector<double>>();
        const std::vector<double> at = args["target"].get<std::vector<double>>();
        if (eye.size() != 3 || at.size() != 3) {
            return textAnswer("'eye' and 'target' want three numbers each.", true);
        }
        const std::vector<double> up =
            args.contains("up") ? args["up"].get<std::vector<double>>() : std::vector<double>{0.0, 1.0, 0.0};
        render::Camera own = render::Camera::lookingAt({eye[0], eye[1], eye[2]}, {at[0], at[1], at[2]},
                                                       {up[0], up[1], up[2]});
        own.lens.focal = args.value("focal", 35.0);
        // A diaphragm, where one is asked for: the raster spreads each splat's
        // covariance by its circle of confusion, the path tracer samples the
        // disk, and both focus at `focus`.
        own.lens.fStop = args.value("fStop", 0.0);
        own.lens.focusDistance = args.value("focus", 0.0);
        image = renderer.render(own, time, width, height, technique);
        drawnFrom = own;
    } else if (camera.empty() && renderer.cameras().empty()) {
        auto framed = renderer.framingCamera(time, args.value("focal", 35.0), technique);
        if (!framed) return textAnswer(errorText(framed.error()), true);
        image = renderer.render(*framed, time, width, height, technique);
        drawnFrom = *framed;
    } else {
        image = renderer.render(camera, time, width, height, technique);
    }
    if (!image) return textAnswer(errorText(image.error()), true);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    session.lastWidth = width;
    session.lastHeight = height;
    session.lastTime = time;
    session.lastCamera = camera;

    std::ostringstream out;
    out << "Rendered " << session.stage.filename().string() << " at " << width << "x" << height << ", " << technique
        << ", time " << time << " in " << std::fixed << ms << " ms";
    // THE FRAME FOR A PROGRAM THAT COMPOSITES IT (the OpenFX plugin): its
    // own numbers, no picture of them. `raw` names a file the caller reads
    // back -- the colour, four floats a pixel, then the depth, one a pixel,
    // both bottom row first, linear and premultiplied, as an OpenFX image
    // lies -- and the answer is text alone, since a PNG of the frame is time
    // the caller would wait for and throw away. After them come the `layers`
    // asked for, four floats a pixel each in the order asked: P (where each
    // pixel lies in the world, alpha 1 where something drew) and the
    // Cryptomatte's layers as a Hydra buffer holds them, (id, coverage, id,
    // coverage). A frame with a matte answers its manifest on a line of its
    // own, the names and ids an EXR's metadata would carry.
    if (args.contains("raw")) {
        const std::string raw = args["raw"].get<std::string>();
        std::ofstream file(raw, std::ios::binary | std::ios::trunc);
        file.write(reinterpret_cast<const char*>(image->rgba.data()),
                   static_cast<std::streamsize>(image->rgba.size() * sizeof(float)));
        file.write(reinterpret_cast<const char*>(image->depth.data()),
                   static_cast<std::streamsize>(image->depth.size() * sizeof(float)));
        const size_t layerBytes = size_t{width} * height * 4 * sizeof(float);
        bool matte = false;
        for (const std::string& layer : layers) {
            if (layer == "P") {
                if (!drawnFrom) {
                    return textAnswer("layer P needs the tool's own camera ('xform', 'eye'): a stage camera's "
                                      "projection is the delegate's.", true);
                }
                auto position = renderer.worldPosition(*drawnFrom);
                if (!position) return textAnswer(errorText(position.error()), true);
                file.write(reinterpret_cast<const char*>(position->data()), static_cast<std::streamsize>(layerBytes));
                continue;
            }
            auto bytes = renderer.mappedOutput(layer);
            if (!bytes) return textAnswer(errorText(bytes.error()), true);
            if (bytes->size() != layerBytes) {
                return textAnswer("layer " + layer + " is not four floats a pixel", true);
            }
            file.write(reinterpret_cast<const char*>(bytes->data()), static_cast<std::streamsize>(layerBytes));
            matte = true;
        }
        if (!file) {
            return textAnswer("could not write the frame to " + raw, true);
        }
        out << "\nraw " << raw;
        if (matte) {
            json manifest = json::object();
            for (const auto& [name, id] : renderer.cryptoManifest()) {
                manifest[name] = core::hex8(id);
            }
            out << "\nmanifest " << manifest.dump();
        }
        return textAnswer(out.str());
    }
    const std::string aov = args.value("aov", std::string("color"));
    const std::string file = args.value("output", std::string{});
    const bool wantsPng = file.size() > 4 && file.compare(file.size() - 4, 4, ".png") == 0;
    auto png = preview(renderer, aov, width, height, args.value("view", std::string("agx")),
                       static_cast<float>(args.value("exposure", 0.0)), wantsPng ? file : std::string{});
    // WHAT `output` WRITES, and why a PNG is not the same file as an EXR.
    //
    // An EXR is the frame's own numbers, linear and whatever the beauty holds.
    // A PNG is the frame **as it is shown**: the AOV that was asked for, put
    // through the view transform -- which is the only way to keep a
    // Cryptomatte's colours, a normal's or a depth's, since none of them is a
    // colour and the beauty's numbers are not theirs. A caller shooting a
    // sequence of an AOV asks for `.png`; one keeping the light asks for
    // `.exr`.
    if (!file.empty()) {
        if (wantsPng) {
            out << (png ? "\nwrote " + file : "\nPNG not written: " + errorText(png.error()));
        } else if (auto written = io::writeExr(file, width, height, image->rgba, image->depth, false);
                   !written) {
            out << "\nEXR not written: " << errorText(written.error());
        } else {
            out << "\nwrote " << file;
        }
    }
    if (!png) {
        out << "\nno preview: " << errorText(png.error());
        return textAnswer(out.str());
    }
    return imageAnswer(out.str(), *png);
}

json pickTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    if (session.lastWidth == 0) {
        return textAnswer("nothing has been rendered yet: render first, then pick a pixel of that frame.", true);
    }
    const uint32_t x = args.value("x", 0u);
    const uint32_t y = args.value("y", 0u);
    auto picked = session.renderer->pick(x, y);
    if (!picked) return textAnswer(errorText(picked.error()), true);
    if (!picked->has_value()) {
        return textAnswer("nothing at (" + std::to_string(x) + ", " + std::to_string(y) + ")");
    }
    std::ostringstream out;
    out << "prim " << (*picked)->prim << "\nrprim " << (*picked)->rprim;
    if ((*picked)->instance >= 0) {
        out << "\ninstance " << (*picked)->instance;
    }
    return textAnswer(out.str());
}

json boundsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    auto bounds = session.renderer->bounds();
    if (!bounds) return textAnswer(errorText(bounds.error()), true);
    if (!bounds->has_value()) {
        return textAnswer("nothing drawn yet, so nothing has bounds: render first.");
    }
    const scene::Bounds& b = **bounds;
    std::ostringstream out;
    out << "min " << b.min[0] << ", " << b.min[1] << ", " << b.min[2] << "\n";
    out << "max " << b.max[0] << ", " << b.max[1] << ", " << b.max[2];
    return textAnswer(out.str());
}

json productsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    const std::string prim = args.value("settings", std::string{});
    if (prim.empty()) {
        return textAnswer("render_products wants 'settings': a UsdRenderSettings prim path.", true);
    }
    auto written = session.renderer->renderProducts(prim, args.value("time", 0.0),
                                                    args.value("directory", std::string{}));
    if (!written) return textAnswer(errorText(written.error()), true);
    std::ostringstream out;
    out << "wrote " << written->size() << " file(s)";
    for (const std::filesystem::path& file : *written) {
        out << "\n  " << file.string();
    }
    return textAnswer(out.str());
}

json convertTool(Session& session, const json& args) {
    const std::string input = args.value("input", std::string{});
    const std::string output = args.value("output", std::string{});
    if (input.empty() || output.empty()) {
        return textAnswer("convert wants 'input' (a .ply, .splat, .spz or .sog) and 'output' (a .usda or .usdc).",
                          true);
    }
    // A capture becomes a stage on the device this session already opened,
    // which is also why the answer can say how many splats survived the read.
    if (!session.open()) {
        return textAnswer("convert needs a device, which the server opens with a stage: open_stage first (any "
                          "stage will do).", true);
    }
    auto raw = io::readSplats(input);
    if (!raw) return textAnswer(errorText(raw.error()), true);
    usd::ExportOptions options;
    options.maxDegree = args.value("degree", 3u);
    options.addCamera = args.value("camera", true);
    options.rotateXDegrees = args.value("rotateX", 0.0);
    if (auto written = usd::writeParticleFieldStage(session.renderer->library(), *raw, output, options); !written) {
        return textAnswer(errorText(written.error()), true);
    }
    std::ostringstream out;
    out << "wrote " << output << "\n" << raw->count << " splats from " << input;
    return textAnswer(out.str());
}

/// What a frame costs, measured the way the CLI measures it: the median of
/// several, since the first pays for every kernel the frame needs.
json timingsTool(Session& session, const json& args) {
    if (auto ready = ensureStage(session, args); !ready) return textAnswer(errorText(ready.error()), true);
    applySettings(session, args);
    usd::StageRenderer& renderer = *session.renderer;
    const uint32_t width = args.value("width", 640u);
    const uint32_t height = args.value("height", 360u);
    const uint32_t frames = std::max(args.value("frames", 5u), 1u);
    const std::string technique = args.value("technique", std::string("raster"));
    const std::string camera = args.value("camera", std::string{});
    // The same camera rules a render follows, so a timing is of the frame a
    // caller would get: a stage without a camera is framed by one of ours.
    std::optional<render::Camera> framing;
    if (camera.empty() && renderer.cameras().empty()) {
        auto framed = renderer.framingCamera(args.value("time", 0.0), args.value("focal", 35.0), technique);
        if (!framed) return textAnswer(errorText(framed.error()), true);
        framing = *framed;
    }
    std::vector<double> ms;
    for (uint32_t k = 0; k < frames; ++k) {
        const auto start = std::chrono::steady_clock::now();
        auto image = framing.has_value()
                         ? renderer.render(*framing, args.value("time", 0.0), width, height, technique)
                         : renderer.render(camera, args.value("time", 0.0), width, height, technique);
        if (!image) return textAnswer(errorText(image.error()), true);
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(ms.begin(), ms.end());
    std::ostringstream out;
    out << frames << " frames at " << width << "x" << height << ", " << technique << "\n";
    out << "median " << ms[ms.size() / 2] << " ms, first " << ms.front() << ", last " << ms.back()
        << " (sorted; the first frame pays for the kernels the frame needs)";
    return textAnswer(out.str());
}

/// What the session holds, so a caller need not remember what it set.
json settingsTool(Session& session, const json& args) {
    if (session.open()) {
        applySettings(session, args);
    }
    std::ostringstream out;
    if (!session.open()) {
        out << "no stage open\n";
        return textAnswer(out.str());
    }
    out << "stage          " << session.stage.string() << "\n";
    out << "last frame     " << session.lastWidth << "x" << session.lastHeight << " at time " << session.lastTime
        << "\n";
    const std::pair<double, double> range = session.renderer->animationRange();
    out << "animated over  " << range.first << ".." << range.second << "\n";
    for (const usd::StageVariantSet& set : session.renderer->variantSets()) {
        out << "variant set    " << set.prim << "{" << set.name << "} = " << set.selected << " of "
            << set.variants.size() << "\n";
    }
    out << "cloud shadows  a map a light, read with no ray: cloudShadows, cloudShadowTexels, cloudShadowTerms, "
           "cloudShadowDensity\n";
    out << "lens           fStop and focus on a camera of your own: depth of field in both techniques\n";
    out << "AOVs a render can show: color, depth, primId, instanceId, elementId, Neye, normal, albedo, "
           "shadingNormal, primvars:<name>, lightGroup:<name>\n";
    out << "view transforms for the image: agx, aces, standard";
    return textAnswer(out.str());
}

}   // namespace

// --- live streams from another process ----------------------------------------
//
// A producer -- a Python script, a host application -- asks for a ring of
// shared memory, writes SPZ pages into it, and commits an epoch by naming
// where each page lies. The pages are copied out of the ring onto the stream's
// queue before the answer, so the producer may write over them as soon as it
// has it. This thread only enqueues; the stage's render thread decodes.

json streamOpen(Session& session, const json& args) {
    const std::string name = args.value("name", "");
    if (name.empty()) {
        return textAnswer("stream_open wants 'name': the stream an AtheneaLiveSourceAPI prim names.", true);
    }
    if (session.streams.count(name) != 0) {
        return textAnswer("stream '" + name + "' is open already; stream_close it first.", true);
    }
    const uint64_t bytes = args.value("ringBytes", uint64_t{64} << 20);
    static uint32_t serial = 0;
    char ring[32];
    std::snprintf(ring, sizeof ring, "athlive-%08x-%u", core::murmurHash3x86_32(name), ++serial);
    auto memory = platform::SharedMemory::create(ring, static_cast<size_t>(bytes));
    if (!memory) return textAnswer(errorText(memory.error()), true);
    LiveStream stream;
    stream.ring = std::move(*memory);
    stream.queue = lod::PageQueue::named(name);
    const std::string sync = args.value("sync", "");
    if (!sync.empty() && sync != "latest" && sync != "clock") {
        return textAnswer("stream_open: 'sync' is latest or clock, not '" + sync + "'.", true);
    }
    stream.queue->setSyncHint(sync == "clock" ? 2 : sync == "latest" ? 1 : 0);
    stream.chunks = args.value("chunks", 64u);
    session.streams.emplace(name, std::move(stream));
    const json answer{{"ring", ring}, {"bytes", bytes}};
    return textAnswer(answer.dump());
}

json streamCommit(Session& session, const json& args) {
    const std::string name = args.value("name", "");
    const auto found = session.streams.find(name);
    if (found == session.streams.end()) {
        return textAnswer("no stream '" + name + "': stream_open it first.", true);
    }
    LiveStream& stream = found->second;
    const uint64_t epoch = args.value("epoch", uint64_t{0});
    const json& pages = args.contains("pages") ? args["pages"] : json::array();
    const uint32_t epochPages = args.value("epochPages", static_cast<uint32_t>(pages.size()));
    if (epoch == 0 || !pages.is_array() || pages.empty() || epochPages == 0) {
        return textAnswer("stream_commit wants an 'epoch' from 1 and 'pages'.", true);
    }
    const std::span<std::byte> ring = stream.ring.bytes();
    std::vector<lod::PageUpdate> updates;
    for (const json& page : pages) {
        lod::PageUpdate u;
        u.stream = 0;
        u.epoch = epoch;
        u.pages = epochPages;
        u.chunk = page.value("chunk", 0u);
        u.ptpTime = page.value("ptpTime", uint64_t{0});
        const std::string op = page.value("op", "replace");
        u.op = op == "remove" ? lod::PageUpdate::Op::Remove : op == "add" ? lod::PageUpdate::Op::Add
                                                                           : lod::PageUpdate::Op::Replace;
        if (u.chunk >= stream.chunks) {
            return textAnswer("chunk " + std::to_string(u.chunk) + " is past the stream's " +
                                  std::to_string(stream.chunks) + "; nothing of the epoch was queued.",
                              true);
        }
        if (page.contains("bounds") && page["bounds"].is_array() && page["bounds"].size() == 6) {
            for (size_t k = 0; k < 3; ++k) {
                u.bounds.min[k] = page["bounds"][k].get<float>();
                u.bounds.max[k] = page["bounds"][k + 3].get<float>();
            }
        }
        if (u.op != lod::PageUpdate::Op::Remove) {
            const uint64_t offset = page.value("offset", uint64_t{0});
            const uint64_t size = page.value("bytes", uint64_t{0});
            if (size == 0 || offset > ring.size() || size > ring.size() - offset) {
                return textAnswer("a page of chunk " + std::to_string(u.chunk) +
                                      " lies outside the ring; nothing of the epoch was queued.",
                                  true);
            }
            u.bytes.assign(ring.begin() + static_cast<std::ptrdiff_t>(offset),
                           ring.begin() + static_cast<std::ptrdiff_t>(offset + size));
        }
        updates.push_back(std::move(u));
    }
    for (lod::PageUpdate& u : updates) {
        stream.queue->push(std::move(u));
    }
    return textAnswer("queued " + std::to_string(updates.size()) + " pages of epoch " + std::to_string(epoch));
}

json streamClose(Session& session, const json& args) {
    const std::string name = args.value("name", "");
    if (session.streams.erase(name) == 0) {
        return textAnswer("no stream '" + name + "'.", true);
    }
    return textAnswer("closed '" + name + "'");
}

json textAnswer(std::string text, bool isError) {
    json answer{{"content", json::array({json{{"type", "text"}, {"text", std::move(text)}}})}};
    if (isError) {
        answer["isError"] = true;
    }
    return answer;
}

const std::vector<ToolEntry>& toolTable() {
    static const std::vector<ToolEntry> table = [] {
        std::vector<ToolEntry> tools;
        tools.push_back({"open_stage",
                         "Open a USD stage and keep it open. Answers with its cameras, up axis and time range.",
                         object({{"stage", field("string", "path to a .usd, .usda or .usdc")}}, {"stage"}),
                         openStage});
        tools.push_back({"stage_tree", "What is under a prim path ('/' for the root): name, type, whether it has children.",
                         object({{"path", field("string", "prim path, '/' by default")},
                                 {"stage", field("string", "open this stage first")}}),
                         stageTree});
        tools.push_back({"variants",
                         "The variant sets the open stage carries -- USD's \"pick one of these\" -- and a selection "
                         "made on one of them. An asset may package its animations, its levels of detail or its "
                         "looks this way; selecting one recomposes the stage for every render that follows.",
                         object({{"select", field("string", "a selection as USD writes one in a path: "
                                                            "/World{clip=air_fly_A0}")},
                                 {"all", field("boolean", "list every variant, however many there are")},
                                 {"stage", field("string", "open this stage first")}}),
                         variantsTool});
        tools.push_back({"device_info", "The GPU the engine opened and what it can do: rasterisation, inline rays, ray tracing pipelines, OptiX, the denoiser, OCIO.",
                         object({}), deviceInfo});
        tools.push_back(
            {"render",
             "Render the open stage and answer with the image and its timing. 'technique' is raster or rt (path "
             "traced); path tracing draws until it holds 'pathTotal' paths. Every setting is remembered for the "
             "next call.",
             object({{"stage", field("string", "open this stage first")},
                     {"camera", field("string", "a camera prim; omitted, the first, or a framing camera")},
                     {"width", field("integer", "pixels across, 640 by default")},
                     {"height", field("integer", "pixels down, 360 by default")},
                     {"time", field("number", "USD time code")},
                     {"technique", field("string", "raster | rt")},
                     {"aov", field("string", "which AOV the image shows: color, depth, primId, instanceId, elementId, Neye, normal, albedo, cryptomatte, CryptoObject00..02")},
                     {"aovs", json{{"type", "array"}, {"items", {{"type", "string"}}},
                                   {"description", "AOVs to compute this frame"}}},
                     {"view", field("string", "view transform for the image: agx (default), aces, standard")},
                     {"exposure", field("number", "stops, for the image only")},
                     {"output", field("string", "also write the frame to this file: an EXR of the beauty's own numbers, or, where the name ends in .png, the frame as it is shown -- the AOV asked for, through the view transform")},
                     {"eye", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                  {"description", "a camera of your own: where it is (x y z), with 'target'"}}},
                     {"target", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                     {"description", "where that camera looks"}}},
                     {"up", json{{"type", "array"}, {"items", {{"type", "number"}}}, {"description", "its up vector"}}},
                     {"focal", field("number", "focal length, mm, of a camera of your own or a framing one")},
                     {"xform", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                    {"description", "a camera of your own as a matrix: camera to world, sixteen "
                                                    "numbers row by row, looking down -z with +y up"}}},
                     {"haperture", field("number", "that camera's horizontal aperture, mm (24.576)")},
                     {"vaperture", field("number", "its vertical aperture, mm (the horizontal one at the image's aspect)")},
                     {"near", field("number", "its near clipping plane, scene units")},
                     {"far", field("number", "its far clipping plane, scene units")},
                     {"sky", field("string", "an image for every dome of the stage (a lat-long HDR or EXR); a "
                                             "stage with no dome gets the default dome and sun; empty gives back "
                                             "the stage's own")},
                     {"skyRotation", field("number", "degrees every dome turns about the stage's up axis")},
                     {"skyVisible", field("boolean", "false leaves the background transparent; the sky still lights")},
                     {"splatView", field("string", "a parameter of the gaussians drawn in place of their light: light "
                                                   "(the default), albedo, roughness, metallic, transmission, ior, "
                                                   "normal, opacity, size, skyVisibility, bounce, openCells, "
                                                   "emission, field, baked, materialRoughness")},
                     {"layers", json{{"type", "array"}, {"items", {{"type", "string"}}},
                                     {"description", "with 'raw': layers written after the depth, four floats a "
                                                     "pixel each, in this order: P (world position, from the "
                                                     "tool's own camera), CryptoObject00..02"}}},
                     {"raw", field("string", "write the frame's own numbers to this file instead of answering with "
                                             "an image: RGBA floats then depth floats, bottom row first, linear "
                                             "and premultiplied")},
                     {"lightSamples", field("integer", "samples per light per pixel")},
                     {"chooseLights", field("boolean", "one light a sample, chosen by power")},
                     {"splatShadows", field("boolean", "a relit splat cloud shadows itself")},
                     {"splatReflections", field("boolean", "rt: a gaussian reflects the cloud it belongs to rather than only the sky, at one ray each")},
                     {"cloudShadows", field("boolean", "the frame's clouds shadow its meshes and each other, "
                                                       "from a map built at each light -- no ray, on by default")},
                     {"cloudShadowTexels", field("integer", "texels a side of that map, per light (1024)")},
                     {"cloudShadowTerms", field("integer", "terms a texel keeps: 1 the total optical depth (a "
                                                           "floor under a cloud), 3/5/7 with Fourier pairs so a "
                                                           "cloud shadows itself; 0 chooses by what receives")},
                     {"cloudShadowDensity", field("number", "what the cloud's optical depth is multiplied by: "
                                                            "1 is what its opacity says, less lets light "
                                                            "through it")},
                     {"fStop", field("number", "a diaphragm for a camera of your own: depth of field at "
                                               "'focus' (0 is a pinhole)")},
                     {"focus", field("number", "the depth in focus, scene units")},
                     {"pathSamples", field("integer", "rt: paths a pixel each pass")},
                     {"pathBounces", field("integer", "rt: bounces after the first hit")},
                     {"pathTotal", field("integer", "rt: paths a pixel the image is drawn until it holds")},
                     {"denoise", field("boolean", "rt: denoise once the total is reached (OIDN)")},
                     {"defaultLights", field("boolean", "a dome and a sun for a stage that carries no light")},
                     {"motionBuckets", field("integer", "rt: shutter slices")},
                     {"shutter", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                      {"description", "[open, close] in frames: what a cloud's motion blur is "
                                                      "integrated over (0.5 is a 180 degree shutter)"}}},
                     {"variant", field("string", "a variant selection to make first: /World{clip=air_fly_A0}")},
                     {"refine", field("integer", "subdivision levels")},
                     {"visibility", field("string", "how meshes are found: automatic | raster | rays | bvh")}}),
             renderTool});
        tools.push_back({"pick", "What the last frame drew at a pixel: the prim, and which instance of it.",
                         object({{"x", field("integer", "from the left")}, {"y", field("integer", "from the top")}},
                                {"x", "y"}),
                         pickTool});
        tools.push_back({"bounds", "Where what the last frame drew is, in world space.", object({}), boundsTool});
        tools.push_back({"render_products",
                         "Render a UsdRenderSettings prim's products: every render var a layer of its EXR.",
                         object({{"settings", field("string", "the settings prim path")},
                                 {"time", field("number", "USD time code")},
                                 {"directory", field("string", "where the files go")}},
                                {"settings"}),
                         productsTool});
        tools.push_back({"convert",
                         "Turn a splat capture into a USD stage this server can open. The values are computed on "
                         "the device the session already has.",
                         object({{"input", field("string", "a .ply, .splat, .spz or .sog")},
                                 {"output", field("string", "a .usda or .usdc")},
                                 {"degree", field("integer", "harmonic degree cap, 0..3")},
                                 {"rotateX", field("number", "turn the cloud about x (COLMAP captures: 180)")},
                                 {"camera", field("boolean", "add /World/Camera framing it")}},
                                {"input", "output"}),
                         convertTool});
        tools.push_back({"timings",
                         "What a frame costs: several frames of the open stage, and the median of them.",
                         object({{"frames", field("integer", "how many, 5 by default")},
                                 {"width", field("integer", "pixels across")},
                                 {"height", field("integer", "pixels down")},
                                 {"technique", field("string", "raster | rt")},
                                 {"camera", field("string", "a camera prim")},
                                 {"time", field("number", "USD time code")},
                                 {"shutter", json{{"type", "array"}, {"items", {{"type", "number"}}},
                                                  {"description", "[open, close] in frames: what a cloud's motion "
                                                                  "blur is integrated over"}}},
                                 {"variant", field("string", "a variant selection: /World{clip=air_fly_A0}")}}),
                         timingsTool});
        tools.push_back({"settings",
                         "What this session holds: the stage, the last frame, and what a render can be asked for.",
                         object({}), settingsTool});
        tools.push_back({"stream_open",
                         "Open a live stream another process feeds: answers with the shared-memory ring "
                         "({\"ring\", \"bytes\"}) to write SPZ pages into. The stage's AtheneaLiveSourceAPI prim "
                         "names the same stream.",
                         object({{"name", field("string", "the stream's name, as primvars:athenea:live:stream says")},
                                 {"chunks", field("integer", "chunk ids a page may name, 64 by default")},
                                 {"ringBytes", field("integer", "the ring's size in bytes, 64 MiB by default")},
                                 {"sync", field("string", "latest (default) | clock: which epoch a frame shows, where the stage does not say")}},
                                {"name"}),
                         streamOpen});
        tools.push_back({"stream_commit",
                         "Queue one epoch's pages, each an SPZ file the producer wrote into the ring. The bytes "
                         "are copied before the answer: the ring may be written over once it comes.",
                         object({{"name", field("string", "the stream")},
                                 {"epoch", field("integer", "from 1, increasing")},
                                 {"epochPages", field("integer", "pages the epoch has in all (this call's, by default)")},
                                 {"pages", json{{"type", "array"},
                                                {"description", "{chunk, op: add|replace|remove, offset, bytes, "
                                                                "bounds: [minx, miny, minz, maxx, maxy, maxz], ptpTime}"},
                                                {"items", {{"type", "object"}}}}}},
                                {"name", "epoch", "pages"}),
                         streamCommit});
        tools.push_back({"stream_close", "Close a stream this session opened and remove its ring.",
                         object({{"name", field("string", "the stream")}}, {"name"}), streamClose});
        return tools;
    }();
    return table;
}

}   // namespace athenea::mcp
