// Copyright (c) 2026 jesus luque.
//
// `athenea convert` and `athenea render --stage`: USD in and out.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Readers.h"
#include "athenea/lod/Athc.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {

void addConvert(CLI::App& app) {
    struct Options {
        std::string input, output;
        unsigned degree = 3;
        double rotateX = 0.0;
        bool noCamera = false;
        uint32_t chunkSplats = uint32_t{1} << 16;
        float maxGroupFraction = 0.5F;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("convert",
                                   "a splat file into a USD ParticleField stage, or into a .athc with levels of detail");
    cmd->add_option("input", o->input, ".ply / .splat / .spz / .sog / meta.json")->required();
    cmd->add_option("output", o->output, ".usda / .usdc / .usd / .athc")->required();
    cmd->add_option("--chunk-splats", o->chunkSplats, ".athc: splats per streamed chunk");
    cmd->add_option("--max-group-fraction", o->maxGroupFraction,
                    ".athc: the finest merged level holds at most this many groups per splat");
    cmd->add_option("--degree", o->degree, "harmonic degree cap 0..3");
    cmd->add_option("--rotate-x", o->rotateX, "turn the cloud about x (COLMAP clouds: 180)");
    cmd->add_flag("--no-camera", o->noCamera, "do not add /World/Camera");
    cmd->callback([o] {
        auto device = gpu::Device::create();
        if (!device) {
            std::fprintf(stderr, "%s\n", device.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        gpu::ShaderLibrary library(*device);
        auto loader = scene::CloudLoader::create(library);
        if (!loader) {
            std::fprintf(stderr, "%s\n", loader.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        if (lod::isAthc(o->output)) {
            if (o->rotateX != 0.0) {
                std::fprintf(stderr, "a .athc keeps the cloud as it is: turn it where it is used\n");
                throw CLI::RuntimeError(1);
            }
            auto splats = scene::loadSplatFile(*loader, o->input, o->degree);
            auto builder = splats ? lod::LodBuilder::create(library) : Result<lod::LodBuilder>(splats.error());
            if (!builder) {
                std::fprintf(stderr, "%s\n", builder.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            lod::LodBuildSettings settings;
            settings.chunkSplats = o->chunkSplats;
            settings.maxGroupFraction = o->maxGroupFraction;
            auto built = builder->build(*splats, settings);
            auto written = built ? lod::writeAthc(**device, *built, o->output) : Result<void>(built.error());
            if (!written) {
                std::fprintf(stderr, "%s\n", written.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            std::printf("wrote %s\n", o->output.c_str());
            return;
        }
        auto raw = scene::readSplatRecords(*loader, o->input, o->degree);
        if (!raw) {
            std::fprintf(stderr, "%s\n", raw.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        usd::ExportOptions options;
        options.maxDegree = o->degree;
        options.addCamera = !o->noCamera;
        // metersPerUnit stays 1: no splat file says what its unit is, and a
        // capture's scale is taken as metres (operations.md, 2.4).
        options.rotateXDegrees = o->rotateX;
        if (auto written = usd::writeParticleFieldStage(library, *raw, o->output, options); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("wrote %s\n", o->output.c_str());
    });
}

void addStage(CLI::App& app) {
    struct Options {
        std::string stage, camera, output = "out.exr", size = "1920x1080", technique = "raster",
                    visibility = "automatic", renderSettings;
        double time = 0.0;
        std::vector<double> eye, target, up{0.0, 1.0, 0.0};
        // 0: the near plane the camera derives from how far it stands.
        double focal = 35.0, nearZ = 0.0, farZ = 100000.0;
        double fStop = 0.0, focus = 0.0;
        uint32_t frames = 1;
        uint32_t pathSamples = 1, pathBounces = 1, pathTotal = 1, motionBuckets = 4, refine = 0;
        std::string shutter;
        std::vector<std::string> variants;
        std::vector<std::string> splatOverrides;
        uint32_t lightSamples = 1;
        bool denoise = false, frameAll = false, splatShadows = false, defaultLights = false;
        bool transferIndirect = true;
        bool splatReflections = false;
        bool cloudShadows = true;
        bool antialias = true;
        uint32_t cloudShadowTexels = 1024;
        uint32_t cloudShadowTerms = 0;
        double cloudShadowDensity = 1.0;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("stage", "render a USD stage through the engine's Hydra delegate");
    cmd->add_option("stage", o->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", o->camera, "camera prim path (default: the first)");
    cmd->add_option("--time", o->time, "USD time code");
    cmd->add_option("--size", o->size, "WIDTHxHEIGHT");
    cmd->add_option("--technique", o->technique, "raster | rt (the delegate's athenea:technique setting)");
    cmd->add_option("--visibility", o->visibility, "how meshes are seen: automatic | raster | rays | bvh");
    cmd->add_option("--path-samples", o->pathSamples, "rt: paths a pixel each pass");
    cmd->add_option("--path-bounces", o->pathBounces, "rt: bounces after the first hit");
    cmd->add_option("--path-total", o->pathTotal, "rt: paths a pixel the image is drawn until it holds");
    cmd->add_flag("--denoise", o->denoise, "rt: denoise the image once it holds its total (OIDN)");
    cmd->add_flag("--default-lights", o->defaultLights,
                  "a dome and a sun in the session layer, for a stage that brings no lights "
                  "(what athenea view offers)");
    cmd->add_option("--shutter", o->shutter,
                    "OPEN:CLOSE in frames, for a camera of our own (--eye): what a cloud's motion "
                    "blur is integrated over. A stage camera's own shutter is taken without it");
    cmd->add_option("--variant", o->variants,
                    "a variant selection, as USD writes one in a path: /World{clip=air_fly_A0}. "
                    "Repeatable, applied in order, in the session layer");
    cmd->add_option("--splat-override", o->splatOverrides,
                    "PRIM=METALLIC,ROUGHNESS,TRANSMISSION[,R,G,B[,REPLACE]]: what the gaussians that came from "
                    "PRIM (a path the cloud's Cryptomatte manifest names, or * for all of them) are "
                    "made of in this frame; -1 leaves a value as the gaussian carries it, and REPLACE 1 makes "
                    "R,G,B the colour instead of a tint on it. Repeatable");
    cmd->add_option("--motion-buckets", o->motionBuckets,
                    "rt: shutter slices for motion blur, 1 to 8 (the shutter is the camera's)");
    cmd->add_option("--refine", o->refine, "subdivision surfaces refined this many levels (0: the control mesh)");
    cmd->add_option("--light-samples", o->lightSamples, "samples per light per pixel");
    cmd->add_flag("--splat-shadows", o->splatShadows,
                  "a relit cloud shadows itself: one ray a splat against its own proxies");
    cmd->add_flag("--splat-reflections", o->splatReflections,
                  "rt: a gaussian reflects the cloud it belongs to rather than only the prepared sky, "
                  "at one ray each -- a collar under a glass ball reflects the ball");
    cmd->add_flag("!--no-transfer-indirect", o->transferIndirect,
                  "a cloud that carries a transfer is drawn without the indirect half of it -- the "
                  "light it bounced off the scene it was converted in (it is added by default), so "
                  "what that half is worth can be seen on the same file");
    cmd->add_flag("!--no-antialias", o->antialias,
                  "every ray through the middle of its pixel: a gathered frame is not averaged over "
                  "the pixel's area (it is by default)");
    cmd->add_flag("!--no-cloud-shadows", o->cloudShadows,
                  "the frame's clouds do not shadow its meshes (they do by default: a map from each "
                  "light, no ray, so it works where nothing can be traced)");
    cmd->add_option("--cloud-shadow-texels", o->cloudShadowTexels,
                    "texels a side of that map, per light (1024)");
    cmd->add_option("--cloud-shadow-density", o->cloudShadowDensity,
                    "what the cloud's optical depth is multiplied by: 1 is what its opacity says, "
                    "less lets light through it, more darkens it");
    cmd->add_option("--cloud-shadow-terms", o->cloudShadowTerms,
                    "terms a texel keeps: 1 the total optical depth (a floor under a cloud), 3/5/7 "
                    "with Fourier pairs so a cloud shadows itself. 0: chosen by what receives");
    cmd->add_option("--eye", o->eye, "a camera of its own at x y z (with --target), not one on the stage")->expected(3);
    cmd->add_option("--target", o->target, "where that camera looks")->expected(3);
    cmd->add_option("--up", o->up, "its up vector")->expected(3);
    cmd->add_flag("--frame-all", o->frameAll,
                  "a camera of its own framing what the stage draws, as athenea view opens (the default on a stage "
                  "without cameras)");
    cmd->add_option("--focal", o->focal, "its focal length, mm (24.576 mm aperture)");
    cmd->add_option("--fstop", o->fStop,
                    "its diaphragm: depth of field at --focus (0 is a pinhole). The raster spreads each "
                    "splat by its circle of confusion; the path tracer samples the disk");
    cmd->add_option("--focus", o->focus, "the depth in focus, scene units");
    cmd->add_option("--near", o->nearZ, "its near clipping distance");
    cmd->add_option("--far", o->farZ, "its far clipping distance");
    cmd->add_option("-o,--output", o->output, "EXR path");
    cmd->add_option("--render-settings", o->renderSettings,
                    "a UsdRenderSettings prim: render its products, each var a layer of its EXR, and stop");
    cmd->add_option("--frames", o->frames,
                    "render this many times and print the time a frame takes (Hydra sync, drawing and the readback)");
    cmd->callback([o] {
        uint32_t width = 0, height = 0;
        if (std::sscanf(o->size.c_str(), "%ux%u", &width, &height) != 2) {
            std::fprintf(stderr, "--size wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(1);
        }
        auto renderer = usd::StageRenderer::open(o->stage);
        if (!renderer) {
            std::fprintf(stderr, "%s\n", renderer.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        for (const std::string& selection : o->variants) {
            if (auto set = (*renderer)->setVariantSelection(selection); !set) {
                std::fprintf(stderr, "%s\n", set.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
        }
        if (auto set = (*renderer)->setMeshVisibility(o->visibility); !set) {
            std::fprintf(stderr, "%s\n", set.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        (*renderer)->setPathSamples(o->pathSamples);
        (*renderer)->setPathBounces(o->pathBounces);
        (*renderer)->setPathTotal(o->pathTotal);
        (*renderer)->setDenoise(o->denoise);
        (*renderer)->setMotionBuckets(o->motionBuckets);
        if (!o->shutter.empty()) {
            double open = 0.0;
            double close = 0.0;
            if (std::sscanf(o->shutter.c_str(), "%lf:%lf", &open, &close) != 2) {
                std::fprintf(stderr, "--shutter wants OPEN:CLOSE, in frames\n");
                throw CLI::RuntimeError(1);
            }
            (*renderer)->setShutter(open, close);
        }
        (*renderer)->setRefineLevel(o->refine);
        (*renderer)->setLightSamples(o->lightSamples);
        (*renderer)->setSplatShadows(o->splatShadows);
        (*renderer)->setSplatTransferIndirect(o->transferIndirect);
        (*renderer)->setSplatReflections(o->splatReflections);
        (*renderer)->setCloudShadows(o->cloudShadows);
        (*renderer)->setCloudShadowResolution(o->cloudShadowTexels);
        (*renderer)->setCloudShadowTerms(o->cloudShadowTerms);
        (*renderer)->setCloudShadowDensity(static_cast<float>(o->cloudShadowDensity));
        (*renderer)->setAntialias(o->antialias);
        if (o->defaultLights) {
            if (auto lit = (*renderer)->setDefaultLights(true); !lit) {
                std::fprintf(stderr, "%s\n", lit.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
        }
        if (!o->renderSettings.empty()) {
            const std::filesystem::path directory =
                o->output.empty() ? std::filesystem::path() : std::filesystem::path(o->output).parent_path();
            auto written = (*renderer)->renderProducts(o->renderSettings, o->time, directory);
            if (!written) {
                std::fprintf(stderr, "%s\n", written.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            for (const std::filesystem::path& file : *written) {
                std::printf("wrote %s\n", file.string().c_str());
            }
            return;
        }
        std::optional<render::Camera> own;
        if (o->eye.size() == 3 && o->target.size() == 3) {
            render::Camera camera = render::Camera::lookingAt({o->eye[0], o->eye[1], o->eye[2]},
                                                              {o->target[0], o->target[1], o->target[2]},
                                                              {o->up[0], o->up[1], o->up[2]});
            camera.lens.focal = o->focal;
            camera.lens.fStop = o->fStop;
            camera.lens.focusDistance = o->focus;
            if (o->nearZ > 0.0) {
                camera.lens.nearZ = o->nearZ;
            }
            camera.lens.farZ = o->farZ;
            own = camera;
        } else if (o->frameAll || (o->camera.empty() && (*renderer)->cameras().empty())) {
            auto framed = (*renderer)->framingCamera(o->time, o->focal, o->technique);
            if (!framed) {
                std::fprintf(stderr, "%s\n", framed.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            own = *framed;
        }
        if (!o->splatOverrides.empty()) {
            // The manifest is the cloud's, known once the stage has been
            // committed and a matte asked of it: one small frame that asks,
            // then the rows by prim, and the frames after ask for nothing.
            (*renderer)->requestOutputs({"CryptoObject00"});
            auto warm = own ? (*renderer)->render(*own, o->time, 16, 16, "raster")
                            : (*renderer)->render(o->camera, o->time, 16, 16, "raster");
            if (!warm) {
                std::fprintf(stderr, "%s\n", warm.error().toString().c_str());
                throw CLI::RuntimeError(1);
            }
            const std::map<std::string, uint32_t> manifest = (*renderer)->cryptoManifest();
            (*renderer)->requestOutputs({});
            for (const std::string& said : o->splatOverrides) {
                const size_t equals = said.rfind('=');
                std::array<float, 7> v{-1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F, 0.0F};
                const int read = equals == std::string::npos
                                     ? 0
                                     : std::sscanf(said.c_str() + equals + 1, "%f,%f,%f,%f,%f,%f,%f", &v[0],
                                                   &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]);
                if (read != 3 && read != 6 && read != 7) {
                    std::fprintf(stderr, "--splat-override wants PRIM=METALLIC,ROUGHNESS,TRANSMISSION[,R,G,B[,REPLACE]]\n");
                    throw CLI::RuntimeError(1);
                }
                const std::string prim = said.substr(0, equals);
                size_t matched = 0;
                for (const auto& [path, id] : manifest) {
                    if (prim != "*" && path != prim) {
                        continue;
                    }
                    render::SplatOverride row;
                    row.id = id;
                    row.metallic = v[0];
                    row.roughness = v[1];
                    row.transmission = v[2];
                    row.tint = {v[3], v[4], v[5]};
                    row.replaceColour = v[6] > 0.5F;
                    (*renderer)->setSplatOverride(row);
                    ++matched;
                }
                if (matched == 0) {
                    std::fprintf(stderr, "--splat-override: no cloud names '%s' in its Cryptomatte manifest\n",
                                 prim.c_str());
                    throw CLI::RuntimeError(1);
                }
            }
        }
        Result<usd::StageImage> image = Error(ErrorCode::InvalidArgument, "no image");
        std::vector<double> ms;
        for (uint32_t frame = 0; frame < std::max(o->frames, uint32_t{1}); ++frame) {
            const auto start = std::chrono::steady_clock::now();
            if (own) {
                image = (*renderer)->render(*own, o->time, width, height, o->technique);
            } else {
                image = (*renderer)->render(o->camera, o->time, width, height, o->technique);
            }
            if (!image) {
                break;
            }
            ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        if (ms.size() > 1) {
            // The first frame loads the stage onto the device and compiles; the rest are steady.
            std::vector<double> steady(ms.begin() + 1, ms.end());
            std::sort(steady.begin(), steady.end());
            std::printf("first frame %.1f ms; then median %.2f ms, fastest %.2f ms over %zu frames\n", ms.front(),
                        steady[steady.size() / 2], steady.front(), steady.size());
        }
        if (!image) {
            std::fprintf(stderr, "%s\n", image.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        if (auto written = io::writeExr(o->output, width, height, image->rgba, image->depth); !written) {
            std::fprintf(stderr, "%s\n", written.error().toString().c_str());
            throw CLI::RuntimeError(1);
        }
        std::printf("wrote %s\n", o->output.c_str());
    });
}

}   // namespace athenea::cli
