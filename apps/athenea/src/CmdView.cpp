// Copyright (c) 2026 jesus luque.
//
// `athenea view`: a window onto a USD stage.
#include <cstdio>
#include <memory>
#include <string>

#include "Commands.h"
#include "athenea/view/Viewer.h"

namespace athenea::cli {

void addView(CLI::App& app) {
    auto options = std::make_shared<view::ViewOptions>();
    auto size = std::make_shared<std::string>("1600x900");
    auto* cmd = app.add_subcommand("view", "look at a USD stage in a window: free camera, stage cameras, outputs, picking");
    cmd->add_option("stage", options->stage, ".usd / .usda / .usdc")->required();
    cmd->add_option("--camera", options->camera, "start from this camera prim (default: a free camera framing the stage)");
    cmd->add_option("--technique", options->technique, "raster | rt");
    cmd->add_option("--visibility", options->visibility, "how meshes are seen: automatic | raster | rays | bvh");
    cmd->add_option("--size", *size, "the window's WIDTHxHEIGHT, in points");
    cmd->add_option("--frames", options->frames, "close after this many frames and print their timings");
    cmd->add_option("--light-samples", options->lightSamples, "samples per light per pixel (1 is interactive)");
    cmd->add_flag("--choose-lights", options->chooseLights, "one light a sample, chosen by power");
    cmd->add_option("--path-samples", options->pathSamples, "rt: paths a pixel each frame");
    cmd->add_option("--path-bounces", options->pathBounces, "rt: bounces after the first hit");
    cmd->add_option("--path-total", options->pathTotal, "rt: paths a pixel at which the frame counts as converged (and is denoised)");
    cmd->add_flag("--denoise", options->denoise, "rt: denoise once converged (OIDN)");
    cmd->add_flag("!--no-default-lights", options->defaultLights,
                  "a stage with no lights of its own stays unlit, rather than getting a sky and a sun");
    cmd->add_flag("--edr", options->edr, "extended dynamic range: a float surface and ACES 2.0 up to the screen's peak");
    cmd->add_option("--ocio-config", options->ocioConfig, "an OpenColorIO config (default with --ocio-display/--ocio-view: ocio://studio-config-latest)");
    cmd->add_option("--ocio-display", options->ocioDisplay, "the OCIO display (default: the config's)");
    cmd->add_option("--ocio-view", options->ocioView, "the OCIO view (default: the display's)");
    cmd->add_option("--snapshot", options->snapshot, "with --frames: the last frame as shown, to this EXR");
    cmd->add_option("--capture", options->capture,
                    "every frame as it was shown, panels and all, into this directory: one EXR a frame "
                    "named frame_00000.exr and up. What --snapshot writes once, this writes for the whole "
                    "run, which is how a capture of the viewer playing a timeline is made. It costs a "
                    "readback and a file a frame, so a captured run does not time like a drawn one");
    cmd->add_option("--isolate", options->isolate,
                    "a prim path whose Cryptomatte alone is shown, white on black (implies --aov cryptomatte)");
    cmd->add_option("--aov", options->aov,
                    "the output to start on: color | depth | primId | instanceId | elementId | Neye | normal | "
                    "cryptomatte | CryptoObject00 | CryptoObject01 | CryptoObject02");
    cmd->add_option("--fstop", options->fStop,
                    "the free camera's diaphragm: depth of field at --focus (0 is a pinhole)");
    cmd->add_option("--focus", options->focus, "the depth in focus, scene units");
    cmd->add_option("--variant", options->variants,
                    "a variant selection, as USD writes one in a path: /World{clip=air_fly_A0}. "
                    "Repeatable; the panel offers every variant set the stage carries");
    cmd->add_option("--hdri", options->hdriPaths,
                    "a directory of .hdr / .exr skies the Sky combo offers for the stage's dome "
                    "lights. Repeatable. Without it the folder each dome's own image sits in is "
                    "offered, which is usually the library it came from")
        ->check(CLI::ExistingDirectory);
    cmd->add_flag("--play", options->play, "start with the timeline playing");
    cmd->add_flag("--every-frame", options->everyFrame,
                  "play a frame of the timeline a drawn frame rather than by the clock: every "
                  "frame is drawn, slower than life");
    cmd->add_option("--shutter", options->shutter,
                    "how long the shutter is open, in frames: what a cloud's motion blur is "
                    "integrated over (0.5 is a 180 degree shutter, 0 none). The panel has it too");
    cmd->callback([options, size] {
        if (std::sscanf(size->c_str(), "%ux%u", &options->width, &options->height) != 2) {
            std::fprintf(stderr, "--size wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(1);
        }
        auto stats = view::runViewer(*options);
        if (!stats) {
            cli::fail(stats.error());
        }
        std::printf("%u frames: draw %.2f ms, frame %.2f ms (medians); %u distinct times, the last %.2f\n",
                    stats->frames, stats->medianDrawMs, stats->medianFrameMs, stats->distinctTimes, stats->lastTime);
    });
}

}   // namespace athenea::cli
