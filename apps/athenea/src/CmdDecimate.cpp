// Copyright (c) 2026 jesus luque.
//
// `athenea decimate`: fewer gaussians, kept where they matter. A cloud's levels
// of detail are built on the device and cut once by what merging each cell
// would lose (lod::Decimator); what survives is written as a ParticleField.
#include "Commands.h"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/lod/Lod.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {

void addDecimate(CLI::App& app) {
    struct Options {
        std::string input, output, prim;
        unsigned degree = 3;
        float colourTolerance = 0.05F;
        float outliers = 0.05F;
        float flatTolerance = 0.08F;
        float reach = 1.5F;
        bool noCamera = false;
        float ratio = 0.0F;
        float edge = 0.15F;
        float maxAngle = 60.0F;
        float highlightCone = 1.0F;
        float surfelFlatness = -1.0F;
        bool noSurfels = false;
        std::string camera;
        unsigned cameraPixels = 1920;
        std::string tree;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand("decimate",
                                   "fewer gaussians: merge the cells of a cloud whose gaussians one gaussian stands for");
    cmd->add_option("input", o->input, ".usd / .usda / .usdc (its ParticleField) or .ply / .splat / .spz / .sog")
        ->required();
    cmd->add_option("output", o->output, ".usda / .usdc / .usd")->required();
    cmd->add_option("--prim", o->prim, "the ParticleField to read, where a stage has more than one");
    cmd->add_option("--colour-tolerance", o->colourTolerance,
                    "how far a gaussian's colour or opacity may be from the one that replaces it before it "
                    "counts as different, 0..1")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--outliers", o->outliers,
                    "the share of what a merge stands for that may be different: a few is noise, many is an edge")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--flat-tolerance", o->flatTolerance,
                    "where the gaussians are discs, how thick against its width the one that replaces them may be")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--reach", o->reach,
                    "how many of its standard deviations a gaussian's centre may stand from the one that replaces it")
        ->check(CLI::Range(0.1F, 10.0F));
    cmd->add_option("--degree", o->degree, "harmonic degree cap 0..3");
    cmd->add_flag("--no-camera", o->noCamera, "do not add /World/Camera");
    cmd->add_option("--ratio", o->ratio,
                    "merge by pairs (NanoGS) until this many times fewer gaussians, the cheapest merges first, "
                    "instead of cutting the octree")
        ->check(CLI::Range(1.0F, 10000.0F));
    cmd->add_option("--edge", o->edge, "--ratio: a colour difference (sRGB, 0..1) never merged across")
        ->check(CLI::Range(0.0F, 2.0F));
    cmd->add_option("--max-angle", o->maxAngle, "--ratio: normals further apart than this are never merged, degrees")
        ->check(CLI::Range(0.0F, 180.0F));
    cmd->add_option("--highlight-cone", o->highlightCone,
                    "--ratio: on a glossy surface, normals further apart than this times sqrt(alpha) radians "
                    "never merge (the highlight would move); large lets the ratio decide alone")
        ->check(CLI::Range(0.0F, 1000.0F));
    cmd->add_option("--surfel-flatness", o->surfelFlatness,
                    "--ratio: a merge of discs whose thinnest axis is under this fraction of its middle one is "
                    "written a surfel (default 0.12)")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_flag("--no-surfels", o->noSurfels, "--ratio: keep a flat merge a thin gaussian, not a surfel");
    cmd->add_option("--camera", o->camera,
                    "--ratio: the camera the cloud is made for (default /World/Camera where there is one)");
    cmd->add_option("--camera-pixels", o->cameraPixels, "--ratio: pixels across that camera's image")
        ->check(CLI::Range(1u, 65536u));
    cmd->add_option("--tree", o->tree,
                    "--ratio: also write the whole merge tree here -- leaves in depth-first order, then every "
                    "merge with its parent and cost -- for a reader that cuts it");
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            cli::fail(error);
        };
        auto device = gpu::Device::create();
        if (!device) fail(device.error());
        gpu::ShaderLibrary library(*device);
        usd::DecimateStageOptions options;
        options.prim = o->prim;
        options.degree = o->degree;
        options.addCamera = !o->noCamera;
        options.settings.colourTolerance = o->colourTolerance;
        options.settings.outliers = o->outliers;
        options.settings.flatTolerance = o->flatTolerance;
        options.settings.reach = o->reach;
        options.pairs = o->ratio > 0.0F;
        options.pairSettings.ratio = o->ratio;
        options.pairSettings.edge = o->edge;
        options.pairSettings.maxAngle = o->maxAngle;
        options.pairSettings.highlightCone = o->highlightCone;
        if (o->surfelFlatness >= 0.0F) {
            options.pairSettings.flatTolerance = o->surfelFlatness;
        }
        options.pairSettings.surfels = !o->noSurfels;
        options.camera = o->camera;
        options.cameraPixels = o->cameraPixels;
        options.treeOutput = o->tree;
        auto decimated = usd::decimateStage(library, o->input, o->output, options);
        if (!decimated) fail(decimated.error());
        const lod::DecimateStats& stats = *decimated;
        if (options.pairs) {
            std::printf("decimate: %u gaussians of %u (%.1f%%, %.2fx) in %u passes by pairs, %u of them surfels; "
                        "wrote %s\n",
                        stats.merged, stats.before, 100.0 * stats.merged / std::max(stats.before, 1u),
                        static_cast<double>(stats.before) / std::max(stats.merged, 1u), stats.passes, stats.surfels,
                        o->output.c_str());
            return;
        }
        std::printf("decimate: %u gaussians of %u (%.1f%%): %u kept as they were, %u merged; wrote %s\n",
                    stats.splats + stats.merged, stats.before,
                    100.0 * (stats.splats + stats.merged) / std::max(stats.before, 1u), stats.splats, stats.merged,
                    o->output.c_str());
    });
}

}   // namespace athenea::cli
