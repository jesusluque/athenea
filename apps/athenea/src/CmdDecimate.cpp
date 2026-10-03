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
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            std::fprintf(stderr, "%s\n", error.toString().c_str());
            throw CLI::RuntimeError(1);
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
        auto decimated = usd::decimateStage(library, o->input, o->output, options);
        if (!decimated) fail(decimated.error());
        const lod::DecimateStats& stats = *decimated;
        std::printf("decimate: %u gaussians of %u (%.1f%%): %u kept as they were, %u merged; wrote %s\n",
                    stats.splats + stats.merged, stats.before,
                    100.0 * (stats.splats + stats.merged) / std::max(stats.before, 1u), stats.splats, stats.merged,
                    o->output.c_str());
    });
}

}   // namespace athenea::cli
