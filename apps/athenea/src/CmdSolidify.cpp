// Copyright (c) 2026 jesus luque.
//
// `athenea solidify`: a cloud's opaque surfaces as a solid layer (SSF), drawn
// as Mode A -- the blobs as flat gaussians, then the residuals -- in a copy of
// the stage it came from (usd::solidifyStage).
#include "Commands.h"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/usd/Export.h"

namespace athenea::cli {

void addSolidify(CLI::App& app) {
    struct Options {
        std::string input, output, prim;
        unsigned degree = 3;
        lod::SolidifySettings s;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "solidify", "SSF: a cloud's opaque surfaces as blobs of a summed field, the rest kept as they are; "
                    "written as Mode A (the blobs drawn as flat gaussians) into a copy of the stage");
    cmd->add_option("input", o->input, ".usd / .usda / .usdc (its ParticleField)")->required();
    cmd->add_option("output", o->output, ".usda / .usdc / .usd")->required();
    cmd->add_option("--prim", o->prim, "the ParticleField to read, where a stage has more than one");
    cmd->add_option("--degree", o->degree, "harmonic degree cap 0..3");
    cmd->add_option("--opacity-min", o->s.opacityMin, "a gaussian less opaque than this stays residual")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--flat-max", o->s.flatMax,
                    "a gaussian whose thin scale is not under this of its middle one stays residual (0: none absorbed)")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--coherence-min", o->s.coherenceMin,
                    "the weighted mean |n.n'| over its neighbours below which a blob goes back to the residuals")
        ->check(CLI::Range(0.0F, 1.0F));
    cmd->add_option("--radius-scale", o->s.radiusScale, "a blob's support over its gaussian's scale")
        ->check(CLI::Range(0.5F, 8.0F));
    cmd->add_option("--thickness", o->s.thicknessRatio,
                    "the anchor's thickness over the candidates' mean middle radius")
        ->check(CLI::Range(0.01F, 4.0F));
    cmd->add_option("--thin-ratio", o->s.thinRatio, "Mode A: a surfel's thin axis over its smaller radius")
        ->check(CLI::Range(0.001F, 1.0F));
    cmd->callback([o] {
        auto device = gpu::Device::create();
        if (!device) fail(device.error());
        gpu::ShaderLibrary library(*device);
        usd::SolidifyStageOptions options;
        options.prim = o->prim;
        options.degree = o->degree;
        options.settings = o->s;
        auto stats = usd::solidifyStage(library, o->input, o->output, options);
        if (!stats) fail(stats.error());
        const double absorbed = stats->gaussians > 0 ? 100.0 * stats->absorbed / stats->gaussians : 0.0;
        out("solidify: %u gaussians: %u candidates, %u absorbed (%.1f %%), %u blobs, %u residuals\n",
            stats->gaussians, stats->candidates, stats->absorbed, absorbed, stats->blobs, stats->residuals);
        out("solidify: read %.2f s, solidify %.2f s, write %.2f s\n", stats->readSeconds, stats->solidifySeconds,
            stats->writeSeconds);
        std::error_code ec;
        const auto in = std::filesystem::file_size(o->input, ec);
        const auto outBytes = std::filesystem::file_size(o->output, ec);
        out("solidify: %s %.1f MB -> %s %.1f MB\n", o->input.c_str(), in / 1048576.0, o->output.c_str(),
            outBytes / 1048576.0);
    });
}

}   // namespace athenea::cli
