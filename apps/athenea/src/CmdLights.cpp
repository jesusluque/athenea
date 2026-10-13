// Copyright (c) 2026 jesus luque.
//
// `athenea lights merge`: a cloud with its lamps. The base cloud (every lamp
// off) and one cloud a group of lamps -- the same conversion of the same
// geometry, the group lit at radiance one and nothing else -- become the base
// with a light layer a group, which a frame adds times each group's weight:
// a lamp switched or dimmed with nothing baked again (docs/decisions.md,
// light groups; usd/LightGroups.h).
#include "Commands.h"

#include <CLI/CLI.hpp>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/usd/LightGroups.h"

namespace athenea::cli {

void addLights(CLI::App& app) {
    struct Options {
        std::string              base, output, lights, prim;
        std::vector<std::string> layers;   ///< group=file
        float                    tolerance = 1.0e-4F;
        float                    dropBelow = 0.0F;
    };
    auto o = std::make_shared<Options>();
    auto* lights = app.add_subcommand("lights", "a car's lamps over its cloud: light groups switched at run time");
    lights->require_subcommand(1);
    auto* merge = lights->add_subcommand(
        "merge", "the base cloud and one cloud a group of lamps into the base with a light layer a group");
    merge->add_option("base", o->base, "the cloud with every lamp off (.usd / .usda / .usdc)")->required();
    merge->add_option("--layer", o->layers,
                      "group=cloud: the same conversion with that group alone lit at radiance 1 (repeatable)")
        ->required()
        ->allow_extra_args(false);
    merge->add_option("--lights", o->lights,
                      "the .lights.usda sidecar whose LightGroups, LightStates and LightRules are copied under "
                      "the cloud");
    merge->add_option("--prim", o->prim, "the base's ParticleField, where it has more than one");
    merge->add_option("--tolerance", o->tolerance,
                      "how far a layer's gaussian may stand from the base's, relative to the coordinate");
    merge->add_option("--drop-below", o->dropBelow,
                      "a layer none of whose gaussians sends out more than this per unit of radiance is dropped");
    merge->add_option("-o,--output", o->output, ".usda / .usdc / .usd")->required();
    merge->callback([o] {
        usd::LightMergeOptions options;
        options.base = o->base;
        options.output = o->output;
        options.lights = o->lights;
        options.prim = o->prim;
        options.tolerance = o->tolerance;
        options.dropBelow = o->dropBelow;
        for (const std::string& layer : o->layers) {
            const size_t equals = layer.find('=');
            if (equals == std::string::npos || equals == 0 || equals + 1 == layer.size()) {
                std::fprintf(stderr, "lights merge: --layer %s: write it group=cloud\n", layer.c_str());
                throw CLI::RuntimeError(kExitFailure);
            }
            options.layers.push_back({layer.substr(0, equals), layer.substr(equals + 1)});
        }
        auto device = gpu::Device::create();
        if (!device) cli::fail(device.error());
        gpu::ShaderLibrary library(*device);
        auto merged = usd::mergeLightLayers(library, options);
        if (!merged) {
            std::fprintf(stderr, "lights merge: ");
            cli::fail(merged.error());
        }
        for (const usd::LightMergeReport::Layer& layer : merged->layers) {
            std::printf("lights merge: %s: %u of %u gaussians lit%s\n", layer.group.c_str(), layer.lit,
                        merged->gaussians, layer.dropped ? ", dropped" : "");
        }
        for (const std::string& scope : merged->copied) {
            std::printf("lights merge: %s from %s\n", scope.c_str(), o->lights.c_str());
        }
        std::printf("lights merge: wrote %s\n", o->output.c_str());
    });
}

}   // namespace athenea::cli
