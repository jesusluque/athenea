// Copyright (c) 2026 jesus luque.
//
// `athenea decimate`: fewer gaussians, kept where they matter. A cloud's levels
// of detail are built on the device and cut once by what merging each cell
// would lose (lod::Decimator); what survives is written as a ParticleField.
#include "Commands.h"

#include <CLI/CLI.hpp>
#include <algorithm>

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
        float maxAspect = 1000.0F;
        float maxAngle = 60.0F;
        float highlightCone = 1.0F;
        float surfelFlatness = -1.0F;
        bool noSurfels = false;
        bool mergeGlass = false;
        std::string camera;
        unsigned cameraPixels = 1920;
        std::string tree;
        bool dropHidden = false;
        uint32_t cutBudget = 0;
        std::string cameraStage;
        float cutPixels = 0.0F;
        float cutError = 0.0F;
        bool cutBounds = false;
        bool cutCullBack = false;
        bool cutCoverage = false;
        bool treeCoverage = false;
        uint32_t occlusionClusters = 0;
        std::vector<std::string> occlusionGroup;
        uint32_t cutPages = 0;
        uint32_t treePages = 0;
        uint32_t objectGroups = 0;
        uint32_t passes = 0;
        uint32_t objectGroup = 0;
        std::string transferBasis;
        uint32_t transferClusters = 256;
        uint32_t transferComponents = 8;
        uint32_t transferBits = 8;
        uint32_t shadowCodebook = 0;
        uint32_t shadowRounds = 12;
        uint32_t quantizeChunk = 0;
        unsigned hiddenViews = 256;
        unsigned hiddenSize = 1024;
        float hiddenPixels = 0.02F;
        float hiddenBelow = -90.0F;
        std::string mergeOpacity = "mean";
        float pruneError = 0.0F;
        float maxCost = 0.0F;
        float maxPixels = 0.0F;
        bool noLightBands = false;
        bool hardness = false;
        float maxHardness = 16.0F;
        unsigned pruneRounds = 8;
        unsigned pruneViews = 64;
        unsigned pruneSize = 512;
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
    cmd->add_option("--max-aspect", o->maxAspect,
                    "--ratio: a merge longer than this against its width (or 1.25 times its members) is refused")
        ->check(CLI::Range(1.0F, 1000.0F));
    cmd->add_flag("--merge-glass", o->mergeGlass,
                  "--ratio: merge transmitting gaussians (glass) among themselves too; off, they are kept");
    cmd->add_flag("--no-surfels", o->noSurfels, "--ratio: keep a flat merge a thin gaussian, not a surfel");
    cmd->add_option("--camera", o->camera,
                    "--ratio: the camera the cloud is made for (default /World/Camera where there is one)");
    cmd->add_option("--camera-pixels", o->cameraPixels, "--ratio: pixels across that camera's image")
        ->check(CLI::Range(1u, 65536u));
    cmd->add_option("--tree", o->tree,
                    "--ratio: also write the whole merge tree here -- leaves in depth-first order, then every "
                    "merge with its parent and cost -- for a reader that cuts it");
    cmd->add_option("--camera-stage", o->cameraStage,
                    "--camera is read from this stage rather than the input (a view the cloud was not baked for)");
    cmd->add_option("--cut-budget", o->cutBudget,
                    "the input is a merge tree (--tree): keep only the cut a renderer would draw from --camera "
                    "with at most this many gaussians (with --ratio 1: nothing more merges)");
    cmd->add_option("--cut-error", o->cutError,
                    "the input is a merge tree: keep the cut whose error, projected on --camera's screen, is no "
                    "more than this many pixels while its parent's is more (research 100; in place of --cut-pixels)");
    cmd->add_flag("--cut-bounds", o->cutBounds,
                  "--cut-error: a node's error also bounds its leaves' colour spread and normal cone (research 100)");
    cmd->add_flag("--cut-cull-back", o->cutCullBack,
                  "--cut-bounds: leave out the nodes whose whole cone of normals faces away from --camera (a closed, "
                  "one-sided part: not glass, sheets or two-sided parts)");
    cmd->add_flag("--cut-coverage", o->cutCoverage,
                  "--cut-error: a kept merge's opacity is what its leaves really cover, composited on its plane");
    cmd->add_flag("--tree-coverage", o->treeCoverage,
                  "the input is a merge tree: every merge's opacity becomes what its leaves really cover, and the "
                  "whole tree is written (with --ratio 1)");
    cmd->add_option("--occlusion-mask", o->occlusionClusters,
                    "the input is a merge tree with 16 x 16 shadow bits: write each node's occlusion mask (where any "
                    "leaf under it is seen from, a cell of margin, open on glass) as an index into a codebook of this "
                    "many entries, and the whole tree (with --ratio 1)");
    cmd->add_option("--occlusion-group", o->occlusionGroup,
                    "with --occlusion-mask: more trees sharing one codebook with the input (an asset's pieces); each "
                    "is written beside the output, under its own name, with the masks alone");
    cmd->add_option("--cut-pages", o->cutPages,
                    "--cut-error: page the tree at this many gaussians a page and say how many the cut's pages hold");
    cmd->add_option("--tree-pages", o->treePages,
                    "the input is a merge tree: write its pages by error of this many gaussians (lodPage, "
                    "lodPageInfo, lodPageBounds), and the whole tree (with --ratio 1)");
    cmd->add_option("--passes", o->passes,
                    "--ratio: at most this many passes by pairs (default 60; a tree of many small parts may need more "
                    "to reach its roots)");
    cmd->add_option("--object-groups", o->objectGroups,
                    "decimate only one group of the asset's parts: those whose cryptoObject modulo this is "
                    "--object-group (an asset too large for one pass)");
    cmd->add_option("--object-group", o->objectGroup, "with --object-groups: which group, 0 ..");
    cmd->add_option("--cut-pixels", o->cutPixels,
                    "the input is a merge tree: keep the cut whose merges are no wider than this on --camera's "
                    "screen (in place of --cut-budget)");
    cmd->add_option("--transfer-basis", o->transferBasis,
                    "a clustered PCA basis (f32): the transfer is coded with it and decoded again before "
                    "anything else, so the output draws what that codec keeps");
    cmd->add_option("--transfer-clusters", o->transferClusters, "--transfer-basis: its centroids")
        ->check(CLI::Range(1u, 65536u));
    cmd->add_option("--transfer-components", o->transferComponents, "--transfer-basis: its axes a block")
        ->check(CLI::Range(1u, 16u));
    cmd->add_option("--transfer-bits", o->transferBits, "--transfer-basis: bits a weight (8 or 4)")
        ->check(CLI::IsMember({4u, 8u}));
    cmd->add_option("--shadow-codebook", o->shadowCodebook,
                    "the shadow bits replaced by the nearest of this many centroids (Hamming k-means on the GPU), "
                    "so the output draws what one index a gaussian and a codebook keep (0: none)")
        ->check(CLI::Range(0u, 65536u));
    cmd->add_option("--shadow-rounds", o->shadowRounds, "--shadow-codebook: k-means rounds");
    cmd->add_option("--quantize", o->quantizeChunk,
                    "the geometry at a phone chunk's bits, in chunks of this many gaussians (8192): 16 bit positions "
                    "in the chunk's box, 8 bit scales, 32 bit rotations, 8 bit opacity and colour, 16 bit normals, "
                    "8 bit curvature -- decoded again, so the output draws what that layout keeps (0: none)");
    cmd->add_flag("--drop-hidden", o->dropHidden,
                  "first leave out what no camera round the cloud sees (an orbit drawn by raster); with no "
                  "--ratio, only that");
    cmd->add_option("--hidden-views", o->hiddenViews, "--drop-hidden: directions, each seen from three distances");
    cmd->add_option("--hidden-size", o->hiddenSize, "--drop-hidden: pixels across each view (square)");
    cmd->add_option("--hidden-pixels", o->hiddenPixels,
                    "--drop-hidden: what a gaussian must give all the views together to stay, in pixels");
    cmd->add_option("--hidden-below", o->hiddenBelow,
                    "--drop-hidden: the lowest elevation a view is taken from, degrees over the horizon");
    cmd->add_option("--max-cost", o->maxCost,
                    "--ratio: no merge costs more than this; the ratio is then the most it goes to (0: off)");
    cmd->add_option("--max-pixels", o->maxPixels,
                    "--ratio: no merge wider than this many pixels on the camera's screen (0: off)");
    cmd->add_flag("--hardness", o->hardness,
                  "--ratio: a flat merge's edge as hard as its members' patch is even (a surfel drawn "
                  "exp(-(|u|^p + |v|^p)/2), p in athenea:splat:hardness)");
    cmd->add_option("--max-hardness", o->maxHardness, "--hardness: the largest p (2 is the gaussian)");
    cmd->add_flag("--no-light-bands", o->noLightBands,
                  "--ratio: merge across a lamp group's light changing fourfold (off: kept apart)");
    cmd->add_option("--prune-error", o->pruneError,
                    "first this many times fewer by removing what leaves the least error in an orbit of frames "
                    "(after --drop-hidden, before --ratio merges)");
    cmd->add_option("--prune-rounds", o->pruneRounds, "--prune-error: rounds, each scoring what is left");
    cmd->add_option("--prune-views", o->pruneViews, "--prune-error: directions, each seen from three distances");
    cmd->add_option("--prune-size", o->pruneSize, "--prune-error: pixels across each view (square)");
    cmd->add_option("--merge-opacity", o->mergeOpacity,
                    "--ratio: a merge's opacity: mean (its members', area weighted, widened as they overlapped), "
                    "mass (its mass over its widened area), mass-tight (the same, not widened) or falloff (its mass over "
                    "its own area up to 5: Spark's harder kernel where the members overlapped)")
        ->check(CLI::IsMember({"mean", "mass", "mass-tight", "falloff"}));
    cmd->callback([o] {
        const auto fail = [](const Error& error) {
            cli::fail(error);
        };
        auto device = gpu::Device::create();
        if (!device) fail(device.error());
        gpu::ShaderLibrary library(*device);
        if (!o->occlusionGroup.empty()) {
            // ONE CODEBOOK OVER SEVERAL TREES: the input to the output, the
            // rest beside it under their own names.
            if (o->occlusionClusters == 0) fail(Error(ErrorCode::InvalidArgument, "--occlusion-group needs --occlusion-mask"));
            std::vector<std::filesystem::path> inputs{o->input};
            std::vector<std::filesystem::path> outputs{o->output};
            const std::filesystem::path dir = std::filesystem::path(o->output).parent_path();
            for (const std::string& tree : o->occlusionGroup) {
                inputs.emplace_back(tree);
                outputs.push_back(dir / std::filesystem::path(tree).filename());
            }
            auto shared = usd::occlusionMasksShared(library, inputs, outputs, o->occlusionClusters);
            if (!shared) fail(shared.error());
            return;
        }
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
        options.pairSettings.mergeGlass = o->mergeGlass;
        options.pairSettings.maxAspect = o->maxAspect;
        options.camera = o->camera;
        options.cameraPixels = o->cameraPixels;
        options.treeOutput = o->tree;
        options.cutBudget = o->cutBudget;
        options.cameraStage = o->cameraStage;
        options.cutPixels = o->cutPixels;
        options.cutError = o->cutError;
        options.cutBounds = o->cutBounds;
        options.cutCullBack = o->cutCullBack;
        options.cutCoverage = o->cutCoverage;
        options.treeCoverage = o->treeCoverage;
        options.occlusionClusters = o->occlusionClusters;
        options.cutPages = o->cutPages;
        options.treePages = o->treePages;
        options.objectGroups = o->objectGroups;
        if (o->passes > 0) {
            options.pairSettings.passes = o->passes;
        }
        options.objectGroup = o->objectGroup;
        options.transferBasis = o->transferBasis;
        options.transferClusters = o->transferClusters;
        options.transferComponents = o->transferComponents;
        options.transferBits = o->transferBits;
        options.shadowCodebook = o->shadowCodebook;
        options.shadowRounds = std::max(o->shadowRounds, 1u);
        options.quantizeChunk = o->quantizeChunk;
        options.dropHidden = o->dropHidden;
        options.hiddenViews = std::max(o->hiddenViews, 1u);
        options.hiddenSize = std::max(o->hiddenSize, 16u);
        options.hiddenPixels = o->hiddenPixels;
        options.hiddenBelow = o->hiddenBelow;
        options.pairSettings.opacityRule = o->mergeOpacity == "mass"         ? 1u
                                           : o->mergeOpacity == "mass-tight" ? 2u
                                           : o->mergeOpacity == "falloff"    ? 3u
                                                                             : 0u;
        options.pruneError = o->pruneError;
        options.lightBands = !o->noLightBands;
        options.pairSettings.maxCost = o->maxCost;
        options.pairSettings.maxPixels = o->maxPixels;
        options.pairSettings.hardness = o->hardness;
        options.pairSettings.maxHardness = std::clamp(o->maxHardness, 2.0F, 32.0F);
        options.pruneRounds = o->pruneRounds;
        options.pruneViews = std::max(o->pruneViews, 1u);
        options.pruneSize = std::max(o->pruneSize, 16u);
        if (!options.pairs && (options.dropHidden || options.pruneError > 1.0F)) {
            options.pairs = true;   // pruned, then every splat its own cluster: nothing merges
            options.pairSettings.ratio = 1.0F;
            options.pairSettings.surfels = false;   // nor does a thin one turn into a disc
        }
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
