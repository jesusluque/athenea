// Copyright (c) 2026 jesus luque.
//
// athenea: the engine from a terminal. Headless but for `athenea view` -- a render node
// has no window, and nothing else here opens one.
#include <filesystem>
#include <system_error>

#include <CLI/CLI.hpp>

#include "Commands.h"
#include "athenea/core/Log.h"
#include "athenea/core/Platform.h"
#include "athenea/usd/StageRenderer.h"

int main(int argc, char** argv) {
    // THE ENGINE'S OWN USD PLUGINS, before any stage is opened or written:
    // the codeless schemas a converted cloud applies, and hdAthenea. A stage
    // written without them names `AtheneaSplatCryptomatteAPI` as an unknown
    // token and drops the schema -- which is what mesh2splat did unless the
    // shell had set PXR_PLUGINPATH_NAME. Beside the binary, as the shaders
    // are, so a build or an install that keeps the layout finds them.
    const std::filesystem::path plugins = athenea::platform::executableDir() / ".." / "plugin" / "usd";
    std::error_code missing;
    if (std::filesystem::exists(plugins / "plugInfo.json", missing)) {
        athenea::usd::registerPlugins(std::filesystem::weakly_canonical(plugins, missing));
    }

    CLI::App app{"athenea: Gaussian splats and point clouds on the GPU"};
    app.set_version_flag("--version", "athenea " ATHENEA_VERSION);
    app.require_subcommand(1);
    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "debug logging");
    app.parse_complete_callback([&verbose] {
        if (verbose) {
            athenea::log::setMinimum(athenea::log::Level::Debug);
        }
    });

    athenea::cli::addInfo(app);
    athenea::cli::addCompare(app);
    athenea::cli::addRender(app);
    athenea::cli::addBench(app);
    athenea::cli::addConvert(app);
    athenea::cli::addDecimate(app);
    athenea::cli::addMigrate(app);
    athenea::cli::addStage(app);
    athenea::cli::addAofx(app);
    athenea::cli::addMesh2Splat(app);
    athenea::cli::addFlatten(app);
    athenea::cli::addVisibility(app);
    athenea::cli::addLive(app);
#if ATHENEA_HAVE_VIEW
    athenea::cli::addView(app);
#endif

    CLI11_PARSE(app, argc, argv);
    return 0;
}
