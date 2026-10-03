// Copyright (c) 2026 jesus luque.
//
// athenea: the engine from a terminal. Headless but for `athenea view` -- a render node
// has no window, and nothing else here opens one.
#include <CLI/CLI.hpp>

#include "Commands.h"
#include "athenea/core/Log.h"

int main(int argc, char** argv) {
    CLI::App app{"athenea: Gaussian splats and point clouds on the GPU"};
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
    athenea::cli::addStage(app);
    athenea::cli::addAofx(app);
    athenea::cli::addMesh2Splat(app);
    athenea::cli::addVisibility(app);
    athenea::cli::addLive(app);
#if ATHENEA_HAVE_VIEW
    athenea::cli::addView(app);
#endif

    CLI11_PARSE(app, argc, argv);
    return 0;
}
