// Copyright (c) 2026 jesus luque.
#pragma once

#include <CLI/CLI.hpp>

namespace athenea::cli {

/// Each subcommand registers itself on the app and returns the callback to run
/// when it was the one chosen.
void addInfo(CLI::App& app);
void addCompare(CLI::App& app);
void addRender(CLI::App& app);
void addBench(CLI::App& app);
void addConvert(CLI::App& app);
void addDecimate(CLI::App& app);
void addMigrate(CLI::App& app);
void addStage(CLI::App& app);
void addAofx(CLI::App& app);
void addMesh2Splat(CLI::App& app);
void addVisibility(CLI::App& app);
void addLive(CLI::App& app);
#if ATHENEA_HAVE_VIEW
void addView(CLI::App& app);
#endif

}   // namespace athenea::cli
