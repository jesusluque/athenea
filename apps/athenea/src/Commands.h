// Copyright (c) 2026 jesus luque.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

#include <CLI/CLI.hpp>

#include "Output.h"
#include "athenea/core/Error.h"

namespace athenea::cli {

/// THE EXIT CODES: 0 done, 1 any failure, 3 the GPU out of memory -- the one
/// failure a script can do something about by asking for less or waiting
/// for the GPU to be freed, so it is told apart (docs/operations.md §1.5).
inline constexpr int kExitFailure = 1;
inline constexpr int kExitOutOfMemory = 3;

/// Prints `error` to stderr -- and for the GPU out of memory, what to do
/// about it -- and returns the code to exit with.
[[nodiscard]] inline int report(const Error& error) {
    err("%s\n", error.toString().c_str());
    if (error.code() == ErrorCode::OutOfMemory) {
        err("athenea: the GPU ran out of memory. Free what other programs hold on it, or ask for less: "
            "a smaller --size, a streamed asset's smaller budget, no splat shadows; ATHENEA_GPU_BUDGET "
            "holds a run to a size in MiB.\n");
        return kExitOutOfMemory;
    }
    return kExitFailure;
}

/// The same, ending the subcommand with that code.
[[noreturn]] inline void fail(const Error& error) {
    throw CLI::RuntimeError(report(error));
}

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
void addFlatten(CLI::App& app);
void addLightmap(CLI::App& app);
void addVisibility(CLI::App& app);
void addLights(CLI::App& app);
void addLive(CLI::App& app);
void addAthzCensus(CLI::App& app);

/// What `athenea convert` takes for a .athz (CmdAthz.cpp).
struct AthzConvertOptions {
    uint32_t degree = 3;
    uint32_t chunkSplats = uint32_t{1} << 16;
    uint32_t run = 256;
    uint32_t pageRuns = 12;
    uint32_t tableLog = 11;
    bool     raw = false;
    uint32_t spzVersion = 3;
};
/// A conversion into or out of a .athz (or a .athc into SPZ): done and true,
/// or false when it is not one.
bool convertWithAthz(const std::string& input, const std::string& output, const AthzConvertOptions& options);
#if ATHENEA_HAVE_VIEW
void addView(CLI::App& app);
#endif

}   // namespace athenea::cli
