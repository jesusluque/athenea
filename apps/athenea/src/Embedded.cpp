// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat`, `athenea flatten` and `athenea compare` from inside
// another program, through hdAthenea.
//
// A program that draws through hdAthenea has loaded one USD already -- its
// own -- and the plugin is built against it. Running the conversion there,
// rather than as an `athenea` process with a USD of its own, is what lets a
// host that cannot ship a second USD (Blender, whose libusd_ms is in a
// namespace of its own) convert a model with the code it already loaded: the
// stage it reads and the stage it writes are that USD's, and the device work
// is the same effect bundle the command runs.
//
// The entries are C, so a host reaches them with dlsym or Python's ctypes.
// Each takes its command's arguments as `athenea <command>` does and its
// lines go to the host's sink. One command at a time, whichever: a command's
// options and its sink are process state.
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "Commands.h"
#include "Output.h"

namespace {

/// Puts the sink back however the command ends.
struct SinkScope {
    SinkScope(athenea::cli::LineSink sink, void* user) { athenea::cli::setLineSink(sink, user); }
    ~SinkScope() { athenea::cli::setLineSink(nullptr, nullptr); }
    SinkScope(const SinkScope&) = delete;
    SinkScope& operator=(const SinkScope&) = delete;
};

/// What a command a host stopped returns.
constexpr int kExitCancelled = 4;

/// One command at a time, mesh2splat or flatten.
std::mutex& running() {
    static std::mutex held;
    return held;
}

/// `athenea <name> <argv...>`, with `add` registering the subcommand.
int runCommand(const char* name, void (*add)(CLI::App&), int argc, const char* const* argv,
               athenea::cli::LineSink sink, void* user) {
    std::unique_lock<std::mutex> held(running(), std::try_to_lock);
    if (!held.owns_lock()) {
        const std::string busy = std::string(name) + ": another command is already running\n";
        if (sink != nullptr) sink(1, busy.c_str(), user);
        return 2;
    }
    const SinkScope scope(sink, user);
    athenea::cli::requestCancel(false);
    std::vector<const char*> args{"athenea", name};
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    CLI::App app{"athenea"};
    app.require_subcommand(1);
    add(app);
    try {
        app.parse(static_cast<int>(args.size()), args.data());
    } catch (const CLI::ParseError& error) {
        std::ostringstream out;
        std::ostringstream err;
        const int code = app.exit(error, out, err);
        if (!out.str().empty()) athenea::cli::out("%s", out.str().c_str());
        if (!err.str().empty()) athenea::cli::err("%s", err.str().c_str());
        return code != 0 && athenea::cli::cancelRequested() ? kExitCancelled : code;
    } catch (const std::exception& error) {
        athenea::cli::err("%s: %s\n", name, error.what());
        return athenea::cli::kExitFailure;
    }
    return 0;
}

}   // namespace

extern "C" {

/// What the entry points below take; raised when one changes. 2: flatten,
/// compare and the cancel request.
__attribute__((visibility("default"))) int athenea_embedded_abi() { return 2; }

/// Asks the running command to stop, from any thread. mesh2splat looks
/// between the transfer's slices and between levels of detail; it then
/// returns 4. A command that is not looking finishes as it would have.
__attribute__((visibility("default"))) void athenea_request_cancel() { athenea::cli::requestCancel(true); }

/// Runs `athenea mesh2splat <argv...>` in this process. `argv` is the
/// command's arguments, without `athenea mesh2splat`. Returns the command's
/// exit code (0 done, 1 failure, 2 another command already running, 3 the
/// GPU out of memory, 4 cancelled); its lines go to `sink` (stdout and stderr when null).
/// Blocks until the conversion is written; call it off a host's UI thread.
__attribute__((visibility("default"))) int athenea_mesh2splat(int argc, const char* const* argv,
                                                               athenea::cli::LineSink sink, void* user) {
    return runCommand("mesh2splat", athenea::cli::addMesh2Splat, argc, argv, sink, user);
}

/// Runs `athenea flatten <argv...>`: a stage's TX clouds under its sky and
/// lights, written as standard Gaussian Splatting files. The same contract.
__attribute__((visibility("default"))) int athenea_flatten(int argc, const char* const* argv,
                                                            athenea::cli::LineSink sink, void* user) {
    return runCommand("flatten", athenea::cli::addFlatten, argc, argv, sink, user);
}

/// Runs `athenea compare <argv...>`: two EXRs measured against each other by
/// the Measure effect (relMSE, p99, a heatmap). The same contract.
__attribute__((visibility("default"))) int athenea_compare(int argc, const char* const* argv,
                                                            athenea::cli::LineSink sink, void* user) {
    return runCommand("compare", athenea::cli::addCompare, argc, argv, sink, user);
}

}   // extern "C"
