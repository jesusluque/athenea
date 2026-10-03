// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat` from inside another program, through hdAthenea.
//
// A program that draws through hdAthenea has loaded one USD already -- its
// own -- and the plugin is built against it. Running the conversion there,
// rather than as an `athenea` process with a USD of its own, is what lets a
// host that cannot ship a second USD (Blender, whose libusd_ms is in a
// namespace of its own) convert a model with the code it already loaded: the
// stage it reads and the stage it writes are that USD's, and the device work
// is the same effect bundle the command runs.
//
// The entry is C, so a host reaches it with dlsym or Python's ctypes. It
// takes the command's arguments as `athenea mesh2splat` does and its lines go
// to the host's sink. One conversion at a time: the command's options and its
// sink are process state.
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

}   // namespace

extern "C" {

/// What the entry points below take; raised when one changes.
__attribute__((visibility("default"))) int athenea_embedded_abi() { return 1; }

/// Runs `athenea mesh2splat <argv...>` in this process. `argv` is the
/// command's arguments, without `athenea mesh2splat`. Returns the command's
/// exit code (0 done, 1 failure, 2 a conversion already running, 3 the GPU
/// out of memory); its lines go to `sink` (stdout and stderr when null).
/// Blocks until the conversion is written; call it off a host's UI thread.
__attribute__((visibility("default"))) int athenea_mesh2splat(int argc, const char* const* argv,
                                                               athenea::cli::LineSink sink, void* user) {
    static std::mutex running;
    std::unique_lock<std::mutex> held(running, std::try_to_lock);
    if (!held.owns_lock()) {
        if (sink != nullptr) sink(1, "mesh2splat: a conversion is already running\n", user);
        return 2;
    }
    const SinkScope scope(sink, user);
    std::vector<const char*> args{"athenea", "mesh2splat"};
    for (int i = 0; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    CLI::App app{"athenea"};
    app.require_subcommand(1);
    athenea::cli::addMesh2Splat(app);
    try {
        app.parse(static_cast<int>(args.size()), args.data());
    } catch (const CLI::ParseError& error) {
        std::ostringstream out;
        std::ostringstream err;
        const int code = app.exit(error, out, err);
        if (!out.str().empty()) athenea::cli::out("%s", out.str().c_str());
        if (!err.str().empty()) athenea::cli::err("%s", err.str().c_str());
        return code;
    } catch (const std::exception& error) {
        athenea::cli::err("mesh2splat: %s\n", error.what());
        return athenea::cli::kExitFailure;
    }
    return 0;
}

}   // extern "C"
