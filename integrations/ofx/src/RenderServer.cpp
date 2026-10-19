// Copyright (c) 2026 jesus luque.
//
// The client side of RenderServer.h: a child process on two pipes, one
// JSON-RPC message a line each way, and a file a frame.
#include "RenderServer.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

extern char** environ;

namespace athenea::ofx {

using nlohmann::json;

namespace {

/// The directory of the image this code was linked into: the plugin's own
/// binary inside its bundle, which says where the server ships beside it.
std::filesystem::path moduleDirectory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&moduleDirectory), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path();
}

}   // namespace

struct RenderServer::Impl {
    std::filesystem::path binary;
    pid_t                 pid = -1;
    int                   toServer = -1;     // its stdin
    int                   fromServer = -1;   // its stdout
    std::string           pending;           // read past the last line
    int                   nextId = 1;
    std::filesystem::path rawFile;           // where it writes each frame

    ~Impl() { stop(); }

    void stop() {
        if (toServer >= 0) ::close(toServer);
        if (fromServer >= 0) ::close(fromServer);
        toServer = fromServer = -1;
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            int status = 0;
            ::waitpid(pid, &status, 0);
        }
        pid = -1;
        pending.clear();
        if (!rawFile.empty()) {
            std::error_code ignored;
            std::filesystem::remove(rawFile, ignored);
        }
    }

    bool alive() {
        if (pid <= 0) return false;
        int status = 0;
        return ::waitpid(pid, &status, WNOHANG) == 0;
    }

    bool start(std::string& error) {
        stop();
        int in[2] = {-1, -1};
        int out[2] = {-1, -1};
        if (::pipe(in) != 0 || ::pipe(out) != 0) {
            error = "cannot make pipes for the render server";
            return false;
        }
        ::fcntl(in[1], F_SETFD, FD_CLOEXEC);
        ::fcntl(out[0], F_SETFD, FD_CLOEXEC);
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, in[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, out[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, in[0]);
        posix_spawn_file_actions_addclose(&actions, out[1]);
        const std::string path = binary.string();
        char* argv[] = {const_cast<char*>(path.c_str()), nullptr};
        const int spawned = binary.has_parent_path()
                                ? posix_spawn(&pid, path.c_str(), &actions, nullptr, argv, environ)
                                : posix_spawnp(&pid, path.c_str(), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        ::close(in[0]);
        ::close(out[1]);
        if (spawned != 0) {
            ::close(in[1]);
            ::close(out[0]);
            pid = -1;
            error = "cannot start the render server " + path + ": " + std::strerror(spawned);
            return false;
        }
        toServer = in[1];
        fromServer = out[0];
        rawFile = std::filesystem::temp_directory_path() /
                  ("athenea-ofx-" + std::to_string(::getpid()) + "-" + std::to_string(pid) + ".raw");
        json hello{{"jsonrpc", "2.0"},
                   {"id", nextId++},
                   {"method", "initialize"},
                   {"params",
                    {{"protocolVersion", "2025-06-18"},
                     {"capabilities", json::object()},
                     {"clientInfo", {{"name", "athenea-ofx"}, {"version", "0.1"}}}}}};
        json reply;
        if (!call(hello, reply, error)) return false;
        return send(json{{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}}, error);
    }

    bool send(const json& message, std::string& error) {
        const std::string line = message.dump() + "\n";
        size_t done = 0;
        while (done < line.size()) {
            const ssize_t n = ::write(toServer, line.data() + done, line.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                error = "the render server stopped listening";
                return false;
            }
            done += static_cast<size_t>(n);
        }
        return true;
    }

    bool readLine(std::string& line, std::string& error) {
        for (;;) {
            if (const size_t end = pending.find('\n'); end != std::string::npos) {
                line = pending.substr(0, end);
                pending.erase(0, end + 1);
                return true;
            }
            char block[65536];
            const ssize_t n = ::read(fromServer, block, sizeof block);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) {
                error = "the render server closed (crashed?)";
                return false;
            }
            pending.append(block, static_cast<size_t>(n));
        }
    }

    /// Sends a request and reads lines until its answer: a notification the
    /// server sends meanwhile is not the answer.
    bool call(const json& request, json& reply, std::string& error) {
        if (!send(request, error)) return false;
        for (;;) {
            std::string line;
            if (!readLine(line, error)) return false;
            json message = json::parse(line, nullptr, false);
            if (message.is_discarded()) continue;
            if (message.contains("id") && message["id"] == request["id"]) {
                if (message.contains("error")) {
                    error = message["error"].value("message", std::string("the render server refused"));
                    return false;
                }
                reply = message.value("result", json::object());
                return true;
            }
        }
    }
};

RenderServer::RenderServer(std::filesystem::path binary) : impl_(std::make_unique<Impl>()) {
    impl_->binary = binary.empty() ? defaultBinary() : std::move(binary);
}

RenderServer::~RenderServer() = default;

std::filesystem::path RenderServer::defaultBinary() {
    if (const char* given = std::getenv("ATHENEA_MCP"); given != nullptr && given[0] != '\0') {
        return given;
    }
    // Bundle/Contents/MacOS/Athenea.ofx -> Bundle/Contents/Resources/bin
    const std::filesystem::path beside = moduleDirectory().parent_path() / "Resources" / "bin" / "athenea-mcp";
    std::error_code ignored;
    if (std::filesystem::exists(beside, ignored)) {
        return beside;
    }
    return "athenea-mcp";
}

bool RenderServer::render(const FrameRequest& request, Frame& frame, std::string& error) {
    // Twice at most: a server that died since the last frame is started once
    // more before the frame is given up on.
    for (int attempt = 0; attempt < 2; ++attempt) {
        if (!impl_->alive() && !impl_->start(error)) {
            continue;
        }
        json args{{"stage", request.cloud.string()},
                  {"xform", request.camera.cameraToWorld},
                  {"focal", request.camera.focal},
                  {"haperture", request.camera.haperture},
                  {"near", request.camera.nearZ},
                  {"far", request.camera.farZ},
                  {"time", request.time},
                  {"width", request.width},
                  {"height", request.height},
                  {"technique", "raster"},
                  {"sky", request.sky},
                  {"skyRotation", request.skyRotation},
                  {"skyVisible", request.skyVisible},
                  {"layers", request.layers},
                  {"raw", impl_->rawFile.string()}};
        if (request.camera.vaperture > 0.0) {
            args["vaperture"] = request.camera.vaperture;
        }
        // A number that is not finite travels as null and is refused: say
        // which before sending (a host that hands an unset knob as NaN).
        for (const auto& [key, value] : args.items()) {
            const auto finite = [](const json& v) { return !v.is_number_float() || std::isfinite(v.get<double>()); };
            bool ok = finite(value);
            if (value.is_array()) {
                for (const json& item : value) ok = ok && finite(item);
            }
            if (!ok) {
                error = "'" + key + "' is not a finite number";
                if (std::getenv("ATHENEA_OFX_DEBUG") != nullptr) {
                    std::fprintf(stderr, "athenea-ofx: %s: %s\n", error.c_str(), args.dump().c_str());
                }
                return false;
            }
        }
        if (std::getenv("ATHENEA_OFX_DEBUG") != nullptr) {
            std::fprintf(stderr, "athenea-ofx: render %s\n", args.dump().c_str());
        }
        json call{{"jsonrpc", "2.0"},
                  {"id", impl_->nextId++},
                  {"method", "tools/call"},
                  {"params", {{"name", "render"}, {"arguments", args}}}};
        json reply;
        if (!impl_->call(call, reply, error)) {
            impl_->stop();
            continue;
        }
        if (std::getenv("ATHENEA_OFX_DEBUG") != nullptr && reply.contains("content") && reply["content"].is_array() &&
            !reply["content"].empty()) {
            std::fprintf(stderr, "athenea-ofx: %s\n", reply["content"][0].value("text", std::string{}).c_str());
        }
        if (reply.value("isError", false)) {
            error = reply["content"].is_array() && !reply["content"].empty()
                        ? reply["content"][0].value("text", std::string("render failed"))
                        : std::string("render failed");
            return false;
        }
        const size_t pixels = size_t{request.width} * request.height;
        frame.width = request.width;
        frame.height = request.height;
        frame.rgba.resize(pixels * 4);
        frame.depth.resize(pixels);
        std::ifstream file(impl_->rawFile, std::ios::binary);
        file.read(reinterpret_cast<char*>(frame.rgba.data()), static_cast<std::streamsize>(pixels * 16));
        file.read(reinterpret_cast<char*>(frame.depth.data()), static_cast<std::streamsize>(pixels * 4));
        frame.layers.clear();
        for (const std::string& layer : request.layers) {
            std::vector<float>& values = frame.layers[layer];
            values.resize(pixels * 4);
            file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(pixels * 16));
        }
        // The answer's text: a line "manifest {...}" where the frame kept a matte.
        frame.manifest.clear();
        if (reply.contains("content") && reply["content"].is_array() && !reply["content"].empty()) {
            const std::string text = reply["content"][0].value("text", std::string{});
            if (const size_t at = text.find("\nmanifest "); at != std::string::npos) {
                const size_t from = at + 10;
                frame.manifest = text.substr(from, text.find('\n', from) - from);
            }
        }
        if (!file) {
            error = "the render server's frame is short: " + impl_->rawFile.string();
            return false;
        }
        return true;
    }
    return false;
}

}   // namespace athenea::ofx
