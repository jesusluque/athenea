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
#include <algorithm>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include <nlohmann/json.hpp>

namespace athenea::ofx {

using nlohmann::json;

namespace {

/// The directory of the image this code was linked into: the plugin's own
/// binary inside its bundle, which says where the server ships beside it.
std::filesystem::path moduleDirectory() {
#ifdef _WIN32
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&moduleDirectory), &module)) {
        return {};
    }
    wchar_t name[32768];
    const DWORD length = GetModuleFileNameW(module, name, static_cast<DWORD>(std::size(name)));
    if (length == 0 || length >= std::size(name)) {
        return {};
    }
    return std::filesystem::path(std::wstring(name, length)).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&moduleDirectory), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

/// This process's id, for the name of the file a frame is written to.
unsigned long currentProcessId() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return static_cast<unsigned long>(::getpid());
#endif
}

}   // namespace

/// The server as a child process on two pipes: its stdin and its stdout.
/// The only part of the plugin that differs between systems.
class Child {
public:
    ~Child() { stop(); }

    bool start(const std::filesystem::path& binary, std::string& error) {
        stop();
#ifdef _WIN32
        SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE childIn = nullptr;
        HANDLE childOut = nullptr;
        if (!CreatePipe(&childIn, &toServer_, &inherit, 0) || !CreatePipe(&fromServer_, &childOut, &inherit, 0)) {
            error = "cannot make pipes for the render server";
            closeAll(childIn, childOut);
            return false;
        }
        // The ends this process keeps are not the child's to inherit.
        SetHandleInformation(toServer_, HANDLE_FLAG_INHERIT, 0);
        SetHandleInformation(fromServer_, HANDLE_FLAG_INHERIT, 0);
        STARTUPINFOW startup{};
        startup.cb = sizeof startup;
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = childIn;
        startup.hStdOutput = childOut;
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        std::wstring commandLine = L"\"" + binary.wstring() + L"\"";
        PROCESS_INFORMATION created{};
        // A bare name is looked for on the PATH: no application name given.
        const BOOL ok = CreateProcessW(binary.has_parent_path() ? binary.c_str() : nullptr, commandLine.data(),
                                       nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                                       &created);
        const DWORD failure = ok ? 0 : GetLastError();
        CloseHandle(childIn);
        CloseHandle(childOut);
        if (!ok) {
            closeAll(toServer_, fromServer_);
            toServer_ = fromServer_ = nullptr;
            error = "cannot start the render server " + binary.string() + ": error " + std::to_string(failure);
            return false;
        }
        CloseHandle(created.hThread);
        process_ = created.hProcess;
        id_ = created.dwProcessId;
        return true;
#else
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
                                ? posix_spawn(&pid_, path.c_str(), &actions, nullptr, argv, environ)
                                : posix_spawnp(&pid_, path.c_str(), &actions, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&actions);
        ::close(in[0]);
        ::close(out[1]);
        if (spawned != 0) {
            ::close(in[1]);
            ::close(out[0]);
            pid_ = -1;
            error = "cannot start the render server " + path + ": " + std::strerror(spawned);
            return false;
        }
        toServer_ = in[1];
        fromServer_ = out[0];
        return true;
#endif
    }

    void stop() {
#ifdef _WIN32
        closeAll(toServer_, fromServer_);
        toServer_ = fromServer_ = nullptr;
        if (process_ != nullptr) {
            TerminateProcess(process_, 1);
            WaitForSingleObject(process_, INFINITE);
            CloseHandle(process_);
        }
        process_ = nullptr;
        id_ = 0;
#else
        if (toServer_ >= 0) ::close(toServer_);
        if (fromServer_ >= 0) ::close(fromServer_);
        toServer_ = fromServer_ = -1;
        if (pid_ > 0) {
            ::kill(pid_, SIGTERM);
            int status = 0;
            ::waitpid(pid_, &status, 0);
        }
        pid_ = -1;
#endif
    }

    bool alive() {
#ifdef _WIN32
        return process_ != nullptr && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
#else
        if (pid_ <= 0) return false;
        int status = 0;
        return ::waitpid(pid_, &status, WNOHANG) == 0;
#endif
    }

    /// The child's process id.
    [[nodiscard]] unsigned long id() const {
#ifdef _WIN32
        return id_;
#else
        return static_cast<unsigned long>(pid_);
#endif
    }

    /// Writes all of `data`; false once the child has stopped reading.
    bool write(const char* data, size_t size) {
        size_t done = 0;
        while (done < size) {
#ifdef _WIN32
            DWORD n = 0;
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(size - done, 1u << 30));
            if (!WriteFile(toServer_, data + done, chunk, &n, nullptr) || n == 0) return false;
#else
            const ssize_t n = ::write(toServer_, data + done, size - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) return false;
#endif
            done += static_cast<size_t>(n);
        }
        return true;
    }

    /// Reads what is there, up to `size`; 0 once the child has closed.
    size_t read(char* data, size_t size) {
        for (;;) {
#ifdef _WIN32
            DWORD n = 0;
            if (!ReadFile(fromServer_, data, static_cast<DWORD>(size), &n, nullptr)) return 0;
            return n;
#else
            const ssize_t n = ::read(fromServer_, data, size);
            if (n < 0 && errno == EINTR) continue;
            return n > 0 ? static_cast<size_t>(n) : 0;
#endif
        }
    }

private:
#ifdef _WIN32
    static void closeAll(HANDLE a, HANDLE b) {
        if (a != nullptr) CloseHandle(a);
        if (b != nullptr) CloseHandle(b);
    }
    HANDLE process_ = nullptr;
    DWORD  id_ = 0;
    HANDLE toServer_ = nullptr;     // its stdin
    HANDLE fromServer_ = nullptr;   // its stdout
#else
    pid_t pid_ = -1;
    int   toServer_ = -1;     // its stdin
    int   fromServer_ = -1;   // its stdout
#endif
};

struct RenderServer::Impl {
    std::filesystem::path binary;
    Child                 child;
    std::string           pending;    // read past the last line
    int                   nextId = 1;
    std::filesystem::path rawFile;    // where it writes each frame

    ~Impl() { stop(); }

    void stop() {
        child.stop();
        pending.clear();
        if (!rawFile.empty()) {
            std::error_code ignored;
            std::filesystem::remove(rawFile, ignored);
        }
    }

    bool alive() { return child.alive(); }

    bool start(std::string& error) {
        stop();
        if (!child.start(binary, error)) {
            return false;
        }
        rawFile = std::filesystem::temp_directory_path() /
                  ("athenea-ofx-" + std::to_string(currentProcessId()) + "-" + std::to_string(child.id()) + ".raw");
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
        if (!child.write(line.data(), line.size())) {
            error = "the render server stopped listening";
            return false;
        }
        return true;
    }

    bool readLine(std::string& line, std::string& error) {
        for (;;) {
            if (const size_t end = pending.find('\n'); end != std::string::npos) {
                line = pending.substr(0, end);
                if (!line.empty() && line.back() == '\r') line.pop_back();
                pending.erase(0, end + 1);
                return true;
            }
            char block[65536];
            const size_t n = child.read(block, sizeof block);
            if (n == 0) {
                error = "the render server closed (crashed?)";
                return false;
            }
            pending.append(block, n);
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
#ifdef _WIN32
    const char* name = "athenea-mcp.exe";
#else
    const char* name = "athenea-mcp";
#endif
    // Bundle/Contents/<MacOS|Linux-x86-64|Win64>/Athenea.ofx -> Bundle/Contents/Resources/bin
    const std::filesystem::path beside = moduleDirectory().parent_path() / "Resources" / "bin" / name;
    std::error_code ignored;
    if (std::filesystem::exists(beside, ignored)) {
        return beside;
    }
    return name;
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
