// Copyright (c) 2026 jesus luque.
//
// The operating system's side of handing a frame to another process: Mach
// ports and file descriptors carried between processes (HandleServer,
// fetchHandles), Metal's shared events as ports, and a child process on pipes.
// Part of core's Platform: the only place these calls are made.
#include "athenea/core/Platform.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <thread>
#include <utility>
#include <optional>

#if !defined(_WIN32)
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#if defined(__APPLE__)
#include <bootstrap.h>
#include <mach/mach.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <sys/fileport.h>
#include <stdlib.h>
extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void  objc_autoreleasePoolPop(void* pool);
#endif

namespace athenea::platform {
namespace {

constexpr uint32_t kMaxHandles = 8;
constexpr uint32_t kStatusServed = 0;
constexpr uint32_t kStatusUnknownToken = 1;
constexpr uint32_t kStatusUnsupported = 2;

std::string hex64(uint64_t value) {
    char text[17];
    std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
    return text;
}

/// The offers, shared by both transports' threads.
class Offers {
public:
    void offer(uint64_t token, std::vector<OsHandle> handles) {
        std::lock_guard lock(mutex_);
        auto& slot = offers_[token];
        for (const OsHandle& old : slot) closeOsHandle(old);
        slot = std::move(handles);
    }
    void withdraw(uint64_t token) {
        std::lock_guard lock(mutex_);
        if (auto it = offers_.find(token); it != offers_.end()) {
            for (const OsHandle& h : it->second) closeOsHandle(h);
            offers_.erase(it);
        }
    }
    /// Copies of the handles under `token`, or nothing.
    std::optional<std::vector<OsHandle>> copies(uint64_t token) {
        std::lock_guard lock(mutex_);
        const auto it = offers_.find(token);
        if (it == offers_.end()) return std::nullopt;
        std::vector<OsHandle> out;
        for (const OsHandle& h : it->second) {
            auto copy = duplicateOsHandle(h);
            if (!copy) {
                for (const OsHandle& made : out) closeOsHandle(made);
                return std::nullopt;
            }
            out.push_back(*copy);
        }
        return out;
    }
    ~Offers() {
        for (auto& [token, handles] : offers_) {
            for (const OsHandle& h : handles) closeOsHandle(h);
        }
    }

private:
    std::mutex                                     mutex_;
    std::map<uint64_t, std::vector<OsHandle>>      offers_;
};

#if !defined(_WIN32)

// --- the Unix socket transport (any POSIX system) ------------------------------

struct SocketReply {
    uint32_t status = kStatusServed;
    uint32_t count = 0;
    uint32_t kinds[kMaxHandles] = {};
};

class SocketServer final : public HandleServer {
public:
    static Result<std::unique_ptr<HandleServer>> make(const std::string& prefix) {
        auto server = std::unique_ptr<SocketServer>(new SocketServer());
        server->path_ = "/tmp/" + prefix + "-" + std::to_string(processId()) + "-" + hex64(randomToken()).substr(0, 8) +
                        ".sock";
        sockaddr_un address{};
        if (server->path_.size() >= sizeof(address.sun_path)) {
            return Error::make(ErrorCode::InvalidArgument, "handle server: socket path '{}' too long", server->path_);
        }
        server->listen_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (server->listen_ < 0) {
            return Error::make(ErrorCode::IoFailure, "handle server: socket: {}", std::strerror(errno));
        }
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, server->path_.c_str(), server->path_.size() + 1);
        ::unlink(server->path_.c_str());
        if (::bind(server->listen_, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0 ||
            ::listen(server->listen_, 16) != 0) {
            return Error::make(ErrorCode::IoFailure, "handle server: bind {}: {}", server->path_, std::strerror(errno));
        }
        ::chmod(server->path_.c_str(), 0600);
        if (::pipe(server->stop_) != 0) {
            return Error::make(ErrorCode::IoFailure, "handle server: pipe: {}", std::strerror(errno));
        }
        server->name_ = "unix:" + server->path_;
        server->thread_ = std::thread([raw = server.get()] { raw->run(); });
        return std::unique_ptr<HandleServer>(std::move(server));
    }

    ~SocketServer() override {
        if (thread_.joinable()) {
            const char byte = 1;
            (void)!::write(stop_[1], &byte, 1);
            thread_.join();
        }
        if (listen_ >= 0) ::close(listen_);
        for (int fd : stop_) {
            if (fd >= 0) ::close(fd);
        }
        if (!path_.empty()) ::unlink(path_.c_str());
    }

    const std::string& name() const noexcept override { return name_; }
    void offer(uint64_t token, std::vector<OsHandle> handles) override { offers_.offer(token, std::move(handles)); }
    void withdraw(uint64_t token) override { offers_.withdraw(token); }
    uint64_t served() const noexcept override { return served_.load(); }
    uint64_t refused() const noexcept override { return refused_.load(); }

private:
    SocketServer() = default;

    void run() {
        for (;;) {
            pollfd waits[2] = {{listen_, POLLIN, 0}, {stop_[0], POLLIN, 0}};
            if (::poll(waits, 2, -1) < 0) {
                if (errno == EINTR) continue;
                return;
            }
            if ((waits[1].revents & POLLIN) != 0) return;
            if ((waits[0].revents & POLLIN) == 0) continue;
            const int peer = ::accept(listen_, nullptr, nullptr);
            if (peer < 0) continue;
            answer(peer);
            ::close(peer);
        }
    }

    void answer(int peer) {
        uint64_t token = 0;
        size_t got = 0;
        while (got < sizeof token) {
            const ssize_t n = ::read(peer, reinterpret_cast<char*>(&token) + got, sizeof token - got);
            if (n <= 0) return;
            got += static_cast<size_t>(n);
        }
        SocketReply reply;
        std::vector<OsHandle> handles;
        if (auto copies = offers_.copies(token); !copies) {
            reply.status = kStatusUnknownToken;
            refused_.fetch_add(1);
        } else {
            handles = std::move(*copies);
            for (const OsHandle& h : handles) {
                if (h.kind != OsHandle::Kind::Fd) reply.status = kStatusUnsupported;
            }
            reply.count = static_cast<uint32_t>(std::min<size_t>(handles.size(), kMaxHandles));
            for (uint32_t k = 0; k < reply.count; ++k) reply.kinds[k] = static_cast<uint32_t>(handles[k].kind);
        }
        if (reply.status != kStatusServed) reply.count = 0;
        iovec io{&reply, sizeof reply};
        msghdr message{};
        message.msg_iov = &io;
        message.msg_iovlen = 1;
        alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * kMaxHandles)] = {};
        if (reply.count > 0) {
            message.msg_control = control;
            message.msg_controllen = CMSG_SPACE(sizeof(int) * reply.count);
            cmsghdr* header = CMSG_FIRSTHDR(&message);
            header->cmsg_level = SOL_SOCKET;
            header->cmsg_type = SCM_RIGHTS;
            header->cmsg_len = CMSG_LEN(sizeof(int) * reply.count);
            int* fds = reinterpret_cast<int*>(CMSG_DATA(header));
            for (uint32_t k = 0; k < reply.count; ++k) fds[k] = static_cast<int>(handles[k].value);
        }
        if (::sendmsg(peer, &message, 0) >= 0 && reply.status == kStatusServed) {
            served_.fetch_add(1);
        }
        for (const OsHandle& h : handles) closeOsHandle(h);
    }

    std::string           path_, name_;
    int                   listen_ = -1;
    int                   stop_[2] = {-1, -1};
    std::thread           thread_;
    Offers                offers_;
    std::atomic<uint64_t> served_{0}, refused_{0};
};

Result<std::vector<OsHandle>> fetchOverSocket(const std::string& path, uint64_t token,
                                              std::chrono::milliseconds timeout) {
    sockaddr_un address{};
    if (path.size() >= sizeof(address.sun_path)) {
        return Error::make(ErrorCode::InvalidArgument, "handle server '{}': path too long", path);
    }
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return Error::make(ErrorCode::IoFailure, "socket: {}", std::strerror(errno));
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    if (::connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof address) != 0) {
        const int failed = errno;
        ::close(fd);
        return Error::make(ErrorCode::IoFailure, "handle server '{}' cannot be reached: {}", path,
                           std::strerror(failed));
    }
    if (::write(fd, &token, sizeof token) != static_cast<ssize_t>(sizeof token)) {
        ::close(fd);
        return Error::make(ErrorCode::IoFailure, "handle server '{}': the request was not sent", path);
    }
    pollfd wait{fd, POLLIN, 0};
    if (::poll(&wait, 1, static_cast<int>(timeout.count())) <= 0) {
        ::close(fd);
        return Error::make(ErrorCode::IoFailure, "handle server '{}' did not answer in {} ms", path, timeout.count());
    }
    SocketReply reply;
    iovec io{&reply, sizeof reply};
    msghdr message{};
    message.msg_iov = &io;
    message.msg_iovlen = 1;
    alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int) * kMaxHandles)] = {};
    message.msg_control = control;
    message.msg_controllen = sizeof control;
    const ssize_t got = ::recvmsg(fd, &message, MSG_WAITALL);
    ::close(fd);
    if (got != static_cast<ssize_t>(sizeof reply)) {
        return Error::make(ErrorCode::IoFailure, "handle server '{}': a short answer", path);
    }
    std::vector<OsHandle> out;
    for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr; header = CMSG_NXTHDR(&message, header)) {
        if (header->cmsg_level == SOL_SOCKET && header->cmsg_type == SCM_RIGHTS) {
            const size_t n = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
            const int* fds = reinterpret_cast<const int*>(CMSG_DATA(header));
            for (size_t k = 0; k < n; ++k) out.push_back({OsHandle::Kind::Fd, static_cast<uint64_t>(fds[k])});
        }
    }
    if (reply.status == kStatusUnknownToken) {
        for (const OsHandle& h : out) closeOsHandle(h);
        return Error(ErrorCode::NotFound, "handle server: no handles under that token");
    }
    if (reply.status != kStatusServed || out.size() != reply.count) {
        for (const OsHandle& h : out) closeOsHandle(h);
        return Error(ErrorCode::Unsupported, "handle server: a handle that a socket cannot carry");
    }
    return out;
}

#endif   // !_WIN32

#if defined(__APPLE__)

// --- the Mach transport (macOS) --------------------------------------------------
//
// A bootstrap name the server checks in itself (launchd gives a process a
// name it registers dynamically, as long as it is not sandboxed); a request
// carries the token and a send-once right to answer on; the answer carries
// the ports, a descriptor turned into a fileport.

constexpr mach_msg_id_t kRequestId = 0x41544842;   // "ATHB"
constexpr mach_msg_id_t kStopId = 0x41544853;
constexpr mach_msg_id_t kReplyId = 0x41544852;

struct MachRequest {
    mach_msg_header_t header;
    uint64_t          token;
};
struct MachRequestReceived {
    MachRequest        request;
    mach_msg_trailer_t trailer;
};
struct MachReply {
    mach_msg_header_t          header;
    mach_msg_body_t            body;
    mach_msg_port_descriptor_t ports[kMaxHandles];
    uint32_t                   status;
    uint32_t                   count;
    uint32_t                   kinds[kMaxHandles];
};
struct MachReplyReceived {
    MachReply          reply;
    mach_msg_trailer_t trailer;
};

class MachServer final : public HandleServer {
public:
    static Result<std::unique_ptr<HandleServer>> make(const std::string& prefix) {
        auto server = std::unique_ptr<MachServer>(new MachServer());
        server->name_ = prefix + "." + std::to_string(processId()) + "." + hex64(randomToken()).substr(0, 8);
        mach_port_t port = MACH_PORT_NULL;
        const kern_return_t checked =
            bootstrap_check_in(bootstrap_port, const_cast<char*>(server->name_.c_str()), &port);
        if (checked != KERN_SUCCESS) {
            return Error::make(ErrorCode::IoFailure, "handle server '{}': bootstrap_check_in: {}", server->name_,
                               bootstrap_strerror(checked));
        }
        server->port_ = port;
        // A send right of our own, to wake the thread when the server goes.
        mach_port_insert_right(mach_task_self(), port, port, MACH_MSG_TYPE_MAKE_SEND);
        server->thread_ = std::thread([raw = server.get()] { raw->run(); });
        return std::unique_ptr<HandleServer>(std::move(server));
    }

    ~MachServer() override {
        if (thread_.joinable()) {
            mach_msg_header_t stop{};
            stop.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);
            stop.msgh_size = sizeof stop;
            stop.msgh_remote_port = port_;
            stop.msgh_id = kStopId;
            mach_msg(&stop, MACH_SEND_MSG, sizeof stop, 0, MACH_PORT_NULL, MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            thread_.join();
        }
        if (port_ != MACH_PORT_NULL) {
            mach_port_mod_refs(mach_task_self(), port_, MACH_PORT_RIGHT_SEND, -1);
            mach_port_mod_refs(mach_task_self(), port_, MACH_PORT_RIGHT_RECEIVE, -1);
        }
    }

    const std::string& name() const noexcept override { return name_; }
    void offer(uint64_t token, std::vector<OsHandle> handles) override { offers_.offer(token, std::move(handles)); }
    void withdraw(uint64_t token) override { offers_.withdraw(token); }
    uint64_t served() const noexcept override { return served_.load(); }
    uint64_t refused() const noexcept override { return refused_.load(); }

private:
    MachServer() = default;

    void run() {
        for (;;) {
            MachRequestReceived in{};
            const kern_return_t got = mach_msg(&in.request.header, MACH_RCV_MSG, 0, sizeof in, port_,
                                               MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
            if (got != KERN_SUCCESS) {
                if (got == MACH_RCV_TOO_LARGE) continue;
                return;
            }
            if (in.request.header.msgh_id == kStopId) return;
            if (in.request.header.msgh_id != kRequestId || in.request.header.msgh_remote_port == MACH_PORT_NULL) {
                mach_msg_destroy(&in.request.header);
                continue;
            }
            answer(in.request.header.msgh_remote_port, in.request.token);
        }
    }

    void answer(mach_port_t replyTo, uint64_t token) {
        MachReply reply{};
        reply.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_MOVE_SEND_ONCE, 0) | MACH_MSGH_BITS_COMPLEX;
        reply.header.msgh_size = sizeof reply;
        reply.header.msgh_remote_port = replyTo;
        reply.header.msgh_id = kReplyId;
        std::vector<OsHandle> handles;
        if (auto copies = offers_.copies(token); !copies) {
            reply.status = kStatusUnknownToken;
            refused_.fetch_add(1);
        } else {
            handles = std::move(*copies);
            reply.status = kStatusServed;
        }
        uint32_t count = 0;
        for (const OsHandle& h : handles) {
            if (count == kMaxHandles) break;
            mach_port_t port = MACH_PORT_NULL;
            if (h.kind == OsHandle::Kind::MachPort) {
                port = static_cast<mach_port_t>(h.value);
                mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1);   // the message moves one
            } else if (h.kind == OsHandle::Kind::Fd) {
                fileport_t file = MACH_PORT_NULL;
                if (fileport_makeport(static_cast<int>(h.value), &file) != 0) {
                    reply.status = kStatusUnsupported;
                    break;
                }
                port = file;
            }
            reply.ports[count].name = port;
            reply.ports[count].disposition = MACH_MSG_TYPE_MOVE_SEND;
            reply.ports[count].type = MACH_MSG_PORT_DESCRIPTOR;
            reply.kinds[count] = static_cast<uint32_t>(h.kind);
            ++count;
        }
        if (reply.status != kStatusServed) {
            for (uint32_t k = 0; k < count; ++k) {
                mach_port_mod_refs(mach_task_self(), reply.ports[k].name, MACH_PORT_RIGHT_SEND, -1);
            }
            count = 0;
        }
        // Unused descriptors still travel: they carry no port.
        for (uint32_t k = count; k < kMaxHandles; ++k) {
            reply.ports[k].name = MACH_PORT_NULL;
            reply.ports[k].disposition = MACH_MSG_TYPE_COPY_SEND;
            reply.ports[k].type = MACH_MSG_PORT_DESCRIPTOR;
        }
        reply.body.msgh_descriptor_count = kMaxHandles;
        reply.count = count;
        const kern_return_t sent = mach_msg(&reply.header, MACH_SEND_MSG, sizeof reply, 0, MACH_PORT_NULL,
                                            MACH_MSG_TIMEOUT_NONE, MACH_PORT_NULL);
        if (sent != KERN_SUCCESS) {
            mach_msg_destroy(&reply.header);
        } else if (reply.status == kStatusServed) {
            served_.fetch_add(1);
        }
        for (const OsHandle& h : handles) closeOsHandle(h);
    }

    std::string           name_;
    mach_port_t           port_ = MACH_PORT_NULL;
    std::thread           thread_;
    Offers                offers_;
    std::atomic<uint64_t> served_{0}, refused_{0};
};

Result<std::vector<OsHandle>> fetchOverMach(const std::string& name, uint64_t token,
                                            std::chrono::milliseconds timeout) {
    mach_port_t server = MACH_PORT_NULL;
    const kern_return_t found = bootstrap_look_up(bootstrap_port, name.c_str(), &server);
    if (found != KERN_SUCCESS) {
        return Error::make(ErrorCode::IoFailure, "handle server '{}' cannot be reached: {}", name,
                           bootstrap_strerror(found));
    }
    mach_port_t replyPort = MACH_PORT_NULL;
    mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &replyPort);
    MachRequest request{};
    request.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, MACH_MSG_TYPE_MAKE_SEND_ONCE);
    request.header.msgh_size = sizeof request;
    request.header.msgh_remote_port = server;
    request.header.msgh_local_port = replyPort;
    request.header.msgh_id = kRequestId;
    request.token = token;
    const kern_return_t sent = mach_msg(&request.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT, sizeof request, 0,
                                        MACH_PORT_NULL, static_cast<mach_msg_timeout_t>(timeout.count()),
                                        MACH_PORT_NULL);
    mach_port_deallocate(mach_task_self(), server);
    if (sent != KERN_SUCCESS) {
        mach_msg_destroy(&request.header);
        mach_port_mod_refs(mach_task_self(), replyPort, MACH_PORT_RIGHT_RECEIVE, -1);
        return Error::make(ErrorCode::IoFailure, "handle server '{}': the request was not sent ({:#x})", name,
                           static_cast<unsigned>(sent));
    }
    MachReplyReceived in{};
    const kern_return_t got = mach_msg(&in.reply.header, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof in, replyPort,
                                       static_cast<mach_msg_timeout_t>(timeout.count()), MACH_PORT_NULL);
    mach_port_mod_refs(mach_task_self(), replyPort, MACH_PORT_RIGHT_RECEIVE, -1);
    if (got != KERN_SUCCESS) {
        return Error::make(ErrorCode::IoFailure, "handle server '{}' did not answer in {} ms ({:#x})", name,
                           timeout.count(), static_cast<unsigned>(got));
    }
    const MachReply& reply = in.reply;
    if (reply.header.msgh_id != kReplyId || (reply.header.msgh_bits & MACH_MSGH_BITS_COMPLEX) == 0 ||
        reply.body.msgh_descriptor_count != kMaxHandles || reply.count > kMaxHandles) {
        mach_msg_destroy(&in.reply.header);
        return Error::make(ErrorCode::IoFailure, "handle server '{}': an answer of another shape", name);
    }
    std::vector<OsHandle> out;
    for (uint32_t k = 0; k < kMaxHandles; ++k) {
        const mach_port_t port = reply.ports[k].name;
        if (k >= reply.count || reply.status != kStatusServed) {
            if (port != MACH_PORT_NULL) mach_port_deallocate(mach_task_self(), port);
            continue;
        }
        if (reply.kinds[k] == static_cast<uint32_t>(OsHandle::Kind::Fd)) {
            const int fd = fileport_makefd(port);
            mach_port_deallocate(mach_task_self(), port);
            if (fd < 0) {
                for (const OsHandle& h : out) closeOsHandle(h);
                return Error::make(ErrorCode::IoFailure, "handle server '{}': a descriptor that did not arrive", name);
            }
            out.push_back({OsHandle::Kind::Fd, static_cast<uint64_t>(fd)});
        } else {
            out.push_back({OsHandle::Kind::MachPort, static_cast<uint64_t>(port)});
        }
    }
    if (reply.status == kStatusUnknownToken) {
        return Error(ErrorCode::NotFound, "handle server: no handles under that token");
    }
    if (reply.status != kStatusServed) {
        for (const OsHandle& h : out) closeOsHandle(h);
        return Error(ErrorCode::Unsupported, "handle server: a handle it could not send");
    }
    return out;
}

#endif   // __APPLE__

}   // namespace

uint64_t randomToken() {
#if defined(__APPLE__)
    uint64_t value = 0;
    arc4random_buf(&value, sizeof value);
    return value;
#else
    static std::mutex          guard;
    static std::random_device  device;
    std::lock_guard            lock(guard);
    return (static_cast<uint64_t>(device()) << 32) ^ static_cast<uint64_t>(device());
#endif
}

void closeOsHandle(OsHandle handle) noexcept {
    switch (handle.kind) {
        case OsHandle::Kind::MachPort:
#if defined(__APPLE__)
            if (handle.value != 0) mach_port_deallocate(mach_task_self(), static_cast<mach_port_t>(handle.value));
#endif
            break;
        case OsHandle::Kind::Fd:
#if !defined(_WIN32)
            ::close(static_cast<int>(handle.value));
#endif
            break;
        case OsHandle::Kind::None:
            break;
    }
}

Result<OsHandle> duplicateOsHandle(OsHandle handle) {
    switch (handle.kind) {
        case OsHandle::Kind::MachPort:
#if defined(__APPLE__)
            if (mach_port_mod_refs(mach_task_self(), static_cast<mach_port_t>(handle.value), MACH_PORT_RIGHT_SEND, 1) !=
                KERN_SUCCESS) {
                return Error(ErrorCode::IoFailure, "a Mach port that is no send right");
            }
            return handle;
#else
            return Error(ErrorCode::Unsupported, "Mach ports are macOS's");
#endif
        case OsHandle::Kind::Fd: {
#if !defined(_WIN32)
            const int copy = ::dup(static_cast<int>(handle.value));
            if (copy < 0) return Error::make(ErrorCode::IoFailure, "dup: {}", std::strerror(errno));
            return OsHandle{OsHandle::Kind::Fd, static_cast<uint64_t>(copy)};
#else
            return Error(ErrorCode::Unsupported, "descriptors are POSIX's");
#endif
        }
        case OsHandle::Kind::None:
            break;
    }
    return OsHandle{};
}

Result<std::unique_ptr<HandleServer>> HandleServer::create(const std::string& prefix, HandleTransport transport) {
    if (prefix.empty() || prefix.size() > 40 || prefix.find('/') != std::string::npos) {
        return Error::make(ErrorCode::InvalidArgument, "handle server: a prefix of 1 to 40 characters without '/'");
    }
#if defined(__APPLE__)
    if (transport == HandleTransport::Native) return MachServer::make(prefix);
#endif
#if !defined(_WIN32)
    (void)transport;
    return SocketServer::make(prefix);
#else
    (void)transport;
    return Error(ErrorCode::Unsupported, "handle server: Windows duplicates handles into a process instead");
#endif
}

Result<std::vector<OsHandle>> fetchHandles(const std::string& server, uint64_t token,
                                           std::chrono::milliseconds timeout) {
#if !defined(_WIN32)
    if (server.rfind("unix:", 0) == 0) return fetchOverSocket(server.substr(5), token, timeout);
#endif
#if defined(__APPLE__)
    return fetchOverMach(server, token, timeout);
#else
    (void)token;
    (void)timeout;
    return Error::make(ErrorCode::Unsupported, "handle server '{}': not a transport this system has", server);
#endif
}


// --- Metal's shared events as ports ------------------------------------------------

#if defined(__APPLE__)
namespace {

template <typename R, typename... A>
R send(void* receiver, const char* selector, A... args) {
    return reinterpret_cast<R (*)(void*, SEL, A...)>(objc_msgSend)(receiver, sel_registerName(selector), args...);
}

bool answers(void* receiver, const char* selector) {
    return receiver != nullptr &&
           send<BOOL>(receiver, "respondsToSelector:", sel_registerName(selector)) == YES;
}

}   // namespace
#endif

Result<OsHandle> exportMetalSharedEvent(void* event) {
#if defined(__APPLE__)
    if (event == nullptr || !answers(event, "newSharedEventHandle")) {
        return Error(ErrorCode::InvalidArgument, "not an MTLSharedEvent");
    }
    void* pool = objc_autoreleasePoolPush();
    void* handle = send<void*>(event, "newSharedEventHandle");
    if (!answers(handle, "eventPort")) {
        if (handle != nullptr) releaseMetalObject(handle);
        objc_autoreleasePoolPop(pool);
        return Error(ErrorCode::Unsupported, "this system's MTLSharedEventHandle names no port (eventPort)");
    }
    const mach_port_t port = send<unsigned>(handle, "eventPort");
    // The handle owns its right; one of our own outlives it.
    const kern_return_t kept = mach_port_mod_refs(mach_task_self(), port, MACH_PORT_RIGHT_SEND, 1);
    releaseMetalObject(handle);
    objc_autoreleasePoolPop(pool);
    if (port == MACH_PORT_NULL || kept != KERN_SUCCESS) {
        return Error(ErrorCode::IoFailure, "an MTLSharedEvent whose port could not be kept");
    }
    return OsHandle{OsHandle::Kind::MachPort, port};
#else
    (void)event;
    return Error(ErrorCode::Unsupported, "Metal shared events are macOS's");
#endif
}

Result<void*> importMetalSharedEvent(void* device, OsHandle port) {
#if defined(__APPLE__)
    if (port.kind != OsHandle::Kind::MachPort || port.value == 0) {
        return Error(ErrorCode::InvalidArgument, "an MTLSharedEvent comes as a Mach port");
    }
    if (!answers(device, "newSharedEventWithMachPort:")) {
        return Error(ErrorCode::Unsupported, "this system's MTLDevice makes no event from a port "
                                             "(newSharedEventWithMachPort:)");
    }
    // The event may take the right it is given: it gets one of its own.
    const mach_port_t right = static_cast<mach_port_t>(port.value);
    mach_port_mod_refs(mach_task_self(), right, MACH_PORT_RIGHT_SEND, 1);
    void* pool = objc_autoreleasePoolPush();
    void* event = send<void*>(device, "newSharedEventWithMachPort:", static_cast<unsigned>(right));
    objc_autoreleasePoolPop(pool);
    if (event == nullptr) {
        mach_port_mod_refs(mach_task_self(), right, MACH_PORT_RIGHT_SEND, -1);
        return Error(ErrorCode::DeviceFailure, "the device made no event from that port");
    }
    return event;
#else
    (void)device;
    (void)port;
    return Error(ErrorCode::Unsupported, "Metal shared events are macOS's");
#endif
}

uint64_t metalSharedEventValue(void* event) noexcept {
#if defined(__APPLE__)
    return event != nullptr ? send<uint64_t>(event, "signaledValue") : 0;
#else
    (void)event;
    return 0;
#endif
}

void setMetalSharedEventValue(void* event, uint64_t value) noexcept {
#if defined(__APPLE__)
    if (event != nullptr) send<void>(event, "setSignaledValue:", value);
#else
    (void)event;
    (void)value;
#endif
}

Result<void> metalChainEvents(void* queue, void* waitEvent, uint64_t waitValue, void* signalEvent,
                              uint64_t signalValue) {
#if defined(__APPLE__)
    if (queue == nullptr || waitEvent == nullptr || signalEvent == nullptr) {
        return Error(ErrorCode::InvalidArgument, "an event chain wants a queue and two events");
    }
    void* pool = objc_autoreleasePoolPush();
    void* commands = send<void*>(queue, "commandBuffer");
    if (commands == nullptr) {
        objc_autoreleasePoolPop(pool);
        return Error(ErrorCode::DeviceFailure, "the queue gave no command buffer");
    }
    send<void>(commands, "setLabel:", nullptr);
    send<void>(commands, "encodeWaitForEvent:value:", waitEvent, waitValue);
    send<void>(commands, "encodeSignalEvent:value:", signalEvent, signalValue);
    send<void>(commands, "commit");
    objc_autoreleasePoolPop(pool);
    return ok();
#else
    (void)queue;
    (void)waitEvent;
    (void)waitValue;
    (void)signalEvent;
    (void)signalValue;
    return Error(ErrorCode::Unsupported, "Metal events are macOS's");
#endif
}

void releaseMetalObject(void* object) noexcept {
#if defined(__APPLE__)
    if (object != nullptr) send<void>(object, "release");
#else
    (void)object;
#endif
}

// --- a child process on pipes --------------------------------------------------------

Result<ChildProcess> ChildProcess::spawn(const std::vector<std::string>& argv, const std::vector<std::string>& env) {
#if !defined(_WIN32)
    if (argv.empty()) return Error(ErrorCode::InvalidArgument, "a child process wants a program");
    int toChild[2] = {-1, -1};
    int fromChild[2] = {-1, -1};
    if (::pipe(toChild) != 0 || ::pipe(fromChild) != 0) {
        return Error::make(ErrorCode::IoFailure, "pipe: {}", std::strerror(errno));
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, toChild[0], 0);
    posix_spawn_file_actions_adddup2(&actions, fromChild[1], 1);
    posix_spawn_file_actions_addclose(&actions, toChild[1]);
    posix_spawn_file_actions_addclose(&actions, fromChild[0]);
    std::vector<char*> args;
    for (const std::string& a : argv) args.push_back(const_cast<char*>(a.c_str()));
    args.push_back(nullptr);
    std::vector<std::string> environment;
    for (char** e = environ; *e != nullptr; ++e) environment.emplace_back(*e);
    for (const std::string& extra : env) environment.push_back(extra);
    std::vector<char*> envp;
    for (std::string& e : environment) envp.push_back(e.data());
    envp.push_back(nullptr);
    pid_t pid = -1;
    const int spawned = ::posix_spawn(&pid, argv[0].c_str(), &actions, nullptr, args.data(), envp.data());
    posix_spawn_file_actions_destroy(&actions);
    ::close(toChild[0]);
    ::close(fromChild[1]);
    if (spawned != 0) {
        ::close(toChild[1]);
        ::close(fromChild[0]);
        return Error::make(ErrorCode::IoFailure, "cannot start {}: {}", argv[0], std::strerror(spawned));
    }
    ::signal(SIGPIPE, SIG_IGN);
    ChildProcess child;
    child.pid_ = pid;
    child.in_ = toChild[1];
    child.out_ = fromChild[0];
    return child;
#else
    (void)argv;
    (void)env;
    return Error(ErrorCode::Unsupported, "a child process on pipes: POSIX only");
#endif
}

ChildProcess::ChildProcess(ChildProcess&& other) noexcept
    : pid_(std::exchange(other.pid_, -1)),
      in_(std::exchange(other.in_, -1)),
      out_(std::exchange(other.out_, -1)),
      pending_(std::move(other.pending_)) {}

ChildProcess& ChildProcess::operator=(ChildProcess&& other) noexcept {
    if (this != &other) {
        (void)wait();
        pid_ = std::exchange(other.pid_, -1);
        in_ = std::exchange(other.in_, -1);
        out_ = std::exchange(other.out_, -1);
        pending_ = std::move(other.pending_);
    }
    return *this;
}

ChildProcess::~ChildProcess() {
    (void)wait();
}

Result<void> ChildProcess::writeLine(const std::string& line) {
#if !defined(_WIN32)
    if (in_ < 0) return Error(ErrorCode::IoFailure, "the child's input is closed");
    const std::string text = line + "\n";
    size_t sent = 0;
    while (sent < text.size()) {
        const ssize_t n = ::write(in_, text.data() + sent, text.size() - sent);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return Error::make(ErrorCode::IoFailure, "writing to the child: {}", std::strerror(errno));
        sent += static_cast<size_t>(n);
    }
    return ok();
#else
    (void)line;
    return Error(ErrorCode::Unsupported, "POSIX only");
#endif
}

Result<std::string> ChildProcess::readLine(std::chrono::milliseconds timeout) {
#if !defined(_WIN32)
    const auto until = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        if (const size_t end = pending_.find('\n'); end != std::string::npos) {
            std::string line = pending_.substr(0, end);
            pending_.erase(0, end + 1);
            return line;
        }
        if (out_ < 0) return Error(ErrorCode::IoFailure, "the child's output is closed");
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(until - std::chrono::steady_clock::now());
        if (left.count() <= 0) return Error(ErrorCode::IoFailure, "the child wrote no line in time");
        pollfd wait{out_, POLLIN, 0};
        const int ready = ::poll(&wait, 1, static_cast<int>(left.count()));
        if (ready < 0 && errno == EINTR) continue;
        if (ready <= 0) continue;
        char chunk[65536];
        const ssize_t n = ::read(out_, chunk, sizeof chunk);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            ::close(out_);
            out_ = -1;
            continue;
        }
        pending_.append(chunk, static_cast<size_t>(n));
    }
#else
    (void)timeout;
    return Error(ErrorCode::Unsupported, "POSIX only");
#endif
}

Result<int> ChildProcess::wait() {
#if !defined(_WIN32)
    if (in_ >= 0) {
        ::close(in_);
        in_ = -1;
    }
    if (out_ >= 0) {
        ::close(out_);
        out_ = -1;
    }
    if (pid_ <= 0) return 0;
    int status = 0;
    while (::waitpid(static_cast<pid_t>(pid_), &status, 0) < 0) {
        if (errno != EINTR) {
            pid_ = -1;
            return Error::make(ErrorCode::IoFailure, "waitpid: {}", std::strerror(errno));
        }
    }
    pid_ = -1;
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#else
    return 0;
#endif
}

}   // namespace athenea::platform
