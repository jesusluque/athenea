// Copyright (c) 2026 jesus luque.
//
// Every operating-system call the engine makes outside its dependencies lives
// behind this header. Linux and macOS are implemented; Windows is a later
// port, and this is the file it starts from.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::platform {

/// A read-only mapping of a whole file.
///
/// Mapped rather than read: a trained splat cloud is hundreds of megabytes and
/// the loader touches it in blocks from several threads, each of which should
/// fault in only the pages it reads.
class MappedFile {
public:
    [[nodiscard]] static Result<MappedFile> open(const std::filesystem::path& path);

    MappedFile() = default;
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile();

    [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
        return {static_cast<const std::byte*>(data_), size_};
    }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    /// Modification time in nanoseconds, for cache keys (path@mtime:size).
    [[nodiscard]] int64_t modifiedNs() const noexcept { return mtimeNs_; }

private:
    void*   data_ = nullptr;
    size_t  size_ = 0;
    int64_t mtimeNs_ = 0;
};

/// MEMORY ANOTHER PROCESS SEES: a named region both map, read and write.
/// POSIX shared memory on macOS and Linux (`shm_open`; a name of at most 30
/// characters, no '/', which the POSIX name gets in front -- what Python's
/// `multiprocessing.shared_memory` takes too), a named file mapping on Windows
/// (`Local\<name>`). The creator owns the name and removes it when it goes;
/// one that opens it only maps it. Any thread.
class SharedMemory {
public:
    [[nodiscard]] static Result<SharedMemory> create(const std::string& name, size_t bytes);
    [[nodiscard]] static Result<SharedMemory> open(const std::string& name, size_t bytes);

    SharedMemory() = default;
    SharedMemory(SharedMemory&&) noexcept;
    SharedMemory& operator=(SharedMemory&&) noexcept;
    SharedMemory(const SharedMemory&) = delete;
    SharedMemory& operator=(const SharedMemory&) = delete;
    ~SharedMemory();

    [[nodiscard]] std::span<std::byte> bytes() const noexcept { return {static_cast<std::byte*>(data_), size_}; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }

private:
    void release() noexcept;
    void*       data_ = nullptr;
    size_t      size_ = 0;
    std::string name_;
    bool        owner_ = false;
    void*       handle_ = nullptr;   // Windows: the mapping
};

/// This process's id. Any thread.
[[nodiscard]] uint32_t processId() noexcept;

// --- handles another process can be given (PlatformIpc.cpp) --------------------
//
// A frame shared with another process is memory both map (SharedMemory) and
// device objects only the system can carry across: a Metal shared event is a
// Mach port, a Vulkan buffer's memory or timeline semaphore a file
// descriptor. A HandleServer hands them to whoever asks with the right token.
// Nothing here touches a device except where it says so.

/// A handle the system carries between processes: a Mach send right (macOS)
/// or a file descriptor (POSIX). The holder closes it (closeOsHandle).
struct OsHandle {
    enum class Kind : uint32_t { None = 0, MachPort = 1, Fd = 2 };
    Kind     kind = Kind::None;
    uint64_t value = 0;
};
/// Gives the right or the descriptor back. Any thread.
void closeOsHandle(OsHandle handle) noexcept;
/// A second reference to the same port or descriptor, for a holder that
/// gives its own away. Any thread.
[[nodiscard]] Result<OsHandle> duplicateOsHandle(OsHandle handle);

/// How a HandleServer is reached. `Native` is a Mach bootstrap name on macOS
/// (a descriptor travels as a fileport there) and a Unix socket elsewhere;
/// `Socket` is the Unix socket on any POSIX system, descriptors only.
enum class HandleTransport { Native, Socket };

/// ANSWERS fetchHandles: a thread of its own receives a request carrying a
/// token and replies with copies of the handles offered under it, so one
/// offer serves any number of fetches. The token is a capability: only a
/// process told it (in an MCP answer) can fetch. Every method: any thread.
class HandleServer {
public:
    [[nodiscard]] static Result<std::unique_ptr<HandleServer>> create(
        const std::string& prefix, HandleTransport transport = HandleTransport::Native);
    virtual ~HandleServer() = default;
    /// What fetchHandles takes to reach this server.
    [[nodiscard]] virtual const std::string& name() const noexcept = 0;
    /// The server owns `handles` from now until withdrawn or gone.
    virtual void offer(uint64_t token, std::vector<OsHandle> handles) = 0;
    virtual void withdraw(uint64_t token) = 0;
    /// Fetches answered with handles, and fetches refused (a token not offered).
    [[nodiscard]] virtual uint64_t served() const noexcept = 0;
    [[nodiscard]] virtual uint64_t refused() const noexcept = 0;
};

/// The handles `server` offers under `token`, the caller's now. NotFound for
/// a token the server does not know; IoFailure where the server cannot be
/// reached within `timeout`. Any thread.
[[nodiscard]] Result<std::vector<OsHandle>> fetchHandles(const std::string& server, uint64_t token,
                                                         std::chrono::milliseconds timeout = std::chrono::seconds(5));

/// A random 64-bit token from the system's generator. Any thread.
[[nodiscard]] uint64_t randomToken();

/// THE METAL SIDE OF A SHARED EVENT (macOS; elsewhere Unsupported). An
/// MTLSharedEvent crosses processes as the Mach port its MTLSharedEventHandle
/// holds. Apple carries that port only inside XPC messages, and XPC needs a
/// service launchd was told of, which a process started by hand is not; so
/// the port is taken from the handle (`eventPort`) and the event made again
/// from it (`-[MTLDevice newSharedEventWithMachPort:]`). Both are Metal's own
/// methods, absent from its public headers: asked for by name, and refused
/// with Unsupported where a system does not answer to them.
/// `mtlSharedEvent` is an id<MTLSharedEvent>. Any thread.
[[nodiscard]] Result<OsHandle> exportMetalSharedEvent(void* mtlSharedEvent);
/// An id<MTLSharedEvent> on `mtlDevice` over the event `port` names, retained
/// once for the caller (releaseMetalObject). `port` stays the caller's. Any thread.
[[nodiscard]] Result<void*> importMetalSharedEvent(void* mtlDevice, OsHandle port);
/// The value the event holds now, and a value set from the host. Any
/// thread; neither waits.
[[nodiscard]] uint64_t metalSharedEventValue(void* mtlSharedEvent) noexcept;
void setMetalSharedEventValue(void* mtlSharedEvent, uint64_t value) noexcept;
/// ONE COMMAND BUFFER ON `mtlQueue` THAT WAITS, THEN SIGNALS: the device waits
/// for `waitEvent` to reach `waitValue` and then sets `signalEvent` to
/// `signalValue`. How an event slang-rhi did not make is tied to one it did
/// (gpu::SharedFence): the chain relies on the two events alone, not on any
/// order between command buffers. Committed, never waited for. The device's
/// one caller (the frame thread).
[[nodiscard]] Result<void> metalChainEvents(void* mtlQueue, void* waitEvent, uint64_t waitValue, void* signalEvent,
                                            uint64_t signalValue);
/// -[NSObject release], for what importMetalSharedEvent made. Any thread.
void releaseMetalObject(void* object) noexcept;

/// A PROCESS STARTED WITH ITS STANDARD INPUT AND OUTPUT AS PIPES: a test's
/// second process, or a client starting athenea-mcp. Standard error is
/// inherited. POSIX only (Unsupported on Windows). Every method: the thread
/// that holds it.
class ChildProcess {
public:
    /// `argv[0]` is the program's path; `env` entries ("NAME=value") are
    /// added to this process's environment.
    [[nodiscard]] static Result<ChildProcess> spawn(const std::vector<std::string>& argv,
                                                    const std::vector<std::string>& env = {});
    ChildProcess() = default;
    ChildProcess(ChildProcess&&) noexcept;
    ChildProcess& operator=(ChildProcess&&) noexcept;
    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;
    /// Closes its input and waits for it to end.
    ~ChildProcess();

    [[nodiscard]] Result<void> writeLine(const std::string& line);
    /// The next line it writes, without the newline; IoFailure when it closes
    /// its output or `timeout` passes first.
    [[nodiscard]] Result<std::string> readLine(std::chrono::milliseconds timeout = std::chrono::seconds(60));
    /// Closes its input and waits for its exit status.
    [[nodiscard]] Result<int> wait();
    [[nodiscard]] int64_t pid() const noexcept { return pid_; }

private:
    int64_t     pid_ = -1;
    int         in_ = -1;    // its standard input, written here
    int         out_ = -1;   // its standard output, read here
    std::string pending_;
};

/// Removes the shared memory regions named `<prefix><pid>-...` whose process
/// is gone -- what a server that crashed with a region open left. Where the
/// system lists its regions (Linux, /dev/shm); macOS lists none, and a
/// Windows mapping goes with its last process: nothing to do there. Returns
/// how many it removed. Any thread.
size_t removeStaleSharedMemory(const std::string& prefix) noexcept;

/// Sleeps for `duration` and wakes on time, to tens of microseconds.
///
/// A plain sleep may overrun by the timer slack the OS allows itself: on macOS
/// that measured 7 ms on average and 10 at worst, a quarter of a 25 fps frame.
/// On macOS the thread holds a real-time (time constraint) policy only while
/// it sleeps -- held while it renders, a thread that overruns its computation
/// budget is demoted -- and wakes 36 µs late at worst. On Linux the thread's
/// timer slack goes to 1 ns.
void sleepPrecisely(std::chrono::nanoseconds duration);

/// An OpenGL 4.5 context made current on this thread with no window, for a
/// library that draws with OpenGL itself -- USD's Storm on Linux, whose
/// HgiGL expects one current. Through EGL on a GPU device (EGL_EXT_platform_
/// device), with a 1x1 pbuffer; made once per process. False where none can
/// be made, or on macOS, where Storm draws through Metal and needs none.
[[nodiscard]] bool makeHeadlessGlContextCurrent();

/// A second descriptor for the same open file or memory object, for a
/// library that takes ownership of the one it is given (OIDN importing a
/// Vulkan buffer's memory). -1 where there is none (Windows).
[[nodiscard]] int duplicateDescriptor(int descriptor);

/// Every shared library the process has loaded, by path.
[[nodiscard]] std::vector<std::string> loadedLibraries();

/// Where a user's caches go: ~/Library/Caches on macOS, $XDG_CACHE_HOME or
/// ~/.cache on Linux.
[[nodiscard]] std::filesystem::path cacheDirectory();

/// The directory holding the running executable.
[[nodiscard]] std::filesystem::path executableDir();

/// The directory holding the image this code was linked into: the
/// executable for a program, the library for a plugin another program
/// loads (hdAthenea inside usdview, Houdini or Blender), whose executable
/// says nothing about where the engine's files are. Empty where the image
/// cannot be named.
[[nodiscard]] std::filesystem::path moduleDir();

/// `name` from the environment, or empty.
[[nodiscard]] std::string env(const char* name);

/// A window's Metal layer drawn at the window's backing scale (2 on a Retina
/// display), so a drawable of the framebuffer's pixels maps one to one.
/// `nsWindow` is an NSWindow*; a no-op elsewhere.
void matchLayerToBacking(void* nsWindow);

/// The extended dynamic range a window's screen offers: its peak over its
/// reference white (1.0 on a standard display; 2 to 16 on an HDR one, as
/// its brightness stands). `nsWindow` is an NSWindow*; 1.0 elsewhere.
[[nodiscard]] double extendedRangeHeadroom(void* nsWindow);

/// A window's Metal layer asked for extended range content in linear
/// Display P3 (1.0 the reference white, values above it the headroom):
/// what a float surface shows as HDR. Returns whether it was set;
/// false elsewhere.
bool enableExtendedRange(void* nsWindow);

/// A Metal buffer with private storage and hazard tracking, on `mtlDevice`:
/// what OIDN will share, and what slang-rhi -- which tracks nothing and
/// orders its own work -- will not make. Null off macOS or on failure. The
/// caller releases it with releaseMetalBuffer.
[[nodiscard]] void* newTrackedMetalBuffer(void* mtlDevice, uint64_t bytes);
void releaseMetalBuffer(void* mtlBuffer);

/// How much memory `mtlDevice` (an MTLDevice*) can use before it starts to
/// hurt: Metal's `recommendedMaxWorkingSetSize`, in bytes. 0 off macOS or
/// for a null device.
[[nodiscard]] uint64_t metalRecommendedWorkingSet(void* mtlDevice);
/// What `mtlDevice` has allocated for this process now, in bytes: Metal's
/// `currentAllocatedSize`, every buffer, texture and structure counted.
/// 0 off macOS or for a null device.
[[nodiscard]] uint64_t metalAllocatedSize(void* mtlDevice);

/// Physical memory the system could hand out now without swapping, in
/// bytes: free, inactive and purgeable pages (Mach's vm statistics) on
/// macOS, MemAvailable on Linux. 0 where it cannot be read. On unified
/// memory this is the device's memory too, and what a budget has to leave
/// the window server and everything else on the machine.
[[nodiscard]] uint64_t availablePhysicalMemory();

/// The size of a page of virtual memory.
[[nodiscard]] uint64_t pageSize();

/// `bytes` rounded up to whole pages, mapped readable and writable, zeroed and
/// page-aligned: memory a device can be handed in place. Null on failure.
/// Given back with unmapPages and the same byte count.
[[nodiscard]] void* mapPages(uint64_t bytes);
void unmapPages(void* pages, uint64_t bytes);

/// A shared-storage Metal buffer over `pages` as they are, no copy, where the
/// device reads host memory in place (unified memory). `pages` must be
/// page-aligned and `bytes` whole pages, and the pages must outlive the
/// buffer. Null off macOS, on a discrete GPU, for memory that is not pages, or
/// on failure; released with releaseMetalBuffer. What AOFX's `Gpu::borrow`
/// binds.
[[nodiscard]] void* newMetalBufferOverPages(void* mtlDevice, const void* pages, uint64_t bytes);

/// WRITING A FILE WHOLE OR NOT AT ALL.
///
/// A conversion that fails half way -- the device runs out, the disk fills,
/// the process is killed -- must not leave a file under the name it was
/// asked for: the next step reads that name and takes a stage of half a
/// cloud for a cloud. So a file is written under another name beside it and
/// takes its own name only once it is complete.
///
/// `partialPathFor` is that other name: the same directory, so the rename
/// does not cross a filesystem and is one step; the same extension, so a
/// writer that chooses its format by the extension (USD's `.usda`/`.usdc`)
/// writes the same format; and this process's id in it, so two conversions
/// writing the same output do not share a partial file.
[[nodiscard]] std::filesystem::path partialPathFor(const std::filesystem::path& path);

/// `from` renamed to `to`, replacing a file already there in one step (POSIX
/// `rename`; `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING` in the Windows
/// port).
[[nodiscard]] Result<void> replaceFile(const std::filesystem::path& from, const std::filesystem::path& to);

/// Removes `path` if it is there. Whether it was is nobody's business: what
/// calls this is cleaning up after a failure it is already reporting.
void removeFile(const std::filesystem::path& path) noexcept;

/// `write` given `partialPathFor(path)`, and that file renamed to `path` when
/// it succeeds. When it fails the partial file is removed and `path` is left
/// as it was -- absent, or the previous complete file.
[[nodiscard]] Result<void> writeAtomically(
    const std::filesystem::path& path,
    const std::function<Result<void>(const std::filesystem::path& partial)>& write);

}   // namespace athenea::platform
