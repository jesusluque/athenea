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
