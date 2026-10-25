// Copyright (c) 2026 jesus luque.
#include "athenea/core/Platform.h"

#if defined(ATHENEA_HAVE_EGL)
#define EGL_EGLEXT_PROTOTYPES 1   // the device extensions' declarations
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#include <psapi.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#elif defined(__linux__)
#include <link.h>
#include <sys/prctl.h>
#endif

#include <thread>

namespace athenea::platform {

#if defined(_WIN32)
Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const HANDLE handle = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    LARGE_INTEGER size{};
    FILETIME written{};
    if (!::GetFileSizeEx(handle, &size) || !::GetFileTime(handle, nullptr, nullptr, &written)) {
        ::CloseHandle(handle);
        return Error::make(ErrorCode::IoFailure, "cannot stat '{}'", path.string());
    }
    MappedFile file;
    file.size_ = static_cast<size_t>(size.QuadPart);
    // FILETIME counts 100 ns from 1601; the Unix epoch is 11644473600 s later.
    const uint64_t ticks = (uint64_t{written.dwHighDateTime} << 32) | written.dwLowDateTime;
    file.mtimeNs_ = (static_cast<int64_t>(ticks) - 116444736000000000LL) * 100;
    if (file.size_ > 0) {
        const HANDLE mapping = ::CreateFileMappingW(handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        void* mapped = mapping != nullptr ? ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) : nullptr;
        if (mapping != nullptr) {
            ::CloseHandle(mapping);   // the view keeps the mapping alive
        }
        if (mapped == nullptr) {
            ::CloseHandle(handle);
            return Error::make(ErrorCode::IoFailure, "cannot map '{}'", path.string());
        }
        file.data_ = mapped;
    }
    ::CloseHandle(handle);
    return file;
}
#else
Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return Error::make(ErrorCode::IoFailure, "cannot open '{}'", path.string());
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0) {
        ::close(fd);
        return Error::make(ErrorCode::IoFailure, "cannot stat '{}'", path.string());
    }
    MappedFile file;
    file.size_ = static_cast<size_t>(info.st_size);
#if defined(__APPLE__)
    file.mtimeNs_ = int64_t{info.st_mtimespec.tv_sec} * 1'000'000'000 +
                    info.st_mtimespec.tv_nsec;
#else
    file.mtimeNs_ = int64_t{info.st_mtim.tv_sec} * 1'000'000'000 + info.st_mtim.tv_nsec;
#endif
    if (file.size_ > 0) {
        void* mapped = ::mmap(nullptr, file.size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            ::close(fd);
            return Error::make(ErrorCode::IoFailure, "cannot map '{}'", path.string());
        }
        file.data_ = mapped;
    }
    ::close(fd);   // the mapping keeps the file alive
    return file;
}
#endif

namespace {

void unmapFile(void* data, size_t size) {
#if defined(_WIN32)
    (void)size;
    ::UnmapViewOfFile(data);
#else
    ::munmap(data, size);
#endif
}

}   // namespace

MappedFile::MappedFile(MappedFile&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      mtimeNs_(other.mtimeNs_) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (data_ != nullptr) {
            unmapFile(data_, size_);
        }
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        mtimeNs_ = other.mtimeNs_;
    }
    return *this;
}

MappedFile::~MappedFile() {
    if (data_ != nullptr) {
        unmapFile(data_, size_);
    }
}

namespace {

Result<void> sharedName(const std::string& name) {
    if (name.empty() || name.size() > 30 || name.find('/') != std::string::npos ||
        name.find('\\') != std::string::npos) {
        return Error::make(ErrorCode::InvalidArgument,
                           "shared memory '{}': a name of 1 to 30 characters without '/' or '\\'", name);
    }
    return ok();
}

}   // namespace

Result<SharedMemory> SharedMemory::create(const std::string& name, size_t bytes) {
    ATHENEA_TRY(sharedName(name));
    if (bytes == 0) {
        return Error::make(ErrorCode::InvalidArgument, "shared memory '{}' of no bytes", name);
    }
    SharedMemory region;
    region.name_ = name;
    region.size_ = bytes;
    region.owner_ = true;
#if defined(_WIN32)
    const std::wstring wide = L"Local\\" + std::wstring(name.begin(), name.end());
    const HANDLE mapping = ::CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                                static_cast<DWORD>(uint64_t{bytes} >> 32),
                                                static_cast<DWORD>(bytes & 0xFFFFFFFFu), wide.c_str());
    if (mapping == nullptr || ::GetLastError() == ERROR_ALREADY_EXISTS) {
        if (mapping != nullptr) ::CloseHandle(mapping);
        return Error::make(ErrorCode::IoFailure, "cannot create shared memory '{}'", name);
    }
    region.handle_ = mapping;
    region.data_ = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
    if (region.data_ == nullptr) {
        return Error::make(ErrorCode::IoFailure, "cannot map shared memory '{}'", name);
    }
#else
    const std::string posix = "/" + name;
    const int fd = ::shm_open(posix.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        return Error::make(ErrorCode::IoFailure, "cannot create shared memory '{}': {}", name, std::strerror(errno));
    }
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        ::close(fd);
        ::shm_unlink(posix.c_str());
        return Error::make(ErrorCode::IoFailure, "cannot size shared memory '{}'", name);
    }
    void* mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        ::shm_unlink(posix.c_str());
        return Error::make(ErrorCode::IoFailure, "cannot map shared memory '{}'", name);
    }
    region.data_ = mapped;
#endif
    return region;
}

Result<SharedMemory> SharedMemory::open(const std::string& name, size_t bytes) {
    ATHENEA_TRY(sharedName(name));
    SharedMemory region;
    region.name_ = name;
    region.size_ = bytes;
#if defined(_WIN32)
    const std::wstring wide = L"Local\\" + std::wstring(name.begin(), name.end());
    const HANDLE mapping = ::OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wide.c_str());
    if (mapping == nullptr) {
        return Error::make(ErrorCode::NotFound, "no shared memory '{}'", name);
    }
    region.handle_ = mapping;
    region.data_ = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, bytes);
    if (region.data_ == nullptr) {
        return Error::make(ErrorCode::IoFailure, "cannot map shared memory '{}'", name);
    }
#else
    const std::string posix = "/" + name;
    const int fd = ::shm_open(posix.c_str(), O_RDWR, 0600);
    if (fd < 0) {
        return Error::make(ErrorCode::NotFound, "no shared memory '{}': {}", name, std::strerror(errno));
    }
    void* mapped = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (mapped == MAP_FAILED) {
        return Error::make(ErrorCode::IoFailure, "cannot map shared memory '{}'", name);
    }
    region.data_ = mapped;
#endif
    return region;
}

void SharedMemory::release() noexcept {
    if (data_ != nullptr) {
        unmapFile(data_, size_);
        data_ = nullptr;
    }
#if defined(_WIN32)
    if (handle_ != nullptr) {
        ::CloseHandle(static_cast<HANDLE>(handle_));
        handle_ = nullptr;
    }
#else
    if (owner_ && !name_.empty()) {
        ::shm_unlink(("/" + name_).c_str());
    }
#endif
    owner_ = false;
}

SharedMemory::SharedMemory(SharedMemory&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      name_(std::move(other.name_)),
      owner_(std::exchange(other.owner_, false)),
      handle_(std::exchange(other.handle_, nullptr)) {}

SharedMemory& SharedMemory::operator=(SharedMemory&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        size_ = std::exchange(other.size_, 0);
        name_ = std::move(other.name_);
        owner_ = std::exchange(other.owner_, false);
        handle_ = std::exchange(other.handle_, nullptr);
    }
    return *this;
}

SharedMemory::~SharedMemory() {
    release();
}

void sleepPrecisely(std::chrono::nanoseconds duration) {
    if (duration.count() <= 0) {
        return;
    }
#if defined(__APPLE__)
    const mach_port_t self = pthread_mach_thread_np(pthread_self());
    static const double absPerNs = [] {
        mach_timebase_info_data_t base{};
        mach_timebase_info(&base);
        return static_cast<double>(base.denom) / static_cast<double>(base.numer);
    }();
    thread_time_constraint_policy_data_t realtime{};
    realtime.period = 0;
    realtime.computation = static_cast<uint32_t>(1e6 * absPerNs);
    realtime.constraint = static_cast<uint32_t>(2e6 * absPerNs);
    realtime.preemptible = 1;
    thread_policy_set(self, THREAD_TIME_CONSTRAINT_POLICY, reinterpret_cast<thread_policy_t>(&realtime),
                      THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    std::this_thread::sleep_for(duration);
    thread_standard_policy_data_t standard{};
    thread_policy_set(self, THREAD_STANDARD_POLICY, reinterpret_cast<thread_policy_t>(&standard),
                      THREAD_STANDARD_POLICY_COUNT);
#elif defined(__linux__)
    thread_local const bool slack = prctl(PR_SET_TIMERSLACK, 1UL, 0UL, 0UL, 0UL) == 0;
    (void)slack;
    std::this_thread::sleep_for(duration);
#else
    std::this_thread::sleep_for(duration);
#endif
}

int duplicateDescriptor(int descriptor) {
#if defined(_WIN32)
    (void)descriptor;
    return -1;
#else
    return ::dup(descriptor);
#endif
}

bool makeHeadlessGlContextCurrent() {
#if defined(ATHENEA_HAVE_EGL)
    static EGLDisplay display = EGL_NO_DISPLAY;
    static EGLContext context = EGL_NO_CONTEXT;
    static EGLSurface surface = EGL_NO_SURFACE;
    static bool tried = false;
    if (!tried) {
        tried = true;
        // A display on the first GPU device, needing no X server.
        // glvnd's libEGL dispatches the device extension without exporting
        // it: found by name.
        using QueryDevices = EGLBoolean (*)(EGLint, EGLDeviceEXT*, EGLint*);
        const auto queryDevices = reinterpret_cast<QueryDevices>(eglGetProcAddress("eglQueryDevicesEXT"));
        EGLDeviceEXT devices[8];
        EGLint count = 0;
        if (queryDevices != nullptr && queryDevices(8, devices, &count) && count > 0) {
            for (EGLint k = 0; k < count && display == EGL_NO_DISPLAY; ++k) {
                EGLDisplay candidate = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[k], nullptr);
                EGLint major = 0;
                EGLint minor = 0;
                if (candidate != EGL_NO_DISPLAY && eglInitialize(candidate, &major, &minor)) {
                    display = candidate;
                }
            }
        }
        if (display == EGL_NO_DISPLAY) {
            return false;
        }
        const EGLint attributes[] = {EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_RED_SIZE,        8,               EGL_GREEN_SIZE,      8,
                                     EGL_BLUE_SIZE,       8,               EGL_DEPTH_SIZE,      24,
                                     EGL_NONE};
        EGLConfig config = nullptr;
        EGLint configs = 0;
        if (!eglChooseConfig(display, attributes, &config, 1, &configs) || configs == 0 ||
            !eglBindAPI(EGL_OPENGL_API)) {
            return false;
        }
        const EGLint pbuffer[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
        surface = eglCreatePbufferSurface(display, config, pbuffer);
        // Compatibility, not core: Storm's state holder and draws use enums a
        // core profile refuses (measured: GL "invalid enum" and "invalid
        // operation" at every draw, and Kitchen_set drawn as nothing).
        const EGLint version[] = {EGL_CONTEXT_MAJOR_VERSION,       4,
                                  EGL_CONTEXT_MINOR_VERSION,       5,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_COMPATIBILITY_PROFILE_BIT,
                                  EGL_NONE};
        context = eglCreateContext(display, config, EGL_NO_CONTEXT, version);
        if (surface == EGL_NO_SURFACE || context == EGL_NO_CONTEXT) {
            return false;
        }
    }
    return context != EGL_NO_CONTEXT && eglMakeCurrent(display, surface, surface, context) == EGL_TRUE;
#else
    return false;
#endif
}

std::vector<std::string> loadedLibraries() {
    std::vector<std::string> out;
#if defined(__APPLE__)
    const uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i) {
        if (const char* name = _dyld_get_image_name(i); name != nullptr) {
            out.emplace_back(name);
        }
    }
#elif defined(__linux__)
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) {
            if (info->dlpi_name != nullptr && info->dlpi_name[0] != '\0') {
                static_cast<std::vector<std::string>*>(data)->emplace_back(info->dlpi_name);
            }
            return 0;
        },
        &out);
#elif defined(_WIN32)
    HMODULE modules[1024];
    DWORD needed = 0;
    if (::EnumProcessModules(::GetCurrentProcess(), modules, sizeof(modules), &needed)) {
        const DWORD count = std::min<DWORD>(needed / sizeof(HMODULE), 1024);
        for (DWORD i = 0; i < count; ++i) {
            wchar_t name[MAX_PATH];
            if (::GetModuleFileNameW(modules[i], name, MAX_PATH) > 0) {
                out.emplace_back(std::filesystem::path(name).string());
            }
        }
    }
#endif
    return out;
}

std::filesystem::path cacheDirectory() {
#if defined(__APPLE__)
    return std::filesystem::path(env("HOME")) / "Library" / "Caches";
#elif defined(_WIN32)
    return std::filesystem::path(env("LOCALAPPDATA"));
#else
    if (std::string xdg = env("XDG_CACHE_HOME"); !xdg.empty()) {
        return xdg;
    }
    return std::filesystem::path(env("HOME")) / ".cache";
#endif
}

std::filesystem::path executableDir() {
#if defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string buffer(size, '\0');
    if (_NSGetExecutablePath(buffer.data(), &size) == 0) {
        return std::filesystem::weakly_canonical(buffer.c_str()).parent_path();
    }
    return {};
#elif defined(_WIN32)
    wchar_t name[MAX_PATH];
    const DWORD length = ::GetModuleFileNameW(nullptr, name, MAX_PATH);
    return length > 0 && length < MAX_PATH ? std::filesystem::path(name).parent_path() : std::filesystem::path{};
#else
    std::error_code failed;
    auto self = std::filesystem::read_symlink("/proc/self/exe", failed);
    return failed ? std::filesystem::path{} : self.parent_path();
#endif
}

std::filesystem::path moduleDir() {
#if defined(_WIN32)
    // The address of a function in this file names the module it was linked
    // into, as dladdr does elsewhere.
    HMODULE module = nullptr;
    if (!::GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCWSTR>(&moduleDir), &module)) {
        return {};
    }
    wchar_t name[MAX_PATH];
    const DWORD length = ::GetModuleFileNameW(module, name, MAX_PATH);
    return length > 0 && length < MAX_PATH ? std::filesystem::path(name).parent_path() : std::filesystem::path{};
#else
    // The address of a function in this file names the image it was linked
    // into: core is a static library, so that is whichever executable or
    // shared library holds this copy of it.
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&moduleDir), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    std::error_code failed;
    auto canonical = std::filesystem::weakly_canonical(info.dli_fname, failed);
    return failed ? std::filesystem::path{} : canonical.parent_path();
#endif
}

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr ? std::string(value) : std::string();
}

void matchLayerToBacking(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return;
    }
    // Through the Objective-C runtime, so this file stays C++.
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    const double scale =
        reinterpret_cast<double (*)(void*, SEL)>(objc_msgSend)(nsWindow, sel_registerName("backingScaleFactor"));
    void* view = send(nsWindow, "contentView");
    void* layer = view != nullptr ? send(view, "layer") : nullptr;
    if (layer != nullptr) {
        reinterpret_cast<void (*)(void*, SEL, double)>(objc_msgSend)(layer, sel_registerName("setContentsScale:"),
                                                                     scale);
    }
#else
    (void)nsWindow;
#endif
}

double extendedRangeHeadroom(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return 1.0;
    }
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    void* screen = send(nsWindow, "screen");
    if (screen == nullptr) {
        return 1.0;
    }
    const double headroom = reinterpret_cast<double (*)(void*, SEL)>(objc_msgSend)(
        screen, sel_registerName("maximumExtendedDynamicRangeColorComponentValue"));
    return headroom > 1.0 ? headroom : 1.0;
#else
    (void)nsWindow;
    return 1.0;
#endif
}

bool enableExtendedRange(void* nsWindow) {
#if defined(__APPLE__)
    if (nsWindow == nullptr) {
        return false;
    }
    const auto send = [](void* receiver, const char* selector) {
        return reinterpret_cast<void* (*)(void*, SEL)>(objc_msgSend)(receiver, sel_registerName(selector));
    };
    void* view = send(nsWindow, "contentView");
    void* layer = view != nullptr ? send(view, "layer") : nullptr;
    if (layer == nullptr) {
        return false;
    }
    // CAMetalLayer: wantsExtendedDynamicRangeContent, and a colour space
    // whose 1.0 is the reference white with the range beyond it kept. The
    // colour space's name is the constant's own string.
    reinterpret_cast<void (*)(void*, SEL, BOOL)>(objc_msgSend)(
        layer, sel_registerName("setWantsExtendedDynamicRangeContent:"), YES);
    CFStringRef name = CFStringCreateWithCString(nullptr, "kCGColorSpaceExtendedLinearDisplayP3",
                                                 kCFStringEncodingUTF8);
    CGColorSpaceRef space = CGColorSpaceCreateWithName(name);
    CFRelease(name);
    if (space == nullptr) {
        return false;
    }
    reinterpret_cast<void (*)(void*, SEL, CGColorSpaceRef)>(objc_msgSend)(layer, sel_registerName("setColorspace:"),
                                                                          space);
    CGColorSpaceRelease(space);
    return true;
#else
    (void)nsWindow;
    return false;
#endif
}

void* newTrackedMetalBuffer(void* mtlDevice, uint64_t bytes) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr || bytes == 0) {
        return nullptr;
    }
    // MTLResourceStorageModePrivate (2 << 4) | MTLResourceHazardTrackingModeTracked (2 << 8).
    const unsigned long options = (2UL << 4) | (2UL << 8);
    return reinterpret_cast<void* (*)(void*, SEL, unsigned long, unsigned long)>(objc_msgSend)(
        mtlDevice, sel_registerName("newBufferWithLength:options:"), static_cast<unsigned long>(bytes), options);
#else
    (void)mtlDevice;
    (void)bytes;
    return nullptr;
#endif
}

uint64_t availablePhysicalMemory() {
#if defined(__APPLE__)
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const mach_port_t host = mach_host_self();
    const kern_return_t got =
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
    mach_port_deallocate(mach_task_self(), host);
    if (got != KERN_SUCCESS) {
        return 0;
    }
    const uint64_t page = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
    // What the system hands back on demand counts as available, as
    // memory_pressure counts it: the file cache too (`external_page_count`,
    // file-backed pages, active or not), and the speculative read-ahead.
    // Free, inactive and purgeable alone read 1.7 GB on a machine
    // memory_pressure called 79% free after a large process had left its
    // files cached, and a cloud that fitted was refused. The two counts
    // overlap (a cached file page may be inactive), so the larger is taken.
    const uint64_t reclaimable = uint64_t{stats.free_count} + stats.inactive_count + stats.purgeable_count +
                                 stats.speculative_count;
    const uint64_t cached = uint64_t{stats.free_count} + stats.external_page_count + stats.purgeable_count +
                            stats.speculative_count;
    return std::max(reclaimable, cached) * page;
#elif defined(__linux__)
    FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) {
        return 0;
    }
    char line[256];
    uint64_t kib = 0;
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        unsigned long long value = 0;
        if (std::sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
            kib = value;
            break;
        }
    }
    std::fclose(file);
    return kib * 1024;
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return ::GlobalMemoryStatusEx(&status) ? uint64_t{status.ullAvailPhys} : 0;
#else
    return 0;
#endif
}

uint64_t metalRecommendedWorkingSet(void* mtlDevice) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr) {
        return 0;
    }
    return reinterpret_cast<uint64_t (*)(void*, SEL)>(objc_msgSend)(mtlDevice,
                                                                    sel_registerName("recommendedMaxWorkingSetSize"));
#else
    (void)mtlDevice;
    return 0;
#endif
}

uint64_t metalAllocatedSize(void* mtlDevice) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr) {
        return 0;
    }
    return static_cast<uint64_t>(reinterpret_cast<unsigned long (*)(void*, SEL)>(objc_msgSend)(
        mtlDevice, sel_registerName("currentAllocatedSize")));
#else
    (void)mtlDevice;
    return 0;
#endif
}

uint64_t pageSize() {
#if defined(_WIN32)
    SYSTEM_INFO info{};
    ::GetSystemInfo(&info);
    return info.dwPageSize;
#else
    return static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
#endif
}

void* mapPages(uint64_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const uint64_t page = pageSize();
    const uint64_t whole = (bytes + page - 1) / page * page;
#if defined(_WIN32)
    return ::VirtualAlloc(nullptr, static_cast<size_t>(whole), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    void* pages = ::mmap(nullptr, static_cast<size_t>(whole), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    return pages == MAP_FAILED ? nullptr : pages;
#endif
}

void unmapPages(void* pages, uint64_t bytes) {
    if (pages == nullptr || bytes == 0) {
        return;
    }
#if defined(_WIN32)
    ::VirtualFree(pages, 0, MEM_RELEASE);
#else
    const uint64_t page = pageSize();
    ::munmap(pages, static_cast<size_t>((bytes + page - 1) / page * page));
#endif
}

void* newMetalBufferOverPages(void* mtlDevice, const void* pages, uint64_t bytes) {
#if defined(__APPLE__)
    if (mtlDevice == nullptr || pages == nullptr || bytes == 0) {
        return nullptr;
    }
    const uint64_t page = pageSize();
    if (reinterpret_cast<uintptr_t>(pages) % page != 0 || bytes % page != 0) {
        return nullptr;
    }
    // A discrete GPU would read these pages over the bus: not what borrowing
    // promises, so it is refused and the effect keeps a copy instead.
    const BOOL unified = reinterpret_cast<BOOL (*)(void*, SEL)>(objc_msgSend)(mtlDevice, sel_registerName("hasUnifiedMemory"));
    if (!unified) {
        return nullptr;
    }
    // MTLResourceStorageModeShared (0 << 4), no deallocator: the pages stay
    // the caller's.
    const unsigned long options = 0UL;
    return reinterpret_cast<void* (*)(void*, SEL, void*, unsigned long, unsigned long, void*)>(objc_msgSend)(
        mtlDevice, sel_registerName("newBufferWithBytesNoCopy:length:options:deallocator:"),
        const_cast<void*>(pages), static_cast<unsigned long>(bytes), options, nullptr);
#else
    (void)mtlDevice;
    (void)pages;
    (void)bytes;
    return nullptr;
#endif
}

void releaseMetalBuffer(void* mtlBuffer) {
#if defined(__APPLE__)
    if (mtlBuffer != nullptr) {
        reinterpret_cast<void (*)(void*, SEL)>(objc_msgSend)(mtlBuffer, sel_registerName("release"));
    }
#else
    (void)mtlBuffer;
#endif
}

std::filesystem::path partialPathFor(const std::filesystem::path& path) {
    // `.name.partial-<pid>.ext`: hidden where a listing hides dot files, and
    // still ending in what the writer chooses its format by.
#if defined(_WIN32)
    const int pid = ::_getpid();
#else
    const int pid = static_cast<int>(::getpid());
#endif
    const std::string name =
        "." + path.stem().string() + ".partial-" + std::to_string(pid) + path.extension().string();
    return path.parent_path() / name;
}

Result<void> replaceFile(const std::filesystem::path& from, const std::filesystem::path& to) {
#if defined(_WIN32)
    // rename refuses an existing target here; MoveFileEx replaces it.
    if (!::MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        return Error::make(ErrorCode::IoFailure, "cannot move '{}' to '{}': error {}", from.string(), to.string(),
                           static_cast<unsigned long>(::GetLastError()));
    }
    return ok();
#else
    if (::rename(from.c_str(), to.c_str()) != 0) {
        return Error::make(ErrorCode::IoFailure, "cannot move '{}' to '{}': {}", from.string(), to.string(),
                           std::strerror(errno));
    }
    return ok();
#endif
}

void removeFile(const std::filesystem::path& path) noexcept {
#if defined(_WIN32)
    ::DeleteFileW(path.c_str());
#else
    ::unlink(path.c_str());
#endif
}

Result<void> writeAtomically(const std::filesystem::path& path,
                             const std::function<Result<void>(const std::filesystem::path& partial)>& write) {
    const std::filesystem::path partial = partialPathFor(path);
    removeFile(partial);   // a partial file a killed run of this same pid left
    Result<void> wrote = write(partial);
    if (!wrote) {
        removeFile(partial);
        return wrote;
    }
    if (Result<void> moved = replaceFile(partial, path); !moved) {
        removeFile(partial);
        return moved;
    }
    return ok();
}

}   // namespace athenea::platform
