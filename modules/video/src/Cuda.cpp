// Copyright (c) 2026 jesus luque.
//
// Cuda.h: the CUDA driver opened once a process, for NVENC and NVDEC.
#include "Cuda.h"

#include <memory>
#include <mutex>

namespace athenea::video::cuda {
namespace {

#if defined(_WIN32)
constexpr const char* kCudaLibrary = "nvcuda.dll";
#else
constexpr const char* kCudaLibrary = "libcuda.so.1";
#endif

}   // namespace

Result<const Driver*> driver(const char* user) {
    static std::mutex              guard;
    static std::unique_ptr<Driver> loaded;
    static std::string             failed;
    static bool                    tried = false;
    const std::lock_guard<std::mutex> held(guard);
    if (loaded != nullptr) return static_cast<const Driver*>(loaded.get());
    if (tried) return Error::make(ErrorCode::Unsupported, "video: {} for {}", failed, user);
    tried = true;
    auto d = std::make_unique<Driver>();
    auto library = platform::SharedLibrary::open(kCudaLibrary);
    if (!library) {
        failed = "no NVIDIA driver (" + library.error().message() + ")";
        return Error::make(ErrorCode::Unsupported, "video: {} for {}", failed, user);
    }
    d->library = std::move(*library);
    const platform::SharedLibrary& c = d->library;
    if (!bind(c, d->cuInit, "cuInit") || !bind(c, d->cuDeviceGet, "cuDeviceGet") ||
        !bind(c, d->cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") ||
        !bind(c, d->cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2") ||
        !bind(c, d->cuCtxPushCurrent, "cuCtxPushCurrent_v2") || !bind(c, d->cuCtxPopCurrent, "cuCtxPopCurrent_v2") ||
        !bind(c, d->cuMemAllocPitch, "cuMemAllocPitch_v2") || !bind(c, d->cuMemFree, "cuMemFree_v2") ||
        !bind(c, d->cuMemcpy2DAsync, "cuMemcpy2DAsync_v2") || !bind(c, d->cuMemsetD8Async, "cuMemsetD8Async") ||
        !bind(c, d->cuMemsetD16Async, "cuMemsetD16Async") ||
        !bind(c, d->cuStreamSynchronize, "cuStreamSynchronize") || !bind(c, d->cuGetErrorName, "cuGetErrorName")) {
        failed = "a CUDA driver without an entry point the codec needs";
        return Error::make(ErrorCode::Unsupported, "video: {} for {}", failed, user);
    }
    if (d->cuInit(0) != CUDA_SUCCESS) {
        failed = "cuInit failed (no CUDA device)";
        return Error::make(ErrorCode::Unsupported, "video: {} for {}", failed, user);
    }
    loaded = std::move(d);
    return static_cast<const Driver*>(loaded.get());
}

Result<void> check(const Driver& d, CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return ok();
    const char* name = nullptr;
    d.cuGetErrorName(r, &name);
    constexpr int kOutOfMemory = 2;   // CUDA_ERROR_OUT_OF_MEMORY
    return Error::make(static_cast<int>(r) == kOutOfMemory ? ErrorCode::OutOfMemory : ErrorCode::DeviceFailure,
                       "video: CUDA failed at {} ({})", what, name != nullptr ? name : "?");
}

}   // namespace athenea::video::cuda
