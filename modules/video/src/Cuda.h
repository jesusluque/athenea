// Copyright (c) 2026 jesus luque.
//
// The CUDA driver's few entry points the NVIDIA codecs use (src/Nvenc.cpp,
// src/Nvdec.cpp), opened at run time: nvcuda.dll or libcuda.so.1 through
// platform::SharedLibrary, so nothing links the driver and a build runs where
// there is none. The types are nv-codec-headers' dynlink_cuda.h (MIT). Only
// copies, fills and synchronisation: nothing here computes on a sample.
#pragma once

#include <ffnvcodec/dynlink_cuda.h>

#include "athenea/core/Platform.h"
#include "athenea/core/Result.h"

namespace athenea::video::cuda {

// The two the dynlink header has no typedef for.
typedef CUresult CUDAAPI tcuMemsetD8Async_(CUdeviceptr, unsigned char, size_t, CUstream);
typedef CUresult CUDAAPI tcuMemsetD16Async_(CUdeviceptr, unsigned short, size_t, CUstream);

/// The driver's entry points, loaded and cuInit'ed once a process.
struct Driver {
    platform::SharedLibrary     library;
    tcuInit*                    cuInit = nullptr;
    tcuDeviceGet*               cuDeviceGet = nullptr;
    tcuDevicePrimaryCtxRetain*  cuDevicePrimaryCtxRetain = nullptr;
    tcuDevicePrimaryCtxRelease* cuDevicePrimaryCtxRelease = nullptr;
    tcuCtxPushCurrent_v2*       cuCtxPushCurrent = nullptr;
    tcuCtxPopCurrent_v2*        cuCtxPopCurrent = nullptr;
    tcuMemAllocPitch_v2*        cuMemAllocPitch = nullptr;
    tcuMemFree_v2*              cuMemFree = nullptr;
    tcuMemcpy2DAsync_v2*        cuMemcpy2DAsync = nullptr;
    tcuMemsetD8Async_*          cuMemsetD8Async = nullptr;
    tcuMemsetD16Async_*         cuMemsetD16Async = nullptr;
    tcuStreamSynchronize*       cuStreamSynchronize = nullptr;
    tcuGetErrorName*            cuGetErrorName = nullptr;
};

/// The driver, or Unsupported with the reason (`user`: who asks, for the
/// message). Thread safe.
[[nodiscard]] Result<const Driver*> driver(const char* user);

/// `r` as a Result: DeviceFailure (OutOfMemory for CUDA's) naming `what`.
[[nodiscard]] Result<void> check(const Driver& d, CUresult r, const char* what);

/// `symbol` of `library` into `to`; false when it is not there.
template <typename F>
bool bind(const platform::SharedLibrary& library, F*& to, const char* symbol) {
    to = reinterpret_cast<F*>(library.symbol(symbol));
    return to != nullptr;
}

/// The context current on this thread while it lives.
class Pushed {
public:
    Pushed(const Driver& d, CUcontext context) : d_(d) { d_.cuCtxPushCurrent(context); }
    ~Pushed() {
        CUcontext popped = nullptr;
        d_.cuCtxPopCurrent(&popped);
    }
    Pushed(const Pushed&) = delete;
    Pushed& operator=(const Pushed&) = delete;

private:
    const Driver& d_;
};

}   // namespace athenea::video::cuda
