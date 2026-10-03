// Copyright (c) 2026 jesus luque.
#include "athenea/gpu_host/Views.h"

#include "gpe/adopt.h"
#include "gpe/device.h"
#include "gpe/pool.h"

namespace athenea::gpu_host {
namespace {

rhi::NativeHandleType bufferHandleType(gpu::Backend backend) {
    switch (backend) {
    case gpu::Backend::Metal: return rhi::NativeHandleType::MTLBuffer;
    case gpu::Backend::CUDA: return rhi::NativeHandleType::CUdeviceptr;
    default: return rhi::NativeHandleType::Undefined;
    }
}

}   // namespace

std::unique_ptr<gpe::Device> adoptForCompute(gpu::Device& device) {
    const gpu::NativeHandles native = device.native();
    switch (device.backend()) {
    case gpu::Backend::Metal:
        return gpe::adoptMetalDevice(reinterpret_cast<void*>(native.device.value),
                                     reinterpret_cast<void*>(native.queue.value));
    case gpu::Backend::CUDA:
        return gpe::adoptCudaContext(reinterpret_cast<void*>(native.device.value),
                                     reinterpret_cast<void*>(native.queue.value));
    default:
        return nullptr;
    }
}

Result<gpu::Buffer> renderView(gpu::Device& device, gpe::PooledDevice& compute,
                               uint64_t gpeBuffer, size_t bytes, uint32_t elementBytes,
                               std::string label) {
    const uint64_t object = compute.backendBuffer(gpeBuffer);
    if (object == 0) {
        return Error(ErrorCode::InvalidArgument, "not a live gpe buffer");
    }
    rhi::NativeHandle handle;
    handle.type = bufferHandleType(device.backend());
    handle.value = object;
    gpu::BufferDesc desc;
    desc.bytes = bytes;
    desc.elementBytes = elementBytes;
    desc.label = std::move(label);
    return gpu::Buffer::wrap(device, handle, desc);
}

Result<uint64_t> computeView(gpe::PooledDevice& compute, const gpu::Buffer& buffer) {
    const rhi::NativeHandle handle = buffer.native();
    if (handle.value == 0) {
        return Error(ErrorCode::InvalidArgument, "the buffer has no native handle");
    }
    const uint64_t adopted = compute.adopt(handle.value, buffer.bytes());
    if (adopted == 0) {
        return Error(ErrorCode::DeviceFailure, "gpe refused the buffer");
    }
    return adopted;
}

}   // namespace athenea::gpu_host
