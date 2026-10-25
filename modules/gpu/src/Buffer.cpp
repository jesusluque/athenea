// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/Buffer.h"

#include <cstring>
#include <format>

#include "athenea/core/Platform.h"
#include "athenea/gpu/Device.h"

namespace athenea::gpu {
namespace {

rhi::BufferDesc toRhi(const BufferDesc& desc) {
    rhi::BufferDesc out;
    out.size = desc.bytes;
    out.elementSize = desc.elementBytes;
    out.label = desc.label.empty() ? nullptr : desc.label.c_str();
    switch (desc.memory) {
    case Memory::Device:
        out.memoryType = rhi::MemoryType::DeviceLocal;
        out.usage = rhi::BufferUsage::ShaderResource | rhi::BufferUsage::UnorderedAccess |
                    rhi::BufferUsage::CopySource | rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::UnorderedAccess;
        break;
    case Memory::Upload:
        out.memoryType = rhi::MemoryType::Upload;
        out.usage = rhi::BufferUsage::CopySource | rhi::BufferUsage::ShaderResource;
        out.defaultState = rhi::ResourceState::General;
        break;
    case Memory::Readback:
        out.memoryType = rhi::MemoryType::ReadBack;
        out.usage = rhi::BufferUsage::CopyDestination;
        out.defaultState = rhi::ResourceState::CopyDestination;
        break;
    }
    out.usage = out.usage | desc.extraUsage;
    if (desc.shared) {
        out.usage = out.usage | rhi::BufferUsage::Shared;
    }
    return out;
}

}   // namespace

Result<Buffer> Buffer::create(Device& device, const BufferDesc& desc, const void* initial,
                              std::source_location where) {
    if (desc.bytes == 0) {
        return Error::make(ErrorCode::InvalidArgument, "buffer '{}' of zero bytes", desc.label);
    }
    ATHENEA_TRY(device.admit(desc.bytes, std::format("buffer '{}'", desc.label)));
    Buffer buffer;
    const rhi::BufferDesc rhiDesc = toRhi(desc);
    // DATA FOR A DEVICE BUFFER GOES ON THE QUEUE where the backend's own
    // copy would wait for it: slang-rhi's CUDA device copies with a
    // synchronous cuMemcpy, which waits for everything the queue's stream
    // holds -- a whole frame's work, each time a frame made a buffer of a few
    // bytes. Staged and copied in order instead, as `write` does, it waits
    // for nothing, and whatever uses the buffer is submitted after it. Metal's
    // memory is the host's and its copy waits for nothing either. What a
    // load uploads -- a mesh's pools, a cloud, tens of megabytes -- keeps the
    // direct copy rather than a staging copy as large.
    constexpr uint64_t kQueuedBytes = uint64_t{16} << 20;
    const bool queued = initial != nullptr && device.backend() != Backend::Metal && desc.memory == Memory::Device &&
                        desc.bytes <= kQueuedBytes;
    if (initial != nullptr && !queued && device.backend() != Backend::Metal) {
        device.noteSync("buffer data", where);
    }
    if (SLANG_FAILED(device.rhi()->createBuffer(rhiDesc, queued ? nullptr : initial, buffer.buffer_.writeRef()))) {
        return Error::make(ErrorCode::OutOfMemory, "cannot allocate buffer '{}' ({} bytes)",
                           desc.label, desc.bytes);
    }
    buffer.bytes_ = desc.bytes;
    buffer.elementBytes_ = desc.elementBytes;
    if (queued) {
        rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
        encoder->uploadBufferData(buffer.buffer_.get(), 0, desc.bytes, initial);
        device.beforeSubmit();
        ATHENEA_TRY(Device::queueResult(device.queue()->submit(encoder->finish()), "a new buffer's data"));
        return buffer;
    }
    // A BUFFER WITHOUT DATA STARTS AT ZERO, on every backend. Metal hands out
    // new memory cleared and the engine came to rely on it: counters and
    // histograms are made and then added to. CUDA's and Vulkan's allocations
    // hold whatever was there before, so a test that counted violations read
    // 138 million of them on the first run on Linux. The clear goes on the
    // queue, which everything that uses the buffer comes after.
    if (initial == nullptr && device.backend() != Backend::Metal &&
        (rhiDesc.usage & rhi::BufferUsage::CopyDestination) != rhi::BufferUsage::None) {
        rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
        encoder->clearBuffer(buffer.buffer_.get());
        device.beforeSubmit();
        ATHENEA_TRY(Device::queueResult(device.queue()->submit(encoder->finish()), "a new buffer's clear"));
    }
    return buffer;
}

Result<Buffer> Buffer::wrap(Device& device, rhi::NativeHandle handle, const BufferDesc& desc) {
    Buffer buffer;
    const rhi::BufferDesc rhiDesc = toRhi(desc);
    if (SLANG_FAILED(device.rhi()->createBufferFromNativeHandle(handle, rhiDesc,
                                                                buffer.buffer_.writeRef()))) {
        return Error::make(ErrorCode::DeviceFailure,
                           "cannot wrap native buffer for '{}' ({} bytes)", desc.label,
                           desc.bytes);
    }
    buffer.bytes_ = desc.bytes;
    buffer.elementBytes_ = desc.elementBytes;
    return buffer;
}

Result<Buffer> Buffer::overSharedPages(Device& device, void* pages, uint64_t bytes, uint32_t elementBytes,
                                       std::string label) {
    if (device.backend() != Backend::Metal) {
        return Error::make(ErrorCode::Unsupported, "buffer '{}': host pages are read in place on Metal only", label);
    }
    void* mtlDevice = reinterpret_cast<void*>(device.native().device.value);
    void* native = platform::newMetalBufferOverPages(mtlDevice, pages, bytes);
    if (native == nullptr) {
        return Error::make(ErrorCode::DeviceFailure,
                           "buffer '{}': the device took no buffer over {} bytes of shared pages (page-aligned, "
                           "whole pages, unified memory)", label, bytes);
    }
    BufferDesc desc;
    desc.bytes = bytes;
    desc.elementBytes = elementBytes;
    desc.label = std::move(label);
    auto wrapped = wrap(device, rhi::NativeHandle{rhi::NativeHandleType::MTLBuffer, reinterpret_cast<uint64_t>(native)},
                        desc);
    // slang-rhi keeps its own reference; ours goes.
    platform::releaseMetalBuffer(native);
    return wrapped;
}

Result<rhi::NativeHandle> Buffer::sharedHandle() const {
    rhi::NativeHandle handle;
    if (buffer_ == nullptr || SLANG_FAILED(buffer_->getSharedHandle(&handle)) || !handle) {
        return Error(ErrorCode::Unsupported, "this backend exports no handle for a buffer (made shared?)");
    }
    return handle;
}

rhi::NativeHandle Buffer::native() const {
    rhi::NativeHandle handle;
    if (buffer_ != nullptr) {
        (void)buffer_->getNativeHandle(&handle);
    }
    return handle;
}

Result<void> Buffer::read(Device& device, uint64_t offset, uint64_t bytes, void* into,
                          std::source_location where) const {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "read outside the buffer");
    }
    device.beforeSubmit();
    // The copy back waits for everything before it anyway (the queue's
    // fence); waiting here first is how a command buffer that failed on the
    // way -- the device out of memory -- is reported instead of read.
    device.waitIdle(where);
    ATHENEA_TRY(device.takeQueueError());
    if (const SlangResult read = device.rhi()->readBuffer(buffer_.get(), offset, bytes, into); SLANG_FAILED(read)) {
        return read == SLANG_E_OUT_OF_MEMORY ? Error(ErrorCode::OutOfMemory, "the GPU ran out of memory reading a buffer back")
                                             : Error(ErrorCode::DeviceFailure, "readBuffer failed");
    }
    return ok();
}

Result<void> Buffer::write(Device& device, uint64_t offset, uint64_t bytes, const void* from) {
    if (buffer_ == nullptr || offset + bytes > bytes_) {
        return Error(ErrorCode::InvalidArgument, "write outside the buffer");
    }
    rhi::ComPtr<rhi::ICommandEncoder> encoder = device.queue()->createCommandEncoder();
    encoder->uploadBufferData(buffer_.get(), offset, bytes, const_cast<void*>(from));
    device.beforeSubmit();
    return Device::queueResult(device.queue()->submit(encoder->finish()), "an upload");
}

}   // namespace athenea::gpu
