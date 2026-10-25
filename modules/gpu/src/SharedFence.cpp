// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/SharedFence.h"

#include <utility>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::gpu {
namespace {

void* eventOf(rhi::IFence* fence) {
    rhi::NativeHandle handle;
    if (fence == nullptr || SLANG_FAILED(fence->getNativeHandle(&handle)) ||
        handle.type != rhi::NativeHandleType::MTLSharedEvent) {
        return nullptr;
    }
    return reinterpret_cast<void*>(handle.value);
}

Result<rhi::ComPtr<rhi::IFence>> makeFence(Device& device, uint64_t initial, bool shared) {
    rhi::FenceDesc desc;
    desc.initialValue = initial;
    // Metal's every fence is an MTLSharedEvent already; slang-rhi refuses
    // nothing for isShared there, and on Vulkan it is what makes the
    // semaphore exportable.
    desc.isShared = shared;
    rhi::ComPtr<rhi::IFence> fence;
    if (SLANG_FAILED(device.rhi()->createFence(desc, fence.writeRef()))) {
        return Error(ErrorCode::DeviceFailure, "the device made no fence");
    }
    return fence;
}

}   // namespace

SharedFence::SharedFence(SharedFence&& other) noexcept
    : device_(std::exchange(other.device_, nullptr)),
      fence_(std::move(other.fence_)),
      imported_(std::exchange(other.imported_, nullptr)),
      local_(std::exchange(other.local_, 0)) {}

SharedFence& SharedFence::operator=(SharedFence&& other) noexcept {
    if (this != &other) {
        platform::releaseMetalObject(imported_);
        device_ = std::exchange(other.device_, nullptr);
        fence_ = std::move(other.fence_);
        imported_ = std::exchange(other.imported_, nullptr);
        local_ = std::exchange(other.local_, 0);
    }
    return *this;
}

SharedFence::~SharedFence() {
    platform::releaseMetalObject(imported_);
}

Result<SharedFence> SharedFence::create(Device& device, uint64_t initial) {
    if (device.backend() == Backend::CUDA) {
        return Error(ErrorCode::Unsupported, "a CUDA fence is a host value slang-rhi sets at submit: nothing "
                                             "another process can wait on");
    }
    SharedFence out;
    out.device_ = &device;
    auto fence = makeFence(device, initial, true);
    if (!fence) return std::move(fence).error();
    out.fence_ = std::move(*fence);
    return out;
}

Result<SharedFence> SharedFence::import(Device& device, platform::OsHandle handle) {
    if (device.backend() != Backend::Metal) {
        return Error::make(ErrorCode::Unsupported, "importing another process's fence on {}: not done (Metal only)",
                           toString(device.backend()));
    }
    auto event = platform::importMetalSharedEvent(reinterpret_cast<void*>(device.native().device.value), handle);
    if (!event) return std::move(event).error();
    SharedFence out;
    out.device_ = &device;
    out.imported_ = *event;
    auto local = makeFence(device, 0, false);
    if (!local) return std::move(local).error();
    out.fence_ = std::move(*local);
    return out;
}

Result<platform::OsHandle> SharedFence::exportHandle() const {
    if (imported_ != nullptr) {
        return platform::exportMetalSharedEvent(imported_);
    }
    if (fence_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "an empty fence");
    }
    if (void* event = eventOf(fence_.get()); event != nullptr) {
        return platform::exportMetalSharedEvent(event);
    }
    rhi::NativeHandle shared;
    if (SLANG_FAILED(fence_->getSharedHandle(&shared)) || !shared) {
        return Error(ErrorCode::Unsupported, "this backend exports no fence");
    }
    if (shared.type == rhi::NativeHandleType::FileDescriptor) {
        // The fence keeps its own descriptor; the caller gets one to give away.
        return platform::duplicateOsHandle({platform::OsHandle::Kind::Fd, shared.value});
    }
    return Error(ErrorCode::Unsupported, "a fence handle that is no descriptor (Windows: not done)");
}

uint64_t SharedFence::value() const {
    if (imported_ != nullptr) {
        return platform::metalSharedEventValue(imported_);
    }
    uint64_t current = 0;
    if (fence_ != nullptr) {
        (void)fence_->getCurrentValue(&current);
    }
    return current;
}

void SharedFence::setValue(uint64_t value) {
    if (imported_ != nullptr) {
        platform::setMetalSharedEventValue(imported_, value);
    } else if (fence_ != nullptr) {
        (void)fence_->setCurrentValue(value);
    }
}

Result<void> SharedFence::signalAfter(CommandBatch& batch, uint64_t value) {
    if (fence_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "an empty fence");
    }
    if (imported_ == nullptr) {
        return batch.submitSignalling(fence_.get(), value);
    }
    ATHENEA_TRY(batch.submitSignalling(fence_.get(), ++local_));
    rhi::NativeHandle queue;
    (void)device_->queue()->getNativeHandle(&queue);
    return platform::metalChainEvents(reinterpret_cast<void*>(queue.value), eventOf(fence_.get()), local_, imported_,
                                      value);
}

Result<void> SharedFence::waitBefore(CommandBatch& batch, uint64_t value) {
    if (fence_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "an empty fence");
    }
    if (imported_ == nullptr) {
        batch.waitBefore(fence_.get(), value);
        return ok();
    }
    rhi::NativeHandle queue;
    (void)device_->queue()->getNativeHandle(&queue);
    ATHENEA_TRY(platform::metalChainEvents(reinterpret_cast<void*>(queue.value), imported_, value,
                                           eventOf(fence_.get()), ++local_));
    batch.waitBefore(fence_.get(), local_);
    return ok();
}

}   // namespace athenea::gpu
