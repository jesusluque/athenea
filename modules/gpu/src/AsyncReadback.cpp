// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/AsyncReadback.h"

#include <algorithm>
#include <cstring>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"

namespace athenea::gpu {

Result<AsyncReadback> AsyncReadback::create(Device& device, uint64_t bytes, uint32_t depth, std::string label) {
    if (bytes == 0 || depth == 0) {
        return Error(ErrorCode::InvalidArgument, "an asynchronous readback needs bytes and a slot");
    }
    AsyncReadback readback;
    rhi::FenceDesc fence;
    fence.initialValue = 0;
    if (SLANG_FAILED(device.rhi()->createFence(fence, readback.fence_.writeRef()))) {
        return Error(ErrorCode::DeviceFailure, "cannot make a fence for an asynchronous readback");
    }
    for (uint32_t k = 0; k < depth; ++k) {
        BufferDesc desc;
        desc.bytes = bytes;
        desc.memory = Memory::Readback;
        desc.label = label + "." + std::to_string(k);
        auto made = Buffer::create(device, desc);
        if (!made) return std::move(made).error();
        readback.slots_.push_back({std::move(*made), 0, 0});
    }
    readback.device_ = &device;
    readback.bytes_ = bytes;
    return readback;
}

Result<void> AsyncReadback::submit(CommandBatch& batch, const Buffer& source, uint64_t tag, bool wait) {
    if (device_ == nullptr) {
        return Error(ErrorCode::InvalidArgument, "an asynchronous readback that was never made");
    }
    if (source.bytes() < bytes_) {
        return Error(ErrorCode::InvalidArgument, "a readback larger than what it copies");
    }
    Slot& slot = slots_[next_];
    next_ = (next_ + 1) % static_cast<uint32_t>(slots_.size());
    batch.encoder()->copyBuffer(slot.buffer.rhi(), 0, source.rhi(), 0, bytes_);
    batch.markDirty();
    slot.sequence = ++sequence_;
    slot.tag = tag;
    return batch.submitSignalling(fence_.get(), slot.sequence, wait);
}

std::optional<uint64_t> AsyncReadback::latest(std::span<uint8_t> into) {
    if (device_ == nullptr) {
        return std::nullopt;
    }
    uint64_t done = 0;
    if (SLANG_FAILED(fence_->getCurrentValue(&done))) {
        return std::nullopt;
    }
    // The newest slot whose last copy has finished. A slot written again
    // since carries the newer sequence, so it is not taken until that one
    // has finished too: what is read is never being written.
    const Slot* best = nullptr;
    for (const Slot& slot : slots_) {
        if (slot.sequence != 0 && slot.sequence <= done && (best == nullptr || slot.sequence > best->sequence)) {
            best = &slot;
        }
    }
    if (best == nullptr) {
        return std::nullopt;
    }
    void* mapped = nullptr;
    if (SLANG_FAILED(device_->rhi()->mapBuffer(best->buffer.rhi(), rhi::CpuAccessMode::Read, &mapped)) ||
        mapped == nullptr) {
        return std::nullopt;
    }
    std::memcpy(into.data(), mapped, static_cast<size_t>(std::min<uint64_t>(into.size(), bytes_)));
    (void)device_->rhi()->unmapBuffer(best->buffer.rhi());
    return best->tag;
}

}   // namespace athenea::gpu
