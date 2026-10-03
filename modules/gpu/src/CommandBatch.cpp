// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/CommandBatch.h"

#include "athenea/gpu/Device.h"

namespace athenea::gpu {

CommandBatch::CommandBatch(Device& device)
    : device_(device), encoder_(device.queue()->createCommandEncoder()) {}

CommandBatch::~CommandBatch() {
    if (dirty_) {
        (void)submit(false);
    }
}

Result<void> CommandBatch::submit(bool wait) {
    if (dirty_) {
        device_.beforeSubmit();
        if (SLANG_FAILED(device_.queue()->submit(encoder_->finish()))) {
            return Error(ErrorCode::DeviceFailure, "command submit failed");
        }
        encoder_ = device_.queue()->createCommandEncoder();
        dirty_ = false;
    }
    if (wait) {
        device_.beforeSubmit();
        device_.queue()->waitOnHost();
    }
    return ok();
}

Result<void> CommandBatch::submitSignalling(rhi::IFence* fence, uint64_t value, bool wait) {
    device_.beforeSubmit();
    // Submitted even when nothing was recorded: the fence has to reach its
    // value, and an empty command buffer is what carries it there.
    rhi::ComPtr<rhi::ICommandBuffer> commands = encoder_->finish();
    rhi::ICommandBuffer* list[] = {commands.get()};
    rhi::IFence* fences[] = {fence};
    const uint64_t values[] = {value};
    rhi::SubmitDesc desc;
    desc.commandBuffers = list;
    desc.commandBufferCount = 1;
    desc.signalFences = fences;
    desc.signalFenceValues = values;
    desc.signalFenceCount = 1;
    if (SLANG_FAILED(device_.queue()->submit(desc))) {
        return Error(ErrorCode::DeviceFailure, "command submit failed");
    }
    encoder_ = device_.queue()->createCommandEncoder();
    dirty_ = false;
    if (wait) {
        device_.queue()->waitOnHost();
    }
    return ok();
}

}   // namespace athenea::gpu
