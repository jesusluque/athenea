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
    // A command buffer that failed before this one -- the device out of
    // memory -- is reported by the first submit or wait after it, whichever
    // batch makes it: the work it fed is garbage, and the caller is told so
    // rather than shown it.
    Result<void> earlier = device_.takeQueueError();
    if (dirty_) {
        device_.beforeSubmit();
        const SlangResult submitted = device_.queue()->submit(encoder_->finish());
        encoder_ = device_.queue()->createCommandEncoder();
        dirty_ = false;
        ATHENEA_TRY(earlier);
        ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    }
    ATHENEA_TRY(earlier);
    if (wait) {
        device_.beforeSubmit();
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
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
    Result<void> earlier = device_.takeQueueError();
    const SlangResult submitted = device_.queue()->submit(desc);
    encoder_ = device_.queue()->createCommandEncoder();
    dirty_ = false;
    ATHENEA_TRY(earlier);
    ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    if (wait) {
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

}   // namespace athenea::gpu
