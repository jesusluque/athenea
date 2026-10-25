// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/CommandBatch.h"

#include <algorithm>

#include "athenea/gpu/Device.h"

namespace athenea::gpu {

CommandBatch::CommandBatch(Device& device)
    : device_(device), encoder_(device.queue()->createCommandEncoder()) {}

CommandBatch::~CommandBatch() {
    if (dirty_) {
        (void)submit(false);
    }
}

Result<void> CommandBatch::submit(bool wait, std::source_location where) {
    // A command buffer that failed before this one -- the device out of
    // memory -- is reported by the first submit or wait after it, whichever
    // batch makes it: the work it fed is garbage, and the caller is told so
    // rather than shown it.
    Result<void> earlier = device_.takeQueueError();
    if (dirty_) {
        device_.beforeSubmit();
        const SlangResult submitted = submitNow(nullptr, 0);
        ATHENEA_TRY(earlier);
        ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    }
    ATHENEA_TRY(earlier);
    if (wait) {
        device_.noteSync("batch", where);
        device_.beforeSubmit();
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

Result<void> CommandBatch::submitSignalling(rhi::IFence* fence, uint64_t value, bool wait,
                                                std::source_location where) {
    device_.beforeSubmit();
    // Submitted even when nothing was recorded: the fence has to reach its
    // value, and an empty command buffer is what carries it there.
    Result<void> earlier = device_.takeQueueError();
    const SlangResult submitted = submitNow(fence, value);
    ATHENEA_TRY(earlier);
    ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    if (wait) {
        device_.noteSync("batch", where);
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

void CommandBatch::waitBefore(rhi::IFence* fence, uint64_t value) {
    if (fence == nullptr) {
        return;
    }
    for (size_t k = 0; k < waits_.size(); ++k) {
        if (waits_[k].get() == fence) {
            waitValues_[k] = std::max(waitValues_[k], value);
            return;
        }
    }
    waits_.emplace_back(fence);
    waitValues_.push_back(value);
}

SlangResult CommandBatch::submitNow(rhi::IFence* signal, uint64_t signalValue) {
    rhi::ComPtr<rhi::ICommandBuffer> commands = encoder_->finish();
    rhi::ICommandBuffer* list[] = {commands.get()};
    std::vector<rhi::IFence*> waits;
    for (const auto& fence : waits_) {
        waits.push_back(fence.get());
    }
    rhi::SubmitDesc desc;
    desc.commandBuffers = list;
    desc.commandBufferCount = 1;
    desc.waitFences = waits.data();
    desc.waitFenceValues = waitValues_.data();
    desc.waitFenceCount = static_cast<uint32_t>(waits.size());
    rhi::IFence* signals[] = {signal};
    const uint64_t values[] = {signalValue};
    if (signal != nullptr) {
        desc.signalFences = signals;
        desc.signalFenceValues = values;
        desc.signalFenceCount = 1;
    }
    const SlangResult submitted = device_.queue()->submit(desc);
    encoder_ = device_.queue()->createCommandEncoder();
    dirty_ = false;
    waits_.clear();
    waitValues_.clear();
    return submitted;
}

}   // namespace athenea::gpu
