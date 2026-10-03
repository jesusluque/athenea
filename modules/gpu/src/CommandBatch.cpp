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

Result<void> CommandBatch::submit(bool wait, std::source_location where) {
    // A command buffer that failed before this one -- the device out of
    // memory -- is reported by the first submit or wait after it, whichever
    // batch makes it: the work it fed is garbage, and the caller is told so
    // rather than shown it.
    Result<void> earlier = device_.takeQueueError();
    if (dirty_) {
        device_.beforeSubmit();
        closeGroups();
        const SlangResult submitted = device_.queue()->submit(encoder_->finish());
        encoder_ = device_.queue()->createCommandEncoder();
        reopenGroups();
        dirty_ = false;
        ATHENEA_TRY(earlier);
        ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    }
    ATHENEA_TRY(earlier);
    if (wait) {
        device_.beforeSubmit();
        device_.noteHostWait(where);
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

Result<uint64_t> CommandBatch::submitNative() {
    if (!dirty_) {
        return uint64_t{0};
    }
    Result<void> earlier = device_.takeQueueError();
    device_.beforeSubmit();
    closeGroups();
    rhi::ComPtr<rhi::ICommandBuffer> commands = encoder_->finish();
    rhi::NativeHandle handle;
    const bool named = commands != nullptr && SLANG_SUCCEEDED(commands->getNativeHandle(&handle));
    const SlangResult submitted = device_.queue()->submit(commands.get());
    encoder_ = device_.queue()->createCommandEncoder();
    reopenGroups();
    dirty_ = false;
    ATHENEA_TRY(earlier);
    ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    return named ? handle.value : uint64_t{0};
}

Result<void> CommandBatch::submitSignalling(rhi::IFence* fence, uint64_t value, bool wait,
                                           std::source_location where) {
    device_.beforeSubmit();
    // Submitted even when nothing was recorded: the fence has to reach its
    // value, and an empty command buffer is what carries it there.
    closeGroups();
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
    reopenGroups();
    dirty_ = false;
    ATHENEA_TRY(earlier);
    ATHENEA_TRY(Device::queueResult(submitted, "a command batch"));
    if (wait) {
        device_.noteHostWait(where);
        ATHENEA_TRY(Device::queueResult(device_.queue()->waitOnHost(), "a command batch"));
    }
    return ok();
}

namespace {

constexpr rhi::MarkerColor kGroupColour{0.35F, 0.6F, 0.9F};

}   // namespace

void CommandBatch::pushGroup(const char* name) {
    groups_.emplace_back(name);
    encoder_->pushDebugGroup(name, kGroupColour);
}

void CommandBatch::popGroup() {
    if (groups_.empty()) {
        return;
    }
    groups_.pop_back();
    encoder_->popDebugGroup();
}

void CommandBatch::closeGroups() {
    for (size_t k = 0; k < groups_.size(); ++k) {
        encoder_->popDebugGroup();
    }
}

void CommandBatch::reopenGroups() {
    for (const std::string& name : groups_) {
        encoder_->pushDebugGroup(name.c_str(), kGroupColour);
    }
}

}   // namespace athenea::gpu
