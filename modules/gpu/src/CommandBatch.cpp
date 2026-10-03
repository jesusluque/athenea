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

}   // namespace athenea::gpu
