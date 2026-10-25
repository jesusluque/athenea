// Copyright (c) 2026 jesus luque.
//
// HOW LONG A BATCH TOOK ON THE DEVICE, read from the GPU's own clock where
// the backend gives one: an ATHENEA_STAGES-style timing (the batch submitted
// alone and waited for) that does not count the host's side of the submit.
// Metal: the command buffer's GPUStartTime to GPUEndTime -- slang-rhi writes
// no Metal timestamps. Elsewhere, for now, the host's time from submit to
// done, and `fromDevice` says so.
#pragma once

#include "athenea/core/Result.h"

namespace athenea::gpu {

class CommandBatch;

class GpuTimer {
public:
    GpuTimer() = default;
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    ~GpuTimer();

    /// Submits and waits for what `batch` holds, timed: whatever was recorded
    /// before it should have been submitted already, or it is counted too.
    [[nodiscard]] Result<void> run(CommandBatch& batch);
    /// The last run's milliseconds.
    [[nodiscard]] double milliseconds() const noexcept { return milliseconds_; }
    /// Whether they are the device's (else the host's submit to done).
    [[nodiscard]] bool fromDevice() const noexcept { return fromDevice_; }

private:
    void*  commandBuffer_ = nullptr;   ///< retained MTLCommandBuffer of the last run
    double milliseconds_ = 0.0;
    bool   fromDevice_ = false;
};

}   // namespace athenea::gpu
