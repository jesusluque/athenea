// Copyright (c) 2026 jesus luque.
#pragma once

#include <source_location>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {

class Device;

/// Commands recorded together and submitted once.
///
/// One per frame stage, not one per dispatch: a command buffer per dispatch is
/// the Metal cost gpe pays today and this engine does not.
class CommandBatch {
public:
    explicit CommandBatch(Device& device);
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch();

    [[nodiscard]] rhi::ICommandEncoder* encoder() noexcept { return encoder_.get(); }
    /// For callers that record straight into `encoder()` (copies, clears).
    void markDirty() noexcept { dirty_ = true; }

    /// Submits what was recorded and starts a fresh encoder. `wait` blocks
    /// until the device has run it, and is counted as a host wait at `where`
    /// (Device::hostWaits). Work submitted without waiting still runs in the
    /// order it was submitted: one queue, and every encoder waits on the
    /// queue's fence. Wait only when the host is about to read what the
    /// device wrote.
    [[nodiscard]] Result<void> submit(bool wait = false,
                                      std::source_location where = std::source_location::current());
    /// The same, and `fence` is set to `value` once the device has run it:
    /// how a caller learns that later without waiting (gpu::AsyncReadback).
    [[nodiscard]] Result<void> submitSignalling(rhi::IFence* fence, uint64_t value, bool wait = false,
                                                std::source_location where = std::source_location::current());
    /// Submits what was recorded, without waiting, and hands back the
    /// command buffer's native handle (an MTLCommandBuffer on Metal), still
    /// owned by slang-rhi: 0 when nothing was recorded. What
    /// gpu::FrameProfiler times a phase by where the device has no
    /// timestamps.
    [[nodiscard]] Result<uint64_t> submitNative();
    [[nodiscard]] bool dirty() const noexcept { return dirty_; }

    /// A named group around what is recorded next, for a capture tool:
    /// Xcode's Metal debugger shows it, and slang-rhi's CUDA backend marks
    /// nothing, so there it costs nothing. Groups still open when the batch
    /// is submitted are closed in that command buffer and opened again in
    /// the next, so each command buffer is balanced. gpu::Phase opens one.
    void pushGroup(const char* name);
    void popGroup();
    [[nodiscard]] Device& device() noexcept { return device_; }

private:
    Device&                           device_;
    rhi::ComPtr<rhi::ICommandEncoder> encoder_;
    bool                              dirty_ = false;
    std::vector<std::string>          groups_;
    /// Closes the open groups in the encoder about to finish, and opens them
    /// in the fresh one.
    void closeGroups();
    void reopenGroups();
    friend class ComputeKernel;
};

}   // namespace athenea::gpu
