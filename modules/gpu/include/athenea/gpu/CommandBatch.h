// Copyright (c) 2026 jesus luque.
#pragma once

#include <source_location>
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
    /// until the device has run it.
    [[nodiscard]] Result<void> submit(bool wait = false,
                                      std::source_location where = std::source_location::current());
    /// THE NEXT SUBMIT WAITS ON THE DEVICE for `fence` to reach `value`
    /// before it runs anything (rhi::SubmitDesc::waitFences): what another
    /// process wrote is read only after it says so, and the host never
    /// waits. Kept until a submit that runs commands. The device's caller.
    void waitBefore(rhi::IFence* fence, uint64_t value);
    /// The waits the next submit will make.
    [[nodiscard]] size_t pendingWaits() const noexcept { return waits_.size(); }

    /// The same, and `fence` is set to `value` once the device has run it:
    /// how a caller learns that later without waiting (gpu::AsyncReadback).
    [[nodiscard]] Result<void> submitSignalling(rhi::IFence* fence, uint64_t value, bool wait = false,
                                                std::source_location where = std::source_location::current());

private:
    Device&                           device_;
    rhi::ComPtr<rhi::ICommandEncoder> encoder_;
    bool                              dirty_ = false;
    std::vector<rhi::ComPtr<rhi::IFence>> waits_;
    std::vector<uint64_t>             waitValues_;
    [[nodiscard]] SlangResult submitNow(rhi::IFence* signal, uint64_t signalValue);
    friend class ComputeKernel;
};

}   // namespace athenea::gpu
