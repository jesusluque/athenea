// Copyright (c) 2026 jesus luque.
//
// NUMBERS FROM THE DEVICE, A FRAME OR TWO LATE, WITHOUT WAITING FOR THEM.
//
// A panel that shows what a frame counted must not make the frame wait for
// the counting: `Buffer::read` blocks until the queue drains, which on a
// frame path is a stall the panel would be the cause of. So the counts are
// copied, in the frame's own command buffer, into one of a few readback
// buffers, and the submit that carries the copy signals a fence. Later --
// the next frame, or the one after -- `latest` asks the fence how far the
// device has got and copies out the newest slot it has finished. It never
// waits; when nothing is finished yet it says so.
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"

namespace athenea::gpu {

class CommandBatch;
class Device;

class AsyncReadback {
public:
    /// `depth` slots of `bytes` each: how many copies may be in flight at once
    /// before the oldest is overwritten.
    [[nodiscard]] static Result<AsyncReadback> create(Device& device, uint64_t bytes, uint32_t depth = 3,
                                                      std::string label = "readback");

    AsyncReadback() = default;

    /// Records a copy of `source` (its first `bytes()`) into the next slot,
    /// tagged `tag`, and submits `batch` signalling this readback's fence --
    /// so the copy counts as finished once everything recorded before it has
    /// run. `wait` is passed on to the submit, for a caller that was going to
    /// wait anyway.
    [[nodiscard]] Result<void> submit(CommandBatch& batch, const Buffer& source, uint64_t tag, bool wait = false);

    /// The newest slot the device has finished: its bytes into `into` (at most
    /// `bytes()`), and its tag. Empty when no copy has finished yet. Never
    /// waits.
    [[nodiscard]] std::optional<uint64_t> latest(std::span<uint8_t> into);

    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] bool valid() const noexcept { return device_ != nullptr; }

private:
    struct Slot {
        Buffer   buffer;
        uint64_t sequence = 0;   ///< the fence value its last copy signals; 0 is never written
        uint64_t tag = 0;
    };
    Device*                 device_ = nullptr;
    rhi::ComPtr<rhi::IFence> fence_;
    std::vector<Slot>       slots_;
    uint64_t                bytes_ = 0;
    uint64_t                sequence_ = 0;
    uint32_t                next_ = 0;
};

}   // namespace athenea::gpu
