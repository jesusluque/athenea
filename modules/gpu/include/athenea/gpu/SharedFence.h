// Copyright (c) 2026 jesus luque.
//
// A FENCE TWO PROCESSES SHARE: a 64-bit value one side's device raises and
// the other's waits on, so a frame handed to another process -- or a page
// it wrote -- is ordered on the devices, never by a host waiting.
//
//   made here    slang-rhi's fence (FenceDesc::isShared): this process
//                signals or waits on it through its own submits, and
//                exports it (exportHandle) for the other side.
//   imported     another process's export, made again on this device. On
//                Metal that is an MTLSharedEvent slang-rhi did not make and
//                cannot submit with, so it is tied to a local fence by an
//                event chain (platform::metalChainEvents): signalAfter
//                signals the local fence with the batch and the chain raises
//                the imported one once the local one is reached; waitBefore
//                chains the other way and the batch waits on the local one.
//                Correct by the events alone, whatever order the command
//                buffers start in.
//
// Backends: Metal exports and imports (an event's Mach port). Vulkan exports
// (an opaque descriptor of a timeline semaphore); importing one is not done.
// CUDA: slang-rhi's fence is a value the host sets at submit, not one the
// device reaches, and exports nothing -- Unsupported, and a CUDA frame is
// handed over by polling (the frame ring's `completed` word).
#pragma once

#include <cstdint>
#include <string>

#include <slang-rhi.h>

#include "athenea/core/Platform.h"
#include "athenea/core/Result.h"

namespace athenea::gpu {

class CommandBatch;
class Device;

class SharedFence {
public:
    SharedFence() = default;
    SharedFence(SharedFence&&) noexcept;
    SharedFence& operator=(SharedFence&&) noexcept;
    SharedFence(const SharedFence&) = delete;
    SharedFence& operator=(const SharedFence&) = delete;
    ~SharedFence();

    /// A fence of this process's, exportable, at `initial`. The device's
    /// caller (the frame thread).
    [[nodiscard]] static Result<SharedFence> create(Device& device, uint64_t initial = 0);
    /// Another process's fence, from what it exported. The handle stays the
    /// caller's. The device's caller.
    [[nodiscard]] static Result<SharedFence> import(Device& device, platform::OsHandle handle);

    [[nodiscard]] bool valid() const noexcept { return fence_ != nullptr || imported_ != nullptr; }
    [[nodiscard]] bool isImported() const noexcept { return imported_ != nullptr; }

    /// What another process imports this fence by; the caller owns it. Any thread.
    [[nodiscard]] Result<platform::OsHandle> exportHandle() const;
    /// The value the fence holds now, never waiting. Any thread.
    [[nodiscard]] uint64_t value() const;
    /// Raises the fence from the host (a host's signal: the device need not
    /// run anything). Any thread.
    void setValue(uint64_t value);

    /// THE DEVICE RAISES IT: submits `batch` and the fence takes `value`
    /// once the device has run it. The frame thread.
    [[nodiscard]] Result<void> signalAfter(CommandBatch& batch, uint64_t value);
    /// THE DEVICE WAITS ON IT: the next submit of `batch` runs nothing until
    /// the fence reaches `value`. The frame thread.
    [[nodiscard]] Result<void> waitBefore(CommandBatch& batch, uint64_t value);

    /// The slang-rhi fence this one submits with (the local one, for an
    /// imported fence). The frame thread.
    [[nodiscard]] rhi::IFence* rhi() const noexcept { return fence_.get(); }

private:
    Device*                  device_ = nullptr;
    rhi::ComPtr<rhi::IFence> fence_;            ///< made here, or the local half of an import
    void*                    imported_ = nullptr;   ///< an id<MTLSharedEvent>, retained
    uint64_t                 local_ = 0;        ///< the local half's last value
};

}   // namespace athenea::gpu
