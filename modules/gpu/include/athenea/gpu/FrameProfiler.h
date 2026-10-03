// Copyright (c) 2026 jesus luque.
//
// GPU TIME BY PHASE, AND THE NAMES A CAPTURE TOOL SHOWS.
//
// A frame is cut into phases (`gpu::Phase`, a scope over a command batch).
// Every phase opens a debug group in its batch, always: Xcode's Metal
// debugger shows it, and where slang-rhi's backend marks nothing (CUDA) it
// costs nothing. With profiling on, every phase boundary is also a point in
// GPU time, and the time between two points is the innermost phase's:
//
//   - where the device writes timestamps (CUDA, Vulkan, D3D12), a timestamp
//     is written into the batch at each boundary, and read back once the
//     device has passed it (`IQueryPool::getResultState`, never waiting);
//   - on Metal, where slang-rhi's `writeTimestamp` writes nothing (Apple's
//     GPUs sample counters only at an encoder's edges), each boundary ends
//     the batch's command buffer, and a phase's time is its command buffers'
//     GPUStartTime..GPUEndTime -- measured from the end of the one before,
//     since a buffer may start before its predecessor finishes and wait on
//     the queue's fence.
//
// Either way a phase counts only what its own batch records: work another
// batch submits in the middle runs outside it. Nothing here waits: a frame's
// times are taken by `latest` once the device has finished it, which for a
// caller that waits at the frame's end is the same frame.
#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {

class CommandBatch;
class Device;

struct PhaseTime {
    std::string name;
    double      ms = 0.0;
};

struct FrameTimes {
    uint64_t               tag = 0;
    /// Summed by name, in the order each name first appeared in the frame.
    std::vector<PhaseTime> phases;
    /// From the frame's first boundary to its last, on the device.
    double                 spanMs = 0.0;
    /// "timestamps" or "command buffers": how they were measured.
    const char*            source = "";
};

class FrameProfiler {
public:
    explicit FrameProfiler(Device& device);
    FrameProfiler(const FrameProfiler&) = delete;
    FrameProfiler& operator=(const FrameProfiler&) = delete;
    ~FrameProfiler();

    /// Off by default: a phase is then a debug group and nothing more.
    void setEnabled(bool on) noexcept { enabled_ = on; }
    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    /// Opens a frame tagged `tag` (the caller's frame number); its first
    /// boundary goes into `batch`. A frame already open is closed first.
    [[nodiscard]] Result<void> beginFrame(CommandBatch& batch, uint64_t tag);
    /// Closes the open frame; its last boundary goes into `batch`, which the
    /// caller submits (on Metal it is submitted here).
    [[nodiscard]] Result<void> endFrame(CommandBatch& batch);
    [[nodiscard]] bool inFrame() const noexcept { return open_.has_value(); }

    /// A boundary: the phase `name` starts (push) or the innermost ends (pop).
    [[nodiscard]] Result<void> push(CommandBatch& batch, const char* name);
    [[nodiscard]] Result<void> pop(CommandBatch& batch);

    /// The newest frame the device has finished, once; empty when none has
    /// since the last call. Never waits.
    [[nodiscard]] std::optional<FrameTimes> latest();

private:
    struct Segment {
        int      owner = -1;          ///< index into Frame::names; -1 is outside every phase
        uint64_t commandBuffer = 0;   ///< Metal: the MTLCommandBuffer, retained
    };
    struct Frame {
        uint64_t                 tag = 0;
        std::vector<std::string> names;
        std::vector<int>         stack;      ///< open phases, innermost last
        std::vector<Segment>     segments;   ///< Metal: one a command buffer; else one between timestamps
        uint32_t                 pool = 0;   ///< the query pool its timestamps went into
        uint32_t                 first = 0;  ///< its first timestamp in that pool
        uint32_t                 stamps = 0;
    };

    [[nodiscard]] Result<void> boundary(CommandBatch& batch);
    [[nodiscard]] int top() const noexcept;
    [[nodiscard]] std::optional<FrameTimes> resolve(Frame& frame, bool& pending);
    void release(Frame& frame);

    Device*                                 device_ = nullptr;
    bool                                    enabled_ = false;
    bool                                    timestamps_ = false;
    std::optional<Frame>                    open_;
    std::deque<Frame>                       finished_;
    std::vector<rhi::ComPtr<rhi::IQueryPool>> pools_;
    uint32_t                                nextPool_ = 0;
    static constexpr uint32_t               kStamps = 256;
};

/// A phase of a frame over `batch`: a debug group always, and a span of GPU
/// time when the device's profiler is on (Device::profiler).
class Phase {
public:
    Phase(CommandBatch& batch, const char* name);
    Phase(const Phase&) = delete;
    Phase& operator=(const Phase&) = delete;
    ~Phase();

private:
    CommandBatch& batch_;
    bool          timed_ = false;
};

}   // namespace athenea::gpu
