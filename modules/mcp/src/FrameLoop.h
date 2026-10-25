// Copyright (c) 2026 jesus luque.
//
// THE SESSION'S FRAME THREAD: the one thread that calls the device. Every
// tool that touches the stage or the device is run here (`run`), between
// frames; the server's thread only waits for its answer. While a
// subscription ticks, the thread also draws a frame at every frame of the
// clock -- SMPTE ST 2059-1's alignment, sched::FrameClock where the build
// has genlock, the same arithmetic on the system clock otherwise -- and
// waits for nothing but the clock (and the tools queued meanwhile).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

#include <nlohmann/json.hpp>

#include "athenea/core/Result.h"

namespace athenea::mcp {

/// Frames a second as a fraction (30000/1001).
struct FrameRate {
    int64_t numerator = 30;
    int64_t denominator = 1;
};

class FrameLoop {
public:
    /// What a tick draws: the frame's index on the clock and the TAI
    /// nanoseconds it begins at. FRAME THREAD.
    using Tick = std::function<void(int64_t index, int64_t startTaiNs)>;

    /// Any thread: starts the frame thread.
    FrameLoop();
    /// Any thread but the frame thread: stops ticking, runs what is queued, joins.
    ~FrameLoop();

    /// ANY THREAD: `task` run on the frame thread, its answer handed back.
    /// From the frame thread itself it runs at once.
    nlohmann::json run(const std::function<nlohmann::json()>& task);
    /// FRAME THREAD: tick at `rate` from the next frame; an empty tick stops.
    [[nodiscard]] Result<void> setTicker(Tick tick, FrameRate rate);
    /// Whether the caller is the frame thread. Any thread.
    [[nodiscard]] bool current() const noexcept { return std::this_thread::get_id() == thread_.get_id(); }
    /// TAI minus UTC, ns, as the clock applies it. Any thread.
    [[nodiscard]] int64_t taiMinusUtcNs() const noexcept;
    /// The clock's own name: "sched::FrameClock" or "system clock". Any thread.
    [[nodiscard]] const char* clockName() const noexcept;

private:
    void loop();
    struct Clock;
    std::mutex                               mutex_;
    std::condition_variable                  wake_;
    std::deque<std::function<void()>>        tasks_;
    bool                                     stop_ = false;
    Tick                                     tick_;
    std::unique_ptr<Clock>                   clock_;
    std::thread                              thread_;
};

}   // namespace athenea::mcp
