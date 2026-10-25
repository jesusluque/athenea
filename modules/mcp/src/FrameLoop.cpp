// Copyright (c) 2026 jesus luque.
#include "FrameLoop.h"

#include <chrono>
#include <future>

#include "athenea/core/Platform.h"

#if defined(ATHENEA_MCP_SCHED)
#include "athenea/sched/FrameClock.h"
#endif

namespace athenea::mcp {
namespace {

int64_t systemTaiNs(int64_t taiMinusUtc) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
               .count() +
           taiMinusUtc;
}

}   // namespace

/// Where frames fall: frame N of the rate begins N periods after the epoch.
struct FrameLoop::Clock {
    FrameRate rate;
#if defined(ATHENEA_MCP_SCHED)
    std::unique_ptr<sched::FrameClock> sched;
#endif
    static constexpr int64_t kTaiMinusUtcNs = int64_t{37} * 1'000'000'000;

    int64_t taiMinusUtc() const {
#if defined(ATHENEA_MCP_SCHED)
        if (sched) return sched->taiMinusUtcNs();
#endif
        return kTaiMinusUtcNs;
    }
    int64_t nowTai() const {
#if defined(ATHENEA_MCP_SCHED)
        if (sched) return sched->nowTaiNs();
#endif
        return systemTaiNs(kTaiMinusUtcNs);
    }
    int64_t frameAt(int64_t tai) const {
#if defined(ATHENEA_MCP_SCHED)
        if (sched) return sched->frameAt(tai);
#endif
        // floor(tai * num / D), D = den * 1e9, without 128 bits: tai = a D + r.
        const int64_t d = rate.denominator * 1'000'000'000;
        const int64_t a = tai / d;
        const int64_t r = tai % d;
        return a * rate.numerator + r * rate.numerator / d;
    }
    int64_t frameStart(int64_t index) const {
#if defined(ATHENEA_MCP_SCHED)
        if (sched) return sched->frameStartTaiNs(index);
#endif
        // ceil(index * D / num): index = b num + c.
        const int64_t d = rate.denominator * 1'000'000'000;
        const int64_t b = index / rate.numerator;
        const int64_t c = index % rate.numerator;
        return b * d + (c * d + rate.numerator - 1) / rate.numerator;
    }
};

FrameLoop::FrameLoop() : thread_([this] { loop(); }) {}

FrameLoop::~FrameLoop() {
    {
        std::lock_guard lock(mutex_);
        stop_ = true;
        tick_ = nullptr;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
}

nlohmann::json FrameLoop::run(const std::function<nlohmann::json()>& task) {
    if (current()) {
        return task();
    }
    std::promise<nlohmann::json> answer;
    std::future<nlohmann::json>  got = answer.get_future();
    {
        std::lock_guard lock(mutex_);
        tasks_.push_back([&] {
            try {
                answer.set_value(task());
            } catch (...) {
                answer.set_exception(std::current_exception());
            }
        });
    }
    wake_.notify_all();
    return got.get();   // rethrows what the task threw
}

Result<void> FrameLoop::setTicker(Tick tick, FrameRate rate) {
    if (!current()) {
        return Error(ErrorCode::InvalidArgument, "FrameLoop::setTicker called off the frame thread");
    }
    if (tick && (rate.numerator <= 0 || rate.denominator <= 0 || rate.numerator > 240 * rate.denominator)) {
        return Error::make(ErrorCode::InvalidArgument, "a frame rate of {}/{} (above 0, at most 240)", rate.numerator,
                           rate.denominator);
    }
    auto clock = std::make_unique<Clock>();
    clock->rate = rate;
#if defined(ATHENEA_MCP_SCHED)
    if (tick) {
        sched::ClockSettings settings;
        settings.rate = sched::Rate{rate.numerator, rate.denominator};
        auto made = sched::FrameClock::create(settings);
        if (!made) return std::move(made).error();
        clock->sched = std::move(*made);
    }
#endif
    std::lock_guard lock(mutex_);
    tick_ = std::move(tick);
    clock_ = tick_ ? std::move(clock) : nullptr;
    return ok();
}

int64_t FrameLoop::taiMinusUtcNs() const noexcept {
    return clock_ ? clock_->taiMinusUtc() : Clock::kTaiMinusUtcNs;
}

const char* FrameLoop::clockName() const noexcept {
#if defined(ATHENEA_MCP_SCHED)
    return "sched::FrameClock (free run)";
#else
    return "the system clock (ST 2059-1 alignment)";
#endif
}

void FrameLoop::loop() {
    // The last 2 ms before a frame are slept precisely, not waited on a
    // condition, whose wake may come late (a process at background priority
    // is woken tens of milliseconds late): a wake past the frame's start
    // draws that frame at once, late, rather than waiting for the next.
    constexpr int64_t kSpinNs = 2'000'000;
    int64_t target = -1;   // the frame being waited for
    for (;;) {
        std::function<void()> task;
        Tick                  tick;
        int64_t               start = 0;
        {
            std::unique_lock lock(mutex_);
            for (;;) {
                if (!tasks_.empty()) {
                    task = std::move(tasks_.front());
                    tasks_.pop_front();
                    break;
                }
                if (stop_) return;
                if (!tick_) {
                    target = -1;
                    wake_.wait(lock);
                    continue;
                }
                const int64_t now = clock_->nowTai();
                const int64_t current = clock_->frameAt(now);
                // The next frame; a frame missed by more than one is not drawn
                // late: the newest one begun is.
                if (target < 0 || target < current) {
                    target = target < 0 ? current + 1 : current;
                }
                start = clock_->frameStart(target);
                if (start - now > kSpinNs) {
                    wake_.wait_for(lock, std::chrono::nanoseconds(start - now - kSpinNs));
                    continue;   // a task may have come; the frame is still the one waited for
                }
                tick = tick_;
                break;
            }
        }
        if (task) {
            task();
            continue;
        }
        const int64_t left = start - clock_->nowTai();
        if (left > 0) platform::sleepPrecisely(std::chrono::nanoseconds(left));
        tick(target, start);
        ++target;
    }
}

}   // namespace athenea::mcp
