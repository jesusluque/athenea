// Copyright (c) 2026 jesus luque.
//
// PROGRESS: what a long render, bake or denoise has done, how fast, and how
// long it has left. Host bookkeeping only -- counts the caller already holds
// (paths a pixel gathered, points traced, frames drawn), timed on the CPU
// when they are handed over. Nothing here waits for the GPU: a report is
// made where the work it counts has already been waited for.
//
// A `Report` is what a renderer hands a callback; a `Tracker` adds the time
// (elapsed, rate, ETA) to one task's reports; `format` is the line; and a
// `Printer` writes those lines to stderr, redrawn in place on a terminal
// (throttled to two a second) and as plain lines every ten seconds when
// stderr is a file or a pipe -- a log.
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace athenea::progress {

using Clock = std::chrono::steady_clock;

/// One task's state after a step of it.
struct Report {
    std::string task;      ///< "render", "bake", "bake extra", "denoise", "frames"
    std::string unit;      ///< what `done` and `total` count: "paths/px", "points", "frames"
    uint64_t    done = 0;  ///< counted so far
    uint64_t    total = 0; ///< what the task holds when finished; 0 when not known
    uint32_t    pass = 0;  ///< passes made so far (0: the task has none)
    uint32_t    passes = 0;///< passes it will make at most (0: not known)
    bool        finished = false;
};

/// What a renderer calls with each report, on the thread that renders.
using Callback = std::function<void(const Report&)>;

/// A task's time, as of its latest report.
struct Timing {
    double elapsed = 0.0;   ///< seconds since the task began
    double rate = 0.0;      ///< units a second (0: not yet known)
    double eta = -1.0;      ///< seconds left; negative when not known
};

/// One task's clock. It begins at `begin` or at its first report, and again
/// whenever `done` goes backwards (a viewport whose camera moved starts its
/// gathering over). The rate is taken from the first report on, so what
/// came before it -- a stage loaded, kernels compiled -- is in the elapsed
/// time but not in the rate.
class Tracker {
public:
    void   begin(Clock::time_point now);
    Timing update(const Report& report, Clock::time_point now);

private:
    bool              begun_ = false;
    bool              reported_ = false;
    Clock::time_point start_{};
    Clock::time_point first_{};
    uint64_t          firstDone_ = 0;
    uint64_t          lastDone_ = 0;
};

/// "12.3s", "4m05s", "1h02m".
[[nodiscard]] std::string formatDuration(double seconds);

/// The line for a report: `context` (say "frame 2/10") first when not empty,
/// then the task, done/total and the share of it, the pass, the elapsed time,
/// the rate and the ETA -- each only where it is known.
///   frame 2/10  render 512/2048 paths/px (25.0%)  pass 64/256  1m12s  7.11 paths/px/s  ETA 3m36s
[[nodiscard]] std::string format(const Report& report, const Timing& timing, std::string_view context = {});

/// At most one line every `interval`; a forced line always, and it restarts
/// the interval.
class Throttle {
public:
    explicit Throttle(Clock::duration interval) : interval_(interval) {}
    [[nodiscard]] bool due(Clock::time_point now, bool force = false);

private:
    Clock::duration   interval_;
    bool              any_ = false;
    Clock::time_point last_{};
};

/// How a printer writes.
enum class Mode {
    Off,        ///< nothing (--quiet, ATHENEA_PROGRESS=0)
    Terminal,   ///< one line redrawn in place ("\r"), two a second
    Lines,      ///< a plain line every ten seconds, for a log
};

/// The mode for stderr: Off when `quiet` or ATHENEA_PROGRESS is 0, Terminal
/// when stderr is a terminal, Lines otherwise.
[[nodiscard]] Mode modeForStderr(bool quiet);

/// Progress lines to a stream. Not thread safe: reports come from the one
/// thread that renders.
class Printer {
public:
    explicit Printer(Mode mode, std::FILE* stream = stderr);
    /// The intervals the throttle keeps: two a second on a terminal, one
    /// every ten seconds in a log.
    static constexpr std::chrono::milliseconds kTerminalInterval{500};
    static constexpr std::chrono::milliseconds kLinesInterval{10000};

    [[nodiscard]] Mode mode() const noexcept { return mode_; }
    /// What every line begins with until it changes ("frame 2/10").
    void setContext(std::string context) { context_ = std::move(context); }
    /// `task`'s clock starts now, rather than at its first report.
    void begin(const std::string& task);
    void begin(const std::string& task, Clock::time_point now);
    /// A report, written if the throttle allows it or it finishes its task.
    void report(const Report& report);
    void report(const Report& report, Clock::time_point now);
    /// Ends a line left open on a terminal; called before other output and
    /// at exit.
    void close();
    /// The printer as a callback, for StageRenderer::setProgress.
    [[nodiscard]] Callback callback();
    /// Lines written so far (tests).
    [[nodiscard]] uint64_t written() const noexcept { return written_; }

    ~Printer() { close(); }
    Printer(const Printer&) = delete;
    Printer& operator=(const Printer&) = delete;

private:
    Mode                           mode_;
    std::FILE*                     stream_;
    Throttle                       throttle_;
    std::map<std::string, Tracker> trackers_;
    std::string                    context_;
    size_t                         open_ = 0;   ///< characters of a line left open on a terminal
    uint64_t                       written_ = 0;
};

}   // namespace athenea::progress
