// Copyright (c) 2026 jesus luque.
#include "athenea/core/Progress.h"

#include <algorithm>
#include <cmath>

#include "athenea/core/Platform.h"

namespace athenea::progress {

namespace {

double seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

/// A number as short as it can be read: three significant figures.
std::string number(double value) {
    char text[32];
    if (value >= 100.0) {
        std::snprintf(text, sizeof(text), "%.0f", value);
    } else if (value >= 10.0) {
        std::snprintf(text, sizeof(text), "%.1f", value);
    } else {
        std::snprintf(text, sizeof(text), "%.2f", value);
    }
    return text;
}

}   // namespace

void Tracker::begin(Clock::time_point now) {
    begun_ = true;
    reported_ = false;
    start_ = now;
    firstDone_ = 0;
    lastDone_ = 0;
}

Timing Tracker::update(const Report& report, Clock::time_point now) {
    if (!begun_ || (reported_ && report.done < lastDone_)) {
        begin(now);
    }
    Timing timing;
    timing.elapsed = std::max(seconds(now - start_), 0.0);
    if (!reported_) {
        reported_ = true;
        first_ = now;
        firstDone_ = report.done;
        // Begun before this report: what was done since is the only rate known.
        if (timing.elapsed > 0.0 && report.done > 0) {
            timing.rate = static_cast<double>(report.done) / timing.elapsed;
        }
    } else if (now > first_ && report.done > firstDone_) {
        timing.rate = static_cast<double>(report.done - firstDone_) / seconds(now - first_);
    } else if (timing.elapsed > 0.0 && report.done > 0) {
        timing.rate = static_cast<double>(report.done) / timing.elapsed;
    }
    lastDone_ = report.done;
    if (report.finished || (report.total > 0 && report.done >= report.total)) {
        timing.eta = 0.0;
    } else if (report.total > 0 && timing.rate > 0.0) {
        timing.eta = static_cast<double>(report.total - report.done) / timing.rate;
    }
    return timing;
}

std::string formatDuration(double value) {
    char text[32];
    if (!(value >= 0.0)) {
        value = 0.0;
    }
    if (value < 60.0) {
        std::snprintf(text, sizeof(text), "%.1fs", value);
    } else if (value < 3600.0) {
        const auto whole = static_cast<uint64_t>(value);
        std::snprintf(text, sizeof(text), "%llum%02llus", static_cast<unsigned long long>(whole / 60),
                      static_cast<unsigned long long>(whole % 60));
    } else {
        const auto minutes = static_cast<uint64_t>(value / 60.0);
        std::snprintf(text, sizeof(text), "%lluh%02llum", static_cast<unsigned long long>(minutes / 60),
                      static_cast<unsigned long long>(minutes % 60));
    }
    return text;
}

std::string format(const Report& report, const Timing& timing, std::string_view context) {
    std::string line;
    if (!context.empty()) {
        line.append(context);
        line.append("  ");
    }
    line.append(report.task);
    char text[96];
    if (report.total > 0) {
        const double share = 100.0 * static_cast<double>(std::min(report.done, report.total)) /
                             static_cast<double>(report.total);
        std::snprintf(text, sizeof(text), " %llu/%llu", static_cast<unsigned long long>(report.done),
                      static_cast<unsigned long long>(report.total));
        line.append(text);
        if (!report.unit.empty()) {
            line.append(" ").append(report.unit);
        }
        std::snprintf(text, sizeof(text), " (%.1f%%)", share);
        line.append(text);
    } else if (report.done > 0 || !report.unit.empty()) {
        std::snprintf(text, sizeof(text), " %llu", static_cast<unsigned long long>(report.done));
        line.append(text);
        if (!report.unit.empty()) {
            line.append(" ").append(report.unit);
        }
    }
    if (report.pass > 0) {
        if (report.passes > 0) {
            std::snprintf(text, sizeof(text), "  pass %u/%u", report.pass, report.passes);
        } else {
            std::snprintf(text, sizeof(text), "  pass %u", report.pass);
        }
        line.append(text);
    }
    line.append("  ").append(formatDuration(timing.elapsed));
    if (timing.rate > 0.0) {
        line.append("  ").append(number(timing.rate));
        if (!report.unit.empty()) {
            line.append(" ").append(report.unit);
        }
        line.append("/s");
    }
    if (report.finished) {
        line.append("  done");
    } else if (timing.eta >= 0.0) {
        line.append("  ETA ").append(formatDuration(timing.eta));
    }
    return line;
}

bool Throttle::due(Clock::time_point now, bool force) {
    if (force || !any_ || now - last_ >= interval_) {
        any_ = true;
        last_ = now;
        return true;
    }
    return false;
}

Mode modeForStderr(bool quiet) {
    if (quiet || platform::env("ATHENEA_PROGRESS") == "0") {
        return Mode::Off;
    }
    return platform::isTerminal(stderr) ? Mode::Terminal : Mode::Lines;
}

Printer::Printer(Mode mode, std::FILE* stream)
    : mode_(stream != nullptr ? mode : Mode::Off), stream_(stream),
      throttle_(mode == Mode::Terminal ? Clock::duration(kTerminalInterval) : Clock::duration(kLinesInterval)) {}

void Printer::begin(const std::string& task) { begin(task, Clock::now()); }

void Printer::begin(const std::string& task, Clock::time_point now) { trackers_[task].begin(now); }

void Printer::report(const Report& report) { this->report(report, Clock::now()); }

void Printer::report(const Report& report, Clock::time_point now) {
    if (mode_ == Mode::Off) {
        return;
    }
    // The clock runs on every report, written or not.
    const Timing timing = trackers_[report.task].update(report, now);
    if (!throttle_.due(now, report.finished)) {
        return;
    }
    const std::string line = format(report, timing, context_);
    if (mode_ == Mode::Terminal) {
        // Redrawn over the last: what is left of a longer one cleared.
        std::fprintf(stream_, "\r%s\x1b[K", line.c_str());
        open_ = line.size();
        if (report.finished) {
            std::fputc('\n', stream_);
            open_ = 0;
        }
    } else {
        std::fprintf(stream_, "%s\n", line.c_str());
    }
    std::fflush(stream_);
    ++written_;
}

void Printer::close() {
    if (mode_ == Mode::Terminal && open_ > 0) {
        std::fputc('\n', stream_);
        std::fflush(stream_);
        open_ = 0;
    }
}

Callback Printer::callback() {
    return [this](const Report& report) { this->report(report); };
}

}   // namespace athenea::progress
