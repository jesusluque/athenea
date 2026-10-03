// Copyright (c) 2026 jesus luque.
//
// Progress is bookkeeping on the CPU: counts handed over, timed when they
// arrive. Its line, its clock and its throttle are checked here with times
// the test chooses, and the printer against a file.
#include "athenea/core/Progress.h"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdio>
#include <string>

using namespace athenea;
using namespace std::chrono_literals;

namespace {

progress::Report pathReport(uint64_t done, uint32_t pass, bool finished = false) {
    progress::Report report;
    report.task = "render";
    report.unit = "paths/px";
    report.done = done;
    report.total = 2048;
    report.pass = pass;
    report.passes = 256;
    report.finished = finished;
    return report;
}

std::string readAll(std::FILE* file) {
    std::rewind(file);
    std::string text;
    char buffer[256];
    size_t n = 0;
    while ((n = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
        text.append(buffer, n);
    }
    return text;
}

size_t count(const std::string& text, const std::string& what) {
    size_t found = 0;
    for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + what.size())) {
        ++found;
    }
    return found;
}

}   // namespace

TEST_CASE("a duration reads as seconds, minutes or hours", "[core][progress]") {
    CHECK(progress::formatDuration(0.0) == "0.0s");
    CHECK(progress::formatDuration(12.34) == "12.3s");
    CHECK(progress::formatDuration(245.0) == "4m05s");
    CHECK(progress::formatDuration(3720.0) == "1h02m");
    CHECK(progress::formatDuration(-1.0) == "0.0s");
}

TEST_CASE("a progress line says done, total, pass, time, rate and ETA", "[core][progress]") {
    progress::Timing timing;
    timing.elapsed = 72.0;
    timing.rate = 7.11;
    timing.eta = 216.0;
    CHECK(progress::format(pathReport(512, 64), timing, "frame 2/10") ==
          "frame 2/10  render 512/2048 paths/px (25.0%)  pass 64/256  1m12s  7.11 paths/px/s  ETA 3m36s");
    // Not known yet: no rate, no ETA.
    CHECK(progress::format(pathReport(8, 1), progress::Timing{}) ==
          "render 8/2048 paths/px (0.4%)  pass 1/256  0.0s");
    timing.eta = 0.0;
    CHECK(progress::format(pathReport(2048, 256, true), timing) ==
          "render 2048/2048 paths/px (100.0%)  pass 256/256  1m12s  7.11 paths/px/s  done");
    progress::Report denoise;
    denoise.task = "denoise";
    CHECK(progress::format(denoise, progress::Timing{}) == "denoise  0.0s");
}

TEST_CASE("a task's rate is taken from its first report on", "[core][progress]") {
    const progress::Clock::time_point t0{};
    progress::Tracker tracker;
    tracker.begin(t0);
    // The first pass carried the stage's load: in the elapsed time only.
    progress::Timing timing = tracker.update(pathReport(8, 1), t0 + 10s);
    CHECK(timing.elapsed == 10.0);
    // 8 a second, from what began to the first report; known, but slow.
    CHECK(timing.rate == 0.8);
    timing = tracker.update(pathReport(88, 11), t0 + 20s);
    CHECK(timing.elapsed == 20.0);
    CHECK(timing.rate == 8.0);
    CHECK(timing.eta == (2048.0 - 88.0) / 8.0);
    timing = tracker.update(pathReport(2048, 256, true), t0 + 265s);
    CHECK(timing.eta == 0.0);
    // Going backwards starts the task over: a viewport's camera moved.
    timing = tracker.update(pathReport(8, 1), t0 + 300s);
    CHECK(timing.elapsed == 0.0);
    CHECK(timing.rate == 0.0);
    CHECK(timing.eta < 0.0);
}

TEST_CASE("the throttle lets a line through once an interval, and a forced one always", "[core][progress]") {
    const progress::Clock::time_point t0{};
    progress::Throttle throttle(500ms);
    CHECK(throttle.due(t0));
    CHECK_FALSE(throttle.due(t0 + 100ms));
    CHECK_FALSE(throttle.due(t0 + 499ms));
    CHECK(throttle.due(t0 + 500ms));
    CHECK_FALSE(throttle.due(t0 + 600ms));
    CHECK(throttle.due(t0 + 601ms, true));
    CHECK_FALSE(throttle.due(t0 + 700ms));
    CHECK(throttle.due(t0 + 1101ms));
}

TEST_CASE("a printer redraws on a terminal and logs lines otherwise", "[core][progress]") {
    const progress::Clock::time_point t0{};
    SECTION("lines: one every ten seconds, and the last") {
        std::FILE* file = std::tmpfile();
        REQUIRE(file != nullptr);
        {
            progress::Printer printer(progress::Mode::Lines, file);
            printer.begin("render", t0);
            // A pass a second for 256 passes.
            for (uint32_t pass = 1; pass <= 256; ++pass) {
                printer.report(pathReport(uint64_t{pass} * 8, pass, pass == 256), t0 + std::chrono::seconds(pass));
            }
            // At 1 s, then 11, 21, ... 251: 26 lines, and the finished one.
            CHECK(printer.written() == 27);
        }
        const std::string text = readAll(file);
        std::fclose(file);
        CHECK(count(text, "\n") == 27);
        CHECK(count(text, "\r") == 0);
        CHECK(text.find("render 8/2048 paths/px (0.4%)  pass 1/256") == 0);
        CHECK(text.find("render 2048/2048 paths/px (100.0%)  pass 256/256") != std::string::npos);
        CHECK(text.find("done\n") == text.size() - 5);
    }
    SECTION("terminal: two a second, in place, a newline when finished") {
        std::FILE* file = std::tmpfile();
        REQUIRE(file != nullptr);
        {
            progress::Printer printer(progress::Mode::Terminal, file);
            printer.setContext("frame 1/2");
            for (uint32_t pass = 1; pass <= 20; ++pass) {
                printer.report(pathReport(uint64_t{pass} * 8, pass), t0 + std::chrono::milliseconds(100 * pass));
            }
            // 0.1 s, then 0.6, 1.1, 1.6: four.
            CHECK(printer.written() == 4);
            printer.close();
        }
        const std::string text = readAll(file);
        std::fclose(file);
        CHECK(count(text, "\r") == 4);
        CHECK(count(text, "\x1b[K") == 4);
        CHECK(count(text, "\n") == 1);
        CHECK(text.find("\rframe 1/2  render 8/2048") == 0);
    }
    SECTION("off: nothing") {
        std::FILE* file = std::tmpfile();
        REQUIRE(file != nullptr);
        {
            progress::Printer printer(progress::Mode::Off, file);
            printer.report(pathReport(2048, 256, true), t0);
            CHECK(printer.written() == 0);
        }
        CHECK(readAll(file).empty());
        std::fclose(file);
    }
}
