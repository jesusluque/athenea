// Copyright (c) 2026 jesus luque.
//
// `athenea stream`: a sequence of SPZ pages (`athenea flatten --range`, one
// epoch a frame) played into the live stream a stage names
// (AtheneaLiveSourceAPI), and the stage drawn while it plays. A producer
// thread pushes each frame's pages at the sequence's rate, with the epoch's
// PTP time; the frame thread draws the stage, which shows the last epoch whose
// every page is on the device.
//
// Two ways to play:
//   by the clock   (default) the producer keeps the rate whatever the frames
//                  do; the frames are drawn as fast as they come, and what
//                  each showed is counted: epochs shown, skipped, and how late.
//   --lockstep     each frame pushes its epoch and draws until the stage shows
//                  it: every epoch drawn once, for comparing frame by frame
//                  (--reference: the source stage drawn at the frame's time
//                  code and measured against it on the device).
// The processor reads files, counts and keeps time; the pages are decoded and
// drawn by the device, and images are compared there.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::cli {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

/// One frame of the sequence: its time code and its pages' bytes.
struct Frame {
    double                              time = 0.0;
    uint64_t                            bytes = 0;
    std::vector<std::vector<std::byte>> pages;
};

struct Sequence {
    double             rate = 30.0;
    uint32_t           chunks = 0;
    uint32_t           chunkSplats = 0;
    std::vector<Frame> frames;
};

/// The manifest `athenea flatten --range` writes, and every page it names.
Result<Sequence> readSequence(const fs::path& dir) {
    std::ifstream in(dir / "stream.txt");
    if (!in) {
        return Error::make(ErrorCode::NotFound, "stream: no {} (athenea flatten --range writes it)",
                           (dir / "stream.txt").string());
    }
    Sequence s;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream words(line);
        std::string key;
        words >> key;
        if (key == "rate") {
            words >> s.rate;
        } else if (key == "chunks") {
            words >> s.chunks;
        } else if (key == "chunkSplats") {
            words >> s.chunkSplats;
        } else if (key == "frame") {
            unsigned long long index = 0, gaussians = 0, bytes = 0;
            size_t pages = 0;
            Frame f;
            words >> index >> f.time >> pages >> gaussians >> bytes;
            for (size_t c = 0; c < pages; ++c) {
                char name[64];
                std::snprintf(name, sizeof name, "f%05llu_c%04zu.spz", index, c);
                std::ifstream page(dir / name, std::ios::binary);
                if (!page) {
                    return Error::make(ErrorCode::NotFound, "stream: page {} is missing", (dir / name).string());
                }
                std::vector<char> raw((std::istreambuf_iterator<char>(page)), std::istreambuf_iterator<char>());
                std::vector<std::byte> b(raw.size());
                std::memcpy(b.data(), raw.data(), raw.size());
                f.bytes += b.size();
                f.pages.push_back(std::move(b));
            }
            s.frames.push_back(std::move(f));
        }
    }
    if (s.frames.empty() || s.rate <= 0.0) {
        return Error::make(ErrorCode::InvalidArgument, "stream: {} holds no frames", (dir / "stream.txt").string());
    }
    return s;
}

/// HH:MM:SS:FF of frame `n` at a whole rate: a label, nothing computed from it.
std::string timecodeOf(uint64_t n, double rate) {
    const uint64_t fps = static_cast<uint64_t>(std::lround(rate));
    const uint64_t f = n % fps, s = (n / fps) % 60, m = (n / fps / 60) % 60, h = n / fps / 3600;
    char out[32];
    std::snprintf(out, sizeof out, "%02llu:%02llu:%02llu:%02llu", static_cast<unsigned long long>(h),
                  static_cast<unsigned long long>(m), static_cast<unsigned long long>(s),
                  static_cast<unsigned long long>(f));
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size())))];
}

}   // namespace

void addStream(CLI::App& app) {
    struct Options {
        std::string stage, pages, stream, camera, size = "960x540", output, reference;
        double rate = 0.0;
        uint32_t loops = 1;
        bool lockstep = false, displayBlend = false;
    };
    auto o = std::make_shared<Options>();
    auto* cmd = app.add_subcommand(
        "stream", "play a sequence of SPZ pages (athenea flatten --range) into the live stream a stage names, "
                  "drawing the stage as it plays");
    cmd->add_option("stage", o->stage, "a stage with a ParticleField under AtheneaLiveSourceAPI")->required();
    cmd->add_option("pages", o->pages, "the pages' directory (its stream.txt)")->required();
    cmd->add_option("--stream", o->stream, "the stream pushed to (default: the one the stage names)");
    cmd->add_option("--camera", o->camera, "camera prim (default: the stage's first)");
    cmd->add_option("--size", o->size, "WIDTHxHEIGHT");
    cmd->add_option("--rate", o->rate, "epochs a second (default: the sequence's)");
    cmd->add_option("--loops", o->loops, "times the sequence is played, its epochs counting on");
    cmd->add_flag("--lockstep", o->lockstep, "each frame waits for its epoch: every epoch drawn once");
    cmd->add_flag("--splat-display-blend", o->displayBlend,
                  "the stage's splats blended as a standard viewer blends them, sRGB values summed -- how the "
                  "colours flatten fits are meant to be drawn (athenea:splatDisplayBlend)");
    cmd->add_option("-o,--output", o->output, "a directory: each frame as frame_NNNNN.exr, with its timecode");
    cmd->add_option("--reference", o->reference,
                    "with --lockstep: a stage drawn at each frame's time code (the source, as it moves) and "
                    "measured against the frame on the device: p99, max and pixels past 2 code values");

    cmd->callback([o] {
        uint32_t width = 0, height = 0;
        if (std::sscanf(o->size.c_str(), "%ux%u", &width, &height) != 2) {
            cli::err("stream: --size wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        if (!o->reference.empty() && !o->lockstep) {
            cli::err("stream: --reference compares frame by frame, and needs --lockstep\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        auto sequence = readSequence(o->pages);
        if (!sequence) cli::fail(sequence.error());
        const Sequence& seq = *sequence;
        const double rate = o->rate > 0.0 ? o->rate : seq.rate;
        std::vector<double> perEpoch;
        for (const Frame& f : seq.frames) perEpoch.push_back(static_cast<double>(f.bytes));
        cli::out("stream: %zu frames of up to %u pages (%u gaussians a page), %.2f MB an epoch (median), at %g a "
                 "second\n",
                 seq.frames.size(), seq.chunks, seq.chunkSplats, median(perEpoch) / 1.0e6, rate);

        auto device = gpu::Device::create();
        if (!device) cli::fail(device.error());
        auto renderer = usd::StageRenderer::open(o->stage, *device);
        if (!renderer) cli::fail(renderer.error());
        std::unique_ptr<usd::StageRenderer> reference;
        if (!o->reference.empty()) {
            auto opened = usd::StageRenderer::open(o->reference, *device);
            if (!opened) cli::fail(opened.error());
            reference = std::move(*opened);
        }
        (*renderer)->setSplatDisplayBlend(o->displayBlend);
        gpu::ShaderLibrary library(*device);
        const std::string camera = o->camera.empty() ? ((*renderer)->cameras().empty() ? std::string()
                                                                                         : (*renderer)->cameras()[0])
                                                     : o->camera;
        // A first frame: the stage committed, its live pool opened, the
        // stream it names known.
        auto first = (*renderer)->render(camera, 0.0, width, height);
        if (!first) cli::fail(first.error());
        std::string stream = o->stream;
        if (stream.empty()) {
            const auto live = (*renderer)->liveStreams();
            if (live.empty()) {
                cli::err("stream: %s names no live stream (AtheneaLiveSourceAPI)\n", o->stage.c_str());
                throw CLI::RuntimeError(kExitFailure);
            }
            stream = live.front().stream;
        }
        const auto shown = [&]() -> usd::StageRenderer::LiveStream {
            for (const auto& l : (*renderer)->liveStreams()) {
                if (l.stream == stream) return l;
            }
            return {};
        };
        auto queue = lod::PageQueue::named(stream);
        const uint64_t total = uint64_t{seq.frames.size()} * std::max(o->loops, 1u);
        // The epoch's PTP time: the wall clock's nanoseconds at the start,
        // then a frame period an epoch.
        const uint64_t ptpStart = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        const auto ptpOf = [&](uint64_t epoch) {
            return ptpStart + static_cast<uint64_t>(std::llround(static_cast<double>(epoch - 1) * 1.0e9 / rate));
        };
        const auto push = [&](uint64_t epoch) {
            const Frame& f = seq.frames[(epoch - 1) % seq.frames.size()];
            const uint32_t pages = std::max<uint32_t>(seq.chunks, static_cast<uint32_t>(f.pages.size()));
            for (uint32_t c = 0; c < pages; ++c) {
                lod::PageUpdate u;
                u.epoch = epoch;
                u.chunk = c;
                u.pages = pages;
                u.ptpTime = ptpOf(epoch);
                if (c < f.pages.size()) {
                    u.op = lod::PageUpdate::Op::Replace;
                    u.bytes = f.pages[c];
                } else {
                    u.op = lod::PageUpdate::Op::Remove;   // a frame of fewer pages than the most
                }
                queue->push(std::move(u));
            }
        };
        const auto write = [&](uint64_t n, const std::vector<float>& rgba, const std::vector<float>& depth,
                               const char* prefix, uint64_t epoch) -> Result<void> {
            if (o->output.empty()) return ok();
            std::error_code made;
            fs::create_directories(o->output, made);
            char name[64];
            std::snprintf(name, sizeof name, "%s_%05llu.exr", prefix, static_cast<unsigned long long>(n));
            const std::vector<io::ExrAttribute> attributes{
                io::ExrAttribute::text("athenea:timecode", timecodeOf(epoch - 1, rate)),
                io::ExrAttribute::float64("athenea:ptpTime", static_cast<double>(ptpOf(epoch)) * 1.0e-9),
                io::ExrAttribute::float64("athenea:epoch", static_cast<double>(epoch))};
            return io::writeExr(fs::path(o->output) / name, width, height, rgba, depth, true, attributes);
        };

        if (o->lockstep) {
            std::vector<double> p99s, maxes, over2s, frameMs;
            uint32_t worstMax = 0;
            uint64_t worstAt = 0, renders = 0;
            for (uint64_t epoch = 1; epoch <= total; ++epoch) {
                const Clock::time_point start = Clock::now();
                push(epoch);
                Result<usd::StageImage> image = Error(ErrorCode::InvalidArgument, "no frame");
                for (int k = 0; k < 400; ++k) {
                    image = (*renderer)->render(camera, 0.0, width, height);
                    ++renders;
                    if (!image) cli::fail(image.error());
                    if (shown().epoch >= epoch) break;
                }
                frameMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
                const usd::StageRenderer::LiveStream now = shown();
                if (now.epoch != epoch) {
                    cli::err("stream: epoch %llu never shown (the stage shows %llu; %llu dropped without room, "
                             "%llu corrupt)\n",
                             static_cast<unsigned long long>(epoch), static_cast<unsigned long long>(now.epoch),
                             static_cast<unsigned long long>(now.droppedNoRoom),
                             static_cast<unsigned long long>(now.droppedCorrupt));
                    throw CLI::RuntimeError(kExitFailure);
                }
                if (auto w = write(epoch, image->rgba, image->depth, "frame", epoch); !w) cli::fail(w.error());
                if (reference != nullptr) {
                    const double time = seq.frames[(epoch - 1) % seq.frames.size()].time;
                    auto ref = reference->render(camera, time, width, height);
                    if (!ref) cli::fail(ref.error());
                    if (auto w = write(epoch, ref->rgba, ref->depth, "reference", epoch); !w) cli::fail(w.error());
                    gpu::BufferDesc desc;
                    desc.bytes = image->rgba.size() * sizeof(float);
                    desc.elementBytes = 16;
                    auto a = gpu::Buffer::create(**device, desc, image->rgba.data());
                    auto b = gpu::Buffer::create(**device, desc, ref->rgba.data());
                    if (!a) cli::fail(a.error());
                    if (!b) cli::fail(b.error());
                    auto d = render::compareImages(library, *a, *b, width, height);
                    if (!d) cli::fail(d.error());
                    p99s.push_back(d->p99);
                    maxes.push_back(d->max);
                    over2s.push_back(100.0 * static_cast<double>(d->over2) / static_cast<double>(d->pixels));
                    if (d->max >= worstMax) {
                        worstMax = d->max;
                        worstAt = epoch;
                    }
                    cli::out("stream: epoch %llu (time %g, %s): p99 %u, max %u, over2 %.3f%%\n",
                             static_cast<unsigned long long>(epoch), time, timecodeOf(epoch - 1, rate).c_str(),
                             d->p99, d->max, over2s.back());
                }
            }
            cli::out("stream: lockstep, %llu epochs in %llu frames drawn; a frame's epoch shown after %.1f ms median "
                     "(p95 %.1f)\n",
                     static_cast<unsigned long long>(total), static_cast<unsigned long long>(renders),
                     median(frameMs), percentile(frameMs, 0.95));
            if (!p99s.empty()) {
                cli::out("stream: against --reference: p99 median %.0f (worst %.0f), max median %.0f (worst %u at "
                         "epoch %llu), over2 median %.3f%% (worst %.3f%%)\n",
                         median(p99s), percentile(p99s, 1.0), median(maxes), worstMax,
                         static_cast<unsigned long long>(worstAt), median(over2s), percentile(over2s, 1.0));
            }
            return;
        }

        // BY THE CLOCK: the producer keeps the rate; the frames take what
        // has come whole.
        std::atomic<bool> produced{false};
        const Clock::time_point t0 = Clock::now() + std::chrono::milliseconds(100);
        const auto dueOf = [&](uint64_t epoch) {
            return t0 + std::chrono::nanoseconds(
                            static_cast<int64_t>(std::llround(static_cast<double>(epoch - 1) * 1.0e9 / rate)));
        };
        std::thread producer([&] {
            for (uint64_t epoch = 1; epoch <= total; ++epoch) {
                std::this_thread::sleep_until(dueOf(epoch));
                push(epoch);
            }
            produced = true;
        });
        std::vector<double> frameMs, lateMs;
        std::vector<uint8_t> seen(total + 1, 0);
        uint64_t frames = 0, last = 0, backwards = 0;
        Clock::time_point previous = Clock::now();
        const Clock::time_point stopAt = dueOf(total) + std::chrono::seconds(2);
        while (Clock::now() < stopAt) {
            auto image = (*renderer)->render(camera, 0.0, width, height);
            if (!image) {
                produced = true;
                producer.join();
                cli::fail(image.error());
            }
            const Clock::time_point now = Clock::now();
            frameMs.push_back(std::chrono::duration<double, std::milli>(now - previous).count());
            previous = now;
            const usd::StageRenderer::LiveStream s = shown();
            if (s.epoch > 0 && s.epoch <= total) {
                if (seen[s.epoch] == 0) {
                    lateMs.push_back(std::chrono::duration<double, std::milli>(now - dueOf(s.epoch)).count());
                    if (auto w = write(s.epoch, image->rgba, image->depth, "frame", s.epoch); !w) {
                        produced = true;
                        producer.join();
                        cli::fail(w.error());
                    }
                }
                seen[s.epoch] = 1;
            }
            backwards += s.epoch < last ? 1 : 0;
            last = s.epoch;
            ++frames;
            if (produced && s.epoch == total) break;
        }
        producer.join();
        uint64_t shownCount = 0;
        for (uint64_t e = 1; e <= total; ++e) shownCount += seen[e];
        const usd::StageRenderer::LiveStream end = shown();
        cli::out("stream: by the clock at %g a second: %llu frames drawn (median %.1f ms, p95 %.1f); %llu of %llu "
                 "epochs shown, %llu skipped, %llu backwards; an epoch shown %.1f ms after it was due (median, p95 "
                 "%.1f); dropped without room %llu, corrupt %llu\n",
                 rate, static_cast<unsigned long long>(frames), median(frameMs), percentile(frameMs, 0.95),
                 static_cast<unsigned long long>(shownCount), static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(total - shownCount), static_cast<unsigned long long>(backwards),
                 median(lateMs), percentile(lateMs, 0.95), static_cast<unsigned long long>(end.droppedNoRoom),
                 static_cast<unsigned long long>(end.droppedCorrupt));
        cli::out("stream: the rate %s: %s\n", shownCount == total ? "holds" : "does not hold",
                 shownCount == total ? "every epoch was drawn" : "epochs were skipped for later ones");
    });
}

}   // namespace athenea::cli
