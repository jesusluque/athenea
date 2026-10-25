// Copyright (c) 2026 jesus luque.
//
// `athenea video-encode`: splat frames -- a sequence of SPZ pages (`athenea
// flatten --range`, one frame each) or splat files taken as one frame -- laid
// into video planes on the device (video_planes_encode.slang), through the
// hardware HEVC encoder, into a `.athv` (modules/video/include/athenea/video/
// Athv.h).
//
// The processor reads files, copies samples into the encoder's buffers and
// keeps counts; the planes and the quantisation are kernels.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Readers.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/video/Athv.h"
#include "athenea/video/Codec.h"
#include "athenea/video/Planes.h"

namespace athenea::cli {
namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct InputFrame {
    double                 time = 0.0;
    std::vector<fs::path>  files;
};

/// The frames to encode: a pages directory's stream.txt, or the files as one.
Result<std::vector<InputFrame>> inputFrames(const std::vector<std::string>& inputs, double& rate) {
    std::vector<InputFrame> frames;
    if (inputs.size() == 1 && fs::is_directory(inputs[0])) {
        const fs::path dir = inputs[0];
        std::ifstream in(dir / "stream.txt");
        if (!in) {
            return Error::make(ErrorCode::NotFound, "video-encode: no {} (athenea flatten --range writes it)",
                               (dir / "stream.txt").string());
        }
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream words(line);
            std::string key;
            words >> key;
            if (key == "rate") {
                double r = 0.0;
                words >> r;
                if (r > 0.0 && rate <= 0.0) rate = r;
            } else if (key == "frame") {
                unsigned long long index = 0;
                size_t pages = 0;
                InputFrame f;
                words >> index >> f.time >> pages;
                for (size_t c = 0; c < pages; ++c) {
                    char name[64];
                    std::snprintf(name, sizeof name, "f%05llu_c%04zu.spz", index, c);
                    f.files.push_back(dir / name);
                }
                frames.push_back(std::move(f));
            }
        }
    } else {
        InputFrame f;
        for (const auto& i : inputs) f.files.emplace_back(i);
        frames.push_back(std::move(f));
    }
    if (frames.empty()) {
        return Error(ErrorCode::InvalidArgument, "video-encode: no frames to encode");
    }
    if (rate <= 0.0) rate = 30.0;
    return frames;
}

/// A frame's files decoded on the device and laid end to end in one cloud.
Result<scene::GpuSplats> loadFrame(gpu::Device& device, scene::CloudLoader& loader, const InputFrame& f,
                                   uint32_t maxDegree, bool& antialiased) {
    std::vector<scene::GpuSplats> parts;
    uint64_t total = 0;
    for (const fs::path& file : f.files) {
        auto raw = io::readSplats(file);
        if (!raw) return std::move(raw).error();
        if (parts.empty()) antialiased = raw->antialiased.value_or(false);
        auto cloud = loader.upload(*raw, maxDegree);
        if (!cloud) return std::move(cloud).error();
        if (!parts.empty() && cloud->restPerColour != parts[0].restPerColour) {
            return Error::make(ErrorCode::InvalidArgument, "video-encode: {} has degree {} harmonics, not {}'s",
                               file.string(), cloud->restPerColour, f.files[0].string());
        }
        total += cloud->count;
        parts.push_back(std::move(*cloud));
    }
    if (parts.size() == 1) return std::move(parts[0]);
    auto whole = video::PlanesCodec::allocate(device, static_cast<uint32_t>(total), parts[0].restPerColour,
                                              "video.frame");
    if (!whole) return std::move(whole).error();
    gpu::CommandBatch batch(device);
    uint64_t at = 0;
    for (const scene::GpuSplats& p : parts) {
        batch.encoder()->copyBuffer(whole->positions.rhi(), at * 16, p.positions.rhi(), 0, uint64_t{p.count} * 16);
        batch.encoder()->copyBuffer(whole->shape.rhi(), at * 16, p.shape.rhi(), 0, uint64_t{p.count} * 16);
        batch.encoder()->copyBuffer(whole->sh.rhi(), at * whole->shWords * 4, p.sh.rhi(), 0,
                                    uint64_t{p.count} * p.shWords * 4);
        at += p.count;
    }
    batch.markDirty();
    ATHENEA_TRY(batch.submit(true));
    return std::move(*whole);
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

std::string timecodeOf(uint64_t n, double rate) {
    const uint64_t fps = std::max<uint64_t>(1, static_cast<uint64_t>(std::lround(rate)));
    const uint64_t f = n % fps, s = (n / fps) % 60, m = (n / fps / 60) % 60, h = n / fps / 3600;
    char out[32];
    std::snprintf(out, sizeof out, "%02llu:%02llu:%02llu:%02llu", static_cast<unsigned long long>(h),
                  static_cast<unsigned long long>(m), static_cast<unsigned long long>(s),
                  static_cast<unsigned long long>(f));
    return out;
}

}   // namespace

void addVideo(CLI::App& app) {
    struct Options {
        std::vector<std::string> inputs;
        std::string output, hevc, range, hi = "side", order = "frame", ranges = "group";
        uint32_t bits = 8, keyInterval = 0, columns = 0, shDegree = 3;
        double quality = 0.5, megabits = 0.0, rate = 0.0;
        bool intra = false;
    };
    auto o = std::make_shared<Options>();
    auto* enc = app.add_subcommand("video-encode",
                                   "splat frames (a directory of SPZ pages, or splat files as one frame) laid into "
                                   "video planes and encoded by the hardware HEVC encoder into a .athv");
    enc->add_option("input", o->inputs, "a pages directory (stream.txt) or splat files")->required();
    enc->add_option("-o,--output", o->output, "the .athv written")->required();
    enc->add_option("--bits", o->bits, "bits a sample: 8 (HEVC Main) or 10 (Main 10)")
        ->check(CLI::IsMember({8u, 10u}));
    enc->add_option("--range", o->range, "FIRST:LAST: the sequence's frames by index, from 0, both included");
    enc->add_option("--quality", o->quality, "the encoder's constant quality, 0 to 1")->check(CLI::Range(0.0, 1.0));
    enc->add_option("--megabits", o->megabits, "an average bit rate in Mbit/s instead of a quality");
    enc->add_option("--keyint", o->keyInterval, "frames from one key frame to the next (default: a second)");
    enc->add_flag("--intra", o->intra, "every frame a key frame");
    enc->add_option("--hi", o->hi, "the position high parts: side (deflated beside the video, exact) or video "
                                   "(planes of their own, through the codec)")
        ->check(CLI::IsMember({"side", "video"}));
    enc->add_option("--order", o->order, "first (the first frame's Morton order for every frame of as many splats) "
                                         "or frame (each frame sorted again)")
        ->check(CLI::IsMember({"first", "frame"}));
    enc->add_option("--ranges", o->ranges, "group (each attribute's range over a group of pictures, so what "
                                           "stays put keeps its samples) or frame (each frame's own)")
        ->check(CLI::IsMember({"group", "frame"}));
    enc->add_option("--sh-degree", o->shDegree, "harmonics degree carried, at most the source's")
        ->check(CLI::Range(0u, 3u));
    enc->add_option("--columns", o->columns, "tiles a mosaic row (default: the squarest mosaic)");
    enc->add_option("--rate", o->rate, "frames a second (default: the sequence's, or 30)");
    enc->add_option("--hevc", o->hevc, "also the video alone as an Annex B .hevc, for a player or a probe");

    enc->callback([o] {
        double rate = o->rate;
        auto frames = inputFrames(o->inputs, rate);
        if (!frames) cli::fail(frames.error());
        size_t first = 0, last = frames->size() - 1;
        if (!o->range.empty()) {
            unsigned long long a = 0, b = 0;
            if (std::sscanf(o->range.c_str(), "%llu:%llu", &a, &b) != 2 || a > b || b >= frames->size()) {
                cli::err("video-encode: --range wants FIRST:LAST within 0:%zu\n", frames->size() - 1);
                throw CLI::RuntimeError(kExitFailure);
            }
            first = a;
            last = b;
        }
        auto device = gpu::Device::create();
        if (!device) cli::fail(device.error());
        gpu::ShaderLibrary library(*device);
        auto loader = scene::CloudLoader::create(library);
        if (!loader) cli::fail(loader.error());
        auto codec = video::PlanesCodec::create(library);
        if (!codec) cli::fail(codec.error());
        const uint32_t restAsked = o->shDegree == 0 ? 0u : (o->shDegree + 1) * (o->shDegree + 1) - 1;

        std::unique_ptr<video::AthvWriter> writer;
        std::unique_ptr<video::VideoEncoder> encoder;
        std::ofstream hevc;
        if (!o->hevc.empty()) {
            hevc.open(o->hevc, std::ios::binary | std::ios::trunc);
            if (!hevc) {
                cli::err("video-encode: cannot write %s\n", o->hevc.c_str());
                throw CLI::RuntimeError(kExitFailure);
            }
        }
        video::PlaneLayout layout;   // the stream's pictures: every one this size
        uint32_t sliceCap = 0;       // the most splats a picture holds
        gpu::Buffer order, ranges, frameRanges;
        uint32_t orderedFor = 0, picture = 0;
        std::vector<double> unitBytes, sideBytes, keyBytes, interBytes, layMs, encodeMs;
        uint64_t keyframes = 0;
        bool antialiased = false;
        uint32_t keyInterval =
            o->intra ? 1u : (o->keyInterval != 0 ? o->keyInterval : static_cast<uint32_t>(std::lround(rate)));
        const bool groupRanges = o->ranges == "group";
        // A frame loaded; on the first, the pictures' size, the encoder and
        // the file made.
        const auto prepare = [&](size_t k) -> scene::GpuSplats {
            bool aa = false;
            auto cloud = loadFrame(**device, *loader, (*frames)[k], o->shDegree, aa);
            if (!cloud) cli::fail(cloud.error());
            if (encoder != nullptr) return std::move(*cloud);
            antialiased = aa;
            const uint32_t rest = std::min(cloud->restPerColour, restAsked);
            auto cap = video::PlaneLayout::sliceSplats(cloud->count, o->bits, rest, o->hi == "video",
                                                       video::PlaneLayout::kBandSplats, o->columns);
            if (!cap) cli::fail(cap.error());
            sliceCap = *cap;
            auto made = video::PlaneLayout::make(sliceCap, o->bits, rest, o->hi == "video",
                                                 video::PlaneLayout::kBandSplats, o->columns);
            if (!made) cli::fail(made.error());
            layout = *made;
            // A frame cut into slices: each picture holds other splats than
            // the one before it, so none is predicted from another.
            if (sliceCap < cloud->count) keyInterval = 1;
            video::EncoderSettings es;
            es.width = layout.width;
            es.height = layout.height;
            es.bits = layout.bits;
            es.rate = rate;
            es.quality = o->quality;
            es.megabits = o->megabits;
            es.keyInterval = keyInterval;
            auto e = video::createHardwareEncoder(es);
            if (!e) cli::fail(e.error());
            encoder = std::move(*e);
            video::AthvHeader header;
            header.bits = layout.bits;
            header.restPerColour = layout.restPerColour;
            header.chunkSplats = layout.chunkSplats;
            header.columns = layout.columns;
            header.maxSplats = sliceCap;
            header.hiInVideo = layout.hiInVideo;
            header.antialiased = antialiased;
            header.rate = rate;
            auto w = video::AthvWriter::create(o->output, header);
            if (!w) cli::fail(w.error());
            writer = std::move(*w);
            cli::out("video-encode: %zu frames of %u splats, %u planes of %u x %u in a %u x %u mosaic, %u slice%s a "
                     "frame, %u bits, high parts %s, encoder %s\n",
                     last - first + 1, cloud->count, layout.planes(), video::PlaneLayout::kTileWidth,
                     layout.tileHeight, layout.width, layout.height, (cloud->count + sliceCap - 1) / sliceCap,
                     sliceCap < cloud->count ? "s" : "", layout.bits, layout.hiInVideo ? "in the video" : "beside it",
                     encoder->hardware() ? "hardware" : "software");
            return std::move(*cloud);
        };
        for (size_t g = first; g <= last; g += keyInterval) {
            const size_t groupEnd = std::min(last, g + keyInterval - 1);
            if (groupRanges) {
                // The group's ranges first, over every frame of it.
                for (size_t k = g; k <= groupEnd; ++k) {
                    scene::GpuSplats cloud = prepare(k);
                    if (auto r = codec->ranges(cloud, layout, cloud.count, frameRanges); !r) cli::fail(r.error());
                    if (auto r = codec->widen(layout, frameRanges, ranges, k == g); !r) cli::fail(r.error());
                }
            }
            for (size_t k = g; k <= groupEnd; ++k) {
                const InputFrame& in = (*frames)[k];
                scene::GpuSplats cloud = prepare(k);
                if (!groupRanges) {
                    if (auto r = codec->ranges(cloud, layout, cloud.count, ranges); !r) cli::fail(r.error());
                }
                if (o->order != "first" || orderedFor != cloud.count) {
                    if (auto r = codec->order(cloud, cloud.count, ranges, order); !r) cli::fail(r.error());
                    orderedFor = cloud.count;
                }
                for (uint32_t at = 0; at < cloud.count; at += sliceCap) {
                    const uint32_t count = std::min(sliceCap, cloud.count - at);
                    auto sliced = video::PlaneLayout::make(count, layout.bits, layout.restPerColour, layout.hiInVideo,
                                                           layout.chunkSplats, layout.columns, sliceCap);
                    if (!sliced) cli::fail(sliced.error());
                    if (sliced->width != layout.width || sliced->height != layout.height) {
                        cli::err("video-encode: frame %zu needs a %u x %u mosaic, not the stream's %u x %u\n", k,
                                 sliced->width, sliced->height, layout.width, layout.height);
                        throw CLI::RuntimeError(kExitFailure);
                    }
                    Clock::time_point t = Clock::now();
                    auto planes = codec->encode(cloud, *sliced, order, at, ranges);
                    if (!planes) cli::fail(planes.error());
                    layMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    t = Clock::now();
                    auto units = encoder->encode(planes->luma, picture);
                    if (!units) cli::fail(units.error());
                    encodeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    if (units->size() != 1) {
                        cli::err("video-encode: the encoder gave %zu units for picture %u\n", units->size(), picture);
                        throw CLI::RuntimeError(kExitFailure);
                    }
                    video::EncodedUnit& unit = units->front();
                    video::AthvFrame f;
                    f.index = picture++;
                    f.source = static_cast<uint32_t>(k - first);
                    f.firstSplat = at;
                    f.sourceSplats = cloud.count;
                    f.splats = count;
                    f.keyframe = unit.keyframe;
                    f.time = in.time;
                    f.timecode = timecodeOf(k - first, rate);
                    f.ranges = planes->ranges;
                    f.high = planes->high;
                    f.parameterSets = unit.parameterSets;
                    f.unit = std::move(unit.unit);
                    if (auto w = writer->write(f); !w) cli::fail(w.error());
                    keyframes += f.keyframe ? 1 : 0;
                    unitBytes.push_back(static_cast<double>(f.unit.size()));
                    (f.keyframe ? keyBytes : interBytes).push_back(static_cast<double>(f.unit.size()));
                    sideBytes.push_back(static_cast<double>(f.recordBytes - f.unit.size()));
                    if (hevc.is_open()) {
                        const std::vector<uint8_t> b = video::annexB(f);
                        hevc.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
                    }
                }
            }
        }
        if (auto fin = writer->finish(0); !fin) cli::fail(fin.error());
        // Per frame (an epoch: one picture, or a frame's slices together).
        const double n = static_cast<double>(last - first + 1);
        double unitTotal = 0.0, sideTotal = 0.0;
        for (double b : unitBytes) unitTotal += b;
        for (double b : sideBytes) sideTotal += b;
        const double perFrame = (unitTotal + sideTotal) / n / 1.0e6;
        cli::out("video-encode: wrote %s: %.0f frames in %u pictures (%llu key), %.3f MB a frame (video %.3f, side "
                 "%.3f), %.1f Mbit/s at 24 a second, %.1f at 30\n",
                 o->output.c_str(), n, picture, static_cast<unsigned long long>(keyframes), perFrame,
                 unitTotal / n / 1.0e6, sideTotal / n / 1.0e6, perFrame * 8.0 * 24.0, perFrame * 8.0 * 30.0);
        cli::out("video-encode: a key picture's video %.3f MB, a predicted one's %.3f MB (medians)\n",
                 median(keyBytes) / 1.0e6, median(interBytes) / 1.0e6);
        cli::out("video-encode: a picture laid into planes in %.1f ms (median, read back included), encoded in %.1f "
                 "ms\n",
                 median(layMs), median(encodeMs));
    });
}

}   // namespace athenea::cli
