// Copyright (c) 2026 jesus luque.
//
// `athenea video-encode`: splat frames -- a sequence of SPZ pages (`athenea
// flatten --range`, one frame each) or splat files taken as one frame -- laid
// into video planes on the device (video_planes_encode.slang), through the
// hardware HEVC encoder, into a `.athv` (modules/video/include/athenea/video/
// Athv.h). With --check each frame is decoded again by the hardware decoder,
// turned back into splats on the device and measured against its source,
// attribute by attribute.
//
// `athenea video-decode`: every frame of a `.athv` through the hardware
// decoder and the planes -> gaussians kernel, timed: what a player pays a
// frame.
//
// The processor reads files, copies samples into the encoder's buffers and
// keeps counts; the planes, the quantisation and the measures are kernels.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "Commands.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/algo/PrefixSum.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Readers.h"
#include "athenea/lod/Lod.h"
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

/// HOW A FRAME IS CUT TO FEWER GAUSSIANS (--gaussians): merged by pairs
/// (lod::Decimator's NanoGS merge, optionally refusing a merge wider than
/// `growth` times its wider member), pruned by contribution (opacity x the
/// area of the two widest axes, video_reduce.slang), both (`mix`: half the
/// excess pruned, the rest merged), or the octree's own cut by error
/// (lod::Decimator::decimate, its tolerances scaled until the count is near).
/// Whatever a capped merge leaves above the target is pruned.
struct Reduction {
    std::string mode = "pairs";
    float       growth = 0.0F;
    /// lod::PairSettings::opacityRule: 0 the members' mean, widened, under
    /// 0.99; 3 their mass, not widened, up to 5 (Spark's harder kernel).
    uint32_t    opacityRule = 0;
    /// `mix`: the share of the excess pruned before the merges.
    float       pruneShare = 0.5F;
    /// video_reduce.slang's score: 0 opacity x area, 1 thin gaussians kept.
    uint32_t    score = 0;
    /// lod::PairSettings::maxAspect: no merge longer than this against its width.
    float       maxAspect = 1000.0F;
};

struct Reducer {
    gpu::ShaderLibrary* library = nullptr;
    std::optional<lod::LodBuilder> builder;
    std::optional<lod::Decimator>  decimator;
    std::optional<gpu::ComputeKernel> score;
    std::optional<gpu::ComputeKernel> keepFlag, keepScatter;
    std::optional<gpu::PrefixSum>     prefix;
    /// `keep`: the first frame's choice, a flag a record, held for every frame.
    gpu::Buffer flags, at, total;
    uint32_t    keepRecords = 0;

    /// The same records every frame: the first frame's highest scored, about
    /// `target`, chosen once; each frame keeps exactly those, in order.
    Result<scene::GpuSplats> keep(const scene::GpuSplats& cloud, uint32_t target) {
        gpu::Device& device = library->device();
        const uint32_t n = cloud.count;
        const auto bindAll = [&](rhi::ShaderCursor c, const scene::GpuSplats& out, const gpu::Buffer& scores,
                                 float threshold) {
            c["keepScore"].setBinding(scores.rhi());
            c["keepFlag"].setBinding(flags.rhi());
            c["keepAt"].setBinding(at.rhi());
            c["inPositions"].setBinding(cloud.positions.rhi());
            c["inShape"].setBinding(cloud.shape.rhi());
            c["inSh"].setBinding(cloud.sh.rhi());
            c["outPositions"].setBinding(out.positions.rhi());
            c["outShape"].setBinding(out.shape.rhi());
            c["outSh"].setBinding(out.sh.rhi());
            c["keepParams"]["count"].setData(n);
            c["keepParams"]["shWords"].setData(cloud.shWords);
            c["keepParams"]["threshold"].setData(threshold);
        };
        if (!flags.valid()) {
            // The first frame: scored, a threshold from the histogram, flagged.
            keepRecords = n;
            gpu::BufferDesc desc;
            desc.bytes = uint64_t{std::max(n, 1u)} * 4;
            desc.elementBytes = 4;
            desc.label = "video.keep";
            auto scores = gpu::Buffer::create(device, desc);
            auto f = gpu::Buffer::create(device, desc);
            auto a = gpu::Buffer::create(device, desc);
            desc.bytes = 4;
            auto t = gpu::Buffer::create(device, desc);
            if (!scores || !f || !a || !t) return Error(ErrorCode::OutOfMemory, "video: the kept records' flags");
            flags = std::move(*f);
            at = std::move(*a);
            total = std::move(*t);
            {
                gpu::CommandBatch batch(device);
                score->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
                    c["positions"].setBinding(cloud.positions.rhi());
                    c["shape"].setBinding(cloud.shape.rhi());
                    c["score"].setBinding(scores->rhi());
                    c["params"]["count"].setData(n);
                    c["params"]["kind"].setData(scoreKind);
                });
                ATHENEA_TRY(batch.submit(true));
            }
            auto bins = decimator->scoreHistogram(*scores, n);
            if (!bins) return std::move(bins).error();
            const uint64_t drop = n > target ? n - target : 0;
            uint64_t sum = 0;
            float threshold = 0.0F;
            for (uint32_t k = 0; k < lod::Decimator::kScoreBins && drop > 0; ++k) {
                sum += (*bins)[k];
                if (sum >= drop) {
                    threshold = lod::Decimator::scoreOfBin(k + 1);
                    break;
                }
            }
            gpu::CommandBatch batch(device);
            keepFlag->dispatch(batch, {n, 1, 1},
                               [&](rhi::ShaderCursor c) { bindAll(c, cloud, *scores, threshold); });
            batch.markDirty();
            ATHENEA_TRY(prefix->apply(batch, flags, at, total, n));
            ATHENEA_TRY(batch.submit(true));
        }
        if (n != keepRecords) {
            return Error::make(ErrorCode::InvalidArgument,
                               "video-encode: --reduce keep wants every frame of the first's {} gaussians, not {}",
                               keepRecords, n);
        }
        uint32_t kept = 0;
        ATHENEA_TRY(total.read(device, 0, 4, &kept));
        auto out = video::PlanesCodec::allocate(device, std::max(kept, 1u), cloud.restPerColour, "video.kept");
        if (!out) return std::move(out).error();
        out->count = kept;
        out->antialiased = cloud.antialiased;
        out->linear = cloud.linear;
        gpu::CommandBatch batch(device);
        keepScatter->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) { bindAll(c, *out, flags, 0.0F); });
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
        return std::move(*out);
    }

    Result<lod::LodCloud> build(scene::GpuSplats& cloud) {
        if (!cloud.origin.valid()) {
            // A frame put together from pages has no record index of its own:
            // its splats are its records (an index list, bookkeeping).
            std::vector<uint32_t> index(cloud.count);
            for (uint32_t k = 0; k < cloud.count; ++k) index[k] = k;
            auto origin = gpu::Buffer::fromSpan(library->device(), std::span<const uint32_t>(index), "video.origin");
            if (!origin) return std::move(origin).error();
            cloud.origin = std::move(*origin);
        }
        return builder->build(cloud);
    }

    /// The store of `lod` without its least contributing splats, about `goal` left.
    uint32_t scoreKind = 0;

    Result<lod::LodCloud> pruneTo(const lod::LodCloud& lod, uint32_t goal) {
        gpu::Device& device = library->device();
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{std::max(lod.count, 1u)} * 4;
        desc.elementBytes = 4;
        desc.label = "video.score";
        auto scores = gpu::Buffer::create(device, desc);
        if (!scores) return std::move(scores).error();
        {
            gpu::CommandBatch batch(device);
            score->dispatch(batch, {lod.count, 1, 1}, [&](rhi::ShaderCursor c) {
                c["positions"].setBinding(lod.splats.positions.rhi());
                c["shape"].setBinding(lod.splats.shape.rhi());
                c["score"].setBinding(scores->rhi());
                c["params"]["count"].setData(lod.count);
                c["params"]["kind"].setData(scoreKind);
            });
            ATHENEA_TRY(batch.submit(true));
        }
        auto bins = decimator->scoreHistogram(*scores, lod.count);
        if (!bins) return std::move(bins).error();
        const uint64_t drop = lod.count > goal ? lod.count - goal : 0;
        uint64_t sum = 0;
        float threshold = 0.0F;
        for (uint32_t k = 0; k < lod::Decimator::kScoreBins && drop > 0; ++k) {
            sum += (*bins)[k];
            if (sum >= drop) {
                threshold = lod::Decimator::scoreOfBin(k + 1);
                break;
            }
        }
        return decimator->prune(lod, *scores, threshold);
    }

    static scene::GpuSplats storeOf(lod::LodCloud&& lod, const scene::GpuSplats& like) {
        scene::GpuSplats out = std::move(lod.splats);
        out.count = lod.count;
        out.antialiased = like.antialiased;
        return out;
    }

    Result<scene::GpuSplats> reduce(scene::GpuSplats cloud, uint32_t target, const Reduction& r) {
        if (target == 0 || cloud.count <= target) return cloud;
        scoreKind = r.score;
        if (r.mode == "keep") return keep(cloud, target);
        auto lod = build(cloud);
        if (!lod) return std::move(lod).error();
        if (r.mode == "prune") {
            auto kept = pruneTo(*lod, target);
            if (!kept) return std::move(kept).error();
            return storeOf(std::move(*kept), cloud);
        }
        if (r.mode == "cut") {
            // The octree's cut by error, its four tolerances scaled together
            // (bisected in their logarithm: the count falls as they grow).
            double lo = std::log(1.0 / 64.0), hi = std::log(256.0);
            std::optional<scene::GpuSplats> best;
            uint32_t bestCount = 0;
            for (int k = 0; k < 12; ++k) {
                const double mid = 0.5 * (lo + hi);
                const float m = static_cast<float>(std::exp(mid));
                lod::DecimateSettings d;
                d.colourTolerance *= m;
                d.outliers = std::min(1.0F, d.outliers * m);
                d.flatTolerance *= m;
                d.reach *= m;
                lod::DecimateStats stats;
                auto kept = decimator->decimate(*lod, cloud.origin, d, {}, &stats);
                if (!kept) return std::move(kept).error();
                const uint32_t n = kept->cloud.count;
                if (!best || (n >= target && (bestCount < target || n < bestCount)) ||
                    (n < target && bestCount < target && n > bestCount)) {
                    best = std::move(kept->cloud);
                    bestCount = n;
                }
                (n > target ? lo : hi) = mid;
            }
            best->antialiased = cloud.antialiased;
            if (bestCount <= target + target / 33) return std::move(*best);
            auto again = build(*best);
            if (!again) return std::move(again).error();
            auto kept = pruneTo(*again, target);
            if (!kept) return std::move(kept).error();
            return storeOf(std::move(*kept), cloud);
        }
        // Pairs, after a prune of half the excess for `mix`.
        const lod::LodCloud* from = &*lod;
        std::optional<lod::LodCloud> pruned;
        if (r.mode == "mix") {
            const auto excess = static_cast<double>(cloud.count - target);
            const double share = std::clamp(static_cast<double>(r.pruneShare), 0.0, 1.0);
            auto kept = pruneTo(*lod, cloud.count - static_cast<uint32_t>(excess * share));
            if (!kept) return std::move(kept).error();
            pruned = std::move(*kept);
            from = &*pruned;
        }
        lod::PairSettings settings;
        settings.ratio = static_cast<float>(from->count) / static_cast<float>(target);
        settings.surfels = false;
        settings.maxGrowth = r.growth;
        settings.opacityRule = r.opacityRule;
        settings.maxAspect = r.maxAspect;
        auto kept = decimator->decimatePairs(*from, cloud.origin, settings);
        if (!kept) return std::move(kept).error();
        kept->cloud.antialiased = cloud.antialiased;
        if (kept->cloud.count <= target + target / 33) return std::move(kept->cloud);
        // A capped merge stopped short: the rest pruned.
        auto again = build(kept->cloud);
        if (!again) return std::move(again).error();
        auto rest = pruneTo(*again, target);
        if (!rest) return std::move(rest).error();
        return storeOf(std::move(*rest), cloud);
    }
};

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

std::string timecodeOf(uint64_t n, double rate) {
    const uint64_t fps = std::max<uint64_t>(1, static_cast<uint64_t>(std::lround(rate)));
    const uint64_t f = n % fps, s = (n / fps) % 60, m = (n / fps / 60) % 60, h = n / fps / 3600;
    char out[32];
    std::snprintf(out, sizeof out, "%02llu:%02llu:%02llu:%02llu", static_cast<unsigned long long>(h),
                  static_cast<unsigned long long>(m), static_cast<unsigned long long>(s),
                  static_cast<unsigned long long>(f));
    return out;
}

/// A frame's side data on the device as a decode reads it: ranges, then the
/// high parts.
Result<gpu::Buffer> sidePage(gpu::Device& device, const std::vector<float>& ranges,
                             const std::vector<std::byte>& high, const std::vector<std::byte>& geometry = {}) {
    std::vector<std::byte> bytes(ranges.size() * sizeof(float) + high.size() + geometry.size() + 4);
    std::memcpy(bytes.data(), ranges.data(), ranges.size() * sizeof(float));
    if (!high.empty()) std::memcpy(bytes.data() + ranges.size() * sizeof(float), high.data(), high.size());
    if (!geometry.empty()) {
        std::memcpy(bytes.data() + ranges.size() * sizeof(float) + high.size(), geometry.data(), geometry.size());
    }
    gpu::BufferDesc desc;
    desc.bytes = (bytes.size() + 3) / 4 * 4;
    desc.elementBytes = 4;
    desc.label = "video.side";
    bytes.resize(desc.bytes);
    return gpu::Buffer::create(device, desc, bytes.data());
}

const char* groupOf(uint32_t a) {
    if (a < 3) return "position";
    if (a == 3) return "opacity";
    if (a < 7) return "log scale";
    if (a < 11) return "rotation";
    if (a < 14) return "base colour";
    return "rest";
}

/// Another decoder's gaussians of one frame (a CSV, a row a splat in the
/// frame's order) against this decode of it, on the device: what
/// `video-decode --dump-compare` prints.
void dumpCompare(gpu::Device& device, gpu::ShaderLibrary& library, video::PlanesCodec& codec,
                 video::AthvReader& reader, video::VideoDecoder& decoder, video::LumaTextures& textures,
                 uint32_t frame, const std::string& csv) {
    const video::AthvHeader& h = reader.header();
    if (frame >= reader.sourceFrames()) {
        cli::err("video-decode: no frame %u\n", frame);
        throw CLI::RuntimeError(kExitFailure);
    }
    scene::GpuSplats store;
    uint32_t total = 0;
    // A geometry layer of differences: every frame from the first applied,
    // the asked one decoded.
    gpu::Buffer state;
    if (h.geometry) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(uint64_t{reader.mostSourceSplats()} * 32, 32);
        desc.elementBytes = 4;
        desc.label = "video.geometryState";
        auto made = gpu::Buffer::create(device, desc);
        if (!made) cli::fail(made.error());
        state = std::move(*made);
    }
    for (uint32_t before = h.geometry ? 0 : frame; before < frame; ++before) {
        for (const uint32_t k : reader.slicesOf(before)) {
            auto f = reader.read(k);
            if (!f) cli::fail(f.error());
            auto layout = video::PlaneLayout::make(f->splats, h.bits, h.restPerColour, h.hiInVideo, h.chunkSplats,
                                                   h.columns, h.maxSplats, 0, 0, h.chroma, h.geometry);
            if (!layout) cli::fail(layout.error());
            layout->positionHighBits = h.positionHighBits;
            layout->geometryPositionBits = h.geometryPositionBits;
            layout->geometryScaleBits = h.geometryScaleBits;
            layout->geometryRotationBits = h.geometryRotationBits;
            auto side = sidePage(device, f->ranges, f->high, f->geometry);
            if (!side) cli::fail(side.error());
            gpu::CommandBatch batch(device);
            const video::GeometryFrame at{f->source, h.geometryInterval, h.geometryStagger, h.geometryOrder};
            if (auto r = codec.geometryApply(batch, *layout, side->rhi(), layout->rangeBytes() + f->high.size(), at,
                                             f->firstSplat, 0, f->splats, state);
                !r) {
                cli::fail(r.error());
            }
            if (auto r = batch.submit(true); !r) cli::fail(r.error());
        }
    }
    for (const uint32_t k : reader.slicesOf(frame)) {
        auto f = reader.read(k);
        if (!f) cli::fail(f.error());
        if (!store.positions.valid()) {
            auto s = video::PlanesCodec::allocate(device, f->sourceSplats, h.restPerColour, "video.dump");
            if (!s) cli::fail(s.error());
            store = std::move(*s);
            total = f->sourceSplats;
        }
        auto picture = decoder.decode(f->unit, f->parameterSets, f->keyframe);
        if (!picture) cli::fail(picture.error());
        auto layout = video::PlaneLayout::make(f->splats, h.bits, h.restPerColour, h.hiInVideo, h.chunkSplats,
                                               h.columns, h.maxSplats, 0, 0, h.chroma, h.geometry);
        if (!layout) cli::fail(layout.error());
        layout->positionBlock = h.positionBlock;
        layout->positionHighBits = h.positionHighBits;
            layout->geometryPositionBits = h.geometryPositionBits;
            layout->geometryScaleBits = h.geometryScaleBits;
            layout->geometryRotationBits = h.geometryRotationBits;
            layout->positionTarget = h.positionTarget;
        auto side = sidePage(device, f->ranges, f->high, f->geometry);
        if (!side) cli::fail(side.error());
        auto luma = textures.wrap(**picture);
        if (!luma) cli::fail(luma.error());
        gpu::CommandBatch batch(device);
        if (h.geometry) {
            const video::GeometryFrame at{f->source, h.geometryInterval, h.geometryStagger, h.geometryOrder};
            if (auto r = codec.geometryApply(batch, *layout, side->rhi(), layout->rangeBytes() + f->high.size(), at,
                                             f->firstSplat, 0, f->splats, state);
                !r) {
                cli::fail(r.error());
            }
        }
        if (auto r = codec.decode(batch, *layout, luma->texture.get(), luma->chroma.get(), side->rhi(), 0,
                                  layout->rangeBytes(), 0, f->splats, store, f->firstSplat, h.restPerColour,
                                  h.geometry ? &state : nullptr, f->firstSplat);
            !r) {
            cli::fail(r.error());
        }
        if (auto s = batch.submit(true); !s) cli::fail(s.error());
    }
    // The CSV: read and parsed here, ten numbers a row (the file's i and
    // anything past the base colour are not compared).
    std::ifstream in(csv);
    if (!in) {
        cli::err("video-decode: cannot read %s\n", csv.c_str());
        throw CLI::RuntimeError(kExitFailure);
    }
    constexpr uint32_t kStride = 10;
    std::vector<float> rows;
    std::string line;
    std::getline(in, line);
    uint32_t count = 0;
    while (std::getline(in, line) && count < total) {
        std::stringstream words(line);
        std::string cell;
        std::getline(words, cell, ',');   // i
        for (uint32_t c = 0; c < kStride; ++c) {
            if (!std::getline(words, cell, ',')) break;
            rows.push_back(std::strtof(cell.c_str(), nullptr));
        }
        rows.resize(size_t{count + 1} * kStride, 0.0F);
        ++count;
    }
    auto rowBuffer = gpu::Buffer::fromSpan(device, std::span<const float>(rows), "video.dumpRows");
    if (!rowBuffer) cli::fail(rowBuffer.error());
    std::vector<uint32_t> zero(14 + 256, 0);
    auto counters = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(zero), "video.dumpCounters");
    if (!counters) cli::fail(counters.error());
    auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/video_dump_compare", "videoDumpCompare");
    if (!kernel) cli::fail(kernel.error());
    const float tolerances[4] = {1.0e-3F, 0.02F, 0.1F, 0.02F};
    {
        gpu::CommandBatch batch(device);
        kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor c) {
            c["positions"].setBinding(store.positions.rhi());
            c["shape"].setBinding(store.shape.rhi());
            c["rows"].setBinding(rowBuffer->rhi());
            c["counters"].setBinding(counters->rhi());
            c["params"]["count"].setData(count);
            c["params"]["stride"].setData(kStride);
            c["params"]["positionTolerance"].setData(tolerances[0]);
            c["params"]["opacityTolerance"].setData(tolerances[1]);
            c["params"]["scaleTolerance"].setData(tolerances[2]);
            c["params"]["colourTolerance"].setData(tolerances[3]);
        });
        if (auto s = batch.submit(true); !s) cli::fail(s.error());
    }
    std::vector<uint32_t> got(zero.size());
    if (auto r = counters->read(device, 0, got.size() * 4, got.data()); !r) cli::fail(r.error());
    const auto asFloat = [](uint32_t bits) {
        float v = 0;
        std::memcpy(&v, &bits, 4);
        return v;
    };
    const char* names[4] = {"position (m, largest axis)", "opacity", "log scale (largest axis)",
                            "base colour (largest channel)"};
    const double units[4] = {1.0e-6, 1.0e-4, 1.0e-3, 1.0e-4};
    cli::out("video-decode: frame %u of %s, %u splats decoded, %u rows compared\n", frame, reader.header().chroma
             ? "a chroma .athv" : "a luma .athv", total, count);
    cli::out("| attribute | max abs diff | mean abs diff | rows past tolerance (tolerance) |\n|---|---:|---:|---:|\n");
    for (int g = 0; g < 4; ++g) {
        cli::out("| %s | %.6g | %.6g | %u (%g) |\n", names[g], static_cast<double>(asFloat(got[g * 3])),
                 got[g * 3 + 1] * units[g] / std::max(count, 1u), got[g * 3 + 2], static_cast<double>(tolerances[g]));
    }
    uint64_t seen = 0, below = 0;
    for (uint32_t b = 0; b < 256; ++b) seen += got[14 + b];
    double p99 = 0.0;
    for (uint32_t b = 0; b < 256; ++b) {
        below += got[14 + b];
        if (below >= seen * 99 / 100) {
            p99 = -16.0 + (b + 1) / 16.0;
            break;
        }
    }
    cli::out("video-decode: this decode: opacity max %.4f, mean %.4f; largest log scale p99 %.3f (bins of 1/16)\n",
             static_cast<double>(asFloat(got[12])), got[13] * 1.0e-4 / std::max(count, 1u), p99);
}

}   // namespace

void addVideo(CLI::App& app) {
    struct Options {
        std::vector<std::string> inputs;
        std::string output, hevc, range, hi = "side", order = "frame", ranges = "group", maxPicture;
        uint32_t bits = 8, keyInterval = 0, columns = 0, shDegree = 3, rendition = 0, gaussians = 0, rangeGroup = 0;
        uint32_t positionBlock = 0;
        int      positionHighBits = -1;
        float    positionTarget = 0.0F;
        std::string reduce = "pairs";
        float    mergeGrowth = 0.0F;
        uint32_t mergeOpacity = 0;
        float    pruneShare = 0.5F;
        float    mergeAspect = 1000.0F;
        std::string pruneScore = "area";
        double quality = 0.5, megabits = 0.0, rate = 0.0;
        bool check = false, intra = false, chroma = false, holdBoxes = false, geometry = false, geometryStagger = false;
        uint32_t geometryKey = 0, geometryPositionBits = 16, geometryScaleBits = 10, geometryPredict = 1;
        std::vector<uint32_t> geometryHold;
        uint32_t geometryRotationBits = 10;
        std::string chromaWeight = "mean";
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
    enc->add_option("--order", o->order, "first (the first frame's Morton order for every frame of as many splats), "
                                         "frame (each frame sorted again) or stable (first, for frames that list "
                                         "the same gaussians in the same order: index i is one gaussian throughout, "
                                         "said in the file)")
        ->check(CLI::IsMember({"first", "frame", "stable"}));
    enc->add_flag("--geometry", o->geometry,
                  "position, scale and rotation exact beside the video (16-bit positions and 10-bit log scales over "
                  "the sequence's ranges, the store's rotation word), the planes only opacity and colour");
    enc->add_option("--geometry-key", o->geometryKey,
                    "with --geometry: frames from one geometry key to the next, the frames between sent as each "
                    "gaussian's difference from the frame before (needs --order stable); 0: every frame a key");
    enc->add_option("--geometry-position-bits", o->geometryPositionBits, "with --geometry: bits a position axis")
        ->check(CLI::Range(10u, 16u));
    enc->add_option("--geometry-scale-bits", o->geometryScaleBits, "with --geometry: bits a log scale")
        ->check(CLI::Range(6u, 10u));
    enc->add_option("--geometry-rotation-bits", o->geometryRotationBits,
                    "with --geometry: bits a rotation component (of the store's smallest three)")
        ->check(CLI::Range(5u, 10u));
    enc->add_option("--geometry-hold", o->geometryHold,
                    "with --geometry-key and --geometry-predict 1: P,S,R -- between keys a position axis, a log "
                    "scale or a rotation (all three components) that moved no more than that many of its steps "
                    "from what the decoder holds keeps that value, sent as no change (conditional replenishment)")
        ->expected(3)
        ->delimiter(',')
        ->check(CLI::Range(0u, 64u));
    enc->add_option("--geometry-predict", o->geometryPredict,
                    "with --geometry-key: what a frame between keys is a residual from -- 1 the frame before, 2 the "
                    "frame before's motion carried on (2 b - c)")
        ->check(CLI::Range(1u, 2u));
    enc->add_flag("--geometry-stagger", o->geometryStagger,
                  "with --geometry-key: no whole key frames, blocks of 256 gaussians refreshed in turn");
    enc->add_flag("--hold-boxes", o->holdBoxes,
                  "with --order stable and --position-block: each block's box the union over its range group's "
                  "frames, so it does not jump from frame to frame");
    enc->add_option("--range-group", o->rangeGroup,
                    "frames whose attribute ranges are one union (default: the key interval); a whole sequence's "
                    "keeps every quantisation step put");
    enc->add_option("--ranges", o->ranges, "group (each attribute's range over a group of pictures, so what "
                                           "stays put keeps its samples) or frame (each frame's own)")
        ->check(CLI::IsMember({"group", "frame"}));
    enc->add_option("--sh-degree", o->shDegree, "harmonics degree carried, at most the source's")
        ->check(CLI::Range(0u, 3u));
    enc->add_option("--columns", o->columns, "tiles a mosaic row (default: the squarest mosaic)");
    enc->add_option("--gaussians", o->gaussians,
                    "about this many gaussians a frame: each frame merged by pairs where merging loses least "
                    "(the decimation's NanoGS merge), not dropped");
    enc->add_option("--position-block", o->positionBlock,
                    "positions over the box of each run of this many splats of the Morton order (64 .. 32768, a "
                    "power of two; 0: over the frame's range)");
    enc->add_option("--position-target", o->positionTarget,
                    "with --position-block: each block's own high bits, none where its box is no wider than this many "
                    "metres, else enough that a step is no wider than this over 255 (adaptive precision; at most "
                    "--position-high-bits, default 8)");
    enc->add_option("--position-high-bits", o->positionHighBits,
                    "a position's bits past --bits, sent exact beside the video (0 .. --bits; default --bits)");
    enc->add_flag("--chroma", o->chroma,
                  "colours as YCoCg: Y a luma plane, Co and Cg the chroma of its tile, one a 2 x 2 block of four "
                  "Morton neighbours (8 bits)");
    enc->add_option("--chroma-weight", o->chromaWeight, "a block's chroma: mean, or area (weighed by opacity x area)")
        ->check(CLI::IsMember({"mean", "area"}));
    enc->add_option("--reduce", o->reduce,
                    "with --gaussians: pairs (merged), prune (the least contributing dropped), mix (half the excess "
                    "pruned, then merged), cut (the octree's cut by error) or keep (the first frame's highest scored "
                    "records, the same ones every frame: what --order stable needs)")
        ->check(CLI::IsMember({"pairs", "prune", "mix", "cut", "keep"}));
    enc->add_option("--merge-growth", o->mergeGrowth,
                    "pairs, mix: no merge wider than this times the wider of the two (0: no limit); what it leaves "
                    "above --gaussians is pruned");
    enc->add_option("--prune-share", o->pruneShare, "mix: the share of the excess pruned before the merges")
        ->check(CLI::Range(0.0F, 1.0F));
    enc->add_option("--prune-score", o->pruneScore,
                    "prune, mix: area (opacity x the area of the two widest axes) or edge (opacity x sqrt(area) x "
                    "widest / middle: thin gaussians kept)")
        ->check(CLI::IsMember({"area", "edge"}));
    enc->add_option("--merge-aspect", o->mergeAspect,
                    "pairs, mix: no merge longer than this against its width (default 1000: no limit)");
    enc->add_option("--merge-opacity", o->mergeOpacity,
                    "pairs, mix: a merge's opacity -- 0 its members' mean, the merge widened, under 0.99; 1 their "
                    "mass over the widened area; 2 their mass, not widened; 3 their mass, not widened, up to 5")
        ->check(CLI::Range(0u, 3u));
    enc->add_option("--rendition", o->rendition, "the rendition's id, which every record carries (0..65535)")
        ->check(CLI::Range(0u, 65535u));
    enc->add_option("--max-picture", o->maxPicture,
                    "WIDTHxHEIGHT: the largest picture, for a decoder that takes less; a frame that needs more "
                    "goes as slices");
    enc->add_option("--rate", o->rate, "frames a second (default: the sequence's, or 30)");
    enc->add_option("--hevc", o->hevc, "also the video alone as an Annex B .hevc, for a player or a probe");
    enc->add_flag("--check", o->check, "decode each frame again and measure it against its source");

    enc->callback([o] {
        uint32_t mostWidth = 0, mostHeight = 0;
        if (!o->maxPicture.empty() &&
            (std::sscanf(o->maxPicture.c_str(), "%ux%u", &mostWidth, &mostHeight) != 2 || mostWidth == 0 ||
             mostHeight == 0)) {
            cli::err("video-encode: --max-picture wants WIDTHxHEIGHT\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        const uint32_t highBits = o->positionHighBits < 0 ? o->bits : static_cast<uint32_t>(o->positionHighBits);
        if (highBits > o->bits ||
            (o->positionBlock != 0 &&
             (o->positionBlock < 64 || o->positionBlock > video::PlaneLayout::kBandSplats ||
              (o->positionBlock & (o->positionBlock - 1)) != 0)) ||
            (o->hi == "video" && (o->positionBlock != 0 || highBits != o->bits))) {
            cli::err("video-encode: --position-block is 0 or a power of two from 64 to %u, --position-high-bits at "
                     "most --bits, and neither goes with --hi video\n", video::PlaneLayout::kBandSplats);
            throw CLI::RuntimeError(kExitFailure);
        }
        if (o->chroma && o->bits != 8) {
            cli::err("video-encode: --chroma is 8 bits only\n");
            throw CLI::RuntimeError(kExitFailure);
        }
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
        Reducer reducer;
        reducer.library = &library;
        if (o->gaussians != 0) {
            auto b = lod::LodBuilder::create(library);
            if (!b) cli::fail(b.error());
            auto d = lod::Decimator::create(library);
            if (!d) cli::fail(d.error());
            auto k = gpu::ComputeKernel::create(library, "athenea/scene/video_reduce", "videoReduceScore");
            if (!k) cli::fail(k.error());
            auto kf = gpu::ComputeKernel::create(library, "athenea/scene/video_keep", "videoKeepFlag");
            if (!kf) cli::fail(kf.error());
            auto ks = gpu::ComputeKernel::create(library, "athenea/scene/video_keep", "videoKeepScatter");
            if (!ks) cli::fail(ks.error());
            auto ps = gpu::PrefixSum::create(library);
            if (!ps) cli::fail(ps.error());
            reducer.keepFlag = std::move(*kf);
            reducer.keepScatter = std::move(*ks);
            reducer.prefix = std::move(*ps);
            reducer.builder = std::move(*b);
            reducer.decimator = std::move(*d);
            reducer.score = std::move(*k);
        }
        const uint32_t restAsked = o->shDegree == 0 ? 0u : (o->shDegree + 1) * (o->shDegree + 1) - 1;

        std::unique_ptr<video::AthvWriter> writer;
        std::unique_ptr<video::VideoEncoder> encoder;
        std::unique_ptr<video::VideoDecoder> decoder;
        std::unique_ptr<video::LumaTextures> textures;
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
        std::vector<double> unitBytes, sideBytes, keyBytes, interBytes, layMs, encodeMs, decodeMs, checkMs;
        std::vector<double> geometryBytes;
        std::vector<video::AttributeError> worst;
        std::vector<double> meanSum;
        uint64_t keyframes = 0, checked = 0;
        bool antialiased = false;
        uint32_t keyInterval =
            o->intra ? 1u : (o->keyInterval != 0 ? o->keyInterval : static_cast<uint32_t>(std::lround(rate)));
        const bool groupRanges = o->ranges == "group";
        const bool stable = o->order == "stable";
        if (o->geometry && (o->positionBlock != 0 || o->hi == "video" || o->positionTarget > 0.0F ||
                            (o->geometryKey > 1 && o->order != "stable"))) {
            cli::err("video-encode: --geometry takes no --position-block, --position-target or --hi video, and a "
                     "--geometry-key past 1 wants --order stable\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        if (o->positionTarget > 0.0F && o->positionBlock == 0) {
            cli::err("video-encode: --position-target wants --position-block\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        if (o->holdBoxes && (!stable || o->positionBlock == 0 || o->ranges != "group")) {
            cli::err("video-encode: --hold-boxes wants --order stable, --position-block and group ranges\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        gpu::Buffer heldBoxes;
        if (stable && o->gaussians != 0 && o->reduce != "keep") {
            cli::err("video-encode: --order stable keeps a gaussian's index only with --reduce keep\n");
            throw CLI::RuntimeError(kExitFailure);
        }
        // A frame loaded; on the first, the pictures' size, the encoder, the
        // file and the checker made.
        const auto prepare = [&](size_t k) -> scene::GpuSplats {
            bool aa = false;
            auto cloud = loadFrame(**device, *loader, (*frames)[k], o->shDegree, aa);
            if (!cloud) cli::fail(cloud.error());
            if (o->gaussians != 0) {
                cloud = reducer.reduce(std::move(*cloud), o->gaussians,
                                       Reduction{o->reduce, o->mergeGrowth, o->mergeOpacity, o->pruneShare,
                                                 o->pruneScore == "edge" ? 1u : 0u, o->mergeAspect});
                if (!cloud) cli::fail(cloud.error());
            }
            if (encoder != nullptr) return std::move(*cloud);
            antialiased = aa;
            const uint32_t rest = std::min(cloud->restPerColour, restAsked);
            auto cap = video::PlaneLayout::sliceSplats(cloud->count, o->bits, rest, o->hi == "video",
                                                       video::PlaneLayout::kBandSplats, o->columns, mostWidth,
                                                       mostHeight, o->chroma, o->geometry);
            if (!cap) cli::fail(cap.error());
            sliceCap = *cap;
            if (mostWidth != 0) {
                // Held to a size, the pictures are sized in whole chunks for
                // the first frame and a twentieth more, so a later frame of a
                // few more splats (a decimation's count varies) takes no
                // extra picture.
                auto most = video::PlaneLayout::sliceSplats(uint32_t{1} << 30, o->bits, rest, o->hi == "video",
                                                            video::PlaneLayout::kBandSplats, o->columns, mostWidth,
                                                            mostHeight, o->chroma, o->geometry);
                if (!most) cli::fail(most.error());
                const uint32_t band = video::PlaneLayout::kBandSplats;
                const uint64_t roomy = (uint64_t{cloud->count} * 21 / 20 + band - 1) / band * band;
                sliceCap = static_cast<uint32_t>(std::min<uint64_t>(*most, roomy));
            }
            auto made = video::PlaneLayout::make(sliceCap, o->bits, rest, o->hi == "video",
                                                 video::PlaneLayout::kBandSplats, o->columns, 0, mostWidth,
                                                 mostHeight, o->chroma, o->geometry);
            if (!made) cli::fail(made.error());
            layout = *made;
            layout.positionBlock = o->positionBlock;
            layout.positionHighBits = o->positionTarget > 0.0F && o->positionHighBits < 0 ? o->bits : highBits;
            layout.positionTarget = o->positionTarget;
            if (o->geometry) {
                layout.positionHighBits = 0;   // nothing beside the video but the layer
                layout.geometryPositionBits = o->geometryPositionBits;
                layout.geometryScaleBits = o->geometryScaleBits;
                layout.geometryRotationBits = o->geometryRotationBits;
            }
            layout.chromaWeight = o->chromaWeight == "area" ? 1u : 0u;
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
            // The rate as a fraction: whole, or NTSC's thousand-and-first.
            if (std::abs(rate - std::round(rate)) < 1.0e-6) {
                header.rateNumerator = static_cast<uint32_t>(std::lround(rate));
                header.rateDenominator = 1;
            } else if (std::abs(rate * 1.001 - std::round(rate * 1.001)) < 1.0e-3) {
                header.rateNumerator = static_cast<uint32_t>(std::lround(rate * 1.001)) * 1000;
                header.rateDenominator = 1001;
            } else {
                header.rateNumerator = static_cast<uint32_t>(std::lround(rate * 1000.0));
                header.rateDenominator = 1000;
            }
            header.rendition = o->rendition;
            header.quality = o->megabits > 0.0 ? 0.0F : static_cast<float>(o->quality);
            header.positionBlock = layout.positionBlock;
            header.positionHighBits = layout.positionHighBits;
            header.positionTarget = layout.positionTarget;
            header.geometry = layout.geometry;
            header.geometryInterval = o->geometryKey;
            header.geometryStagger = o->geometryStagger;
            header.geometryPositionBits = o->geometryPositionBits;
            header.geometryScaleBits = o->geometryScaleBits;
            header.geometryRotationBits = o->geometryRotationBits;
            header.geometryOrder = o->geometryPredict;
            header.chroma = layout.chroma;
            header.stableOrder = stable;
            auto w = video::AthvWriter::create(o->output, header);
            if (!w) cli::fail(w.error());
            writer = std::move(*w);
            if (o->check) {
                auto d = video::createHardwareDecoder(header.codec, layout.bits);
                if (!d) cli::fail(d.error());
                decoder = std::move(*d);
                auto t = video::createLumaTextures(**device);
                if (!t) cli::fail(t.error());
                textures = std::move(*t);
            }
            cli::out("video-encode: %zu frames of %u splats, %u planes of %u x %u in a %u x %u mosaic, %u slice%s a "
                     "frame, %u bits, high parts %s, encoder %s\n",
                     last - first + 1, cloud->count, layout.planes(), video::PlaneLayout::kTileWidth,
                     layout.tileHeight, layout.width, layout.height, (cloud->count + sliceCap - 1) / sliceCap,
                     sliceCap < cloud->count ? "s" : "", layout.bits, layout.hiInVideo ? "in the video" : "beside it",
                     encoder->hardware() ? "hardware" : "software");
            return std::move(*cloud);
        };
        const uint32_t groupFrames = o->rangeGroup != 0 ? o->rangeGroup : keyInterval;
        // THE GEOMETRY LAYER: each frame's quantised geometry and the frame
        // before's, frame-wide (a stable order's index), and the check's
        // decoded state. Its quantisation is over ranges that must not move
        // between keys: the whole sequence's.
        gpu::Buffer geomNow, geomBefore, geomBefore2, geomState;
        const bool deltas = o->geometry && o->geometryKey != 1;
        if (o->geometry && o->geometryKey != 1 && groupFrames < last - first + 1) {
            cli::err("video-encode: a geometry layer sent as differences wants --range-group %zu (the sequence's)\n",
                     last - first + 1);
            throw CLI::RuntimeError(kExitFailure);
        }
        const auto frameBuffer = [&](gpu::Buffer& b, uint32_t splats, const char* label, uint32_t frames = 1) {
            if (b.valid() && b.bytes() >= uint64_t{splats} * 16 * frames) return;
            gpu::BufferDesc desc;
            desc.bytes = uint64_t{std::max(splats, 1u)} * 16 * frames;
            desc.elementBytes = 4;
            desc.label = label;
            std::vector<std::byte> zeros(desc.bytes);
            auto made = gpu::Buffer::create(**device, desc, zeros.data());
            if (!made) cli::fail(made.error());
            b = std::move(*made);
        };
        for (size_t g = first; g <= last; g += groupFrames) {
            const size_t groupEnd = std::min(last, g + groupFrames - 1);
            if (groupRanges) {
                // The group's ranges first, over every frame of it.
                for (size_t k = g; k <= groupEnd; ++k) {
                    scene::GpuSplats cloud = prepare(k);
                    if (auto r = codec->ranges(cloud, layout, cloud.count, frameRanges); !r) cli::fail(r.error());
                    if (auto r = codec->widen(layout, frameRanges, ranges, k == g); !r) cli::fail(r.error());
                    if (o->holdBoxes) {
                        // The stable order, from the first frame, and every
                        // frame's boxes in it, their union held.
                        if (orderedFor == 0) {
                            if (auto r = codec->order(cloud, cloud.count, frameRanges, order); !r) cli::fail(r.error());
                            orderedFor = cloud.count;
                        }
                        video::PlaneLayout whole = layout;
                        whole.splats = cloud.count;
                        if (auto r = codec->holdBoxes(cloud, whole, order, heldBoxes, k == g); !r) cli::fail(r.error());
                    }
                }
            }
            if (deltas && o->geometryStagger && g == first) {
                // Staggered refreshes: the first frame's blocks that are not
                // refreshed are differences from the last frame, which the
                // loop shows before it.
                scene::GpuSplats head = prepare(first);
                if (orderedFor == 0) {
                    if (auto r = codec->order(head, head.count, ranges, order); !r) cli::fail(r.error());
                    orderedFor = head.count;
                }
                if (o->geometryHold.size() == 3) {
                    // Held values depend on every frame since a block's
                    // key: the loop's last `interval` frames, encoded as
                    // the stream does, leave in geomBefore what the decoder
                    // holds when the loop comes round (each block keyed once
                    // in them).
                    const size_t from = last + 1 - std::min<size_t>(o->geometryKey, last + 1 - first);
                    for (size_t back = from; back <= last; ++back) {
                        scene::GpuSplats tail = prepare(back);
                        frameBuffer(geomNow, tail.count, "video.geometryNow");
                        frameBuffer(geomBefore, tail.count, "video.geometryBefore");
                        frameBuffer(geomBefore2, tail.count, "video.geometryBefore2");
                        video::PlaneLayout whole = layout;
                        whole.splats = tail.count;
                        video::GeometryFrame at{static_cast<uint32_t>(back - first), o->geometryKey, true, 1};
                        at.holdPosition = o->geometryHold[0];
                        at.holdScale = o->geometryHold[1];
                        at.holdRotation = o->geometryHold[2];
                        auto held = codec->geometryEncode(tail, whole, order, 0, ranges, at, geomNow, geomBefore, true);
                        if (!held) cli::fail(held.error());
                        std::swap(geomBefore2, geomBefore);
                        std::swap(geomBefore, geomNow);
                    }
                }
                // The loop's last two frames, what frame 0's first blocks
                // are predicted from.
                for (const size_t back : {last - 1, last}) {
                    if (o->geometryHold.size() == 3) break;
                    if (back < first) continue;
                    scene::GpuSplats tail = prepare(back);
                    gpu::Buffer& into = back == last ? geomBefore : geomBefore2;
                    frameBuffer(into, tail.count, "video.geometryBefore");
                    video::PlaneLayout whole = layout;
                    whole.splats = tail.count;
                    const video::GeometryFrame at{static_cast<uint32_t>(back - first), o->geometryKey, true};
                    auto quantised = codec->geometryEncode(tail, whole, order, 0, ranges, at, into, into, true);
                    if (!quantised) cli::fail(quantised.error());
                }
            }
            for (size_t k = g; k <= groupEnd; ++k) {
                const InputFrame& in = (*frames)[k];
                scene::GpuSplats cloud = prepare(k);
                if (o->geometry) {
                    frameBuffer(geomNow, cloud.count, "video.geometryNow");
                    frameBuffer(geomBefore, cloud.count, "video.geometryBefore");
                    frameBuffer(geomBefore2, cloud.count, "video.geometryBefore2");
                    frameBuffer(geomState, cloud.count, "video.geometryState", 2);
                }
                video::GeometryFrame geomAt{static_cast<uint32_t>(k - first), o->geometryKey, o->geometryStagger,
                                            o->geometryPredict};
                if (o->geometryHold.size() == 3) {
                    geomAt.holdPosition = o->geometryHold[0];
                    geomAt.holdScale = o->geometryHold[1];
                    geomAt.holdRotation = o->geometryHold[2];
                }
                if (!groupRanges) {
                    if (auto r = codec->ranges(cloud, layout, cloud.count, ranges); !r) cli::fail(r.error());
                }
                if (stable && orderedFor != 0 && cloud.count != orderedFor) {
                    cli::err("video-encode: --order stable: frame %zu has %u gaussians, the first %u\n", k, cloud.count,
                             orderedFor);
                    throw CLI::RuntimeError(kExitFailure);
                }
                if (o->order == "frame" || orderedFor != cloud.count) {
                    if (auto r = codec->order(cloud, cloud.count, ranges, order); !r) cli::fail(r.error());
                    orderedFor = cloud.count;
                }
                for (uint32_t at = 0; at < cloud.count; at += sliceCap) {
                    const uint32_t count = std::min(sliceCap, cloud.count - at);
                    auto sliced = video::PlaneLayout::make(count, layout.bits, layout.restPerColour, layout.hiInVideo,
                                                           layout.chunkSplats, layout.columns, sliceCap, 0, 0,
                                                           layout.chroma, layout.geometry);
                    if (!sliced) cli::fail(sliced.error());
                    sliced->positionBlock = layout.positionBlock;
                    sliced->positionHighBits = layout.positionHighBits;
                    sliced->positionTarget = layout.positionTarget;
                    sliced->geometryPositionBits = layout.geometryPositionBits;
                    sliced->geometryScaleBits = layout.geometryScaleBits;
                    sliced->geometryRotationBits = layout.geometryRotationBits;
                    sliced->chromaWeight = layout.chromaWeight;
                    if (sliced->width != layout.width || sliced->height != layout.height) {
                        cli::err("video-encode: frame %zu needs a %u x %u mosaic, not the stream's %u x %u\n", k,
                                 sliced->width, sliced->height, layout.width, layout.height);
                        throw CLI::RuntimeError(kExitFailure);
                    }
                    Clock::time_point t = Clock::now();
                    auto planes =
                        codec->encode(cloud, *sliced, order, at, ranges, o->holdBoxes ? &heldBoxes : nullptr);
                    if (!planes) cli::fail(planes.error());
                    layMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    t = Clock::now();
                    auto units = encoder->encode(planes->luma, planes->chroma, picture);
                    if (!units) cli::fail(units.error());
                    encodeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    if (units->size() != 1) {
                        cli::err("video-encode: the encoder gave %zu units for picture %u\n", units->size(), picture);
                        throw CLI::RuntimeError(kExitFailure);
                    }
                    video::EncodedUnit& unit = units->front();
                    video::AthvFrame f;
                    f.index = picture++;
                    f.frame = static_cast<uint32_t>(k - first);
                    f.slice = at / sliceCap;
                    f.slices = (cloud.count + sliceCap - 1) / sliceCap;
                    f.source = static_cast<uint32_t>(k - first);
                    f.firstSplat = at;
                    f.sourceSplats = cloud.count;
                    f.splats = count;
                    f.keyframe = unit.keyframe;
                    f.time = in.time;
                    f.timecode = timecodeOf(k - first, rate);
                    f.ranges = planes->ranges;
                    f.high = planes->high;
                    if (o->geometry) {
                        auto g = codec->geometryEncode(cloud, *sliced, order, at, ranges, geomAt, geomNow, geomBefore,
                                                       false, &geomBefore2);
                        if (!g) cli::fail(g.error());
                        f.geometry = std::move(*g);
                        f.geometryKey =
                            geomAt.interval <= 1 || (!geomAt.stagger && geomAt.frame % geomAt.interval == 0);
                        geometryBytes.push_back(0.0);
                    }
                    f.parameterSets = unit.parameterSets;
                    f.unit = std::move(unit.unit);
                    if (auto w = writer->write(f); !w) cli::fail(w.error());
                    if (o->geometry) geometryBytes.back() = static_cast<double>(f.geometryStored);
                    keyframes += f.keyframe ? 1 : 0;
                    unitBytes.push_back(static_cast<double>(f.unit.size()));
                    (f.keyframe ? keyBytes : interBytes).push_back(static_cast<double>(f.unit.size()));
                    sideBytes.push_back(static_cast<double>(f.recordBytes - f.unit.size()));
                    if (hevc.is_open()) {
                        const std::vector<uint8_t> b = video::annexB(f);
                        hevc.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
                    }
                    if (!o->check) continue;
                    // The picture back through the hardware decoder and the
                    // planes decode, measured against its source on the device.
                    t = Clock::now();
                    auto decoded = decoder->decode(f.unit, f.parameterSets, f.keyframe);
                    if (!decoded) cli::fail(decoded.error());
                    decodeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    t = Clock::now();
                    auto luma = textures->wrap(**decoded);
                    if (!luma) cli::fail(luma.error());
                    auto side = sidePage(**device, f.ranges, f.high, f.geometry);
                    if (!side) cli::fail(side.error());
                    auto back = video::PlanesCodec::allocate(**device, count, layout.restPerColour, "video.check");
                    if (!back) cli::fail(back.error());
                    {
                        gpu::CommandBatch batch(**device);
                        if (o->geometry) {
                            if (auto r = codec->geometryApply(batch, *sliced, side->rhi(),
                                                              sliced->rangeBytes() + f.high.size(), geomAt, at, 0,
                                                              count, geomState);
                                !r) {
                                cli::fail(r.error());
                            }
                        }
                        if (auto r = codec->decode(batch, *sliced, luma->texture.get(), luma->chroma.get(),
                                                   side->rhi(), 0, sliced->rangeBytes(), 0, count, *back, 0,
                                                   layout.restPerColour, o->geometry ? &geomState : nullptr, at);
                            !r) {
                            cli::fail(r.error());
                        }
                        if (auto r = batch.submit(true); !r) cli::fail(r.error());
                    }
                    checkMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
                    auto errors = codec->compare(*sliced, *back, cloud, order, at, ranges, *side,
                                                 sliced->rangeBytes());
                    if (!errors) cli::fail(errors.error());
                    if (worst.empty()) {
                        worst.resize(errors->size());
                        meanSum.assign(errors->size(), 0.0);
                    }
                    for (size_t a = 0; a < errors->size(); ++a) {
                        worst[a].maxSteps = std::max(worst[a].maxSteps, (*errors)[a].maxSteps);
                        worst[a].far += (*errors)[a].far;
                        meanSum[a] += (*errors)[a].meanSteps;
                    }
                    ++checked;
                }
                if (o->geometry) {
                    // The frames move down a place: now -> before -> before2.
                    std::swap(geomBefore2, geomBefore);
                    std::swap(geomBefore, geomNow);
                }
            }
        }
        if (auto fin = writer->finish(0, static_cast<uint32_t>(last - first + 1)); !fin) cli::fail(fin.error());
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
        if (!geometryBytes.empty()) {
            double total = 0.0;
            for (double b : geometryBytes) total += b;
            cli::out("video-encode: the geometry layer %.4f MB a frame (deflated, in the side figure), a picture's "
                     "median %.4f MB, the largest %.4f MB\n", total / n / 1.0e6, median(geometryBytes) / 1.0e6,
                     *std::max_element(geometryBytes.begin(), geometryBytes.end()) / 1.0e6);
        }
        cli::out("video-encode: a key picture's video %.3f MB, a predicted one's %.3f MB (medians)\n",
                 median(keyBytes) / 1.0e6, median(interBytes) / 1.0e6);
        cli::out("video-encode: a picture laid into planes in %.1f ms (median, read back included), encoded in %.1f "
                 "ms\n",
                 median(layMs), median(encodeMs));
        if (checked != 0) {
            cli::out("video-encode: --check over %llu pictures: hardware decode %.2f ms a picture (median, p95 %.2f), "
                     "planes -> gaussians %.2f ms (median, submitted and waited for)\n",
                     static_cast<unsigned long long>(checked), median(decodeMs), percentile(decodeMs, 0.95),
                     median(checkMs));
            // Grouped: the worst attribute of each group, in its own steps.
            std::map<std::string, std::pair<double, double>> groups;   // max, mean
            uint64_t far = 0;
            for (size_t a = 0; a < worst.size(); ++a) {
                auto& g = groups[groupOf(static_cast<uint32_t>(a))];
                g.first = std::max(g.first, worst[a].maxSteps);
                g.second = std::max(g.second, meanSum[a] / static_cast<double>(checked));
                if (a < 3) far += worst[a].far;
            }
            for (const char* name : {"position", "opacity", "log scale", "rotation", "base colour", "rest"}) {
                const auto it = groups.find(name);
                if (it == groups.end()) continue;
                cli::out("video-encode: %-12s max %.2f steps, mean %.3f%s\n", name, it->second.first,
                         it->second.second,
                         std::string(name) == "position" ? " (steps of the 2 x bits position)" : "");
            }
            cli::out("video-encode: position high parts %s: %llu splat axes whose high part is not the source's\n",
                     far == 0 ? "exact" : "NOT exact", static_cast<unsigned long long>(far));
        }
    });

    struct FlickerOptions {
        std::string video, source;
        uint32_t first = 1, last = 61;
        std::vector<uint32_t> rect;
    };
    auto fl = std::make_shared<FlickerOptions>();
    auto* flick = app.add_subcommand("video-flicker",
                                     "how much a decode's error changes from frame to frame: per pixel, |(video_k - "
                                     "source_k) - (video_k-1 - source_k-1)| in sRGB codes, mean and p99 over the "
                                     "frames, from two directories of frame_NNNNN.exr (athenea stream -o)");
    flick->add_option("video", fl->video, "frames drawn from the video")->required();
    flick->add_option("source", fl->source, "the same frames drawn from the source")->required();
    flick->add_option("--first", fl->first, "the first frame number");
    flick->add_option("--last", fl->last, "the last frame number");
    flick->add_option("--rect", fl->rect, "X0 Y0 X1 Y1: only these pixels (top row 0, x1 and y1 past the last)")
        ->expected(4);
    flick->callback([fl] {
        auto device = gpu::Device::create();
        if (!device) cli::fail(device.error());
        gpu::ShaderLibrary library(*device);
        auto kernel = gpu::ComputeKernel::create(library, "athenea/scene/video_flicker", "videoFlicker");
        if (!kernel) cli::fail(kernel.error());
        uint32_t width = 0, height = 0;
        const auto load = [&](const std::string& dir, uint32_t k) {
            char name[32];
            std::snprintf(name, sizeof name, "frame_%05u.exr", k);
            auto px = io::readExr(fs::path(dir) / name);
            if (!px) cli::fail(px.error());
            auto b = gpu::Buffer::fromSpan(**device, std::span<const float>(px->rgba), "video.flicker.frame");
            if (!b) cli::fail(b.error());
            width = px->width;
            height = px->height;
            return std::make_pair(std::move(*b), uint64_t{px->width} * px->height);
        };
        std::vector<uint32_t> zero(257, 0);
        auto counters = gpu::Buffer::fromSpan(**device, std::span<const uint32_t>(zero), "video.flicker.counters");
        if (!counters) cli::fail(counters.error());
        std::vector<uint64_t> bins(256, 0);
        uint64_t sum = 0, pixels = 0;
        auto beforeV = load(fl->video, fl->first);
        auto beforeS = load(fl->source, fl->first);
        for (uint32_t k = fl->first + 1; k <= fl->last; ++k) {
            auto nowV = load(fl->video, k);
            auto nowS = load(fl->source, k);
            const uint32_t all = static_cast<uint32_t>(nowV.second);
            // The rect in the engine's rows (bottom first); the whole image without one.
            uint32_t x0 = 0, y0 = 0, x1 = width, y1 = height;
            if (fl->rect.size() == 4) {
                x0 = std::min(fl->rect[0], width);
                x1 = std::min(std::max(fl->rect[2], x0), width);
                y1 = height - std::min(fl->rect[1], height);
                y0 = height - std::min(std::max(fl->rect[3], fl->rect[1]), height);
            }
            const uint32_t n = (x1 - x0) * (y1 - y0);
            if (nowS.second != all || beforeV.second != all) {
                cli::err("video-flicker: frame %u is not the others' size\n", k);
                throw CLI::RuntimeError(kExitFailure);
            }
            gpu::CommandBatch batch(**device);
            batch.encoder()->clearBuffer(counters->rhi(), 0, 257 * 4);
            batch.markDirty();
            kernel->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
                c["videoNow"].setBinding(nowV.first.rhi());
                c["sourceNow"].setBinding(nowS.first.rhi());
                c["videoBefore"].setBinding(beforeV.first.rhi());
                c["sourceBefore"].setBinding(beforeS.first.rhi());
                c["counters"].setBinding(counters->rhi());
                c["params"]["count"].setData(n);
                c["params"]["width"].setData(width);
                c["params"]["x0"].setData(x0);
                c["params"]["y0"].setData(y0);
                c["params"]["rectWidth"].setData(std::max(x1 - x0, 1u));
            });
            if (auto r = batch.submit(true); !r) cli::fail(r.error());
            std::vector<uint32_t> got(257);
            if (auto r = counters->read(**device, 0, 257 * 4, got.data()); !r) cli::fail(r.error());
            for (uint32_t b = 0; b < 256; ++b) bins[b] += got[b];
            sum += got[256];
            pixels += n;
            beforeV = std::move(nowV);
            beforeS = std::move(nowS);
        }
        uint64_t below = 0;
        double p99 = 0.0, p999 = 0.0;
        for (uint32_t b = 0; b < 256; ++b) {
            below += bins[b];
            if (p99 == 0.0 && below * 100 >= pixels * 99) p99 = b + 1;
            if (p999 == 0.0 && below * 1000 >= pixels * 999) p999 = b + 1;
        }
        cli::out("video-flicker: frames %u-%u, %llu pixel pairs: mean %.4f codes, p99 %.2f, p99.9 %.2f (sRGB codes, "
                 "the largest channel of the error's change)\n", fl->first, fl->last,
                 static_cast<unsigned long long>(pixels),
                 static_cast<double>(sum) / 16.0 / static_cast<double>(std::max<uint64_t>(pixels, 1)),
                 p99, p999);
    });

    struct DecodeOptions {
        std::string input, dump;
        uint32_t frames = 0, frame = 0;
        bool histogram = false;
    };
    auto d = std::make_shared<DecodeOptions>();
    auto* dec = app.add_subcommand("video-decode",
                                   "every frame of a .athv through the hardware decoder and the planes -> gaussians "
                                   "kernel, timed");
    dec->add_option("input", d->input, "the .athv")->required();
    dec->add_option("--frames", d->frames, "at most this many pictures (default: all)");
    dec->add_option("--dump-compare", d->dump,
                    "a CSV of another decoder's gaussians of --frame (i,x,y,z,opacity,lnsx,lnsy,lnsz,r,g,b,...): "
                    "compared, splat by splat, with this decode");
    dec->add_option("--frame", d->frame, "with --dump-compare: the source frame, from 0");
    dec->add_flag("--luma-histogram", d->histogram,
                  "each picture's plane 0 as the decoder left it: its lowest and highest sample and how many are 128 "
                  "(a full-range picture spans 0-255; one squeezed to video range, 16-235)");
    dec->callback([d] {
        auto reader = video::AthvReader::open(d->input);
        if (!reader) cli::fail(reader.error());
        const video::AthvHeader& h = (*reader)->header();
        auto device = gpu::Device::create();
        if (!device) cli::fail(device.error());
        gpu::ShaderLibrary library(*device);
        auto codec = video::PlanesCodec::create(library);
        if (!codec) cli::fail(codec.error());
        auto decoder = video::createHardwareDecoder(h.codec, h.bits);
        if (!decoder) cli::fail(decoder.error());
        auto textures = video::createLumaTextures(**device);
        if (!textures) cli::fail(textures.error());
        if (!d->dump.empty()) {
            dumpCompare(**device, library, *codec, **reader, **decoder, **textures, d->frame, d->dump);
            return;
        }
        const uint32_t count = d->frames != 0 ? std::min(d->frames, (*reader)->frames()) : (*reader)->frames();
        std::vector<double> readMs, decodeMs, kernelMs;
        std::optional<gpu::ComputeKernel> histogramKernel;
        gpu::Buffer geometryState;
        gpu::Buffer histogramBuffer;
        if (d->histogram) {
            auto k = gpu::ComputeKernel::create(library, "athenea/scene/video_dump_compare", "videoLumaHistogram");
            if (!k) cli::fail(k.error());
            histogramKernel = std::move(*k);
            std::vector<uint32_t> zero(256, 0);
            auto b = gpu::Buffer::fromSpan(**device, std::span<const uint32_t>(zero), "video.histogram");
            if (!b) cli::fail(b.error());
            histogramBuffer = std::move(*b);
        }
        scene::GpuSplats store;
        for (uint32_t k = 0; k < count; ++k) {
            Clock::time_point t = Clock::now();
            auto f = (*reader)->read(k);
            if (!f) cli::fail(f.error());
            readMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
            t = Clock::now();
            auto frame = (*decoder)->decode(f->unit, f->parameterSets, f->keyframe);
            if (!frame) cli::fail(frame.error());
            decodeMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
            auto layout = video::PlaneLayout::make(f->splats, h.bits, h.restPerColour, h.hiInVideo, h.chunkSplats,
                                                   h.columns, h.maxSplats, 0, 0, h.chroma, h.geometry);
            if (!layout) cli::fail(layout.error());
            layout->positionBlock = h.positionBlock;
            layout->positionHighBits = h.positionHighBits;
            layout->geometryPositionBits = h.geometryPositionBits;
            layout->geometryScaleBits = h.geometryScaleBits;
            layout->geometryRotationBits = h.geometryRotationBits;
            layout->positionTarget = h.positionTarget;
            if (store.count < f->splats) {
                auto s = video::PlanesCodec::allocate(**device, f->splats, h.restPerColour, "video.decoded");
                if (!s) cli::fail(s.error());
                store = std::move(*s);
            }
            auto side = sidePage(**device, f->ranges, f->high, f->geometry);
            if (!side) cli::fail(side.error());
            t = Clock::now();
            auto luma = (*textures)->wrap(**frame);
            if (!luma) cli::fail(luma.error());
            gpu::CommandBatch batch(**device);
            if (h.geometry) {
                if (!geometryState.valid()) {
                    gpu::BufferDesc desc;
                    desc.bytes = std::max<uint64_t>(uint64_t{(*reader)->mostSourceSplats()} * 32, 32);
                    desc.elementBytes = 4;
                    desc.label = "video.geometryState";
                    auto made = gpu::Buffer::create(**device, desc);
                    if (!made) cli::fail(made.error());
                    geometryState = std::move(*made);
                }
                const video::GeometryFrame at{f->source, h.geometryInterval, h.geometryStagger, h.geometryOrder};
                if (auto r = codec->geometryApply(batch, *layout, side->rhi(), layout->rangeBytes() + f->high.size(),
                                                  at, f->firstSplat, 0, f->splats, geometryState);
                    !r) {
                    cli::fail(r.error());
                }
            }
            if (auto r = codec->decode(batch, *layout, luma->texture.get(), luma->chroma.get(), side->rhi(), 0,
                                       layout->rangeBytes(), 0, f->splats, store, 0, h.restPerColour,
                                       h.geometry ? &geometryState : nullptr, f->firstSplat);
                !r) {
                cli::fail(r.error());
            }
            if (auto s = batch.submit(true); !s) cli::fail(s.error());
            kernelMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t).count());
            if (histogramKernel) {
                const uint32_t w = (*frame)->width(), hgt = (*frame)->height();
                gpu::CommandBatch hb(**device);
                hb.encoder()->clearBuffer(histogramBuffer.rhi(), 0, 256 * 4);
                hb.markDirty();
                histogramKernel->dispatch(hb, {w * hgt, 1, 1}, [&](rhi::ShaderCursor c) {
                    c["lumaPlane"].setBinding(luma->texture.get());
                    c["histogram"].setBinding(histogramBuffer.rhi());
                    c["lumaParams"]["count"].setData(w);
                    c["lumaParams"]["stride"].setData(hgt);
                });
                if (auto s = hb.submit(true); !s) cli::fail(s.error());
                std::vector<uint32_t> bins(256);
                if (auto r = histogramBuffer.read(**device, 0, 1024, bins.data()); !r) cli::fail(r.error());
                uint32_t lo = 255, hi = 0;
                uint64_t total = 0;
                for (uint32_t b = 0; b < 256; ++b) {
                    if (bins[b] == 0) continue;
                    lo = std::min(lo, b);
                    hi = std::max(hi, b);
                    total += bins[b];
                }
                const auto share = [&](uint32_t from, uint32_t to) {
                    uint64_t n = 0;
                    for (uint32_t b = from; b < to; ++b) n += bins[b];
                    return 100.0 * static_cast<double>(n) / static_cast<double>(std::max<uint64_t>(total, 1));
                };
                cli::out("video-decode: picture %u (%u x %u): plane 0 from %u to %u, %.2f%% at 128, %.2f%% under 16, "
                         "%.2f%% over 235\n", k, w, hgt, lo, hi, share(128, 129), share(0, 16), share(236, 256));
            }
        }
        cli::out("video-decode: %u pictures of %s (%u frames, %.2f MB a picture): read %.2f ms, hardware decode %.2f "
                 "ms (p95 %.2f, %s), planes -> gaussians %.2f ms (p95 %.2f) a picture, medians\n",
                 count, d->input.c_str(), (*reader)->sourceFrames(),
                 static_cast<double>((*reader)->bytes()) / (*reader)->frames() / 1.0e6,
                 median(readMs), median(decodeMs), percentile(decodeMs, 0.95),
                 (*decoder)->hardware() ? "hardware" : "software", median(kernelMs), percentile(kernelMs, 0.95));
    });
}

}   // namespace athenea::cli
