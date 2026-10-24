// Copyright (c) 2026 jesus luque.
//
// LIVE STREAMS UNDER WHAT PRODUCERS GET WRONG, OR DO TOO FAST. Pages
// repeated, out of order, out of range, empty, broken; epochs faster than the
// frames; many producers on many streams; pools opened and closed with pages in
// flight. Whatever comes, the source and the pool must not crash or hang, and a
// frame must never show half an epoch.
//
// The cases that need no device drive a LiveSource alone (its poll and fetch
// make no device call) and read its changes, which are bookkeeping. The GPU
// cases drive a StreamingPool and compare every frame on the device with the
// epoch status() names, drawn straight from its SPZ pages.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <thread>
#include <vector>

#include "load-spz.h"
#include "athenea/io/Readers.h"
#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;

namespace {

// --- a source alone ------------------------------------------------------------

/// A stream of .athc chunk blocks, four splats a chunk at most, one SH word.
lod::LiveSettings blockStream(uint32_t chunks, uint32_t stream = 0) {
    lod::LiveSettings s;
    s.stream = stream;
    s.chunks = chunks;
    s.format.chunkSplats = 4;
    s.format.shWords = 1;
    s.name = "qa-stream-" + std::to_string(stream);
    return s;
}

lod::PageUpdate blockPage(const lod::LiveSettings& s, uint64_t epoch, uint32_t chunk, uint32_t pages,
                          uint32_t splats = 2) {
    lod::PageUpdate u;
    u.stream = s.stream;
    u.epoch = epoch;
    u.chunk = chunk;
    u.pages = pages;
    u.op = lod::PageUpdate::Op::Replace;
    u.bytes.assign(splats * s.format.elementBytes(), std::byte{0});
    return u;
}

/// What one poll hands over, by epoch.
std::map<uint64_t, std::vector<lod::ChunkChange>> pollEpochs(lod::LiveSource& source) {
    std::vector<lod::ChunkChange> changes;
    source.poll(changes);
    std::map<uint64_t, std::vector<lod::ChunkChange>> by;
    for (const lod::ChunkChange& c : changes) {
        by[c.epoch].push_back(c);
    }
    return by;
}

/// An SPZ file of `count` random gaussians (degree `degree`), as bytes.
std::vector<uint8_t> spzBytes(uint32_t count, uint64_t seed, bool antialiased = true, int version = 4,
                              int degree = 0, float minSize = 0.02F, float maxSize = 0.3F) {
    test::CloudBuilder built = test::randomCloud(count, seed, minSize, maxSize);
    const auto& r = built.raw;
    const uint32_t stride = r.encoding.floatsPerRecord;
    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(r.count);
    cloud.shDegree = degree;
    cloud.antialiased = antialiased;
    const uint32_t shDim = degree == 0 ? 0 : degree == 1 ? 3 : degree == 2 ? 8 : 15;
    for (uint32_t i = 0; i < r.count; ++i) {
        const float* f = r.records.data() + size_t{i} * stride;
        cloud.positions.insert(cloud.positions.end(), {f[0], f[1], f[2]});
        cloud.alphas.push_back(f[3]);
        cloud.scales.insert(cloud.scales.end(), {f[4], f[5], f[6]});
        cloud.rotations.insert(cloud.rotations.end(), {f[8], f[9], f[10], f[7]});
        cloud.colors.insert(cloud.colors.end(), {f[11], f[12], f[13]});
        for (uint32_t k = 0; k < shDim * 3; ++k) {
            cloud.sh.push_back(0.1F * static_cast<float>((i + k) % 7) - 0.3F);
        }
    }
    spz::PackOptions options;
    options.version = version;
    options.from = spz::CoordinateSystem::RDF;
    std::vector<uint8_t> bytes;
    REQUIRE(spz::saveSpz(cloud, options, &bytes));
    return bytes;
}

std::vector<std::byte> asBytes(const std::vector<uint8_t>& v) {
    std::vector<std::byte> b(v.size());
    if (!v.empty()) std::memcpy(b.data(), v.data(), v.size());
    return b;
}

struct Lcg64 {
    uint64_t state = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return state >> 17;
    }
};

}   // namespace

TEST_CASE("qa: a page sent twice does not stand in for a page that never came", "[lod][stream][live][qa]") {
    // The epoch has three pages, one per chunk. Chunk 0's page comes twice
    // (a producer that resends; the later replaces the earlier) and chunk 2's
    // not yet: the epoch is not whole, and must not be handed over.
    auto queue = std::make_shared<lod::PageQueue>();
    const lod::LiveSettings s = blockStream(3);
    auto source = lod::LiveSource::create(queue, s);
    REQUIRE(source);
    queue->push(blockPage(s, 1, 0, 3));
    queue->push(blockPage(s, 1, 0, 3));
    queue->push(blockPage(s, 1, 1, 3));
    const auto first = pollEpochs(**source);
    std::printf("  after chunk 0 twice and chunk 1: %zu epochs handed over (%zu changes)\n", first.size(),
                first.empty() ? size_t{0} : first.begin()->second.size());
    CHECK(first.empty());
    queue->push(blockPage(s, 1, 2, 3));
    const auto second = pollEpochs(**source);
    REQUIRE(second.count(1) == 1);
    CHECK(second.at(1).size() == 3);
    CHECK((*source)->stats().late == 0);
}

TEST_CASE("qa: pages out of order and epochs interleaved are handed over whole and in order",
          "[lod][stream][live][qa]") {
    auto queue = std::make_shared<lod::PageQueue>();
    const lod::LiveSettings s = blockStream(3);
    auto source = lod::LiveSource::create(queue, s);
    REQUIRE(source);
    queue->push(blockPage(s, 1, 2, 3));
    queue->push(blockPage(s, 1, 0, 3));
    queue->push(blockPage(s, 2, 1, 2));
    queue->push(blockPage(s, 1, 1, 3));
    queue->push(blockPage(s, 2, 0, 2));
    const auto got = pollEpochs(**source);
    REQUIRE(got.size() == 2);
    REQUIRE(got.at(1).size() == 3);
    REQUIRE(got.at(2).size() == 2);
    for (uint32_t c = 0; c < 3; ++c) CHECK(got.at(1)[c].request.chunk == c);
    CHECK((*source)->stats().dropped == 0);
    CHECK((*source)->stats().epochs == 2);
}

TEST_CASE("qa: pages a stream cannot hold are refused and change nothing", "[lod][stream][live][qa]") {
    auto queue = std::make_shared<lod::PageQueue>();
    const lod::LiveSettings s = blockStream(3);
    auto source = lod::LiveSource::create(queue, s);
    REQUIRE(source);
    std::vector<lod::PageUpdate> bad;
    bad.push_back(blockPage(s, 1, 3, 1));                         // chunk past the stream's
    bad.push_back(blockPage(s, 1, 0xFFFFFFFFu, 1));               // far past it
    bad.push_back(blockPage(s, 1, 0, 1, 0));                      // no splats
    bad.push_back(blockPage(s, 1, 0, 1, 5));                      // more than a chunk holds
    bad.push_back(blockPage(s, 0, 0, 1));                         // epoch 0
    bad.push_back(blockPage(s, 1, 0, 0));                         // no page count
    {
        lod::PageUpdate odd = blockPage(s, 1, 0, 1);
        odd.bytes.resize(odd.bytes.size() + 3);                   // not whole splats
        bad.push_back(std::move(odd));
        lod::PageUpdate remove = blockPage(s, 1, 0, 1);
        remove.op = lod::PageUpdate::Op::Remove;                  // a remove with bytes
        bad.push_back(std::move(remove));
    }
    const size_t n = bad.size();
    for (auto& p : bad) queue->push(std::move(p));
    CHECK(pollEpochs(**source).empty());
    CHECK((*source)->stats().refused == n);
    CHECK((*source)->stats().pages == 0);
    // An epoch whose pages disagree on how many it has: the odd one refused.
    queue->push(blockPage(s, 2, 0, 2));
    queue->push(blockPage(s, 2, 1, 3));
    queue->push(blockPage(s, 2, 1, 2));
    const auto got = pollEpochs(**source);
    REQUIRE(got.count(2) == 1);
    CHECK(got.at(2).size() == 2);
    CHECK((*source)->stats().refused == n + 1);
}

TEST_CASE("qa: many producers on many streams, polled while they push", "[lod][stream][live][qa]") {
    // Eight streams on one queue, four producer threads each. Every epoch has
    // eight pages, two from each producer; a producer starts epoch e + 1 only
    // when all four have pushed e (the contract: an epoch's pages before any
    // of the next). One consumer thread polls all eight sources meanwhile.
    constexpr uint32_t kStreams = 8, kProducers = 4, kChunks = 8, kEpochs = 300;
    auto queue = std::make_shared<lod::PageQueue>();
    std::vector<lod::LiveSettings> settings;
    std::vector<std::unique_ptr<lod::LiveSource>> sources;
    for (uint32_t st = 0; st < kStreams; ++st) {
        settings.push_back(blockStream(kChunks, st));
        auto source = lod::LiveSource::create(queue, settings.back());
        REQUIRE(source);
        sources.push_back(std::move(*source));
    }
    std::vector<std::atomic<uint32_t>> pushed(kStreams);
    for (auto& p : pushed) p = 0;
    std::vector<std::thread> producers;
    for (uint32_t st = 0; st < kStreams; ++st) {
        for (uint32_t p = 0; p < kProducers; ++p) {
            producers.emplace_back([&, st, p] {
                for (uint64_t e = 1; e <= kEpochs; ++e) {
                    while (pushed[st].load() < (e - 1) * kProducers) std::this_thread::yield();
                    for (uint32_t c = p; c < kChunks; c += kProducers) {
                        auto page = blockPage(settings[st], e, c, kChunks, 1 + (c + e) % 4);
                        if (const auto ok = lod::LiveSource::check(page, settings[st]); !ok) std::abort();
                        queue->push(std::move(page));
                    }
                    pushed[st].fetch_add(1);
                }
            });
        }
    }
    std::vector<uint64_t> last(kStreams, 0);
    uint64_t wrong = 0, backwards = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    bool all = false;
    while (!all && std::chrono::steady_clock::now() < deadline) {
        all = true;
        for (uint32_t st = 0; st < kStreams; ++st) {
            for (const auto& [epoch, changes] : pollEpochs(*sources[st])) {
                wrong += changes.size() == kChunks ? 0 : 1;
                backwards += epoch > last[st] ? 0 : 1;
                last[st] = epoch;
            }
            all = all && last[st] == kEpochs;
        }
    }
    for (auto& t : producers) t.join();
    uint64_t dropped = 0, late = 0, refused = 0;
    for (const auto& s : sources) {
        dropped += s->stats().dropped;
        late += s->stats().late;
        refused += s->stats().refused;
    }
    std::printf("  %u streams x %u producers x %u epochs: %llu not whole, %llu backwards, dropped %llu, late %llu, "
                "refused %llu\n",
                kStreams, kProducers, kEpochs, static_cast<unsigned long long>(wrong),
                static_cast<unsigned long long>(backwards), static_cast<unsigned long long>(dropped),
                static_cast<unsigned long long>(late), static_cast<unsigned long long>(refused));
    CHECK(all);
    CHECK(wrong == 0);
    CHECK(backwards == 0);
    CHECK(dropped == 0);
    CHECK(late == 0);
    CHECK(refused == 0);
}

TEST_CASE("qa: broken SPZ bytes never crash the reader or overrun a staging slot", "[io][spz][stream][qa]") {
    // Valid files (versions 3 and 4, degree 1), then every kind of damage:
    // truncated at many lengths, bits flipped, and version 4's plain header
    // patched (magic, version, count, degree, streams, table offset). Each is
    // read from memory and fetched into a staging slot of a stream whose
    // chunks hold exactly the valid file's count; whatever reads must fit.
    constexpr uint32_t kCount = 300;
    lod::LiveSettings live;
    live.chunks = 1;
    live.format.payload = lod::ChunkFormat::Payload::Spz;
    live.format.chunkSplats = kCount;
    live.format.restPerColour = 3;
    live.format.shWords = 5;
    live.name = "qa-fuzz";
    uint64_t tried = 0, readable = 0, fetched = 0, inconsistent = 0;
    Lcg64 rng;
    const auto attempt = [&](const std::vector<uint8_t>& bytes) {
        ++tried;
        auto raw = io::readSpz(std::span<const uint8_t>(bytes), "fuzz");
        if (raw) {
            ++readable;
            const io::SpzStreams& at = raw->spz;
            const uint64_t size = raw->packed.size();
            if (raw->count == 0 || at.positions > size || at.alphas > size || at.colours > size ||
                at.scales > size || at.rotations > size || at.sh > size) {
                ++inconsistent;
            }
        }
        // Through a live source, as a loader would: one page, one fetch.
        auto queue = std::make_shared<lod::PageQueue>();
        auto source = lod::LiveSource::create(queue, live);
        REQUIRE(source);
        lod::PageUpdate u;
        u.epoch = 1;
        u.pages = 1;
        u.op = lod::PageUpdate::Op::Replace;
        u.bytes = asBytes(bytes);
        if (u.bytes.empty()) return;
        queue->push(std::move(u));
        std::vector<lod::ChunkChange> changes;
        (*source)->poll(changes);
        REQUIRE(changes.size() == 1);
        // Exactly the slot's size, in a vector of its own: an overrun is ASan's.
        std::vector<std::byte> staging(changes[0].request.bytes);
        if ((*source)->fetch(changes[0].request, staging)) ++fetched;
    };
    for (const int version : {3, 4}) {
        const std::vector<uint8_t> good = spzBytes(kCount, 77 + version, true, version, 1);
        attempt(good);
        const size_t step = std::max<size_t>(1, good.size() / 300);
        for (size_t n = 0; n < good.size(); n += step) {
            attempt(std::vector<uint8_t>(good.begin(), good.begin() + static_cast<ptrdiff_t>(n)));
        }
        for (int k = 0; k < 1500; ++k) {
            std::vector<uint8_t> flipped = good;
            const int flips = 1 + static_cast<int>(rng.next() % 4);
            for (int f = 0; f < flips; ++f) {
                // Most damage near the front, where the headers are.
                const size_t span = (rng.next() % 4 == 0) ? flipped.size() : std::min<size_t>(flipped.size(), 96);
                flipped[rng.next() % span] ^= static_cast<uint8_t>(1u << (rng.next() % 8));
            }
            attempt(flipped);
        }
        if (version == 4) {
            const auto patched = [&](size_t at, const void* value, size_t n) {
                std::vector<uint8_t> b = good;
                std::memcpy(b.data() + at, value, n);
                attempt(b);
            };
            for (const uint32_t magic : {0u, 0x5053474Eu ^ 1u, 0xFFFFFFFFu}) patched(0, &magic, 4);
            for (const uint32_t v : {0u, 1u, 2u, 3u, 5u, 255u, 0xFFFFFFFFu}) patched(4, &v, 4);
            for (const uint32_t n : {0u, 1u, kCount - 1, kCount + 1, 1u << 20, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu}) {
                patched(8, &n, 4);
            }
            for (const uint8_t d : {uint8_t{2}, uint8_t{3}, uint8_t{4}, uint8_t{5}, uint8_t{255}}) patched(12, &d, 1);
            for (const uint8_t fb : {uint8_t{0}, uint8_t{24}, uint8_t{31}, uint8_t{32}, uint8_t{255}}) patched(13, &fb, 1);
            for (const uint8_t flags : {uint8_t{2}, uint8_t{0xFF}}) patched(14, &flags, 1);
            for (const uint8_t streams : {uint8_t{0}, uint8_t{1}, uint8_t{7}, uint8_t{255}}) patched(15, &streams, 1);
            for (const uint32_t toc : {0u, 31u, 33u, 1u << 30, 0xFFFFFFFFu}) patched(16, &toc, 4);
        }
    }
    std::printf("  %llu damaged SPZ files: %llu read (%llu with streams past their bytes), %llu fetched into staging\n",
                static_cast<unsigned long long>(tried), static_cast<unsigned long long>(readable),
                static_cast<unsigned long long>(inconsistent), static_cast<unsigned long long>(fetched));
    CHECK(inconsistent == 0);
    CHECK(readable >= 2);   // the two undamaged files, at least
}

// --- a pool --------------------------------------------------------------------

namespace {

/// A pool over an SPZ stream, the device's references for page sets, and the
/// frame loop: everything a GPU case below shares.
struct SpzRig {
    test::Gpu*                         gpu = nullptr;
    std::unique_ptr<scene::CloudLoader> loader;
    std::unique_ptr<render::TileRasterizer> raster;
    std::unique_ptr<lod::CutSelector>  cut;
    render::RenderSettings             settings;
    render::Projection                 projection;
    lod::LiveSettings                  live;
    std::shared_ptr<lod::PageQueue>    queue;
    lod::LiveSource*                   source = nullptr;   // owned by the pool
    std::unique_ptr<lod::StreamingPool> pool;

    explicit SpzRig(test::Gpu* g) : gpu(g) {
        auto l = scene::CloudLoader::create(*gpu->library);
        auto r = render::TileRasterizer::create(*gpu->library);
        auto c = lod::CutSelector::create(*gpu->library);
        REQUIRE(l);
        REQUIRE(r);
        REQUIRE(c);
        loader = std::make_unique<scene::CloudLoader>(std::move(*l));
        raster = std::make_unique<render::TileRasterizer>(std::move(*r));
        cut = std::make_unique<lod::CutSelector>(std::move(*c));
        settings.width = 160;
        settings.height = 120;
        projection = render::projectionFor(render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0}),
                                           settings.width, settings.height);
    }

    void open(uint32_t chunks, uint32_t chunkSplats, lod::StreamingSettings streaming = {}) {
        live.stream = 0;
        live.chunks = chunks;
        live.format.payload = lod::ChunkFormat::Payload::Spz;
        live.format.chunkSplats = chunkSplats;
        live.format.restPerColour = 0;
        live.format.shWords = 1;
        live.name = "qa-spz";
        queue = std::make_shared<lod::PageQueue>();
        auto s = lod::LiveSource::create(queue, live);
        REQUIRE(s);
        source = s->get();
        streaming.library = gpu->library.get();
        auto p = lod::StreamingPool::open(*gpu->device, std::unique_ptr<lod::ChunkSource>(std::move(*s)), streaming);
        if (!p) FAIL(p.error().toString());
        pool = std::move(*p);
    }

    scene::GpuSplats cloudOf(const std::vector<uint8_t>& bytes) {
        auto raw = io::readSpz(std::span<const uint8_t>(bytes), "page");
        REQUIRE(raw);
        auto c = loader->upload(*raw, 3);
        REQUIRE(c);
        return std::move(*c);
    }

    render::RenderTargets draw(const std::vector<const scene::GpuSplats*>& clouds) {
        std::vector<render::SplatInstance> instances;
        for (const auto* c : clouds) {
            if (c != nullptr) instances.push_back({c, render::Mat4::identity()});
        }
        render::RenderTargets t;
        REQUIRE(raster->render(projection, instances, settings, t));
        return t;
    }

    render::RenderTargets shown() {
        const std::vector<lod::LodInstance> instances{{&pool->cloud(), render::Mat4::identity()}};
        auto selected = cut->select(projection, instances, 0.0F);
        if (!selected) FAIL(selected.error().toString());
        render::RenderTargets t;
        REQUIRE(raster->render(projection, *selected, settings, t));
        return t;
    }

    uint32_t maxDiff(const render::RenderTargets& a, const render::RenderTargets& b) {
        auto d = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(d);
        return d->max;
    }

    /// Epoch `epoch`: chunk c takes pages[c] (null: removed).
    void push(uint64_t epoch, const std::vector<const std::vector<uint8_t>*>& pages, uint64_t ptp = 0) {
        for (uint32_t c = 0; c < pages.size(); ++c) {
            lod::PageUpdate u;
            u.stream = live.stream;
            u.epoch = epoch;
            u.chunk = c;
            u.pages = static_cast<uint32_t>(pages.size());
            u.ptpTime = ptp;
            u.op = pages[c] == nullptr ? lod::PageUpdate::Op::Remove : lod::PageUpdate::Op::Replace;
            if (pages[c] != nullptr) u.bytes = asBytes(*pages[c]);
            queue->push(std::move(u));
        }
    }
};

}   // namespace

TEST_CASE("qa: the epoch being built does not change how the shown one is drawn", "[lod][gpu][stream][live][spz][qa]") {
    // Small gaussians, so the antialiasing compensation shows. Epoch 1's
    // pages say antialiased; epoch 2's say not, and come a page a frame
    // (bytesPerFrame 1). While epoch 2 is being placed, epoch 1 is shown and
    // must be drawn as its own pages say.
    ATHENEA_REQUIRE_GPU(gpu);
    SpzRig rig(gpu);
    lod::StreamingSettings streaming;
    streaming.bytesPerFrame = 1;
    rig.open(3, 600, streaming);
    std::vector<std::vector<uint8_t>> a, b;
    std::vector<scene::GpuSplats> ca, cb;
    for (uint32_t c = 0; c < 3; ++c) {
        a.push_back(spzBytes(500, 300 + c, true, 4, 0, 0.002F, 0.01F));
        b.push_back(spzBytes(500, 300 + c, false, 4, 0, 0.002F, 0.01F));
        ca.push_back(rig.cloudOf(a.back()));
        cb.push_back(rig.cloudOf(b.back()));
    }
    const auto refA = rig.draw({&ca[0], &ca[1], &ca[2]});
    const auto refB = rig.draw({&cb[0], &cb[1], &cb[2]});
    const uint32_t flagMatters = rig.maxDiff(refA, refB);
    rig.push(1, {&a[0], &a[1], &a[2]});
    for (int k = 0; k < 8 && rig.pool->status().epoch != 1; ++k) REQUIRE(rig.pool->update(true));
    REQUIRE(rig.pool->status().epoch == 1);
    CHECK(rig.maxDiff(refA, rig.shown()) == 0);

    rig.push(2, {&b[0], &b[1], &b[2]});
    int frames = 0, mismatched = 0, building = 0;
    for (; frames < 400 && rig.pool->status().epoch != 2; ++frames) {
        REQUIRE(rig.pool->update(false));
        const uint64_t epoch = rig.pool->status().epoch;
        building += epoch == 1 && rig.pool->status().pendingEpochs > 0 ? 1 : 0;
        mismatched += rig.maxDiff(epoch == 2 ? refB : refA, rig.shown()) == 0 ? 0 : 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::printf("  the flag alone: max %u; %d frames, %d of them while epoch 2 was built, %d not the epoch shown\n",
                flagMatters, frames, building, mismatched);
    REQUIRE(flagMatters > 0);
    CHECK(rig.pool->status().epoch == 2);
    CHECK(mismatched == 0);

    // One epoch whose pages disagree: each must be drawn as its own says.
    rig.push(3, {&a[0], &b[1], &a[2]});
    for (int k = 0; k < 8 && rig.pool->status().epoch != 3; ++k) REQUIRE(rig.pool->update(true));
    REQUIRE(rig.pool->status().epoch == 3);
    const uint32_t mixed = rig.maxDiff(rig.draw({&ca[0], &cb[1], &ca[2]}), rig.shown());
    std::printf("  an epoch of mixed flags against its pages drawn alone: max %u\n", mixed);
    CHECK(mixed == 0);
}

TEST_CASE("qa: a backlog of epochs is caught up at once, not played a frame each", "[lod][gpu][stream][live][spz][qa]") {
    // Sixty complete epochs wait before the first update (a stage that opened
    // late, a frame that stalled). A live stream wants the latest: the pool
    // should show epoch 60 within a few updates, not one epoch a frame.
    ATHENEA_REQUIRE_GPU(gpu);
    SpzRig rig(gpu);
    rig.open(3, 400);
    std::vector<std::vector<std::vector<uint8_t>>> sets(4);
    std::vector<std::vector<scene::GpuSplats>> clouds(4);
    std::vector<render::RenderTargets> refs;
    for (uint32_t k = 0; k < 4; ++k) {
        for (uint32_t c = 0; c < 3; ++c) {
            sets[k].push_back(spzBytes(300 + 10 * k, 1000 * k + c));
            clouds[k].push_back(rig.cloudOf(sets[k].back()));
        }
        refs.push_back(rig.draw({&clouds[k][0], &clouds[k][1], &clouds[k][2]}));
    }
    constexpr uint64_t kEpochs = 60;
    for (uint64_t e = 1; e <= kEpochs; ++e) {
        const auto& s = sets[e % 4];
        rig.push(e, {&s[0], &s[1], &s[2]});
    }
    int updates = 0, mismatched = 0;
    uint32_t mostPending = 0;
    for (; updates < 1000 && rig.pool->status().epoch != kEpochs; ++updates) {
        REQUIRE(rig.pool->update(false));
        const auto st = rig.pool->status();
        mostPending = std::max(mostPending, st.pendingEpochs);
        if (st.epoch > 0) mismatched += rig.maxDiff(refs[st.epoch % 4], rig.shown()) == 0 ? 0 : 1;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::printf("  %llu epochs waiting: epoch %llu shown after %d updates (%llu shown on the way, %u pending at most); "
                "%d frames not their epoch\n",
                static_cast<unsigned long long>(kEpochs), static_cast<unsigned long long>(rig.pool->status().epoch),
                updates, static_cast<unsigned long long>(rig.pool->status().epochs), mostPending, mismatched);
    CHECK(mismatched == 0);
    CHECK(rig.pool->status().epoch == kEpochs);
    CHECK(updates <= 10);
}

TEST_CASE("qa: a producer faster than the frames: every frame whole, and the pool keeps up",
          "[lod][gpu][stream][live][spz][qa]") {
    // A producer thread pushes an epoch every millisecond while the frame
    // thread updates, draws and takes 4 ms more (a frame of a real view):
    // four epochs or so a frame, of one page, then of sixteen. Every frame
    // must be the whole epoch status() names, and the epochs waiting must
    // stay few -- the latest is what a live stream wants.
    ATHENEA_REQUIRE_GPU(gpu);
    for (const uint32_t chunks : {1u, 16u}) {
        SpzRig rig(gpu);
        rig.open(chunks, 200);
        std::vector<std::vector<std::vector<uint8_t>>> sets(3);
        std::vector<std::vector<scene::GpuSplats>> clouds(3);
        std::vector<render::RenderTargets> refs;
        for (uint32_t k = 0; k < 3; ++k) {
            std::vector<const scene::GpuSplats*> all;
            for (uint32_t c = 0; c < chunks; ++c) {
                sets[k].push_back(spzBytes(120 + 7 * k + c, 5000 * k + c));
            }
            for (uint32_t c = 0; c < chunks; ++c) clouds[k].push_back(rig.cloudOf(sets[k][c]));
            for (auto& c : clouds[k]) all.push_back(&c);
            refs.push_back(rig.draw(all));
        }
        constexpr uint64_t kEpochs = 400;
        std::atomic<bool> done{false};
        std::thread producer([&] {
            for (uint64_t e = 1; e <= kEpochs; ++e) {
                std::vector<const std::vector<uint8_t>*> pages;
                for (auto& p : sets[e % 3]) pages.push_back(&p);
                rig.push(e, pages, e * 1000000ull);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            done = true;
        });
        int frames = 0, mismatched = 0, after = 0;
        uint32_t mostPending = 0;
        uint64_t last = 0;
        bool ordered = true, ptp = true;
        for (; frames < 20000; ++frames) {
            REQUIRE(rig.pool->update(false));
            const auto st = rig.pool->status();
            mostPending = std::max(mostPending, st.pendingEpochs);
            if (st.epoch > 0) {
                mismatched += rig.maxDiff(refs[st.epoch % 3], rig.shown()) == 0 ? 0 : 1;
                ptp = ptp && st.ptpTime == st.epoch * 1000000ull;
            }
            ordered = ordered && st.epoch >= last;
            last = st.epoch;
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
            if (done) {
                ++after;
                if (st.epoch == kEpochs) break;
            }
        }
        producer.join();
        std::printf("  %u chunk(s): %d frames, %d after the producer ended; %u epochs pending at most; "
                    "%llu epochs shown; %d frames not their epoch\n",
                    chunks, frames, after, mostPending, static_cast<unsigned long long>(rig.pool->status().epochs),
                    mismatched);
        CHECK(mismatched == 0);
        CHECK(ordered);
        CHECK(ptp);
        CHECK(last == kEpochs);
        CHECK(mostPending <= 4);
    }
}

TEST_CASE("qa: pages a loader cannot read drop their epoch, whatever is wrong with them",
          "[lod][gpu][stream][live][spz][qa]") {
    ATHENEA_REQUIRE_GPU(gpu);
    SpzRig rig(gpu);
    rig.open(2, 300);
    const std::vector<uint8_t> p0 = spzBytes(250, 1), p1 = spzBytes(260, 2);
    const scene::GpuSplats c0 = rig.cloudOf(p0), c1 = rig.cloudOf(p1);
    const auto good = rig.draw({&c0, &c1});
    rig.push(1, {&p0, &p1});
    REQUIRE(rig.pool->update(true));
    REQUIRE(rig.pool->status().epoch == 1);

    std::vector<std::pair<const char*, std::vector<uint8_t>>> broken;
    broken.emplace_back("truncated", std::vector<uint8_t>(p0.begin(), p0.begin() + p0.size() / 2));
    broken.emplace_back("one byte", std::vector<uint8_t>(1, 0x4E));
    broken.emplace_back("past a chunk", spzBytes(301, 3));
    broken.emplace_back("far past a chunk", spzBytes(5000, 4));
    broken.emplace_back("version 3 truncated", [&] {
        auto v3 = spzBytes(250, 5, true, 3);
        v3.resize(v3.size() - 9);
        return v3;
    }());
    {
        auto wrong = p0;
        wrong[0] ^= 0xFF;   // the magic
        broken.emplace_back("wrong magic", std::move(wrong));
        auto version = p0;
        version[4] = 99;
        broken.emplace_back("unknown version", std::move(version));
        auto count = p0;
        const uint32_t huge = 0x7FFFFFF0u;
        std::memcpy(count.data() + 8, &huge, 4);
        broken.emplace_back("a count of 2^31", std::move(count));
    }
    uint64_t epoch = 1, corrupt = 0;
    for (const auto& [what, bytes] : broken) {
        rig.push(++epoch, {&bytes, &p1});
        for (int k = 0; k < 4; ++k) REQUIRE(rig.pool->update(true));
        const auto st = rig.pool->status();
        const uint32_t diff = rig.maxDiff(good, rig.shown());
        std::printf("  %-20s: epoch shown %llu, dropped as corrupt %llu, against epoch 1 max %u\n", what,
                    static_cast<unsigned long long>(st.epoch), static_cast<unsigned long long>(st.droppedCorrupt),
                    diff);
        CHECK(st.epoch == 1);
        CHECK(st.droppedCorrupt == ++corrupt);
        CHECK(diff == 0);
    }
    // And the stream goes on.
    rig.push(++epoch, {&p1, &p0});
    REQUIRE(rig.pool->update(true));
    CHECK(rig.pool->status().epoch == epoch);
    CHECK(rig.maxDiff(rig.draw({&c1, &c0}), rig.shown()) == 0);
}

TEST_CASE("qa: a store's budget at its edges", "[lod][gpu][stream][live][spz][qa]") {
    // 0 and 1 gaussian: one slot, the least a store takes -- the first epoch
    // of one chunk shows, and one that replaces it has no room beside it.
    // A huge budget is held to two epochs of every chunk.
    ATHENEA_REQUIRE_GPU(gpu);
    const std::vector<uint8_t> p0 = spzBytes(150, 11), p1 = spzBytes(160, 12);
    for (const uint64_t budget : {uint64_t{0}, uint64_t{1}, ~uint64_t{0}}) {
        SpzRig rig(gpu);
        lod::StreamingSettings streaming;
        streaming.budgetSplats = budget;
        rig.open(2, 200, streaming);
        const scene::GpuSplats c0 = rig.cloudOf(p0), c1 = rig.cloudOf(p1);
        const auto expectSlots = budget == ~uint64_t{0} ? 4u : 1u;
        CHECK(rig.pool->status().slots == expectSlots);
        lod::PageUpdate add;
        add.epoch = 1;
        add.pages = 1;
        add.chunk = 0;
        add.bytes = asBytes(p0);
        rig.queue->push(std::move(add));
        REQUIRE(rig.pool->update(true));
        CHECK(rig.pool->status().epoch == 1);
        CHECK(rig.maxDiff(rig.draw({&c0}), rig.shown()) == 0);
        lod::PageUpdate replace;
        replace.epoch = 2;
        replace.pages = 1;
        replace.chunk = 0;
        replace.op = lod::PageUpdate::Op::Replace;
        replace.bytes = asBytes(p1);
        rig.queue->push(std::move(replace));
        REQUIRE(rig.pool->update(true));
        const auto st = rig.pool->status();
        if (expectSlots == 1) {
            CHECK(st.epoch == 1);
            CHECK(st.droppedNoRoom == 1);
            CHECK(rig.maxDiff(rig.draw({&c0}), rig.shown()) == 0);
        } else {
            CHECK(st.epoch == 2);
            CHECK(rig.maxDiff(rig.draw({&c1}), rig.shown()) == 0);
        }
    }
}

TEST_CASE("qa: pools opened and closed with pages in flight", "[lod][gpu][stream][live][spz][qa]") {
    // Forty pools over one queue, each closed right after an update that set
    // its loaders going and recorded copies: nothing may outlive its pool.
    ATHENEA_REQUIRE_GPU(gpu);
    SpzRig rig(gpu);
    std::vector<std::vector<uint8_t>> pages;
    for (uint32_t c = 0; c < 4; ++c) pages.push_back(spzBytes(500, 40 + c));
    std::vector<scene::GpuSplats> clouds;
    for (auto& p : pages) clouds.push_back(rig.cloudOf(p));
    for (int round = 0; round < 40; ++round) {
        rig.open(4, 600);
        rig.push(1, {&pages[0], &pages[1], &pages[2], &pages[3]});
        REQUIRE(rig.pool->update(false));
        if (round % 3 == 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            REQUIRE(rig.pool->update(false));
        }
        rig.pool.reset();
    }
    rig.open(4, 600);
    rig.push(1, {&pages[0], &pages[1], &pages[2], &pages[3]});
    for (int k = 0; k < 8 && rig.pool->status().epoch != 1; ++k) REQUIRE(rig.pool->update(true));
    CHECK(rig.pool->status().epoch == 1);
    CHECK(rig.maxDiff(rig.draw({&clouds[0], &clouds[1], &clouds[2], &clouds[3]}), rig.shown()) == 0);
}

TEST_CASE("qa: a producer that stops mid-epoch leaves the shown epoch whole", "[lod][gpu][stream][live][spz][qa]") {
    ATHENEA_REQUIRE_GPU(gpu);
    SpzRig rig(gpu);
    rig.open(3, 300);
    const std::vector<uint8_t> a = spzBytes(200, 61), b = spzBytes(210, 62), c = spzBytes(220, 63);
    const scene::GpuSplats ca = rig.cloudOf(a), cb = rig.cloudOf(b), cc = rig.cloudOf(c);
    rig.push(1, {&a, &b, &c});
    REQUIRE(rig.pool->update(true));
    REQUIRE(rig.pool->status().epoch == 1);
    std::thread([&] {
        // Two of epoch 2's three pages, and the producer is gone.
        for (uint32_t ch = 0; ch < 2; ++ch) {
            lod::PageUpdate u;
            u.epoch = 2;
            u.chunk = ch;
            u.pages = 3;
            u.op = lod::PageUpdate::Op::Replace;
            u.bytes = asBytes(ch == 0 ? c : a);
            rig.queue->push(std::move(u));
        }
    }).join();
    for (int k = 0; k < 10; ++k) {
        REQUIRE(rig.pool->update(k % 2 == 0));
        CHECK(rig.pool->status().epoch == 1);
    }
    CHECK(rig.maxDiff(rig.draw({&ca, &cb, &cc}), rig.shown()) == 0);
    rig.push(3, {&c, nullptr, &a});
    REQUIRE(rig.pool->update(true));
    CHECK(rig.pool->status().epoch == 3);
    CHECK(rig.source->stats().dropped == 1);
    CHECK(rig.maxDiff(rig.draw({&cc, &ca}), rig.shown()) == 0);
}

TEST_CASE("qa: an SPZ whose fixed point is past what a word holds is refused", "[io][spz][stream][qa]") {
    // Version 4 keeps its header in the clear: byte 13 is the fixed point's
    // fractional bits. 1 << bits on a 32-bit word is undefined from 32 on
    // (UBSan: SplatReaders.cpp, spzEncoding), and no position needs more than
    // the 24 bits SPZ stores; such a file is broken and must be refused, not
    // read with a scale that means nothing.
    const std::vector<uint8_t> good = spzBytes(64, 5);
    REQUIRE(io::readSpz(std::span<const uint8_t>(good), "good"));
    int accepted = 0;
    for (const uint8_t bits : {uint8_t{32}, uint8_t{40}, uint8_t{76}, uint8_t{255}}) {
        std::vector<uint8_t> bad = good;
        bad[13] = bits;
        if (io::readSpz(std::span<const uint8_t>(bad), "bad")) {
            ++accepted;
            std::printf("  fractional bits %u: read\n", bits);
        }
    }
    CHECK(accepted == 0);
}
