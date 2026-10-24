// Copyright (c) 2026 jesus luque.
//
// Streamed clouds through a ChunkSource: a .athc file as before, and a live
// stream whose pages producers push in epochs. Everything is checked by
// renders compared on the device: the pool's cut against a cloud uploaded
// straight from the same chunk bytes, in the same order.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <optional>
#include <thread>
#include <vector>

#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
using test::CloudBuilder;
using test::randomCloud;

namespace {

constexpr uint32_t kChunkSplats = 1000;
constexpr uint32_t kChunks = 6;

struct Harness {
    test::Gpu*             gpu;
    scene::CloudLoader     loader;
    render::TileRasterizer raster;
    lod::LodBuilder        builder;
    lod::CutSelector       cut;
    render::RenderSettings settings;
    render::Projection     projection;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto builder = lod::LodBuilder::create(*gpu->library);
    auto cut = lod::CutSelector::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!cut) FAIL(cut.error().toString());
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Projection projection = render::projectionFor(
        render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0}), settings.width, settings.height);
    return std::unique_ptr<Harness>(new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*builder),
                                                std::move(*cut), settings, projection});
}

/// A cloud of kChunks chunks written as a .athc; `seed` makes it another.
std::filesystem::path writeCloud(Harness& h, uint64_t seed, const char* name) {
    CloudBuilder built = randomCloud(kChunks * kChunkSplats, seed, 0.003F, 0.08F);
    auto cloud = h.loader.upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = kChunkSplats;
    auto lod = h.builder.build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(lod->chunks() == kChunks);
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    REQUIRE(lod::writeAthc(*h.gpu->device, *lod, path));
    return path;
}

/// A chunk's block as a producer would send it: copied off the file.
std::vector<std::byte> chunkBytes(lod::AthcSource& source, uint32_t chunk) {
    const lod::ChunkRequest r = source.request(chunk);
    std::vector<std::byte> bytes(r.bytes);
    REQUIRE(source.fetch(r, bytes));
    return bytes;
}

/// The cloud the chunk blocks `blocks` make, in that order, uploaded as it
/// is: the arrays of each block put one after the other (bytes copied, no
/// arithmetic) -- what a pool showing them must draw.
scene::GpuSplats uploaded(gpu::Device& device, const lod::ChunkFormat& f,
                          const std::vector<const std::vector<std::byte>*>& blocks) {
    const uint64_t per = f.elementBytes();
    std::vector<std::byte> positions, shape, sh;
    uint32_t count = 0;
    for (const std::vector<std::byte>* b : blocks) {
        const uint64_t n = b->size() / per;
        const std::byte* at = b->data();
        positions.insert(positions.end(), at, at + n * 16);
        shape.insert(shape.end(), at + n * 16, at + n * 32);
        sh.insert(sh.end(), at + n * 32, at + n * 32 + n * 4 * f.shWords);
        count += static_cast<uint32_t>(n);
    }
    scene::GpuSplats s;
    s.source = "reference";
    s.count = count;
    s.declared = count;
    s.restPerColour = f.restPerColour;
    s.shWords = f.shWords;
    s.linear = f.linear;
    const auto make = [&](const std::vector<std::byte>& data, uint32_t element) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(data.size(), element);
        desc.elementBytes = element;
        desc.label = "test.reference";
        auto b = gpu::Buffer::create(device, desc, data.empty() ? nullptr : data.data());
        REQUIRE(b);
        return *b;
    };
    s.positions = make(positions, 16);
    s.shape = make(shape, 4);
    s.sh = make(sh, 4);
    return s;
}

render::RenderTargets renderCloud(Harness& h, const scene::GpuSplats& cloud) {
    render::RenderTargets t;
    std::vector<render::SplatInstance> instances;
    if (cloud.count > 0) {
        instances.push_back({&cloud, render::Mat4::identity()});
    }
    REQUIRE(h.raster.render(h.projection, instances, h.settings, t));
    return t;
}

struct Cut {
    lod::CutStats         stats;
    render::RenderTargets targets;
};

Cut renderPool(Harness& h, const lod::LodCloud& cloud, float threshold) {
    Cut cut;
    std::vector<lod::CutStats> stats;
    const std::vector<lod::LodInstance> instances{{&cloud, render::Mat4::identity()}};
    auto selected = h.cut.select(h.projection, instances, threshold, &stats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(stats.size() == 1);
    cut.stats = stats.front();
    REQUIRE(h.raster.render(h.projection, *selected, h.settings, cut.targets));
    return cut;
}

render::ImageDifference compare(Harness& h, const render::RenderTargets& a, const render::RenderTargets& b) {
    auto diff = render::compareImages(*h.gpu->library, a.colour, b.colour, h.settings.width, h.settings.height);
    REQUIRE(diff);
    return *diff;
}

/// Cut, want, update -- frame after frame, without waiting -- until a frame
/// places nothing and nothing is in flight. Returns the frames it took.
int settle(Harness& h, lod::StreamingPool& pool, float threshold, Cut* last, int most = 400) {
    for (int frame = 1; frame <= most; ++frame) {
        *last = renderPool(h, pool.cloud(), threshold);
        pool.want(last->stats.needs);
        auto placed = pool.update(false);
        if (!placed) FAIL(placed.error().toString());
        const auto s = pool.status();
        if (*placed == 0 && s.inFlight == 0 && s.missing == 0) {
            return frame;
        }
        if (*placed == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));   // the loaders' turn
        }
    }
    FAIL("the stream never settled");
    return 0;
}

/// A ChunkSource over a file whose loaders try to reach the device: the
/// first fetch calls the pool's update, which must refuse it.
class IntrudingSource final : public lod::ChunkSource {
public:
    explicit IntrudingSource(std::unique_ptr<lod::AthcSource> file) : file_(std::move(file)) {}
    const lod::ChunkFormat& format() const noexcept override { return file_->format(); }
    uint32_t chunkCount() const noexcept override { return file_->chunkCount(); }
    scene::Bounds bounds(uint32_t c) const override { return file_->bounds(c); }
    std::string name() const override { return file_->name(); }
    Result<lod::LodCloud> openCloud(gpu::Device& device) override { return file_->openCloud(device); }
    lod::ChunkRequest request(uint32_t c) const override { return file_->request(c); }
    Result<void> fetch(const lod::ChunkRequest& r, std::span<std::byte> staging) override {
        if (lod::StreamingPool* pool = pool_.exchange(nullptr); pool != nullptr) {
            auto refused = pool->update(false);
            code_ = refused ? ErrorCode::Unknown : refused.error().code();
            tried_ = true;
        }
        return file_->fetch(r, staging);
    }
    std::atomic<lod::StreamingPool*> pool_{nullptr};
    std::atomic<bool>                tried_{false};
    std::atomic<ErrorCode>           code_{ErrorCode::Unknown};

private:
    std::unique_ptr<lod::AthcSource> file_;
};

}   // namespace

TEST_CASE("a pool over an AthcSource settles on the image the cloud in memory draws", "[lod][gpu][stream]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const std::filesystem::path path = writeCloud(*h, 47, "athenea_test_source.athc");
    auto whole = lod::readAthc(*gpu->device, path);
    if (!whole) FAIL(whole.error().toString());

    for (const float threshold : {0.0F, 24.0F}) {
        const Cut memory = renderPool(*h, *whole, threshold);
        // One staging slot: every chunk goes through the same bytes of the
        // mapped buffer, given back only when the fence says its copy ran.
        lod::StreamingSettings settings;
        settings.budgetSplats = kChunks * kChunkSplats;
        settings.stagingChunks = 1;
        auto source = lod::AthcSource::open(path);
        REQUIRE(source);
        auto pool = lod::StreamingPool::open(*gpu->device, std::unique_ptr<lod::ChunkSource>(std::move(*source)),
                                             settings);
        if (!pool) FAIL(pool.error().toString());
        Cut stream;
        const int frames = settle(*h, **pool, threshold, &stream);
        const auto diff = compare(*h, memory.targets, stream.targets);
        std::printf("  threshold %.0f px: settled in %d frames; %u splats + %u merged; max %u\n",
                    static_cast<double>(threshold), frames, stream.stats.splats, stream.stats.merged, diff.max);
        CHECK(stream.stats.splats == memory.stats.splats);
        CHECK(stream.stats.merged == memory.stats.merged);
        CHECK(diff.max == 0);

        // And by path, waiting, as before the source existed.
        auto byPath = lod::StreamingPool::open(*gpu->device, path, {kChunks * kChunkSplats, 2});
        if (!byPath) FAIL(byPath.error().toString());
        Cut waited;
        for (int round = 0; round < 4; ++round) {
            waited = renderPool(*h, (*byPath)->cloud(), threshold);
            (*byPath)->want(waited.stats.needs);
            REQUIRE((*byPath)->update(true));
        }
        CHECK(compare(*h, memory.targets, waited.targets).max == 0);
    }
    std::filesystem::remove(path);
}

TEST_CASE("a frame budget spreads the chunks over frames and ends on the same image", "[lod][gpu][stream]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const std::filesystem::path path = writeCloud(*h, 53, "athenea_test_budget.athc");
    auto file = lod::AthcSource::open(path);
    REQUIRE(file);
    const uint64_t chunk = (*file)->format().chunkBytes();

    auto uncapped = lod::StreamingPool::open(*gpu->device, path, {kChunks * kChunkSplats, 2});
    if (!uncapped) FAIL(uncapped.error().toString());
    Cut free;
    const int freeFrames = settle(*h, **uncapped, 0.0F, &free);

    lod::StreamingSettings capped;
    capped.budgetSplats = kChunks * kChunkSplats;
    capped.bytesPerFrame = chunk + chunk / 2;   // one chunk a frame, never two
    auto pool = lod::StreamingPool::open(*gpu->device, path, capped);
    if (!pool) FAIL(pool.error().toString());
    int frames = 0;
    uint32_t mostPlaced = 0;
    Cut last;
    for (frames = 1; frames <= 200; ++frames) {
        last = renderPool(*h, (*pool)->cloud(), 0.0F);
        (*pool)->want(last.stats.needs);
        auto placed = (*pool)->update(true);   // loads waited for: only the budget holds chunks back
        if (!placed) FAIL(placed.error().toString());
        const auto s = (*pool)->status();
        mostPlaced = std::max(mostPlaced, *placed);
        CHECK(s.placedBytes <= capped.bytesPerFrame);
        if (*placed == 0 && s.inFlight == 0 && s.missing == 0) {
            break;
        }
    }
    const auto diff = compare(*h, free.targets, last.targets);
    std::printf("  uncapped: %d frames; one chunk a frame: %d frames; max %u\n", freeFrames, frames, diff.max);
    CHECK(mostPlaced == 1);
    CHECK(frames > static_cast<int>(kChunks));
    CHECK(last.stats.splats == free.stats.splats);
    CHECK(diff.max == 0);
    std::filesystem::remove(path);
}

TEST_CASE("a pool refuses a device call from any thread but the one that opened it", "[lod][gpu][stream]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const std::filesystem::path path = writeCloud(*h, 59, "athenea_test_threads.athc");
    auto file = lod::AthcSource::open(path);
    REQUIRE(file);
    auto* intruder = new IntrudingSource(std::move(*file));
    auto pool = lod::StreamingPool::open(*gpu->device, std::unique_ptr<lod::ChunkSource>(intruder),
                                         {kChunks * kChunkSplats, 2});
    if (!pool) FAIL(pool.error().toString());

    SECTION("another thread") {
        std::optional<Error> refused;
        std::thread other([&] {
            auto r = (*pool)->update(false);
            if (!r) refused = r.error();
            (void)(*pool)->status();   // any thread may ask how it stands
        });
        other.join();
        REQUIRE(refused.has_value());
        CHECK(refused->code() == ErrorCode::InvalidArgument);
        CHECK(refused->message().find("frame thread") != std::string::npos);
    }

    SECTION("a loader thread, from inside a fetch") {
        intruder->pool_ = pool->get();
        Cut cut;
        settle(*h, **pool, 0.0F, &cut);
        CHECK(intruder->tried_);
        CHECK(intruder->code_ == ErrorCode::InvalidArgument);
        // Refused, and nothing broken: the pool settles on the whole cloud.
        auto whole = lod::readAthc(*gpu->device, path);
        REQUIRE(whole);
        CHECK(compare(*h, renderPool(*h, *whole, 0.0F).targets, cut.targets).max == 0);
    }
    std::filesystem::remove(path);
}

namespace {

/// Two clouds whose chunks a live stream mixes, and their chunk blocks.
struct Pages {
    std::filesystem::path                 pathA, pathB;
    std::unique_ptr<lod::AthcSource>      a, b;
    std::vector<std::vector<std::byte>>   blocksA, blocksB;
    lod::LiveSettings                     settings;
};

Pages pages(Harness& h) {
    Pages p;
    p.pathA = writeCloud(h, 61, "athenea_test_live_a.athc");
    p.pathB = writeCloud(h, 67, "athenea_test_live_b.athc");
    auto a = lod::AthcSource::open(p.pathA);
    auto b = lod::AthcSource::open(p.pathB);
    REQUIRE(a);
    REQUIRE(b);
    p.a = std::move(*a);
    p.b = std::move(*b);
    REQUIRE(p.a->format() == p.b->format());
    for (uint32_t c = 0; c < kChunks; ++c) {
        p.blocksA.push_back(chunkBytes(*p.a, c));
        p.blocksB.push_back(chunkBytes(*p.b, c));
    }
    p.settings.stream = 7;
    p.settings.chunks = kChunks;
    p.settings.format = p.a->format();
    p.settings.bounds = p.a->bounds(0);
    p.settings.name = "test-live";
    return p;
}

/// What a stream shows: per chunk, the block it holds, or none.
using State = std::vector<const std::vector<std::byte>*>;

scene::GpuSplats uploadedState(gpu::Device& device, const lod::ChunkFormat& f, const State& state) {
    std::vector<const std::vector<std::byte>*> present;
    for (const auto* b : state) {
        if (b != nullptr) present.push_back(b);
    }
    return uploaded(device, f, present);
}

lod::PageUpdate page(uint32_t stream, uint64_t epoch, uint32_t chunk, lod::PageUpdate::Op op,
                     const std::vector<std::byte>* bytes, uint32_t pages, const scene::Bounds& bounds) {
    lod::PageUpdate u;
    u.stream = stream;
    u.epoch = epoch;
    u.chunk = chunk;
    u.op = op;
    if (bytes != nullptr) u.bytes = *bytes;
    u.bounds = bounds;
    u.ptpTime = epoch * 40'000'000;   // 25 epochs a second
    u.pages = pages;
    return u;
}

}   // namespace

TEST_CASE("a live stream shows whole epochs only, while a producer pushes the next", "[lod][gpu][stream][live]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    Pages p = pages(*h);
    gpu::Device& device = *gpu->device;

    // Epoch e: chunk c removed where (c + e) % 3 == 0, else A's or B's by the
    // parity of c + e -- each epoch a different image, every one complete.
    constexpr uint64_t kEpochs = 8;
    std::vector<State> states(kEpochs + 1, State(kChunks, nullptr));
    std::vector<render::RenderTargets> references(kEpochs + 1);
    references[0] = renderCloud(*h, uploadedState(device, p.settings.format, states[0]));
    for (uint64_t e = 1; e <= kEpochs; ++e) {
        for (uint32_t c = 0; c < kChunks; ++c) {
            if ((c + e) % 3 != 0) {
                states[e][c] = (c + e) % 2 == 0 ? &p.blocksA[c] : &p.blocksB[c];
            }
        }
        references[e] = renderCloud(*h, uploadedState(device, p.settings.format, states[e]));
    }
    // The references differ from each other: a half epoch would show.
    for (uint64_t e = 1; e < kEpochs; ++e) {
        CHECK(compare(*h, references[e], references[e + 1]).max > 0);
    }

    auto queue = std::make_shared<lod::PageQueue>();
    auto source = lod::LiveSource::create(queue, p.settings);
    REQUIRE(source);
    lod::LiveSource* live = source->get();
    lod::StreamingSettings settings;
    settings.budgetSplats = 2 * kChunks * kChunkSplats;
    settings.bytesPerFrame = p.settings.format.chunkBytes();   // a page a frame: an epoch takes frames
    settings.stagingChunks = 3;
    auto pool = lod::StreamingPool::open(device, std::unique_ptr<lod::ChunkSource>(std::move(*source)), settings);
    if (!pool) FAIL(pool.error().toString());

    // The producer: every chunk of every epoch, a replace or a remove, with
    // a pause between pages so that epochs arrive while frames are drawn.
    std::atomic<bool> produced{false};
    std::thread producer([&] {
        for (uint64_t e = 1; e <= kEpochs; ++e) {
            for (uint32_t c = 0; c < kChunks; ++c) {
                const auto* bytes = states[e][c];
                const auto op = bytes == nullptr ? lod::PageUpdate::Op::Remove
                                                 : (e == 1 ? lod::PageUpdate::Op::Add : lod::PageUpdate::Op::Replace);
                queue->push(page(p.settings.stream, e, c, op, bytes, kChunks, p.a->bounds(c)));
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            }
        }
        produced = true;
    });

    std::map<uint64_t, int> framesOf;   // epoch shown: frames that showed it
    int mismatched = 0;
    int building = 0;                   // frames drawn while an epoch was being built
    uint64_t previous = 0;
    bool ordered = true;
    for (int frame = 0; frame < 3000; ++frame) {
        auto placed = (*pool)->update(false);
        if (!placed) FAIL(placed.error().toString());
        const auto s = (*pool)->status();
        const Cut cut = renderPool(*h, (*pool)->cloud(), 0.0F);
        const auto diff = compare(*h, references[s.epoch], cut.targets);
        mismatched += diff.max == 0 ? 0 : 1;
        ++framesOf[s.epoch];
        building += s.pendingEpochs > 0 ? 1 : 0;
        ordered = ordered && s.epoch >= previous;
        previous = s.epoch;
        if (produced && s.epoch == kEpochs && s.pendingEpochs == 0) {
            break;
        }
    }
    producer.join();
    const auto s = (*pool)->status();
    const auto stats = live->stats();
    std::printf("  %zu epochs shown over %d frames, %d of them while one was built; %d mismatched; "
                "%llu pages, %llu dropped\n",
                framesOf.size(), [&] { int n = 0; for (auto& [e, f] : framesOf) n += f; return n; }(), building,
                mismatched, static_cast<unsigned long long>(stats.pages),
                static_cast<unsigned long long>(stats.dropped));
    CHECK(mismatched == 0);
    CHECK(ordered);
    CHECK(s.epoch == kEpochs);
    CHECK(s.epochs == kEpochs);
    CHECK(building > 0);
    CHECK(framesOf.size() >= 3);
    CHECK(stats.pages == kEpochs * kChunks);
    CHECK(stats.dropped == 0);
    CHECK(stats.refused == 0);
    std::filesystem::remove(p.pathA);
    std::filesystem::remove(p.pathB);
}

TEST_CASE("a live stream's replaces and removes change only their chunks", "[lod][gpu][stream][live]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    Pages p = pages(*h);
    gpu::Device& device = *gpu->device;
    const scene::Bounds bounds = p.a->bounds(0);
    const auto push = [&](lod::PageQueue& q, uint64_t epoch, std::vector<std::pair<uint32_t, const std::vector<std::byte>*>> changes,
                          bool add = false) {
        for (const auto& [chunk, bytes] : changes) {
            const auto op = bytes == nullptr ? lod::PageUpdate::Op::Remove
                                             : (add ? lod::PageUpdate::Op::Add : lod::PageUpdate::Op::Replace);
            q.push(page(p.settings.stream, epoch, chunk, op, bytes, static_cast<uint32_t>(changes.size()), bounds));
        }
    };
    const auto shows = [&](lod::StreamingPool& pool, const State& state) {
        const Cut cut = renderPool(*h, pool.cloud(), 0.0F);
        return compare(*h, renderCloud(*h, uploadedState(device, p.settings.format, state)), cut.targets).max;
    };

    SECTION("replace, remove, add again") {
        auto queue = std::make_shared<lod::PageQueue>();
        auto source = lod::LiveSource::create(queue, p.settings);
        REQUIRE(source);
        lod::LiveSource* live = source->get();
        auto pool = lod::StreamingPool::open(device, std::unique_ptr<lod::ChunkSource>(std::move(*source)),
                                             {2 * kChunks * kChunkSplats, 2});
        if (!pool) FAIL(pool.error().toString());
        State state(kChunks, nullptr);
        CHECK(shows(**pool, state) == 0);   // nothing yet

        std::vector<std::pair<uint32_t, const std::vector<std::byte>*>> all;
        for (uint32_t c = 0; c < kChunks; ++c) {
            all.emplace_back(c, &p.blocksA[c]);
            state[c] = &p.blocksA[c];
        }
        push(*queue, 1, all, true);
        REQUIRE((*pool)->update(true));
        CHECK((*pool)->status().epoch == 1);
        CHECK((*pool)->status().resident == kChunks);
        CHECK(shows(**pool, state) == 0);

        push(*queue, 2, {{2, &p.blocksB[2]}, {4, nullptr}});
        state[2] = &p.blocksB[2];
        state[4] = nullptr;
        REQUIRE((*pool)->update(true));
        CHECK((*pool)->status().epoch == 2);
        CHECK((*pool)->status().resident == kChunks - 1);
        CHECK(shows(**pool, state) == 0);

        push(*queue, 3, {{4, &p.blocksB[4]}, {0, nullptr}}, true);
        state[4] = &p.blocksB[4];
        state[0] = nullptr;
        REQUIRE((*pool)->update(true));
        CHECK(shows(**pool, state) == 0);

        // Many replaces of everything: slots are given back as epochs show,
        // so a store of two epochs never runs out.
        for (uint64_t e = 4; e < 14; ++e) {
            std::vector<std::pair<uint32_t, const std::vector<std::byte>*>> flip;
            for (uint32_t c = 0; c < kChunks; ++c) {
                const auto* b = e % 2 == 0 ? &p.blocksA[c] : &p.blocksB[c];
                flip.emplace_back(c, b);
                state[c] = b;
            }
            push(*queue, e, flip);
            auto placed = (*pool)->update(true);
            if (!placed) FAIL(placed.error().toString());
        }
        CHECK((*pool)->status().epoch == 13);
        CHECK(shows(**pool, state) == 0);

        // An epoch that never comes whole is dropped when a later one does,
        // and its stray pages change nothing.
        push(*queue, 14, {{1, &p.blocksA[1]}});   // says one page: whole
        queue->push(page(p.settings.stream, 15, 3, lod::PageUpdate::Op::Remove, nullptr, 2, bounds));   // 1 of 2
        push(*queue, 16, {{5, nullptr}});
        state[1] = &p.blocksA[1];
        state[5] = nullptr;
        REQUIRE((*pool)->update(true));
        CHECK((*pool)->status().epoch == 16);
        CHECK(live->stats().dropped == 1);
        queue->push(page(p.settings.stream, 15, 0, lod::PageUpdate::Op::Remove, nullptr, 2, bounds));   // too late
        REQUIRE((*pool)->update(true));
        CHECK(live->stats().late == 1);
        CHECK(shows(**pool, state) == 0);

        // A page of another format or chunk is refused, not shown.
        lod::PageUpdate bad = page(p.settings.stream, 17, 0, lod::PageUpdate::Op::Replace, &p.blocksA[0], 1, bounds);
        bad.bytes.resize(bad.bytes.size() - 4);
        CHECK(!lod::LiveSource::check(bad, p.settings));
        queue->push(bad);
        REQUIRE((*pool)->update(true));
        CHECK(live->stats().refused == 1);
        CHECK((*pool)->status().epoch == 16);
    }

    SECTION("a store without room for an epoch beside the one shown says so") {
        auto queue = std::make_shared<lod::PageQueue>();
        auto source = lod::LiveSource::create(queue, p.settings);
        REQUIRE(source);
        auto pool = lod::StreamingPool::open(device, std::unique_ptr<lod::ChunkSource>(std::move(*source)),
                                             {kChunks * kChunkSplats, 2});
        if (!pool) FAIL(pool.error().toString());
        std::vector<std::pair<uint32_t, const std::vector<std::byte>*>> all;
        for (uint32_t c = 0; c < kChunks; ++c) {
            all.emplace_back(c, &p.blocksA[c]);
        }
        push(*queue, 1, all, true);
        REQUIRE((*pool)->update(true));
        std::vector<std::pair<uint32_t, const std::vector<std::byte>*>> flip;
        for (uint32_t c = 0; c < kChunks; ++c) {
            flip.emplace_back(c, &p.blocksB[c]);
        }
        push(*queue, 2, flip);
        auto refused = (*pool)->update(true);
        REQUIRE(!refused);
        CHECK(refused.error().code() == ErrorCode::OutOfMemory);
        CHECK((*pool)->status().epoch == 1);
    }
    std::filesystem::remove(p.pathA);
    std::filesystem::remove(p.pathB);
}
