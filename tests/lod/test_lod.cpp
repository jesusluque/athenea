// Copyright (c) 2026 jesus luque.
//
// Levels of detail, built and cut on the device, checked by renders compared on
// the device.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Exr.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/Athc.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
using test::CloudBuilder;
using test::randomCloud;

namespace {

struct Harness {
    test::Gpu*             gpu;
    scene::CloudLoader     loader;
    render::TileRasterizer raster;
    lod::LodBuilder        builder;
    lod::CutSelector       cut;
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
    return std::unique_ptr<Harness>(
        new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*builder), std::move(*cut)});
}

render::ImageDifference renderBoth(Harness& h, const render::Camera& camera, const scene::GpuSplats& full,
                                   const lod::LodCloud& lod, float threshold, const render::RenderSettings& settings,
                                   lod::CutStats* stats = nullptr) {
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    render::RenderTargets a, b;
    REQUIRE(h.raster.render(projection, std::vector<render::SplatInstance>{{&full, render::Mat4::identity()}},
                            settings, a));
    std::vector<lod::CutStats> cutStats;
    const std::vector<lod::LodInstance> instances{{&lod, render::Mat4::identity()}};
    auto selected = h.cut.select(projection, instances, threshold, &cutStats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(h.raster.render(projection, *selected, settings, b));
    if (stats != nullptr && !cutStats.empty()) {
        *stats = cutStats.front();
    }
    if (const char* dump = std::getenv("ATHENEA_TEST_DUMP"); dump != nullptr) {
        static int serial = 0;
        ++serial;
        for (const auto& [t, name] : {std::pair{&a, "full"}, std::pair{&b, "lod"}}) {
            auto c = t->colour.readAll<float>(*h.gpu->device);
            REQUIRE(c);
            REQUIRE(io::writeExr(std::string(dump) + "/lod" + std::to_string(serial) + "_" + name + ".exr",
                                 settings.width, settings.height, *c));
        }
    }
    auto diff = render::compareImages(*h.gpu->library, a.colour, b.colour, settings.width, settings.height);
    REQUIRE(diff);
    return *diff;
}

}   // namespace

TEST_CASE("a cut with threshold zero draws the cloud as it is", "[lod][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(5000, 41);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    CHECK(!lod->levels.empty());
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    lod::CutStats stats;
    const auto diff = renderBoth(*h, camera, *cloud, *lod, 0.0F, settings, &stats);
    std::printf("  threshold 0: %u splats, %u merged; p99 %u, max %u\n", stats.splats, stats.merged, diff.p99, diff.max);
    CHECK(stats.splats == cloud->count);
    CHECK(stats.merged == 0);
    // Not bit-identical: the cut draws the splats in Morton order, and splats
    // whose 24-bit depth keys tie (nearer than 3e-5 relative -- common among
    // 5000 large splats in a few units of depth) keep index order, which is
    // now a different order. Measured: p99 1, 0.6% of pixels over 2.
    CHECK(diff.p99 <= 1);
    CHECK(diff.over2 * 100 <= diff.pixels);
}

TEST_CASE("a cell holding one splat merges into that splat", "[lod][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // Splats far enough apart that every level-1 cell holds at most one: each
    // merged Gaussian is its single splat's moments, turned back into a splat.
    CloudBuilder b;
    const std::array<std::array<float, 3>, 6> at{{{-2, -2, -2}, {2, -2, -2}, {-2, 2, -2}, {2, 2, 2}, {-2, 2, 2}, {2, -2, 2}}};
    test::Lcg rng;
    for (const auto& p : at) {
        // In names, in order: a call's arguments are evaluated in no order
        // the language fixes (SplatFixtures.h says what that did).
        const float opacity = rng.range(0.3F, 0.95F);
        const float sx = rng.range(0.1F, 0.5F);
        const float sy = rng.range(0.1F, 0.5F);
        const float sz = rng.range(0.1F, 0.5F);
        const std::array<float, 4> turn{rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1), rng.range(-1, 1)};
        const std::array<float, 3> dc{rng.range(0.1F, 1.0F), rng.range(0.1F, 1.0F), rng.range(0.1F, 1.0F)};
        b.add(p[0], p[1], p[2], opacity, sx, sy, sz, {turn[0], turn[1], turn[2], turn[3]}, {dc[0], dc[1], dc[2]},
              {0.2F, -0.1F, 0.3F, 0.0F, 0.1F, -0.2F, 0.1F, 0.0F, -0.3F});
    }
    auto cloud = h->loader.upload(b.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings one;
    one.coarsestLevel = 1;
    one.maxGroupFraction = 1.0F;   // keep level 1 though it merges nothing
    auto lod = h->builder.build(*cloud, one);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(!lod->levels.empty());
    CHECK(lod->levels.front().gaussians.count == 6);
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    render::Camera camera = render::Camera::lookingAt({3.0, 2.0, 12.0}, {0.0, 0.0, 0.0});
    lod::CutStats stats;
    // A threshold every cell meets: everything is drawn merged, at level 1.
    const auto diff = renderBoth(*h, camera, *cloud, *lod, 1e6F, settings, &stats);
    std::printf("  single-splat cells: %u splats, %u merged; p99 %u, max %u\n", stats.splats, stats.merged,
                diff.p99, diff.max);
    CHECK(stats.splats == 0);
    CHECK(stats.merged == 6);
    // The same covariance to 1e-4 (the 10-bit quaternion); what differs is
    // the outermost ring of pixels, where a coverage of 1/255 more or less is
    // a dozen code values once encoded.
    CHECK(diff.p99 <= 1);
    CHECK(diff.over2 * 100 <= diff.pixels);
}

TEST_CASE("far away, a dense cloud draws a fraction of its splats and looks the same", "[lod][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(60000, 43, 0.003F, 0.05F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::Camera camera = render::Camera::lookingAt({6.0, 4.0, 30.0}, {0.0, 0.0, 0.0});
    for (const float threshold : {1.0F, 2.0F, 4.0F}) {
        lod::CutStats stats;
        const auto diff = renderBoth(*h, camera, *cloud, *lod, threshold, settings, &stats);
        std::printf("  threshold %.0f px: %u splats + %u merged of %u (%.1f%%); p99 %u, max %u\n", threshold,
                    stats.splats, stats.merged, stats.available,
                    100.0 * (stats.splats + stats.merged) / stats.available, diff.p99, diff.max);
        // Random colours splat to splat are the worst case merging has: every
        // merged Gaussian is an average of colours no pixel showed. Bounds
        // measured on this cloud (p99 0 / 9 / 29 at 1 / 2 / 4 px).
        if (threshold == 1.0F) {
            CHECK(diff.p99 <= 1);
        } else if (threshold == 2.0F) {
            CHECK(stats.splats + stats.merged < stats.available * 3 / 4);
            CHECK(diff.p99 <= 12);
        } else {
            CHECK(stats.splats + stats.merged < stats.available / 4);
            CHECK(diff.p99 <= 36);
        }
    }
}


namespace {

/// The cloud as a stream would hold it: chunk c in slot chunks-1-c (copied on
/// the device), and only the chunks `keep` says on it.
lod::LodCloud streamedCopy(Harness& h, const lod::LodCloud& lod, const std::vector<bool>& keep) {
    gpu::Device& device = *h.gpu->device;
    lod::LodCloud out = lod;
    out.streamed = true;
    const auto like = [&](const gpu::Buffer& b) {
        gpu::BufferDesc desc;
        desc.bytes = b.bytes();
        desc.elementBytes = b.elementBytes();
        desc.label = "test.store";
        auto made = gpu::Buffer::create(device, desc);
        REQUIRE(made);
        return *made;
    };
    out.splats.positions = like(lod.splats.positions);
    out.splats.shape = like(lod.splats.shape);
    out.splats.sh = like(lod.splats.sh);
    out.groups = like(lod.groups);
    std::vector<uint32_t> resident(lod.chunks(), 0);
    gpu::CommandBatch batch(device);
    for (uint32_t c = 0; c < lod.chunks(); ++c) {
        if (!keep[c]) {
            out.slots[c] = -1;
            continue;
        }
        const uint32_t slot = lod.chunks() - 1 - c;
        out.slots[c] = static_cast<int32_t>(slot);
        resident[c] = 1;
        const uint64_t from = uint64_t{c} * lod.chunkSplats;
        const uint64_t to = uint64_t{slot} * lod.chunkSplats;
        const uint64_t n = lod.chunkCount(c);
        const auto copy = [&](const gpu::Buffer& dst, const gpu::Buffer& src, uint64_t perSplat) {
            batch.encoder()->copyBuffer(dst.rhi(), to * perSplat, src.rhi(), from * perSplat, n * perSplat);
        };
        copy(out.splats.positions, lod.splats.positions, 16);
        copy(out.splats.shape, lod.splats.shape, 16);
        copy(out.splats.sh, lod.splats.sh, 4 * uint64_t{lod.splats.shWords});
        copy(out.groups, lod.groups, 4);
    }
    batch.markDirty();
    REQUIRE(batch.submit(true));
    auto flags = gpu::Buffer::fromSpan(device, std::span<const uint32_t>(resident), "test.resident");
    REQUIRE(flags);
    out.resident = *flags;
    return out;
}

struct Cut {
    lod::CutStats           stats;
    render::RenderTargets   targets;
};

Cut cutAndRender(Harness& h, const render::Projection& projection, const lod::LodCloud& lod, float threshold,
                 const render::RenderSettings& settings) {
    Cut cut;
    std::vector<lod::CutStats> stats;
    const std::vector<lod::LodInstance> instances{{&lod, render::Mat4::identity()}};
    auto selected = h.cut.select(projection, instances, threshold, &stats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(stats.size() == 1);
    cut.stats = stats.front();
    REQUIRE(h.raster.render(projection, *selected, settings, cut.targets));
    return cut;
}

}   // namespace

TEST_CASE("chunks not on the device are drawn merged, and chunks not wanted change nothing", "[lod][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(20000, 47, 0.003F, 0.05F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto lod = h->builder.build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(lod->chunks() == 20);
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::Camera camera = render::Camera::lookingAt({4.0, 3.0, 18.0}, {0.0, 0.0, 0.0});
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    const auto compare = [&](const Cut& a, const Cut& b) {
        auto diff = render::compareImages(*gpu->library, a.targets.colour, b.targets.colour, settings.width,
                                          settings.height);
        REQUIRE(diff);
        return *diff;
    };

    SECTION("every chunk there, in other slots: the same cut, the same image") {
        const lod::LodCloud all = streamedCopy(*h, *lod, std::vector<bool>(20, true));
        for (const float threshold : {0.0F, 2.0F, 1e6F}) {
            const Cut memory = cutAndRender(*h, projection, *lod, threshold, settings);
            const Cut stream = cutAndRender(*h, projection, all, threshold, settings);
            const auto diff = compare(memory, stream);
            CHECK(stream.stats.splats == memory.stats.splats);
            CHECK(stream.stats.merged == memory.stats.merged);
            CHECK(diff.max == 0);
            REQUIRE(stream.stats.needs.size() == 20);
            const auto wanted = std::count_if(stream.stats.needs.begin(), stream.stats.needs.end(), [](uint32_t n) { return n != 0; });
            if (threshold == 0.0F) {
                CHECK(wanted == 20);
            } else if (threshold == 1e6F) {
                CHECK(wanted == 0);
                CHECK(stream.stats.splats == 0);
            }
        }
    }

    SECTION("chunks missing: their places drawn merged until they come") {
        std::vector<bool> keep(20, true);
        uint32_t missing = 0;
        for (uint32_t c = 0; c < 20; c += 3) {
            keep[c] = false;
            missing += lod->chunkCount(c);
        }
        const lod::LodCloud partial = streamedCopy(*h, *lod, keep);
        const Cut memory = cutAndRender(*h, projection, *lod, 0.0F, settings);
        const Cut stream = cutAndRender(*h, projection, partial, 0.0F, settings);
        const auto diff = compare(memory, stream);
        std::printf("  %u of %u splats missing: %u splats + %u merged drawn; p99 %u, max %u\n", missing, lod->count,
                    stream.stats.splats, stream.stats.merged, diff.p99, diff.max);
        CHECK(stream.stats.splats <= lod->count - missing);
        CHECK(stream.stats.merged > 0);
        CHECK(std::count_if(stream.stats.needs.begin(), stream.stats.needs.end(), [](uint32_t n) { return n != 0; }) == 20);
        CHECK(diff.max > 0);
    }

    SECTION("dropping the chunks a view does not want leaves it as it was") {
        // Close to one side of the cloud: the near chunks are wanted, the far
        // ones merged away.
        const render::Projection near = render::projectionFor(
            render::Camera::lookingAt({0.0, 0.5, 4.5}, {0.0, 0.0, 0.0}), settings.width, settings.height);
        const lod::LodCloud all = streamedCopy(*h, *lod, std::vector<bool>(20, true));
        for (const float threshold : {36.0F, 48.0F}) {
            const Cut full = cutAndRender(*h, near, all, threshold, settings);
            std::vector<bool> keep(20);
            for (uint32_t c = 0; c < 20; ++c) {
                keep[c] = full.stats.needs[c] != 0;
            }
            const auto wanted = std::count(keep.begin(), keep.end(), true);
            const lod::LodCloud needed = streamedCopy(*h, *lod, keep);
            const Cut trimmed = cutAndRender(*h, near, needed, threshold, settings);
            const auto diff = compare(full, trimmed);
            std::printf("  threshold %.0f px: %ld of 20 chunks wanted; %u splats + %u merged; max %u\n", static_cast<double>(threshold),
                        static_cast<long>(wanted), trimmed.stats.splats, trimmed.stats.merged, diff.max);
            CHECK(wanted < 20);
            CHECK(trimmed.stats.splats == full.stats.splats);
            CHECK(trimmed.stats.merged == full.stats.merged);
            CHECK(diff.max == 0);
        }
    }
}

TEST_CASE("a .athc reads back as it was built, and a stream settles on the same image", "[lod][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(20000, 47, 0.003F, 0.05F);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto lod = h->builder.build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "athenea_test_stream.athc";
    REQUIRE(lod::writeAthc(*gpu->device, *lod, path));
    CHECK(std::filesystem::file_size(path) % 4096 == 0);

    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::Projection near = render::projectionFor(render::Camera::lookingAt({0.0, 0.5, 4.5}, {0.0, 0.0, 0.0}),
                                                    settings.width, settings.height);
    const auto compare = [&](const Cut& a, const Cut& b) {
        auto diff = render::compareImages(*gpu->library, a.targets.colour, b.targets.colour, settings.width,
                                          settings.height);
        REQUIRE(diff);
        return *diff;
    };
    // Cut, tell the pool, let it load; until a frame places nothing more.
    const auto settle = [&](lod::StreamingPool& pool, float threshold, int* rounds) {
        for (*rounds = 1; *rounds <= 8; ++*rounds) {
            Cut cut = cutAndRender(*h, near, pool.cloud(), threshold, settings);
            pool.want(cut.stats.needs);
            auto placed = pool.update(true);
            REQUIRE(placed);
            if (*placed == 0) {
                return cut;
            }
        }
        FAIL("the stream never settled");
        return Cut{};
    };

    SECTION("read whole") {
        auto read = lod::readAthc(*gpu->device, path);
        if (!read) FAIL(read.error().toString());
        CHECK(read->count == lod->count);
        CHECK(read->chunks() == lod->chunks());
        CHECK(read->levels.size() == lod->levels.size());
        for (const float threshold : {0.0F, 36.0F}) {
            const Cut memory = cutAndRender(*h, near, *lod, threshold, settings);
            const Cut file = cutAndRender(*h, near, *read, threshold, settings);
            CHECK(file.stats.splats == memory.stats.splats);
            CHECK(file.stats.merged == memory.stats.merged);
            CHECK(compare(memory, file).max == 0);
        }
    }

    SECTION("streamed with room for all of it") {
        auto pool = lod::StreamingPool::open(*gpu->device, path, {uint64_t{1} << 20, 2});
        if (!pool) FAIL(pool.error().toString());
        for (const float threshold : {0.0F, 36.0F}) {
            int rounds = 0;
            const Cut memory = cutAndRender(*h, near, *lod, threshold, settings);
            const Cut stream = settle(**pool, threshold, &rounds);
            const auto status = (*pool)->status();
            std::printf("  threshold %.0f px: settled in %d rounds, %u of %u chunks on the device\n", static_cast<double>(threshold),
                        rounds, status.resident, lod->chunks());
            CHECK(status.missing == 0);
            CHECK(stream.stats.splats == memory.stats.splats);
            CHECK(stream.stats.merged == memory.stats.merged);
            CHECK(compare(memory, stream).max == 0);
        }
    }

    SECTION("streamed into a store too small for the view, then a view it fits") {
        auto pool = lod::StreamingPool::open(*gpu->device, path, {8000, 2});
        if (!pool) FAIL(pool.error().toString());
        int rounds = 0;
        const Cut memory36 = cutAndRender(*h, near, *lod, 36.0F, settings);
        const Cut tight = settle(**pool, 36.0F, &rounds);
        auto status = (*pool)->status();
        const auto diff36 = compare(memory36, tight);
        std::printf("  8 slots, threshold 36 px: %u chunks missing; %u splats + %u merged (whole: %u + %u); p99 %u\n",
                    status.missing, tight.stats.splats, tight.stats.merged, memory36.stats.splats,
                    memory36.stats.merged, diff36.p99);
        CHECK(status.slots == 8);
        CHECK(status.resident == 8);
        CHECK(status.missing > 0);
        CHECK(tight.stats.splats < memory36.stats.splats);

        const Cut memory48 = cutAndRender(*h, near, *lod, 48.0F, settings);
        const Cut fits = settle(**pool, 48.0F, &rounds);
        status = (*pool)->status();
        std::printf("  8 slots, threshold 48 px: %u missing, %llu loads, %llu evictions\n", status.missing,
                    static_cast<unsigned long long>(status.loads), static_cast<unsigned long long>(status.evictions));
        CHECK(status.missing == 0);
        CHECK(fits.stats.splats == memory48.stats.splats);
        CHECK(fits.stats.merged == memory48.stats.merged);
        CHECK(compare(memory48, fits).max == 0);

        // The other side: other chunks wanted, the store full of chunks that
        // no longer are.
        near = render::projectionFor(render::Camera::lookingAt({0.0, 0.5, -4.5}, {0.0, 0.0, 0.0}), settings.width,
                                     settings.height);
        const uint64_t evictionsBefore = status.evictions;
        const Cut behindMemory = cutAndRender(*h, near, *lod, 48.0F, settings);
        const Cut behind = settle(**pool, 48.0F, &rounds);
        status = (*pool)->status();
        std::printf("  8 slots, from behind: %u missing, %llu evictions\n", status.missing,
                    static_cast<unsigned long long>(status.evictions - evictionsBefore));
        CHECK(status.evictions > evictionsBefore);
        CHECK(status.missing == 0);
        CHECK(behind.stats.splats == behindMemory.stats.splats);
        CHECK(compare(behindMemory, behind).max == 0);
    }
    std::filesystem::remove(path);
}

// FEWER GAUSSIANS, KEPT WHERE THEY MATTER. A floor of discs, 96 by 96, each
// as wide as their spacing -- what a conversion from a mesh gives -- seen from
// above: one colour all over, it takes a ninth and draws the same image,
// which is what the widening of a kept merge is for (the moments alone left
// holes); under a checker of 12-disc squares the edges keep their discs and
// the image stays within a few code values; and with one disc in thirty a
// bright glint, it still merges, less -- a few that differ within a merge's
// reach are noise, which the share of outliers is for, and none may lie in
// its tail.
TEST_CASE("a decimated floor draws as it did, with fewer gaussians where it is the same",
          "[lod][gpu][decimate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    auto decimator = lod::Decimator::create(*gpu->library);
    if (!decimator) FAIL(decimator.error().toString());
    constexpr int kSide = 96;
    constexpr float kSpacing = 2.0F / kSide;
    const auto floor = [&](int pattern) {
        CloudBuilder b;
        test::Lcg lcg{7};
        for (int y = 0; y < kSide; ++y) {
            for (int x = 0; x < kSide; ++x) {
                std::array<float, 3> colour{0.55F, 0.45F, 0.35F};
                // The other colour a hair above the first, so that where the
                // two overlap the same one is on top whatever order the discs
                // are drawn in -- and a merge, which stands at its discs'
                // mean depth, keeps it there. Discs of one colour may tie:
                // which is on top changes nothing.
                float z = 0.0F;
                if (pattern == 1 && ((x / 12 + y / 12) & 1) != 0) {
                    colour = {0.15F, 0.2F, 0.3F};
                    z = 2.0e-3F;
                }
                if (pattern == 2 && lcg.next() < 1.0 / 30.0) {
                    colour = {0.95F, 0.9F, 0.7F};
                    z = 2.0e-3F;
                }
                b.add(-1.0F + (x + 0.5F) * kSpacing, -1.0F + (y + 0.5F) * kSpacing, z, 0.95F, kSpacing,
                      kSpacing, kSpacing * 0.1F, {1.0F, 0.0F, 0.0F, 0.0F}, colour);
            }
        }
        return b;
    };
    // About two and a half discs a pixel: where two colours meet, which disc
    // is drawn last decides a pixel of one disc's width, and a merge changes
    // that order without changing what the floor is.
    render::RenderSettings settings;
    settings.width = 40;
    settings.height = 40;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 2.6}, {0.0, 0.0, 0.0});
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    for (const auto& [pattern, name] : {std::pair{0, "one colour"}, std::pair{1, "a checker"},
                                        std::pair{2, "one glint in thirty"}}) {
        CloudBuilder built = floor(pattern);
        auto cloud = h->loader.upload(built.raw, 0);
        REQUIRE(cloud);
        auto lod = h->builder.build(*cloud);
        if (!lod) FAIL(lod.error().toString());
        lod::DecimateStats stats;
        auto result = decimator->decimate(*lod, cloud->origin, lod::DecimateSettings{}, {}, &stats);
        if (!result) FAIL(result.error().toString());
        const scene::GpuSplats* decimated = &result->cloud;
        render::RenderTargets a, b;
        REQUIRE(h->raster.render(projection, std::vector<render::SplatInstance>{{&*cloud, render::Mat4::identity()}},
                                 settings, a));
        REQUIRE(h->raster.render(projection,
                                 std::vector<render::SplatInstance>{{&*decimated, render::Mat4::identity()}}, settings,
                                 b));
        if (const char* dump = std::getenv("ATHENEA_TEST_DUMP"); dump != nullptr) {
            for (const auto& [t, which] : {std::pair{&a, "full"}, std::pair{&b, "decimated"}}) {
                auto c = t->colour.readAll<float>(*gpu->device);
                REQUIRE(c);
                REQUIRE(io::writeExr(std::string(dump) + "/decimate" + std::to_string(pattern) + "_" + which + ".exr",
                                     settings.width, settings.height, *c));
            }
        }
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(diff);
        const uint32_t kept = stats.splats + stats.merged;
        std::printf("  %-20s %u of %u kept (%u merged); p99 %u, max %u, %.1f%% over 2\n", name, kept, stats.before,
                    stats.merged, diff->p99, diff->max, 100.0 * double(diff->over2) / double(diff->pixels));
        CHECK(stats.before == uint32_t{kSide * kSide});
        // Measured: 1024 of 9216 at p99 1; 6408 at p99 6, the checker's edges
        // drawn a shade differently where the merges beside them end; 6541 at
        // p99 1, since a glint in a merge's tail stops the merge.
        if (pattern == 0) {
            CHECK(kept * 4 < stats.before);
            CHECK(diff->p99 <= 2);
        } else if (pattern == 1) {
            CHECK(kept * 4 < stats.before * 3);
            CHECK(diff->p99 <= 8);
        } else {
            CHECK(kept * 5 < stats.before * 4);
            CHECK(diff->p99 <= 2);
        }
    }
}

// WHAT A CLOUD CARRIES GOES WITH ITS GAUSSIANS. A floor of one colour whose
// left half is one prim (id 7), carried by joint 1, with one pattern of bits,
// and whose right half is another (id 9, joint 2, another pattern), and a
// float that is each record's own x. Decimated with the ids and the rig in
// its key, it still merges -- it is one colour -- but never across the
// middle: every kept gaussian has its side's id, is carried wholly by its
// side's joint, keeps its side's bits, and its float is where it stands.
TEST_CASE("a decimation carries ids, rigs, bits and floats with the gaussians, and merges none across",
          "[lod][gpu][decimate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    auto decimator = lod::Decimator::create(*gpu->library);
    if (!decimator) FAIL(decimator.error().toString());
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/decimate_check", "decimateCheck");
    if (!check) FAIL(check.error().toString());
    constexpr int kSide = 64;
    constexpr float kSpacing = 2.0F / kSide;
    CloudBuilder built;
    std::vector<uint32_t> xs, ids, joints, weights, bits;
    const auto word = [](float f) {
        uint32_t w;
        std::memcpy(&w, &f, 4);
        return w;
    };
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const float px = -1.0F + (x + 0.5F) * kSpacing;
            built.add(px, -1.0F + (y + 0.5F) * kSpacing, 0.0F, 0.95F, kSpacing, kSpacing, kSpacing * 0.1F,
                      {1.0F, 0.0F, 0.0F, 0.0F}, {0.5F, 0.4F, 0.3F});
            const bool left = x < kSide / 2;
            xs.push_back(word(px));
            ids.push_back(left ? 7u : 9u);
            // The side's joint first, then one that carries nothing.
            for (const uint32_t j : {left ? 1u : 2u, 5u, 6u, 7u}) {
                joints.push_back(j);
            }
            for (const float w : {1.0F, 0.0F, 0.0F, 0.0F}) {
                weights.push_back(word(w));
            }
            bits.push_back(left ? 0xF0F0F0F0u : 0x0F0F0F0Fu);
            bits.push_back(left ? 1u : 2u);
        }
    }
    const uint32_t records = kSide * kSide;
    const auto upload = [&](const std::vector<uint32_t>& v, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = v.size() * 4;
        desc.elementBytes = 4;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, v.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer xBuffer = upload(xs, "check.x");
    const gpu::Buffer idBuffer = upload(ids, "check.ids");
    const gpu::Buffer jointBuffer = upload(joints, "check.joints");
    const gpu::Buffer weightBuffer = upload(weights, "check.weights");
    const gpu::Buffer bitBuffer = upload(bits, "check.bits");
    gpu::Buffer keys = upload(std::vector<uint32_t>(records, 0u), "check.keys");
    REQUIRE(decimator->addKey(keys, records, idBuffer, 1, lod::AttributeMerge::First));
    REQUIRE(decimator->addKey(keys, records, jointBuffer, 4, lod::AttributeMerge::Rig, &weightBuffer));

    auto cloud = h->loader.upload(built.raw, 0);
    REQUIRE(cloud);
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    lod::DecimateCarried carried;
    carried.keys = &keys;
    lod::DecimateStats stats;
    auto kept = decimator->decimate(*lod, cloud->origin, lod::DecimateSettings{}, carried, &stats);
    if (!kept) FAIL(kept.error().toString());
    const uint32_t count = kept->cloud.count;
    std::printf("  %u of %u kept, %u merged\n", count, stats.before, stats.merged);
    CHECK(stats.merged > 0);
    CHECK(count * 2 < records);

    auto meanX = decimator->mergeAttribute(*lod, *kept, xBuffer, 1, lod::AttributeMerge::Mean);
    auto keptIds = decimator->mergeAttribute(*lod, *kept, idBuffer, 1, lod::AttributeMerge::First);
    auto rig = decimator->mergeAttribute(*lod, *kept, jointBuffer, 4, lod::AttributeMerge::Rig, &weightBuffer);
    auto keptBits = decimator->mergeAttribute(*lod, *kept, bitBuffer, 2, lod::AttributeMerge::Bits);
    REQUIRE(meanX);
    REQUIRE(keptIds);
    REQUIRE(rig);
    REQUIRE(keptBits);
    gpu::Buffer counts = upload(std::vector<uint32_t>(8, 0u), "check.counts");
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["positions"].setBinding(kept->cloud.positions.rhi());
            cursor["meanX"].setBinding(meanX->first.rhi());
            cursor["ids"].setBinding(keptIds->first.rhi());
            cursor["joints"].setBinding(rig->first.rhi());
            cursor["weights"].setBinding(rig->second.rhi());
            cursor["bits"].setBinding(keptBits->first.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["tolerance"].setData(1.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    auto violations = counts.readAll<uint32_t>(*gpu->device);
    REQUIRE(violations);
    std::printf("  off their mean %u, another id %u, another joint %u, other bits %u\n", (*violations)[0],
                (*violations)[1], (*violations)[2], (*violations)[3]);
    CHECK((*violations)[0] == 0);
    CHECK((*violations)[1] == 0);
    CHECK((*violations)[2] == 0);
    CHECK((*violations)[3] == 0);
}

// A MERGE OF SHADING NORMALS IS A NORMAL BETWEEN THEM.
//
// A converted cloud keeps a shading normal apart from each gaussian's frame
// (GpuSplats::normals). The levels of detail merge it as they merge a colour
// -- the sum weighed by what each covers, made unit again -- and the cut
// hands it to the frame with the rest; a decimation merges the file's normal
// as a direction (AttributeMerge::Normal), not as three floats averaged,
// which are shorter than one wherever two normals disagree. A floor whose
// normals alternate between two tilts, cell by cell, has every merge mix the
// two, so every merged normal must be a unit vector on the arc between them.
TEST_CASE("merged shading normals stay unit vectors between the ones they merge", "[lod][gpu][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    auto inArc = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalsInArc");
    auto unit = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalFloatsUnit");
    auto decimator = lod::Decimator::create(*gpu->library);
    if (!inArc) FAIL(inArc.error().toString());
    if (!unit) FAIL(unit.error().toString());
    if (!decimator) FAIL(decimator.error().toString());
    constexpr int kSide = 64;
    constexpr float kSpacing = 2.0F / kSide;
    const std::array<float, 3> n1{0.6F, 0.0F, 0.8F};
    const std::array<float, 3> n2{0.0F, 0.6F, 0.8F};
    CloudBuilder built;
    std::vector<float> normals;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            built.add(-1.0F + (x + 0.5F) * kSpacing, -1.0F + (y + 0.5F) * kSpacing, 0.0F, 0.95F, kSpacing, kSpacing,
                      kSpacing * 0.1F, {1.0F, 0.0F, 0.0F, 0.0F}, {0.5F, 0.4F, 0.3F});
            const std::array<float, 3>& n = (x + y) % 2 == 0 ? n1 : n2;
            normals.insert(normals.end(), n.begin(), n.end());
        }
    }
    test::giveNormals(built, normals);
    auto cloud = h->loader.upload(built.raw, 0);
    REQUIRE(cloud);
    REQUIRE(cloud->hasNormals());
    auto lod = h->builder.build(*cloud);
    if (!lod) FAIL(lod.error().toString());
    REQUIRE(lod->splats.hasNormals());

    gpu::BufferDesc countsDesc;
    countsDesc.bytes = 8 * 4;
    countsDesc.elementBytes = 4;
    countsDesc.label = "normals.counts";
    auto counts = gpu::Buffer::create(*gpu->device, countsDesc);
    REQUIRE(counts);
    const auto arcCheck = [&](const gpu::Buffer& those, uint32_t count) {
        const uint32_t zero[8] = {};
        REQUIRE(counts->write(*gpu->device, 0, sizeof(zero), zero));
        gpu::CommandBatch batch(*gpu->device);
        inArc->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["normalsA"].setBinding(those.rhi());
            cursor["normalsB"].setBinding(those.rhi());
            cursor["floats"].setBinding(those.rhi());
            cursor["counts"].setBinding(counts->rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            const std::array<float, 4> a{n1[0], n1[1], n1[2], 0.0F};
            const std::array<float, 4> b{n2[0], n2[1], n2[2], 0.0F};
            p["n1"].setData(a.data(), 16);
            p["n2"].setData(b.data(), 16);
            p["tolerance"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts->read(*gpu->device, 0, sizeof(seen), seen));
        return std::pair{seen[0], seen[1]};
    };
    for (const lod::LodLevel& level : lod->levels) {
        REQUIRE(level.gaussians.hasNormals());
        const auto [compared, outside] = arcCheck(level.gaussians.normals, level.gaussians.count);
        std::printf("  level %u: %u merged normals, %u off the arc\n", level.level, compared, outside);
        CHECK(compared == level.gaussians.count);
        CHECK(outside == 0);
    }

    // And the cut draws them: merged where the floor is far, the cloud's own
    // where it is near, all of them carried into the frame's cloud.
    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 120;
    const render::Projection far = render::projectionFor(
        render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0}), settings.width, settings.height);
    const std::vector<lod::LodInstance> instances{{&*lod, render::Mat4::identity()}};
    std::vector<lod::CutStats> stats;
    auto selected = h->cut.select(far, instances, 8.0F, &stats);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(selected->size() == 1);
    const scene::GpuSplats& drawn = *selected->front().splats;
    REQUIRE(drawn.hasNormals());
    const auto [cutCompared, cutOutside] = arcCheck(drawn.normals, drawn.count);
    std::printf("  the cut: %u drawn (%u merged), %u off the arc\n", cutCompared, stats.front().merged, cutOutside);
    CHECK(stats.front().merged > 0);
    CHECK(cutOutside == 0);

    // A decimation merges the file's own normal: three floats a record.
    auto kept = decimator->decimate(*lod, cloud->origin);
    if (!kept) FAIL(kept.error().toString());
    std::vector<uint32_t> words(normals.size());
    std::memcpy(words.data(), normals.data(), normals.size() * 4);
    auto values = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, words, "normals.values");
    REQUIRE(values);
    const auto unitCheck = [&](lod::AttributeMerge how) {
        auto merged = decimator->mergeAttribute(*lod, *kept, *values, 3, how);
        REQUIRE(merged);
        const uint32_t zero[8] = {};
        REQUIRE(counts->write(*gpu->device, 0, sizeof(zero), zero));
        gpu::CommandBatch batch(*gpu->device);
        unit->dispatch(batch, {kept->cloud.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["normalsA"].setBinding(merged->first.rhi());
            cursor["normalsB"].setBinding(merged->first.rhi());
            cursor["floats"].setBinding(merged->first.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["count"].setData(kept->cloud.count);
            cursor["params"]["width"].setData(3u);
            cursor["params"]["tolerance"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts->read(*gpu->device, 0, sizeof(seen), seen));
        return std::pair{seen[0], seen[1]};
    };
    const auto [asNormals, notUnit] = unitCheck(lod::AttributeMerge::Normal);
    const auto [asFloats, shortened] = unitCheck(lod::AttributeMerge::Mean);
    std::printf("  decimated: %u kept, %u not unit merged as normals, %u of %u shorter merged as three floats\n",
                asNormals, notUnit, shortened, asFloats);
    CHECK(asNormals == kept->cloud.count);
    CHECK(notUnit == 0);
    // What the test tells apart: averaged as floats, they are not.
    CHECK(shortened > 0);
}
