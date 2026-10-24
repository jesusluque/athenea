// Copyright (c) 2026 jesus luque.
//
// A live stream whose pages are SPZ files: each decompressed by a loader
// thread, turned into the store's layout by the device, and shown a whole
// epoch at a time -- the same image the SPZ files draw when loaded alone.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
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
namespace fs = std::filesystem;

namespace {

/// An SPZ file of `count` random degree-3 gaussians, as bytes.
std::vector<uint8_t> spzPage(uint32_t count, uint64_t seed) {
    test::CloudBuilder built = test::randomCloud(count, seed, 0.02F, 0.3F);
    const auto& r = built.raw;
    const uint32_t stride = r.encoding.floatsPerRecord;
    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(r.count);
    cloud.shDegree = 0;
    cloud.antialiased = true;
    for (uint32_t i = 0; i < r.count; ++i) {
        const float* f = r.records.data() + size_t{i} * stride;
        cloud.positions.insert(cloud.positions.end(), {f[0], f[1], f[2]});
        cloud.alphas.push_back(f[3]);
        cloud.scales.insert(cloud.scales.end(), {f[4], f[5], f[6]});
        cloud.rotations.insert(cloud.rotations.end(), {f[8], f[9], f[10], f[7]});
        cloud.colors.insert(cloud.colors.end(), {f[11], f[12], f[13]});
    }
    spz::PackOptions options;
    options.version = 4;
    options.from = spz::CoordinateSystem::RDF;
    std::vector<uint8_t> bytes;
    REQUIRE(spz::saveSpz(cloud, options, &bytes));
    return bytes;
}

}   // namespace

TEST_CASE("a live stream of SPZ pages shows, epoch by epoch, what its SPZ files draw", "[lod][gpu][stream][live][spz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    auto cut = lod::CutSelector::create(*gpu->library);
    REQUIRE(cut);
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Projection projection = render::projectionFor(
        render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0}), settings.width, settings.height);

    constexpr uint32_t kChunks = 3;
    constexpr uint32_t kChunkSplats = 600;
    std::vector<std::vector<uint8_t>> pagesA, pagesB;
    std::vector<scene::GpuSplats> cloudsA, cloudsB;
    for (uint32_t c = 0; c < kChunks; ++c) {
        pagesA.push_back(spzPage(500 + 30 * c, 100 + c));
        pagesB.push_back(spzPage(450 + 20 * c, 200 + c));
        for (auto* pair : {&pagesA, &pagesB}) {
            auto raw = io::readSpz(std::span<const uint8_t>(pair->back()), "page");
            REQUIRE(raw);
            auto cloud = loader->upload(*raw, 3);
            REQUIRE(cloud);
            (pair == &pagesA ? cloudsA : cloudsB).push_back(std::move(*cloud));
        }
    }
    // What a state draws: its chunks' clouds, one instance each, in chunk order.
    const auto reference = [&](const std::vector<const scene::GpuSplats*>& state) {
        std::vector<render::SplatInstance> instances;
        for (const auto* c : state) {
            if (c != nullptr) instances.push_back({c, render::Mat4::identity()});
        }
        render::RenderTargets t;
        REQUIRE(raster->render(projection, instances, settings, t));
        return t;
    };
    const auto shown = [&](const lod::LodCloud& cloud) {
        const std::vector<lod::LodInstance> instances{{&cloud, render::Mat4::identity()}};
        auto selected = cut->select(projection, instances, 0.0F);
        if (!selected) FAIL(selected.error().toString());
        render::RenderTargets t;
        REQUIRE(raster->render(projection, *selected, settings, t));
        return t;
    };
    const auto compare = [&](const render::RenderTargets& a, const render::RenderTargets& b) {
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(diff);
        return *diff;
    };

    lod::LiveSettings live;
    live.stream = 3;
    live.chunks = kChunks;
    live.format.payload = lod::ChunkFormat::Payload::Spz;
    live.format.chunkSplats = kChunkSplats;
    live.format.restPerColour = cloudsA.front().restPerColour;
    live.format.shWords = cloudsA.front().shWords;
    live.name = "spz-live";
    auto queue = std::make_shared<lod::PageQueue>();
    auto source = lod::LiveSource::create(queue, live);
    REQUIRE(source);
    lod::StreamingSettings streaming;
    streaming.budgetSplats = 2 * kChunks * kChunkSplats;
    streaming.library = gpu->library.get();
    auto pool = lod::StreamingPool::open(device, std::unique_ptr<lod::ChunkSource>(std::move(*source)), streaming);
    if (!pool) FAIL(pool.error().toString());

    const auto push = [&](uint64_t epoch, std::vector<std::pair<uint32_t, const std::vector<uint8_t>*>> pages) {
        for (const auto& [chunk, bytes] : pages) {
            lod::PageUpdate u;
            u.stream = live.stream;
            u.epoch = epoch;
            u.chunk = chunk;
            u.op = bytes == nullptr ? lod::PageUpdate::Op::Remove : lod::PageUpdate::Op::Replace;
            if (bytes != nullptr) {
                u.bytes.resize(bytes->size());
                std::memcpy(u.bytes.data(), bytes->data(), bytes->size());
            }
            u.pages = static_cast<uint32_t>(pages.size());
            queue->push(std::move(u));
        }
    };

    push(1, {{0, &pagesA[0]}, {1, &pagesA[1]}, {2, &pagesA[2]}});
    REQUIRE((*pool)->update(true));
    CHECK((*pool)->status().epoch == 1);
    const auto first = compare(reference({&cloudsA[0], &cloudsA[1], &cloudsA[2]}), shown((*pool)->cloud()));
    REQUIRE((*pool)->cloud().splats.antialiased.has_value());

    push(2, {{1, &pagesB[1]}, {2, nullptr}});
    REQUIRE((*pool)->update(true));
    CHECK((*pool)->status().epoch == 2);
    const auto second = compare(reference({&cloudsA[0], &cloudsB[1], nullptr}), shown((*pool)->cloud()));

    push(3, {{2, &pagesB[2]}, {0, &pagesB[0]}});
    REQUIRE((*pool)->update(true));
    const auto third = compare(reference({&cloudsB[0], &cloudsB[1], &cloudsB[2]}), shown((*pool)->cloud()));
    std::printf("  SPZ pages against their files: epoch 1 max %u, epoch 2 max %u, epoch 3 max %u\n", first.max,
                second.max, third.max);
    CHECK(first.max == 0);
    CHECK(second.max == 0);
    CHECK(third.max == 0);
    CHECK((*pool)->cloud().chunkCount(1) == 450 + 20);

    // A page that is no SPZ at all: its epoch is dropped and counted, the one
    // shown stays, and the next epoch is shown.
    std::vector<uint8_t> corrupt(pagesA[0].size(), 0x5A);
    push(4, {{0, &corrupt}, {1, &pagesA[1]}});
    REQUIRE((*pool)->update(true));
    CHECK((*pool)->status().epoch == 3);
    CHECK((*pool)->status().droppedCorrupt == 1);
    CHECK(compare(reference({&cloudsB[0], &cloudsB[1], &cloudsB[2]}), shown((*pool)->cloud())).max == 0);
    push(5, {{1, &pagesA[1]}});
    REQUIRE((*pool)->update(true));
    CHECK((*pool)->status().epoch == 5);
    CHECK(compare(reference({&cloudsB[0], &cloudsA[1], &cloudsB[2]}), shown((*pool)->cloud())).max == 0);
}
