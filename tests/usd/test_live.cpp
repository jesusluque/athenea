// Copyright (c) 2026 jesus luque.
//
// A live stream named by a stage (AtheneaLiveSourceAPI) through Hydra: a
// producer thread pushes SPZ pages, epoch by epoch, onto the stream's queue
// while the stage is drawn frame after frame; every frame is one whole epoch
// -- the clouds of its pages drawn directly -- and the last epoch is reached.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "load-spz.h"
#include "athenea/io/Readers.h"
#include "athenea/lod/SplatStream.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

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

TEST_CASE("a live stream a stage names is drawn through Hydra a whole epoch at a time", "[usd][gpu][live]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr uint32_t kChunks = 3;
    constexpr uint32_t kEpochs = 5;
    const std::string stream = "athenea-test-live-hydra";
    const fs::path dir = fs::temp_directory_path() / "athenea-tests";
    fs::create_directories(dir);
    const fs::path shot = dir / "live-shot.usda";
    {
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def ParticleField3DGaussianSplat \"Live\" (\n"
               "        prepend apiSchemas = [\"AtheneaLiveSourceAPI\"]\n    )\n    {\n"
               "        string primvars:athenea:live:stream = \"" << stream << "\" ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:chunks = " << kChunks << " ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:chunkSplats = 600 ( interpolation = \"constant\" )\n"
               "        int primvars:athenea:live:degree = 0 ( interpolation = \"constant\" )\n"
               "        float3[] extent = [(-5, -4, -3), (5, 4, 3)]\n"
               "    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.3, 0.4, 6.0)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    // Epoch e: chunk c is page (e, c), and chunk (e mod 3) is left out.
    std::vector<std::vector<std::vector<uint8_t>>> pages(kEpochs + 1);
    std::vector<std::vector<scene::GpuSplats>> clouds(kEpochs + 1);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    for (uint32_t e = 1; e <= kEpochs; ++e) {
        for (uint32_t c = 0; c < kChunks; ++c) {
            pages[e].push_back(spzPage(400 + 50 * c, 1000 * e + c));
            auto raw = io::readSpz(std::span<const uint8_t>(pages[e].back()), "page");
            REQUIRE(raw);
            auto cloud = loader->upload(*raw, 3);
            REQUIRE(cloud);
            clouds[e].push_back(std::move(*cloud));
        }
    }

    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.3, 0.4, 6.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);
    std::vector<render::RenderTargets> references(kEpochs + 1);
    for (uint32_t e = 0; e <= kEpochs; ++e) {
        std::vector<render::SplatInstance> instances;
        for (uint32_t c = 0; e > 0 && c < kChunks; ++c) {
            if (c != e % kChunks) instances.push_back({&clouds[e][c], render::Mat4::identity()});
        }
        REQUIRE(raster->render(projection, instances, settings, references[e]));
    }

    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());
    // Before any epoch: the prim's authored extent is the stream's bounds.
    {
        auto image = (*renderer)->render("/World/Shot", 0.0, settings.width, settings.height);
        REQUIRE(image);
        auto bounds = (*renderer)->bounds();
        REQUIRE(bounds);
        REQUIRE(bounds->has_value());
        CHECK((*bounds)->min == std::array<float, 3>{-5, -4, -3});
        CHECK((*bounds)->max == std::array<float, 3>{5, 4, 3});
    }
    // What each page says it covers: chunk c from (c, 0, 0) to (c + 1, 1, 1).
    const auto pageBounds = [](uint32_t c) {
        scene::Bounds b;
        b.min = {static_cast<float>(c), 0.0F, 0.0F};
        b.max = {static_cast<float>(c) + 1.0F, 1.0F, 1.0F};
        return b;
    };

    auto queue = lod::PageQueue::named(stream);
    std::atomic<bool> produced{false};
    std::thread producer([&] {
        for (uint32_t e = 1; e <= kEpochs; ++e) {
            for (uint32_t c = 0; c < kChunks; ++c) {
                lod::PageUpdate u;
                u.stream = 0;
                u.epoch = e;
                u.chunk = c;
                u.pages = kChunks;
                if (c == e % kChunks) {
                    u.op = lod::PageUpdate::Op::Remove;
                } else {
                    u.op = lod::PageUpdate::Op::Replace;
                    u.bounds = pageBounds(c);
                    u.bytes.resize(pages[e][c].size());
                    std::memcpy(u.bytes.data(), pages[e][c].data(), pages[e][c].size());
                }
                queue->push(std::move(u));
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        }
        produced = true;
    });

    int frames = 0, mismatched = 0;
    uint32_t last = 0, seen = 0;
    bool ordered = true;
    for (; frames < 400; ++frames) {
        auto image = (*renderer)->render("/World/Shot", 0.0, settings.width, settings.height);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(hydra);
        // Which epoch it is: the reference it is closest to, which must be
        // as close as Hydra and a direct draw ever are.
        uint32_t best = 0;
        uint32_t bestMax = UINT32_MAX;
        uint32_t bestP99 = UINT32_MAX;
        for (uint32_t e = 0; e <= kEpochs; ++e) {
            auto diff = render::compareImages(*gpu->library, *hydra, references[e].colour, settings.width,
                                              settings.height);
            REQUIRE(diff);
            if (diff->max < bestMax) {
                best = e;
                bestMax = diff->max;
                bestP99 = diff->p99;
            }
        }
        mismatched += (bestP99 <= 1 && bestMax <= 2) ? 0 : 1;
        ordered = ordered && best >= last;
        last = best;
        seen |= 1u << best;
        if (produced && best == kEpochs) {
            break;
        }
    }
    producer.join();
    std::printf("  %d frames; epochs seen %s%s%s%s%s%s; %d not one whole epoch\n", frames,
                (seen & 1) ? "0 " : "", (seen & 2) ? "1 " : "", (seen & 4) ? "2 " : "", (seen & 8) ? "3 " : "",
                (seen & 16) ? "4 " : "", (seen & 32) ? "5" : "", mismatched);
    CHECK(mismatched == 0);
    CHECK(ordered);
    CHECK(last == kEpochs);
    CHECK(__builtin_popcount(seen) >= 3);
    // Once an epoch is shown, its pages' bounds: the last leaves chunk
    // kEpochs % 3 = 2 out, so chunks 0 and 1, from (0, 0, 0) to (2, 1, 1).
    auto bounds = (*renderer)->bounds();
    REQUIRE(bounds);
    REQUIRE(bounds->has_value());
    CHECK((*bounds)->min == std::array<float, 3>{0, 0, 0});
    CHECK((*bounds)->max == std::array<float, 3>{2, 1, 1});
}
