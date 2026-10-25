// Copyright (c) 2026 jesus luque.
//
// A live stream fed from outside: a producer that speaks only JSON-RPC to the
// server and writes into shared memory (stream_open, stream_commit) -- as a
// process of its own would -- while a stage that names the stream is drawn
// frame after frame. Every frame is one whole epoch.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include <nlohmann/json.hpp>

#include "load-spz.h"
#include "athenea/core/Platform.h"
#include "athenea/io/Readers.h"
#include "athenea/mcp/Server.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
using nlohmann::json;
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

json call(mcp::Server& server, const std::string& tool, const json& arguments) {
    const auto reply = server.handle(json{{"jsonrpc", "2.0"},
                                          {"id", 3},
                                          {"method", "tools/call"},
                                          {"params", {{"name", tool}, {"arguments", arguments}}}}
                                         .dump());
    REQUIRE(reply.has_value());
    return json::parse(*reply);
}

std::string textOf(const json& answer) {
    std::string text;
    for (const json& block : answer["result"]["content"]) {
        text += block.value("text", std::string{});
    }
    return text;
}

}   // namespace

TEST_CASE("a producer that speaks JSON-RPC and shared memory feeds a stage whole epochs", "[mcp][gpu][live]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr uint32_t kChunks = 2;
    constexpr uint32_t kEpochs = 4;
    const std::string stream = "athenea-test-live-mcp";
    const fs::path dir = fs::temp_directory_path() / "athenea-mcp-tests";
    fs::create_directories(dir);
    const fs::path shot = dir / "live-mcp.usda";
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
               "    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.3, 0.4, 6.0)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    std::vector<std::vector<std::vector<uint8_t>>> pages(kEpochs + 1);
    std::vector<std::vector<scene::GpuSplats>> clouds(kEpochs + 1);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    for (uint32_t e = 1; e <= kEpochs; ++e) {
        for (uint32_t c = 0; c < kChunks; ++c) {
            pages[e].push_back(spzPage(500 + 40 * c, 77 * e + c));
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
            instances.push_back({&clouds[e][c], render::Mat4::identity()});
        }
        REQUIRE(raster->render(projection, instances, settings, references[e]));
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());

    // The producer: a server of its own, as another process would have, and
    // the ring it is given.
    mcp::Server server;
    const json opened = call(server, "stream_open", {{"name", stream}, {"chunks", kChunks}, {"ringBytes", 1 << 20}});
    const json ring = json::parse(textOf(opened));
    auto memory = platform::SharedMemory::open(ring["ring"].get<std::string>(), ring["bytes"].get<size_t>());
    if (!memory) FAIL(memory.error().toString());
    std::atomic<bool> produced{false};
    std::atomic<int> refused{0};
    std::thread producer([&] {
        for (uint32_t e = 1; e <= kEpochs; ++e) {
            json list = json::array();
            uint64_t at = 0;
            for (uint32_t c = 0; c < kChunks; ++c) {
                std::memcpy(memory->bytes().data() + at, pages[e][c].data(), pages[e][c].size());
                list.push_back({{"chunk", c}, {"op", "replace"}, {"offset", at}, {"bytes", pages[e][c].size()}});
                at += pages[e][c].size();
            }
            const json answer = call(server, "stream_commit", {{"name", stream}, {"epoch", e}, {"pages", list}});
            refused += answer["result"].value("isError", false) ? 1 : 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }
        produced = true;
    });

    int frames = 0, mismatched = 0;
    uint32_t last = 0;
    bool ordered = true;
    for (; frames < 400; ++frames) {
        auto image = (*renderer)->render("/World/Shot", 0.0, settings.width, settings.height);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(hydra);
        uint32_t best = 0, bestMax = UINT32_MAX, bestP99 = UINT32_MAX;
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
        if (produced && best == kEpochs) {
            break;
        }
    }
    producer.join();
    std::printf("  %d frames, the last epoch %u; %d not one whole epoch; %d commits refused\n", frames, last,
                mismatched, refused.load());
    CHECK(refused == 0);
    CHECK(mismatched == 0);
    CHECK(ordered);
    CHECK(last == kEpochs);
    CHECK(textOf(call(server, "stream_close", {{"name", stream}})).find("closed") != std::string::npos);
}

TEST_CASE("shared memory a name stands for is seen by whoever opens the name", "[core][platform]") {
    auto made = platform::SharedMemory::create("athenea-test-shm", 4096);
    if (!made) FAIL(made.error().toString());
    std::memcpy(made->bytes().data(), "live", 4);
    auto seen = platform::SharedMemory::open("athenea-test-shm", 4096);
    if (!seen) FAIL(seen.error().toString());
    CHECK(std::memcmp(seen->bytes().data(), "live", 4) == 0);
    CHECK(!platform::SharedMemory::create("athenea-test-shm", 4096));   // the name is taken
    CHECK(!platform::SharedMemory::create("a/b", 4096));
}
