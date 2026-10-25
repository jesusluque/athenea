// Copyright (c) 2026 jesus luque.
//
// LIVE STREAMS A STAGE NAMES, UNDER WHAT A STAGE DOES TO THEM: two prims
// naming one stream, and a stream renamed, removed and named again (variant
// switches) while a producer pushes. Each frame is identified by the epoch it
// is drawn closest to, among references drawn straight from the pages' clouds;
// a frame must be one whole epoch, and the latest epoch must be reached.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "load-spz.h"
#include "athenea/core/Platform.h"
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

constexpr uint32_t kChunks = 3;
constexpr uint32_t kWidth = 160, kHeight = 120;

const char* kCamera =
    "    def Camera \"Shot\"\n    {\n"
    "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
    "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
    "        double3 xformOp:translate = (0.3, 0.4, 6.0)\n"
    "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n";

std::string livePrim(const std::string& name, const std::string& stream, const std::string& indent) {
    return indent + "def ParticleField3DGaussianSplat \"" + name + "\" (\n" + indent +
           "    prepend apiSchemas = [\"AtheneaLiveSourceAPI\"]\n" + indent + ")\n" + indent + "{\n" + indent +
           "    string primvars:athenea:live:stream = \"" + stream + "\" ( interpolation = \"constant\" )\n" +
           indent + "    int primvars:athenea:live:chunks = 3 ( interpolation = \"constant\" )\n" + indent +
           "    int primvars:athenea:live:chunkSplats = 600 ( interpolation = \"constant\" )\n" + indent +
           "    int primvars:athenea:live:degree = 0 ( interpolation = \"constant\" )\n" + indent + "}\n";
}

/// A set of pages (one an chunk) and what they draw, `times` over.
struct PageSet {
    std::vector<std::vector<uint8_t>> pages;
    std::vector<scene::GpuSplats>     clouds;
};

struct Rig {
    test::Gpu*                              gpu;
    std::unique_ptr<scene::CloudLoader>     loader;
    std::unique_ptr<render::TileRasterizer> raster;
    render::RenderSettings                  settings;
    render::Projection                      projection;

    explicit Rig(test::Gpu* g) : gpu(g) {
        auto l = scene::CloudLoader::create(*gpu->library);
        auto r = render::TileRasterizer::create(*gpu->library);
        REQUIRE(l);
        REQUIRE(r);
        loader = std::make_unique<scene::CloudLoader>(std::move(*l));
        raster = std::make_unique<render::TileRasterizer>(std::move(*r));
        render::Camera camera;
        camera.cameraToWorld = aofx::xform::translation({0.3, 0.4, 6.0});
        camera.lens.focal = 30.0;
        camera.lens.nearZ = 0.1;
        camera.lens.farZ = 1000.0;
        settings.width = kWidth;
        settings.height = kHeight;
        projection = render::projectionFor(camera, kWidth, kHeight);
    }

    PageSet set(uint64_t seed) {
        PageSet s;
        for (uint32_t c = 0; c < kChunks; ++c) {
            s.pages.push_back(spzPage(350 + 40 * c, seed * 10 + c));
            auto raw = io::readSpz(std::span<const uint8_t>(s.pages.back()), "page");
            REQUIRE(raw);
            auto cloud = loader->upload(*raw, 3);
            REQUIRE(cloud);
            s.clouds.push_back(std::move(*cloud));
        }
        return s;
    }

    /// What `sets` draw together (each cloud once per set listed).
    render::RenderTargets draw(const std::vector<const PageSet*>& sets) {
        std::vector<render::SplatInstance> instances;
        for (const PageSet* s : sets) {
            for (const auto& c : s->clouds) instances.push_back({&c, render::Mat4::identity()});
        }
        render::RenderTargets t;
        REQUIRE(raster->render(projection, instances, settings, t));
        return t;
    }

    /// The reference a stage frame is closest to, and whether it is as close
    /// as Hydra and a direct draw are (p99 1, max 2).
    std::pair<int, bool> classify(usd::StageRenderer& renderer, const std::vector<render::RenderTargets>& refs) {
        auto image = renderer.render("/World/Shot", 0.0, kWidth, kHeight);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(hydra);
        int best = -1;
        uint32_t bestMax = UINT32_MAX, bestP99 = UINT32_MAX;
        for (size_t k = 0; k < refs.size(); ++k) {
            auto d = render::compareImages(*gpu->library, *hydra, refs[k].colour, kWidth, kHeight);
            REQUIRE(d);
            if (d->max < bestMax) {
                best = static_cast<int>(k);
                bestMax = d->max;
                bestP99 = d->p99;
            }
        }
        return {best, bestP99 <= 1 && bestMax <= 2};
    }
};

void push(lod::PageQueue& queue, uint64_t epoch, const PageSet& s) {
    for (uint32_t c = 0; c < kChunks; ++c) {
        lod::PageUpdate u;
        u.epoch = epoch;
        u.chunk = c;
        u.pages = kChunks;
        u.op = lod::PageUpdate::Op::Replace;
        u.bytes.resize(s.pages[c].size());
        std::memcpy(u.bytes.data(), s.pages[c].data(), s.pages[c].size());
        queue.push(std::move(u));
    }
}

fs::path writeStage(const std::string& name, const std::string& body) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests";
    fs::create_directories(dir);
    const fs::path path = dir / name;
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n" << body;
    return path;
}

}   // namespace

TEST_CASE("qa: two prims naming one stream both draw every epoch", "[usd][gpu][live][qa]") {
    ATHENEA_REQUIRE_GPU(gpu);
    Rig rig(gpu);
    const std::string stream = "athenea-qa-two-prims";
    const fs::path shot = writeStage("live-qa-two.usda", "def Xform \"World\"\n{\n" + livePrim("A", stream, "    ") +
                                                             livePrim("B", stream, "    ") + kCamera + "}\n");
    std::vector<PageSet> sets;
    for (uint64_t e = 1; e <= 3; ++e) sets.push_back(rig.set(700 + e));
    // 0: nothing; 1..3: an epoch drawn once (one prim has it); 4..6: twice (both).
    std::vector<render::RenderTargets> refs;
    refs.push_back(rig.draw({}));
    for (auto& s : sets) refs.push_back(rig.draw({&s}));
    for (auto& s : sets) refs.push_back(rig.draw({&s, &s}));

    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());
    auto queue = lod::PageQueue::named(stream);
    int last = 0, frames = 0, notWhole = 0, once = 0;
    for (uint64_t e = 1; e <= 3; ++e) {
        push(*queue, e, sets[e - 1]);
        for (int k = 0; k < 30; ++k, ++frames) {
            const auto [best, close] = rig.classify(**renderer, refs);
            notWhole += close ? 0 : 1;
            once += (best >= 1 && best <= 3) ? 1 : 0;
            last = best;
            if (best == static_cast<int>(3 + e)) break;
        }
    }
    std::printf("  %d frames: the last drawn as reference %d (6: epoch 3 by both prims); %d frames drawn by one prim "
                "alone; %d not whole\n",
                frames, last, once, notWhole);
    CHECK(last == 6);
    CHECK(once == 0);
    CHECK(notWhole == 0);
}

TEST_CASE("qa: a stream renamed, removed and named again shows its latest epoch", "[usd][gpu][live][qa]") {
    // A variant set swaps the prim's stream: "a", then "b", then no prim at
    // all while a's producer keeps pushing, then "a" again. Named again, the
    // stream must show its latest epoch -- not replay what was pushed while no
    // prim read it.
    ATHENEA_REQUIRE_GPU(gpu);
    Rig rig(gpu);
    const std::string a = "athenea-qa-feed-a", b = "athenea-qa-feed-b";
    const fs::path shot = writeStage(
        "live-qa-feed.usda",
        "def Xform \"World\" (\n    variants = {\n        string feed = \"a\"\n    }\n"
        "    prepend variantSets = \"feed\"\n)\n{\n    variantSet \"feed\" = {\n"
        "        \"a\" {\n" + livePrim("Live", a, "            ") + "        }\n"
        "        \"b\" {\n" + livePrim("Live", b, "            ") + "        }\n"
        "        \"off\" {\n        }\n    }\n" + kCamera + "}\n");
    // 0 nothing; 1..3 a's sets; 4..5 b's; 6 the epoch a ends on.
    std::vector<PageSet> sets;
    for (uint64_t k = 1; k <= 6; ++k) sets.push_back(rig.set(900 + k));
    std::vector<render::RenderTargets> refs;
    refs.push_back(rig.draw({}));
    for (auto& s : sets) refs.push_back(rig.draw({&s}));

    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());
    auto qa = lod::PageQueue::named(a);
    auto qb = lod::PageQueue::named(b);
    int notWhole = 0;
    const auto until = [&](int want, int most) {
        for (int k = 1; k <= most; ++k) {
            const auto [best, close] = rig.classify(**renderer, refs);
            notWhole += close ? 0 : 1;
            if (best == want) return k;
        }
        return -1;
    };

    for (uint64_t e = 1; e <= 3; ++e) push(*qa, e, sets[e - 1]);
    const int toA = until(3, 60);
    REQUIRE((*renderer)->setVariantSelection("/World", "feed", "b"));
    push(*qb, 1, sets[3]);
    push(*qb, 2, sets[4]);
    const int toB = until(5, 60);

    // No prim reads a while its producer goes on.
    std::atomic<bool> stop{false};
    std::thread producer([&] {
        for (uint64_t e = 4; e <= 40 && !stop; ++e) {
            push(*qa, e, sets[(e - 1) % 3]);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });
    REQUIRE((*renderer)->setVariantSelection("/World", "feed", "off"));
    const int toNothing = until(0, 60);
    for (int k = 0; k < 20; ++k) (void)rig.classify(**renderer, refs);
    producer.join();
    push(*qa, 41, sets[5]);
    REQUIRE((*renderer)->setVariantSelection("/World", "feed", "a"));
    const int toLatest = until(6, 120);
    std::printf("  frames to a's epoch 3: %d; to b's epoch 2: %d; to nothing: %d; named again, to a's latest: %d; "
                "%d frames not whole\n",
                toA, toB, toNothing, toLatest, notWhole);
    CHECK(toA > 0);
    CHECK(toB > 0);
    CHECK(toNothing > 0);
    CHECK(notWhole == 0);
    CHECK(toLatest > 0);
    CHECK(toLatest <= 5);
}

TEST_CASE("qa soak: a producer at 30 epochs a second for ATHENEA_SOAK_SECONDS while the stage draws",
          "[.soak][usd][gpu][live][qa]") {
    // Odd epochs replace all eight chunks with set k; even ones replace the
    // first four and remove the rest -- so the page count changes every
    // epoch. Every frame is compared with the state the epoch the stage says
    // it shows must draw; the counters are printed as it goes, for a script
    // that samples the process's memory beside it.
    ATHENEA_REQUIRE_GPU(gpu);
    const std::string said = platform::env("ATHENEA_SOAK_SECONDS");
    const double seconds = said.empty() ? 60.0 : std::atof(said.c_str());
    Rig rig(gpu);
    const std::string stream = "athenea-qa-soak";
    std::string prim = livePrim("Live", stream, "    ");
    prim.replace(prim.find("chunks = 3"), 10, "chunks = 8");
    const fs::path shot = writeStage("live-qa-soak.usda", "def Xform \"World\"\n{\n" + prim + kCamera + "}\n");
    constexpr uint32_t kSets = 4, kHalf = 4, kAll = 8;
    std::vector<std::vector<std::vector<uint8_t>>> pages(kSets);
    std::vector<std::vector<scene::GpuSplats>> clouds(kSets);
    std::vector<render::RenderTargets> full, half;
    for (uint32_t k = 0; k < kSets; ++k) {
        for (uint32_t c = 0; c < kAll; ++c) {
            pages[k].push_back(spzPage(200 + 30 * c + 7 * k, 5000 + 100 * k + c));
            auto raw = io::readSpz(std::span<const uint8_t>(pages[k].back()), "page");
            REQUIRE(raw);
            auto cloud = rig.loader->upload(*raw, 3);
            REQUIRE(cloud);
            clouds[k].push_back(std::move(*cloud));
        }
        for (const uint32_t n : {kAll, kHalf}) {
            std::vector<render::SplatInstance> instances;
            for (uint32_t c = 0; c < n; ++c) instances.push_back({&clouds[k][c], render::Mat4::identity()});
            render::RenderTargets t;
            REQUIRE(rig.raster->render(rig.projection, instances, rig.settings, t));
            (n == kAll ? full : half).push_back(std::move(t));
        }
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());
    auto queue = lod::PageQueue::named(stream);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> pushed{0};
    std::thread producer([&] {
        const auto t0 = std::chrono::steady_clock::now();
        for (uint64_t e = 1; !stop; ++e) {
            std::this_thread::sleep_until(t0 + std::chrono::microseconds(33333 * e));
            const uint32_t k = static_cast<uint32_t>(e % kSets);
            for (uint32_t c = 0; c < kAll; ++c) {
                lod::PageUpdate u;
                u.epoch = e;
                u.chunk = c;
                u.pages = kAll;
                u.ptpTime = e;
                if (e % 2 == 1 || c < kHalf) {
                    u.op = lod::PageUpdate::Op::Replace;
                    u.bytes.resize(pages[k][c].size());
                    std::memcpy(u.bytes.data(), pages[k][c].data(), pages[k][c].size());
                } else {
                    u.op = lod::PageUpdate::Op::Remove;
                }
                queue->push(std::move(u));
            }
            pushed = e;
        }
    });
    uint64_t frames = 0, wrong = 0, backwards = 0, last = 0;
    const auto start = std::chrono::steady_clock::now();
    auto report = start;
    const auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
    usd::StageRenderer::LiveStream s;
    while (elapsed() < seconds) {
        auto image = (*renderer)->render("/World/Shot", 0.0, kWidth, kHeight);
        if (!image) FAIL(image.error().toString());
        const auto live = (*renderer)->liveStreams();
        REQUIRE(live.size() == 1);
        s = live.front();
        ++frames;
        backwards += s.epoch < last ? 1 : 0;
        last = s.epoch;
        if (s.epoch > 0) {
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(hydra);
            const uint32_t k = static_cast<uint32_t>(s.epoch % kSets);
            const auto& want = s.epoch % 2 == 1 ? full[k] : half[k];
            auto d = render::compareImages(*gpu->library, *hydra, want.colour, kWidth, kHeight);
            REQUIRE(d);
            wrong += d->p99 <= 1 && d->max <= 2 ? 0 : 1;
        }
        if (std::chrono::steady_clock::now() - report > std::chrono::seconds(30)) {
            report = std::chrono::steady_clock::now();
            std::printf("soak %.0f s: %llu frames, epoch %llu of %llu pushed, %llu shown, %u pending, dropped %llu/%llu, "
                        "%llu not their epoch, %llu backwards\n",
                        elapsed(), static_cast<unsigned long long>(frames), static_cast<unsigned long long>(s.epoch),
                        static_cast<unsigned long long>(pushed.load()), static_cast<unsigned long long>(s.epochs),
                        s.pendingEpochs, static_cast<unsigned long long>(s.droppedNoRoom),
                        static_cast<unsigned long long>(s.droppedCorrupt), static_cast<unsigned long long>(wrong),
                        static_cast<unsigned long long>(backwards));
            std::fflush(stdout);
        }
    }
    stop = true;
    producer.join();
    std::printf("soak end %.0f s: %llu frames, %llu epochs shown of %llu pushed, %llu not their epoch\n", elapsed(),
                static_cast<unsigned long long>(frames), static_cast<unsigned long long>(s.epochs),
                static_cast<unsigned long long>(pushed.load()), static_cast<unsigned long long>(wrong));
    CHECK(wrong == 0);
    CHECK(backwards == 0);
    CHECK(s.droppedNoRoom == 0);
    CHECK(s.droppedCorrupt == 0);
    CHECK(s.epochs + 2 >= pushed.load());
}
