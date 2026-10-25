// Copyright (c) 2026 jesus luque.
//
// .athz: a cloud coded into pages on the device and decoded back on the
// device. Everything is checked there: records compared word for word and
// hashed by a kernel (athz_check.slang), images compared by compareImages.
// What crosses back is counters.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "athenea/core/Platform.h"
#include "athenea/lod/Athc.h"
#include "athenea/lod/Athz.h"
#include "athenea/lod/Lod.h"
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
    gpu::ComputeKernel     compareWords, hashWords;
    render::RenderSettings settings;
    render::Projection     projection;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto builder = lod::LodBuilder::create(*gpu->library);
    auto cut = lod::CutSelector::create(*gpu->library);
    auto compareWords = gpu::ComputeKernel::create(*gpu->library, "athenea/lod/athz_check", "athzCompare");
    auto hashWords = gpu::ComputeKernel::create(*gpu->library, "athenea/lod/athz_check", "athzHash");
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!builder) FAIL(builder.error().toString());
    if (!cut) FAIL(cut.error().toString());
    if (!compareWords) FAIL(compareWords.error().toString());
    if (!hashWords) FAIL(hashWords.error().toString());
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Projection projection = render::projectionFor(
        render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0}), settings.width, settings.height);
    return std::unique_ptr<Harness>(new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*builder),
                                                std::move(*cut), std::move(*compareWords), std::move(*hashWords),
                                                settings, projection});
}

/// A random cloud with its levels; with shading normals where asked (one
/// more array a record, so one more word an element).
lod::LodCloud cloudOf(Harness& h, uint32_t count, uint64_t seed, bool normals) {
    CloudBuilder built = randomCloud(count, seed, 0.003F, 0.08F);
    if (normals) {
        std::vector<float> n;
        test::Lcg rng;
        rng.state ^= seed * 977;
        for (uint32_t i = 0; i < count; ++i) {
            n.insert(n.end(), {rng.range(-1, 1), rng.range(-1, 1), 1.0F});
        }
        test::giveNormals(built, n);
    }
    auto cloud = h.loader.upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 4096;
    auto lod = h.builder.build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    return std::move(*lod);
}

std::filesystem::path temp(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

/// Words of `a` and `b` that differ, from word `fromA` and `fromB`.
uint32_t differing(Harness& h, const gpu::Buffer& a, uint64_t fromA, const gpu::Buffer& b, uint64_t fromB,
                   uint64_t words) {
    if (words == 0) {
        return 0;
    }
    gpu::Buffer counters = test::uintBuffer(*h.gpu->device, 2, "test.differing");
    gpu::CommandBatch batch(*h.gpu->device);
    h.compareWords.dispatch(batch, {static_cast<uint32_t>(words), 1, 1}, [&](rhi::ShaderCursor c) {
        c["a"].setBinding(a.rhi());
        c["b"].setBinding(b.rhi());
        c["counters"].setBinding(counters.rhi());
        c["positions"].setBinding(a.rhi());
        c["shape"].setBinding(a.rhi());
        c["sh"].setBinding(a.rhi());
        c["flat"].setBinding(counters.rhi());
        c["params"]["words"].setData(static_cast<uint32_t>(words));
        c["params"]["offsetA"].setData(static_cast<uint32_t>(fromA));
        c["params"]["offsetB"].setData(static_cast<uint32_t>(fromB));
    });
    REQUIRE(batch.submit(true));
    uint32_t n = 0;
    REQUIRE(counters.read(*h.gpu->device, 0, 4, &n));
    return n;
}

std::array<uint32_t, 2> hashOf(Harness& h, const gpu::Buffer& a, uint64_t words) {
    gpu::Buffer counters = test::uintBuffer(*h.gpu->device, 2, "test.hash");
    gpu::CommandBatch batch(*h.gpu->device);
    h.hashWords.dispatch(batch, {static_cast<uint32_t>(words), 1, 1}, [&](rhi::ShaderCursor c) {
        c["a"].setBinding(a.rhi());
        c["b"].setBinding(a.rhi());
        c["counters"].setBinding(counters.rhi());
        c["positions"].setBinding(a.rhi());
        c["shape"].setBinding(a.rhi());
        c["sh"].setBinding(a.rhi());
        c["flat"].setBinding(counters.rhi());
        c["params"]["words"].setData(static_cast<uint32_t>(words));
        c["params"]["offsetA"].setData(0u);
        c["params"]["offsetB"].setData(0u);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 2> out{};
    REQUIRE(counters.read(*h.gpu->device, 0, 8, out.data()));
    return out;
}

/// Every word two clouds' levels, starts and splats differ by.
uint64_t cloudDifference(Harness& h, const lod::LodCloud& a, const lod::LodCloud& b) {
    REQUIRE(a.levels.size() == b.levels.size());
    REQUIRE(a.count == b.count);
    uint64_t n = 0;
    const auto arrays = [&](const scene::GpuSplats& x, const scene::GpuSplats& y, uint64_t from, uint64_t count) {
        n += differing(h, x.positions, from * 4, y.positions, from * 4, count * 4);
        n += differing(h, x.shape, from * 4, y.shape, from * 4, count * 4);
        n += differing(h, x.sh, from * x.shWords, y.sh, from * y.shWords, count * x.shWords);
        if (x.hasNormals()) {
            REQUIRE(y.hasNormals());
            n += differing(h, x.normals, from, y.normals, from, count);
        }
    };
    for (size_t l = 0; l < a.levels.size(); ++l) {
        const auto& la = a.levels[l].gaussians;
        const auto& lb = b.levels[l].gaussians;
        REQUIRE(la.count == lb.count);
        arrays(la, lb, 0, la.count);
        n += differing(h, a.levels[l].cells, 0, b.levels[l].cells, 0, la.count);
    }
    n += differing(h, a.starts, 0, b.starts, 0, a.levels.back().gaussians.count);
    // Both stores hold chunk c in slot c, so the splats are in the same places.
    arrays(a.splats, b.splats, 0, a.count);
    n += differing(h, a.groups, 0, b.groups, 0, a.count);
    return n;
}

render::RenderTargets renderCloud(Harness& h, const scene::GpuSplats& cloud) {
    render::RenderTargets t;
    std::vector<render::SplatInstance> instances{{&cloud, render::Mat4::identity()}};
    REQUIRE(h.raster.render(h.projection, instances, h.settings, t));
    return t;
}

struct Cut {
    lod::CutStats         stats;
    render::RenderTargets targets;
};

Cut renderCut(Harness& h, const lod::LodCloud& cloud, float threshold) {
    Cut cut;
    std::vector<lod::CutStats> stats;
    const std::vector<lod::LodInstance> instances{{&cloud, render::Mat4::identity()}};
    auto selected = h.cut.select(h.projection, instances, threshold, &stats);
    if (!selected) FAIL(selected.error().toString());
    cut.stats = stats.front();
    REQUIRE(h.raster.render(h.projection, *selected, h.settings, cut.targets));
    return cut;
}

render::ImageDifference compare(Harness& h, const render::RenderTargets& a, const render::RenderTargets& b) {
    auto diff = render::compareImages(*h.gpu->library, a.colour, b.colour, h.settings.width, h.settings.height);
    REQUIRE(diff);
    return *diff;
}

int settle(Harness& h, lod::StreamingPool& pool, float threshold, Cut* last, int most = 400) {
    for (int frame = 1; frame <= most; ++frame) {
        *last = renderCut(h, pool.cloud(), threshold);
        pool.want(last->stats.needs);
        auto placed = pool.update(false);
        if (!placed) FAIL(placed.error().toString());
        const auto s = pool.status();
        if (*placed == 0 && s.inFlight == 0 && s.missing == 0) {
            return frame;
        }
        if (*placed == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    FAIL("the stream never settled");
    return 0;
}

/// A file's bytes on the device (a file's bytes as they are, for a kernel
/// to compare).
gpu::Buffer fileOnDevice(Harness& h, const std::filesystem::path& path, uint64_t* bytes) {
    auto file = platform::MappedFile::open(path);
    REQUIRE(file);
    *bytes = file->size();
    gpu::BufferDesc desc;
    desc.bytes = (file->size() + 3) / 4 * 4;
    desc.label = "test.file";
    std::vector<std::byte> padded(desc.bytes, std::byte{0});
    std::copy(file->bytes().begin(), file->bytes().end(), padded.begin());
    auto made = gpu::Buffer::create(*h.gpu->device, desc, padded.data());
    REQUIRE(made);
    return *made;
}

void copyFile(const std::filesystem::path& from, const std::filesystem::path& to) {
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing);
}

}   // namespace

TEST_CASE("an .athz decodes to the records it was coded from, bit for bit", "[lod][gpu][athz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    for (const bool normals : {false, true}) {
        const lod::LodCloud cloud = cloudOf(*h, 30000, normals ? 81 : 80, normals);
        struct Config {
            uint32_t run, pageRuns, tableLog;
            bool     rans, bytePlanes;
        };
        for (const Config c : {Config{256, 12, 11, true, false}, Config{64, 48, 12, true, false},
                               Config{32, 3, 11, false, false}, Config{256, 2, 9, true, false},
                               Config{128, 5, 11, false, true}}) {
            lod::AthzSettings s;
            s.runGaussians = c.run;
            s.pageRuns = c.pageRuns;
            s.tableLog = c.tableLog;
            s.rans = c.rans;
            s.bytePlanes = c.bytePlanes;
            const auto path = temp("athenea_test_roundtrip.athz");
            auto stats = lod::writeAthz(*gpu->library, cloud, path, s);
            if (!stats) FAIL(stats.error().toString());
            auto back = lod::readAthz(*gpu->library, path);
            if (!back) FAIL(back.error().toString());
            const uint64_t words = cloudDifference(*h, cloud, *back);
            const uint64_t record = uint64_t{cloud.count} * 4;
            const auto hashA = hashOf(*h, cloud.splats.positions, record);
            const auto hashB = hashOf(*h, back->splats.positions, record);
            std::printf("  normals %d, runs %u x %u, 2^%u, %s: %llu pages, %.1f B a splat; %llu words differ\n",
                        normals ? 1 : 0, c.pageRuns, c.run, c.tableLog,
                        c.bytePlanes ? "byte planes" : (c.rans ? "rANS" : "raw"),
                        static_cast<unsigned long long>(stats->pages),
                        static_cast<double>(stats->fileBytes) / cloud.count, static_cast<unsigned long long>(words));
            CHECK(words == 0);
            CHECK(hashA == hashB);
            CHECK(back->splats.hasNormals() == normals);
            // Only the layers asked for: a reader of the geometry alone gets
            // the same positions, and no colour.
            auto geometry = lod::readAthz(*gpu->library, path, lod::kAthzGeometry);
            if (!geometry) FAIL(geometry.error().toString());
            CHECK(differing(*h, cloud.splats.positions, 0, geometry->splats.positions, 0, record) == 0);
            CHECK(differing(*h, cloud.splats.sh, 0, geometry->splats.sh, 0, uint64_t{cloud.count} * cloud.splats.shWords) > 0);
        }
    }
}

TEST_CASE("an .athz's rANS tables do not depend on the order its pages are counted in", "[lod][gpu][athz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const lod::LodCloud cloud = cloudOf(*h, 30000, 90, false);
    lod::AthzSettings s;
    s.runGaussians = 64;
    s.pageRuns = 8;
    const auto a = temp("athenea_test_order_a.athz");
    const auto b = temp("athenea_test_order_b.athz");
    REQUIRE(lod::writeAthz(*gpu->library, cloud, a, s));
    s.histogramShuffle = 0x5EED;
    REQUIRE(lod::writeAthz(*gpu->library, cloud, b, s));
    uint64_t bytesA = 0, bytesB = 0;
    const gpu::Buffer fa = fileOnDevice(*h, a, &bytesA);
    const gpu::Buffer fb = fileOnDevice(*h, b, &bytesB);
    REQUIRE(bytesA == bytesB);
    CHECK(differing(*h, fa, 0, fb, 0, bytesA / 4) == 0);
}

TEST_CASE("a corrupt or cut .athz page is refused, never decoded", "[lod][gpu][athz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const lod::LodCloud cloud = cloudOf(*h, 20000, 100, false);
    const auto path = temp("athenea_test_good.athz");
    auto stats = lod::writeAthz(*gpu->library, cloud, path);
    REQUIRE(stats);
    // One byte flipped in the last section, inside a page.
    const auto corrupt = temp("athenea_test_corrupt.athz");
    copyFile(path, corrupt);
    {
        std::fstream f(corrupt, std::ios::in | std::ios::out | std::ios::binary);
        const auto at = static_cast<std::streamoff>(std::filesystem::file_size(corrupt) - 4096 - 100);
        f.seekg(at);
        char c = 0;
        f.read(&c, 1);
        c = static_cast<char>(c ^ 0x10);
        f.seekp(at);
        f.write(&c, 1);
    }
    auto read = lod::readAthz(*gpu->library, corrupt);
    REQUIRE_FALSE(read);
    std::printf("  corrupt: %s\n", read.error().message().c_str());
    CHECK(read.error().message().find("corrupt") != std::string::npos);
    // Through a pool: the loader refuses the page, the update fails.
    {
        auto source = lod::AthzSource::open(*gpu->library, corrupt);
        REQUIRE(source);
        lod::StreamingSettings settings;
        settings.budgetSplats = uint64_t{1} << 30;   // a slot every chunk
        auto pool = lod::StreamingPool::open(*gpu->device, std::unique_ptr<lod::ChunkSource>(std::move(*source)),
                                             settings);
        REQUIRE(pool);
        std::vector<uint32_t> all((*pool)->cloud().chunks(), 1000);
        (*pool)->want(all);
        auto placed = (*pool)->update(true);
        CHECK_FALSE(placed);
    }
    // Cut short: half the file.
    const auto cut = temp("athenea_test_cut.athz");
    copyFile(path, cut);
    std::filesystem::resize_file(cut, std::filesystem::file_size(cut) / 2);
    auto shortRead = lod::readAthz(*gpu->library, cut);
    REQUIRE_FALSE(shortRead);
    std::printf("  cut short: %s\n", shortRead.error().message().c_str());
    // And the header's own CRC.
    const auto header = temp("athenea_test_header.athz");
    copyFile(path, header);
    {
        std::fstream f(header, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(20);
        const char c = 0x7F;
        f.write(&c, 1);
    }
    CHECK_FALSE(lod::readAthz(*gpu->library, header));
}

TEST_CASE("a prefix of an .athz's bands draws as a coarser cut", "[lod][gpu][athz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const lod::LodCloud cloud = cloudOf(*h, 30000, 110, false);
    lod::AthzSettings s;
    s.runGaussians = 64;
    s.pageRuns = 4;
    const auto path = temp("athenea_test_full.athz");
    REQUIRE(lod::writeAthz(*gpu->library, cloud, path, s));
    auto full = lod::readAthz(*gpu->library, path);
    REQUIRE(full);
    const auto levels = static_cast<uint32_t>(full->levels.size());
    REQUIRE(levels >= 2);

    // Every merged level and none of the splats: drawn at 0 pixels, the cut
    // wants every splat, has none, and draws the finest level instead.
    const auto merged = temp("athenea_test_merged.athz");
    REQUIRE(lod::writeAthzPrefix(path, merged, levels));
    REQUIRE(std::filesystem::file_size(merged) < std::filesystem::file_size(path));
    auto coarse = lod::readAthz(*gpu->library, merged);
    if (!coarse) FAIL(coarse.error().toString());
    const Cut drawn = renderCut(*h, *coarse, 0.0F);
    const auto finest = renderCloud(*h, full->levels.back().gaussians);
    const auto diff = compare(*h, drawn.targets, finest);
    std::printf("  every level, no splats: %u merged drawn of %u; p99 %u max %u\n", drawn.stats.merged,
                full->levels.back().gaussians.count, diff.p99, diff.max);
    CHECK(drawn.stats.splats == 0);
    CHECK(drawn.stats.merged == full->levels.back().gaussians.count);
    CHECK(diff.p99 == 0);
    CHECK(diff.over2 * 1000 <= diff.pixels);

    // The same prefix streamed: nothing to load, the same image.
    {
        auto source = lod::AthzSource::open(*gpu->library, merged);
        REQUIRE(source);
        auto pool = lod::StreamingPool::open(*gpu->device, std::unique_ptr<lod::ChunkSource>(std::move(*source)));
        if (!pool) FAIL(pool.error().toString());
        Cut streamed;
        settle(*h, **pool, 0.0F, &streamed);
        CHECK(streamed.stats.splats == 0);
        CHECK(compare(*h, streamed.targets, drawn.targets).max == 0);
    }

    // The coarsest band alone: a fixed cut, the coarsest level's gaussians.
    const auto first = temp("athenea_test_first.athz");
    REQUIRE(lod::writeAthzPrefix(path, first, 1));
    uint32_t band = 99;
    auto fixed = lod::readAthzCut(*gpu->library, first, &band);
    if (!fixed) FAIL(fixed.error().toString());
    CHECK(band == 0);
    CHECK(fixed->count == full->levels.front().gaussians.count);
    const auto coarsest = compare(*h, renderCloud(*h, *fixed), renderCloud(*h, full->levels.front().gaussians));
    CHECK(coarsest.max == 0);
    // A cloud to stream needs every merged level.
    CHECK_FALSE(lod::readAthz(*gpu->library, first));
}

TEST_CASE("a pool over an AthzSource settles on the image the cloud in memory draws", "[lod][gpu][athz][stream]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const lod::LodCloud cloud = cloudOf(*h, 30000, 120, true);
    lod::AthzSettings s;
    s.runGaussians = 128;
    s.pageRuns = 8;
    const auto path = temp("athenea_test_stream.athz");
    REQUIRE(lod::writeAthz(*gpu->library, cloud, path, s));
    auto whole = lod::readAthz(*gpu->library, path);
    REQUIRE(whole);
    for (const float threshold : {0.0F, 24.0F}) {
        const Cut memory = renderCut(*h, *whole, threshold);
        auto source = lod::AthzSource::open(*gpu->library, path);
        REQUIRE(source);
        lod::StreamingSettings settings;
        settings.budgetSplats = uint64_t{1} << 30;   // a slot every chunk
        settings.stagingChunks = 3;
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
    }
}

TEST_CASE("an .athz exported to SPZ draws as the cloud does, within SPZ's steps", "[lod][gpu][athz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const lod::LodCloud cloud = cloudOf(*h, 20000, 130, false);
    const auto path = temp("athenea_test_export.athz");
    REQUIRE(lod::writeAthz(*gpu->library, cloud, path));
    auto splats = lod::readAthzCut(*gpu->library, path);
    if (!splats) FAIL(splats.error().toString());
    REQUIRE(splats->count == cloud.count);
    const auto spz = temp("athenea_test_export.spz");
    if (auto written = lod::writeSpzOfCloud(*gpu->library, *splats, spz); !written) FAIL(written.error().toString());
    auto back = scene::loadSplatFile(h->loader, spz, 3);
    if (!back) FAIL(back.error().toString());
    CHECK(back->count == cloud.count);
    const auto diff = compare(*h, renderCloud(*h, *splats), renderCloud(*h, *back));
    std::printf("  SPZ %.1f KB: p99 %u max %u, %llu of %llu pixels past 2 codes\n",
                static_cast<double>(std::filesystem::file_size(spz)) / 1024, diff.p99, diff.max,
                static_cast<unsigned long long>(diff.over2), static_cast<unsigned long long>(diff.pixels));
    // The bounds the SPZ readers' own tests hold a file to against the PLY it
    // came from (docs/decisions.md, "SPZ as the input contract"): SPZ's steps.
    CHECK(diff.p99 <= 4);
    CHECK(diff.max <= 16);
}
