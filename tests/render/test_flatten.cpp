// Copyright (c) 2026 jesus luque.
//
// `athenea flatten` (render/Flatten.h, docs/decisions.md task TXF), checked
// in the two places a wrong sign or order would hide:
//
// - the fit: degree 3 harmonics known in a file's frame, seen from a
//   gaussian's random frame and fitted through the matrix the device solved,
//   come back as they went in -- for a surface's weighting and a thin wall's;
// - the files: the same flattened gaussians written as PLY and as SPZ
//   (versions 3 and 4) by this engine's packers, read back by its readers
//   (the SPZ one is Niantic's decompression), render alike. SPZ is right-up-
//   back and the engine reads it as right-down-front, so its frame is drawn
//   turned half about x -- which is what puts it where the PLY is.
//
// Everything compared on the device; only counters and the image metrics
// come back.
#include "../gpu/GpuTest.h"

#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "athenea/io/Readers.h"
#include "athenea/io/SplatWriters.h"
#include "athenea/render/Flatten.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
namespace fs = std::filesystem;

TEST_CASE("flatten's fit gives back degree 3 harmonics in any frame", "[render][flatten][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t directions = GENERATE(64u, 256u);
    auto fit = render::FlattenFit::create(*gpu->library);
    REQUIRE(fit);
    render::FlattenSettings settings;
    settings.directions = directions;
    // Unregularised: data in the span of the basis is fitted exactly
    // whatever the weights, so what comes back is what went in.
    settings.lambda = 0.0F;
    REQUIRE(fit->prepare(settings));
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/flatten_check", "flattenCheckFit");
    REQUIRE(check);
    for (uint32_t cls = 0; cls < 2; ++cls) {
        const std::array<uint32_t, 2> zero{0, 0};
        auto stats = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, zero, "test.flatten.stats");
        REQUIRE(stats);
        constexpr uint32_t kCases = 4096;
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {kCases, 1, 1}, [&](rhi::ShaderCursor c) {
            c["basisRows"].setBinding(fit->basis().rhi());
            c["stats"].setBinding(stats->rhi());
            c["records"].setBinding(stats->rhi());
            c["params"]["cases"].setData(kCases);
            c["params"]["directions"].setData(directions);
            c["params"]["cls"].setData(cls);
            c["params"]["tolerance"].setData(2.0e-3F);
            c["params"]["count"].setData(0u);
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 2> got{};
        REQUIRE(stats->read(*gpu->device, 0, sizeof(got), got.data()));
        std::printf("  %u directions, %s: %u of %u cases off, worst %.6f\n", directions,
                    cls == 0 ? "surface" : "thin wall", got[0], kCases, static_cast<double>(got[1]) * 1.0e-6);
        CHECK(got[0] == 0);
    }
    // And the roughness floor was found among the candidates.
    auto floor = fit->floorRoughness();
    REQUIRE(floor);
    std::printf("  roughness floor %.4f\n", static_cast<double>(*floor));
    CHECK(*floor > 0.0F);
    CHECK(*floor < 1.0F);
}

TEST_CASE("a flattened cloud written as PLY and as SPZ renders alike", "[render][flatten][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t version = GENERATE(3u, 4u);
    if (version == 4 && !io::writesSpzVersion4()) {
        SKIP("this build writes no SPZ version 4 (no ZSTD)");
    }
    constexpr uint32_t kCount = 1500;
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{kCount} * render::kFlatStride * 4;
    desc.elementBytes = 4;
    desc.label = "test.flatten.records";
    auto records = gpu::Buffer::create(*gpu->device, desc);
    REQUIRE(records);
    auto make = gpu::ComputeKernel::create(*gpu->library, "athenea/test/flatten_check", "flattenCheckRecords");
    REQUIRE(make);
    {
        gpu::CommandBatch batch(*gpu->device);
        make->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor c) {
            c["basisRows"].setBinding(records->rhi());
            c["stats"].setBinding(records->rhi());
            c["records"].setBinding(records->rhi());
            c["params"]["count"].setData(kCount);
        });
        REQUIRE(batch.submit(true));
    }
    render::FlatCloud cloud;
    cloud.count = kCount;
    cloud.records = *records;
    const std::vector<render::FlatCloud> clouds{cloud};
    auto pack = render::FlattenPack::create(*gpu->library);
    REQUIRE(pack);
    auto stats = pack->stats(clouds, -12.0F);
    REQUIRE(stats);
    CHECK(stats->spzSaturated == 0);
    CHECK(stats->min[0] >= -1.0F);
    CHECK(stats->max[0] <= 1.0F);

    const fs::path dir = fs::temp_directory_path() / "athenea-tests";
    fs::create_directories(dir);
    const fs::path plyPath = dir / "flattened.ply";
    const fs::path spzPath = dir / ("flattened-v" + std::to_string(version) + ".spz");
    auto ply = pack->ply(cloud, -12.0F);
    REQUIRE(ply);
    const std::vector<std::vector<float>> chunks{*ply};
    REQUIRE(io::writePly3dgs(plyPath, chunks, std::vector<std::string>{"test"}));
    auto streams = pack->spz(clouds, stats->fractionalBits, 8, 8);
    REQUIRE(streams);
    io::SpzHeader header;
    header.version = version;
    header.count = kCount;
    header.fractionalBits = stats->fractionalBits;
    REQUIRE(io::writeSpz(spzPath, header, streams->streams));

    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto plyRaw = io::readSplats(plyPath);
    if (!plyRaw) {
        FAIL(plyRaw.error().toString());
    }
    auto spzRaw = io::readSplats(spzPath);
    if (!spzRaw) {
        FAIL(spzRaw.error().toString());
    }
    CHECK(plyRaw->count == kCount);
    CHECK(spzRaw->count == kCount);
    auto fromPly = loader->upload(*plyRaw, 3);
    REQUIRE(fromPly);
    auto fromSpz = loader->upload(*spzRaw, 3);
    REQUIRE(fromSpz);
    CHECK(fromPly->degree() == 3);
    CHECK(fromSpz->degree() == 3);

    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 4.5}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    // SPZ read as right-down-front: half a turn about x puts it back.
    render::Mat4 turned = render::Mat4::identity();
    turned.m[5] = -1.0;
    turned.m[10] = -1.0;
    for (uint32_t degree : {0u, 3u}) {
        settings.maxShDegree = degree;
        render::RenderTargets a, b;
        std::vector<render::SplatInstance> plyInstance(1), spzInstance(1);
        plyInstance[0].splats = &*fromPly;
        spzInstance[0].splats = &*fromSpz;
        spzInstance[0].objectToWorld = turned;
        REQUIRE(raster->render(camera, plyInstance, settings, a));
        REQUIRE(raster->render(camera, spzInstance, settings, b));
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(diff);
        std::printf("  SPZ v%u vs PLY, degree %u: p99 %u, max %u, %llu of %llu pixels over 2\n", version, degree,
                    diff->p99, diff->max, static_cast<unsigned long long>(diff->over2),
                    static_cast<unsigned long long>(diff->pixels));
        // 8-bit colour, 1/16-stop sizes and 8-bit harmonics: a few code
        // values; a wrong axis or sign is tens.
        CHECK(diff->p99 <= (degree == 0 ? 4u : 8u));
    }
}
