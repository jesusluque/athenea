// Copyright (c) 2026 jesus luque.
//
// What `athenea mesh2splat` wrote, checked on the device.
//
// The conversion is the command's own (apps/athenea/src/CmdMesh2Splat.cpp), so
// it is run as a command: ctest runs `athenea mesh2splat` over the stages in
// tests/data into ATHENEA_M2S_OUTPUTS first (the `mesh2splat_outputs`
// fixture, tests/CMakeLists.txt), and these cases read what it wrote. Every
// number is counted by a kernel (`athenea/test/mesh2splat_output_check`); what
// crosses back is the counts.
#include "../gpu/GpuTest.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;

namespace {

namespace fs = std::filesystem;

fs::path output(const char* name) { return fs::path(ATHENEA_M2S_OUTPUTS) / name; }

/// Red, blue, green, other and every gaussian that is there, in a converted
/// stage.
struct Colours {
    uint32_t red = 0, blue = 0, green = 0, other = 0, there = 0;
};

Colours coloursOf(test::Gpu& gpu, const fs::path& stage) {
    REQUIRE(fs::exists(stage));
    auto raw = usd::readParticleFieldRecords(stage);
    if (!raw) FAIL(raw.error().toString());
    REQUIRE(raw->count > 0);
    auto records = gpu::Buffer::fromSpan<float>(*gpu.device, raw->records, "test.records");
    REQUIRE(records);
    auto check = gpu::ComputeKernel::create(*gpu.library, "athenea/test/mesh2splat_output_check", "m2sColourClasses");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 5, "counts");
    gpu::CommandBatch batch(*gpu.device);
    check->dispatch(batch, {raw->count, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["records"].setBinding(records->rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["params"]["count"].setData(raw->count);
        cursor["params"]["stride"].setData(raw->encoding.floatsPerRecord);
        cursor["params"]["opacity"].setData(raw->encoding.opacity);
        cursor["params"]["dc0"].setData(raw->encoding.dc0);
    });
    REQUIRE(batch.submit(true));
    uint32_t seen[5] = {0, 0, 0, 0, 0};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(seen), seen));
    return {seen[0], seen[1], seen[2], seen[3], seen[4]};
}

}   // namespace

// A MESH WHOSE GEOMSUBSETS BIND DIFFERENT MATERIALS IS CONVERTED A SUBSET AT A
// TIME. tests/data/two_subsets.usda: one mesh of two equal faces, the left
// bound red and the right blue by GeomSubsets, the mesh itself green. The
// cloud carries red and blue, about as many of each, and no green -- which is
// all it carried when the mesh was converted with its own material alone.
TEST_CASE("a mesh's GeomSubsets are converted each with its own material", "[mesh2splat][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Colours c = coloursOf(*gpu, output("two_subsets.usda"));
    std::printf("  two subsets: %u red, %u blue, %u green, %u other of %u\n", c.red, c.blue, c.green, c.other,
                c.there);
    CHECK(c.red > 0);
    CHECK(c.blue > 0);
    CHECK(c.green == 0);
    CHECK(c.other == 0);
    // Two faces of one size, one cell: as many gaussians each, give or take
    // the cells along the edge they share.
    CHECK(c.red * 10 >= c.blue * 9);
    CHECK(c.blue * 10 >= c.red * 9);
}

// THE BUDGET IS SHARED, NOT SPENT IN MESH ORDER. tests/data/two_cards.usda:
// two equal cards, red first and blue second, under one grid at resolution
// 64 -- about nine hundred gaussians each -- converted with --max-splats 800.
// Spent in mesh order the red card took all 800 and the blue one none; shared
// in proportion, each walks a cell coarsened alike and takes about half.
TEST_CASE("a budget too small is shared between the meshes, not spent on the first", "[mesh2splat][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Colours c = coloursOf(*gpu, output("two_cards_budget.usda"));
    std::printf("  budget 800: %u red, %u blue, %u other of %u\n", c.red, c.blue, c.green + c.other, c.there);
    CHECK(c.there <= 800);
    CHECK(c.red >= 300);
    CHECK(c.blue >= 300);
}

// A CAMERA'S PIXEL DECIDES EACH MESH'S CELL (`--cell-from-camera`, Mesh2GS).
// The red card stands three units from the lens and the blue one about seven:
// a pixel covers 2.3 times as much at the blue card, so its cell is that much
// coarser and it holds about a fifth of the red card's gaussians -- where one
// grid over the pair gave them as many each.
TEST_CASE("a camera's pixel decides each mesh's cell", "[mesh2splat][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Colours c = coloursOf(*gpu, output("two_cards_camera.usda"));
    std::printf("  from the camera: %u red, %u blue of %u\n", c.red, c.blue, c.there);
    CHECK(c.blue > 0);
    CHECK(c.red > 3 * c.blue);
}

// `-o x.athc` WRITES WHAT `-o x.usda` WRITES, WITH ITS LEVELS OF DETAIL. The
// two cards converted both ways: the stage read back and decoded onto the
// device, the .athc read whole, and both drawn by the rasteriser from the
// same camera -- the .athc through a cut that keeps every splat. The same
// image, but for splats whose depths tie and draw in the other order
// (Morton order against the conversion's; tests/lod measures that at p99 1).
TEST_CASE("a cloud written as a .athc draws as the one written as a stage", "[mesh2splat][gpu][lod]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path stage = output("two_cards.usda");
    const fs::path athc = output("two_cards.athc");
    REQUIRE(fs::exists(stage));
    REQUIRE(fs::exists(athc));
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto cut = lod::CutSelector::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!cut) FAIL(cut.error().toString());

    auto raw = usd::readParticleFieldRecords(stage);
    if (!raw) FAIL(raw.error().toString());
    auto fromStage = loader->upload(*raw, 3);
    if (!fromStage) FAIL(fromStage.error().toString());
    auto fromAthc = lod::readAthc(*gpu->device, athc);
    if (!fromAthc) FAIL(fromAthc.error().toString());
    std::printf("  %u gaussians in the stage, %u in the .athc (%zu levels)\n", fromStage->count, fromAthc->count,
                fromAthc->levels.size());
    CHECK(fromAthc->count == fromStage->count);

    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 120;
    const render::Projection projection = render::projectionFor(
        render::Camera::lookingAt({1.25, 0.5, 3.0}, {1.25, 0.5, -2.0}), settings.width, settings.height);
    render::RenderTargets a, b;
    REQUIRE(raster->render(projection, std::vector<render::SplatInstance>{{&*fromStage, render::Mat4::identity()}},
                           settings, a));
    const std::vector<lod::LodInstance> instances{{&*fromAthc, render::Mat4::identity()}};
    auto selected = cut->select(projection, instances, 0.0F, nullptr);
    if (!selected) FAIL(selected.error().toString());
    REQUIRE(raster->render(projection, *selected, settings, b));
    auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
    REQUIRE(diff);
    std::printf("  .athc against stage: p99 %u, max %u, %llu of %llu over 2\n", diff->p99, diff->max,
                static_cast<unsigned long long>(diff->over2), static_cast<unsigned long long>(diff->pixels));
    CHECK(diff->p99 <= 1);
    CHECK(diff->over2 * 100 <= diff->pixels);
}

// A SKINNED CLOUD KEEPS ITS TRANSFER (proposal 014 B). tests/data/skinned_corner.usda
// converted with `--skinned --transfer --time 1`: the cloud is carried by its
// joint and its transfer is kept as ten values a gaussian -- two zonal lobes in
// the gaussian's frame -- where `--skinned` used to drop the transfer. Under a
// sky, the corner draws at every instant of its range, the wall taking light
// off the floor whichever way the joint has turned them.
TEST_CASE("a skinned conversion keeps its transfer as zonal lobes and draws in every pose",
          "[mesh2splat][gpu][skinning][transfer]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path stage = output("skinned_transfer.usda");
    REQUIRE(fs::exists(stage));
    const fs::path lit = output("skinned_transfer_lit.usda");
    {
        std::ofstream out(lit);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    subLayers = [@" << stage.string() << "@]\n)\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1, 6)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    auto renderer = usd::StageRenderer::open(lit);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setGaussianStats(true);
    const uint32_t w = 96, h = 72;
    for (const double time : {0.0, 1.0, 2.0}) {
        auto image = (*renderer)->render("/Camera", time, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        auto stats = render::imageStats(*gpu->library, *made, w, h);
        REQUIRE(stats);
        std::printf("  skinned corner at %.0f: mean %.4f, max %.3f\n", time, stats->mean[0],
                    static_cast<double>(stats->max[0]));
        CHECK(stats->mean[0] > 0.0);
        CHECK(stats->max[0] <= 2.0F);
    }
    // What the frames were handed: one cloud, carried and transferred.
    const usd::GaussianStats report = (*renderer)->gaussianStats();
    uint32_t zonal = 0;
    for (const ui::GaussianCloud& cloud : report.clouds) {
        zonal += cloud.transfer == scene::GpuSplats::kTransferZonalCount && cloud.skinned ? 1u : 0u;
    }
    std::printf("  %zu cloud%s, %u skinned with a zonal transfer\n", report.clouds.size(),
                report.clouds.size() == 1 ? "" : "s", zonal);
    CHECK(zonal == 1);
}
