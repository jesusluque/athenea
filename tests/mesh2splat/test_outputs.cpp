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
#include <string>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/usd/Export.h"

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
