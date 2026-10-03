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
