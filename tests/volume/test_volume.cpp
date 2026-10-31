// Copyright (c) 2026 jesus luque.
//
// Volumes (M9): a .vdb written from boxes of known extent, read and laid
// out as NanoVDB, uploaded, and checked by kernels -- the active voxels
// counted through the leaf masks against the count the boxes make, the
// bounds against the boxes' extent, each leaf's maximum recomputed voxel by
// voxel against the prepare kernel's, the majorant against the largest
// value filled. Then the medium on it: Beer-Lambert through the dense box
// by ratio and delta tracking within the standard error the kernel
// measures, no density over its majorant, and the leaf walk visiting the
// leaves a voxel walk passes through. Nothing about the values is read on
// the host.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <array>
#include <filesystem>
#include <string>

#include "athenea/gpu/algo/RadixSort.h"
#include "athenea/io/Vdb.h"
#include "athenea/world/VolumeSet.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

/// The fixture: 32^3 voxels at 0.5 from the origin, 8^3 at 2.0 apart, voxel 0.1.
fs::path boxesFixture() {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume";
    fs::create_directories(dir);
    const fs::path path = dir / "boxes.vdb";
    const io::VdbBox boxes[] = {{{0, 0, 0}, {32, 32, 32}, 0.5F}, {{40, 0, 0}, {48, 8, 8}, 2.0F}};
    if (auto written = io::writeVdbBoxes(path, "density", 0.1, boxes); !written) FAIL(written.error().toString());
    return path;
}

}   // namespace

TEST_CASE("a VDB grid laid out as NanoVDB holds the voxels it was filled with, and its bounds and maxima are the "
          "kernels'",
          "[volume][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path path = boxesFixture();
    auto names = io::vdbGridNames(path);
    if (!names) FAIL(names.error().toString());
    REQUIRE(*names == std::vector<std::string>{"density"});
    auto grid = io::readVdbGrid(path, "density");
    if (!grid) FAIL(grid.error().toString());
    std::printf("  %s: %zu words, %u leaves, voxel %.3f\n", grid->name.c_str(), grid->words.size(), grid->leafCount,
                grid->voxelSize);
    CHECK(grid->leafCount == 64 + 1);
    CHECK(grid->voxelSize == 0.1);

    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    world::VolumeInput input;
    input.grid = &*grid;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    auto countKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_check", "volumeCountActive");
    auto maxKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_check", "volumeLeafMaxCheck");
    if (!countKernel) FAIL(countKernel.error().toString());
    if (!maxKernel) FAIL(maxKernel.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "volume.counts");
    const world::VolumeGridPlacement& p = set->placement(0);
    {
        gpu::CommandBatch batch(*gpu->device);
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["words"].setBinding(set->words().rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["check"]["gridWord"].setData(p.gridWord);
            cursor["check"]["leafCount"].setData(p.leafCount);
            cursor["check"]["leafBase"].setData(p.leafBase);
        };
        countKernel->dispatch(batch, {p.leafCount, 1, 1}, bind);
        maxKernel->dispatch(batch, {p.leafCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    uint32_t c[3] = {};
    int32_t bounds[6] = {};
    uint32_t majorantBits = 0;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(set->bounds().read(*gpu->device, 0, sizeof(bounds), bounds));
    REQUIRE(set->majorant().read(*gpu->device, 0, 4, &majorantBits));
    float majorant = 0.0F;
    std::memcpy(&majorant, &majorantBits, 4);
    std::printf("  %u active voxels (32^3 + 8^3 = %u); bounds [%d %d %d, %d %d %d); %u of %u leaf maxima off the "
                "voxel-by-voxel recount; majorant %.3f\n",
                c[0], 32u * 32u * 32u + 8u * 8u * 8u, bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
                c[1], c[2], static_cast<double>(majorant));
    CHECK(c[0] == 32u * 32u * 32u + 8u * 8u * 8u);
    CHECK(bounds[0] == 0);
    CHECK(bounds[1] == 0);
    CHECK(bounds[2] == 0);
    CHECK(bounds[3] == 48);
    CHECK(bounds[4] == 32);
    CHECK(bounds[5] == 32);
    CHECK(c[2] == 65);
    CHECK(c[1] == 0);
    CHECK(majorant == 2.0F);
}

TEST_CASE("a medium on the grid is Beer-Lambert's within its own standard error, never over its majorant, and its "
          "leaf walk visits the leaves a voxel walk does",
          "[volume][gpu][medium]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    auto grid = io::readVdbGrid(boxesFixture(), "density");
    if (!grid) FAIL(grid.error().toString());
    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    // The dense box is 32 voxels of 0.1: 3.2 world units at density 0.5,
    // so sigma L = 1.6 along an axis through it; a scale of 1.
    world::VolumeInput input;
    input.grid = &*grid;
    input.densityScale = 1.0F;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    auto rays = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumRays");
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumReduce");
    auto majorant = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumMajorant");
    auto steps = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumSteps");
    if (!rays) FAIL(rays.error().toString());
    if (!reduce) FAIL(reduce.error().toString());
    if (!majorant) FAIL(majorant.error().toString());
    if (!steps) FAIL(steps.error().toString());
    constexpr uint32_t kRays = 16384;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "medium.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "medium.sums");
    gpu::Buffer perRay = test::uintBuffer(*gpu->device, uint64_t{kRays} * 2, "medium.perRay");
    struct Ray {
        float origin[3];
        float direction[3];
        float tMin, tMax;
        float expectedT;
        const char* name;
        uint32_t leaves;   // the leaf walk's expected count; 0: judge against the voxel walk
    };
    // The chord a ray cuts through the dense box [0, 3.2)^3, for the
    // expectation of an oblique ray (the fixture's geometry, by construction).
    const auto chord = [](const float* o, const float* d) {
        float t0 = -1e9F, t1 = 1e9F;
        for (int a = 0; a < 3; ++a) {
            float ta = (0.0F - o[a]) / d[a];
            float tb = (3.2F - o[a]) / d[a];
            if (ta > tb) std::swap(ta, tb);
            t0 = std::max(t0, ta);
            t1 = std::min(t1, tb);
        }
        return std::max(t1 - t0, 0.0F);
    };
    const float oblique[3] = {-0.5F, -0.3F, -0.2F};
    const float obliqueDir[3] = {0.68599F, 0.51449F, 0.51449F};   // (0.8, 0.6, 0.6) normalised
    // Through the middle of the dense box along x: 4 leaves, 6 leaf cells
    // with the empty ones past it; obliquely across it, judged against the
    // voxel walk; and along x through the small box from beyond the dense
    // one: 1 leaf.
    const Ray cases[] = {
        {{-1.0F, 1.6F, 1.6F}, {1.0F, 0.0F, 0.0F}, 0.0F, 10.0F, std::exp(-1.6F), "x through the dense box", 4},
        {{oblique[0], oblique[1], oblique[2]}, {obliqueDir[0], obliqueDir[1], obliqueDir[2]}, 0.0F, 20.0F,
         std::exp(-0.5F * chord(oblique, obliqueDir)), "obliquely across the dense box", 0},
        {{3.5F, 0.4F, 0.4F}, {1.0F, 0.0F, 0.0F}, 0.0F, 10.0F, std::exp(-2.0F * 0.8F), "x through the small box", 1},
        // The same box, walked the other way along each axis: a negative
        // direction steps the leaf cells down.
        {{4.2F, 1.6F, 1.6F}, {-1.0F, 0.0F, 0.0F}, 0.0F, 10.0F, std::exp(-1.6F), "-x through the dense box", 4},
        {{1.6F, 4.2F, 1.6F}, {0.0F, -1.0F, 0.0F}, 0.0F, 10.0F, std::exp(-1.6F), "-y through the dense box", 4},
        {{1.6F, 1.5F, 4.2F}, {0.0F, 0.0F, -1.0F}, 0.0F, 10.0F, std::exp(-1.6F), "-z through the dense box", 4},
    };
    for (const Ray& ray : cases) {
        const auto bind = [&](rhi::ShaderCursor c) {
            c["medium"]["rays"].setData(kRays);
            c["medium"]["volume"].setData(uint32_t{0});
            c["medium"]["expectedT"].setData(ray.expectedT);
            c["medium"]["origin"].setData(ray.origin, sizeof(ray.origin));
            c["medium"]["tMin"].setData(ray.tMin);
            c["medium"]["direction"].setData(ray.direction, sizeof(ray.direction));
            c["medium"]["tMax"].setData(ray.tMax);
            const float boxMin[3] = {0.0F, 0.0F, 0.0F};
            const float boxMax[3] = {4.8F, 3.2F, 3.2F};
            c["medium"]["boxMin"].setData(boxMin, sizeof(boxMin));
            c["medium"]["boxMax"].setData(boxMax, sizeof(boxMax));
            c["words"].setBinding(set->words().rhi());
            c["counts"].setBinding(counts.rhi());
            c["sums"].setBinding(sums.rhi());
            c["perRay"].setBinding(perRay.rhi());
        };
        const uint32_t zeros[6] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        {
            gpu::CommandBatch batch(*gpu->device);
            rays->dispatch(batch, {kRays, 1, 1}, bind);
            reduce->dispatch(batch, {1, 1, 1}, bind);
            majorant->dispatch(batch, {kRays, 1, 1}, bind);
            steps->dispatch(batch, {1, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t c[6] = {};
        float s[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
        const double n = kRays;
        const double meanT = s[0] / n;
        const double varT = std::max(s[1] / n - meanT * meanT, 0.0);
        const double zT = (meanT - ray.expectedT) / std::max(std::sqrt(varT / n), 1e-9);
        const double meanS = s[2] / n;
        const double varS = std::max(s[3] / n - meanS * meanS, 0.0);
        const double zS = (meanS - (1.0 - ray.expectedT)) / std::max(std::sqrt(varS / n), 1e-9);
        std::printf("  %s: T %.4f by ratio tracking (expected %.4f, z %.2f); scattered %.4f by delta tracking "
                    "(expected %.4f, z %.2f); %u of %u probes over their majorant; leaf walk %u steps, %u with a "
                    "leaf, voxel walk %u distinct leaves\n",
                    ray.name, meanT, static_cast<double>(ray.expectedT), zT, meanS, 1.0 - ray.expectedT, zS, c[1],
                    c[5], c[2], c[3], c[4]);
        CHECK(std::abs(zT) < 3.5);
        CHECK(std::abs(zS) < 3.5);
        CHECK(c[1] == 0);
        CHECK(c[2] == c[4]);   // every leaf cell the ray crosses, once
        if (ray.leaves != 0) {
            CHECK(c[3] == ray.leaves);
        }
    }
}

// LEVEL SETS: OpenVDB's level set of a 32^3 box, a band of three voxels a
// side, read and made a fog on the device. The class and background are the
// header's; the medium then reads 0 outside, a ramp over the band's inner
// half and 1 beyond it -- its interior the level set's inactive tiles, with
// no leaf -- so along an axis through the middle the optical depth is
// sigma (L - halfWidth): exp(-0.5 x 2.9) for L 3.2, half width 0.3. And no
// density, in the band or a tile, over the majorant over it.
TEST_CASE("a level set volume is read as a fog: Beer-Lambert through its interior tiles and its band, never over its "
          "majorant",
          "[volume][gpu][medium][levelset]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume";
    fs::create_directories(dir);
    const fs::path path = dir / "levelset_box.vdb";
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 1.0F};
    if (auto written = io::writeVdbLevelSetBox(path, "surface", 0.1, box, 3.0F); !written) {
        FAIL(written.error().toString());
    }
    auto grid = io::readVdbGrid(path, "surface");
    if (!grid) FAIL(grid.error().toString());
    std::printf("  level set: %u leaves, background %.3f\n", grid->leafCount, double(grid->background));
    CHECK(grid->levelSet);
    CHECK(grid->background == Catch::Approx(0.3).margin(1e-5));
    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    world::VolumeInput input;
    input.grid = &*grid;
    input.densityScale = 0.5F;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
        REQUIRE(batch.submit(true));
    }
    auto rays = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumRays");
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumReduce");
    auto majorant = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumMajorant");
    if (!rays) FAIL(rays.error().toString());
    if (!reduce) FAIL(reduce.error().toString());
    if (!majorant) FAIL(majorant.error().toString());
    constexpr uint32_t kRays = 16384;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "levelset.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "levelset.sums");
    gpu::Buffer perRay = test::uintBuffer(*gpu->device, uint64_t{kRays} * 2, "levelset.perRay");
    struct Ray {
        float origin[3];
        float direction[3];
        const char* name;
    };
    const float expected = std::exp(-0.5F * 2.9F);
    const Ray cases[] = {
        {{-1.0F, 1.65F, 1.65F}, {1.0F, 0.0F, 0.0F}, "x"},
        {{1.65F, 4.2F, 1.65F}, {0.0F, -1.0F, 0.0F}, "-y"},
        {{1.65F, 1.65F, -1.0F}, {0.0F, 0.0F, 1.0F}, "z"},
    };
    for (const Ray& ray : cases) {
        const auto bind = [&](rhi::ShaderCursor c) {
            c["medium"]["rays"].setData(kRays);
            c["medium"]["volume"].setData(uint32_t{0});
            c["medium"]["expectedT"].setData(expected);
            c["medium"]["origin"].setData(ray.origin, sizeof(ray.origin));
            c["medium"]["tMin"].setData(0.0F);
            c["medium"]["direction"].setData(ray.direction, sizeof(ray.direction));
            c["medium"]["tMax"].setData(10.0F);
            const float boxMin[3] = {-0.4F, -0.4F, -0.4F};
            const float boxMax[3] = {3.6F, 3.6F, 3.6F};
            c["medium"]["boxMin"].setData(boxMin, sizeof(boxMin));
            c["medium"]["boxMax"].setData(boxMax, sizeof(boxMax));
            c["words"].setBinding(set->words().rhi());
            c["counts"].setBinding(counts.rhi());
            c["sums"].setBinding(sums.rhi());
            c["perRay"].setBinding(perRay.rhi());
        };
        const uint32_t zeros[6] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        {
            gpu::CommandBatch batch(*gpu->device);
            rays->dispatch(batch, {kRays, 1, 1}, bind);
            reduce->dispatch(batch, {1, 1, 1}, bind);
            majorant->dispatch(batch, {kRays, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t c[6] = {};
        float s[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
        const double n = kRays;
        const double meanT = s[0] / n;
        const double varT = std::max(s[1] / n - meanT * meanT, 0.0);
        const double zT = (meanT - expected) / std::max(std::sqrt(varT / n), 1e-9);
        const double meanS = s[2] / n;
        const double varS = std::max(s[3] / n - meanS * meanS, 0.0);
        const double zS = (meanS - (1.0 - expected)) / std::max(std::sqrt(varS / n), 1e-9);
        std::printf("  %s through the level set: T %.4f by ratio tracking (expected %.4f, z %.2f); scattered %.4f "
                    "(z %.2f); %u of %u probes over their majorant\n",
                    ray.name, meanT, static_cast<double>(expected), zT, meanS, zS, c[1], c[5]);
        CHECK(std::abs(zT) < 3.5);
        CHECK(std::abs(zS) < 3.5);
        CHECK(c[1] == 0);
    }
}

// GRID TYPES: a box filled in each value type a .vdb holds -- float, a
// float stored as half, double, int32, Vec3f, Vec3d and a mask -- read and
// laid out as NanoVDB's own grid of that type. Every voxel of a region
// about the box, read on the device through the typed accessor, is the
// value it was filled with (a double or a half that a float holds exactly,
// so the comparison is equality), and 0 outside; and the medium on each
// is Beer-Lambert's with the scalar it reads -- a vector's length, an
// int's value, a mask's 1 -- within its own standard error.
TEST_CASE("every grid type reads on the device as it was filled, and a medium on each is Beer-Lambert's",
          "[volume][gpu][medium][types]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume";
    fs::create_directories(dir);
    struct Case {
        const char* name;
        io::VdbType type;
        bool        half;
        uint32_t    nanoType;   // PNANOVDB_GRID_TYPE_*
        double      written[3]; // what the box is filled with (a scalar: the first)
        float       inside[4];  // the value read back: xyz, components
        float       scalar;     // what the medium reads
    };
    // Written values a float does not hold exactly show the conversions: a
    // half stores 0.1 as 0.0999755859375, read back as that; a double 0.1
    // reads as the float nearest it, 0.1F, which a truncation would miss by
    // one bit; 0.3 and 0.4 as doubles become the floats nearest them.
    const Case cases[] = {
        {"float", io::VdbType::Float, false, 1, {0.5, 0, 0}, {0.5F, 0.0F, 0.0F, 1.0F}, 0.5F},
        {"float saved as half", io::VdbType::Float, true, 1, {0.1, 0, 0}, {0.0999755859375F, 0.0F, 0.0F, 1.0F},
         0.0999755859375F},
        {"double", io::VdbType::Double, false, 2, {0.1, 0, 0}, {0.1F, 0.0F, 0.0F, 1.0F}, 0.1F},
        {"int32", io::VdbType::Int32, false, 4, {2.0, 0, 0}, {2.0F, 0.0F, 0.0F, 1.0F}, 2.0F},
        {"vec3f", io::VdbType::Vec3f, false, 6, {0.375, 0.5, 0}, {0.375F, 0.5F, 0.0F, 3.0F}, 0.625F},
        {"vec3d", io::VdbType::Vec3d, false, 7, {0.3, 0.4, 0}, {0.3F, 0.4F, 0.0F, 3.0F}, 0.5F},
        {"mask", io::VdbType::Mask, false, 8, {1.0, 0, 0}, {1.0F, 0.0F, 0.0F, 1.0F}, 1.0F},
    };
    auto typedKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_check", "volumeTypedCheck");
    auto rays = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumRays");
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "athenea/test/medium_check", "mediumReduce");
    if (!typedKernel) FAIL(typedKernel.error().toString());
    if (!rays) FAIL(rays.error().toString());
    if (!reduce) FAIL(reduce.error().toString());
    constexpr uint32_t kRays = 16384;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "types.counts");
    gpu::Buffer typedCounts = test::uintBuffer(*gpu->device, 4, "types.typedCounts");
    gpu::Buffer scalarCounts = test::uintBuffer(*gpu->device, 4, "types.scalarCounts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "types.sums");
    gpu::Buffer perRay = test::uintBuffer(*gpu->device, uint64_t{kRays} * 2, "types.perRay");
    for (const Case& tc : cases) {
        const fs::path path = dir / (std::string("typed_") + std::to_string(int(tc.type)) + (tc.half ? "h" : "") + ".vdb");
        io::VdbBox box{{0, 0, 0}, {16, 16, 16}, tc.written[0], {tc.written[0], tc.written[1], tc.written[2]}};
        if (auto written = io::writeVdbBoxes(path, "field", 0.1, std::span<const io::VdbBox>(&box, 1), tc.type, tc.half);
            !written) {
            FAIL(written.error().toString());
        }
        auto grid = io::readVdbGrid(path, "field");
        if (!grid) FAIL(grid.error().toString());
        CHECK(grid->type == tc.type);
        CHECK(grid->savedAsHalf == tc.half);
        CHECK(grid->nanoType == tc.nanoType);
        auto set = world::VolumeSet::create(*gpu->library);
        if (!set) FAIL(set.error().toString());
        world::VolumeInput input;
        input.grid = &*grid;
        input.densityScale = 1.0F;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
            REQUIRE(batch.submit(true));
        }
        const uint32_t zeros[6] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        REQUIRE(typedCounts.write(*gpu->device, 0, 16, zeros));
        REQUIRE(scalarCounts.write(*gpu->device, 0, 16, zeros));
        const float expectedT = std::exp(-tc.scalar * 1.6F);
        const float origin[3] = {-1.0F, 0.75F, 0.75F};
        const float direction[3] = {1.0F, 0.0F, 0.0F};
        // A grid not of floats is laid out after its float topology, which
        // the medium walks: the typed grid is placement 1, the filled
        // float one placement 0.
        CHECK((grid->scalar != nullptr) == (tc.nanoType != 1));
        CHECK(set->count() == (grid->scalar != nullptr ? 2u : 1u));
        const uint32_t typedPlacement = grid->scalar != nullptr ? 1u : 0u;
        const float scalarInside[4] = {tc.scalar, 0.0F, 0.0F, 1.0F};
        {
            gpu::CommandBatch batch(*gpu->device);
            const auto check = [&](gpu::Buffer& into, uint32_t placement, uint32_t type, const float* inside) {
                typedKernel->dispatch(batch, {20 * 20 * 20, 1, 1}, [&](rhi::ShaderCursor c) {
                    c["words"].setBinding(set->words().rhi());
                    c["counts"].setBinding(into.rhi());
                    c["typed"]["gridWord"].setData(set->placement(placement).gridWord);
                    c["typed"]["type"].setData(type);
                    c["typed"]["lo"].setData(int32_t{-2});
                    c["typed"]["hi"].setData(int32_t{18});
                    c["typed"]["inside"].setData(inside, sizeof(float) * 4);
                    const int32_t lo[3] = {0, 0, 0};
                    const int32_t hi[3] = {16, 16, 16};
                    c["typed"]["boxMin"].setData(lo, sizeof(lo));
                    c["typed"]["boxMax"].setData(hi, sizeof(hi));
                });
            };
            check(typedCounts, typedPlacement, tc.nanoType, tc.inside);
            check(scalarCounts, 0, 1u, scalarInside);
            const auto bind = [&](rhi::ShaderCursor c) {
                c["medium"]["rays"].setData(kRays);
                c["medium"]["volume"].setData(uint32_t{0});
                c["medium"]["expectedT"].setData(expectedT);
                c["medium"]["origin"].setData(origin, sizeof(origin));
                c["medium"]["tMin"].setData(0.0F);
                c["medium"]["direction"].setData(direction, sizeof(direction));
                c["medium"]["tMax"].setData(10.0F);
                c["words"].setBinding(set->words().rhi());
                c["counts"].setBinding(counts.rhi());
                c["sums"].setBinding(sums.rhi());
                c["perRay"].setBinding(perRay.rhi());
            };
            rays->dispatch(batch, {kRays, 1, 1}, bind);
            reduce->dispatch(batch, {1, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t c[4] = {};
        uint32_t f[4] = {};
        float s[4] = {};
        REQUIRE(typedCounts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(scalarCounts.read(*gpu->device, 0, sizeof(f), f));
        REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
        const double n = kRays;
        const double meanT = s[0] / n;
        const double varT = std::max(s[1] / n - meanT * meanT, 0.0);
        const double zT = (meanT - expectedT) / std::max(std::sqrt(varT / n), 1e-9);
        const double meanS = s[2] / n;
        const double varS = std::max(s[3] / n - meanS * meanS, 0.0);
        const double zS = (meanS - (1.0 - expectedT)) / std::max(std::sqrt(varS / n), 1e-9);
        std::printf("  %s: %zu words, %u leaves; %u of %u voxels off their value, header type %s; the medium's "
                    "float grid %u of %u off its scalar; T %.4f (expected %.4f, z %.2f), scattered z %.2f\n",
                    tc.name, grid->words.size(), grid->leafCount, c[0], c[2], c[1] == 0 ? "right" : "WRONG", f[0],
                    f[2], meanT, double(expectedT), zT, zS);
        CHECK(f[0] == 0);
        CHECK(f[1] == 0);
        CHECK(c[2] == 20u * 20u * 20u);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
        CHECK(std::abs(zT) < 3.5);
        CHECK(std::abs(zS) < 3.5);
    }
}

// FIELD SELECTION (UsdVol's fieldName, fieldIndex, fieldDataType): a file
// holding three grids named density -- float, vec3f, double -- and one
// named temperature. The name picks the grids of that name in the file's
// order, the index counts among them, the declared type must agree with the
// grid the index picks and, without an index, picks the first that does.
// Each grid chosen is checked on the device by its values, which tell the
// grids apart. A vector grid's role comes from the header's vector type.
TEST_CASE("a field's name, index and data type pick the grid UsdVol says, checked on the device by its values",
          "[volume][gpu][types][fields]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume";
    fs::create_directories(dir);
    const fs::path path = dir / "selection.vdb";
    const auto spec = [](const char* name, io::VdbType type, double value, std::array<double, 3> vector,
                         io::VectorRole role = io::VectorRole::None) {
        io::VdbGridSpec g;
        g.name = name;
        g.type = type;
        g.voxelSize = 0.1;
        g.boxes = {io::VdbBox{{0, 0, 0}, {16, 16, 16}, value, vector}};
        g.vectorRole = role;
        return g;
    };
    const io::VdbGridSpec grids[] = {
        spec("density", io::VdbType::Float, 0.25, {0, 0, 0}),
        spec("density", io::VdbType::Vec3f, 0, {0.375, 0.5, 0}, io::VectorRole::Vector),
        spec("density", io::VdbType::Double, 0.75, {0, 0, 0}),
        spec("temperature", io::VdbType::Float, 3.0, {0, 0, 0}),
    };
    if (auto written = io::writeVdbGrids(path, grids); !written) FAIL(written.error().toString());
    auto names = io::vdbGridNames(path);
    if (!names) FAIL(names.error().toString());
    std::printf("  the file's grids: %zu\n", names->size());
    CHECK(names->size() == 4);

    auto typedKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_check", "volumeTypedCheck");
    if (!typedKernel) FAIL(typedKernel.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "selection.counts");
    struct Pick {
        const char* what;
        io::VdbSelection selection;
        uint32_t nanoType;
        float inside[4];
    };
    const Pick picks[] = {
        {"density alone: the first", {"density", -1, ""}, 1, {0.25F, 0, 0, 1}},
        {"density, index 1", {"density", 1, ""}, 6, {0.375F, 0.5F, 0, 3}},
        {"density, index 2, double", {"density", 2, "double"}, 2, {0.75F, 0, 0, 1}},
        {"density, float3 without an index", {"density", -1, "float3"}, 6, {0.375F, 0.5F, 0, 3}},
        {"temperature", {"temperature", -1, ""}, 1, {3.0F, 0, 0, 1}},
        {"no name, double", {"", -1, "double"}, 2, {0.75F, 0, 0, 1}},
        {"no name, index 3", {"", 3, ""}, 1, {3.0F, 0, 0, 1}},
    };
    for (const Pick& pick : picks) {
        auto grid = io::readVdbGrid(path, pick.selection);
        if (!grid) FAIL(std::string(pick.what) + ": " + grid.error().toString());
        auto set = world::VolumeSet::create(*gpu->library);
        if (!set) FAIL(set.error().toString());
        world::VolumeInput input;
        input.grid = &*grid;
        {
            gpu::CommandBatch batch(*gpu->device);
            REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
            REQUIRE(batch.submit(true));
        }
        const uint32_t zeros[4] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        {
            gpu::CommandBatch batch(*gpu->device);
            typedKernel->dispatch(batch, {20 * 20 * 20, 1, 1}, [&](rhi::ShaderCursor c) {
                c["words"].setBinding(set->words().rhi());
                c["counts"].setBinding(counts.rhi());
                c["typed"]["gridWord"].setData(set->placement(grid->scalar != nullptr ? 1 : 0).gridWord);
                c["typed"]["type"].setData(pick.nanoType);
                c["typed"]["lo"].setData(int32_t{-2});
                c["typed"]["hi"].setData(int32_t{18});
                c["typed"]["inside"].setData(pick.inside, sizeof(pick.inside));
                const int32_t lo[3] = {0, 0, 0};
                const int32_t hi[3] = {16, 16, 16};
                c["typed"]["boxMin"].setData(lo, sizeof(lo));
                c["typed"]["boxMax"].setData(hi, sizeof(hi));
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  %s: grid '%s' (%s), %u of %u voxels off its values, type %s\n", pick.what,
                    grid->name.c_str(), io::vdbTypeName(grid->type), c[0], c[2], c[1] == 0 ? "right" : "WRONG");
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
        CHECK(c[2] == 8000);
        if (pick.nanoType == 6) {
            CHECK(grid->vectorRole == io::VectorRole::Vector);
        }
    }
    // What is refused, with the reason.
    const auto refused = [&](const io::VdbSelection& selection, ErrorCode code) {
        auto grid = io::readVdbGrid(path, selection);
        REQUIRE_FALSE(grid);
        std::printf("  refused: %s\n", grid.error().toString().c_str());
        CHECK(grid.error().code() == code);
    };
    refused({"density", 1, "float"}, ErrorCode::InvalidArgument);   // index 1 is the vector
    refused({"density", 3, ""}, ErrorCode::NotFound);               // three of that name
    refused({"temperature", -1, "double3"}, ErrorCode::InvalidArgument);
    refused({"density", -1, "string"}, ErrorCode::Unsupported);
    refused({"smoke", -1, ""}, ErrorCode::NotFound);
    CHECK(io::vectorRoleFromHint("Color").value() == io::VectorRole::Color);
    CHECK_FALSE(io::vectorRoleFromHint("Colour"));
}

// EVERY FIELD BY NAME: a volume with eight fields of every type, at two
// voxel sizes -- temperature at 0.2 where the density is at 0.1 -- laid out
// in one VolumeSet, each a table entry in the record. A kernel names each
// field by its hash and reads it at a world point inside the boxes and one
// outside: every value is the one it was filled with (0 outside), with the
// role its name gives and the vector role it was given; a name the volume
// does not have finds no slot.
TEST_CASE("a volume's fields are each reachable on the device by name, at their own voxel size",
          "[volume][gpu][fields]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume";
    fs::create_directories(dir);
    struct Field {
        const char*    name;
        io::VdbType    type;
        double         voxel;
        int            side;   // the box [0, side)^3 in its own voxels: 1.6 world units either way
        double         written[3];
        float          read[4];
        world::FieldRole role;
        io::VectorRole vectorRole;
    };
    const Field fields[] = {
        {"density", io::VdbType::Float, 0.1, 16, {0.5, 0, 0}, {0.5F, 0, 0, 1}, world::FieldRole::Density,
         io::VectorRole::None},
        {"temperature", io::VdbType::Float, 0.2, 8, {300.0, 0, 0}, {300.0F, 0, 0, 1}, world::FieldRole::Temperature,
         io::VectorRole::None},
        {"flame", io::VdbType::Double, 0.1, 16, {0.1, 0, 0}, {0.1F, 0, 0, 1}, world::FieldRole::Flame,
         io::VectorRole::None},
        {"Cd", io::VdbType::Vec3f, 0.1, 16, {0.25, 0.5, 0.75}, {0.25F, 0.5F, 0.75F, 3}, world::FieldRole::Color,
         io::VectorRole::Color},
        {"vel", io::VdbType::Vec3d, 0.2, 8, {0.3, 0.4, 0}, {0.3F, 0.4F, 0, 3}, world::FieldRole::Velocity,
         io::VectorRole::Vector},
        {"active", io::VdbType::Mask, 0.1, 16, {1, 0, 0}, {1.0F, 0, 0, 1}, world::FieldRole::Other,
         io::VectorRole::None},
        {"count", io::VdbType::Int32, 0.1, 16, {7, 0, 0}, {7.0F, 0, 0, 1}, world::FieldRole::Other,
         io::VectorRole::None},
        {"rest", io::VdbType::Vec3f, 0.1, 16, {1, 2, 3}, {1.0F, 2.0F, 3.0F, 3}, world::FieldRole::Other,
         io::VectorRole::Point},
    };
    std::vector<io::NanoGrid> grids;
    grids.reserve(8);
    for (const Field& f : fields) {
        const fs::path path = dir / (std::string("field_") + f.name + ".vdb");
        io::VdbGridSpec spec;
        spec.name = f.name;
        spec.type = f.type;
        spec.voxelSize = f.voxel;
        spec.boxes = {io::VdbBox{{0, 0, 0}, {f.side, f.side, f.side}, f.written[0],
                                 {f.written[0], f.written[1], f.written[2]}}};
        if (auto written = io::writeVdbGrids(path, std::span<const io::VdbGridSpec>(&spec, 1)); !written) {
            FAIL(written.error().toString());
        }
        auto grid = io::readVdbGrid(path, f.name);
        if (!grid) FAIL(grid.error().toString());
        grids.push_back(std::move(*grid));
    }
    world::VolumeInput input;
    input.grid = &grids[0];
    for (size_t k = 0; k < 8; ++k) {
        world::VolumeFieldInput field;
        field.grid = &grids[k];
        field.nameHash = world::fieldNameHash(fields[k].name);
        field.role = world::fieldRoleOf(fields[k].name);
        field.vectorRole = fields[k].vectorRole;
        CHECK(field.role == fields[k].role);
        input.fields.push_back(field);
    }
    // A second volume after it, translated, reading the same grids: the
    // table is the record's own, and world points move with it.
    world::VolumeInput moved = input;
    moved.objectToWorld = render::Mat4::identity();
    moved.objectToWorld.at(0, 3) = 10.0;
    const world::VolumeInput volumes[] = {input, moved};
    auto set = world::VolumeSet::create(*gpu->library);
    if (!set) FAIL(set.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(set->set(batch, volumes));
        REQUIRE(batch.submit(true));
    }
    CHECK(set->count() == 8);
    // The probes: each field inside its box at (0.85, 0.65, 1.05) and outside
    // at (2.5, 0.5, 0.5), in each volume; and a name neither has.
    std::vector<uint32_t> probes;
    const auto bits = [](float v) {
        uint32_t b = 0;
        std::memcpy(&b, &v, 4);
        return b;
    };
    const auto probe = [&](float x, float y, float z, uint32_t hash, const float* value, uint32_t role,
                           uint32_t vectorRole, int32_t slot) {
        probes.insert(probes.end(), {bits(x), bits(y), bits(z), hash, bits(value[0]), bits(value[1]), bits(value[2]),
                                     bits(value[3]), role, vectorRole, static_cast<uint32_t>(slot), 0u});
    };
    for (float dx : {0.0F, 10.0F}) {
        probes.clear();
        for (size_t k = 0; k < 8; ++k) {
            const uint32_t hash = world::fieldNameHash(fields[k].name);
            const float outside[4] = {0.0F, 0.0F, 0.0F, fields[k].read[3]};
            probe(dx + 0.85F, 0.65F, 1.05F, hash, fields[k].read, uint32_t(fields[k].role),
                  uint32_t(fields[k].vectorRole), int32_t(k));
            probe(dx + 2.5F, 0.5F, 0.5F, hash, outside, uint32_t(fields[k].role), uint32_t(fields[k].vectorRole),
                  int32_t(k));
        }
        const float none[4] = {};
        probe(dx + 0.85F, 0.65F, 1.05F, world::fieldNameHash("smoke"), none, 0, 0, -1);
        auto probeBuffer = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, probes, "fields.probes");
        REQUIRE(probeBuffer);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "fields.counts");
        auto kernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/fields_check", "volumeFieldsCheck");
        if (!kernel) FAIL(kernel.error().toString());
        const uint32_t n = static_cast<uint32_t>(probes.size() / 12);
        {
            gpu::CommandBatch batch(*gpu->device);
            kernel->dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
                c["fieldsCheck"]["probes"].setData(n);
                c["fieldsCheck"]["volume"].setData(uint32_t{dx == 0.0F ? 0u : 1u});
                c["words"].setBinding(set->words().rhi());
                c["probes"].setBinding(probeBuffer->rhi());
                c["counts"].setBinding(counts.rhi());
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[4] = {};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        std::printf("  volume %s: %u probes; %u found the wrong slot, %u read the wrong value, %u the wrong role\n",
                    dx == 0.0F ? "at the origin" : "moved by 10", c[3], c[0], c[1], c[2]);
        CHECK(c[3] == 17);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
        CHECK(c[2] == 0);
    }
    // A ninth field is refused.
    world::VolumeInput tooMany = input;
    tooMany.fields.push_back(input.fields[0]);
    gpu::CommandBatch batch(*gpu->device);
    CHECK_FALSE(set->set(batch, std::span<const world::VolumeInput>(&tooMany, 1)));
}

// THE MEDIUM'S RANDOM STREAMS DO NOT MEET. One pixel's streams as the path
// tracer opens them -- 4096 paths, 8 bounces, the 4 purposes a medium draws
// for -- each over its first 64 draws: 8.4 million states, sorted on the
// device and counted where two neighbours are equal. Two streams through
// one state draw the same numbers from there on, so a pixel's paths share
// randomness and its error stops falling with their number. The legacy
// construction (one LCG word seeded from a 24-bit float: every stream a
// stretch of one cycle of 2^32, started at one of 2^24 points) meets itself
// by the thousands; the keyed streams only where two 32-bit keys collide.
TEST_CASE("a pixel's medium random streams do not run into one another", "[volume][gpu][random]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto states = gpu::ComputeKernel::create(*gpu->library, "athenea/test/streams_check", "streamStates");
    auto equals = gpu::ComputeKernel::create(*gpu->library, "athenea/test/streams_check", "streamEquals");
    auto sort = gpu::RadixSort::create(*gpu->library);
    if (!states) FAIL(states.error().toString());
    if (!equals) FAIL(equals.error().toString());
    if (!sort) FAIL(sort.error().toString());
    constexpr uint32_t kPaths = 4096, kBounces = 8, kSteps = 64;
    constexpr uint32_t kStreams = kPaths * kBounces * 4;
    constexpr uint32_t kStates = kStreams * kSteps;
    gpu::SortBuffers buffers;
    buffers.keysLo = test::uintBuffer(*gpu->device, kStates, "streams.keysLo");
    buffers.keysHi = test::uintBuffer(*gpu->device, kStates, "streams.keysHi");
    buffers.values = test::uintBuffer(*gpu->device, kStates, "streams.values");
    buffers.scratchKeysLo = test::uintBuffer(*gpu->device, kStates, "streams.scratchKeysLo");
    buffers.scratchKeysHi = test::uintBuffer(*gpu->device, kStates, "streams.scratchKeysHi");
    buffers.scratchValues = test::uintBuffer(*gpu->device, kStates, "streams.scratchValues");
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "streams.counts");
    uint32_t equal[2] = {};
    for (uint32_t legacy = 0; legacy < 2; ++legacy) {
        const uint32_t zeros[4] = {};
        REQUIRE(counts.write(*gpu->device, 0, sizeof(zeros), zeros));
        const auto bind = [&](rhi::ShaderCursor c) {
            c["streams"]["paths"].setData(kPaths);
            c["streams"]["bounces"].setData(kBounces);
            c["streams"]["steps"].setData(kSteps);
            c["streams"]["legacy"].setData(legacy);
            c["streams"]["pixel"].setData(uint32_t{1234});
            c["streams"]["seed"].setData(uint32_t{0});
            c["keysLo"].setBinding(buffers.keysLo.rhi());
            c["keysHi"].setBinding(buffers.keysHi.rhi());
            c["counts"].setBinding(counts.rhi());
        };
        {
            gpu::CommandBatch batch(*gpu->device);
            states->dispatch(batch, {kStreams, 1, 1}, bind);
            REQUIRE(sort->sort(batch, buffers, kStates, legacy != 0 ? 32 : 64));
            equals->dispatch(batch, {kStates, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        REQUIRE(counts.read(*gpu->device, 0, 4, &equal[legacy]));
    }
    // Keyed: a state repeats only where two of the 131072 keys collide,
    // n^2 / 2^33 = 2 pairs expected, 64 states each.
    std::printf("  %u streams of %u draws: %u repeated states keyed, %u with the legacy LCG\n", kStreams, kSteps,
                equal[0], equal[1]);
    CHECK(equal[0] <= 8u * kSteps);
    CHECK(equal[1] > 1000u);
}
