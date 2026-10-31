// Copyright (c) 2026 jesus luque.
//
// A volume into volumetric gaussians (proposal 103), through the Vdb2Splat
// effect as `athenea vdb2splat` runs it. The fixtures are made on the
// device: a box of active voxels (writeVdbBoxes, every value zero) whose
// values a kernel writes -- a known mixture of gaussians evaluated at the
// voxel centres (world::VolumeSplats::randomMixture, then the way back,
// splatToGrid), or a smooth noise blob. Then:
//  - one gaussian is recovered exactly: one gaussian in, a budget of one,
//    and the grid it gives back on the same topology within 0.5% (relative
//    L2) of the grid it came from;
//  - a mixture of sixteen, with a budget of sixty-four, within 3%;
//  - on the blob, the error falls as the budget grows;
//  - the closed-form transmittance of the known mixture along random rays
//    is ratio tracking's through its grid, within five standard errors and
//    what the grid's sampling costs (0.01).
// What crosses to the processor is the effect's counts and the comparison
// kernels' sums.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "aofx/Effect.h"
#include "athenea/aofx/EffectRegistry.h"
#include "athenea/aofx/EffectRender.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"
#include "athenea/gpu_host/ImageStorage.h"
#include "athenea/image/Image.h"
#include "athenea/io/Vdb.h"
#include "athenea/world/VolumeSet.h"
#include "athenea/world/VolumeSplats.h"

using namespace athenea;

namespace {

gpu::Buffer viewOf(gpu_host::Context& context, const image::ImagePtr& picture) {
    gpu_host::ImageStorage* storage = context.sharedStorage();
    REQUIRE(storage != nullptr);
    const uint64_t buffer = storage->bufferFor(picture->address());
    REQUIRE(buffer != 0);
    auto view = context.renderView(buffer, picture->sizeBytes(), 16, "test.image");
    REQUIRE(view);
    return std::move(*view);
}

image::ImagePtr pictureOf(uint64_t entries) {
    const uint64_t rows = (entries + 4095) / 4096;
    auto made = image::Image::create({0, 0, 4096, static_cast<int32_t>(std::max<uint64_t>(rows, 1))});
    REQUIRE(made);
    return *made;
}

/// A box of `size` voxels a side, every voxel active at zero.
io::NanoGrid boxGrid(int32_t size) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "athenea-tests" / "vdb2splat";
    std::filesystem::create_directories(dir);
    const std::filesystem::path path = dir / ("box" + std::to_string(size) + ".vdb");
    const io::VdbBox box{{0, 0, 0}, {size, size, size}, 0.0F};
    REQUIRE(io::writeVdbBoxes(path, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    auto grid = io::readVdbGrid(path, "density");
    REQUIRE(grid);
    return std::move(*grid);
}

struct Fixture {
    io::NanoGrid       grid;
    world::PackedGrids packed;
    gpu::Buffer        words;
    gpu::Buffer        mixture;   // the known gaussians, where there are
    uint32_t           known = 0;
};

Fixture mixtureFixture(gpu::Device& device, const world::VolumeSplats& splats, int32_t size, uint32_t count,
                       float minStd, float maxStd, uint32_t seed) {
    Fixture f;
    f.grid = boxGrid(size);
    f.packed = world::PackedGrids::of(f.grid, nullptr, nullptr);
    auto words = gpu::Buffer::fromSpan<uint32_t>(device, f.packed.words, "test.words");
    REQUIRE(words);
    f.words = std::move(*words);
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{count} * 64;
    desc.elementBytes = 16;
    desc.label = "test.mixture";
    auto mixture = gpu::Buffer::create(device, desc);
    REQUIRE(mixture);
    f.mixture = std::move(*mixture);
    f.known = count;
    {
        gpu::CommandBatch batch(device);
        const float s = static_cast<float>(size);
        splats.randomMixture(batch, f.mixture, count, {s, s, s}, s * 0.3F, minStd, maxStd, seed);
        REQUIRE(batch.submit(true));
    }
    REQUIRE(splats.splatToGrid(f.mixture, count, f.words, f.packed.density, f.packed.leafCount));
    return f;
}

Fixture blobFixture(gpu::Device& device, const world::VolumeSplats& splats, int32_t size) {
    Fixture f;
    f.grid = boxGrid(size);
    f.packed = world::PackedGrids::of(f.grid, nullptr, nullptr);
    auto words = gpu::Buffer::fromSpan<uint32_t>(device, f.packed.words, "test.words");
    REQUIRE(words);
    f.words = std::move(*words);
    gpu::CommandBatch batch(device);
    const float c = static_cast<float>(size) * 0.5F;
    splats.fixture(batch, f.words, f.packed, 1, {c, c, c}, static_cast<float>(size) * 0.4F, 3u);
    REQUIRE(batch.submit(true));
    return f;
}

struct Fitted {
    uint32_t        count = 0;
    float           ownError = 0.0F;   // the effect's relative L2
    double          backError = 0.0;   // the way back's, against the fixture
    image::ImagePtr records;
};

Fitted fit(gpu_host::Context& context, aofx::Effect& effect, const world::VolumeSplats& splats, gpu::Device& device,
           const Fixture& f, uint32_t target, uint32_t iterations, double initFraction = 0.6) {
    const image::ImagePtr leaves = pictureOf(uint64_t{f.packed.leafCount} * world::VolumeSplats::kLeafEntries);
    {
        const gpu::Buffer view = viewOf(context, leaves);
        gpu::CommandBatch batch(device);
        splats.packLeaves(batch, f.words, f.packed, view);
        REQUIRE(batch.submit(true));
        leaves->deviceWrote();
    }
    const image::ImagePtr out = pictureOf(uint64_t{target} * 4);
    aofx_host::EffectJob job;
    job.bounds = out->bounds();
    job.inputs.push_back({"Grid", leaves});
    const auto number = [&](const char* name, double value) {
        job.params.push_back(aofx::ParamValue{name, {value}, {}});
    };
    number("leaves", f.packed.leafCount);
    number("target", target);
    number("iterations", iterations);
    number("initFraction", initFraction);
    auto rendered = aofx_host::renderEffect(context, effect, job);
    REQUIRE(rendered);
    const std::vector<float>* told = (*rendered)->attached("gaussians");
    REQUIRE(told != nullptr);
    REQUIRE(told->size() >= 8);
    Fitted r;
    r.count = static_cast<uint32_t>((*told)[0]);
    r.ownError = (*told)[3];
    r.records = *rendered;
    // The way back, on a copy of the fixture's words.
    auto back = gpu::Buffer::fromSpan<uint32_t>(device, f.packed.words, "test.back");
    REQUIRE(back);
    const gpu::Buffer records = viewOf(context, r.records);
    REQUIRE(splats.splatToGrid(records, r.count, *back, f.packed.density, f.packed.leafCount));
    auto error = splats.compare(*back, f.words, f.packed.density, f.packed.leafCount);
    REQUIRE(error);
    r.backError = error->relativeL2;
    return r;
}

struct Setup {
    gpu_host::Context*           gpu = nullptr;
    aofx_host::EffectRegistry    registry;
    aofx::Effect*                effect = nullptr;
};

}   // namespace

TEST_CASE("a volume's gaussians give its grid back: one exactly, a mixture of sixteen within 3%",
          "[aofx][vdb2splat]") {
    if (!io::haveOpenVdb()) SKIP("built without OpenVDB");
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) SKIP("no gpe device");
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* effect = registry.find("rt.sparrow.aofx.vdb2splat");
    if (effect == nullptr) SKIP("the Vdb2Splat bundle is not built here");
    gpu::ShaderLibrary library(gpu->deviceShared());
    REQUIRE(gpu->run([&] {
        auto splats = world::VolumeSplats::create(library);
        REQUIRE(splats);
        gpu::Device& device = library.device();

        const Fixture one = mixtureFixture(device, *splats, 40, 1, 2.5F, 5.0F, 11u);
        const Fitted single = fit(*gpu, *effect, *splats, device, one, 1, 150, 1.0);
        std::printf("  one gaussian, a budget of one: %u out, the fit's relative L2 %.5f, the way back's %.5f\n",
                    single.count, static_cast<double>(single.ownError), single.backError);
        CHECK(single.count == 1);
        CHECK(single.backError < 0.005);
        CHECK(std::abs(single.backError - static_cast<double>(single.ownError)) < 1e-3);

        const Fixture sixteen = mixtureFixture(device, *splats, 48, 16, 2.0F, 4.0F, 5u);
        const Fitted many = fit(*gpu, *effect, *splats, device, sixteen, 64, 200);
        std::printf("  sixteen gaussians, a budget of 64: %u out, relative L2 %.5f (the way back's %.5f)\n",
                    many.count, static_cast<double>(many.ownError), many.backError);
        CHECK(many.count <= 64);
        CHECK(many.backError < 0.03);
    }));
}

TEST_CASE("a smoke blob's density error falls as its gaussians grow", "[aofx][vdb2splat]") {
    if (!io::haveOpenVdb()) SKIP("built without OpenVDB");
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) SKIP("no gpe device");
    aofx_host::EffectRegistry registry;
    registry.addSearchPath(ATHENEA_AOFX_BUNDLES);
    registry.scan(gpu);
    aofx::Effect* effect = registry.find("rt.sparrow.aofx.vdb2splat");
    if (effect == nullptr) SKIP("the Vdb2Splat bundle is not built here");
    gpu::ShaderLibrary library(gpu->deviceShared());
    REQUIRE(gpu->run([&] {
        auto splats = world::VolumeSplats::create(library);
        REQUIRE(splats);
        gpu::Device& device = library.device();
        const Fixture blob = blobFixture(device, *splats, 64);
        double last = 1e30;
        for (const uint32_t n : {250u, 1000u, 4000u}) {
            const Fitted r = fit(*gpu, *effect, *splats, device, blob, n, 100);
            std::printf("  blob 64^3, %u gaussians (budget %u): relative L2 %.5f, the way back's %.5f\n", r.count, n,
                        static_cast<double>(r.ownError), r.backError);
            CHECK(r.count <= n);
            CHECK(r.backError < last);
            CHECK(std::abs(r.backError - static_cast<double>(r.ownError)) < 1e-3);
            last = r.backError;
        }
        CHECK(last < 0.25);
    }));
}

TEST_CASE("a mixture's closed-form transmittance is ratio tracking's through its grid", "[aofx][vdb2splat]") {
    if (!io::haveOpenVdb()) SKIP("built without OpenVDB");
    gpu_host::Context* gpu = gpu_host::installProcessContext();
    if (gpu == nullptr || gpu->compute() == nullptr) SKIP("no gpe device");
    gpu::ShaderLibrary library(gpu->deviceShared());
    REQUIRE(gpu->run([&] {
        auto splats = world::VolumeSplats::create(library);
        REQUIRE(splats);
        gpu::Device& device = library.device();
        const Fixture f = mixtureFixture(device, *splats, 48, 16, 2.5F, 5.0F, 9u);
        // Read back as the grid the medium walks: the fixture's words.
        std::vector<uint32_t> words(f.packed.words.size());
        REQUIRE(f.words.read(device, 0, words.size() * 4, words.data()));
        world::PackedGrids filled = f.packed;
        filled.words = std::move(words);
        const io::NanoGrid grid = filled.unpack(filled.density, f.grid);
        auto set = world::VolumeSet::create(library);
        REQUIRE(set);
        world::VolumeInput input;
        input.grid = &grid;
        input.densityScale = 1.0F;   // voxel 0.1: an optical depth of a few across the box
        {
            gpu::CommandBatch batch(device);
            REQUIRE(set->set(batch, std::span<const world::VolumeInput>(&input, 1)));
            REQUIRE(batch.submit(true));
        }
        auto rays = splats->rays(set->words(), f.mixture, f.known, grid.indexToWorld, 2048, 256, 1u, 0.01F);
        REQUIRE(rays);
        std::printf("  %u rays: mean |T_closed - T_ratio| %.5f, %u beyond 5 standard errors + 0.01, largest "
                    "optical depth %.3f\n",
                    rays->rays, rays->meanDifference, rays->beyond, rays->largestDepth);
        CHECK(rays->largestDepth > 0.5);
        CHECK(rays->beyond <= rays->rays / 500);
        CHECK(rays->meanDifference < 0.01);
    }));
}
