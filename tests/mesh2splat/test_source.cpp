// Copyright (c) 2026 jesus luque.
//
// WHERE ON ITS SOURCE A GAUSSIAN STANDS, AND THE ORDER THAT FOLLOWS FROM IT
// (docs/decisions.md, "A gaussian's place on its source"). ctest runs the
// conversions first (the `mesh2splat_outputs` fixture, tests/CMakeLists.txt):
// tests/data/source_patch.usda -- a patch whose st is turned a quarter and a
// bare quad without st -- twice at resolution 32 and once at 16, its TX
// conversion flattened over three frames with the order kept, and the first
// decimated by pairs. These cases read the files back; every comparison is a
// kernel's (`athenea/test/source_order_check`), and what crosses back is counts.
#include "../gpu/GpuTest.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Readers.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

using namespace athenea;

namespace {

namespace fs = std::filesystem;

fs::path output(const char* name) { return fs::path(ATHENEA_M2S_OUTPUTS) / name; }

/// A cloud's source coordinates and opacities, on the device.
struct Sources {
    uint32_t    count = 0;
    gpu::Buffer mesh, triangle, cell, uv, id, opacity, keys;
};

gpu::Buffer words(test::Gpu& gpu, const void* data, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    auto made = gpu::Buffer::create(*gpu.device, desc, data);
    REQUIRE(made);
    return *made;
}

gpu::ComputeKernel kernel(test::Gpu& gpu, const char* entry) {
    auto k = gpu::ComputeKernel::create(*gpu.library, "athenea/test/source_order_check", entry);
    if (!k) FAIL(k.error().toString());
    return std::move(*k);
}

/// Read back and keyed (srcKeys): a gaussian the file holds empty keyed last.
Sources sourcesOf(test::Gpu& gpu, const fs::path& stage) {
    REQUIRE(fs::exists(stage));
    auto read = usd::readSplatSources(stage);
    if (!read) FAIL(read.error().toString());
    auto raw = usd::readParticleFieldRecords(stage);
    if (!raw) FAIL(raw.error().toString());
    Sources s;
    s.count = raw->count;
    REQUIRE(read->mesh.size() == s.count);
    // Each record's opacity, a field of the records (a copy, not a value).
    std::vector<float> opacity(s.count);
    for (uint32_t i = 0; i < s.count; ++i) {
        opacity[i] = raw->records[size_t{i} * raw->encoding.floatsPerRecord + raw->encoding.opacity];
    }
    s.mesh = words(gpu, read->mesh.data(), s.count, 4, "src.mesh");
    s.triangle = words(gpu, read->triangle.data(), s.count, 4, "src.triangle");
    s.cell = words(gpu, read->cell.data(), s.count, 8, "src.cell");
    s.uv = words(gpu, read->uv.data(), s.count, 8, "src.uv");
    s.id = words(gpu, read->id.data(), uint64_t{s.count} * 2, 4, "src.id");
    s.opacity = words(gpu, opacity.data(), s.count, 4, "src.opacity");
    s.keys = words(gpu, nullptr, s.count, 8, "src.keys");
    auto keys = kernel(gpu, "srcKeys");
    gpu::CommandBatch batch(*gpu.device);
    keys.dispatch(batch, {s.count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["mesh"].setBinding(s.mesh.rhi());
        c["triangle"].setBinding(s.triangle.rhi());
        c["cell"].setBinding(s.cell.rhi());
        c["uv"].setBinding(s.uv.rhi());
        c["opacity"].setBinding(s.opacity.rhi());
        c["keys"].setBinding(s.keys.rhi());
        c["params"]["count"].setData(s.count);
    });
    REQUIRE(batch.submit(true));
    return s;
}

/// Runs one check over the bindings `bind` sets, and hands back its two counts.
std::pair<uint32_t, uint32_t> counted(test::Gpu& gpu, const char* entry, uint32_t threads,
                                      const std::function<void(rhi::ShaderCursor)>& bind) {
    auto k = kernel(gpu, entry);
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "src.counts");
    gpu::CommandBatch batch(*gpu.device);
    k.dispatch(batch, {std::max(threads, 1u), 1, 1}, [&](rhi::ShaderCursor c) {
        bind(c);
        c["counts"].setBinding(counts.rhi());
    });
    REQUIRE(batch.submit(true));
    uint32_t back[2] = {0, 0};
    REQUIRE(counts.read(*gpu.device, 0, sizeof back, back));
    return {back[0], back[1]};
}

uint32_t orderBreaks(test::Gpu& gpu, const Sources& s) {
    return counted(gpu, "srcOrderBreaks", s.count, [&](rhi::ShaderCursor c) {
               c["keys"].setBinding(s.keys.rhi());
               c["params"]["count"].setData(s.count);
           }).first;
}

uint32_t wordsDiffer(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b, uint32_t count, uint32_t width) {
    return counted(gpu, "srcWordsDiffer", count * width, [&](rhi::ShaderCursor c) {
               c["wordsA"].setBinding(a.rhi());
               c["wordsB"].setBinding(b.rhi());
               c["params"]["count"].setData(count);
               c["params"]["width"].setData(width);
           }).first;
}

}   // namespace

// THE SAME POINT OF THE SURFACE IS THE SAME INDEX IN EVERY RUN. Converted twice
// at one density, the two clouds list the same gaussians in the same order:
// every record's words, every source coordinate and every id alike.
TEST_CASE("a gaussian keeps its index and its id from one run to the next", "[mesh2splat][source][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Sources a = sourcesOf(*gpu, output("source_a.usda"));
    const Sources b = sourcesOf(*gpu, output("source_b.usda"));
    REQUIRE(a.count == b.count);
    auto ra = usd::readParticleFieldRecords(output("source_a.usda"));
    auto rb = usd::readParticleFieldRecords(output("source_b.usda"));
    REQUIRE(ra);
    REQUIRE(rb);
    REQUIRE(ra->encoding.floatsPerRecord == rb->encoding.floatsPerRecord);
    const uint32_t width = ra->encoding.floatsPerRecord;
    const gpu::Buffer recordsA = words(*gpu, ra->records.data(), uint64_t{a.count} * width, 4, "src.recordsA");
    const gpu::Buffer recordsB = words(*gpu, rb->records.data(), uint64_t{b.count} * width, 4, "src.recordsB");
    const uint32_t records = wordsDiffer(*gpu, recordsA, recordsB, a.count, width);
    const uint32_t ids = wordsDiffer(*gpu, a.id, b.id, a.count, 2);
    const uint32_t cells = wordsDiffer(*gpu, a.cell, b.cell, a.count, 2);
    const uint32_t uvs = wordsDiffer(*gpu, a.uv, b.uv, a.count, 2);
    std::printf("  two runs of %u gaussians: %u record words, %u id words, %u cell words, %u st words differ\n", a.count,
                records, ids, cells, uvs);
    CHECK(records == 0);
    CHECK(ids == 0);
    CHECK(cells == 0);
    CHECK(uvs == 0);
}

// THE ORDER FOLLOWS THE SURFACE: the cloud is sorted by its source key -- the
// mesh by its path (the bare quad first), then st along a Hilbert curve (the
// patch, whose st is turned a quarter from the order its triangles are walked
// in), or triangle and cell where there is no st -- and what a reader drops
// comes last. And the same holds of a skinned conversion (the corner a joint
// turns, which has no st).
TEST_CASE("a converted cloud is in the order of its surface", "[mesh2splat][source][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    for (const char* name : {"source_a.usda", "source_coarse.usda", "skinned_transfer.usda"}) {
        const Sources s = sourcesOf(*gpu, output(name));
        const uint32_t breaks = orderBreaks(*gpu, s);
        std::printf("  %s: %u gaussians, %u places where the key goes down\n", name, s.count, breaks);
        CHECK(breaks == 0);
    }
}

// A COARSER SET IS ORDERED AS THE FINER ONE IS. The patch at resolution 16
// and at 32: each coarse gaussian's key, placed among the fine keys by
// bisection, falls next to a fine gaussian of the same mesh within a coarse
// cell in st (an eighth: the patch's st is its position turned) -- the coarse
// order is the fine order restricted to it.
TEST_CASE("a coarser conversion's order is the finer one's restricted to it", "[mesh2splat][source][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Sources coarse = sourcesOf(*gpu, output("source_coarse.usda"));
    const Sources fine = sourcesOf(*gpu, output("source_a.usda"));
    REQUIRE(coarse.count < fine.count);
    const auto [far, looked] = counted(*gpu, "srcCoarseNear", coarse.count, [&](rhi::ShaderCursor c) {
        c["mesh"].setBinding(coarse.mesh.rhi());
        c["uv"].setBinding(coarse.uv.rhi());
        c["opacity"].setBinding(coarse.opacity.rhi());
        c["keys"].setBinding(coarse.keys.rhi());
        c["otherKeys"].setBinding(fine.keys.rhi());
        c["otherMesh"].setBinding(fine.mesh.rhi());
        c["otherUv"].setBinding(fine.uv.rhi());
        c["params"]["count"].setData(coarse.count);
        c["params"]["other"].setData(fine.count);
        c["params"]["tolerance"].setData(0.125F);
    });
    std::printf("  %u coarse gaussians with st among %u fine: %u land away from their place\n", looked, fine.count, far);
    CHECK(looked > 50);
    CHECK(far == 0);
}

// THROUGH A FLATTEN: `athenea flatten --range --keep-order` writes each frame's
// pages in the stage's order, so page gaussian i is cloud gaussian i in every
// frame -- its source coordinates and its id are the cloud's. Three frames of
// the patch's TX conversion, each page's gaussians against the cloud's by
// index: each one's distance from the first, which the axes an SPZ file is
// written in (right-up-back) do not change.
TEST_CASE("a flattened frame lists the cloud's gaussians at their indices", "[mesh2splat][source][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto raw = usd::readParticleFieldRecords(output("source_tx.usdc"));
    if (!raw) FAIL(raw.error().toString());
    auto loader = scene::CloudLoader::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    auto cloud = loader->upload(*raw, 0);
    if (!cloud) FAIL(cloud.error().toString());
    for (const char* page : {"f00000_c0000.spz", "f00001_c0000.spz", "f00002_c0000.spz"}) {
        const fs::path path = output("source_pages") / page;
        REQUIRE(fs::exists(path));
        auto spz = io::readSplats(path);
        if (!spz) FAIL(spz.error().toString());
        auto flat = loader->upload(*spz, 0);
        if (!flat) FAIL(flat.error().toString());
        REQUIRE(flat->count == cloud->count);
        const uint32_t moved = counted(*gpu, "srcPositionsMatch", cloud->count, [&](rhi::ShaderCursor c) {
                                   c["positionsA"].setBinding(cloud->positions.rhi());
                                   c["positionsB"].setBinding(flat->positions.rhi());
                                   c["params"]["count"].setData(cloud->count);
                                   c["params"]["other"].setData(flat->count);
                                   c["params"]["tolerance"].setData(2.0e-3F);
                               }).first;
        std::printf("  %s: %u gaussians, %u not where the cloud's of that index is\n", page, flat->count, moved);
        CHECK(moved == 0);
    }
}

// THROUGH A MERGE: a decimation by pairs gives each merge the source
// coordinates of the member that weighs most in it -- its id is one of the
// cloud's -- and lays the result out in the order of the surface again.
TEST_CASE("a merge keeps a member's id and the cloud stays in surface order", "[mesh2splat][source][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Sources whole = sourcesOf(*gpu, output("source_a.usda"));
    const Sources merged = sourcesOf(*gpu, output("source_decimated.usda"));
    REQUIRE(merged.count < whole.count);
    const uint32_t missing = counted(*gpu, "srcIdsFound", merged.count, [&](rhi::ShaderCursor c) {
                                 c["wordsA"].setBinding(whole.id.rhi());
                                 c["wordsB"].setBinding(merged.id.rhi());
                                 c["params"]["count"].setData(whole.count);
                                 c["params"]["other"].setData(merged.count);
                             }).first;
    const uint32_t breaks = orderBreaks(*gpu, merged);
    std::printf("  %u of %u gaussians after the merge: %u ids not the cloud's, %u places where the key goes down\n",
                merged.count, whole.count, missing, breaks);
    CHECK(missing == 0);
    CHECK(breaks == 0);
}
