// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER'S FIELD (SSF phase 1), checked on the device against
// itself: the analytic gradient against central differences, the grid
// against the sum over every blob, one blob's zero set against its
// ellipsoid, -tau exactly where nothing reaches, and the smooth minimum.
// Layers and probes are made by kernels (test/ssf_field_check.slang); only
// counters come back.
#include "../gpu/GpuTest.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/SolidField.h"

using namespace athenea;

namespace {

constexpr const char* kChecks = "athenea/test/ssf_field_check";

struct Probe {
    uint32_t count = 0;
    uint32_t blobs = 0;
    uint32_t side = 0;
    uint32_t seed = 0x5F1Du;
    float    size = 1.0F;
    float    radius = 0.06F;
    float    thickness = 0.05F;
    float    tau = 0.5F;
    float    step = 1.0F / 2048.0F;
    float    gradFloor = 2.0F;   // a tenth of w / c
    float    k = 0.0F;
    float    weight = 1.0F;
    std::array<float, 4> centre{0, 0, 0, 0};
    std::array<float, 4> normal{0, 0, 1, 0};
};

void setProbe(rhi::ShaderCursor p, const Probe& q) {
    p["count"].setData(q.count);
    p["blobs"].setData(q.blobs);
    p["side"].setData(q.side);
    p["seed"].setData(q.seed);
    p["size"].setData(q.size);
    p["radius"].setData(q.radius);
    p["thickness"].setData(q.thickness);
    p["tau"].setData(q.tau);
    p["step"].setData(q.step);
    p["gradFloor"].setData(q.gradFloor);
    p["k"].setData(q.k);
    p["weight"].setData(q.weight);
    p["centre"].setData(q.centre.data(), sizeof q.centre);
    p["normal"].setData(q.normal.data(), sizeof q.normal);
}

gpu::Buffer deviceWords(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    REQUIRE(made);
    return std::move(*made);
}

gpu::ComputeKernel check(test::Gpu& gpu, const char* entry) {
    auto made = gpu::ComputeKernel::create(*gpu.library, kChecks, entry);
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

/// A layer one generator makes, its normals, and its field built.
struct Layer {
    lod::SolidLayer  solid;
    gpu::Buffer      normals;
    lod::SolidField  field;
};

Layer made(test::Gpu& gpu, const char* generator, const Probe& probe) {
    Layer l;
    const uint32_t n = probe.count;
    l.solid.blobs = n;
    l.solid.anchors = 1;
    l.solid.positions = deviceWords(*gpu.device, n, 16, "test.ssf.positions");
    l.solid.shape = deviceWords(*gpu.device, uint64_t{n} * 4, 4, "test.ssf.shape");
    l.solid.anchorWords = deviceWords(*gpu.device, 4, 4, "test.ssf.anchors");
    l.normals = deviceWords(*gpu.device, n, 4, "test.ssf.normals");
    auto field = lod::SolidField::create(*gpu.library);
    if (!field) FAIL(field.error().toString());
    l.field = std::move(*field);
    gpu::ComputeKernel generate = check(gpu, generator);
    gpu::Buffer none = test::uintBuffer(*gpu.device, 4, "test.ssf.none");
    gpu::CommandBatch batch(*gpu.device);
    generate.dispatch(batch, {n, 1, 1}, [&](rhi::ShaderCursor c) {
        c["outPositions"].setBinding(l.solid.positions.rhi());
        c["outShape"].setBinding(l.solid.shape.rhi());
        c["outNormals"].setBinding(l.normals.rhi());
        c["outAnchors"].setBinding(l.solid.anchorWords.rhi());
        c["probes"].setBinding(l.solid.positions.rhi());
        c["wrong"].setBinding(none.rhi());
        c["blobPositions"].setBinding(l.solid.positions.rhi());
        c["blobShape"].setBinding(l.solid.shape.rhi());
        c["blobNormals"].setBinding(l.normals.rhi());
        c["anchors"].setBinding(l.solid.anchorWords.rhi());
        c["gridKeys"].setBinding(none.rhi());
        c["gridOrder"].setBinding(none.rhi());
        c["gridInfo"].setBinding(none.rhi());
        setProbe(c["params"], probe);
    });
    REQUIRE(batch.submit(true));
    REQUIRE(l.field.build(batch, l.solid, l.normals));
    REQUIRE(batch.submit(true));
    return l;
}

/// A check over `probe.count` threads; its four counters.
std::array<uint32_t, 4> run(test::Gpu& gpu, const Layer& l, const char* entry, const Probe& probe,
                            const gpu::Buffer& probes) {
    gpu::ComputeKernel kernel = check(gpu, entry);
    gpu::Buffer wrong = test::uintBuffer(*gpu.device, 4, "test.ssf.wrong");
    gpu::Buffer none = test::uintBuffer(*gpu.device, 4, "test.ssf.none");
    gpu::CommandBatch batch(*gpu.device);
    kernel.dispatch(batch, {probe.count, 1, 1}, [&](rhi::ShaderCursor c) {
        l.field.bind(c);
        c["outPositions"].setBinding(none.rhi());
        c["outShape"].setBinding(none.rhi());
        c["outNormals"].setBinding(none.rhi());
        c["outAnchors"].setBinding(none.rhi());
        c["probes"].setBinding(probes.rhi());
        c["wrong"].setBinding(wrong.rhi());
        setProbe(c["params"], probe);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 4> counts{};
    REQUIRE(wrong.read(*gpu.device, 0, sizeof(counts), counts.data()));
    return counts;
}

gpu::Buffer pointsNear(test::Gpu& gpu, const Layer& l, uint32_t count) {
    gpu::Buffer probes = deviceWords(*gpu.device, count, 16, "test.ssf.probes");
    Probe p;
    p.count = count;
    p.blobs = l.solid.blobs;
    (void)run(gpu, l, "ssfPointsNear", p, probes);
    return probes;
}

}   // namespace

TEST_CASE("a solid field's gradient is its differences', and its grid finds every blob", "[lod][gpu][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    for (const char* generator : {"ssfSphere", "ssfCube"}) {
        DYNAMIC_SECTION(generator) {
            Probe g;
            const bool cube = std::string(generator) == "ssfCube";
            g.side = 40;
            g.count = cube ? 6 * g.side * g.side : 4000;
            const Layer l = made(*gpu, generator, g);
            constexpr uint32_t kProbes = 1 << 15;
            const gpu::Buffer probes = pointsNear(*gpu, l, kProbes);
            Probe p;
            p.count = kProbes;
            p.blobs = l.solid.blobs;
            const std::array<uint32_t, 4> grad = run(*gpu, l, "ssfGradCheck", p, probes);
            const uint32_t steep = grad[3] & 0xFFFFu, edge = grad[3] >> 16;
            std::printf("  %s, %u blobs: (a) gradient off its differences by more than 1e-3 at %u of %u points "
                        "(%u with |grad F| above the floor; %u apart, their stencil across a support's edge); "
                        "worst %.2e\n",
                        generator, l.solid.blobs, grad[0], grad[1], steep, edge, grad[2] * 1.0e-6);
            CHECK(grad[1] + edge == kProbes);
            // The margin is a bound (|r - 1| over the largest inverse axis, 1/c), and a
            // probe sees some ten blobs: a third of them are set apart, measured.
            CHECK(grad[1] > kProbes / 2);
            CHECK(steep > grad[1] / 2);       // most probes are on the steep part
            CHECK(grad[0] == 0);
            const std::array<uint32_t, 4> grid = run(*gpu, l, "ssfGridCheck", p, probes);
            std::printf("  %s: the grid against every blob: %u of %u points off (%u reached); worst %.2e\n",
                        generator, grid[0], grid[1], grid[3], grid[2] * 1.0e-6);
            CHECK(grid[1] == kProbes);
            CHECK(grid[3] > kProbes / 2);
            CHECK(grid[0] == 0);
            Probe o;
            o.count = 1 << 14;
            o.size = g.size;
            const std::array<uint32_t, 4> outside = run(*gpu, l, "ssfOutsideCheck", o, probes);
            std::printf("  %s: (d) F not -tau exactly, or a gradient, at %u of %u points no blob reaches\n",
                        generator, outside[0], outside[1]);
            CHECK(outside[1] == o.count);
            CHECK(outside[0] == 0);
        }
    }
}

TEST_CASE("one solid blob's zero set is its ellipsoid, scaled by the threshold", "[lod][gpu][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    Probe g;
    g.count = 1;
    g.radius = 0.08F;
    g.thickness = 0.02F;
    g.weight = 1.5F;
    g.centre = {0.3F, -0.2F, 0.7F, 1.1F};   // theta 1.1
    const float len = std::sqrt(0.3F * 0.3F + 0.5F * 0.5F + 0.8F * 0.8F);
    g.normal = {0.3F / len, 0.5F / len, 0.8F / len, 0.03F};   // b = 0.03
    const Layer l = made(*gpu, "ssfOneBlob", g);
    Probe p;
    p.count = 4096;
    const gpu::Buffer none = deviceWords(*gpu->device, 1, 16, "test.ssf.none");
    const std::array<uint32_t, 4> counts = run(*gpu, l, "ssfIsolatedCheck", p, none);
    std::printf("  (b) one blob: %u of %u directions off (bits %u: 1 zero set, 2 inside, 4 outside, 8 past support)\n",
                counts[0], counts[1], counts[3]);
    CHECK(counts[1] == p.count);
    CHECK(counts[0] == 0);
}

TEST_CASE("the smooth minimum is symmetric, the minimum away from k, and C1", "[lod][gpu][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // Any layer binds the field's names; the check reads none of it.
    Probe g;
    g.count = 1;
    const Layer l = made(*gpu, "ssfOneBlob", g);
    const gpu::Buffer none = deviceWords(*gpu->device, 1, 16, "test.ssf.none");
    for (const float k : {0.0F, 0.01F, 0.2F, 1.0F}) {
        Probe p;
        p.count = 1 << 14;
        p.k = k;
        const std::array<uint32_t, 4> counts = run(*gpu, l, "ssfSminCheck", p, none);
        std::printf("  (c) smin k %.2f: %u of %u samples off (bits %u: 1 symmetry, 2 minimum, 4 derivative, "
                    "8 C1, 16 bound)\n",
                    k, counts[0], counts[1], counts[3]);
        CHECK(counts[1] == p.count);
        CHECK(counts[0] == 0);
    }
}
