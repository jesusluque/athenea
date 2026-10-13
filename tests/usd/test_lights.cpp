// Copyright (c) 2026 jesus luque.
//
// LIGHT GROUPS (docs/decisions.md; usd/LightGroups.h): a cloud's lamps,
// merged from one cloud a group by `athenea lights merge` and switched at run
// time. Light is linear, so a frame with two groups lit is the sum of the
// frames with each, less the one with neither, and a group at level two is
// twice its share -- checked on frames drawn through Hydra, combined and
// compared on the device. A layer cloud that is not the base's gaussians is
// refused. A light catcher adds its lamps' light over what is behind it and
// never covers it.
#include "../gpu/GpuTest.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>

#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/LightGroups.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "lights";
    fs::create_directories(dir);
    return dir / name;
}

/// A cloud of degree 0 in records: `count` gaussians in a slab `depth` deep
/// at z = `z`, their constant terms drawn from [dcLow, dcHigh) -- for a
/// layer, the light it sends out (0.5 + SH0 dc). The same `seed` gives the
/// same gaussians; the constant terms come from their own sequence, so a
/// layer is the base's gaussians with another light on them.
io::RawSplats slab(uint32_t count, uint64_t seed, float z, float depth, float logitOpacity, float dcLow,
                   float dcHigh, uint64_t colourSeed) {
    io::RawSplats raw;
    raw.source = "lights synthetic";
    raw.linear = true;
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 0;
    uint64_t state = seed;
    uint64_t colours = colourSeed;
    const auto next = [](uint64_t& s) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((s >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t i = 0; i < count; ++i) {
        raw.records.resize(raw.records.size() + 14, 0.0F);
        float* r = raw.records.data() + raw.records.size() - 14;
        r[0] = next(state) * 3.0F - 1.5F;
        r[1] = next(state) * 2.2F - 1.1F;
        r[2] = z + (next(state) - 0.5F) * depth;
        r[3] = logitOpacity;
        r[4] = -2.6F + next(state);
        r[5] = -2.6F + next(state);
        r[6] = -3.5F;
        r[7] = 1.0F;
        r[8] = next(state) * 0.2F;
        r[9] = next(state) * 0.2F;
        r[10] = next(state) - 0.5F;
        for (int k = 11; k < 14; ++k) {
            r[k] = dcLow + (dcHigh - dcLow) * next(colours);
        }
        raw.count += 1;
    }
    return raw;
}

void write(gpu::ShaderLibrary& library, const io::RawSplats& raw, const fs::path& path, bool lightCatcher = false) {
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.linear = true;
    options.upAxis = 'y';
    options.lightCatcher = lightCatcher;
    REQUIRE(usd::writeParticleFieldStage(library, raw, path, options));
}

/// The sidecar's Scopes as a `.lights.usda` holds them.
void writeSidecar(const fs::path& path) {
    std::ofstream out(path);
    out << "#usda 1.0\n"
           "over \"Car\"\n{\n"
           "    def Scope \"LightGroups\"\n    {\n"
           "        def Scope \"a\"\n        {\n"
           "            float athenea:lightGroup:emissionLuminance = 2\n"
           "            color3f athenea:lightGroup:emissionColor = (1, 0.5, 0.25)\n"
           "            bool athenea:lightGroup:initialOn = false\n        }\n"
           "        def Scope \"b\"\n        {\n"
           "            float athenea:lightGroup:emissionLuminance = 1\n"
           "            color3f athenea:lightGroup:emissionColor = (0.25, 0.5, 1)\n"
           "            bool athenea:lightGroup:initialOn = false\n        }\n    }\n"
           "    def Scope \"LightStates\"\n    {\n"
           "        def Scope \"dark\"\n        {\n        }\n"
           "        def Scope \"both\"\n        {\n"
           "            double athenea:lightState:target:a = 1\n"
           "            double athenea:lightState:target:b = 1\n        }\n    }\n}\n";
}

/// A shot of the clouds at `references` (each a file whose cloud is at
/// /World/Splats), from a camera of its own.
void writeShot(const fs::path& path, const std::vector<std::string>& references) {
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
           "def Xform \"World\"\n{\n";
    for (size_t k = 0; k < references.size(); ++k) {
        out << "    def \"Cloud" << k << "\" ( references = @" << references[k] << "@</World/Splats> )\n    {\n    }\n";
    }
    out << "    def Camera \"Shot\"\n    {\n"
           "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
           "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
           "        double3 xformOp:translate = (0, 0, 4)\n"
           "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
}

constexpr uint32_t kWidth = 160;
constexpr uint32_t kHeight = 120;

gpu::Buffer frame(test::Gpu& gpu, usd::StageRenderer& renderer) {
    auto image = renderer.render("/World/Shot", 0.0, kWidth, kHeight);
    if (!image) {
        FAIL(image.error().toString());
    }
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "lights.frame";
    auto made = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(made);
    return *made;
}

/// ka a + kb b + kc c, on the device (light_layers_check).
gpu::Buffer combine(test::Gpu& gpu, const gpu::Buffer& a, float ka, const gpu::Buffer& b, float kb,
                    const gpu::Buffer& c, float kc) {
    static gpu::ComputeKernel kCombine = test::kernel(gpu, "athenea/test/light_layers_check");
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{kWidth} * kHeight * 16;
    desc.elementBytes = 16;
    desc.label = "lights.combined";
    auto out = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(out);
    gpu::CommandBatch batch(*gpu.device);
    kCombine.dispatch(batch, {kWidth * kHeight, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["a"].setBinding(a.rhi());
        cursor["b"].setBinding(b.rhi());
        cursor["c"].setBinding(c.rhi());
        cursor["combined"].setBinding(out->rhi());
        cursor["params"]["pixels"].setData(kWidth * kHeight);
        cursor["params"]["ka"].setData(ka);
        cursor["params"]["kb"].setData(kb);
        cursor["params"]["kc"].setData(kc);
    });
    REQUIRE(batch.submit(true));
    return *out;
}

render::ImageDifference differ(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b) {
    auto diff = render::compareImages(*gpu.library, a, b, kWidth, kHeight);
    REQUIRE(diff);
    return *diff;
}

}   // namespace

TEST_CASE("lamp groups add linearly: two lit are the sum of each, a level of two twice its share",
          "[usd][gpu][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path base = scratch("base.usdc");
    const fs::path layerA = scratch("layer_a.usdc");
    const fs::path layerB = scratch("layer_b.usdc");
    const fs::path sidecar = scratch("car.lights.usda");
    const fs::path merged = scratch("merged.usdc");
    // One slab, three lights on it: the base's, and each group's per unit of
    // its radiance (0.02 to 0.22).
    write(*gpu->library, slab(3000, 7, 0.0F, 0.6F, 0.5F, -1.0F, 1.0F, 101), base);
    write(*gpu->library, slab(3000, 7, 0.0F, 0.6F, 0.5F, -1.7F, -1.0F, 202), layerA);
    write(*gpu->library, slab(3000, 7, 0.0F, 0.6F, 0.5F, -1.7F, -1.0F, 303), layerB);
    writeSidecar(sidecar);
    usd::LightMergeOptions options;
    options.base = base;
    options.layers = {{"a", layerA}, {"b", layerB}};
    options.lights = sidecar;
    options.output = merged;
    fs::remove(merged);
    auto report = usd::mergeLightLayers(*gpu->library, options);
    if (!report) {
        FAIL(report.error().toString());
    }
    REQUIRE(report->layers.size() == 2);
    CHECK(report->layers[0].lit == 3000);
    CHECK(report->copied.size() == 2);

    const fs::path shot = scratch("shot.usda");
    writeShot(shot, {merged.string()});
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) {
        FAIL(renderer.error().toString());
    }
    usd::StageRenderer& r = **renderer;
    REQUIRE(r.lightRig().groups.size() == 2);
    REQUIRE(r.lightRig().states.size() == 2);

    r.setLightState("dark");
    const gpu::Buffer off = frame(*gpu, r);
    const uint64_t uploads = r.cloudUploads();
    r.setLightLevel("a", 1.0F);
    const gpu::Buffer a = frame(*gpu, r);
    r.setLightLevel("a", 2.0F);
    const gpu::Buffer a2 = frame(*gpu, r);
    r.setLightLevel("a", 0.0F);
    r.setLightLevel("b", 1.0F);
    const gpu::Buffer b = frame(*gpu, r);
    r.setLightLevel("a", 1.0F);
    const gpu::Buffer ab = frame(*gpu, r);
    r.setLightLevel("a", std::nullopt);
    r.setLightLevel("b", std::nullopt);
    r.setLightState("both");
    const gpu::Buffer both = frame(*gpu, r);
    // A lamp switched is a weight: the cloud was not uploaded again.
    CHECK(r.cloudUploads() == uploads);

    // The layers are drawn at all.
    const render::ImageDifference shows = differ(*gpu, a, off);
    INFO("group a against none: max " << shows.max);
    CHECK(shows.max >= 10);
    CHECK(differ(*gpu, b, off).max >= 10);
    // image(A) + image(B) - image(off) is image(A and B).
    const render::ImageDifference sum = differ(*gpu, combine(*gpu, a, 1.0F, b, 1.0F, off, -1.0F), ab);
    INFO("A + B - off against AB: p99 " << sum.p99 << " max " << sum.max);
    CHECK(sum.p99 <= 1);
    CHECK(sum.max <= 2);
    // 2 image(A) - image(off) is image(A at level 2).
    const render::ImageDifference twice = differ(*gpu, combine(*gpu, a, 2.0F, off, -1.0F, off, 0.0F), a2);
    INFO("2A - off against A at 2: p99 " << twice.p99 << " max " << twice.max);
    CHECK(twice.p99 <= 1);
    CHECK(twice.max <= 2);
    // A state is its levels.
    const render::ImageDifference state = differ(*gpu, both, ab);
    CHECK(state.max <= 1);
}

TEST_CASE("lights merge refuses a layer whose gaussians are not the base's", "[usd][gpu][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path base = scratch("refuse_base.usdc");
    const fs::path fewer = scratch("refuse_fewer.usdc");
    const fs::path moved = scratch("refuse_moved.usdc");
    const fs::path same = scratch("refuse_same.usdc");
    const fs::path output = scratch("refuse_out.usdc");
    write(*gpu->library, slab(800, 11, 0.0F, 0.4F, 0.5F, -1.0F, 1.0F, 1), base);
    write(*gpu->library, slab(799, 11, 0.0F, 0.4F, 0.5F, -1.7F, -1.0F, 2), fewer);
    // The same count, other places: another draw of the slab.
    write(*gpu->library, slab(800, 12, 0.0F, 0.4F, 0.5F, -1.7F, -1.0F, 2), moved);
    write(*gpu->library, slab(800, 11, 0.0F, 0.4F, 0.5F, -1.7F, -1.0F, 2), same);

    usd::LightMergeOptions options;
    options.base = base;
    options.output = output;
    for (const fs::path& wrong : {fewer, moved}) {
        fs::remove(output);
        options.layers = {{"a", same}, {"b", wrong}};
        auto refused = usd::mergeLightLayers(*gpu->library, options);
        REQUIRE_FALSE(refused);
        INFO(refused.error().toString());
        CHECK(refused.error().message().find(wrong.string()) != std::string::npos);
        // Nothing is written for a merge that was refused.
        CHECK_FALSE(fs::exists(output));
    }
    // And the same gaussians are taken.
    options.layers = {{"a", same}};
    fs::remove(output);
    auto taken = usd::mergeLightLayers(*gpu->library, options);
    if (!taken) {
        FAIL(taken.error().toString());
    }
    CHECK(fs::exists(output));
}

TEST_CASE("a light catcher adds its lamp's light over what is behind it, linearly, and never covers it",
          "[usd][gpu][lights][catcher]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // An opaque wall behind, and in front of it a sheet of gaussians lying
    // across the view: the catcher, with one group's light on it.
    const fs::path wall = scratch("catcher_wall.usdc");
    const fs::path base = scratch("catcher_base.usdc");
    const fs::path layer = scratch("catcher_layer.usdc");
    const fs::path sidecar = scratch("catcher.lights.usda");
    const fs::path merged = scratch("catcher_merged.usdc");
    write(*gpu->library, slab(3000, 21, -1.0F, 0.3F, 4.0F, -1.0F, 1.0F, 5), wall);
    write(*gpu->library, slab(2000, 22, 0.5F, 0.05F, 1.0F, -1.0F, 1.0F, 6), base, true);
    write(*gpu->library, slab(2000, 22, 0.5F, 0.05F, 1.0F, -1.7F, -1.0F, 7), layer, true);
    writeSidecar(sidecar);
    usd::LightMergeOptions options;
    options.base = base;
    options.layers = {{"a", layer}};
    options.lights = sidecar;
    options.output = merged;
    fs::remove(merged);
    REQUIRE(usd::mergeLightLayers(*gpu->library, options));

    const fs::path alone = scratch("catcher_alone.usda");
    const fs::path with = scratch("catcher_with.usda");
    writeShot(alone, {wall.string()});
    writeShot(with, {wall.string(), merged.string()});
    auto bare = usd::StageRenderer::open(alone);
    auto lit = usd::StageRenderer::open(with);
    if (!bare || !lit) {
        FAIL("cannot open the catcher's shots");
    }
    const gpu::Buffer behind = frame(*gpu, **bare);
    (*lit)->setLightLevel("a", 0.0F);
    const gpu::Buffer dark = frame(*gpu, **lit);
    (*lit)->setLightLevel("a", 1.0F);
    const gpu::Buffer one = frame(*gpu, **lit);
    (*lit)->setLightLevel("a", 2.0F);
    const gpu::Buffer two = frame(*gpu, **lit);

    // Off, it is not there: its own colours are not drawn, and it takes
    // nothing from the wall behind it.
    const render::ImageDifference covers = differ(*gpu, dark, behind);
    INFO("catcher off against the wall alone: max " << covers.max);
    CHECK(covers.max <= 1);
    // On, it adds.
    CHECK(differ(*gpu, one, behind).max >= 10);
    // And linearly in its weight: 2 image(1) - image(off) is image(2).
    const render::ImageDifference twice = differ(*gpu, combine(*gpu, one, 2.0F, behind, -1.0F, behind, 0.0F), two);
    INFO("2 one - off against two: p99 " << twice.p99 << " max " << twice.max);
    CHECK(twice.p99 <= 1);
    CHECK(twice.max <= 2);
}
