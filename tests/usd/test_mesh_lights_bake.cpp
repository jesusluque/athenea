// Copyright (c) 2026 jesus luque.
//
// THE EMISSION GROUPS' BAKED INDIRECT LIGHT (stage E, `mesh2splat
// --light-groups`): every group's light at a set of points in one bake
// (`StageRenderer::bakeGroupsOnDevice`), and what a frame does with the layer
// a conversion writes of it.
//
// Linear: two groups baked together add up to the same emitters baked as one
// group, and all their light adds up to the whole bake. Then a white ball
// beside a strip on a red floor, converted with a TX transfer and the strip's
// layer, rasterised against the mesh path traced: the red the floor bounces
// onto the ball is there, closer than without the layer, and the strip's
// level scales its direct and its indirect light together.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hio/image.h>

#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "mesh_lights_bake";
    fs::create_directories(dir);
    return dir / name;
}

/// A white floor, a red wall behind and two glowing strips over the floor by
/// the wall, in the groups named (`a` and `b`, or one name for both).
fs::path roomStage(const std::string& name, const char* groupA, const char* groupB) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\"\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
           "def Mesh \"Wall\"\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, -2), (6, 0, -2), (6, 3, -2), (-6, 3, -2)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    color3f[] primvars:displayColor = [(0.8, 0.1, 0.1)] ( interpolation = \"constant\" )\n}\n";
    const char* groups[2] = {groupA, groupB};
    for (int k = 0; k < 2; ++k) {
        const float x = k == 0 ? -1.5F : 1.5F;
        char points[256];
        std::snprintf(points, sizeof(points), "[(%g, 1.4, -1.8), (%g, 1.4, -1.8), (%g, 1.4, -1), (%g, 1.4, -1)]",
                      x - 0.3, x + 0.3, x + 0.3, x - 0.3);
        out << "def Mesh \"Strip" << k << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
            << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
            << "    point3f[] points = " << points << "\n"
            << "    uniform token subdivisionScheme = \"none\"\n"
            << "    string primvars:athenea:lightGroup = \"" << groups[k] << "\" ( interpolation = \"constant\" )\n"
            << "    rel material:binding = </Materials/Glow" << k << ">\n}\n";
    }
    out << "def Scope \"Materials\"\n{\n";
    for (int k = 0; k < 2; ++k) {
        out << "    def Material \"Glow" << k << "\"\n    {\n"
            << "        token outputs:mtlx:surface.connect = </Materials/Glow" << k << "/Surface.outputs:out>\n"
            << "        def Shader \"Surface\"\n        {\n"
            << "            uniform token info:id = \"ND_surface_unlit\"\n"
            << "            float inputs:emission = 4\n"
            << "            color3f inputs:emission_color = " << (k == 0 ? "(1, 0.8, 0.6)" : "(0.7, 0.8, 1)") << "\n"
            << "            token outputs:out\n        }\n    }\n";
    }
    out << "}\n";
    return path;
}

struct LayerSum {
    uint32_t apart = 0;
    uint32_t lit = 0;
    float    worst = 0.0F;
    uint32_t red = 0;
};

/// meshLayerSum: `a`'s groups added against `b`'s (or a bake's colour where
/// `groupsB` is 0), point by point.
LayerSum layerSum(test::Gpu& gpu, const gpu::Buffer& a, uint32_t groupsA, const gpu::Buffer& b, uint32_t groupsB,
                  uint32_t count, float tolerance, float floor) {
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/mesh_light_check", "meshLayerSum");
    REQUIRE(kernel);
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 4, "layerCounts");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["layersA"].setBinding(a.rhi());
            cursor["layersB"].setBinding(b.rhi());
            cursor["layerCounts"].setBinding(counts.rhi());
            cursor["layerSum"]["count"].setData(count);
            cursor["layerSum"]["groupsA"].setData(groupsA);
            cursor["layerSum"]["groupsB"].setData(groupsB);
            cursor["layerSum"]["tolerance"].setData(tolerance);
            cursor["layerSum"]["floor"].setData(floor);
            cursor["layerSum"]["lit"].setData(1.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t got[4] = {0, 0, 0, 0};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(got), got));
    LayerSum s;
    s.apart = got[0];
    s.lit = got[1];
    std::memcpy(&s.worst, &got[2], sizeof(float));
    s.red = got[3];
    return s;
}

}   // namespace

TEST_CASE("emission groups baked in one pass add up as their light does",
          "[usd][gpu][mesh][bake][meshlights][light-groups]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    // Points on the floor along the wall, under and between the strips: the
    // wall is lit by them and bounces red onto the floor.
    constexpr uint32_t kCount = 40;
    std::vector<float> rays(size_t{kCount} * 12, 0.0F);
    for (uint32_t k = 0; k < kCount; ++k) {
        float* ray = rays.data() + size_t{k} * 12;
        ray[0] = -2.5F + 5.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(kCount);
        ray[1] = 0.0F;
        ray[2] = -1.4F;
        ray[3] = 1.0e-3F;
        ray[5] = 1.0F;
    }
    gpu::Device& device = *gpu->device;
    gpu::BufferDesc desc;
    desc.bytes = rays.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "test.rays";
    auto rayBuffer = gpu::Buffer::create(device, desc, rays.data());
    REQUIRE(rayBuffer);
    constexpr uint32_t kSamples = 4096;
    constexpr uint32_t kBounces = 3;

    auto split = usd::StageRenderer::open(roomStage("groups_ab.usda", "a", "b"), gpu->device);
    if (!split) FAIL(split.error().toString());
    auto indirect = (*split)->bakeGroupsOnDevice(*rayBuffer, kCount, 0.0, {"a", "b"}, kSamples, kBounces, true);
    if (!indirect) FAIL(indirect.error().toString());
    auto whole = (*split)->bakeGroupsOnDevice(*rayBuffer, kCount, 0.0, {"a", "b"}, kSamples, kBounces, false);
    if (!whole) FAIL(whole.error().toString());
    auto plain = (*split)->bakePointsOnDevice(*rayBuffer, kCount, 0.0, kSamples, kBounces, 0);
    if (!plain) FAIL(plain.error().toString());

    auto joined = usd::StageRenderer::open(roomStage("groups_one.usda", "ab", "ab"), gpu->device);
    if (!joined) FAIL(joined.error().toString());
    auto union_ = (*joined)->bakeGroupsOnDevice(*rayBuffer, kCount, 0.0, {"ab"}, kSamples, kBounces, true);
    if (!union_) FAIL(union_.error().toString());

    // a + b, indirect, against a u b: the same light, other paths.
    const LayerSum added = layerSum(*gpu, *indirect, 2, *union_, 1, kCount, 0.1F, 0.02F);
    std::printf("  indirect, a + b against a u b: %u of %u points apart, worst %.3f; %u lit, %u red\n", added.apart,
                kCount, static_cast<double>(added.worst), added.lit, added.red);
    CHECK(added.apart == 0);
    CHECK(added.lit == kCount);
    // The light the red wall bounces is red.
    CHECK(added.red == kCount);
    // Every group's light, direct and indirect, against the bake whole: the
    // strips are the stage's only light, so their groups are all of it.
    const LayerSum all = layerSum(*gpu, *whole, 2, *plain, 0, kCount, 0.1F, 0.05F);
    std::printf("  all light, a + b against the whole bake: %u of %u points apart, worst %.3f\n", all.apart, kCount,
                static_cast<double>(all.worst));
    CHECK(all.apart == 0);
    CHECK(all.lit == kCount);
}

namespace {

gpu::Buffer drawFrame(test::Gpu& gpu, const fs::path& path, const char* technique, uint32_t w, uint32_t h,
                      float level = 1.0F, uint32_t bounces = 6) {
    // (The ball is convex: what reaches it after a bounce came off the floor.)
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    if (std::string(technique) == "rt") {
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(1024);
        (*renderer)->setPathBounces(bounces);
    } else {
        (*renderer)->setLightSamples(64);
        (*renderer)->setCloudShadows(false);
    }
    (*renderer)->setLightLevel("strip", level);
    auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

/// The bleed stage with the converted cloud in place of the ball.
fs::path withCloud(const std::string& name, const fs::path& cloud) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    subLayers = [@" << cloud.string() << "@, @"
        << (fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_bleed.usda").string() << "@]\n    upAxis = \"Y\"\n)\n"
        << "over \"World\"\n{\n    over \"Ball\" (\n        active = false\n    )\n    {\n    }\n}\n";
    return path;
}

/// a | b | |a - b| x 4, written as a PNG where ATHENEA_MESH_LIGHT_IMAGES asks.
void picture(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b, uint32_t w, uint32_t h,
             const std::string& name) {
    const char* folder = std::getenv("ATHENEA_MESH_LIGHT_IMAGES");
    if (folder == nullptr || *folder == '\0') {
        return;
    }
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/triptych", "triptych");
    REQUIRE(kernel);
    gpu::Buffer out = test::uintBuffer(*gpu.device, uint64_t{w} * 3 * h, "triptych");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(a.rhi());
            cursor["b"].setBinding(b.rhi());
            cursor["out"].setBinding(out.rhi());
            cursor["width"].setData(w);
            cursor["height"].setData(h);
            cursor["exposure"].setData(1.0F);
            cursor["gain"].setData(4.0F);
        });
        REQUIRE(batch.submit(true));
    }
    auto pixels = out.readAll<uint32_t>(*gpu.device);
    REQUIRE(pixels);
    fs::create_directories(folder);
    pxr::HioImageSharedPtr image = pxr::HioImage::OpenForWriting((fs::path(folder) / (name + ".png")).string());
    REQUIRE(image);
    pxr::HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w * 3);
    spec.height = static_cast<int>(h);
    spec.format = pxr::HioFormatUNorm8Vec4srgb;
    spec.data = pixels->data();
    CHECK(image->Write(spec));
}

/// Where the ball is: the mesh stage with the ball an unlit white of
/// radiance one and nothing else in it.
fs::path maskStage() {
    const fs::path path = scratch("bleed_mask.usda");
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    subLayers = [@" << (fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_bleed.usda").string()
        << "@]\n    upAxis = \"Y\"\n)\n"
        << "over \"World\"\n{\n"
           "    over \"Ground\" (\n        active = false\n    )\n    {\n    }\n"
           "    over \"Strip\" (\n        active = false\n    )\n    {\n    }\n"
           "    over \"Ball\"\n    {\n        rel material:binding = </World/Mask>\n    }\n"
           "    def Material \"Mask\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </World/Mask/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_surface_unlit\"\n"
           "            float inputs:emission = 1\n"
           "            color3f inputs:emission_color = (1, 1, 1)\n"
           "            token outputs:out\n        }\n    }\n}\n";
    return path;
}

struct Masked {
    double relMse = 0.0;
    double pixels = 0.0;
    float  a[3] = {0, 0, 0};
    float  b[3] = {0, 0, 0};
};

Masked masked(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b, const gpu::Buffer& mask, uint32_t pixels) {
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/mesh_light_check", "meshMaskedError");
    REQUIRE(kernel);
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = 4;
    auto out = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(out);
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["maskedA"].setBinding(a.rhi());
            cursor["maskedB"].setBinding(b.rhi());
            cursor["maskFrame"].setBinding(mask.rhi());
            cursor["maskedOut"].setBinding(out->rhi());
            cursor["maskedPixels"].setData(pixels);
        });
        REQUIRE(batch.submit(true));
    }
    float got[8] = {};
    REQUIRE(out->read(*gpu.device, 0, sizeof(got), got));
    Masked m;
    m.pixels = got[1];
    m.relMse = got[1] > 0.0F ? got[0] / got[1] : 0.0;
    for (int c = 0; c < 3; ++c) {
        m.a[c] = got[1] > 0.0F ? got[2 + c] / got[1] : 0.0F;
        m.b[c] = got[1] > 0.0F ? got[5 + c] / got[1] : 0.0F;
    }
    return m;
}

struct Scaled {
    uint32_t apart = 0;
    uint32_t lit = 0;
};

Scaled scaled(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b, uint32_t pixels, float scale) {
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/mesh_light_check", "meshScaled");
    REQUIRE(kernel);
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "scaledCounts");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["frameA"].setBinding(a.rhi());
            cursor["frameB"].setBinding(b.rhi());
            cursor["scaledCounts"].setBinding(counts.rhi());
            cursor["scaledPixels"].setData(pixels);
            cursor["scale"].setData(scale);
            cursor["scaledTolerance"].setData(2.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t got[2] = {0, 0};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(got), got));
    return {got[0], got[1]};
}

}   // namespace

TEST_CASE("a TX cloud with its strip's baked bounce rasterises the red the floor sends up",
          "[.meshlights-bleed][usd][gpu][splat][transfer][meshlights][light-groups]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization || !gpu->device->caps().rayQuery) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path converted(ATHENEA_TX_DIR);
    const fs::path withLayer = converted / "bleed_groups.usdc";
    const fs::path without = converted / "bleed_plain.usdc";
    if (!fs::exists(withLayer) || !fs::exists(without)) {
        SKIP("the bleed conversions are not there: ctest's mesh2splat_light_groups_* convert them");
    }
    const uint32_t w = 192, h = 192;
    const gpu::Buffer traced =
        drawFrame(*gpu, fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_bleed.usda", "rt", w, h);
    const fs::path stageWith = withCloud("bleed_with.usda", withLayer);
    const fs::path stageWithout = withCloud("bleed_without.usda", without);
    const gpu::Buffer layered = drawFrame(*gpu, stageWith, "raster", w, h);
    const gpu::Buffer direct = drawFrame(*gpu, stageWithout, "raster", w, h);
    auto withDiff = render::compareHdr(*gpu->library, layered, traced, w, h);
    REQUIRE(withDiff);
    auto withoutDiff = render::compareHdr(*gpu->library, direct, traced, w, h);
    REQUIRE(withoutDiff);
    picture(*gpu, layered, traced, w, h, "e_bleed_direct_and_layer_raster_pt");
    picture(*gpu, direct, traced, w, h, "e_bleed_direct_only_raster_pt");
    std::printf("  the whole frame against the mesh path traced: direct and the strip's layer relMSE %.4f; "
                "direct alone %.4f\n",
                withDiff->relMse, withoutDiff->relMse);
    // Most of a whole frame's error is the ground's: the cloud casts no
    // shadow of a mesh light onto it (stage C). On the ball alone:
    const gpu::Buffer mask = drawFrame(*gpu, maskStage(), "raster", w, h);
    const Masked ballWith = masked(*gpu, layered, traced, mask, w * h);
    const Masked ballWithout = masked(*gpu, direct, traced, mask, w * h);
    std::printf("  on the ball (%.0f pixels): relMSE %.4f with the layer, %.4f without; mean colour "
                "(%.4f %.4f %.4f) with, (%.4f %.4f %.4f) without, (%.4f %.4f %.4f) path traced\n",
                ballWith.pixels, ballWith.relMse, ballWithout.relMse, ballWith.a[0], ballWith.a[1], ballWith.a[2],
                ballWithout.a[0], ballWithout.a[1], ballWithout.a[2], ballWith.b[0], ballWith.b[1], ballWith.b[2]);
    CHECK(ballWith.pixels > 1000.0);
    // WHAT THE LAYER ADDS, against what the path tracer finds past the first
    // bounce: the ball path traced on its red floor less the ball path traced
    // on a black one -- the ball is convex, so what reaches it after a bounce
    // came off the floor -- on the ball, channel by channel. Measured apart
    // from the direct light, which is the frame's (stage C), not the layer's.
    const fs::path black = scratch("bleed_black_floor.usda");
    {
        std::ofstream out(black);
        out << "#usda 1.0\n(\n    subLayers = [@"
            << (fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_bleed.usda").string()
            << "@]\n    upAxis = \"Y\"\n)\nover \"World\"\n{\n    over \"Ground\"\n    {\n"
               "        color3f[] primvars:displayColor = [(0, 0, 0)] ( interpolation = \"constant\" )\n    }\n}\n";
    }
    const gpu::Buffer tracedDirect = drawFrame(*gpu, black, "rt", w, h);
    const Masked directTraced = masked(*gpu, tracedDirect, traced, mask, w * h);
    float layerAdds[3];
    float tracedIndirect[3];
    for (int c = 0; c < 3; ++c) {
        layerAdds[c] = ballWith.a[c] - ballWithout.a[c];
        tracedIndirect[c] = directTraced.b[c] - directTraced.a[c];
    }
    std::printf("  on the ball, past the first bounce: the layer adds (%.4f %.4f %.4f), the path tracer finds "
                "(%.4f %.4f %.4f)\n",
                layerAdds[0], layerAdds[1], layerAdds[2], tracedIndirect[0], tracedIndirect[1], tracedIndirect[2]);
    picture(*gpu, layered, direct, w, h, "e_bleed_layer_added");
    // The bounce is red, and the layer's is the path tracer's to a quarter.
    CHECK(tracedIndirect[0] > 2.0F * tracedIndirect[1]);
    CHECK(layerAdds[0] > 2.0F * layerAdds[1]);
    for (int c = 0; c < 3; ++c) {
        INFO("channel " << c);
        CHECK(std::abs(layerAdds[c] - tracedIndirect[c]) < 0.25F * tracedIndirect[0]);
    }

    // The strip's level: half is half of everything it gives, direct and
    // baked; nought is none of it, and nothing else lights the stage.
    const gpu::Buffer half = drawFrame(*gpu, stageWith, "raster", w, h, 0.5F);
    const Scaled halved = scaled(*gpu, layered, half, w * h, 0.5F);
    std::printf("  level 0.5: %u of %u lit pixels not half of level 1\n", halved.apart, halved.lit);
    CHECK(halved.lit > w * h / 4);
    CHECK(halved.apart <= halved.lit / 1000);
    const gpu::Buffer off = drawFrame(*gpu, stageWith, "raster", w, h, 0.0F);
    const Scaled dark = scaled(*gpu, off, off, w * h, 1.0F);
    std::printf("  level 0: %u lit pixels\n", dark.lit);
    CHECK(dark.lit == 0);
}
