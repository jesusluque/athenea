// Copyright (c) 2026 jesus luque.
//
// EMISSION GROUPS AND THEIR LEVELS (stage D): two strips of light in groups
// `a` and `b` (`primvars:athenea:lightGroup`) over a floor. Their groups'
// outputs sum to the beauty, raster and path traced; a level of one half is
// half the light, linearly, seen and cast; a level of nought is no lamp; and
// a level changed refits and uploads the header, rebuilding nothing.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>

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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "mesh_lights_groups";
    fs::create_directories(dir);
    return dir / name;
}

/// The floor and the two strips; `emission` is what the strips' material
/// gives off, `sphere` a small sphere light besides (in no group) so a frame
/// is never lit by the headlight.
fs::path stage(const std::string& name, float emission, bool sphere) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\"\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n";
    const char* names[2] = {"a", "b"};
    for (int k = 0; k < 2; ++k) {
        const float x = k == 0 ? -1.5F : 1.5F;
        char points[256];
        std::snprintf(points, sizeof(points), "[(%g, 1.4, -4), (%g, 1.4, -4), (%g, 1.4, -1.5), (%g, 1.4, -1.5)]",
                      x - 0.1, x + 0.1, x + 0.1, x - 0.1);
        out << "def Mesh \"Strip" << names[k] << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
            << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
            << "    point3f[] points = " << points << "\n"
            << "    uniform token subdivisionScheme = \"none\"\n"
            << "    string primvars:athenea:lightGroup = \"" << names[k] << "\" ( interpolation = \"constant\" )\n"
            << "    rel material:binding = </Materials/Glow" << names[k] << ">\n}\n";
    }
    out << "def Scope \"Materials\"\n{\n";
    for (int k = 0; k < 2; ++k) {
        out << "    def Material \"Glow" << names[k] << "\"\n    {\n"
            << "        token outputs:mtlx:surface.connect = </Materials/Glow" << names[k] << "/Surface.outputs:out>\n"
            << "        def Shader \"Surface\"\n        {\n"
            << "            uniform token info:id = \"ND_surface_unlit\"\n"
            << "            float inputs:emission = " << emission << "\n"
            << "            color3f inputs:emission_color = " << (k == 0 ? "(1, 0.7, 0.4)" : "(0.4, 0.7, 1)") << "\n"
            << "            token outputs:out\n        }\n    }\n";
    }
    out << "}\n";
    if (sphere) {
        out << "def SphereLight \"Fill\"\n{\n    float inputs:intensity = 2\n    float inputs:radius = 0.2\n"
               "    bool inputs:shadow:enable = 0\n"
               "    double3 xformOp:translate = (0, 3, 1)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    out << "def Camera \"Camera\"\n{\n"
           "    float focalLength = 24\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n"
           "    double3 xformOp:translate = (0, 2.2, 2.5)\n"
           "    double xformOp:rotateX = -30\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    return path;
}

gpu::Buffer upload(test::Gpu& gpu, const std::vector<float>& rgba) {
    auto made = gpu::Buffer::fromSpan<float>(*gpu.device, rgba, "frame");
    REQUIRE(made);
    return std::move(*made);
}

}   // namespace

TEST_CASE("an emission group's output holds its emitters' light, and the groups sum to the beauty",
          "[usd][gpu][mesh][meshlights][light-groups]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = stage("groups_sum.usda", 3.0F, false);
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setAntialias(false);
    auto sumKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/light_group_check", "lightGroupSum");
    REQUIRE(sumKernel);
    const uint32_t w = 96, h = 72;
    const size_t pixels = size_t{w} * h;
    for (const char* technique : {"raster", "rt"}) {
        if (std::string(technique) == "rt") {
            (*renderer)->setPathSamples(4);
            (*renderer)->setPathTotal(8);
        }
        (*renderer)->requestOutputs({"lightGroup:a", "lightGroup:b"});
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto beauty = (*renderer)->mappedOutput("color");
        REQUIRE(beauty);
        std::vector<uint8_t> planes;
        for (const char* group : {"lightGroup:a", "lightGroup:b"}) {
            auto plane = (*renderer)->mappedOutput(group);
            REQUIRE(plane);
            planes.insert(planes.end(), plane->begin(), plane->end());
        }
        auto b = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, *beauty, "groups.beauty");
        auto p = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, planes, "groups.planes");
        REQUIRE(b);
        REQUIRE(p);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "groups.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "groups.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            sumKernel->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) {
                c["check"]["pixels"].setData(static_cast<uint32_t>(pixels));
                c["check"]["groups"].setData(uint32_t{2});
                c["check"]["empty"].setData(0xFFFFFFFFu);
                c["check"]["tolerance"].setData(1.0e-3F);
                c["beauty"].setBinding(b->rhi());
                c["planes"].setBinding(p->rhi());
                c["counts"].setBinding(counts.rhi());
                c["worst"].setBinding(worst.rhi());
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t n[4] = {};
        float largest = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, 16, n));
        REQUIRE(worst.read(*gpu->device, 0, 4, &largest));
        std::printf("  %-6s: the two groups summed off the beauty at %u of %u pixels (worst %.2e relative)\n",
                    technique, n[0], n[3], static_cast<double>(largest));
        CHECK(n[0] == 0);
    }
}

TEST_CASE("an emission group's level scales its light, nought is no lamp, and a level is a refit",
          "[usd][gpu][mesh][meshlights][light-groups]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    auto scaled = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh_light_check", "meshScaled");
    REQUIRE(scaled);
    const uint32_t w = 96, h = 72;
    const auto parts = [&](const gpu::Buffer& a, const gpu::Buffer& b, float scale) {
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "scaled.counts");
        gpu::CommandBatch batch(*gpu->device);
        scaled->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor c) {
            c["frameA"].setBinding(a.rhi());
            c["frameB"].setBinding(b.rhi());
            c["scaledCounts"].setBinding(counts.rhi());
            c["scaledPixels"].setData(w * h);
            c["scale"].setData(scale);
            c["scaledTolerance"].setData(1.0e-3F);
        });
        REQUIRE(batch.submit(true));
        uint32_t n[2] = {};
        REQUIRE(counts.read(*gpu->device, 0, 8, n));
        return std::pair{n[0], n[1]};
    };
    // Group b off throughout: what is left is a's, and a's level scales it.
    auto renderer = usd::StageRenderer::open(stage("groups_levels.usda", 3.0F, false));
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setLightLevel("b", 0.0F);
    (*renderer)->setLightLevel("a", 1.0F);
    auto full = (*renderer)->render("/Camera", 0.0, w, h, "raster");
    if (!full) FAIL(full.error().toString());
    const auto before = (*renderer)->meshLightCounts();
    (*renderer)->setLightLevel("a", 0.5F);
    auto half = (*renderer)->render("/Camera", 0.0, w, h, "raster");
    if (!half) FAIL(half.error().toString());
    const auto after = (*renderer)->meshLightCounts();
    const auto [off, lit] = parts(upload(*gpu, full->rgba), upload(*gpu, half->rgba), 0.5F);
    std::printf("  group a at 0.5 against 1: %u of %u lit pixels not half; the level change: %llu rebuilds, %llu "
                "refits, %llu header uploads (%u clusters)\n",
                off, lit, static_cast<unsigned long long>(after.rebuilds - before.rebuilds),
                static_cast<unsigned long long>(after.refits - before.refits),
                static_cast<unsigned long long>(after.headerWrites - before.headerWrites), after.clusters);
    CHECK(lit > 1000);
    CHECK(off == 0);
    CHECK(after.rebuilds == before.rebuilds);
    CHECK(after.headerWrites == before.headerWrites + 1);
    // Both at nought, against the strips with no emission at all, a sphere
    // light beside in both so neither is lit by the headlight.
    auto dimmed = usd::StageRenderer::open(stage("groups_zero.usda", 3.0F, true));
    auto dark = usd::StageRenderer::open(stage("groups_none.usda", 0.0F, true));
    if (!dimmed) FAIL(dimmed.error().toString());
    if (!dark) FAIL(dark.error().toString());
    (*dimmed)->setLightLevel("a", 0.0F);
    (*dimmed)->setLightLevel("b", 0.0F);
    auto zero = (*dimmed)->render("/Camera", 0.0, w, h, "raster");
    auto none = (*dark)->render("/Camera", 0.0, w, h, "raster");
    if (!zero) FAIL(zero.error().toString());
    if (!none) FAIL(none.error().toString());
    auto differ = render::compareHdr(*gpu->library, upload(*gpu, zero->rgba), upload(*gpu, none->rgba), w, h);
    REQUIRE(differ);
    std::printf("  both groups at nought against strips that give off nothing: relMSE %.3e, max relative %.3e\n",
                differ->relMse, differ->maxRelative);
    CHECK(differ->maxRelative < 1.0e-3);
}
