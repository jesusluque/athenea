// Copyright (c) 2026 jesus luque.
//
// THE SAME FRAME TWICE (docs/decisions.md, 18 Nov, "What a thread's memory
// bears"). A path traced frame's random numbers are keyed by pixel, sample
// and bounce, so two renderers opened on one stage draw one picture, bit for
// bit. On Metal they did not where the stage's materials made the path
// tracer's kernel large -- OpenPBR's eleven lobes beside UsdPreviewSurface,
// with textures, glass and an emitter: runs of pixels the width of a SIMD
// group differed between two runs of one build, a hundred or two of two
// million with no bounce, and a line added to the kernel that changed no
// meaning made it half the frame. What a thread keeps in memory was past what
// the device bears without saying so. Held here on a stage of those
// materials: two renders, compared by a kernel, no code apart.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>

#include "athenea/io/Vdb.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

struct OpenPbr {
    const char* name;
    const char* inputs;
};

/// The materials of scripts/examples/neural-scene-qa's stage, by kind: what
/// OpenPBR pushes most lobes with (a coat, a transmission), an emitter, and
/// UsdPreviewSurface plain, textured and with a normal map.
constexpr OpenPbr kOpenPbr[] = {
    {"Matte", "color3f inputs:base_color = (0.8, 0.8, 0.8)\n float inputs:specular_roughness = 1\n"
              " float inputs:specular_weight = 0\n"},
    {"Gloss", "color3f inputs:base_color = (0.05, 0.15, 0.7)\n float inputs:specular_roughness = 0.08\n"},
    {"Chrome", "color3f inputs:base_color = (0.95, 0.95, 0.95)\n float inputs:base_metalness = 1\n"
               " float inputs:specular_roughness = 0.03\n"},
    {"Glass", "color3f inputs:base_color = (1, 1, 1)\n float inputs:transmission_weight = 1\n"
              " float inputs:specular_roughness = 0\n float inputs:specular_ior = 1.5\n"},
    {"Frosted", "color3f inputs:base_color = (1, 1, 1)\n float inputs:transmission_weight = 1\n"
                " float inputs:specular_roughness = 0.35\n float inputs:specular_ior = 1.5\n"},
    {"Emissive", "color3f inputs:base_color = (0.1, 0.1, 0.1)\n float inputs:emission_luminance = 4\n"
                 " color3f inputs:emission_color = (1, 0.55, 0.2)\n"},
    {"Coated", "color3f inputs:base_color = (0.6, 0.03, 0.03)\n float inputs:specular_roughness = 0.6\n"
               " float inputs:coat_weight = 1\n float inputs:coat_roughness = 0.03\n"},
};

void quad(std::ofstream& out, const char* name, const char* material, const char* points, const char* st) {
    out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
        << "    rel material:binding = </Materials/" << material << ">\n"
        << "    uniform token subdivisionScheme = \"none\"\n"
        << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
        << "    point3f[] points = [" << points << "]\n"
        << "    texCoord2f[] primvars:st = [" << st << "] (\n        interpolation = \"vertex\"\n    )\n}\n";
}

/// A floor, a wall behind it and two cards, of UsdPreviewSurface; before
/// them a row of closed boxes, one of each OpenPBR material; a sky, a sun and
/// a camera above the near edge of the floor. With `medium`, a fog over the
/// boxes (the file at `medium`, a density): the kernel with media is another
/// kernel.
fs::path stage(const fs::path& medium = {}) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "path_repeat";
    fs::create_directories(dir);
    const fs::path path = dir / (medium.empty() ? "materials.usda" : "materials_medium.usda");
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    metersPerUnit = 1\n)\n"
           "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n    color3f inputs:color = (0.62, 0.74, 1)\n}\n"
           "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 5\n    float inputs:angle = 2\n"
           "    float xformOp:rotateX = -50\n    float xformOp:rotateY = 30\n"
           "    uniform token[] xformOpOrder = [\"xformOp:rotateY\", \"xformOp:rotateX\"]\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 24\n"
           "    float horizontalAperture = 36\n    float verticalAperture = 20.25\n"
           "    float2 clippingRange = (0.05, 100)\n"
           "    double3 xformOp:translate = (0, 1.1, 3.2)\n    float xformOp:rotateX = -14\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    out << "def Scope \"Materials\"\n{\n";
    for (const OpenPbr& m : kOpenPbr) {
        out << "    def Material \"" << m.name << "\"\n    {\n"
            << "        token outputs:mtlx:surface.connect = </Materials/" << m.name << "/Surface.outputs:out>\n"
            << "        def Shader \"Surface\"\n        {\n"
            << "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n            " << m.inputs
            << "            token outputs:out\n        }\n    }\n";
    }
    const auto preview = [&](const char* name, const char* inputs, const char* texture) {
        out << "    def Material \"" << name << "\"\n    {\n"
            << "        token outputs:surface.connect = </Materials/" << name << "/Surface.outputs:surface>\n"
            << "        def Shader \"Surface\"\n        {\n"
            << "            uniform token info:id = \"UsdPreviewSurface\"\n            float inputs:roughness = 0.6\n"
            << "            " << inputs << "\n            token outputs:surface\n        }\n";
        if (texture != nullptr) {
            out << "        def Shader \"Coords\"\n        {\n"
                << "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                << "            string inputs:varname = \"st\"\n            float2 outputs:result\n        }\n"
                << "        def Shader \"Image\"\n        {\n"
                << "            uniform token info:id = \"UsdUVTexture\"\n" << texture
                << "            token inputs:wrapS = \"repeat\"\n            token inputs:wrapT = \"repeat\"\n"
                << "            float2 inputs:st.connect = </Materials/" << name << "/Coords.outputs:result>\n"
                << "            float3 outputs:rgb\n        }\n";
        }
        out << "    }\n";
    };
    // The images are not there: a texture that did not load reads its
    // fallback, through the same nodes.
    preview("Plain", "color3f inputs:diffuseColor = (0.1, 0.35, 0.9)", nullptr);
    preview("Textured", "color3f inputs:diffuseColor.connect = </Materials/Textured/Image.outputs:rgb>",
            "            asset inputs:file = @./chequer.png@\n            token inputs:sourceColorSpace = \"sRGB\"\n"
            "            float4 inputs:fallback = (0.6, 0.6, 0.6, 1)\n");
    preview("Bumped",
            "color3f inputs:diffuseColor = (0.5, 0.5, 0.5)\n"
            "            normal3f inputs:normal.connect = </Materials/Bumped/Image.outputs:rgb>",
            "            asset inputs:file = @./bumps.png@\n            token inputs:sourceColorSpace = \"raw\"\n"
            "            float4 inputs:fallback = (0.5, 0.5, 1, 1)\n"
            "            float4 inputs:scale = (2, 2, 2, 1)\n            float4 inputs:bias = (-1, -1, -1, 0)\n");
    out << "}\n";
    quad(out, "Floor", "Textured", "(-4, 0, -2), (-4, 0, 4), (4, 0, 4), (4, 0, -2)", "(0, 0), (0, 6), (8, 6), (8, 0)");
    quad(out, "Wall", "Plain", "(-4, 0, -2), (4, 0, -2), (4, 3, -2), (-4, 3, -2)", "(0, 0), (1, 0), (1, 1), (0, 1)");
    quad(out, "Card", "Bumped", "(-2.6, 0, -1), (-1.2, 0, -1.4), (-1.2, 1.2, -1.4), (-2.6, 1.2, -1)",
         "(0, 0), (2, 0), (2, 2), (0, 2)");
    int index = 0;
    for (const OpenPbr& m : kOpenPbr) {
        // Outward faces, counter-clockwise from outside: near, far, -x, +x,
        // top, bottom.
        const double x0 = -1.9 + 0.55 * index;
        const double x1 = x0 + 0.4;
        out << "def Mesh \"Box" << index << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
            << "    rel material:binding = </Materials/" << m.name << ">\n"
            << "    uniform token subdivisionScheme = \"none\"\n"
            << "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n"
            << "    int[] faceVertexIndices = [4, 5, 6, 7, 1, 0, 3, 2, 0, 4, 7, 3, 5, 1, 2, 6, 3, 7, 6, 2, 0, 1, 5, 4]\n"
            << "    point3f[] points = [(" << x0 << ", 0, -0.2), (" << x1 << ", 0, -0.2), (" << x1 << ", 0.4, -0.2), ("
            << x0 << ", 0.4, -0.2), (" << x0 << ", 0, 0.2), (" << x1 << ", 0, 0.2), (" << x1 << ", 0.4, 0.2), (" << x0
            << ", 0.4, 0.2)]\n}\n";
        ++index;
    }
    if (!medium.empty()) {
        out << "def Volume \"Fog\"\n{\n"
               "    double3 xformOp:translate = (-2, 0, -1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    rel field:density = </Fog/density>\n"
               "    float primvars:athenea:densityScale = 1\n"
               "    color3f primvars:athenea:albedo = (0.8, 0.8, 0.8)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath = @" << medium.string() << "@\n"
               "        token fieldName = \"density\"\n    }\n}\n";
    }
    return path;
}

constexpr uint32_t kWidth = 1920;
constexpr uint32_t kHeight = 1080;

/// 32 paths a pixel in passes of 8, from a renderer of its own: the frame
/// the first of a process draws.
Result<gpu::Buffer> frame(test::Gpu* gpu, const fs::path& path, uint32_t bounces, const char* technique = "rt") {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) return std::move(renderer).error();
    (*renderer)->setPathSamples(8);
    (*renderer)->setPathTotal(32);
    (*renderer)->setPathBounces(bounces);
    auto image = (*renderer)->render("/Camera", 0.0, kWidth, kHeight, technique);
    if (!image) return std::move(image).error();
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    return gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
}

/// Two frames drawn the same way, compared by kernels: a picture, and not a
/// bit apart.
void twice(test::Gpu* gpu, const fs::path& path, uint32_t bounces, const char* technique, const char* what) {
    auto first = frame(gpu, path, bounces, technique);
    if (!first) FAIL(first.error().toString());
    auto second = frame(gpu, path, bounces, technique);
    if (!second) FAIL(second.error().toString());
    auto stats = render::imageStats(*gpu->library, *first, kWidth, kHeight);
    REQUIRE(stats);
    auto difference = render::compareImages(*gpu->library, *first, *second, kWidth, kHeight);
    REQUIRE(difference);
    auto words = render::countDifferent(*gpu->library, *first, *second, kWidth * kHeight * 4);
    REQUIRE(words);
    std::printf("  %s, %u bounces: mean (%.4f, %.4f, %.4f); twice: p99 %u, max %u, %llu of %llu pixels over two codes, "
                "%llu floats differ\n",
                what, bounces, stats->mean[0], stats->mean[1], stats->mean[2], difference->p99, difference->max,
                static_cast<unsigned long long>(difference->over2),
                static_cast<unsigned long long>(difference->pixels), static_cast<unsigned long long>(*words));
    // A picture, not a black frame that would repeat as well.
    CHECK(stats->mean[1] > 0.05);
    CHECK(difference->max == 0);
    CHECK(*words == 0);
}

}   // namespace

TEST_CASE("a path traced frame of many materials is the same frame from two renderers",
          "[usd][gpu][mesh][path][emissive][glass][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = stage();
    for (const uint32_t bounces : {0u, 2u}) {
        twice(gpu, path, bounces, "rt", "paths");
    }
}

// ONE RENDERER, THE VIEW AGAIN: an image is its own (docs/decisions.md, 22
// Nov). A view's first pass took the pixel offset of the view drawn before
// it, and a view asked for twice came back with a pass more than its total.
TEST_CASE("a path traced view is the same view again, at once and after another view",
          "[usd][gpu][mesh][path][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    auto renderer = usd::StageRenderer::open(stage());
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(8);
    (*renderer)->setPathTotal(32);
    (*renderer)->setPathBounces(2);
    const auto view = [&]() {
        auto image = (*renderer)->render("/Camera", 0.0, kWidth, kHeight, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    const gpu::Buffer first = view();
    const gpu::Buffer atOnce = view();
    auto other = (*renderer)->framingCamera(0.0, 50.0, "rt");
    REQUIRE(other);
    REQUIRE((*renderer)->render(*other, 0.0, kWidth, kHeight, "rt"));
    const gpu::Buffer after = view();
    auto again = render::countDifferent(*gpu->library, first, atOnce, kWidth * kHeight * 4);
    REQUIRE(again);
    auto later = render::countDifferent(*gpu->library, first, after, kWidth * kHeight * 4);
    REQUIRE(later);
    std::printf("  one renderer, the view again: %llu floats differ at once, %llu after another view\n",
                static_cast<unsigned long long>(*again), static_cast<unsigned long long>(*later));
    CHECK(*again == 0);
    CHECK(*later == 0);
}

// THE OTHER KERNELS THAT SHADE, each held the same way (docs/decisions.md,
// 18 Nov, "The margin of every kernel"): what a thread's memory bears is a
// kernel's own, and the one with media stood at the edge when the plain one
// no longer did.

TEST_CASE("a path traced frame through a medium is the same frame from two renderers",
          "[usd][gpu][mesh][path][volume][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "path_repeat";
    fs::create_directories(dir);
    const fs::path vdb = dir / "fog.vdb";
    // 4 m by 1.2 m by 2 m of thin fog over the boxes.
    const io::VdbBox box{{0, 0, 0}, {80, 24, 40}, 0.4F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.05, std::span<const io::VdbBox>(&box, 1)));
    twice(gpu, stage(vdb), 2, "rt", "paths through a medium");
}

TEST_CASE("a raster frame of many materials is the same frame from two renderers",
          "[usd][gpu][mesh][raster][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("needs rasterisation");
    }
    twice(gpu, stage(), 0, "raster", "raster");
}

TEST_CASE("the denoiser's guides of many materials are the same from two renderers",
          "[usd][gpu][mesh][path][aov][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = stage();
    // Both on the test's device, so the planes are compared where they are.
    const auto guides = [&](std::unique_ptr<usd::StageRenderer>& renderer) {
        auto opened = usd::StageRenderer::open(path, gpu->device);
        if (!opened) FAIL(opened.error().toString());
        renderer = std::move(*opened);
        renderer->requestOutputs({"albedo", "shadingNormal", "material", "throughPosition"});
        renderer->setPathSamples(8);
        renderer->setPathTotal(8);
        renderer->setPathBounces(0);
        auto image = renderer->render("/Camera", 0.0, kWidth, kHeight, "rt");
        if (!image) FAIL(image.error().toString());
        auto source = renderer->displaySource("albedo");
        if (!source) FAIL(source.error().toString());
        REQUIRE(source->buffer != nullptr);
        return *source;
    };
    std::unique_ptr<usd::StageRenderer> one;
    std::unique_ptr<usd::StageRenderer> two;
    const technique::DisplaySource first = guides(one);
    const technique::DisplaySource second = guides(two);
    REQUIRE(first.stride == second.stride);
    // Every plane the guides' kernel wrote: `stride` float4 a pixel.
    const uint32_t words = kWidth * kHeight * first.stride * 4;
    auto different = render::countDifferent(*gpu->library, *first.buffer, *second.buffer, words);
    REQUIRE(different);
    auto stats = render::imageStats(*gpu->library, *first.buffer, kWidth, kHeight);
    REQUIRE(stats);
    std::printf("  guides: %u planes, albedo's mean (%.4f, %.4f, %.4f); twice: %llu floats differ\n", first.stride,
                stats->mean[0], stats->mean[1], stats->mean[2], static_cast<unsigned long long>(*different));
    CHECK(stats->mean[1] > 0.05);
    CHECK(*different == 0);
}

TEST_CASE("a bake of points among many materials is the same bake from two renderers",
          "[usd][gpu][mesh][path][bake][repeat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = stage();
    // A grid of points a millimetre over the floor, under and around the
    // boxes, facing up: 2^16 of them, as many SIMD groups as a small frame.
    constexpr uint32_t kSide = 256;
    constexpr uint32_t kCount = kSide * kSide;
    constexpr uint32_t kDegree = 3;
    std::vector<float> rays(size_t{kCount} * 8, 0.0F);
    for (uint32_t k = 0; k < kCount; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -3.0F + 6.0F * (static_cast<float>(k % kSide) + 0.5F) / static_cast<float>(kSide);
        ray[1] = 0.0F;
        ray[2] = -1.5F + 4.0F * (static_cast<float>(k / kSide) + 0.5F) / static_cast<float>(kSide);
        ray[3] = 1.0e-3F;   // how far off the surface the rays start
        ray[5] = 1.0F;      // the normal, +y
    }
    const uint32_t planes = (kDegree + 1) * (kDegree + 1);
    const auto bake = [&]() {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto baked = (*renderer)->bakePoints(rays, kCount, 0.0, 32, 2, kDegree);
        if (!baked) FAIL(baked.error().toString());
        REQUIRE(baked->size() == size_t{kCount} * planes * 4);
        gpu::BufferDesc desc;
        desc.bytes = baked->size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, baked->data());
        if (!buffer) FAIL(buffer.error().toString());
        return std::move(*buffer);
    };
    const gpu::Buffer first = bake();
    const gpu::Buffer second = bake();
    // The constant's plane as a picture kSide square, for its mean.
    auto stats = render::imageStats(*gpu->library, first, kSide, kSide);
    REQUIRE(stats);
    auto different = render::countDifferent(*gpu->library, first, second, kCount * planes * 4);
    REQUIRE(different);
    std::printf("  bake: %u points, %u planes, first plane's mean (%.4f, %.4f, %.4f); twice: %llu floats differ\n", kCount,
                planes, stats->mean[0], stats->mean[1], stats->mean[2], static_cast<unsigned long long>(*different));
    CHECK(*different == 0);
}
