// Copyright (c) 2026 jesus luque.
//
// GLASS SHADOWS (docs/decisions.md, "A lamp behind its lens"): a lamp is an
// emitting mesh behind a lens, and next event estimation's shadow ray from
// what it lights used to stop at the lens, so the floor in front of a car's
// headlamp was lit only by the paths that refracted through the glass by
// chance -- black but for fireflies. The shadow ray now goes on through glass
// whose transmission is a delta, tinted by what each face lets through, and
// the material's rays that reach the lamp through the same glass are weighed
// against it. Checked through Hydra on a small emitting quad inside a closed
// box of thin glass over a diffuse floor: the floor is lit by the shadow
// rays (far less noise than the material's rays alone at the same paths),
// the two strategies together are the same light as the material's rays
// alone, a tinted glass tints it, and for a thin wall the estimator is
// unbiased -- the floor reads what the same scene without the glass reads
// with the emitter scaled by the glass's transmittance.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "glass_shadows";
    fs::create_directories(dir);
    return dir / name;
}

/// What the stage holds: a floor; an emitting 0.5 x 0.5 quad facing down at
/// y 1, `emission` times `colour`; and, with `glass`, a closed unit box of
/// thin glass around it (OpenPBR, smooth, index `ior`, transmission colour
/// `tint`). The camera looks down at the floor in front of the box, so the
/// lower third of the frame is floor the lamp lights through the box's near
/// face, and nothing else.
struct GlassScene {
    bool        glass = true;
    float       ior = 1.5F;
    const char* tint = "1, 1, 1";
    float       emission = 200.0F;
    const char* colour = "1, 1, 1";
};

fs::path stage(const std::string& name, const GlassScene& scene) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\"\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
           "def Mesh \"Lamp\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-0.25, 1, -1.25), (0.25, 1, -1.25), (0.25, 1, -0.75), (-0.25, 1, -0.75)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    uniform bool doubleSided = 1\n"
           "    rel material:binding = </Materials/Glow>\n}\n";
    if (scene.glass) {
        // Outward faces, counter-clockwise from outside: back, front, -x,
        // +x, bottom, top.
        out << "def Mesh \"Box\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n"
               "    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2]\n"
               "    point3f[] points = [(-0.5, 0.3, -1.5), (0.5, 0.3, -1.5), (0.5, 1.3, -1.5), (-0.5, 1.3, -1.5),\n"
               "                        (-0.5, 0.3, -0.5), (0.5, 0.3, -0.5), (0.5, 1.3, -0.5), (-0.5, 1.3, -0.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform bool doubleSided = 1\n"
               "    rel material:binding = </Materials/Glass>\n}\n";
    }
    out << "def Scope \"Materials\"\n{\n"
           "    def Material \"Glow\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Glow/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_surface_unlit\"\n"
           "            float inputs:emission = "
        << scene.emission
        << "\n"
           "            color3f inputs:emission_color = ("
        << scene.colour
        << ")\n"
           "            token outputs:out\n        }\n    }\n"
           "    def Material \"Glass\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Glass/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
           "            color3f inputs:base_color = (1, 1, 1)\n"
           "            float inputs:transmission_weight = 1\n"
           "            color3f inputs:transmission_color = ("
        << scene.tint
        << ")\n"
           "            float inputs:specular_roughness = 0\n"
           "            float inputs:specular_ior = "
        << scene.ior
        << "\n"
           "            bool inputs:geometry_thin_walled = 1\n"
           "            token outputs:out\n        }\n    }\n}\n"
           "def Camera \"Camera\"\n{\n"
           "    float focalLength = 24\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n"
           "    double3 xformOp:translate = (0, 1.2, 2)\n"
           "    double xformOp:rotateX = -30\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    return path;
}

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;

/// A path traced frame of `path`, gathered to `total` paths a pixel, on the device.
gpu::Buffer frame(test::Gpu* gpu, const fs::path& path, bool glassShadows, uint32_t total) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setGlassShadows(glassShadows);
    (*renderer)->setPathSamples(std::min<uint32_t>(total, 256));
    (*renderer)->setPathTotal(total);
    (*renderer)->setPathBounces(1);
    auto image = (*renderer)->render("/Camera", 0.0, kWidth, kHeight, "rt");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

/// The floor in front of the box: the frame's lower third (rows bottom first).
render::ImageStats floorOf(test::Gpu* gpu, const gpu::Buffer& image) {
    auto stats = render::imageStats(*gpu->library, image, kWidth, kHeight, 0, 0, kWidth, kHeight / 3);
    REQUIRE(stats);
    return *stats;
}

bool tracesRays(test::Gpu* gpu) {
    const gpu::Caps& caps = gpu->device->caps();
    return caps.rasterization && caps.rayQuery && caps.accelerationStructure;
}

}   // namespace

// Before: the shadow ray stopped at the box, and the floor was lit only by
// the material's rays that happened to cross the glass onto the lamp -- at
// 32 paths a field of fireflies, as noisy as with the setting off. After:
// lit by next event estimation through the glass, and the deep frames with
// and without it are the same light (a real Fresnel, ior 1.5: the weights of
// the two strategies through the glass sum to one).
TEST_CASE("a lamp in a glass box lights the floor through the glass by its shadow rays",
          "[usd][gpu][mesh][path][mis][emissive][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!tracesRays(gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path boxed = stage("glass_box_clear.usda", GlassScene{});
    const gpu::Buffer deepOff = frame(gpu, boxed, false, 8192);
    const gpu::Buffer deepOn = frame(gpu, boxed, true, 8192);
    const gpu::Buffer off = frame(gpu, boxed, false, 32);
    const gpu::Buffer on = frame(gpu, boxed, true, 32);
    auto errOff = render::compareHdr(*gpu->library, off, deepOff, kWidth, kHeight);
    auto errOn = render::compareHdr(*gpu->library, on, deepOff, kWidth, kHeight);
    REQUIRE(errOff);
    REQUIRE(errOn);
    const render::ImageStats floorOff = floorOf(gpu, deepOff);
    const render::ImageStats floorOn = floorOf(gpu, deepOn);
    const render::ImageStats floorShallow = floorOf(gpu, on);
    std::printf("  a lamp in a clear glass box: the floor, deep, %.4f through the material's rays alone and %.4f "
                "with shadow rays through the glass (%.2f%%); at 32 paths against the deep frame, relMSE %.3e "
                "without and %.3e with (%.1f times less); the floor at 32 paths with them %.4f\n",
                floorOff.mean[1], floorOn.mean[1], 100.0 * (floorOn.mean[1] / floorOff.mean[1] - 1.0),
                errOff->relMse, errOn->relMse, errOff->relMse / std::max(errOn->relMse, 1e-30),
                floorShallow.mean[1]);
    CHECK(floorOff.mean[1] > 0.05);                                   // the lamp does reach the floor
    CHECK(floorShallow.mean[1] > 0.5 * floorOff.mean[1]);             // and 32 paths see it
    CHECK(errOn->relMse < errOff->relMse / 8.0);                      // by the shadow rays: fails before
    // The same light, counted once -- twice would be nearly double -- within
    // what the two strategies disagree by: a shadow ray pays the thin wall's
    // 2R / (1 + R), and the material's ray pays that and the specular layer's
    // Fresnel over again, (1 - F) at index 1.5 (lobes.slang takes a
    // transmission's Fresnel twice under OpenPBR's layer: docs/decisions.md,
    // "A ray leaves a surface by a hair"). Their weighed sum lies between
    // the material's rays alone and those over (1 - F): 3.45% over, measured.
    const double fresnel = (0.5 / 2.5) * (0.5 / 2.5);
    CHECK(floorOn.mean[1] > 0.99 * floorOff.mean[1]);
    CHECK(floorOn.mean[1] < 1.01 * floorOff.mean[1] / (1.0 - fresnel));
}

// A tinted glass tints the light its shadow rays let through, as the glass
// tints what the material's rays carry: the floor under a glass of colour
// (1, 0.5, 0.25) reads what the same floor reads with no glass and the lamp
// that colour, channel by channel. Index one, so a thin wall lets through
// its colour whole and nothing else.
TEST_CASE("through tinted glass the floor's light takes the glass's colour", "[usd][gpu][mesh][path][emissive][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!tracesRays(gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    GlassScene tinted;
    tinted.ior = 1.0F;
    tinted.tint = "1, 0.5, 0.25";
    GlassScene bare;
    bare.glass = false;
    bare.colour = "1, 0.5, 0.25";
    const gpu::Buffer through = frame(gpu, stage("glass_box_tinted.usda", tinted), true, 256);
    const gpu::Buffer reference = frame(gpu, stage("glass_box_tinted_bare.usda", bare), true, 8192);
    auto difference = render::compareHdr(*gpu->library, through, reference, kWidth, kHeight);
    REQUIRE(difference);
    const render::ImageStats floor = floorOf(gpu, through);
    const render::ImageStats truth = floorOf(gpu, reference);
    std::printf("  a lamp in a tinted glass box: the floor (%.4f, %.4f, %.4f), its green and blue over its red "
                "%.3f and %.3f; with no glass and the lamp that colour (%.4f, %.4f, %.4f); relMSE %.3e\n",
                floor.mean[0], floor.mean[1], floor.mean[2], floor.mean[1] / floor.mean[0],
                floor.mean[2] / floor.mean[0], truth.mean[0], truth.mean[1], truth.mean[2], difference->relMse);
    REQUIRE(floor.mean[0] > 0.05);
    CHECK(floor.mean[1] / floor.mean[0] == Catch::Approx(0.5).epsilon(0.05));
    CHECK(floor.mean[2] / floor.mean[0] == Catch::Approx(0.25).epsilon(0.05));
    for (int c = 0; c < 3; ++c) {
        CHECK(floor.mean[c] == Catch::Approx(truth.mean[c]).epsilon(0.05));
    }
    // At 256 paths the material's rays alone (before) leave the floor a
    // field of fireflies whose error against the deep frame is of order one.
    CHECK(difference->relMse < 0.05);
}

// UNBIASED FOR A THIN WALL: next event estimation through the glass and the
// material's rays through it, weighed against each other, are the light the
// lamp would give with no glass and its emission scaled by what the glass
// lets through -- deep, the floor within its noise and the frame pixel by
// pixel.
TEST_CASE("a thin glass's shadow rays are unbiased: the glass is the lamp scaled by its transmittance",
          "[usd][gpu][mesh][path][mis][emissive][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!tracesRays(gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    GlassScene grey;
    grey.ior = 1.0F;
    grey.tint = "0.5, 0.5, 0.5";
    GlassScene bare;
    bare.glass = false;
    bare.colour = "0.5, 0.5, 0.5";
    const gpu::Buffer through = frame(gpu, stage("glass_box_grey.usda", grey), true, 4096);
    const gpu::Buffer reference = frame(gpu, stage("glass_box_grey_bare.usda", bare), true, 8192);
    // The glass's own noise at 64 paths, not the bare scene's: the material's
    // rays that cross the glass onto the lamp make the glass frame far the
    // noisier of the two (relMSE 3.3e-2 at 4096 paths, where the bare one
    // reads 6.4e-4 at 64), and that is variance, not bias. Unbiased, the
    // error falls as the paths grow: 64 times the paths, at least 8 times
    // less error, where a bias would hold it at a floor.
    const gpu::Buffer shallow = frame(gpu, stage("glass_box_grey.usda", grey), true, 64);
    auto same = render::compareHdr(*gpu->library, through, reference, kWidth, kHeight);
    auto noise = render::compareHdr(*gpu->library, shallow, reference, kWidth, kHeight);
    REQUIRE(same);
    REQUIRE(noise);
    const render::ImageStats floor = floorOf(gpu, through);
    const render::ImageStats truth = floorOf(gpu, reference);
    std::printf("  a lamp in a grey thin glass box against the bare lamp at half its emission: the floor %.4f "
                "against %.4f (%.2f%%); relMSE %.3e, where 64 paths of the glass read %.3e\n",
                floor.mean[1], truth.mean[1], 100.0 * (floor.mean[1] / truth.mean[1] - 1.0), same->relMse,
                noise->relMse);
    REQUIRE(truth.mean[1] > 0.05);
    CHECK(std::abs(floor.mean[1] / truth.mean[1] - 1.0) < 0.02);
    CHECK(same->relMse < noise->relMse / 8.0);
}
