// Copyright (c) 2026 jesus luque.
//
// A THIN LENS IS TWO FACES (docs/decisions.md, "A ray leaves a surface by a
// hair"): the Corvette's tail lamps are white emitters behind a red lens, a
// closed solid 1.1 mm thick of OpenPBR glass whose transmission colour is
// sqrt(base_color) = (0.8319, 0, 0.1049), so that the two faces give the base
// colour between them. The path tracer started a ray that went into the lens
// some 7 mm past its entry point -- 1e-3 of the position's magnitude along the
// normal, along the ray and again as TMin, three metres from the origin --
// so it never met the far face: one tint and one Fresnel, blue 12.6% of red
// where Cycles reads 1.6%, and the lamps came out magenta. Checked through
// Hydra on an emitting quad behind a slab of glass three metres from the
// origin, the camera at the origin looking straight through: what reaches
// the camera is the slab's transmittance, channel by channel.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "glass_tint";
    fs::create_directories(dir);
    return dir / name;
}

/// What the stage holds: a white emitting 2 x 2 quad at z -3.2, emission 1;
/// in front of it, a 1 x 1 slab of OpenPBR glass whose near face is at z -3
/// and which is `thickness` deep (a closed box), or with `sheet` a single
/// quad at z -3 that is a thin wall; smooth, index 1.6, transmission colour
/// `tint`. The camera at the origin looks down -z, so the middle of the frame
/// sees the emitter through the glass, nearly at normal incidence.
struct SlabScene {
    float       thickness = 0.001F;
    bool        sheet = false;
    const char* tint = "0.8319, 0, 0.1049";
    const char* base = "0.692, 0, 0.011";
};

fs::path stage(const std::string& name, const SlabScene& scene) {
    const fs::path path = scratch(name);
    const float zNear = -3.0F;
    const float zFar = zNear - scene.thickness;
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Lamp\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-1, -1, -3.2), (1, -1, -3.2), (1, 1, -3.2), (-1, 1, -3.2)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    uniform bool doubleSided = 1\n"
           "    rel material:binding = </Materials/Glow>\n}\n";
    if (scene.sheet) {
        out << "def Mesh \"Lens\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -0.5, "
            << zNear << "), (0.5, -0.5, " << zNear << "), (0.5, 0.5, " << zNear << "), (-0.5, 0.5, " << zNear
            << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform bool doubleSided = 1\n"
               "    rel material:binding = </Materials/Glass>\n}\n";
    } else {
        // Outward faces, counter-clockwise from outside: far, near, -x, +x,
        // bottom, top.
        out << "def Mesh \"Lens\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n"
               "    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 4, 7, 3, 1, 2, 6, 5, 0, 1, 5, 4, 3, 7, 6, 2]\n"
               "    point3f[] points = [(-0.5, -0.5, "
            << zFar << "), (0.5, -0.5, " << zFar << "), (0.5, 0.5, " << zFar << "), (-0.5, 0.5, " << zFar
            << "),\n"
               "                        (-0.5, -0.5, "
            << zNear << "), (0.5, -0.5, " << zNear << "), (0.5, 0.5, " << zNear << "), (-0.5, 0.5, " << zNear
            << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform bool doubleSided = 1\n"
               "    rel material:binding = </Materials/Glass>\n}\n";
    }
    out << "def Scope \"Materials\"\n{\n"
           "    def Material \"Glow\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Glow/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_surface_unlit\"\n"
           "            float inputs:emission = 1\n"
           "            color3f inputs:emission_color = (1, 1, 1)\n"
           "            token outputs:out\n        }\n    }\n"
           "    def Material \"Glass\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Glass/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
           "            color3f inputs:base_color = ("
        << scene.base
        << ")\n"
           "            float inputs:transmission_weight = 1\n"
           "            color3f inputs:transmission_color = ("
        << scene.tint
        << ")\n"
           "            float inputs:specular_roughness = 0\n"
           "            float inputs:specular_ior = 1.6\n"
           "            bool inputs:geometry_thin_walled = "
        << (scene.sheet ? 1 : 0)
        << "\n"
           "            token outputs:out\n        }\n    }\n}\n"
           "def Camera \"Camera\"\n{\n"
           "    float focalLength = 50\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n}\n";
    return path;
}

constexpr uint32_t kWidth = 48;
constexpr uint32_t kHeight = 36;

/// The middle third of a path traced frame of `path`, 256 paths a pixel:
/// the emitter through the glass, within a few degrees of normal incidence.
render::ImageStats middle(test::Gpu* gpu, const fs::path& path) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(256);
    (*renderer)->setPathTotal(256);
    (*renderer)->setPathBounces(1);
    auto image = (*renderer)->render("/Camera", 0.0, kWidth, kHeight, "rt");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(made);
    auto stats = render::imageStats(*gpu->library, *made, kWidth, kHeight, kWidth / 3, kHeight / 3,
                                    2 * kWidth / 3, 2 * kHeight / 3);
    REQUIRE(stats);
    return *stats;
}

bool tracesRays(test::Gpu* gpu) {
    const gpu::Caps& caps = gpu->device->caps();
    return caps.rasterization && caps.rayQuery && caps.accelerationStructure;
}

/// What a smooth dielectric of index 1.6 lets through at normal incidence,
/// one face: 1 - ((n - 1) / (n + 1))^2.
constexpr double kFace = 1.0 - (0.6 / 2.6) * (0.6 / 2.6);

void report(const char* what, const render::ImageStats& s) {
    std::printf("  %s: (%.4f, %.5f, %.5f), green and blue over red %.5f and %.5f\n", what, s.mean[0], s.mean[1],
                s.mean[2], s.mean[1] / s.mean[0], s.mean[2] / s.mean[0]);
}

}   // namespace

// Before: B/R 0.126 and R 0.79 -- one face of the 1 mm slab, the ray into it
// started past the other. After: both faces, each tinting and each taking
// its Fresnel, as OpenPBR's transmission tint (transmission_depth 0) and
// Cycles have it: B/R (0.1049 / 0.8319)^2 = 0.0159 and R 0.8319^2 x 0.947^2.
TEST_CASE("a red lens 1 mm thick tints the lamp behind it at both its faces",
          "[usd][gpu][mesh][path][emissive][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!tracesRays(gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const double tR = 0.8319;
    const double tB = 0.1049;

    SECTION("solid, 1 mm: two tints and two Fresnels") {
        const render::ImageStats s = middle(gpu, stage("slab_1mm.usda", SlabScene{}));
        report("a 1 mm red slab three metres out", s);
        REQUIRE(s.mean[0] > 0.1);
        CHECK(s.mean[1] / s.mean[0] < 1.0e-3);
        CHECK(s.mean[2] / s.mean[0] == Catch::Approx((tB / tR) * (tB / tR)).epsilon(0.15));
        CHECK(s.mean[0] == Catch::Approx(tR * tR * kFace * kFace).epsilon(0.03));
    }
    SECTION("solid, 10 cm: the same, and the same before the fix") {
        SlabScene thick;
        thick.thickness = 0.1F;
        const render::ImageStats s = middle(gpu, stage("slab_10cm.usda", thick));
        report("a 10 cm red slab three metres out", s);
        REQUIRE(s.mean[0] > 0.1);
        CHECK(s.mean[1] / s.mean[0] < 1.0e-3);
        CHECK(s.mean[2] / s.mean[0] == Catch::Approx((tB / tR) * (tB / tR)).epsilon(0.15));
        CHECK(s.mean[0] == Catch::Approx(tR * tR * kFace * kFace).epsilon(0.03));
    }
    SECTION("a pure red, 1 mm: green and blue nothing") {
        SlabScene red;
        red.tint = "0.8, 0, 0";
        red.base = "0.64, 0, 0";
        const render::ImageStats s = middle(gpu, stage("slab_1mm_red.usda", red));
        report("a 1 mm pure red slab", s);
        REQUIRE(s.mean[0] > 0.1);
        CHECK(s.mean[1] / s.mean[0] < 1.0e-3);
        CHECK(s.mean[2] / s.mean[0] < 1.0e-3);
        CHECK(s.mean[0] == Catch::Approx(0.8 * 0.8 * kFace * kFace).epsilon(0.03));
    }
    SECTION("a thin wall: one face, one tint") {
        SlabScene sheet;
        sheet.sheet = true;
        const render::ImageStats s = middle(gpu, stage("sheet.usda", sheet));
        report("a thin red wall", s);
        // A thin wall reflects 2R / (1 + R) and lets the rest through once.
        const double r = 1.0 - kFace;
        REQUIRE(s.mean[0] > 0.1);
        CHECK(s.mean[1] / s.mean[0] < 1.0e-3);
        CHECK(s.mean[2] / s.mean[0] == Catch::Approx(tB / tR).epsilon(0.05));
        CHECK(s.mean[0] == Catch::Approx(tR * (1.0 - 2.0 * r / (1.0 + r))).epsilon(0.03));
    }
}
