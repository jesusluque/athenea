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

void report(const char* what, const render::ImageStats& s) {
    std::printf("  %s: (%.4f, %.5f, %.5f), green and blue over red %.5f and %.5f\n", what, s.mean[0], s.mean[1],
                s.mean[2], s.mean[1] / s.mean[0], s.mean[2] / s.mean[0]);
}

}   // namespace

// Before: B/R 0.1261 -- one face of the 1 mm slab, the ray into it started
// past the other. After: both faces, each tinting, as OpenPBR's transmission
// tint (transmission_depth 0) and Cycles have it: B/R (0.1049 / 0.8319)^2 =
// 0.0159, measured 0.01590. The level is held to the same slab 10 cm thick,
// which no margin ever stepped through -- the same kernel, the same faces --
// rather than to a closed form: the lobe library takes each face's Fresnel
// twice for a transmission under OpenPBR's specular layer (0.5556 =
// 0.692 x 0.9467^4 where one Fresnel a face is 0.620), a defect of its own
// (docs/decisions.md), and a closed form here would hold that in place.
TEST_CASE("a red lens 1 mm thick tints the lamp behind it at both its faces",
          "[usd][gpu][mesh][path][emissive][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!tracesRays(gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const double tR = 0.8319;
    const double tB = 0.1049;
    SlabScene thick;
    thick.thickness = 0.1F;
    SlabScene red;
    red.tint = "0.8, 0, 0";
    red.base = "0.64, 0, 0";
    SlabScene redThick = red;
    redThick.thickness = 0.1F;
    SlabScene sheet;
    sheet.sheet = true;
    const render::ImageStats thin = middle(gpu, stage("slab_1mm.usda", SlabScene{}));
    const render::ImageStats deep = middle(gpu, stage("slab_10cm.usda", thick));
    const render::ImageStats redThin = middle(gpu, stage("slab_1mm_red.usda", red));
    const render::ImageStats redDeep = middle(gpu, stage("slab_10cm_red.usda", redThick));
    const render::ImageStats wall = middle(gpu, stage("sheet.usda", sheet));
    report("a 1 mm red slab three metres out", thin);
    report("the same 10 cm thick", deep);
    report("a 1 mm pure red slab", redThin);
    report("the same 10 cm thick", redDeep);
    report("a thin red wall", wall);
    for (const render::ImageStats* s : {&thin, &deep, &redThin, &redDeep, &wall}) {
        REQUIRE(s->mean[0] > 0.1);
        CHECK(s->mean[1] / s->mean[0] < 1.0e-3);   // the tint has no green
    }
    // Two faces, two tints: blue over red the tint's ratio squared.
    CHECK(thin.mean[2] / thin.mean[0] == Catch::Approx((tB / tR) * (tB / tR)).epsilon(0.02));
    CHECK(deep.mean[2] / deep.mean[0] == Catch::Approx((tB / tR) * (tB / tR)).epsilon(0.02));
    // And the light of two faces: what the thick slab passes (one face of the
    // thin one let a third more through).
    CHECK(thin.mean[0] == Catch::Approx(deep.mean[0]).epsilon(0.01));
    CHECK(thin.mean[2] == Catch::Approx(deep.mean[2]).epsilon(0.02));
    // A pure red passes no blue at all, at either thickness.
    CHECK(redThin.mean[2] / redThin.mean[0] < 1.0e-3);
    CHECK(redThin.mean[0] == Catch::Approx(redDeep.mean[0]).epsilon(0.01));
    // A thin wall is one face: one tint.
    CHECK(wall.mean[2] / wall.mean[0] == Catch::Approx(tB / tR).epsilon(0.02));
}
