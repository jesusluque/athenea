// Copyright (c) 2026 jesus luque.
//
// WHAT A CONVERTED SURFACE COVERS.
//
// `athenea mesh2splat` lays a gaussian a cell over a surface, `sigma` cells
// wide, so a point of it is under several at once. A partial opacity written
// straight into each of them is not the surface's: a mask of 0.5 covered 96%
// of what stood behind it. The conversion now takes every opacity it reads --
// the material's constant, a map's value, what a glass keeps -- as the
// surface's coverage and gives each gaussian what one of the stack needs for
// that (`scene::coverageOpacity`, `m2sCoverageAlpha`).
//
// The planes are tests/data/coverage/planes.usda, converted by the
// `mesh2splat_coverage` fixture before this runs (ctest does both; run alone,
// the case skips). Each is measured against the mesh with the same material:
// its alpha with nothing behind it, and its colour over a backdrop of 1 --
// the planes are black and reflect nothing, so what comes through is what
// they let through. At the screen size the mesh fills, a quarter of it and a
// twentieth, where a converted gaussian is a fraction of a pixel and the
// rasteriser's filter spreads it.
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "../gpu/GpuTest.h"
#include "athenea/geom/Mesh.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/MeshStage.h"
#include "athenea/usd/StageRenderer.h"

#include <pxr/base/plug/registry.h>

using namespace athenea;
namespace fs = std::filesystem;

namespace {

// Before any case builds USD's schema registry, as athenea_usd_tests does:
// the converted cloud applies the engine's own schemas.
[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "coverage";
    fs::create_directories(dir);
    return dir / name;
}

constexpr int      kPlanes = 8;
constexpr float    kScales[3] = {1.0F, 0.25F, 0.05F};
constexpr uint32_t kSide = 256;
// The camera's aperture over its focal length: how wide the view is a unit
// away, so a unit plane fills `scale` of the frame at 1 / (scale * this).
constexpr double   kAperture = 24.576;
constexpr double   kFocal = 35.0;

/// A layer over `source`: a camera a plane and a size, and a white backdrop
/// behind them all if asked.
fs::path shot(const fs::path& source, const std::string& name, bool backdrop) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    subLayers = [@" << source.string() << "@]\n)\n"
        << "def Scope \"Cams\"\n{\n";
    for (int plane = 0; plane < kPlanes; ++plane) {
        for (int k = 0; k < 3; ++k) {
            const double distance = 1.0 / (kScales[k] * kAperture / kFocal);
            out << "    def Camera \"P" << plane << "S" << k << "\"\n    {\n"
                << "        float focalLength = " << kFocal << "\n"
                << "        float horizontalAperture = " << kAperture << "\n"
                << "        float verticalAperture = " << kAperture << "\n"
                << "        float2 clippingRange = (0.1, 1000)\n"
                << "        double3 xformOp:translate = (" << plane * 1.5 << ", 0, " << distance << ")\n"
                << "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n";
        }
    }
    out << "}\n";
    if (backdrop) {
        out << "def Mesh \"Backdrop\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-200, -200, -2), (200, -200, -2), (200, 200, -2), (-200, 200, -2)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </White>\n}\n"
               "def Material \"White\"\n{\n"
               "    token outputs:mtlx:surface.connect = </White/Surface.outputs:out>\n"
               "    def Shader \"Surface\"\n    {\n"
               "        uniform token info:id = \"ND_standard_surface_surfaceshader\"\n"
               "        float inputs:base = 0\n        float inputs:specular = 0\n"
               "        float inputs:emission = 1\n        color3f inputs:emission_color = (1, 1, 1)\n"
               "        token outputs:out\n    }\n}\n";
    }
    return path;
}

/// The mean of a frame over the middle three fifths of the plane -- or, in
/// x, over [from, to) of it, the plane's centre at 0 and its sides at -0.5
/// and 0.5 -- as the image-statistics kernel measures it.
std::array<double, 4> meanOver(test::Gpu& gpu, const usd::StageImage& image, float scale, float from = -0.3F,
                               float to = 0.3F) {
    gpu::BufferDesc desc;
    desc.bytes = image.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto frame = gpu::Buffer::create(*gpu.device, desc, image.rgba.data());
    REQUIRE(frame);
    const float size = scale * static_cast<float>(kSide);
    const float centre = static_cast<float>(kSide) * 0.5F;
    const auto at = [&](float x) { return static_cast<uint32_t>(std::lround(centre + x * size)); };
    const uint32_t x0 = at(from);
    const uint32_t x1 = std::max(at(to), x0 + 1);
    const uint32_t y0 = at(-0.3F);
    const uint32_t y1 = std::max(at(0.3F), y0 + 1);
    auto stats = render::imageStats(*gpu.library, *frame, image.width, image.height, x0, y0, x1, y1);
    if (!stats) FAIL(stats.error().toString());
    return stats->mean;
}

}   // namespace

TEST_CASE("a converted surface covers what its material's opacity says, at any size, on both routes",
          "[usd][gpu][mesh2splat][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path source = ATHENEA_COVERAGE_STAGE;
    const fs::path cloud = ATHENEA_COVERAGE_CLOUD;
    if (!fs::exists(cloud)) {
        SKIP("the converted planes are the mesh2splat_coverage fixture's: run this through ctest");
    }
    const fs::path meshBare = shot(source, "mesh_bare.usda", false);
    const fs::path cloudBare = shot(cloud, "cloud_bare.usda", false);
    const fs::path cloudOver = shot(cloud, "cloud_over.usda", true);

    // What the mesh covers, at the size it fills: its alpha with nothing
    // behind it. Coverage is a property of the surface, not of the screen,
    // and the mesh's map planes are cut by a pixel's lot -- this is where
    // that lot averages over most pixels.
    std::array<double, kPlanes> wanted{};
    {
        auto renderer = usd::StageRenderer::open(meshBare);
        if (!renderer) FAIL(renderer.error().toString());
        for (int plane = 0; plane < kPlanes; ++plane) {
            auto image = (*renderer)->render("/Cams/P" + std::to_string(plane) + "S0", 0.0, kSide, kSide);
            if (!image) FAIL(image.error().toString());
            wanted[size_t(plane)] = meanOver(*gpu, *image, 1.0F)[3];
        }
    }
    // The planes' own numbers, so that a mesh that drew nothing is not a
    // reference: 0.25, 0.5 and 0.75 constant, the same off maps of one value
    // (64, 128 and 191 of 255), opaque, and a cut-out of half.
    const std::array<double, kPlanes> nominal{0.25, 0.5, 0.75, 64 / 255.0, 128 / 255.0, 191 / 255.0, 1.0, 0.5};
    for (int plane = 0; plane < kPlanes; ++plane) {
        INFO("plane " << plane);
        CHECK(std::abs(wanted[size_t(plane)] - nominal[size_t(plane)]) < 0.02);
    }

    const gpu::Caps& caps = gpu->device->caps();
    const bool traces = caps.accelerationStructure && (caps.rayQuery || caps.rayTracing);
    for (const char* technique : {"raster", "rt"}) {
        if (std::string(technique) == "rt" && !traces) {
            WARN("no ray tracing here: the traced route is not measured");
            continue;
        }
        auto bare = usd::StageRenderer::open(cloudBare);
        auto over = usd::StageRenderer::open(cloudOver);
        if (!bare) FAIL(bare.error().toString());
        if (!over) FAIL(over.error().toString());
        for (auto* renderer : {bare->get(), over->get()}) {
            renderer->setPathSamples(4);
            renderer->setPathTotal(4);
        }
        for (int plane = 0; plane < kPlanes; ++plane) {
            for (int k = 0; k < 3; ++k) {
                const std::string camera = "/Cams/P" + std::to_string(plane) + "S" + std::to_string(k);
                auto alone = (*bare)->render(camera, 0.0, kSide, kSide, technique);
                auto backed = (*over)->render(camera, 0.0, kSide, kSide, technique);
                if (!alone) FAIL(alone.error().toString());
                if (!backed) FAIL(backed.error().toString());
                const double alpha = meanOver(*gpu, *alone, kScales[k])[3];
                const double through = meanOver(*gpu, *backed, kScales[k])[1];
                const double want = wanted[size_t(plane)];
                std::printf("  %-6s plane %d at %.2fx: covers %.4f (alpha), %.4f (colour), the mesh %.4f\n",
                            technique, plane, double(kScales[k]), alpha, 1.0 - through, want);
                INFO(technique << " plane " << plane << " at " << kScales[k] << "x");
                if (plane == 3) {
                    // A map of 0.25 is under the default cut: no surface at all.
                    CHECK(alpha < 0.02);
                    continue;
                }
                // Whole stays whole.
                if (plane == 6) {
                    CHECK(alpha >= 0.99);
                    CHECK(1.0 - through >= 0.99);
                    continue;
                }
                // A cut-out still cuts: its left half (0.25, under the
                // threshold) is not there and its right (0.75, over it) is
                // whole, not three quarters. Measured clear of the band the
                // two meet in, where an edge of whole gaussians spills past
                // the cut by a fraction of a cell -- an edge's softness, not
                // a coverage: over the middle three fifths it read 0.536.
                if (plane == 7) {
                    const double left = meanOver(*gpu, *alone, kScales[k], -0.4F, -0.15F)[3];
                    const double right = meanOver(*gpu, *alone, kScales[k], 0.15F, 0.4F)[3];
                    std::printf("  %-6s the cut-out at %.2fx: left half %.4f, right half %.4f\n", technique,
                                double(kScales[k]), left, right);
                    CHECK(left < 0.02);
                    CHECK(right > 0.98);
                    continue;
                }
                CHECK(std::abs(alpha - want) < 0.03);
                CHECK(std::abs((1.0 - through) - want) < 0.03);
            }
        }
    }
}

// WHAT A MATERIAL SAYS ITS OPACITY IS. A constant under one is coverage in
// MaterialX (and in UsdPreviewSurface's `presence` mode), a threshold makes a
// cut-out of a map and decides a constant, and glTF says which by its
// `alpha_mode`: OPAQUE ignores the alpha, MASK cuts at `alpha_cutoff`, BLEND
// is coverage. UsdPreviewSurface's default `transparent` mode keeps reading a
// constant as a thin wall's transmission.
TEST_CASE("a material's opacity is read as coverage, as a cut-out by threshold, or as a thin wall",
          "[usd][gpu][mesh2splat][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    struct Case {
        const char* name;
        const char* id;
        const char* inputs;
    };
    const std::array<Case, 9> cases{{
        {"Standard", "ND_standard_surface_surfaceshader", "color3f inputs:opacity = (0.3, 0.3, 0.3)"},
        {"Open", "ND_open_pbr_surface_surfaceshader", "float inputs:geometry_opacity = 0.4"},
        {"GltfOpaque", "ND_gltf_pbr_surfaceshader", "float inputs:alpha = 0.3"},
        {"GltfMask", "ND_gltf_pbr_surfaceshader",
         "float inputs:alpha = 0.3\n            int inputs:alpha_mode = 1\n            float inputs:alpha_cutoff = 0.2"},
        {"GltfBlend", "ND_gltf_pbr_surfaceshader", "float inputs:alpha = 0.3\n            int inputs:alpha_mode = 2"},
        {"Preview", "UsdPreviewSurface", "float inputs:opacity = 0.3"},
        {"PreviewPresence", "UsdPreviewSurface",
         "float inputs:opacity = 0.3\n            token inputs:opacityMode = \"presence\""},
        {"PreviewKept", "UsdPreviewSurface", "float inputs:opacity = 0.6\n            float inputs:opacityThreshold = 0.5"},
        {"PreviewGone", "UsdPreviewSurface", "float inputs:opacity = 0.4\n            float inputs:opacityThreshold = 0.5"},
    }};
    const fs::path path = scratch("opacity_reading.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        for (size_t k = 0; k < cases.size(); ++k) {
            out << "def Mesh \"" << cases[k].name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                << "    point3f[] points = [(" << k << ", 0, 0), (" << k + 1 << ", 0, 0), (" << k << ", 1, 0)]\n"
                << "    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] ( interpolation = \"vertex\" )\n"
                << "    uniform token subdivisionScheme = \"none\"\n"
                << "    rel material:binding = </Looks/" << cases[k].name << ">\n}\n";
        }
        out << "def Scope \"Looks\"\n{\n";
        for (const Case& c : cases) {
            const bool preview = std::string(c.id) == "UsdPreviewSurface";
            const std::string at = std::string("/Looks/") + c.name;
            out << "    def Material \"" << c.name << "\"\n    {\n"
                << "        token outputs:" << (preview ? "surface" : "mtlx:surface") << ".connect = <" << at
                << "/Surface.outputs:" << (preview ? "surface" : "out") << ">\n"
                << "        def Shader \"Surface\"\n        {\n"
                << "            uniform token info:id = \"" << c.id << "\"\n"
                << "            " << c.inputs << "\n"
                << "            token outputs:" << (preview ? "surface" : "out") << "\n        }\n    }\n";
        }
        out << "}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == cases.size());
    const auto material = [&](const char* name) -> const usd::StageMaterial& {
        for (const usd::StageMesh& mesh : *meshes) {
            if (mesh.path == std::string("/") + name) {
                return mesh.material;
            }
        }
        FAIL("no mesh " << name);
        return meshes->front().material;
    };
    const auto near = [](float a, float b) { return std::abs(a - b) < 1e-5F; };
    CHECK(near(material("Standard").opacity, 0.3F));        // color3, its mean
    CHECK(near(material("Open").opacity, 0.4F));
    CHECK(near(material("GltfOpaque").opacity, 1.0F));      // OPAQUE: the alpha is not the surface's
    CHECK(near(material("GltfMask").opacity, 1.0F));        // 0.3 at least the cutoff of 0.2: whole
    CHECK(near(material("GltfBlend").opacity, 0.3F));
    CHECK(near(material("Preview").opacity, 1.0F));         // transparent: a thin wall
    CHECK(near(material("Preview").transmission, 0.7F));
    CHECK(material("Preview").thinWalled);
    CHECK(near(material("PreviewPresence").opacity, 0.3F)); // presence: coverage
    CHECK(near(material("PreviewPresence").transmission, 0.0F));
    CHECK(near(material("PreviewKept").opacity, 1.0F));     // over its threshold: whole
    CHECK(near(material("PreviewKept").transmission, 0.0F));
    CHECK(near(material("PreviewGone").opacity, 0.0F));     // under it: not there
    for (const usd::StageMesh& mesh : *meshes) {
        CHECK(mesh.material.opacityThreshold == 0.0F);      // no map, so no threshold kept
    }
}
