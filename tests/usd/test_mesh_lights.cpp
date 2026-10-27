// Copyright (c) 2026 jesus luque.
//
// MESH LIGHTS THROUGH HYDRA (technique::MeshLights): emitting meshes and
// UsdLux MeshLightAPI prims as lights. The path tracer through the clusters'
// tree draws the image the flat table draws, with less noise where emitters
// are many; a MeshLightAPI quad lights a floor as a RectLight of its size
// and radiance does, in each sync mode. Frames are compared on the device.
//
// With ATHENEA_MESH_LIGHT_IMAGES set to a folder, each case also leaves its
// frames there side by side (a | b | difference) as PNG, for a person.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "mesh_lights";
    fs::create_directories(dir);
    return dir / name;
}

const char* kFloorAndCamera =
    "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
    "def Mesh \"Floor\"\n{\n"
    "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
    "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
    "    uniform token subdivisionScheme = \"none\"\n"
    "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
    "def Camera \"Camera\"\n{\n"
    "    float focalLength = 24\n"
    "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
    "    float2 clippingRange = (0.1, 1000)\n"
    "    double3 xformOp:translate = (0, 1.2, 2)\n"
    "    double xformOp:rotateX = -30\n"
    "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";

/// A quad from (x0, z0) to (x1, z1) at height y facing down.
std::string downQuad(const std::string& name, float x0, float z0, float x1, float z1, float y,
                     const std::string& extra) {
    char points[256];
    std::snprintf(points, sizeof(points), "[(%g, %g, %g), (%g, %g, %g), (%g, %g, %g), (%g, %g, %g)]", x0, y, z0, x1,
                  y, z0, x1, y, z1, x0, y, z1);
    return "def Mesh \"" + name + "\" (\n    prepend apiSchemas = [" + extra + "]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = " + points + "\n"
           "    uniform token subdivisionScheme = \"none\"\n";
}

const char* kGlowMaterial =
    "def Scope \"Materials\"\n{\n"
    "    def Material \"Glow\"\n    {\n"
    "        token outputs:mtlx:surface.connect = </Materials/Glow/Surface.outputs:out>\n"
    "        def Shader \"Surface\"\n        {\n"
    "            uniform token info:id = \"ND_surface_unlit\"\n"
    "            float inputs:emission = 3\n"
    "            color3f inputs:emission_color = (1, 0.8, 0.6)\n"
    "            token outputs:out\n        }\n    }\n}\n";

struct Frame {
    gpu::Buffer rgba;
    uint32_t    width = 0;
    uint32_t    height = 0;
};

Frame frameOf(test::Gpu& gpu, const fs::path& path, uint32_t w, uint32_t h, const char* technique, bool meshLights,
             uint32_t total, const std::function<void(usd::StageRenderer&)>& more = {}) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setMeshLights(meshLights);
    (*renderer)->setPathMis(true);
    (*renderer)->setPathSamples(std::min<uint32_t>(total, 256));
    (*renderer)->setPathTotal(total);
    (*renderer)->setPathBounces(1);
    if (more) {
        more(**renderer);
    }
    auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(made);
    return {std::move(*made), w, h};
}

/// a | b | |a - b| x gain, written as a PNG where ATHENEA_MESH_LIGHT_IMAGES asks.
void picture(test::Gpu& gpu, const Frame& a, const Frame& b, const std::string& name, float exposure, float gain) {
    const char* folder = std::getenv("ATHENEA_MESH_LIGHT_IMAGES");
    if (folder == nullptr || *folder == '\0') {
        return;
    }
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/triptych", "triptych");
    REQUIRE(kernel);
    const uint32_t w = a.width;
    const uint32_t h = a.height;
    gpu::Buffer out = test::uintBuffer(*gpu.device, uint64_t{w} * 3 * h, "triptych");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(a.rgba.rhi());
            cursor["b"].setBinding(b.rgba.rhi());
            cursor["out"].setBinding(out.rhi());
            cursor["width"].setData(w);
            cursor["height"].setData(h);
            cursor["exposure"].setData(exposure);
            cursor["gain"].setData(gain);
        });
        REQUIRE(batch.submit(true));
    }
    auto pixels = out.readAll<uint32_t>(*gpu.device);
    REQUIRE(pixels);
    fs::create_directories(folder);
    const fs::path file = fs::path(folder) / (name + ".png");
    pxr::HioImageSharedPtr image = pxr::HioImage::OpenForWriting(file.string());
    REQUIRE(image);
    pxr::HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w * 3);
    spec.height = static_cast<int>(h);
    spec.format = pxr::HioFormatUNorm8Vec4srgb;
    spec.flipped = false;
    spec.data = pixels->data();
    CHECK(image->Write(spec));
    std::printf("  wrote %s\n", file.string().c_str());
}

}   // namespace

// The tree is a better way to choose the same light, not another light: deep
// frames through it and through EmissiveTable's flat distribution agree to
// within their own noise, on a floor under two emitters of different size.
TEST_CASE("the path tracer through the mesh lights' tree draws the flat table's image",
          "[usd][gpu][mesh][path][meshlights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("tree_against_flat.usda");
    {
        std::ofstream out(path);
        out << kFloorAndCamera << downQuad("Panel", -0.5F, -3.5F, 0.5F, -2.5F, 2.5F, "\"MaterialBindingAPI\"")
            << "    rel material:binding = </Materials/Glow>\n}\n"
            << downQuad("Strip", -3.0F, -5.0F, 2.0F, -4.8F, 1.5F, "\"MaterialBindingAPI\"")
            << "    rel material:binding = </Materials/Glow>\n}\n"
            << kGlowMaterial;
    }
    const uint32_t w = 64, h = 48;
    const Frame flatDeep = frameOf(*gpu, path, w, h, "rt", false, 8192);
    const Frame treeDeep = frameOf(*gpu, path, w, h, "rt", true, 8192);
    const Frame flat = frameOf(*gpu, path, w, h, "rt", false, 32);
    const Frame tree = frameOf(*gpu, path, w, h, "rt", true, 32);
    auto same = render::compareHdr(*gpu->library, treeDeep.rgba, flatDeep.rgba, w, h);
    auto errFlat = render::compareHdr(*gpu->library, flat.rgba, flatDeep.rgba, w, h);
    auto errTree = render::compareHdr(*gpu->library, tree.rgba, flatDeep.rgba, w, h);
    REQUIRE(same);
    REQUIRE(errFlat);
    REQUIRE(errTree);
    std::printf("  tree against flat, 8192 paths each: relMSE %.3e (p99 relative %.3e); at 32 paths against the "
                "deep flat frame: flat %.3e, tree %.3e\n",
                same->relMse, same->p99Relative, errFlat->relMse, errTree->relMse);
    picture(*gpu, flatDeep, treeDeep, "tree_against_flat", 1.0F, 8.0F);
    // The deep frames part by their own noise: 8192 paths carry 256 times
    // less of it than 32, and two of them sum theirs.
    CHECK(same->relMse < errFlat->relMse / 32.0);
}

// MANY EMITTERS: sixty thin strips over a floor, each its own cluster or
// few. A flat distribution spends its samples on the strips by power alone,
// wherever the shading point is; the tree spends them on the strips near it.
TEST_CASE("sixty strips are sampled with less noise through the tree than by the flat table",
          "[usd][gpu][mesh][path][meshlights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("sixty_strips.usda");
    {
        std::ofstream out(path);
        out << kFloorAndCamera;
        for (int i = 0; i < 60; ++i) {
            const float x = -5.5F + 11.0F * float(i % 10) / 9.0F;
            const float z = 1.0F - 9.0F * float(i / 10) / 5.0F;
            out << downQuad("Strip" + std::to_string(i), x - 0.4F, z - 0.03F, x + 0.4F, z + 0.03F, 0.6F,
                            "\"MaterialBindingAPI\"")
                << "    rel material:binding = </Materials/Glow>\n}\n";
        }
        out << kGlowMaterial;
    }
    const uint32_t w = 64, h = 48;
    const Frame deep = frameOf(*gpu, path, w, h, "rt", false, 8192);
    const Frame flat = frameOf(*gpu, path, w, h, "rt", false, 32);
    const Frame tree = frameOf(*gpu, path, w, h, "rt", true, 32);
    const Frame treeDeep = frameOf(*gpu, path, w, h, "rt", true, 8192);
    auto errFlat = render::compareHdr(*gpu->library, flat.rgba, deep.rgba, w, h);
    auto errTree = render::compareHdr(*gpu->library, tree.rgba, deep.rgba, w, h);
    auto same = render::compareHdr(*gpu->library, treeDeep.rgba, deep.rgba, w, h);
    REQUIRE(errFlat);
    REQUIRE(errTree);
    REQUIRE(same);
    std::printf("  sixty strips at 32 paths against 8192 of the flat table: flat %.3e, tree %.3e (%.2f times less); "
                "the deep frames part by %.3e\n",
                errFlat->relMse, errTree->relMse, errFlat->relMse / std::max(errTree->relMse, 1e-30), same->relMse);
    picture(*gpu, flat, tree, "sixty_strips_32_flat_tree", 1.0F, 4.0F);
    CHECK(errTree->relMse < errFlat->relMse);
    CHECK(same->relMse < errFlat->relMse / 32.0);
}

// MESHLIGHTAPI: a quad with the API and no material lights a floor as a
// RectLight of its size and intensity does, in each sync mode -- with no
// emission of its own a material neither tints nor adds -- and a bound
// emitting material is dropped where the mode says noMaterialResponse.
TEST_CASE("a MeshLightAPI quad lights a floor as a rect light of its size and intensity, in each sync mode",
          "[usd][gpu][mesh][path][meshlights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 64, h = 48;
    const fs::path rectPath = scratch("rect_reference.usda");
    {
        std::ofstream out(rectPath);
        out << kFloorAndCamera
            << "def RectLight \"Panel\"\n{\n"
               "    float inputs:intensity = 3\n    float inputs:width = 1\n    float inputs:height = 1\n"
               "    color3f inputs:color = (1, 0.8, 0.6)\n"
               "    double3 xformOp:translate = (0, 2.5, -3)\n"
               "    double xformOp:rotateX = -90\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    }
    const Frame rect = frameOf(*gpu, rectPath, w, h, "rt", true, 8192);
    struct Mode {
        const char* name;
        bool        material;   // an emitting material bound as well
    };
    for (const Mode& mode : {Mode{"materialGlowTintsLight", false}, Mode{"independent", false},
                             Mode{"noMaterialResponse", true}}) {
        const fs::path path = scratch(std::string("meshlight_") + mode.name + ".usda");
        {
            std::ofstream out(path);
            out << kFloorAndCamera
                << downQuad("Panel", -0.5F, -3.5F, 0.5F, -2.5F, 2.5F,
                            mode.material ? "\"MeshLightAPI\", \"MaterialBindingAPI\"" : "\"MeshLightAPI\"")
                << "    float inputs:intensity = 3\n"
                   "    color3f inputs:color = (1, 0.8, 0.6)\n"
                   "    token light:materialSyncMode = \"" << mode.name << "\"\n"
                << (mode.material ? "    rel material:binding = </Materials/Glow>\n" : "") << "}\n"
                << (mode.material ? kGlowMaterial : "");
        }
        const Frame deep = frameOf(*gpu, path, w, h, "rt", true, 8192);
        const Frame shallow = frameOf(*gpu, path, w, h, "rt", true, 32);
        auto same = render::compareHdr(*gpu->library, deep.rgba, rect.rgba, w, h);
        auto noise = render::compareHdr(*gpu->library, shallow.rgba, rect.rgba, w, h);
        REQUIRE(same);
        REQUIRE(noise);
        std::printf("  MeshLightAPI (%s) against a rect light, 8192 paths: relMSE %.3e (p99 relative %.3e); its "
                    "32-path frame %.3e\n",
                    mode.name, same->relMse, same->p99Relative, noise->relMse);
        picture(*gpu, deep, rect, std::string("meshlightapi_") + mode.name + "_vs_rect", 1.0F, 8.0F);
        CHECK(same->relMse < noise->relMse / 32.0);
    }
}

// THE PICTURES (hidden: `athenea_usd_tests "[.meshlights-pictures]"` with
// ATHENEA_MESH_LIGHT_IMAGES set): a showcase stage -- a floor, a box, strips
// over them and a MeshLightAPI lamp -- drawn by the raster and path traced,
// side by side with their difference, at a size a person can judge.
TEST_CASE("mesh lights: the raster against the path tracer, as pictures", "[.meshlights-pictures]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("showcase.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-8, 0, 4), (8, 0, 4), (8, 0, -10), (-8, 0, -10)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.7, 0.7, 0.7)] ( interpolation = \"constant\" )\n}\n"
               "def Cube \"Box\"\n{\n    double size = 1\n"
               "    color3f[] primvars:displayColor = [(0.6, 0.3, 0.2)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0.3, 0.5, -3.5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        for (int i = 0; i < 8; ++i) {
            const float x = -3.5F + float(i);
            out << downQuad("Strip" + std::to_string(i), x - 0.05F, -6.0F, x + 0.05F, -1.0F, 2.2F,
                            "\"MaterialBindingAPI\"")
                << "    rel material:binding = </Materials/Glow>\n}\n";
        }
        out << downQuad("Lamp", -1.0F, -8.0F, 1.0F, -7.0F, 1.2F, "\"MeshLightAPI\"")
            << "    float inputs:intensity = 4\n    color3f inputs:color = (0.6, 0.8, 1)\n}\n"
            << kGlowMaterial
            << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 36\n    float verticalAperture = 20.25\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 2.6, 2.5)\n"
               "    double xformOp:rotateX = -22\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    }
    const uint32_t w = 1024, h = 576;
    const auto samples = [](usd::StageRenderer& r) { r.setLightSamples(16); };
    const Frame raster = frameOf(*gpu, path, w, h, "raster", true, 1, samples);
    const Frame sampled = frameOf(*gpu, path, w, h, "raster", true, 1,
                                  [](usd::StageRenderer& r) { r.setMeshLightSamples(16); });
    // The direct light alone, through the pixels' middles: what the raster draws.
    const Frame traced = frameOf(*gpu, path, w, h, "rt", true, 1024, [](usd::StageRenderer& r) {
        r.setPathBounces(0);
        r.setAntialias(false);
    });
    picture(*gpu, sampled, traced, "showcase_sampled16_pt_diff", 1.0F, 2.0F);
    auto diff = render::compareHdr(*gpu->library, raster.rgba, traced.rgba, w, h);
    REQUIRE(diff);
    std::printf("  showcase, raster against 1024 paths: relMSE %.3e, p99 relative %.3e\n", diff->relMse,
                diff->p99Relative);
    picture(*gpu, raster, traced, "showcase_raster_pt_diff", 1.0F, 2.0F);
}
