// Copyright (c) 2026 jesus luque.
//
// THE WORKING SPACE CHANGES NO PIXEL. Every kernel that took Rec.709 for
// granted -- luminance weights, the XYZ matrix of a blackbody, a capture's
// sRGB -- reads it from the working-space record now, and for
// lin_rec709_scene the record holds the constants the kernels held. These
// cases render the stages that reach every one of those kernels, by raster,
// by the path tracer and as splats, and compare them, word for word, with
// what the engine rendered before the record existed.
//
// The references are files in tests/data/working_space, written by this
// same source built against the engine before the change (it uses nothing
// the change added): run with ATHENEA_WRITE_GOLDENS=<dir> and the cases write
// their frames there instead of comparing. Without the files a case skips.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hio/image.h>

#include "athenea/colour/ColourCompiler.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Readers.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratchDir() {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "working_space";
    fs::create_directories(dir);
    return dir;
}

/// A cloud of `count` gaussians from a fixed sequence: the same cloud on
/// every run and every build, so the frames it draws can be compared.
io::RawSplats fixedCloud(uint32_t count) {
    io::RawSplats raw;
    raw.source = "working space synthetic";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 23;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
    uint64_t state = 2026;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t i = 0; i < count; ++i) {
        raw.records.resize(raw.records.size() + 23, 0.0F);
        float* r = raw.records.data() + raw.records.size() - 23;
        r[0] = next() * 2 - 1; r[1] = next() * 1.5F; r[2] = next() * 2 - 3;
        r[3] = next() * 6 - 2;
        r[4] = std::log(0.02F + next() * 0.15F); r[5] = std::log(0.02F + next() * 0.15F);
        r[6] = std::log(0.01F + next() * 0.08F);
        r[7] = next() - 0.5F; r[8] = next() - 0.5F; r[9] = next() - 0.5F; r[10] = next() - 0.5F;
        for (int k = 11; k < 23; ++k) {
            r[k] = next() * 2 - 1;
        }
        raw.count += 1;
    }
    return raw;
}

/// A lat-long sky with a bright, warm spot: luminance that is not grey, so
/// the dome's importance warp, its sun and its prefilter all weigh colour.
fs::path writeSky(const fs::path& dir) {
    const fs::path png = dir / "sky.png";
    fs::remove(png);
    const uint32_t w = 64;
    const uint32_t h = 32;
    std::vector<uint32_t> texels(size_t{w} * h);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t r = 40 + (x * 3) % 120;
            uint32_t g = 60 + (y * 5) % 140;
            uint32_t b = 200 - (y * 4) % 100;
            if (x >= 20 && x < 23 && y >= 6 && y < 9) {
                r = 255; g = 230; b = 160;
            }
            texels[size_t{y} * w + x] = 0xFF000000u | (b << 16) | (g << 8) | r;   // ABGR
        }
    }
    HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
    REQUIRE(made);
    HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = HioFormatUNorm8Vec4;
    spec.data = texels.data();
    REQUIRE(made->Write(spec));
    return png;
}

/// The edit every case's cloud stands under: a grade (tint and saturation,
/// the kernel's luminance weights) over part of it.
const char* kEdit =
    "        bool primvars:athenea:edit:active = 1 ( interpolation = \"constant\" )\n"
    "        token primvars:athenea:edit:shape = \"sphere\" ( interpolation = \"constant\" )\n"
    "        token primvars:athenea:edit:mode = \"grade\" ( interpolation = \"constant\" )\n"
    "        float3 primvars:athenea:edit:centre = (0.3, 0.6, -2) ( interpolation = \"constant\" )\n"
    "        float3 primvars:athenea:edit:size = (0.9, 0, 0) ( interpolation = \"constant\" )\n"
    "        color3f primvars:athenea:edit:tint = (1, 0.5, 0.2) ( interpolation = \"constant\" )\n"
    "        float primvars:athenea:edit:saturation = 0.4 ( interpolation = \"constant\" )\n";

const char* kCamera =
    "def Camera \"Camera\"\n{\n"
    "    float focalLength = 24\n"
    "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
    "    float2 clippingRange = (0.1, 1000)\n"
    "    double3 xformOp:translate = (0, 1.2, 2.5)\n"
    "    double xformOp:rotateX = -10\n"
    "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";

/// Meshes under every kind of light the record reaches -- a blackbody sphere
/// and sun, a coloured rect, a sky with an image -- in a UsdPreviewSurface
/// with a clearcoat and a MaterialX standard_surface with a coat and a
/// specular (lobe stacks the luminance weights choose in), and an edited
/// cloud among them.
fs::path writeLitStage(const fs::path& dir, const fs::path& sky, const fs::path& cloud) {
    const fs::path path = dir / "lit.usda";
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n"
           "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Coated>\n}\n"
           "def Sphere \"Ball\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    double radius = 0.6\n"
           "    double3 xformOp:translate = (-1.1, 0.6, -2.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
           "    rel material:binding = </Materials/Standard>\n}\n"
           "def Cube \"Box\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    double size = 0.8\n"
           "    double3 xformOp:translate = (1.2, 0.4, -3)\n"
           "    double xformOp:rotateY = 30\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateY\"]\n"
           "    rel material:binding = </Materials/Coated>\n}\n"
           "def Xform \"Group\" (\n    prepend apiSchemas = [\"AtheneaSplatEditAPI\"]\n)\n{\n"
        << kEdit
        << "    def \"Cloud\" ( references = @" << cloud.string() << "@</World/Splats> )\n    {\n"
           "        double3 xformOp:translate = (0.3, 0.2, -2.2)\n"
           "        double3 xformOp:scale = (0.6, 0.6, 0.6)\n"
           "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n    }\n}\n"
           "def SphereLight \"Bulb\"\n{\n"
           "    float inputs:intensity = 12\n    float inputs:radius = 0.3\n"
           "    bool inputs:enableColorTemperature = 1\n    float inputs:colorTemperature = 2700\n"
           "    double3 xformOp:translate = (-1.5, 2.2, -1.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
           "def RectLight \"Panel\"\n{\n"
           "    float inputs:intensity = 3\n    float inputs:width = 2\n    float inputs:height = 1\n"
           "    color3f inputs:color = (0.3, 0.6, 1)\n"
           "    double3 xformOp:translate = (1.5, 2.5, -3)\n"
           "    double xformOp:rotateX = -90\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
           "def DistantLight \"Sun\"\n{\n"
           "    float inputs:intensity = 1.5\n    float inputs:angle = 2\n"
           "    bool inputs:enableColorTemperature = 1\n    float inputs:colorTemperature = 9000\n"
           "    double xformOp:rotateX = -50\n    double xformOp:rotateY = 25\n"
           "    uniform token[] xformOpOrder = [\"xformOp:rotateY\", \"xformOp:rotateX\"]\n}\n"
           "def DomeLight \"Sky\"\n{\n"
           "    float inputs:intensity = 0.6\n"
           "    asset inputs:texture:file = @" << sky.string() << "@\n}\n"
        << kCamera
        << "def Scope \"Materials\"\n{\n"
           "    def Material \"Coated\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Coated/Preview.outputs:surface>\n"
           "        def Shader \"Preview\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor = (0.7, 0.35, 0.15)\n"
           "            float inputs:metallic = 0.2\n"
           "            float inputs:roughness = 0.35\n"
           "            float inputs:clearcoat = 0.8\n"
           "            float inputs:clearcoatRoughness = 0.05\n"
           "            token outputs:surface\n        }\n    }\n"
           "    def Material \"Standard\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Standard/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_standard_surface_surfaceshader\"\n"
           "            color3f inputs:base_color = (0.2, 0.5, 0.8)\n"
           "            float inputs:specular_roughness = 0.3\n"
           "            float inputs:coat = 0.6\n"
           "            color3f inputs:coat_color = (1, 0.9, 0.7)\n"
           "            float inputs:sheen = 0.4\n"
           "            color3f inputs:sheen_color = (0.9, 0.2, 0.4)\n"
           "            token outputs:out\n        }\n    }\n}\n";
    return path;
}

/// The edited cloud alone, unlit: the splat rasteriser's own colours.
fs::path writeSplatStage(const fs::path& dir, const fs::path& cloud) {
    const fs::path path = dir / "splats.usda";
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Xform \"Group\" (\n    prepend apiSchemas = [\"AtheneaSplatEditAPI\"]\n)\n{\n"
        << kEdit
        << "    def \"Cloud\" ( references = @" << cloud.string() << "@</World/Splats> )\n    {\n"
           "        double3 xformOp:translate = (0, 0, -0.5)\n"
           "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n"
        << kCamera;
    return path;
}

struct Frame {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;
};

Frame renderFrame(const fs::path& stage, const std::string& technique, const std::string& working = {},
                  bool chooseLights = true) {
    auto renderer = usd::StageRenderer::open(stage);
    if (!renderer) FAIL(renderer.error().toString());
    if (!working.empty()) {
        (*renderer)->setWorkingSpace(working);
    }
    (*renderer)->setChooseLights(chooseLights);
    (*renderer)->setLightSamples(2);
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathBounces(3);
    (*renderer)->setDenoise(false);
    auto image = (*renderer)->render("/Camera", 0.0, 128, 96, technique);
    if (!image) FAIL(image.error().toString());
    return Frame{image->width, image->height, std::move(image->rgba)};
}

gpu::Buffer upload(gpu::Device& device, const std::vector<float>& rgba) {
    gpu::BufferDesc desc;
    desc.bytes = rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto buffer = gpu::Buffer::create(device, desc, rgba.data());
    REQUIRE(buffer);
    return std::move(*buffer);
}

/// `frame` against the reference named `name`: every word the same, which
/// compareHdr and compareImages also say as a maximum of zero.
void matchReference(test::Gpu& gpu, const std::string& name, const Frame& frame) {
    if (const char* write = std::getenv("ATHENEA_WRITE_GOLDENS"); write != nullptr && *write != '\0') {
        fs::create_directories(write);
        REQUIRE(io::writeExr(fs::path(write) / (name + ".exr"), frame.width, frame.height, frame.rgba, {},
                             /*half=*/false));
        std::printf("  wrote %s.exr to %s\n", name.c_str(), write);
        return;
    }
    const fs::path file = fs::path(ATHENEA_TEST_DATA_DIR) / "working_space" / (name + ".exr");
    if (!fs::exists(file)) {
        SKIP("no reference " + file.string() + " (ATHENEA_WRITE_GOLDENS writes it)");
    }
    auto reference = io::readExr(file);
    if (!reference) FAIL(reference.error().toString());
    REQUIRE(reference->width == frame.width);
    REQUIRE(reference->height == frame.height);
    REQUIRE(reference->rgba.size() == frame.rgba.size());

    const gpu::Buffer a = upload(*gpu.device, frame.rgba);
    const gpu::Buffer b = upload(*gpu.device, reference->rgba);
    auto words = render::countDifferent(*gpu.library, a, b, static_cast<uint32_t>(frame.rgba.size()));
    auto hdr = render::compareHdr(*gpu.library, a, b, frame.width, frame.height);
    auto ldr = render::compareImages(*gpu.library, a, b, frame.width, frame.height);
    REQUIRE(words);
    REQUIRE(hdr);
    REQUIRE(ldr);
    std::printf("  %s: %llu of %zu words differ; compareHdr max %.3g, p99 %.3g; compareImages max %u, p99 %u\n",
                name.c_str(), static_cast<unsigned long long>(*words), frame.rgba.size(), hdr->maxRelative,
                hdr->p99Relative, ldr->max, ldr->p99);
    CHECK(*words == 0);
    CHECK(hdr->maxRelative == 0.0);
    CHECK(ldr->max == 0);
}

/// A frame that is not the same twice in one process cannot be compared
/// with another build's: said apart, so a failure below is not mistaken for
/// a changed pixel.
Frame renderTwice(test::Gpu& gpu, const fs::path& stage, const std::string& technique) {
    Frame first = renderFrame(stage, technique);
    const Frame second = renderFrame(stage, technique);
    REQUIRE(first.rgba.size() == second.rgba.size());
    const gpu::Buffer a = upload(*gpu.device, first.rgba);
    const gpu::Buffer b = upload(*gpu.device, second.rgba);
    auto words = render::countDifferent(*gpu.library, a, b, static_cast<uint32_t>(first.rgba.size()));
    REQUIRE(words);
    INFO(stage.filename().string() << " by " << technique << " differs from itself in " << *words << " words");
    REQUIRE(*words == 0);
    return first;
}

struct Scene {
    fs::path lit;
    fs::path splats;
};

Scene writeScenes(test::Gpu& gpu) {
    const fs::path dir = scratchDir();
    const fs::path cloud = dir / "cloud.usdc";
    fs::remove(cloud);
    REQUIRE(usd::writeParticleFieldStage(*gpu.library, fixedCloud(3000), cloud, {.addCamera = false}));
    const fs::path sky = writeSky(dir);
    return Scene{writeLitStage(dir, sky, cloud), writeSplatStage(dir, cloud)};
}

}   // namespace

TEST_CASE("lin_rec709_scene as the working space draws the raster's frame as it did before the record",
          "[usd][gpu][working_space][mesh]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const Scene scene = writeScenes(*gpu);
    matchReference(*gpu, "lit_raster", renderTwice(*gpu, scene.lit, "raster"));
}

TEST_CASE("lin_rec709_scene as the working space path traces the frame it did before the record",
          "[usd][gpu][working_space][mesh][path]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const Scene scene = writeScenes(*gpu);
    matchReference(*gpu, "lit_rt", renderTwice(*gpu, scene.lit, "rt"));
}

TEST_CASE("lin_rec709_scene as the working space draws a capture's splats as it did before the record",
          "[usd][gpu][working_space][splat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const Scene scene = writeScenes(*gpu);
    matchReference(*gpu, "splats_raster", renderTwice(*gpu, scene.splats, "raster"));
    const gpu::Caps& caps = gpu->device->caps();
    if (caps.rayQuery && caps.accelerationStructure) {
        matchReference(*gpu, "splats_rt", renderTwice(*gpu, scene.splats, "rt"));
    }
}

// PHASES 2B, 2C AND 3: A WORKING SPACE THAT IS NOT REC.709, AND COLOURS
// AUTHORED IN ANOTHER SPACE THAN THE ONE RENDERED IN.

TEST_CASE("lin_rec709_scene asked for by any of its names draws the frames the default draws, word for word",
          "[usd][gpu][working_space][mesh]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const Scene scene = writeScenes(*gpu);
    for (const char* name : {"lin_rec709_scene", "lin_rec709", "Linear Rec.709 (sRGB)"}) {
        INFO(name);
        matchReference(*gpu, "lit_raster", renderFrame(scene.lit, "raster", name));
        matchReference(*gpu, "splats_raster", renderFrame(scene.splats, "raster", name));
    }
}

namespace {

/// What phase 3 converts, authored in ACEScg -- a UsdPreviewSurface's
/// diffuse colour, a MaterialX standard_surface's base colour, a rect
/// light's colour, a mesh's displayColor -- or (`literal`) the same colours
/// converted to linear Rec.709 beforehand: AP1 to Rec.709 with a Bradford
/// adaptation from D60 to D65 (colour-science's matrix), to seven digits.
fs::path writeAuthoredStage(const fs::path& dir, bool literal) {
    const fs::path path = dir / (literal ? "authored_rec709.usda" : "authored_acescg.usda");
    const auto colour = [&](const char* type, const char* name, const char* acescg, const char* rec709) {
        return std::string("    ") + type + " " + name + " = " + (literal ? rec709 : acescg) +
               (literal ? "\n" : " (\n        colorSpace = \"acescg\"\n    )\n");
    };
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n"
           "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Preview>\n}\n"
           "def Sphere \"Ball\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    double radius = 0.6\n"
           "    double3 xformOp:translate = (-1.1, 0.6, -2.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
           "    rel material:binding = </Materials/Standard>\n}\n"
           "def Cube \"Box\"\n{\n"
           "    double size = 0.8\n"
           "    double3 xformOp:translate = (1.2, 0.4, -3)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
        << colour("color3f[]", "primvars:displayColor", "[(0.3, 0.7, 0.2)]", "[(0.0596090, 0.7573767, 0.1331152)]")
        << "}\n"
           "def RectLight \"Panel\"\n{\n"
           "    float inputs:intensity = 4\n    float inputs:width = 2\n    float inputs:height = 1\n"
        << colour("color3f", "inputs:color", "(0.3, 0.6, 1)", "(0.0551812, 0.6348576, 1.0683899)")
        << "    double3 xformOp:translate = (1.5, 2.5, -3)\n"
           "    double xformOp:rotateX = -90\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
           "def SphereLight \"Bulb\"\n{\n"
           "    float inputs:intensity = 10\n    float inputs:radius = 0.3\n"
           "    double3 xformOp:translate = (-1.5, 2.2, -1.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
        << kCamera
        << "def Scope \"Materials\"\n{\n"
           "    def Material \"Preview\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Preview/Shader.outputs:surface>\n"
           "        def Shader \"Shader\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n        "
        << colour("color3f", "inputs:diffuseColor", "(0.8, 0.6, 0.2)", "(0.9743137, 0.5781680, 0.1340104)")
        << "            float inputs:roughness = 0.4\n"
           "            token outputs:surface\n        }\n    }\n"
           "    def Material \"Standard\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Standard/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_standard_surface_surfaceshader\"\n        "
        << colour("color3f", "inputs:base_color", "(0.2, 0.5, 0.8)", "(-0.0364930, 0.5359124, 0.8530927)")
        << "            float inputs:specular_roughness = 0.3\n"
           "            token outputs:out\n        }\n    }\n}\n";
    return path;
}

/// A grey stage: achromatic surfaces, coloured lights and an 8-bit sRGB
/// texture whose colour nobody said, and a linear cloud in lin_rec709_scene.
/// A grey times a colour commutes with a 3x3, so drawn in ACEScg and brought
/// to Rec.709 it is the stage drawn in Rec.709, but for rounding -- and the
/// halves a converted cloud keeps.
fs::path writeGreyStage(const fs::path& dir, const fs::path& texture, const fs::path& cloud) {
    const fs::path path = dir / "grey.usda";
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n"
           "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
           "    texCoord2f[] primvars:st = [(0, 0), (4, 0), (4, 4), (0, 4)] ( interpolation = \"vertex\" )\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Textured>\n}\n"
           "def Sphere \"Ball\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    double radius = 0.6\n"
           "    double3 xformOp:translate = (-1.1, 0.6, -2.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
           "    rel material:binding = </Materials/Grey>\n}\n"
           "def \"Cloud\" ( references = @" << cloud.string() << "@</World/Splats> )\n{\n"
           "    double3 xformOp:translate = (0.6, 0.3, -2.2)\n"
           "    double3 xformOp:scale = (0.5, 0.5, 0.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n}\n"
           "def RectLight \"Panel\"\n{\n"
           "    float inputs:intensity = 4\n    float inputs:width = 2\n    float inputs:height = 1\n"
           "    color3f inputs:color = (0.3, 0.6, 1)\n"
           "    double3 xformOp:translate = (1.5, 2.5, -3)\n"
           "    double xformOp:rotateX = -90\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
           "def SphereLight \"Bulb\"\n{\n"
           "    float inputs:intensity = 10\n    float inputs:radius = 0.3\n"
           "    color3f inputs:color = (1, 0.6, 0.3)\n"
           "    double3 xformOp:translate = (-1.5, 2.2, -1.5)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
        << kCamera
        << "def Scope \"Materials\"\n{\n"
           "    def Material \"Grey\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Grey/Shader.outputs:surface>\n"
           "        def Shader \"Shader\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor = (0.5, 0.5, 0.5)\n"
           "            float inputs:roughness = 0.4\n"
           "            token outputs:surface\n        }\n    }\n"
           "    def Material \"Textured\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Textured/Shader.outputs:surface>\n"
           "        def Shader \"Shader\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor.connect = </Materials/Textured/Image.outputs:rgb>\n"
           "            float inputs:roughness = 0.6\n"
           "            token outputs:surface\n        }\n"
           "        def Shader \"Reader\"\n        {\n"
           "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
           "            string inputs:varname = \"st\"\n"
           "            float2 outputs:result\n        }\n"
           "        def Shader \"Image\"\n        {\n"
           "            uniform token info:id = \"UsdUVTexture\"\n"
           "            asset inputs:file = @" << texture.string() << "@\n"
           "            float2 inputs:st.connect = </Materials/Textured/Reader.outputs:result>\n"
           "            float3 outputs:rgb\n        }\n    }\n}\n";
    return path;
}

/// `frame`'s pixels from `space` into linear Rec.709, on the device.
gpu::Buffer toRec709(test::Gpu& gpu, const Frame& frame, const std::string& space) {
    gpu::Buffer pixels = upload(*gpu.device, frame.rgba);
    auto compiler = colour::ColourCompiler::create(*gpu.library);
    if (!compiler) FAIL(compiler.error().toString());
    gpu::CommandBatch batch(*gpu.device);
    colour::ColourWordsRun run;
    run.count = frame.width * frame.height;
    run.stride = 4;
    auto converted = (*compiler)->convertWords(batch, pixels, space, run);
    if (!converted) FAIL(converted.error().toString());
    REQUIRE(*converted);
    REQUIRE(batch.submit(true));
    return pixels;
}

}   // namespace

/// Phase 3: colours authored in ACEScg -- a material's constants, a light's
/// colour, a displayColor -- drawn in Rec.709 are the same colours converted
/// beforehand and authored in Rec.709. The literals are the reference's
/// matrix to seven digits, and OCIO's adaptation is its own: the bound is
/// what those two leave, far below the 30 percent a colour taken as the
/// wrong space is off by.
TEST_CASE("colours authored in ACEScg draw as the same colours converted beforehand",
          "[usd][gpu][working_space][mesh][phase3]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    const fs::path dir = scratchDir();
    const Frame authored = renderFrame(writeAuthoredStage(dir, false), "raster", {}, false);
    const Frame literal = renderFrame(writeAuthoredStage(dir, true), "raster", {}, false);
    const gpu::Buffer a = upload(*gpu->device, authored.rgba);
    const gpu::Buffer b = upload(*gpu->device, literal.rgba);
    auto hdr = render::compareHdr(*gpu->library, a, b, authored.width, authored.height);
    REQUIRE(hdr);
    std::printf("  authored in ACEScg against converted beforehand: compareHdr max %.3g, p99 %.3g\n",
                hdr->maxRelative, hdr->p99Relative);
    CHECK(hdr->p99Relative <= 1e-3);
    CHECK(hdr->maxRelative <= 1e-2);
}

/// Phase 2b/2c: ACEScg as the working space. A grey stage drawn in it, its
/// frame brought to Rec.709 on the device, is the stage drawn in Rec.709:
/// a grey times a colour commutes with a 3x3. What is left is rounding, the
/// sRGB texture's matrix after the filter, and the halves the linear cloud's
/// converted coefficients are stored in (2^-11 relative).
TEST_CASE("a grey stage drawn in ACEScg and brought to Rec.709 is the stage drawn in Rec.709",
          "[usd][gpu][working_space][mesh][splat][acescg]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    const fs::path dir = scratchDir();
    const fs::path cloud = dir / "linear_cloud.usdc";
    fs::remove(cloud);
    io::RawSplats raw = fixedCloud(2000);
    raw.linear = true;   // light in lin_rec709_scene, as a bake writes it
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, cloud, {.addCamera = false}));
    const fs::path stage = writeGreyStage(dir, writeSky(dir), cloud);
    const Frame rec709 = renderFrame(stage, "raster", "lin_rec709_scene", false);
    const Frame acescg = renderFrame(stage, "raster", "acescg", false);
    const gpu::Buffer a = upload(*gpu->device, rec709.rgba);
    const gpu::Buffer b = toRec709(*gpu, acescg, "acescg");
    const gpu::Buffer untouched = upload(*gpu->device, acescg.rgba);
    auto hdr = render::compareHdr(*gpu->library, b, a, rec709.width, rec709.height);
    auto apart = render::compareHdr(*gpu->library, untouched, a, rec709.width, rec709.height);
    REQUIRE(hdr);
    REQUIRE(apart);
    std::printf("  ACEScg brought to Rec.709 against Rec.709: max %.3g, p99 %.3g; not brought: p99 %.3g\n",
                hdr->maxRelative, hdr->p99Relative, apart->p99Relative);
    CHECK(hdr->p99Relative <= 1e-3);
    CHECK(hdr->maxRelative <= 2e-2);
    CHECK(apart->p99Relative > 1e-2);   // the frames are in two spaces: the conversion is what joins them
}

/// The trap Blender sets: its stages carry ColorSpaceAPI lin_rec709_scene at
/// the root, while its splats are sRGB captures. A cloud without
/// `athenea:splat:linear` is a capture whatever an ancestor says, so in
/// ACEScg as in Rec.709 Blender's points draw as the PLY they came from.
TEST_CASE("Blender's captured splats stay captures in ACEScg, whatever the root's ColorSpaceAPI says",
          "[usd][gpu][working_space][splat][blender]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    const fs::path data = fs::path(ATHENEA_TEST_DATA_DIR) / "splats";
    auto raw = io::readSplatPly(data / "sh3.ply");
    REQUIRE(raw);
    const fs::path dir = scratchDir();
    usd::ExportOptions options;
    options.addCamera = false;
    options.upAxis = 'z';
    const fs::path field = dir / "blender_field.usdc";
    fs::remove(field);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, *raw, field, options));
    const auto stage = [&](const std::string& name, const std::string& cloud) {
        const fs::path path = dir / name;
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
            << "def \"Cloud\" ( prepend references = " << cloud << " )\n{\n}\n"
            << "def Camera \"Camera\"\n{\n"
               "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (1.2, -6.5, 1.5)\n    double3 xformOp:rotateXYZ = (78, 0, 10)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n}\n";
        return path;
    };
    const fs::path fromBlender = stage("ws_blender_points.usda", "@" + (data / "sh3_blender.usda").string() + "@");
    const fs::path fromPly = stage("ws_blender_ply.usda", "@" + field.string() + "@</World/Splats>");
    const Frame points = renderFrame(fromBlender, "raster", "acescg");
    const Frame ply = renderFrame(fromPly, "raster", "acescg");
    const gpu::Buffer a = upload(*gpu->device, points.rgba);
    const gpu::Buffer b = upload(*gpu->device, ply.rgba);
    auto same = render::compareImages(*gpu->library, a, b, points.width, points.height);
    REQUIRE(same);
    std::printf("  in ACEScg, Blender's points against the PLY's field: p99 %u, max %u\n", same->p99, same->max);
    CHECK(same->p99 == 0);
    CHECK(same->max <= 1);
}
