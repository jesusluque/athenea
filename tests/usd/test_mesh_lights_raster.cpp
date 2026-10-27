// Copyright (c) 2026 jesus luque.
//
// MESHES LIT BY MESH LIGHTS IN THE RASTER (technique::MaterialShading): a
// strip of light and a MeshLightAPI lamp over a Lambert floor and a glossy
// one, a box between them casting its shadow (tests/data/mesh_light_strip.usda),
// against the path tracer's direct light; a MeshLightAPI quad against a
// RectLight of its size; and a frame drawn twice, against itself. Frames are
// compared on the device.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cmath>
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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "mesh_lights_raster";
    fs::create_directories(dir);
    return dir / name;
}

struct Frame {
    gpu::Buffer rgba;
    uint32_t    width = 0;
    uint32_t    height = 0;
};

Frame frameOf(test::Gpu& gpu, const fs::path& path, uint32_t w, uint32_t h, const char* technique,
              const std::function<void(usd::StageRenderer&)>& settings) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // The direct light alone, so the raster and the path tracer estimate one
    // integral.
    (*renderer)->setPathBounces(0);
    (*renderer)->setPathMis(true);
    // Every ray through the pixel's middle, as the raster's one sample: the
    // strip's edges, seen directly, are then the same pixels in both.
    (*renderer)->setAntialias(false);
    if (settings) {
        settings(**renderer);
    }
    auto image = (*renderer)->render("/World/Camera", 0.0, w, h, technique);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(made);
    return {std::move(*made), w, h};
}

const auto traced = [](uint32_t paths) {
    return [paths](usd::StageRenderer& r) {
        r.setPathSamples(std::min<uint32_t>(paths, 256));
        r.setPathTotal(paths);
    };
};

/// a | b | |a - b| x gain, written as a PNG where ATHENEA_MESH_LIGHT_IMAGES asks.
void picture(test::Gpu& gpu, const Frame& a, const Frame& b, const std::string& name, float gain) {
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
            cursor["exposure"].setData(1.0F);
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
    spec.data = pixels->data();
    CHECK(image->Write(spec));
    std::printf("  wrote %s\n", file.string().c_str());
}

bool rays(test::Gpu& gpu) {
    const gpu::Caps& caps = gpu.device->caps();
    return caps.rasterization && caps.rayQuery && caps.accelerationStructure;
}

}   // namespace

// SAMPLED: the raster draws the path tracer's next event estimate at each
// pixel -- a cluster through the tree, a triangle, a point, a shadow ray --
// so it converges to the path tracer's direct light, at the rate a Monte
// Carlo estimate does: four times the samples, a quarter of the error.
TEST_CASE("the raster's sampled mesh lights converge to the path tracer's direct light",
          "[usd][gpu][mesh][meshlights][raster]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!rays(*gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_strip.usda";
    const uint32_t w = 384, h = 216;
    const Frame reference = frameOf(*gpu, path, w, h, "rt", traced(8192));
    const auto sampled = [](uint32_t n) { return [n](usd::StageRenderer& r) { r.setMeshLightSamples(n); }; };
    const Frame four = frameOf(*gpu, path, w, h, "raster", sampled(4));
    const Frame sixtyFour = frameOf(*gpu, path, w, h, "raster", sampled(64));
    auto err4 = render::compareHdr(*gpu->library, four.rgba, reference.rgba, w, h);
    auto err64 = render::compareHdr(*gpu->library, sixtyFour.rgba, reference.rgba, w, h);
    REQUIRE(err4);
    REQUIRE(err64);
    std::printf("  sampled raster against 8192 paths of direct light: 4 samples relMSE %.3e, 64 samples %.3e "
                "(%.1f times less; p99 relative %.3e)\n",
                err4->relMse, err64->relMse, err4->relMse / std::max(err64->relMse, 1e-30), err64->p99Relative);
    picture(*gpu, sixtyFour, reference, "b1_strip_raster64_pt", 4.0F);
    // And the closed form (athenea:meshLightSamples 0, the default), against
    // the same reference: what the on-air path draws.
    const Frame closed = frameOf(*gpu, path, w, h, "raster", sampled(0));
    auto errClosed = render::compareHdr(*gpu->library, closed.rgba, reference.rgba, w, h);
    REQUIRE(errClosed);
    std::printf("  closed form against the same: relMSE %.3e (p99 relative %.3e)\n", errClosed->relMse,
                errClosed->p99Relative);
    picture(*gpu, closed, reference, "b2_strip_closed_pt", 4.0F);
    // The rectangles stand in for the strip and the lamp exactly here (each
    // a plane); what is left is the LTC fit on the glossy floor and the
    // shadow's ratio at the box.
    CHECK(errClosed->relMse < 2.0e-3);
    // Sixteen times the samples: about sixteen times less error, unless the
    // two estimate different things -- a bias would stay as the noise went.
    CHECK(err64->relMse < err4->relMse / 8.0);
    CHECK(err64->relMse < 5.0e-3);
}

// A MeshLightAPI quad is a RectLight of its size and intensity to the raster
// as to the path tracer: each against the path tracer's frame of the other.
TEST_CASE("a MeshLightAPI quad lights the raster's floor as a rect light does",
          "[usd][gpu][mesh][meshlights][raster]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!rays(*gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const auto stage = [](const char* name, bool mesh) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\ndef Xform \"World\"\n{\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n";
        if (mesh) {
            out << "def Mesh \"Panel\" (\n    prepend apiSchemas = [\"MeshLightAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-0.5, 2.5, -3.5), (0.5, 2.5, -3.5), (0.5, 2.5, -2.5), (-0.5, 2.5, -2.5)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    float inputs:intensity = 3\n}\n";
        } else {
            out << "def RectLight \"Panel\"\n{\n"
                   "    float inputs:intensity = 3\n    float inputs:width = 1\n    float inputs:height = 1\n"
                   "    double3 xformOp:translate = (0, 2.5, -3)\n"
                   "    double xformOp:rotateX = -90\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.2, 2)\n"
               "    double xformOp:rotateX = -30\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n}\n";
        return path;
    };
    const fs::path meshPath = stage("raster_meshlight_quad.usda", true);
    const fs::path rectPath = stage("raster_rect.usda", false);
    const uint32_t w = 128, h = 96;
    const Frame rectTraced = frameOf(*gpu, rectPath, w, h, "rt", traced(4096));
    const Frame meshRaster = frameOf(*gpu, meshPath, w, h, "raster",
                                     [](usd::StageRenderer& r) { r.setMeshLightSamples(64); });
    const Frame rectRaster = frameOf(*gpu, rectPath, w, h, "raster",
                                     [](usd::StageRenderer& r) { r.setLightSamples(64); });
    auto mesh = render::compareHdr(*gpu->library, meshRaster.rgba, rectTraced.rgba, w, h);
    auto rect = render::compareHdr(*gpu->library, rectRaster.rgba, rectTraced.rgba, w, h);
    REQUIRE(mesh);
    REQUIRE(rect);
    std::printf("  against the rect light path traced: the MeshLightAPI quad in the raster relMSE %.3e, the rect "
                "light in the raster %.3e\n",
                mesh->relMse, rect->relMse);
    picture(*gpu, meshRaster, rectRaster, "b_meshlightapi_vs_rect_raster", 8.0F);
    // In closed form: the quad's rectangle is the rect light's, integrated.
    const Frame meshClosed = frameOf(*gpu, meshPath, w, h, "raster",
                                     [](usd::StageRenderer& r) { r.setMeshLightSamples(0); });
    auto closed = render::compareHdr(*gpu->library, meshClosed.rgba, rectTraced.rgba, w, h);
    REQUIRE(closed);
    std::printf("  the quad in closed form against the rect light path traced: relMSE %.3e\n", closed->relMse);
    picture(*gpu, meshClosed, rectTraced, "b2_meshlightapi_closed_vs_rect_pt", 8.0F);
    CHECK(closed->relMse < 1.0e-4);
    // The rect light's own raster sets the bar: its 64 samples of the area,
    // the quad's 64 of its two triangles.
    CHECK(mesh->relMse < 2.0 * rect->relMse + 1.0e-4);
}

// THE SAME FRAME TWICE IS THE SAME FRAME: the samples are the pixel's own,
// not the frame's, and nothing in the mesh lights moves between two draws of
// a still stage.
TEST_CASE("a still stage lit by mesh lights draws the same raster frame twice",
          "[usd][gpu][mesh][meshlights][raster]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!rays(*gpu)) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = fs::path(ATHENEA_TEST_DATA_DIR) / "mesh_light_strip.usda";
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160, h = 90;
    std::vector<gpu::Buffer> frames;
    for (int k = 0; k < 2; ++k) {
        auto image = (*renderer)->render("/World/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        auto made = gpu::Buffer::fromSpan<float>(*gpu->device, image->rgba, "frame");
        REQUIRE(made);
        frames.push_back(std::move(*made));
    }
    auto differ = render::countDifferent(*gpu->library, frames[0], frames[1], w * h * 4);
    REQUIRE(differ);
    std::printf("  two raster frames of a still stage: %llu words differ\n", static_cast<unsigned long long>(*differ));
    CHECK(*differ == 0);
}
