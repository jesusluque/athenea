// Copyright (c) 2026 jesus luque.
//
// HOW FAR A RAY LEAVES ITS SURFACE (docs/decisions.md, "A ray's margin is the
// frame's to say"): a ray that crosses nothing starts off its surface by a
// share of how far the position is from the origin -- a thousandth, which at
// twenty metres in centimetres is two centimetres along the normal and two
// along the ray. At a concave join nearer than that the ray starts past the
// wall it should have met and goes on to the sky: Intel's Sponza drew a
// bright line wherever a wall met its floor. Checked through Hydra in a
// closed box two thousand units from the origin under a dome: nothing lights
// its inside, and with the default margin the joins are lit all the same.
#include "../gpu/GpuTest.h"

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

constexpr uint32_t kWidth = 96;
constexpr uint32_t kHeight = 64;

/// A box of six separate walls, a hundred units a side, centred `far` units
/// along x, a dome all round it and a camera inside looking at a corner's joins.
fs::path stage(float far) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "ray_margin";
    fs::create_directories(dir);
    const fs::path path = dir / "closed_box.usda";
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    // Each wall: the axis it is across, and which side of the box.
    for (int wall = 0; wall < 6; ++wall) {
        const int   axis = wall / 2;
        const float side = (wall % 2) != 0 ? 50.0F : -50.0F;
        float       corner[4][3];
        const float u[4] = {-50.0F, 50.0F, 50.0F, -50.0F};
        const float v[4] = {-50.0F, -50.0F, 50.0F, 50.0F};
        for (int c = 0; c < 4; ++c) {
            corner[c][axis] = side;
            corner[c][(axis + 1) % 3] = u[c];
            corner[c][(axis + 2) % 3] = v[c];
            corner[c][0] += far;
        }
        out << "def Mesh \"Wall" << wall << "\"\n{\n    uniform bool doubleSided = 1\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n    point3f[] points = [";
        for (int c = 0; c < 4; ++c) {
            out << (c != 0 ? ", " : "") << "(" << corner[c][0] << ", " << corner[c][1] << ", " << corner[c][2] << ")";
        }
        out << "]\n}\n";
    }
    out << "def Camera \"Camera\"\n{\n    float focalLength = 12\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
           "    float2 clippingRange = (0.1, 10000)\n"
           "    double3 xformOp:translate = ("
        << far << ", 0, 30)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    return path;
}

render::ImageStats inside(test::Gpu* gpu, const fs::path& path, float margin) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    if (margin > 0.0F) (*renderer)->setRayMargin(margin);
    (*renderer)->setPathSamples(64);
    (*renderer)->setPathTotal(64);
    (*renderer)->setPathBounces(2);
    auto image = (*renderer)->render("/Camera", 0.0, kWidth, kHeight, "rt");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(made);
    auto stats = render::imageStats(*gpu->library, *made, kWidth, kHeight, 0, 0, kWidth, kHeight);
    REQUIRE(stats);
    return *stats;
}

}   // namespace

TEST_CASE("a closed box far from the origin is dark inside once its rays leave by less",
          "[usd][gpu][mesh][path][margin]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!(caps.rasterization && caps.rayQuery && caps.accelerationStructure)) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path box = stage(2000.0F);
    const render::ImageStats byDefault = inside(gpu, box, 0.0F);
    const render::ImageStats byLess = inside(gpu, box, 2.0e-5F);
    std::printf("  a closed box 2000 units out, under a dome of 1: its inside reads %.5f on average (the brightest "
                "pixel %.3f) with rays leaving by a thousandth, and %.6f (%.4f) by 2e-5\n",
                byDefault.mean[1], byDefault.max[1], byLess.mean[1], byLess.max[1]);
    CHECK(byDefault.mean[1] > 1.0e-3);                       // the joins leak: what Sponza's lines were
    // 34 times less. What is left is the pixels on a join itself, whose camera
    // ray meets the edge two walls share.
    CHECK(byLess.mean[1] < 0.05 * byDefault.mean[1]);
}
