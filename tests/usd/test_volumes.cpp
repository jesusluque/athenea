// Copyright (c) 2026 jesus luque.
//
// VOLUMES THROUGH HYDRA (M9), past the first medium: the grid cache held to
// its budget over an animated sequence. Fixtures are written by
// io::writeVdbBoxes; frames are compared on the device, and what the host
// reads back are counters.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
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

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volumes";
    fs::create_directories(dir);
    return dir / name;
}

/// A sun-lit plane at z = -4 the camera at the origin looks at down -z; the
/// sun casts no shadow, so a volume dims only the camera's chord.
const char* kPlaneAndSun =
    "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 1\n    endTimeCode = 8\n)\n"
    "def Mesh \"Plane\"\n{\n"
    "    int[] faceVertexCounts = [4]\n"
    "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
    "    point3f[] points = [(-4, -4, -4), (4, -4, -4), (4, 4, -4), (-4, 4, -4)]\n"
    "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
    "    uniform token subdivisionScheme = \"none\"\n}\n"
    "def DistantLight \"Sun\"\n{\n"
    "    float inputs:intensity = 3\n    float inputs:angle = 0\n"
    "    bool inputs:shadow:enable = 0\n}\n";

render::Camera frontCamera() {
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    return camera;
}

gpu::Buffer upload(test::Gpu& gpu, const usd::StageImage& image) {
    gpu::BufferDesc desc;
    desc.bytes = image.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image.rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

}   // namespace

// THE GRID CACHE (athenea:volumeCacheMB): a Volume whose field's filePath
// is time-sampled, a file a frame, eight frames, under a budget that holds
// two of its grids. Each frame reads its grid; the grids it no longer draws
// leave the cache least recently used first, so the bytes held stay under
// the budget however long the sequence; and a frame whose grid was evicted
// reads it again and draws what it drew the first time.
TEST_CASE("an animated volume's grid cache stays within its budget, and a frame read again draws as it did",
          "[usd][gpu][volume][cache]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    // 64^3 voxels of 0.05: 512 leaves, a little over a megabyte a grid.
    constexpr int kFrames = 8;
    std::vector<fs::path> files;
    for (int f = 1; f <= kFrames; ++f) {
        files.push_back(scratch("sequence." + std::to_string(f) + ".vdb"));
        const io::VdbBox box{{0, 0, 0}, {64, 64, 64}, 0.25F + 0.05F * float(f)};
        REQUIRE(io::writeVdbBoxes(files.back(), "density", 0.05, std::span<const io::VdbBox>(&box, 1)));
    }
    const fs::path path = scratch("sequence.usda");
    {
        std::ofstream out(path);
        out << kPlaneAndSun
            << "def Volume \"Smoke\"\n{\n"
               "    double3 xformOp:translate = (-1.6, -1.6, -3.8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    rel field:density = </Smoke/density>\n"
               "    float primvars:athenea:densityScale = 1\n"
               "    color3f primvars:athenea:albedo = (0, 0, 0)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath.timeSamples = {\n";
        for (int f = 1; f <= kFrames; ++f) {
            out << "            " << f << ": @" << files[static_cast<size_t>(f - 1)].string() << "@,\n";
        }
        out << "        }\n        token fieldName = \"density\"\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setAntialias(false);
    (*renderer)->setPathSamples(64);
    (*renderer)->setPathBounces(0);
    (*renderer)->setPathTotal(64);
    // Two grids' worth and a little: the third evicts the first.
    (*renderer)->setVolumeCacheMB(3.0);
    const uint32_t w = 64, h = 48;
    const render::Camera camera = frontCamera();
    std::optional<gpu::Buffer> first;
    uint64_t largest = 0;
    usd::StageRenderer::VolumeCacheCounts counts;
    for (int f = 1; f <= kFrames; ++f) {
        auto image = (*renderer)->render(camera, double(f), w, h, "rt");
        if (!image) FAIL(image.error().toString());
        if (f == 1) {
            first = upload(*gpu, *image);
        }
        counts = (*renderer)->volumeCacheCounts();
        largest = std::max(largest, counts.bytes);
        std::printf("  frame %d: %llu grids, %.2f MB held (%.2f MB on the device), %llu read, %llu evicted\n", f,
                    static_cast<unsigned long long>(counts.grids), double(counts.bytes) / 1048576.0,
                    double(counts.deviceBytes) / 1048576.0, static_cast<unsigned long long>(counts.reads),
                    static_cast<unsigned long long>(counts.evictions));
    }
    CHECK(counts.reads == kFrames);
    CHECK(counts.grids == 2);
    CHECK(counts.evictions == kFrames - 2);
    CHECK(largest <= uint64_t{3} * 1048576);
    // Frame 1's grid left long ago: read again, and the same frame.
    auto again = (*renderer)->render(camera, 1.0, w, h, "rt");
    if (!again) FAIL(again.error().toString());
    const usd::StageRenderer::VolumeCacheCounts after = (*renderer)->volumeCacheCounts();
    CHECK(after.reads == kFrames + 1);
    gpu::Buffer second = upload(*gpu, *again);
    auto same = render::compareImages(*gpu->library, *first, second, w, h);
    REQUIRE(same);
    std::printf("  frame 1 read again: p99 %u max %u, %llu pixels over 2\n", same->p99, same->max,
                static_cast<unsigned long long>(same->over2));
    CHECK(same->p99 <= 1);
    CHECK(same->over2 < 10);
}
