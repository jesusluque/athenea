// Copyright (c) 2026 jesus luque.
//
// VOLUMES THROUGH HYDRA (M9), past the first medium: the grid cache held to
// its budget over an animated sequence, instanced volumes, and light
// linking in a medium. Fixtures are written by
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

// INSTANCED VOLUMES: a PointInstancer whose prototype is a Volume, two
// instances side by side, draws the frame two Volumes authored where the
// instancer puts them draw -- one record an instance, the grid laid out
// once -- and not the frame without them.
TEST_CASE("a PointInstancer of a volume draws its instances where two authored volumes stand",
          "[usd][gpu][volume][instancing]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("instanced_box.vdb");
    const io::VdbBox box{{0, 0, 0}, {16, 16, 16}, 0.8F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    const auto volume = [&](const std::string& name, const std::string& at, const std::string& translate) {
        return "def Volume \"" + name + "\"\n{\n" +
               (translate.empty() ? std::string()
                                  : "    double3 xformOp:translate = " + translate +
                                        "\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n") +
               "    rel field:density = <" + at + "/density>\n"
               "    float primvars:athenea:densityScale = 1\n"
               "    color3f primvars:athenea:albedo = (0, 0, 0)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath = @" + vdb.string() + "@\n"
               "        token fieldName = \"density\"\n    }\n}\n";
    };
    const fs::path authored = scratch("volumes_authored.usda");
    {
        std::ofstream out(authored);
        out << kPlaneAndSun << volume("Left", "/Left", "(-1.7, -0.8, -3.8)")
            << volume("Right", "/Right", "(0.1, -0.8, -3.8)");
    }
    const fs::path instanced = scratch("volumes_instanced.usda");
    {
        std::ofstream out(instanced);
        out << kPlaneAndSun
            << "def PointInstancer \"Fogs\"\n{\n"
               "    point3f[] positions = [(-1.7, -0.8, -3.8), (0.1, -0.8, -3.8)]\n"
               "    int[] protoIndices = [0, 0]\n"
               "    rel prototypes = </Fogs/Prototypes/Fog>\n"
               "    def Scope \"Prototypes\"\n    {\n"
            << volume("Fog", "/Fogs/Prototypes/Fog", "") << "    }\n}\n";
    }
    const fs::path empty = scratch("volumes_none.usda");
    {
        std::ofstream out(empty);
        out << kPlaneAndSun;
    }
    const uint32_t w = 161, h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        (*renderer)->setPathSamples(256);
        (*renderer)->setPathBounces(0);
        (*renderer)->setPathTotal(256);
        auto image = (*renderer)->render(frontCamera(), 1.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        return upload(*gpu, *image);
    };
    gpu::Buffer byInstancer = frame(instanced);
    gpu::Buffer byPrims = frame(authored);
    gpu::Buffer without = frame(empty);
    auto same = render::compareImages(*gpu->library, byInstancer, byPrims, w, h);
    auto differs = render::compareImages(*gpu->library, byInstancer, without, w, h);
    REQUIRE(same);
    REQUIRE(differs);
    std::printf("  instanced against authored: p99 %u max %u, %llu pixels over 2; against no volume: %llu over 2\n",
                same->p99, same->max, static_cast<unsigned long long>(same->over2),
                static_cast<unsigned long long>(differs->over2));
    CHECK(same->p99 <= 1);
    CHECK(same->over2 < 40);
    CHECK(differs->over2 > 2000);
}

// LIGHT LINKING IN A MEDIUM: a scattering volume alone, lit by one sun.
// Linked so that its collection leaves the volume out, the sun lights none
// of the medium and the frame is black; linked so that it includes the
// volume, it lights it as an unlinked sun does.
TEST_CASE("a light whose link leaves a volume out lights none of its medium, and one that includes it lights it",
          "[usd][gpu][volume][lights][linking]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("linked_box.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    // link: "" unlinked, "excludes" or "includes" the volume.
    const auto stage = [&](const std::string& name, const std::string& link) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def DistantLight \"Sun\"" << (link.empty() ? "" : " (\n    prepend apiSchemas = [\"CollectionAPI:lightLink\"]\n)")
            << "\n{\n    float inputs:intensity = 3\n    float inputs:angle = 0\n";
        if (link == "excludes") {
            out << "    uniform bool collection:lightLink:includeRoot = 1\n"
                   "    rel collection:lightLink:excludes = </Fog>\n";
        } else if (link == "includes") {
            out << "    uniform bool collection:lightLink:includeRoot = 0\n"
                   "    rel collection:lightLink:includes = </Fog>\n";
        }
        out << "}\n"
               "def Volume \"Fog\"\n{\n"
               "    double3 xformOp:translate = (-1.6, -1.6, -5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    rel field:density = </Fog/density>\n"
               "    float primvars:athenea:densityScale = 1\n"
               "    color3f primvars:athenea:albedo = (1, 1, 1)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath = @" << vdb.string() << "@\n"
               "        token fieldName = \"density\"\n    }\n}\n";
        return path;
    };
    const uint32_t w = 81, h = 61;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setPathTotal(64);
        auto image = (*renderer)->render(frontCamera(), 1.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        // Opacity aside: a medium's collision is opaque whatever lights it,
        // so the colour alone is what the link decides.
        std::vector<float> rgb = image->rgba;
        for (size_t i = 3; i < rgb.size(); i += 4) rgb[i] = 1.0F;
        usd::StageImage opaque = *image;
        opaque.rgba = std::move(rgb);
        return upload(*gpu, opaque);
    };
    gpu::Buffer unlinked = frame(stage("linked_none.usda", ""));
    gpu::Buffer excluded = frame(stage("linked_excludes.usda", "excludes"));
    gpu::Buffer included = frame(stage("linked_includes.usda", "includes"));
    std::vector<float> zeros(size_t{w} * h * 4, 0.0F);
    for (size_t i = 3; i < zeros.size(); i += 4) zeros[i] = 1.0F;
    gpu::BufferDesc desc;
    desc.bytes = zeros.size() * sizeof(float);
    desc.elementBytes = 16;
    auto black = gpu::Buffer::create(*gpu->device, desc, zeros.data());
    REQUIRE(black);
    auto lit = render::compareImages(*gpu->library, unlinked, *black, w, h);
    auto dark = render::compareImages(*gpu->library, excluded, *black, w, h);
    auto same = render::compareImages(*gpu->library, included, unlinked, w, h);
    REQUIRE(lit);
    REQUIRE(dark);
    REQUIRE(same);
    std::printf("  unlinked sun: %llu pixels over 2 against black; excluded: %llu (max %u); included against "
                "unlinked: p99 %u, %llu over 2\n",
                static_cast<unsigned long long>(lit->over2), static_cast<unsigned long long>(dark->over2), dark->max,
                same->p99, static_cast<unsigned long long>(same->over2));
    CHECK(lit->over2 > 1000);
    CHECK(dark->max == 0);
    CHECK(same->p99 <= 1);
    CHECK(same->over2 < 20);
}
