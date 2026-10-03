// Copyright (c) 2026 jesus luque.
//
// A CLOUD'S SHADOW ON A FLOOR, THROUGH HYDRA, AS A FRAME SEES IT.
//
// What `[shadowmap]` in the technique tests measures with probes, measured
// here in pictures: a floor far enough from the map that a pixel covers
// several of its texels, and a skinned cloud whose wing flaps over the floor
// for nine frames. Both are compared on the device (`compareImages`,
// `compareFlicker`); nothing is read back but the metrics.
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

fs::path shadowScratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "usd";
    fs::create_directories(dir);
    return dir / name;
}

/// A frame's pixels on the device, as a kernel reads them.
gpu::Buffer onDevice(gpu::Device& device, const usd::StageImage& image) {
    gpu::BufferDesc desc;
    desc.bytes = image.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto buffer = gpu::Buffer::create(device, desc, image.rgba.data());
    REQUIRE(buffer);
    return std::move(*buffer);
}

/// A ground quad, grey, wide enough for every shadow here.
std::string ground() {
    return "def Mesh \"Ground\"\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-2, -3, 0), (4, -3, 0), (4, 3, 0), (-2, 3, 0)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n";
}

}   // namespace

// A FLOOR FAR FROM THE MAP READS THE MEAN OF WHAT IS OVER A PIXEL.
//
// Sixteen thousand gaussians smaller than a texel, scattered over half a unit
// two units up, and a sun at 45 degrees: the floor under them is a speckle a
// texel across, and the camera over it sees four texels a pixel. Read at one
// place a pixel (level zero, even filtered over four texels) that speckle
// aliases; read at the level of the pixel's footprint in the transmittance
// chain, it is the mean of the transmittances under the pixel -- what the
// same frame rendered four times larger and boxed down to this size holds.
TEST_CASE("a floor four texels a pixel from the map reads what a frame four times larger averages to",
          "[usd][gpu][mesh][splat][shadowmap][mips]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n" << ground();
    // The cloud: positions from a fixed sequence, so the speckle is the same
    // in both frames.
    uint64_t state = 2026;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>((state >> 40) & 0xFFFFFF) / 16777216.0;
    };
    const int count = 16384;
    out << "def ParticleField3DGaussianSplat \"Cloud\"\n{\n    point3f[] positions = [";
    for (int k = 0; k < count; ++k) {
        out << (k ? ", " : "") << "(" << next() * 0.5 - 0.25 << ", " << next() * 0.5 - 0.25 << ", 2)";
    }
    out << "]\n    quatf[] orientations = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
    out << "]\n    float3[] scales = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(0.0008, 0.0008, 0.0008)";
    out << "]\n    float[] opacities = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "0.8";
    out << "]\n    int radiance:sphericalHarmonicsDegree = 0\n"
           "    float3[] radiance:sphericalHarmonicsCoefficients = [";
    for (int k = 0; k < count; ++k) out << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
    // The camera one unit over the shadow, which the sun throws two units to
    // the right, looking straight down at 0.6 units of floor: the cloud is
    // behind it.
    out << "]\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 40.96\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
           "    float2 clippingRange = (0.1, 100)\n"
           "    double3 xformOp:translate = (2, 0, 1)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
           "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n"
           "    float inputs:angle = 0\n"
           "    float3 xformOp:rotateXYZ = (0, -45, 0)\n    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n";
    const fs::path path = shadowScratch("cloud_shadow_mips.usda");
    {
        std::ofstream file(path);
        file << out.str();
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setAntialias(false);
    (*renderer)->setCloudShadows(true);
    (*renderer)->setCloudShadowResolution(1024);

    const uint32_t size = 128;
    auto small = (*renderer)->render("/Camera", 0.0, size, size, "raster");
    if (!small) FAIL(small.error().toString());
    auto large = (*renderer)->render("/Camera", 0.0, size * 4, size * 4, "raster");
    if (!large) FAIL(large.error().toString());
    const gpu::Buffer frame = onDevice(*gpu->device, *small);
    const gpu::Buffer wide = onDevice(*gpu->device, *large);

    // The large frame boxed down to the small one's size, on the device.
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{size} * size * 16;
    desc.elementBytes = 16;
    desc.label = "test.boxed";
    auto boxed = gpu::Buffer::create(*gpu->device, desc);
    REQUIRE(boxed);
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "athenea/test/box_reduce", "boxReduce");
    if (!reduce) FAIL(reduce.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        reduce->dispatch(batch, {size, size, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["source"].setBinding(wide.rhi());
            cursor["reduced"].setBinding(boxed->rhi());
            cursor["boxParams"]["width"].setData(size);
            cursor["boxParams"]["height"].setData(size);
            cursor["boxParams"]["factor"].setData(uint32_t{4});
        });
        REQUIRE(batch.submit(true));
    }
    auto diff = render::compareImages(*gpu->library, frame, *boxed, size, size);
    REQUIRE(diff);
    // And that there is a shadow to alias at all: the frame is not the
    // unshadowed floor.
    (*renderer)->setCloudShadows(false);
    auto bare = (*renderer)->render("/Camera", 0.0, size, size, "raster");
    if (!bare) FAIL(bare.error().toString());
    auto shadowed = render::compareImages(*gpu->library, frame, onDevice(*gpu->device, *bare), size, size);
    REQUIRE(shadowed);
    std::printf("  a floor four texels a pixel from the map, against four times the frame boxed down: p99 %u, max %u, "
                "%llu pixels beyond 2 codes; against no shadow at all: p99 %u\n",
                diff->p99, diff->max, static_cast<unsigned long long>(diff->over2), shadowed->p99);
    CHECK(shadowed->p99 > 20);
    CHECK(diff->p99 <= 8);
}

namespace {

/// A rest transform as UsdSkel writes it: identity, translated by (x, y, z)
/// in the last row.
std::string translated(double x, double y, double z) {
    std::ostringstream m;
    m << "( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (" << x << ", " << y << ", " << z << ", 1) )";
    return m.str();
}

/// A body and a wing hinged to it, 0.6 over the ground: the skeleton both
/// the cloud and the mesh below are bound to. The wing turns about the x
/// axis through (0, 0.1, 0.6), 30 degrees up and down over eight frames.
std::string rig() {
    std::ostringstream out;
    out << "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
           "        uniform token[] joints = [\"body\", \"body/wing\"]\n"
           "        uniform matrix4d[] bindTransforms = ["
        << translated(0, 0, 0) << ", " << translated(0, 0.1, 0.6)
        << "]\n        uniform matrix4d[] restTransforms = [" << translated(0, 0, 0) << ", "
        << translated(0, 0.1, 0.6)
        << "]\n        rel skel:animationSource = </Bird/Anim>\n    }\n"
           "    def SkelAnimation \"Anim\"\n    {\n"
           "        uniform token[] joints = [\"body/wing\"]\n"
           "        float3[] translations = [(0, 0.1, 0.6)]\n"
           "        half3[] scales = [(1, 1, 1)]\n"
           "        quatf[] rotations.timeSamples = {\n";
    for (int t = 0; t <= 8; ++t) {
        const double angle = 0.5 * (30.0 * 3.14159265358979 / 180.0) * std::sin(2.0 * 3.14159265358979 * t / 8.0);
        out << "            " << t << ": [(" << std::cos(angle) << ", " << std::sin(angle) << ", 0, 0)],\n";
    }
    out << "        }\n    }\n";
    return out.str();
}

/// The bound prim's influences: every point of the body on joint 0, every
/// point of the wing on joint 1.
std::string influences(const std::vector<int>& joints) {
    std::ostringstream out;
    out << "        rel skel:skeleton = </Bird/Skel>\n        int[] primvars:skel:jointIndices = [";
    for (size_t k = 0; k < joints.size(); ++k) out << (k ? ", " : "") << joints[k];
    out << "] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
           "        float[] primvars:skel:jointWeights = [";
    for (size_t k = 0; k < joints.size(); ++k) out << (k ? ", " : "") << "1";
    out << "] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
           "        matrix4d primvars:skel:geomBindTransform = "
        << translated(0, 0, 0) << "\n";
    return out.str();
}

/// The stage around a bound prim: the ground, the sun at 40 degrees of
/// elevation, nine frames, and a camera over the shadow that does not see
/// the bird.
std::string flightStage(const std::string& bound) {
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n    startTimeCode = 0\n    endTimeCode = 8\n)\n"
        << ground() << "def SkelRoot \"Bird\"\n{\n" << rig() << bound << "}\n"
        << "def Camera \"Camera\"\n{\n    float focalLength = 38.2\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 100)\n"
           "    double3 xformOp:translate = (0.715, 0.2, 1.4)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
           "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n"
           "    float inputs:angle = 0\n"
           "    float3 xformOp:rotateXYZ = (0, -50, 0)\n    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n";
    return out.str();
}

}   // namespace

// THE FLICKER OF A FLAPPING WING'S SHADOW, AGAINST THE MESH OF THE SAME RIG.
//
// Flicker is the temporal second difference: frame t against the mean of
// frames t - 1 and t + 1 (`render::compareFlicker`). A smooth motion is
// nearly linear over three frames, so its second difference is small;
// flicker is not. The wing of a two-joint rig flaps over a floor for nine
// frames with the camera still, once as a cloud and once as the mesh the
// same skeleton carries, and the cloud's shadow may flicker no more than the
// mesh's by ten per cent -- the bar proposal 005 set. The mesh's shadow is a
// shadow ray's, so the case needs a device that traces.
TEST_CASE("a flapping cloud's shadow on the floor flickers no more than its mesh's",
          "[usd][gpu][skinning][shadowmap][flicker]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("the mesh's shadow is traced: no ray tracing on this device");
    }
    // The body, 0.3 x 0.2, and the wing beside it, 0.3 x 0.4, both flat at
    // 0.6: a gaussian every centimetre for the cloud, two quads for the mesh.
    std::vector<std::array<double, 3>> points;
    std::vector<int> joints;
    for (int i = 0; i <= 30; ++i) {
        for (int j = 0; j <= 60; ++j) {
            const double y = -0.1 + 0.01 * j;
            points.push_back({-0.15 + 0.01 * i, y, 0.6});
            joints.push_back(y > 0.1 ? 1 : 0);
        }
    }
    std::ostringstream cloud;
    cloud << "    def ParticleField3DGaussianSplat \"Cloud\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n"
             "    )\n    {\n        point3f[] positions = [";
    for (size_t k = 0; k < points.size(); ++k) {
        cloud << (k ? ", " : "") << "(" << points[k][0] << ", " << points[k][1] << ", " << points[k][2] << ")";
    }
    cloud << "]\n        quatf[] orientations = [";
    for (size_t k = 0; k < points.size(); ++k) cloud << (k ? ", " : "") << "(1, 0, 0, 0)";
    cloud << "]\n        float3[] scales = [";
    for (size_t k = 0; k < points.size(); ++k) cloud << (k ? ", " : "") << "(0.008, 0.008, 0.002)";
    cloud << "]\n        float[] opacities = [";
    for (size_t k = 0; k < points.size(); ++k) cloud << (k ? ", " : "") << "0.9";
    cloud << "]\n        int radiance:sphericalHarmonicsDegree = 0\n"
             "        float3[] radiance:sphericalHarmonicsCoefficients = [";
    for (size_t k = 0; k < points.size(); ++k) cloud << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
    cloud << "]\n" << influences(joints) << "    }\n";

    std::ostringstream mesh;
    mesh << "    def Mesh \"Wing\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
            "        int[] faceVertexCounts = [4, 4]\n"
            "        int[] faceVertexIndices = [0, 1, 2, 3, 4, 5, 6, 7]\n"
            "        point3f[] points = [(-0.155, -0.105, 0.6), (0.155, -0.105, 0.6), (0.155, 0.1, 0.6), "
            "(-0.155, 0.1, 0.6), (-0.155, 0.1, 0.6), (0.155, 0.1, 0.6), (0.155, 0.505, 0.6), (-0.155, 0.505, 0.6)]\n"
            "        uniform token subdivisionScheme = \"none\"\n"
            "        uniform bool doubleSided = 1\n"
            "        color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)] ( interpolation = \"constant\" )\n"
         << influences({0, 0, 0, 0, 1, 1, 1, 1}) << "    }\n";

    const uint32_t w = 160, h = 120;
    // Nine frames of a stage, and the worst of its seven second differences.
    const auto flicker = [&](const char* name, const std::string& bound) {
        const fs::path path = shadowScratch(std::string("flicker_") + name + ".usda");
        {
            std::ofstream file(path);
            file << flightStage(bound);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        std::vector<gpu::Buffer> frames;
        for (int t = 0; t <= 8; ++t) {
            auto image = (*renderer)->render("/Camera", double(t), w, h, "raster");
            if (!image) FAIL(image.error().toString());
            frames.push_back(onDevice(*gpu->device, *image));
        }
        double relMse = 0.0;
        uint32_t p99 = 0;
        for (size_t t = 1; t + 1 < frames.size(); ++t) {
            auto second = render::compareFlicker(*gpu->library, frames[t - 1], frames[t], frames[t + 1], w, h);
            if (!second) FAIL(second.error().toString());
            relMse = std::max(relMse, second->hdr.relMse);
            p99 = std::max(p99, second->codes.p99);
        }
        // And that the wing's shadow is in the picture: the first frame and
        // the fifth (wing up against wing level) differ.
        auto moved = render::compareImages(*gpu->library, frames[0], frames[2], w, h);
        REQUIRE(moved);
        std::printf("  %-5s flicker: worst relMSE %.3e, worst p99 %u codes; the wing moves %llu pixels by more "
                    "than 2\n",
                    name, relMse, p99, static_cast<unsigned long long>(moved->over2));
        CHECK(moved->over2 > 100);
        return std::make_pair(relMse, p99);
    };
    const auto [cloudMse, cloudP99] = flicker("cloud", cloud.str());
    const auto [meshMse, meshP99] = flicker("mesh", mesh.str());
    CHECK(cloudMse <= 1.1 * meshMse + 1.0e-6);
    CHECK(cloudP99 <= meshP99 + meshP99 / 10 + 1);
}
