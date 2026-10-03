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

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

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
