// Copyright (c) 2026 jesus luque.
//
// USD as the scene: a cloud written to a stage and rendered through the Hydra
// delegate draws what the same cloud draws directly; points prims too; and
// the delegate loads as a plugin the way a USD application finds it.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <memory>
#include <optional>
#include <set>

#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/base/plug/registry.h>
#include <pxr/imaging/hio/image.h>
#include <pxr/imaging/hd/pluginRenderDelegateUniqueHandle.h>
#include <pxr/imaging/hd/rendererPluginRegistry.h>
#include <pxr/usd/usd/primDefinition.h>
#include <pxr/usd/usd/schemaRegistry.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>

#include "athenea/geom/Mesh.h"
#include "athenea/usd/MeshStage.h"
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/vec3h.h>
#include <cstdio>

#include "athenea/technique/Visibility.h"
#include "athenea/core/Hash.h"
#include "athenea/core/Platform.h"
#include "athenea/io/Exr.h"
#include "athenea/io/Vdb.h"
#include "athenea/io/Readers.h"
#include "athenea/lod/Lod.h"
#include "athenea/lod/Athc.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/PrimData.h"
#include "athenea/technique/Denoiser.h"
#include "athenea/technique/PathTracer.h"
#include "athenea/scene/ThinWall.h"
#include "athenea/colour/ColourCompiler.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

// Before any case builds USD's schema registry: registered later, a plugin's
// schemas are not seen by a registry that already exists.
[[maybe_unused]] const bool kPluginsRegistered = [] {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "usd";
    fs::create_directories(dir);
    return dir / name;
}

io::RawSplats cloud(uint32_t count) {
    io::RawSplats raw;
    raw.source = "usd synthetic";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 23;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
    uint64_t state = 99;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t i = 0; i < count; ++i) {
        for (int k = 0; k < 23; ++k) {
            raw.records.push_back(0.0F);
        }
        float* r = raw.records.data() + raw.records.size() - 23;
        r[0] = next() * 4 - 2; r[1] = next() * 3 - 1.5F; r[2] = next() * 4 - 2;
        r[3] = next() * 6 - 3;
        r[4] = std::log(0.02F + next() * 0.2F); r[5] = std::log(0.02F + next() * 0.2F);
        r[6] = std::log(0.01F + next() * 0.1F);
        r[7] = next() - 0.5F; r[8] = next() - 0.5F; r[9] = next() - 0.5F; r[10] = next() - 0.5F;
        for (int k = 11; k < 23; ++k) {
            r[k] = next() * 2 - 1;
        }
        raw.count += 1;
    }
    return raw;
}

}   // namespace

TEST_CASE("a cloud written to USD and drawn through Hydra matches the cloud drawn directly",
          "[usd][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path path = scratch("cloud.usdc");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, {.addCamera = false}));

    // A shot, in a layer of its own over the exported one.
    const fs::path shot = scratch("shot.usda");
    {
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def \"Cloud\" ( references = @./cloud.usdc@</World/Splats> )\n    {\n"
               "        double3 xformOp:rotateXYZ = (0, 35, 0)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) {
        FAIL(renderer.error().toString());
    }
    auto image = (*renderer)->render("/World/Shot", 0.0, 240, 180);
    if (!image) {
        FAIL(image.error().toString());
    }

    // The same cloud, the same camera, straight to the rasteriser.
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(raw);
    REQUIRE(splats);
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.4, 0.2, 7.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::RenderTargets direct;
    const std::vector<render::SplatInstance> instances{{&*splats, aofx::xform::rotationY(35.0)}};
    REQUIRE(rasterizer->render(camera, instances, settings, direct));

    auto hydra = gpu::Buffer::fromSpan<float>(*gpu->device, image->rgba, "hydra");
    REQUIRE(hydra);
    gpu::BufferDesc desc;
    desc.bytes = hydra->bytes();
    desc.elementBytes = 16;
    // Reinterpret the float buffer as float4 for the comparison kernel.
    auto hydraRgba = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(hydraRgba);
    auto diff = render::compareImages(*gpu->library, *hydraRgba, direct.colour, 240, 180);
    REQUIRE(diff);
    CHECK(diff->p99 <= 1);
    CHECK(diff->max <= 2);

    // And ray traced: the delegate's athenea:technique against the ray tracer.
    auto traced = (*renderer)->render("/World/Shot", 0.0, 240, 180, "rt");
    if (!traced) {
        FAIL(traced.error().toString());
    }
    auto tracer = render::GaussianRayTracer::create(*gpu->library);
    REQUIRE(tracer);
    render::RenderTargets directTraced;
    REQUIRE(tracer->render(camera, instances, settings, directTraced));
    auto tracedRgba = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(tracedRgba);
    auto tracedDiff = render::compareImages(*gpu->library, *tracedRgba, directTraced.colour, 240, 180);
    REQUIRE(tracedDiff);
    CHECK(tracedDiff->p99 <= 1);
    CHECK(tracedDiff->max <= 2);
}

// A CLOUD GOES OUT AND COMES BACK AS IT WAS. What `athenea decimate` rests on: a
// cloud on the device read back into records (CloudLoader::records), written
// as a ParticleField, and that file read back into records
// (usd::readParticleFieldRecords) -- each uploaded again draws the image the
// first did.
TEST_CASE("a cloud unpacked from the device and read back from its ParticleField draws as it did",
          "[usd][gpu][decimate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(3000);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto first = loader->upload(raw);
    REQUIRE(first);
    auto unpacked = loader->records(*first);
    if (!unpacked) FAIL(unpacked.error().toString());
    CHECK(unpacked->count == first->count);
    auto second = loader->upload(*unpacked);
    REQUIRE(second);
    const fs::path path = scratch("round_trip.usdc");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, *unpacked, path, {.addCamera = false}));
    std::string where;
    bool moved = true;
    auto read = usd::readParticleFieldRecords(path, {}, &where, &moved);
    if (!read) FAIL(read.error().toString());
    CHECK(where == "/World/Splats");
    CHECK_FALSE(moved);
    CHECK(read->count == first->count);
    auto third = loader->upload(*read);
    REQUIRE(third);
    CHECK_FALSE(usd::readParticleFieldRecords(path, "/World/Nothing"));

    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.4, 0.2, 7.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const auto draw = [&](const scene::GpuSplats& splats) {
        render::RenderTargets out;
        REQUIRE(rasterizer->render(camera, std::vector<render::SplatInstance>{{&splats, render::Mat4::identity()}},
                                   settings, out));
        return out;
    };
    const render::RenderTargets a = draw(*first);
    const render::RenderTargets b = draw(*second);
    const render::RenderTargets c = draw(*third);
    auto unpackedDiff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
    auto fileDiff = render::compareImages(*gpu->library, a.colour, c.colour, settings.width, settings.height);
    REQUIRE(unpackedDiff);
    REQUIRE(fileDiff);
    std::printf("  unpacked: p99 %u, max %u; through the file: p99 %u, max %u\n", unpackedDiff->p99,
                unpackedDiff->max, fileDiff->p99, fileDiff->max);
    CHECK(unpackedDiff->p99 <= 1);
    CHECK(unpackedDiff->max <= 2);
    CHECK(fileDiff->p99 <= 1);
    CHECK(fileDiff->max <= 2);
}

// THE STAGE AS IT WAS, WITH FEWER GAUSSIANS. A floor of one colour written
// as a ParticleField, then given what a converted cloud carries: a float and
// an id a gaussian, a rig of four joints, a constant flag, a camera beside it
// and an asset named relative to its layer. Decimated into another folder,
// the stage keeps the flag, the camera and the manifest-like constants as
// they were, its relative asset still finds its file, and every array a
// gaussian long is exactly as long as the gaussians now are.
TEST_CASE("a decimated stage keeps everything, every array a gaussian long as long as the gaussians",
          "[usd][gpu][decimate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr int kSide = 48;
    io::RawSplats raw;
    raw.source = "decimate floor";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    const float spacing = 2.0F / kSide;
    for (int y = 0; y < kSide; ++y) {
        for (int x = 0; x < kSide; ++x) {
            const float r[14] = {-1.0F + (x + 0.5F) * spacing, -1.0F + (y + 0.5F) * spacing, 0.0F, 0.95F,
                                 spacing, spacing, spacing * 0.1F, 1.0F, 0.0F, 0.0F, 0.0F, 0.3F, 0.2F, 0.1F};
            raw.records.insert(raw.records.end(), r, r + 14);
            raw.count += 1;
        }
    }
    const fs::path folder = scratch("decimate_in");
    fs::create_directories(folder);
    const fs::path input = folder / "floor.usda";
    fs::remove(input);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, input, {.addCamera = true}));
    {
        UsdStageRefPtr stage = UsdStage::Open(input.string());
        REQUIRE(stage);
        const UsdPrim field = stage->GetPrimAtPath(SdfPath("/World/Splats"));
        REQUIRE(field);
        UsdGeomPrimvarsAPI primvars(field);
        VtFloatArray floats;
        VtIntArray ids;
        VtIntArray joints;
        VtFloatArray weights;
        for (uint32_t k = 0; k < raw.count; ++k) {
            floats.push_back(0.5F);
            ids.push_back(k % kSide < kSide / 2 ? 7 : 9);
            for (int j = 0; j < 4; ++j) {
                joints.push_back(j == 0 ? (k % kSide < kSide / 2 ? 1 : 2) : 0);
                weights.push_back(j == 0 ? 1.0F : 0.0F);
            }
        }
        primvars.CreatePrimvar(TfToken("athenea:splat:roughness"), SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex)
            .Set(floats);
        primvars.CreatePrimvar(TfToken("athenea:splat:cryptoObject"), SdfValueTypeNames->IntArray, UsdGeomTokens->vertex)
            .Set(ids);
        primvars.CreatePrimvar(TfToken("skel:jointIndices"), SdfValueTypeNames->IntArray, UsdGeomTokens->vertex, 4)
            .Set(joints);
        primvars.CreatePrimvar(TfToken("skel:jointWeights"), SdfValueTypeNames->FloatArray, UsdGeomTokens->vertex, 4)
            .Set(weights);
        primvars.CreatePrimvar(TfToken("athenea:splat:relight"), SdfValueTypeNames->Int, UsdGeomTokens->constant)
            .Set(1);
        field.CreateAttribute(TfToken("athenea:test:texture"), SdfValueTypeNames->Asset)
            .Set(SdfAssetPath("./floor_texture.png"));
        // In centimetres, which the copy must keep: the gaussians' own stage
        // on the way is written in metres.
        UsdGeomSetStageMetersPerUnit(stage, 0.01);
        REQUIRE(stage->GetRootLayer()->Save());
    }
    { std::ofstream(folder / "floor_texture.png") << "not an image, only a file that is there"; }
    const fs::path outFolder = scratch("decimate_out");
    fs::create_directories(outFolder);
    const fs::path output = outFolder / "floor_fewer.usda";
    fs::remove(output);
    auto stats = usd::decimateStage(*gpu->library, input, output);
    if (!stats) FAIL(stats.error().toString());
    const uint32_t kept = stats->splats + stats->merged;
    std::printf("  %u of %u kept\n", kept, stats->before);
    CHECK(stats->before == raw.count);
    CHECK(kept * 2 < raw.count);

    auto records = usd::readParticleFieldRecords(output);
    if (!records) FAIL(records.error().toString());
    CHECK(records->count == kept);
    auto arrays = usd::readGaussianArrays(output, "/World/Splats", kept);
    if (!arrays) FAIL(arrays.error().toString());
    std::set<std::string> names;
    for (const usd::GaussianArray& a : *arrays) {
        names.insert(a.name);
        CHECK(a.values[0].size() == size_t{kept} * a.width);
    }
    CHECK(names.contains("primvars:athenea:splat:roughness"));
    CHECK(names.contains("primvars:athenea:splat:cryptoObject"));
    CHECK(names.contains("primvars:skel:jointIndices"));
    CHECK(names.contains("primvars:skel:jointWeights"));
    UsdStageRefPtr stage = UsdStage::Open(output.string());
    REQUIRE(stage);
    const UsdPrim field = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(field);
    int relight = 0;
    CHECK(UsdGeomPrimvarsAPI(field).GetPrimvar(TfToken("athenea:splat:relight")).Get(&relight));
    CHECK(relight == 1);
    CHECK(stage->GetPrimAtPath(SdfPath("/World/Camera")));
    SdfAssetPath texture;
    REQUIRE(field.GetAttribute(TfToken("athenea:test:texture")).Get(&texture));
    CHECK(!texture.GetResolvedPath().empty());   // found from the other folder
    CHECK(UsdGeomGetStageMetersPerUnit(stage) == 0.01);
}

TEST_CASE("splats and points behind a mesh leave it as it is; in front of it they show", "[usd][gpu][mesh][layers]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device: the layers are composited from rasterised passes");
    }
    const io::RawSplats raw = cloud(2500);
    const fs::path clouds = scratch("layers_cloud.usdc");
    fs::remove(clouds);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, clouds, {.addCamera = false}));
    // A wall (single sided, facing the camera) at z, splats within |z| <= 2
    // and a sheet of points at z = 0, seen from z = 9.
    const auto stage = [&](const std::string& name, double wallZ, bool withSplats, bool withPoints) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-9, -9, 0), (9, -9, 0), (9, 9, 0), (-9, 9, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.3, 0.6, 0.9)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0, 0, " << wallZ << ")\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "}\n";
        if (withSplats) {
            out << "def \"Cloud\" ( references = @./layers_cloud.usdc@</World/Splats> )\n{\n}\n";
        }
        if (withPoints) {
            out << "def Points \"Sheet\"\n{\n    point3f[] points = [";
            for (int y = -4; y <= 4; ++y) {
                for (int x = -4; x <= 4; ++x) {
                    out << (x == -4 && y == -4 ? "" : ", ") << "(" << x * 0.3 << ", " << y * 0.3 << ", 0)";
                }
            }
            out << "]\n    float[] widths = [0.12] ( interpolation = \"constant\" )\n"
                   "    color3f[] primvars:displayColor = [(1, 0.5, 0.25)] ( interpolation = \"constant\" )\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (0.2, 0.1, 9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 240;
    const uint32_t h = 180;
    const auto render = [&](const fs::path& path, const char* route) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const fs::path hiddenPath = stage("layers_hidden.usda", 3.0, true, true);
    const fs::path frontWallPath = stage("layers_front_wall.usda", 3.0, false, false);
    const fs::path shownPath = stage("layers_shown.usda", -5.0, true, true);
    const fs::path backWallPath = stage("layers_back_wall.usda", -5.0, false, false);
    for (const char* route : {"raster", "rays", "bvh"}) {
        const gpu::Buffer hidden = render(hiddenPath, route);
        const gpu::Buffer frontWall = render(frontWallPath, route);
        const gpu::Buffer shown = render(shownPath, route);
        const gpu::Buffer backWall = render(backWallPath, route);
        auto behind = render::compareImages(*gpu->library, hidden, frontWall, w, h);
        auto before = render::compareImages(*gpu->library, shown, backWall, w, h);
        REQUIRE(behind);
        REQUIRE(before);
        std::printf("  %s: splats and points behind the wall change %llu pixels (max %u); in front, %llu (max %u)\n",
                    route, static_cast<unsigned long long>(behind->over2), behind->max,
                    static_cast<unsigned long long>(before->over2), before->max);
        CHECK(behind->max == 0);
        CHECK(before->over2 > uint64_t{w} * h / 10);
    }
}

TEST_CASE("a UsdGeomPoints prim draws through Hydra", "[usd][gpu][points]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("points.usda");
    fs::remove(path);
    {
        UsdStageRefPtr stage = UsdStage::CreateNew(path.string());
        auto points = UsdGeomPoints::Define(stage, SdfPath("/Points"));
        VtVec3fArray positions;
        VtVec3fArray colours;
        for (int y = -10; y <= 10; ++y) {
            for (int x = -10; x <= 10; ++x) {
                positions.push_back(GfVec3f(x * 0.1F, y * 0.1F, 0.0F));
                colours.push_back(GfVec3f(1.0F, 0.5F, 0.25F));
            }
        }
        points.CreatePointsAttr(VtValue(positions));
        points.CreateWidthsAttr(VtValue(VtFloatArray{0.08F}));
        points.SetWidthsInterpolation(UsdGeomTokens->constant);
        points.CreateDisplayColorAttr(VtValue(colours));
        auto camera = UsdGeomCamera::Define(stage, SdfPath("/Camera"));
        camera.CreateFocalLengthAttr(VtValue(35.0F));
        camera.CreateHorizontalApertureAttr(VtValue(24.576F));
        camera.CreateVerticalApertureAttr(VtValue(24.576F));
        UsdGeomXformCommonAPI(camera.GetPrim()).SetTranslate(GfVec3d(0, 0, 4));
        stage->GetRootLayer()->Save();
    }
    auto renderer = usd::StageRenderer::open(path);
    REQUIRE(renderer);
    auto image = (*renderer)->render("", 0.0, 128, 128);
    if (!image) {
        FAIL(image.error().toString());
    }
    const size_t centre = (64 * 128 + 64) * 4;
    CHECK(image->rgba[centre + 3] > 0.99F);
    CHECK(image->rgba[centre] > 0.9F);
    CHECK(image->rgba[centre + 1] == Catch::Approx(0.5F).margin(0.02F));
    // View z: the points plane sits 4 units from the camera; the corner is empty.
    CHECK(image->depth[64 * 128 + 64] == Catch::Approx(4.0F).margin(0.05F));
    CHECK(image->depth[2 * 128 + 2] == 0.0F);

    // What a host that maps Hydra's render buffers reads: converted on the
    // device, bottom row first, depth in the projection's [0, 1].
    auto colour = (*renderer)->mappedOutput("color");
    auto depth = (*renderer)->mappedOutput("depth");
    REQUIRE(colour);
    REQUIRE(depth);
    REQUIRE(colour->size() == size_t{128} * 128 * 16);
    REQUIRE(depth->size() == size_t{128} * 128 * 4);
    const auto floatAt = [](const std::vector<uint8_t>& bytes, size_t index) {
        float v = 0.0F;
        std::memcpy(&v, bytes.data() + index * 4, 4);
        return v;
    };
    CHECK(floatAt(*colour, (63 * 128 + 64) * 4 + 3) == image->rgba[(63 * 128 + 64) * 4 + 3]);
    CHECK(floatAt(*colour, (63 * 128 + 64) * 4 + 1) == image->rgba[(63 * 128 + 64) * 4 + 1]);
    const float centreDepth = floatAt(*depth, 64 * 128 + 64);
    CHECK(centreDepth > 0.0F);
    CHECK(centreDepth < 1.0F);
    CHECK(floatAt(*depth, 2 * 128 + 2) == 1.0F);   // nothing: the far plane
}

// BLENDER'S SPLATS ARE SPLATS. Blender's Gaussian-splat PointCloud reaches
// Hydra as a UsdGeomPoints whose primvars are its attributes: `radiance:base`
// (the DC coefficient and the opacity, written by the athenea_hydra add-on's
// export hook), `radiance:sh_N` one array a basis function, `rotation` and
// `scale`. The fixture is `sh3.ply` imported by Blender 5.3 and exported to
// USD with that hook; it must draw as the same PLY does through a
// ParticleField athenea wrote itself, the harmonics included.
TEST_CASE("Blender's Gaussian-splat points draw as the PLY they were imported from", "[usd][gpu][points]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path data = fs::path(ATHENEA_TEST_DATA_DIR) / "splats";
    auto raw = io::readSplatPly(data / "sh3.ply");
    REQUIRE(raw);
    usd::ExportOptions options;
    options.addCamera = false;
    options.upAxis = 'z';
    const fs::path field = scratch("blender_field.usdc");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, *raw, field, options));

    // The same camera over each: framing the 3-unit cube the cloud fills.
    const auto stage = [&](const std::string& name, const std::string& cloud, const std::string& over) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
            << "def \"Cloud\" ( prepend references = " << cloud << " )\n{\n" << over << "}\n"
            << "def Camera \"Camera\"\n{\n"
               "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (1.2, -6.5, 1.5)\n    double3 xformOp:rotateXYZ = (78, 0, 10)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n}\n";
        return path;
    };
    const std::string blender = "@" + (data / "sh3_blender.usda").string() + "@";
    const fs::path fromBlender = stage("blender_points.usda", blender, "");
    const fs::path fromPly = stage("blender_ply.usda", "@" + field.string() + "@</World/Splats>", "");
    // The control: the hook's half taken away, as Blender's own export leaves it.
    const fs::path withoutBase = stage(
        "blender_points_nobase.usda", blender,
        "    over \"sh3\"\n    {\n        over \"sh3\"\n        {\n"
        "            float4[] primvars:radiance:base = None\n        }\n    }\n");

    const uint32_t w = 240;
    const uint32_t h = 180;
    const auto render = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer points = render(fromBlender);
    const gpu::Buffer ply = render(fromPly);
    const gpu::Buffer bare = render(withoutBase);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    gpu::BufferDesc desc;
    desc.bytes = blank.size() * sizeof(float);
    desc.elementBytes = 16;
    auto empty = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(empty);
    auto same = render::compareImages(*gpu->library, points, ply, w, h);
    auto drawn = render::compareImages(*gpu->library, ply, *empty, w, h);
    auto control = render::compareImages(*gpu->library, bare, ply, w, h);
    REQUIRE(same);
    REQUIRE(drawn);
    REQUIRE(control);
    std::printf("  Blender's points against the PLY's field: p99 %u, max %u; against nothing: %llu pixels over 2; "
                "without radiance:base: p99 %u\n",
                same->p99, same->max, static_cast<unsigned long long>(drawn->over2), control->p99);
    CHECK(drawn->over2 > uint64_t{w} * h / 20);   // the cloud is in the picture
    // Measured: p99 0, max 0 -- the same floats reach the same records -- and
    // 100 for the control, which is drawn opaque and grey.
    CHECK(same->p99 == 0);
    CHECK(same->max <= 1);
    CHECK(control->p99 >= 50);
}

TEST_CASE("a UsdGeomMesh draws through Hydra where, how deep and how lit it analytically is", "[usd][gpu][mesh]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("mesh.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.4, 0.2)] ( interpolation = \"constant\" )\n"
               "}\n"
               "def Mesh \"Guide\"\n{\n"
               "    uniform token purpose = \"guide\"\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-3, -3, -2), (3, -3, -2), (3, 3, -2), (-3, 3, -2)]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 161;
    const uint32_t h = 121;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colour = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colour);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::BufferDesc wordsDesc;
    wordsDesc.bytes = 12;
    wordsDesc.elementBytes = 4;
    auto counts = gpu::Buffer::create(*gpu->device, wordsDesc);
    wordsDesc.bytes = 4;
    auto worst = gpu::Buffer::create(*gpu->device, wordsDesc);
    REQUIRE(counts);
    REQUIRE(worst);
    gpu::CommandBatch batch(*gpu->device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colour->rhi());
        cursor["depth"].setBinding(depth->rhi());
        cursor["counts"].setBinding(counts->rhi());
        cursor["worst"].setBinding(worst->rhi());
        rhi::ShaderCursor c = cursor["camera"];
        c["width"].setData(w);
        c["height"].setData(h);
        c["focalX"].setData(static_cast<float>(projection.focalX));
        c["focalY"].setData(static_cast<float>(projection.focalY));
        c["centreX"].setData(static_cast<float>(projection.centreX));
        c["centreY"].setData(static_cast<float>(projection.centreY));
        c["nearZ"].setData(0.1F);
        c["farZ"].setData(1000.0F);
        c["orthographic"].setData(uint32_t{0});
        cursor["plane"]["z"].setData(5.0F);
        cursor["plane"]["half"].setData(1.0F);
        cursor["plane"]["colourR"].setData(0.8F);
        cursor["plane"]["colourG"].setData(0.4F);
        cursor["plane"]["colourB"].setData(0.2F);
    });
    REQUIRE(batch.submit(true));
    uint32_t c[3] = {0, 0, 0};
    float depthError = 0.0F;
    REQUIRE(counts->read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(depthError), &depthError));
    std::printf("  Hydra mesh: %u pixels covered, %u coverage and %u colour mismatches, depth off by %.2e\n", c[2],
                c[0], c[1], static_cast<double>(depthError));
    // The guide-purpose mesh in front is not in the geometry collection's render tags.
    CHECK(c[2] > 800);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(depthError < 1e-3F);
}

namespace {

/// A stage's /Camera image against the analytic headlit square at z = -5
/// (visibility_check.slang's planeCheck): coverage, depth and colour mismatches.
std::array<uint32_t, 3> squareMismatches(test::Gpu& gpu, const usd::StageImage& image, std::array<float, 3> colour,
                                         float z = 5.0F, float half = 1.0F) {
    const uint32_t w = image.width;
    const uint32_t h = image.height;
    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image.rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu.device, colourDesc, image.rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu.device, image.depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu.library, "athenea/test/visibility_check", "planeCheck");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 3, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu.device, 1, "worst");
    gpu::CommandBatch batch(*gpu.device);
    check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["colour"].setBinding(colours->rhi());
        cursor["depth"].setBinding(depth->rhi());
        cursor["counts"].setBinding(counts.rhi());
        cursor["worst"].setBinding(worst.rhi());
        rhi::ShaderCursor c = cursor["camera"];
        c["width"].setData(w);
        c["height"].setData(h);
        c["focalX"].setData(static_cast<float>(projection.focalX));
        c["focalY"].setData(static_cast<float>(projection.focalY));
        c["centreX"].setData(static_cast<float>(projection.centreX));
        c["centreY"].setData(static_cast<float>(projection.centreY));
        c["nearZ"].setData(0.1F);
        c["farZ"].setData(1000.0F);
        c["orthographic"].setData(uint32_t{0});
        cursor["plane"]["z"].setData(z);
        cursor["plane"]["half"].setData(half);
        cursor["plane"]["colourR"].setData(colour[0]);
        cursor["plane"]["colourG"].setData(colour[1]);
        cursor["plane"]["colourB"].setData(colour[2]);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 3> c{};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(c), c.data()));
    return c;
}

const char* kSquareStage = R"(#usda 1.0
(
    upAxis = "Y"
)
def Mesh "Square" (
    prepend apiSchemas = ["MaterialBindingAPI"]
)
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
    uniform token subdivisionScheme = "none"
    color3f[] primvars:displayColor = [(0.1, 0.9, 0.1)] (
        interpolation = "constant"
    )
    rel material:binding = </Materials/Mat>
}
def Camera "Camera"
{
    float focalLength = 35
    float horizontalAperture = 24.576
    float verticalAperture = 18.432
    float2 clippingRange = (0.1, 1000)
}
)";

}   // namespace

TEST_CASE("materials bound in USD shade a mesh: MaterialX with a texture, and UsdPreviewSurface",
          "[usd][gpu][mesh][materials]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A texture of one colour, exactly representable in 8 bits.
    const fs::path png = scratch("orange.png");
    {
        fs::remove(png);
        std::vector<uint32_t> texels(16 * 16, 0xFF3399CCu);   // ABGR: r 0xCC, g 0x99, b 0x33
        HioImageSharedPtr image = HioImage::OpenForWriting(png.string());
        REQUIRE(image);
        HioImage::StorageSpec spec;
        spec.width = 16;
        spec.height = 16;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(image->Write(spec));
    }
    const std::array<float, 3> textureColour{0xCC / 255.0F, 0x99 / 255.0F, 0x33 / 255.0F};
    SECTION("MaterialX: oren_nayar coloured by an image") {
        const fs::path path = scratch("material_mtlx.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
                   "        def Shader \"Surface\"\n        {\n"
                   "            uniform token info:id = \"ND_surface\"\n"
                   "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
                   "            token outputs:out\n        }\n"
                   "        def Shader \"Diffuse\"\n        {\n"
                   "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
                   "            color3f inputs:color.connect = </Materials/Mat/Texture.outputs:out>\n"
                   "            token outputs:out\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"ND_image_color3\"\n"
                   "            asset inputs:file = @" << png.string() << "@ ( colorSpace = \"lin_rec709\" )\n"
                   "            color3f outputs:out\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const auto c = squareMismatches(*gpu, *image, textureColour);
        // The same material seen by rays instead of the rasteriser: shading
        // reads the visibility buffer, whichever route filled it.
        if (gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure) {
            REQUIRE((*renderer)->setMeshVisibility("rays"));
            auto traced = (*renderer)->render("/Camera", 0.0, 160, 120);
            REQUIRE((*renderer)->setMeshVisibility("raster"));
            if (!traced) FAIL(traced.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto a = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            auto b = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
            REQUIRE(a);
            REQUIRE(b);
            auto diff = render::compareImages(*gpu->library, *a, *b, 160, 120);
            REQUIRE(diff);
            std::printf("  raster against rays, textured material: p99 %u, max %u, %llu pixels beyond 2\n", diff->p99,
                        diff->max, static_cast<unsigned long long>(diff->over2));
            CHECK(diff->max == 0);
        }
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  MaterialX image material: %u covered, %u coverage and %u colour mismatches; centre %.4f %.4f "
                    "%.4f (texture %.4f %.4f %.4f)\n",
                    c[2], c[0], c[1], double(centre[0]), double(centre[1]), double(centre[2]),
                    double(textureColour[0]), double(textureColour[1]), double(textureColour[2]));
        CHECK(c[2] > 800);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
    }
    SECTION("a file's colour space reaches the decode: srgb_texture is linearised, and its absence is not") {
        // The same picture, read twice: once as the sRGB values it holds and
        // once as linear ones. What the colour space says has to survive the
        // journey a delegate takes it on -- USD attribute, scene index,
        // MaterialX document, the compiled material's slot -- and it used not
        // to: `HdMaterialNode2::parameters` is values and nothing else, so
        // every texture was read raw and an 8-bit sRGB marble came out pale.
        const auto linearOf = [](float value) {
            return value <= 0.04045F ? value / 12.92F : std::pow((value + 0.055F) / 1.055F, 2.4F);
        };
        const std::array<float, 3> linearised{linearOf(textureColour[0]), linearOf(textureColour[1]),
                                              linearOf(textureColour[2])};
        const auto shadeWith = [&](const char* space, const char* name) {
            const fs::path path = scratch(name);
            {
                std::ofstream out(path);
                out << kSquareStage
                    << "def Scope \"Materials\"\n{\n"
                       "    def Material \"Mat\"\n    {\n"
                       "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
                       "        def Shader \"Surface\"\n        {\n"
                       "            uniform token info:id = \"ND_surface\"\n"
                       "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
                       "            token outputs:out\n        }\n"
                       "        def Shader \"Diffuse\"\n        {\n"
                       "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
                       "            color3f inputs:color.connect = </Materials/Mat/Texture.outputs:out>\n"
                       "            token outputs:out\n        }\n"
                       "        def Shader \"Texture\"\n        {\n"
                       "            uniform token info:id = \"ND_image_color3\"\n"
                       "            asset inputs:file = @" << png.string() << "@ ( colorSpace = \"" << space
                    << "\" )\n"
                       "            color3f outputs:out\n        }\n    }\n}\n";
            }
            auto renderer = usd::StageRenderer::open(path);
            if (!renderer) FAIL(renderer.error().toString());
            auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
            if (!image) FAIL(image.error().toString());
            return std::move(*image);
        };
        const auto srgb = shadeWith("srgb_texture", "material_srgb.usda");
        const auto raw = shadeWith("lin_rec709", "material_raw.usda");
        const float* srgbCentre = srgb.rgba.data() + (60 * 160 + 80) * 4;
        const float* rawCentre = raw.rgba.data() + (60 * 160 + 80) * 4;
        // Two claims, and the kernel carries the one it can carry exactly:
        // every pixel of the square is the same colour as its centre (the
        // check's tolerance is 1e-5, which a colour computed on the host with
        // `pow` would not survive), and that centre is the texture's values
        // put through the sRGB curve.
        const auto wrongSrgb =
            squareMismatches(*gpu, srgb, {srgbCentre[0], srgbCentre[1], srgbCentre[2]});
        const auto wrongRaw = squareMismatches(*gpu, raw, textureColour);
        std::printf("  srgb_texture: centre %.4f %.4f %.4f (linearised %.4f %.4f %.4f); "
                    "lin_rec709: centre %.4f %.4f %.4f (as held %.4f %.4f %.4f)\n",
                    double(srgbCentre[0]), double(srgbCentre[1]), double(srgbCentre[2]), double(linearised[0]),
                    double(linearised[1]), double(linearised[2]), double(rawCentre[0]), double(rawCentre[1]),
                    double(rawCentre[2]), double(textureColour[0]), double(textureColour[1]),
                    double(textureColour[2]));
        CHECK(wrongSrgb[2] > 800);
        CHECK(wrongSrgb[0] == 0);
        CHECK(wrongSrgb[1] == 0);
        CHECK(wrongRaw[0] == 0);
        CHECK(wrongRaw[1] == 0);
        for (size_t k = 0; k < 3; ++k) {
            CHECK(srgbCentre[k] == Catch::Approx(linearised[k]).epsilon(0.005));
            CHECK(rawCentre[k] == Catch::Approx(textureColour[k]).epsilon(0.005));
        }
        // Any other space the config knows: ACEScg is brought into linear
        // Rec.709 by the function OpenColorIO compiled for it, inside the
        // decode kernel, and a data space by any of its names is read as held.
        // Before colour::ColourNames both were read raw.
        if (colour::ocioBuilt()) {
            const auto acescg = shadeWith("acescg", "material_acescg.usda");
            const float* acescgCentre = acescg.rgba.data() + (60 * 160 + 80) * 4;
            const auto wrongAcescg =
                squareMismatches(*gpu, acescg, {acescgCentre[0], acescgCentre[1], acescgCentre[2]});
            // ACES AP1 to Rec.709, D60 to D65 by Bradford: the published matrix,
            // what the colour tests check the compiled function against on the
            // device; here only the journey is in question.
            const std::array<std::array<float, 3>, 3> ap1To709{{{1.70505F, -0.62179F, -0.08326F},
                                                                {-0.13026F, 1.14080F, -0.01055F},
                                                                {-0.02400F, -0.12897F, 1.15297F}}};
            std::printf("  acescg: centre %.4f %.4f %.4f\n", double(acescgCentre[0]), double(acescgCentre[1]),
                        double(acescgCentre[2]));
            CHECK(wrongAcescg[2] > 800);
            CHECK(wrongAcescg[0] == 0);
            CHECK(wrongAcescg[1] == 0);
            for (size_t k = 0; k < 3; ++k) {
                const float want = ap1To709[k][0] * textureColour[0] + ap1To709[k][1] * textureColour[1] +
                                   ap1To709[k][2] * textureColour[2];
                CHECK(acescgCentre[k] == Catch::Approx(want).epsilon(0.005));
            }
        }
        const auto data = shadeWith("Non-Color", "material_noncolor.usda");
        const float* dataCentre = data.rgba.data() + (60 * 160 + 80) * 4;
        const auto wrongData = squareMismatches(*gpu, data, {dataCentre[0], dataCentre[1], dataCentre[2]});
        std::printf("  Non-Color: centre %.4f %.4f %.4f\n", double(dataCentre[0]), double(dataCentre[1]),
                    double(dataCentre[2]));
        CHECK(wrongData[2] > 800);
        CHECK(wrongData[0] == 0);
        CHECK(wrongData[1] == 0);
        for (size_t k = 0; k < 3; ++k) {
            CHECK(dataCentre[k] == Catch::Approx(textureColour[k]).epsilon(0.005));
        }
    }
    SECTION("UsdPreviewSurface with a UsdUVTexture read through UsdPrimvarReader") {
        const fs::path path = scratch("material_preview_texture.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Texture.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @" << png.string() << "@\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            color3f outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  UsdPreviewSurface textured centre: %.4f %.4f %.4f (texture %.4f %.4f %.4f)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]), double(textureColour[0]),
                    double(textureColour[1]), double(textureColour[2]));
        for (int k = 0; k < 3; ++k) {
            CHECK(centre[k] == Catch::Approx(textureColour[size_t(k)]).epsilon(0.01));
        }
    }
    SECTION("a texture authored relative to its layer, read from another directory, wrapping as USD spells it") {
        // The layer in a directory of its own, the image beside it under
        // textures/: "./textures/orange.png" names it only relative to the
        // layer, never to the process's working directory.
        const fs::path directory = scratch("relative_layer");
        fs::create_directories(directory / "textures");
        fs::copy_file(png, directory / "textures" / "orange.png", fs::copy_options::overwrite_existing);
        const fs::path path = directory / "material_relative.usda";
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Texture.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Texture\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @./textures/orange.png@\n"
                   "            token inputs:wrapS = \"repeat\"\n"
                   "            token inputs:wrapT = \"repeat\"\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            color3f outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        REQUIRE(fs::current_path() != directory);
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  relative texture centre: %.4f %.4f %.4f (texture %.4f %.4f %.4f)\n", double(centre[0]),
                    double(centre[1]), double(centre[2]), double(textureColour[0]), double(textureColour[1]),
                    double(textureColour[2]));
        for (int k = 0; k < 3; ++k) {
            CHECK(centre[k] == Catch::Approx(textureColour[size_t(k)]).epsilon(0.01));
        }
    }
    SECTION("UsdPreviewSurface's diffuseColor from a colour primvar through a float3 reader") {
        // The reader is a vector to MaterialX and diffuseColor a colour: the
        // material failed its declaration and the mesh fell back to its
        // displayColor, (0.1, 0.9, 0.1). The primvar read here is another
        // colour, so the fallback cannot pass for the material.
        const fs::path path = scratch("material_preview_reader3.usda");
        {
            std::ofstream out(path);
            std::string square = kSquareStage;
            const std::string binding = "    rel material:binding";
            square.insert(square.find(binding),
                          "    color3f[] primvars:tint = [(0.7, 0.3, 0.5)] (\n        interpolation = \"constant\"\n    )\n");
            out << square
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float3\"\n"
                   "            string inputs:varname = \"tint\"\n"
                   "            float3 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  a colour primvar through a float3 reader: centre %.4f %.4f %.4f (primvar 0.7 0.3 0.5)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.7F).margin(0.07F));
        CHECK(centre[1] == Catch::Approx(0.3F).margin(0.03F));
        CHECK(centre[2] == Catch::Approx(0.5F).margin(0.05F));
    }
    SECTION("a primvar authored on the material reaches the mesh bound to it") {
        // The blend shape test stage's material carries primvars:displayColor
        // itself and reads it back: the mesh has none of its own. Transferred
        // by hdsi, the square is the material's colour; untransferred, the
        // reader reads its default, black.
        const fs::path path = scratch("material_primvar_transfer.usda");
        {
            std::string square = kSquareStage;
            const std::string colour = "    color3f[] primvars:displayColor = [(0.1, 0.9, 0.1)] (\n"
                                       "        interpolation = \"constant\"\n    )\n";
            square.erase(square.find(colour), colour.size());
            std::ofstream out(path);
            out << square
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        color3f primvars:tint = (0.3, 0.6, 0.9)\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float3\"\n"
                   "            string inputs:varname = \"tint\"\n"
                   "            float3 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  a material's own primvar: centre %.4f %.4f %.4f (primvar 0.3 0.6 0.9)\n", double(centre[0]),
                    double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.3F).margin(0.03F));
        CHECK(centre[1] == Catch::Approx(0.6F).margin(0.06F));
        CHECK(centre[2] == Catch::Approx(0.9F).margin(0.09F));
    }
    SECTION("a normal map authored as USD writes it: float4 scale and bias, a colour into a vector normal") {
        // UsdUVTexture's scale and bias are float4 in USD and color4 in
        // MaterialX; UsdPreviewSurface's normal is a vector fed from the
        // texture's rgb, a colour. As authored (McUsd writes every material
        // so) the whole material failed and fell back to displayColor.
        const fs::path flat = scratch("flat_normal.png");
        {
            fs::remove(flat);
            std::vector<uint32_t> texels(16 * 16, 0xFFFF8080u);   // ABGR: (128, 128, 255), the unperturbed normal
            HioImageSharedPtr image = HioImage::OpenForWriting(flat.string());
            REQUIRE(image);
            HioImage::StorageSpec spec;
            spec.width = 16;
            spec.height = 16;
            spec.depth = 1;
            spec.format = HioFormatUNorm8Vec4;
            spec.data = texels.data();
            REQUIRE(image->Write(spec));
        }
        const fs::path path = scratch("material_preview_normalmap.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor = (0.8, 0.4, 0.2)\n"
                   "            float3 inputs:normal.connect = </Materials/Mat/Normal.outputs:rgb>\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n"
                   "        def Shader \"Normal\"\n        {\n"
                   "            uniform token info:id = \"UsdUVTexture\"\n"
                   "            asset inputs:file = @" << flat.string() << "@\n"
                   "            float4 inputs:bias = (-1, -1, -1, -1)\n"
                   "            float4 inputs:scale = (2, 2, 2, 2)\n"
                   "            token inputs:sourceColorSpace = \"raw\"\n"
                   "            float2 inputs:st.connect = </Materials/Mat/Reader.outputs:result>\n"
                   "            float3 outputs:rgb\n        }\n"
                   "        def Shader \"Reader\"\n        {\n"
                   "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                   "            string inputs:varname = \"st\"\n"
                   "            float2 outputs:result\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  USD-typed normal map: centre %.4f %.4f %.4f (diffuse 0.8 0.4 0.2; the fallback is 0.1 0.9 0.1)\n",
                    double(centre[0]), double(centre[1]), double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.8F).margin(0.08F));
        CHECK(centre[1] == Catch::Approx(0.4F).margin(0.04F));
        CHECK(centre[2] == Catch::Approx(0.2F).margin(0.02F));
    }
    SECTION("UsdPreviewSurface, rough, specular workflow without specular") {
        const fs::path path = scratch("material_preview.usda");
        {
            std::ofstream out(path);
            out << kSquareStage
                << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Mat\"\n    {\n"
                   "        token outputs:surface.connect = </Materials/Mat/Preview.outputs:surface>\n"
                   "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            color3f inputs:diffuseColor = (0.8, 0.4, 0.2)\n"
                   "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        // Not the displayColor fallback: the material's colour, less what the
        // (black) specular layer's albedo fit keeps of it.
        const float* centre = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  UsdPreviewSurface centre: %.4f %.4f %.4f\n", double(centre[0]), double(centre[1]),
                    double(centre[2]));
        CHECK(centre[0] == Catch::Approx(0.8F).margin(0.08F));
        CHECK(centre[1] == Catch::Approx(0.4F).margin(0.04F));
        CHECK(centre[2] == Catch::Approx(0.2F).margin(0.02F));
    }
}

namespace {

/// Two squares, one in front of the other, each with its own material: the
/// front one's is a UsdPreviewSurface whose opacity `opacity` is cut at
/// `threshold`, the back one's a MaterialX diffuse of a known colour. With
/// `alpha` the opacity is that texture's alpha channel instead.
std::string cutoutStage(const std::string& opacity, float threshold, const std::array<float, 3>& back,
                        const std::string& extra = {}) {
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Front\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-1, -1, -3), (1, -1, -3), (1, 1, -3), (-1, 1, -3)]\n"
           "    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (\n        interpolation = \"vertex\"\n    )\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Cut>\n}\n"
           "def Mesh \"Back\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
           "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
           "    uniform token subdivisionScheme = \"none\"\n"
           "    rel material:binding = </Materials/Back>\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n}\n"
           "def Scope \"Materials\"\n{\n"
           "    def Material \"Cut\"\n    {\n"
           "        token outputs:surface.connect = </Materials/Cut/Preview.outputs:surface>\n"
           "        def Shader \"Preview\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor = (0.2, 0.4, 0.8)\n"
           "            int inputs:useSpecularWorkflow = 1\n"
           "            color3f inputs:specularColor = (0, 0, 0)\n"
           "            float inputs:roughness = 1\n"
        << "            " << opacity << "\n"
        << "            float inputs:opacityThreshold = " << threshold << "\n"
           "            token outputs:surface\n        }\n"
        << extra
        << "    }\n"
           "    def Material \"Back\"\n    {\n"
           "        token outputs:mtlx:surface.connect = </Materials/Back/Surface.outputs:out>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"ND_surface\"\n"
           "            token inputs:bsdf.connect = </Materials/Back/Diffuse.outputs:out>\n"
           "            token outputs:out\n        }\n"
           "        def Shader \"Diffuse\"\n        {\n"
           "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        << "            color3f inputs:color = (" << back[0] << ", " << back[1] << ", " << back[2] << ")\n"
        << "            token outputs:out\n        }\n    }\n}\n";
    return out.str();
}

}   // namespace

TEST_CASE("a material's opacityThreshold cuts its samples out of visibility itself, in every route",
          "[usd][gpu][mesh][materials][cutout]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    const bool rays = caps.rayQuery && caps.accelerationStructure;
    std::vector<const char*> routes{"raster", "bvh"};
    if (rays) {
        routes.insert(routes.begin() + 1, "rays");
    }
    const std::array<float, 3> backColour{0.8F, 0.4F, 0.2F};
    const uint32_t w = 160;
    const uint32_t h = 120;

    SECTION("cut away, what is behind shows exactly as if it were alone") {
        const fs::path path = scratch("cutout_all.usda");
        {
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity = 0.2", 0.5F, backColour);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // The back square at z = -5, analytically: the front one is gone,
            // and nothing of it is left in the depth either.
            const auto c = squareMismatches(*gpu, *image, backColour);
            std::printf("  cutout by %s: %u covered, %u coverage and %u colour mismatches against the square behind\n",
                        route, c[2], c[0], c[1]);
            CHECK(c[2] > 800);
            CHECK(c[0] == 0);
            CHECK(c[1] == 0);
        }
    }

    SECTION("an opacity above the threshold is not cut") {
        const fs::path path = scratch("cutout_none.usda");
        {
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity = 0.2", 0.1F, backColour);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // The front square at z = -3 covers the back one: its coverage and
            // depth, not the colour, which is the preview surface's.
            const auto c = squareMismatches(*gpu, *image, {0.2F, 0.4F, 0.8F}, 3.0F);
            std::printf("  kept by %s: %u covered, %u coverage mismatches against the square in front (%u colour)\n",
                        route, c[2], c[0], c[1]);
            CHECK(c[2] > 2000);
            CHECK(c[0] == 0);
        }
    }

    SECTION("half a square cut by a texture's alpha, the same in every route") {
        // Alpha 0 on the left half of the texture, 1 on the right.
        const fs::path png = scratch("cutout_alpha.png");
        {
            fs::remove(png);
            std::vector<uint32_t> texels(size_t{16} * 16, 0u);
            for (size_t y = 0; y < 16; ++y) {
                for (size_t x = 0; x < 16; ++x) {
                    texels[y * 16 + x] = x < 8 ? 0x00FFFFFFu : 0xFFFFFFFFu;   // ABGR
                }
            }
            HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
            REQUIRE(made);
            HioImage::StorageSpec spec;
            spec.width = 16;
            spec.height = 16;
            spec.depth = 1;
            spec.format = HioFormatUNorm8Vec4;
            spec.data = texels.data();
            REQUIRE(made->Write(spec));
        }
        const fs::path path = scratch("cutout_texture.usda");
        {
            const std::string alpha =
                "        def Shader \"Alpha\"\n        {\n"
                "            uniform token info:id = \"UsdUVTexture\"\n"
                "            asset inputs:file = @" + png.string() + "@\n"
                "            token inputs:sourceColorSpace = \"raw\"\n"
                "            float2 inputs:st.connect = </Materials/Cut/Reader.outputs:result>\n"
                "            float outputs:a\n        }\n"
                "        def Shader \"Reader\"\n        {\n"
                "            uniform token info:id = \"UsdPrimvarReader_float2\"\n"
                "            string inputs:varname = \"st\"\n"
                "            float2 outputs:result\n        }\n";
            std::ofstream out(path);
            out << cutoutStage("float inputs:opacity.connect = </Materials/Cut/Alpha.outputs:a>", 0.5F, backColour,
                               alpha);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        std::vector<gpu::Buffer> images;
        for (const char* route : routes) {
            REQUIRE((*renderer)->setMeshVisibility(route));
            auto image = (*renderer)->render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            // Left of the square: the back one shows through the hole. Right:
            // the front one, nearer, is kept.
            const float* left = image->rgba.data() + (size_t{h} / 2 * w + w / 2 - 20) * 4;
            const float* right = image->rgba.data() + (size_t{h} / 2 * w + w / 2 + 20) * 4;
            std::printf("  half cut by %s: left %.3f %.3f %.3f, right %.3f %.3f %.3f\n", route, double(left[0]),
                        double(left[1]), double(left[2]), double(right[0]), double(right[1]), double(right[2]));
            CHECK(left[0] > left[2]);    // the back square's orange
            CHECK(right[2] > right[0]);  // the front square's blue
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(buffer);
            images.push_back(std::move(*buffer));
        }
        for (size_t k = 1; k < images.size(); ++k) {
            auto diff = render::compareImages(*gpu->library, images[0], images[k], w, h);
            REQUIRE(diff);
            std::printf("  raster against %s: p99 %u, max %u, %llu pixels beyond 2\n", routes[k], diff->p99, diff->max,
                        static_cast<unsigned long long>(diff->over2));
            // The cut is decided at the pixel's centre by the same material in
            // every route, as the silhouettes are: a few edge pixels apart.
            CHECK(diff->over2 * 100 <= uint64_t{w} * h);
        }
    }
}

TEST_CASE("a GeomSubset's material shades its faces; the mesh's the rest", "[usd][gpu][mesh][materials]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto diffuse = [](const std::string& name, const std::string& colour) {
        return "    def Material \"" + name + "\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/" + name + "/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/" + name + "/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (" + colour + ")\n"
               "            token outputs:out\n        }\n    }\n";
    };
    const fs::path path = scratch("subsets.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Faces\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]\n"
               "    point3f[] points = [(-2, -1, -5), (0, -1, -5), (2, -1, -5), (-2, 1, -5), (0, 1, -5), (2, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    uniform token subsetFamily:materialBind:familyType = \"nonOverlapping\"\n"
               "    rel material:binding = </Materials/Green>\n"
               "    def GeomSubset \"Left\" (\n        prepend apiSchemas = [\"MaterialBindingAPI\"]\n    )\n    {\n"
               "        uniform token elementType = \"face\"\n"
               "        uniform token familyName = \"materialBind\"\n"
               "        int[] indices = [0]\n"
               "        rel material:binding = </Materials/Red>\n    }\n}\n"
               "def Scope \"Materials\"\n{\n" << diffuse("Red", "0.9, 0.1, 0.1") << diffuse("Green", "0.1, 0.9, 0.1")
            << "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 20\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    const auto pixel = [&](uint32_t x, uint32_t y) {
        const float* p = image->rgba.data() + (size_t{y} * w + x) * 4;
        return std::array<float, 3>{p[0], p[1], p[2]};
    };
    const auto left = pixel(w / 2 - 20, h / 2);
    const auto right = pixel(w / 2 + 20, h / 2);
    std::printf("  subset face: %.3f %.3f %.3f; the rest: %.3f %.3f %.3f\n", double(left[0]), double(left[1]),
                double(left[2]), double(right[0]), double(right[1]), double(right[2]));
    CHECK(left[0] > 0.8F);
    CHECK(left[1] < 0.15F);
    CHECK(right[1] > 0.8F);
    CHECK(right[0] < 0.15F);
}

TEST_CASE("displayColor reaches the pixels per face and per indexed face-vertex", "[usd][gpu][mesh][primvars]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("primvars.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               // Two faces, one colour each.
               "def Mesh \"PerFace\"\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 4, 3, 1, 2, 5, 4]\n"
               "    point3f[] points = [(-2, 0.1, -5), (0, 0.1, -5), (2, 0.1, -5), (-2, 1.5, -5), (0, 1.5, -5), (2, 1.5, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(1, 0, 0), (0, 1, 0)] ( interpolation = \"uniform\" )\n"
               "}\n"
               // One face, red on its left corners and blue on its right, indexed.
               "def Mesh \"PerCorner\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -1.5, -5), (2, -1.5, -5), (2, -0.1, -5), (-2, -0.1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(1, 0, 0), (0, 0, 1)] ( interpolation = \"faceVarying\" )\n"
               "    int[] primvars:displayColor:indices = [0, 1, 1, 0]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 20\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    // Pixels well inside each region (bottom row first): what they read.
    const auto pixel = [&](uint32_t x, uint32_t y) {
        const float* p = image->rgba.data() + (size_t{y} * w + x) * 4;
        return std::array<float, 3>{p[0], p[1], p[2]};
    };
    const auto left = pixel(w / 2 - 20, h / 2 + 15);
    const auto right = pixel(w / 2 + 20, h / 2 + 15);
    const auto cornerLeft = pixel(w / 2 - 45, h / 2 - 15);
    const auto cornerMiddle = pixel(w / 2, h / 2 - 15);
    const auto cornerRight = pixel(w / 2 + 45, h / 2 - 15);
    std::printf("  per face: (%.2f %.2f %.2f) | (%.2f %.2f %.2f); per corner: r/b %.2f/%.2f, %.2f/%.2f, %.2f/%.2f\n",
                double(left[0]), double(left[1]), double(left[2]), double(right[0]), double(right[1]),
                double(right[2]), double(cornerLeft[0]), double(cornerLeft[2]), double(cornerMiddle[0]),
                double(cornerMiddle[2]), double(cornerRight[0]), double(cornerRight[2]));
    CHECK(left[0] > 0.5F);
    CHECK(left[1] == 0.0F);
    CHECK(right[1] > 0.5F);
    CHECK(right[0] == 0.0F);
    CHECK(cornerLeft[0] > cornerLeft[2]);
    CHECK(cornerRight[2] > cornerRight[0]);
    CHECK(std::abs(cornerMiddle[0] - cornerMiddle[2]) < 0.1F);
    CHECK(cornerMiddle[1] == 0.0F);

    // The same frame's ids, normals and a primvar, as Hydra's outputs.
    (*renderer)->requestOutputs({"primId", "elementId", "Neye", "normal", "primvars:displayColor"});
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto primId = (*renderer)->mappedOutput("primId");
    auto elementId = (*renderer)->mappedOutput("elementId");
    auto eye = (*renderer)->mappedOutput("Neye");
    auto world = (*renderer)->mappedOutput("normal");
    auto colour = (*renderer)->mappedOutput("primvars:displayColor");
    REQUIRE(primId);
    REQUIRE(elementId);
    REQUIRE(eye);
    REQUIRE(world);
    REQUIRE(colour);
    // Hydra's buffers are bottom row first, as the engine's images.
    const auto at = [&](const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y, uint32_t channels, uint32_t c) {
        int32_t v = 0;
        std::memcpy(&v, bytes.data() + (size_t{y} * w + x) * channels * 4 + c * 4, 4);
        return v;
    };
    const auto atFloat = [&](const std::vector<uint8_t>& bytes, uint32_t x, uint32_t y, uint32_t channels,
                             uint32_t c) {
        float v = 0.0F;
        std::memcpy(&v, bytes.data() + (size_t{y} * w + x) * channels * 4 + c * 4, 4);
        return v;
    };
    const uint32_t lx = w / 2 - 20, rx = w / 2 + 20, fy = h / 2 + 15, cy = h / 2 - 15;
    std::printf("  primId %d %d %d, background %d; elementId %d %d; Neye z %.3f; normal z %.3f; displayColor r %.3f\n",
                at(*primId, lx, fy, 1, 0), at(*primId, rx, fy, 1, 0), at(*primId, w / 2, cy, 1, 0),
                at(*primId, 2, 2, 1, 0), at(*elementId, lx, fy, 1, 0), at(*elementId, rx, fy, 1, 0),
                double(atFloat(*eye, lx, fy, 3, 2)), double(atFloat(*world, lx, fy, 3, 2)),
                double(atFloat(*colour, lx, fy, 3, 0)));
    CHECK(at(*primId, lx, fy, 1, 0) == at(*primId, rx, fy, 1, 0));
    CHECK(at(*primId, lx, fy, 1, 0) != at(*primId, w / 2, cy, 1, 0));
    CHECK(at(*primId, lx, fy, 1, 0) >= 0);
    CHECK(at(*primId, 2, 2, 1, 0) == -1);
    CHECK(at(*elementId, lx, fy, 1, 0) == 0);
    CHECK(at(*elementId, rx, fy, 1, 0) == 1);
    CHECK(atFloat(*eye, lx, fy, 3, 2) == Catch::Approx(1.0F).margin(1e-4F));
    CHECK(atFloat(*world, lx, fy, 3, 2) == Catch::Approx(1.0F).margin(1e-4F));
    CHECK(atFloat(*colour, lx, fy, 3, 0) == Catch::Approx(1.0F).margin(1e-5F));
    CHECK(atFloat(*colour, lx, fy, 3, 1) == Catch::Approx(0.0F).margin(1e-5F));

    // What a viewer asks of the same frame: the prim under a pixel (from the
    // top left) and where the stage is.
    auto overFace = (*renderer)->pick(lx, h - 1 - fy);
    auto overCorner = (*renderer)->pick(w / 2, h - 1 - cy);
    auto overNothing = (*renderer)->pick(2, 2);
    REQUIRE(overFace);
    REQUIRE(overCorner);
    REQUIRE(overNothing);
    REQUIRE(overFace->has_value());
    REQUIRE(overCorner->has_value());
    CHECK((*overFace)->prim == "/PerFace");
    CHECK((*overCorner)->prim == "/PerCorner");
    CHECK(!overNothing->has_value());
    auto box = (*renderer)->bounds();
    REQUIRE(box);
    REQUIRE(box->has_value());
    std::printf("  picked %s and %s; bounds (%.2f %.2f %.2f)-(%.2f %.2f %.2f)\n", (*overFace)->prim.c_str(),
                (*overCorner)->prim.c_str(), double((*box)->min[0]), double((*box)->min[1]), double((*box)->min[2]),
                double((*box)->max[0]), double((*box)->max[1]), double((*box)->max[2]));
    CHECK((*box)->min[0] == Catch::Approx(-2.0F).margin(1e-5F));
    CHECK((*box)->min[1] == Catch::Approx(-1.5F).margin(1e-5F));
    CHECK((*box)->max[1] == Catch::Approx(1.5F).margin(1e-5F));
    CHECK((*box)->max[2] == Catch::Approx(-5.0F).margin(1e-5F));
}

TEST_CASE("a PointInstancer draws as its instances authored one by one", "[usd][gpu][mesh][instancing]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string square =
        "        int[] faceVertexCounts = [4]\n"
        "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
        "        point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
        "        uniform token subdivisionScheme = \"none\"\n"
        "        color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    double3 xformOp:translate = (0.5, 1, 9)\n"
                               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    const fs::path instanced = scratch("instancer.usda");
    {
        std::ofstream out(instanced);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def PointInstancer \"Many\"\n{\n"
               "    rel prototypes = [</Many/Prototypes/Square>]\n"
               "    int[] protoIndices = [0, 0, 0]\n"
               "    point3f[] positions = [(-2, 0, 0), (0, 1, -1), (2, -0.5, 0.5)]\n"
               "    quath[] orientations = [(1, 0, 0, 0), (0.7071068, 0, 0.7071068, 0), (0.9238795, 0.3826834, 0, 0)]\n"
               "    float3[] scales = [(1, 1, 1), (2, 1, 1), (1, 1.5, 1)]\n"
               "    double3 xformOp:rotateXYZ = (0, 10, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n"
               "    def Scope \"Prototypes\"\n    {\n"
               "        def Mesh \"Square\"\n        {\n" << square << "        }\n    }\n}\n" << camera;
    }
    const fs::path authored = scratch("authored.usda");
    {
        std::ofstream out(authored);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Many\"\n{\n"
               "    double3 xformOp:rotateXYZ = (0, 10, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n";
        // translate, orient (as a rotation about one axis), scale: the
        // instancer's T * R * S, authored as ops.
        const char* ops[3] = {
            "        double3 xformOp:translate = (-2, 0, 0)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n",
            "        double3 xformOp:translate = (0, 1, -1)\n        double xformOp:rotateY = 90\n"
            "        float3 xformOp:scale = (2, 1, 1)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateY\", \"xformOp:scale\"]\n",
            "        double3 xformOp:translate = (2, -0.5, 0.5)\n        double xformOp:rotateX = 45\n"
            "        float3 xformOp:scale = (1, 1.5, 1)\n"
            "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\", \"xformOp:scale\"]\n"};
        for (int k = 0; k < 3; ++k) {
            out << "    def Mesh \"Square" << k << "\"\n    {\n" << ops[k] << square << "    }\n";
        }
        out << "}\n" << camera;
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto a = usd::StageRenderer::open(instanced);
    auto b = usd::StageRenderer::open(authored);
    if (!a) FAIL(a.error().toString());
    if (!b) FAIL(b.error().toString());
    auto imageA = (*a)->render("/Camera", 0.0, w, h);
    auto imageB = (*b)->render("/Camera", 0.0, w, h);
    if (!imageA) FAIL(imageA.error().toString());
    if (!imageB) FAIL(imageB.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = imageA->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, imageA->rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, imageB->rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareHdr(*gpu->library, *bufferA, *bufferB, w, h);
    REQUIRE(diff);
    auto blank = std::vector<float>(imageA->rgba.size(), 0.0F);
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto drawn = render::compareHdr(*gpu->library, *bufferA, *blankBuffer, w, h);
    REQUIRE(drawn);
    std::printf("  PointInstancer against authored xforms: relMSE %.2e, p99 relative %.2e (against blank: relMSE %.2e)\n",
                diff->relMse, diff->p99Relative, drawn->relMse);
    CHECK(drawn->relMse > 1.0);    // something was drawn
    CHECK(diff->relMse < 1e-5);    // half-precision orientations
}

TEST_CASE("Hydra gets the same ids whichever route finds the meshes", "[usd][gpu][mesh][visibility]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries, to compare all three routes");
    }
    const fs::path path = scratch("routes.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               // Instanced squares, turned and scaled; a single-sided floor
               // facing the camera and a wall behind it facing away; a
               // mirrored single-sided square.
               "def PointInstancer \"Many\"\n{\n"
               "    rel prototypes = [</Many/Prototypes/Square>]\n"
               "    int[] protoIndices = [0, 0, 0, 0]\n"
               "    point3f[] positions = [(-2, 0, 0), (0, 1, -1), (2, -0.5, 0.5), (0.5, -1.2, 1)]\n"
               "    quath[] orientations = [(1, 0, 0, 0), (0.7071068, 0, 0.7071068, 0), (0.9238795, 0.3826834, 0, 0),"
               " (0.9659258, 0, 0, 0.258819)]\n"
               "    float3[] scales = [(1, 1, 1), (2, 1, 1), (1, 1.5, 1), (0.7, 0.7, 0.7)]\n"
               "    def Scope \"Prototypes\"\n    {\n"
               "        def Mesh \"Square\"\n        {\n"
               "            int[] faceVertexCounts = [4]\n"
               "            int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "            point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
               "            uniform token subdivisionScheme = \"none\"\n"
               "            uniform bool doubleSided = 1\n"
               "        }\n    }\n}\n"
               "def Mesh \"Ground\"\n{\n"
               "    int[] faceVertexCounts = [4, 4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3, 3, 5, 4, 2]\n"
               "    point3f[] points = [(-4, -2, 3), (4, -2, 3), (4, -2, -3), (-4, -2, -3), (4, 1, -5), (-4, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "}\n"
               "def Mesh \"Mirrored\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (-0.5, 0.5, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    double3 xformOp:translate = (-1.5, 1.5, 1)\n    float3 xformOp:scale = (-1, 1, 1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    double3 xformOp:translate = (0.5, 0.5, 9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"primId", "instanceId", "elementId"});
    const uint32_t w = 320;
    const uint32_t h = 240;
    struct Frame {
        gpu::Buffer ids;   // primId, instanceId, elementId: three words a pixel
        uint64_t    covered = 0;
    };
    const auto frame = [&](const char* route) {
        REQUIRE((*renderer)->setMeshVisibility(route));
        REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
        std::vector<uint8_t> bytes;
        for (const char* aov : {"primId", "instanceId", "elementId"}) {
            auto mapped = (*renderer)->mappedOutput(aov);
            REQUIRE(mapped);
            REQUIRE(mapped->size() == size_t{w} * h * 4);
            bytes.insert(bytes.end(), mapped->begin(), mapped->end());
        }
        gpu::BufferDesc desc;
        desc.bytes = bytes.size();
        desc.elementBytes = 4;
        auto ids = gpu::Buffer::create(*gpu->device, desc, bytes.data());
        REQUIRE(ids);
        // Coverage: the primId plane against a cleared one (-1 everywhere).
        const std::vector<int32_t> cleared(size_t{w} * h, -1);
        gpu::BufferDesc plane;
        plane.bytes = cleared.size() * 4;
        plane.elementBytes = 4;
        auto blank = gpu::Buffer::create(*gpu->device, plane, cleared.data());
        auto prims = gpu::Buffer::create(*gpu->device, plane, bytes.data());
        REQUIRE(blank);
        REQUIRE(prims);
        auto covered = render::countDifferent(*gpu->library, *prims, *blank, w * h);
        REQUIRE(covered);
        return Frame{std::move(*ids), *covered};
    };
    const Frame raster = frame("raster");
    const Frame rays = frame("rays");
    const Frame bvh = frame("bvh");
    const Frame automatic = frame("automatic");
    auto raysDiffer = render::countDifferent(*gpu->library, raster.ids, rays.ids, 3 * w * h);
    auto bvhDiffer = render::countDifferent(*gpu->library, raster.ids, bvh.ids, 3 * w * h);
    auto automaticDiffer = render::countDifferent(*gpu->library, rays.ids, automatic.ids, 3 * w * h);
    REQUIRE(raysDiffer);
    REQUIRE(bvhDiffer);
    REQUIRE(automaticDiffer);
    std::printf("  ids through Hydra, %u x %u, raster covering %llu: rays differ in %llu words, the BVH in %llu, "
                "automatic in %llu\n",
                w, h, static_cast<unsigned long long>(raster.covered), static_cast<unsigned long long>(*raysDiffer),
                static_cast<unsigned long long>(*bvhDiffer), static_cast<unsigned long long>(*automaticDiffer));
    CHECK(raster.covered > uint64_t{w} * h / 8);
    CHECK(*automaticDiffer == 0);   // this device has ray queries: automatic is rays
    // Raster samples pixel centres as the rays do, so they part only where a
    // centre falls on an edge -- here along the grazing floor: measured 15
    // and 6 words of 3 x 76800.
    CHECK(*raysDiffer * 1000 <= raster.covered);
    CHECK(*bvhDiffer * 1000 <= raster.covered);
}

TEST_CASE("the hdAthenea plugin loads through USD's renderer plugin registry", "[usd][plugin]") {
    const fs::path plugins = ATHENEA_HYDRA_PLUGIN_DIR;
    PlugRegistry::GetInstance().RegisterPlugins(plugins.string());
    HfPluginDescVector descs;
    HdRendererPluginRegistry::GetInstance().GetPluginDescs(&descs);
    bool found = false;
    for (const HfPluginDesc& desc : descs) {
        found = found || desc.id == TfToken("HdAtheneaRendererPlugin");
    }
    CHECK(found);
    HdPluginRenderDelegateUniqueHandle delegate =
        HdRendererPluginRegistry::GetInstance().CreateRenderDelegate(TfToken("HdAtheneaRendererPlugin"));
    CHECK(delegate);
}

TEST_CASE("a SplatEdit authored on an ancestor Xform reaches the cloud through Hydra", "[usd][gpu][edit]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path path = scratch("edit-cloud.usdc");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, {.addCamera = false}));

    // The edit on a group above the cloud, as constant primvars: inherited by
    // everything under it, in each cloud's own space.
    const fs::path shot = scratch("edit-shot.usda");
    {
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def Xform \"Group\" (\n        prepend apiSchemas = [\"AtheneaSplatEditAPI\"]\n    )\n    {\n"
               "        bool primvars:athenea:edit:active = 1 ( interpolation = \"constant\" )\n"
               "        token primvars:athenea:edit:shape = \"sphere\" ( interpolation = \"constant\" )\n"
               "        token primvars:athenea:edit:mode = \"grade\" ( interpolation = \"constant\" )\n"
               "        float3 primvars:athenea:edit:centre = (0.5, 0, 0) ( interpolation = \"constant\" )\n"
               "        float3 primvars:athenea:edit:size = (1.5, 0, 0) ( interpolation = \"constant\" )\n"
               "        color3f primvars:athenea:edit:tint = (1, 0.3, 0.1) ( interpolation = \"constant\" )\n"
               "        float primvars:athenea:edit:saturation = 0.5 ( interpolation = \"constant\" )\n"
               "        float primvars:athenea:edit:opacity = 0.6 ( interpolation = \"constant\" )\n"
               "        bool primvars:athenea:edit:invert = 1 ( interpolation = \"constant\" )\n"
               "        def \"Cloud\" ( references = @./edit-cloud.usdc@</World/Splats> )\n        {\n"
               "            double3 xformOp:rotateXYZ = (0, 35, 0)\n"
               "            uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n        }\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) {
        FAIL(renderer.error().toString());
    }
    auto image = (*renderer)->render("/World/Shot", 0.0, 240, 180);
    if (!image) {
        FAIL(image.error().toString());
    }

    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(raw);
    REQUIRE(splats);
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.4, 0.2, 7.0});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    render::SplatEdit edit;
    edit.active = true;
    edit.shape = render::SplatEdit::Shape::Sphere;
    edit.mode = render::SplatEdit::Mode::Grade;
    edit.centre = {0.5F, 0.0F, 0.0F};
    edit.size = {1.5F, 0.0F, 0.0F};
    edit.tint = {1.0F, 0.3F, 0.1F};
    edit.saturation = 0.5F;
    edit.opacity = 0.6F;
    edit.invert = true;
    render::RenderTargets direct, plain;
    REQUIRE(rasterizer->render(camera, std::vector<render::SplatInstance>{{&*splats, aofx::xform::rotationY(35.0), edit}},
                               settings, direct));
    REQUIRE(rasterizer->render(camera, std::vector<render::SplatInstance>{{&*splats, aofx::xform::rotationY(35.0)}},
                               settings, plain));

    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(hydra);
    auto diff = render::compareImages(*gpu->library, *hydra, direct.colour, 240, 180);
    REQUIRE(diff);
    CHECK(diff->p99 <= 1);
    CHECK(diff->max <= 2);
    auto unedited = render::compareImages(*gpu->library, *hydra, plain.colour, 240, 180);
    REQUIRE(unedited);
    CHECK(unedited->over2 > unedited->pixels / 20);   // the edit really arrived
}

TEST_CASE("a .athc referenced from USD is cut and streamed through Hydra as it is directly", "[usd][gpu][lod]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(cloud(20000));
    REQUIRE(splats);
    auto builder = lod::LodBuilder::create(*gpu->library);
    REQUIRE(builder);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto built = builder->build(*splats, chunked);
    REQUIRE(built);
    REQUIRE(lod::writeAthc(*gpu->device, *built, scratch("stream.athc")));

    const float threshold = 12.0F;
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.3, 0.4, 4.5});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = 240;
    settings.height = 180;
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    auto cutter = lod::CutSelector::create(*gpu->library);
    REQUIRE(cutter);
    const render::Projection projection = render::projectionFor(camera, settings.width, settings.height);

    for (const uint64_t budget : {uint64_t{0}, uint64_t{8000}}) {
        INFO("budget " << budget);
        const fs::path shot = scratch("stream-shot-" + std::to_string(budget) + ".usda");
        {
            std::ofstream out(shot);
            out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
                   "def Xform \"World\"\n{\n"
                   "    def ParticleField3DGaussianSplat \"Cloud\" (\n"
                   "        prepend apiSchemas = [\"AtheneaStreamedAssetAPI\"]\n    )\n    {\n"
                   "        asset primvars:athenea:asset = @./stream.athc@ ( interpolation = \"constant\" )\n"
                   "        float primvars:athenea:lod:threshold = " << threshold << " ( interpolation = \"constant\" )\n"
                   "        int64 primvars:athenea:stream:budget = " << budget << " ( interpolation = \"constant\" )\n"
                   "    }\n"
                   "    def Camera \"Shot\"\n    {\n"
                   "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
                   "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
                   "        double3 xformOp:translate = (0.3, 0.4, 4.5)\n"
                   "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
        }
        auto renderer = usd::StageRenderer::open(shot);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/World/Shot", 0.0, settings.width, settings.height);
        if (!image) FAIL(image.error().toString());

        // The same, directly: the file read whole, or the same stream settled.
        std::unique_ptr<lod::StreamingPool> pool;
        std::optional<lod::LodCloud> whole;
        if (budget == 0) {
            auto read = lod::readAthc(*gpu->device, scratch("stream.athc"));
            REQUIRE(read);
            whole.emplace(std::move(*read));
        } else {
            auto opened = lod::StreamingPool::open(*gpu->device, scratch("stream.athc"), {budget, 2});
            REQUIRE(opened);
            pool = std::move(*opened);
        }
        const lod::LodCloud& lodCloud = pool ? pool->cloud() : *whole;
        std::vector<lod::CutStats> stats;
        std::vector<render::SplatInstance> drawn;
        for (int round = 0; round < 16; ++round) {
            auto selected = cutter->select(projection, std::vector<lod::LodInstance>{{&lodCloud}}, threshold, &stats);
            REQUIRE(selected);
            drawn = *selected;
            if (!pool) {
                break;
            }
            pool->want(stats.front().needs);
            auto placed = pool->update(true);
            REQUIRE(placed);
            if (*placed == 0) {
                break;
            }
        }
        std::printf("  budget %llu: %u splats + %u merged drawn\n", static_cast<unsigned long long>(budget),
                    stats.front().splats, stats.front().merged);
        CHECK(stats.front().merged > 0);
        CHECK(stats.front().splats > 0);
        render::RenderTargets direct;
        REQUIRE(rasterizer->render(projection, drawn, settings, direct));

        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto hydra = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(hydra);
        auto diff = render::compareImages(*gpu->library, *hydra, direct.colour, settings.width, settings.height);
        REQUIRE(diff);
        std::printf("  budget %llu: Hydra against direct p99 %u, max %u\n", static_cast<unsigned long long>(budget),
                    diff->p99, diff->max);
        CHECK(diff->p99 <= 1);
        CHECK(diff->max <= 2);
    }
}

TEST_CASE("a cloud authored in half floats draws as the same cloud in floats", "[usd][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path floats = scratch("floats.usdc");
    const fs::path halves = scratch("halves.usdc");
    fs::remove(floats);
    fs::remove(halves);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, floats, {.addCamera = false}));
    {
        // The fixture: the same attributes, authored as their half twins.
        UsdStageRefPtr from = UsdStage::Open(floats.string());
        REQUIRE(from);
        const UsdVolParticleField3DGaussianSplat source(from->GetPrimAtPath(SdfPath("/World/Splats")));
        UsdStageRefPtr to = UsdStage::CreateNew(halves.string());
        UsdGeomXform::Define(to, SdfPath("/World"));
        auto target = UsdVolParticleField3DGaussianSplat::Define(to, SdfPath("/World/Splats"));
        const auto halve3 = [](const VtVec3fArray& in) {
            VtVec3hArray out(in.size());
            for (size_t i = 0; i < in.size(); ++i) {
                out[i] = GfVec3h(in[i]);
            }
            return out;
        };
        VtVec3fArray positions, scales, coefficients;
        VtQuatfArray orientations;
        VtFloatArray opacities;
        int degree = 0;
        REQUIRE(source.GetPositionsAttr().Get(&positions));
        REQUIRE(source.GetScalesAttr().Get(&scales));
        REQUIRE(source.GetOrientationsAttr().Get(&orientations));
        REQUIRE(source.GetOpacitiesAttr().Get(&opacities));
        REQUIRE(source.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&coefficients));
        REQUIRE(source.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree));
        VtQuathArray orientationsh(orientations.size());
        for (size_t i = 0; i < orientations.size(); ++i) {
            orientationsh[i] = GfQuath(orientations[i]);
        }
        target.CreatePositionshAttr(VtValue(halve3(positions)));
        target.CreateScaleshAttr(VtValue(halve3(scales)));
        target.CreateOrientationshAttr(VtValue(orientationsh));
        target.CreateOpacitieshAttr(VtValue(VtHalfArray(opacities.begin(), opacities.end())));
        target.CreateRadianceSphericalHarmonicsCoefficientshAttr(VtValue(halve3(coefficients)));
        target.CreateRadianceSphericalHarmonicsDegreeAttr(VtValue(degree));
        REQUIRE(to->GetRootLayer()->Save());
    }
    const auto shotOf = [&](const std::string& name, const std::string& cloudFile) {
        const fs::path shot = scratch(name);
        std::ofstream out(shot);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def \"Cloud\" ( references = @./" << cloudFile << "@</World/Splats> )\n    {\n    }\n"
               "    def Camera \"Shot\"\n    {\n"
               "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
               "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
               "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
        return shot;
    };
    auto a = usd::StageRenderer::open(shotOf("floats-shot.usda", "floats.usdc"));
    auto b = usd::StageRenderer::open(shotOf("halves-shot.usda", "halves.usdc"));
    if (!a) FAIL(a.error().toString());
    if (!b) FAIL(b.error().toString());
    auto imageA = (*a)->render("/World/Shot", 0.0, 240, 180);
    auto imageB = (*b)->render("/World/Shot", 0.0, 240, 180);
    if (!imageA) FAIL(imageA.error().toString());
    if (!imageB) FAIL(imageB.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = imageA->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, imageA->rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, imageB->rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareImages(*gpu->library, *bufferA, *bufferB, 240, 180);
    REQUIRE(diff);
    std::printf("  half against float: p99 %u, max %u\n", diff->p99, diff->max);
    // Half precision moves a unit-scale position by up to 1/1024: sub-pixel here.
    CHECK(diff->p99 <= 1);
    CHECK(diff->over2 * 100 <= diff->pixels);
}

TEST_CASE("the codeless athenea schemas register, with their defaults", "[usd][schema]") {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    const UsdSchemaRegistry& registry = UsdSchemaRegistry::GetInstance();
    const UsdPrimDefinition* edit = registry.FindAppliedAPIPrimDefinition(TfToken("AtheneaSplatEditAPI"));
    REQUIRE(edit != nullptr);
    CHECK(registry.FindAppliedAPIPrimDefinition(TfToken("AtheneaPointStyleAPI")) != nullptr);
    const UsdPrimDefinition* lighting = registry.FindAppliedAPIPrimDefinition(TfToken("AtheneaSplatLightingAPI"));
    REQUIRE(lighting != nullptr);
    // Baked is the default: a capture shows the light it was captured under
    // until something asks otherwise.
    VtValue relight;
    CHECK(lighting->GetAttributeFallbackValue(TfToken("primvars:athenea:splat:relight"), &relight));
    CHECK(relight.IsHolding<bool>());
    CHECK(!relight.UncheckedGet<bool>());
    // Built on UsdVol's radiance base, the way the schema's own radiance
    // definition is: applying the one applies the other.
    {
        const TfTokenVector applied = lighting->GetAppliedAPISchemas();
        CHECK(std::find(applied.begin(), applied.end(), TfToken("ParticleFieldRadianceBaseAPI")) != applied.end());
    }

    // Every schema the engine reads must be one the registry knows: the
    // skinning one was read by name for a whole milestone without being
    // registered at all, and three sets of primvars were consumed with no
    // declaration anywhere.
    for (const char* name : {"AtheneaStreamedAssetAPI", "AtheneaSplatSkinningAPI", "AtheneaSplatVisibilityAPI", "AtheneaVolumeAPI"}) {
        INFO(name);
        CHECK(registry.FindAppliedAPIPrimDefinition(TfToken(name)) != nullptr);
    }
    const UsdPrimDefinition* skinning = registry.FindAppliedAPIPrimDefinition(TfToken("AtheneaSplatSkinningAPI"));
    REQUIRE(skinning != nullptr);
    VtValue elementSize;
    CHECK(skinning->GetPropertyMetadata(TfToken("primvars:athenea:splat:jointIndices"), TfToken("elementSize"),
                                        &elementSize));
    CHECK(elementSize.IsHolding<int>());
    CHECK(elementSize.UncheckedGet<int>() == 4);
    // The weights' gradients: two components for each of three joints.
    VtValue gradientSize;
    CHECK(skinning->GetPropertyMetadata(TfToken("primvars:athenea:splat:jointWeightGradients"),
                                        TfToken("elementSize"), &gradientSize));
    CHECK(gradientSize.IsHolding<int>());
    CHECK(gradientSize.UncheckedGet<int>() == 6);
    VtValue litBody;
    CHECK(lighting->GetAttributeFallbackValue(TfToken("primvars:athenea:splat:litBody"), &litBody));
    CHECK(litBody.IsHolding<bool>());
    const UsdPrimDefinition* volume = registry.FindAppliedAPIPrimDefinition(TfToken("AtheneaVolumeAPI"));
    REQUIRE(volume != nullptr);
    VtValue albedo;
    CHECK(volume->GetAttributeFallbackValue(TfToken("primvars:athenea:albedo"), &albedo));
    CHECK(albedo.IsHolding<GfVec3f>());
    CHECK(albedo.UncheckedGet<GfVec3f>() == GfVec3f(0.8F, 0.8F, 0.8F));

    UsdStageRefPtr stage = UsdStage::CreateInMemory();
    UsdPrim group = stage->DefinePrim(SdfPath("/Group"), TfToken("Xform"));
    CHECK(group.ApplyAPI(TfToken("AtheneaSplatEditAPI")));
    CHECK(group.HasAPI(TfToken("AtheneaSplatEditAPI")));
    // Unauthored, the schema's defaults answer.
    float saturation = 0.0F;
    CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:saturation")).Get(&saturation));
    CHECK(saturation == 1.0F);
    TfToken mode;
    CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:mode")).Get(&mode));
    CHECK(mode == TfToken("grade"));
    VtValue allowed;
    CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:shape")).GetMetadata(TfToken("allowedTokens"), &allowed));
}

TEST_CASE("a UsdLux light through Hydra lights a Lambert plane as the closed form says",
          "[usd][gpu][mesh][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_sphere.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\" (\n    prepend apiSchemas = [\"ShadowAPI\"]\n)\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    // The same stage with no light at all: the headlight, which the analytic
    // square checks exactly. If this is 0 and 0, the material is Lambert of
    // albedo 0.8 and whatever the lit image does is the light's doing.
    {
        const fs::path dark = scratch("light_none.usda");
        {
            std::ifstream in(path);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            const size_t at = text.find("def SphereLight");
            const size_t end = text.find("def Scope \"Materials\"");
            REQUIRE(at != std::string::npos);
            REQUIRE(end != std::string::npos);
            text.erase(at, end - at);
            std::ofstream out(dark);
            out << text;
        }
        auto plain = usd::StageRenderer::open(dark);
        if (!plain) FAIL(plain.error().toString());
        auto lit = (*plain)->render("/Camera", 0.0, w, h);
        if (!lit) FAIL(lit.error().toString());
        const auto c = squareMismatches(*gpu, *lit, {0.8F, 0.8F, 0.8F});
        std::printf("  the same plane by the headlight: %u covered, %u coverage and %u colour mismatches\n", c[2],
                    c[0], c[1]);
        CHECK(c[0] == 0);
        CHECK(c[1] == 0);
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // The closed form is what the estimator converges to: ask for enough
    // samples that what is left is the light, not the noise.
    (*renderer)->setLightSamples(4096);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(uint32_t{0});   // a sphere
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.4F};
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 0.0F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
            const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.02F};
            const float none[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            p["centre"].setData(centre, sizeof(centre));
            p["axisX"].setData(axisX, sizeof(axisX));
            p["axisY"].setData(axisY, sizeof(axisY));
            p["normal"].setData(normal, sizeof(normal));
            p["radiance"].setData(radiance, sizeof(radiance));
            p["occluder"].setData(none, sizeof(none));
            p["occluderAxes"].setData(none, sizeof(none));
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[2] = {0, 0};
    float probe[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
    const float relative = probe[0];
    // The centre of the plane sees the sphere on its axis: d = 2, r = 0.4,
    // so E = pi r^2 / d^2 and a Lambert surface returns albedo E / pi.
    const float* centre = image->rgba.data() + (size_t{h} / 2 * w + w / 2) * 4;
    const float* offAxis = image->rgba.data() + (size_t{h} / 2 * w + w / 2 + 20) * 4;
    const float expected = 0.8F * 0.4F * 0.4F / (2.0F * 2.0F);
    std::printf("  UsdLuxSphereLight: %u pixels, %u beyond 2%%, worst %.4f; centre %.5f (closed form %.5f), "
                "20 px off axis %.5f against %.5f at (%.3f, %.3f, %.3f)\n",
                c[0], c[1], double(relative), double(centre[0]), double(expected), double(offAxis[0]),
                double(probe[1]), double(probe[2]), double(probe[3]), double(probe[4]));
    CHECK(c[0] > 800);
    CHECK(c[1] == 0);
}

TEST_CASE("a dome light's image lights a Lambert plane, and shows where nothing is drawn",
          "[usd][gpu][mesh][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A lat-long image of one colour: whatever the mapping does, every
    // direction reads the same radiance, so the closed form still holds.
    const fs::path png = scratch("dome.png");
    {
        fs::remove(png);
        std::vector<uint32_t> texels(size_t{32} * 16, 0xFFCCCCCCu);   // ABGR, 0xCC per channel
        HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
        REQUIRE(made);
        HioImage::StorageSpec spec;
        spec.width = 32;
        spec.height = 16;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(made->Write(spec));
    }
    // Requested as auto, so an 8-bit image is read as sRGB.
    const float code = 0xCC / 255.0F;
    const float linear = std::pow((code + 0.055F) / 1.055F, 2.4F);
    const fs::path path = scratch("light_dome.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def DomeLight \"Sky\"\n{\n"
               "    float inputs:intensity = 1\n"
               "    color3f inputs:color = (1, 1, 1)\n"
            << "    asset inputs:texture:file = @" << png.string() << "@\n"
            << "}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 161;
    const uint32_t h = 121;
    (*renderer)->setLightSamples(std::getenv("ATHENEA_DOME_SAMPLES") != nullptr ? 16384u : 1024u);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(uint32_t{5});   // a dome
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.0F};
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 0.0F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
            const float radiance[4] = {linear, linear, linear, 0.02F};
            const float none[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            p["centre"].setData(centre, sizeof(centre));
            p["axisX"].setData(axisX, sizeof(axisX));
            p["axisY"].setData(axisY, sizeof(axisY));
            p["normal"].setData(normal, sizeof(normal));
            p["radiance"].setData(radiance, sizeof(radiance));
            p["occluder"].setData(none, sizeof(none));
            p["occluderAxes"].setData(none, sizeof(none));
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[2] = {0, 0};
    float probe[5] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(probe), probe));
    // A corner of the image sees no geometry: it shows the dome itself.
    const float* corner = image->rgba.data();
    std::printf("  dome image: %u pixels, %u beyond 2%%, worst %.4f; background %.4f (image %.4f)\n", c[0], c[1],
                double(probe[0]), double(corner[0]), double(linear));
    CHECK(c[0] > 800);
    CHECK(c[1] == 0);
    CHECK(corner[0] == Catch::Approx(linear).margin(0.01F));

    // The same image authored raw: its colour space reaches the texture
    // store through the light's network in the scene index, and the sky
    // shows the code values as they are. Before, every dome was read as the
    // file said (sRGB for 8 bits).
    const fs::path rawPath = scratch("light_dome_raw.usda");
    {
        std::ofstream out(rawPath);
        out << kSquareStage
            << "def DomeLight \"Sky\"\n{\n"
               "    float inputs:intensity = 1\n"
               "    color3f inputs:color = (1, 1, 1)\n"
            << "    asset inputs:texture:file = @" << png.string() << "@ ( colorSpace = \"raw\" )\n"
            << "}\n";
    }
    auto rawRenderer = usd::StageRenderer::open(rawPath);
    if (!rawRenderer) FAIL(rawRenderer.error().toString());
    auto rawImage = (*rawRenderer)->render("/Camera", 0.0, w, h);
    if (!rawImage) FAIL(rawImage.error().toString());
    std::printf("  dome image authored raw: background %.4f (code %.4f)\n", double(rawImage->rgba[0]), double(code));
    CHECK(rawImage->rgba[0] == Catch::Approx(code).margin(0.01F));
}

/// The scene index's half of light linking, through Hydra. Hidden until
/// the chain's order was found: HdsiLightLinkingSceneIndex builds its
/// collection cache from the added-prim notices that pass through it, and
/// StageRenderer gave the stage to UsdImaging's scene indices before this
/// renderer's filters existed, so they never heard of the stage's prims and
/// every category came out empty. The stage is now given once the render
/// index observes the chain.
TEST_CASE("a UsdLux light's collection reaches only what it includes",
          "[usd][gpu][mesh][lights][linking]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_linked.usda");
    {
        std::ofstream out(path);
        // Two squares side by side, one material, and a light whose collection
        // includes the left one alone. USD resolves that into a category the
        // left square carries, which is what the engine tests.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        for (int k = 0; k < 2; ++k) {
            const float x = k == 0 ? -1.0F : 1.0F;
            out << "def Mesh \"" << (k == 0 ? "Left" : "Right") << "\" (\n"
                   "    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                << "    point3f[] points = [(" << x - 0.9F << ", -0.9, -5), (" << x + 0.9F << ", -0.9, -5), ("
                << x + 0.9F << ", 0.9, -5), (" << x - 0.9F << ", 0.9, -5)]\n"
                << "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/Mat>\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def SphereLight \"Key\" (\n    prepend apiSchemas = [\"ShadowAPI\"]\n)\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token collection:lightLink:mode = \"expression\"\n"
               "    uniform pathExpression collection:lightLink:membershipExpression = \"/Left\"\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t w = 160;
    const uint32_t h = 120;
    (*renderer)->setLightSamples(16);
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());

    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto counters = gpu::ComputeKernel::create(*gpu->library, "athenea/test/light_check", "lightLinkCounters");
    if (!counters) FAIL(counters.error().toString());
    gpu::Buffer halves = test::uintBuffer(*gpu->device, 4, "halves");
    {
        gpu::CommandBatch batch(*gpu->device);
        counters->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["image"].setBinding(colours->rhi());
            cursor["imageDepth"].setBinding(depth->rhi());
            cursor["halves"].setBinding(halves.rhi());
            cursor["params"]["thetaBins"].setData(h);
            cursor["params"]["phiBins"].setData(w);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[4] = {0, 0, 0, 0};
    REQUIRE(halves.read(*gpu->device, 0, sizeof(n), n));
    std::printf("  collection:lightLink includes /Left: left %u drawn %u lit, right %u drawn %u lit\n", n[0], n[1],
                n[2], n[3]);
    CHECK(n[0] > 1000);
    CHECK(n[1] == n[0]);   // the light reaches what its collection includes
    CHECK(n[2] > 1000);
    CHECK(n[3] == 0);      // and nothing else
}

// A stage without lights: the raster lights it with the headlight, and the
// path tracer drew it black. Both now light it alike -- the headlight at the
// first vertex, nothing after -- so an unlit asset looks the same whichever
// technique opens it, bounces or not.
TEST_CASE("a stage without lights draws the same under rt as under the raster's headlight",
          "[usd][gpu][mesh][path][headlight]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("unlit.usda");
    {
        std::ofstream out(path);
        out << kSquareStage;
    }
    const uint32_t w = 160, h = 120;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto raster = (*renderer)->render("/Camera", 0.0, w, h, "raster");
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathBounces(3);
    auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!raster) FAIL(raster.error().toString());
    if (!traced) FAIL(traced.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = raster->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto a = gpu::Buffer::create(*gpu->device, desc, raster->rgba.data());
    auto b = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(a);
    REQUIRE(b);
    auto diff = render::compareImages(*gpu->library, *a, *b, w, h);
    REQUIRE(diff);
    const uint64_t lit = [&] {
        gpu::BufferDesc zeros = desc;
        auto empty = gpu::Buffer::create(*gpu->device, zeros);
        REQUIRE(empty);
        auto words = render::countDifferent(*gpu->library, *b, *empty, w * h * 4);
        REQUIRE(words);
        return *words;
    }();
    std::printf("  unlit stage, rt against raster: p99 %u, max %u, %llu pixels beyond 2; rt words lit %llu\n",
                diff->p99, diff->max, static_cast<unsigned long long>(diff->over2), static_cast<unsigned long long>(lit));
    CHECK(lit > 8000);
    CHECK(diff->max <= 1);
}

// Multiple importance sampling (M6): next event estimation and the
// material's own sampling, weighed by the power heuristic, on the case that
// needs both -- a glossy metal floor under a large rect light, whose
// highlight light sampling alone finds by luck. Two things are checked. That
// it is the same image: a deep frame with MIS against a deep frame without,
// within the noise either still carries. And that it is a better estimate of
// it: at equal paths, the error against the deep frame without MIS falls
// several times over.
TEST_CASE("MIS weighs light and material sampling into the same image with less noise on a glossy floor",
          "[usd][gpu][mesh][path][mis]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    struct Light {
        const char* name;
        const char* prim;
        double      atLeast;   // how many times less error MIS must reach
        const char* surface = "            float inputs:metallic = 1\n            float inputs:roughness = 0.2\n";
        /// How far the deep frames may part, over the shallow light-sampling
        /// frame's error: light sampling of a dome on a glossy floor is heavy
        /// tailed, and its deep frame carries fireflies that a relative MSE
        /// weighs by their square -- measured apart by 3.4e-2 at 8192 against
        /// 32768 paths while MIS at the two agreed to 1.8e-6. Its sameness
        /// is the rough floor's to show, where light sampling converges.
        /// Two independent deep frames of 8192 paths part by their summed
        /// noise, errNee * 32 / 8192 each: errNee / 128, a factor of two
        /// under the rough floor's bound.
        double      sameOver = 32.0;
    };
    const Light cases[] = {
        {"rect", "def RectLight \"Panel\"\n{\n"
                 "    float inputs:intensity = 4\n    float inputs:width = 4\n    float inputs:height = 2\n"
                 "    double3 xformOp:translate = (0, 2.5, -4)\n"
                 "    double xformOp:rotateX = -90\n"
                 "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n", 3.0},
        {"dome", "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n", 3.0,
         "            float inputs:metallic = 1\n            float inputs:roughness = 0.2\n", 2.0},
        {"dome, rough", "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n", 0.8,
         "            float inputs:metallic = 0\n            float inputs:roughness = 1\n", 64.0},
        {"sphere", "def SphereLight \"Bulb\"\n{\n"
                   "    float inputs:intensity = 20\n    float inputs:radius = 0.8\n"
                   "    double3 xformOp:translate = (0, 2.5, -4)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n", 1.5},
    };
    for (const Light& light : cases) {
    std::string file = std::string("mis_") + light.name + ".usda";
    std::replace(file.begin(), file.end(), ' ', '_');
    std::replace(file.begin(), file.end(), ',', '_');
    const fs::path path = scratch(file);
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Materials/Metal>\n}\n"
            << light.prim
            << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.2, 2)\n"
               "    double xformOp:rotateX = -12\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Metal\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Metal/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.9, 0.9, 0.9)\n"
            << light.surface
            << "            token outputs:surface\n        }\n    }\n}\n";
    }
    const uint32_t w = 64, h = 48;
    const auto frame = [&](bool mis, uint32_t total) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathMis(mis);
        (*renderer)->setPathSamples(std::min<uint32_t>(total, 256));
        (*renderer)->setPathTotal(total);
        (*renderer)->setPathBounces(1);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer deepNee = frame(false, 8192);
    const gpu::Buffer deepMis = frame(true, 8192);
    const gpu::Buffer nee = frame(false, 32);
    const gpu::Buffer mis = frame(true, 32);
    auto same = render::compareHdr(*gpu->library, deepMis, deepNee, w, h);
    auto errNee = render::compareHdr(*gpu->library, nee, deepNee, w, h);
    auto errMis = render::compareHdr(*gpu->library, mis, deepNee, w, h);
    REQUIRE(same);
    REQUIRE(errNee);
    REQUIRE(errMis);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto lit = render::compareHdr(*gpu->library, deepNee, *empty, w, h);
    REQUIRE(lit);
    std::printf("  %-11s: deep MIS against deep light sampling relMSE %.3e; at 32 paths against the deep frame, light "
                "sampling alone %.3e, MIS %.3e (%.1f times less); the frame against blank %.2e\n",
                light.name, same->relMse, errNee->relMse, errMis->relMse,
                errNee->relMse / std::max(errMis->relMse, 1e-30), lit->relMse);
    CHECK(lit->relMse > 0.1);
    // Same image: the deep frames differ by their own noise, which the
    // shallow light-sampling frame carries 256 times more of.
    CHECK(same->relMse < errNee->relMse / light.sameOver);
    CHECK(errMis->relMse < errNee->relMse / light.atLeast);
    }
}

// Emitting geometry as a light: a quad whose material emits, sampled by next
// event estimation and weighed against the material's rays, must light a
// floor as a UsdLux rect light of its size and radiance does -- both above
// the frame, so only the light they lay on the floor is seen -- and with
// less noise than the material's rays alone find it.
TEST_CASE("an emitting quad lights a floor as a rect light of its radiance, sampled as a light",
          "[usd][gpu][mesh][path][mis][emissive]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const auto stage = [&](const char* name, bool emitter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, 0, 3), (6, 0, 3), (6, 0, -9), (-6, 0, -9)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n";
        if (emitter) {
            // A 1 x 1 quad at y 2.5 facing down, emitting 3 each side.
            out << "def Mesh \"Panel\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-0.5, 2.5, -3.5), (0.5, 2.5, -3.5), (0.5, 2.5, -2.5), (-0.5, 2.5, -2.5)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/Glow>\n}\n"
                   "def Scope \"Materials\"\n{\n"
                   "    def Material \"Glow\"\n    {\n"
                   "        token outputs:mtlx:surface.connect = </Materials/Glow/Surface.outputs:out>\n"
                   "        def Shader \"Surface\"\n        {\n"
                   "            uniform token info:id = \"ND_surface_unlit\"\n"
                   "            float inputs:emission = 3\n"
                   "            color3f inputs:emission_color = (1, 1, 1)\n"
                   "            token outputs:out\n        }\n    }\n}\n";
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
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return path;
    };
    const uint32_t w = 64, h = 48;
    const auto frame = [&](const fs::path& path, bool mis, uint32_t total) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathMis(mis);
        (*renderer)->setPathSamples(std::min<uint32_t>(total, 256));
        (*renderer)->setPathTotal(total);
        (*renderer)->setPathBounces(1);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const fs::path quad = stage("emissive_quad.usda", true);
    const fs::path rect = stage("emissive_rect.usda", false);
    const gpu::Buffer rectDeep = frame(rect, true, 8192);
    const gpu::Buffer quadDeep = frame(quad, true, 8192);
    const gpu::Buffer quadLight = frame(quad, true, 32);
    const gpu::Buffer quadRays = frame(quad, false, 32);
    auto same = render::compareHdr(*gpu->library, quadDeep, rectDeep, w, h);
    auto errLight = render::compareHdr(*gpu->library, quadLight, rectDeep, w, h);
    auto errRays = render::compareHdr(*gpu->library, quadRays, rectDeep, w, h);
    REQUIRE(same);
    REQUIRE(errLight);
    REQUIRE(errRays);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto lit = render::compareHdr(*gpu->library, rectDeep, *empty, w, h);
    REQUIRE(lit);
    std::printf("  an emitting quad against a rect light, deep: relMSE %.3e (p99 relative %.3e); at 32 paths, the "
                "material's rays alone %.3e, sampled as a light %.3e (%.1f times less); lit against blank %.2e\n",
                same->relMse, same->p99Relative, errRays->relMse, errLight->relMse,
                errRays->relMse / std::max(errLight->relMse, 1e-30), lit->relMse);
    CHECK(lit->relMse > 0.1);
    CHECK(same->relMse < errLight->relMse / 32.0);
    CHECK(errLight->relMse < errRays->relMse / 4.0);
}

// The traced technique over a mesh, through Hydra: what used to trace splats
// and return now path traces the surfaces. A plane alone under one light has
// nothing for a bounce to find, so the traced frame and the raster frame are
// two estimators of the same direct light and must agree to within the noise
// the paths still carry. This says nothing about accumulation: every frame
// here is gathered in one pass.
TEST_CASE("the rt technique path traces a mesh and agrees with the raster's direct light",
          "[usd][gpu][mesh][path]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("rt_surface.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setLightSamples(64);
    // Enough paths that what is left between the two frames is the estimator
    // and not the noise: Monte Carlo error falls as 1/sqrt(N).
    (*renderer)->setPathSamples(1024);
    (*renderer)->setPathBounces(1);
    auto rasterised = (*renderer)->render("/Camera", 0.0, w, h);
    if (!rasterised) FAIL(rasterised.error().toString());
    auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!traced) FAIL(traced.error().toString());
    // That the traced frame drew the surface where the surface is. Only the
    // coverage is read here: squareMismatches compares colour against albedo
    // times the cosine to the eye, which is the headlight's answer and not
    // this scene's, and the closed form of a sphere light is what the M5 case
    // checks with athenea/test/lambert_irradiance. What this case is for is the
    // line below it.
    const auto counts = squareMismatches(*gpu, *traced, {0.8F, 0.8F, 0.8F});
    std::printf("  the rt technique over a mesh: %u covered, %u coverage mismatches\n", counts[2], counts[0]);
    CHECK(counts[2] > 8000);
    CHECK(counts[0] == 0);
    gpu::BufferDesc desc;
    desc.bytes = rasterised->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto rasterRgba = gpu::Buffer::create(*gpu->device, desc, rasterised->rgba.data());
    auto tracedRgba = gpu::Buffer::create(*gpu->device, desc, traced->rgba.data());
    REQUIRE(rasterRgba);
    REQUIRE(tracedRgba);
    auto diff = render::compareImages(*gpu->library, *rasterRgba, *tracedRgba, w, h);
    REQUIRE(diff);
    std::printf("  rt against raster over the same light: p99 %u, max %u, %llu pixels beyond 2\n", diff->p99,
                diff->max, static_cast<unsigned long long>(diff->over2));
    CHECK(diff->p99 <= 8);
}

// Progressive accumulation, through the whole chain: the delegate's settings,
// the render pass, the engine's mean. Drawing the same camera again is the same
// frame continued and its paths are added; moving the camera or changing a
// setting is a different frame and the mean starts over. The second half is
// what says the revision is armed rather than decorative -- a revision nothing
// ever raises would let the mean keep accumulating over a scene that changed,
// and no image would look wrong enough to say so.
TEST_CASE("a path traced frame accumulates over passes and starts again when it must",
          "[usd][gpu][mesh][path]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("rt_progressive.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 96;
    const uint32_t h = 72;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathBounces(1);
    (*renderer)->setPathTotal(32);
    // Eight passes of four paths reach the thirty-two it was told to gather.
    std::vector<uint32_t> held;
    for (uint32_t pass = 0; pass < 8; ++pass) {
        if (auto drawn = (*renderer)->draw("/Camera", 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
        held.push_back((*renderer)->pathAccumulated());
    }
    std::printf("  paths a pixel held after each pass:");
    for (uint32_t n : held) {
        std::printf(" %u", n);
    }
    std::printf("\n  converged: %s\n", (*renderer)->pathConverged() ? "yes" : "no");
    for (uint32_t pass = 0; pass < 8; ++pass) {
        CHECK(held[pass] == 4 * (pass + 1));
    }
    CHECK((*renderer)->pathConverged());

    // A camera somewhere else: a different frame, and the mean starts again.
    render::Camera moved = render::Camera::lookingAt({0.5, 0.25, 0.0}, {0.0, 0.0, -1.0});
    moved.lens.focal = 35.0;
    moved.lens.haperture = 24.576;
    if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
        FAIL(drawn.error().toString());
    }
    const uint32_t afterMove = (*renderer)->pathAccumulated();
    std::printf("  after the camera moved: %u\n", afterMove);
    CHECK(afterMove == 4);
    CHECK_FALSE((*renderer)->pathConverged());

    // And a setting that changes what a path finds: the same again.
    for (uint32_t pass = 0; pass < 3; ++pass) {
        if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
    }
    const uint32_t beforeSetting = (*renderer)->pathAccumulated();
    (*renderer)->setPathBounces(2);
    if (auto drawn = (*renderer)->draw(moved, 0.0, w, h, "rt"); !drawn) {
        FAIL(drawn.error().toString());
    }
    const uint32_t afterSetting = (*renderer)->pathAccumulated();
    std::printf("  before the bounce count changed: %u, after: %u\n", beforeSetting, afterSetting);
    CHECK(beforeSetting == 16);
    CHECK(afterSetting == 4);

    // An image, not a viewport: render() draws until the total is gathered.
    (*renderer)->setPathBounces(1);
    (*renderer)->setPathTotal(32);
    auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!image) FAIL(image.error().toString());
    std::printf("  render() with a total of 32 and 4 a pass left %u paths gathered\n",
                (*renderer)->pathAccumulated());
    CHECK((*renderer)->pathAccumulated() == 32);
    CHECK((*renderer)->pathConverged());

    // Adaptive: with a total no image would reach, render() ends when every
    // covered pixel's error is below the target -- the other gate.
    (*renderer)->setPathTotal(100000);
    (*renderer)->setPathAdaptive(true);
    (*renderer)->setPathError(0.1F);
    auto adaptive = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!adaptive) FAIL(adaptive.error().toString());
    std::printf("  adaptive at 10%% against a total of 100000: gathered %u a pixel at most, converged %s\n",
                (*renderer)->pathAccumulated(), (*renderer)->pathConverged() ? "yes" : "no");
    CHECK((*renderer)->pathConverged());
    CHECK((*renderer)->pathAccumulated() < 100000);
}

// The denoiser from the engine: athenea:denoise runs OIDN over a path traced
// frame once it has gathered athenea:pathTotal, in place. Two frames of the same
// stage and paths, one with the setting and one without, must differ once the
// total is reached -- and must not differ while it is not, which is the gate.
TEST_CASE("athenea:denoise runs once a path traced frame is gathered, and not before",
          "[usd][gpu][mesh][path][denoise]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    if (!technique::denoiserBuilt()) {
        SKIP("built without OIDN");
    }
    if (auto probe = technique::Denoiser::create(*gpu->library); !probe) {
        SKIP(std::string("no denoiser here: ") + probe.error().toString());
    }
    const fs::path path = scratch("rt_denoise.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def SphereLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 96;
    const uint32_t h = 72;
    const auto frame = [&](bool denoise, uint32_t total) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathBounces(1);
        (*renderer)->setPathTotal(total);
        (*renderer)->setDenoise(denoise);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const uint32_t words = w * h * 4;
    // Gathered in one pass: the denoiser runs, and the frame is not the mean.
    const gpu::Buffer plain = frame(false, 4);
    const gpu::Buffer denoised = frame(true, 4);
    auto changed = render::countDifferent(*gpu->library, plain, denoised, words);
    REQUIRE(changed);
    // Not yet gathered: the gate holds, and the two frames are the same frame.
    // render() draws until the total, so this half is one draw() -- four of
    // sixty-four paths -- read back as the host maps the colour output.
    const auto drawnOnce = [&](bool denoise) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathBounces(1);
        (*renderer)->setPathTotal(64);
        (*renderer)->setDenoise(denoise);
        if (auto drawn = (*renderer)->draw("/Camera", 0.0, w, h, "rt"); !drawn) {
            FAIL(drawn.error().toString());
        }
        CHECK((*renderer)->pathAccumulated() == 4);
        CHECK_FALSE((*renderer)->pathConverged());
        auto bytes = (*renderer)->mappedOutput("color");
        if (!bytes) FAIL(bytes.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = bytes->size();
        desc.elementBytes = 4;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, bytes->data());
        REQUIRE(buffer);
        return std::pair<gpu::Buffer, uint32_t>(*buffer, static_cast<uint32_t>(bytes->size() / 4));
    };
    const auto [plainEarly, earlyWords] = drawnOnce(false);
    const auto [gated, gatedWords] = drawnOnce(true);
    REQUIRE(earlyWords == gatedWords);
    auto unchanged = render::countDifferent(*gpu->library, plainEarly, gated, earlyWords);
    REQUIRE(unchanged);
    std::printf("  denoise at the total: %llu of %u words changed; before the total: %llu\n",
                static_cast<unsigned long long>(*changed), words, static_cast<unsigned long long>(*unchanged));
    CHECK(*changed > 0);
    CHECK(*unchanged == 0);
}

// The camera's exposure: UsdGeomCamera's `exposure`, in stops, reaches the
// engine through HdCamera and scales the composed frame by 2^stops -- exactly,
// since a power of two is an exponent bump in float. So the frame with
// exposure 1 authored must be the frame without it, doubled on the device.
TEST_CASE("a camera's exposure scales the frame by a power of two, exactly", "[usd][gpu][mesh][camera]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stageWith = [&](const char* name, const char* exposureLine) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        std::string stage = kSquareStage;
        // kSquareStage authors /Camera; the exposure goes in beside its focal length.
        const size_t at = stage.find("float focalLength");
        REQUIRE(at != std::string::npos);
        stage.insert(at, exposureLine);
        out << stage;
        return path;
    };
    const fs::path plainPath = stageWith("exposure_plain.usda", "");
    const fs::path stopPath = stageWith("exposure_one_stop.usda", "float exposure = 1\n        ");
    const uint32_t w = 96;
    const uint32_t h = 72;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    gpu::Buffer plain = frame(plainPath);
    const gpu::Buffer stop = frame(stopPath);
    // Double the plain frame on the device with the same kernel the engine
    // uses, and the two must be the same words.
    auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/technique/exposure", "applyExposure");
    if (!made) FAIL(made.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(plain.rhi());
            cursor["params"]["scale"].setData(2.0F);
            cursor["params"]["pixels"].setData(w * h);
        });
        REQUIRE(batch.submit(true));
    }
    auto differing = render::countDifferent(*gpu->library, plain, stop, w * h * 4);
    REQUIRE(differing);
    std::printf("  exposure 1 authored against the plain frame doubled: %llu of %u words differ\n",
                static_cast<unsigned long long>(*differing), w * h * 4);
    CHECK(*differing == 0);
}

// A UsdLux cylinder light through Hydra: radius and length as authored, the
// axis along the prim's x, lighting a Lambert plane as the closed form says.
TEST_CASE("a UsdLux cylinder light through Hydra lights a Lambert plane as the closed form says",
          "[usd][gpu][mesh][lights][cylinder]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("light_cylinder.usda");
    {
        std::ofstream out(path);
        out << kSquareStage
            << "def CylinderLight \"Key\"\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.3\n"
               "    float inputs:length = 1.4\n"
               "    float inputs:intensity = 1\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n"
               "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
    }
    const uint32_t w = 81;
    const uint32_t h = 61;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setLightSamples(65536);   // a one-sided curved emitter: sigma ~0.6% here, see the technique case
    auto image = (*renderer)->render("/Camera", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    test::dumpPpm("cylinder_light_hydra", image->rgba.data(), w, h);

    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    gpu::BufferDesc colourDesc;
    colourDesc.bytes = image->rgba.size() * sizeof(float);
    colourDesc.elementBytes = 16;
    auto colours = gpu::Buffer::create(*gpu->device, colourDesc, image->rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
    REQUIRE(colours);
    REQUIRE(depth);
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/lambert_irradiance", "lambertIrradiance");
    if (!check) FAIL(check.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 5, "worst");
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["colour"].setBinding(colours->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor p = cursor["plane"];
            p["kind"].setData(uint32_t{6});
            p["vertices"].setData(uint32_t{512});
            p["row0"].setData(toWorld.data(), sizeof(float) * 4);
            p["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            p["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            const float centre[4] = {0.0F, 0.0F, -3.0F, 0.0F};
            const float axisX[4] = {1.0F, 0.0F, 0.0F, 0.3F};
            const float axisY[4] = {0.0F, 1.0F, 0.0F, 1.4F};
            const float normal[4] = {0.0F, 0.0F, 1.0F, 0.8F};
            const float radiance[4] = {1.0F, 1.0F, 1.0F, 0.03F};
            const float none[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            p["centre"].setData(centre, sizeof(centre));
            p["axisX"].setData(axisX, sizeof(axisX));
            p["axisY"].setData(axisY, sizeof(axisY));
            p["normal"].setData(normal, sizeof(normal));
            p["radiance"].setData(radiance, sizeof(radiance));
            p["occluder"].setData(none, sizeof(none));
            p["occluderAxes"].setData(none, sizeof(none));
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[2] = {0, 0};
    float relative = 0.0F;
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(relative), &relative));
    std::printf("  cylinder light through Hydra: %u pixels, %u beyond 3%%, worst %.4f\n", c[0], c[1],
                double(relative));
    CHECK(c[0] > 2000);
    CHECK(c[1] == 0);
}

// A UsdLux IES profile through Hydra: a cutoff at 20 degrees, one inside and
// zero outside, on a small sphere light over the plane. Inside the cone the
// frame must be the frame without the profile, word for word; outside it
// must be black. A band about the cutoff is the interpolation's and skipped.
TEST_CASE("a UsdLux IES profile shapes a light through Hydra: lit inside its cone as without, black outside",
          "[usd][gpu][mesh][lights][ies]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path ies = scratch("cutoff20.ies");
    {
        std::ofstream out(ies);
        out << "IESNA:LM-63-2002\n[TEST] cutoff 20\nTILT=NONE\n1 1000 1 37 1 1 2 0 0 0\n1 1 0\n";
        for (int k = 0; k < 37; ++k) out << (k * 5) << (k == 36 ? "\n" : " ");
        out << "0\n";
        for (int k = 0; k < 37; ++k) out << (k * 5 <= 20 ? "1" : "0") << (k == 36 ? "\n" : " ");
    }
    const auto stageWith = [&](const char* name, bool shaped) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << kSquareStage << "def SphereLight \"Key\"";
        if (shaped) out << " (\n    prepend apiSchemas = [\"ShapingAPI\"]\n)";
        out << "\n{\n"
               "    bool inputs:shadow:enable = 0\n"
               "    float inputs:radius = 0.05\n"
               "    float inputs:intensity = 400\n"
               "    bool inputs:normalize = 0\n"
               "    color3f inputs:color = (1, 1, 1)\n";
        if (shaped) out << "    asset inputs:shaping:ies:file = @" << ies.string() << "@\n";
        out << "    double3 xformOp:translate = (0, 0, -3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"Diffuse\"\n        {\n"
               "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
               "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:out\n        }\n    }\n}\n";
        return path;
    };
    const uint32_t w = 161;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(64);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage plain = frame(stageWith("ies_plain.usda", false));
    const usd::StageImage shaped = frame(stageWith("ies_cutoff.usda", true));
    test::dumpPpm("ies_cutoff_hydra", shaped.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = plain.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto with = gpu::Buffer::create(*gpu->device, desc, shaped.rgba.data());
    auto without = gpu::Buffer::create(*gpu->device, desc, plain.rgba.data());
    auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, shaped.depth, "depth");
    REQUIRE(with);
    REQUIRE(without);
    REQUIRE(depth);
    render::Camera camera;
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
    auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/test/ies_check", "iesCutoff");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 6, "ies.counts");
    gpu::Buffer worst = test::uintBuffer(*gpu->device, 2, "ies.worst");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["with"].setBinding(with->rhi());
            cursor["without"].setBinding(without->rhi());
            cursor["depth"].setBinding(depth->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["worst"].setBinding(worst.rhi());
            technique::setCamera(cursor["camera"], projection, w, h);
            rhi::ShaderCursor c = cursor["check"];
            const float centre[4] = {0.0F, 0.0F, -3.0F, 20.0F};
            c["lightCentre"].setData(centre, sizeof(centre));
            c["row0"].setData(toWorld.data(), sizeof(float) * 4);
            c["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            c["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[6] = {};
    float e[2] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    REQUIRE(worst.read(*gpu->device, 0, sizeof(e), e));
    std::printf("  IES cutoff through Hydra: inside the cone %u of %u pixels differ from the unshaped frame; outside "
                "it %u of %u are lit (largest %.3e, farthest at %.1f degrees)\n",
                c[2], c[4], c[3], c[5], double(e[0]), double(e[1]));
    CHECK(c[4] > 100);
    CHECK(c[5] > 100);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
}


// Light instancing through Hydra: a PointInstancer whose prototype is a
// sphere light, against the same three lights authored one by one, on the
// square. The frames must agree to what a differently ordered light table
// allows -- the instancer's copies are the authored lights, placed by the
// device.
TEST_CASE("a PointInstancer of lights lights the scene as its instances authored one by one",
          "[usd][gpu][mesh][lights][instancing]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const char* positions[3] = {"(-1, 0.3, -3)", "(0, -0.5, -2.5)", "(1.2, 0.2, -3.5)"};
    const std::string light =
        "    bool inputs:shadow:enable = 0\n"
        "    float inputs:radius = 0.1\n"
        "    float inputs:intensity = 60\n"
        "    bool inputs:normalize = 0\n"
        "    color3f inputs:color = (1, 0.9, 0.8)\n";
    const std::string materials =
        "def Scope \"Materials\"\n{\n"
        "    def Material \"Mat\"\n    {\n"
        "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
        "        def Shader \"Surface\"\n        {\n"
        "            uniform token info:id = \"ND_surface\"\n"
        "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
        "            token outputs:out\n        }\n"
        "        def Shader \"Diffuse\"\n        {\n"
        "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
        "            float inputs:roughness = 0\n"
        "            token outputs:out\n        }\n    }\n}\n";
    const fs::path instanced = scratch("light_instancer.usda");
    {
        std::ofstream out(instanced);
        out << kSquareStage << "def PointInstancer \"Many\"\n{\n"
            << "    rel prototypes = [</Many/Prototypes/Key>]\n"
            << "    int[] protoIndices = [0, 0, 0]\n"
            << "    point3f[] positions = [" << positions[0] << ", " << positions[1] << ", " << positions[2] << "]\n"
            << "    def Scope \"Prototypes\"\n    {\n"
            << "        def SphereLight \"Key\"\n        {\n" << light << "        }\n    }\n}\n" << materials;
    }
    const fs::path authored = scratch("light_authored.usda");
    {
        std::ofstream out(authored);
        out << kSquareStage;
        for (int k = 0; k < 3; ++k) {
            out << "def SphereLight \"Key" << k << "\"\n{\n" << light
                << "    double3 xformOp:translate = " << positions[k] << "\n"
                << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        }
        out << materials;
    }
    const uint32_t w = 161;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(64);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage a = frame(instanced);
    const usd::StageImage b = frame(authored);
    test::dumpPpm("light_instancer_hydra", a.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = a.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto bufferA = gpu::Buffer::create(*gpu->device, desc, a.rgba.data());
    auto bufferB = gpu::Buffer::create(*gpu->device, desc, b.rgba.data());
    REQUIRE(bufferA);
    REQUIRE(bufferB);
    auto diff = render::compareHdr(*gpu->library, *bufferA, *bufferB, w, h);
    REQUIRE(diff);
    std::vector<float> blank(a.rgba.size(), 0.0F);
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto drawn = render::compareHdr(*gpu->library, *bufferA, *blankBuffer, w, h);
    REQUIRE(drawn);
    std::printf("  PointInstancer of sphere lights against authored lights: relMSE %.2e, max relative %.2e "
                "(against blank: relMSE %.2e)\n",
                diff->relMse, diff->maxRelative, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(diff->relMse < 1e-8);
}

// UsdGeomCamera's diaphragm through Hydra: fStop and focusDistance reach the
// path tracer's own primary rays. With the square in focus the frame is the
// pinhole camera's, exactly; with the focus in front of it the square's
// edges blur, which the frame shows as a difference. Distortion is checked
// to the pixel in the technique; here it only has to arrive.
TEST_CASE("a UsdGeomCamera's fStop and focusDistance reach the path tracer through Hydra",
          "[usd][gpu][mesh][path][camera]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const std::string rest =
        "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n"
        "def Scope \"Materials\"\n{\n"
        "    def Material \"Mat\"\n    {\n"
        "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
        "        def Shader \"Surface\"\n        {\n"
        "            uniform token info:id = \"ND_surface\"\n"
        "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
        "            token outputs:out\n        }\n"
        "        def Shader \"Diffuse\"\n        {\n"
        "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
        "            float inputs:roughness = 0\n"
        "            token outputs:out\n        }\n    }\n}\n";
    const auto stageWith = [&](const char* name, const char* lens) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        // kSquareStage's camera, with the lens authored inside it.
        std::string text = kSquareStage;
        const size_t at = text.find("    float2 clippingRange");
        REQUIRE(at != std::string::npos);
        text.insert(at, lens);
        out << text << rest;
        return path;
    };
    // An even width: with the square's triangles meeting on a diagonal
    // through the frame's centre, an odd width put pixel centres exactly on
    // that shared edge, and one lens ray in four converging on such a point
    // fell through the seam.
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(4);
        (*renderer)->setPathTotal(4);
        (*renderer)->setPathBounces(0);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        return *image;
    };
    const usd::StageImage pinhole = frame(stageWith("lens_pinhole.usda", ""));
    const usd::StageImage inFocus =
        frame(stageWith("lens_focus5.usda", "    float fStop = 8\n    float focusDistance = 5\n"));
    const usd::StageImage outOfFocus =
        frame(stageWith("lens_focus2.usda", "    float fStop = 8\n    float focusDistance = 2.5\n"));
    const usd::StageImage distorted = frame(stageWith(
        "lens_distorted.usda", "    token lensDistortion:type = \"standard\"\n    float lensDistortion:k1 = 0.3\n"));
    test::dumpPpm("dof_hydra", outOfFocus.rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = pinhole.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto a = gpu::Buffer::create(*gpu->device, desc, pinhole.rgba.data());
    auto b = gpu::Buffer::create(*gpu->device, desc, inFocus.rgba.data());
    auto c = gpu::Buffer::create(*gpu->device, desc, outOfFocus.rgba.data());
    auto d = gpu::Buffer::create(*gpu->device, desc, distorted.rgba.data());
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(c);
    REQUIRE(d);
    auto focused = render::compareHdr(*gpu->library, *a, *b, w, h);
    auto blurred = render::compareHdr(*gpu->library, *a, *c, w, h);
    auto bent = render::compareHdr(*gpu->library, *a, *d, w, h);
    REQUIRE(focused);
    REQUIRE(blurred);
    REQUIRE(bent);
    std::printf("  UsdGeomCamera lens through Hydra: in focus relMSE %.2e (p99 relative %.2e, max %.2e) against the "
                "pinhole; out of focus %.2e; distorted %.2e\n",
                focused->relMse, focused->p99Relative, focused->maxRelative, blurred->relMse, bent->relMse);
    CHECK(focused->relMse < 1e-8);
    CHECK(blurred->relMse > 1e-3);
    CHECK(bent->relMse > 1e-3);
}

// UsdLux's DistantLight, in its own words (usdLux/schema.usda, LightAPI's
// `intensity` and `normalize`, DistantLight's `angle`): `intensity` is the
// radiance of the disc the light covers, and `normalize` divides it by
// sizeFactor = pi sin^2(angle / 2) (for half angles up to 90 degrees), so
// that the intensity becomes the illuminance on a surface facing the light.
// An angle of 0 is a parallel light whose irradiance is the intensity. A
// white Lambert square faces the sun at normal incidence; its radiance is
// E / pi, measured over the square's middle by `imageStats`, in the raster's
// shading and in the path tracer, for four angles with normalize off and on.
TEST_CASE("a DistantLight lays the irradiance UsdLux's intensity, angle and normalize say",
          "[usd][gpu][mesh][path][light][distant]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const std::string material =
        "def Scope \"Materials\"\n{\n"
        "    def Material \"Mat\"\n    {\n"
        "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
        "        def Shader \"Surface\"\n        {\n"
        "            uniform token info:id = \"ND_surface\"\n"
        "            token inputs:bsdf.connect = </Materials/Mat/Diffuse.outputs:out>\n"
        "            token outputs:out\n        }\n"
        "        def Shader \"Diffuse\"\n        {\n"
        "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
        "            color3f inputs:color = (1, 1, 1)\n"
        "            float inputs:roughness = 0\n"
        "            token outputs:out\n        }\n    }\n}\n";
    struct Case {
        float degrees;
        bool  normalize;
    };
    const std::array<Case, 8> cases{Case{0.0F, false},  Case{0.0F, true},  Case{0.53F, false}, Case{0.53F, true},
                                    Case{20.0F, false}, Case{20.0F, true}, Case{90.0F, false}, Case{90.0F, true}};
    const float intensity = 2.0F;
    const uint32_t w = 96, h = 72;
    for (const Case& c : cases) {
        const fs::path path = scratch("distant_units.usda");
        {
            std::ofstream out(path);
            out << kSquareStage << material << "def DistantLight \"Sun\"\n{\n    float inputs:intensity = " << intensity
                << "\n    float inputs:angle = " << c.degrees << "\n    bool inputs:normalize = " << (c.normalize ? 1 : 0)
                << "\n    bool inputs:shadow:enable = 0\n}\n";
        }
        // The schema's sizeFactor, a constant of the case: the expected
        // radiance of the white Lambert square facing the light is E / pi.
        const double half = 0.5 * double(c.degrees) * 3.14159265358979 / 180.0;
        const double sizeFactor = half > 0.0 ? 3.14159265358979 * std::sin(half) * std::sin(half) : 1.0;
        const double irradiance = double(intensity) * (c.normalize ? 1.0 : sizeFactor);
        const double expected = irradiance / 3.14159265358979;
        for (const char* technique : {"raster", "rt"}) {
            auto renderer = usd::StageRenderer::open(path);
            if (!renderer) FAIL(renderer.error().toString());
            (*renderer)->setLightSamples(64);
            (*renderer)->setPathSamples(64);
            (*renderer)->setPathTotal(64);
            (*renderer)->setPathBounces(0);
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(buffer);
            // The square covers the middle 57% of the frame: the middle 40%.
            auto stats =
                render::imageStats(*gpu->library, *buffer, w, h, w * 3 / 10, h * 3 / 10, w * 7 / 10, h * 7 / 10);
            if (!stats) FAIL(stats.error().toString());
            const double got = stats->mean[1];
            const double relative = std::abs(got - expected) / expected;
            std::printf("  DistantLight angle %5.2f normalize %d, %-6s: mean %.6e, expected %.6e (E %.6e), off %.2e\n",
                        double(c.degrees), c.normalize ? 1 : 0, technique, got, expected, irradiance, relative);
            CHECK(stats->pixels > 1000);
            CHECK(relative < 0.01);
        }
    }
}

// A mesh whose points are time sampled, through Hydra: at the second time
// Hydra hands the delegate new points and the same topology, the engine
// rebuilds the mesh under the same key, and the scene takes it as a
// deformation -- the pools' generation stands while their positions revision
// rises -- so the hardware and compute structures are refit. Checked by the
// three routes agreeing on the deformed frame, and by that frame differing
// from the first time's.
TEST_CASE("time sampled points deform a mesh in place through Hydra, and every route sees the deformation",
          "[usd][gpu][mesh][deformation][refit]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries, to compare all three routes");
    }
    const int n = 12;
    const auto grid = [&](double amplitude) {
        std::string out = "[";
        for (int y = 0; y <= n; ++y) {
            for (int x = 0; x <= n; ++x) {
                const double fx = static_cast<double>(x) / n * 4.0 - 2.0;
                const double fy = static_cast<double>(y) / n * 3.0 - 1.5;
                const double fz = -6.0 + amplitude * std::sin(fx * 1.9) * std::cos(fy * 2.3);
                char buffer[96];
                std::snprintf(buffer, sizeof(buffer), "%s(%.4f, %.4f, %.4f)", (x == 0 && y == 0) ? "" : ", ", fx, fy,
                              fz);
                out += buffer;
            }
        }
        return out + "]";
    };
    std::string counts = "[";
    std::string indices = "[";
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const int p = y * (n + 1) + x;
            char buffer[96];
            std::snprintf(buffer, sizeof(buffer), "%s4", (x == 0 && y == 0) ? "" : ", ");
            counts += buffer;
            // Counter-clockwise seen from +z, where the camera is: single sided
            // by USD's default, the sheet faces it.
            std::snprintf(buffer, sizeof(buffer), "%s%d, %d, %d, %d", (x == 0 && y == 0) ? "" : ", ", p, p + 1,
                          p + n + 2, p + n + 1);
            indices += buffer;
        }
    }
    counts += "]";
    indices += "]";
    const fs::path path = scratch("deforming.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Sheet\"\n{\n"
            << "    int[] faceVertexCounts = " << counts << "\n"
            << "    int[] faceVertexIndices = " << indices << "\n"
            << "    point3f[] points.timeSamples = {\n        0: " << grid(0.0) << ",\n        1: " << grid(1.2)
            << ",\n    }\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.6, 0.7, 0.3)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto frame = [&](double time, const char* route) {
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer flat = frame(0.0, "raster");
    const gpu::Buffer flatRays = frame(0.0, "rays");
    const uint64_t generation = (*renderer)->meshGeneration();
    const uint64_t revision = (*renderer)->meshPositionsRevision();
    const gpu::Buffer bentRaster = frame(1.0, "raster");
    CHECK((*renderer)->meshGeneration() == generation);
    CHECK((*renderer)->meshPositionsRevision() > revision);
    const gpu::Buffer bentRays = frame(1.0, "rays");
    const gpu::Buffer bentBvh = frame(1.0, "bvh");
    CHECK((*renderer)->meshGeneration() == generation);
    auto moved = render::compareImages(*gpu->library, flat, bentRaster, w, h);
    auto baseline = render::compareImages(*gpu->library, flat, flatRays, w, h);
    auto rays = render::compareImages(*gpu->library, bentRaster, bentRays, w, h);
    auto bvh = render::compareImages(*gpu->library, bentRaster, bentBvh, w, h);
    REQUIRE(moved);
    REQUIRE(baseline);
    REQUIRE(rays);
    REQUIRE(bvh);
    // The routes differ along triangle edges (the rasteriser's coverage rule
    // against a ray at the pixel's centre); the interior agrees, which the
    // technique's check counts exactly. Here the deformed frame's edge
    // disagreement must be no more than the flat frame's kind: a stale
    // structure would disagree over the whole sheet.
    std::printf("  time 1 against time 0: %llu pixels beyond 2 (max %u); raster against rays: %llu beyond 2 at "
                "time 1, %llu at time 0; against the compute BVH %llu; generation %llu, positions revision %llu "
                "-> %llu\n",
                static_cast<unsigned long long>(moved->over2), moved->max,
                static_cast<unsigned long long>(rays->over2), static_cast<unsigned long long>(baseline->over2),
                static_cast<unsigned long long>(bvh->over2), static_cast<unsigned long long>(generation),
                static_cast<unsigned long long>(revision),
                static_cast<unsigned long long>((*renderer)->meshPositionsRevision()));
    CHECK(moved->over2 > uint64_t{w} * h / 20);
    CHECK(rays->over2 <= 2 * baseline->over2 + 50);
    CHECK(bvh->over2 <= 2 * baseline->over2 + 50);
    CHECK(rays->over2 < uint64_t{w} * h / 50);
    CHECK(bvh->over2 < uint64_t{w} * h / 50);
}

// Motion blur through Hydra: a square sliding between two frames under a
// camera whose shutter is open about the frame, path traced in eight
// slices. The blurred frame differs from the frame the same stage gives
// with the shutter closed; and the same with the square's points sliding
// instead of its transform. The staircase itself is checked in the
// technique; here the shutter, the samples and the buckets have to arrive.
TEST_CASE("a camera's shutter blurs a moving mesh through Hydra, rigid and deforming",
          "[usd][gpu][mesh][path][motion]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const std::string sun =
        "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    const auto stageWith = [&](const char* name, bool shutter, bool deform) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n";
        if (deform) {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0: [(-1.5, -1, -5), (-0.5, -1, -5), (-0.5, 1, -5), (-1.5, 1, -5)],\n"
                   "        1: [(0.5, -1, -5), (1.5, -1, -5), (1.5, 1, -5), (0.5, 1, -5)],\n    }\n";
        } else {
            out << "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
                   "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        }
        out << "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n" << sun;
        return path;
    };
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    for (const bool deform : {false, true}) {
        const gpu::Buffer sharp = frame(stageWith(deform ? "blur_points_closed.usda" : "blur_closed.usda", false, deform));
        const gpu::Buffer blurred = frame(stageWith(deform ? "blur_points_open.usda" : "blur_open.usda", true, deform));
        auto diff = render::compareHdr(*gpu->library, sharp, blurred, w, h);
        REQUIRE(diff);
        std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
        gpu::BufferDesc desc;
        desc.bytes = blank.size() * sizeof(float);
        desc.elementBytes = 16;
        auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
        REQUIRE(blankBuffer);
        auto drawn = render::compareHdr(*gpu->library, blurred, *blankBuffer, w, h);
        REQUIRE(drawn);
        std::printf("  %s through Hydra: shutter open against closed relMSE %.2e (blurred frame against blank %.2e)\n",
                    deform ? "sliding points" : "sliding transform", diff->relMse, drawn->relMse);
        CHECK(drawn->relMse > 1.0);
        CHECK(diff->relMse > 1e-2);
    }
}

// A PointInstancer whose positions move under the shutter blurs its
// instances as the same squares authored one by one, each sliding by its
// own transform: the set's chain is composed at the two samples and its
// records copied per shutter slice between them, as a single instance's
// are. And the blur is there: against the shutter closed, the frames differ.
TEST_CASE("a moving PointInstancer's instances blur as the same prims authored one by one",
          "[usd][gpu][mesh][path][motion][instancing]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const char* square = "    int[] faceVertexCounts = [4]\n"
                         "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                         "    point3f[] points = [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]\n"
                         "    uniform token subdivisionScheme = \"none\"\n"
                         "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n";
    const auto stage = [&](const char* name, bool instanced, bool shutter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n";
        if (instanced) {
            out << "def PointInstancer \"Squares\"\n{\n"
                   "    int[] protoIndices = [0, 0]\n"
                   "    point3f[] positions.timeSamples = {\n"
                   "        0: [(-1.5, -0.5, -5), (0.5, 0.6, -6)],\n"
                   "        1: [(-0.5, -0.5, -5), (1.5, 0.9, -6)],\n    }\n"
                   "    rel prototypes = </Squares/Proto>\n"
                   "    def Mesh \"Proto\"\n    {\n" << square << "    }\n}\n";
        } else {
            const char* from[] = {"(-1.5, -0.5, -5)", "(0.5, 0.6, -6)"};
            const char* to[] = {"(-0.5, -0.5, -5)", "(1.5, 0.9, -6)"};
            for (int k = 0; k < 2; ++k) {
                out << "def Mesh \"Square" << k << "\"\n{\n" << square
                    << "    double3 xformOp:translate.timeSamples = {\n        0: " << from[k] << ",\n        1: " << to[k]
                    << ",\n    }\n"
                       "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
            }
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer instancedBlur = frame(stage("instancer_blur.usda", true, true));
    const gpu::Buffer singlesBlur = frame(stage("instancer_blur_singles.usda", false, true));
    const gpu::Buffer instancedSharp = frame(stage("instancer_sharp.usda", true, false));
    auto same = render::compareHdr(*gpu->library, instancedBlur, singlesBlur, w, h);
    auto blurred = render::compareHdr(*gpu->library, instancedBlur, instancedSharp, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto drawn = render::compareHdr(*gpu->library, instancedBlur, *empty, w, h);
    REQUIRE(drawn);
    std::printf("  a moving instancer against the same squares one by one: relMSE %.2e (max relative %.2e); against "
                "the shutter closed %.2e; against blank %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}

// A camera that moves under the shutter: everything it sees blurs as if the
// scene moved the other way. Two squares under a sun that lights by
// direction alone (no shadows), so where a square is does not change its
// shade -- the camera sliding +x over still squares must draw what a still
// camera draws of the squares sliding -x, slice by slice. And the blur is
// there: against the same camera with the shutter closed, the frames differ.
TEST_CASE("a camera moving under the shutter blurs the frame as the scene moving the other way does",
          "[usd][gpu][mesh][path][motion][camera]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const char* square = "    int[] faceVertexCounts = [4]\n"
                         "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                         "    point3f[] points = [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]\n"
                         "    uniform token subdivisionScheme = \"none\"\n"
                         "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n";
    // cameraMoves: the camera slides from x 0 to 1 over frames 0..1 and the
    // squares stand; otherwise the camera stands and the squares slide 0..-1.
    const auto stage = [&](const char* name, bool cameraMoves, bool shutter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n";
        const char* at[] = {"(-0.8, -0.3, -5)", "(0.9, 0.5, -6)"};
        const char* moved[] = {"(-1.8, -0.3, -5)", "(-0.1, 0.5, -6)"};
        for (int k = 0; k < 2; ++k) {
            out << "def Mesh \"Square" << k << "\"\n{\n" << square;
            if (cameraMoves) {
                out << "    double3 xformOp:translate = " << at[k] << "\n";
            } else {
                out << "    double3 xformOp:translate.timeSamples = {\n        0: " << at[k] << ",\n        1: " << moved[k]
                    << ",\n    }\n";
            }
            out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (cameraMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 0, 0),\n        1: (1, 0, 0),\n    }\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        }
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160, h = 120;
    // Drawn at frame 0.5 with the camera's frame position subtracted: the
    // still-camera stage at 0.5 has the squares half way along, and the
    // moving camera's frame view is half way along too.
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer cameraBlur = frame(stage("camera_blur.usda", true, true));
    const gpu::Buffer sceneBlur = frame(stage("camera_blur_scene.usda", false, true));
    const gpu::Buffer cameraSharp = frame(stage("camera_sharp.usda", true, false));
    auto same = render::compareHdr(*gpu->library, cameraBlur, sceneBlur, w, h);
    auto blurred = render::compareHdr(*gpu->library, cameraBlur, cameraSharp, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::vector<float> blank(size_t{w} * h * 4, 0.0F);
    auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
    REQUIRE(empty);
    auto drawn = render::compareHdr(*gpu->library, cameraBlur, *empty, w, h);
    REQUIRE(drawn);
    std::printf("  a moving camera against the scene moving the other way: relMSE %.2e (max relative %.2e); "
                "against the shutter closed %.2e; against blank %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse, drawn->relMse);
    CHECK(drawn->relMse > 1.0);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}

// A CLOUD MOVING UNDER THE SHUTTER BLURS AS THE CAMERA MOVING THE OTHER WAY
// DOES. The rasteriser smears a gaussian along what the shutter changes of
// object-to-view (`SplatInstance::viewStep`), and that has two sources: the
// prim's own transform, sampled over the shutter as a mesh's is, and the
// camera's. A glass sparrow crossing the frame drew sharp while only its
// wings blurred -- nothing ever filled the step in. The same relative motion
// arranged either way must give the same frame, and a different one from the
// shutter closed.
TEST_CASE("a cloud moving under the shutter blurs as the camera moving the other way does",
          "[usd][gpu][splat][motion][camera]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const auto stage = [&](const char* name, bool cameraMoves, bool shutter) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def ParticleField3DGaussianSplat \"Cloud\"\n{\n    point3f[] positions = [";
        const int side = 6;
        for (int k = 0; k < side * side; ++k) {
            out << (k ? ", " : "") << "(" << (-0.5 + (k % side) * 0.2) << ", " << (-0.5 + (k / side) * 0.2)
                << ", 0)";
        }
        out << "]\n    quatf[] orientations = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
        out << "]\n    float3[] scales = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(0.04, 0.04, 0.04)";
        out << "]\n    float[] opacities = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "0.9";
        out << "]\n    uniform int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(1.5, 1.2, 0.8)";
        out << "]\n";
        if (cameraMoves) {
            out << "    double3 xformOp:translate = (0, 0, -4)\n";
        } else {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 0, -4),\n        1: (-1, 0, -4),\n"
                   "    }\n";
        }
        out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (cameraMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 0, 0),\n        1: (1, 0, 0),\n    }\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        }
        if (shutter) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        }
        out << "}\n";
        return path;
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        // Twice: a shutter reaches the prims on the frame after it is read.
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "raster");
        image = (*renderer)->render("/Camera", 0.5, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer cloudBlur = frame(stage("cloud_blur.usda", false, true));
    const gpu::Buffer cameraBlur = frame(stage("cloud_blur_camera.usda", true, true));
    const gpu::Buffer cloudSharp = frame(stage("cloud_sharp.usda", false, false));
    auto same = render::compareHdr(*gpu->library, cloudBlur, cameraBlur, w, h);
    auto blurred = render::compareHdr(*gpu->library, cloudBlur, cloudSharp, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::printf("  a cloud moving against the camera moving the other way: relMSE %.2e; against the shutter "
                "closed %.2e\n",
                same->relMse, blurred->relMse);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->relMse < blurred->relMse * 0.01);
}

// A light that moves under the shutter: its light and its shadows move with
// it, slice by slice. The same instants arranged the other way round -- the
// light still and everything else (the camera, a floor, an occluder)
// sliding the opposite way -- are the same relative scene at every time, so
// the two frames must agree; and against the light standing still, they
// must not.
TEST_CASE("a light moving under the shutter lights and shadows as the scene moving the other way does",
          "[usd][gpu][mesh][path][motion][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    // LightMoves: the light slides x 0 -> 2 and all else stands at x 0.
    // SceneMoves: the light stands at x 1 and the camera, floor and occluder
    // slide x 1 -> -1, so that everything less the light is -2t either way.
    // Still: the light at its frame position, x 1, nothing moving.
    // InstancedMoves: LightMoves with the light a PointInstancer's prototype,
    // its one position sliding x 0 -> 2 instead of the light's transform.
    enum class Arrangement { LightMoves, SceneMoves, Still, InstancedMoves };
    const auto stage = [&](const char* name, Arrangement arrangement) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        const bool sceneMoves = arrangement == Arrangement::SceneMoves;
        const auto slide = [&](const char* at, const char* to) {
            std::string text;
            if (sceneMoves) {
                text = std::string("    double3 xformOp:translate.timeSamples = {\n        0: ") + at + ",\n        1: " + to +
                       ",\n    }\n";
            } else {
                text = "    double3 xformOp:translate = (0, 0, 0)\n";
            }
            return text + "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        };
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-8, -1, 0), (8, -1, 0), (8, -1, -12), (-8, -1, -12)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
            << slide("(1, 0, 0)", "(-1, 0, 0)") << "}\n"
            << "def Mesh \"Occluder\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, 0.2, -4.5), (0.5, 0.2, -4.5), (0.5, 0.2, -5.5), (-0.5, 0.2, -5.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
            << slide("(1, 0, 0)", "(-1, 0, 0)") << "}\n";
        if (arrangement == Arrangement::InstancedMoves) {
            out << "def PointInstancer \"Many\"\n{\n"
                   "    rel prototypes = [</Many/Prototypes/Bulb>]\n"
                   "    int[] protoIndices = [0]\n"
                   "    point3f[] positions.timeSamples = {\n        0: [(0, 2, -5)],\n        1: [(2, 2, -5)],\n    }\n"
                   "    def Scope \"Prototypes\"\n    {\n"
                   "        def SphereLight \"Bulb\"\n        {\n"
                   "            float inputs:intensity = 30\n            float inputs:radius = 0.2\n"
                   "        }\n    }\n}\n";
        } else {
            out << "def SphereLight \"Bulb\"\n{\n    float inputs:intensity = 30\n    float inputs:radius = 0.2\n";
            if (arrangement == Arrangement::LightMoves) {
                out << "    double3 xformOp:translate.timeSamples = {\n        0: (0, 2, -5),\n        1: (2, 2, -5),\n    }\n";
            } else {
                out << "    double3 xformOp:translate = (1, 2, -5)\n";
            }
            out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double shutter:open = -0.25\n    double shutter:close = 0.25\n";
        if (sceneMoves) {
            out << "    double3 xformOp:translate.timeSamples = {\n        0: (1, 1, 1),\n        1: (-1, 1, 1),\n    }\n";
        } else {
            out << "    double3 xformOp:translate = (0, 1, 1)\n";
        }
        out << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 128, h = 96;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(128);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer lightMoves = frame(stage("light_moves.usda", Arrangement::LightMoves));
    const gpu::Buffer sceneMoves = frame(stage("light_moves_scene.usda", Arrangement::SceneMoves));
    const gpu::Buffer still = frame(stage("light_still.usda", Arrangement::Still));
    const gpu::Buffer instancedMoves = frame(stage("light_moves_instanced.usda", Arrangement::InstancedMoves));
    auto same = render::compareHdr(*gpu->library, lightMoves, sceneMoves, w, h);
    auto moved = render::compareHdr(*gpu->library, lightMoves, still, w, h);
    auto instanced = render::compareHdr(*gpu->library, instancedMoves, lightMoves, w, h);
    REQUIRE(same);
    REQUIRE(moved);
    REQUIRE(instanced);
    std::printf("  an instancer moving its light against the light moving: relMSE %.2e (max relative %.2e)\n",
                instanced->relMse, instanced->maxRelative);
    CHECK(instanced->relMse < moved->relMse / 1000.0);
    std::printf("  a moving light against the scene moving the other way: relMSE %.2e (p99 relative %.2e); against the "
                "light still %.2e\n",
                same->relMse, same->p99Relative, moved->relMse);
    // The blur is a soft shadow sweeping part of the floor: small over the
    // frame (2.8e-4), and real -- with the light's samples withheld from the
    // kernel, the moving light drew the still frame to 3.7e-13 and parted
    // from the other arrangement by 3.1e-4.
    CHECK(moved->relMse > 1e-5);
    CHECK(same->relMse < moved->relMse / 1000.0);
}

// A shutter opened after the first frame: the prims already synced must
// sample again about it. The same renderer draws a sliding square with the
// shutter closed, the camera's shutter is then authored open through the
// layer the stage shares, and the next frame must be the frame a renderer
// opened on the edited stage draws -- not the sharp one again.
TEST_CASE("a shutter opened after the first frame blurs the next as a stage authored so does",
          "[usd][gpu][mesh][path][motion][shutter]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("shutter_opened.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
               "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    }
    const uint32_t w = 160, h = 120;
    const auto settle = [](usd::StageRenderer& r) {
        r.setPathSamples(64);
        r.setPathTotal(64);
        r.setPathBounces(0);
        r.setMotionBuckets(8);
    };
    const auto upload = [&](const usd::StageImage& image) {
        gpu::BufferDesc desc;
        desc.bytes = image.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image.rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    settle(**renderer);
    auto sharp = (*renderer)->render("/Camera", 0.5, w, h, "rt");
    if (!sharp) FAIL(sharp.error().toString());
    // The shutter, authored on the layer both stages share.
    {
        UsdStageRefPtr editing = UsdStage::Open(path.string());
        REQUIRE(editing);
        UsdGeomCamera camera(editing->GetPrimAtPath(SdfPath("/Camera")));
        REQUIRE(camera);
        camera.GetShutterOpenAttr().Set(-0.25);
        camera.GetShutterCloseAttr().Set(0.25);
    }
    auto edited = (*renderer)->render("/Camera", 0.5, w, h, "rt");
    if (!edited) FAIL(edited.error().toString());
    auto fresh = usd::StageRenderer::open(path);
    if (!fresh) FAIL(fresh.error().toString());
    settle(**fresh);
    auto authored = (*fresh)->render("/Camera", 0.5, w, h, "rt");
    if (!authored) FAIL(authored.error().toString());
    const gpu::Buffer a = upload(*edited);
    const gpu::Buffer b = upload(*authored);
    const gpu::Buffer c = upload(*sharp);
    auto same = render::compareHdr(*gpu->library, a, b, w, h);
    auto blurred = render::compareHdr(*gpu->library, b, c, w, h);
    REQUIRE(same);
    REQUIRE(blurred);
    std::printf("  shutter opened after a frame: the next frame against the authored stage's relMSE %.2e (max relative "
                "%.2e); the authored blur against the sharp frame %.2e\n",
                same->relMse, same->maxRelative, blurred->relMse);
    CHECK(blurred->relMse > 1e-2);
    CHECK(same->maxRelative < 1e-3);
}

// Velocities: a mesh that authors one sample of points and `velocities`
// must blur as one that authors the two samples those velocities reach --
// UsdGeom's velocity interpolation, resolved by hdsi's scene index ahead
// of the delegate, so the shutter samples read the same either way.
TEST_CASE("authored velocities blur a mesh as the equivalent time samples do", "[usd][gpu][mesh][path][motion][velocity]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    // 24 time codes a second: a velocity of 48 units a second slides the
    // square 2 units a frame, from x -1 at frame 0 to x 1 at frame 1. The
    // velocity stage authors its one sample at the frame drawn, 0.5: the
    // scene index extrapolates from the sample the frame reads.
    const auto stageWith = [&](const char* name, bool velocities) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n"
               "    timeCodesPerSecond = 24\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n";
        if (velocities) {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0.5: [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)],\n    }\n"
                   "    vector3f[] velocities.timeSamples = {\n"
                   "        0.5: [(48, 0, 0), (48, 0, 0), (48, 0, 0), (48, 0, 0)],\n    }\n";
        } else {
            out << "    point3f[] points.timeSamples = {\n"
                   "        0: [(-1.5, -1, -5), (-0.5, -1, -5), (-0.5, 1, -5), (-1.5, 1, -5)],\n"
                   "        1: [(0.5, -1, -5), (1.5, -1, -5), (1.5, 1, -5), (0.5, 1, -5)],\n    }\n";
        }
        out << "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double shutter:open = -0.25\n    double shutter:close = 0.25\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
        return path;
    };
    const uint32_t w = 160;
    const uint32_t h = 121;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(64);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer sampled = frame(stageWith("velocity_samples.usda", false));
    const gpu::Buffer velocities = frame(stageWith("velocity_authored.usda", true));
    auto diff = render::compareHdr(*gpu->library, sampled, velocities, w, h);
    REQUIRE(diff);
    std::printf("  velocities against the equivalent samples: relMSE %.2e, max relative %.2e\n", diff->relMse,
                diff->maxRelative);
    CHECK(diff->relMse < 1e-6);
}

// Skinning through Hydra: a square bound to one joint of a two-joint
// skeleton whose animation slides that joint. usdSkelImaging hands the
// delegate the skinning as ext computation prims; the engine runs them on
// the device (geom::Skinner) and builds the mesh from the skinned points.
// With every weight on the sliding joint the skinned square is the square
// authored with that slide as its transform -- to the pixel, both routes --
// and at the rest pose it is the square as authored. The skinned mesh keeps
// its topology key across frames, so the animation is a refit, not a
// repack.
TEST_CASE("a skeleton's animation skins a mesh on the device, as the authored transform draws it",
          "[usd][gpu][mesh][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string square =
        "    int[] faceVertexCounts = [4]\n"
        "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
        "    point3f[] points = [(-0.5, -0.5, -5), (0.5, -0.5, -5), (0.5, 0.5, -5), (-0.5, 0.5, -5)]\n"
        "    uniform token subdivisionScheme = \"none\"\n"
        "    color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const fs::path skinned = scratch("skinned.usda");
    {
        std::ofstream out(skinned);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"root\", \"root/arm\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"root/arm\"]\n"
               "        float3[] translations.timeSamples = {\n            0: [(0, 0, 0)],\n            1: [(1.2, 0.4, 0)],\n        }\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n    }\n"
               "    def Mesh \"Square\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
            << square
            << "        rel skel:skeleton = </Root/Skel>\n"
               "        int[] primvars:skel:jointIndices = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        float[] primvars:skel:jointWeights = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        matrix4d primvars:skel:geomBindTransform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )\n"
               "    }\n}\n"
            << camera;
    }
    const auto plain = [&](const char* name, const char* translate) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
            << square << "    double3 xformOp:translate = " << translate << "\n"
            << "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
            << camera;
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](usd::StageRenderer& renderer, double time, const char* route) {
        REQUIRE(renderer.setMeshVisibility(route));
        auto image = renderer.render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    auto skel = usd::StageRenderer::open(skinned);
    auto rest = usd::StageRenderer::open(plain("skinned_rest.usda", "(0, 0, 0)"));
    auto moved = usd::StageRenderer::open(plain("skinned_moved.usda", "(1.2, 0.4, 0)"));
    if (!skel) FAIL(skel.error().toString());
    if (!rest) FAIL(rest.error().toString());
    if (!moved) FAIL(moved.error().toString());
    const gpu::Buffer skelRest = frame(**skel, 0.0, "raster");
    const uint64_t generation = (*skel)->meshGeneration();
    const gpu::Buffer skelMoved = frame(**skel, 1.0, "raster");
    const gpu::Buffer authoredRest = frame(**rest, 0.0, "raster");
    const gpu::Buffer authoredMoved = frame(**moved, 0.0, "raster");
    auto atRest = render::compareImages(*gpu->library, skelRest, authoredRest, w, h);
    auto atOne = render::compareImages(*gpu->library, skelMoved, authoredMoved, w, h);
    auto changed = render::compareImages(*gpu->library, skelRest, skelMoved, w, h);
    REQUIRE(atRest);
    REQUIRE(atOne);
    REQUIRE(changed);
    std::printf("  skinned square: at rest %llu pixels beyond 2 of the authored square (max %u); slid %llu (max %u); "
                "the slide changes %llu pixels; generation %llu -> %llu\n",
                static_cast<unsigned long long>(atRest->over2), atRest->max,
                static_cast<unsigned long long>(atOne->over2), atOne->max,
                static_cast<unsigned long long>(changed->over2), static_cast<unsigned long long>(generation),
                static_cast<unsigned long long>((*skel)->meshGeneration()));
    CHECK(changed->over2 > 1000);
    CHECK(atRest->max == 0);
    CHECK(atOne->max == 0);
    CHECK((*skel)->meshGeneration() == generation);
    if (gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure) {
        const gpu::Buffer rays = frame(**skel, 1.0, "rays");
        auto viaRays = render::compareImages(*gpu->library, rays, authoredMoved, w, h);
        REQUIRE(viaRays);
        std::printf("  and through rays: %llu beyond 2 (max %u)\n", static_cast<unsigned long long>(viaRays->over2),
                    viaRays->max);
        CHECK(viaRays->over2 < uint64_t{w} * h / 50);
    }
}

// Blend shapes through Hydra: a shape with an inbetween, weighed by the
// animation. At weight 1 the square is the square authored with the
// shape's offsets; at weight 0.5 it is the inbetween authored at 0.5 -- not
// half the shape's offsets, which is what proves the inbetween is resolved
// -- both to the pixel.
TEST_CASE("blend shapes and their inbetweens deform a mesh through Hydra as the authored shapes draw it",
          "[usd][gpu][mesh][skinning][blendshapes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto squareWith = [](const char* p0, const char* p1, const char* p2, const char* p3) {
        return std::string("    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                           "    point3f[] points = [") +
               p0 + ", " + p1 + ", " + p2 + ", " + p3 +
               "]\n    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.3, 0.7, 0.5)] ( interpolation = \"constant\" )\n";
    };
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const fs::path shaped = scratch("blendshape.usda");
    {
        std::ofstream out(shaped);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 2\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"root\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"root\"]\n"
               "        float3[] translations = [(0, 0, 0)]\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n"
               "        uniform token[] blendShapes = [\"puff\"]\n"
               "        float[] blendShapeWeights.timeSamples = {\n            0: [0],\n            1: [0.5],\n            2: [1],\n        }\n"
               "    }\n"
               "    def Mesh \"Square\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
            << squareWith("(-0.5, -0.5, -5)", "(0.5, -0.5, -5)", "(0.5, 0.5, -5)", "(-0.5, 0.5, -5)")
            << "        rel skel:skeleton = </Root/Skel>\n"
               "        uniform token[] skel:blendShapes = [\"puff\"]\n"
               "        rel skel:blendShapeTargets = [</Root/Square/Puff>]\n"
               "        int[] primvars:skel:jointIndices = [0, 0, 0, 0] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        float[] primvars:skel:jointWeights = [1, 1, 1, 1] ( elementSize = 1\n            interpolation = \"vertex\" )\n"
               "        def BlendShape \"Puff\"\n        {\n"
               "            uniform vector3f[] offsets = [(0.8, 0, 0), (0, 0.6, 0)]\n"
               "            uniform int[] pointIndices = [1, 2]\n"
               "            uniform vector3f[] inbetweens:Half:offsets = [(0.1, -0.3, 0), (-0.2, 0.1, 0)] (\n"
               "                weight = 0.5\n            )\n        }\n"
               "    }\n}\n"
            << camera;
    }
    const auto plain = [&](const char* name, const std::string& square) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\ndef Mesh \"Square\"\n{\n" << square << "}\n" << camera;
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](usd::StageRenderer& renderer, double time) {
        auto image = renderer.render("/Camera", time, w, h);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    auto skel = usd::StageRenderer::open(shaped);
    // Weight 1: the shape's offsets on points 1 and 2. Weight 0.5: the
    // inbetween's own offsets, as authored at 0.5.
    auto full = usd::StageRenderer::open(plain(
        "blendshape_full.usda", squareWith("(-0.5, -0.5, -5)", "(1.3, -0.5, -5)", "(0.5, 1.1, -5)", "(-0.5, 0.5, -5)")));
    auto half = usd::StageRenderer::open(plain(
        "blendshape_half.usda", squareWith("(-0.5, -0.5, -5)", "(0.6, -0.8, -5)", "(0.3, 0.6, -5)", "(-0.5, 0.5, -5)")));
    if (!skel) FAIL(skel.error().toString());
    if (!full) FAIL(full.error().toString());
    if (!half) FAIL(half.error().toString());
    const gpu::Buffer rest = frame(**skel, 0.0);
    const gpu::Buffer atHalf = frame(**skel, 1.0);
    const gpu::Buffer atFull = frame(**skel, 2.0);
    const gpu::Buffer authoredFull = frame(**full, 0.0);
    const gpu::Buffer authoredHalf = frame(**half, 0.0);
    auto fullDiff = render::compareImages(*gpu->library, atFull, authoredFull, w, h);
    auto halfDiff = render::compareImages(*gpu->library, atHalf, authoredHalf, w, h);
    auto moved = render::compareImages(*gpu->library, rest, atFull, w, h);
    REQUIRE(fullDiff);
    REQUIRE(halfDiff);
    REQUIRE(moved);
    std::printf("  blend shape at weight 1: %llu pixels beyond 2 of the authored shape (max %u); at 0.5, %llu beyond "
                "the authored inbetween (max %u); the shape changes %llu pixels\n",
                static_cast<unsigned long long>(fullDiff->over2), fullDiff->max,
                static_cast<unsigned long long>(halfDiff->over2), halfDiff->max,
                static_cast<unsigned long long>(moved->over2));
    CHECK(moved->over2 > 1000);
    CHECK(fullDiff->max == 0);
    CHECK(halfDiff->max == 0);
}

// hdsi's conversions ahead of the delegate: a UsdGeomSphere becomes a mesh
// (checked against the analytic sphere, to the tessellation's chord), a
// TetMesh becomes its surface triangles and a degree-one NurbsPatch its
// quad -- each drawn as the mesh authored by hand, to the pixel.
TEST_CASE("implicit surfaces, tetrahedral meshes and NURBS patches arrive as meshes through Hydra",
          "[usd][gpu][mesh][hdsi]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const std::string camera = "def Camera \"Camera\"\n{\n"
                               "    float focalLength = 35\n"
                               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                               "    float2 clippingRange = (0.1, 1000)\n}\n";
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto image = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto out = (*renderer)->render("/Camera", 0.0, w, h);
        if (!out) FAIL(out.error().toString());
        return *out;
    };
    const auto buffer = [&](const usd::StageImage& img) {
        gpu::BufferDesc desc;
        desc.bytes = img.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, img.rgba.data());
        REQUIRE(made);
        return *made;
    };
    // The depth as an image, for a comparison that does not care which way
    // a face winds: the same triangles at the same places, however shaded.
    const auto depthImage = [&](const usd::StageImage& img) {
        std::vector<float> packed(img.depth.size() * 4, 1.0F);
        for (size_t i = 0; i < img.depth.size(); ++i) {
            packed[i * 4] = img.depth[i];
        }
        gpu::BufferDesc desc;
        desc.bytes = packed.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, packed.data());
        REQUIRE(made);
        return *made;
    };
    // The sphere, against the analytic one: hdsi tessellates with 10 axial
    // and 10 radial segments, so the chord's sag is r (1 - cos(pi / 10)).
    {
        const fs::path path = scratch("hdsi_sphere.usda");
        {
            std::ofstream out(path);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Sphere \"Ball\"\n{\n    double radius = 1.2\n"
                   "    double3 xformOp:translate = (0, 0, -5)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
                << camera;
        }
        const usd::StageImage img = image(path);
        test::dumpPpm("hdsi_sphere", img.rgba.data(), w, h);
        render::Camera cam;
        cam.lens.focal = 35.0;
        cam.lens.haperture = 24.576;
        const render::Projection projection = render::projectionFor(cam, w, h);
        auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, img.depth, "depth");
        REQUIRE(depth);
        const gpu::Buffer colours = buffer(img);
        auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/visibility_check", "sphereCheck");
        if (!check) FAIL(check.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "sphere.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "sphere.worst");
        const float radius = 1.2F;
        const float sag = radius * (1.0F - std::cos(3.14159265F / 10.0F));
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(colours.rhi());
                cursor["depth"].setBinding(depth->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                cursor["sphere"]["z"].setData(5.0F);
                cursor["sphere"]["radius"].setData(radius);
                cursor["sphere"]["sag"].setData(sag);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[3] = {0, 0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  UsdGeomSphere as a mesh: %u of %u pixels judged wrong outside the chord band, %u squarely "
                    "covered, depth within %.4f of the analytic sphere (sag %.4f)\n",
                    c[0], c[1], c[2], static_cast<double>(e), static_cast<double>(sag));
        CHECK(c[2] > 2000);
        CHECK(c[0] == 0);
        CHECK(e <= sag / 0.6F + 1e-4F);
    }
    // A tetrahedron as a TetMesh, against its four faces authored as a mesh.
    {
        const char* points = "[(-1, -0.6, -5), (1, -0.6, -5), (0, -0.6, -6.5), (0, 0.9, -5.5)]";
        const fs::path tet = scratch("hdsi_tet.usda");
        {
            std::ofstream out(tet);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def TetMesh \"Tet\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int4[] tetVertexIndices = [(0, 1, 2, 3)]\n"
                   "    int3[] surfaceFaceVertexIndices = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]\n"
                   "    color3f[] primvars:displayColor = [(0.6, 0.5, 0.2)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const fs::path faces = scratch("hdsi_tet_faces.usda");
        {
            std::ofstream out(faces);
            // The surface hdsi derives: each face wound outward.
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Tet\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int[] faceVertexCounts = [3, 3, 3, 3]\n"
                   "    int[] faceVertexIndices = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    bool doubleSided = 1\n"
                   "    color3f[] primvars:displayColor = [(0.6, 0.5, 0.2)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        // Depth against depth: hdsi winds the surface its own way, and the
        // headlight shades a face by the side it sees.
        const usd::StageImage a = image(tet);
        const usd::StageImage b = image(faces);
        const gpu::Buffer depthA = depthImage(a);
        const gpu::Buffer depthB = depthImage(b);
        auto diff = render::compareHdr(*gpu->library, depthA, depthB, w, h);
        REQUIRE(diff);
        std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
        gpu::BufferDesc desc;
        desc.bytes = blank.size() * sizeof(float);
        desc.elementBytes = 16;
        auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
        REQUIRE(blankBuffer);
        auto drawn = render::compareImages(*gpu->library, buffer(a), *blankBuffer, w, h);
        REQUIRE(drawn);
        std::printf("  TetMesh against its faces as a mesh, by depth: relMSE %.2e, max relative %.2e; drawn %llu\n",
                    diff->relMse, diff->maxRelative, static_cast<unsigned long long>(drawn->over2));
        CHECK(drawn->over2 > 1000);
        CHECK(diff->relMse < 1e-8);
    }
    // A bilinear NurbsPatch, against its quad.
    {
        const char* points = "[(-1, -0.7, -5), (1, -0.7, -5), (-1, 0.7, -5), (1, 0.7, -5)]";
        const fs::path patch = scratch("hdsi_nurbs.usda");
        {
            std::ofstream out(patch);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def NurbsPatch \"Patch\"\n{\n"
                   "    int uVertexCount = 2\n    int vVertexCount = 2\n"
                   "    int uOrder = 2\n    int vOrder = 2\n"
                   "    double[] uKnots = [0, 0, 1, 1]\n    double[] vKnots = [0, 0, 1, 1]\n"
                << "    point3f[] points = " << points << "\n"
                << "    color3f[] primvars:displayColor = [(0.2, 0.5, 0.7)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const fs::path quad = scratch("hdsi_nurbs_quad.usda");
        {
            std::ofstream out(quad);
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Patch\"\n{\n"
                << "    point3f[] points = " << points << "\n"
                << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 3, 2]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    bool doubleSided = 1\n"
                   "    color3f[] primvars:displayColor = [(0.2, 0.5, 0.7)] ( interpolation = \"constant\" )\n}\n"
                << camera;
        }
        const gpu::Buffer a = buffer(image(patch));
        const gpu::Buffer b = buffer(image(quad));
        auto diff = render::compareImages(*gpu->library, a, b, w, h);
        REQUIRE(diff);
        std::printf("  NurbsPatch against its quad: %llu pixels beyond 2 (max %u)\n",
                    static_cast<unsigned long long>(diff->over2), diff->max);
        CHECK(diff->max == 0);
    }
}

// BasisCurves through Hydra: a straight linear curve of width w along x is
// a tube -- a cylinder of radius w/2 tessellated with `sides` sides -- so
// its depth on the row through its axis is the front of that cylinder to
// within a facet's sag, and the rows it covers are those within w/2 of the
// axis, to the same sag. The three visibility routes must agree on it.
TEST_CASE("a linear BasisCurves prim draws as a tube of its width through Hydra, on every route",
          "[usd][gpu][mesh][curves]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("curve.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def BasisCurves \"Hair\"\n{\n"
               "    uniform token type = \"linear\"\n"
               "    int[] curveVertexCounts = [2]\n"
               "    point3f[] points = [(-3, 0.2, -5), (3, 0.2, -5)]\n"
               "    float[] widths = [0.4] ( interpolation = \"constant\" )\n"
               "    color3f[] primvars:displayColor = [(0.7, 0.5, 0.3)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 200;
    const uint32_t h = 150;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    render::Camera cam;
    cam.lens.focal = 35.0;
    cam.lens.haperture = 24.576;
    const render::Projection projection = render::projectionFor(cam, w, h);
    const float radius = 0.2F;
    const float sag = radius * (1.0F - std::cos(3.14159265F / 8.0F));   // 8 sides
    for (const char* route : {"raster", "rays", "bvh"}) {
        if (std::string(route) != "raster" && !(gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure)) {
            continue;
        }
        REQUIRE((*renderer)->setMeshVisibility(route));
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        auto depth = gpu::Buffer::fromSpan<float>(*gpu->device, image->depth, "depth");
        REQUIRE(depth);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto colours = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(colours);
        auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/visibility_check", "cylinderCheck");
        if (!check) FAIL(check.error().toString());
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "cylinder.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "cylinder.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["colour"].setBinding(colours->rhi());
                cursor["depth"].setBinding(depth->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                technique::setCamera(cursor["camera"], projection, w, h);
                rhi::ShaderCursor c = cursor["cylinder"];
                c["y"].setData(0.2F);
                c["z"].setData(5.0F);
                c["radius"].setData(radius);
                c["sag"].setData(sag);
                c["xMin"].setData(-3.0F);
                c["xMax"].setData(3.0F);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t c[3] = {0, 0, 0};
        float e = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(e), &e));
        std::printf("  %s: %u pixels squarely on the tube, %u coverage wrong beyond the facet band, depth within "
                    "%.4f of the cylinder's front (sag %.4f)\n",
                    route, c[2], c[0], static_cast<double>(e), static_cast<double>(sag));
        CHECK(c[2] > 500);
        CHECK(c[0] == 0);
        CHECK(e <= sag / 0.7F + 1e-3F);
    }
}

// A hair material on a curve through Hydra: chiang_hair_bsdf bound to a
// BasisCurves prim, lit by a sun. The tube draws lit and not black, and
// differently from the same curve under a Lambert material -- the lobe is
// the one shading it, with the tube's own tangent as the fibre's direction.
TEST_CASE("a chiang hair material shades a curve through Hydra", "[usd][gpu][mesh][curves][hair]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stageWith = [&](const char* name, bool hair) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def BasisCurves \"Hair\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    uniform token type = \"cubic\"\n    uniform token basis = \"bspline\"\n"
               "    int[] curveVertexCounts = [6]\n"
               "    point3f[] points = [(-3, -0.5, -5), (-2, 0.6, -5), (-1, -0.4, -4.5), (1, 0.5, -5.5), (2, -0.6, -5), (3, 0.5, -5)]\n"
               "    float[] widths = [0.5] ( interpolation = \"constant\" )\n"
               "    rel material:binding = </Materials/Mat>\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n"
               "    double xformOp:rotateX = -30\n    uniform token[] xformOpOrder = [\"xformOp:rotateX\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface\"\n"
               "            token inputs:bsdf.connect = </Materials/Mat/Bsdf.outputs:out>\n"
               "            token outputs:out\n        }\n";
        if (hair) {
            out << "        def Shader \"Bsdf\"\n        {\n"
                   "            uniform token info:id = \"ND_chiang_hair_bsdf\"\n"
                   "            color3f inputs:tint_R = (1, 1, 1)\n"
                   "            color3f inputs:tint_TT = (1, 0.8, 0.6)\n"
                   "            color3f inputs:tint_TRT = (1, 0.9, 0.8)\n"
                   "            float3 inputs:absorption_coefficient = (0.2, 0.4, 0.8)\n"
                   "            token outputs:out\n        }\n    }\n}\n";
        } else {
            out << "        def Shader \"Bsdf\"\n        {\n"
                   "            uniform token info:id = \"ND_oren_nayar_diffuse_bsdf\"\n"
                   "            color3f inputs:color = (0.8, 0.8, 0.8)\n"
                   "            token outputs:out\n        }\n    }\n}\n";
        }
        return path;
    };
    const uint32_t w = 200;
    const uint32_t h = 150;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(16);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        test::dumpPpm(path.stem().string(), image->rgba.data(), w, h);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer hair = frame(stageWith("hair_chiang.usda", true));
    const gpu::Buffer lambert = frame(stageWith("hair_lambert.usda", false));
    std::vector<float> blank(static_cast<size_t>(w) * h * 4, 0.0F);
    gpu::BufferDesc desc;
    desc.bytes = blank.size() * sizeof(float);
    desc.elementBytes = 16;
    auto blankBuffer = gpu::Buffer::create(*gpu->device, desc, blank.data());
    REQUIRE(blankBuffer);
    auto lit = render::compareHdr(*gpu->library, hair, *blankBuffer, w, h);
    auto differs = render::compareHdr(*gpu->library, hair, lambert, w, h);
    REQUIRE(lit);
    REQUIRE(differs);
    std::printf("  hair material on a curve: against blank relMSE %.2e; against Lambert relMSE %.2e\n", lit->relMse,
                differs->relMse);
    CHECK(lit->relMse > 0.1);
    CHECK(differs->relMse > 1e-2);
}

// Subdivision through Hydra: a catmullClark cube drawn at refine levels 0,
// 1 and 2. The face centre of the +z face is a vertex from level 1 on and
// its limit is one point, so the depth at the pixel looking straight at it
// is the same at levels 1 and 2 to a facet's sag (the limit projection),
// while the flat control face at level 0 sits nearer the camera: the
// surface bulges inward there, so the centre depth grows.
TEST_CASE("a catmullClark cube refines through Hydra, its limit the same at every level", "[usd][gpu][mesh][subdivision]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("subdiv_cube.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Cube\"\n{\n"
               "    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]\n"
               "    int[] faceVertexIndices = [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 2, 3, 7, 6, 1, 2, 6, 5, 0, 4, 7, 3]\n"
               "    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), "
               "(1, 1, 1), (-1, 1, 1)]\n"
               "    uniform token subdivisionScheme = \"catmullClark\"\n"
               "    color3f[] primvars:displayColor = [(0.7, 0.6, 0.4)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0, 0, -6)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    std::array<float, 3> centreDepth{};
    std::array<uint32_t, 3> covered{};
    for (uint32_t level = 0; level < 3; ++level) {
        (*renderer)->setRefineLevel(level);
        auto image = (*renderer)->render("/Camera", 0.0, w, h);
        if (!image) FAIL(image.error().toString());
        centreDepth[level] = image->depth[(h / 2) * w + w / 2];
        test::dumpPpm("subdiv_cube_" + std::to_string(level), image->rgba.data(), w, h);
        for (const float d : image->depth) {
            covered[level] += d > 0.0F ? 1u : 0u;
        }
    }
    std::printf("  refine 0, 1, 2: centre depth %.5f %.5f %.5f; covered %u %u %u pixels\n",
                static_cast<double>(centreDepth[0]), static_cast<double>(centreDepth[1]),
                static_cast<double>(centreDepth[2]), covered[0], covered[1], covered[2]);
    CHECK(std::abs(centreDepth[0] - 5.0F) < 1e-4F);            // the control face at z = -5
    CHECK(centreDepth[1] > centreDepth[0] + 0.05F);           // the limit surface sits inside the cube
    // One surface whatever the level: the centre pixel's ray meets a facet
    // beside the centre vertex, so the two levels differ by a facet's sag
    // (0.004 here), not by a level's worth of motion (0.17).
    CHECK(std::abs(centreDepth[1] - centreDepth[2]) < 0.01F);
    CHECK(covered[1] < covered[0]);                           // the corners pull in
    CHECK(covered[2] < covered[0]);
}

// Coordinate systems through Hydra: UsdShadeCoordSysAPI binds a name on
// the mesh to an xformable prim; hdsi makes a coordSys prim under that
// target, the delegate takes the sprim, and the mesh's Sync reads its
// bindings with the target's transform -- the name and the translation
// arrive exactly. What a material does with them is not yet wired: this
// is the resolution the plan asked for, by the scene index's prim and not
// by parsing paths.
TEST_CASE("a coordinate system bound to a mesh resolves to its target's transform through Hydra",
          "[usd][gpu][mesh][coordSys]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("coordsys.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Frame\"\n{\n"
               "    double3 xformOp:translate = (1, 2, 3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Mesh \"Square\" (\n    prepend apiSchemas = [\"CoordSysAPI:paint\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel coordSys:paint:binding = </Frame>\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render("/Camera", 0.0, 64, 48);
    if (!image) FAIL(image.error().toString());
    const std::vector<usd::CoordSysBinding> bindings = (*renderer)->coordSysBindings("/Square");
    std::printf("  %zu coordinate systems bound to /Square", bindings.size());
    for (const usd::CoordSysBinding& b : bindings) {
        const render::Vec3 t = b.toWorld.translation();
        std::printf("; '%s' at (%.1f, %.1f, %.1f)", b.name.c_str(), t.x, t.y, t.z);
    }
    std::printf("\n");
    REQUIRE(bindings.size() == 1);
    CHECK(bindings[0].name == "paint");
    const render::Vec3 t = bindings[0].toWorld.translation();
    CHECK(t.x == 1.0);
    CHECK(t.y == 2.0);
    CHECK(t.z == 3.0);
}

// MaterialX's transforms between spaces, on a mesh that is not at the origin,
// and to a coordinate system bound to it (M8.2). Each is checked against a
// graph that reaches the same value without the transform node: position in
// world space against object position carried object -> world, and so on;
// both are emitted, offset to keep them positive, and the frames compared.
// Before, genslang's transforms read world matrices nothing set -- identity
// -- and knew no coordinate system at all.
TEST_CASE("MaterialX transforms between object, world and a bound coordinate system match the spaces they name",
          "[usd][gpu][mesh][materials][coordSys]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A square tilted in object space (its normal is not along an axis), under
    // a rotation, a non-uniform scale and a translation; a frame translated
    // and scaled, bound as "paint".
    const auto stage = [&](const std::string& name, const std::string& graph) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"Frame\"\n{\n"
               "    double3 xformOp:translate = (1, 2, 3)\n"
               "    double3 xformOp:scale = (2, 2, 2)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n}\n"
               "def Mesh \"Square\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\", \"CoordSysAPI:paint\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -0.5, -0.2), (0.5, -0.5, 0.2), (0.5, 0.5, 0.2), (-0.5, 0.5, -0.2)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    double3 xformOp:translate = (0.2, -0.1, -5)\n"
               "    double3 xformOp:rotateXYZ = (10, 25, 5)\n"
               "    double3 xformOp:scale = (2.5, 1.5, 1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\", \"xformOp:scale\"]\n"
               "    rel coordSys:paint:binding = </Frame>\n"
               "    rel material:binding = </Materials/Mat>\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Mat\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/Mat/Surface.outputs:out>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"ND_surface_unlit\"\n"
               "            color3f inputs:emission_color.connect = </Materials/Mat/Out.outputs:out>\n"
               "            token outputs:out\n        }\n"
            << graph
            << "        def Shader \"Out\"\n        {\n"
               "            uniform token info:id = \"ND_convert_vector3_color3\"\n"
               "            vector3f inputs:in.connect = </Materials/Mat/Offset.outputs:out>\n"
               "            color3f outputs:out\n        }\n"
               "    }\n}\n";
        return path;
    };
    // Graph pieces: a node named N of type ND_... with inputs.
    const auto node = [](const std::string& name, const std::string& id, const std::string& inputs) {
        return "        def Shader \"" + name + "\"\n        {\n            uniform token info:id = \"" + id + "\"\n" +
               inputs + "            vector3f outputs:out\n        }\n";
    };
    const auto offset = [&](const std::string& from, float by) {
        return node("Offset", "ND_add_vector3FA",
                    "            vector3f inputs:in1.connect = </Materials/Mat/" + from + ".outputs:out>\n"
                    "            float inputs:in2 = " + std::to_string(by) + "\n");
    };
    const auto position = [&](const char* name, const char* space) {
        return node(name, "ND_position_vector3", std::string("            string inputs:space = \"") + space + "\"\n");
    };
    const auto transform = [&](const char* id, const char* from, const char* fromSpace, const char* toSpace) {
        return node("Xf", id,
                    std::string("            vector3f inputs:in.connect = </Materials/Mat/") + from + ".outputs:out>\n" +
                        "            string inputs:fromspace = \"" + fromSpace + "\"\n" +
                        "            string inputs:tospace = \"" + toSpace + "\"\n");
    };
    struct Case {
        const char* what;
        std::string transformed;
        std::string direct;
    };
    const std::vector<Case> cases{
        {"object point to world", position("P", "object") + transform("ND_transformpoint_vector3", "P", "object", "world") + offset("Xf", 10.0F),
         position("P", "world") + offset("P", 10.0F)},
        {"world point to object", position("P", "world") + transform("ND_transformpoint_vector3", "P", "world", "object") + offset("Xf", 10.0F),
         position("P", "object") + offset("P", 10.0F)},
        {"world point to the bound 'paint'",
         position("P", "world") + transform("ND_transformpoint_vector3", "P", "world", "paint") + offset("Xf", 10.0F),
         position("P", "world") +
             node("Moved", "ND_subtract_vector3", "            vector3f inputs:in1.connect = </Materials/Mat/P.outputs:out>\n"
                                                  "            vector3f inputs:in2 = (1, 2, 3)\n") +
             node("Scaled", "ND_divide_vector3FA", "            vector3f inputs:in1.connect = </Materials/Mat/Moved.outputs:out>\n"
                                                   "            float inputs:in2 = 2\n") +
             offset("Scaled", 10.0F)},
        {"object normal to world",
         node("N", "ND_normal_vector3", "            string inputs:space = \"object\"\n") +
             transform("ND_transformnormal_vector3", "N", "object", "world") + offset("Xf", 2.0F),
         node("N", "ND_normal_vector3", "            string inputs:space = \"world\"\n") + offset("N", 2.0F)},
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    int k = 0;
    for (const Case& c : cases) {
        gpu::Buffer a = frame(stage("spaces_" + std::to_string(k) + "_transformed.usda", c.transformed));
        gpu::Buffer b = frame(stage("spaces_" + std::to_string(k) + "_direct.usda", c.direct));
        auto diff = render::compareHdr(*gpu->library, a, b, w, h);
        REQUIRE(diff);
        std::vector<float> blank(size_t{w} * h * 4, 0.0F);
        auto empty = gpu::Buffer::fromSpan<float>(*gpu->device, blank, "blank");
        REQUIRE(empty);
        auto drawn = render::compareHdr(*gpu->library, a, *empty, w, h);
        REQUIRE(drawn);
        std::printf("  %-34s: relMSE %.2e, max relative %.2e (against blank %.2e)\n", c.what, diff->relMse,
                    diff->maxRelative, drawn->relMse);
        CHECK(drawn->relMse > 0.1);
        CHECK(diff->maxRelative < 1e-4);
        ++k;
    }
    // The normals agree to the bit, which is also what a check comparing a
    // thing with itself would say: the object normal left untransformed must
    // not agree with the world normal.
    gpu::Buffer untransformed = frame(stage("spaces_normal_untransformed.usda",
        node("N", "ND_normal_vector3", "            string inputs:space = \"object\"\n") + offset("N", 2.0F)));
    gpu::Buffer world = frame(stage("spaces_normal_world.usda",
        node("N", "ND_normal_vector3", "            string inputs:space = \"world\"\n") + offset("N", 2.0F)));
    auto control = render::compareHdr(*gpu->library, untransformed, world, w, h);
    REQUIRE(control);
    std::printf("  control, object normal untransformed : max relative %.2e against the world normal\n",
                control->maxRelative);
    CHECK(control->maxRelative > 1e-2);
}

// Render settings through Hydra (M10): a UsdRenderSettings prim with its
// products and vars reaches the delegate's renderSettings bprim once it is
// the scene's active one; `renderProducts` renders each product at its own
// resolution with its vars as the layers of one EXR, `includedPurposes` as
// the render tags. The check is the plan's, literal: every layer of the
// file, read back and uploaded, is bit for bit the AOV rendered on its own
// (countDifferent 0 words), and a second settings prim that includes guides
// covers more pixels than the first, which is what proves the purposes
// reached the tags and not a default.
TEST_CASE("a render settings prim's products come out as the AOVs rendered one at a time, bit for bit",
          "[usd][gpu][mesh][render-settings]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("render_settings.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "    color3f[] primvars:displayColor = [(0.2, 0.7, 0.3)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Guide\"\n{\n"
               "    uniform token purpose = \"guide\"\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(1, -2, -4), (3, -2, -4), (3, 2, -4), (1, 2, -4)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Settings\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] includedPurposes = [\"default\"]\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (96, 64)\n"
               "        int athenea:pathSamples = 3\n"
               "        token athenea:technique = \"raster\"\n"
               "    }\n"
               "    def RenderSettings \"WithGuides\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] includedPurposes = [\"default\", \"guide\"]\n"
               "        rel products = </Render/GuideProduct>\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"product.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/depth>, </Render/Vars/primId>, </Render/Vars/Neye>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def RenderProduct \"GuideProduct\"\n    {\n"
               "        token productName = \"guides.exr\"\n"
               "        rel orderedVars = [</Render/Vars/primId>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n            token dataType = \"color3f\"\n        }\n"
               "        def RenderVar \"depth\"\n        {\n            string sourceName = \"z\"\n            token dataType = \"float\"\n        }\n"
               "        def RenderVar \"primId\"\n        {\n            string sourceName = \"primId\"\n            token dataType = \"int\"\n        }\n"
               "        def RenderVar \"Neye\"\n        {\n            string sourceName = \"Neye\"\n            token dataType = \"normal3f\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto info = (*renderer)->renderSettings("/Render/Settings");
    if (!info) FAIL(info.error().toString());
    std::printf("  /Render/Settings: %s, synced %u times, %zu products, purposes:", info->active ? "active" : "inactive",
                info->syncs, info->products.size());
    for (const std::string& p : info->includedPurposes) std::printf(" %s", p.c_str());
    std::printf("; camera %s; settings:", info->camera.c_str());
    for (const auto& [k, v] : info->settings) std::printf(" %s=%s", k.c_str(), v.c_str());
    std::printf("\n");
    REQUIRE(info->active);
    REQUIRE(info->products.size() == 1);
    CHECK(info->products[0].width == 96);
    CHECK(info->products[0].height == 64);
    CHECK(info->products[0].name == "product.exr");
    REQUIRE(info->products[0].vars.size() == 4);
    CHECK(info->products[0].vars[0].name == "beauty");
    CHECK(info->products[0].vars[0].sourceName == "Ci");
    CHECK(info->includedPurposes == std::vector<std::string>{"default"});
    CHECK(info->settings["athenea:pathSamples"] == "3");
    CHECK(info->settings["athenea:technique"] == "raster");

    const fs::path directory = path.parent_path();
    auto written = (*renderer)->renderProducts("/Render/Settings", 0.0, directory);
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 1);
    CHECK(written->front() == directory / "product.exr");
    auto file = io::readExrChannels(written->front());
    if (!file) FAIL(file.error().toString());
    REQUIRE(file->width == 96);
    REQUIRE(file->height == 64);
    std::printf("  product.exr: %zu channels:", file->channels.size());
    for (const io::ExrChannelData& c : file->channels) std::printf(" %s", c.name.c_str());
    std::printf("\n");
    REQUIRE(file->channels.size() == 4 + 1 + 1 + 3);
    const auto channel = [&](const std::string& name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : file->channels) {
            if (c.name == name) return &c;
        }
        return nullptr;
    };
    for (const char* name : {"beauty.R", "beauty.G", "beauty.B", "beauty.A", "Z", "primId", "Neye.x", "Neye.y", "Neye.z"}) {
        INFO(name);
        REQUIRE(channel(name) != nullptr);
    }
    CHECK(channel("primId")->type == io::ExrChannelType::Uint);
    CHECK(channel("Z")->type == io::ExrChannelType::Float);

    // The same AOVs rendered one at a time, through the same Hydra buffers.
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto upload = [&](std::span<const uint32_t> words, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = words.size() * 4;
        desc.elementBytes = 4;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, words.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const auto differing = [&](const std::string& name, std::span<const uint32_t> expected) {
        const io::ExrChannelData* c = channel(name);
        REQUIRE(c != nullptr);
        REQUIRE(c->words.size() == pixels);
        gpu::Buffer a = upload(c->words, "exr.layer");
        gpu::Buffer b = upload(expected, "exr.alone");
        auto count = render::countDifferent(*gpu->library, a, b, static_cast<uint32_t>(pixels));
        REQUIRE(count);
        return *count;
    };
    const auto plane = [&](const std::vector<uint8_t>& bytes, size_t components, size_t c) {
        std::vector<uint32_t> words(pixels);
        for (size_t p = 0; p < pixels; ++p) {
            std::memcpy(&words[p], bytes.data() + (p * components + c) * 4, 4);
        }
        return words;
    };
    (*renderer)->requestOutputs({"primId"});
    auto alone = (*renderer)->render("/Camera", 0.0, w, h);
    if (!alone) FAIL(alone.error().toString());
    auto primId = (*renderer)->mappedOutput("primId");
    REQUIRE(primId);
    const std::vector<uint32_t> ids = plane(*primId, 1, 0);
    std::vector<uint32_t> depth(pixels);
    std::memcpy(depth.data(), alone->depth.data(), pixels * 4);
    std::vector<uint32_t> colour[4];
    for (size_t c = 0; c < 4; ++c) {
        colour[c].resize(pixels);
        for (size_t p = 0; p < pixels; ++p) std::memcpy(&colour[c][p], alone->rgba.data() + p * 4 + c, 4);
    }
    (*renderer)->requestOutputs({"Neye"});
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto eye = (*renderer)->mappedOutput("Neye");
    REQUIRE(eye);
    const uint64_t offId = differing("primId", ids);
    const uint64_t offZ = differing("Z", depth);
    const uint64_t offR = differing("beauty.R", colour[0]);
    const uint64_t offG = differing("beauty.G", colour[1]);
    const uint64_t offB = differing("beauty.B", colour[2]);
    const uint64_t offA = differing("beauty.A", colour[3]);
    const uint64_t offNx = differing("Neye.x", plane(*eye, 3, 0));
    const uint64_t offNy = differing("Neye.y", plane(*eye, 3, 1));
    const uint64_t offNz = differing("Neye.z", plane(*eye, 3, 2));
    // Coverage of the ids plane: how many pixels the square took.
    const std::vector<uint32_t> cleared(pixels, 0xFFFFFFFFu);
    const uint64_t covered = differing("primId", cleared);
    std::printf("  layers against the AOVs alone: primId %llu words off, Z %llu, beauty %llu %llu %llu %llu, Neye %llu %llu %llu;"
                " %llu of %zu pixels covered\n",
                static_cast<unsigned long long>(offId), static_cast<unsigned long long>(offZ),
                static_cast<unsigned long long>(offR), static_cast<unsigned long long>(offG),
                static_cast<unsigned long long>(offB), static_cast<unsigned long long>(offA),
                static_cast<unsigned long long>(offNx), static_cast<unsigned long long>(offNy),
                static_cast<unsigned long long>(offNz), static_cast<unsigned long long>(covered), pixels);
    CHECK(offId == 0);
    CHECK(offZ == 0);
    CHECK(offR == 0);
    CHECK(offG == 0);
    CHECK(offB == 0);
    CHECK(offA == 0);
    CHECK(offNx == 0);
    CHECK(offNy == 0);
    CHECK(offNz == 0);
    CHECK(covered > 500);

    // The settings that include guides draw the guide square too.
    auto guides = (*renderer)->renderProducts("/Render/WithGuides", 0.0, directory);
    if (!guides) FAIL(guides.error().toString());
    REQUIRE(guides->size() == 1);
    auto guideFile = io::readExrChannels(guides->front());
    if (!guideFile) FAIL(guideFile.error().toString());
    REQUIRE(guideFile->channels.size() == 1);
    REQUIRE(guideFile->channels[0].words.size() == pixels);
    gpu::Buffer guideIds = upload(guideFile->channels[0].words, "exr.guides");
    gpu::Buffer blank = upload(cleared, "exr.cleared");
    auto guideCovered = render::countDifferent(*gpu->library, guideIds, blank, static_cast<uint32_t>(pixels));
    REQUIRE(guideCovered);
    auto second = (*renderer)->renderSettings("/Render/WithGuides");
    REQUIRE(second);
    std::printf("  with guides (%s): %llu pixels covered against %llu without\n", second->active ? "active" : "inactive",
                static_cast<unsigned long long>(*guideCovered), static_cast<unsigned long long>(covered));
    CHECK(second->active);
    CHECK(*guideCovered > covered + 200);
    // And the first is no longer the active one.
    auto first = (*renderer)->renderSettings("/Render/Settings");
    REQUIRE(first);
    CHECK(first->active);   // asking makes it active again
}

// Material binding purposes: a square bound three ways -- all-purpose to
// red, `material:binding:full` to blue, `material:binding:preview` to green.
// The delegate resolved Hydra's default purpose, "preview", so a
// production binding was never seen (the shader ball's walls are bound
// `full` alone and drew their fallback). It now resolves "full" unless a
// settings prim's `materialBindingPurposes` names another, with the
// all-purpose binding behind either.
TEST_CASE("a mesh takes the material bound for the purpose render settings name, full by default",
          "[usd][gpu][mesh][materials][purposes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("binding_purposes.usda");
    {
        std::ofstream out(path);
        std::string square = kSquareStage;
        const std::string binding = "    rel material:binding = </Materials/Mat>\n";
        square.replace(square.find(binding), binding.size(),
                       "    rel material:binding = </Materials/Red>\n"
                       "    rel material:binding:full = </Materials/Blue>\n"
                       "    rel material:binding:preview = </Materials/Green>\n");
        out << square << "def Scope \"Materials\"\n{\n";
        const auto material = [&](const char* name, const char* colour) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:surface.connect = </Materials/" << name << "/Preview.outputs:surface>\n"
                << "        def Shader \"Preview\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                << "            color3f inputs:diffuseColor = " << colour << "\n"
                << "            int inputs:useSpecularWorkflow = 1\n"
                   "            color3f inputs:specularColor = (0, 0, 0)\n"
                   "            float inputs:roughness = 1\n"
                   "            token outputs:surface\n        }\n    }\n";
        };
        material("Red", "(0.8, 0.1, 0.1)");
        material("Green", "(0.1, 0.8, 0.1)");
        material("Blue", "(0.1, 0.1, 0.8)");
        out << "}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Preview\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        uniform token[] materialBindingPurposes = [\"preview\", \"\"]\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"purposes.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>]\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n"
               "            token dataType = \"color3f\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto centre = [&](const char* label) {
        auto image = (*renderer)->render("/Camera", 0.0, 160, 120);
        if (!image) FAIL(image.error().toString());
        const float* c = image->rgba.data() + (60 * 160 + 80) * 4;
        std::printf("  %-34s: centre %.3f %.3f %.3f\n", label, double(c[0]), double(c[1]), double(c[2]));
        // Which of the three: the channel that is high.
        return c[0] > 0.5F ? 'r' : c[1] > 0.5F ? 'g' : c[2] > 0.5F ? 'b' : '?';
    };
    CHECK(centre("default (full)") == 'b');
    (*renderer)->setMaterialBindingPurposes({"preview", ""});
    CHECK(centre("preview, then all-purpose") == 'g');
    (*renderer)->setMaterialBindingPurposes({""});
    CHECK(centre("all-purpose alone") == 'r');
    (*renderer)->setMaterialBindingPurposes({});
    CHECK(centre("back to the default") == 'b');
    // From a settings prim: its products are drawn with its purposes, and
    // the renders after keep them.
    auto written = (*renderer)->renderProducts("/Render/Preview", 0.0, path.parent_path());
    if (!written) FAIL(written.error().toString());
    CHECK(centre("after /Render/Preview's products") == 'g');
}

// A render product's disableMotionBlur and disableDepthOfField (M10): read
// since the products were, applied now. One stage -- a square sliding under
// a shutter open about the frame, seen through a lens focused in front of it
// -- and three products: both switches on, which must be bit for bit the
// same stage authored with neither shutter nor lens; and each switch alone,
// which must not be, since the other effect is still drawn. The switches
// hold for their product and nothing after.
TEST_CASE("a render product's disableMotionBlur and disableDepthOfField draw it without either",
          "[usd][gpu][mesh][path][render-settings][camera][motion]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto stage = [&](const char* name, bool effects) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
               "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n";
        if (effects) {
            out << "    double shutter:open = -0.25\n    double shutter:close = 0.25\n"
                   "    float fStop = 8\n    float focusDistance = 2.5\n";
        }
        out << "}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Settings\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        rel products = [</Render/Plain>, </Render/Lens>, </Render/Blur>]\n"
               "        int2 resolution = (96, 64)\n"
               "        token athenea:technique = \"rt\"\n"
               "        int athenea:pathSamples = 16\n        int athenea:pathTotal = 16\n        int athenea:pathBounces = 0\n"
               "        int athenea:motionBuckets = 8\n"
               "    }\n";
        const auto product = [&](const char* productName, bool noBlur, bool noLens) {
            out << "    def RenderProduct \"" << productName << "\"\n    {\n"
                << "        token productName = \"switches_" << productName << ".exr\"\n"
                << "        rel orderedVars = </Render/Vars/beauty>\n"
                   "        int2 resolution = (96, 64)\n"
                << "        uniform bool disableMotionBlur = " << (noBlur ? 1 : 0) << "\n"
                << "        uniform bool disableDepthOfField = " << (noLens ? 1 : 0) << "\n    }\n";
        };
        product("Plain", true, true);
        product("Lens", true, false);
        product("Blur", false, true);
        out << "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n"
               "            token dataType = \"color3f\"\n        }\n    }\n}\n";
        return path;
    };
    // The reference: neither shutter nor lens, drawn as the settings say.
    std::vector<uint32_t> reference[3];
    {
        auto renderer = usd::StageRenderer::open(stage("switches_reference.usda", false));
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(16);
        (*renderer)->setPathTotal(16);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        auto image = (*renderer)->render("/Camera", 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        for (size_t c = 0; c < 3; ++c) {
            reference[c].resize(pixels);
            for (size_t p = 0; p < pixels; ++p) std::memcpy(&reference[c][p], image->rgba.data() + p * 4 + c, 4);
        }
    }
    const fs::path path = stage("switches.usda", true);
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto written = (*renderer)->renderProducts("/Render/Settings", 0.5, path.parent_path());
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 3);
    const auto differing = [&](const fs::path& file) {
        auto read = io::readExrChannels(file);
        if (!read) FAIL(read.error().toString());
        uint64_t off = 0;
        const char* names[] = {"beauty.R", "beauty.G", "beauty.B"};
        for (size_t c = 0; c < 3; ++c) {
            const io::ExrChannelData* found = nullptr;
            for (const io::ExrChannelData& data : read->channels) {
                if (data.name == names[c]) found = &data;
            }
            REQUIRE(found != nullptr);
            REQUIRE(found->words.size() == pixels);
            auto a = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, found->words, "product");
            auto b = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, reference[c], "reference");
            REQUIRE(a);
            REQUIRE(b);
            auto count = render::countDifferent(*gpu->library, *a, *b, static_cast<uint32_t>(pixels));
            REQUIRE(count);
            off += *count;
        }
        return off;
    };
    const uint64_t plain = differing((*written)[0]);
    const uint64_t lens = differing((*written)[1]);
    const uint64_t blur = differing((*written)[2]);
    std::printf("  words off the plain stage's frame: both switched off %llu, lens drawn %llu, blur drawn %llu (of %zu)\n",
                static_cast<unsigned long long>(plain), static_cast<unsigned long long>(lens),
                static_cast<unsigned long long>(blur), pixels * 3);
    CHECK(plain == 0);
    CHECK(lens > 300);
    CHECK(blur > 300);
    // And after the products, the stage's own effects again.
    (*renderer)->setPathSamples(16);
    (*renderer)->setPathTotal(16);
    (*renderer)->setPathBounces(0);
    (*renderer)->setMotionBuckets(8);
    auto after = (*renderer)->render("/Camera", 0.5, w, h, "rt");
    if (!after) FAIL(after.error().toString());
    std::vector<uint32_t> red(pixels);
    for (size_t p = 0; p < pixels; ++p) std::memcpy(&red[p], after->rgba.data() + p * 4, 4);
    auto a = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, red, "after");
    auto b = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, reference[0], "reference");
    REQUIRE(a);
    REQUIRE(b);
    auto again = render::countDifferent(*gpu->library, *a, *b, static_cast<uint32_t>(pixels));
    REQUIRE(again);
    std::printf("  a render after the products: %llu red words off the plain frame\n",
                static_cast<unsigned long long>(*again));
    CHECK(*again > 100);
}

// A dome seen by the camera is in its light group -- C.*<L.'NAME'> matches
// the camera's ray meeting the dome with `.*` empty -- and the camera's
// exposure scales the groups as it scales the beauty. Before, both left
// the groups short of the beauty: the sky's pixels in no group, and every
// pixel by the exposure's factor.
TEST_CASE("a dome's background is in its light group, and the exposure in every group, raster and path traced",
          "[usd][gpu][mesh][lights][light-groups][dome]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("light_groups_dome.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    string athenea:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def DomeLight \"Sky\"\n{\n"
               "    float inputs:intensity = 0.5\n    color3f inputs:color = (0.4, 0.6, 1)\n"
               "    bool inputs:shadow:enable = 0\n"
               "    string athenea:lightGroup = \"sky\"\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    float exposure = 1\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -8\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // Pixel by pixel, so the camera stays in the middle of the pixel: an
    // antialiased edge is a mean over the passes and this compares frames.
    REQUIRE(renderer);
    (*renderer)->setAntialias(false);
    auto sumKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/light_group_check", "lightGroupSum");
    if (!sumKernel) FAIL(sumKernel.error().toString());
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    struct Counts {
        uint32_t off, emptyNonzero, covered, checked;
        float    worst;
    };
    const auto run = [&](const char* technique) {
        (*renderer)->requestOutputs({"lightGroup:key", "lightGroup:sky", "lightGroup:none"});
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto beauty = (*renderer)->mappedOutput("color");
        REQUIRE(beauty);
        std::vector<uint8_t> planes;
        for (const char* group : {"lightGroup:key", "lightGroup:sky", "lightGroup:none"}) {
            auto plane = (*renderer)->mappedOutput(group);
            REQUIRE(plane);
            planes.insert(planes.end(), plane->begin(), plane->end());
        }
        REQUIRE(beauty->size() == pixels * 16);
        auto b = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, *beauty, "dome.beauty");
        auto p = gpu::Buffer::fromSpan<uint8_t>(*gpu->device, planes, "dome.planes");
        REQUIRE(b);
        REQUIRE(p);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "dome.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "dome.worst");
        gpu::CommandBatch batch(*gpu->device);
        sumKernel->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) {
            c["check"]["pixels"].setData(static_cast<uint32_t>(pixels));
            c["check"]["groups"].setData(uint32_t{3});
            c["check"]["empty"].setData(uint32_t{2});
            c["check"]["tolerance"].setData(1.0e-4F);
            c["beauty"].setBinding(b->rhi());
            c["planes"].setBinding(p->rhi());
            c["counts"].setBinding(counts.rhi());
            c["worst"].setBinding(worst.rhi());
        });
        REQUIRE(batch.submit(true));
        Counts out{};
        REQUIRE(counts.read(*gpu->device, 0, 16, &out));
        REQUIRE(worst.read(*gpu->device, 0, 4, &out.worst));
        std::printf("  %-6s: %u of %u pixels covered (floor and sky); groups summed off the beauty at %u (worst %.2e "
                    "relative); the empty group nonzero at %u\n",
                    technique, out.covered, out.checked, out.off, static_cast<double>(out.worst), out.emptyNonzero);
        return out;
    };
    const Counts raster = run("raster");
    CHECK(raster.covered == pixels);
    CHECK(raster.off == 0);
    CHECK(raster.emptyNonzero == 0);
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathTotal(8);
    const Counts traced = run("rt");
    CHECK(traced.covered == pixels);
    CHECK(traced.off == 0);
    CHECK(traced.emptyNonzero == 0);
}

// Light groups (M10): a light's `athenea:lightGroup` puts its direct light,
// at every bounce, into a plane of its own beside the beauty, asked for as
// the "lightGroup:NAME" output -- or, through a render product, as the var
// with the light path expression C.*<L.'NAME'>. The plan's check: the
// groups summed are the beauty, and a group no light belongs to is exactly
// zero -- by a kernel over every pixel, for the raster's direct light and
// the path tracer alike.
TEST_CASE("light groups sum to the beauty and an empty group is zero, raster and path traced",
          "[usd][gpu][mesh][lights][light-groups]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("light_groups.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    string athenea:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Key2\"\n{\n"
               "    float inputs:intensity = 20\n    float inputs:radius = 0.3\n"
               "    color3f inputs:color = (1, 0.6, 0.3)\n"
               "    string athenea:lightGroup = \"key\"\n"
               "    double3 xformOp:translate = (3, 2.5, -9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Fill\"\n{\n"
               "    float inputs:intensity = 10\n    float inputs:radius = 0.5\n"
               "    color3f inputs:color = (0.3, 0.5, 1)\n"
               "    string ri:light:lightGroup = \"fill\"\n"
               "    double3 xformOp:translate = (0, 3, -4)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Traced\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (96, 64)\n"
               "        token athenea:technique = \"rt\"\n"
               "        int athenea:pathSamples = 4\n"
               "        int athenea:pathTotal = 16\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"groups.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/key>, </Render/Vars/fill>, </Render/Vars/rim>]\n"
               "        int2 resolution = (96, 64)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"color\"\n        }\n"
               "        def RenderVar \"key\"\n        {\n            string sourceName = \"C.*<L.'key'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "        def RenderVar \"fill\"\n        {\n            string sourceName = \"C.*<L.'fill'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "        def RenderVar \"rim\"\n        {\n            string sourceName = \"C.*<L.'rim'>\"\n            token sourceType = \"lpe\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // Pixel by pixel, so the camera stays in the middle of the pixel: an
    // antialiased edge is a mean over the passes and this compares frames.
    REQUIRE(renderer);
    (*renderer)->setAntialias(false);
    auto sumKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/light_group_check", "lightGroupSum");
    if (!sumKernel) FAIL(sumKernel.error().toString());
    const uint32_t w = 96, h = 64;
    const size_t pixels = size_t{w} * h;
    const auto upload = [&](const void* bytes, size_t count, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = count * 16;
        desc.elementBytes = 16;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, bytes);
        REQUIRE(made);
        return std::move(*made);
    };
    // The check over the beauty and three planes: key, fill, rim (empty).
    struct Counts {
        uint32_t off, emptyNonzero, covered, checked;
        float    worst;
    };
    const auto check = [&](const std::vector<uint8_t>& beauty, const std::vector<uint8_t>& planes) {
        REQUIRE(beauty.size() == pixels * 16);
        REQUIRE(planes.size() == pixels * 16 * 3);
        gpu::Buffer b = upload(beauty.data(), pixels, "groups.beauty");
        gpu::Buffer p = upload(planes.data(), pixels * 3, "groups.planes");
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "groups.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 1, "groups.worst");
        gpu::CommandBatch batch(*gpu->device);
        sumKernel->dispatch(batch, {static_cast<uint32_t>(pixels), 1, 1}, [&](rhi::ShaderCursor c) {
            c["check"]["pixels"].setData(static_cast<uint32_t>(pixels));
            c["check"]["groups"].setData(uint32_t{3});
            c["check"]["empty"].setData(uint32_t{2});
            c["check"]["tolerance"].setData(1.0e-4F);
            c["beauty"].setBinding(b.rhi());
            c["planes"].setBinding(p.rhi());
            c["counts"].setBinding(counts.rhi());
            c["worst"].setBinding(worst.rhi());
        });
        REQUIRE(batch.submit(true));
        Counts out{};
        REQUIRE(counts.read(*gpu->device, 0, 16, &out));
        REQUIRE(worst.read(*gpu->device, 0, 4, &out.worst));
        return out;
    };
    const auto planesOf = [&](const char* technique) {
        (*renderer)->requestOutputs({"lightGroup:key", "lightGroup:fill", "lightGroup:rim"});
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto beauty = (*renderer)->mappedOutput("color");
        REQUIRE(beauty);
        std::vector<uint8_t> planes;
        for (const char* group : {"lightGroup:key", "lightGroup:fill", "lightGroup:rim"}) {
            auto plane = (*renderer)->mappedOutput(group);
            REQUIRE(plane);
            planes.insert(planes.end(), plane->begin(), plane->end());
        }
        {
            const std::string prefix = std::string("light_groups_") + technique;
            test::dumpPpm((prefix + "_beauty").c_str(), image->rgba.data(), w, h);
            std::vector<float> plane(pixels * 4);
            for (size_t g = 0; g < 3; ++g) {
                std::memcpy(plane.data(), planes.data() + g * pixels * 16, pixels * 16);
                test::dumpPpm((prefix + "_" + std::to_string(g)).c_str(), plane.data(), w, h);
            }
        }
        return check(*beauty, planes);
    };
    const Counts raster = planesOf("raster");
    std::printf("  raster: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e relative); the empty "
                "group nonzero at %u\n",
                raster.covered, raster.checked, raster.off, static_cast<double>(raster.worst), raster.emptyNonzero);
    CHECK(raster.covered > 1000);
    CHECK(raster.off == 0);
    CHECK(raster.emptyNonzero == 0);
    (*renderer)->setPathSamples(4);
    (*renderer)->setPathTotal(16);
    const Counts traced = planesOf("rt");
    std::printf("  path traced: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e relative); the "
                "empty group nonzero at %u\n",
                traced.covered, traced.checked, traced.off, static_cast<double>(traced.worst), traced.emptyNonzero);
    CHECK(traced.covered > 1000);
    CHECK(traced.off == 0);
    CHECK(traced.emptyNonzero == 0);

    // And through a render product: the light path expressions name the
    // groups, and the file's layers are the same planes.
    const fs::path directory = path.parent_path();
    auto written = (*renderer)->renderProducts("/Render/Traced", 0.0, directory);
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 1);
    auto file = io::readExrChannels(written->front());
    if (!file) FAIL(file.error().toString());
    std::printf("  groups.exr: %zu channels:", file->channels.size());
    for (const io::ExrChannelData& c : file->channels) std::printf(" %s", c.name.c_str());
    std::printf("\n");
    REQUIRE(file->channels.size() == 16);
    const auto channel = [&](const std::string& name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : file->channels) {
            if (c.name == name) return &c;
        }
        return nullptr;
    };
    std::vector<uint8_t> beauty(pixels * 16);
    std::vector<uint8_t> planes(pixels * 16 * 3);
    const auto gather = [&](std::vector<uint8_t>& into, size_t plane, const std::string& prefix) {
        for (size_t c = 0; c < 4; ++c) {
            const io::ExrChannelData* data = channel(prefix + "." + "RGBA"[c]);
            REQUIRE(data != nullptr);
            for (size_t p = 0; p < pixels; ++p) {
                std::memcpy(into.data() + (plane * pixels + p) * 16 + c * 4, &data->words[p], 4);
            }
        }
    };
    gather(beauty, 0, "beauty");
    gather(planes, 0, "key");
    gather(planes, 1, "fill");
    gather(planes, 2, "rim");
    const Counts product = check(beauty, planes);
    std::printf("  the product's layers: %u of %u pixels covered; groups summed off the beauty at %u (worst %.2e); the "
                "empty group nonzero at %u\n",
                product.covered, product.checked, product.off, static_cast<double>(product.worst),
                product.emptyNonzero);
    CHECK(product.covered > 1000);
    CHECK(product.off == 0);
    CHECK(product.emptyNonzero == 0);
}

// Shadow rays over a floor nothing occludes must change nothing: the raster
// technique's shading with the lights' shadows on is bit for bit the
// shading with them off. This is the regression that caught a Metal
// compiler problem: with a local copy of the lobe stack live across the
// shadow ray's intersector call, the shadowed kernel wrote rows of garbage
// in blocks of half a threadgroup (MaterialShading.cpp says the rest).
TEST_CASE("shadow rays over an unoccluded floor change nothing through Hydra, raster technique",
          "[usd][gpu][mesh][lights][shadows]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const auto stage = [&](bool shadows) {
        const fs::path path = scratch(shadows ? "floor_shadows_on.usda" : "floor_shadows_off.usda");
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "    double3 xformOp:translate = (-2, 2, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def SphereLight \"Fill\"\n{\n"
               "    float inputs:intensity = 10\n    float inputs:radius = 0.5\n"
               "    color3f inputs:color = (0.3, 0.5, 1)\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "    double3 xformOp:translate = (0, 3, -4)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 64;
    const auto render = [&](bool shadows) {
        auto renderer = usd::StageRenderer::open(stage(shadows));
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * 4;
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    // Three shadowed renders: the corruption was nondeterministic, and one
    // clean frame proved nothing.
    gpu::Buffer off = render(false);
    for (int k = 0; k < 3; ++k) {
        gpu::Buffer on = render(true);
        auto words = render::countDifferent(*gpu->library, on, off, w * h * 4);
        REQUIRE(words);
        std::printf("  shadows on against off, run %d: %llu of %u words differ\n", k + 1,
                    static_cast<unsigned long long>(*words), w * h * 4);
        CHECK(*words == 0);
    }
}


// A light's shadowLink collection through Hydra: an occluder between a
// sphere light and a floor, both out of the camera's view, so the only
// thing the occluder can change in the image is the shadow. With the
// collection left to include everything the floor is shadowed; with a
// membership expression naming the floor alone the occluder casts nothing,
// and the frame is bit for bit the frame without the occluder -- under
// the raster's shading and under the path tracer's direct light alike.
TEST_CASE("a UsdLux light's shadowLink collection decides what casts its shadow",
          "[usd][gpu][mesh][lights][linking][shadows]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    enum class Occluder { None, Casting, Unlinked };
    const auto stage = [&](Occluder occluder) {
        const char* names[] = {"shadow_link_none.usda", "shadow_link_casting.usda", "shadow_link_unlinked.usda"};
        const fs::path path = scratch(names[int(occluder)]);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n";
        if (occluder != Occluder::None) {
            // Above the top of the frame (it ends 0.9 degrees above the
            // horizon; this is 3.6 degrees up), below the light.
            out << "def Mesh \"Occluder\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n"
                   "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-1, 2, -7), (1, 2, -7), (1, 2, -9), (-1, 2, -9)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n}\n";
        }
        out << "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = 40\n    float inputs:radius = 0.3\n"
               "    double3 xformOp:translate = (0, 3, -8)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";
        if (occluder == Occluder::Unlinked) {
            out << "    uniform token collection:shadowLink:mode = \"expression\"\n"
                   "    uniform pathExpression collection:shadowLink:membershipExpression = \"/Floor\"\n";
        }
        out << "}\n"
               "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n"
               "    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 64;
    for (const char* technique : {"raster", "rt"}) {
        const auto render = [&](Occluder occluder) {
            auto renderer = usd::StageRenderer::open(stage(occluder));
            if (!renderer) FAIL(renderer.error().toString());
            (*renderer)->setPathSamples(16);
            (*renderer)->setPathBounces(0);
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * 4;
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        gpu::Buffer none = render(Occluder::None);
        gpu::Buffer casting = render(Occluder::Casting);
        gpu::Buffer unlinked = render(Occluder::Unlinked);
        auto shadowed = render::countDifferent(*gpu->library, casting, none, w * h * 4);
        auto ignored = render::countDifferent(*gpu->library, unlinked, none, w * h * 4);
        REQUIRE(shadowed);
        REQUIRE(ignored);
        std::printf("  %-6s: occluder in the shadow link changes %llu words, outside it %llu, of %u\n", technique,
                    static_cast<unsigned long long>(*shadowed), static_cast<unsigned long long>(*ignored), w * h * 4);
        CHECK(*shadowed > 200);
        CHECK(*ignored == 0);
    }
}

// Volumes through Hydra (M9): a UsdVolVolume whose density field is a
// UsdVolOpenVDBAsset, between the camera and a sun-lit plane, rendered by
// the rt technique. The volume's constant primvars make it absorb
// (athenea:albedo 0); the sun casts no shadows. Judged as the technique-level
// test judges it: each pixel's ratio against the stage without the volume
// on Beer-Lambert along its own ray, within five binomial deviations of the
// path count, and the pixels outside the box unchanged.
TEST_CASE("a UsdVol volume with an OpenVDB field absorbs as Beer-Lambert says through Hydra",
          "[usd][gpu][mesh][volume]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("hydra_box.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    // The volume's medium said three ways: not at all, by this engine's
    // primvars, and by the Material UsdVol binds it to -- a MaterialX
    // `volume` whose `absorption_vdf` absorbs one per unit density, which is
    // densityScale 1 and albedo 0 in the primvars' words -- so the last two
    // must draw the same frame. And, for the emission check below, the same
    // Material with a `uniform_edf`.
    const auto stage = [&](const char* name, bool volume, bool material, const char* emission) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Plane\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -4), (4, -4, -4), (4, 4, -4), (-4, 4, -4)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DistantLight \"Sun\"\n{\n"
               "    float inputs:intensity = 3\n    float inputs:angle = 0\n"
               "    bool inputs:shadow:enable = 0\n}\n";
        if (volume) {
            out << "def Volume \"Box\"" << (material ? " (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)" : "")
                << "\n{\n"
                   "    double3 xformOp:translate = (0, -1.6, -3.8)\n"
                   "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
                   "    rel field:density = </Box/density>\n";
            if (material) {
                out << "    rel material:binding = </Materials/Fog>\n";
            } else {
                out << "    float primvars:athenea:densityScale = 1\n"
                       "    color3f primvars:athenea:albedo = (0, 0, 0)\n";
            }
            out << "    def OpenVDBAsset \"density\"\n    {\n"
                   "        asset filePath = @" << vdb.string() << "@\n"
                   "        token fieldName = \"density\"\n    }\n}\n";
        }
        if (material) {
            out << "def Scope \"Materials\"\n{\n"
                   "    def Material \"Fog\"\n    {\n"
                   "        token outputs:mtlx:volume.connect = </Materials/Fog/Volume.outputs:out>\n"
                   "        def Shader \"Volume\"\n        {\n"
                   "            uniform token info:id = \"ND_volume\"\n"
                   "            token inputs:vdf.connect = </Materials/Fog/Absorb.outputs:out>\n"
                << (emission != nullptr ? "            token inputs:edf.connect = </Materials/Fog/Glow.outputs:out>\n" : "")
                << "            token outputs:out\n        }\n"
                   "        def Shader \"Absorb\"\n        {\n"
                   "            uniform token info:id = \"ND_absorption_vdf\"\n"
                   "            float3 inputs:absorption = (1, 1, 1)\n"
                   "            token outputs:out\n        }\n";
            if (emission != nullptr) {
                out << "        def Shader \"Glow\"\n        {\n"
                       "            uniform token info:id = \"ND_uniform_edf\"\n"
                       "            color3f inputs:color = " << emission << "\n"
                       "            token outputs:out\n        }\n";
            }
            out << "    }\n}\n";
        }
        return path;
    };
    const uint32_t w = 161;
    const uint32_t h = 121;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    const render::Projection projection = render::projectionFor(camera, w, h);
    constexpr uint32_t kPaths = 1024;
    const auto render = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
    // Pixel by pixel, so the camera stays in the middle of the pixel: an
    // antialiased edge is a mean over the passes and this compares frames.
    REQUIRE(renderer);
    (*renderer)->setAntialias(false);
        (*renderer)->setPathSamples(256);
        (*renderer)->setPathBounces(0);
        (*renderer)->setPathTotal(kPaths);
        auto image = (*renderer)->render(camera, 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * 4;
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    gpu::Buffer without = render(stage("volume_off.usda", false, false, nullptr));
    gpu::Buffer with = render(stage("volume_on.usda", true, false, nullptr));
    gpu::Buffer byMaterial = render(stage("volume_material.usda", true, true, nullptr));
    {
        // The same medium either way, to the noise a medium's free flight
        // leaves between two runs (its random walk is not bit-exact across
        // dispatches; a run against itself differs by a handful of words).
        auto same = render::compareImages(*gpu->library, with, byMaterial, w, h);
        REQUIRE(same);
        std::printf("  by Material against by primvars: p99 %u max %u, %llu pixels over 2\n", same->p99, same->max,
                    static_cast<unsigned long long>(same->over2));
        CHECK(same->p99 <= 1);
        CHECK(same->over2 < 20);
    }
    // And glowing: with the volume absorbing one per unit density and albedo
    // 0, what a pixel gains over the absorbing frame is L_e (1 - T), where
    // T is what the absorbing frame is of the frame without (E - A + L_e A/B
    // = L_e, volume_emission_check.slang).
    {
        gpu::Buffer glowing = render(stage("volume_glowing.usda", true, true, "(1, 0.5, 0.25)"));
        auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_emission_check", "volumeEmissionCheck");
        if (!check) FAIL(check.error().toString());
        gpu::Buffer glowSums = test::uintBuffer(*gpu->device, 4, "volume.glow.sums");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor c) {
                c["without"].setBinding(without.rhi());
                c["absorbing"].setBinding(with.rhi());
                c["glowing"].setBinding(glowing.rhi());
                c["sums"].setBinding(glowSums.rhi());
                c["check"]["pixels"].setData(w * h);
                c["check"]["minBackground"].setData(0.2F);
                c["check"]["scale"].setData(4096.0F);
                const float emission[3] = {1.0F, 0.5F, 0.25F};
                c["check"]["emission"].setData(emission, sizeof(emission));
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 4> read{};
        REQUIRE(glowSums.read(*gpu->device, 0, sizeof(read), read.data()));
        const double n = std::max<double>(read[3], 1.0) * 4096.0;
        std::printf("  glowing: over %u pixels through the volume, E - A + L_e A/B reads (%.3f, %.3f, %.3f) for L_e "
                    "(1, 0.5, 0.25)\n",
                    read[3], read[0] / n, read[1] / n, read[2] / n);
        CHECK(read[3] > 1000);
        CHECK(read[0] / n == Catch::Approx(1.0).margin(0.05));
        CHECK(read[1] / n == Catch::Approx(0.5).margin(0.03));
        CHECK(read[2] / n == Catch::Approx(0.25).margin(0.02));
    }
    auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_render_check", "volumeRenderCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 5, "volume.hydra.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 4, "volume.hydra.sums");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
            technique::setCamera(c["camera"], projection, w, h);
            const std::array<float, 12> rows = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
            c["toWorld"]["row0"].setData(rows.data(), sizeof(float) * 4);
            c["toWorld"]["row1"].setData(rows.data() + 4, sizeof(float) * 4);
            c["toWorld"]["row2"].setData(rows.data() + 8, sizeof(float) * 4);
            const float boxMin[3] = {0.0F, -1.6F, -3.8F};
            const float boxMax[3] = {3.2F, 1.6F, -0.6F};
            const float towardLight[3] = {0.0F, 0.0F, 1.0F};
            c["check"]["boxMin"].setData(boxMin, sizeof(boxMin));
            c["check"]["boxMax"].setData(boxMax, sizeof(boxMax));
            c["check"]["towardLight"].setData(towardLight, sizeof(towardLight));
            c["check"]["planeZ"].setData(-4.0F);
            c["check"]["sigma"].setData(0.5F);
            c["check"]["paths"].setData(static_cast<float>(kPaths));
            c["check"]["deviations"].setData(5.0F);
            c["check"]["shadows"].setData(uint32_t{0});
            c["with"].setBinding(with.rhi());
            c["without"].setBinding(without.rhi());
            c["counts"].setBinding(counts.rhi());
            c["zBits"].setBinding(sums.rhi());
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[5] = {};
    float s[4] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
    REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
    const double meanSquare = n[0] > 0 ? s[0] / n[0] : 0.0;
    std::printf("  through Hydra: %u pixels through the box, %u beyond 5 binomial deviations, mean z^2 %.3f, mean "
                "ratio %.4f against %.4f; %u outside, %u changed; %u uncovered\n",
                n[0], n[1], meanSquare, n[0] ? s[1] / n[0] : 0.0F, n[0] ? s[2] / n[0] : 0.0F, n[2], n[3], n[4]);
    CHECK(n[0] > 3000);
    CHECK(n[1] == 0);
    CHECK(meanSquare > 0.7);
    CHECK(meanSquare < 1.3);
    CHECK(n[2] > 3000);
    CHECK(n[3] == 0);
}


// A frame of volumes alone (M9): no mesh, a medium that scatters everything
// and absorbs nothing, under a dome of radiance 1. The traced technique
// takes the mesh layer for the volume even with no mesh, every sample walks
// its camera ray, and a pixel the medium touched reads the dome's radiance
// over its opacity -- judged as the technique-level furnace is, within five
// standard errors measured from the pixels' spread.
TEST_CASE("a frame of volumes alone is path traced through Hydra, and an albedo-one medium reads the dome",
          "[usd][gpu][volume][furnace]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const fs::path vdb = scratch("hydra_furnace.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.5F};
    REQUIRE(io::writeVdbBoxes(vdb, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    const fs::path path = scratch("volume_alone.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Volume \"Cloud\"\n{\n"
               "    double3 xformOp:translate = (-1.6, -1.6, -5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    rel field:density = </Cloud/density>\n"
               "    float primvars:athenea:densityScale = 0.2\n"
               "    color3f primvars:athenea:albedo = (1, 1, 1)\n"
               "    def OpenVDBAsset \"density\"\n    {\n"
               "        asset filePath = @" << vdb.string() << "@\n"
               "        token fieldName = \"density\"\n    }\n}\n";
    }
    const uint32_t w = 81;
    const uint32_t h = 61;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -1.0});
    camera.lens.focal = 35.0;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathSamples(64);
    (*renderer)->setPathBounces(32);
    (*renderer)->setPathTotal(256);
    auto image = (*renderer)->render(camera, 0.0, w, h, "rt");
    if (!image) FAIL(image.error().toString());
    test::dumpPpm("volume_alone_hydra", image->rgba.data(), w, h);
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * 4;
    desc.elementBytes = 16;
    auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(frame);
    auto made = gpu::ComputeKernel::create(*gpu->library, "athenea/test/volume_render_check", "volumeFurnaceCheck");
    if (!made) FAIL(made.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "furnace.hydra.counts");
    gpu::Buffer sums = test::uintBuffer(*gpu->device, 2, "furnace.hydra.sums");
    {
        gpu::CommandBatch batch(*gpu->device);
        made->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor c) {
            c["furnace"]["pixels"].setData(w * h);
            c["furnace"]["minAlpha"].setData(0.25F);
            c["with"].setBinding(frame->rhi());
            c["counts"].setBinding(counts.rhi());
            c["zBits"].setBinding(sums.rhi());
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t n[2] = {};
    float s[2] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(n), n));
    REQUIRE(sums.read(*gpu->device, 0, sizeof(s), s));
    const double count = n[0];
    const double mean = count > 0 ? s[0] / count : 0.0;
    const double spread = count > 1 ? std::sqrt(std::max(s[1] / count - mean * mean, 0.0)) : 0.0;
    const double standardError = spread / std::sqrt(std::max(count, 1.0));
    std::printf("  volumes alone: %u pixels with opacity over 0.25 (of %u touched): mean radiance %.5f, spread %.4f, "
                "%.2f standard errors from 1\n",
                n[0], n[1], mean, spread, (mean - 1.0) / std::max(standardError, 1e-9));
    CHECK(n[0] > 1000);
    CHECK(std::abs(mean - 1.0) < 5.0 * standardError);
}

// Authored normals and a dome, through Hydra. Under a dome of radiance 1
// with no image, a Lambert plane of albedo 0.18 reads exactly 0.18 --
// cosine sampling makes the estimate the albedo at every sample -- whether
// its normals are computed or authored, on both techniques. With authored
// normals it read 0: the view-space geometric normal came out reversed
// under the view's reflection, the backface test flipped the authored
// normal away, and only the dome, which samples about that normal, showed
// it (surface.slang says the rest).
TEST_CASE("a plane with authored normals under a dome reads its albedo through Hydra, on both techniques",
          "[usd][gpu][mesh][lights][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 64, h = 48;
    std::vector<float> expected(size_t{w} * h * 4);
    for (size_t p = 0; p < size_t{w} * h; ++p) {
        expected[p * 4] = expected[p * 4 + 1] = expected[p * 4 + 2] = 0.18F;
        expected[p * 4 + 3] = 1.0F;
    }
    gpu::BufferDesc desc;
    desc.bytes = expected.size() * 4;
    desc.elementBytes = 16;
    auto reference = gpu::Buffer::create(*gpu->device, desc, expected.data());
    REQUIRE(reference);
    for (const bool authored : {false, true}) {
        const fs::path path = scratch(authored ? "dome_normals_authored.usda" : "dome_normals_computed.usda");
        {
            std::ofstream out(path);
            // The square fills the view: 4 wide at distance 1.5 against a 35mm lens.
            out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
                   "def Mesh \"Square\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
                << (authored ? "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
                             : "")
                << "    uniform token subdivisionScheme = \"none\"\n}\n"
                   "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
                   "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
                   "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                   "    float2 clippingRange = (0.1, 1000)\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const char* technique : {"raster", "rt"}) {
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(frame);
            auto difference = render::compareHdr(*gpu->library, *frame, *reference, w, h);
            REQUIRE(difference);
            std::printf("  normals %s, %s: %llu pixels, worst %.2e relative to 0.18\n", authored ? "authored" : "computed",
                        technique, static_cast<unsigned long long>(difference->pixels), difference->relMse, difference->p99Relative, difference->maxRelative);
            CHECK(difference->pixels == uint64_t{w} * h);
            CHECK(difference->maxRelative < 1e-4);
        }
    }
}

// What the viewer's Technique selector does: the same renderer, one frame
// after another, told a different technique. The frame it draws must be the
// frame a renderer drawing that technique from the start draws -- not the
// technique it was drawing before. A floor and a red wall under one sphere
// light, so that the path tracer's bounce puts red on the floor where the
// raster has none, and the two techniques cannot pass for each other.
TEST_CASE("a renderer told another technique between frames draws that technique", "[usd][gpu][mesh][technique]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs rasterisation and rays");
    }
    const uint32_t w = 64, h = 48;
    const fs::path path = scratch("technique_switch.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -1, 2), (4, -1, 2), (4, -1, -6), (-4, -1, -6)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -1, -4), (4, -1, -4), (4, 4, -4), (-4, 4, -4)]\n"
               "    color3f[] primvars:displayColor = [(0.9, 0.1, 0.1)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def SphereLight \"Bulb\"\n{\n    float inputs:intensity = 30\n    float inputs:radius = 0.3\n"
               "    double3 xformOp:translate = (0, 2, -2)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    render::Camera camera = render::Camera::lookingAt({0.0, 0.5, 2.0}, {0.0, 0.0, -3.0});
    camera.lens.focal = 30.0;
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{w} * h * 16;
    desc.elementBytes = 16;
    // Draws `sequence` in order on one renderer and returns the last frame.
    const auto lastOf = [&](std::initializer_list<const char*> sequence) -> gpu::Buffer {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        const char* last = nullptr;
        size_t k = 0;
        for (const char* technique : sequence) {
            if (++k == sequence.size()) {
                last = technique;
                break;
            }
            auto drawn = (*renderer)->draw(camera, 0.0, w, h, technique);
            if (!drawn) FAIL(drawn.error().toString());
        }
        auto image = (*renderer)->render(camera, 0.0, w, h, last);
        if (!image) FAIL(image.error().toString());
        auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        if (!frame) FAIL(frame.error().toString());
        return std::move(*frame);
    };
    const gpu::Buffer raster = lastOf({"raster", "raster"});
    const gpu::Buffer rt = lastOf({"rt", "rt"});
    const gpu::Buffer rasterThenRt = lastOf({"raster", "raster", "rt", "rt"});
    const gpu::Buffer rtThenRaster = lastOf({"rt", "rt", "raster", "raster"});
    const auto compare = [&](const gpu::Buffer& a, const gpu::Buffer& b, const char* what) {
        auto d = render::compareHdr(*gpu->library, a, b, w, h);
        REQUIRE(d);
        std::printf("  %s: relMse %.3e, worst %.2e\n", what, d->relMse, d->maxRelative);
        return *d;
    };
    // The control: the techniques differ, or the test cannot see a switch.
    CHECK(compare(raster, rt, "raster against rt").maxRelative > 1e-2);
    CHECK(compare(rasterThenRt, rt, "raster then rt, against rt").maxRelative < 1e-4);
    CHECK(compare(rtThenRaster, raster, "rt then raster, against raster").maxRelative < 1e-4);

    // And what a window does between switches: frame after frame of one
    // camera. A viewport keeps gathering -- a frame adds its paths whatever
    // the total, which only says when the frame counts as converged -- and a
    // switch back to raster reads as finished.
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    REQUIRE((*renderer)->draw(camera, 0.0, w, h, "rt"));
    REQUIRE((*renderer)->draw(camera, 0.0, w, h, "rt"));
    const uint32_t stuck = (*renderer)->pathAccumulated();
    (*renderer)->setPathTotal(8);
    std::vector<uint32_t> gathered;
    for (int frame = 0; frame < 10; ++frame) {
        REQUIRE((*renderer)->draw(camera, 0.0, w, h, "rt"));
        gathered.push_back((*renderer)->pathAccumulated());
    }
    std::printf("  %u paths after two frames; with a total of 8, frame by frame:", stuck);
    for (const uint32_t g : gathered) {
        std::printf(" %u", g);
    }
    std::printf("%s\n", (*renderer)->pathConverged() ? " (converged)" : "");
    CHECK(stuck == 2);
    for (size_t k = 0; k < gathered.size(); ++k) {
        CHECK(gathered[k] == stuck + 1 + k);
    }
    CHECK((*renderer)->pathConverged());
    REQUIRE((*renderer)->draw(camera, 0.0, w, h, "raster"));
    CHECK((*renderer)->pathConverged());
}

// A stage that authors no lights, as the chess set and Kitchen_set do: path
// traced, it is lit from the eye like the raster and has nothing to bounce.
// The viewer's default lights are a sky and a sun in the session layer. They
// must reach the frame (it changes), leave the file alone (the stage still
// authors none), and go away again exactly (the frame is the unlit one).
TEST_CASE("default lights in the session layer light a stage that has none, and leave it as it was",
          "[usd][gpu][mesh][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs rasterisation and rays");
    }
    const uint32_t w = 64, h = 48;
    const fs::path path = scratch("default_lights.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -1, 2), (4, -1, 2), (4, -1, -6), (-4, -1, -6)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -1, -4), (4, -1, -4), (4, 4, -4), (-4, 4, -4)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n";
    }
    render::Camera camera = render::Camera::lookingAt({0.0, 0.5, 2.0}, {0.0, 0.0, -3.0});
    camera.lens.focal = 30.0;
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{w} * h * 16;
    desc.elementBytes = 16;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    // Pixel by pixel, so the camera stays in the middle of the pixel: an
    // antialiased edge is a mean over the passes and this compares frames.
    REQUIRE(renderer);
    (*renderer)->setAntialias(false);
    const auto frame = [&](const char* technique) -> gpu::Buffer {
        auto image = (*renderer)->render(camera, 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    CHECK_FALSE((*renderer)->hasLights());
    for (const char* technique : {"raster", "rt"}) {
        const gpu::Buffer unlit = frame(technique);
        REQUIRE((*renderer)->setDefaultLights(true));
        CHECK_FALSE((*renderer)->hasLights());   // the session's lights are not the stage's
        const gpu::Buffer lit = frame(technique);
        REQUIRE((*renderer)->setDefaultLights(false));
        const gpu::Buffer again = frame(technique);
        auto changed = render::compareHdr(*gpu->library, lit, unlit, w, h);
        auto restored = render::compareHdr(*gpu->library, again, unlit, w, h);
        REQUIRE(changed);
        REQUIRE(restored);
        std::printf("  %s: lit against unlit relMse %.3e; lights removed against unlit, worst %.2e\n", technique,
                    changed->relMse, restored->maxRelative);
        CHECK(changed->relMse > 1e-2);
        CHECK(restored->maxRelative < 1e-4);
    }
}

// A glass pane under a dome, drawn by the raster, which has no rays for what
// is behind it. Before the raster looked the dome up along what light
// sampling cannot reach -- the far side of the surface, and a delta lobe
// that answers no light sample -- glass drew black: the chess set's pawn
// heads. Two checks, each with an answer that owes nothing to the raster:
//
//  - a bare smooth dielectric_bsdf in scatter mode RT is lossless: every
//    sample goes to reflection or transmission with a weight of exactly one,
//    so under a uniform dome of radiance 1 every pixel reads 1;
//  - standard_surface with transmission (the pawns' own material) is not --
//    MaterialX layers its specular reflection over the transmission, which
//    it attenuates by one minus the reflection's albedo, so a pane reads
//    F + (1 - F)^2 and not 1 -- but with nothing behind the pane the path
//    tracer sees exactly what the raster should, so the two must agree.
TEST_CASE("the raster sees a dome through glass as the path tracer does", "[usd][gpu][mesh][lights][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs rasterisation and rays");
    }
    const uint32_t w = 64, h = 48;
    const auto stageWith = [&](const std::string& name, const std::string& shader) {
        const fs::path material = scratch(name + ".mtlx");
        {
            std::ofstream out(material);
            out << "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n" << shader
                << "  <surfacematerial name=\"M_Glass\" type=\"material\">\n"
                   "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"Glass\" />\n"
                   "  </surfacematerial>\n</materialx>\n";
        }
        const fs::path path = scratch(name + ".usda");
        std::ofstream out(path);
        // The pane fills the view: 8 wide at distance 1.5 against a 35mm lens.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Scope \"Looks\" (\n    prepend references = @./" << name << ".mtlx@</MaterialX/Materials>\n)\n{\n}\n"
               "def Mesh \"Pane\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Looks/M_Glass>\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
        return path;
    };
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{w} * h * 16;
    desc.elementBytes = 16;
    const auto upload = [&](const std::vector<float>& rgba) {
        auto made = gpu::Buffer::create(*gpu->device, desc, rgba.data());
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };

    {
        const fs::path path = stageWith(
            "glass_lossless",
            "  <dielectric_bsdf name=\"b\" type=\"BSDF\">\n"
            "    <input name=\"roughness\" type=\"vector2\" value=\"0, 0\" />\n"
            "    <input name=\"scatter_mode\" type=\"string\" value=\"RT\" />\n"
            "  </dielectric_bsdf>\n"
            "  <surface name=\"Glass\" type=\"surfaceshader\">\n"
            "    <input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" />\n  </surface>\n");
        const gpu::Buffer ones = upload(std::vector<float>(size_t{w} * h * 4, 1.0F));
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        for (const uint32_t lightSamples : {1u, 4u}) {
            (*renderer)->setLightSamples(lightSamples);
            auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
            if (!image) FAIL(image.error().toString());
            auto difference = render::compareHdr(*gpu->library, upload(image->rgba), ones, w, h);
            REQUIRE(difference);
            std::printf("  lossless dielectric, %u light samples: %llu pixels, worst %.2e relative to the dome\n",
                        lightSamples, static_cast<unsigned long long>(difference->pixels), difference->maxRelative);
            CHECK(difference->pixels == uint64_t{w} * h);
            CHECK(difference->maxRelative < 1e-3);
        }
    }
    {
        const fs::path path = stageWith(
            "glass_standard",
            "  <standard_surface name=\"Glass\" type=\"surfaceshader\">\n"
            "    <input name=\"base_color\" type=\"color3\" value=\"1, 1, 1\" />\n"
            "    <input name=\"specular_roughness\" type=\"float\" value=\"0\" />\n"
            "    <input name=\"transmission\" type=\"float\" value=\"1\" />\n"
            "    <input name=\"transmission_color\" type=\"color3\" value=\"0.3, 0.5, 0.45\" />\n"
            "  </standard_surface>\n");
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        // The path tracer first, deep, as the answer; then the raster at two
        // sample counts. Its lobe choice between reflection and a tinted
        // transmission is noise, so agreement is stated as the error falling
        // with the samples as one over their number, not as a threshold met
        // once.
        (*renderer)->setPathBounces(2);
        (*renderer)->setPathTotal(4096);
        auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!traced) FAIL(traced.error().toString());
        const gpu::Buffer answer = upload(traced->rgba);
        double errors[2] = {0.0, 0.0};
        const uint32_t counts[2] = {2u, 32u};
        for (size_t c = 0; c < 2; ++c) {
            (*renderer)->setLightSamples(counts[c]);
            auto raster = (*renderer)->render("/Camera", 0.0, w, h, "raster");
            if (!raster) FAIL(raster.error().toString());
            auto difference = render::compareHdr(*gpu->library, upload(raster->rgba), answer, w, h);
            REQUIRE(difference);
            std::printf("  standard_surface glass, raster at %u samples against rt at 4096 paths: relMse %.2e, p99 %.2e\n",
                        counts[c], difference->relMse, difference->p99Relative);
            CHECK(difference->pixels == uint64_t{w} * h);
            errors[c] = difference->relMse;
        }
        // Sixteen times the samples -- within the 32 lobe samples the raster
        // draws a pixel -- and an unbiased estimator's squared error is a
        // sixteenth, less the path tracer's own noise in both; a bias would
        // leave it where it was.
        CHECK(errors[0] / errors[1] > 8.0);   // 16.0 measured
        CHECK(errors[1] < 5e-4);   // 1.24e-04 measured at 32
    }
    {
        // A polished metal: a lobe too narrow for a dome's light samples to
        // find and not narrow enough to be a delta, which is where the two
        // strategies are weighed against each other. Before, the chess set's
        // rims drew black; the same one-over-N as the glass says the weights
        // add up to one.
        const fs::path path = stageWith(
            "metal_glossy",
            "  <conductor_bsdf name=\"b\" type=\"BSDF\">\n"
            "    <input name=\"roughness\" type=\"vector2\" value=\"0.1, 0.1\" />\n"
            "  </conductor_bsdf>\n"
            "  <surface name=\"Glass\" type=\"surfaceshader\">\n"
            "    <input name=\"bsdf\" type=\"BSDF\" nodename=\"b\" />\n  </surface>\n");
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathBounces(1);
        (*renderer)->setPathTotal(4096);
        auto traced = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!traced) FAIL(traced.error().toString());
        const gpu::Buffer answer = upload(traced->rgba);
        double errors[2] = {0.0, 0.0};
        const uint32_t counts[2] = {2u, 32u};
        for (size_t c = 0; c < 2; ++c) {
            (*renderer)->setLightSamples(counts[c]);
            auto raster = (*renderer)->render("/Camera", 0.0, w, h, "raster");
            if (!raster) FAIL(raster.error().toString());
            auto difference = render::compareHdr(*gpu->library, upload(raster->rgba), answer, w, h);
            REQUIRE(difference);
            std::printf("  polished conductor, raster at %u samples against rt at 4096 paths: relMse %.2e, p99 %.2e\n",
                        counts[c], difference->relMse, difference->p99Relative);
            CHECK(difference->pixels == uint64_t{w} * h);
            errors[c] = difference->relMse;
        }
        CHECK(errors[0] / errors[1] > 8.0);
    }
}

namespace {

/// A lat-long sky of `columns` x `rows` cells in four colours (ABGR), so what
/// a glass shows through it is a picture and where it bends to can be seen.
/// Four and not two: a ball lens turns what is behind it over, and a
/// two-colour checker turned over about a corner is the same checker.
fs::path checkerSky(const std::string& name, uint32_t columns, uint32_t rows) {
    const fs::path png = scratch(name);
    fs::remove(png);
    const uint32_t palette[4] = {0xFF2050E0u, 0xFFE0D0A0u, 0xFF30B040u, 0xFF101010u};
    const uint32_t w = 128, h = 64;
    std::vector<uint32_t> texels(size_t{w} * h);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            texels[size_t{y} * w + x] = palette[((x * columns / w) + 2 * (y * rows / h)) % 4];
        }
    }
    HioImageSharedPtr image = HioImage::OpenForWriting(png.string());
    REQUIRE(image);
    HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = HioFormatUNorm8Vec4;
    spec.data = texels.data();
    REQUIRE(image->Write(spec));
    return png;
}

/// A standard_surface glass as a MaterialX file, `M_Glass` under /MaterialX.
fs::path glassLook(const std::string& name, float roughness, const std::string& tint) {
    const fs::path material = scratch(name + ".mtlx");
    std::ofstream out(material);
    out << "<?xml version=\"1.0\"?>\n<materialx version=\"1.39\">\n"
           "  <standard_surface name=\"Glass\" type=\"surfaceshader\">\n"
           "    <input name=\"base_color\" type=\"color3\" value=\"1, 1, 1\" />\n"
           "    <input name=\"specular_roughness\" type=\"float\" value=\""
        << roughness
        << "\" />\n"
           "    <input name=\"specular_IOR\" type=\"float\" value=\"1.5\" />\n"
           "    <input name=\"transmission\" type=\"float\" value=\"1\" />\n"
           "    <input name=\"transmission_color\" type=\"color3\" value=\""
        << tint
        << "\" />\n"
           "  </standard_surface>\n"
           "  <surfacematerial name=\"M_Glass\" type=\"material\">\n"
           "    <input name=\"surfaceshader\" type=\"surfaceshader\" nodename=\"Glass\" />\n"
           "  </surfacematerial>\n</materialx>\n";
    return material;
}

/// A ball of radius one at the origin, `slices` around and `stacks` from pole
/// to pole, as a Mesh prim `name` bound to `material`. Hydra's own sphere is
/// ten facets around, which is a lens of facets.
std::string ballMesh(const std::string& name, const std::string& material, bool doubleSided,
                     uint32_t slices = 96, uint32_t stacks = 48) {
    std::ostringstream out;
    out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
        << "    uniform bool doubleSided = " << (doubleSided ? "1" : "0") << "\n"
        << "    uniform token subdivisionScheme = \"none\"\n    rel material:binding = <" << material << ">\n";
    const auto vertices = [&](const char* attribute) {
        out << "    " << attribute << " = [";
        for (uint32_t j = 0; j <= stacks; ++j) {
            const double theta = 3.14159265358979 * j / stacks;
            for (uint32_t i = 0; i < slices; ++i) {
                const double phi = 2.0 * 3.14159265358979 * i / slices;
                out << (j + i ? ", " : "") << "(" << std::sin(theta) * std::cos(phi) << ", " << std::cos(theta)
                    << ", " << -std::sin(theta) * std::sin(phi) << ")";
            }
        }
        out << "]";
    };
    vertices("point3f[] points");
    out << "\n";
    vertices("normal3f[] normals");
    out << " (interpolation = \"vertex\")\n    int[] faceVertexCounts = [";
    for (uint32_t k = 0; k < slices * stacks; ++k) {
        out << (k ? ", " : "") << 4;
    }
    // Counter-clockwise seen from outside, which is USD's right-handed front.
    out << "]\n    int[] faceVertexIndices = [";
    for (uint32_t j = 0; j < stacks; ++j) {
        for (uint32_t i = 0; i < slices; ++i) {
            const uint32_t a = j * slices + i;
            const uint32_t b = j * slices + (i + 1) % slices;
            const uint32_t c = (j + 1) * slices + (i + 1) % slices;
            const uint32_t d = (j + 1) * slices + i;
            out << (j + i ? ", " : "") << a << ", " << d << ", " << c << ", " << b;
        }
    }
    out << "]\n}\n";
    return out.str();
}

/// The camera every glass ball below is seen from: the ball of radius one
/// at the origin nearly fills it.
const char* const kBallCamera =
    "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
    "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
    "    float2 clippingRange = (0.1, 1000)\n"
    "    double3 xformOp:translate = (0, 0, 3.5)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";

}   // namespace

// A RAY THAT WENT INTO A GLASS MEETS ITS FAR FACE.
//
// Back faces were culled for every ray the path tracer traced after the
// first hit, which is what a single-sided mesh asks of a ray that bounced off
// it -- and wrong for one that went through: inside a solid glass ball the
// far face is a back face, so the ray left without bending again and the
// ball showed the room bent once, a thick lens drawn as one interface. The
// chess pawn's glass head was the reference a converted cloud was measured
// against, and the cloud, which bends twice, came out 0.05 relMSE from it for
// being right. A double-sided ball was not culled and drew black instead: at
// the default of one bounce the ray met the far face with nothing left to
// leave by. Now a crossing sees back faces and costs no bounce, so the two
// balls are one picture, and one bounce draws what four do.
TEST_CASE("a glass ball bends at its far face whether or not it is double-sided, at one bounce",
          "[usd][gpu][mesh][path][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path sky = checkerSky("ball_sky.png", 8, 4);
    glassLook("ball_glass", 0.0F, "1, 1, 1");
    const auto stage = [&](const std::string& name, bool doubleSided) {
        const fs::path path = scratch(name + ".usda");
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Scope \"Looks\" (\n    prepend references = @./ball_glass.mtlx@</MaterialX/Materials>\n)\n{\n}\n"
            << ballMesh("Ball", "/Looks/M_Glass", doubleSided)
            << "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n"
               "    asset inputs:texture:file = @"
            << sky.string() << "@\n}\n"
            << kBallCamera;
        return path;
    };
    const uint32_t w = 64, h = 64;
    const auto render = [&](const fs::path& path, uint32_t bounces) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathBounces(bounces);
        (*renderer)->setPathTotal(256);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    const gpu::Buffer single = render(stage("ball_single", false), 1);
    const gpu::Buffer twice = render(stage("ball_double", true), 1);
    const gpu::Buffer deep = render(stage("ball_double_deep", true), 4);
    auto sided = render::compareHdr(*gpu->library, single, twice, w, h);
    auto bounces = render::compareHdr(*gpu->library, twice, deep, w, h);
    REQUIRE(sided);
    REQUIRE(bounces);
    std::printf("  single- against double-sided: relMse %.2e, p99 %.3g; one bounce against four: relMse %.2e, p99 %.3g\n",
                sided->relMse, sided->p99Relative, bounces->relMse, bounces->p99Relative);
    CHECK(sided->pixels == uint64_t{w} * h);
    CHECK(sided->relMse < 0.01);
    CHECK(sided->p99Relative < 0.3);
    CHECK(bounces->relMse < 0.01);
    CHECK(bounces->p99Relative < 0.3);
}

namespace {

/// A GLASS BALL AS mesh2splat MAKES ONE: a shell of flat gaussians on the
/// sphere of radius one, each facing out, a cell wide, at the conversion's
/// glass opacity (0.6), its colour the transmission tint (base colour one),
/// transmitting and as rough as the material. Written as the conversion
/// writes it -- relit, with the index -- and placed with the mesh's camera
/// and `sky` in a stage of its own. Fibonacci points: evenly spaced, no seam.
///
/// With `card`, the cloud also holds an opaque square of that colour behind
/// the ball -- 6 wide at z = -6, facing the camera -- which is what a ray
/// through the glass meets instead of the sky.
fs::path glassBallCloudStage(test::Gpu* gpu, const std::string& name, uint32_t count,
                             const std::array<float, 3>& tint, float roughness, const std::string& sky,
                             const std::array<float, 3>* card = nullptr) {
    io::RawSplats raw;
    raw.source = "glass ball";
    io::SplatEncoding& e = raw.encoding;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.metallic = 14; e.roughness = 15; e.transmission = 16;
    e.restBase = 17; e.restPerColour = 0; e.restColourOuter = 0;
    e.floatsPerRecord = 17;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::LinearLight;
    e.rest = io::SplatEncoding::Rest::Float;
    e.rotation = io::SplatEncoding::Rotation::Float;
    const double cell = std::sqrt(4.0 * 3.14159265358979 / count);
    const double golden = 3.14159265358979 * (3.0 - std::sqrt(5.0));
    for (uint32_t i = 0; i < count; ++i) {
        const double y = 1.0 - 2.0 * (i + 0.5) / count;
        const double ring = std::sqrt(std::max(0.0, 1.0 - y * y));
        const double x = ring * std::cos(golden * i);
        const double z = ring * std::sin(golden * i);
        // The rotation taking the third axis, the short one, to the normal.
        double qw = 1.0 + z, qx = -y, qy = x;
        if (qw < 1e-6) {
            qw = 0.0; qx = 1.0; qy = 0.0;
        }
        const double qn = std::sqrt(qw * qw + qx * qx + qy * qy);
        const float record[17] = {float(x), float(y), float(z), 0.6F,
                                  float(cell), float(cell), float(0.1 * cell),
                                  float(qw / qn), float(qx / qn), float(qy / qn), 0.0F,
                                  tint[0], tint[1], tint[2],
                                  0.0F, roughness, 1.0F};
        raw.records.insert(raw.records.end(), record, record + 17);
        raw.count += 1;
    }
    if (card != nullptr) {
        const uint32_t side = 120;
        const float cell = 6.0F / side;
        for (uint32_t j = 0; j < side; ++j) {
            for (uint32_t i = 0; i < side; ++i) {
                const float record[17] = {-3.0F + cell * (i + 0.5F), -3.0F + cell * (j + 0.5F), -6.0F, 0.99F,
                                          cell, cell, 0.1F * cell,
                                          1.0F, 0.0F, 0.0F, 0.0F,
                                          (*card)[0], (*card)[1], (*card)[2],
                                          0.0F, 1.0F, 0.0F};
                raw.records.insert(raw.records.end(), record, record + 17);
                raw.count += 1;
            }
        }
    }
    const fs::path cloud = scratch(name + "_splats.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    options.ior = 1.5F;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, cloud, options));
    const fs::path path = scratch(name + ".usda");
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Xform \"Ball\" (\n    prepend references = @./" << cloud.filename().string() << "@</World>\n)\n{\n}\n"
        << sky << kBallCamera;
    return path;
}

/// The same ball as a mesh with a `standard_surface` glass of that tint and
/// roughness, under the same `sky`.
///
/// With `card`, the square glassBallCloudStage puts behind the ball, as a
/// mesh with a diffuse UsdPreviewSurface of that colour.
fs::path glassBallMeshStage(const std::string& name, const std::string& tint, float roughness,
                            const std::string& sky, const std::string& card = "") {
    glassLook(name, roughness, tint);
    const fs::path path = scratch(name + ".usda");
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Scope \"Looks\" (\n    prepend references = @./" << name << ".mtlx@</MaterialX/Materials>\n)\n{\n}\n"
        << ballMesh("Ball", "/Looks/M_Glass", false) << sky << kBallCamera;
    if (!card.empty()) {
        out << "def Material \"Card\"\n{\n"
               "    token outputs:surface.connect = </Card/Surface.outputs:surface>\n"
               "    def Shader \"Surface\"\n    {\n"
               "        uniform token info:id = \"UsdPreviewSurface\"\n"
               "        color3f inputs:diffuseColor = (" << card << ")\n"
               "        float inputs:roughness = 1\n"
               "        token outputs:surface\n    }\n}\n"
               "def Mesh \"Back\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    rel material:binding = </Card>\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-3, -3, -6), (3, -3, -6), (3, 3, -6), (-3, 3, -6)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n}\n";
    }
    return path;
}

/// A frame, rt, `paths` a pixel, on the device.
gpu::Buffer renderBall(test::Gpu* gpu, const fs::path& path, uint32_t w, uint32_t h, uint32_t paths) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->setPathTotal(paths);
    auto image = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

/// The mean colour of the `side` x `side` pixels at the middle of a frame.
std::array<double, 3> middleMean(test::Gpu* gpu, const gpu::Buffer& frame, uint32_t w, uint32_t h,
                                 uint32_t side) {
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/patch_mean", "patchMean");
    if (!kernel) FAIL(kernel.error().toString());
    const std::array<uint32_t, 4> zero{};
    auto sums = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "ball.sums");
    REQUIRE(sums);
    {
        gpu::CommandBatch batch(*gpu->device);
        kernel->dispatch(batch, {(side + 7) / 8, (side + 7) / 8, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["frame"].setBinding(frame.rhi());
            cursor["sums"].setBinding(sums->rhi());
            cursor["params"]["width"].setData(w);
            cursor["params"]["x0"].setData(w / 2 - side / 2);
            cursor["params"]["y0"].setData(h / 2 - side / 2);
            cursor["params"]["w"].setData(side);
            cursor["params"]["h"].setData(side);
            cursor["params"]["scale"].setData(65536.0F);
            cursor["params"]["withAlpha"].setData(0u);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 4> read{};
    REQUIRE(sums->read(*gpu->device, 0, sizeof(read), read.data()));
    const double n = std::max<double>(read[3], 1.0) * 65536.0;
    return {read[0] / n, read[1] / n, read[2] / n};
}

}   // namespace

// WHAT A GLASS CLOUD'S FAR FACE LETS OUT.
//
// A ray through a solid crosses two interfaces. The cloud bent at both
// (rt_glass finds the far face) and weighed only the first: what the near
// face did not reflect, tinted once by the colour mesh2splat folds the
// transmission colour into. The mesh's dielectric lobe reflects its Fresnel
// share at the far face too and tints again, so through the middle of a ball
// of tint (1, 1, 0.5) the cloud read blue at half the light where the mesh
// reads a quarter. A sky of one colour, so what is measured is the weight and
// nothing of where the ray went.
TEST_CASE("a glass cloud lets out at its far face what the mesh's glass does", "[usd][gpu][splat][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const std::string sky = "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    const uint32_t w = 96, h = 96;
    const gpu::Buffer mesh = renderBall(gpu, glassBallMeshStage("exit_mesh", "1, 1, 0.5", 0.0F, sky), w, h, 512);
    const gpu::Buffer cloud =
        renderBall(gpu, glassBallCloudStage(gpu, "exit_cloud", 60000, {1.0F, 1.0F, 0.5F}, 0.0F, sky), w, h, 16);
    const std::array<double, 3> m = middleMean(gpu, mesh, w, h, 16);
    const std::array<double, 3> c = middleMean(gpu, cloud, w, h, 16);
    std::printf("  through the middle of a ball of tint (1, 1, 0.5): mesh %.4f %.4f %.4f, cloud %.4f %.4f %.4f\n",
                m[0], m[1], m[2], c[0], c[1], c[2]);
    // 0.98 and 0.51 before, against 0.92 and 0.26; 2.6 % and 3.3 % now.
    CHECK(std::abs(c[1] - m[1]) < 0.045 * m[1]);
    CHECK(std::abs(c[2] - m[2]) < 0.045 * m[2]);
}

// WHAT A RAY MEETS BEHIND THE GLASS LEAVES THROUGH THE FAR FACE TOO.
//
// The colour a ray through a glass cloud meets behind it -- a gold collar
// under a pawn's glass head -- is the colour that particle was shaded with,
// and it crosses the far face as the sky does: its Fresnel and a second
// tint. By the precedence of `?:` the far face weighed only the sky, so a
// grey card behind a ball of tint (1, 1, 0.5) came through at half its blue
// where the mesh's glass lets a quarter of it through. The ratio of blue to
// green is the tint squared whatever the card's own shading, which is what
// is held to the mesh; green is held too, more loosely, since a relit card
// and a path-traced one need not agree to the percent.
TEST_CASE("what a glass cloud shows behind it leaves through the far face as the mesh's does",
          "[usd][gpu][splat][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const std::string sky = "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    const uint32_t w = 96, h = 96;
    const std::array<float, 3> grey{0.8F, 0.8F, 0.8F};
    const gpu::Buffer mesh =
        renderBall(gpu, glassBallMeshStage("behind_mesh", "1, 1, 0.5", 0.0F, sky, "0.8, 0.8, 0.8"), w, h, 512);
    const gpu::Buffer cloud = renderBall(
        gpu, glassBallCloudStage(gpu, "behind_cloud", 60000, {1.0F, 1.0F, 0.5F}, 0.0F, sky, &grey), w, h, 16);
    const std::array<double, 3> m = middleMean(gpu, mesh, w, h, 16);
    const std::array<double, 3> c = middleMean(gpu, cloud, w, h, 16);
    std::printf("  a grey card through a ball of tint (1, 1, 0.5): mesh %.4f %.4f %.4f, cloud %.4f %.4f %.4f\n",
                m[0], m[1], m[2], c[0], c[1], c[2]);
    REQUIRE(m[1] > 0.0);
    REQUIRE(c[1] > 0.0);
    const double meshRatio = m[2] / m[1];
    const double cloudRatio = c[2] / c[1];
    // 1.6 % and 4.5 % measured; 84 % and 8.6 % before.
    CHECK(std::abs(cloudRatio - meshRatio) < 0.06 * meshRatio);
    CHECK(std::abs(c[1] - m[1]) < 0.08 * m[1]);
}

// A COMPILER GIVEN MATERIALX LIBRARIES OF ITS OWN READS ITS DEFINITIONS THERE.
//
// $ATHENEA_MATERIALX_ROOT names the libraries hdAthenea's material compiler
// loads in place of the host USD's, and the document hdMtlx builds carries
// the host's: the compiler's definitions must win over the document's. The
// root here is a copy of the build's own libraries in which UsdPreviewSurface's
// diffuseColor defaults to red instead of 0.18 grey, and a quad whose
// UsdPreviewSurface authors no colour must come out red. Grey is what either
// half missing gives: the variable not read, or the host's node definition
// kept from the document. Hidden: ctest runs it as `materialx_root`, with the
// variable set, since the engine reads it once for the process; its one tag
// keeps a run by any other tag ([usd], [materials]) from selecting it.
TEST_CASE("a material compiler given its own MaterialX libraries takes its definitions from them",
          "[.materialx_root]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const std::string root = platform::env("ATHENEA_MATERIALX_ROOT");
    REQUIRE_FALSE(root.empty());
    const fs::path libraries = fs::path(root) / "libraries";
    std::error_code ec;
    fs::remove_all(libraries, ec);
    fs::create_directories(root);
    fs::copy(ATHENEA_TEST_MATERIALX_LIBRARIES, libraries, fs::copy_options::recursive);
    const fs::path preview = libraries / "bxdf" / "usd_preview_surface.mtlx";
    std::string text;
    {
        std::ifstream in(preview);
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    const std::string grey = "name=\"diffuseColor\" type=\"color3\" value=\"0.18, 0.18, 0.18\"";
    const size_t at = text.find(grey);
    REQUIRE(at != std::string::npos);
    text.replace(at, grey.size(), "name=\"diffuseColor\" type=\"color3\" value=\"0.8, 0.05, 0.05\"");
    {
        std::ofstream out(preview, std::ios::trunc);
        out << text;
    }
    const fs::path path = scratch("materialx_root.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Quad\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Looks/Paint>\n}\n"
               "def Scope \"Looks\"\n{\n    def Material \"Paint\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Paint/Surface.outputs:surface>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            float inputs:roughness = 1\n"
               "            token outputs:surface\n        }\n    }\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 3)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    const uint32_t w = 64, h = 64;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    REQUIRE(frame);
    const std::array<double, 3> m = middleMean(gpu, *frame, w, h, 16);
    std::printf("  an unauthored diffuseColor under the compiler's own libraries: %.4f %.4f %.4f\n", m[0], m[1], m[2]);
    // 0.810 0.063 0.063 measured: the red default, lit by a dome of one.
    CHECK(m[0] > 0.7);
    CHECK(m[0] > 8.0 * m[1]);
    CHECK(m[0] > 8.0 * m[2]);
}

// THE ROOM THROUGH A ROUGH GLASS IS SHARPER THAN ITS REFLECTION.
//
// The prepared sky's levels are reflection lobes, and the transmitted half
// read them at the material's own roughness: a microfacet tilted by theta
// turns a reflection by 2 theta and a ray through two faces of index 1.5 by
// about 0.7 theta, so a rough ball showed the room through a reflection's
// blur. A ball of roughness 0.3 under a checker sky, cloud against mesh.
TEST_CASE("a rough glass cloud blurs the room behind it as the mesh's glass does", "[usd][gpu][splat][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path png = checkerSky("rough_sky.png", 16, 8);
    const std::string sky = "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n"
                            "    asset inputs:texture:file = @" + png.string() + "@\n}\n";
    const uint32_t w = 96, h = 96;
    const gpu::Buffer mesh = renderBall(gpu, glassBallMeshStage("rough_mesh", "1, 1, 1", 0.3F, sky), w, h, 1024);
    const gpu::Buffer cloud =
        renderBall(gpu, glassBallCloudStage(gpu, "rough_cloud", 60000, {1.0F, 1.0F, 1.0F}, 0.3F, sky), w, h, 16);
    auto difference = render::compareHdr(*gpu->library, cloud, mesh, w, h);
    REQUIRE(difference);
    std::printf("  a ball of roughness 0.3, cloud against mesh: relMse %.4f, p99 %.3f\n", difference->relMse,
                difference->p99Relative);
    CHECK(difference->pixels == uint64_t{w} * h);
    // 0.119 read at the reflection's roughness, 0.045 now.
    CHECK(difference->relMse < 0.07);
}

// THE BAKE'S ANSWER CANNOT DEPEND ON HOW MANY HARMONICS IT IS ASKED FOR.
//
// A Lambertian surface sends the same radiance in every direction of the half
// of the sphere it faces. Every basis function above the constant is therefore
// zero, and the constant is the radiance -- whatever degree the bake is asked
// to fit. Two degrees that disagree about a flat surface are a defect in the
// fit and nothing else, which is what this pins down.
//
// It is what the projection did. Fitting by projecting over the sphere with
// the far half taken as nothing, a plane of albedo 0.18 under a dome came back
// at 0.045 at degree 0, 0.094 at degree 2 and 0.069 at degree 3 -- three
// answers to a question with one, and a dark rim around every silhouette in
// the model, since a silhouette is the surface seen from the equator of that
// half, where the step the projection puts there is worth half the light.
TEST_CASE("a Lambertian surface bakes to the same constant at every degree", "[usd][gpu][mesh][bake]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_lambert.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    // A row of points on the plane, each with the plane's normal and a step to
    // start its rays off the surface.
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[1] = 0.0F;
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;   // how far off the surface the rays start
        ray[6] = 1.0F;      // the normal, +z
    }
    // What the surface sends, in linear light, which is where a cloud is
    // blended and so where a bake fits: the plane reads 0.18 in a frame (the
    // test above measures it). It was 0.461, the sRGB code of 0.18, while
    // clouds were blended encoded.
    const float expected = 0.18F;
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_check", "bakeCheck");
    if (!check) FAIL(check.error().toString());

    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    for (const uint32_t degree : {0u, 2u, 3u}) {
        auto baked = (*renderer)->bakePoints(rays, count, 0.0, 1024, 1, degree);
        if (!baked) FAIL(baked.error().toString());
        const uint32_t planes = (degree + 1) * (degree + 1);
        REQUIRE(baked->size() == size_t{count} * planes * 4);
        gpu::BufferDesc desc;
        desc.bytes = baked->size() * 4;
        desc.elementBytes = 16;
        auto fitted = gpu::Buffer::create(*gpu->device, desc, baked->data());
        REQUIRE(fitted);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["coefficients"].setBinding(fitted->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["planes"].setData(planes);
                // The renderer reads the basis along the eye's line to the
                // point, which for a plane looked at straight on is -z.
                const float direction[4] = {0.0F, 0.0F, -1.0F, 0.05F};
                const float want[4] = {expected, expected, expected, 0.0F};
                p["direction"].setData(direction, sizeof(direction));
                p["expected"].setData(want, sizeof(want));
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[2] = {0, 0};
        float    got[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(counts.read(*gpu->device, 0, sizeof(seen), seen));
        REQUIRE(worst.read(*gpu->device, 0, sizeof(got), got));
        std::printf("  degree %u: %u points, %u beyond 5%%, worst %.4f, first %.4f/%.4f/%.4f against %.4f\n",
                    degree, seen[0], seen[1], double(got[0]), double(got[1]), double(got[2]), double(got[3]),
                    double(expected));
        CHECK(seen[0] == count);
        CHECK(seen[1] == 0);
    }
}

// THE BAKE'S BOX AND RAYS, ON THE DEVICE, ANSWER AS THEY DID ON THE HOST.
//
// `athenea mesh2splat` folded the cloud's box over its records on the processor,
// wrote a ray a gaussian there -- `1e-4` of the box's diagonal off the
// surface -- and uploaded them to `bakePoints`. Now `mesh2splat_span` folds
// the box and writes the offset on the device, and `bakePointsOnDevice`
// traces the rays where they are. On a known plane: every ray's offset is
// that fraction of the box the records were built in, and the device bake
// answers what the host bake answers for the same rays, entry for entry.
TEST_CASE("a bake from rays set up on the device answers as the host's did", "[usd][gpu][mesh][bake][mesh2splat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_device_plane.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def SphereLight \"Lamp\"\n{\n    float inputs:intensity = 30\n    float inputs:radius = 0.3\n"
               "    double3 xformOp:translate = (1, 0.5, 1)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    // Records as the conversion lays them out, twenty floats a gaussian, on
    // an 8 x 8 grid over a box the test knows: x in [-1.5, 1.5], y in
    // [-0.5, 0.5], on the plane. And the rays the gather would have written
    // for them: the point, the plane's normal, no facing, no offset yet.
    constexpr uint32_t kSide = 8;
    constexpr uint32_t kCount = kSide * kSide;
    constexpr uint32_t kPerRecord = 20;
    std::vector<float> records(size_t{kCount} * kPerRecord, 0.0F);
    std::vector<float> rays(size_t{kCount} * 12, 0.0F);
    for (uint32_t j = 0; j < kSide; ++j) {
        for (uint32_t i = 0; i < kSide; ++i) {
            const uint32_t k = j * kSide + i;
            const float x = -1.5F + 3.0F * static_cast<float>(i) / static_cast<float>(kSide - 1);
            const float y = -0.5F + 1.0F * static_cast<float>(j) / static_cast<float>(kSide - 1);
            float* record = records.data() + size_t{k} * kPerRecord;
            record[0] = x;
            record[1] = y;
            record[2] = -1.5F;
            record[3] = 1.0F;
            record[19] = 1.0F;
            float* ray = rays.data() + size_t{k} * 12;
            ray[0] = x;
            ray[1] = y;
            ray[2] = -1.5F;
            ray[6] = 1.0F;
        }
    }
    gpu::Device& device = *gpu->device;
    auto recordBuffer = gpu::Buffer::fromSpan<float>(device, records, "test.records");
    REQUIRE(recordBuffer);
    gpu::BufferDesc desc;
    desc.bytes = rays.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "test.rays";
    auto rayBuffer = gpu::Buffer::create(device, desc, rays.data());
    REQUIRE(rayBuffer);

    // The conversion's own kernels, as `Converter::spanRays` runs them.
    auto chunks = gpu::ComputeKernel::create(*gpu->library, "athenea/usd/mesh2splat_span", "m2sRecordChunks");
    auto reduce = gpu::ComputeKernel::create(*gpu->library, "athenea/scene/bounds_reduce", "boundsReduce");
    auto span = gpu::ComputeKernel::create(*gpu->library, "athenea/usd/mesh2splat_span", "m2sRaySpan");
    auto spanCheck = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh2splat_host_check", "m2sSpanCheck");
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh2splat_host_check", "m2sAnswerCompare");
    REQUIRE(chunks);
    REQUIRE(reduce);
    REQUIRE(span);
    REQUIRE(spanCheck);
    REQUIRE(compare);
    constexpr uint32_t kChunk = 16;
    constexpr uint32_t kChunks = kCount / kChunk;
    desc.bytes = uint64_t{kChunks} * 2 * 16;
    desc.label = "test.extents";
    auto extents = gpu::Buffer::create(device, desc);
    desc.bytes = 2 * 16;
    desc.label = "test.box";
    auto box = gpu::Buffer::create(device, desc);
    REQUIRE(extents);
    REQUIRE(box);
    {
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["records"].setBinding(recordBuffer->rhi());
            cursor["extents"].setBinding(extents->rhi());
            cursor["box"].setBinding(box->rhi());
            cursor["rays"].setBinding(rayBuffer->rhi());
            cursor["span"]["count"].setData(kCount);
            cursor["span"]["perRecord"].setData(kPerRecord);
            cursor["span"]["chunkSize"].setData(kChunk);
            cursor["span"]["chunkCount"].setData(kChunks);
        };
        gpu::CommandBatch batch(device);
        chunks->dispatch(batch, {kChunks, 1, 1}, bind);
        reduce->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["extents"].setBinding(extents->rhi());
            cursor["result"].setBinding(box->rhi());
            cursor["params"]["count"].setData(kCount);
            cursor["params"]["chunkSize"].setData(kChunk);
            cursor["params"]["chunkCount"].setData(kChunks);
        });
        span->dispatch(batch, {kCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    {
        gpu::Buffer counts = test::uintBuffer(device, 4, "counts");
        gpu::CommandBatch batch(device);
        spanCheck->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(rayBuffer->rhi());
            cursor["b"].setBinding(rayBuffer->rhi());
            cursor["counts"].setBinding(counts.rhi());
            const float low[4] = {-1.5F, -0.5F, -1.5F, 0.0F};
            const float high[4] = {1.5F, 0.5F, -1.5F, 0.0F};
            cursor["params"]["low"].setData(low, sizeof(low));
            cursor["params"]["high"].setData(high, sizeof(high));
            cursor["params"]["count"].setData(kCount);
            cursor["params"]["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts.read(device, 0, sizeof(seen), seen));
        CHECK(seen[0] == kCount);
        CHECK(seen[1] == 0);   // every ray a ten-thousandth of the diagonal off
    }

    // The same rays, traced from the host and from the device.
    auto renderer = usd::StageRenderer::open(path, gpu->device);
    if (!renderer) FAIL(renderer.error().toString());
    std::vector<float> laid(size_t{kCount} * 12);
    REQUIRE(rayBuffer->read(device, 0, laid.size() * sizeof(float), laid.data()));
    std::vector<float> hostRays(size_t{kCount} * 8);
    for (uint32_t k = 0; k < kCount; ++k) {   // the host form is the first two float4, copied
        std::copy_n(laid.data() + size_t{k} * 12, 8, hostRays.data() + size_t{k} * 8);
    }
    constexpr uint32_t kDegree = 2;
    constexpr uint32_t kEntries = (kDegree + 1) * (kDegree + 1);
    auto fromHost = (*renderer)->bakePoints(hostRays, kCount, 0.0, 64, 1, kDegree);
    if (!fromHost) FAIL(fromHost.error().toString());
    auto fromDevice = (*renderer)->bakePointsOnDevice(*rayBuffer, kCount, 0.0, 64, 1, kDegree);
    if (!fromDevice) FAIL(fromDevice.error().toString());
    REQUIRE(fromHost->size() == size_t{kCount} * kEntries * 4);
    REQUIRE(fromDevice->bytes() >= uint64_t{kCount} * kEntries * 16);
    auto hostAnswer = gpu::Buffer::create(device, [&] {
        gpu::BufferDesc d;
        d.bytes = fromHost->size() * sizeof(float);
        d.elementBytes = 16;
        d.label = "test.hostAnswer";
        return d;
    }(), fromHost->data());
    REQUIRE(hostAnswer);
    gpu::Buffer counts = test::uintBuffer(device, 4, "counts");
    {
        gpu::CommandBatch batch(device);
        compare->dispatch(batch, {kCount * kEntries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(hostAnswer->rhi());
            cursor["b"].setBinding(fromDevice->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["count"].setData(kCount * kEntries);
            cursor["params"]["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t seen[3] = {0, 0, 0};
    REQUIRE(counts.read(device, 0, sizeof(seen), seen));
    float worst = 0.0F;
    std::memcpy(&worst, &seen[2], sizeof(worst));
    std::printf("  device bake against host bake: %u entries, %u apart, worst %.3g\n", seen[0], seen[1],
                static_cast<double>(worst));
    CHECK(seen[0] == kCount * kEntries);
    CHECK(seen[1] == 0);
}

// A BAKE IN PASSES ANSWERS AS A BAKE IN ONE.
//
// A cloud of ten million gaussians in one pass had the tracer hold a plane an
// entry for every one of them at once -- 2.6 GB at degree 3 -- besides its
// own sums. `bakePointsOnDevice` takes it in passes of at most `batch`
// points, each laid into the answer at its place. On a Lambertian plane under
// a dome, where every point's answer is the same: a bake in passes of 7
// points (so the last is short) against one pass, entry for entry, within
// what the paths' noise moves an answer by.
TEST_CASE("a bake taken in passes answers as one taken whole", "[usd][gpu][mesh][bake][mesh2splat]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_batches.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    constexpr uint32_t kCount = 40;
    std::vector<float> rays(size_t{kCount} * 12, 0.0F);
    for (uint32_t k = 0; k < kCount; ++k) {
        float* ray = rays.data() + size_t{k} * 12;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(kCount);
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    gpu::Device& device = *gpu->device;
    gpu::BufferDesc desc;
    desc.bytes = rays.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "test.rays";
    auto rayBuffer = gpu::Buffer::create(device, desc, rays.data());
    REQUIRE(rayBuffer);
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh2splat_host_check", "m2sAnswerCompare");
    REQUIRE(compare);
    auto renderer = usd::StageRenderer::open(path, gpu->device);
    if (!renderer) FAIL(renderer.error().toString());
    constexpr uint32_t kDegree = 2;
    constexpr uint32_t kEntries = (kDegree + 1) * (kDegree + 1);
    auto whole = (*renderer)->bakePointsOnDevice(*rayBuffer, kCount, 0.0, 1024, 1, kDegree, false, 0);
    if (!whole) FAIL(whole.error().toString());
    auto passes = (*renderer)->bakePointsOnDevice(*rayBuffer, kCount, 0.0, 1024, 1, kDegree, false, 7);
    if (!passes) FAIL(passes.error().toString());
    gpu::Buffer counts = test::uintBuffer(device, 4, "counts");
    {
        gpu::CommandBatch batch(device);
        compare->dispatch(batch, {kCount * kEntries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(whole->rhi());
            cursor["b"].setBinding(passes->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["count"].setData(kCount * kEntries);
            // The constant term is about 0.46 here and the harmonics about
            // zero; 1024 stratified paths hold both within 0.0053 (M5 Pro).
            cursor["params"]["tolerance"].setData(0.01F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t seen[3] = {0, 0, 0};
    REQUIRE(counts.read(device, 0, sizeof(seen), seen));
    float worst = 0.0F;
    std::memcpy(&worst, &seen[2], sizeof(worst));
    std::printf("  six passes of 7 and one of 40: %u entries, %u apart, worst %.3g\n", seen[0], seen[1],
                static_cast<double>(worst));
    CHECK(seen[0] == kCount * kEntries);
    CHECK(seen[1] == 0);
}

// A BAKE SPLIT IN TWO SUMS TO THE BAKE WHOLE.
//
// `bakeSplitOnDevice` keeps the direct light's sums apart from the indirect's
// and fits each on its own; `combineBake` adds them and bounds the sum as the
// tracer's own bake does. The fit is linear and the paths are the same paths
// (the same seed), so the two must agree to rounding. A floor beside a wall
// under a dome, so the indirect half is not nothing: the floor sees the wall
// lit, and the check also counts the points whose indirect light is there.
TEST_CASE("a bake split into direct and indirect sums to the bake whole", "[usd][gpu][mesh][bake][split]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_split.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, 0), (4, -4, 0), (4, 4, 0), (-4, 4, 0)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, 0.5, 0), (4, 0.5, 0), (4, 0.5, 4), (-4, 0.5, 4)]\n"
               "    normal3f[] normals = [(0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    constexpr uint32_t kCount = 32;
    std::vector<float> rays(size_t{kCount} * 12, 0.0F);
    for (uint32_t k = 0; k < kCount; ++k) {
        float* ray = rays.data() + size_t{k} * 12;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(kCount);
        ray[1] = 0.25F;
        ray[2] = 0.0F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    gpu::Device& device = *gpu->device;
    gpu::BufferDesc desc;
    desc.bytes = rays.size() * sizeof(float);
    desc.elementBytes = 16;
    desc.label = "test.rays";
    auto rayBuffer = gpu::Buffer::create(device, desc, rays.data());
    REQUIRE(rayBuffer);
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/mesh2splat_host_check", "m2sAnswerCompare");
    REQUIRE(compare);
    auto seen = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_split_check", "splitIndirectSeen");
    REQUIRE(seen);
    auto renderer = usd::StageRenderer::open(path, gpu->device);
    if (!renderer) FAIL(renderer.error().toString());
    constexpr uint32_t kDegree = 2;
    constexpr uint32_t kEntries = (kDegree + 1) * (kDegree + 1);
    auto whole = (*renderer)->bakePointsOnDevice(*rayBuffer, kCount, 0.0, 256, 2, kDegree);
    if (!whole) FAIL(whole.error().toString());
    usd::BakeOptions options;
    options.samples = 256;
    options.bounces = 2;
    options.degree = kDegree;
    auto split = (*renderer)->bakeSplitOnDevice(*rayBuffer, kCount, 0.0, options);
    if (!split) FAIL(split.error().toString());
    auto combined = (*renderer)->combineBake(*split, *rayBuffer);
    if (!combined) FAIL(combined.error().toString());
    gpu::Buffer counts = test::uintBuffer(device, 4, "counts");
    gpu::Buffer lit = test::uintBuffer(device, 4, "lit");
    {
        gpu::CommandBatch batch(device);
        compare->dispatch(batch, {kCount * kEntries, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(whole->rhi());
            cursor["b"].setBinding(combined->rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["count"].setData(kCount * kEntries);
            cursor["params"]["tolerance"].setData(1.0e-4F);
        });
        seen->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["guides"].setBinding(split->guides.rhi());
            cursor["sums"].setBinding(split->sums.rhi());
            cursor["allot"].setBinding(split->allot.rhi());
            cursor["counts"].setBinding(lit.rhi());
            cursor["check"]["count"].setData(kCount);
            cursor["check"]["threshold"].setData(1.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t got[3] = {0, 0, 0};
    REQUIRE(counts.read(device, 0, sizeof(got), got));
    uint32_t indirect = 0;
    REQUIRE(lit.read(device, 0, sizeof(indirect), &indirect));
    float worst = 0.0F;
    std::memcpy(&worst, &got[2], sizeof(worst));
    std::printf("  split against whole: %u entries, %u apart, worst %.3g; %u of %u points with indirect light\n",
                got[0], got[1], static_cast<double>(worst), indirect, kCount);
    CHECK(got[0] == kCount * kEntries);
    CHECK(got[1] == 0);
    CHECK(indirect == kCount);
}

// THE EXTRA PATHS GO WHERE THE NOISE IS.
//
// `allotBakePasses` shares a budget out by sqrt(relative variance / cost). On
// made-up sums whose answer is known -- every gaussian's mean the same, its
// cost the same, the first half's variance a hundred times smaller than the
// second's -- the second half must get ten times the passes (sqrt of a
// hundred), and the whole the budget asked for.
TEST_CASE("an adaptive bake allots its extra paths where the variance is", "[usd][gpu][bake][adaptive]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    constexpr uint32_t kCount = 4096;
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{kCount} * 5 * 16;
    desc.elementBytes = 16;
    desc.label = "test.sums";
    auto sums = gpu::Buffer::create(device, desc);
    REQUIRE(sums);
    gpu::Buffer allot = test::uintBuffer(device, kCount, "allot");
    gpu::Buffer halves = test::uintBuffer(device, 4, "halves");
    auto synth = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_split_check", "allotSynth");
    REQUIRE(synth);
    auto count = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_split_check", "allotHalves");
    REQUIRE(count);
    const auto bind = [&](rhi::ShaderCursor cursor) {
        cursor["guides"].setBinding(sums->rhi());
        cursor["sums"].setBinding(sums->rhi());
        cursor["allot"].setBinding(allot.rhi());
        cursor["counts"].setBinding(halves.rhi());
        cursor["check"]["count"].setData(kCount);
        cursor["check"]["low"].setData(0.001F);
        cursor["check"]["high"].setData(0.1F);
    };
    {
        gpu::CommandBatch batch(device);
        synth->dispatch(batch, {kCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    usd::BakeOptions options;
    options.extraSamples = 128;   // two passes of 64 a gaussian on average
    options.passSamples = 64;
    options.maxPasses = 64;
    REQUIRE(usd::allotBakePasses(*gpu->library, *sums, kCount, 1, options, allot));
    {
        gpu::CommandBatch batch(device);
        count->dispatch(batch, {kCount, 1, 1}, bind);
        REQUIRE(batch.submit(true));
    }
    uint32_t got[2] = {0, 0};
    REQUIRE(halves.read(device, 0, sizeof(got), got));
    const double quiet = static_cast<double>(got[0]);
    const double noisy = static_cast<double>(got[1]);
    std::printf("  passes allotted: %.0f to the quiet half, %.0f to the noisy one (budget %u)\n", quiet, noisy,
                kCount * 2);
    CHECK(noisy > 8.0 * quiet);
    CHECK(noisy < 12.0 * quiet);
    CHECK(std::abs(quiet + noisy - kCount * 2.0) < kCount * 2.0 * 0.05);
}

// A BAKE GIVES NO MORE LIGHT THAN ANY PATH SAW, FROM ANY SIDE.
//
// The surface that broke it: polished metal under one small, bright light,
// so its mean is dark and its light swings with the direction. The bands
// were taken into the cloud's space to first order, scaled by the sRGB
// curve's slope at the mean -- 12.92 at black -- and the series, fitted on
// the half of the sphere the surface faces, was read from every side. The
// pawn's glass head baked at degree 2 decoded to 1094 under a dome of 1, and
// at degree 3 to infinity. Here the series is evaluated over the whole
// sphere and may not decode to more than the light sends.
TEST_CASE("a glossy bake stays within the light the scene sends, from every side", "[usd][gpu][mesh][bake]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_glossy.usda");
    // The light: a sphere whose radiance is `kRadiance` (UsdLux: intensity is
    // the light's radiance while `normalize` is off).
    constexpr float kRadiance = 50.0F;
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Metal>\n}\n"
               "def Material \"Metal\"\n{\n"
               "    token outputs:surface.connect = </Metal/Preview.outputs:surface>\n"
               "    def Shader \"Preview\"\n    {\n"
               "        uniform token info:id = \"UsdPreviewSurface\"\n"
               "        color3f inputs:diffuseColor = (0.9, 0.9, 0.9)\n"
               "        float inputs:metallic = 1\n"
               "        float inputs:roughness = 0.08\n"
               "        token outputs:surface\n    }\n}\n"
               "def SphereLight \"Key\"\n{\n"
               "    float inputs:intensity = " << kRadiance << "\n"
               "    float inputs:radius = 0.15\n"
               "    double3 xformOp:translate = (0.8, 0.5, 0.5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    auto bound = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_check", "bakeBound");
    if (!bound) FAIL(bound.error().toString());
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    for (const uint32_t degree : {1u, 2u, 3u}) {
        auto baked = (*renderer)->bakePoints(rays, count, 0.0, 256, 1, degree);
        if (!baked) FAIL(baked.error().toString());
        const uint32_t planes = (degree + 1) * (degree + 1);
        REQUIRE(baked->size() == size_t{count} * planes * 4);
        gpu::BufferDesc desc;
        desc.bytes = baked->size() * 4;
        desc.elementBytes = 16;
        auto fitted = gpu::Buffer::create(*gpu->device, desc, baked->data());
        REQUIRE(fitted);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 3, "counts");
        gpu::Buffer largest = test::uintBuffer(*gpu->device, 1, "largest");
        {
            gpu::CommandBatch batch(*gpu->device);
            bound->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["coefficients"].setBinding(fitted->rhi());
                cursor["boundCounts"].setBinding(counts.rhi());
                cursor["brightest"].setBinding(largest.rhi());
                rhi::ShaderCursor p = cursor["boundParams"];
                p["count"].setData(count);
                p["planes"].setData(planes);
                // Gibbs: a projection of a bounded function overshoots it by
                // a fraction, never by multiples.
                p["bound"].setData(kRadiance * 1.25F);
            });
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[3] = {0, 0, 0};
        float    most = 0.0F;
        REQUIRE(counts.read(*gpu->device, 0, sizeof(seen), seen));
        REQUIRE(largest.read(*gpu->device, 0, sizeof(most), &most));
        std::printf("  degree %u: %u points, %u beyond the light, %u not a number, largest %.4g\n", degree, seen[0],
                    seen[1], seen[2], double(most));
        CHECK(seen[0] == count);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }
}

// A CONVERTED GLASS CARRIES WHAT IT BENDS BY, AND SAYS WHICH SCHEMA SAYS SO.
//
// A transmitting gaussian refracts only with the cloud's index (`ior > 1`),
// and mesh2splat knew the material's and never wrote it: the chess pawn's
// glass head drew as a milky ball whether relit, transferred or baked. And
// the lighting primvars were written without AtheneaSplatLightingAPI
// applied, so `relight` came out as a custom attribute nobody declared.
TEST_CASE("a converted cloud keeps its glass index and applies the lighting schema", "[usd][gpu][export]") {
    ATHENEA_REQUIRE_GPU(gpu);
    io::RawSplats raw = cloud(64);
    const fs::path path = scratch("export_ior.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    options.ior = 1.5F;
    options.metersPerUnit = 1.0;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));

    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(prim);
    CHECK(prim.HasAPI(TfToken("AtheneaSplatLightingAPI")));
    float ior = 0.0F;
    REQUIRE(prim.GetAttribute(TfToken("primvars:athenea:splat:ior")).Get(&ior));
    CHECK(ior == 1.5F);
    CHECK_FALSE(prim.GetAttribute(TfToken("primvars:athenea:splat:relight")).IsCustom());
    CHECK(UsdGeomGetStageMetersPerUnit(stage) == 1.0);

    // A cloud with nothing to bend writes no index: 0 is the schema's own.
    const fs::path plain = scratch("export_no_ior.usda");
    usd::ExportOptions none;
    none.addCamera = false;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, plain, none));
    UsdStageRefPtr opened = UsdStage::Open(plain.string());
    REQUIRE(opened);
    CHECK_FALSE(opened->GetPrimAtPath(SdfPath("/World/Splats")).GetAttribute(TfToken("primvars:athenea:splat:ior")).HasAuthoredValue());
}

// A SLOT WITH NOTHING IN IT IS STILL A SLOT.
//
// The writer used to skip a record that decoded to nothing -- a scale that
// overflowed, an opacity below a 255th -- so the file came out shorter than
// the conversion that made it. That is harmless for one still frame and
// impossible for a sequence: a gaussian is followed from one pose to the next
// by being the same element of the array, and a triangle that goes degenerate
// in one pose alone would put every later gaussian out of step in that frame
// and in no other. The slot is kept now, written with no opacity and no size.
TEST_CASE("a splat that decodes to nothing keeps its place in the written cloud", "[usd][gpu][export]") {
    ATHENEA_REQUIRE_GPU(gpu);
    io::RawSplats raw = cloud(512);
    // Three of them asked to be dropped: one with no opacity, one whose scale
    // overflows, one whose position is not a number.
    raw.records[size_t{7} * 23 + 3] = -40.0F;
    raw.records[size_t{100} * 23 + 4] = 200.0F;
    raw.records[size_t{300} * 23 + 0] = std::numeric_limits<float>::quiet_NaN();

    const fs::path path = scratch("export-keeps-slots.usda");
    fs::remove(path);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, {.addCamera = false}));

    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdVolParticleField3DGaussianSplat splats(stage->GetPrimAtPath(SdfPath("/World/Splats")));
    REQUIRE(splats);
    VtVec3fArray positions;
    VtFloatArray opacities;
    VtVec3fArray scales;
    REQUIRE(splats.GetPositionsAttr().Get(&positions));
    REQUIRE(splats.GetOpacitiesAttr().Get(&opacities));
    REQUIRE(splats.GetScalesAttr().Get(&scales));

    // Counts and slots, which is what the processor is for.
    CHECK(positions.size() == raw.count);
    CHECK(opacities.size() == raw.count);
    CHECK(scales.size() == raw.count);
    for (const uint32_t at : {7u, 100u, 300u}) {
        CHECK(opacities[at] == 0.0F);
        CHECK(scales[at] == GfVec3f(0.0F, 0.0F, 0.0F));
        CHECK(std::isfinite(positions[at][0]));
        CHECK(std::isfinite(positions[at][1]));
        CHECK(std::isfinite(positions[at][2]));
    }
    // And the extent is the cloud's, not stretched to where an empty slot
    // happens to stand.
    VtVec3fArray extent;
    REQUIRE(splats.GetExtentAttr().Get(&extent));
    REQUIRE(extent.size() == 2);
    CHECK(std::isfinite(extent[0][0]));
    CHECK(extent[1][0] >= extent[0][0]);
}

// A CONVERSION READS THE POSE, NOT THE REST.
//
// `MeshStage` goes round Hydra on purpose -- a conversion wants none of a
// render index -- and a skinned mesh's `points` attribute does not animate,
// because the deformation is the skeleton's. So a stage with a SkelRoot used
// to convert as the rest pose whatever time was asked for, and `--time`
// reached only the bake's ray tracing: the rays stood where the mesh used to
// be while the scene they traced was somewhere else.
//
// It is posed with `UsdSkelBakeSkinning`, into the session layer and for one
// instant, which writes the posed points onto the meshes themselves. The
// reads then need to know nothing about skinning. Here one joint carries the
// whole square and slides by (1.2, 0.4, 0) between time 0 and time 1, so the
// mesh the conversion builds must move by exactly that -- and its bounds are
// the device's own fold over the points, not the host's.
TEST_CASE("a skinned stage is read in the pose it holds at the time asked for", "[usd][gpu][mesh][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("mesh-stage-skinned.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"root\", \"root/arm\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) ), "
               "( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"root/arm\"]\n"
               "        float3[] translations.timeSamples = {\n            0: [(0, 0, 0)],\n"
               "            1: [(1.2, 0.4, 0)],\n        }\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n    }\n"
               "    def Mesh \"Square\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        int[] faceVertexCounts = [4]\n"
               "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-0.5, -0.5, -5), (0.5, -0.5, -5), (0.5, 0.5, -5), (-0.5, 0.5, -5)]\n"
               "        uniform token subdivisionScheme = \"none\"\n"
               "        rel skel:skeleton = </Root/Skel>\n"
               "        int[] primvars:skel:jointIndices = [1, 1, 1, 1] ( elementSize = 1\n"
               "            interpolation = \"vertex\" )\n"
               "        float[] primvars:skel:jointWeights = [1, 1, 1, 1] ( elementSize = 1\n"
               "            interpolation = \"vertex\" )\n"
               "        matrix4d primvars:skel:geomBindTransform = ( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )\n"
               "    }\n}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());

    const auto boundsAt = [&](double time) {
        auto stage = usd::MeshStage::open(path);
        if (!stage) FAIL(stage.error().toString());
        usd::MeshStageOptions options;
        options.time = time;
        auto meshes = stage->read(*builder, options);
        if (!meshes) FAIL(meshes.error().toString());
        REQUIRE(meshes->size() == 1);
        return (*meshes)[0].mesh.bounds;
    };
    const scene::Bounds rest = boundsAt(0.0);
    const scene::Bounds posed = boundsAt(1.0);
    std::printf("  rest %.3f %.3f, posed %.3f %.3f\n", double(rest.min[0]), double(rest.min[1]),
                double(posed.min[0]), double(posed.min[1]));

    // The square, as authored, at time 0.
    CHECK(rest.min[0] == Catch::Approx(-0.5).margin(1e-4));
    CHECK(rest.min[1] == Catch::Approx(-0.5).margin(1e-4));
    // And carried by its one joint at time 1.
    CHECK(posed.min[0] == Catch::Approx(0.7).margin(1e-3));
    CHECK(posed.min[1] == Catch::Approx(-0.1).margin(1e-3));
    CHECK(posed.max[0] == Catch::Approx(1.7).margin(1e-3));
    CHECK(posed.max[1] == Catch::Approx(0.9).margin(1e-3));
    // The plane it stands in does not move: the joint slides in x and y.
    CHECK(posed.min[2] == Catch::Approx(rest.min[2]).margin(1e-4));

    // AND ASKED FOR ITS JOINTS INSTEAD, it comes back in the bind pose with
    // the influences on it -- because a cloud that keeps its joints is built
    // where the skeleton's transforms expect to find it, and posing it first
    // would skin it twice.
    {
        auto stage = usd::MeshStage::open(path);
        if (!stage) FAIL(stage.error().toString());
        usd::MeshStageOptions options;
        options.time = 1.0;        // the pose it would otherwise be read in
        options.skinned = true;
        auto meshes = stage->read(*builder, options);
        if (!meshes) FAIL(meshes.error().toString());
        REQUIRE(meshes->size() == 1);
        const usd::StageMesh& mesh = (*meshes)[0];
        CHECK(mesh.mesh.bounds.min[0] == Catch::Approx(-0.5).margin(1e-4));
        CHECK(mesh.mesh.bounds.max[0] == Catch::Approx(0.5).margin(1e-4));

        const usd::StageSkinning& skin = mesh.skinning;
        CHECK(skin.bound);
        CHECK(skin.skeleton == "/Root/Skel");
        CHECK(skin.joints.size() == 2);
        CHECK(skin.perPoint == 1);
        REQUIRE(skin.influences.size() == 4 * 2);
        for (size_t point = 0; point < 4; ++point) {
            CHECK(skin.influences[point * 2] == 1.0F);       // the arm, the skeleton's joint 1
            CHECK(skin.influences[point * 2 + 1] == 1.0F);   // and all of it
        }
    }
}

// A MATERIALX GLASS KEEPS ITS CUT-OUT. UsdPreviewSurface has no transmission,
// so a map on its opacity was the only cut-out a conversion read; a feather
// made glass in MaterialX -- OpenPBR's `geometry_opacity`, standard_surface's
// `opacity` -- lost its shape and became a pane. Both are coverage there too,
// and the map reads its first channel by the UV set its `texcoord` names.
TEST_CASE("a MaterialX glass's opacity map is a cut-out, by its own UV set", "[usd][mesh][materials]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("mx_cut.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        const char* surfaces[2][3] = {{"Open", "ND_open_pbr_surface_surfaceshader", "geometry_opacity"},
                                      {"Standard", "ND_standard_surface_surfaceshader", "opacity"}};
        for (const auto& surface : surfaces) {
            out << "def Mesh \"" << surface[0] << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                   "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    texCoord2f[] primvars:st2 = [(0, 0), (1, 0), (0, 1)] ( interpolation = \"vertex\" )\n"
                << "    rel material:binding = </Looks/" << surface[0] << ">\n}\n";
        }
        out << "def Scope \"Looks\"\n{\n";
        for (const auto& surface : surfaces) {
            const bool open = std::string(surface[0]) == "Open";
            const std::string at = std::string("/Looks/") + surface[0];
            out << "    def Material \"" << surface[0] << "\"\n    {\n"
                << "        token outputs:mtlx:surface.connect = <" << at << "/Surface.outputs:out>\n"
                << "        def Shader \"Surface\"\n        {\n"
                << "            uniform token info:id = \"" << surface[1] << "\"\n"
                << "            float inputs:" << (open ? "transmission_weight" : "transmission") << " = 1\n"
                << "            " << (open ? "float" : "color3f") << " inputs:" << surface[2] << ".connect = <" << at
                << "/Mask.outputs:out>\n"
                << "            token outputs:out\n        }\n"
                << "        def Shader \"Mask\"\n        {\n"
                << "            uniform token info:id = \"" << (open ? "ND_image_float" : "ND_image_color3") << "\"\n"
                << "            asset inputs:file = @./mask.png@\n"
                << "            float2 inputs:texcoord.connect = <" << at << "/Uv.outputs:out>\n"
                << "            " << (open ? "float" : "color3f") << " outputs:out\n        }\n"
                << "        def Shader \"Uv\"\n        {\n"
                << "            uniform token info:id = \"ND_geompropvalue_vector2\"\n"
                << "            string inputs:geomprop = \"st2\"\n"
                << "            float2 outputs:out\n        }\n    }\n";
        }
        out << "}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == 2);
    for (const usd::StageMesh& mesh : *meshes) {
        INFO(mesh.path);
        const usd::StageMaterial& m = mesh.material;
        CHECK(m.transmission == 1.0F);                   // glass
        CHECK(m.opacityMap.file.find("mask.png") != std::string::npos);   // and cut
        CHECK(m.opacityMap.channel == 'r');              // by the channel an image node gives
        CHECK(m.opacityMap.uvSet == "st2");              // off the UV set its texcoord names
    }
}

// EACH SURFACE IS READ IN ITS OWN WORDS AND AT ITS OWN DEFAULTS. glTF's
// `gltf_pbr` calls its inputs `metallic`, `roughness` and `ior`, which read
// as standard_surface's were simply not there; and an input nobody authored
// is worth what that surface says, which is what the mesh renders with --
// standard_surface a 0.8 grey of roughness 0.2, OpenPBR 0.3, glTF a fully
// rough metal, UsdPreviewSurface 0.18 at 0.5. And a packed map's missing
// channel is a factor of one: the kernel multiplies it into the value.
TEST_CASE("a surface's inputs are read in its own vocabulary and at its own defaults", "[usd][mesh][materials]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("vocabulary_read.usda");
    const char* names[] = {"Standard", "Open", "Gltf", "Preview", "GltfAuthored", "Weighted", "Clear"};
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        for (const char* name : names) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                   "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                << "    rel material:binding = </Looks/" << name << ">\n}\n";
        }
        const auto mx = [&out](const char* name, const char* id, const char* inputs) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:mtlx:surface.connect = </Looks/" << name << "/S.outputs:out>\n"
                << "        def Shader \"S\"\n        {\n"
                << "            uniform token info:id = \"" << id << "\"\n" << inputs
                << "            token outputs:out\n        }\n    }\n";
        };
        out << "def Scope \"Looks\"\n{\n";
        mx("Standard", "ND_standard_surface_surfaceshader", "");
        mx("Open", "ND_open_pbr_surface_surfaceshader", "");
        mx("Gltf", "ND_gltf_pbr_surfaceshader", "");
        mx("GltfAuthored", "ND_gltf_pbr_surfaceshader",
           "            float inputs:metallic = 0\n            float inputs:roughness = 0.25\n"
           "            float inputs:ior = 1.7\n            float inputs:transmission = 1\n");
        mx("Weighted", "ND_standard_surface_surfaceshader",
           "            float inputs:base = 0.5\n            color3f inputs:base_color = (1, 0.5, 0)\n");
        out << "    def Material \"Preview\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Preview/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            token outputs:surface\n        }\n    }\n"
               "    def Material \"Clear\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Clear/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            float inputs:opacity = 0.4\n"
               "            token outputs:surface\n        }\n    }\n}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == std::size(names));
    for (const usd::StageMesh& mesh : *meshes) {
        INFO(mesh.path);
        const usd::StageMaterial& m = mesh.material;
        if (mesh.path == "/Standard") {
            CHECK(m.baseColour[0] == Catch::Approx(0.8F));
            CHECK(m.metallic == 0.0F);
            CHECK(m.roughness == Catch::Approx(0.2F));
        } else if (mesh.path == "/Open") {
            CHECK(m.baseColour[1] == Catch::Approx(0.8F));
            CHECK(m.roughness == Catch::Approx(0.3F));
        } else if (mesh.path == "/Gltf") {
            CHECK(m.baseColour[2] == 1.0F);
            CHECK(m.metallic == 1.0F);
            CHECK(m.roughness == 1.0F);
        } else if (mesh.path == "/Preview") {
            CHECK(m.baseColour[0] == Catch::Approx(0.18F));
            CHECK(m.roughness == Catch::Approx(0.5F));
        } else if (mesh.path == "/GltfAuthored") {
            CHECK(m.metallic == 0.0F);
            CHECK(m.roughness == Catch::Approx(0.25F));
            CHECK(m.ior == Catch::Approx(1.7F));
            CHECK(m.transmission == 1.0F);
        } else if (mesh.path == "/Clear") {
            // A preview surface's opacity is a coverage: seen through, not
            // bent, which is a thin wall's transmission.
            CHECK(m.transmission == Catch::Approx(0.6F));
            CHECK(m.thinWalled);
        } else {
            // base 0.5 over (1, 0.5, 0): the weight scales the colour.
            CHECK(m.baseColour[0] == Catch::Approx(0.5F));
            CHECK(m.baseColour[1] == Catch::Approx(0.25F));
            CHECK(m.baseColour[2] == 0.0F);
        }
    }
}

// A HEIGHT IS READ WHICHEVER WAY IT IS WRITTEN. UsdPreviewSurface puts it on
// the surface's `displacement`, read through UsdUVTexture's own scale and
// bias on the channel connected; MaterialX puts it on a `displacement` node
// the material's displacement terminal names, whose `scale` multiplies it;
// and a constant, with no map, moves the whole surface. A mesh scaled by two
// in the world displaces twice as far, since a height is authored in the
// mesh's own units.
TEST_CASE("a material's displacement is read as a height, from either vocabulary", "[usd][mesh][materials][displacement]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("displace_read.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        for (const char* name : {"Preview", "Mx", "Constant"}) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                   "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] ( interpolation = \"vertex\" )\n"
                << (std::string(name) == "Preview"
                        ? "    float3 xformOp:scale = (2, 2, 2)\n    uniform token[] xformOpOrder = [\"xformOp:scale\"]\n"
                        : "")
                << "    rel material:binding = </Looks/" << name << ">\n}\n";
        }
        out << "def Scope \"Looks\"\n{\n"
               "    def Material \"Preview\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Preview/S.outputs:surface>\n"
               "        token outputs:displacement.connect = </Looks/Preview/S.outputs:displacement>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            float inputs:displacement.connect = </Looks/Preview/H.outputs:g>\n"
               "            token outputs:surface\n            token outputs:displacement\n        }\n"
               "        def Shader \"H\"\n        {\n"
               "            uniform token info:id = \"UsdUVTexture\"\n"
               "            asset inputs:file = @./height.png@\n"
               "            float4 inputs:scale = (1, 0.06, 1, 1)\n"
               "            float4 inputs:bias = (0, -0.03, 0, 0)\n"
               "            float outputs:g\n        }\n    }\n"
               "    def Material \"Mx\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Looks/Mx/S.outputs:out>\n"
               "        token outputs:mtlx:displacement.connect = </Looks/Mx/D.outputs:out>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"D\"\n        {\n"
               "            uniform token info:id = \"ND_displacement_float\"\n"
               "            float inputs:displacement.connect = </Looks/Mx/H.outputs:out>\n"
               "            float inputs:scale = 0.2\n"
               "            token outputs:out\n        }\n"
               "        def Shader \"H\"\n        {\n"
               "            uniform token info:id = \"ND_image_float\"\n"
               "            asset inputs:file = @./height.png@\n"
               "            float outputs:out\n        }\n    }\n"
               "    def Material \"Constant\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Constant/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            float inputs:displacement = 0.1\n"
               "            token outputs:surface\n        }\n    }\n"
               "}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == 3);
    for (const usd::StageMesh& mesh : *meshes) {
        INFO(mesh.path);
        const usd::StageMaterial& m = mesh.material;
        CHECK(m.displaces());
        if (mesh.path == "/Preview") {
            CHECK(m.displacementMap.file.find("height.png") != std::string::npos);
            CHECK(m.displacementMap.channel == 'g');          // the channel connected
            CHECK(m.displacementScale == Catch::Approx(0.06F));   // and that channel's scale and bias
            CHECK(m.displacementBias == Catch::Approx(-0.03F));
            CHECK(mesh.displacementUnit == Catch::Approx(2.0F));   // in a mesh scaled by two
        } else if (mesh.path == "/Mx") {
            CHECK(m.displacementMap.file.find("height.png") != std::string::npos);
            CHECK(m.displacementMap.channel == 'r');          // an image node's one value
            CHECK(m.displacementScale == Catch::Approx(0.2F));   // the node's scale
            CHECK(m.displacementBias == 0.0F);
            CHECK(mesh.displacementUnit == Catch::Approx(1.0F));
        } else {
            CHECK(m.displacementMap.empty());                 // no map, and the surface moved
            CHECK(m.displacementBias == Catch::Approx(0.1F));
        }
    }
}

// A THIN WALL'S CARD STOPS WHAT THE SHEET REFLECTS, AND NO MORE. A glass sheet
// sends what it does not reflect straight on, so the conversion makes its
// gaussians exactly as transparent as the sheet is and lets the blend be the
// transmission. That only holds if a card of them composites to the sheet's
// opacity: a point is under several gaussians (their footprints sum to
// 2 pi sigma^2 on the conversion's grid) and a faint one is drawn short of its
// tails. Given one each at the sheet's own 0.077, a card came out 0.375.
// Here a card laid out as mesh2splat lays one out -- a gaussian a cell,
// sigma cells wide -- at `scene::thinWallOpacity`, drawn by both routes.
TEST_CASE("a thin wall's card is as opaque as the glass sheet reflects, on both routes",
          "[usd][gpu][splat][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const float ior = 1.5F;
    const double sigma = 1.0;
    const float alpha = scene::thinWallOpacity(ior, sigma);
    const float wanted = scene::thinWallReflectance(ior);
    const fs::path path = scratch("thin_card.usda");
    {
        std::ofstream out(path);
        const int side = 120;
        const double cell = 2.0 / side;
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def ParticleField3DGaussianSplat \"Card\"\n{\n    point3f[] positions = [";
        for (int k = 0; k < side * side; ++k) {
            out << (k ? ", " : "") << "(" << (-1.0 + ((k % side) + 0.5) * cell) << ", "
                << (-1.0 + ((k / side) + 0.5) * cell) << ", 0)";
        }
        out << "]\n    quatf[] orientations = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
        out << "]\n    float3[] scales = [";
        for (int k = 0; k < side * side; ++k) {
            out << (k ? ", " : "") << "(" << sigma * cell << ", " << sigma * cell << ", " << 0.1 * sigma * cell
                << ")";
        }
        out << "]\n    float[] opacities = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << alpha;
        out << "]\n    uniform int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(0, 0, 0)";
        out << "]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 3)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    auto meanKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/patch_mean", "patchMean");
    if (!meanKernel) FAIL(meanKernel.error().toString());
    // Each gaussian a few pixels wide, as the bird's are: at one pixel the
    // rasteriser's antialiasing filter widens a splat and lowers its peak,
    // and a card of them read 0.068.
    const uint32_t w = 512, h = 512;
    const auto coverage = [&](const char* technique) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(frame);
        const std::array<uint32_t, 5> zero{};
        auto sums = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "thin.sums");
        REQUIRE(sums);
        {
            gpu::CommandBatch batch(*gpu->device);
            meanKernel->dispatch(batch, {20, 20, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["frame"].setBinding(frame->rhi());
                cursor["sums"].setBinding(sums->rhi());
                cursor["params"]["width"].setData(w);
                cursor["params"]["x0"].setData(w / 2 - 80);
                cursor["params"]["y0"].setData(h / 2 - 80);
                cursor["params"]["w"].setData(160u);
                cursor["params"]["h"].setData(160u);
                cursor["params"]["scale"].setData(65536.0F);
                cursor["params"]["withAlpha"].setData(1u);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 5> read{};
        REQUIRE(sums->read(*gpu->device, 0, sizeof(read), read.data()));
        return read[4] / (std::max<double>(read[3], 1.0) * 65536.0);
    };
    const double raster = coverage("raster");
    const double traced = coverage("rt");
    std::printf("  a thin wall's card, each gaussian at %.4f: raster %.4f, traced %.4f, against the sheet's %.4f\n",
                double(alpha), raster, traced, double(wanted));
    CHECK(std::abs(raster - wanted) < 0.006);
    CHECK(std::abs(traced - wanted) < 0.006);
}

// OPENPBR'S OPACITY IS COVERAGE. `geometry_opacity` is what `opacity` is on a
// standard_surface: where it reads low the surface is not there. Only the
// latter was read as a cut-out, so a glass sparrow's OpenPBR feather cards
// were drawn whole, their mask a fractional opacity nothing cut by, and its
// bake found nothing under 8% of its gaussians. A card at zero in front of a
// coloured square must draw as the square alone, on both routes.
TEST_CASE("an OpenPBR geometry_opacity of zero cuts the surface away, and its shadow, on both routes",
          "[usd][gpu][mesh][materials][openpbr]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const auto stage = [&](const char* name, bool card, bool shadows) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Back\"\n{\n    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.3, 0.1)] ( interpolation = \"constant\" )\n}\n";
        if (card) {
            out << "def Mesh \"Card\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-0.6, -0.6, 0), (0.6, -0.6, 0), (0.6, 0.6, 0), (-0.6, 0.6, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Looks/Cut>\n}\n"
                   "def Scope \"Looks\"\n{\n    def Material \"Cut\"\n    {\n"
                   "        token outputs:mtlx:surface.connect = </Looks/Cut/Surface.outputs:out>\n"
                   "        def Shader \"Surface\"\n        {\n"
                   "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
                   "            color3f inputs:base_color = (0.1, 0.9, 0.2)\n"
                   "            float inputs:geometry_opacity = 0\n"
                   "            token outputs:out\n        }\n    }\n}\n";
        }
        // With shadows on: a card that is not there casts
        // none either, on both routes: the raster asks a shadow's cut-out in
        // traceShadows, a kernel apart from the shading.
        out << "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 2\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n") << "}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 4)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 72;
    const auto frame = [&](const fs::path& path, const char* technique) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(16);
        (*renderer)->setPathTotal(16);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    for (const char* technique : {"raster", "rt"}) {
        INFO(technique);
        const bool shadows = true;
        const gpu::Buffer a = frame(stage("openpbr_cut.usda", true, shadows), technique);
        const gpu::Buffer b = frame(stage("openpbr_cut_none.usda", false, shadows), technique);
        auto same = render::compareHdr(*gpu->library, a, b, w, h);
        REQUIRE(same);
        std::printf("  %s: a card at geometry_opacity 0 against no card, relMSE %.2e\n", technique, same->relMse);
        CHECK(same->relMse < 1e-3);
    }
}

// A CLOUD THAT CARRIES ITS RIG RATHER THAN ITS FRAMES.
//
// Four joints a gaussian and their weights, the transform out of the cloud's
// space, and the joints' own transforms as time samples -- which is the only
// thing about an animated cloud that changes from frame to frame. Sixty
// joints are four kilobytes a frame, against forty bytes a gaussian a frame
// for the arrays this replaces.
TEST_CASE("a cloud writes the skeleton that carries it, and its joints over time", "[usd][gpu][export][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(256);

    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.joints = 3;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        // Two joints each, and the two weights a whole.
        float* one = rig.influences.data() + size_t{k} * 8;
        one[0] = static_cast<float>(k % 3);        one[1] = 0.75F;
        one[2] = static_cast<float>((k + 1) % 3);  one[3] = 0.25F;
    }
    rig.times = {0.0, 1.0, 2.0};
    rig.xforms.resize(rig.times.size() * rig.joints * 16, 0.0F);
    for (size_t frame = 0; frame < rig.times.size(); ++frame) {
        for (uint32_t joint = 0; joint < rig.joints; ++joint) {
            float* m = rig.xforms.data() + (frame * rig.joints + joint) * 16;
            m[0] = m[5] = m[10] = m[15] = 1.0F;
            // The translation is in the last row, as GfMatrix4f holds it.
            m[12] = static_cast<float>(frame) + static_cast<float>(joint) * 0.1F;
        }
    }
    REQUIRE(rig.valid());

    const fs::path path = scratch("cloud-rigged.usda");
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.skinning = &rig;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));

    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(prim);
    const UsdGeomPrimvarsAPI primvars(prim);

    // The binding is UsdSkel's own: SkelBindingAPI applied, the influences
    // under its names, four a gaussian, and the Skeleton by relationship.
    REQUIRE(prim.HasAPI<UsdSkelBindingAPI>());
    const UsdSkelBindingAPI bound(prim);
    VtIntArray indices;
    VtFloatArray weights;
    REQUIRE(bound.GetJointIndicesPrimvar().Get(&indices));
    REQUIRE(bound.GetJointWeightsPrimvar().Get(&weights));
    CHECK(indices.size() == size_t{raw.count} * 4);
    CHECK(weights.size() == size_t{raw.count} * 4);
    CHECK(indices[0] == 0);
    CHECK(indices[1] == 1);
    CHECK(weights[0] == 0.75F);
    CHECK(weights[1] == 0.25F);
    CHECK(bound.GetJointIndicesPrimvar().GetElementSize() == 4);
    SdfPathVector skeletons;
    REQUIRE(bound.GetSkeletonRel().GetTargets(&skeletons));
    REQUIRE(skeletons.size() == 1);
    CHECK(skeletons[0] == SdfPath("/Root/Skel"));
    CHECK(!primvars.HasPrimvar(TfToken("primvars:athenea:splat:jointIndices")));

    // The joints move, and only the joints: one array a time code, and the
    // stage's range says where they are.
    const UsdGeomPrimvar moved = primvars.GetPrimvar(TfToken("primvars:athenea:splat:skinningXforms"));
    REQUIRE(moved);
    std::vector<double> times;
    REQUIRE(moved.GetTimeSamples(&times));
    REQUIRE(times.size() == 3);
    CHECK(stage->GetStartTimeCode() == 0.0);
    CHECK(stage->GetEndTimeCode() == 2.0);
    VtMatrix4dArray at;
    REQUIRE(moved.Get(&at, UsdTimeCode(2.0)));
    REQUIRE(at.size() == 3);
    CHECK(at[0][3][0] == Catch::Approx(2.0).margin(1e-6));
    CHECK(at[2][3][0] == Catch::Approx(2.2).margin(1e-6));

    // And the gaussians are all still there, one slot each.
    const UsdVolParticleField3DGaussianSplat splats(prim);
    VtVec3fArray positions;
    REQUIRE(splats.GetPositionsAttr().Get(&positions));
    CHECK(positions.size() == raw.count);
}

// AND THE ENGINE PUTS IT WHERE THE SKELETON IS.
//
// The whole reading path in one measurement: Sync finds the rig's primvars, a
// time change dirties the one of them that is time sampled, the engine skins
// the bind-pose cloud into a posed one, and that is what the frame draws. One
// joint carries every gaussian and slides a unit in x a frame, so the cloud's
// bounds -- folded on the device -- must slide exactly that.
TEST_CASE("a cloud its file says a skeleton carries moves with it", "[usd][gpu][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(512);

    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.joints = 1;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        float* one = rig.influences.data() + size_t{k} * 8;
        one[0] = 0.0F;   // the one joint
        one[1] = 1.0F;   // carrying all of it
    }
    rig.times = {0.0, 1.0, 2.0};
    rig.xforms.resize(rig.times.size() * 16, 0.0F);
    for (size_t frame = 0; frame < rig.times.size(); ++frame) {
        float* m = rig.xforms.data() + frame * 16;
        m[0] = m[5] = m[10] = m[15] = 1.0F;
        m[12] = static_cast<float>(frame);   // the slide, in GfMatrix4f's last row
    }
    REQUIRE(rig.valid());

    const fs::path path = scratch("cloud-carried.usda");
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.skinning = &rig;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));

    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto boundsAt = [&](double time) {
        // A frame of its own puts the stage on the device at that instant.
        auto camera = (*renderer)->framingCamera(time, 35.0, "raster");
        if (!camera) FAIL(camera.error().toString());
        if (auto drawn = (*renderer)->draw(*camera, time, 32, 32, "raster"); !drawn) {
            FAIL(drawn.error().toString());
        }
        auto box = (*renderer)->bounds();
        if (!box) FAIL(box.error().toString());
        REQUIRE(box->has_value());
        return **box;
    };
    const scene::Bounds first = boundsAt(0.0);
    const scene::Bounds later = boundsAt(2.0);
    std::printf("  x from %.3f..%.3f to %.3f..%.3f\n", double(first.min[0]), double(first.max[0]),
                double(later.min[0]), double(later.max[0]));

    // Two time codes on, the one joint has slid two units in x and nothing
    // else has moved at all.
    CHECK(later.min[0] - first.min[0] == Catch::Approx(2.0).margin(1e-3));
    CHECK(later.max[0] - first.max[0] == Catch::Approx(2.0).margin(1e-3));
    CHECK(later.min[1] == Catch::Approx(first.min[1]).margin(1e-3));
    CHECK(later.min[2] == Catch::Approx(first.min[2]).margin(1e-3));
}

// A SKINNED CLOUD'S TRANSFER TURNS WITH IT (proposal 014 B).
//
// A transfer kept as zonal lobes in each gaussian's own frame is turned into
// the world by whatever frame the gaussian has: the one a skeleton's pose gave
// it, or the one a prim's transform puts it in. So the same cloud turned by a
// rigid rotation must shade the same either way -- carried by one joint whose
// transform at time 1 is the rotation, or still and under an Xform of that
// rotation -- under a sky whose image is not the same in any two directions.
// And the same turned cloud without its transfer must shade otherwise: that is
// what says the transfer is what the comparison compares.
TEST_CASE("a skinned cloud's zonal transfer turns with it as a turned still cloud's does",
          "[usd][gpu][skinning][transfer][zonal]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(512);
    // Two lobes a gaussian: an axis in its frame (the octahedral square's u,
    // v) and three coefficients, values a bake would give (a cosine-like
    // lobe and a weaker second one).
    std::vector<float> zonal(size_t{raw.count} * 10, 0.0F);
    uint64_t state = 7;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t k = 0; k < raw.count; ++k) {
        float* one = zonal.data() + size_t{k} * 10;
        one[0] = next();
        one[1] = next();
        one[2] = 0.5F + 0.4F * next();
        one[3] = 0.3F + 0.4F * next();
        one[4] = 0.1F * next();
        one[5] = next();
        one[6] = next();
        one[7] = -0.2F * next();
        one[8] = 0.2F * next() - 0.1F;
        one[9] = 0.1F * next() - 0.05F;
    }
    // The turn: 60 degrees about (1, 2, 0.5).
    const double angle = 60.0 * 3.14159265358979 / 180.0;
    const double ax = 1.0 / std::sqrt(5.25), ay = 2.0 / std::sqrt(5.25), az = 0.5 / std::sqrt(5.25);
    const double c = std::cos(angle), sn = std::sin(angle), t = 1.0 - c;
    const double r[3][3] = {{t * ax * ax + c, t * ax * ay - sn * az, t * ax * az + sn * ay},
                            {t * ax * ay + sn * az, t * ay * ay + c, t * ay * az - sn * ax},
                            {t * ax * az - sn * ay, t * ay * az + sn * ax, t * az * az + c}};
    // As USD holds a transform, vectors on the left: the transpose, by rows.
    std::array<float, 16> turned{};
    std::ostringstream matrix;
    matrix << "( ";
    for (int row = 0; row < 4; ++row) {
        matrix << "(";
        for (int column = 0; column < 4; ++column) {
            const double value = row < 3 && column < 3 ? r[column][row] : (row == column ? 1.0 : 0.0);
            turned[static_cast<size_t>(row * 4 + column)] = static_cast<float>(value);
            matrix << value << (column < 3 ? ", " : "");
        }
        matrix << ")" << (row < 3 ? ", " : " ");
    }
    matrix << ")";

    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    options.linear = true;
    options.transferZonal = zonal;
    const fs::path still = scratch("zonal-still.usda");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, still, options));
    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.joints = 1;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        rig.influences[size_t{k} * 8 + 1] = 1.0F;   // joint 0, all of it
    }
    rig.times = {0.0, 1.0};
    rig.xforms.assign(32, 0.0F);
    rig.xforms[0] = rig.xforms[5] = rig.xforms[10] = rig.xforms[15] = 1.0F;
    std::copy(turned.begin(), turned.end(), rig.xforms.begin() + 16);
    REQUIRE(rig.valid());
    options.skinning = &rig;
    const fs::path carried = scratch("zonal-carried.usda");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, carried, options));

    // A sky that differs in every direction: red across, green up, blue in a
    // spot.
    const fs::path png = scratch("zonal-sky.png");
    {
        std::vector<uint32_t> texels(size_t{64} * 32);
        for (uint32_t y = 0; y < 32; ++y) {
            for (uint32_t x = 0; x < 64; ++x) {
                const uint32_t red = 40 + x * 3;
                const uint32_t green = 30 + y * 6;
                const uint32_t blue = (x > 40 && x < 50 && y > 8 && y < 16) ? 255 : 20;
                texels[size_t{y} * 64 + x] = 0xFF000000u | (blue << 16) | (green << 8) | red;
            }
        }
        HioImageSharedPtr image = HioImage::OpenForWriting(png.string());
        REQUIRE(image);
        HioImage::StorageSpec spec;
        spec.width = 64;
        spec.height = 32;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(image->Write(spec));
    }
    const auto wrapper = [&](const char* name, const fs::path& layer, const std::string& transform) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    subLayers = [@" << layer.string() << "@]\n)\n";
        if (!transform.empty()) {
            out << "over \"World\"\n{\n    over \"Splats\"\n    {\n"
                   "        matrix4d xformOp:transform = " << transform << "\n"
                   "        uniform token[] xformOpOrder = [\"xformOp:transform\"]\n    }\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 9)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n"
               "    asset inputs:texture:file = @" << png.string() << "@\n}\n";
        return path;
    };
    const fs::path stillTurned = wrapper("zonal-still-turned.usda", still, matrix.str());
    usd::ExportOptions bare = options;
    bare.transferZonal = {};
    bare.skinning = nullptr;
    const fs::path plain = scratch("zonal-plain.usda");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, plain, bare));
    const fs::path plainTurned = wrapper("zonal-plain-turned.usda", plain, matrix.str());
    const fs::path skinned = wrapper("zonal-carried-turned.usda", carried, "");

    const uint32_t w = 160, h = 120;
    std::vector<std::string> routes{"raster"};
    const gpu::Caps& caps = gpu->device->caps();
    if (caps.accelerationStructure && (caps.rayQuery || caps.rayTracing)) {
        routes.push_back("rt");
    }
    for (const std::string& route : routes) {
        const auto frame = [&](const fs::path& path) {
            auto renderer = usd::StageRenderer::open(path);
            if (!renderer) FAIL(renderer.error().toString());
            auto image = (*renderer)->render("/Camera", 1.0, w, h, route);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        const gpu::Buffer byXform = frame(stillTurned);
        const gpu::Buffer bySkeleton = frame(skinned);
        const gpu::Buffer untransferred = frame(plainTurned);
        auto same = render::compareHdr(*gpu->library, bySkeleton, byXform, w, h);
        auto moved = render::compareHdr(*gpu->library, untransferred, byXform, w, h);
        REQUIRE(same);
        REQUIRE(moved);
        std::printf("  zonal transfer (%s): carried by the skeleton against the turned prim relMSE %.2e, p99 %.2e, "
                    "max %.2e; the turned cloud without its transfer relMSE %.2e\n",
                    route.c_str(), same->relMse, same->p99Relative, same->maxRelative, moved->relMse);
        // The skinner's frame is re-packed in ten-bit quaternions and its
        // eigen-decomposition rounds, so not bit for bit: within a hundredth
        // at the 99th percentile.
        CHECK(same->relMse < 1.0e-4);
        CHECK(same->p99Relative < 1.0e-2);
        CHECK(moved->relMse > 1.0e-3);
    }
}

// THE SAME CLOUD, BOUND THE SPECIFICATION'S WAY.
//
// SkelBindingAPI on the ParticleField names a Skeleton, the Skeleton's
// animation says where its joint is at each instant, and the cloud's cache of
// the joints is blocked so nothing but the binding can move it. The frame it
// draws must be the frame the cache drew, to the pixel: it is the same
// transform, resolved at render time instead of read from the file.
// A RIG IS A RECORD EACH, AND VALIDATION DROPS RECORDS. A cloud's influences
// were checked against the splats kept rather than the records the file
// holds, so a cloud that lost any gaussian to validation lost its whole rig
// with it, in silence, and drew in its bind pose: a glass sparrow whose
// faintest feathers fell under 1/255 of opacity never flapped. A quarter of
// this cloud is too faint to draw, and the rest must still move.
// A TIME STEP IS NOT A NEW CLOUD. A skinned cloud's joints have samples, so
// every step of the timeline dirties every primvar it has; the file keeps
// integer arrays compressed, and reading one again gave back the same values
// in a new buffer -- which the engine took for a new cloud, uploading all of
// it every frame: 43% of a playing sparrow's main thread. An array with no
// samples is read once (HdAtheneaParticleField's `_held`).
// LEVELS OF DETAIL OF A CLOUD THAT MOVES. A skinned cloud cannot be merged
// into coarser cells (a cell that took wing and body would not know which to
// move with), but it can be converted again at a coarser cell: the levels are
// prims of one `athenea:lod:group`, each with its own `athenea:lod:cell`, and a view
// draws the coarsest whose cell spans no more than a pixel where the cloud is
// nearest. Near, the fine level; far, the coarse one; never both.
TEST_CASE("a cloud in levels of detail draws the coarsest one whose cell a pixel holds",
          "[usd][gpu][lod]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats fine = cloud(4096);
    const io::RawSplats coarse = cloud(256);
    const fs::path fineFile = scratch("lod-fine.usda");
    const fs::path coarseFile = scratch("lod-coarse.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, fine, fineFile, options));
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, coarse, coarseFile, options));
    // The cloud spans about 4 units: cells of 1/400 and 1/25 of that.
    const fs::path assembly = scratch("lod-assembly.usda");
    REQUIRE(usd::writeLodAssembly(assembly, {{fineFile, 0.01}, {coarseFile, 0.16}}, "test"));
    auto renderer = usd::StageRenderer::open(assembly);
    if (!renderer) FAIL(renderer.error().toString());
    const auto drawnFrom = [&](double distance) {
        render::Camera camera = render::Camera::lookingAt({0.0, 0.0, distance}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
        camera.lens.focal = 35.0;
        if (auto drawn = (*renderer)->draw(camera, 0.0, 400, 300, "raster"); !drawn) {
            FAIL(drawn.error().toString());
        }
        return (*renderer)->counters().splats;
    };
    const uint32_t nearBy = drawnFrom(6.0);
    const uint32_t farAway = drawnFrom(4000.0);
    std::printf("  near: %u splats drawn; far: %u\n", nearBy, farAway);
    CHECK(nearBy == fine.count);
    CHECK(farAway == coarse.count);
}

// OUT OF DEVICE MEMORY, THE ENGINE STEPS DOWN. A frame the device's budget
// cannot hold fails as a Result -- the render pass gave back what it could,
// gave up a level of detail and tried once more -- and the frames after it
// are drawn a level coarser than asked, under the same budget, rather than
// the process ending (athenea view on a 5.9 M cloud, docs/decisions.md).
TEST_CASE("a frame past the device's memory budget fails as a Result and the next draws a coarser level",
          "[usd][gpu][lod][memory]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats fine = cloud(4096);
    const io::RawSplats coarse = cloud(256);
    const fs::path fineFile = scratch("oom-fine.usda");
    const fs::path coarseFile = scratch("oom-coarse.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, fine, fineFile, options));
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, coarse, coarseFile, options));
    const fs::path assembly = scratch("oom-assembly.usda");
    REQUIRE(usd::writeLodAssembly(assembly, {{fineFile, 0.01}, {coarseFile, 0.16}}, "test"));
    auto renderer = usd::StageRenderer::open(assembly);
    if (!renderer) FAIL(renderer.error().toString());
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    camera.lens.focal = 35.0;

    // Near: the fine level, as asked.
    REQUIRE((*renderer)->draw(camera, 0.0, 64, 64, "raster"));
    CHECK((*renderer)->counters().splats == fine.count);
    CHECK((*renderer)->memoryRelief().times == 0);

    // Eight MiB more than the device holds, and a frame whose colour alone
    // is sixty-four: refused, given back, tried again, refused, reported.
    gpu::Device& device = (*renderer)->device();
    const uint64_t budget = device.memoryBudget();
    device.setMemoryBudget(device.memoryInUse() + (uint64_t{8} << 20));
    auto big = (*renderer)->draw(camera, 0.0, 2048, 2048, "raster");
    REQUIRE_FALSE(big);
    CHECK(big.error().code() == ErrorCode::OutOfMemory);
    const usd::StageRenderer::MemoryRelief relief = (*renderer)->memoryRelief();
    std::printf("  refused: %s\n  relief: %u, '%s', %u levels coarser\n", big.error().toString().c_str(),
                relief.times, relief.last.c_str(), relief.lodBias);
    CHECK(relief.times == 1);
    CHECK(relief.lodBias == 1);

    // The same view under the same budget: drawn, and from the coarser level.
    REQUIRE((*renderer)->draw(camera, 0.0, 64, 64, "raster"));
    CHECK((*renderer)->counters().splats == coarse.count);
    device.setMemoryBudget(budget);
}

TEST_CASE("the Gaussians panel's numbers say which level a view drew and what the device kept of it",
          "[usd][gpu][lod][counters]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats fine = cloud(4096);
    const io::RawSplats coarse = cloud(256);
    const fs::path fineFile = scratch("stats-fine.usda");
    const fs::path coarseFile = scratch("stats-coarse.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, fine, fineFile, options));
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, coarse, coarseFile, options));
    const fs::path assembly = scratch("stats-assembly.usda");
    REQUIRE(usd::writeLodAssembly(assembly, {{fineFile, 0.01}, {coarseFile, 0.16}}, "test"));
    auto renderer = usd::StageRenderer::open(assembly);
    if (!renderer) FAIL(renderer.error().toString());
    // Not asked for, not gathered.
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 4000.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    camera.lens.focal = 35.0;
    REQUIRE((*renderer)->draw(camera, 0.0, 400, 300, "raster"));
    CHECK((*renderer)->gaussianStats().clouds.empty());

    (*renderer)->setGaussianStats(true);
    REQUIRE((*renderer)->draw(camera, 0.0, 400, 300, "raster"));
    const usd::GaussianStats far = (*renderer)->gaussianStats();
    CHECK(far.route == "raster");
    CHECK(far.inStage == fine.count + coarse.count);
    CHECK(far.submitted == coarse.count);
    REQUIRE(far.clouds.size() == 2);
    uint32_t drawnClouds = 0;
    for (const usd::GaussianCloudStats& c : far.clouds) {
        CHECK(c.lod.find("in 'test'") != std::string::npos);
        CHECK(c.bytes > 0);
        if (c.drawn) {
            ++drawnClouds;
            CHECK(c.gaussians == coarse.count);
            CHECK(c.submitted == coarse.count);
            CHECK(c.lod.rfind("level 1 of 2", 0) == 0);
            CHECK(c.counted);
            CHECK(c.visible == far.visible);
        }
    }
    CHECK(drawnClouds == 1);
    // The device's counts: of the frame just drawn here, since the
    // rasteriser waits for itself; and they add up.
    REQUIRE(far.counted);
    CHECK(far.countedFrame == far.frame);
    CHECK(far.countedSlots == coarse.count);
    uint32_t culled = 0;
    for (uint32_t n : far.culled) {
        culled += n;
    }
    CHECK(far.visible + culled == far.countedSlots);
    CHECK(far.cloudBytes > 0);
    std::printf("  far: %u visible of %u, %u pairs\n", far.visible, far.countedSlots, far.pairs);

    // Near: the fine level, and most of it kept.
    camera = render::Camera::lookingAt({0.0, 0.0, 6.0}, {0.0, 0.0, 0.0}, {0.0, 1.0, 0.0});
    camera.lens.focal = 35.0;
    REQUIRE((*renderer)->draw(camera, 0.0, 400, 300, "raster"));
    const usd::GaussianStats near = (*renderer)->gaussianStats();
    CHECK(near.frame > far.frame);
    CHECK(near.submitted == fine.count);
    CHECK(near.countedSlots == fine.count);
    CHECK(near.visible > 0);
    CHECK(near.pairs >= near.visible);
    for (const usd::GaussianCloudStats& c : near.clouds) {
        if (c.drawn) {
            CHECK(c.lod.rfind("level 0 of 2", 0) == 0);
            CHECK(c.visible == near.visible);
            CHECK(c.pairs == near.pairs);
        }
    }
    std::printf("  near: %u visible of %u, %u pairs\n", near.visible, near.countedSlots, near.pairs);
}

TEST_CASE("a skinned cloud stepping through time is uploaded once", "[usd][gpu][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(512);
    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.jointNames = {"j"};
    rig.joints = 1;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        rig.influences[size_t{k} * 8 + 1] = 1.0F;
    }
    rig.times = {0.0, 1.0, 2.0};
    rig.xforms.resize(rig.times.size() * 16, 0.0F);
    for (size_t frame = 0; frame < rig.times.size(); ++frame) {
        float* m = rig.xforms.data() + frame * 16;
        m[0] = m[5] = m[10] = m[15] = 1.0F;
        m[12] = static_cast<float>(frame);
    }
    REQUIRE(rig.valid());
    // A crate file, as a conversion writes: its integer arrays compressed.
    const fs::path path = scratch("cloud-steps.usdc");
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.skinning = &rig;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto camera = (*renderer)->framingCamera(0.0, 35.0, "raster");
    if (!camera) FAIL(camera.error().toString());
    std::vector<scene::Bounds> boxes;
    for (const double time : {0.0, 1.0, 2.0}) {
        if (auto drawn = (*renderer)->draw(*camera, time, 32, 32, "raster"); !drawn) {
            FAIL(drawn.error().toString());
        }
        auto box = (*renderer)->bounds();
        if (!box) FAIL(box.error().toString());
        REQUIRE(box->has_value());
        boxes.push_back(**box);
    }
    std::printf("  three steps: %llu upload(s); x from %.3f to %.3f\n",
                static_cast<unsigned long long>((*renderer)->cloudUploads()), double(boxes.front().min[0]),
                double(boxes.back().min[0]));
    CHECK((*renderer)->cloudUploads() == 1);
    // And it still moved.
    CHECK(boxes.back().min[0] - boxes.front().min[0] == Catch::Approx(2.0).margin(1e-3));
}

TEST_CASE("a cloud a skeleton carries still moves when validation drops some of its gaussians",
          "[usd][gpu][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    io::RawSplats raw = cloud(512);
    for (uint32_t k = 0; k < raw.count; k += 4) {
        raw.records[size_t{k} * raw.encoding.floatsPerRecord + raw.encoding.opacity] = -20.0F;
    }
    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.jointNames = {"j"};
    rig.joints = 1;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        rig.influences[size_t{k} * 8 + 1] = 1.0F;
    }
    rig.times = {0.0, 1.0};
    rig.xforms.resize(rig.times.size() * 16, 0.0F);
    for (size_t frame = 0; frame < rig.times.size(); ++frame) {
        float* m = rig.xforms.data() + frame * 16;
        m[0] = m[5] = m[10] = m[15] = 1.0F;
        m[12] = 2.0F * static_cast<float>(frame);
    }
    REQUIRE(rig.valid());
    const fs::path path = scratch("cloud-skel-dropped.usda");
    fs::remove(path);
    usd::ExportOptions options;
    options.addCamera = false;
    options.skinning = &rig;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto boundsAt = [&](double time) {
        auto camera = (*renderer)->framingCamera(time, 35.0, "raster");
        if (!camera) FAIL(camera.error().toString());
        if (auto drawn = (*renderer)->draw(*camera, time, 32, 32, "raster"); !drawn) {
            FAIL(drawn.error().toString());
        }
        auto box = (*renderer)->bounds();
        if (!box) FAIL(box.error().toString());
        REQUIRE(box->has_value());
        return **box;
    };
    const scene::Bounds first = boundsAt(0.0);
    const scene::Bounds later = boundsAt(1.0);
    std::printf("  a quarter dropped: x from %.3f..%.3f at t=0 to %.3f..%.3f at t=1\n", double(first.min[0]),
                double(first.max[0]), double(later.min[0]), double(later.max[0]));
    CHECK(later.min[0] - first.min[0] == Catch::Approx(2.0).margin(1e-3));
    CHECK(later.max[0] - first.max[0] == Catch::Approx(2.0).margin(1e-3));
}

TEST_CASE("a cloud bound to a Skeleton by SkelBindingAPI draws where its cached joints put it",
          "[usd][gpu][skinning][skel]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(512);
    usd::SplatSkinning rig;
    rig.skeleton = "/Root/Skel";
    rig.jointNames = {"j"};
    rig.joints = 1;
    rig.influences.resize(size_t{raw.count} * 8, 0.0F);
    for (uint32_t k = 0; k < raw.count; ++k) {
        rig.influences[size_t{k} * 8 + 1] = 1.0F;
    }
    rig.times = {0.0, 1.0, 2.0};
    rig.xforms.resize(rig.times.size() * 16, 0.0F);
    for (size_t frame = 0; frame < rig.times.size(); ++frame) {
        float* m = rig.xforms.data() + frame * 16;
        m[0] = m[5] = m[10] = m[15] = 1.0F;
        m[12] = static_cast<float>(frame);
    }
    REQUIRE(rig.valid());
    const fs::path cached = scratch("cloud-skel.usda");
    fs::remove(cached);
    usd::ExportOptions options;
    options.addCamera = false;
    options.skinning = &rig;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, cached, options));

    // The Skeleton the file names, over the file with its cache blocked.
    // Its animation puts the joint at t=1 where the cache has it at t=2 and
    // the other way about, so a frame that came out of the cache instead of
    // the Skeleton would be the wrong frame, not the same one.
    const fs::path bound = scratch("cloud-skel-bound.usda");
    {
        std::ofstream out(bound);
        out << "#usda 1.0\n(\n    subLayers = [@" << cached.filename().string() << "@]\n)\n"
               "def SkelRoot \"Root\"\n{\n"
               "    def Skeleton \"Skel\" (\n        prepend apiSchemas = [\"SkelBindingAPI\"]\n    )\n    {\n"
               "        uniform token[] joints = [\"j\"]\n"
               "        uniform matrix4d[] bindTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        uniform matrix4d[] restTransforms = [( (1,0,0,0), (0,1,0,0), (0,0,1,0), (0,0,0,1) )]\n"
               "        rel skel:animationSource = </Root/Anim>\n    }\n"
               "    def SkelAnimation \"Anim\"\n    {\n"
               "        uniform token[] joints = [\"j\"]\n"
               "        float3[] translations.timeSamples = {\n            0: [(0, 0, 0)],\n"
               "            1: [(2, 0, 0)],\n            2: [(1, 0, 0)],\n        }\n"
               "        quatf[] rotations = [(1, 0, 0, 0)]\n"
               "        half3[] scales = [(1, 1, 1)]\n    }\n}\n"
               "over \"World\"\n{\n    over \"Splats\"\n    {\n"
               "        matrix4d[] primvars:athenea:splat:skinningXforms = None\n    }\n}\n";
    }
    auto byCache = usd::StageRenderer::open(cached);
    auto bySkel = usd::StageRenderer::open(bound);
    if (!byCache) FAIL(byCache.error().toString());
    if (!bySkel) FAIL(bySkel.error().toString());
    // The blocked cache leaves the bound file nothing to skin by but the
    // Skeleton: the same slide has to come out of it.
    const auto boundsAt = [&](usd::StageRenderer& renderer, double time) {
        auto camera = renderer.framingCamera(time, 35.0, "raster");
        if (!camera) FAIL(camera.error().toString());
        if (auto drawn = renderer.draw(*camera, time, 32, 32, "raster"); !drawn) {
            FAIL(drawn.error().toString());
        }
        auto box = renderer.bounds();
        if (!box) FAIL(box.error().toString());
        REQUIRE(box->has_value());
        return **box;
    };
    const scene::Bounds first = boundsAt(**bySkel, 0.0);
    const scene::Bounds later = boundsAt(**bySkel, 1.0);
    std::printf("  by SkelBindingAPI: x from %.3f..%.3f at t=0 to %.3f..%.3f at t=1\n", double(first.min[0]),
                double(first.max[0]), double(later.min[0]), double(later.max[0]));
    CHECK(later.min[0] - first.min[0] == Catch::Approx(2.0).margin(1e-3));
    CHECK(later.max[0] - first.max[0] == Catch::Approx(2.0).margin(1e-3));

    // And pixel for pixel what the cache draws, from one camera.
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto camera = (*byCache)->framingCamera(2.0, 35.0, "raster");
    if (!camera) FAIL(camera.error().toString());
    const auto frame = [&](usd::StageRenderer& renderer, double time) {
        auto image = renderer.render(*camera, time, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer cacheFrame = frame(**byCache, 2.0);
    const gpu::Buffer skelFrame = frame(**bySkel, 1.0);
    const gpu::Buffer skelRest = frame(**bySkel, 0.0);
    auto same = render::compareImages(*gpu->library, cacheFrame, skelFrame, w, h);
    auto moved = render::compareImages(*gpu->library, skelFrame, skelRest, w, h);
    REQUIRE(same);
    REQUIRE(moved);
    std::printf("  cache at t=2 against Skeleton at t=1: p99 %.3g max %.3g; Skeleton t=1 against t=0: max %.3g\n",
                double(same->p99), double(same->max), double(moved->max));
    CHECK(same->p99 < 1e-4);
    CHECK(moved->max > 0.1);
}

// A UsdPreviewSurface's opacity under one, by the specification's two modes
// (2.6, `opacityMode`). In `presence` it is coverage: the surface is there
// for a sample or it is not, drawn by lot, never a lens of glass (MaterialX's
// own reading, which made a feather's soft edge a glass edge). A red card of
// opacity a over a white card then reads a of red and 1 - a of white; at 0
// the card is not there at all, at 1 it is all there is. In `transparent`,
// the default, the diffuse goes down in favour of what is behind WHILE THE
// EMISSION AND THE SPECULAR STAY WHOLE: the same red card, which emits,
// reads its whole red plus 1 - a of the white behind it.
TEST_CASE("a fractional opacity is coverage: a half-clear red card over a white one reads half of each",
          "[usd][gpu][rt][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const auto stage = [&](float opacity, const char* mode) {
        const fs::path path = scratch("coverage_" + std::string(mode) + std::to_string(int(opacity * 100)) + ".usda");
        std::ofstream out(path);
        const auto card = [&](const char* name, double z, const char* material) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-3, -3, " << z << "), (3, -3, " << z << "), (3, 3, " << z
                << "), (-3, 3, " << z << ")]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/" << material << ">\n}\n";
        };
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        card("Front", 0.0, "Red");
        card("Back", -1.0, "White");
        out << "def Camera \"Camera\"\n{\n"
               "    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, 0, 5)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Red\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Red/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0, 0, 0)\n"
               "            color3f inputs:emissiveColor = (1, 0, 0)\n"
               "            int inputs:useSpecularWorkflow = 1\n"
               "            color3f inputs:specularColor = (0, 0, 0)\n"
               "            float inputs:roughness = 1\n"
               "            float inputs:opacity = " << opacity << "\n"
            << mode
            << "            token outputs:surface\n        }\n    }\n"
               "    def Material \"White\"\n    {\n"
               "        token outputs:surface.connect = </Materials/White/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0, 0, 0)\n"
               "            color3f inputs:emissiveColor = (1, 1, 1)\n"
               "            int inputs:useSpecularWorkflow = 1\n"
               "            color3f inputs:specularColor = (0, 0, 0)\n"
               "            float inputs:roughness = 1\n"
               "            token outputs:surface\n        }\n    }\n}\n";
        return path;
    };
    // Black diffuse, no specular: the cards emit and reflect nothing, so
    // the pixel is the emission of what is there. (With a specular lobe the
    // white card caught the red card as a light -- the emissive table does
    // not draw the lot, see the decisions -- and the headlight of a stage
    // without lights.)
    const uint32_t w = 64, h = 48;
    // Presence is asked for by name; transparent is the default and needs no
    // input at all.
    const char* kPresence = "            int inputs:opacityMode = 1\n";
    const char* kTransparent = "";
    const auto centre = [&](float opacity, const char* technique,
                            const char* mode = "            int inputs:opacityMode = 1\n") {
        auto renderer = usd::StageRenderer::open(stage(opacity, mode));
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(256);
        (*renderer)->setPathTotal(256);
        (*renderer)->setPathBounces(0);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        const size_t at = (size_t{h / 2} * w + w / 2) * 4;
        std::printf("  opacity %.2f, %s: centre (%.3f, %.3f, %.3f, %.3f)\n", double(opacity), technique,
                    double(image->rgba[at]), double(image->rgba[at + 1]), double(image->rgba[at + 2]),
                    double(image->rgba[at + 3]));
        return std::array<float, 4>{image->rgba[at], image->rgba[at + 1], image->rgba[at + 2], image->rgba[at + 3]};
    };
    // Half: red and white by lot, 256 paths a pixel -- a binomial's three
    // sigma is 0.094.
    const auto half = centre(0.5F, "rt");
    CHECK(half[0] == Catch::Approx(1.0F).margin(0.02F));
    CHECK(half[1] == Catch::Approx(0.5F).margin(0.1F));
    CHECK(half[2] == Catch::Approx(0.5F).margin(0.1F));
    CHECK(half[3] == Catch::Approx(1.0F).margin(0.02F));
    // None: the card is cut away by the visibility pass, and the white card is
    // the pixel's -- on both routes.
    for (const char* technique : {"rt", "raster"}) {
        const auto none = centre(0.0F, technique);
        CHECK(none[0] == Catch::Approx(1.0F).margin(0.02F));
        CHECK(none[1] == Catch::Approx(1.0F).margin(0.02F));
        CHECK(none[2] == Catch::Approx(1.0F).margin(0.02F));
    }
    // Whole: red, and nothing of the white behind.
    const auto whole = centre(1.0F, "rt");
    CHECK(whole[0] == Catch::Approx(1.0F).margin(0.02F));
    CHECK(whole[1] == Catch::Approx(0.0F).margin(0.02F));
    CHECK(whole[2] == Catch::Approx(0.0F).margin(0.02F));

    // TRANSPARENT, the default: the emission is whole whatever the opacity,
    // and 1 - a of the white shows through. The lot that draws it keeps the
    // surface in a twentieth of the samples at opacity 0, twenty times the
    // weight, so the margin is the lot's, a tenth at 256 paths.
    const auto clear = centre(0.0F, "rt", kTransparent);
    CHECK(clear[0] == Catch::Approx(2.0F).margin(0.12F));
    CHECK(clear[1] == Catch::Approx(1.0F).margin(0.05F));
    CHECK(clear[2] == Catch::Approx(1.0F).margin(0.05F));
    const auto halfClear = centre(0.5F, "rt", kTransparent);
    CHECK(halfClear[0] == Catch::Approx(1.5F).margin(0.1F));
    CHECK(halfClear[1] == Catch::Approx(0.5F).margin(0.1F));
    CHECK(halfClear[2] == Catch::Approx(0.5F).margin(0.1F));
    // The raster has no way past a surface but the lot, and cuts by it in
    // either mode: at opacity 0 the white card is the pixel's.
    const auto rasterClear = centre(0.0F, "raster", kTransparent);
    CHECK(rasterClear[0] == Catch::Approx(1.0F).margin(0.02F));
    CHECK(rasterClear[1] == Catch::Approx(1.0F).margin(0.02F));
}

// Variant sets: USD's own "pick one of these". A host that offers them offers
// whatever an asset packages that way -- seventy-one animations of a bird, a
// level of detail, a shirt -- so the engine's part is to list them, select one
// and say what interval the selected one's animation occupies, since the
// declared range belongs to the root layer and describes all of them at once.
TEST_CASE("a variant set is listed, selected and drawn, and the timeline follows the selection",
          "[usd][gpu][mesh][variants]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("variants.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n"
               "    startTimeCode = 0\n    endTimeCode = 10\n)\n"
               "def Xform \"World\" (\n"
               "    variants = {\n        string pose = \"left\"\n    }\n"
               "    prepend variantSets = \"pose\"\n)\n{\n"
               // The two variants move the same square and animate it over
               // different intervals: two time codes and eight.
               "    variantSet \"pose\" = {\n"
               "        \"left\" {\n"
               "            over \"Square\"\n            {\n"
               "                double3 xformOp:translate.timeSamples = {\n"
               "                    0: (-1, 0, 0),\n                    2: (-1, 0, 0),\n                }\n"
               "            }\n        }\n"
               "        \"right\" {\n"
               "            over \"Square\"\n            {\n"
               "                double3 xformOp:translate.timeSamples = {\n"
               "                    0: (1, 0, 0),\n                    8: (1, 0, 0),\n                }\n"
               "            }\n        }\n"
               "    }\n"
               "    def Mesh \"Square\"\n    {\n"
               "        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-0.5, -0.5, -5), (0.5, -0.5, -5), (0.5, 0.5, -5), (-0.5, 0.5, -5)]\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "        uniform token subdivisionScheme = \"none\"\n"
               "        color3f[] primvars:displayColor = [(0.9, 0.9, 0.9)] ( interpolation = \"constant\" )\n"
               "    }\n"
               "}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    usd::StageRenderer& stage = **renderer;

    const std::vector<usd::StageVariantSet> sets = stage.variantSets();
    REQUIRE(sets.size() == 1);
    CHECK(sets[0].prim == "/World");
    CHECK(sets[0].name == "pose");
    CHECK(sets[0].variants == std::vector<std::string>{"left", "right"});
    CHECK(sets[0].selected == "left");

    // The interval the selection's samples occupy, not the ten the root layer
    // declares for every variant it holds.
    auto range = stage.animationRange();
    CHECK(range.first == Catch::Approx(0.0));
    CHECK(range.second == Catch::Approx(2.0));

    const uint32_t w = 160, h = 120;
    const auto upload = [&](const usd::StageImage& image) {
        gpu::BufferDesc desc;
        desc.bytes = image.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image.rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const auto draw = [&] {
        auto image = stage.render("/Camera", 1.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        return upload(*image);
    };
    const gpu::Buffer left = draw();
    // Selected as USD writes a selection in a path.
    if (auto set = stage.setVariantSelection("/World{pose=right}"); !set) FAIL(set.error().toString());
    const gpu::Buffer right = draw();
    range = stage.animationRange();
    CHECK(range.second == Catch::Approx(8.0));
    CHECK(stage.variantSets()[0].selected == "right");
    // And back: the same selection draws the same frame again.
    if (auto set = stage.setVariantSelection("/World", "pose", "left"); !set) FAIL(set.error().toString());
    const gpu::Buffer again = draw();

    auto moved = render::compareHdr(*gpu->library, left, right, w, h);
    auto returned = render::compareHdr(*gpu->library, left, again, w, h);
    REQUIRE(moved);
    REQUIRE(returned);
    std::printf("  variant selection: left against right relMSE %.2e, left against left again %.2e\n",
                moved->relMse, returned->relMse);
    CHECK(moved->relMse > 1e-2);
    CHECK(returned->maxRelative < 1e-6);

    // What a host may not do.
    CHECK_FALSE(stage.setVariantSelection("/World", "pose", "sideways"));
    CHECK_FALSE(stage.setVariantSelection("/World", "colour", "red"));
    CHECK_FALSE(stage.setVariantSelection("/Nothing", "pose", "left"));
    CHECK_FALSE(stage.setVariantSelection("/World.pose"));
    // The file itself is untouched: the selection lives in the session layer.
    CHECK(std::string(std::istreambuf_iterator<char>(*std::make_unique<std::ifstream>(path)),
                      std::istreambuf_iterator<char>())
              .find("string pose = \"right\"") == std::string::npos);
}

// A DOME IS THE ONE LIGHT WHOSE WHOLE CHARACTER IS A FILE, so a host can offer
// to change it: `athenea view` lists every sky beside the one a dome carries and
// puts the chosen one on it. What the engine owes that host is three things --
// say which domes there are and what each holds, put an image on one, and turn
// it -- with the stage on disk untouched, as a variant selection leaves it.
TEST_CASE("a dome's sky is listed, swapped and turned, and the stage on disk is not touched",
          "[usd][gpu][mesh][dome]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const auto sky = [&](const std::string& name, uint32_t left, uint32_t right) {
        const fs::path png = scratch(name);
        fs::remove(png);
        // Lat-long: the left half of the image is one colour and the right
        // half another, so turning the dome changes which one faces the
        // square and the frame has to move.
        std::vector<uint32_t> texels(size_t{64} * 32, left);
        for (uint32_t y = 0; y < 32; ++y) {
            for (uint32_t x = 32; x < 64; ++x) {
                texels[size_t{y} * 64 + x] = right;
            }
        }
        HioImageSharedPtr image = HioImage::OpenForWriting(png.string());
        REQUIRE(image);
        HioImage::StorageSpec spec;
        spec.width = 64;
        spec.height = 32;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(image->Write(spec));
        return png;
    };
    const fs::path pale = sky("dome_pale.png", 0xFFFFFFFFu, 0xFF404040u);
    const fs::path dark = sky("dome_dark.png", 0xFF303030u, 0xFF202020u);

    const fs::path path = scratch("dome_swap.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def Mesh \"Square\"\n    {\n"
               "        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "        uniform token subdivisionScheme = \"none\"\n"
               "        color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
               "    }\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n"
               "    uniform token poleAxis = \"scene\"\n"
               "    asset inputs:texture:file = @" << pale.string() << "@\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    usd::StageRenderer& stage = **renderer;

    std::vector<usd::StageDome> domes = stage.domes();
    REQUIRE(domes.size() == 1);
    CHECK(domes[0].prim == "/Sky");
    CHECK(domes[0].name == "Sky");
    CHECK(fs::path(domes[0].texture).filename() == pale.filename());
    CHECK(domes[0].rotation == Catch::Approx(0.0F));

    const uint32_t w = 96, h = 72;
    const auto draw = [&] {
        auto image = stage.render("/Camera", 1.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer first = draw();

    if (auto set = stage.setDomeTexture("/Sky", dark.string()); !set) FAIL(set.error().toString());
    CHECK(fs::path(stage.domes()[0].texture).filename() == dark.filename());
    const gpu::Buffer swapped = draw();

    // Back to the stage's own opinion: an empty path clears the session's,
    // and the frame is the one the file asked for, to the last bit.
    if (auto set = stage.setDomeTexture("/Sky", std::string()); !set) FAIL(set.error().toString());
    CHECK(fs::path(stage.domes()[0].texture).filename() == pale.filename());
    const gpu::Buffer restored = draw();

    // Turned half a turn: the half of the sky that was lighting the square is
    // now behind it.
    if (auto set = stage.setDomeRotation("/Sky", 180.0F); !set) FAIL(set.error().toString());
    CHECK(stage.domes()[0].rotation == Catch::Approx(180.0F));
    const gpu::Buffer turned = draw();

    auto changed = render::compareHdr(*gpu->library, first, swapped, w, h);
    auto back = render::compareHdr(*gpu->library, first, restored, w, h);
    auto moved = render::compareHdr(*gpu->library, first, turned, w, h);
    REQUIRE(changed);
    REQUIRE(back);
    REQUIRE(moved);
    std::printf("  dome: swapped relMSE %.2e, restored maxRel %.2e, turned relMSE %.2e\n",
                changed->relMse, back->maxRelative, moved->relMse);
    CHECK(changed->relMse > 1e-2);
    CHECK(back->maxRelative < 1e-6);
    CHECK(moved->relMse > 1e-3);

    // What a host may not do.
    CHECK_FALSE(stage.setDomeTexture("/World/Square", pale.string()));
    CHECK_FALSE(stage.setDomeTexture("/Nothing", pale.string()));
    CHECK_FALSE(stage.setDomeRotation("/World/Square", 90.0F));

    // The file itself never learns about any of it.
    const std::string authored(std::istreambuf_iterator<char>(*std::make_unique<std::ifstream>(path)),
                               std::istreambuf_iterator<char>());
    CHECK(authored.find(dark.filename().string()) == std::string::npos);
    CHECK(authored.find("rotateXYZ") == std::string::npos);
}

// A STAGE BRINGS ITS OWN LIGHTS, and a host trying skies has to be able to
// take the rest away: a sun left on under every image is a highlight no sky
// explains. What the engine owes is to list them, the ones switched off
// included, and to switch one off and back on through the session layer --
// back on to the last bit, and with the file never told.
TEST_CASE("a stage's lights are listed and switched off and on, and the stage on disk is not touched",
          "[usd][gpu][mesh][lights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("light_switch.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def Mesh \"Square\"\n    {\n"
               "        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-1, -1, -5), (1, -1, -5), (1, 1, -5), (-1, 1, -5)]\n"
               "        uniform token subdivisionScheme = \"none\"\n"
               "        color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n"
               "    }\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def Xform \"Lights\"\n{\n"
               "    def DomeLight \"Sky\"\n    {\n        float inputs:intensity = 0.3\n    }\n"
               "    def DistantLight \"Sun\"\n    {\n        bool inputs:normalize = 1\n        float inputs:intensity = 3\n    }\n"
               "    def SphereLight \"Lamp\" ( active = false )\n    {\n        float inputs:intensity = 50\n"
               "        double3 xformOp:translate = (0, 0, -3)\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n"
               "}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    usd::StageRenderer& stage = **renderer;

    const auto on = [&](const std::string& prim) {
        for (const usd::StageLight& light : stage.lights()) {
            if (light.prim == prim) {
                return light.on;
            }
        }
        FAIL("no light " << prim << " listed");
        return false;
    };
    std::vector<usd::StageLight> lights = stage.lights();
    REQUIRE(lights.size() == 3);
    CHECK(lights[0].prim == "/Lights/Sky");
    CHECK(lights[1].name == "Sun");
    CHECK(lights[1].type == "DistantLight");
    CHECK(lights[0].on);
    CHECK(lights[1].on);
    CHECK_FALSE(lights[2].on);

    const uint32_t w = 96, h = 72;
    const auto draw = [&] {
        auto image = stage.render("/Camera", 1.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer first = draw();

    if (auto set = stage.setLightOn("/Lights/Sun", false); !set) FAIL(set.error().toString());
    CHECK_FALSE(on("/Lights/Sun"));
    const gpu::Buffer sunless = draw();
    if (auto set = stage.setLightOn("/Lights/Sun", true); !set) FAIL(set.error().toString());
    CHECK(on("/Lights/Sun"));
    const gpu::Buffer restored = draw();

    // A light the file left off is switched on by saying so, and the lamp in
    // front of the square changes the frame.
    if (auto set = stage.setLightOn("/Lights/Lamp", true); !set) FAIL(set.error().toString());
    CHECK(on("/Lights/Lamp"));
    const gpu::Buffer lamp = draw();
    if (auto set = stage.setLightOn("/Lights/Lamp", false); !set) FAIL(set.error().toString());

    // A dome switched off leaves the domes a host offers skies for.
    REQUIRE(stage.domes().size() == 1);
    if (auto set = stage.setLightOn("/Lights/Sky", false); !set) FAIL(set.error().toString());
    CHECK(stage.domes().empty());
    CHECK(stage.lights().size() == 3);

    auto dimmed = render::compareHdr(*gpu->library, first, sunless, w, h);
    auto back = render::compareHdr(*gpu->library, first, restored, w, h);
    auto lit = render::compareHdr(*gpu->library, first, lamp, w, h);
    REQUIRE(dimmed);
    REQUIRE(back);
    REQUIRE(lit);
    std::printf("  lights: sun off relMSE %.2e, back on maxRel %.2e, lamp on relMSE %.2e\n",
                dimmed->relMse, back->maxRelative, lit->relMse);
    CHECK(dimmed->relMse > 1e-2);
    CHECK(back->maxRelative < 1e-6);
    CHECK(lit->relMse > 1e-2);

    CHECK_FALSE(stage.setLightOn("/World/Square", false));
    CHECK_FALSE(stage.setLightOn("/Nothing", false));

    const std::string authored(std::istreambuf_iterator<char>(*std::make_unique<std::ifstream>(path)),
                               std::istreambuf_iterator<char>());
    CHECK(authored.find("active = false") != std::string::npos);   // the lamp's own, and only it
    CHECK(authored.find("active = false") == authored.rfind("active = false"));
    CHECK(authored.find("active = true") == std::string::npos);
}

// ANTIALIASING: A PATH TRACED FRAME IS A MEAN, AND THE CAMERA MOVES INSIDE
// THE PIXEL BETWEEN THE PASSES IT IS A MEAN OF.
//
// Every ray goes through the middle of its pixel and the visibility buffer
// holds one hit a pixel, so one pass draws a staircase however many paths it
// casts. Gathering the frame over passes with a sub-pixel offset between them
// turns that mean into an average over the pixel's area -- which is what
// antialiasing is, at no cost in rays. A diagonal edge says it plainly: with
// one pass every pixel is either the quad or the background, and with many a
// band of them holds the fraction each side covers.
TEST_CASE("a path traced frame antialiases its edges as it gathers, and one pass is what it always was",
          "[usd][gpu][mesh][path][antialias]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path path = scratch("antialias.usda");
    {
        std::ofstream out(path);
        // A quad turned about the view axis, so its edges cross the pixel
        // grid at an angle no rasteriser can hide.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Slant\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1.1, -0.35, -4), (0.35, -1.1, -4), (1.1, 0.35, -4), (-0.35, 1.1, -4)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.9, 0.9, 0.9)] ( interpolation = \"constant\" )\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n";
    }
    const uint32_t w = 200, h = 150;
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto counter = gpu::ComputeKernel::create(*gpu->library, "athenea/test/edge_count", "edgeCount");
    if (!counter) FAIL(counter.error().toString());
    // How many pixels are neither the quad nor the background: the width of
    // the edge, counted on the device.
    const auto between = [&](const usd::StageImage& image) {
        gpu::BufferDesc desc;
        desc.bytes = image.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto frame = gpu::Buffer::create(*gpu->device, desc, image.rgba.data());
        REQUIRE(frame);
        const std::array<uint32_t, 3> zero{0, 0, 0};
        auto counts = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "edge.counts");
        REQUIRE(counts);
        {
            gpu::CommandBatch batch(*gpu->device);
            counter->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["frame"].setBinding(frame->rhi());
                cursor["counts"].setBinding(counts->rhi());
                cursor["params"]["pixels"].setData(w * h);
                // The quad is displayColor 0.9 under a distant light of 3,
                // facing it: 0.9/pi * 3 = 0.859, and the background is black.
                // Anything between a twentieth of that and four fifths of it
                // is a pixel the edge runs through.
                cursor["params"]["low"].setData(0.05F);
                cursor["params"]["high"].setData(0.70F);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 3> read{};
        REQUIRE(counts->read(*gpu->device, 0, sizeof(read), read.data()));
        return read;
    };
    (*renderer)->setPathSamples(1);
    (*renderer)->setPathBounces(0);

    // One pass: the frame the engine has always drawn, sampled at the pixel's
    // middle and nowhere else.
    (*renderer)->setPathTotal(1);
    auto one = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!one) FAIL(one.error().toString());
    const std::array<uint32_t, 3> sharp = between(*one);

    // And gathered over many, each with its own offset inside the pixel.
    (*renderer)->setPathTotal(64);
    auto many = (*renderer)->render("/Camera", 0.0, w, h, "rt");
    if (!many) FAIL(many.error().toString());
    const std::array<uint32_t, 3> soft = between(*many);

    std::printf("  antialias: one pass between %u, dark %u, bright %u; 64 passes between %u, dark %u, bright %u\n",
                sharp[0], sharp[1], sharp[2], soft[0], soft[1], soft[2]);
    // One pass leaves nothing between: every pixel is the quad or the
    // background. The gathered frame holds partial coverage along the whole
    // perimeter -- measured, 148 pixels of the roughly 460 the edge runs
    // through, the rest of them being covered in nearly all the passes or
    // nearly none, which is what a straight edge does to a box filter.
    CHECK(sharp[0] < 10);
    CHECK(soft[0] > 100);
    // And it is the same quad: what is fully covered and fully empty barely
    // moves, since the edge is where the difference lives.
    CHECK(std::abs(int(soft[1]) - int(sharp[1])) < int(w * h) / 20);
}

// OPACITY MODE, UsdPreviewSurface 2.6.
//
// `transparent` (the default) says a fractional opacity takes the diffuse
// down in favour of what is behind while the specular and the emission stay
// whole: a window at opacity 0 still reflects the sky. `presence` scales the
// whole response, and the surface is there by lot. The path tracer draws the
// first as the specification says, and the raster by the same lot.
TEST_CASE("a UsdPreviewSurface at opacity 0 keeps its specular in transparent mode and vanishes in presence mode",
          "[usd][gpu][mesh][materials][opacity][path]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    // A glossy front square over a matte back one, a sun placed so the
    // front's reflection of it lands in the frame; `front` is the preview
    // surface's inputs, empty for a stage with the back square alone.
    const auto stageText = [](const std::string& front) {
        std::ostringstream out;
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        if (!front.empty()) {
            out << "def Mesh \"Front\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-1, -1, -3), (1, -1, -3), (1, 1, -3), (-1, 1, -3)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/Front>\n}\n";
        }
        out << "def Mesh \"Back\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-1.5, -1.5, -5), (1.5, -1.5, -5), (1.5, 1.5, -5), (-1.5, 1.5, -5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Materials/Back>\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 4\n    float inputs:angle = 2\n"
               "    bool inputs:shadow:enable = 0\n"
               "    double3 xformOp:rotateXYZ = (-20, 15, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"Front\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Front/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.8, 0.1, 0.1)\n"
               "            int inputs:useSpecularWorkflow = 1\n"
               "            color3f inputs:specularColor = (1, 1, 1)\n"
               "            float inputs:roughness = 0.35\n"
            << front
            << "            token outputs:surface\n        }\n    }\n"
               "    def Material \"Back\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Back/Preview.outputs:surface>\n"
               "        def Shader \"Preview\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.2, 0.5, 0.9)\n"
               "            float inputs:roughness = 1\n"
               "            token outputs:surface\n        }\n    }\n}\n";
        return out.str();
    };
    const uint32_t w = 160, h = 120;
    const auto frame = [&](const char* name, const std::string& front, const char* technique) {
        const fs::path path = scratch(std::string("opacitymode_") + name + ".usda");
        {
            std::ofstream out(path);
            out << stageText(front);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        (*renderer)->setPathSamples(16);
        (*renderer)->setPathTotal(256);
        (*renderer)->setPathBounces(1);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    // The frame's premultiplied luminance, in fixed point: what a specular
    // adds shows here as more light than the back square alone.
    auto energyKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/image_average", "imageEnergy");
    if (!energyKernel) FAIL(energyKernel.error().toString());
    const auto energy = [&](const gpu::Buffer& colour) {
        const uint32_t zero = 0;
        auto total = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(&zero, 1), "energy");
        REQUIRE(total);
        gpu::CommandBatch batch(*gpu->device);
        energyKernel->dispatch(batch, {w * h, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["accum"].setBinding(total->rhi());
            cursor["frame"].setBinding(colour.rhi());
            cursor["total"].setBinding(total->rhi());
            cursor["params"]["pixels"].setData(w * h);
            cursor["params"]["weight"].setData(0.0F);
            cursor["params"]["scale"].setData(1024.0F);
        });
        REQUIRE(batch.submit(true));
        uint32_t out = 0;
        REQUIRE(total->read(*gpu->device, 0, 4, &out));
        return out;
    };

    const gpu::Buffer alone = frame("alone", "", "rt");
    const gpu::Buffer presence = frame("presence", "            float inputs:opacity = 0\n            int inputs:opacityMode = 1\n", "rt");
    const gpu::Buffer transparent = frame("transparent", "            float inputs:opacity = 0\n", "rt");
    const gpu::Buffer opaque = frame("opaque", "            float inputs:opacity = 1\n", "rt");
    auto presenceGone = render::compareHdr(*gpu->library, presence, alone, w, h);
    auto transparentSeen = render::compareHdr(*gpu->library, transparent, alone, w, h);
    auto opaqueSeen = render::compareHdr(*gpu->library, opaque, alone, w, h);
    REQUIRE(presenceGone);
    REQUIRE(transparentSeen);
    REQUIRE(opaqueSeen);
    const uint32_t eAlone = energy(alone), ePresence = energy(presence), eTransparent = energy(transparent),
                   eOpaque = energy(opaque);
    std::printf("  opacity 0, path traced: presence against the back alone relMSE %.2e; transparent %.2e; opaque "
                "%.2e. Energy: alone %u, presence %u, transparent %u, opaque %u\n",
                presenceGone->relMse, transparentSeen->relMse, opaqueSeen->relMse, eAlone, ePresence, eTransparent,
                eOpaque);
    // Presence at opacity 0: the surface is not there, and the back is as
    // it would be alone.
    CHECK(presenceGone->relMse < 1e-3);
    // Transparent at opacity 0: the back shows through, the diffuse is gone
    // (no red), and the sun's reflection is added -- more light than alone,
    // and a different frame.
    CHECK(transparentSeen->relMse > 1e-3);
    CHECK(eTransparent > eAlone);
    // And opaque is the red square, whichever mode.
    CHECK(opaqueSeen->relMse > 1e-2);

    // The estimator, by its linearity. At opacity 1/2, presence gives half of
    // (specular + diffuse) and half of the back; transparent gives the whole
    // specular, half the diffuse and half the back. Their difference is half
    // the specular -- which at opacity 0 was the whole of what transparent
    // added over the back alone. So the ratio is a half, and it is measured,
    // not assumed.
    const gpu::Buffer presenceHalf =
        frame("presence_half", "            float inputs:opacity = 0.5\n            int inputs:opacityMode = 1\n", "rt");
    const gpu::Buffer transparentHalf = frame("transparent_half", "            float inputs:opacity = 0.5\n", "rt");
    const double half = double(energy(transparentHalf)) - double(energy(presenceHalf));
    const double whole = double(eTransparent) - double(eAlone);
    std::printf("  at opacity 1/2, transparent less presence is %.3f of the specular transparent added at 0\n",
                half / whole);
    CHECK(half / whole > 0.42);
    CHECK(half / whole < 0.58);

    // The raster draws the same two readings. Presence at opacity 0 is gone,
    // as it always was. Transparent is kept by the tracer's lot, one pixel in
    // twenty at twenty times its specular: what it adds over the back alone
    // is, over the frame, what the tracer's transparent card adds. It used
    // to be cut away like presence -- a feather card's clear texels
    // reflected nothing under the raster and a glass sheet's worth under the
    // tracer, and the sparrow came out two different birds.
    const gpu::Buffer rasterAlone = frame("raster_alone", "", "raster");
    const gpu::Buffer rasterPresence =
        frame("raster_presence", "            float inputs:opacity = 0\n            int inputs:opacityMode = 1\n",
              "raster");
    const gpu::Buffer rasterTransparent = frame("raster_transparent", "            float inputs:opacity = 0\n", "raster");
    auto rasterGone = render::compareHdr(*gpu->library, rasterPresence, rasterAlone, w, h);
    REQUIRE(rasterGone);
    const double rasterAdded = double(energy(rasterTransparent)) - double(energy(rasterAlone));
    std::printf("  raster: presence at opacity 0 against the back alone relMSE %.2e; transparent adds %.3f of what "
                "it adds path traced\n",
                rasterGone->relMse, rasterAdded / whole);
    CHECK(rasterGone->relMse < 1e-3);
    // Within what one thing the raster cannot draw: a sample its lot passed
    // shows the back whole, where the tracer takes off what the sheet
    // reflected (transparentPasses) -- here a white specular, nearly all.
    CHECK(rasterAdded / whole > 0.8);
    CHECK(rasterAdded / whole < 1.4);
}

namespace {

/// A stage written from `text`, rendered by `technique` at w x h with the
/// settings `configure` gives, its colour on the device.
gpu::Buffer renderStageText(const std::string& name, const std::string& text, const char* technique, uint32_t w,
                            uint32_t h, const std::function<void(usd::StageRenderer&)>& configure = {}) {
    const fs::path path = scratch(name);
    {
        std::ofstream out(path);
        out << text;
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    if (configure) configure(**renderer);
    auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*::athenea::test::gpuOrNull()->device, desc, image->rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

}   // namespace

// THE RASTER'S SHADOW RAYS IN A KERNEL OF THEIR OWN.
//
// The shading kernel with a ray query in it miscompiled on Metal (Apple M5
// Pro): rows of garbage in blocks of half a threadgroup. A copy of the lobe
// stack kept out of it, and the shadow walk kept from asking a cut-out's
// opacity, kept the floor of the test above clean -- and the sparrow's frame
// striped red and yellow all the same, because it merely *held* a cut-out
// material. One unbound card material of opacity one half beside the same
// grey floor is enough: 200 718 of 518 400 pixels differed at 960 x 540 with
// the shadows on against off. Now the rays are traceShadows' and the frames
// are the same word for word.
TEST_CASE("shadow rays change nothing over an unoccluded floor when the frame holds a cut-out material",
          "[usd][gpu][mesh][lights][shadows]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stage = [](bool shadows) {
        std::ostringstream out;
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -1, -2), (6, -1, -2), (6, -1, -14), (-6, -1, -14)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Looks/Grey>\n}\n"
               "def Scope \"Looks\"\n{\n"
               "    def Material \"Grey\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Grey/Surface.outputs:surface>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.8, 0.8, 0.8)\n"
               "            token outputs:surface\n        }\n    }\n"
               // Bound to nothing: it is in the frame's materials, and that is all.
               "    def Material \"Card\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Card/Surface.outputs:surface>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            float inputs:opacity = 0.5\n"
               "            token outputs:surface\n        }\n    }\n}\n"
               "def DomeLight \"Dome\"\n{\n    float inputs:intensity = 0.5\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "}\n"
               "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 3\n    float inputs:angle = 2\n"
            << (shadows ? "" : "    bool inputs:shadow:enable = 0\n")
            << "    float3 xformOp:rotateXYZ = (-60, 0, 25)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1.5, 0)\n    double xformOp:rotateX = -18\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return out.str();
    };
    const uint32_t w = 192, h = 128;
    const gpu::Buffer off = renderStageText("cutmaterial_shadows_off.usda", stage(false), "raster", w, h);
    for (int k = 0; k < 3; ++k) {
        const gpu::Buffer on = renderStageText("cutmaterial_shadows_on.usda", stage(true), "raster", w, h);
        auto words = render::countDifferent(*gpu->library, on, off, w * h * 4);
        REQUIRE(words);
        std::printf("  a cut-out material in the frame, shadows on against off, run %d: %llu of %u words differ\n",
                    k + 1, static_cast<unsigned long long>(*words), w * h * 4);
        CHECK(*words == 0);
    }
}

// A CUT-OUT'S SHADOW IS ITS COVERAGE, ON THE RASTER ROUTE TOO. A card at
// opacity one half (presence) between the sun and a floor, behind the
// camera: the floor under it gets half the sun, by the shadow rays' lots.
// While the shadow ray shared the shading kernel it could not ask the card's
// material, and a card cast its whole shadow -- the black hole in the
// sparrow's belly, under dozens of soft feather cards.
TEST_CASE("a half-clear card casts half a shadow on the raster route", "[usd][gpu][mesh][lights][shadows][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const auto stage = [](bool card) {
        std::ostringstream out;
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-20, 0, 20), (20, 0, 20), (20, 0, -20), (-20, 0, -20)]\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n";
        if (card) {
            out << "def Mesh \"Card\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-20, 3, 20), (20, 3, 20), (20, 3, -20), (-20, 3, -20)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Looks/Half>\n}\n"
                   "def Scope \"Looks\"\n{\n    def Material \"Half\"\n    {\n"
                   "        token outputs:surface.connect = </Looks/Half/Surface.outputs:surface>\n"
                   "        def Shader \"Surface\"\n        {\n"
                   "            uniform token info:id = \"UsdPreviewSurface\"\n"
                   "            float inputs:opacity = 0.5\n            int inputs:opacityMode = 1\n"
                   "            token outputs:surface\n        }\n    }\n}\n";
        }
        // The sun straight down; the camera at height 1 looking down, under
        // the card, which it does not see.
        out << "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 2\n"
               "    double3 xformOp:rotateXYZ = (-90, 0, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 1, 0)\n    double xformOp:rotateX = -90\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return out.str();
    };
    const uint32_t w = 128, h = 96;
    const gpu::Buffer lit = renderStageText("half_shadow_none.usda", stage(false), "raster", w, h);
    const gpu::Buffer half = renderStageText("half_shadow_card.usda", stage(true), "raster", w, h);
    auto open = render::imageStats(*gpu->library, lit, w, h);
    auto under = render::imageStats(*gpu->library, half, w, h);
    REQUIRE(open);
    REQUIRE(under);
    const double ratio = under->mean[0] / std::max(open->mean[0], 1e-9);
    std::printf("  a floor under a card at opacity 1/2 gets %.3f of the open sun (0.5 wanted)\n", ratio);
    CHECK(std::abs(ratio - 0.5) < 0.05);
}

// EVERY LAYER DRAWS ITS OWN LOT. Three emissive cards one behind the other --
// red at opacity one half, green at one half, an opaque blue back -- seen
// through by the raster's lot: red half the frame, green a quarter, blue a
// quarter, which is what the path tracer draws. One lot a pixel for every
// layer drew blue half the frame and green never: a sample the front's lot
// cut was cut from the middle card too.
TEST_CASE("the raster's lot is drawn a layer at a time: two half-clear cards let a quarter through",
          "[usd][gpu][mesh][materials][coverage]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
    const auto card = [&](const char* name, double z, double half, const char* emit, const char* opacity) {
        out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
            << "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
            << "    point3f[] points = [(" << -half << ", " << -half << ", " << z << "), (" << half << ", " << -half
            << ", " << z << "), (" << half << ", " << half << ", " << z << "), (" << -half << ", " << half << ", "
            << z << ")]\n"
            << "    uniform token subdivisionScheme = \"none\"\n"
            << "    rel material:binding = </" << name << "/Look>\n"
            << "    def Material \"Look\"\n    {\n"
            << "        token outputs:surface.connect = </" << name << "/Look/Surface.outputs:surface>\n"
            << "        def Shader \"Surface\"\n        {\n"
            << "            uniform token info:id = \"UsdPreviewSurface\"\n"
            << "            color3f inputs:diffuseColor = (0, 0, 0)\n"
            << "            int inputs:useSpecularWorkflow = 1\n"
            << "            color3f inputs:specularColor = (0, 0, 0)\n"
            << "            color3f inputs:emissiveColor = " << emit << "\n"
            << "            float inputs:opacity = " << opacity << "\n"
            << "            int inputs:opacityMode = 1\n"
            << "            token outputs:surface\n        }\n    }\n}\n";
    };
    card("Front", -3.0, 1.0, "(1, 0, 0)", "0.5");
    card("Middle", -4.0, 2.0, "(0, 1, 0)", "0.5");
    card("Back", -5.0, 3.0, "(0, 0, 1)", "1");
    out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n}\n";
    const uint32_t w = 160, h = 120;
    // The middle of the frame, inside the front card's square.
    const uint32_t x0 = 60, y0 = 40, x1 = 100, y1 = 80;
    for (const char* technique : {"raster", "rt"}) {
        INFO(technique);
        const gpu::Buffer frame = renderStageText(std::string("layer_lots_") + technique + ".usda", out.str(),
                                                  technique, w, h, [](usd::StageRenderer& r) {
                                                      r.setPathSamples(16);
                                                      r.setPathTotal(64);
                                                  });
        auto stats = render::imageStats(*gpu->library, frame, w, h, x0, y0, x1, y1);
        REQUIRE(stats);
        std::printf("  %s: red %.3f, green %.3f, blue %.3f of the window (0.5, 0.25, 0.25 wanted)\n", technique,
                    stats->mean[0], stats->mean[1], stats->mean[2]);
        CHECK(std::abs(stats->mean[0] - 0.5) < 0.05);
        CHECK(std::abs(stats->mean[1] - 0.25) < 0.05);
        CHECK(std::abs(stats->mean[2] - 0.25) < 0.05);
    }
}

// A WHITE FURNACE FOR TRANSPARENT OPACITY. Twenty clear mirror sheets one
// behind the other (UsdPreviewSurface at opacity 0, transparent by default,
// metallic and white) under a dome of radiance one: nothing in it absorbs
// and nothing emits, so it returns at most what arrives. The specification
// keeps each sheet's specular whole; taken as (1 - opacity) of what is behind
// as well, each sheet reflected and passed everything, and the stack read
// 206 in the path tracer. What passes is what the sheet does not reflect.
TEST_CASE("a stack of transparent sheets returns no more light than a dome of radiance one gives it",
          "[usd][gpu][mesh][materials][opacity][path]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    std::ostringstream out;
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
           "def Mesh \"Sheets\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
           "    uniform bool doubleSided = 1\n    int[] faceVertexCounts = [";
    for (int i = 0; i < 20; ++i) out << (i ? ", " : "") << 4;
    out << "]\n    int[] faceVertexIndices = [";
    for (int i = 0; i < 80; ++i) out << (i ? ", " : "") << i;
    out << "]\n    point3f[] points = [";
    for (int i = 0; i < 20; ++i) {
        const double z = -0.05 * i;
        out << (i ? ", " : "") << "(-1, -1, " << z << "), (1, -1, " << z << "), (1, 1, " << z << "), (-1, 1, " << z
            << ")";
    }
    out << "]\n    uniform token subdivisionScheme = \"none\"\n    rel material:binding = </Looks/Sheet>\n}\n"
           "def Scope \"Looks\"\n{\n    def Material \"Sheet\"\n    {\n"
           "        token outputs:surface.connect = </Looks/Sheet/Surface.outputs:surface>\n"
           "        def Shader \"Surface\"\n        {\n"
           "            uniform token info:id = \"UsdPreviewSurface\"\n"
           "            color3f inputs:diffuseColor = (1, 1, 1)\n"
           "            float inputs:metallic = 1\n            float inputs:roughness = 0.4\n"
           "            float inputs:opacity = 0\n"
           "            token outputs:surface\n        }\n    }\n}\n"
           "def DomeLight \"Dome\"\n{\n    float inputs:intensity = 1\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    float2 clippingRange = (0.1, 1000)\n"
           "    double3 xformOp:translate = (0, 0, 4)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    const uint32_t w = 96, h = 72;
    const gpu::Buffer frame = renderStageText("transparent_furnace.usda", out.str(), "rt", w, h,
                                              [](usd::StageRenderer& r) {
                                                  r.setPathSamples(16);
                                                  r.setPathTotal(64);
                                              });
    // The sheets' middle.
    auto stats = render::imageStats(*gpu->library, frame, w, h, 38, 26, 58, 46);
    REQUIRE(stats);
    std::printf("  twenty transparent mirror sheets under a white dome: mean %.3f (at most 1)\n", stats->mean[0]);
    CHECK(stats->mean[0] < 1.05);
    CHECK(stats->mean[0] > 0.5);
}

// A LOBE SAMPLE IS SHADOWED. A polished metal floor under a plate far wider
// than the view, a dome of radiance one above both: the floor sees the
// plate's underside, which nothing lights, so the raster -- direct light
// only -- draws it black. While the trace shared the shading kernel a lobe's
// sample traced no shadow ray, so the floor reflected the sky through the
// plate; a sparrow's belly, under the bird, read 0.26 where the tracer
// reads 0.16. And a broad lobe's light samples are weighed against its own
// samples by their true densities: the same floor at roughness one half
// under the open sky returns at most the sky's one, not a dome sample at
// forty times it at a grazing pixel.
TEST_CASE("the raster shadows a lobe's own samples, and weighs them against the light's at any roughness",
          "[usd][gpu][mesh][lights][shadows]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const auto stage = [](bool plate, const char* roughness) {
        std::ostringstream out;
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-60, 0, 60), (60, 0, 60), (60, 0, -60), (-60, 0, -60)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Looks/Metal>\n}\n";
        if (plate) {
            out << "def Mesh \"Plate\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(-60, 1, 60), (-60, 1, -60), (60, 1, -60), (60, 1, 60)]\n"
                   "    color3f[] primvars:displayColor = [(0, 0, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n}\n";
        }
        out << "def Scope \"Looks\"\n{\n    def Material \"Metal\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Metal/Surface.outputs:surface>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (1, 1, 1)\n"
               "            float inputs:metallic = 1\n"
               "            float inputs:roughness = "
            << roughness
            << "\n            token outputs:surface\n        }\n    }\n}\n"
               "def DomeLight \"Dome\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 24\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 16.384\n"
               "    float2 clippingRange = (0.01, 1000)\n"
               "    double3 xformOp:translate = (0, 0.5, 0)\n    double xformOp:rotateX = -30\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n}\n";
        return out.str();
    };
    const uint32_t w = 128, h = 96;
    const gpu::Buffer covered = renderStageText("lobe_shadow_plate.usda", stage(true, "0.1"), "raster", w, h);
    auto under = render::imageStats(*gpu->library, covered, w, h, 0, 0, w, h / 2);
    REQUIRE(under);
    std::printf("  a polished floor under a plate, raster: mean %.4f (black wanted)\n", under->mean[0]);
    CHECK(under->mean[0] < 0.02);

    const gpu::Buffer open = renderStageText("lobe_mis_open.usda", stage(false, "0.5"), "raster", w, h);
    auto sky = render::imageStats(*gpu->library, open, w, h, 0, 0, w, h / 2);
    REQUIRE(sky);
    std::printf("  a rough metal floor under the open sky, raster: mean %.3f, brightest pixel %.2f\n",
                sky->mean[0], static_cast<double>(sky->max[0]));
    CHECK(sky->mean[0] < 1.05);
    CHECK(sky->max[0] < 4.0F);
}

// UsdLux ShadowAPI on a cloud's shadow: `shadow:color` tints what the cloud
// stops (a non-physical control, and the specification says so),
// `shadow:enable` turns it off, and `shadow:distance` ends it short of a
// receiver further from the light than that. All three reach the map the
// raster reads and the factors a relit cloud reads, through the light record.
// A CLOUD SHADOWS A MESH UNDER A SKY, ON THE RASTER ROUTE (task TX).
//
// The map from the lights had no slot for a dome -- a dome has no direction
// to build one about -- so a car converted to gaussians cast nothing on the
// ground under a sky (CV2's Corvette: the ground under the car 0.217 where
// the path traced mesh reads 0.182, and identical without cloud shadows). The
// dome now gets maps along its zenith and a ring forty degrees up, and a
// mesh's dome samples read the nearest. A wide slab of opaque gaussians low
// over a floor under a plain sky: the floor under it, seen from the side,
// must go well darker with the cloud's shadows than without.
TEST_CASE("a cloud shadows a mesh under a dome on the raster route", "[usd][gpu][mesh][splat][lights][dome]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const fs::path path = scratch("cloud_dome_shadow.usda");
    {
        std::ofstream out(path);
        const int n = 21;
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
               "def Mesh \"Ground\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-6, -6, 0), (6, -6, 0), (6, 6, 0), (-6, 6, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def ParticleField3DGaussianSplat \"Cloud\"\n{\n    point3f[] positions = [";
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                out << ((i || j) ? ", " : "") << "(" << (i - n / 2) * 0.09 << ", " << (j - n / 2) * 0.09 << ", 0.4)";
            }
        }
        out << "]\n    quatf[] orientations = [";
        for (int k = 0; k < n * n; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
        out << "]\n    float3[] scales = [";
        for (int k = 0; k < n * n; ++k) out << (k ? ", " : "") << "(0.07, 0.07, 0.02)";
        out << "]\n    float[] opacities = [";
        for (int k = 0; k < n * n; ++k) out << (k ? ", " : "") << "0.99";
        out << "]\n    int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [";
        for (int k = 0; k < n * n; ++k) out << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
        out << "]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, -2.5, 0.25)\n    float3 xformOp:rotateXYZ = (85, 0, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    const uint32_t w = 160, h = 120;
    auto meanKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/patch_mean", "patchMean");
    if (!meanKernel) FAIL(meanKernel.error().toString());
    const auto under = [&](bool shadows) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        (*renderer)->setLightSamples(16);
        (*renderer)->setCloudShadows(shadows);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(frame);
        const std::array<uint32_t, 4> zero{0, 0, 0, 0};
        auto sums = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "dome.sums");
        REQUIRE(sums);
        {
            gpu::CommandBatch batch(*gpu->device);
            meanKernel->dispatch(batch, {12, 12, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["frame"].setBinding(frame->rhi());
                cursor["sums"].setBinding(sums->rhi());
                cursor["params"]["width"].setData(w);
                cursor["params"]["x0"].setData(w / 2 - 6);
                cursor["params"]["y0"].setData(h / 2 + 4);
                cursor["params"]["w"].setData(12u);
                cursor["params"]["h"].setData(12u);
                cursor["params"]["scale"].setData(4096.0F);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 4> read{};
        REQUIRE(sums->read(*gpu->device, 0, sizeof(read), read.data()));
        const double count = std::max<double>(read[3], 1.0) * 4096.0;
        return (read[0] + read[1] + read[2]) / (3.0 * count);
    };
    const double with = under(true);
    const double without = under(false);
    std::printf("  the floor under a cloud, under a sky: %.4f with the cloud's shadows, %.4f without\n", with,
                without);
    CHECK(without > 0.1);
    CHECK(with < 0.6 * without);
}

TEST_CASE("a cloud's shadow on a plane is tinted, switched off and cut short as the light's ShadowAPI says",
          "[usd][gpu][mesh][splat][lights][shadowapi]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization) {
        SKIP("no rasterisation on this device");
    }
    // A white plane, a slab of opaque gaussians above its middle, the camera
    // straight over them, and a light from the left: the shadow falls to the
    // right of the slab, where the camera sees the plane and not the cloud.
    const auto stageText = [](const std::string& light) {
        std::ostringstream out;
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\n"
               "def Mesh \"Ground\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def ParticleField3DGaussianSplat \"Cloud\"\n{\n"
               "    point3f[] positions = [";
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) {
                out << ((i || j) ? ", " : "") << "(" << (i - 4) * 0.09 << ", " << (j - 4) * 0.09 << ", 0.6)";
            }
        }
        out << "]\n    quatf[] orientations = [";
        for (int k = 0; k < 81; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
        out << "]\n    float3[] scales = [";
        for (int k = 0; k < 81; ++k) out << (k ? ", " : "") << "(0.07, 0.07, 0.02)";
        out << "]\n    float[] opacities = [";
        for (int k = 0; k < 81; ++k) out << (k ? ", " : "") << "0.99";
        out << "]\n    int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [";
        for (int k = 0; k < 81; ++k) out << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
        out << "]\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 30\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, 0, 3)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
            << light;
        return out.str();
    };
    // The sun from 45 degrees to the left: the slab at 0.6 up shadows the
    // plane 0.6 to the right of itself.
    const std::string sun =
        "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    float inputs:angle = 0.5\n"
        "    float3 xformOp:rotateXYZ = (0, -45, 0)\n    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n";
    // A lamp to the left and up: 1.4 from the slab, 2.4 from the shadow it
    // throws, which is what a `shadow:distance` in between tells apart.
    const std::string lamp =
        "def SphereLight \"Lamp\"\n{\n    float inputs:intensity = 1500\n    float inputs:radius = 0.05\n"
        "    double3 xformOp:translate = (-1, 0, 1.6)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n";

    const uint32_t w = 160, h = 120;
    auto meanKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/patch_mean", "patchMean");
    if (!meanKernel) FAIL(meanKernel.error().toString());
    // The mean colour of the patch the shadow lands on: 0.75 to the right of
    // the middle, which the camera's 30 mm lens puts 49 pixels right of it.
    const auto patch = [&](const usd::StageImage& image) {
        gpu::BufferDesc desc;
        desc.bytes = image.rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto frame = gpu::Buffer::create(*gpu->device, desc, image.rgba.data());
        REQUIRE(frame);
        const std::array<uint32_t, 4> zero{0, 0, 0, 0};
        auto sums = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "patch.sums");
        REQUIRE(sums);
        {
            gpu::CommandBatch batch(*gpu->device);
            meanKernel->dispatch(batch, {12, 12, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["frame"].setBinding(frame->rhi());
                cursor["sums"].setBinding(sums->rhi());
                cursor["params"]["width"].setData(w);
                cursor["params"]["x0"].setData(w / 2 + 49 - 6);
                cursor["params"]["y0"].setData(h / 2 - 6);
                cursor["params"]["w"].setData(12u);
                cursor["params"]["h"].setData(12u);
                cursor["params"]["scale"].setData(4096.0F);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 4> read{};
        REQUIRE(sums->read(*gpu->device, 0, sizeof(read), read.data()));
        const double n = std::max<double>(read[3], 1.0) * 4096.0;
        return std::array<double, 3>{read[0] / n, read[1] / n, read[2] / n};
    };
    const auto shadowOf = [&](const char* name, const std::string& light, const char* technique) {
        const fs::path path = scratch(std::string("shadowapi_") + name + ".usda");
        {
            std::ofstream out(path);
            out << stageText(light);
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setAntialias(false);
        (*renderer)->setLightSamples(4);
        // The path tracer shadows a mesh with a cloud only when asked: it is
        // a ray a light a sample through the cloud's proxies.
        (*renderer)->setSplatShadows(true);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        const std::array<double, 3> mean = patch(*image);
        std::printf("  %-6s %-7s the shadow's patch (%.3f, %.3f, %.3f)\n", name, technique, mean[0], mean[1],
                    mean[2]);
        return mean;
    };
    const auto ratio = [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return std::array<double, 3>{a[0] / std::max(b[0], 1e-6), a[1] / std::max(b[1], 1e-6),
                                     a[2] / std::max(b[2], 1e-6)};
    };
    for (const char* technique : {"raster", "rt"}) {
        // Off: no shadow at all, which is what the others are measured against.
        const auto off = shadowOf("off", sun + "    bool inputs:shadow:enable = 0\n}\n", technique);
        CHECK(off[1] > 0.3);
        // Black, as a shadow is: the cloud is opaque and stops nearly all.
        const auto plain = ratio(shadowOf("plain", sun + "}\n", technique), off);
        CHECK(plain[1] < 0.1);
        // Red: what the cloud stops is replaced by the shadow's colour, so red
        // comes through whole and green and blue are stopped.
        const auto red = ratio(shadowOf("red", sun + "    color3f inputs:shadow:color = (1, 0, 0)\n}\n", technique), off);
        CHECK(red[0] > 0.9);
        CHECK(red[1] < 0.1);
        CHECK(red[2] < 0.1);
        // The lamp's shadow reaches 2 units from it: the plane, 2.4 away, is
        // past its end, while without the limit it is in it.
        const auto lampOff = shadowOf("lampoff", lamp + "    bool inputs:shadow:enable = 0\n}\n", technique);
        CHECK(lampOff[1] > 0.05);
        const auto near = ratio(shadowOf("near", lamp + "}\n", technique), lampOff);
        CHECK(near[1] < 0.15);
        const auto cut = ratio(shadowOf("cut", lamp + "    float inputs:shadow:distance = 2\n}\n", technique), lampOff);
        CHECK(cut[1] > 0.9);
    }
}

// THE HINTS ARE HINTS. `projectionModeHint` and `sortingModeHint` are, the
// schema says, a renderer's to ignore, and this one draws the same frame
// whatever they say (and names what they say when the stage is opened).
TEST_CASE("a cloud's projection and sorting hints change nothing about its frame", "[usd][gpu][splat][hints]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const auto stage = [&](const char* name, const char* hints) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n"
               "def ParticleField3DGaussianSplat \"Cloud\"\n{\n"
            << hints
            << "    point3f[] positions = [(-0.3, 0, 0), (0, 0, 0), (0.3, 0, 0), (0, 0.3, 0.1), (0, -0.3, -0.1)]\n"
               "    quatf[] orientations = [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]\n"
               "    float3[] scales = [(0.1, 0.1, 0.1), (0.15, 0.1, 0.05), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]\n"
               "    float[] opacities = [0.9, 0.8, 0.9, 0.7, 0.7]\n"
               "    int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]\n"
               "}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, 0, 2)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 96, h = 72;
    const auto frame = [&](const fs::path& path) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer plain = frame(stage("hints_plain.usda", ""));
    const gpu::Buffer hinted = frame(stage("hints_tangential.usda",
                                           "    uniform token projectionModeHint = \"tangential\"\n"
                                           "    uniform token sortingModeHint = \"rayHitDistance\"\n"));
    auto differ = render::countDifferent(*gpu->library, plain, hinted, w * h * 4);
    REQUIRE(differ);
    std::printf("  %llu of %u words differ between the hinted cloud and the plain one\n",
                static_cast<unsigned long long>(*differ), w * h * 4);
    CHECK(*differ == 0);
}

// DomeLight_1's `poleAxis` (UsdLux, 26.x): where the dome's top is. "scene"
// puts it at the stage's up axis, "Z" at +Z whatever the stage says, and the
// old DomeLight has it at +Y always, which is what made a Z-up stage's sky
// lie on its side. UsdImaging folds the alignment into the light's transform,
// and this engine's dome has +Y of that transform for its pole: so a plane
// facing the pole under an image white above the horizon and black below
// reads its whole albedo, 0.8, and a plane facing the horizon half of it.
TEST_CASE("a dome's pole is where DomeLight_1's poleAxis puts it, on a Y-up and a Z-up stage",
          "[usd][gpu][mesh][lights][dome][poleaxis]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    // The image: white above the horizon, black below, in a lat-long.
    const fs::path png = scratch("dome_halves.png");
    {
        fs::remove(png);
        std::vector<uint32_t> texels(size_t{32} * 16, 0xFFFFFFFFu);
        for (size_t k = size_t{32} * 8; k < texels.size(); ++k) texels[k] = 0xFF000000u;
        HioImageSharedPtr made = HioImage::OpenForWriting(png.string());
        REQUIRE(made);
        HioImage::StorageSpec spec;
        spec.width = 32;
        spec.height = 16;
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(made->Write(spec));
    }
    auto meanKernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/patch_mean", "patchMean");
    if (!meanKernel) FAIL(meanKernel.error().toString());
    const uint32_t w = 96, h = 72;
    // A plane facing +Y or +Z, its camera three units off it looking at it,
    // and the dome as asked.
    const auto reading = [&](const char* name, const char* up, const char* dome, char facing) {
        const fs::path path = scratch(std::string("pole_") + name + ".usda");
        {
            std::ofstream out(path);
            out << "#usda 1.0\n(\n    upAxis = \"" << up << "\"\n)\n"
                   "def Mesh \"Plane\"\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                << (facing == 'Y' ? "    point3f[] points = [(-1, 0, 1), (1, 0, 1), (1, 0, -1), (-1, 0, -1)]\n"
                                  : "    point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]\n")
                << "    uniform token subdivisionScheme = \"none\"\n"
                   "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
                   "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
                   "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
                   "    float2 clippingRange = (0.1, 100)\n"
                << (facing == 'Y' ? "    double3 xformOp:translate = (0, 3, 0)\n"
                                    "    float3 xformOp:rotateXYZ = (-90, 0, 0)\n"
                                    "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n"
                                  : "    double3 xformOp:translate = (0, 0, 3)\n"
                                    "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n")
                << "}\n"
                << dome << "    float inputs:intensity = 1\n    asset inputs:texture:file = @" << png.string()
                << "@\n}\n";
        }
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setLightSamples(512);
        auto image = (*renderer)->render("/Camera", 0.0, w, h, "raster");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto frame = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(frame);
        const std::array<uint32_t, 4> zero{0, 0, 0, 0};
        auto sums = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, std::span<const uint32_t>(zero), "pole.sums");
        REQUIRE(sums);
        {
            gpu::CommandBatch batch(*gpu->device);
            meanKernel->dispatch(batch, {12, 12, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["frame"].setBinding(frame->rhi());
                cursor["sums"].setBinding(sums->rhi());
                cursor["params"]["width"].setData(w);
                cursor["params"]["x0"].setData(w / 2 - 6);
                cursor["params"]["y0"].setData(h / 2 - 6);
                cursor["params"]["w"].setData(12u);
                cursor["params"]["h"].setData(12u);
                cursor["params"]["scale"].setData(4096.0F);
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 4> read{};
        REQUIRE(sums->read(*gpu->device, 0, sizeof(read), read.data()));
        const double mean = read[1] / (std::max<double>(read[3], 1.0) * 4096.0);
        std::printf("  %-22s up %s, plane facing +%c: %.3f\n", name, up, facing, mean);
        return mean;
    };
    const char* dome1 = "def DomeLight_1 \"Sky\"\n{\n";
    const char* dome1Z = "def DomeLight_1 \"Sky\"\n{\n    uniform token poleAxis = \"Z\"\n";
    const char* dome0 = "def DomeLight \"Sky\"\n{\n";
    // The pole at the stage's up axis, whichever it is.
    CHECK(reading("scene, Y-up", "Y", dome1, 'Y') == Catch::Approx(0.8).margin(0.06));
    CHECK(reading("scene, Z-up", "Z", dome1, 'Z') == Catch::Approx(0.8).margin(0.06));
    // Asked for +Z on a Y-up stage: the plane facing +Z sees the sky, the
    // one facing +Y the horizon.
    CHECK(reading("Z, Y-up", "Y", dome1Z, 'Z') == Catch::Approx(0.8).margin(0.06));
    CHECK(reading("Z, Y-up, facing Y", "Y", dome1Z, 'Y') == Catch::Approx(0.4).margin(0.06));
    // The old DomeLight has no pole axis: +Y, and a Z-up stage's sky on its side.
    CHECK(reading("DomeLight, Z-up", "Z", dome0, 'Z') == Catch::Approx(0.4).margin(0.06));
}

// The stage's active render settings prim (`renderSettingsPrimPath`) says
// `disableMotionBlur`: its products drew without blur, and now so does a
// camera of the engine's own with a shutter asked for by `athenea:shutter`.
TEST_CASE("the active render settings switch motion blur off for a camera of the engine's own",
          "[usd][gpu][mesh][path][render-settings][motion]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    const uint32_t w = 96, h = 64;
    const auto stage = [&](const char* name, bool noBlur) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    startTimeCode = 0\n    endTimeCode = 1\n"
               "    renderSettingsPrimPath = \"/Render/Settings\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-0.5, -1, -5), (0.5, -1, -5), (0.5, 1, -5), (-0.5, 1, -5)]\n"
               "    double3 xformOp:translate.timeSamples = {\n        0: (-1, 0, 0),\n        1: (1, 0, 0),\n    }\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.8, 0.8, 0.8)] ( interpolation = \"constant\" )\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 3\n    bool inputs:shadow:enable = 0\n}\n"
               "def Scope \"Render\"\n{\n    def RenderSettings \"Settings\"\n    {\n"
            << "        uniform bool disableMotionBlur = " << (noBlur ? 1 : 0) << "\n    }\n}\n";
        return path;
    };
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 0.0}, {0.0, 0.0, -5.0});
    camera.lens.focal = 35.0;
    camera.lens.haperture = 24.576;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    const auto frame = [&](const fs::path& path, bool shutter) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        (*renderer)->setPathSamples(16);
        (*renderer)->setPathTotal(16);
        (*renderer)->setPathBounces(0);
        (*renderer)->setMotionBuckets(8);
        (*renderer)->setAntialias(false);
        if (shutter) {
            (*renderer)->setShutter(-0.25, 0.25);
        }
        auto image = (*renderer)->render(camera, 0.5, w, h, "rt");
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto buffer = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(buffer);
        return *buffer;
    };
    const gpu::Buffer still = frame(stage("own_still.usda", false), false);
    const gpu::Buffer blurred = frame(stage("own_blurred.usda", false), true);
    const gpu::Buffer switched = frame(stage("own_switched.usda", true), true);
    auto blurs = render::countDifferent(*gpu->library, blurred, still, w * h * 4);
    auto held = render::countDifferent(*gpu->library, switched, still, w * h * 4);
    REQUIRE(blurs);
    REQUIRE(held);
    std::printf("  the shutter changes %llu words; with disableMotionBlur on the active settings, %llu\n",
                static_cast<unsigned long long>(*blurs), static_cast<unsigned long long>(*held));
    CHECK(*blurs > 0);
    CHECK(*held == 0);
}

// A DISPLACED GAUSSIAN IS BAKED AS THE RELIEF FACES. It stands off the mesh,
// and the tracer holds only the flat one, so its bake starts from the flat
// point under it -- and its light must be the light of a surface turned as
// the relief turns it. The reference is the tracer's own answer on a plane
// that is really turned: points on the flat plane facing 40 degrees over
// must bake what points on the plane turned by 40 degrees bake. Facing
// sideways, a sun overhead leaves almost nothing; and a facing given as none
// (w 0) is the bake with no facing at all, to the bit.
TEST_CASE("a displaced gaussian bakes as the relief faces, from the flat surface under it",
          "[usd][gpu][mesh][bake][displacement]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_facing.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n"
               "def Mesh \"Plane\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Materials/White>\n}\n"
               // The same plane turned 40 degrees about y, far to the side:
               // what the relief's facing has to look like.
               "def Mesh \"Turned\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    double3 xformOp:translate = (10, 0, 0)\n"
               "    float3 xformOp:rotateXYZ = (0, 40, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n"
               "    rel material:binding = </Materials/White>\n}\n"
               "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 1\n    float inputs:angle = 0.5\n}\n"
               "def Scope \"Materials\"\n{\n"
               "    def Material \"White\"\n    {\n"
               "        token outputs:surface.connect = </Materials/White/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.8, 0.8, 0.8)\n"
               "            float inputs:roughness = 0\n"
               "            token outputs:surface\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_check", "bakeCheck");
    if (!check) FAIL(check.error().toString());
    // Points standing a little off the plane, as a raised gaussian does: the
    // ray still comes down onto it.
    const uint32_t count = 16;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -1.0F + 2.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    const auto bakedWith = [&](const std::vector<float>* facing) {
        auto baked = (*renderer)->bakePoints(rays, count, 0.0, 256, 1, 0, false, facing);
        if (!baked) FAIL(baked.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = baked->size() * 4;
        desc.elementBytes = 16;
        auto fitted = gpu::Buffer::create(*gpu->device, desc, baked->data());
        REQUIRE(fitted);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "facing.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "facing.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["coefficients"].setBinding(fitted->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["planes"].setData(1u);
                const float direction[4] = {0.0F, 0.0F, -1.0F, 1.0e6F};
                const float want[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                p["direction"].setData(direction, sizeof(direction));
                p["expected"].setData(want, sizeof(want));
            });
            REQUIRE(batch.submit(true));
        }
        float got[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(worst.read(*gpu->device, 0, sizeof(got), got));
        return got[1];
    };
    const auto facingAt = [&](float degrees, float w) {
        const float t = degrees * 3.14159265F / 180.0F;
        std::vector<float> facing(size_t{count} * 4, 0.0F);
        for (uint32_t k = 0; k < count; ++k) {
            float* f = facing.data() + size_t{k} * 4;
            f[0] = std::sin(t);
            f[2] = std::cos(t);
            f[3] = w;
        }
        return facing;
    };
    const std::vector<float> none = facingAt(40.0F, 0.0F);
    const std::vector<float> up = facingAt(0.0F, 1.0F);
    const std::vector<float> turned = facingAt(40.0F, 1.0F);
    const std::vector<float> sideways = facingAt(89.0F, 1.0F);
    const float plain = bakedWith(nullptr);
    const float saidNone = bakedWith(&none);
    const float facingUp = bakedWith(&up);
    const float facingTurned = bakedWith(&turned);
    const float facingSideways = bakedWith(&sideways);
    // The same points' light, from the plane that really is turned.
    const float t = 40.0F * 3.14159265F / 180.0F;
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        const float along = -1.0F + 2.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[0] = 10.0F + along * std::cos(t);
        ray[2] = -along * std::sin(t);
        ray[4] = std::sin(t);
        ray[6] = std::cos(t);
    }
    const float reallyTurned = bakedWith(nullptr);
    std::printf("  facing: none %.4f, said none %.4f, up %.4f; turned 40 deg %.4f against a turned plane's %.4f; "
                "sideways %.4f\n",
                double(plain), double(saidNone), double(facingUp), double(facingTurned), double(reallyTurned),
                double(facingSideways));
    CHECK(saidNone == plain);
    CHECK(std::abs(facingUp - plain) < 0.01F * plain);
    CHECK(std::abs(facingTurned - reallyTurned) < 0.03F * reallyTurned);
    CHECK(facingTurned < 0.97F * facingUp);
    CHECK(facingSideways < 0.15F * facingUp);
}

// A METAL IS NOT ALWAYS A conductor_bsdf, AND A BAKE MUST NOT DROP IT.
//
// What a bake keeps of a material is its body -- the diffuse lobes, what it
// transmits, and a conductor's reflection, which is all a metal has -- and
// what it drops is the dielectric polish, which `splat_relight` puts back
// with a direction in it. MaterialX writes OpenPBR's metal as a generalized
// Schlick rather than as a conductor, so the rule "drop every Schlick" threw
// a metal away: a car's chrome baked to exactly zero, every gaussian of it,
// while the same metal written as UsdPreviewSurface baked correctly. The two
// spellings of one metal must bake alike, and a dielectric's polish must
// still be dropped.
TEST_CASE("a metal bakes its reflection whichever node writes it, and a polish is still dropped",
          "[usd][gpu][mesh][bake][metal]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("bake_metal.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
        const auto plane = [&](const char* name, double x, const char* material) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                   "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
                   "    point3f[] points = [(" << x - 1.8 << ", -1.8, -1.5), (" << x + 1.8 << ", -1.8, -1.5), ("
                << x + 1.8 << ", 1.8, -1.5), (" << x - 1.8 << ", 1.8, -1.5)]\n"
                   "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                   "    rel material:binding = </Materials/" << material << ">\n}\n";
        };
        plane("PreviewMetal", -4.0, "PreviewMetal");
        plane("MxMetal", 0.0, "MxMetal");
        plane("Polish", 4.0, "Polish");
        out << "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Scope \"Materials\"\n{\n"
               // The metal as USD spells it: our own graph, which uses conductor_bsdf.
               "    def Material \"PreviewMetal\"\n    {\n"
               "        token outputs:surface.connect = </Materials/PreviewMetal/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.9, 0.9, 0.9)\n"
               "            float inputs:metallic = 1\n"
               "            float inputs:roughness = 0.25\n"
               "            token outputs:surface\n        }\n    }\n"
               // And as MaterialX spells it: OpenPBR, whose metal is a
               // generalized Schlick, which is what a car arrives with.
               "    def Material \"MxMetal\"\n    {\n"
               "        token outputs:mtlx:surface.connect = </Materials/MxMetal/S.outputs:out>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
               "            color3f inputs:base_color = (0.9, 0.9, 0.9)\n"
               "            float inputs:base_metalness = 1\n"
               "            float inputs:specular_roughness = 0.25\n"
               "            token outputs:out\n        }\n    }\n"
               // A dielectric of a dark body under a bright polish: the body
               // is what a bake keeps, so it stays dark.
               "    def Material \"Polish\"\n    {\n"
               "        token outputs:surface.connect = </Materials/Polish/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.04, 0.04, 0.04)\n"
               "            float inputs:metallic = 0\n"
               "            float inputs:roughness = 0.25\n"
               "            token outputs:surface\n        }\n    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/bake_check", "bakeCheck");
    if (!check) FAIL(check.error().toString());
    // What one plane's points bake to, read back through the check kernel:
    // a tolerance nothing can exceed, so what it reports is the reading.
    const uint32_t count = 32;
    const auto bakedAt = [&](double x) {
        std::vector<float> rays(size_t{count} * 8, 0.0F);
        for (uint32_t k = 0; k < count; ++k) {
            float* ray = rays.data() + size_t{k} * 8;
            ray[0] = static_cast<float>(x) - 1.2F + 2.4F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
            ray[2] = -1.5F;
            ray[3] = 1.0e-3F;
            ray[6] = 1.0F;
        }
        auto baked = (*renderer)->bakePoints(rays, count, 0.0, 256, 1, 0);
        if (!baked) FAIL(baked.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = baked->size() * 4;
        desc.elementBytes = 16;
        auto fitted = gpu::Buffer::create(*gpu->device, desc, baked->data());
        REQUIRE(fitted);
        gpu::Buffer counts = test::uintBuffer(*gpu->device, 2, "metal.counts");
        gpu::Buffer worst = test::uintBuffer(*gpu->device, 4, "metal.worst");
        {
            gpu::CommandBatch batch(*gpu->device);
            check->dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["coefficients"].setBinding(fitted->rhi());
                cursor["counts"].setBinding(counts.rhi());
                cursor["worst"].setBinding(worst.rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["planes"].setData(1u);
                const float direction[4] = {0.0F, 0.0F, -1.0F, 1.0e6F};
                const float want[4] = {0.0F, 0.0F, 0.0F, 0.0F};
                p["direction"].setData(direction, sizeof(direction));
                p["expected"].setData(want, sizeof(want));
            });
            REQUIRE(batch.submit(true));
        }
        float got[4] = {0.0F, 0.0F, 0.0F, 0.0F};
        REQUIRE(worst.read(*gpu->device, 0, sizeof(got), got));
        return std::array<float, 3>{got[1], got[2], got[3]};
    };
    const std::array<float, 3> preview = bakedAt(-4.0);
    const std::array<float, 3> mtlx = bakedAt(0.0);
    const std::array<float, 3> polish = bakedAt(4.0);
    std::printf("  UsdPreviewSurface metal %.3f, OpenPBR metal %.3f, a dielectric's polish %.3f\n",
                double(preview[1]), double(mtlx[1]), double(polish[1]));
    // Both metals send back most of the dome they stand under.
    CHECK(preview[1] > 0.6F);
    CHECK(mtlx[1] > 0.6F);
    // And they agree: it is one metal, written twice.
    CHECK(std::abs(mtlx[1] - preview[1]) < 0.1F);
    // The polish is not the body: what the dark surface keeps is its own.
    CHECK(polish[1] < 0.35F);
}

// WHICH PRIMS A PIXEL SAW, AS CRYPTOMATTE WRITES IT. A cloud converted from a
// mesh knows the prim each of its gaussians came from, so a cloud and a mesh
// are named in the same matte: the layers hold (id, coverage) pairs, the
// coverages of a pixel add up to its alpha, and a product's EXR carries the
// manifest that turns the numbers back into paths.
TEST_CASE("a Cryptomatte names a converted cloud's prims and a mesh's, and the EXR carries the manifest",
          "[usd][gpu][crypto]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const uint32_t w = 96, h = 72;
    const uint32_t idCloud = core::cryptomatteId("/World/Cloud");
    const uint32_t idPlane = core::cryptomatteId("/World/Plane");
    const uint32_t idBadge = core::cryptomatteId("/Source/Badge");
    const fs::path path = scratch("crypto_object.usda");
    {
        std::ofstream out(path);
        // The cloud's gaussians name two source prims: the first three came
        // from /Source/Body, the last two from /Source/Badge, which is what
        // `athenea mesh2splat` writes when it converts two meshes into one cloud.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def ParticleField3DGaussianSplat \"Cloud\"\n    {\n"
               "        point3f[] positions = [(-0.3, 0, 0), (0, 0, 0), (0.3, 0, 0), (0, 0.35, 0), (0, -0.35, 0)]\n"
               "        quatf[] orientations = [(1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]\n"
               "        float3[] scales = [(0.12, 0.12, 0.12), (0.12, 0.12, 0.12), (0.12, 0.12, 0.12), (0.1, 0.1, 0.1), (0.1, 0.1, 0.1)]\n"
               "        float[] opacities = [0.9, 0.9, 0.9, 0.85, 0.85]\n"
               "        int radiance:sphericalHarmonicsDegree = 0\n"
               "        float3[] radiance:sphericalHarmonicsCoefficients = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1)]\n"
               "        int[] primvars:athenea:splat:cryptoObject = ["
            << static_cast<int32_t>(idCloud) << ", " << static_cast<int32_t>(idCloud) << ", "
            << static_cast<int32_t>(idCloud) << ", " << static_cast<int32_t>(idBadge) << ", "
            << static_cast<int32_t>(idBadge) << "] (\n"
               "            interpolation = \"vertex\"\n        )\n"
               "        string primvars:athenea:splat:cryptoManifest = '{\"/World/Cloud\":\""
            << core::hex8(idCloud) << "\",\"/Source/Badge\":\"" << core::hex8(idBadge) << "\"}'\n"
               "    }\n"
               "    def Mesh \"Plane\"\n    {\n"
               "        int[] faceVertexCounts = [4]\n"
               "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-2, -2, -1), (2, -2, -1), (2, 2, -1), (-2, 2, -1)]\n"
               "        color3f[] primvars:displayColor = [(0.2, 0.2, 0.8)]\n"
               "    }\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, 0, 2)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());

    // The frame without the matte first: it must not move when one is kept.
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto plain = (*renderer)->mappedOutput("color");
    REQUIRE(plain);
    auto plainIds = (*renderer)->mappedOutput("primId");
    (void)plainIds;

    (*renderer)->requestOutputs({"CryptoObject00", "CryptoObject01", "CryptoObject02", "primId"});
    REQUIRE((*renderer)->render("/Camera", 0.0, w, h));
    auto matted = (*renderer)->mappedOutput("color");
    REQUIRE(matted);
    CHECK(*plain == *matted);

    std::array<std::vector<uint8_t>, 3> layers;
    for (uint32_t k = 0; k < 3; ++k) {
        auto got = (*renderer)->mappedOutput("CryptoObject0" + std::to_string(k));
        if (!got) FAIL(got.error().toString());
        REQUIRE(got->size() == size_t{w} * h * 16);
        layers[k] = std::move(*got);
    }
    const auto rank = [&](uint32_t layer, uint32_t at, uint32_t which) {
        uint32_t id = 0;
        float coverage = 0.0F;
        std::memcpy(&id, layers[layer].data() + (size_t{at} * 4 + which * 2) * 4, 4);
        std::memcpy(&coverage, layers[layer].data() + (size_t{at} * 4 + which * 2 + 1) * 4, 4);
        return std::pair<uint32_t, float>{id, coverage};
    };
    std::map<uint32_t, uint32_t> pixelsNaming;
    size_t named = 0;
    double worstSum = 0.0;
    for (uint32_t at = 0; at < w * h; ++at) {
        float alpha = 0.0F;
        std::memcpy(&alpha, matted->data() + (size_t{at} * 4 + 3) * 4, 4);
        double sum = 0.0;
        float previous = 3.0e38F;
        for (uint32_t layer = 0; layer < 3; ++layer) {
            for (uint32_t which = 0; which < 2; ++which) {
                const auto [id, coverage] = rank(layer, at, which);
                if (coverage <= 0.0F) {
                    continue;
                }
                CHECK(coverage <= previous + 1.0e-6F);
                previous = coverage;
                sum += coverage;
                ++pixelsNaming[id];
            }
        }
        if (sum > 0.0) {
            ++named;
        }
        worstSum = std::max(worstSum, std::abs(sum - double(alpha)));
    }
    std::printf("  crypto: %zu named pixels of %u, cloud %u, badge %u, plane %u, worst sum error %.2g\n", named,
                w * h, pixelsNaming[idCloud], pixelsNaming[idBadge], pixelsNaming[idPlane], worstSum);
    // The plane fills the frame, so every pixel is named by something.
    CHECK(named == w * h);
    CHECK(worstSum < 2.0e-3);
    // All three prims are named, and nothing else is.
    CHECK(pixelsNaming[idCloud] > 100u);
    CHECK(pixelsNaming[idBadge] > 10u);
    CHECK(pixelsNaming[idPlane] > 1000u);
    CHECK(pixelsNaming.size() == 3);

    // The manifest names every id the frame drew, the cloud's own included.
    const std::map<std::string, uint32_t> manifest = (*renderer)->cryptoManifest();
    CHECK(manifest.at("/World/Plane") == idPlane);
    CHECK(manifest.at("/World/Cloud") == idCloud);
    CHECK(manifest.at("/Source/Badge") == idBadge);

    // AND A PIXEL PICKED. A gaussian writes no prim id, so before the matte a
    // click on a cloud found nothing at all; now the nearest rank names it,
    // and the manifest turns that number into the prim's path.
    auto onCloud = (*renderer)->pick(w / 2, h / 2);
    if (!onCloud) FAIL(onCloud.error().toString());
    REQUIRE(onCloud->has_value());
    std::printf("  picked at the centre: %s (matte %s, %08x)\n", (*onCloud)->prim.c_str(),
                (*onCloud)->cryptoName.c_str(), (*onCloud)->cryptoId);
    CHECK((*onCloud)->cryptoId == idCloud);
    CHECK((*onCloud)->cryptoName == "/World/Cloud");
    CHECK((*onCloud)->cryptoCoverage > 0.1F);
    // A corner is the plane alone: the mesh is named by the same matte, and
    // there it is the prim Hydra picked too.
    auto onPlane = (*renderer)->pick(2, 2);
    if (!onPlane) FAIL(onPlane.error().toString());
    REQUIRE(onPlane->has_value());
    CHECK((*onPlane)->cryptoId == idPlane);
    CHECK((*onPlane)->prim == "/World/Plane");
}

// The same matte through a render product: twelve float channels named as the
// specification names them, and the four attributes a reader needs to turn the
// numbers back into paths.
TEST_CASE("a render product writes the Cryptomatte layers and the manifest beside them",
          "[usd][gpu][crypto][products]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const uint32_t idPlane = core::cryptomatteId("/World/Plane");
    const fs::path path = scratch("crypto_product.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n    renderSettingsPrimPath = \"/Render/Settings\"\n)\n"
               "def Xform \"World\"\n{\n"
               "    def Mesh \"Plane\"\n    {\n"
               "        int[] faceVertexCounts = [4]\n"
               "        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-2, -2, -1), (2, -2, -1), (2, 2, -1), (-2, 2, -1)]\n"
               "        color3f[] primvars:displayColor = [(0.8, 0.3, 0.1)]\n"
               "    }\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 100)\n"
               "    double3 xformOp:translate = (0, 0, 2)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n"
               "def Scope \"Render\"\n{\n"
               "    def RenderSettings \"Settings\"\n    {\n"
               "        rel camera = </Camera>\n"
               "        rel products = </Render/Product>\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def RenderProduct \"Product\"\n    {\n"
               "        token productName = \"crypto.exr\"\n"
               "        rel orderedVars = [</Render/Vars/beauty>, </Render/Vars/CryptoObject00>,"
               " </Render/Vars/CryptoObject01>, </Render/Vars/CryptoObject02>]\n"
               "        int2 resolution = (64, 48)\n"
               "    }\n"
               "    def Scope \"Vars\"\n    {\n"
               "        def RenderVar \"beauty\"\n        {\n            string sourceName = \"Ci\"\n"
               "            token dataType = \"color3f\"\n        }\n"
               "        def RenderVar \"CryptoObject00\"\n        {\n"
               "            string sourceName = \"CryptoObject00\"\n            token dataType = \"color4f\"\n        }\n"
               "        def RenderVar \"CryptoObject01\"\n        {\n"
               "            string sourceName = \"CryptoObject01\"\n            token dataType = \"color4f\"\n        }\n"
               "        def RenderVar \"CryptoObject02\"\n        {\n"
               "            string sourceName = \"CryptoObject02\"\n            token dataType = \"color4f\"\n        }\n"
               "    }\n}\n";
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    auto written = (*renderer)->renderProducts("/Render/Settings", 0.0, path.parent_path());
    if (!written) FAIL(written.error().toString());
    REQUIRE(written->size() == 1);

    auto file = io::readExrChannels(written->front());
    if (!file) FAIL(file.error().toString());
    const auto channel = [&](const std::string& name) -> const io::ExrChannelData* {
        for (const io::ExrChannelData& c : file->channels) {
            if (c.name == name) return &c;
        }
        return nullptr;
    };
    for (uint32_t layer = 0; layer < 3; ++layer) {
        for (const char* component : {"R", "G", "B", "A"}) {
            const std::string name = "CryptoObject0" + std::to_string(layer) + "." + component;
            const io::ExrChannelData* found = channel(name);
            REQUIRE(found != nullptr);
            // Float, never half: an id rounded is another id's name.
            CHECK(found->type == io::ExrChannelType::Float);
        }
    }
    // The nearest rank of the first layer is the plane, covering the pixel whole.
    const io::ExrChannelData* id0 = channel("CryptoObject00.R");
    const io::ExrChannelData* cov0 = channel("CryptoObject00.G");
    REQUIRE(id0 != nullptr);
    REQUIRE(cov0 != nullptr);
    REQUIRE(id0->words.size() == size_t{file->width} * file->height);
    size_t named = 0;
    for (size_t k = 0; k < id0->words.size(); ++k) {
        float coverage = 0.0F;
        std::memcpy(&coverage, &cov0->words[k], 4);
        if (id0->words[k] == idPlane && coverage > 0.99F) {
            ++named;
        }
    }
    std::printf("  crypto.exr: %zu channels, %zu of %zu pixels are the plane covered whole\n",
                file->channels.size(), named, id0->words.size());
    CHECK(named == id0->words.size());

    // And the attributes a Cryptomatte reader looks for, from the same read:
    // a file of layered channels is not one `readExr` can open.
    const std::string key = core::cryptomatteLayerKey("CryptoObject");
    const auto attribute = [&](const std::string& suffix) -> std::string {
        const std::string want = "cryptomatte/" + key + "/" + suffix;
        for (const io::ExrAttribute& a : file->attributes) {
            if (a.name == want && a.type == "string") {
                return std::string(reinterpret_cast<const char*>(a.value.data()), a.value.size());
            }
        }
        return {};
    };
    CHECK(attribute("name").find("CryptoObject") != std::string::npos);
    CHECK(attribute("hash").find("MurmurHash3_32") != std::string::npos);
    CHECK(attribute("conversion").find("uint32_to_float32") != std::string::npos);
    const std::string manifest = attribute("manifest");
    std::printf("  manifest: %s\n", manifest.c_str());
    const auto names = core::parseCryptomatteManifest(manifest);
    CHECK(names.at("/World/Plane") == idPlane);
}

// The ancestry a conversion inherits, written and read back: half the cloud
// from one prim and half from another, the manifest naming both, and the
// frame's own matte holding exactly those two ids.
TEST_CASE("a cloud exports its Cryptomatte ids and the stage gives them back", "[usd][gpu][crypto]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device");
    }
    const uint32_t n = 400;
    io::RawSplats raw = cloud(n);
    const uint32_t idA = core::cryptomatteId("/Source/Body");
    const uint32_t idB = core::cryptomatteId("/Source/Wheel");
    std::vector<uint32_t> ids(n);
    for (uint32_t i = 0; i < n; ++i) {
        ids[i] = i < n / 2 ? idA : idB;
    }
    usd::ExportOptions options;
    options.addCamera = true;
    options.cryptoObject = ids;
    options.cryptoManifest = {{"/Source/Body", idA}, {"/Source/Wheel", idB}};
    const fs::path path = scratch("crypto_export.usda");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));

    // The primvar as USD holds it: one id a record, the two counts kept.
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    UsdPrim splats = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(splats);
    UsdGeomPrimvar written = UsdGeomPrimvarsAPI(splats).GetPrimvar(TfToken("primvars:athenea:splat:cryptoObject"));
    REQUIRE(written);
    VtIntArray back;
    REQUIRE(written.Get(&back));
    REQUIRE(back.size() == n);
    size_t first = 0, second = 0;
    for (const int id : back) {
        first += static_cast<uint32_t>(id) == idA ? 1 : 0;
        second += static_cast<uint32_t>(id) == idB ? 1 : 0;
    }
    CHECK(first == n / 2);
    CHECK(second == n - n / 2);
    std::string manifest;
    REQUIRE(UsdGeomPrimvarsAPI(splats).GetPrimvar(TfToken("primvars:athenea:splat:cryptoManifest")).Get(&manifest));
    const auto names = core::parseCryptomatteManifest(manifest);
    CHECK(names.at("/Source/Body") == idA);
    CHECK(names.at("/Source/Wheel") == idB);

    // And drawn: the frame's matte names those two and nothing else, and the
    // engine's manifest has them beside the cloud's own prim.
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    (*renderer)->requestOutputs({"CryptoObject00", "CryptoObject01", "CryptoObject02"});
    const uint32_t w = 96, h = 72;
    REQUIRE((*renderer)->render("", 0.0, w, h));
    std::set<uint32_t> named;
    size_t covered = 0;
    for (uint32_t layer = 0; layer < 3; ++layer) {
        auto got = (*renderer)->mappedOutput("CryptoObject0" + std::to_string(layer));
        if (!got) FAIL(got.error().toString());
        for (uint32_t at = 0; at < w * h; ++at) {
            for (uint32_t which = 0; which < 2; ++which) {
                uint32_t id = 0;
                float coverage = 0.0F;
                std::memcpy(&id, got->data() + (size_t{at} * 4 + which * 2) * 4, 4);
                std::memcpy(&coverage, got->data() + (size_t{at} * 4 + which * 2 + 1) * 4, 4);
                if (coverage > 0.0F) {
                    named.insert(id);
                    ++covered;
                }
            }
        }
    }
    std::printf("  exported cloud: %zu ranks covered, %zu ids named\n", covered, named.size());
    CHECK(covered > 100);
    CHECK(named == std::set<uint32_t>{idA, idB});
    const auto engineManifest = (*renderer)->cryptoManifest();
    CHECK(engineManifest.at("/Source/Body") == idA);
    CHECK(engineManifest.at("/Source/Wheel") == idB);
}

// A TRANSFER IS NOT THE LIGHT THAT REACHED A POINT BUT HOW MUCH OF AN
// ENVIRONMENT WOULD.
//
// `litBody` keeps the light of the dome that was there when it was baked, so
// under another sky it is wrong. A transfer keeps the geometry instead --
// visibility times the cosine, the surface's own albedo taken as one -- and
// the frame recombines it with whatever sky it is put under. What makes that
// believable is that the unoccluded case has a closed form: the projection of
// a clamped cosine lobe, whose bands are pi, 2pi/3 and pi/4. A plane with
// nothing around it must bake exactly that, and dotted with a constant sky of
// radiance L it must come back as L, because a white Lambert surface under
// such a sky reflects exactly L.
TEST_CASE("an unoccluded point transfers the cosine lobe, and that times a sky is the sky",
          "[usd][gpu][mesh][bake][transfer]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("transfer_plane.usda");
    {
        std::ofstream out(path);
        // The same plane the bake above uses, and the dome is there only so
        // the stage has a light at all: a transfer gathers none.
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
    }
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[1] = 0.0F;
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;   // the normal, +z
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t degree = 2;
    const uint32_t coefficients = (degree + 1) * (degree + 1);
    auto baked = (*renderer)->bakePoints(rays, count, 0.0, 1024, 2, degree, /*transfer=*/true);
    if (!baked) FAIL(baked.error().toString());
    const uint32_t entries = coefficients + 2;   // the coverage, then the open directions
    REQUIRE(baked->size() == size_t{count} * entries * 4);

    gpu::BufferDesc desc;
    desc.bytes = baked->size() * 4;
    desc.elementBytes = 16;
    auto values = gpu::Buffer::create(*gpu->device, desc, baked->data());
    REQUIRE(values);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "transfer.stats");
    const std::array<float, 8> zeros{};
    gpu::BufferDesc worstDesc;
    worstDesc.bytes = sizeof(zeros);
    worstDesc.elementBytes = sizeof(float);
    worstDesc.label = "transfer.worst";
    auto worst = gpu::Buffer::create(*gpu->device, worstDesc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/transfer_check");
    const float skyRadiance = 0.7F;
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["baked"].setBinding(values->rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["entries"].setData(entries);
            p["coefficients"].setData(coefficients);
            p["tolerance"].setData(0.03F);
            const float normal[3] = {0.0F, 0.0F, 1.0F};
            p["normal"].setData(normal, sizeof(normal));
            p["skyRadiance"].setData(skyRadiance);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
    std::printf("  transfer: %u points, %u beyond, %u found nothing; first %.4f / %.4f / %.4f "
                "(want 0.2821 / 0.3257 / 0.1577), sky %.4f (want %.4f); worst %.4f, sky worst %.4f\n",
                counts[1], counts[0], counts[3], double(readings[2]), double(readings[3]),
                double(readings[4]), double(readings[5]), double(skyRadiance),
                double(counts[4]) * 1.0e-6, double(counts[5]) * 1.0e-6);
    CHECK(counts[1] == count);   // every point found its surface
    CHECK(counts[3] == 0);
    CHECK(counts[0] == 0);       // and transfers the cosine lobe
    CHECK(counts[2] == 0);       // and that sky reads as the sky
}

// AND AT DEGREE 3 (task TX, step 3): the clamped cosine lobe's band 3 is
// zero, so an unoccluded point's sixteen coefficients are its first nine and
// seven zeros, in a TX transfer's layout, whose coverage follows the
// coefficients and whose cells and field follow that.
TEST_CASE("an unoccluded point transfers the cosine lobe at degree 3, with nothing in band 3",
          "[usd][gpu][mesh][bake][transfer][degree3]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("transfer_plane_degree3.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -1.5F + 3.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t coefficients = 16;
    auto baked = (*renderer)->bakePoints(rays, count, 0.0, 1024, 2, 3, /*transfer=*/true, nullptr, 16);
    if (!baked) FAIL(baked.error().toString());
    const uint32_t entries = coefficients + technique::transferPlanes(true, 16);
    REQUIRE(baked->size() == size_t{count} * entries * 4);
    gpu::BufferDesc desc;
    desc.bytes = baked->size() * 4;
    desc.elementBytes = 16;
    auto values = gpu::Buffer::create(*gpu->device, desc, baked->data());
    REQUIRE(values);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "transfer3.stats");
    const std::array<float, 8> zeros{};
    gpu::BufferDesc worstDesc;
    worstDesc.bytes = sizeof(zeros);
    worstDesc.elementBytes = sizeof(float);
    auto worst = gpu::Buffer::create(*gpu->device, worstDesc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/transfer_check");
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["baked"].setBinding(values->rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["entries"].setData(entries);
            p["coefficients"].setData(coefficients);
            p["tolerance"].setData(0.03F);
            const float normal[3] = {0.0F, 0.0F, 1.0F};
            p["normal"].setData(normal, sizeof(normal));
            p["skyRadiance"].setData(0.7F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    std::printf("  degree 3: %u points, %u beyond (worst coefficient %.4f), %u found nothing, sky worst %.4f\n",
                counts[1], counts[0], double(counts[4]) * 1.0e-6, counts[3], double(counts[5]) * 1.0e-6);
    CHECK(counts[1] == count);
    CHECK(counts[3] == 0);
    CHECK(counts[0] == 0);
    CHECK(counts[2] == 0);
}

// WHICH WAYS OUT ARE OPEN IS A SHADOW WITH AN EDGE.
//
// The transfer is degree 2: it knows how much of the sky a point sees and
// roughly where from, and a sun taken out of the sky and shadowed by it gets
// a soft falloff where a roof has an edge. So the bake also keeps sixty-four
// bits a point, one traced ray a cell of an 8 x 8 octahedral grid. Under a
// square roof those bits have a closed form -- a line against a square -- and
// every cell the roof does not straddle must read what that says.
TEST_CASE("the open directions a transfer keeps are the ones a roof leaves",
          "[usd][gpu][mesh][bake][transfer][sun]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("transfer_roof.usda");
    const float roofHeight = 1.0F, roofHalf = 0.75F;
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Roof\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 3, 2, 1]\n"
               "    point3f[] points = [(-0.75, -0.75, -0.5), (0.75, -0.75, -0.5), (0.75, 0.75, -0.5), "
               "(-0.75, 0.75, -0.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    // A row of points from under the middle of the roof to well outside it.
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    std::vector<float> where(size_t{count} * 4, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -2.5F + 5.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[1] = 0.1F;
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;   // the normal, +z
        where[size_t{k} * 4] = ray[0];
        where[size_t{k} * 4 + 1] = ray[1];
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t coefficients = 9;
    auto baked = (*renderer)->bakePoints(rays, count, 0.0, 64, 1, 2, /*transfer=*/true);
    if (!baked) FAIL(baked.error().toString());
    const uint32_t entries = coefficients + 2;
    REQUIRE(baked->size() == size_t{count} * entries * 4);

    const auto upload = [&](const std::vector<float>& values, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = values.size() * 4;
        desc.elementBytes = 16;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, values.data());
        REQUIRE(made);
        return *made;
    };
    gpu::Buffer values = upload(*baked, "bits.baked");
    gpu::Buffer points = upload(where, "bits.points");
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "bits.stats");
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/shadow_bits_check");
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["baked"].setBinding(values.rhi());
            cursor["points"].setBinding(points.rhi());
            cursor["stats"].setBinding(stats.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["entries"].setData(entries);
            p["roofHeight"].setData(roofHeight);
            p["roofHalf"].setData(roofHalf);
            const float normal[3] = {0.0F, 0.0F, 1.0F};
            p["normal"].setData(normal, sizeof(normal));
            p["margin"].setData(0.05F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    std::printf("  open directions: %u cells judged, %u open, %u wrong, %u at the edge, "
                "%u below the horizon read open\n",
                counts[1], counts[2], counts[0], counts[4], counts[3]);
    CHECK(counts[1] > count * 20);           // most of each hemisphere was judged
    CHECK(counts[2] > 0);                    // some of it is open
    CHECK(counts[2] < counts[1]);            // and some of it is under the roof
    CHECK(counts[0] == 0);                   // and every bit is what the roof says
    CHECK(counts[3] == 0);                   // nothing was traced below the horizon
}

// AND SIXTEEN OR SIXTY-FOUR TIMES FINER, OVER THE WHOLE SPHERE (task TX).
//
// A TX transfer keeps 256 or 1024 bits a point, one traced ray a cell of a
// 16 x 16 or 32 x 32 octahedral grid, the half below the surface traced as
// well: a glass is looked through and a sheet is seen from behind. The same
// roof, then, and its closed form above the floor; below it the floor is a
// sheet over nothing, and every bit there must read open.
TEST_CASE("the open directions a TX transfer keeps are the ones a roof leaves, over the whole sphere",
          "[usd][gpu][mesh][bake][transfer][cells]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const fs::path path = scratch("transfer_roof_cells.usda");
    const float roofHeight = 1.0F, roofHalf = 0.75F;
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Square\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Roof\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 3, 2, 1]\n"
               "    point3f[] points = [(-0.75, -0.75, -0.5), (0.75, -0.75, -0.5), (0.75, 0.75, -0.5), "
               "(-0.75, 0.75, -0.5)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    const uint32_t count = 64;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    std::vector<float> where(size_t{count} * 4, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -2.5F + 5.0F * (static_cast<float>(k) + 0.5F) / static_cast<float>(count);
        ray[1] = 0.1F;
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
        where[size_t{k} * 4] = ray[0];
        where[size_t{k} * 4 + 1] = ray[1];
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const auto upload = [&](const std::vector<float>& values, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = values.size() * 4;
        desc.elementBytes = 16;
        desc.label = label;
        auto made = gpu::Buffer::create(*gpu->device, desc, values.data());
        REQUIRE(made);
        return *made;
    };
    gpu::Buffer points = upload(where, "cells.points");
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/shadow_cells_check");
    for (const uint32_t side : {16u, 32u}) {
        const uint32_t coefficients = 9;
        auto baked = (*renderer)->bakePoints(rays, count, 0.0, 64, 1, 2, /*transfer=*/true, nullptr, side);
        if (!baked) FAIL(baked.error().toString());
        const uint32_t entries = coefficients + technique::transferPlanes(true, side);
        REQUIRE(entries == coefficients + 1 + side * side / 128 + technique::kTransferFieldPlanes);
        REQUIRE(baked->size() == size_t{count} * entries * 4);
        gpu::Buffer values = upload(*baked, "cells.baked");
        gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "cells.stats");
        {
            gpu::CommandBatch batch(*gpu->device);
            check.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["baked"].setBinding(values.rhi());
                cursor["points"].setBinding(points.rhi());
                cursor["stats"].setBinding(stats.rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["entries"].setData(entries);
                p["firstPlane"].setData(coefficients + 1);
                p["side"].setData(side);
                p["roofHeight"].setData(roofHeight);
                p["roofHalf"].setData(roofHalf);
                p["margin"].setData(0.05F);
                const float normal[3] = {0.0F, 0.0F, 1.0F};
                p["normal"].setData(normal, sizeof(normal));
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 8> counts{};
        REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
        std::printf("  %u x %u cells: %u above judged, %u open, %u wrong; %u below judged, %u of them closed; "
                    "%u left out\n",
                    side, side, counts[1], counts[2], counts[0], counts[5], counts[3], counts[4]);
        INFO(side << " cells a side");
        CHECK(counts[1] > count * side * side / 4);   // most of the upper half was judged
        CHECK(counts[2] > 0);
        CHECK(counts[2] < counts[1]);
        CHECK(counts[0] == 0);                         // every bit is what the roof says
        CHECK(counts[5] > count * side * side / 4);    // the lower half was traced too
        CHECK(counts[3] == 0);                         // and under a sheet it is open
    }
}

// WHAT A TX TRANSFER'S CLOSED DIRECTIONS SHOW IS WHAT STANDS THERE (step 2).
//
// A point on a black floor beside a grey wall, under a white sky. The wall is
// a vertical surface whose upper half of hemisphere is the sky and whose
// lower half is the black floor, so it sends back exactly half its albedo,
// and that is what the field must read toward it; straight up is open, and
// the field -- which holds only what arrived after meeting the scene --
// must read next to nothing there.
TEST_CASE("a TX transfer's reflected field reads the wall that stands beside it", "[usd][gpu][mesh][bake][transfer][field]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const float albedo = 0.5F;
    const fs::path path = scratch("transfer_field_wall.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    color3f[] primvars:displayColor = [(0, 0, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(2, -4, -1.5), (2, -4, 2.5), (2, 4, 2.5), (2, 4, -1.5)]\n"
               "    normal3f[] normals = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)] (interpolation = \"vertex\")\n"
               "    color3f[] primvars:displayColor = [("
            << albedo << ", " << albedo << ", " << albedo << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    // The wall sends back half its albedo where its lower half sees the
    // black floor and a little more higher up, where it sees past the
    // floor's edge to the sky below the horizon.
    const float wallRadiance = albedo * 0.55F;
    // Near the wall, where it fills the half of the sky that faces it.
    const uint32_t count = 32;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = 1.9F - 0.015F * static_cast<float>(k);
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t side = 16;
    auto baked = (*renderer)->bakePoints(rays, count, 0.0, 1024, 3, 2, /*transfer=*/true, nullptr, side);
    if (!baked) FAIL(baked.error().toString());
    const uint32_t entries = 9 + technique::transferPlanes(true, side);
    REQUIRE(baked->size() == size_t{count} * entries * 4);
    gpu::BufferDesc desc;
    desc.bytes = baked->size() * 4;
    desc.elementBytes = 16;
    auto values = gpu::Buffer::create(*gpu->device, desc, baked->data());
    REQUIRE(values);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "field.stats");
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/field_check", "fieldBakeCheck");
    if (!check) FAIL(check.error().toString());
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            for (const char* unused : {"halves", "transfer", "skies", "envSh", "worst"}) {
                cursor[unused].setBinding(stats.rhi());
            }
            cursor["baked"].setBinding(values->rhi());
            cursor["stats"].setBinding(stats.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["entries"].setData(entries);
            p["firstField"].setData(entries - technique::kTransferFieldPlanes);
            p["wallRadiance"].setData(wallRadiance);
            p["tolerance"].setData(0.3F);
            p["openFloor"].setData(wallRadiance * 0.3F);
            p["roughness"].setData(0.0F);
            const float towards[3] = {1.0F, 0.0F, 0.3F};
            // The open side is read away from the wall: straight up lies on
            // the edge of the wall's quarter of the sphere, where degree 3
            // reads half the step, as any truncated series does at a step.
            const float away[3] = {-1.0F, 0.0F, 0.3F};
            p["awayFromWall"].setData(away, sizeof(away));
            p["towardsWall"].setData(towards, sizeof(towards));
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    float worstWall = 0.0F, mostUp = 0.0F, worstFilled = 0.0F;
    std::memcpy(&worstWall, &counts[4], 4);
    std::memcpy(&mostUp, &counts[5], 4);
    std::memcpy(&worstFilled, &counts[6], 4);
    std::printf("  the field beside a wall: %u points, %u off the wall's %.3f (worst %.3f relative), %u reading "
                "the open side (most %.4f); filled over the closed directions, %u off (worst %.3f)\n",
                counts[2], counts[0], double(wallRadiance), double(worstWall), counts[1], double(mostUp), counts[3],
                double(worstFilled));
    CHECK(counts[2] == count);
    CHECK(counts[1] == 0);
    // What the file keeps is the field filled over the closed directions
    // (m2sFieldOverClosed), and that is what is held to the wall. The bake's
    // own projection is printed beside it: a closed direction this near the
    // wall's edge reads it diluted by the open side, 0.317 off at worst where
    // the filled field is 0.284 off.
    CHECK(counts[3] == 0);
}

// WHAT A TX TRANSFER'S GLASS SEES THROUGH ITSELF (step 5).
//
// A pane of clear glass over a grey floor, under a white sky. A transmitting
// gaussian's first directions are drawn over the whole sphere, so its field
// holds what stands behind it: straight down, the floor, which the sky lights
// through the pane and which sends back about half; straight up, the open
// sky, where the field holds next to nothing.
TEST_CASE("a TX transfer's glass holds what stands behind it", "[usd][gpu][mesh][bake][transfer][field][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    const float albedo = 0.5F;
    const fs::path path = scratch("transfer_field_pane.usda");
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-20, -20, -2.5), (20, -20, -2.5), (20, 20, -2.5), (-20, 20, -2.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    color3f[] primvars:displayColor = [("
            << albedo << ", " << albedo << ", " << albedo << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def Mesh \"Pane\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, -1.5), (2, -2, -1.5), (2, 2, -1.5), (-2, 2, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Glass>\n}\n"
               "def Material \"Glass\"\n{\n"
               "    token outputs:mtlx:surface.connect = </Glass/OpenPBR.outputs:out>\n"
               "    def Shader \"OpenPBR\"\n    {\n"
               "        uniform token info:id = \"ND_open_pbr_surface_surfaceshader\"\n"
               "        float inputs:specular_roughness = 0\n        float inputs:specular_ior = 1.45\n"
               "        float inputs:transmission_weight = 1\n        token outputs:out\n    }\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n";
    }
    const uint32_t count = 32;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = -0.8F + 0.05F * static_cast<float>(k);
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    const uint32_t side = 16;
    auto baked = (*renderer)->bakePoints(rays, count, 0.0, 1024, 3, 2, /*transfer=*/true, nullptr, side);
    if (!baked) FAIL(baked.error().toString());
    const uint32_t entries = 9 + technique::transferPlanes(true, side);
    REQUIRE(baked->size() == size_t{count} * entries * 4);
    gpu::BufferDesc desc;
    desc.bytes = baked->size() * 4;
    desc.elementBytes = 16;
    auto values = gpu::Buffer::create(*gpu->device, desc, baked->data());
    REQUIRE(values);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "pane.stats");
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/field_check", "fieldBakeCheck");
    if (!check) FAIL(check.error().toString());
    // What the floor sends back: its albedo times the white sky through the
    // pane (a clear dielectric passes about 0.92 at the angles that matter
    // most) over its upper hemisphere, where the pane is a small part.
    const float floorRadiance = albedo * 0.95F;
    {
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            for (const char* unused : {"halves", "transfer", "skies", "envSh", "worst"}) {
                cursor[unused].setBinding(stats.rhi());
            }
            cursor["baked"].setBinding(values->rhi());
            cursor["stats"].setBinding(stats.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(count);
            p["entries"].setData(entries);
            p["firstField"].setData(entries - technique::kTransferFieldPlanes);
            p["wallRadiance"].setData(floorRadiance);
            p["tolerance"].setData(0.3F);
            p["openFloor"].setData(floorRadiance * 0.3F);
            p["roughness"].setData(0.0F);
            const float towards[3] = {0.0F, 0.0F, -1.0F};
            p["towardsWall"].setData(towards, sizeof(towards));
            const float away[3] = {0.0F, 0.0F, 1.0F};
            p["awayFromWall"].setData(away, sizeof(away));
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    float worstFloor = 0.0F, mostUp = 0.0F;
    std::memcpy(&worstFloor, &counts[4], 4);
    std::memcpy(&mostUp, &counts[5], 4);
    std::printf("  the field of a glass pane: %u points, %u off the floor's %.3f below (worst %.3f relative), %u "
                "reading the open sky above (most %.4f)\n",
                counts[2], counts[0], double(floorRadiance), double(worstFloor), counts[1], double(mostUp));
    CHECK(counts[2] == count);
    CHECK(counts[0] == 0);
    CHECK(counts[1] == 0);
}

// THE TEST THAT SAYS THE WHOLE CHAIN IS LINEAR.
//
// A transfer is worth having only if recombining it with a sky gives what
// baking under that sky would have given. The two keep different things -- a
// radiance bake the light that arrived, a transfer how much of an environment
// would -- so what is compared is their product: under a sky of constant
// radiance L, a Lambert surface of albedo `a` reads `a * L` either way.
//
// It is done twice, and that is the point of doing it twice: at one bounce,
// where only the direct half can be involved, and at three, where the
// indirect half has to make up the difference. A transfer right in its direct
// half and wrong in its indirect one passes the first and fails the second.
TEST_CASE("a transfer recombined with a sky is what a bake under that sky reads",
          "[usd][gpu][mesh][bake][transfer]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.accelerationStructure || !(caps.rayQuery || caps.rayTracing)) {
        SKIP("needs ray tracing");
    }
    // A floor and a wall at a right angle, and the wall's colour is what the
    // two cases differ in. Black, it occludes and sends nothing back, so a
    // bake at one bounce is direct light alone and the transfer's direct half
    // must be all of it. White, it sends plenty back, and only the indirect
    // half can account for the difference.
    const float albedo = 0.6F;
    const auto corner = [&](const char* name, float wall, float wallX = 2.0F) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n"
               "def Mesh \"Floor\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-4, -4, -1.5), (4, -4, -1.5), (4, 4, -1.5), (-4, 4, -1.5)]\n"
               "    normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (interpolation = \"vertex\")\n"
               "    color3f[] primvars:displayColor = [("
            << albedo << ", " << albedo << ", " << albedo << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               // Wound with its front face towards the floor: a nearest trace
               // culls back faces where a shadow ray does not, and a wall the
               // bounce rays pass through occludes the bake and not the
               // transfer, which is exactly how this was first written.
               "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [("
            << wallX << ", -4, -1.5), (" << wallX << ", -4, 2.5), (" << wallX << ", 4, 2.5), ("
            << wallX << ", 4, -1.5)]\n"
               "    normal3f[] normals = [(-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)] (interpolation = \"vertex\")\n"
               "    color3f[] primvars:displayColor = [("
            << wall << ", " << wall << ", " << wall << ")]\n"
               "    uniform token subdivisionScheme = \"none\"\n}\n"
               "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 1\n}\n"
               "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
               "    float2 clippingRange = (0.1, 1000)\n}\n";
        return path;
    };
    const fs::path dark = corner("transfer_corner_dark.usda", 0.0F);
    const fs::path bright = corner("transfer_corner.usda", 1.0F);
    // And the same corner with its wall taken away, which is the case that
    // says the two agree when nothing is in the way at all.
    const fs::path clear = corner("transfer_corner_clear.usda", 0.0F, 200.0F);
    const uint32_t count = 32;
    std::vector<float> rays(size_t{count} * 8, 0.0F);
    for (uint32_t k = 0; k < count; ++k) {
        float* ray = rays.data() + size_t{k} * 8;
        ray[0] = 1.9F - 0.05F * static_cast<float>(k);   // along the floor, near the wall
        ray[1] = 0.0F;
        ray[2] = -1.5F;
        ray[3] = 1.0e-3F;
        ray[6] = 1.0F;
    }
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/transfer_check");
    auto against = gpu::ComputeKernel::create(*gpu->library, "athenea/test/transfer_check", "transferAgainstBake");
    if (!against) FAIL(against.error().toString());
    const uint32_t paths = 4096;
    // A BOUNCE OF THE ONE IS NOT A BOUNCE OF THE OTHER.
    //
    // The radiance bake gathers light at every vertex it shades, so a path
    // allowed one bounce already carries what the wall sent back. A transfer
    // pays only where a path escapes, and a path that ends absorbed on the
    // wall pays nothing at all: what the bake collects at depth d is what the
    // transfer collects on escaping at depth d + 1. So the two are compared
    // one bounce apart -- a bake with no bounce against the transfer's direct
    // half alone, and then the whole chain.
    struct Case {
        const fs::path& stage;
        uint32_t        bounces;
        bool            indirect;
        const char*     what;
    };
    for (const Case one : {Case{clear, 1u, false, "no wall"}, Case{dark, 1u, false, "a black wall"},
                           Case{bright, 3u, true, "a white wall"}}) {
        const uint32_t bounces = one.bounces;
        auto renderer = usd::StageRenderer::open(one.stage);
        if (!renderer) FAIL(renderer.error().toString());
        auto radiance = (*renderer)->bakePoints(rays, count, 0.0, paths, bounces, 0);
        if (!radiance) FAIL(radiance.error().toString());
        auto transfer = (*renderer)->bakePoints(rays, count, 0.0, paths, bounces, 2, /*transfer=*/true);
        if (!transfer) FAIL(transfer.error().toString());
        const uint32_t entries = 11;   // nine, the coverage, the open directions
        REQUIRE(radiance->size() == size_t{count} * 4);
        REQUIRE(transfer->size() == size_t{count} * entries * 4);

        const auto upload = [&](const std::vector<float>& values, const char* label) {
            gpu::BufferDesc desc;
            desc.bytes = values.size() * 4;
            desc.elementBytes = 16;
            desc.label = label;
            auto made = gpu::Buffer::create(*gpu->device, desc, values.data());
            REQUIRE(made);
            return *made;
        };
        gpu::Buffer transferBuffer = upload(*transfer, "transfer.values");
        gpu::Buffer radianceBuffer = upload(*radiance, "transfer.radiance");
        gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "transfer.stats");
        const std::array<float, 8> zeros{};
        gpu::BufferDesc worstDesc;
        worstDesc.bytes = sizeof(zeros);
        worstDesc.elementBytes = sizeof(float);
        worstDesc.label = "transfer.worst";
        auto worst = gpu::Buffer::create(*gpu->device, worstDesc, zeros.data());
        REQUIRE(worst);
        {
            // What the transfer itself reads, for the report below: an
            // occluded point must read less than the 0.2821 of an open one.
            gpu::CommandBatch peek(*gpu->device);
            // Its own counters: sharing the comparison's would count every
            // point twice and the report would read 64 of 32.
            gpu::Buffer peekStats = test::uintBuffer(*gpu->device, 8, "transfer.peek.stats");
            check.dispatch(peek, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["baked"].setBinding(transferBuffer.rhi());
                cursor["stats"].setBinding(peekStats.rhi());
                cursor["worst"].setBinding(worst->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(count);
                p["entries"].setData(entries);
                p["coefficients"].setData(uint32_t{9});
                p["tolerance"].setData(1.0e6F);
                const float normal[3] = {0.0F, 0.0F, 1.0F};
                p["normal"].setData(normal, sizeof(normal));
                p["skyRadiance"].setData(1.0F);
            });
            REQUIRE(peek.submit(true));
            std::array<float, 8> peeked{};
            std::array<uint32_t, 8> peekCounts{};
            REQUIRE(worst->read(*gpu->device, 0, sizeof(peeked), peeked.data()));
            REQUIRE(peekStats.read(*gpu->device, 0, sizeof(peekCounts), peekCounts.data()));
            std::printf("    the transfer's own reading at the first point: %.4f, where an open sky "
                        "would be 0.2821; the farthest any point is from open: %.4f\n",
                        double(peeked[2]), double(peekCounts[4]) * 1.0e-6);
        }
        {
            gpu::CommandBatch batch(*gpu->device);
            against->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["baked"].setBinding(transferBuffer.rhi());
                cursor["radianceBaked"].setBinding(radianceBuffer.rhi());
                cursor["stats"].setBinding(stats.rhi());
                cursor["worst"].setBinding(worst->rhi());
                rhi::ShaderCursor p = cursor["bakeParams"];
                p["count"].setData(count);
                p["entries"].setData(entries);
                p["albedo"].setData(albedo);
                p["skyRadiance"].setData(1.0F);
                // A tenth: one side is a fit of radiance and the other a
                // projection of geometry, and at four thousand paths each
                // carries its own noise. The worst point of the three cases
                // is 4.3% out, and it was a third before the fit was taken
                // out of the encoded space (PathTracer.cpp says what that
                // was doing to an occluded point).
                p["tolerance"].setData(0.10F);
                p["withIndirect"].setData(uint32_t{one.indirect ? 1u : 0u});
            });
            REQUIRE(batch.submit(true));
        }
        std::array<uint32_t, 8> counts{};
        std::array<float, 8> readings{};
        REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
        REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
        std::printf("  %s, %u bounce(s): the transfer reads %.4f where the bake reads %.4f; %u of %u "
                    "points beyond a tenth, worst %.1f%%\n", one.what, bounces,
                    double(readings[1]), double(readings[2]),
                    counts[0], counts[1], double(counts[6]) * 1.0e-4);
        CHECK(counts[1] > 0);
        CHECK(counts[0] == 0);
    }
}

namespace {

/// Gives every record of `raw` the shading normal record `i % 5` of the
/// table splat_normal_check.slang keeps (`tableNormal`): three floats more a
/// record, as `athenea mesh2splat` writes the one its normal map turned.
io::RawSplats withTableNormals(const io::RawSplats& raw) {
    static const float kTable[5][3] = {{0.0F, 0.0F, 1.0F}, {0.6F, 0.0F, 0.8F}, {0.0F, -0.6F, 0.8F},
                                       {-0.48F, 0.6F, 0.64F}, {0.0F, 0.8F, -0.6F}};
    io::RawSplats out = raw;
    const uint32_t stride = raw.encoding.floatsPerRecord;
    out.records.clear();
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* from = raw.records.data() + size_t{i} * stride;
        out.records.insert(out.records.end(), from, from + stride);
        out.records.insert(out.records.end(), kTable[i % 5], kTable[i % 5] + 3);
    }
    out.encoding.floatsPerRecord = stride + 3;
    out.encoding.normal = stride;
    return out;
}

}   // namespace

// A SHADING NORMAL GOES OUT AND COMES BACK.
//
// A converted gaussian keeps the normal its mesh's normal map gave it, apart
// from its frame: `primvars:athenea:splat:normal` (normal3f, declared by
// AtheneaSplatLightingAPI) in a stage, a word a splat in `GpuSplats::normals`,
// and a word an element of every block in a `.athc`. Each way back must give
// the normals that went out: the stage read as records
// (`readParticleFieldRecords`), the stage read as Hydra reads it (its arrays
// interleaved on the device, `splatStreams`), and the levels of detail written
// to a file and read again. And a `.athc` of the version before normals, which
// has none, still reads. Compared on the device; counters come back.
TEST_CASE("a cloud's shading normals survive USD and .athc, and an old .athc still reads",
          "[usd][gpu][export][lod][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalsAgainstTable");
    if (!check) FAIL(check.error().toString());
    const io::RawSplats raw = withTableNormals(cloud(3000));
    auto direct = loader->upload(raw, 0);
    REQUIRE(direct);
    REQUIRE(direct->hasNormals());
    // Validation drops none of these, so splat i is record i and the table
    // says what each should hold.
    REQUIRE(direct->count == raw.count);

    gpu::BufferDesc countsDesc;
    countsDesc.bytes = 8 * 4;
    countsDesc.elementBytes = 4;
    countsDesc.label = "normals.counts";
    auto counts = gpu::Buffer::create(*gpu->device, countsDesc);
    REQUIRE(counts);
    const auto against = [&](const gpu::Buffer& a, const gpu::Buffer& b, uint32_t count) {
        const uint32_t zero[8] = {};
        REQUIRE(counts->write(*gpu->device, 0, sizeof(zero), zero));
        gpu::CommandBatch batch(*gpu->device);
        check->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["normalsA"].setBinding(a.rhi());
            cursor["normalsB"].setBinding(b.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["count"].setData(count);
            // Sixteen bits a component of the octahedral square, and a float
            // normalised on the way: a hundred-thousandth is generous.
            cursor["params"]["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 3> seen{};
        REQUIRE(counts->read(*gpu->device, 0, sizeof(seen), seen.data()));
        return seen;
    };
    const auto self = against(direct->normals, direct->normals, direct->count);
    CHECK(self[0] == raw.count);
    CHECK(self[1] == 0);

    const fs::path path = scratch("normals.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(prim);
    CHECK(prim.HasAPI(TfToken("AtheneaSplatLightingAPI")));
    const UsdGeomPrimvar written = UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken("athenea:splat:normal"));
    REQUIRE(written);
    CHECK(written.GetTypeName() == SdfValueTypeNames->Normal3fArray);
    CHECK(written.GetInterpolation() == UsdGeomTokens->vertex);
    // Declared by the schema, so it is not a custom attribute.
    CHECK_FALSE(written.GetAttr().IsCustom());

    SECTION("read back as records") {
        auto records = usd::readParticleFieldRecords(path);
        REQUIRE(records);
        REQUIRE(records->encoding.normal != io::SplatEncoding::kNoField);
        auto back = loader->upload(*records, 0);
        REQUIRE(back);
        REQUIRE(back->hasNormals());
        REQUIRE(back->count == direct->count);
        const auto seen = against(back->normals, direct->normals, back->count);
        std::printf("  through records: %u compared, %u off the table, %u apart from the cloud written\n", seen[0],
                    seen[1], seen[2]);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }

    SECTION("read back as Hydra reads it") {
        const UsdVolParticleField3DGaussianSplat field(prim);
        usd::ParticleFieldArrays arrays;
        field.GetPositionsAttr().Get(&arrays.positions);
        field.GetOrientationsAttr().Get(&arrays.orientations);
        field.GetScalesAttr().Get(&arrays.scales);
        field.GetOpacitiesAttr().Get(&arrays.opacities);
        int degree = 0;
        field.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree);
        arrays.shDegree = degree;
        field.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&arrays.shCoefficients);
        written.Get(&arrays.normals);
        const scene::SplatStreams streams = usd::splatStreams(arrays, "normals streams");
        REQUIRE_FALSE(streams.normals.empty());
        auto back = loader->upload(streams, 0);
        REQUIRE(back);
        REQUIRE(back->hasNormals());
        REQUIRE(back->count == direct->count);
        const auto seen = against(back->normals, direct->normals, back->count);
        std::printf("  through Hydra's arrays: %u compared, %u off the table, %u apart from the cloud written\n",
                    seen[0], seen[1], seen[2]);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }

    SECTION("through a .athc, and a version 1 file without them") {
        auto builder = lod::LodBuilder::create(*gpu->library);
        REQUIRE(builder);
        lod::LodBuildSettings chunked;
        chunked.chunkSplats = 1000;
        auto built = builder->build(*direct, chunked);
        if (!built) FAIL(built.error().toString());
        REQUIRE(built->splats.hasNormals());
        const fs::path file = scratch("normals.athc");
        REQUIRE(lod::writeAthc(*gpu->device, *built, file));
        // The header's version and flags words: version 2 where a block
        // carries anything besides its four arrays, and only there.
        const auto header = [](const fs::path& path) {
            std::ifstream in(path, std::ios::binary);
            std::array<char, 80> bytes{};
            in.read(bytes.data(), bytes.size());
            uint32_t version = 0, flags = 0;
            std::memcpy(&version, bytes.data() + 4, 4);
            std::memcpy(&flags, bytes.data() + 76, 4);
            return std::pair{version, flags};
        };
        CHECK(header(file) == std::pair{2u, 1u});
        auto read = lod::readAthc(*gpu->device, file);
        if (!read) FAIL(read.error().toString());
        REQUIRE(read->splats.hasNormals());
        REQUIRE(read->levels.size() == built->levels.size());
        // The words themselves: the file holds what the device held.
        auto store = render::countDifferent(*gpu->library, read->splats.normals, built->splats.normals, built->count);
        REQUIRE(store);
        uint64_t levelsApart = 0;
        for (size_t l = 0; l < built->levels.size(); ++l) {
            REQUIRE(read->levels[l].gaussians.hasNormals());
            auto apart = render::countDifferent(*gpu->library, read->levels[l].gaussians.normals,
                                                built->levels[l].gaussians.normals,
                                                built->levels[l].gaussians.count);
            REQUIRE(apart);
            levelsApart += *apart;
        }
        std::printf("  through a .athc: %llu of %u splat normals and %llu merged ones changed\n",
                    static_cast<unsigned long long>(*store), built->count,
                    static_cast<unsigned long long>(levelsApart));
        CHECK(*store == 0);
        CHECK(levelsApart == 0);

        // A cloud with none writes the file the version before wrote, version
        // number included, so a reader of version 1 alone still opens it.
        auto plain = loader->upload(cloud(3000), 0);
        REQUIRE(plain);
        REQUIRE_FALSE(plain->hasNormals());
        auto plainLod = builder->build(*plain, chunked);
        if (!plainLod) FAIL(plainLod.error().toString());
        const fs::path old = scratch("normals_v1.athc");
        REQUIRE(lod::writeAthc(*gpu->device, *plainLod, old));
        CHECK(header(old) == std::pair{1u, 0u});
        auto oldRead = lod::readAthc(*gpu->device, old);
        if (!oldRead) FAIL(oldRead.error().toString());
        CHECK_FALSE(oldRead->splats.hasNormals());
        CHECK(oldRead->count == plainLod->count);
        auto oldStore = render::countDifferent(*gpu->library, oldRead->splats.shape, plainLod->splats.shape,
                                               plainLod->count * 4);
        REQUIRE(oldStore);
        CHECK(*oldStore == 0);
    }
}

namespace {

/// THE CARD AND THE QUAD the shading normal tests draw: a 4 x 4 square at
/// z = 0, as a relit cloud (`card`) of gaussians a cell wide and flat or as a
/// mesh of a grey diffuse paint, carrying the normal `kTiltedNormal` where
/// `tilted`, under a prim scaled `scaleX` in x where that is not one, lit by
/// a distant light turned `lightDegrees` about y, seen from z = 3.
const char* const kTiltedNormal = "(0.573576, 0, 0.819152)";   // 35 degrees towards +x

fs::path tiltedCardStage(const char* name, bool card, bool tilted, float scaleX = 1.0F,
                         float lightDegrees = 40.0F) {
    const char* kTilted = kTiltedNormal;
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
    // Under a prim of that scale, where it is not one.
    const bool scaled = scaleX != 1.0F;
    if (scaled) {
        out << "def Xform \"Scaled\"\n{\n    float3 xformOp:scale = (" << scaleX << ", 1, 1)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:scale\"]\n";
    }
    if (card) {
        // A gaussian a cell, sigma a cell wide and flat, its albedo the
        // mesh's 0.5 kept as a cloud keeps a colour (encoded, so
        // 0.5 + SH0 * dc = 0.735357), relit.
        const int side = 160;
        const double cell = 4.0 / side;
        out << "def ParticleField3DGaussianSplat \"Card\"\n{\n    point3f[] positions = [";
        for (int k = 0; k < side * side; ++k) {
            out << (k ? ", " : "") << "(" << (-2.0 + ((k % side) + 0.5) * cell) << ", "
                << (-2.0 + ((k / side) + 0.5) * cell) << ", 0)";
        }
        out << "]\n    quatf[] orientations = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
        out << "]\n    float3[] scales = [";
        for (int k = 0; k < side * side; ++k) {
            out << (k ? ", " : "") << "(" << cell << ", " << cell << ", " << 1.0e-4 * cell << ")";
        }
        out << "]\n    float[] opacities = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "0.99";
        out << "]\n    uniform int radiance:sphericalHarmonicsDegree = 0\n"
               "    float3[] radiance:sphericalHarmonicsCoefficients = [";
        for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << "(0.834321, 0.834321, 0.834321)";
        out << "]\n    bool primvars:athenea:splat:relight = 1\n";
        if (tilted) {
            out << "    normal3f[] primvars:athenea:splat:normal = [";
            for (int k = 0; k < side * side; ++k) out << (k ? ", " : "") << kTilted;
            out << "] (\n        interpolation = \"vertex\"\n    )\n";
        }
        out << "}\n";
        if (scaled) out << "}\n";
    } else {
        out << "def Mesh \"Quad\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
               "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]\n";
        if (tilted) {
            out << "    normal3f[] normals = [" << kTilted << ", " << kTilted << ", " << kTilted << ", "
                << kTilted << "] (\n        interpolation = \"vertex\"\n    )\n";
        }
        out << "    uniform token subdivisionScheme = \"none\"\n"
               "    rel material:binding = </Looks/Paint>\n}\n";
        if (scaled) out << "}\n";
        out << "def Scope \"Looks\"\n{\n    def Material \"Paint\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Paint/Surface.outputs:surface>\n"
               "        def Shader \"Surface\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:diffuseColor = (0.5, 0.5, 0.5)\n"
               "            float inputs:roughness = 1\n"
               "            float inputs:metallic = 0\n"
               "            token outputs:surface\n        }\n    }\n}\n";
    }
    out << "def DistantLight \"Sun\"\n{\n    bool inputs:normalize = 1\n    float inputs:intensity = 2\n"
           "    bool inputs:shadow:enable = 0\n"
           "    float3 xformOp:rotateXYZ = (0, " << lightDegrees << ", 0)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n"
           "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
           "    float2 clippingRange = (0.1, 1000)\n"
           "    double3 xformOp:translate = (0, 0, 3)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    return path;
}

}   // namespace

// A RELIT CONVERSION KEEPS THE RELIEF ITS NORMAL MAP DREW.
//
// mesh2splat stands each gaussian on its triangle, its disc's axis the face's
// own normal, and works out the normal the normal map turns that into. Until
// a cloud kept that normal, relighting took the disc's axis and the relief was
// gone: a quad whose map tilts every normal one way rendered relit like a quad
// with no map at all. Here the quad is a mesh whose normals are tilted --
// which is what a constant normal map makes of every point -- and a card of
// gaussians laid out as mesh2splat lays one out, flat, carrying that tilted
// normal; a distant light comes from the side the tilt leans to. The card
// must render like the tilted mesh, and not like the flat one, on both routes.
TEST_CASE("a relit card with a tilted shading normal renders like the tilted mesh, not the flat one",
          "[usd][gpu][splat][relight][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // A tilt of 35 degrees towards +x, and a light from 40 degrees that way:
    // the cosine is 0.996 with the tilt and 0.766 without.
    const auto stage = [&](const char* name, bool card, bool tilted) { return tiltedCardStage(name, card, tilted); };
    const fs::path meshTilted = stage("normals_mesh_tilted.usda", false, true);
    const fs::path meshFlat = stage("normals_mesh_flat.usda", false, false);
    const fs::path cardTilted = stage("normals_card_tilted.usda", true, true);
    const fs::path cardFlat = stage("normals_card_flat.usda", true, false);
    const uint32_t w = 160, h = 160;
    for (const char* technique : {"raster", "rt"}) {
        const auto draw = [&](const fs::path& path) {
            auto renderer = usd::StageRenderer::open(path);
            if (!renderer) FAIL(renderer.error().toString());
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        const gpu::Buffer mt = draw(meshTilted);
        const gpu::Buffer mf = draw(meshFlat);
        const gpu::Buffer ct = draw(cardTilted);
        const gpu::Buffer cf = draw(cardFlat);
        const auto hdr = [&](const gpu::Buffer& a, const gpu::Buffer& b) {
            auto diff = render::compareHdr(*gpu->library, a, b, w, h);
            REQUIRE(diff);
            return *diff;
        };
        const auto tiltedPair = hdr(ct, mt);
        const auto flatPair = hdr(cf, mf);
        const auto tiltedAgainstFlat = hdr(ct, mf);
        std::printf("  %s: the tilted card against the tilted mesh p99 %.3f relMSE %.2e; the flat pair p99 %.3f; "
                    "the tilted card against the flat mesh p99 %.3f\n",
                    technique, tiltedPair.p99Relative, tiltedPair.relMse, flatPair.p99Relative,
                    tiltedAgainstFlat.p99Relative);
        // The card is the mesh it came from, tilted or not, to what a cloud
        // of discs can be ...
        CHECK(tiltedPair.p99Relative < 0.08);
        CHECK(flatPair.p99Relative < 0.08);
        // ... and the tilt is what tells the two meshes apart.
        CHECK(tiltedAgainstFlat.p99Relative > 0.2);
    }
}

// A STORED NORMAL GOES TO THE WORLD AS A NORMAL.
//
// A prim scaled (2, 1, 1) stretches its card in x, and a normal tilted 35
// degrees towards +x on it leans 19 degrees in the world -- the inverse
// transpose, which is what the mesh's normals take -- where turned by the
// rows as a direction it leant 54. Lit from straight above, that is a cosine
// of 0.94 against 0.58: the stretched card must render like the stretched
// mesh on both routes.
TEST_CASE("a relit card's stored normal under a scale that is not uniform leans as the scaled mesh's",
          "[usd][gpu][splat][relight][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path mesh = tiltedCardStage("normals_mesh_stretched.usda", false, true, 2.0F, 0.0F);
    const fs::path card = tiltedCardStage("normals_card_stretched.usda", true, true, 2.0F, 0.0F);
    const uint32_t w = 160, h = 160;
    for (const char* technique : {"raster", "rt"}) {
        const auto draw = [&](const fs::path& path) {
            auto renderer = usd::StageRenderer::open(path);
            if (!renderer) FAIL(renderer.error().toString());
            auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
            if (!image) FAIL(image.error().toString());
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        const gpu::Buffer m = draw(mesh);
        const gpu::Buffer c = draw(card);
        auto diff = render::compareHdr(*gpu->library, c, m, w, h);
        REQUIRE(diff);
        std::printf("  %s: the stretched card against the stretched mesh p99 %.3f relMSE %.2e\n", technique,
                    diff->p99Relative, diff->relMse);
        CHECK(diff->pixels == uint64_t{w} * h);
        // 0.014 on both routes; 0.386 turned as a direction (a cosine of 0.58
        // where the mesh is 0.94).
        CHECK(diff->p99Relative < 0.04);
    }
}

// THE LIGHT A MATERIAL GIVES OFF IS READ IN ITS OWN WORDS.
//
// Four vocabularies say it four ways, each with its own defaults:
// standard_surface's `emission` times `emission_color` (white, weighed by
// nothing), OpenPBR's `emission_luminance` times `emission_color` (nits,
// which its graph multiplies in as they stand), glTF's `emissive` times
// `emissive_strength` (black, weighed by one), and UsdPreviewSurface's
// `emissiveColor` alone. A map on the colour is the colour and leaves the
// weight to multiply it; a map on the weight is read on one channel and
// leaves the colour; a map on something that gives off nothing is nothing.
TEST_CASE("a material's emission is read in each vocabulary, with its map", "[usd][mesh][materials][emission]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("splat_emission_read.usda");
    const char* names[] = {"Standard", "Open", "Gltf", "Preview", "Dark", "GltfMapped", "WeightMapped",
                           "PreviewMapped", "MappedDark"};
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        for (const char* name : names) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                   "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                << "    rel material:binding = </Looks/" << name << ">\n}\n";
        }
        const auto mx = [&out](const char* name, const char* id, const std::string& inputs) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:mtlx:surface.connect = </Looks/" << name << "/S.outputs:out>\n"
                << "        def Shader \"S\"\n        {\n"
                << "            uniform token info:id = \"" << id << "\"\n" << inputs
                << "            token outputs:out\n        }\n";
        };
        const auto image = [&out](const char* material, const char* type) {
            out << "        def Shader \"Map\"\n        {\n"
                << "            uniform token info:id = \"ND_image_" << type << "\"\n"
                << "            asset inputs:file = @./lamp_" << material << ".png@\n"
                << "            " << (std::string(type) == "float" ? "float" : "color3f") << " outputs:out\n"
                << "        }\n";
        };
        out << "def Scope \"Looks\"\n{\n";
        mx("Standard", "ND_standard_surface_surfaceshader",
           "            float inputs:emission = 2\n            color3f inputs:emission_color = (1, 0.5, 0.25)\n");
        out << "    }\n";
        mx("Open", "ND_open_pbr_surface_surfaceshader",
           "            float inputs:emission_luminance = 3\n            color3f inputs:emission_color = (0.5, 1, 0)\n");
        out << "    }\n";
        mx("Gltf", "ND_gltf_pbr_surfaceshader",
           "            color3f inputs:emissive = (0.2, 0.4, 0.8)\n            float inputs:emissive_strength = 5\n");
        out << "    }\n";
        // standard_surface with a colour and no weight gives off nothing.
        mx("Dark", "ND_standard_surface_surfaceshader", "            color3f inputs:emission_color = (1, 1, 1)\n");
        out << "    }\n";
        mx("GltfMapped", "ND_gltf_pbr_surfaceshader",
           "            color3f inputs:emissive.connect = </Looks/GltfMapped/Map.outputs:out>\n"
           "            float inputs:emissive_strength = 4\n");
        image("GltfMapped", "color3");
        out << "    }\n";
        mx("WeightMapped", "ND_standard_surface_surfaceshader",
           "            float inputs:emission.connect = </Looks/WeightMapped/Map.outputs:out>\n"
           "            color3f inputs:emission_color = (0.25, 0.5, 1)\n");
        image("WeightMapped", "float");
        out << "    }\n";
        // A map on a weight of nothing: nothing.
        mx("MappedDark", "ND_standard_surface_surfaceshader",
           "            color3f inputs:emission_color.connect = </Looks/MappedDark/Map.outputs:out>\n");
        image("MappedDark", "color3");
        out << "    }\n";
        out << "    def Material \"Preview\"\n    {\n"
               "        token outputs:surface.connect = </Looks/Preview/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:emissiveColor = (1.5, 0.5, 0)\n"
               "            token outputs:surface\n        }\n    }\n"
               "    def Material \"PreviewMapped\"\n    {\n"
               "        token outputs:surface.connect = </Looks/PreviewMapped/S.outputs:surface>\n"
               "        def Shader \"S\"\n        {\n"
               "            uniform token info:id = \"UsdPreviewSurface\"\n"
               "            color3f inputs:emissiveColor.connect = </Looks/PreviewMapped/T.outputs:rgb>\n"
               "            token outputs:surface\n        }\n"
               "        def Shader \"T\"\n        {\n"
               "            uniform token info:id = \"UsdUVTexture\"\n"
               "            asset inputs:file = @./lamp_preview.png@\n"
               "            token inputs:sourceColorSpace = \"sRGB\"\n"
               "            float3 outputs:rgb\n        }\n    }\n}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == std::size(names));
    const auto is = [](const usd::StageMaterial& m, float r, float g, float b) {
        CHECK(m.emission[0] == Catch::Approx(r));
        CHECK(m.emission[1] == Catch::Approx(g));
        CHECK(m.emission[2] == Catch::Approx(b));
    };
    for (const usd::StageMesh& mesh : *meshes) {
        INFO(mesh.path);
        const usd::StageMaterial& m = mesh.material;
        if (mesh.path == "/Standard") {
            is(m, 2.0F, 1.0F, 0.5F);
            CHECK(m.emissionMap.empty());
        } else if (mesh.path == "/Open") {
            is(m, 1.5F, 3.0F, 0.0F);
        } else if (mesh.path == "/Gltf") {
            is(m, 1.0F, 2.0F, 4.0F);
        } else if (mesh.path == "/Preview") {
            is(m, 1.5F, 0.5F, 0.0F);
        } else if (mesh.path == "/Dark" || mesh.path == "/MappedDark") {
            CHECK_FALSE(m.emits());
            CHECK(m.emissionMap.empty());
        } else if (mesh.path == "/GltfMapped") {
            // The map is the colour; the strength multiplies it.
            is(m, 4.0F, 4.0F, 4.0F);
            CHECK(m.emissionMap.file.find("lamp_GltfMapped.png") != std::string::npos);
            CHECK(m.emissionMap.channel == 0);
        } else if (mesh.path == "/WeightMapped") {
            // The map is the weight, one channel of it; the colour multiplies it.
            is(m, 0.25F, 0.5F, 1.0F);
            CHECK(m.emissionMap.file.find("lamp_WeightMapped.png") != std::string::npos);
            CHECK(m.emissionMap.channel == 'r');
        } else {
            is(m, 1.0F, 1.0F, 1.0F);
            CHECK(m.emissionMap.file.find("lamp_preview.png") != std::string::npos);
            CHECK(m.emissionMap.srgb);
        }
    }
}

namespace {

/// The emission record `i % 5` of the table splat_emission_check.slang keeps
/// (`tableEmission`): three floats more a record, as `athenea mesh2splat`
/// writes what a material gives off.
io::RawSplats withTableEmission(const io::RawSplats& raw) {
    static const float kTable[5][3] = {
        {0.0F, 0.0F, 0.0F}, {1.5F, 0.75F, 0.25F}, {0.02F, 0.5F, 4.0F}, {100.0F, 3.0F, 0.0F}, {0.3F, 0.3F, 0.3F}};
    io::RawSplats out = raw;
    const uint32_t stride = raw.encoding.floatsPerRecord;
    out.records.clear();
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* from = raw.records.data() + size_t{i} * stride;
        out.records.insert(out.records.end(), from, from + stride);
        out.records.insert(out.records.end(), kTable[i % 5], kTable[i % 5] + 3);
    }
    out.encoding.floatsPerRecord = stride + 3;
    out.encoding.emission = stride;
    return out;
}

}   // namespace

// THE LIGHT A GAUSSIAN GIVES OFF GOES OUT AND COMES BACK.
//
// A converted gaussian keeps what its material gave off where it stood:
// `primvars:athenea:splat:emission` (color3f, declared by
// AtheneaSplatLightingAPI) in a stage, an RGB9E5 word a splat in
// `GpuSplats::emission`, and a word an element of every block of a `.athc`
// (flags bit 2). Each way back must give the words that went out: the stage
// read as records, the stage read as Hydra reads it, and the levels of detail
// written to a file and read again; and every merged level must give off a
// mean of what it merged, inside the box the table spans. HDR on the way: a
// hundred is kept as a hundred. Compared on the device; counters come back.
TEST_CASE("a cloud's emission survives USD, .athc and the levels of detail", "[usd][gpu][export][lod][emission]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto against = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_emission_check",
                                              "emissionAgainstTable");
    auto within = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_emission_check", "emissionWithin");
    if (!against) FAIL(against.error().toString());
    if (!within) FAIL(within.error().toString());
    const io::RawSplats raw = withTableEmission(cloud(3000));
    auto direct = loader->upload(raw, 0);
    REQUIRE(direct);
    REQUIRE(direct->hasEmission());
    REQUIRE(direct->count == raw.count);

    gpu::BufferDesc countsDesc;
    countsDesc.bytes = 8 * 4;
    countsDesc.elementBytes = 4;
    countsDesc.label = "emission.counts";
    auto counts = gpu::Buffer::create(*gpu->device, countsDesc);
    REQUIRE(counts);
    // RGB9E5 steps by 1/512 of the brightest channel, and rounds to half of
    // that: twice it is the tolerance.
    constexpr float kTolerance = 4.0e-3F;
    const auto check = [&](const gpu::Buffer& a, const gpu::Buffer& b, uint32_t count) {
        const uint32_t zero[8] = {};
        REQUIRE(counts->write(*gpu->device, 0, sizeof(zero), zero));
        gpu::CommandBatch batch(*gpu->device);
        against->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["emissionA"].setBinding(a.rhi());
            cursor["emissionB"].setBinding(b.rhi());
            cursor["counts"].setBinding(counts->rhi());
            cursor["params"]["count"].setData(count);
            cursor["params"]["tolerance"].setData(kTolerance);
        });
        REQUIRE(batch.submit(true));
        std::array<uint32_t, 3> seen{};
        REQUIRE(counts->read(*gpu->device, 0, sizeof(seen), seen.data()));
        return seen;
    };
    const auto self = check(direct->emission, direct->emission, direct->count);
    std::printf("  on the device: %u compared, %u off the table\n", self[0], self[1]);
    CHECK(self[0] == raw.count);
    CHECK(self[1] == 0);

    const fs::path path = scratch("splat_emission.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(prim);
    CHECK(prim.HasAPI(TfToken("AtheneaSplatLightingAPI")));
    const UsdGeomPrimvar written = UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken("athenea:splat:emission"));
    REQUIRE(written);
    CHECK(written.GetTypeName() == SdfValueTypeNames->Color3fArray);
    CHECK(written.GetInterpolation() == UsdGeomTokens->vertex);
    // Declared by the schema, so it is not a custom attribute.
    CHECK_FALSE(written.GetAttr().IsCustom());

    SECTION("read back as records") {
        auto records = usd::readParticleFieldRecords(path);
        REQUIRE(records);
        REQUIRE(records->encoding.emission != io::SplatEncoding::kNoField);
        auto back = loader->upload(*records, 0);
        REQUIRE(back);
        REQUIRE(back->hasEmission());
        REQUIRE(back->count == direct->count);
        const auto seen = check(back->emission, direct->emission, back->count);
        std::printf("  through records: %u compared, %u off the table, %u words apart from the cloud written\n",
                    seen[0], seen[1], seen[2]);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }

    SECTION("read back as Hydra reads it") {
        const UsdVolParticleField3DGaussianSplat field(prim);
        usd::ParticleFieldArrays arrays;
        field.GetPositionsAttr().Get(&arrays.positions);
        field.GetOrientationsAttr().Get(&arrays.orientations);
        field.GetScalesAttr().Get(&arrays.scales);
        field.GetOpacitiesAttr().Get(&arrays.opacities);
        int degree = 0;
        field.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree);
        arrays.shDegree = degree;
        field.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&arrays.shCoefficients);
        written.Get(&arrays.emission);
        const scene::SplatStreams streams = usd::splatStreams(arrays, "emission streams");
        REQUIRE_FALSE(streams.emission.empty());
        auto back = loader->upload(streams, 0);
        REQUIRE(back);
        REQUIRE(back->hasEmission());
        REQUIRE(back->count == direct->count);
        const auto seen = check(back->emission, direct->emission, back->count);
        std::printf("  through Hydra's arrays: %u compared, %u off the table, %u words apart from the cloud "
                    "written\n",
                    seen[0], seen[1], seen[2]);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }

    SECTION("merged by the levels of detail, through a .athc") {
        auto builder = lod::LodBuilder::create(*gpu->library);
        REQUIRE(builder);
        lod::LodBuildSettings chunked;
        chunked.chunkSplats = 1000;
        auto built = builder->build(*direct, chunked);
        if (!built) FAIL(built.error().toString());
        REQUIRE(built->splats.hasEmission());
        // Every merge a mean: inside the box of the table's five.
        uint64_t mergedCompared = 0;
        uint64_t outside = 0;
        for (const lod::LodLevel& level : built->levels) {
            REQUIRE(level.gaussians.hasEmission());
            const uint32_t zero[8] = {};
            REQUIRE(counts->write(*gpu->device, 0, sizeof(zero), zero));
            gpu::CommandBatch batch(*gpu->device);
            within->dispatch(batch, {level.gaussians.count, 1, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["emissionA"].setBinding(level.gaussians.emission.rhi());
                cursor["emissionB"].setBinding(level.gaussians.emission.rhi());
                cursor["counts"].setBinding(counts->rhi());
                rhi::ShaderCursor p = cursor["params"];
                p["count"].setData(level.gaussians.count);
                p["tolerance"].setData(kTolerance);
                const std::array<float, 4> low{0.0F, 0.0F, 0.0F, 0.0F};
                const std::array<float, 4> high{100.0F, 3.0F, 4.0F, 0.0F};
                p["low"].setData(low.data(), 16);
                p["high"].setData(high.data(), 16);
            });
            REQUIRE(batch.submit(true));
            uint32_t seen[2] = {0, 0};
            REQUIRE(counts->read(*gpu->device, 0, sizeof(seen), seen));
            mergedCompared += seen[0];
            outside += seen[1];
        }
        std::printf("  merged: %llu compared over %zu levels, %llu outside what was merged\n",
                    static_cast<unsigned long long>(mergedCompared), built->levels.size(),
                    static_cast<unsigned long long>(outside));
        CHECK(mergedCompared > 0);
        CHECK(outside == 0);

        const fs::path file = scratch("splat_emission.athc");
        REQUIRE(lod::writeAthc(*gpu->device, *built, file));
        auto read = lod::readAthc(*gpu->device, file);
        if (!read) FAIL(read.error().toString());
        REQUIRE(read->splats.hasEmission());
        // Normals are not in this cloud: bit 2 stands alone, after no bit 0.
        CHECK_FALSE(read->splats.hasNormals());
        REQUIRE(read->levels.size() == built->levels.size());
        auto store = render::countDifferent(*gpu->library, read->splats.emission, built->splats.emission, built->count);
        REQUIRE(store);
        uint64_t levelsApart = 0;
        for (size_t l = 0; l < built->levels.size(); ++l) {
            REQUIRE(read->levels[l].gaussians.hasEmission());
            auto apart = render::countDifferent(*gpu->library, read->levels[l].gaussians.emission,
                                                built->levels[l].gaussians.emission,
                                                built->levels[l].gaussians.count);
            REQUIRE(apart);
            levelsApart += *apart;
        }
        std::printf("  through a .athc: %llu of %u splat emissions and %llu merged ones changed\n",
                    static_cast<unsigned long long>(*store), built->count,
                    static_cast<unsigned long long>(levelsApart));
        CHECK(*store == 0);
        CHECK(levelsApart == 0);
    }
}

// AN EMISSIVE QUAD, CONVERTED, GIVES OFF WHAT THE MESH GIVES OFF.
//
// `athenea mesh2splat` dropped a material's emission: a converted lamp was as
// dark as its albedo under the scene's light, relit or transferred. Two quads
// (tests/data/emissive_quad*.usda) -- OpenPBR's luminance times its colour, and
// glTF's emissive map times its strength -- under a dome dim enough that what
// they give off is most of what they show, are converted by ctest beforehand
// (the mesh2splat_emissive_* tests: --no-bake, --transfer, and the radiance
// bake) and drawn here, raster and traced, against the mesh. The relit and
// the transferred clouds add their emission; the baked one holds it in its
// colours already and must not add it again, which would be it twice.
//
// Hidden: it reads what those conversions wrote, so ctest runs it after them
// (emissive_conversions_render_like_the_mesh).
TEST_CASE("an emissive quad converted relit, transferred or baked renders as the mesh does",
          "[.][emissive_conversion][usd][gpu][splat][emission]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path data(ATHENEA_TEST_DATA_DIR);
    const fs::path converted(ATHENEA_EMISSIVE_DIR);
    // The mesh, or a converted cloud in its place with the same sky: the
    // source stage with its quad switched off, and the cloud over it.
    const auto composed = [&](const std::string& name, const fs::path& source, const fs::path& cloud) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    subLayers = [";
        if (!cloud.empty()) {
            out << "@" << cloud.string() << "@, ";
        }
        out << "@" << source.string() << "@]\n    upAxis = \"Y\"\n)\n";
        if (!cloud.empty()) {
            out << "over \"World\"\n{\n    over \"Quad\" (\n        active = false\n    )\n    {\n    }\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 3)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 160, h = 160;
    const auto draw = [&](const fs::path& path, const char* technique) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const auto meanRed = [&](const gpu::Buffer& image) {
        auto stats = render::imageStats(*gpu->library, image, w, h);
        REQUIRE(stats);
        return stats->mean[0];
    };
    for (const char* variant : {"constant", "mapped"}) {
        const fs::path source =
            data / (std::string("emissive_quad") + (std::string(variant) == "mapped" ? "_mapped" : "") + ".usda");
        const fs::path mesh = composed(std::string("splat_emission_mesh_") + variant + ".usda", source, {});
        // THE MESH, BOTH WAYS. Its rasterised image is the reference every
        // cloud is held to pixel by pixel: the traced one samples the map with
        // a path's own jitter, and on the gradient a percent of its pixels sit
        // a tenth off the rasterised ones -- the reference's noise, not the
        // cloud's. The traced mesh is still what the traced cloud's mean is
        // held to.
        const gpu::Buffer meshRaster = draw(mesh, "raster");
        const gpu::Buffer meshTraced = draw(mesh, "rt");
        for (const char* mode : {"relit", "transfer", "baked"}) {
            const fs::path cloudFile = converted / (std::string(variant) + "_" + mode + ".usda");
            if (!fs::exists(cloudFile)) {
                SKIP("'" << cloudFile.string() << "' is not there: ctest converts it first "
                     "(emissive_conversions_render_like_the_mesh)");
            }
            const fs::path card = composed(std::string("splat_emission_card_") + variant + "_" + mode + ".usda",
                                           source, cloudFile);
            for (const char* technique : {"raster", "rt"}) {
                const bool traced = std::string(technique) == "rt";
                const gpu::Buffer c = draw(card, technique);
                auto diff = render::compareHdr(*gpu->library, c, meshRaster, w, h);
                REQUIRE(diff);
                const double meshMean = meanRed(traced ? meshTraced : meshRaster);
                const double cardMean = meanRed(c);
                std::printf("  %-8s %-8s %-6s: p99 %.3f relMSE %.2e against the rasterised mesh; mean r %.3f "
                            "against the mesh's %.3f\n",
                            variant, mode, technique, diff->p99Relative, diff->relMse, cardMean, meshMean);
                INFO(variant << " " << mode << " " << technique);
                // What the mesh gives off is what the cloud gives off, to
                // what a cloud of discs can be; and not twice it, baked.
                CHECK(diff->p99Relative < 0.08);
                CHECK(cardMean == Catch::Approx(meshMean).epsilon(0.03));
            }
        }
    }
}

// WHAT A MATERIAL LAYERS OVER ITS BASE IS READ IN EACH VOCABULARY.
//
// Proposal 026: a gaussian carries its specular's weight, colour and index,
// a clear coat and a sheen, read from the material as constants. OpenPBR
// says `specular_weight`, `coat_weight` with its `coat_darkening`, and
// `fuzz_weight` times `fuzz_color` for its sheen (its coat at 1.6 by default); standard_surface `specular`, `coat` (0.1
// rough at 1.5), `sheen`; UsdPreviewSurface `clearcoat` and
// `clearcoatRoughness` at its own `ior`, and in its specular workflow a
// `specularColor` that is the reflectivity head on; glTF `clearcoat` and a
// `sheen_color` with no weight. A material that names none of it is plain,
// and its conversion carries none.
TEST_CASE("a material's specular, coat and sheen are read in each vocabulary", "[usd][mesh][materials][lobes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path path = scratch("splat_lobes_read.usda");
    const char* names[] = {"Open", "OpenPlain", "Standard", "Preview", "PreviewSpecular", "Gltf"};
    {
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    upAxis = \"Z\"\n)\n";
        for (const char* name : names) {
            out << "def Mesh \"" << name << "\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
                << "    int[] faceVertexCounts = [3]\n    int[] faceVertexIndices = [0, 1, 2]\n"
                   "    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]\n"
                   "    uniform token subdivisionScheme = \"none\"\n"
                << "    rel material:binding = </Looks/" << name << ">\n}\n";
        }
        const auto mx = [&out](const char* name, const char* id, const std::string& inputs) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:mtlx:surface.connect = </Looks/" << name << "/S.outputs:out>\n"
                << "        def Shader \"S\"\n        {\n"
                << "            uniform token info:id = \"" << id << "\"\n" << inputs
                << "            token outputs:out\n        }\n    }\n";
        };
        const auto preview = [&out](const char* name, const std::string& inputs) {
            out << "    def Material \"" << name << "\"\n    {\n"
                << "        token outputs:surface.connect = </Looks/" << name << "/S.outputs:surface>\n"
                << "        def Shader \"S\"\n        {\n"
                << "            uniform token info:id = \"UsdPreviewSurface\"\n" << inputs
                << "            token outputs:surface\n        }\n    }\n";
        };
        out << "def Scope \"Looks\"\n{\n";
        mx("Open", "ND_open_pbr_surface_surfaceshader",
           "            float inputs:specular_weight = 0.5\n"
           "            color3f inputs:specular_color = (1, 0.5, 0.25)\n"
           "            float inputs:specular_ior = 1.45\n"
           "            float inputs:coat_weight = 1\n            float inputs:coat_roughness = 0.2\n"
           "            float inputs:coat_ior = 1.45\n"
           "            float inputs:coat_darkening = 0.25\n"
           "            float inputs:fuzz_weight = 0.5\n            color3f inputs:fuzz_color = (1, 1, 0)\n"
           "            float inputs:fuzz_roughness = 0.4\n");
        mx("OpenPlain", "ND_open_pbr_surface_surfaceshader", "            float inputs:base_metalness = 1\n");
        mx("Standard", "ND_standard_surface_surfaceshader",
           "            float inputs:specular = 0.8\n            float inputs:coat = 0.7\n"
           "            float inputs:sheen = 1\n            color3f inputs:sheen_color = (0.2, 0.4, 0.6)\n");
        preview("Preview", "            float inputs:clearcoat = 1\n            float inputs:clearcoatRoughness = 0.05\n"
                           "            float inputs:ior = 1.45\n            float inputs:metallic = 1\n");
        preview("PreviewSpecular", "            int inputs:useSpecularWorkflow = 1\n"
                                   "            color3f inputs:specularColor = (0.08, 0.04, 0.02)\n"
                                   "            float inputs:metallic = 1\n");
        mx("Gltf", "ND_gltf_pbr_surfaceshader",
           "            float inputs:clearcoat = 0.5\n            float inputs:clearcoat_roughness = 0.3\n"
           "            color3f inputs:sheen_color = (0.3, 0.3, 0.3)\n");
        out << "}\n";
    }
    auto builder = geom::MeshBuilder::create(*gpu->library);
    if (!builder) FAIL(builder.error().toString());
    auto stage = usd::MeshStage::open(path);
    if (!stage) FAIL(stage.error().toString());
    auto meshes = stage->read(*builder, usd::MeshStageOptions{});
    if (!meshes) FAIL(meshes.error().toString());
    REQUIRE(meshes->size() == std::size(names));
    const auto colour = [](const std::array<float, 3>& c, float r, float g, float b) {
        CHECK(c[0] == Catch::Approx(r));
        CHECK(c[1] == Catch::Approx(g));
        CHECK(c[2] == Catch::Approx(b));
    };
    for (const usd::StageMesh& mesh : *meshes) {
        INFO(mesh.path);
        const usd::StageMaterial& m = mesh.material;
        if (mesh.path == "/Open") {
            CHECK(m.layered());
            CHECK(m.specularWeight == Catch::Approx(0.5F));
            colour(m.specularColour, 1.0F, 0.5F, 0.25F);
            CHECK(m.ior == Catch::Approx(1.45F));
            CHECK(m.coatWeight == Catch::Approx(1.0F));
            CHECK(m.coatRoughness == Catch::Approx(0.2F));
            CHECK(m.coatIor == Catch::Approx(1.45F));
            CHECK(m.coatDarkening == Catch::Approx(0.25F));
            colour(m.sheenColour, 0.5F, 0.5F, 0.0F);
            CHECK(m.sheenRoughness == Catch::Approx(0.4F));
        } else if (mesh.path == "/OpenPlain") {
            // A dark metal names no layer: plain, and no conversion of it
            // carries any.
            CHECK_FALSE(m.layered());
            CHECK(m.coatIor == Catch::Approx(1.6F));
            CHECK(m.coatDarkening == Catch::Approx(1.0F));   // OpenPBR's default
            CHECK(m.sheenRoughness == Catch::Approx(0.5F));  // fuzz_roughness's
        } else if (mesh.path == "/Standard") {
            CHECK(m.layered());
            CHECK(m.specularWeight == Catch::Approx(0.8F));
            CHECK(m.coatWeight == Catch::Approx(0.7F));
            CHECK(m.coatRoughness == Catch::Approx(0.1F));
            CHECK(m.coatIor == Catch::Approx(1.5F));
            CHECK(m.coatDarkening == 0.0F);   // standard_surface's coat has none
            colour(m.sheenColour, 0.2F, 0.4F, 0.6F);
        } else if (mesh.path == "/Preview") {
            CHECK(m.coatWeight == Catch::Approx(1.0F));
            CHECK(m.coatRoughness == Catch::Approx(0.05F));
            CHECK(m.coatIor == Catch::Approx(1.45F));
            colour(m.sheenColour, 0.0F, 0.0F, 0.0F);
        } else if (mesh.path == "/PreviewSpecular") {
            // The reflectivity head on: 0.08 at the index that gives it,
            // tinted by the colour over its brightest channel; no metal.
            CHECK(m.metallic == 0.0F);
            colour(m.specularColour, 1.0F, 0.5F, 0.25F);
            const float r = (m.ior - 1.0F) / (m.ior + 1.0F);
            CHECK(r * r == Catch::Approx(0.08F));
        } else {
            CHECK(m.coatWeight == Catch::Approx(0.5F));
            CHECK(m.coatRoughness == Catch::Approx(0.3F));
            colour(m.sheenColour, 0.3F, 0.3F, 0.3F);
        }
    }
}

namespace {

/// The layers record `i % 5` of a table: plain, a car's lacquer, a tinted
/// half specular with a coat and a sheen, the ends of every range, a dim
/// sheen. Thirteen floats more a record, as `athenea mesh2splat` writes them.
io::RawSplats withTableLobes(const io::RawSplats& raw) {
    static const float kTable[5][13] = {
        {1.0F, 1.0F, 1.0F, 1.0F, 1.5F, 0.0F, 0.0F, 1.5F, 0.0F, 0.0F, 0.0F, 0.3F, 0.0F},
        {1.0F, 1.0F, 1.0F, 1.0F, 1.5F, 1.0F, 0.0F, 1.45F, 0.0F, 0.0F, 0.0F, 0.3F, 1.0F},
        {0.5F, 1.0F, 0.5F, 0.25F, 1.45F, 0.25F, 0.4F, 1.6F, 0.2F, 0.4F, 0.6F, 0.5F, 0.0F},
        {0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.75F, 1.0F, 2.5F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F},
        {0.8F, 0.9F, 0.9F, 0.9F, 1.33F, 0.0F, 0.0F, 1.5F, 0.05F, 0.05F, 0.05F, 0.1F, 0.0F}};
    io::RawSplats out = raw;
    const uint32_t stride = raw.encoding.floatsPerRecord;
    out.records.clear();
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* from = raw.records.data() + size_t{i} * stride;
        out.records.insert(out.records.end(), from, from + stride);
        out.records.insert(out.records.end(), kTable[i % 5], kTable[i % 5] + 13);
    }
    out.encoding.floatsPerRecord = stride + 13;
    out.encoding.lobes = stride;
    return out;
}

}   // namespace

// WHAT A MATERIAL LAYERS OVER ITS BASE GOES OUT AND COMES BACK.
//
// Nine primvars of AtheneaSplatLightingAPI in a stage (`specularWeight`,
// `specularColor`, `specularIor`, `coatWeight`, `coatRoughness`, `coatIor`,
// `sheenColor`, `sheenRoughness`, `coatDarkening`), three words a splat on the device. The
// stage read as Hydra reads it must give back the words that went out, and a
// cloud written without them must come back without them.
TEST_CASE("a cloud's specular, coat and sheen survive USD", "[usd][gpu][export][lobes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    const io::RawSplats raw = withTableLobes(cloud(3000));
    auto direct = loader->upload(raw, 0);
    REQUIRE(direct);
    REQUIRE(direct->hasLobes());
    REQUIRE(direct->count == raw.count);

    const fs::path path = scratch("splat_lobes.usda");
    usd::ExportOptions options;
    options.addCamera = false;
    options.relight = true;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
    UsdStageRefPtr stage = UsdStage::Open(path.string());
    REQUIRE(stage);
    const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(prim);
    CHECK(prim.HasAPI(TfToken("AtheneaSplatLightingAPI")));
    usd::ParticleFieldArrays arrays;
    const UsdVolParticleField3DGaussianSplat field(prim);
    field.GetPositionsAttr().Get(&arrays.positions);
    field.GetOrientationsAttr().Get(&arrays.orientations);
    field.GetScalesAttr().Get(&arrays.scales);
    field.GetOpacitiesAttr().Get(&arrays.opacities);
    int degree = 0;
    field.GetRadianceSphericalHarmonicsDegreeAttr().Get(&degree);
    arrays.shDegree = degree;
    field.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(&arrays.shCoefficients);
    const auto primvar = [&](const char* name, pxr::VtValue& into, const SdfValueTypeName& type) {
        const UsdGeomPrimvar written = UsdGeomPrimvarsAPI(prim).GetPrimvar(TfToken(name));
        REQUIRE(written);
        CHECK(written.GetTypeName() == type);
        CHECK(written.GetInterpolation() == UsdGeomTokens->vertex);
        // Declared by the schema, so it is not a custom attribute.
        CHECK_FALSE(written.GetAttr().IsCustom());
        written.Get(&into);
    };
    primvar("athenea:splat:specularWeight", arrays.specularWeight, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:specularColor", arrays.specularColour, SdfValueTypeNames->Color3fArray);
    primvar("athenea:splat:specularIor", arrays.specularIor, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:coatWeight", arrays.coatWeight, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:coatRoughness", arrays.coatRoughness, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:coatIor", arrays.coatIor, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:sheenColor", arrays.sheenColour, SdfValueTypeNames->Color3fArray);
    primvar("athenea:splat:sheenRoughness", arrays.sheenRoughness, SdfValueTypeNames->FloatArray);
    primvar("athenea:splat:coatDarkening", arrays.coatDarkening, SdfValueTypeNames->FloatArray);
    const scene::SplatStreams streams = usd::splatStreams(arrays, "lobes streams");
    auto back = loader->upload(streams, 0);
    REQUIRE(back);
    REQUIRE(back->hasLobes());
    REQUIRE(back->count == direct->count);
    auto apart = render::countDifferent(*gpu->library, back->lobes, direct->lobes, direct->count * 3);
    REQUIRE(apart);
    std::printf("  through Hydra's arrays: %llu of %u words apart from the cloud written\n",
                static_cast<unsigned long long>(*apart), direct->count * 3);
    CHECK(*apart == 0);

    // Without them: nothing written, nothing carried.
    const io::RawSplats plain = cloud(300);
    const fs::path bare = scratch("splat_lobes_bare.usda");
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, plain, bare, options));
    UsdStageRefPtr bareStage = UsdStage::Open(bare.string());
    REQUIRE(bareStage);
    CHECK_FALSE(UsdGeomPrimvarsAPI(bareStage->GetPrimAtPath(SdfPath("/World/Splats")))
                    .GetPrimvar(TfToken("athenea:splat:coatWeight")));
    auto none = loader->upload(plain, 0);
    REQUIRE(none);
    CHECK_FALSE(none->hasLobes());
}

// A BALL OF EACH MATERIAL, CONVERTED, RASTERISED, AGAINST THE MESH PATH TRACED.
//
// Proposal 026, material by material: what the user wants of a conversion is
// the mesh's look in the gaussian rasteriser, and the mesh path traced is
// the ground truth. Five balls under a pale sky and a sun
// (tests/data/lobes/*.usda) -- a car's coated dark metal, chrome, rubber with
// a sheen, a plastic whose specular is weighed and tinted, clear glass --
// converted relit (--no-bake) and baked by ctest beforehand
// (mesh2splat_lobes_*), are drawn here rasterised and held, each, to the mesh
// path traced: the mean within a bound, and the 99th percentile of the
// relative error within another. The paint is what this was written for: its
// metal is a 0.05 green that reflects next to nothing, its colour is the
// lacquer's reflection of the sky, and both a cloud with no coat and a bake
// that dropped its dark metal as polish (`bakeBody`) came out black.
//
// Hidden: it reads what those conversions wrote, so ctest runs it after them
// (lobes_conversions_render_like_the_mesh).
TEST_CASE("a ball of each material converted relit or baked rasterises as the mesh path traces",
          "[.][lobes_conversion][usd][gpu][splat][lobes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path data = fs::path(ATHENEA_TEST_DATA_DIR) / "lobes";
    const fs::path converted(ATHENEA_LOBES_DIR);
    const auto composed = [&](const std::string& name, const fs::path& source, const fs::path& cloud) {
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    subLayers = [";
        if (!cloud.empty()) {
            out << "@" << cloud.string() << "@, ";
        }
        out << "@" << source.string() << "@]\n    upAxis = \"Y\"\n)\n";
        if (!cloud.empty()) {
            out << "over \"World\"\n{\n    over \"Ball\" (\n        active = false\n    )\n    {\n    }\n}\n";
        }
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0, 3.6)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
        return path;
    };
    const uint32_t w = 192, h = 192;
    const auto draw = [&](const fs::path& path, const char* technique, const std::string& keep) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        if (std::string(technique) == "rt") {
            (*renderer)->setPathSamples(64);
            (*renderer)->setPathTotal(1024);
            (*renderer)->setPathBounces(6);
        }
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        // Kept beside the run, for a person to look at (EXR, linear).
        const fs::path picture = scratch(keep + ".exr");
        (void)io::writeExr(picture, w, h, image->rgba);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const auto mean = [&](const gpu::Buffer& image) {
        auto stats = render::imageStats(*gpu->library, image, w, h);
        REQUIRE(stats);
        return (stats->mean[0] + stats->mean[1] + stats->mean[2]) / 3.0;
    };
    // THE BOUNDS, A MATERIAL AND A MODE: the mean of the cloud against the
    // mesh's (relative), and the 99th percentile of the relative error over
    // the frame. Set from the first measurement (decisions.md, proposal 026)
    // with about a quarter of margin: chrome 0.081 relit and 0.115 baked,
    // plastic 0.648 and 0.707 (mean 4.4 % off baked), glass 0.250 and 0.545
    // (means 7.0 % and 4.0 % off), the paint baked 0.459. The paint relit
    // (0.917, before the coat darkened the metal under it) was measured on
    // what this no longer is; its bound is the paint baked's. The rubber was
    // measured again, four runs alike: 0.841 both ways (relMSE 8.4e-3 and
    // 8.6e-3, mean 3.3 % under), and holds to that with the same margin. Its
    // data names the fuzz `sheen_*`, which OpenPBR does not declare, so
    // neither the mesh nor the cloud carries one.
    struct Bound {
        const char* material;
        double      mean;
        double      p99Relit;
        double      p99Baked;
    };
    const Bound bounds[] = {{"paint", 0.07, 0.6, 0.55},   {"chrome", 0.05, 0.12, 0.15},
                            {"rubber", 0.08, 1.05, 1.05},   {"plastic", 0.06, 0.8, 0.85},
                            {"glass", 0.09, 0.32, 0.65}};
    for (const Bound& bound : bounds) {
        const fs::path source = data / (std::string(bound.material) + ".usda");
        const fs::path mesh = composed(std::string("lobes_mesh_") + bound.material + ".usda", source, {});
        const gpu::Buffer meshTraced = draw(mesh, "rt", std::string("lobes_mesh_") + bound.material);
        const double meshMean = mean(meshTraced);
        for (const char* mode : {"relit", "baked"}) {
            const fs::path cloudFile = converted / (std::string(bound.material) + "_" + mode + ".usda");
            if (!fs::exists(cloudFile)) {
                SKIP("'" << cloudFile.string() << "' is not there: ctest converts it first "
                     "(lobes_conversions_render_like_the_mesh)");
            }
            const std::string name = std::string("lobes_cloud_") + bound.material + "_" + mode;
            const fs::path card = composed(name + ".usda", source, cloudFile);
            const gpu::Buffer c = draw(card, "raster", name);
            auto diff = render::compareHdr(*gpu->library, c, meshTraced, w, h);
            REQUIRE(diff);
            const double cloudMean = mean(c);
            std::printf("  %-8s %-6s raster: p99 %.3f relMSE %.2e against the mesh path traced; mean %.4f against "
                        "the mesh's %.4f\n",
                        bound.material, mode, diff->p99Relative, diff->relMse, cloudMean, meshMean);
            INFO(bound.material << " " << mode);
            CHECK(diff->p99Relative < (std::string(mode) == "relit" ? bound.p99Relit : bound.p99Baked));
            CHECK(cloudMean == Catch::Approx(meshMean).epsilon(bound.mean));
        }
    }
}

// A TRANSFER RIGHT UNDER ANY SKY, BALL BY BALL (task TX).
//
// The balls of tests/data/lobes stand on a ground (tests/data/tx) that the
// conversion leaves a mesh. Each ball alone was converted with --transfer
// twice by ctest beforehand (mesh2splat_tx_*): as TX writes it, and as the
// first transfer did (--transfer-cells 0, degree 2). Both are drawn
// rasterised under two skies -- the pale dome and sun the stage was converted
// under, and a sky the conversion never saw: an image with a window, no sun
// -- against the mesh path traced under the same sky. What is held: TX
// within a bound of the mesh, and, for the materials whose look is their
// reflection (the coated paint, the chrome), closer to it than the first
// transfer under both skies: that is what fails before TX.
//
// Hidden: it reads what those conversions wrote, so ctest runs it after them
// (tx_conversions_render_like_the_mesh).
TEST_CASE("a ball converted with a TX transfer rasterises as the mesh path traces, under any sky",
          "[.][tx_conversion][usd][gpu][splat][transfer]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path data = fs::path(ATHENEA_TEST_DATA_DIR) / "tx";
    const fs::path converted(ATHENEA_TX_DIR);
    // The second sky: a lat-long with a blue upper half, a dark brown lower
    // one and a bright window, written by the test.
    const fs::path window = scratch("tx_window_sky.png");
    {
        const uint32_t w = 64, h = 32;
        std::vector<uint32_t> texels(size_t{w} * h);
        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                uint32_t texel = y < h / 2 ? 0xFFE08050u : 0xFF20304Au;   // ABGR: blue above, brown below
                if (y >= 8 && y < 14 && x >= 20 && x < 30) {
                    texel = 0xFFFFFFFFu;
                }
                texels[size_t{y} * w + x] = texel;
            }
        }
        HioImageSharedPtr image = HioImage::OpenForWriting(window.string());
        REQUIRE(image);
        HioImage::StorageSpec spec;
        spec.width = static_cast<int>(w);
        spec.height = static_cast<int>(h);
        spec.depth = 1;
        spec.format = HioFormatUNorm8Vec4;
        spec.data = texels.data();
        REQUIRE(image->Write(spec));
    }
    // 0: the pale dome and the sun it was converted under; 1: the window, no
    // sun; 2: no sky at all, a lamp beside the ball (step 4).
    const auto composed = [&](const std::string& name, const fs::path& source, const fs::path& cloud, int sky) {
        const bool windowSky = sky == 1;
        const fs::path path = scratch(name);
        std::ofstream out(path);
        out << "#usda 1.0\n(\n    subLayers = [";
        if (!cloud.empty()) {
            out << "@" << cloud.string() << "@, ";
        }
        out << "@" << source.string() << "@]\n    upAxis = \"Y\"\n)\n";
        out << "over \"World\"\n{\n";
        if (!cloud.empty()) {
            out << "    over \"Ball\" (\n        active = false\n    )\n    {\n    }\n";
        }
        if (sky == 2) {
            out << "    over \"Sun\" (\n        active = false\n    )\n    {\n    }\n"
                << "    over \"Sky\" (\n        active = false\n    )\n    {\n    }\n"
                << "    def SphereLight \"Lamp\"\n    {\n        float inputs:radius = 0.3\n"
                << "        float inputs:intensity = 40\n        double3 xformOp:translate = (2.2, 2.5, 1.5)\n"
                << "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n";
        }
        if (windowSky) {
            out << "    over \"Sun\" (\n        active = false\n    )\n    {\n    }\n"
                << "    over \"Sky\"\n    {\n        color3f inputs:color = (1, 1, 1)\n"
                << "        float inputs:intensity = 1.5\n"
                << "        asset inputs:texture:file = @" << window.string() << "@\n    }\n";
        }
        out << "}\n";
        out << "def Camera \"Camera\"\n{\n    float focalLength = 35\n"
               "    float horizontalAperture = 24.576\n    float verticalAperture = 24.576\n"
               "    float2 clippingRange = (0.1, 1000)\n"
               "    double3 xformOp:translate = (0, 0.4, 4.2)\n    float3 xformOp:rotateXYZ = (-8, 0, 0)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateXYZ\"]\n}\n";
        return path;
    };
    const uint32_t w = 192, h = 192;
    const auto draw = [&](const fs::path& path, const char* technique, const std::string& keep) {
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        if (std::string(technique) == "rt") {
            (*renderer)->setPathSamples(64);
            (*renderer)->setPathTotal(1024);
            (*renderer)->setPathBounces(6);
        }
        auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
        if (!image) FAIL(image.error().toString());
        const fs::path picture = scratch(keep + ".exr");
        (void)io::writeExr(picture, w, h, image->rgba);
        gpu::BufferDesc desc;
        desc.bytes = image->rgba.size() * sizeof(float);
        desc.elementBytes = 16;
        auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
        REQUIRE(made);
        return std::move(*made);
    };
    // THE BOUNDS, A MATERIAL: TX's relMSE against the mesh over the frame,
    // and whether TX must beat the first transfer. The ground is the same
    // mesh in every frame and adds the same to both.
    struct Bound {
        const char* material;
        double      relMse;
        bool        beatsFirst;
    };
    const Bound bounds[] = {{"paint", 0.30, true}, {"chrome", 0.40, true}, {"rubber", 0.20, false},
                            {"glass", 0.60, false}};
    for (const Bound& bound : bounds) {
        const fs::path source = data / (std::string(bound.material) + ".usda");
        for (const int skyKind : {0, 1, 2}) {
            const std::string sky = skyKind == 0 ? "pale" : skyKind == 1 ? "window" : "lamp";
            const std::string meshName = std::string("tx_mesh_") + bound.material + "_" + sky;
            const gpu::Buffer meshTraced = draw(composed(meshName + ".usda", source, {}, skyKind), "rt", meshName);
            double relMse[2] = {0.0, 0.0};
            const char* modes[2] = {"tx", "first"};
            for (int k = 0; k < 2; ++k) {
                const fs::path cloudFile = converted / (std::string(bound.material) + "_" + modes[k] + ".usdc");
                if (!fs::exists(cloudFile)) {
                    SKIP("'" << cloudFile.string() << "' is not there: ctest converts it first "
                         "(tx_conversions_render_like_the_mesh)");
                }
                const std::string name = std::string("tx_cloud_") + bound.material + "_" + modes[k] + "_" + sky;
                const gpu::Buffer c = draw(composed(name + ".usda", source, cloudFile, skyKind), "raster", name);
                auto diff = render::compareHdr(*gpu->library, c, meshTraced, w, h);
                REQUIRE(diff);
                relMse[k] = diff->relMse;
                std::printf("  %-7s %-6s %-5s raster: relMSE %.4f p99 %.3f against the mesh path traced\n",
                            bound.material, sky.c_str(), modes[k], diff->relMse, diff->p99Relative);
            }
            INFO(bound.material << " under the " << sky << " sky");
            CHECK(relMse[0] < bound.relMse);
            if (bound.beatsFirst) {
                CHECK(relMse[0] < relMse[1]);
            } else {
                // Glass: no worse than the first transfer, which drew a
                // clear ball sharp against the sky it bends (the pawn's head
                // came back milky when the field filled the whole lens).
                CHECK(relMse[0] <= 1.05 * relMse[1]);
            }
        }
    }
}

// A MAP ON A LAYER IS SAMPLED PER GAUSSIAN (task TX). A card whose coat
// weight is a map, half nothing and half whole (tests/data/lobes/coat_map.usda),
// converted by ctest beforehand: its gaussians carry both, about half each,
// where a constant would have given every one the same.
//
// Hidden: it reads what that conversion wrote (a_map_on_a_layer_is_sampled_per_gaussian).
TEST_CASE("a map on a layer is sampled per gaussian", "[.][layer_maps][usd][lobes]") {
    const fs::path cloud = fs::path(ATHENEA_LOBES_DIR) / "coat_map.usda";
    if (!fs::exists(cloud)) {
        SKIP("'" << cloud.string() << "' is not there: ctest converts it first");
    }
    UsdStageRefPtr stage = UsdStage::Open(cloud.string());
    REQUIRE(stage);
    const UsdPrim splats = stage->GetPrimAtPath(SdfPath("/World/Splats"));
    REQUIRE(splats);
    VtFloatArray weights;
    REQUIRE(UsdGeomPrimvarsAPI(splats).GetPrimvar(TfToken("primvars:athenea:splat:coatWeight")).Get(&weights));
    size_t none = 0, whole = 0;
    for (const float w : weights) {
        none += w < 0.1F ? 1 : 0;
        whole += w > 0.9F ? 1 : 0;
    }
    std::printf("  coat map: %zu gaussians, %zu with no coat, %zu with a whole one\n", weights.size(), none, whole);
    REQUIRE(!weights.empty());
    CHECK(none > weights.size() / 3);
    CHECK(whole > weights.size() / 3);
}
