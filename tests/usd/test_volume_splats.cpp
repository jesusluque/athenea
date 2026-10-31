// Copyright (c) 2026 jesus luque.
//
// A MEDIUM OF GAUSSIANS IN THE RASTER (proposal 105), judged against the path
// tracer. The fixture is made on the device: a known mixture of gaussians
// (world::VolumeSplats::randomMixture), given back onto a box of voxels for
// the path tracer (`splatToGrid`, written as a .vdb) and written as the
// ParticleField `athenea vdb2splat` writes (`exportRecords`). The raster
// draws the gaussians by the exact optical depth of each pixel's ray, the
// path tracer walks the grid; the frames are compared on the device, and what
// the host reads back are compareHdr's and compareImages' numbers.
#include "../gpu/GpuTest.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/io/Vdb.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/StageRenderer.h"
#include "athenea/world/VolumeSplats.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    pxr::PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "volume-splats";
    fs::create_directories(dir);
    return dir / name;
}

gpu::Buffer upload(test::Gpu& gpu, const usd::StageImage& image) {
    gpu::BufferDesc desc;
    desc.bytes = image.rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image.rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

/// The fixture's two files: `count` random gaussians in a 32^3 box of voxels
/// 0.1 wide, as a ParticleField and given back onto the box.
struct Mixture {
    fs::path splats;
    fs::path grid;
};

Mixture makeMixture(test::Gpu& gpu, uint32_t count, uint32_t seed) {
    const fs::path topologyPath = scratch("mixture_topology.vdb");
    const io::VdbBox box{{0, 0, 0}, {32, 32, 32}, 0.0F};
    REQUIRE(io::writeVdbBoxes(topologyPath, "density", 0.1, std::span<const io::VdbBox>(&box, 1)));
    auto topology = io::readVdbGrid(topologyPath, "density");
    if (!topology) FAIL(topology.error().toString());
    auto splats = world::VolumeSplats::create(*gpu.library);
    if (!splats) FAIL(splats.error().toString());
    world::PackedGrids packed = world::PackedGrids::of(*topology, nullptr, nullptr);
    auto words = gpu::Buffer::fromSpan<uint32_t>(*gpu.device, packed.words, "volumeSplats.mixture.words");
    REQUIRE(words);
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{count} * 64;
    desc.elementBytes = 16;
    desc.label = "volumeSplats.mixture";
    auto mixture = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(mixture);
    {
        gpu::CommandBatch batch(*gpu.device);
        // Held 11 voxels in from the box's sides, deviations 1.5 to 3: every
        // gaussian's 3.5 deviations inside the box, so the grid holds all of
        // what the raster draws.
        splats->randomMixture(batch, *mixture, count, {32.0F, 32.0F, 32.0F}, 11.0F, 1.5F, 3.0F, seed);
        REQUIRE(batch.submit(true));
    }
    // The path tracer's grid: the mixture at the box's voxels.
    REQUIRE(splats->splatToGrid(*mixture, count, *words, packed.density, packed.leafCount));
    REQUIRE(words->read(*gpu.device, 0, packed.words.size() * 4, packed.words.data()));
    Mixture out;
    out.grid = scratch("mixture_back.vdb");
    const std::vector<io::NanoGrid> grids{packed.unpack(packed.density, *topology)};
    REQUIRE(io::writeNanoGrids(out.grid, grids));
    // The raster's cloud, as `athenea vdb2splat -o` writes it.
    gpu::BufferDesc exportDesc;
    exportDesc.bytes = uint64_t{count} * 19 * 4;
    exportDesc.elementBytes = 4;
    exportDesc.label = "volumeSplats.exported";
    auto exported = gpu::Buffer::create(*gpu.device, exportDesc);
    REQUIRE(exported);
    {
        gpu::CommandBatch batch(*gpu.device);
        splats->exportRecords(batch, *mixture, count, topology->indexToWorld, 1.0F, *exported);
        REQUIRE(batch.submit(true));
    }
    std::vector<uint32_t> halves(size_t{count} * 5);
    REQUIRE(exported->read(*gpu.device, uint64_t{count} * 14 * 4, halves.size() * 4, halves.data()));
    usd::DeviceSplatRecords raw;
    raw.source = "mixture";
    raw.count = count;
    raw.records = *exported;
    raw.linear = true;
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 14;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3;
    e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10;
    e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.opacity_ = io::SplatEncoding::Opacity::Linear;
    e.scale_ = io::SplatEncoding::Scale::Linear;
    e.colour = io::SplatEncoding::Colour::LinearLight;
    e.rotation = io::SplatEncoding::Rotation::Float;
    usd::ExportOptions options;
    options.maxDegree = 0;
    options.linear = true;
    options.volumeHalves = halves;
    out.splats = scratch("mixture_splats.usdc");
    REQUIRE(usd::writeParticleFieldStage(*gpu.library, raw, out.splats, options));
    return out;
}

/// What both stages share: a sun, a dome, a grey ground under the box --
/// each where `lit` asks for light at all.
std::string sceneHead(bool lit, bool ground) {
    std::string s = "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
    if (lit) {
        s += "def DistantLight \"Sun\"\n{\n"
             "    float inputs:intensity = 3\n    bool inputs:normalize = 1\n    float inputs:angle = 1\n"
             "    double3 xformOp:rotateXYZ = (-50, 30, 0)\n"
             "    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n";
    }
    if (ground) {
        s += "def Mesh \"Ground\"\n{\n"
             "    int[] faceVertexCounts = [4]\n    int[] faceVertexIndices = [0, 1, 2, 3]\n"
             "    point3f[] points = [(-20, 0.3, 20), (20, 0.3, 20), (20, 0.3, -20), (-20, 0.3, -20)]\n"
             "    color3f[] primvars:displayColor = [(0.5, 0.5, 0.5)]\n"
             "    uniform token subdivisionScheme = \"none\"\n}\n";
    }
    return s;
}

/// The gaussians, as the raster draws them.
fs::path rasterStage(const Mixture& m, const char* name, float density, const char* albedo, const char* emission,
                     bool lit, bool ground) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << sceneHead(lit, ground) << "def \"Smoke\" (\n    prepend references = @" << m.splats.string()
        << "@</World/Splats>\n)\n{\n"
        << "    float primvars:athenea:densityScale = " << density << "\n"
        << "    color3f primvars:athenea:albedo = " << albedo << "\n"
        << "    color3f primvars:athenea:volume:emission = " << emission << "\n"
        << "    int primvars:athenea:volume:octaves = 1\n}\n";
    return path;
}

/// The grid given back, as the path tracer draws it.
fs::path tracedStage(const Mixture& m, const char* name, float density, const char* albedo, const char* emission,
                     bool lit, bool ground) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << sceneHead(lit, ground);
    if (emission != nullptr) {
        // Emission is the Material's to give (UsdVol): an absorbing medium
        // of `density` a unit of density, glowing.
        out << "def Volume \"Smoke\" (\n    prepend apiSchemas = [\"MaterialBindingAPI\"]\n)\n{\n"
            << "    rel field:density = </Smoke/density>\n    rel material:binding = </Materials/Glow>\n"
            << "    def OpenVDBAsset \"density\"\n    {\n        asset filePath = @" << m.grid.string()
            << "@\n        token fieldName = \"density\"\n    }\n}\n"
            << "def Scope \"Materials\"\n{\n    def Material \"Glow\"\n    {\n"
            << "        token outputs:mtlx:volume.connect = </Materials/Glow/Volume.outputs:out>\n"
            << "        def Shader \"Volume\"\n        {\n            uniform token info:id = \"ND_volume\"\n"
            << "            token inputs:vdf.connect = </Materials/Glow/Absorb.outputs:out>\n"
            << "            token inputs:edf.connect = </Materials/Glow/Emit.outputs:out>\n"
            << "            token outputs:out\n        }\n"
            << "        def Shader \"Absorb\"\n        {\n            uniform token info:id = \"ND_absorption_vdf\"\n"
            << "            float3 inputs:absorption = (" << density << ", " << density << ", " << density << ")\n"
            << "            token outputs:out\n        }\n"
            << "        def Shader \"Emit\"\n        {\n            uniform token info:id = \"ND_uniform_edf\"\n"
            << "            color3f inputs:color = " << emission << "\n            token outputs:out\n        }\n"
            << "    }\n}\n";
    } else {
        out << "def Volume \"Smoke\"\n{\n    rel field:density = </Smoke/density>\n"
            << "    float primvars:athenea:densityScale = " << density << "\n"
            << "    color3f primvars:athenea:albedo = " << albedo << "\n"
            << "    def OpenVDBAsset \"density\"\n    {\n        asset filePath = @" << m.grid.string()
            << "@\n        token fieldName = \"density\"\n    }\n}\n";
    }
    return path;
}

render::Camera boxCamera() {
    render::Camera camera = render::Camera::lookingAt({1.6, 2.6, 8.0}, {1.6, 1.4, 1.6});
    camera.lens.focal = 35.0;
    return camera;
}

}   // namespace

// EMISSION AND ABSORPTION ALONE: a medium that glows with radiance one and
// scatters nothing sends L (1 - T) along a ray, whatever order its gaussians
// come in, so the raster -- each gaussian's alpha its exact optical depth on
// the pixel's ray, its radiance one -- and the path tracer through the grid
// the same gaussians give back must draw one frame. What separates them is
// the grid's sampling of the gaussians and the paths' noise.
TEST_CASE("a medium's gaussians drawn by their optical depth glow as the path traced grid they give back",
          "[usd][volume][vdb2splat][raster]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const Mixture m = makeMixture(*gpu, 12, 4u);
    const fs::path raster = rasterStage(m, "ea_raster.usda", 4.0F, "(0, 0, 0)", "(1, 1, 1)", false, false);
    const fs::path traced = tracedStage(m, "ea_traced.usda", 4.0F, "(0, 0, 0)", "(1, 1, 1)", false, false);
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto rasterRenderer = usd::StageRenderer::open(raster);
    if (!rasterRenderer) FAIL(rasterRenderer.error().toString());
    auto drawn = (*rasterRenderer)->render(boxCamera(), 0.0, w, h, "raster");
    if (!drawn) FAIL(drawn.error().toString());
    auto tracedRenderer = usd::StageRenderer::open(traced);
    if (!tracedRenderer) FAIL(tracedRenderer.error().toString());
    (*tracedRenderer)->setPathSamples(256);
    (*tracedRenderer)->setPathBounces(1);
    (*tracedRenderer)->setPathTotal(2048);
    auto truth = (*tracedRenderer)->render(boxCamera(), 0.0, w, h, "rt");
    if (!truth) FAIL(truth.error().toString());
    const gpu::Buffer a = upload(*gpu, *drawn);
    const gpu::Buffer b = upload(*gpu, *truth);
    auto stats = render::imageStats(*gpu->library, a, w, h);
    auto statsTruth = render::imageStats(*gpu->library, b, w, h);
    REQUIRE(stats);
    REQUIRE(statsTruth);
    auto hdr = render::compareHdr(*gpu->library, a, b, w, h);
    auto codes = render::compareImages(*gpu->library, a, b, w, h);
    REQUIRE(hdr);
    REQUIRE(codes);
    std::printf("  emission-absorption: raster mean %.5f, path traced %.5f; relMSE %.5f, p99 relative %.4f; "
                "8-bit p99 %u max %u, %llu of %llu over 2\n",
                stats->mean[0], statsTruth->mean[0], hdr->relMse, hdr->p99Relative, codes->p99, codes->max,
                static_cast<unsigned long long>(codes->over2), static_cast<unsigned long long>(codes->pixels));
    // The medium is in the frame and glows as much as the grid does. The
    // 8-bit codes are not held: the paths' noise in a faint pixel is a dozen
    // codes once sRGB-encoded.
    CHECK(statsTruth->mean[0] > 0.02);
    CHECK(std::abs(stats->mean[0] - statsTruth->mean[0]) < 0.02 * statsTruth->mean[0] + 1.0e-3);
    CHECK(hdr->relMse < 5.0e-3);
}

// LIT BY A SUN, OVER A GROUND: the medium scatters a normalized sun once
// (`octaves` 1), through the transmittance the cloud shadow map gives each
// gaussian at its centre -- its own Fourier terms -- and its shadow falls on
// the ground through the same map. Against the path tracer with no bounce
// after the first hit, which is single scattering and the ground's direct
// light.
TEST_CASE("a lit medium of gaussians and its shadow on the ground follow the path traced grid",
          "[usd][volume][vdb2splat][raster][shadow]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const gpu::Caps& caps = gpu->device->caps();
    if (!caps.rasterization || !caps.rayQuery || !caps.accelerationStructure) {
        SKIP("needs rasterisation and ray queries");
    }
    if (!io::haveOpenVdb()) {
        SKIP("built without OpenVDB");
    }
    const Mixture m = makeMixture(*gpu, 12, 4u);
    const fs::path raster = rasterStage(m, "lit_raster.usda", 4.0F, "(0.8, 0.8, 0.8)", "(0, 0, 0)", true, true);
    const fs::path traced = tracedStage(m, "lit_traced.usda", 4.0F, "(0.8, 0.8, 0.8)", nullptr, true, true);
    const fs::path bare = scratch("lit_bare.usda");
    {
        std::ofstream out(bare);
        out << sceneHead(true, true);
    }
    const uint32_t w = 160;
    const uint32_t h = 120;
    auto rasterRenderer = usd::StageRenderer::open(raster);
    if (!rasterRenderer) FAIL(rasterRenderer.error().toString());
    auto drawn = (*rasterRenderer)->render(boxCamera(), 0.0, w, h, "raster");
    if (!drawn) FAIL(drawn.error().toString());
    auto bareRenderer = usd::StageRenderer::open(bare);
    if (!bareRenderer) FAIL(bareRenderer.error().toString());
    auto unshadowed = (*bareRenderer)->render(boxCamera(), 0.0, w, h, "raster");
    if (!unshadowed) FAIL(unshadowed.error().toString());
    auto tracedRenderer = usd::StageRenderer::open(traced);
    if (!tracedRenderer) FAIL(tracedRenderer.error().toString());
    (*tracedRenderer)->setPathSamples(256);
    // No bounce after the first hit: single scattering, and the ground's
    // direct light.
    (*tracedRenderer)->setPathBounces(0);
    (*tracedRenderer)->setPathTotal(2048);
    auto truth = (*tracedRenderer)->render(boxCamera(), 0.0, w, h, "rt");
    if (!truth) FAIL(truth.error().toString());
    const gpu::Buffer a = upload(*gpu, *drawn);
    const gpu::Buffer b = upload(*gpu, *truth);
    const gpu::Buffer c = upload(*gpu, *unshadowed);
    auto stats = render::imageStats(*gpu->library, a, w, h);
    auto statsTruth = render::imageStats(*gpu->library, b, w, h);
    auto statsBare = render::imageStats(*gpu->library, c, w, h);
    REQUIRE(stats);
    REQUIRE(statsTruth);
    REQUIRE(statsBare);
    auto hdr = render::compareHdr(*gpu->library, a, b, w, h);
    auto codes = render::compareImages(*gpu->library, a, b, w, h);
    REQUIRE(hdr);
    REQUIRE(codes);
    std::printf("  lit: raster mean %.5f, path traced %.5f, ground alone %.5f; relMSE %.5f, p99 relative %.4f; "
                "8-bit p99 %u max %u, %llu of %llu over 2\n",
                stats->mean[1], statsTruth->mean[1], statsBare->mean[1], hdr->relMse, hdr->p99Relative, codes->p99,
                codes->max, static_cast<unsigned long long>(codes->over2),
                static_cast<unsigned long long>(codes->pixels));
    // The medium and its shadow changed the frame, and as the path tracer
    // says they do.
    CHECK(std::abs(statsTruth->mean[1] - statsBare->mean[1]) > 0.005);
    CHECK(std::abs(stats->mean[1] - statsTruth->mean[1]) < 0.1 * std::abs(statsTruth->mean[1] - statsBare->mean[1]) + 0.01);
    CHECK(hdr->relMse < 0.03);
}
