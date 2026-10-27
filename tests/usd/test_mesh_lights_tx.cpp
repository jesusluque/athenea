// Copyright (c) 2026 jesus luque.
//
// TX GAUSSIANS RELIT BY MESH LIGHTS (stage C, splat_mesh_lights.slang): the
// balls of tests/data/tx converted with a TX transfer (ctest's
// mesh2splat_tx_*), on their ground under a strip of light and nothing else,
// rasterised against the mesh path traced under the same strip. The same
// strip as a UsdLux RectLight is what a relit splat took before -- one sample
// at the light's middle -- and the closed form must do better than it.
//
// Hidden: it reads what the conversions wrote (`[.meshlights-tx]`).
#include "../gpu/GpuTest.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "mesh_lights_tx";
    fs::create_directories(dir);
    return dir / name;
}

/// The ball's stage with its sky and sun off, a strip of light over it --
/// a MeshLightAPI quad, or a RectLight of the same size, radiance and place
/// -- and, where `cloud` is given, the converted cloud in place of the ball.
fs::path composed(const std::string& name, const fs::path& source, const fs::path& cloud, bool meshLight) {
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
    out << "    over \"Sun\" (\n        active = false\n    )\n    {\n    }\n"
        << "    over \"Sky\" (\n        active = false\n    )\n    {\n    }\n";
    // A strip 3 x 0.3 at y 2.4 above the ball, facing down: the ball's shadow
    // falls under it, where the camera sees little of it.
    if (meshLight) {
        out << "    def Mesh \"Strip\" (\n        prepend apiSchemas = [\"MeshLightAPI\"]\n    )\n    {\n"
               "        int[] faceVertexCounts = [4]\n        int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "        point3f[] points = [(-1.5, 2.4, -0.15), (1.5, 2.4, -0.15), (1.5, 2.4, 0.15), (-1.5, 2.4, 0.15)]\n"
               "        uniform token subdivisionScheme = \"none\"\n"
               "        float inputs:intensity = 6\n        color3f inputs:color = (1, 0.85, 0.7)\n    }\n";
    } else {
        out << "    def RectLight \"Strip\"\n    {\n"
               "        float inputs:intensity = 6\n        float inputs:width = 3\n        float inputs:height = 0.3\n"
               "        color3f inputs:color = (1, 0.85, 0.7)\n"
               "        double3 xformOp:translate = (0, 2.4, 0)\n        double xformOp:rotateX = -90\n"
               "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:rotateX\"]\n    }\n";
    }
    out << "}\n";
    return path;
}

struct Picture {
    gpu::Buffer rgba;
};

gpu::Buffer draw(test::Gpu& gpu, const fs::path& path, const char* technique, uint32_t w, uint32_t h) {
    auto renderer = usd::StageRenderer::open(path);
    if (!renderer) FAIL(renderer.error().toString());
    if (std::string(technique) == "rt") {
        (*renderer)->setPathSamples(64);
        (*renderer)->setPathTotal(1024);
        (*renderer)->setPathBounces(6);
    } else {
        // The ground is a mesh either way: lit converged by the rect light's
        // samples as by the mesh light's closed form, and shadowed by the
        // cloud in neither -- a cloud casts no mesh light's shadow on a mesh
        // (decisions.md, stage C), and the rect light's map would be the one
        // thing the two frames do not share -- so what differs is the ball.
        (*renderer)->setLightSamples(64);
        (*renderer)->setCloudShadows(false);
    }
    auto image = (*renderer)->render("/Camera", 0.0, w, h, technique);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu.device, desc, image->rgba.data());
    REQUIRE(made);
    return std::move(*made);
}

/// a | b | |a - b| x 4, written as a PNG where ATHENEA_MESH_LIGHT_IMAGES asks.
void picture(test::Gpu& gpu, const gpu::Buffer& a, const gpu::Buffer& b, uint32_t w, uint32_t h,
             const std::string& name) {
    const char* folder = std::getenv("ATHENEA_MESH_LIGHT_IMAGES");
    if (folder == nullptr || *folder == '\0') {
        return;
    }
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/triptych", "triptych");
    REQUIRE(kernel);
    gpu::Buffer out = test::uintBuffer(*gpu.device, uint64_t{w} * 3 * h, "triptych");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {w, h, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["a"].setBinding(a.rhi());
            cursor["b"].setBinding(b.rhi());
            cursor["out"].setBinding(out.rhi());
            cursor["width"].setData(w);
            cursor["height"].setData(h);
            cursor["exposure"].setData(1.0F);
            cursor["gain"].setData(4.0F);
        });
        REQUIRE(batch.submit(true));
    }
    auto pixels = out.readAll<uint32_t>(*gpu.device);
    REQUIRE(pixels);
    fs::create_directories(folder);
    pxr::HioImageSharedPtr image = pxr::HioImage::OpenForWriting((fs::path(folder) / (name + ".png")).string());
    REQUIRE(image);
    pxr::HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w * 3);
    spec.height = static_cast<int>(h);
    spec.format = pxr::HioFormatUNorm8Vec4srgb;
    spec.data = pixels->data();
    CHECK(image->Write(spec));
}

}   // namespace

TEST_CASE("TX balls under a strip of mesh light rasterise closer to the mesh than under the rect light's middle sample",
          "[.meshlights-tx][usd][gpu][splat][transfer][meshlights]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization || !gpu->device->caps().rayQuery) {
        SKIP("needs rasterisation and ray queries");
    }
    const fs::path data = fs::path(ATHENEA_TEST_DATA_DIR) / "tx";
    const fs::path converted(ATHENEA_TX_DIR);
    const uint32_t w = 192, h = 192;
    for (const char* material : {"paint", "rubber", "chrome"}) {
        const fs::path source = data / (std::string(material) + ".usda");
        const fs::path cloudFile = converted / (std::string(material) + "_tx.usdc");
        if (!fs::exists(cloudFile)) {
            SKIP("'" << cloudFile.string() << "' is not there: mesh2splat_tx_" << material << "_tx converts it");
        }
        double relMse[2] = {0.0, 0.0};
        for (const bool meshLight : {true, false}) {
            const std::string kind = meshLight ? "meshlight" : "rect";
            const gpu::Buffer traced = draw(*gpu, composed(std::string("mesh_") + material + "_" + kind + ".usda",
                                                          source, {}, meshLight),
                                            "rt", w, h);
            const gpu::Buffer cloud = draw(*gpu, composed(std::string("cloud_") + material + "_" + kind + ".usda",
                                                         source, cloudFile, meshLight),
                                           "raster", w, h);
            auto diff = render::compareHdr(*gpu->library, cloud, traced, w, h);
            REQUIRE(diff);
            relMse[meshLight ? 0 : 1] = diff->relMse;
            picture(*gpu, cloud, traced, w, h, std::string("c_") + material + "_" + kind + "_raster_pt");
            // The mesh rasterised, for what the ground and the raster's own
            // lighting leave between the two routes.
            const gpu::Buffer meshRaster = draw(*gpu, composed(std::string("mesh_") + material + "_" + kind + ".usda",
                                                              source, {}, meshLight),
                                                "raster", w, h);
            auto meshDiff = render::compareHdr(*gpu->library, meshRaster, traced, w, h);
            REQUIRE(meshDiff);
            std::printf("  %-6s the mesh rasterised against the same: relMSE %.4f\n", material, meshDiff->relMse);
            picture(*gpu, meshRaster, traced, w, h, std::string("c_") + material + "_" + kind + "_mesh_raster_pt");
            std::printf("  %-6s under the strip as a %-9s: the TX cloud rasterised against the mesh path traced, "
                        "relMSE %.4f p99 %.3f\n",
                        material, meshLight ? "mesh light" : "RectLight", diff->relMse, diff->p99Relative);
        }
        INFO(material);
        // Most of either error is the ground's: the mesh path traced has the
        // ball's shadow under it and neither raster frame has (cloud shadows
        // are off in both). On a rough ball the two lights put much the same
        // on it; on chrome the rect light's middle sample is a point, and its
        // reflection a dot where the strip's is a bar.
        CHECK(relMse[0] < 1.02 * relMse[1]);
        if (std::string(material) == "chrome") {
            CHECK(relMse[0] < 0.5 * relMse[1]);
        }
    }
}
