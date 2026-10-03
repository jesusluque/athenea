// Copyright (c) 2026 jesus luque.
//
// A frame through Hydra and what it costs the host: how often it waits for
// the device, that leaving it in flight or timing its phases draws the same
// image word for word, and that the phases' GPU time fits in the frame's.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <pxr/base/plug/registry.h>

#include "athenea/gpu/FrameProfiler.h"
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

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "waits";
    fs::create_directories(dir);
    return dir / name;
}

/// A wall, a cloud and a sheet of points, any of them, under a distant light
/// and a dome, seen by a camera.
fs::path stage(const std::string& name, bool wall, bool splats, bool points) {
    const fs::path path = scratch(name);
    std::ofstream out(path);
    out << "#usda 1.0\n(\n    upAxis = \"Y\"\n)\n";
    if (wall) {
        out << "def Mesh \"Wall\"\n{\n"
               "    int[] faceVertexCounts = [4]\n"
               "    int[] faceVertexIndices = [0, 1, 2, 3]\n"
               "    point3f[] points = [(-9, -9, 0), (9, -9, 0), (9, 9, 0), (-9, 9, 0)]\n"
               "    uniform token subdivisionScheme = \"none\"\n"
               "    color3f[] primvars:displayColor = [(0.3, 0.6, 0.9)] ( interpolation = \"constant\" )\n"
               "    double3 xformOp:translate = (0, 0, -4)\n"
               "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n"
               "}\n";
    }
    if (splats) {
        out << "def \"Cloud\" ( references = @./waits_cloud.usdc@</World/Splats> )\n{\n}\n";
    }
    if (points) {
        out << "def Points \"Sheet\"\n{\n    point3f[] points = [";
        for (int y = -4; y <= 4; ++y) {
            for (int x = -4; x <= 4; ++x) {
                out << (x == -4 && y == -4 ? "" : ", ") << "(" << x * 0.3 << ", " << y * 0.3 << ", 0.5)";
            }
        }
        out << "]\n    float[] widths = [0.12] ( interpolation = \"constant\" )\n"
               "    color3f[] primvars:displayColor = [(1, 0.5, 0.25)] ( interpolation = \"constant\" )\n}\n";
    }
    out << "def DistantLight \"Sun\"\n{\n    float inputs:intensity = 2\n"
           "    float3 xformOp:rotateXYZ = (-40, 20, 0)\n    uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n}\n"
           "def DomeLight \"Sky\"\n{\n    float inputs:intensity = 0.5\n    color3f inputs:color = (0.6, 0.7, 0.9)\n}\n"
           "def Camera \"Camera\"\n{\n"
           "    float2 clippingRange = (0.1, 1000)\n    float focalLength = 30\n"
           "    float horizontalAperture = 24.576\n    float verticalAperture = 18.432\n"
           "    double3 xformOp:translate = (0.2, 0.1, 9)\n"
           "    uniform token[] xformOpOrder = [\"xformOp:translate\"]\n}\n";
    return path;
}

}   // namespace

/// HOW OFTEN A FRAME WAITS. A frame of splats waits once at its end -- what
/// the rasteriser learns of its counts -- and once more to read its image;
/// meshes add nothing to that, as nothing of a mesh layer is read back. The
/// bound is the number a regression would break; `athenea stage --frames
/// --waits` says where each one is.
TEST_CASE("a frame through Hydra waits a bounded number of times, and draws the same image in flight or timed",
          "[usd][gpu][waits]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rasterization) {
        SKIP("no rasterisation on this device: the mesh layer below the splats is rasterised");
    }
    const fs::path clouds = scratch("waits_cloud.usdc");
    fs::remove(clouds);
    const test::CloudBuilder built = test::randomCloud(4000, 3, 0.01F, 0.3F);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, built.raw, clouds, {.addCamera = false}));
    struct Case {
        const char* name;
        bool        wall, splats, points;
        uint64_t    waits;   // at most, a frame after the first
    };
    const uint32_t w = 240;
    const uint32_t h = 180;
    for (const Case c : {Case{"splats", false, true, false, 2}, Case{"mesh", true, false, false, 2},
                         Case{"mixed", true, true, true, 2}}) {
        INFO(c.name);
        const fs::path path = stage(std::string("waits_") + c.name + ".usda", c.wall, c.splats, c.points);
        auto renderer = usd::StageRenderer::open(path);
        if (!renderer) FAIL(renderer.error().toString());
        usd::StageRenderer& r = **renderer;
        gpu::Device& device = r.device();
        const auto frame = [&](uint64_t* waits, double* ms) {
            const uint64_t before = device.hostWaits();
            const auto start = std::chrono::steady_clock::now();
            auto image = r.render("/Camera", 0.0, w, h);
            if (!image) FAIL(image.error().toString());
            if (ms != nullptr) {
                *ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            }
            if (waits != nullptr) {
                *waits = device.hostWaits() - before;
            }
            gpu::BufferDesc desc;
            desc.bytes = image->rgba.size() * sizeof(float);
            desc.elementBytes = 16;
            auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
            REQUIRE(made);
            return std::move(*made);
        };
        (void)frame(nullptr, nullptr);   // the first: uploads, compiles, sizes
        uint64_t waits = 0;
        const gpu::Buffer waited = frame(&waits, nullptr);
        std::printf("  %s: %llu host waits a frame\n", c.name, static_cast<unsigned long long>(waits));
        CHECK(waits <= c.waits);

        // Left in flight: the rasteriser does not wait, and the image read
        // back is the one wait.
        r.setFramesInFlight(true);
        uint64_t flyingWaits = 0;
        const gpu::Buffer flying = frame(&flyingWaits, nullptr);
        r.setFramesInFlight(false);
        CHECK(flyingWaits <= 1);

        // Timed: on Metal every phase is a command buffer of its own, which
        // must change nothing of what is drawn.
        r.setProfiling(true);
        double ms = 0.0;
        const gpu::Buffer timed = frame(nullptr, &ms);
        r.setProfiling(false);
        std::optional<gpu::FrameTimes> times = r.phaseTimes();
        REQUIRE(times.has_value());
        double sum = 0.0;
        bool raster = false;
        bool shade = false;
        for (const gpu::PhaseTime& phase : times->phases) {
            std::printf("    %s %.3f ms\n", phase.name.c_str(), phase.ms);
            sum += phase.ms;
            raster = raster || phase.name == "raster";
            shade = shade || phase.name == "shade";
        }
        std::printf("  %s: phases %.3f ms of a %.3f ms frame (%s), span %.3f ms\n", c.name, sum, ms, times->source,
                    times->spanMs);
        CHECK(sum > 0.0);
        CHECK(sum <= ms);
        CHECK(times->spanMs <= ms);
        CHECK(raster);   // the splat pass runs in every frame, splats or none
        CHECK(shade == c.wall);

        const uint32_t words = w * h * 4;
        auto inFlight = render::countDifferent(*gpu->library, flying, waited, words);
        auto profiled = render::countDifferent(*gpu->library, timed, waited, words);
        REQUIRE(inFlight);
        REQUIRE(profiled);
        CHECK(*inFlight == 0);
        CHECK(*profiled == 0);
    }
}
