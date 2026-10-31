// Copyright (c) 2026 jesus luque.
//
// A CLOUD SOLIDIFIED (SSF phase 2): a sphere of flat gaussians becomes blobs
// on the sphere with radial normals; gaussians that are not flat, and flat
// ones alone and turned anyhow, stay residuals; and the layer drawn again as
// flat gaussians (Mode A) is the picture the cloud was. Inputs are a
// fixture's records; everything checked is counted by kernels.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/io/Exr.h"
#include "athenea/lod/Solidify.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;

namespace {

constexpr uint32_t kSurface = 12000;   // flat on the unit sphere, first
constexpr uint32_t kFuzz = 1500;       // round, in a shell round it
constexpr uint32_t kLoose = 300;       // flat, turned anyhow, far apart

std::array<float, 4> turningZTo(float nx, float ny, float nz) {
    // The shortest turn from +z to n, as w x y z.
    const float w = 1.0F + nz;
    if (w < 1e-6F) {
        return {0.0F, 1.0F, 0.0F, 0.0F};
    }
    const float len = std::sqrt(w * w + ny * ny + nx * nx);
    return {w / len, -ny / len, nx / len, 0.0F};
}

test::CloudBuilder sphereCloud() {
    test::CloudBuilder b;
    test::Lcg rng{97};
    const float spacing = std::sqrt(4.0F * 3.14159265F / kSurface);
    for (uint32_t i = 0; i < kSurface; ++i) {
        const float z = 1.0F - (2.0F * i + 1.0F) / kSurface;
        const float phi = i * 3.14159265F * (3.0F - std::sqrt(5.0F));
        const float rho = std::sqrt(std::max(1.0F - z * z, 0.0F));
        const float x = rho * std::cos(phi), y = rho * std::sin(phi);
        const float s = 0.6F * spacing;
        b.add(x, y, z, 0.95F, s, s, 0.05F * s, turningZTo(x, y, z),
              {0.5F + 0.4F * x, 0.5F + 0.4F * y, 0.5F + 0.4F * z});
    }
    for (uint32_t i = 0; i < kFuzz; ++i) {
        // Round blurs in a shell from 1.3 to 1.6.
        float x = rng.range(-1, 1), y = rng.range(-1, 1), z = rng.range(-1, 1);
        const float r = std::sqrt(x * x + y * y + z * z) + 1e-6F, at = rng.range(1.3F, 1.6F) / r;
        b.add(x * at, y * at, z * at, 0.6F, 0.02F, 0.02F, 0.02F, {1, 0, 0, 0}, {0.9F, 0.2F, 0.2F});
    }
    for (uint32_t i = 0; i < kLoose; ++i) {
        // Flat but alone, far out, turned anyhow.
        float x = rng.range(-1, 1), y = rng.range(-1, 1), z = rng.range(-1, 1);
        const float r = std::sqrt(x * x + y * y + z * z) + 1e-6F, at = rng.range(2.5F, 4.0F) / r;
        const float a = rng.range(-1, 1), c = rng.range(-1, 1), d = rng.range(-1, 1), e = rng.range(-1, 1);
        const float q = std::sqrt(a * a + c * c + d * d + e * e) + 1e-6F;
        b.add(x * at, y * at, z * at, 0.9F, 0.02F, 0.02F, 0.001F, {a / q, c / q, d / q, e / q}, {0.2F, 0.9F, 0.2F});
    }
    return b;
}

std::array<uint32_t, 4> counted(test::Gpu& gpu, const char* entry, uint32_t count,
                                const std::function<void(rhi::ShaderCursor)>& bind) {
    auto kernel = gpu::ComputeKernel::create(*gpu.library, "athenea/test/ssf_solidify_check", entry);
    if (!kernel) FAIL(kernel.error().toString());
    gpu::Buffer wrong = test::uintBuffer(*gpu.device, 4, "test.ssf.wrong");
    gpu::Buffer none = test::uintBuffer(*gpu.device, 4, "test.ssf.none");
    gpu::CommandBatch batch(*gpu.device);
    kernel->dispatch(batch, {std::max(count, 1u), 1, 1}, [&](rhi::ShaderCursor c) {
        c["sources"].setBinding(none.rhi());
        c["blobPositions"].setBinding(none.rhi());
        c["blobNormals"].setBinding(none.rhi());
        c["wrong"].setBinding(wrong.rhi());
        c["params"]["count"].setData(count);
        bind(c);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 4> counts{};
    REQUIRE(wrong.read(*gpu.device, 0, sizeof(counts), counts.data()));
    return counts;
}

}   // namespace

TEST_CASE("a sphere of flat gaussians solidifies onto the sphere, and draws as it was", "[lod][gpu][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto solidifier = lod::Solidifier::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!solidifier) FAIL(solidifier.error().toString());
    if (!raster) FAIL(raster.error().toString());
    test::CloudBuilder built = sphereCloud();
    auto cloud = loader->upload(built.raw, 0);
    REQUIRE(cloud);
    REQUIRE(cloud->count == kSurface + kFuzz + kLoose);
    const lod::SolidifySettings settings;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const Clock::time_point t0 = Clock::now();
    auto result = solidifier->solidify(*cloud, settings);
    const Clock::time_point t1 = Clock::now();
    if (!result) FAIL(result.error().toString());
    std::printf("  solidify: %.1f ms (this build, every stage waited for)\n", ms(t0, t1));
    std::printf("  %u gaussians: %u candidates, %u absorbed (%.1f %%), %u blobs, %u residuals\n", result->gaussians,
                result->candidates, result->coherent, 100.0 * result->absorbed(), result->solid.blobs,
                result->residuals);

    // What became what. The sphere's gaussians are the first kSurface.
    const auto surfaceIn = [&](const gpu::Buffer& list, uint32_t count) {
        return counted(*gpu, "ssfSourcesCheck", count, [&](rhi::ShaderCursor c) {
            c["sources"].setBinding(list.rhi());
            c["params"]["surface"].setData(kSurface);
        });
    };
    const std::array<uint32_t, 4> solid = surfaceIn(result->solidSource, result->solid.blobs);
    const std::array<uint32_t, 4> residual = surfaceIn(result->residualSource, result->residuals);
    std::printf("  blobs from the sphere: %u of %u; residuals from the sphere: %u of %u\n", solid[0], solid[1],
                residual[0], residual[1]);
    CHECK(solid[1] == result->solid.blobs);
    CHECK(solid[0] == solid[1]);                   // nothing but the sphere solidifies
    CHECK(residual[1] == result->residuals);
    CHECK(result->residuals - residual[0] == kFuzz + kLoose);   // and all the rest stays
    CHECK(result->coherent >= kSurface * 95 / 100);

    // The blobs on the sphere, their normals radial. A normal is the field's
    // gradient on the outer zero set, which ripples a little between blobs.
    const auto sphereCheck = [&](float lean) {
        return counted(*gpu, "ssfSphereCheck", result->solid.blobs, [&](rhi::ShaderCursor c) {
            c["blobPositions"].setBinding(result->solid.positions.rhi());
            c["blobNormals"].setBinding(result->normals.rhi());
            c["params"]["off"].setData(2.0e-3F);
            c["params"]["lean"].setData(lean);
        });
    };
    const std::array<uint32_t, 4> sphere = sphereCheck(1.0e-3F);   // 2.6 degrees
    const std::array<uint32_t, 4> wide = sphereCheck(1.5e-2F);     // 10 degrees
    std::printf("  blobs off the sphere by more than 2e-3: %u; normals off the radius by more than 2.6 deg: %u, "
                "10 deg: %u, of %u; worst off %.2e\n",
                sphere[0], sphere[1], wide[1], sphere[2], sphere[3] * 1.0e-6);
    CHECK(sphere[2] == result->solid.blobs);
    CHECK(sphere[0] == 0);
    CHECK(sphere[1] <= result->solid.blobs / 100);
    CHECK(wide[1] == 0);

    // Mode A against the cloud itself.
    auto splats = solidifier->toSplats(*result, *cloud, settings);
    if (!splats) FAIL(splats.error().toString());
    REQUIRE(splats->count == result->solid.blobs + result->residuals);
    render::RenderSettings rs;
    rs.width = 320;
    rs.height = 240;
    for (const auto& eye : {std::array<double, 3>{0.0, 0.4, 4.0}, std::array<double, 3>{3.0, 1.5, 2.0}}) {
        const render::Projection projection = render::projectionFor(
            render::Camera::lookingAt({eye[0], eye[1], eye[2]}, {0.0, 0.0, 0.0}), rs.width, rs.height);
        render::RenderTargets a, b;
        // Each drawn five times; the median, after one to warm.
        const auto timed = [&](const scene::GpuSplats& c, render::RenderTargets& t) {
            std::vector<double> times;
            for (int k = 0; k < 6; ++k) {
                const Clock::time_point r0 = Clock::now();
                REQUIRE(raster->render(projection, std::vector<render::SplatInstance>{{&c, render::Mat4::identity()}},
                                       rs, t));
                if (k > 0) times.push_back(ms(r0, Clock::now()));
            }
            std::sort(times.begin(), times.end());
            return times[times.size() / 2];
        };
        const double original = timed(*cloud, a);
        const double modeA = timed(*splats, b);
        std::printf("  frame (median of 5, this build): original %.2f ms, mode A %.2f ms\n", original, modeA);
        if (const char* dump = std::getenv("ATHENEA_TEST_DUMP"); dump != nullptr) {
            static int serial = 0;
            ++serial;
            for (const auto& [t, name] : {std::pair{&a, "original"}, std::pair{&b, "modeA"}}) {
                auto c = t->colour.readAll<float>(*gpu->device);
                REQUIRE(c);
                REQUIRE(io::writeExr(std::string(dump) + "/ssf" + std::to_string(serial) + "_" + name + ".exr",
                                     rs.width, rs.height, *c));
            }
        }
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, rs.width, rs.height);
        REQUIRE(diff);
        std::printf("  mode A from (%.1f, %.1f, %.1f): p99 %u, max %u, %llu over 2\n", eye[0], eye[1], eye[2],
                    diff->p99, diff->max, static_cast<unsigned long long>(diff->over2));
        CHECK(diff->p99 <= 4);
    }
}
