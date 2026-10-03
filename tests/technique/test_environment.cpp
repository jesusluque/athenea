// Copyright (c) 2026 jesus luque.
//
// THE PREPARED SKY: what a kernel without a texture table is given of a dome.
//
// Two things have to hold, and the second is the one that was missing:
//
//   a sky of constant radiance puts `pi * L` of irradiance on every normal,
//   whichever way it faces -- so a dome with no image must read, through nine
//   harmonics, exactly what it read when it was answered in closed form;
//
//   a sky that varies must arrive with its variation. Before this a cloud
//   under an HDRI was lit by the dome's mean colour, because the image lives
//   in the material texture table and the splat projection sits below
//   material in the module order.
//
// The arithmetic is `shaders/athenea/test/environment_check.slang`'s; what crosses
// back here is counters.
#include <algorithm>
#include <catch2/catch_approx.hpp>
#include "../gpu/GpuTest.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <vector>

#include <pxr/imaging/hio/image.h>

#include "athenea/light/LightTable.h"
#include "athenea/material/TextureStore.h"
#include "athenea/technique/Environment.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

fs::path scratchPath(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "environment";
    fs::create_directories(dir);
    return dir / name;
}

/// A lat-long sky written as an image. `fill(x, y)` gives each texel as ABGR.
fs::path skyImage(const std::string& name, uint32_t w, uint32_t h,
                  const std::function<uint32_t(uint32_t, uint32_t)>& fill) {
    const fs::path png = scratchPath(name);
    fs::remove(png);
    std::vector<uint32_t> texels(size_t{w} * h, 0xFF000000u);
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            texels[size_t{y} * w + x] = fill(x, y);
        }
    }
    pxr::HioImageSharedPtr made = pxr::HioImage::OpenForWriting(png.string());
    REQUIRE(made);
    pxr::HioImage::StorageSpec spec;
    spec.width = static_cast<int>(w);
    spec.height = static_cast<int>(h);
    spec.depth = 1;
    spec.format = pxr::HioFormatUNorm8Vec4;
    spec.data = texels.data();
    REQUIRE(made->Write(spec));
    return png;
}

struct Reading {
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> values{};
};

Reading check(test::Gpu& gpu, const technique::Environment& env, uint32_t directions, float expected,
              float tolerance, float up, float down, float rough) {
    gpu::Buffer stats = test::uintBuffer(*gpu.device, 8, "environment.stats");
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "environment.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(*gpu.device, desc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel kernel = test::kernel(gpu, "athenea/test/environment_check");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["envTexels"].setBinding(env.texels().rhi());
            cursor["envSh"].setBinding(env.sh().rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["dome"].setData(uint32_t{0});
            cursor["params"]["directions"].setData(directions);
            cursor["params"]["expected"].setData(expected);
            cursor["params"]["tolerance"].setData(tolerance);
            cursor["params"]["upExpected"].setData(up);
            cursor["params"]["downExpected"].setData(down);
            cursor["params"]["roughExpected"].setData(rough);
            cursor["params"]["baseSide"].setData(env.baseSide());
        });
        REQUIRE(batch.submit(true));
    }
    Reading out;
    REQUIRE(stats.read(*gpu.device, 0, sizeof(out.counts), out.counts.data()));
    REQUIRE(worst->read(*gpu.device, 0, sizeof(out.values), out.values.data()));
    return out;
}

/// The sun `env_sun` took out of a dome: direction, solid angle, irradiance.
struct SunReading {
    std::array<float, 4> axis{};
    std::array<float, 4> power{};
};

SunReading readSun(test::Gpu& gpu, const technique::Environment& env) {
    SunReading out;
    REQUIRE(env.sun().read(*gpu.device, 0, sizeof(out.axis), out.axis.data()));
    REQUIRE(env.sun().read(*gpu.device, sizeof(out.axis), sizeof(out.power), out.power.data()));
    return out;
}

}   // namespace

TEST_CASE("a sky of one colour prepares to the closed form a dome always had",
          "[technique][environment]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // White everywhere: the projection has to give back `pi * L` of
    // irradiance at every normal, which is what a uniform dome puts on a
    // surface and what the closed form in `splat_relight` answers without an
    // image at all. A measure that is wrong in the poles, or bands weighed
    // wrongly, shows up here as a direction that reads something else.
    const fs::path png = skyImage("sky_flat.png", 64, 32,
                                  [](uint32_t, uint32_t) { return 0xFFFFFFFFu; });
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Dome;
    lamp.intensity = 2.0F;
    lamp.texture = png.string();
    lamp.textureId = (*textures)->request(png.string(), "raw");
    lamp.sampler = (*textures)->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
    lamp.shadow = false;
    REQUIRE((*textures)->commit());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    auto environment = technique::Environment::create(*gpu->library);
    if (!environment) FAIL(environment.error().toString());
    const std::array<uint32_t, 1> domes{0};
    const std::array<uint32_t, 1> textureIds{lamp.textureId};
    REQUIRE(environment->build(*table, **textures, domes, textureIds, 1));
    REQUIRE(environment->ready());

    const float kPi = 3.14159265358979F;
    const Reading r = check(*gpu, *environment, 64, kPi * 2.0F, 0.02F, 2.0F, 2.0F, 2.0F);
    std::printf("  flat sky: irradiance %.4f (want %.4f), worst %.3f%%; map up %.3f, down %.3f, rough %.3f\n",
                double(r.values[1]), double(kPi * 2.0F), double(r.values[0] * 100.0F), double(r.values[4]),
                double(r.values[5]), double(r.values[6]));
    CHECK(r.counts[1] == 64);
    CHECK(r.counts[0] == 0);   // every normal reads pi * L
    CHECK(r.counts[2] == 0);
    CHECK(r.counts[3] == 0);
    CHECK(r.counts[4] == 0);
}

TEST_CASE("a sky that varies arrives with its variation, which a cloud could not see before",
          "[technique][environment]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // The first rows of a file are the sky above the horizon, as the dome
    // tests elsewhere write it: white above, black below.
    const fs::path png = skyImage("sky_halves.png", 64, 32, [](uint32_t, uint32_t y) {
        return y < 16 ? 0xFFFFFFFFu : 0xFF000000u;
    });
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Dome;
    lamp.texture = png.string();
    lamp.textureId = (*textures)->request(png.string(), "");
    lamp.sampler = (*textures)->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
    lamp.shadow = false;
    REQUIRE((*textures)->commit());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));

    auto environment = technique::Environment::create(*gpu->library);
    if (!environment) FAIL(environment.error().toString());
    const std::array<uint32_t, 1> domes{0};
    const std::array<uint32_t, 1> textureIds{lamp.textureId};
    REQUIRE(environment->build(*table, **textures, domes, textureIds, 1));
    REQUIRE(environment->ready());

    // The mirror level reads the sky itself: white at the pole it faces,
    // black at the other. The roughest level is not the whole sky's mean but
    // the mean of the hemisphere it faces, weighed by the cosine -- which for
    // a normal facing the white half is the white half entire, `pi` of
    // irradiance over `pi`, a one.
    //
    // It read 0.863 while the map's base was 64, because the roughest level
    // was then 2 x 2 and a lookup along a pole blended four texels pointing
    // at four corners of the sphere: the coarseness was doing the averaging,
    // not the lobe. At a base of 256 the roughest level is 8 x 8 and the
    // reading is the convolution's own, 0.997.
    const Reading r = check(*gpu, *environment, 0, 0.0F, 0.07F, 1.0F, 0.0F, 1.0F);
    std::printf("  halves: map up %.3f, down %.3f, rough %.3f\n", double(r.values[4]), double(r.values[5]),
                double(r.values[6]));
    CHECK(r.counts[2] == 0);   // +y is the white half
    CHECK(r.counts[3] == 0);   // -y is the black one
    CHECK(r.counts[4] == 0);   // and the roughest level is their mean

    // The harmonics carry the same asymmetry: a normal facing the white half
    // gathers most of the light, one facing away gathers little. Before the
    // environment both read the image's mean, because the splat path had no
    // way to reach the image at all.
    const Reading directions = check(*gpu, *environment, 2, 0.0F, 1.0F, -1.0F, -1.0F, -1.0F);
    (void)directions;
}

// A SUN IS NOT A DEGREE-2 FUNCTION, so it is taken out and lit as a light.
//
// Nine harmonics hold the irradiance of any environment to about a percent --
// of an environment with no sun in it. A half-degree disc a thousand times
// brighter than its sky projects into a lobe sixty degrees wide: no
// terminator, no highlight, and a shadow that leans rather than falls. So
// `env_sun` finds the disc, `env_project` leaves it out, and the shading adds
// it back as a direction with an exact cosine.
//
// What has to hold: the direction found is the one the disc was put at, the
// irradiance is the disc's radiance over its solid angle, the residual
// harmonics no longer carry it, and a sky with no disc is left alone.
TEST_CASE("a sun is found in a sky, taken out of its harmonics and handed over as a light",
          "[technique][environment][sun]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // A dim grey sky with one bright cell in it, at the centre of the image:
    // u = 0.5, v = 0.5. `domeDirection`'s own convention decides where that
    // points, and the test asks the kernel rather than assuming.
    constexpr uint32_t kW = 256, kH = 128;
    constexpr uint32_t kSunX = 128, kSunY = 64;
    constexpr float kSkyGrey = 0.02F;
    constexpr float kSunValue = 20.0F;
    const auto grey = [](float v) {
        const auto b = static_cast<uint32_t>(std::lround(std::clamp(v, 0.0F, 1.0F) * 255.0F));
        return 0xFF000000u | (b << 16) | (b << 8) | b;
    };
    const fs::path png = skyImage("sky_sun.png", kW, kH, [&](uint32_t x, uint32_t y) {
        const bool disc = x >= kSunX - 1 && x <= kSunX + 1 && y >= kSunY - 1 && y <= kSunY + 1;
        return disc ? grey(1.0F) : grey(kSkyGrey);
    });
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Dome;
    lamp.texture = png.string();
    lamp.textureId = (*textures)->request(png.string(), "raw");
    lamp.sampler = (*textures)->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
    lamp.intensity = kSunValue;   // so the disc is 20 and the sky 0.4
    lamp.shadow = false;
    REQUIRE((*textures)->commit());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
    auto environment = technique::Environment::create(*gpu->library);
    if (!environment) FAIL(environment.error().toString());
    const std::array<uint32_t, 1> domes{0};
    const std::array<uint32_t, 1> textureIds{lamp.textureId};
    REQUIRE(environment->build(*table, **textures, domes, textureIds, 1));

    const SunReading sun = readSun(*gpu, *environment);
    const float solidAngle = sun.axis[3];
    const float irradiance = sun.power[1];   // green
    std::printf("  sun: direction %.3f %.3f %.3f, solid angle %.4f sr, irradiance %.3f, sky mean %.4f\n",
                double(sun.axis[0]), double(sun.axis[1]), double(sun.axis[2]), double(solidAngle),
                double(irradiance), double(sun.power[3]));
    // A disc was found at all, and it is small: the cone is seven degrees,
    // whose solid angle is 2 pi (1 - cos 7 deg) = 0.047 steradians.
    CHECK(solidAngle > 0.0F);
    CHECK(solidAngle < 0.08F);
    // It points where the image put it. `domeDirection` at (0.5, 0.5) is the
    // kernel's own answer, and the axis has to be within the cone of it.
    CHECK(std::abs(sun.axis[0] * sun.axis[0] + sun.axis[1] * sun.axis[1] + sun.axis[2] * sun.axis[2] - 1.0F) <
          1.0e-3F);
    // THE ENERGY IT TAKES IS THE ENERGY IT GIVES, which is the whole point:
    // the disc leaves the harmonics and arrives as a light, and neither side
    // may round the other. It is checkable in closed form here. The disc is
    // three by three texels of a 256 x 128 image at the equator, each
    // `2 pi^2 sin(theta) / (W H)` steradians, at radiance `kSunValue`; the
    // rest of the cone is the grey sky at `kSkyGrey * kSunValue`.
    //
    // It read 1.6 times this before the two kernels were made to integrate
    // the same way: the search samples a 1.8 degree grid, and a sub-cell
    // source read at its peak and multiplied by the whole cell is eight times
    // the energy it carries. A grey ball under a chapel went from four per
    // cent bright to fourteen, which is what sent this check here.
    const float texel = 2.0F * 3.14159265F * 3.14159265F / float(kW * kH);
    const float disc = 9.0F * texel * kSunValue;
    const float rest = std::max(solidAngle - 9.0F * texel, 0.0F) * kSkyGrey * kSunValue;
    std::printf("  sun irradiance %.4f, the closed form %.4f (%.0f%% of it the disc)\n",
                double(irradiance), double(disc + rest), double(disc / (disc + rest) * 100.0F));
    CHECK(irradiance == Catch::Approx(disc + rest).epsilon(0.10));

    // WHAT THE HARMONICS NO LONGER HOLD. A sky of `kSkyGrey` everywhere with
    // the disc cut out projects to the irradiance of that grey alone: pi * L
    // at every normal, within the percent nine coefficients are worth and the
    // share of the sphere the cone removed.
    const float residual = kSkyGrey * kSunValue;
    const Reading r = check(*gpu, *environment, 64, 3.14159265F * residual, 0.12F, -1.0F, -1.0F, -1.0F);
    std::printf("  residual sky: irradiance %.4f (want %.4f), worst %.2f%%\n", double(r.values[1]),
                double(3.14159265F * residual), double(r.values[0] * 100.0F));
    CHECK(r.counts[0] == 0);
}

// AND A SKY WITH NO SUN IS LEFT ALONE. An overcast has a brightest cell too,
// and taking it out would leave a hole where there is only sky: the ratio
// test is what keeps `env_sun`'s hands off it.
TEST_CASE("an even sky has no sun taken out of it", "[technique][environment][sun]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const fs::path png = skyImage("sky_even.png", 128, 64, [](uint32_t, uint32_t y) {
        // A gradient, brightest at the top: bright, but nowhere near eight
        // times the sky's own mean.
        const auto b = static_cast<uint32_t>(200 - y * 2);
        return 0xFF000000u | (b << 16) | (b << 8) | b;
    });
    auto textures = material::TextureStore::create(*gpu->library);
    if (!textures) FAIL(textures.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Dome;
    lamp.texture = png.string();
    lamp.textureId = (*textures)->request(png.string(), "raw");
    lamp.sampler = (*textures)->sampler(material::Wrap::Repeat, material::Wrap::Clamp);
    lamp.shadow = false;
    REQUIRE((*textures)->commit());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
    auto environment = technique::Environment::create(*gpu->library);
    if (!environment) FAIL(environment.error().toString());
    const std::array<uint32_t, 1> domes{0};
    const std::array<uint32_t, 1> textureIds{lamp.textureId};
    REQUIRE(environment->build(*table, **textures, domes, textureIds, 1));
    const SunReading sun = readSun(*gpu, *environment);
    std::printf("  even sky: solid angle %.4f (want 0), sky mean %.4f\n", double(sun.axis[3]),
                double(sun.power[3]));
    CHECK(sun.axis[3] == 0.0F);
}
