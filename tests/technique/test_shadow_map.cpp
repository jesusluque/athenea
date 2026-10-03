// Copyright (c) 2026 jesus luque.
//
// THE MAP FROM THE LIGHT, AGAINST WHAT A PRODUCT OF OPACITIES MUST BE.
//
// A cloud of N particles stacked along the light's own axis, each taking its
// opacity whole where the axis passes through its centre, lets exactly
// (1 - alpha)^N through. That is the number the map has to answer at a point
// behind the stack -- and only the accumulation of optical depth, in fixed
// point, over footprints that overlap, stands between the two.
//
// Nothing here is traced and nothing is sorted, which is the point: the map is
// built on a device with no ray tracing of any kind.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>

#include <catch2/catch_approx.hpp>

#include "../render/SplatFixtures.h"
#include "athenea/light/LightTable.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/technique/SplatShadowMap.h"

using namespace athenea;

TEST_CASE("a cloud's shadow map answers the product of the opacities along the light's axis",
          "[technique][gpu][shadowmap]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    auto map = technique::SplatShadowMap::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!table) FAIL(table.error().toString());
    if (!map) FAIL(map.error().toString());

    // A sun shining down -z: UsdLux shines down its own -Z, so the identity
    // is a light whose rays travel that way.
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.lightToWorld = render::Mat4::identity();
    sun.intensity = 1.0F;
    sun.shadow = true;
    REQUIRE(table->set(std::span<const light::Light>(&sun, 1)));

    // Four particles on the axis, and four columns of one at the corners: the
    // corners give the cloud a box wide enough for the map to have texels to
    // spare, and they are their own answer -- one opacity, not four.
    const float opacity = 0.5F;
    const uint32_t stacked = 4;
    test::CloudBuilder built;
    for (uint32_t k = 0; k < stacked; ++k) {
        built.add(0.0F, 0.0F, -2.0F * float(k), opacity, 0.1F, 0.1F, 0.1F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    }
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sy = -1; sy <= 1; sy += 2) {
            built.add(float(sx), float(sy), -3.0F, opacity, 0.1F, 0.1F, 0.1F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
        }
    }
    auto cloud = loader->upload(built.raw, 0);
    REQUIRE(cloud);

    technique::ShadowMapJob job;
    job.casters.push_back({&*cloud, &cloud->positions, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 0});
    job.lights = &table->records();
    job.lightCount = 1;
    job.resolution = 512;
    job.coefficients = 1;
    job.margin = 0.2F;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(map->build(batch, job));
        REQUIRE(batch.submit(true));
    }
    REQUIRE(map->valid());

    // Behind the stack, behind a corner, in front of everything, and beside
    // the cloud altogether.
    const std::vector<std::array<float, 4>> points{
        {0.0F, 0.0F, -20.0F, 1.0F},    // through all four
        {1.0F, 1.0F, -20.0F, 1.0F},    // through one
        {0.0F, 0.0F, 20.0F, 1.0F},     // nearer the light than anything
        {40.0F, 0.0F, -20.0F, 1.0F},   // beside the map
    };
    auto answers = map->probe(0, points);
    if (!answers) FAIL(answers.error().toString());
    const double four = std::pow(1.0 - double(opacity), double(stacked));
    const double one = 1.0 - double(opacity);
    std::printf("  shadow map: behind four %.4f (closed form %.4f), behind one %.4f (%.4f), in front %.4f, "
                "beside %.4f\n",
                double((*answers)[0]), four, double((*answers)[1]), one, double((*answers)[2]),
                double((*answers)[3]));
    CHECK((*answers)[0] == Catch::Approx(four).margin(0.01));
    CHECK((*answers)[1] == Catch::Approx(one).margin(0.01));
    CHECK((*answers)[2] == Catch::Approx(1.0F));
    CHECK((*answers)[3] == Catch::Approx(1.0F));

    // Optical depth adds, so a stack twice as tall lets the square of it
    // through. Nothing about the map's arithmetic is allowed to saturate.
    test::CloudBuilder twice;
    for (uint32_t k = 0; k < stacked * 2; ++k) {
        twice.add(0.0F, 0.0F, -1.0F * float(k), opacity, 0.1F, 0.1F, 0.1F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
    }
    for (int sx = -1; sx <= 1; sx += 2) {
        for (int sy = -1; sy <= 1; sy += 2) {
            twice.add(float(sx), float(sy), -3.0F, opacity, 0.1F, 0.1F, 0.1F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
        }
    }
    auto taller = loader->upload(twice.raw, 0);
    REQUIRE(taller);
    job.casters.clear();
    job.casters.push_back({&*taller, &taller->positions, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 0});
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(map->build(batch, job));
        REQUIRE(batch.submit(true));
    }
    auto again = map->probe(0, points);
    if (!again) FAIL(again.error().toString());
    std::printf("  twice the stack: %.4f, the square of %.4f is %.4f\n", double((*again)[0]), four, four * four);
    CHECK((*again)[0] == Catch::Approx(four * four).margin(0.005));

    // A light that casts no shadow has no map, and neither does a dome: both
    // answer that everything reaches everywhere.
    light::Light quiet = sun;
    quiet.shadow = false;
    REQUIRE(table->set(std::span<const light::Light>(&quiet, 1)));
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(map->build(batch, job));
        REQUIRE(batch.submit(true));
    }
    auto none = map->probe(0, points);
    if (!none) FAIL(none.error().toString());
    CHECK((*none)[0] == Catch::Approx(1.0F));
}
