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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

namespace {

/// A sun shining straight down -z, the one light of `table`.
void sunDown(light::LightTable& table) {
    light::Light sun;
    sun.kind = light::LightKind::Distant;
    sun.lightToWorld = render::Mat4::identity();
    sun.intensity = 1.0F;
    sun.shadow = true;
    REQUIRE(table.set(std::span<const light::Light>(&sun, 1)));
}

}   // namespace

// A FRAME THAT DOES NOT BREATHE WITH THE WINGS.
//
// A body that stays where it is and a wing that flaps: three poses of the
// wing, which make the posed box larger and smaller. The map used to be framed
// on that box, so every pose changed the size and the phase of a texel under
// the body, and the edge of the body's shadow shimmered although nothing in
// it had moved. Framed on the rest sphere, with its centre on the world's
// texel grid, the body's shadow is the same in every pose: the sums are
// integers, so the answers agree to the last bit or to one unit of fixed
// point where a float rounds the other way.
TEST_CASE("a part that does not move casts the same shadow while another part moves",
          "[technique][gpu][shadowmap][stable]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    auto map = technique::SplatShadowMap::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!table) FAIL(table.error().toString());
    if (!map) FAIL(map.error().toString());
    sunDown(*table);

    // The body: five by five gaussians over [-0.1, 0.1]^2, at z = 0. The wing:
    // three by three about where the pose puts it -- stretched out at rest,
    // folded in and up, folded in and down.
    const std::array<std::array<float, 3>, 3> wing{{{0.5F, 0.0F, 0.0F}, {0.25F, 0.0F, 0.15F}, {0.35F, 0.0F, -0.2F}}};
    std::vector<scene::GpuSplats> poses;
    for (const auto& at : wing) {
        test::CloudBuilder built;
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                built.add(0.05F * float(i), 0.05F * float(j), 0.0F, 0.6F, 0.03F, 0.03F, 0.01F, {1, 0, 0, 0},
                          {0.8F, 0.8F, 0.8F});
            }
        }
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                built.add(at[0] + 0.05F * float(i), at[1] + 0.05F * float(j), at[2], 0.6F, 0.03F, 0.03F, 0.01F,
                          {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
            }
        }
        auto cloud = loader->upload(built.raw, 0);
        REQUIRE(cloud);
        poses.push_back(std::move(*cloud));
    }
    // The box it was bound in is the stretched-out pose's, whatever pose it
    // is drawn in -- as Engine::carryCloud keeps it.
    const scene::Bounds rest = poses.front().bounds;

    // Sixteen points across the body's shadow edge at x = -0.1, on two rows:
    // well behind the cloud for the total, and just behind the body, inside
    // the slab, for the Fourier terms.
    const auto pointsAt = [](float z) {
        std::vector<std::array<float, 4>> points;
        for (int row = 0; row < 2; ++row) {
            for (int k = 0; k < 8; ++k) {
                points.push_back({-0.16F + 0.012F * float(k), 0.031F * float(row), z, 1.0F});
            }
        }
        return points;
    };
    for (const uint32_t terms : {1u, 5u}) {
        const std::vector<std::array<float, 4>> points = pointsAt(terms == 1 ? -0.5F : -0.08F);
        std::vector<std::vector<float>> answers;
        for (const scene::GpuSplats& pose : poses) {
            technique::ShadowMapJob job;
            technique::ShadowMapCaster caster;
            caster.cloud = &pose;
            caster.positions = &pose.positions;
            caster.restBounds = rest;
            job.casters.push_back(caster);
            job.lights = &table->records();
            job.lightCount = 1;
            job.resolution = 512;
            job.coefficients = terms;
            {
                gpu::CommandBatch batch(*gpu->device);
                REQUIRE(map->build(batch, job));
                REQUIRE(batch.submit(true));
            }
            auto read = map->probe(0, points);
            if (!read) FAIL(read.error().toString());
            answers.push_back(std::move(*read));
        }
        float largest = 0.0F;
        float darkest = 1.0F;
        size_t same = 0;
        for (size_t pose = 1; pose < answers.size(); ++pose) {
            for (size_t k = 0; k < points.size(); ++k) {
                const float delta = std::fabs(answers[pose][k] - answers[0][k]);
                largest = std::max(largest, delta);
                same += delta == 0.0F ? 1 : 0;
                darkest = std::min(darkest, answers[0][k]);
            }
        }
        std::printf("  %u term%s: the body's shadow over three wing poses differs by at most %.2e "
                    "(%zu of %zu answers bit for bit), darkest %.4f\n",
                    terms, terms == 1 ? "" : "s", double(largest), same, points.size() * (answers.size() - 1),
                    double(darkest));
        // The edge is an edge: something under it is in shadow.
        CHECK(darkest < 0.8F);
        // One unit of the map's fixed point is 1/4096 of optical depth.
        CHECK(largest <= 5.0e-4F);
    }
}

// A SHADOW EDGE READ BETWEEN TWO TEXELS.
//
// A sheet of gaussians smaller than a texel, ending in a straight edge that
// falls on a texel boundary (the grid is the world's since step 1), and a
// receiver walking across it from one texel's centre to the next in quarters.
// Read at the nearest texel, the answer is one texel's and then the other's:
// a step, which is what a floor under a flapping wing sparkles with. Read
// bilinearly -- by hand, four integer loads, so Metal and CUDA agree -- it is
// the straight line between the two, and halfway is the mean of the ends.
TEST_CASE("a receiver crossing a texel reads the shadow's edge continuously",
          "[technique][gpu][shadowmap][filtered]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto table = light::LightTable::create(*gpu->library);
    auto map = technique::SplatShadowMap::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!table) FAIL(table.error().toString());
    if (!map) FAIL(map.error().toString());
    sunDown(*table);

    // y < 0, x in [-0.05, 0.05], gaussians every 0.4 mm -- half a texel of
    // the map the sheet's rest sphere frames at 256 texels -- each a tenth of
    // a millimetre, so the footprint is the map's own dilation and the edge
    // is as sharp as the map can hold.
    test::CloudBuilder built;
    const float spacing = 0.0004F;
    for (int i = 0; i <= 250; ++i) {
        for (int j = 0; j <= 250; ++j) {
            built.add(-0.05F + spacing * float(i), -spacing * float(j) - 0.5F * spacing, 0.0F, 0.5F, 0.0001F,
                      0.0001F, 0.0001F, {1, 0, 0, 0}, {0.8F, 0.8F, 0.8F});
        }
    }
    auto sheet = loader->upload(built.raw, 0);
    REQUIRE(sheet);
    technique::ShadowMapJob job;
    job.casters.push_back({&*sheet, &sheet->positions, {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}, 0});
    job.lights = &table->records();
    job.lightCount = 1;
    job.resolution = 256;
    job.coefficients = 1;
    {
        gpu::CommandBatch batch(*gpu->device);
        REQUIRE(map->build(batch, job));
        REQUIRE(batch.submit(true));
    }
    auto frame = map->frameInfo(0);
    if (!frame) FAIL(frame.error().toString());
    REQUIRE(frame->valid);
    const float t = frame->texelsPerUnit;
    // The map's u and v axes in the world, and where a texel coordinate
    // lands: what puts a receiver on a texel's centre.
    const std::array<float, 3> axisU{frame->rowU[0] / t, frame->rowU[1] / t, frame->rowU[2] / t};
    const std::array<float, 3> axisV{frame->rowV[0] / t, frame->rowV[1] / t, frame->rowV[2] / t};
    const auto at = [&](float u, float v) {
        const float a = (u - frame->rowU[3]) / t;
        const float b = (v - frame->rowV[3]) / t;
        // Behind the sheet, where one term is the whole answer.
        return std::array<float, 4>{a * axisU[0] + b * axisV[0], a * axisU[1] + b * axisV[1],
                                    a * axisU[2] + b * axisV[2] - 0.5F, 1.0F};
    };
    // The edge is at y = 0; which of u and v that is depends on the frame the
    // light gets, so the walk is across whichever of them moves along y.
    const bool alongU = std::fabs(axisU[1]) > std::fabs(axisV[1]);
    const float edge = alongU ? frame->rowU[3] : frame->rowV[3];   // y = 0, in texels: a whole number
    const float across = std::round(alongU ? frame->rowV[3] : frame->rowU[3]) + 0.5F;   // a texel centre at x = 0
    // From the centre of the texel on one side of the edge to the centre of
    // the one on the other, in quarters.
    std::vector<std::array<float, 4>> points;
    for (int q = 0; q <= 4; ++q) {
        const float walk = std::round(edge) - 0.5F + 0.25F * float(q);
        points.push_back(alongU ? at(walk, across) : at(across, walk));
    }
    auto read = map->probe(0, points);
    if (!read) FAIL(read.error().toString());
    const std::vector<float>& T = *read;
    std::printf("  across the edge in quarters of a texel: %.4f %.4f %.4f %.4f %.4f\n", double(T[0]), double(T[1]),
                double(T[2]), double(T[3]), double(T[4]));
    // The two texels differ: the edge is in between.
    CHECK(std::fabs(T[0] - T[4]) > 0.2F);
    // Halfway is the mean of the two ends, and the walk is monotonic.
    CHECK(std::fabs(T[2] - 0.5F * (T[0] + T[4])) <= 0.02F);
    const float sign = T[4] > T[0] ? 1.0F : -1.0F;
    for (size_t k = 1; k < T.size(); ++k) {
        CHECK(sign * (T[k] - T[k - 1]) >= -1.0e-5F);
    }
}
