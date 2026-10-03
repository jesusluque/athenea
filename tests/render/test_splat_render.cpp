// Copyright (c) 2026 jesus luque.
//
// The tile rasteriser against the GPU reference, and against arithmetic.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstring>

#include <cmath>
#include <cstdio>
#include <vector>

#include "athenea/core/Hash.h"
#include "athenea/io/RawSplats.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/SplatQuery.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/light/LightTable.h"
#include "athenea/render/GaussianRayTracer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/scene/SplatSkinner.h"

using namespace athenea;
using test::CloudBuilder;
using test::randomCloud;

namespace {

struct Harness {
    test::Gpu*                 gpu;
    scene::CloudLoader         loader;
    render::TileRasterizer     raster;
    render::ReferenceRenderer  reference;
};

std::unique_ptr<Harness> harness(test::Gpu* gpu) {
    auto loader = scene::CloudLoader::create(*gpu->library);
    auto raster = render::TileRasterizer::create(*gpu->library);
    auto reference = render::ReferenceRenderer::create(*gpu->library);
    if (!loader) FAIL(loader.error().toString());
    if (!raster) FAIL(raster.error().toString());
    if (!reference) FAIL(reference.error().toString());
    return std::unique_ptr<Harness>(
        new Harness{gpu, std::move(*loader), std::move(*raster), std::move(*reference)});
}

render::ImageDifference compareToReference(Harness& h, const render::Camera& camera,
                                           std::span<const render::SplatInstance> instances,
                                           const render::RenderSettings& settings) {
    render::RenderTargets ours;
    render::RenderTargets truth;
    auto stats = h.raster.render(camera, instances, settings, ours);
    if (!stats) FAIL(stats.error().toString());
    auto overflow = h.reference.render(camera, instances, settings, truth);
    if (!overflow) FAIL(overflow.error().toString());
    CHECK(*overflow == 0);
    auto diff = render::compareImages(*h.gpu->library, ours.colour, truth.colour, settings.width,
                                      settings.height);
    if (!diff) FAIL(diff.error().toString());
    std::printf("  %u splats, %u pairs: p99 %u, max %u, %llu pixels over 2 of %llu\n",
                stats->splats, stats->pairs, diff->p99, diff->max,
                static_cast<unsigned long long>(diff->over2),
                static_cast<unsigned long long>(diff->pixels));
    return *diff;
}

std::vector<float> readColour(test::Gpu& gpu, const render::RenderTargets& targets) {
    auto values = targets.colour.readAll<float>(*gpu.device);
    REQUIRE(values);
    return std::move(*values);
}

}   // namespace

TEST_CASE("the tile rasteriser renders what the GPU reference renders", "[render][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(3000, 7);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};

    render::RenderSettings settings;
    // Not multiples of 16: the last tile row and column are partial.
    settings.width = 250;
    settings.height = 190;

    SECTION("perspective") {
        render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
        camera.lens.focal = 30.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
        CHECK(diff.max <= 16);
    }
    SECTION("close enough that splats cross the frustum edge") {
        render::Camera camera = render::Camera::lookingAt({0.3, -0.2, 2.2}, {0.0, 0.0, 0.0});
        camera.lens.focal = 18.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("orthographic") {
        render::Camera camera = render::Camera::lookingAt({0.0, 3.0, 5.0}, {0.0, 0.0, 0.0});
        camera.lens.projection = render::Lens::Projection::Orthographic;
        camera.lens.focal = 5.0;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("no antialiasing, a window offset and a rolled film back") {
        render::Camera camera = render::Camera::lookingAt({-2.0, 0.5, 5.0}, {0.0, 0.0, 0.0});
        camera.lens.windowTranslate[0] = 0.2;
        camera.lens.windowScale[1] = 1.3;
        camera.lens.windowRoll = 25.0;
        settings.antialias = false;
        const auto diff = compareToReference(*h, camera, instances, settings);
        CHECK(diff.p99 <= 2);
    }
    SECTION("two instances of the cloud, one turned, scaled and moved") {
        std::vector<render::SplatInstance> two = instances;
        two.push_back({&*cloud, aofx::xform::translation({0.5, 0.0, -1.5}) *
                                    aofx::xform::rotationY(40.0) *
                                    aofx::xform::scaling({1.0, 0.7, 1.3})});
        render::Camera camera = render::Camera::lookingAt({2.0, 1.5, 7.0}, {0.0, 0.0, -0.5});
        const auto diff = compareToReference(*h, camera, two, settings);
        CHECK(diff.p99 <= 2);
    }
}

TEST_CASE("a single splat lands on the pixel the camera arithmetic says", "[render][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(1.0F, 0.5F, -10.0F, 0.9F, 0.02F, 0.02F, 0.02F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::Camera camera;   // at the origin, looking down -Z
    camera.lens.focal = camera.lens.haperture;   // fx = width
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    render::RenderTargets targets;
    REQUIRE(h->raster.render(camera, instances, settings, targets));
    const auto pixels = readColour(*gpu, targets);
    size_t best = 0;
    for (size_t i = 0; i < pixels.size() / 4; ++i) {
        if (pixels[i * 4 + 3] > pixels[best * 4 + 3]) {
            best = i;
        }
    }
    // x: 160 + 320 * 1 / 10 = 192; y: 120 + 320 * 0.5 / 10 = 136, bottom-up.
    const int x = static_cast<int>(best % 320);
    const int y = static_cast<int>(best / 320);
    CHECK(std::abs(x - 192) <= 1);
    CHECK(std::abs(y - 136) <= 1);
}

TEST_CASE("the nearer splat wins from either side, and left stays left", "[render][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(0.0F, 0.0F, -1.0F, 0.99F, 0.5F, 0.5F, 0.05F, {1, 0, 0, 0}, {1, 0, 0});   // red, z = -1
    b.add(0.0F, 0.0F, 1.0F, 0.99F, 0.5F, 0.5F, 0.05F, {1, 0, 0, 0}, {0, 0, 1});    // blue, z = +1
    b.add(-3.0F, 0.0F, 0.0F, 0.99F, 0.3F, 0.3F, 0.3F, {1, 0, 0, 0}, {0, 1, 0});    // green, -x
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 100;

    const auto centre = [&](const std::vector<float>& p) {
        const size_t at = (50 * 200 + 100) * 4;
        return std::array<float, 3>{p[at], p[at + 1], p[at + 2]};
    };
    const auto brightestX = [&](const std::vector<float>& p, int channel) {
        int bestX = 0;
        float best = -1.0F;
        for (int x = 0; x < 200; ++x) {
            const float v = p[static_cast<size_t>((50 * 200 + x) * 4 + channel)];
            if (v > best) {
                best = v;
                bestX = x;
            }
        }
        return bestX;
    };

    render::RenderTargets targets;
    // From +Z, looking down -Z: blue (z = +1) is in front; -x is on the left.
    render::Camera front = render::Camera::lookingAt({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});
    REQUIRE(h->raster.render(front, instances, settings, targets));
    auto pixels = readColour(*gpu, targets);
    CHECK(centre(pixels)[2] > 0.9F);
    CHECK(centre(pixels)[0] < 0.1F);
    CHECK(brightestX(pixels, 1) < 100);

    // From -Z, looking down +Z: red is in front; -x is now on the right.
    render::Camera back = render::Camera::lookingAt({0.0, 0.0, -10.0}, {0.0, 0.0, 0.0});
    REQUIRE(h->raster.render(back, instances, settings, targets));
    pixels = readColour(*gpu, targets);
    CHECK(centre(pixels)[0] > 0.9F);
    CHECK(centre(pixels)[2] < 0.1F);
    CHECK(brightestX(pixels, 1) > 100);
}

TEST_CASE("antialiasing pays back the energy dilation adds to a sub-pixel splat", "[render][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder b;
    b.add(0.0F, 0.0F, -50.0F, 0.9F, 0.002F, 0.002F, 0.002F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    render::Camera camera;
    render::RenderSettings settings;
    settings.width = 64;
    settings.height = 64;
    const auto energy = [&](bool antialias) {
        settings.antialias = antialias;
        render::RenderTargets targets;
        REQUIRE(h->raster.render(camera, instances, settings, targets));
        const auto p = readColour(*gpu, targets);
        double sum = 0.0;
        for (size_t i = 0; i < p.size() / 4; ++i) {
            sum += static_cast<double>(p[i * 4 + 3]);
        }
        return sum;
    };
    const double with = energy(true);
    const double without = energy(false);
    CHECK(without > 0.0);
    CHECK(with < 0.6 * without);
}

TEST_CASE("a SplatEdit renders as the GPU reference renders it", "[render][gpu][edit]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder built = randomCloud(3000, 29);
    auto cloud = h->loader.upload(built.raw, 3);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    render::SplatEdit edit;
    edit.active = true;

    SECTION("keep what is inside a box") {
        edit.mode = render::SplatEdit::Mode::Keep;
        edit.centre = {0.5F, 0.0F, 0.0F};
        edit.size = {1.0F, 0.8F, 1.5F};
    }
    SECTION("remove what is inside a sphere, and the grade does nothing to the rest") {
        edit.mode = render::SplatEdit::Mode::Remove;
        edit.shape = render::SplatEdit::Shape::Sphere;
        edit.size = {1.2F, 0.0F, 0.0F};
        edit.tint = {0.2F, 1.0F, 1.0F};
    }
    SECTION("grade inside an inverted box: tint, saturation, brightness, opacity") {
        edit.mode = render::SplatEdit::Mode::Grade;
        edit.invert = true;
        edit.size = {1.0F, 1.0F, 1.0F};
        edit.tint = {1.0F, 0.6F, 0.3F};
        edit.saturation = 0.3F;
        edit.brightness = 1.4F;
        edit.opacity = 0.5F;
    }
    SECTION("the haze filters, wherever the volume is") {
        edit.mode = render::SplatEdit::Mode::Grade;
        edit.size = {100.0F, 100.0F, 100.0F};
        edit.minOpacity = 0.4F;
        edit.maxScale = 0.2F;
    }
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity(), edit}};
    const auto diff = compareToReference(*h, camera, instances, settings);
    CHECK(diff.p99 <= 2);

    // And the edit changed the picture.
    render::RenderTargets edited, plain;
    REQUIRE(h->raster.render(camera, instances, settings, edited));
    const std::vector<render::SplatInstance> untouched{{&*cloud, render::Mat4::identity()}};
    REQUIRE(h->raster.render(camera, untouched, settings, plain));
    auto changed = render::compareImages(*gpu->library, edited.colour, plain.colour, settings.width, settings.height);
    REQUIRE(changed);
    CHECK(changed->over2 > changed->pixels / 20);
}

TEST_CASE("a SplatEdit is in the cloud's own space and per instance", "[render][gpu][edit]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // One opaque white splat at the cloud's origin.
    CloudBuilder b;
    b.add(0.0F, 0.0F, 0.0F, 0.95F, 0.3F, 0.3F, 0.3F, {1, 0, 0, 0}, {1, 1, 1});
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    render::RenderSettings settings;
    settings.width = 65;
    settings.height = 65;
    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 5.0}, {0.0, 0.0, 0.0});
    // Two instances side by side; the edit's box is around the cloud origin,
    // which the left instance's transform moves with it.
    render::SplatEdit keepNothing;
    keepNothing.active = true;
    keepNothing.mode = render::SplatEdit::Mode::Keep;
    keepNothing.centre = {5.0F, 0.0F, 0.0F};   // far from the splat: nothing kept
    render::SplatEdit greenOnly;
    greenOnly.active = true;
    greenOnly.mode = render::SplatEdit::Mode::Grade;
    greenOnly.size = {1.0F, 1.0F, 1.0F};
    greenOnly.tint = {0.0F, 1.0F, 0.0F};
    const std::vector<render::SplatInstance> instances{
        {&*cloud, aofx::xform::translation({-1.0, 0.0, 0.0}), keepNothing},
        {&*cloud, aofx::xform::translation({1.0, 0.0, 0.0}), greenOnly}};
    render::RenderTargets targets;
    REQUIRE(h->raster.render(camera, instances, settings, targets));
    const auto p = readColour(*gpu, targets);
    // 65 px over a 24.576 aperture at 50mm is 132 px per unit at distance 1:
    // x = +-1 at 5 units lands 26 px either side of the centre.
    const auto at = [&](int x, int y) { return &p[static_cast<size_t>((y * 65 + x) * 4)]; };
    const float* left = at(32 - 26, 32);
    const float* right = at(32 + 26, 32);
    CHECK(left[3] < 0.01F);              // removed
    CHECK(right[3] > 0.5F);
    CHECK(right[0] < 0.01F);             // tinted green
    CHECK(right[2] < 0.01F);
    CHECK(right[1] > 0.3F);
}

TEST_CASE("both routes relight a splat, and alike: what the rasteriser does the ray tracer does",
          "[render][splats][relight][rt]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rayQuery || !gpu->device->caps().accelerationStructure) {
        SKIP("no ray queries on this device");
    }
    auto h = harness(gpu);
    auto tracer = render::GaussianRayTracer::create(*gpu->library);
    if (!tracer) FAIL(tracer.error().toString());
    // One flat splat facing the camera, its baked colour a known albedo, and
    // what it reflects with: rough enough that the specular lobe is a sheen
    // rather than a spike a single sample would miss.
    CloudBuilder b;
    b.add(0.0F, 0.0F, 0.0F, 0.99F, 0.6F, 0.6F, 0.02F, {1.0F, 0.0F, 0.0F, 0.0F}, {0.8F, 0.4F, 0.2F});
    b.raw.encoding.metallic = b.raw.encoding.floatsPerRecord;
    b.raw.encoding.roughness = b.raw.encoding.floatsPerRecord + 1;
    b.raw.encoding.floatsPerRecord += 2;
    std::vector<float> widened;
    for (uint32_t k = 0; k < b.raw.count; ++k) {
        widened.insert(widened.end(), b.raw.records.begin() + k * (b.raw.encoding.floatsPerRecord - 2),
                       b.raw.records.begin() + (k + 1) * (b.raw.encoding.floatsPerRecord - 2));
        widened.push_back(0.0F);   // metallic: a dielectric
        widened.push_back(0.4F);   // roughness
    }
    b.raw.records = std::move(widened);
    auto cloud = h->loader.upload(b.raw);
    if (!cloud) FAIL(cloud.error().toString());
    CHECK(cloud->hasPbr());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, 2.0});
    lamp.radius = 0.3F;
    lamp.intensity = 4.0F;
    lamp.shadow = false;
    lamp.lightCategory = light::kLightUnlinked;
    REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
    // After `set`, which is what makes the buffer.
    render::SplatLights lights;
    lights.records = &table->records();
    lights.count = table->count();

    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 3.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 96;
    settings.height = 96;
    const auto both = [&](bool relight, render::RenderTargets& rasterised, render::RenderTargets& traced) {
        std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        instances[0].relight = relight;
        REQUIRE(h->raster.render(camera, instances, settings, rasterised, {}, nullptr, &lights));
        REQUIRE(tracer->render(camera, instances, settings, traced, &lights));
    };
    render::RenderTargets bakedRaster, bakedTraced, relitRaster, relitTraced;
    both(false, bakedRaster, bakedTraced);
    both(true, relitRaster, relitTraced);

    const auto compare = [&](const render::RenderTargets& a, const render::RenderTargets& c) {
        auto diff = render::compareImages(*gpu->library, a.colour, c.colour, settings.width, settings.height);
        if (!diff) FAIL(diff.error().toString());
        return *diff;
    };
    const auto tracedChanged = compare(bakedTraced, relitTraced);
    const auto routes = compare(relitRaster, relitTraced);
    const auto bakedRoutes = compare(bakedRaster, bakedTraced);
    std::printf("  traced baked against traced relit: max %u, %llu pixels beyond 2; "
                "relit raster against relit traced: p99 %u, max %u; baked, the same two: p99 %u, max %u\n",
                tracedChanged.max, static_cast<unsigned long long>(tracedChanged.over2), routes.p99, routes.max,
                bakedRoutes.p99, bakedRoutes.max);
    // The flag means something on the traced route: it used to mean nothing
    // there, and a cloud converted from a mesh came back as its own albedo.
    CHECK(tracedChanged.over2 > 100);
    // And it means the same thing on both. The two renderers differ a little
    // wherever they always differ -- EWA against the exact evaluation -- so
    // the claim is that relighting does not widen that: the relit pair agrees
    // as closely as the baked pair does.
    CHECK(routes.p99 <= bakedRoutes.p99 + 1);
}

TEST_CASE("a translucent splat is lit by a light behind it, and an opaque one is not",
          "[render][splats][relight][translucency]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    // One flat splat facing the camera and a light on the far side of it.
    // What reaches the eye can only have come through: a reflection cannot
    // reach it, and a Lambert lobe stops dead at the terminator.
    const auto cloudWith = [&](float transmission) {
        CloudBuilder b;
        b.add(0.0F, 0.0F, 0.0F, 0.99F, 0.6F, 0.6F, 0.02F, {1.0F, 0.0F, 0.0F, 0.0F}, {0.8F, 0.8F, 0.8F});
        const uint32_t floats = b.raw.encoding.floatsPerRecord;
        b.raw.encoding.metallic = floats;
        b.raw.encoding.roughness = floats + 1;
        b.raw.encoding.transmission = floats + 2;
        b.raw.encoding.floatsPerRecord = floats + 3;
        std::vector<float> widened;
        for (uint32_t k = 0; k < b.raw.count; ++k) {
            widened.insert(widened.end(), b.raw.records.begin() + k * floats,
                           b.raw.records.begin() + (k + 1) * floats);
            widened.push_back(0.0F);           // metallic
            widened.push_back(0.6F);           // roughness
            widened.push_back(transmission);
        }
        b.raw.records = std::move(widened);
        auto cloud = h->loader.upload(b.raw);
        if (!cloud) FAIL(cloud.error().toString());
        return std::move(*cloud);
    };
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, -2.0});   // behind the splat
    lamp.radius = 0.3F;
    lamp.intensity = 8.0F;
    lamp.shadow = false;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 3.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 96;
    settings.height = 96;
    settings.background = {0.0F, 0.0F, 0.0F, 0.0F};
    const auto draw = [&](const scene::GpuSplats& cloud, uint32_t category, render::RenderTargets& into) {
        lamp.lightCategory = category;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        // The table's buffer is made by `set`, so the frame's lights are taken
        // after it and not before.
        render::SplatLights lights;
        lights.records = &table->records();
        lights.count = table->count();
        std::vector<render::SplatInstance> instances{{&cloud, render::Mat4::identity()}};
        instances[0].relight = true;
        REQUIRE(h->raster.render(camera, instances, settings, into, {}, nullptr, &lights));
    };
    const scene::GpuSplats clear = cloudWith(1.0F);
    const scene::GpuSplats opaque = cloudWith(0.0F);
    render::RenderTargets lit;
    render::RenderTargets dark;
    render::RenderTargets unreached;
    draw(clear, light::kLightUnlinked, lit);
    draw(opaque, light::kLightUnlinked, dark);
    // The same opaque splat, with the light in a collection that does not
    // include it: nothing reaches it at all.
    draw(opaque, 0, unreached);

    auto diff = render::compareImages(*gpu->library, lit.colour, dark.colour, settings.width, settings.height);
    if (!diff) FAIL(diff.error().toString());
    auto against = render::compareImages(*gpu->library, dark.colour, unreached.colour, settings.width,
                                         settings.height);
    if (!against) FAIL(against.error().toString());
    std::printf("  translucent against opaque, light behind: max %u, %llu pixels beyond 2; "
                "the opaque one against a light that does not reach it: max %u\n",
                diff->max, static_cast<unsigned long long>(diff->over2), against->max);
    // The translucent splat is lit by a light it faces away from.
    CHECK(diff->over2 > 100);
    // And the one that lets nothing through is not: a light behind it is the
    // same as no light at all.
    CHECK(against->max <= 1);
}

TEST_CASE("a splat asked to be relit shows the scene's light, not the light it was baked with",
          "[render][splats][relight]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // One flat splat facing the camera, its baked colour a known albedo.
    CloudBuilder b;
    b.add(0.0F, 0.0F, 0.0F, 0.99F, 0.6F, 0.6F, 0.02F, {1.0F, 0.0F, 0.0F, 0.0F}, {0.8F, 0.4F, 0.2F});
    auto cloud = h->loader.upload(b.raw);
    if (!cloud) FAIL(cloud.error().toString());
    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());

    render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 3.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 96;
    settings.height = 96;

    const auto draw = [&](bool relight, uint32_t category, render::RenderTargets& into) {
        light::Light lamp;
        lamp.kind = light::LightKind::Sphere;
        lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, 2.0});
        lamp.radius = 0.3F;
        lamp.intensity = 4.0F;
        lamp.shadow = false;
        lamp.lightCategory = category;
        REQUIRE(table->set(std::span<const light::Light>(&lamp, 1)));
        render::SplatLights lights;
        lights.records = &table->records();
        lights.count = table->count();
        std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
        instances[0].relight = relight;
        REQUIRE(h->raster.render(camera, instances, settings, into, {}, nullptr, &lights));
    };

    render::RenderTargets baked;
    render::RenderTargets bakedAgain;
    render::RenderTargets relit;
    render::RenderTargets unreached;
    draw(false, light::kLightUnlinked, baked);
    draw(false, light::kLightUnlinked, bakedAgain);
    draw(true, light::kLightUnlinked, relit);
    // Relit, but by a light whose collection does not include this cloud:
    // nothing reaches it, so nothing lights it.
    draw(true, 0, unreached);

    const auto compare = [&](const render::RenderTargets& a, const render::RenderTargets& c) {
        auto diff = render::compareImages(*gpu->library, a.colour, c.colour, settings.width, settings.height);
        if (!diff) FAIL(diff.error().toString());
        return *diff;
    };
    const auto same = compare(baked, bakedAgain);
    const auto changed = compare(baked, relit);
    const auto dark = compare(relit, unreached);
    std::printf("  baked against itself: max %u; baked against relit: max %u, %llu pixels beyond 2; "
                "relit against unreached: max %u\n",
                same.max, changed.max, static_cast<unsigned long long>(changed.over2), dark.max);
    // Asking for nothing changes nothing: the path a frame without relighting
    // takes is the one it always took.
    CHECK(same.max == 0);
    // Asking for it changes the picture.
    CHECK(changed.over2 > 100);
    // And a light that does not reach the cloud lights none of it.
    CHECK(dark.max > 0);
}

// THE ID A PIXEL HANDS BACK IS A SELECTION, AND THEREFORE A HANDLE.
//
// Every gaussian carrying one Cryptomatte id came from one prim and stood on
// one material when the mesh was walked. So the id keys a table of material
// overrides, and clicking is how a host addresses it: `athenea view` measures what
// those gaussians carry and lets you say otherwise, for the frame, without
// touching the file.
TEST_CASE("a prim's gaussians are measured by their matte id, and said otherwise by it",
          "[render][gpu][crypto][override]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    const uint32_t idA = core::cryptomatteId("/World/Left");
    const uint32_t idB = core::cryptomatteId("/World/Right");
    // Two clouds, one prim each, both carrying a material: a dielectric at
    // roughness 0.4, the same number for every gaussian of both.
    const auto withPbr = [](CloudBuilder b, float metallic, float roughness) {
        b.raw.encoding.metallic = b.raw.encoding.floatsPerRecord;
        b.raw.encoding.roughness = b.raw.encoding.floatsPerRecord + 1;
        b.raw.encoding.floatsPerRecord += 2;
        std::vector<float> widened;
        for (uint32_t k = 0; k < b.raw.count; ++k) {
            widened.insert(widened.end(),
                           b.raw.records.begin() + k * (b.raw.encoding.floatsPerRecord - 2),
                           b.raw.records.begin() + (k + 1) * (b.raw.encoding.floatsPerRecord - 2));
            widened.push_back(metallic);
            widened.push_back(roughness);
        }
        b.raw.records = std::move(widened);
        return b;
    };
    CloudBuilder left = withPbr(randomCloud(800, 31, 0.02F, 0.12F), 0.0F, 0.4F);
    CloudBuilder right = withPbr(randomCloud(800, 97, 0.02F, 0.12F), 0.0F, 0.4F);
    test::giveCryptoId(left, idA);
    test::giveCryptoId(right, idB);
    auto cloudA = h->loader.upload(left.raw, 3);
    auto cloudB = h->loader.upload(right.raw, 3);
    REQUIRE(cloudA);
    REQUIRE(cloudB);
    REQUIRE(cloudA->hasPbr());
    REQUIRE(cloudA->hasCrypto());

    // WHAT THEY CARRY, counted on the device. A constant material has no
    // range to report, and a value the byte quantisation rounds is why the
    // tolerance is a 255th and not an epsilon.
    auto a = render::measureSplatId(*gpu->library, *cloudA, idA);
    auto b = render::measureSplatId(*gpu->library, *cloudA, idB);
    if (!a) FAIL(a.error().toString());
    if (!b) FAIL(b.error().toString());
    std::printf("  measured id A: %u gaussians, roughness %.3f (%.3f to %.3f); id B in cloud A: %u\n",
                a->count, double(a->mean[1]), double(a->low[1]), double(a->high[1]), b->count);
    CHECK(a->count == cloudA->count);
    CHECK(a->hasPbr);
    CHECK(a->mean[1] == Catch::Approx(0.4F).margin(1.0 / 255.0));
    CHECK(a->low[1] == Catch::Approx(a->high[1]));
    CHECK(a->mean[0] == Catch::Approx(0.0F).margin(1.0 / 255.0));
    // An id no gaussian of this cloud carries selects nothing.
    CHECK(b->count == 0);

    // The table, two float4 a row as the shader reads it: the id bit-cast
    // into the first float, then what the frame says over the file.
    const auto table = [&](uint32_t id, float metallic, float roughness, std::array<float, 3> tint) {
        std::array<float, 8> row{};
        std::memcpy(&row[0], &id, sizeof(float));
        row[1] = metallic;
        row[2] = roughness;
        row[3] = -1.0F;   // transmission: leave the gaussian's own
        row[4] = tint[0];
        row[5] = tint[1];
        row[6] = tint[2];
        gpu::BufferDesc desc;
        desc.bytes = sizeof(row);
        desc.elementBytes = 16;
        desc.label = "splat.overrides.test";
        auto made = gpu::Buffer::create(*gpu->device, desc, row.data());
        REQUIRE(made);
        return std::move(*made);
    };
    const gpu::Buffer sayA = table(idA, 1.0F, 0.02F, {1.0F, 0.2F, 0.2F});

    auto lights = light::LightTable::create(*gpu->library);
    if (!lights) FAIL(lights.error().toString());
    light::Light lamp;
    lamp.kind = light::LightKind::Sphere;
    lamp.lightToWorld = aofx::xform::translation({0.0, 0.0, 3.0});
    lamp.radius = 0.4F;
    lamp.intensity = 6.0F;
    lamp.shadow = false;
    lamp.lightCategory = light::kLightUnlinked;
    REQUIRE(lights->set(std::span<const light::Light>(&lamp, 1)));
    render::SplatLights bound;
    bound.records = &lights->records();
    bound.count = lights->count();

    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 120;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.4, 5.0}, {0.0, 0.0, 0.0});
    const auto sideways = [](double x) {
        render::Mat4 m = render::Mat4::identity();
        m.at(0, 3) = x;
        return m;
    };
    const auto draw = [&](bool saying, bool bothClouds, render::RenderTargets& out) {
        std::vector<render::SplatInstance> instances{{&*cloudA, sideways(-1.5)}};
        if (bothClouds) {
            instances.push_back({&*cloudB, sideways(1.5)});
        }
        for (render::SplatInstance& instance : instances) {
            instance.relight = true;
            if (saying) {
                instance.overrides = &sayA;
                instance.overrideCount = 1;
            }
        }
        REQUIRE(h->raster.render(camera, instances, settings, out, {}, nullptr, &bound));
    };
    render::RenderTargets plain, said, otherPlain, otherSaid;
    draw(false, true, plain);
    draw(true, true, said);
    auto changed = render::compareImages(*gpu->library, plain.colour, said.colour, settings.width,
                                         settings.height);
    if (!changed) FAIL(changed.error().toString());

    // AND ONLY THAT PRIM. The same table is bound to the cloud of the other
    // id, whose gaussians carry none of the rows in it: its pixels have to be
    // the ones they were, bit for bit.
    const auto onlyRight = [&](bool saying, render::RenderTargets& out) {
        std::vector<render::SplatInstance> instances{{&*cloudB, sideways(1.5)}};
        instances[0].relight = true;
        if (saying) {
            instances[0].overrides = &sayA;
            instances[0].overrideCount = 1;
        }
        REQUIRE(h->raster.render(camera, instances, settings, out, {}, nullptr, &bound));
    };
    onlyRight(false, otherPlain);
    onlyRight(true, otherSaid);
    auto untouched = render::compareImages(*gpu->library, otherPlain.colour, otherSaid.colour,
                                           settings.width, settings.height);
    if (!untouched) FAIL(untouched.error().toString());
    std::printf("  saying otherwise: max %u, %llu pixels beyond 2; the other prim: max %u\n",
                changed->max, static_cast<unsigned long long>(changed->over2), untouched->max);
    CHECK(changed->over2 > 100);
    CHECK(untouched->max == 0);
}

TEST_CASE("a Cryptomatte names the prims a pixel saw, and by how much they covered it",
          "[render][gpu][crypto]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // Two clouds, each converted from its own prim, so each carries one id:
    // what `athenea mesh2splat` writes, and what the matte has to keep apart.
    const uint32_t idA = core::cryptomatteId("/World/Left");
    const uint32_t idB = core::cryptomatteId("/World/Right");
    CHECK(idA != idB);
    CloudBuilder left = randomCloud(1200, 31, 0.02F, 0.12F);
    CloudBuilder right = randomCloud(1200, 97, 0.02F, 0.12F);
    test::giveCryptoId(left, idA);
    test::giveCryptoId(right, idB);
    auto cloudA = h->loader.upload(left.raw, 3);
    auto cloudB = h->loader.upload(right.raw, 3);
    REQUIRE(cloudA);
    REQUIRE(cloudB);
    CHECK(cloudA->hasCrypto());
    CHECK(cloudB->hasCrypto());
    const auto sideways = [](double x) {
        render::Mat4 m = render::Mat4::identity();
        m.at(0, 3) = x;
        return m;
    };
    const std::vector<render::SplatInstance> instances{{&*cloudA, sideways(-1.8)},
                                                      {&*cloudB, sideways(1.8)}};

    render::RenderSettings settings;
    settings.width = 200;
    settings.height = 150;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.5, 6.0}, {0.0, 0.0, 0.0});

    render::RenderTargets plain;
    auto without = h->raster.render(camera, instances, settings, plain);
    if (!without) FAIL(without.error().toString());

    settings.cryptomatte = true;
    render::RenderTargets matte;
    auto with = h->raster.render(camera, instances, settings, matte);
    if (!with) FAIL(with.error().toString());
    REQUIRE(matte.hasCrypto());

    // The matte costs the colour nothing: the walk is the same walk.
    auto diff = render::compareImages(*h->gpu->library, plain.colour, matte.colour, settings.width,
                                      settings.height);
    if (!diff) FAIL(diff.error().toString());
    CHECK(diff->max == 0);

    const std::array<uint32_t, 2> ids{idA, idB};
    gpu::BufferDesc desc;
    desc.bytes = sizeof(ids);
    desc.elementBytes = 4;
    desc.label = "crypto.allowed";
    auto allowed = gpu::Buffer::create(*gpu->device, desc, ids.data());
    REQUIRE(allowed);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "crypto.stats");
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/crypto_check");
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {settings.width, settings.height, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["crypto"].setBinding(matte.crypto.rhi());
            cursor["colour"].setBinding(matte.colour.rhi());
            cursor["allowed"].setBinding(allowed->rhi());
            cursor["stats"].setBinding(stats.rhi());
            cursor["params"]["width"].setData(settings.width);
            cursor["params"]["height"].setData(settings.height);
            cursor["params"]["allowedCount"].setData(uint32_t{2});
            // A pixel's coverage is a sum of up to six weights against one
            // alpha accumulated in the same order: a few ulps apart at most.
            cursor["params"]["tolerance"].setData(2.0e-3F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> row{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(row), row.data()));
    std::printf("  crypto: %u covered, %u only left, %u only right, %u both\n", row[4],
                row[5] - row[7], row[6] - row[7], row[7]);
    CHECK(row[0] == 0);   // no rank named an id the scene does not hold
    CHECK(row[1] == 0);   // every pixel's ranks descend
    CHECK(row[2] == 0);   // no pixel named an id twice
    CHECK(row[3] == 0);   // the coverages add up to the pixel's alpha
    CHECK(row[4] > 1000);
    CHECK(row[4] < settings.width * settings.height);   // and not the whole frame
    CHECK(row[5] - row[7] > 100);   // pixels only the left cloud covered
    CHECK(row[6] - row[7] > 100);   // and only the right one
    // The two clouds overlap where they meet: a pixel naming both is the case
    // a single id per pixel could not have expressed.
    CHECK(row[7] > 0);
}

TEST_CASE("a cloud with no ancestry is left out of the matte rather than named wrongly",
          "[render][gpu][crypto]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    CloudBuilder capture = randomCloud(800, 13, 0.02F, 0.3F);
    auto cloud = h->loader.upload(capture.raw, 3);
    REQUIRE(cloud);
    CHECK_FALSE(cloud->hasCrypto());
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};

    render::RenderSettings settings;
    settings.width = 120;
    settings.height = 96;
    settings.cryptomatte = true;
    render::RenderTargets targets;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.5, 6.0}, {0.0, 0.0, 0.0});
    auto stats = h->raster.render(camera, instances, settings, targets);
    if (!stats) FAIL(stats.error().toString());
    REQUIRE(targets.hasCrypto());

    // Nothing in the matte at all: id 0 is absence, not a name.
    auto values = targets.crypto.readAll<float>(*gpu->device);
    REQUIRE(values);
    double coverage = 0.0;
    for (size_t k = 0; k < values->size(); k += 2) {
        coverage += (*values)[k + 1];
    }
    CHECK(coverage == 0.0);
}

// THE ONE PIECE OF PHYSICS IN A GAUSSIAN'S GLASS.
//
// A cloud has no surface to refract at, so what a frame does with an index is
// bend the direction the transmitted half looks along, once, at the
// gaussian's own normal (`splat_relight.slang`). Everything else about that
// glass is an approximation on purpose; the bend itself is Snell's and has to
// be exactly Snell's, or a ball turns the room by the wrong amount and
// nothing says so.
TEST_CASE("a gaussian bends what it transmits by Snell's law", "[render][gpu][refract]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "refract.stats");
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "refract.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(*gpu->device, desc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/refract_check");
    constexpr uint32_t kAngles = 64;
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["angles"].setData(kAngles);
            cursor["params"]["ior"].setData(1.5F);
            cursor["params"]["tolerance"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
    std::printf("  refraction: %u angles, %u off Snell, %u on the near side, %u out of plane; first "
                "%.5f against %.5f; worst %.6f\n", counts[1], counts[0], counts[2], counts[3],
                double(readings[1]), double(readings[2]), double(readings[0]));
    CHECK(counts[1] == kAngles);
    CHECK(counts[0] == 0);
    CHECK(counts[2] == 0);
    CHECK(counts[3] == 0);
    CHECK(counts[4] == 0);   // and no index at all is the straight direction
}

// THE DFG TERM, WHICH IS NOW ONE FIT AND NOT TWO.
//
// How much of an environment a GGX lobe sends back was fitted twice in this
// tree: Lazarov's in `splat_relight.slang`, which a gaussian reflected with,
// and MaterialX's `ggxDirectionalAlbedo`, which the surface lobes reflect
// with. Asked the same question over a grid of roughness and `N.V` the two
// came apart by 76 %, worst where a surface grazes -- so a cloud and the mesh
// beside it under one sky could not have agreed, and no comparison of the two
// routes had a floor. A gaussian now reflects with the surface's own fit, and
// this is what says they never part again.
TEST_CASE("the two fits of what a lobe sends back agree", "[render][gpu][dfg]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "dfg.stats");
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "dfg.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(*gpu->device, desc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/dfg_check");
    constexpr uint32_t kSteps = 16;
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["steps"].setData(kSteps);
            cursor["params"]["f0"].setData(0.04F);
            cursor["params"]["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
    std::printf("  the DFG fits: %u points, %u apart; worst %.1f%% (%.4f against %.4f at "
                "roughness %.2f, N.V %.2f)\n", counts[1], counts[0], double(counts[2]) * 1.0e-4,
                double(readings[0]), double(readings[1]), double(readings[2]), double(readings[3]));
    CHECK(counts[1] == kSteps * kSteps);
    CHECK(counts[0] == 0);
}

// WHAT A REFLECTION IS ALLOWED TO SEE, WITH NO RAY TO ASK.
//
// A gaussian's reflection was added with nothing in front of it: the pawn's
// marble disc, with its own stem standing over it, read a tenth bright
// against the mesh path traced beside it. The transfer the cloud already
// carries is the answer -- its constant term is the visible fraction of the
// sky -- and Lagarde and de Rousiers' fit turns that into how much of it a
// lobe of a given sharpness may see. The fit is an approximation; what it
// must never get wrong is checked here.
TEST_CASE("an open sky reflects whole, a closed one not at all", "[render][gpu][openness]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "openness.stats");
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "openness.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(*gpu->device, desc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/openness_check");
    constexpr uint32_t kSteps = 12;
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["steps"].setData(kSteps);
            cursor["params"]["tolerance"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
    std::printf("  the opening: %u readings; %u not whole under an open sky (worst %.6f), %u leaking "
                "under a closed one, %u not rising with it, %u out of range\n", counts[4], counts[0],
                double(readings[0]), counts[1], counts[2], counts[3]);
    CHECK(counts[4] == kSteps * kSteps * (kSteps + 1));
    CHECK(counts[0] == 0);
    CHECK(counts[1] == 0);
    CHECK(counts[2] == 0);
    CHECK(counts[3] == 0);
}

// GLASS SENDS ON WHAT IT DID NOT REFLECT, AND NOT MORE.
//
// A transmitting gaussian's body is the light that came through it, and the
// reflection is added on top whatever the surface does: at full strength the
// two together are more light than arrived, and a glass ball came back half
// again as bright as the mesh beside it. What passes is the complement of
// what reflects, which is what `lobes.slang` writes for a transmitting lobe
// -- and this is what holds the sum below one.
TEST_CASE("what glass passes and what it reflects never add to more than arrived",
          "[render][gpu][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "glass.stats");
    gpu::BufferDesc desc;
    desc.bytes = 8 * sizeof(float);
    desc.elementBytes = sizeof(float);
    desc.label = "glass.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(*gpu->device, desc, zeros.data());
    REQUIRE(worst);
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/glass_energy_check");
    constexpr uint32_t kSteps = 16;
    {
        gpu::CommandBatch batch(*gpu->device);
        check.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["stats"].setBinding(stats.rhi());
            cursor["worst"].setBinding(worst->rhi());
            cursor["params"]["steps"].setData(kSteps);
            cursor["params"]["tolerance"].setData(1.0e-4F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(*gpu->device, 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(*gpu->device, 0, sizeof(readings), readings.data()));
    std::printf("  glass: %u points, %u over what arrived, %u too closed; the largest sum is %.4f at "
                "roughness %.2f, N.V %.2f\n", counts[1], counts[0], counts[2], double(readings[0]),
                double(readings[1]), double(readings[2]));
    CHECK(counts[1] == kSteps * kSteps);
    CHECK(counts[0] == 0);
    CHECK(counts[2] == 0);
}

// WHAT A MATERIAL LAYERS OVER ITS BASE CHANGES NOTHING WHERE IT LAYERS NOTHING.
//
// A gaussian carries its specular's weight, colour and index, a coat and a
// sheen where its conversion read them (`GpuSplats::lobes`), and the plain
// lobes where it did not: every cloud written before them. Those must be
// reflected exactly as before -- the environment's reflection bit for bit,
// a light's lobe to rounding, nothing taken by a coat and nothing given by a
// sheen -- and a coat over anything gives back no more than it takes. The
// twelve values go into three words and come back within a byte's step.
TEST_CASE("the plain lobes reflect as before, and the layers pack into three words",
          "[render][gpu][lobes]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 8, "lobes.counts");
    constexpr uint32_t kSteps = 16;
    constexpr uint32_t kPacked = 100000;
    {
        auto plain = gpu::ComputeKernel::create(*gpu->library, "athenea/test/lobes_check", "lobesPlainCheck");
        if (!plain) FAIL(plain.error().toString());
        gpu::CommandBatch batch(*gpu->device);
        plain->dispatch(batch, {kSteps * kSteps * kSteps, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["counts"].setBinding(counts.rhi());
            cursor["params"]["steps"].setData(kSteps);
            cursor["params"]["count"].setData(0u);
            cursor["params"]["tolerance"].setData(1.0e-5F);
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> seen{};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(seen), seen.data()));
    std::printf("  plain lobes: %u points; %u reflections of a sky changed, %u layers not nothing, %u lobes off "
                "the old one, %u plain lobes not themselves packed, %u coats that gave back more than they took\n",
                seen[0], seen[1], seen[2], seen[3], seen[4], seen[5]);
    CHECK(seen[0] == kSteps * kSteps * kSteps);
    CHECK(seen[1] == 0);
    CHECK(seen[2] == 0);
    CHECK(seen[3] == 0);
    CHECK(seen[4] == 0);
    CHECK(seen[5] == 0);

    gpu::Buffer packed = test::uintBuffer(*gpu->device, 8, "lobes.packed");
    {
        auto trip = gpu::ComputeKernel::create(*gpu->library, "athenea/test/lobes_check", "lobesRoundTrip");
        if (!trip) FAIL(trip.error().toString());
        gpu::CommandBatch batch(*gpu->device);
        trip->dispatch(batch, {kPacked, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["counts"].setBinding(packed.rhi());
            cursor["params"]["steps"].setData(0u);
            cursor["params"]["count"].setData(kPacked);
            cursor["params"]["tolerance"].setData(0.0F);
        });
        REQUIRE(batch.submit(true));
    }
    REQUIRE(packed.read(*gpu->device, 0, sizeof(seen), seen.data()));
    std::printf("  packed: %u sets of twelve; %u with a weight, colour or roughness off by more than half a "
                "byte, %u with an index off by more than half its step\n",
                seen[0], seen[1], seen[2]);
    CHECK(seen[0] == kPacked);
    CHECK(seen[1] == 0);
    CHECK(seen[2] == 0);
}

// A RAY THAT STARTS ON A SURFACE MUST NOT MEET THAT SURFACE.
//
// A convex cloud is a shell of overlapping gaussians, and a mirror ray leaving
// one at a grazing angle runs inside that shell for a chord tens of footprints
// long, meeting neighbour after neighbour. The nearest of them was what the
// gaussian "reflected": a gold ball reflected itself all round its rim and
// lost its blue twice over, a grey one read 0.93 of the mesh beside it, and
// the reference sheets carried both for a week while the bake was suspected.
// `glassNearest` now skips anything that rises less than a few footprints
// above the plane the ray started on. The first half of this holds that a
// ball alone reflects nothing; the second, that a plate above it still does.
TEST_CASE("a convex cloud does not reflect itself, and still reflects what stands over it",
          "[render][gpu][glass][reflections]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rayQuery || !gpu->device->caps().accelerationStructure) {
        SKIP("no ray queries on this device");
    }
    auto h = harness(gpu);
    auto tracer = render::GaussianRayTracer::create(*gpu->library);
    if (!tracer) FAIL(tracer.error().toString());

    // A unit ball of flat gold gaussians, each with its short axis along the
    // radius, as mesh2splat stands them: a spacing of 0.025 and a footprint
    // of 0.65 of it, which is what the shipped conversion does.
    const auto withPbr = [](CloudBuilder b, float metallic, float roughness) {
        b.raw.encoding.metallic = b.raw.encoding.floatsPerRecord;
        b.raw.encoding.roughness = b.raw.encoding.floatsPerRecord + 1;
        b.raw.encoding.floatsPerRecord += 2;
        std::vector<float> widened;
        for (uint32_t k = 0; k < b.raw.count; ++k) {
            widened.insert(widened.end(),
                           b.raw.records.begin() + k * (b.raw.encoding.floatsPerRecord - 2),
                           b.raw.records.begin() + (k + 1) * (b.raw.encoding.floatsPerRecord - 2));
            widened.push_back(metallic);
            widened.push_back(roughness);
        }
        b.raw.records = std::move(widened);
        return b;
    };
    const auto facing = [](float nx, float ny, float nz) {
        // The rotation taking local +z, the short axis, onto the normal.
        const float w = 1.0F + nz;
        float x = -ny, y = nx, z = 0.0F;
        float len = std::sqrt(w * w + x * x + y * y + z * z);
        if (len < 1.0e-6F) {   // pointing straight down: a half turn about x
            return std::array<float, 4>{0.0F, 1.0F, 0.0F, 0.0F};
        }
        return std::array<float, 4>{w / len, x / len, y / len, z / len};
    };
    CloudBuilder shell;
    {
        const uint32_t count = 20000;
        const float golden = 2.39996322972865332F;
        const float size = 0.025F * 0.65F;
        for (uint32_t i = 0; i < count; ++i) {
            const float z = 1.0F - 2.0F * (float(i) + 0.5F) / float(count);
            const float r = std::sqrt(std::max(1.0F - z * z, 0.0F));
            const float phi = golden * float(i);
            const float nx = r * std::cos(phi), ny = r * std::sin(phi), nz = z;
            shell.add(nx, ny, nz, 0.99F, size, size, size / 11.0F, facing(nx, ny, nz),
                      {0.944F, 0.776F, 0.373F});
        }
    }
    auto ball = h->loader.upload(withPbr(shell, 1.0F, 0.05F).raw, 3);
    REQUIRE(ball);
    // The same ball with a dark plate behind the camera, facing it, in the
    // SAME cloud: a gaussian reflects the cloud it belongs to and no other,
    // so what the ball's front legitimately sees has to be part of it. The
    // camera never sees the plate itself.
    CloudBuilder covered = shell;
    for (int i = -20; i <= 20; ++i) {
        for (int j = -20; j <= 20; ++j) {
            covered.add(float(i) * 0.1F, float(j) * 0.1F, 6.5F, 0.99F, 0.08F, 0.08F, 0.008F,
                        facing(0.0F, 0.0F, -1.0F), {0.05F, 0.05F, 0.05F});
        }
    }
    auto plate = h->loader.upload(withPbr(covered, 1.0F, 0.05F).raw, 3);
    REQUIRE(plate);

    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    light::Light sky;
    sky.kind = light::LightKind::Dome;
    sky.intensity = 1.0F;
    sky.shadow = false;
    sky.lightCategory = light::kLightUnlinked;
    REQUIRE(table->set(std::span<const light::Light>(&sky, 1)));
    render::SplatLights lights;
    lights.records = &table->records();
    lights.count = table->count();

    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 160;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.6, 4.5}, {0.0, 0.0, 0.0});
    const auto draw = [&](bool reflecting, bool withPlate, render::RenderTargets& out) {
        std::vector<render::SplatInstance> instances{{withPlate ? &*plate : &*ball, render::Mat4::identity()}};
        instances[0].relight = true;
        instances[0].reflectCloud = reflecting;
        REQUIRE(tracer->render(camera, instances, settings, out, &lights));
    };
    render::RenderTargets aloneOff, aloneOn, coveredOff, coveredOn;
    draw(false, false, aloneOff);
    draw(true, false, aloneOn);
    draw(false, true, coveredOff);
    draw(true, true, coveredOn);
    const auto compare = [&](const render::RenderTargets& a, const render::RenderTargets& b) {
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        if (!diff) FAIL(diff.error().toString());
        return *diff;
    };
    const auto self = compare(aloneOff, aloneOn);
    const auto lidSeen = compare(coveredOff, coveredOn);
    std::printf("  a ball alone, reflections off against on: max %u, %llu pixels beyond 2; "
                "with a plate behind the camera: max %u, %llu beyond 2\n",
                self.max, static_cast<unsigned long long>(self.over2), lidSeen.max,
                static_cast<unsigned long long>(lidSeen.over2));
    // Alone, the ray meets nothing but the sky, so the frame is the one drawn
    // without a ray at all; with the plate behind the camera, the ball's
    // front now reflects it.
    CHECK(self.over2 == 0);
    CHECK(lidSeen.over2 > 50);
}

// A POSED CLOUD REFITS WHAT IT IS TRACED THROUGH, AND DRAWS WHAT A REBUILT ONE DRAWS.
//
// A skinned cloud is the same particles in the same order every frame, only
// moved; its ray tracing structure keeps its shape and has its bounds
// refitted (GaussianRayTracer::sync), where it used to be built again from
// nothing each pose. The scene is the reflection test's ball with a plate in
// the same cloud: the plate stands still on one joint while the ball turns
// and slides on another, so a stale box would lose the ball where it went,
// and the ball's reflection of the plate -- a ray walked through the cloud's
// own tree (rt_glass.slang, glassNearest) on the compute route -- would come
// out of a tree that no longer bounds it. Three tracers draw every pose: one
// that refits, one that rebuilds every pose (refitsPerRebuild 0, the old
// behaviour, and the reference), and one that refits twice and then rebuilds
// in place. Only the stats' counters and compareImages' metrics come back.
TEST_CASE("a posed cloud refits its ray tracing structure and draws what a rebuilt one draws",
          "[render][gpu][rt][skinning][glass]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!gpu->device->caps().rayQuery || !gpu->device->caps().accelerationStructure) {
        SKIP("no ray queries on this device");
    }
    const render::RayTracingRoute route =
        GENERATE(render::RayTracingRoute::Hardware, render::RayTracingRoute::ComputeBvh);
    std::printf("  route: %s\n", route == render::RayTracingRoute::Hardware ? "hardware" : "compute BVH");
    auto h = harness(gpu);
    const auto tracer = [&](uint32_t refitsPerRebuild) {
        render::RayTracerSettings settings;
        settings.route = route;
        settings.refitsPerRebuild = refitsPerRebuild;
        auto made = render::GaussianRayTracer::create(*gpu->library, settings);
        if (!made) FAIL(made.error().toString());
        return std::move(*made);
    };
    render::GaussianRayTracer refitting = tracer(64);
    render::GaussianRayTracer rebuilding = tracer(0);
    render::GaussianRayTracer periodic = tracer(2);
    auto skinner = scene::SplatSkinner::create(*gpu->library);
    if (!skinner) FAIL(skinner.error().toString());

    const auto facing = [](float nx, float ny, float nz) {
        const float w = 1.0F + nz;
        const float x = -ny, y = nx;
        const float len = std::sqrt(w * w + x * x + y * y);
        if (len < 1.0e-6F) {
            return std::array<float, 4>{0.0F, 1.0F, 0.0F, 0.0F};
        }
        return std::array<float, 4>{w / len, x / len, y / len, 0.0F};
    };
    CloudBuilder built;
    constexpr uint32_t kBall = 8000;
    {
        const float golden = 2.39996322972865332F;
        const float size = 0.04F * 0.65F;
        for (uint32_t i = 0; i < kBall; ++i) {
            const float z = 1.0F - 2.0F * (float(i) + 0.5F) / float(kBall);
            const float r = std::sqrt(std::max(1.0F - z * z, 0.0F));
            const float phi = golden * float(i);
            const float nx = r * std::cos(phi), ny = r * std::sin(phi), nz = z;
            built.add(nx, ny, nz, 0.99F, size, size, size / 11.0F, facing(nx, ny, nz), {0.944F, 0.776F, 0.373F});
        }
        for (int i = -20; i <= 20; ++i) {
            for (int j = -20; j <= 20; ++j) {
                built.add(float(i) * 0.1F, float(j) * 0.1F, 6.5F, 0.99F, 0.08F, 0.08F, 0.008F,
                          facing(0.0F, 0.0F, -1.0F), {0.05F, 0.05F, 0.05F});
            }
        }
    }
    {
        // Flat gold: metallic and smooth, two more floats a record.
        built.raw.encoding.metallic = built.raw.encoding.floatsPerRecord;
        built.raw.encoding.roughness = built.raw.encoding.floatsPerRecord + 1;
        built.raw.encoding.floatsPerRecord += 2;
        const uint32_t was = built.raw.encoding.floatsPerRecord - 2;
        std::vector<float> widened;
        for (uint32_t k = 0; k < built.raw.count; ++k) {
            widened.insert(widened.end(), built.raw.records.begin() + k * was,
                           built.raw.records.begin() + (k + 1) * was);
            widened.push_back(1.0F);
            widened.push_back(0.05F);
        }
        built.raw.records = std::move(widened);
    }
    auto rest = h->loader.upload(built.raw, 3);
    REQUIRE(rest);
    const uint32_t count = rest->count;
    REQUIRE(count == built.raw.count);

    // The ball on joint 1, the plate on joint 0: one (joint, weight) a gaussian.
    std::vector<float> pairs(size_t{count} * 2);
    for (uint32_t k = 0; k < count; ++k) {
        pairs[size_t{k} * 2] = k < kBall ? 1.0F : 0.0F;
        pairs[size_t{k} * 2 + 1] = 1.0F;
    }
    auto influences = gpu::Buffer::fromSpan<float>(*gpu->device, pairs, "test.influences");
    REQUIRE(influences);

    // The posed cloud, as the engine keeps one: the rest's channels, its own
    // positions and shape, the same handles every pose.
    scene::GpuSplats posed = *rest;
    {
        gpu::BufferDesc desc;
        desc.bytes = uint64_t{count} * 16;
        desc.elementBytes = 16;
        desc.label = "test.posed.positions";
        auto positions = gpu::Buffer::create(*gpu->device, desc);
        desc.bytes = uint64_t{count} * 16;
        desc.elementBytes = 4;
        desc.label = "test.posed.shape";
        auto shape = gpu::Buffer::create(*gpu->device, desc);
        REQUIRE(positions);
        REQUIRE(shape);
        posed.positions = std::move(*positions);
        posed.shape = std::move(*shape);
    }

    auto table = light::LightTable::create(*gpu->library);
    if (!table) FAIL(table.error().toString());
    light::Light sky;
    sky.kind = light::LightKind::Dome;
    sky.intensity = 1.0F;
    sky.shadow = false;
    sky.lightCategory = light::kLightUnlinked;
    REQUIRE(table->set(std::span<const light::Light>(&sky, 1)));
    render::SplatLights lights;
    lights.records = &table->records();
    lights.count = table->count();

    render::RenderSettings settings;
    settings.width = 160;
    settings.height = 160;
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.6, 4.5}, {0.0, 0.0, 0.0});
    std::vector<render::SplatInstance> instances{{&posed, render::Mat4::identity()}};
    instances[0].relight = true;
    instances[0].reflectCloud = true;

    constexpr uint32_t kPoses = 6;
    uint32_t refitsSeen = 0, rebuildsOfRefitting = 0, periodicRebuilds = 0;
    for (uint32_t pose = 0; pose < kPoses; ++pose) {
        // Joint 0 where it was; joint 1 turned about y and slid. Row major,
        // as GfMatrix4f holds them: the translation is the last row.
        const float a = 0.3F * float(pose);
        const float c = std::cos(a), s = std::sin(a);
        const std::array<float, 32> joints{1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F,
                                           c, 0.0F, -s, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F,
                                           s, 0.0F, c, 0.0F, 0.15F * float(pose), -0.1F * float(pose), 0.0F, 1.0F};
        auto xforms = gpu::Buffer::fromSpan<float>(*gpu->device, joints, "test.joints");
        REQUIRE(xforms);
        {
            gpu::CommandBatch batch(*gpu->device);
            scene::SplatSkinInput input;
            input.rest = &*rest;
            input.influences = &*influences;
            input.perSplat = 1;
            input.skinningXforms = &*xforms;
            REQUIRE(skinner->skin(batch, input, posed.positions, posed.shape));
            REQUIRE(batch.submit(true));
        }
        auto box = h->loader.boundsOf(posed.positions, count);
        REQUIRE(box);
        posed.bounds = *box;
        posed.revision += 1;

        render::RenderTargets a1, b1, c1;
        auto sa = refitting.render(camera, instances, settings, a1, &lights);
        auto sb = rebuilding.render(camera, instances, settings, b1, &lights);
        auto sc = periodic.render(camera, instances, settings, c1, &lights);
        if (!sa) FAIL(sa.error().toString());
        if (!sb) FAIL(sb.error().toString());
        if (!sc) FAIL(sc.error().toString());
        refitsSeen += sa->refitted;
        rebuildsOfRefitting += sa->rebuilt ? 1u : 0u;
        periodicRebuilds += sc->rebuilt ? 1u : 0u;
        CHECK(sb->rebuilt);
        CHECK(sb->refitted == 0);
        auto refitted = render::compareImages(*gpu->library, a1.colour, b1.colour, settings.width, settings.height);
        auto inPlace = render::compareImages(*gpu->library, c1.colour, b1.colour, settings.width, settings.height);
        REQUIRE(refitted);
        REQUIRE(inPlace);
        std::printf("  pose %u: refit %s (%.2f ms build), rebuild %.2f ms build, periodic %s; "
                    "refit against rebuild p99 %u, max %u, %llu over 2; periodic p99 %u, max %u\n",
                    pose, sa->rebuilt ? "rebuilt" : (sa->refitted != 0 ? "refitted" : "kept"), sa->buildMs,
                    sb->buildMs, sc->rebuilt ? "rebuilt" : "refitted", refitted->p99, refitted->max,
                    static_cast<unsigned long long>(refitted->over2), inPlace->p99, inPlace->max);
        // Measured (M5 Pro, both routes): p99 0, max 1, nothing over 2 --
        // a refitted tree is walked in another order than a rebuilt one, and
        // near-equal peaks land a code value apart. A tree left stale (the
        // refit skipped) reads p99 255 and ~17000 pixels over 2.
        CHECK(refitted->p99 == 0);
        CHECK(refitted->max <= 1);
        CHECK(inPlace->p99 == 0);
        CHECK(inPlace->max <= 1);
    }
    // The first pose builds; every one after it refits, and nothing is
    // rebuilt. The periodic tracer builds, refits twice, rebuilds in place,
    // refits twice: poses 0 and 3.
    CHECK(rebuildsOfRefitting == 1);
    CHECK(refitsSeen == kPoses - 1);
    CHECK(periodicRebuilds == 2);
}

namespace {

/// The image `athenea/test/linear_blend` builds: two layers of constant
/// colour, front over back, blended in linear light.
gpu::Buffer analyticBlend(test::Gpu& gpu, uint32_t width, uint32_t height, std::array<float, 4> front,
                          std::array<float, 4> back) {
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{width} * height * 16;
    desc.elementBytes = 16;
    desc.label = "analytic";
    auto image = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(image);
    auto blend = gpu::ComputeKernel::create(*gpu.library, "athenea/test/linear_blend", "linearBlend");
    if (!blend) FAIL(blend.error().toString());
    gpu::CommandBatch batch(*gpu.device);
    blend->dispatch(batch, {width * height, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["analytic"].setBinding(image->rhi());
        rhi::ShaderCursor p = cursor["blendParams"];
        p["front"].setData(front.data(), sizeof(front));
        p["back"].setData(back.data(), sizeof(back));
        p["width"].setData(width);
        p["height"].setData(height);
    });
    REQUIRE(batch.submit(true));
    return std::move(*image);
}

}   // namespace

TEST_CASE("a converted cloud's splats are blended in linear light", "[render][gpu][linear]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // Two discs facing the camera, each far wider than the frame -- a sigma
    // of 1000 against a frame 5 units across at their distance, so a pixel
    // sees each at its peak opacity to five decimals -- of half opacity and
    // colours in linear light, as a conversion writes them (LinearLight).
    // Front red, back blue: a mix that is far from its sRGB counterpart.
    const std::array<float, 3> red{0.8F, 0.05F, 0.05F};
    const std::array<float, 3> blue{0.05F, 0.05F, 0.8F};
    CloudBuilder b;
    b.add(0.0F, 0.0F, 1.0F, 0.5F, 1000.0F, 1000.0F, 0.01F, {1, 0, 0, 0}, {0.5F, 0.5F, 0.5F});
    b.add(0.0F, 0.0F, -1.0F, 0.5F, 1000.0F, 1000.0F, 0.01F, {1, 0, 0, 0}, {0.5F, 0.5F, 0.5F});
    b.raw.encoding.colour = io::SplatEncoding::Colour::LinearLight;
    for (uint32_t k = 0; k < 2; ++k) {
        const std::array<float, 3>& c = k == 0 ? red : blue;
        for (uint32_t ch = 0; ch < 3; ++ch) {
            b.raw.records[size_t{k} * b.raw.encoding.floatsPerRecord + b.raw.encoding.dc0 + ch] = c[ch];
        }
    }
    auto cloud = h->loader.upload(b.raw);
    REQUIRE(cloud);
    const std::vector<render::SplatInstance> instances{{&*cloud, render::Mat4::identity()}};
    const render::Camera camera = render::Camera::lookingAt({0.0, 0.0, 10.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 64;
    settings.height = 48;
    const gpu::Buffer analytic = analyticBlend(*gpu, settings.width, settings.height,
                                               {red[0], red[1], red[2], 0.5F}, {blue[0], blue[1], blue[2], 0.5F});
    const auto against = [&](const render::RenderTargets& drawn, const char* route) {
        auto diff = render::compareImages(*gpu->library, drawn.colour, analytic, settings.width, settings.height);
        if (!diff) FAIL(diff.error().toString());
        std::printf("  %s against the linear mix: p99 %u, max %u, %llu of %llu pixels over 2\n", route, diff->p99,
                    diff->max, static_cast<unsigned long long>(diff->over2),
                    static_cast<unsigned long long>(diff->pixels));
        return *diff;
    };
    // Blended in sRGB and linearised at the end, as clouds were, the red
    // channel is 0.32 where the mix is 0.41: 15 codes off at every pixel.
    render::RenderTargets rasterised, reference;
    REQUIRE(h->raster.render(camera, instances, settings, rasterised));
    const auto raster = against(rasterised, "rasteriser");
    CHECK(raster.max <= 1);
    auto overflow = h->reference.render(camera, instances, settings, reference);
    REQUIRE(overflow);
    const auto truth = against(reference, "reference");
    CHECK(truth.max <= 1);
    if (gpu->device->caps().rayQuery && gpu->device->caps().accelerationStructure) {
        auto tracer = render::GaussianRayTracer::create(*gpu->library);
        if (!tracer) FAIL(tracer.error().toString());
        render::RenderTargets traced;
        REQUIRE(tracer->render(camera, instances, settings, traced));
        const auto rays = against(traced, "ray tracer");
        CHECK(rays.max <= 1);
    }
}

TEST_CASE("a capture keeps its look when its splats are blended in linear light", "[render][gpu][linear]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // A capture: sRGB colours, as every trainer writes them. Drawn now, each
    // splat is made linear before the blend.
    CloudBuilder built = randomCloud(3000, 7);
    auto capture = h->loader.upload(built.raw);
    REQUIRE(capture);
    CHECK_FALSE(capture->linear);
    // What it was drawn as before: the same sRGB values blended as they are
    // (a cloud that says its colours are light is taken as it is) and the
    // finished contribution taken through the curve, which over nothing is
    // exactly what the blend wrote then.
    built.raw.linear = true;
    auto asItWas = h->loader.upload(built.raw);
    REQUIRE(asItWas);
    const render::Camera camera = render::Camera::lookingAt({1.0, 2.0, 6.0}, {0.0, 0.0, 0.0});
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::RenderTargets now, before;
    const std::vector<render::SplatInstance> nowList{{&*capture, render::Mat4::identity()}};
    const std::vector<render::SplatInstance> beforeList{{&*asItWas, render::Mat4::identity()}};
    REQUIRE(h->raster.render(camera, nowList, settings, now));
    REQUIRE(h->raster.render(camera, beforeList, settings, before));
    {
        auto finish = gpu::ComputeKernel::create(*gpu->library, "athenea/test/linear_blend", "srgbFinish");
        if (!finish) FAIL(finish.error().toString());
        const uint32_t pixels = settings.width * settings.height;
        gpu::CommandBatch batch(*gpu->device);
        finish->dispatch(batch, {pixels, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["finished"].setBinding(before.colour.rhi());
            cursor["finishParams"]["pixels"].setData(pixels);
        });
        REQUIRE(batch.submit(true));
    }
    auto diff = render::compareImages(*gpu->library, now.colour, before.colour, settings.width, settings.height);
    if (!diff) FAIL(diff.error().toString());
    std::printf("  a capture, linear blend against the sRGB blend: p99 %u, max %u, %llu of %llu pixels over 2\n",
                diff->p99, diff->max, static_cast<unsigned long long>(diff->over2),
                static_cast<unsigned long long>(diff->pixels));
    // Where one splat covers a pixel the two are the same; where several
    // overlap, the mean of their light is brighter than the light of their
    // mean (the curve is convex). This cloud is the worst case for it --
    // every splat a random colour, so a pixel mixes black with white, which
    // is 128 codes blended encoded and 188 blended as light -- and measured
    // p99 51, max 63. Real captures, whose neighbours agree, move less:
    // decisions.md has train, drjohnson and a beetle at p99 55, 24 and 14
    // over far fewer pixels. The bound is the measurement with a margin of a
    // few codes: what it guards is that the decode stays per splat and sRGB.
    CHECK(diff->p99 <= 55);
    CHECK(diff->max <= 70);
}

TEST_CASE("the counters a panel reads say how many splats were kept and why the rest were not",
          "[render][gpu][counters]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto h = harness(gpu);
    // A camera at the origin looking down -Z, and splats placed so that
    // each one's fate is known: kept, behind the eye, too small to leave a
    // mark, and well outside the frustum's sides.
    CloudBuilder first;
    for (int k = 0; k < 10; ++k) {   // kept
        first.add(-0.9F + 0.2F * float(k), 0.0F, -10.0F, 0.8F, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0}, {1, 1, 1});
    }
    for (int k = 0; k < 4; ++k) {    // behind the eye
        first.add(0.1F * float(k), 0.0F, 5.0F, 0.8F, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0}, {1, 1, 1});
    }
    // Too small to see: the pixel filter spreads each over a pixel and pays
    // for it in opacity, which leaves less than 1/255. (A splat faint in the
    // file never gets this far: the loader drops it.)
    for (int k = 0; k < 3; ++k) {
        first.add(0.1F * float(k), 0.3F, -10.0F, 0.8F, 1.0e-5F, 1.0e-5F, 1.0e-5F, {1, 0, 0, 0}, {1, 1, 1});
    }
    for (int k = 0; k < 5; ++k) {    // far to the side
        first.add(200.0F + float(k), 0.0F, -10.0F, 0.8F, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0}, {1, 1, 1});
    }
    CloudBuilder second;
    for (int k = 0; k < 6; ++k) {    // kept, the second cloud's
        second.add(-0.5F + 0.2F * float(k), -0.4F, -8.0F, 0.8F, 0.05F, 0.05F, 0.05F, {1, 0, 0, 0}, {1, 1, 1});
    }
    auto a = h->loader.upload(first.raw);
    auto b = h->loader.upload(second.raw);
    REQUIRE(a);
    REQUIRE(b);
    REQUIRE(a->count == 22);
    REQUIRE(b->count == 6);
    const std::vector<render::SplatInstance> instances{{&*a, render::Mat4::identity()},
                                                       {&*b, render::Mat4::identity()}};
    render::Camera camera;
    camera.lens.focal = camera.lens.haperture;
    render::RenderSettings settings;
    settings.width = 320;
    settings.height = 240;
    settings.countSplats = true;
    render::RenderTargets targets;

    // Nothing is there before a frame has counted anything.
    CHECK_FALSE(h->raster.latestCounters().has_value());

    for (uint64_t tag : {41u, 42u}) {
        settings.countersTag = tag;
        auto stats = h->raster.render(camera, instances, settings, targets);
        REQUIRE(stats);
        // The frame waits for itself, so its counts are the newest finished.
        auto counted = h->raster.latestCounters();
        REQUIRE(counted.has_value());
        CHECK(counted->tag == tag);
        CHECK(counted->slots == 28);
        CHECK(counted->visible == 16);
        CHECK(counted->visible == stats->visible);
        CHECK(counted->pairs == stats->pairs);
        CHECK(counted->maxTiles >= 1);
        CHECK(counted->culled[render::SplatCounters::Depth] == 4);
        CHECK(counted->culled[render::SplatCounters::Faint] == 3);
        CHECK(counted->culled[render::SplatCounters::Offscreen] == 5);
        CHECK(counted->culled[render::SplatCounters::Edit] == 0);
        CHECK(counted->culled[render::SplatCounters::Unprojected] == 0);
        uint32_t culled = 0;
        for (uint32_t n : counted->culled) {
            culled += n;
        }
        CHECK(culled + counted->visible == counted->slots);
        REQUIRE(counted->clouds.size() == 2);
        CHECK(counted->clouds[0].visible == 10);
        CHECK(counted->clouds[1].visible == 6);
        CHECK(counted->clouds[0].pairs + counted->clouds[1].pairs == counted->pairs);
    }

    // A frame that does not ask leaves the last count where it was.
    settings.countSplats = false;
    auto quiet = h->raster.render(camera, instances, settings, targets);
    REQUIRE(quiet);
    auto still = h->raster.latestCounters();
    REQUIRE(still.has_value());
    CHECK(still->tag == 42);
}
