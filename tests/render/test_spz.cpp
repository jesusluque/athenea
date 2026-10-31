// Copyright (c) 2026 jesus luque.
//
// An SPZ written by Niantic's own packer, from the same Gaussians a PLY
// carries, renders as that PLY does: the check that this engine's reading of
// SPZ's quantisation and axes is Niantic's, not only this repository's
// reading of the format. Niantic's code only writes the fixture; the
// comparison is two GPU renders compared on the GPU.
#include "../gpu/GpuTest.h"
#include "SplatFixtures.h"

#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <vector>

#include "load-spz.h"
#include "athenea/io/Readers.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;
namespace fs = std::filesystem;

// Measured on these 1500 splats and set just above: p99 3 without harmonics,
// then 4, 10 and 13 through degrees 1, 2 and 3 -- SPZ keeps degree 1 in 5 bits
// and the bands above in 4. With the signs of bands 2 and 3 deliberately
// wrong the same comparison gave 136 and 234, so a sign error cannot hide
// inside these bounds.
constexpr uint32_t kFlatP99 = 4;
constexpr uint32_t kShadedP99 = 16;

TEST_CASE("an SPZ from Niantic's packer renders as the PLY it was packed from", "[render][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t version = GENERATE(3u, 4u);   // gzip, zstd
    std::printf("  SPZ version %u\n", version);

    // Degree-3 records in the 3DGS PLY encoding: x y z, logit opacity, log
    // scales, w x y z, SH DC, 45 rest all red then green then blue. Degree 3
    // so every band's change of sign under the axis turn is exercised.
    test::CloudBuilder built = test::randomCloud(1500, 17, 0.02F, 0.4F);
    constexpr size_t kStride = 14 + 45;
    io::RawSplats raw;
    raw.source = "synthetic degree 3";
    raw.count = built.raw.count;
    raw.encoding = built.raw.encoding;
    raw.encoding.floatsPerRecord = kStride;
    raw.encoding.restPerColour = 15;
    raw.records.assign(size_t{raw.count} * kStride, 0.0F);
    test::Lcg rng;
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* from = built.raw.records.data() + i * built.raw.encoding.floatsPerRecord;
        float* to = raw.records.data() + i * kStride;
        std::copy(from, from + 14, to);
        for (size_t k = 0; k < 45; ++k) {
            // Large enough to survive 4-bit quantisation of the upper bands.
            to[14 + k] = rng.range(-0.6F, 0.6F);
        }
    }

    spz::GaussianCloud cloud;
    cloud.numPoints = static_cast<int32_t>(raw.count);
    cloud.shDegree = 3;
    for (uint32_t i = 0; i < raw.count; ++i) {
        const float* r = raw.records.data() + i * kStride;
        cloud.positions.insert(cloud.positions.end(), {r[0], r[1], r[2]});
        cloud.alphas.push_back(r[3]);
        cloud.scales.insert(cloud.scales.end(), {r[4], r[5], r[6]});
        cloud.rotations.insert(cloud.rotations.end(), {r[8], r[9], r[10], r[7]});   // x y z w
        cloud.colors.insert(cloud.colors.end(), {r[11], r[12], r[13]});
        for (int basis = 0; basis < 15; ++basis) {
            for (int colour = 0; colour < 3; ++colour) {
                cloud.sh.push_back(r[14 + colour * 15 + basis]);   // rgb per basis
            }
        }
    }
    // Said trained with the filter, as the frame draws the PLY by default: the
    // file's flag decides how a cloud from it is drawn.
    cloud.antialiased = true;
    spz::PackOptions options;
    options.version = version;
    options.from = spz::CoordinateSystem::RDF;   // what a 3DGS PLY is in
    const fs::path path = fs::temp_directory_path() / "athenea-tests" / ("packed-v" + std::to_string(version) + ".spz");
    fs::create_directories(path.parent_path());
    REQUIRE(spz::saveSpz(cloud, options, path.string()));

    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto fromPly = loader->upload(raw, 3);
    REQUIRE(fromPly);
    auto spzRaw = io::readSplats(path);
    if (!spzRaw) {
        FAIL(spzRaw.error().toString());
    }
    auto fromSpz = loader->upload(*spzRaw, 3);
    REQUIRE(fromSpz);
    CHECK(fromSpz->count == fromPly->count);
    CHECK(fromSpz->degree() == 3);

    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    render::RenderSettings settings;
    settings.width = 250;
    settings.height = 190;
    render::Camera camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
    camera.lens.focal = 30.0;
    const auto compare = [&](uint32_t shDegree) {
        settings.maxShDegree = shDegree;
        render::RenderTargets a, b;
        REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&*fromPly, render::Mat4::identity()}},
                               settings, a));
        REQUIRE(raster->render(camera, std::vector<render::SplatInstance>{{&*fromSpz, render::Mat4::identity()}},
                               settings, b));
        auto diff = render::compareImages(*gpu->library, a.colour, b.colour, settings.width, settings.height);
        REQUIRE(diff);
        std::printf("  PLY vs SPZ, harmonics to degree %u: p99 %u, max %u, %llu of %llu pixels over 2\n",
                    shDegree, diff->p99, diff->max, static_cast<unsigned long long>(diff->over2),
                    static_cast<unsigned long long>(diff->pixels));
        return *diff;
    };
    // Base colour, shape and position alone: 8-bit colour, 1/16-stop scales
    // and 12-bit fixed point cost next to nothing.
    const auto flat = compare(0);
    // Band by band, what their quantisation costs (5 bits for degree 1, 4
    // above). A wrong sign on any basis costs tens of code values.
    const auto first = compare(1);
    const auto second = compare(2);
    const auto third = compare(3);
    CHECK(flat.p99 <= kFlatP99);
    CHECK(first.p99 <= kShadedP99);
    CHECK(second.p99 <= kShadedP99);
    CHECK(third.p99 <= kShadedP99);
}

// --- SPZ 4 as the input contract -----------------------------------------------

#include <fstream>
#include <memory>

#include "coordinate-system-adobe.h"
#include "athenea/io/SplatWriters.h"

namespace {

/// A degree-3 cloud as Niantic's types hold it, from records in the 3DGS PLY
/// encoding (the fixture of the test above, smaller).
struct Fixture {
    io::RawSplats       ply;
    spz::GaussianCloud  cloud;
};

Fixture fixture(uint32_t count, uint64_t seed) {
    test::CloudBuilder built = test::randomCloud(count, seed, 0.02F, 0.4F);
    constexpr size_t kStride = 14 + 45;
    Fixture f;
    f.ply.source = "synthetic degree 3";
    f.ply.count = built.raw.count;
    f.ply.encoding = built.raw.encoding;
    f.ply.encoding.floatsPerRecord = kStride;
    f.ply.encoding.restPerColour = 15;
    f.ply.records.assign(size_t{f.ply.count} * kStride, 0.0F);
    test::Lcg rng;
    rng.state ^= seed;
    for (uint32_t i = 0; i < f.ply.count; ++i) {
        const float* from = built.raw.records.data() + i * built.raw.encoding.floatsPerRecord;
        float* to = f.ply.records.data() + i * kStride;
        std::copy(from, from + 14, to);
        for (size_t k = 0; k < 45; ++k) {
            to[14 + k] = rng.range(-0.6F, 0.6F);
        }
    }
    f.cloud.numPoints = static_cast<int32_t>(f.ply.count);
    f.cloud.shDegree = 3;
    for (uint32_t i = 0; i < f.ply.count; ++i) {
        const float* r = f.ply.records.data() + i * kStride;
        f.cloud.positions.insert(f.cloud.positions.end(), {r[0], r[1], r[2]});
        f.cloud.alphas.push_back(r[3]);
        f.cloud.scales.insert(f.cloud.scales.end(), {r[4], r[5], r[6]});
        f.cloud.rotations.insert(f.cloud.rotations.end(), {r[8], r[9], r[10], r[7]});
        f.cloud.colors.insert(f.cloud.colors.end(), {r[11], r[12], r[13]});
        for (int basis = 0; basis < 15; ++basis) {
            for (int colour = 0; colour < 3; ++colour) {
                f.cloud.sh.push_back(r[14 + colour * 15 + basis]);
            }
        }
    }
    return f;
}

/// `cloud` (in PLY's right-down-front) saved as SPZ `version`; `stored` is
/// the system the file keeps its data in, said by the coordinate-system
/// extension where it is not the format's default right-up-back.
fs::path save(spz::GaussianCloud cloud, uint32_t version, spz::CoordinateSystem stored, bool antialiased,
              const std::string& name) {
    cloud.antialiased = antialiased;
    if (stored != spz::CoordinateSystem::RUB) {
        auto axes = std::make_shared<spz::SpzExtensionCoordinateSystemAdobe>();
        axes->coordinateSystem = stored;
        cloud.extensions.push_back(axes);
    }
    spz::PackOptions options;
    options.version = version;
    options.from = spz::CoordinateSystem::RDF;
    const fs::path path = fs::temp_directory_path() / "athenea-tests" / name;
    fs::create_directories(path.parent_path());
    REQUIRE(spz::saveSpz(cloud, options, path.string()));
    return path;
}

std::vector<uint8_t> bytesOf(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

struct Scene {
    std::unique_ptr<scene::CloudLoader>     loader;
    std::unique_ptr<render::TileRasterizer> raster;
    render::RenderSettings                  settings;
    render::Camera                          camera = render::Camera::lookingAt({1.5, 1.0, 6.0}, {0.0, 0.0, 0.0});
};

Scene sceneFor(test::Gpu* gpu) {
    Scene s;
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto raster = render::TileRasterizer::create(*gpu->library);
    REQUIRE(raster);
    s.loader = std::make_unique<scene::CloudLoader>(std::move(*loader));
    s.raster = std::make_unique<render::TileRasterizer>(std::move(*raster));
    s.settings.width = 250;
    s.settings.height = 190;
    s.camera.lens.focal = 30.0;
    return s;
}

render::ImageDifference compareClouds(test::Gpu* gpu, Scene& s, const scene::GpuSplats& a,
                                      const scene::GpuSplats& b, const render::RenderSettings& settingsB) {
    render::RenderTargets ta, tb;
    REQUIRE(s.raster->render(s.camera, std::vector<render::SplatInstance>{{&a, render::Mat4::identity()}},
                             s.settings, ta));
    REQUIRE(s.raster->render(s.camera, std::vector<render::SplatInstance>{{&b, render::Mat4::identity()}},
                             settingsB, tb));
    auto diff = render::compareImages(*gpu->library, ta.colour, tb.colour, s.settings.width, s.settings.height);
    REQUIRE(diff);
    return *diff;
}

scene::GpuSplats uploadSpz(Scene& s, const fs::path& path) {
    auto raw = io::readSplats(path);
    if (!raw) FAIL(raw.error().toString());
    auto cloud = s.loader->upload(*raw, 3);
    if (!cloud) FAIL(cloud.error().toString());
    return std::move(*cloud);
}

}   // namespace

TEST_CASE("an SPZ read from memory is the SPZ read from its file, kept packed until the device",
          "[render][io][gpu][spz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t version = GENERATE(3u, 4u);
    Fixture f = fixture(800, 23);
    const fs::path path = save(f.cloud, version, spz::CoordinateSystem::RUB, false,
                               "memory-v" + std::to_string(version) + ".spz");
    const std::vector<uint8_t> bytes = bytesOf(path);
    auto fromMemory = io::readSpz(std::span<const uint8_t>(bytes), "memory");
    if (!fromMemory) FAIL(fromMemory.error().toString());
    // The six streams as the file packed them: no record expanded on the CPU.
    CHECK(fromMemory->encoding.record == io::SplatEncoding::Record::SpzPacked);
    CHECK(fromMemory->records.empty());
    CHECK(!fromMemory->packed.empty());
    CHECK(fromMemory->packed.size() <= size_t{f.ply.count} * 65 + 24);   // 65 bytes a splat at degree 3, padded

    Scene s = sceneFor(gpu);
    auto memoryCloud = s.loader->upload(*fromMemory, 3);
    REQUIRE(memoryCloud);
    const scene::GpuSplats fileCloud = uploadSpz(s, path);
    CHECK(memoryCloud->count == fileCloud.count);
    CHECK(compareClouds(gpu, s, *memoryCloud, fileCloud, s.settings).max == 0);

    // What a host-side consumer reads (the USD export): the records the
    // device expanded, read back -- the same cloud again.
    auto records = s.loader->records(*fromMemory);
    if (!records) FAIL(records.error().toString());
    CHECK(records->encoding.record == io::SplatEncoding::Record::Floats);
    CHECK(records->records.size() == size_t{records->count} * records->encoding.floatsPerRecord);
    auto again = s.loader->upload(*records, 3);
    REQUIRE(again);
    CHECK(compareClouds(gpu, s, *again, fileCloud, s.settings).max == 0);
}

TEST_CASE("an SPZ stored right-down-front reads as the one stored right-up-back", "[render][io][gpu][spz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const uint32_t version = GENERATE(3u, 4u);
    Fixture f = fixture(1500, 29);
    const fs::path rub = save(f.cloud, version, spz::CoordinateSystem::RUB, true,
                              "rub-v" + std::to_string(version) + ".spz");
    const fs::path rdf = save(f.cloud, version, spz::CoordinateSystem::RDF, true,
                              "rdf-v" + std::to_string(version) + ".spz");
    Scene s = sceneFor(gpu);
    auto fromPly = s.loader->upload(f.ply, 3);
    REQUIRE(fromPly);
    const scene::GpuSplats a = uploadSpz(s, rub);
    const scene::GpuSplats b = uploadSpz(s, rdf);
    for (const uint32_t degree : {0u, 3u}) {
        s.settings.maxShDegree = degree;
        const auto rubDiff = compareClouds(gpu, s, *fromPly, a, s.settings);
        const auto rdfDiff = compareClouds(gpu, s, *fromPly, b, s.settings);
        std::printf("  v%u, degree %u: PLY vs RUB p99 %u, vs RDF p99 %u\n", version, degree, rubDiff.p99,
                    rdfDiff.p99);
        CHECK(rubDiff.p99 <= (degree == 0 ? kFlatP99 : kShadedP99));
        CHECK(rdfDiff.p99 <= (degree == 0 ? kFlatP99 : kShadedP99));
    }
}

TEST_CASE("an SPZ that says it was trained antialiased is drawn with the filter, whatever the frame's default",
          "[render][io][gpu][spz]") {
    ATHENEA_REQUIRE_GPU(gpu);
    Fixture f = fixture(1500, 31);
    const fs::path on = save(f.cloud, 4, spz::CoordinateSystem::RUB, true, "aa-on.spz");
    const fs::path off = save(f.cloud, 4, spz::CoordinateSystem::RUB, false, "aa-off.spz");
    Scene s = sceneFor(gpu);
    const scene::GpuSplats withFilter = uploadSpz(s, on);
    const scene::GpuSplats without = uploadSpz(s, off);
    REQUIRE(withFilter.antialiased.has_value());
    CHECK(*withFilter.antialiased);
    CHECK(!*without.antialiased);
    for (const bool frameDefault : {false, true}) {
        s.settings.antialias = frameDefault;
        // The file decides: the antialiased one drawn by default is the other
        // drawn with the filter forced on, and the other is not.
        render::RenderSettings forcedOn = s.settings;
        forcedOn.antialiasOverride = true;
        render::RenderSettings forcedOff = s.settings;
        forcedOff.antialiasOverride = false;
        const auto same = compareClouds(gpu, s, withFilter, without, forcedOn);
        const auto apart = compareClouds(gpu, s, withFilter, without, s.settings);
        const auto overridden = compareClouds(gpu, s, without, withFilter, forcedOff);
        std::printf("  frame default %d: forced on max %u; as the files say max %u; forced off vs default max %u\n",
                    frameDefault ? 1 : 0, same.max, apart.max, overridden.max);
        CHECK(same.max == 0);
        CHECK(apart.max > 0);
        CHECK(overridden.max == 0);
    }
}

TEST_CASE("an SPZ this engine writes says its axes are right-up-back", "[io][spz]") {
    constexpr uint32_t kCount = 4;
    std::array<std::vector<uint8_t>, 6> streams{std::vector<uint8_t>(kCount * 9, 1), std::vector<uint8_t>(kCount, 200),
                                                std::vector<uint8_t>(kCount * 3, 128), std::vector<uint8_t>(kCount * 3, 100),
                                                std::vector<uint8_t>(kCount * 4, 0), std::vector<uint8_t>()};
    for (const uint32_t version : {3u, 4u}) {
        if (version == 4 && !io::writesSpzVersion4()) {
            continue;
        }
        io::SpzHeader header;
        header.version = version;
        header.count = kCount;
        header.shDegree = 0;
        const fs::path path = fs::temp_directory_path() / "athenea-tests" / ("axes-v" + std::to_string(version) + ".spz");
        fs::create_directories(path.parent_path());
        REQUIRE(io::writeSpz(path, header, streams));
        const spz::PackedGaussians packed = spz::loadSpzPacked(path.string());
        REQUIRE(packed.numPoints == static_cast<int32_t>(kCount));
        CHECK(!packed.hadSkippedExtensions);
        auto axes = spz::findExtensionByType<spz::SpzExtensionCoordinateSystemAdobe>(packed.extensions);
        REQUIRE(axes != nullptr);
        CHECK(axes->coordinateSystem == spz::CoordinateSystem::RUB);
        auto raw = io::readSplats(path);
        REQUIRE(raw);
        CHECK(raw->encoding.flipYZ);
    }
}
