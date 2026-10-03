// Copyright (c) 2026 jesus luque.
//
// Loading: files written by hand with values whose decoded form is known
// exactly, decoded on the GPU, and unpacked on the GPU for comparison.
#include "../gpu/GpuTest.h"

#include <catch2/catch_approx.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

#ifdef ATHENEA_TEST_SPZ
#include <zlib.h>
#endif

#include "athenea/io/Readers.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/scene/SplatSkinner.h"

using namespace athenea;
using Catch::Approx;
namespace fs = std::filesystem;

namespace {

fs::path scratchFile(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests";
    fs::create_directories(dir);
    return dir / name;
}

/// One 3DGS record's properties, in file order, degree 1 (9 rest coefficients).
struct PlyRecord {
    float x, y, z, nx, ny, nz;
    float dc[3];
    float rest[9];
    float opacity;
    float scale[3];
    float rot[4];   // w x y z
};

void writeSplatPly(const fs::path& path, const std::vector<PlyRecord>& records) {
    std::ofstream out(path, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\n"
        << "element vertex " << records.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property float nx\nproperty float ny\nproperty float nz\n"
        << "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n";
    for (int k = 0; k < 9; ++k) {
        out << "property float f_rest_" << k << "\n";
    }
    out << "property float opacity\n"
        << "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        << "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        << "end_header\n";
    out.write(reinterpret_cast<const char*>(records.data()),
              static_cast<std::streamsize>(records.size() * sizeof(PlyRecord)));
}

std::array<float, 20> inspect(test::Gpu& gpu, const scene::GpuSplats& splats, uint32_t index) {
    static gpu::ComputeKernel kInspect = test::kernel(gpu, "athenea/test/splat_inspect");
    gpu::BufferDesc desc;
    desc.bytes = 20 * sizeof(float);
    desc.elementBytes = sizeof(float);
    auto values = gpu::Buffer::create(*gpu.device, desc);
    REQUIRE(values);
    gpu::CommandBatch batch(*gpu.device);
    kInspect.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["positions"].setBinding(splats.positions.rhi());
        cursor["shape"].setBinding(splats.shape.rhi());
        cursor["sh"].setBinding(splats.sh.rhi());
        cursor["values"].setBinding(values->rhi());
        cursor["params"]["index"].setData(index);
        cursor["params"]["shWords"].setData(splats.shWords);
        cursor["params"]["hasSh"].setData(uint32_t{splats.restPerColour > 0 ? 1u : 0u});
    });
    REQUIRE(batch.submit(true));
    std::array<float, 20> out{};
    REQUIRE(values->read(*gpu.device, 0, sizeof(out), out.data()));
    return out;
}

}   // namespace

TEST_CASE("a 3DGS PLY decodes opacity, scale, rotation, colour and harmonics exactly",
          "[scene][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // logit(0) = 0.5; log scales 0, ln2, -ln2; quaternion (w=2, x=0, y=0, z=0)
    // unnormalised -> identity; f_dc 0 -> base 0.5; rest rrr ggg bbb.
    PlyRecord first{};
    first.x = 1; first.y = 2; first.z = 3;
    first.opacity = 0.0F;
    first.scale[0] = 0.0F; first.scale[1] = 0.6931472F; first.scale[2] = -0.6931472F;
    first.rot[0] = 2.0F;
    for (int k = 0; k < 9; ++k) {
        first.rest[k] = 0.1F * static_cast<float>(k + 1);   // r: .1 .2 .3  g: .4 .5 .6  b: .7 .8 .9
    }
    // Invisible: logit -10 is opacity 4.5e-5 < 1/255. Dropped.
    PlyRecord faint = first;
    faint.opacity = -10.0F;
    // Broken: a NaN position. Dropped.
    PlyRecord broken = first;
    broken.x = nan;
    // Kept, after two dropped ones: compaction must keep file order.
    PlyRecord last = first;
    last.x = -4; last.y = 5; last.z = -6;
    last.opacity = 1.0986123F;   // logit(0.75)
    last.rot[0] = 0.0F; last.rot[3] = -3.0F;   // 180 degrees about z, sign folded
    last.dc[0] = 1.0F / 0.28209479F;   // base colour 1.5

    const fs::path path = scratchFile("decode.ply");
    writeSplatPly(path, {first, faint, broken, last});

    auto raw = io::readSplats(path);
    REQUIRE(raw);
    CHECK(raw->count == 4);
    CHECK(raw->encoding.restPerColour == 3);
    auto loader = scene::CloudLoader::create(*gpu->library);
    if (!loader) {
        FAIL(loader.error().toString());
    }
    auto splats = loader->upload(*raw, 3);
    REQUIRE(splats);
    CHECK(splats->count == 2);
    CHECK(splats->degree() == 1);

    const auto a = inspect(*gpu, *splats, 0);
    CHECK(a[0] == 1.0F);
    CHECK(a[1] == 2.0F);
    CHECK(a[2] == 3.0F);
    CHECK(a[3] == Approx(0.5).margin(1e-6));
    CHECK(a[4] == Approx(1.0).epsilon(2e-3));
    CHECK(a[5] == Approx(2.0).epsilon(2e-3));
    CHECK(a[6] == Approx(0.5).epsilon(2e-3));
    CHECK(a[10] == Approx(1.0).margin(1e-3));   // w
    CHECK(a[11] == Approx(0.5).margin(1e-3));   // base colour r
    // rgb per basis: basis 1 is (r .1, g .4, b .7), basis 2 starts (r .2, ...)
    CHECK(a[14] == Approx(0.1).margin(1e-3));
    CHECK(a[15] == Approx(0.4).margin(1e-3));
    CHECK(a[16] == Approx(0.7).margin(1e-3));
    CHECK(a[17] == Approx(0.2).margin(1e-3));

    const auto b = inspect(*gpu, *splats, 1);
    CHECK(b[0] == -4.0F);
    CHECK(b[3] == Approx(0.75).margin(1e-5));
    CHECK(std::abs(b[9]) == Approx(1.0).margin(1e-3));   // |z| = 1
    CHECK(b[11] == Approx(1.5).margin(1e-3));

    CHECK(splats->bounds.min[0] == -4.0F);
    CHECK(splats->bounds.max[0] == 1.0F);
    CHECK(splats->bounds.min[2] == -6.0F);
    CHECK(splats->bounds.max[2] == 3.0F);

    // Degree capped at 0: no harmonics kept.
    auto flat = loader->upload(*raw, 0);
    REQUIRE(flat);
    CHECK(flat->degree() == 0);
}

TEST_CASE("a .splat file decodes byte-encoded opacity, colour and rotation", "[scene][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    struct Record {
        float pos[3];
        float scale[3];
        unsigned char rgba[4];
        unsigned char rot[4];
    };
    Record r{{0.5F, -0.5F, 2.0F}, {0.25F, 0.5F, 1.0F}, {255, 0, 51, 255}, {255, 128, 128, 128}};
    const fs::path path = scratchFile("decode.splat");
    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&r), sizeof r);
    }
    auto raw = io::readSplats(path);
    REQUIRE(raw);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(*raw);
    REQUIRE(splats);
    REQUIRE(splats->count == 1);
    const auto v = inspect(*gpu, *splats, 0);
    CHECK(v[3] == Approx(1.0).margin(1e-6));
    CHECK(v[4] == Approx(0.25).epsilon(2e-3));
    CHECK(v[6] == Approx(1.0).epsilon(2e-3));
    CHECK(v[10] == Approx(1.0).margin(1e-2));   // w from byte 255 -> 0.992, normalised
    CHECK(v[11] == Approx(1.0).margin(1e-3));
    CHECK(v[13] == Approx(0.2).margin(1e-3));
}

#ifdef ATHENEA_TEST_SPZ
namespace {

/// A legacy SPZ (v2/v3): a 16-byte header and the attribute streams, gzipped
/// together -- written here byte by byte, so the test knows every value.
void writeSpz(const fs::path& path, uint32_t version, uint32_t points, uint8_t shDegree,
              uint8_t fractionalBits, const std::vector<uint8_t>& streams) {
    std::vector<uint8_t> bytes(16, 0);
    const uint32_t magic = 0x5053474e;
    std::memcpy(bytes.data(), &magic, 4);
    std::memcpy(bytes.data() + 4, &version, 4);
    std::memcpy(bytes.data() + 8, &points, 4);
    bytes[12] = shDegree;
    bytes[13] = fractionalBits;
    bytes.insert(bytes.end(), streams.begin(), streams.end());
    gzFile out = gzopen(path.string().c_str(), "wb");
    REQUIRE(out != nullptr);
    REQUIRE(gzwrite(out, bytes.data(), static_cast<unsigned>(bytes.size())) == static_cast<int>(bytes.size()));
    gzclose(out);
}

void appendInt24(std::vector<uint8_t>& out, int32_t v) {
    const auto u = static_cast<uint32_t>(v);
    out.push_back(static_cast<uint8_t>(u & 0xFF));
    out.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((u >> 16) & 0xFF));
}

}   // namespace

TEST_CASE("an SPZ file dequantises on the GPU and turns right-up-back into the PLY's axes",
          "[scene][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);

    SECTION("version 3: smallest-three rotation, degree-1 harmonics") {
        // Two points; the second is transparent (alpha byte 0) and dropped.
        std::vector<uint8_t> s;
        appendInt24(s, 4096); appendInt24(s, 8192); appendInt24(s, -2048);   // (1, 2, -0.5) at 12 bits
        appendInt24(s, 0); appendInt24(s, 0); appendInt24(s, 0);
        s.push_back(191); s.push_back(0);                                  // alphas
        s.insert(s.end(), {255, 128, 0, 128, 128, 128});                   // colours
        s.insert(s.end(), {160, 176, 144, 160, 160, 160});                 // log scales 0, 1, -1
        // Largest component w (index 3), y = +1/sqrt2, x = z = 0.
        const uint32_t comp = (3u << 30) | (511u << 10);
        for (int k = 0; k < 2; ++k) {
            for (int b = 0; b < 4; ++b) {
                s.push_back(static_cast<uint8_t>((comp >> (8 * b)) & 0xFF));
            }
        }
        // Harmonics, rgb per basis: basis 0 (y) +0.5, basis 1 (z) -0.5, basis 2 (x) mixed.
        const std::vector<uint8_t> sh{192, 192, 192, 64, 64, 64, 192, 64, 128};
        s.insert(s.end(), sh.begin(), sh.end());
        s.insert(s.end(), sh.begin(), sh.end());
        const fs::path path = scratchFile("decode-v3.spz");
        writeSpz(path, 3, 2, 1, 12, s);

        auto raw = io::readSplats(path);
        if (!raw) {
            FAIL(raw.error().toString());
        }
        auto splats = loader->upload(*raw, 3);
        REQUIRE(splats);
        REQUIRE(splats->count == 1);
        CHECK(splats->degree() == 1);
        const auto v = inspect(*gpu, *splats, 0);
        // Right-up-back (1, 2, -0.5) is right-down-front (1, -2, 0.5).
        CHECK(v[0] == 1.0F);
        CHECK(v[1] == -2.0F);
        CHECK(v[2] == 0.5F);
        CHECK(v[3] == Approx(191.0 / 255.0).margin(1e-6));
        CHECK(v[4] == Approx(1.0).epsilon(2e-3));
        CHECK(v[5] == Approx(std::exp(1.0)).epsilon(2e-3));
        CHECK(v[6] == Approx(std::exp(-1.0)).epsilon(2e-3));
        // A quarter turn about y becomes one about -y: y and w of opposite
        // sign (the packing may negate the whole quaternion).
        CHECK(std::abs(v[8]) == Approx(0.70710678).margin(3e-3));
        CHECK(std::abs(v[10]) == Approx(0.70710678).margin(3e-3));
        CHECK(v[8] * v[10] < 0.0F);
        CHECK(std::abs(v[7]) < 3e-3F);
        CHECK(std::abs(v[9]) < 3e-3F);
        // Colour through SPZ's DC scale (0.15), then 0.5 + SH0 * dc.
        const auto base = [](double byte) { return 0.5 + 0.28209479 * ((byte / 255.0 - 0.5) / 0.15); };
        CHECK(v[11] == Approx(base(255)).margin(1e-3));
        CHECK(v[12] == Approx(base(128)).margin(1e-3));
        CHECK(v[13] == Approx(base(0)).margin(1e-3));
        // Basis 0 is odd in y and basis 1 in z: both change sign.
        CHECK(v[14] == Approx(-0.5).margin(1e-3));
        CHECK(v[15] == Approx(-0.5).margin(1e-3));
        CHECK(v[17] == Approx(0.5).margin(1e-3));
        CHECK(v[19] == Approx(0.5).margin(1e-3));
    }

    SECTION("version 2: first-three rotation, no harmonics") {
        std::vector<uint8_t> s;
        appendInt24(s, -256); appendInt24(s, 256); appendInt24(s, 512);   // (-1, 1, 2) at 8 bits
        s.push_back(255);
        s.insert(s.end(), {128, 128, 128});
        s.insert(s.end(), {160, 160, 160});
        s.insert(s.end(), {128, 128, 255});   // x, y ~ 0, z = 1: a half turn about z
        const fs::path path = scratchFile("decode-v2.spz");
        writeSpz(path, 2, 1, 0, 8, s);
        auto raw = io::readSplats(path);
        if (!raw) {
            FAIL(raw.error().toString());
        }
        auto splats = loader->upload(*raw, 3);
        REQUIRE(splats);
        REQUIRE(splats->count == 1);
        CHECK(splats->degree() == 0);
        const auto v = inspect(*gpu, *splats, 0);
        CHECK(v[0] == -1.0F);
        CHECK(v[1] == -1.0F);
        CHECK(v[2] == -2.0F);
        CHECK(v[3] == Approx(1.0).margin(1e-6));
        CHECK(std::abs(v[9]) == Approx(1.0).margin(1e-2));
    }
}
#endif

TEST_CASE("point files of every kind load, with 8-bit colour linearised on the GPU",
          "[scene][io][gpu]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);

    const fs::path ascii = scratchFile("points.ply");
    {
        std::ofstream out(ascii);
        out << "ply\nformat ascii 1.0\nelement vertex 3\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
               "0 0 0 255 0 0\n1 2 3 0 255 0\n-1 -2 -3 0 0 255\n";
    }
    auto rawAscii = io::readPoints(ascii);
    REQUIRE(rawAscii);
    CHECK(rawAscii->count == 3);
    CHECK(rawAscii->colourKind == 1);
    auto cloud = loader->upload(*rawAscii);
    REQUIRE(cloud);
    CHECK(cloud->count == 3);
    CHECK(cloud->bounds.min[1] == -2.0F);
    CHECK(cloud->bounds.max[2] == 3.0F);

    const fs::path colmap = scratchFile("points3D.txt");
    {
        std::ofstream out(colmap);
        out << "# 3D point list\n1 0.5 0.5 0.5 128 128 128 0.1 1 2\n2 1.5 0.5 0.5 10 20 30 0.2\n";
    }
    auto rawColmap = io::readPoints(colmap);
    REQUIRE(rawColmap);
    CHECK(rawColmap->count == 2);
    CHECK(rawColmap->records[0] == 0.5F);   // the id was skipped

    // detail keeps a fraction, the same subset every time.
    std::vector<float> many;
    const fs::path xyz = scratchFile("many.xyz");
    {
        std::ofstream out(xyz);
        for (int i = 0; i < 20000; ++i) {
            out << i << " 0 0\n";
        }
    }
    auto rawMany = io::readPoints(xyz);
    REQUIRE(rawMany);
    CHECK(rawMany->count == 20000);
    auto half = loader->upload(*rawMany, 0.5F);
    REQUIRE(half);
    CHECK(half->count > 9000);
    CHECK(half->count < 11000);
    auto again = loader->upload(*rawMany, 0.5F);
    REQUIRE(again);
    CHECK(again->count == half->count);
}

TEST_CASE("a trained cloud from openFXplayer's examples loads", "[scene][io][gpu][data]") {
    const fs::path train = fs::path(std::getenv("HOME") != nullptr ? std::getenv("HOME") : "") /
                           "openFXplayer/examples/media/train_7k.ply";
    std::error_code ignored;
    if (!fs::exists(train, ignored)) {
        SKIP("no " + train.string());
    }
    ATHENEA_REQUIRE_GPU(gpu);
    auto raw = io::readSplats(train);
    REQUIRE(raw);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(*raw);
    REQUIRE(splats);
    CHECK(splats->count > splats->declared / 2);
    CHECK(splats->degree() == 3);
    CHECK(splats->bounds.min[0] < splats->bounds.max[0]);
    std::printf("%s: %u of %u splats, bounds x %.2f..%.2f\n", train.c_str(), splats->count,
                splats->declared, static_cast<double>(splats->bounds.min[0]),
                static_cast<double>(splats->bounds.max[0]));
}

// A SKELETON AT REST LEAVES A CLOUD WHERE IT WAS, AND A SKELETON THAT MOVES
// RIGIDLY CARRIES IT RIGIDLY.
//
// Those are the two invariants a cloud carried by a rig stands on, and they
// are enough to catch every way the transform chain can be wrong: a matrix
// transposed, a translation dropped, the frame turned by the wrong part of
// the chain, a size scaled when it should not be. Both the cloud and the
// answer are made and compared on the device; only four counters come back.
TEST_CASE("a skinned cloud follows its joints, and nothing else moves", "[scene][gpu][skinning]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto make = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatSkinMake");
    auto joints = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatSkinJoints");
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatSkinCompare");
    auto skinner = scene::SplatSkinner::create(*gpu->library);
    if (!make) FAIL(make.error().toString());
    if (!joints) FAIL(joints.error().toString());
    if (!compare) FAIL(compare.error().toString());
    if (!skinner) FAIL(skinner.error().toString());

    constexpr uint32_t kCount = 4096;
    constexpr uint32_t kPerSplat = 4;
    constexpr uint32_t kJoints = 3;
    const auto buffer = [&](uint64_t bytes, uint32_t element, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(gpu->library->device(), desc);
        REQUIRE(made);
        return std::move(*made);
    };
    gpu::Buffer restPositions = buffer(uint64_t{kCount} * 16, 16, "skin.restPositions");
    gpu::Buffer restShape = buffer(uint64_t{kCount} * 4 * 4, 4, "skin.restShape");
    gpu::Buffer influences = buffer(uint64_t{kCount} * kPerSplat * 8, 8, "skin.influences");
    gpu::Buffer xforms = buffer(uint64_t{kJoints} * 4 * 16, 16, "skin.xforms");
    gpu::Buffer positions = buffer(uint64_t{kCount} * 16, 16, "skin.positions");
    gpu::Buffer shape = buffer(uint64_t{kCount} * 4 * 4, 4, "skin.shape");
    gpu::Buffer counts = buffer(8 * 4, 4, "skin.counts");

    scene::GpuSplats rest;
    rest.count = kCount;
    rest.positions = std::move(restPositions);
    rest.shape = std::move(restShape);

    struct Case {
        const char*          name;
        std::array<float, 4> rotation;      // x, y, z, w
        std::array<float, 3> translation;
        float                tolerance;
    };
    const float half = 0.6F;   // a turn of 1.2 radians about a slanted axis
    const float s = std::sin(half);
    const float axis[3] = {0.4082483F, 0.8164966F, 0.4082483F};
    // The tolerance is what the format can hold, not what the arithmetic can:
    // a frame is stored as a smallest-three quaternion at ten bits a
    // component, so an axis cannot be pinned closer than about `sqrt(2)/1023`.
    // At rest the frame written is the frame read, and re-encoding a value
    // that was already the decode of a word lands on that word -- except at
    // the boundary of the rounding, where two gaussians of 4096 did.
    constexpr float kQuantised = 4.0e-3F;
    const std::array<Case, 2> cases{
        Case{"at rest", {0.0F, 0.0F, 0.0F, 1.0F}, {0.0F, 0.0F, 0.0F}, kQuantised},
        Case{"turned and slid",
             {axis[0] * s, axis[1] * s, axis[2] * s, std::cos(half)},
             {1.5F, -0.75F, 0.25F},
             kQuantised}};

    for (const Case& one : cases) {
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["restPositions"].setBinding(rest.positions.rhi());
            cursor["restShape"].setBinding(rest.shape.rhi());
            cursor["influences"].setBinding(influences.rhi());
            cursor["xforms"].setBinding(xforms.rhi());
            cursor["positions"].setBinding(positions.rhi());
            cursor["shape"].setBinding(shape.rhi());
            cursor["counts"].setBinding(counts.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(kCount);
            p["perSplat"].setData(kPerSplat);
            p["joints"].setData(kJoints);
            p["rotation"].setData(one.rotation.data(), 16);
            const std::array<float, 4> slide{one.translation[0], one.translation[1], one.translation[2], 0.0F};
            p["translation"].setData(slide.data(), 16);
            p["tolerance"].setData(one.tolerance);
        };
        {
            gpu::CommandBatch batch(gpu->library->device());
            make->dispatch(batch, {kCount, 1, 1}, bind);
            joints->dispatch(batch, {kJoints, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(gpu->library->device());
            scene::SplatSkinInput input;
            input.rest = &rest;
            input.influences = &influences;
            input.perSplat = kPerSplat;
            input.skinningXforms = &xforms;
            REQUIRE(skinner->skin(batch, input, positions, shape));
            REQUIRE(batch.submit(true));
        }
        {
            const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            REQUIRE(counts.write(gpu->library->device(), 0, sizeof(zero), zero));
            gpu::CommandBatch batch(gpu->library->device());
            compare->dispatch(batch, {kCount, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[4] = {0, 0, 0, 0};
        REQUIRE(counts.read(gpu->library->device(), 0, sizeof(seen), seen));
        std::printf("  %-16s %u compared, %u misplaced, %u resized, %u misturned\n", one.name, seen[0],
                    seen[1], seen[2], seen[3]);
        CHECK(seen[0] == kCount);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
        CHECK(seen[3] == 0);
    }

    // AND WHAT THE SHUTTER MOVED, from the same two poses.
    //
    // The displacement is the difference between the poses, so two of the
    // same must give nothing and a pure slide must give exactly the slide.
    // Both are read back as one counter, as everything else here is.
    auto motionCheck = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatSkinMotion");
    if (!motionCheck) FAIL(motionCheck.error().toString());
    gpu::Buffer xformsEnd = buffer(uint64_t{kJoints} * 4 * 16, 16, "skin.xformsEnd");
    gpu::Buffer motion = buffer(uint64_t{kCount} * 2 * 4, 4, "skin.motion");

    struct Moved {
        const char*          name;
        std::array<float, 3> slide;
    };
    const std::array<Moved, 2> moves{Moved{"the same pose twice", {0.0F, 0.0F, 0.0F}},
                                     Moved{"a slide on every joint", {0.25F, -0.5F, 0.125F}}};
    for (const Moved& one : moves) {
        const auto bindPose = [&](rhi::ShaderCursor cursor, gpu::Buffer& into,
                                  const std::array<float, 3>& slide) {
            cursor["restPositions"].setBinding(rest.positions.rhi());
            cursor["restShape"].setBinding(rest.shape.rhi());
            cursor["influences"].setBinding(influences.rhi());
            cursor["xforms"].setBinding(into.rhi());
            cursor["positions"].setBinding(positions.rhi());
            cursor["shape"].setBinding(shape.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["motion"].setBinding(motion.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(kCount);
            p["perSplat"].setData(kPerSplat);
            p["joints"].setData(kJoints);
            const std::array<float, 4> still{0.0F, 0.0F, 0.0F, 1.0F};
            p["rotation"].setData(still.data(), 16);
            const std::array<float, 4> shift{slide[0], slide[1], slide[2], 0.0F};
            p["translation"].setData(shift.data(), 16);
            p["tolerance"].setData(2.0e-3F);
        };
        {
            gpu::CommandBatch batch(gpu->library->device());
            make->dispatch(batch, {kCount, 1, 1},
                           [&](rhi::ShaderCursor c) { bindPose(c, xforms, {0.0F, 0.0F, 0.0F}); });
            joints->dispatch(batch, {kJoints, 1, 1},
                             [&](rhi::ShaderCursor c) { bindPose(c, xforms, {0.0F, 0.0F, 0.0F}); });
            joints->dispatch(batch, {kJoints, 1, 1},
                             [&](rhi::ShaderCursor c) { bindPose(c, xformsEnd, one.slide); });
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(gpu->library->device());
            scene::SplatSkinInput input;
            input.rest = &rest;
            input.influences = &influences;
            input.perSplat = kPerSplat;
            input.skinningXforms = &xforms;
            input.skinningXformsEnd = &xformsEnd;
            REQUIRE(skinner->skin(batch, input, positions, shape, &motion));
            REQUIRE(batch.submit(true));
        }
        {
            const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            REQUIRE(counts.write(gpu->library->device(), 0, sizeof(zero), zero));
            gpu::CommandBatch batch(gpu->library->device());
            motionCheck->dispatch(batch, {kCount, 1, 1},
                                  [&](rhi::ShaderCursor c) { bindPose(c, xforms, one.slide); });
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[4] = {0, 0, 0, 0};
        REQUIRE(counts.read(gpu->library->device(), 0, sizeof(seen), seen));
        std::printf("  %-24s %u compared, %u displaced wrongly\n", one.name, seen[0], seen[1]);
        CHECK(seen[0] == kCount);
        CHECK(seen[1] == 0);
    }
}

// A SHADING NORMAL TURNS WITH THE LIMB IT IS ON.
//
// A converted gaussian keeps the normal its mesh's normal map gave it, apart
// from its frame (GpuSplats::normals). A skeleton that carried the frame and
// left that normal where the bind pose put it would light a raised arm as
// though it still hung down: the relief would stay with the light, not with
// the arm. So the skinner turns it by the same blend -- as a normal, by the
// inverse transpose, which for a rigid joint is the turn itself and for one
// that stretches is not -- and it stays on the side of its disc it was on.
// Made and compared on the device; three counters come back.
TEST_CASE("a skinned cloud's shading normals turn with its joints", "[scene][gpu][skinning][normals]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = gpu->library->device();
    auto make = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalSkinMake");
    auto joints = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalSkinJoints");
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_normal_check", "normalSkinCompare");
    auto skinner = scene::SplatSkinner::create(*gpu->library);
    if (!make) FAIL(make.error().toString());
    if (!joints) FAIL(joints.error().toString());
    if (!compare) FAIL(compare.error().toString());
    if (!skinner) FAIL(skinner.error().toString());

    constexpr uint32_t kCount = 4096;
    constexpr uint32_t kPerSplat = 4;
    constexpr uint32_t kJoints = 2;
    const auto buffer = [&](uint64_t bytes, uint32_t element, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(device, desc);
        REQUIRE(made);
        return std::move(*made);
    };
    scene::GpuSplats rest;
    rest.count = kCount;
    rest.positions = buffer(uint64_t{kCount} * 16, 16, "normals.restPositions");
    rest.shape = buffer(uint64_t{kCount} * 16, 4, "normals.restShape");
    rest.normals = buffer(uint64_t{kCount} * 4, 4, "normals.restNormals");
    gpu::Buffer influences = buffer(uint64_t{kCount} * kPerSplat * 8, 8, "normals.influences");
    gpu::Buffer xforms = buffer(uint64_t{kJoints} * 4 * 16, 16, "normals.xforms");
    gpu::Buffer positions = buffer(uint64_t{kCount} * 16, 16, "normals.positions");
    gpu::Buffer shape = buffer(uint64_t{kCount} * 16, 4, "normals.shape");
    gpu::Buffer normals = buffer(uint64_t{kCount} * 4, 4, "normals.posed");
    gpu::Buffer counts = buffer(8 * 4, 4, "normals.counts");

    struct Case {
        const char*                         name;
        std::array<std::array<float, 4>, 3> rows;   // linear part, translation in w
    };
    // A turn of 1.2 radians about a slanted axis (Rodrigues' rows), and a
    // joint that stretches and shears -- where a normal turned as a direction
    // would go wrong.
    const float c = std::cos(1.2F), s = std::sin(1.2F);
    const float ax = 0.4082483F, ay = 0.8164966F, az = 0.4082483F;
    const float k = 1.0F - c;
    const std::array<Case, 2> cases{
        Case{"turned",
             {{{c + ax * ax * k, ax * ay * k - az * s, ax * az * k + ay * s, 1.5F},
               {ay * ax * k + az * s, c + ay * ay * k, ay * az * k - ax * s, -0.75F},
               {az * ax * k - ay * s, az * ay * k + ax * s, c + az * az * k, 0.25F}}}},
        Case{"stretched and sheared",
             {{{2.0F, 0.3F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.4F, 0.5F}, {0.0F, 0.0F, 0.5F, 0.0F}}}}};
    for (const Case& one : cases) {
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["restPositions"].setBinding(rest.positions.rhi());
            cursor["restShape"].setBinding(rest.shape.rhi());
            cursor["restNormals"].setBinding(rest.normals.rhi());
            cursor["influences"].setBinding(influences.rhi());
            cursor["xforms"].setBinding(xforms.rhi());
            cursor["shape"].setBinding(shape.rhi());
            cursor["normals"].setBinding(normals.rhi());
            cursor["counts"].setBinding(counts.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(kCount);
            p["perSplat"].setData(kPerSplat);
            p["joints"].setData(kJoints);
            p["row0"].setData(one.rows[0].data(), 16);
            p["row1"].setData(one.rows[1].data(), 16);
            p["row2"].setData(one.rows[2].data(), 16);
            // Sixteen bits a component of the octahedral square: a few
            // thousandths of a degree, far inside this.
            p["tolerance"].setData(1.0e-5F);
        };
        {
            gpu::CommandBatch batch(device);
            make->dispatch(batch, {kCount, 1, 1}, bind);
            joints->dispatch(batch, {kJoints, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(device);
            scene::SplatSkinInput input;
            input.rest = &rest;
            input.influences = &influences;
            input.perSplat = kPerSplat;
            input.skinningXforms = &xforms;
            REQUIRE(skinner->skin(batch, input, positions, shape, nullptr, &normals));
            REQUIRE(batch.submit(true));
        }
        {
            const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            REQUIRE(counts.write(device, 0, sizeof(zero), zero));
            gpu::CommandBatch batch(device);
            compare->dispatch(batch, {kCount, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[3] = {0, 0, 0};
        REQUIRE(counts.read(device, 0, sizeof(seen), seen));
        std::printf("  %-22s %u compared, %u not turned as a normal, %u on the other side of their disc\n",
                    one.name, seen[0], seen[1], seen[2]);
        CHECK(seen[0] == kCount);
        CHECK(seen[1] == 0);
        CHECK(seen[2] == 0);
    }
}

// A GAUSSIAN ACROSS A RAMP OF WEIGHTS IS STRETCHED BY THE WHOLE JACOBIAN.
//
// The posed map of linear blend skinning is `p' = sum_j w_j(p) X_j p`, and a
// gaussian's frame has to follow its derivative -- the blend of the joints'
// linear parts *and* `sum_j (X_j p) grad w_j^T`, what the weights changing
// across the gaussian do. Taking only the first leaves a bent limb's
// gaussians a third too short at the middle of the bend, and squaring their
// axes up afterwards leaves a sheared joint's unsheared. A strip bent a
// quarter turn across a ramp of weights, a rigid control and a joint that
// shears, each compared on the device with `J E S^2 E^T J^T`; one counter
// comes back. And the same bend with no gradients is the blend alone, as a
// cloud converted before them is drawn.
TEST_CASE("a skinned gaussian follows the whole Jacobian of its blend", "[scene][gpu][skinning][jacobian]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = gpu->library->device();
    auto make = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatStripMake");
    auto joints = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatStripJoints");
    auto compare = gpu::ComputeKernel::create(*gpu->library, "athenea/test/splat_skin_check", "splatStripCompare");
    auto skinner = scene::SplatSkinner::create(*gpu->library);
    if (!make) FAIL(make.error().toString());
    if (!joints) FAIL(joints.error().toString());
    if (!compare) FAIL(compare.error().toString());
    if (!skinner) FAIL(skinner.error().toString());

    constexpr uint32_t kCount = 4096;
    constexpr uint32_t kPerSplat = 4;
    constexpr uint32_t kJoints = 2;
    const auto buffer = [&](uint64_t bytes, uint32_t element, const char* label) {
        gpu::BufferDesc desc;
        desc.bytes = bytes;
        desc.elementBytes = element;
        desc.label = label;
        auto made = gpu::Buffer::create(device, desc);
        REQUIRE(made);
        return std::move(*made);
    };
    scene::GpuSplats rest;
    rest.count = kCount;
    rest.positions = buffer(uint64_t{kCount} * 16, 16, "strip.restPositions");
    rest.shape = buffer(uint64_t{kCount} * 16, 4, "strip.restShape");
    gpu::Buffer influences = buffer(uint64_t{kCount} * kPerSplat * 8, 8, "strip.influences");
    gpu::Buffer gradients = buffer(uint64_t{kCount} * (kPerSplat - 1) * 4, 4, "strip.gradients");
    gpu::Buffer xforms = buffer(uint64_t{kJoints} * 4 * 16, 16, "strip.xforms");
    gpu::Buffer positions = buffer(uint64_t{kCount} * 16, 16, "strip.positions");
    gpu::Buffer shape = buffer(uint64_t{kCount} * 16, 4, "strip.shape");
    gpu::Buffer counts = buffer(8 * 4, 4, "strip.counts");

    struct Case {
        const char* name;
        uint32_t    mode;           // 0 bent, 1 rigid, 2 sheared joint
        float       frameAngle;     // the gaussians' first axis from x, radians
        bool        withGradients;
    };
    // Six degrees past a multiple of a right angle, so no frame is squared up
    // by luck, and thirty, where a bend shears the gaussians as well.
    const std::array<Case, 5> cases{Case{"bent, axes along it", 0, 0.1F, true},
                                    Case{"bent, axes turned", 0, 0.5236F, true},
                                    Case{"rigid", 1, 0.5236F, true},
                                    Case{"a sheared joint", 2, 0.5236F, true},
                                    Case{"bent, no gradients kept", 0, 0.5236F, false}};
    const float halfTurn = 0.6F;
    const float s = std::sin(halfTurn);
    for (const Case& one : cases) {
        const auto bind = [&](rhi::ShaderCursor cursor) {
            cursor["restPositions"].setBinding(rest.positions.rhi());
            cursor["restShape"].setBinding(rest.shape.rhi());
            cursor["influences"].setBinding(influences.rhi());
            cursor["gradients"].setBinding(gradients.rhi());
            cursor["xforms"].setBinding(xforms.rhi());
            cursor["shape"].setBinding(shape.rhi());
            cursor["counts"].setBinding(counts.rhi());
            rhi::ShaderCursor p = cursor["params"];
            p["count"].setData(kCount);
            p["perSplat"].setData(kPerSplat);
            p["joints"].setData(kJoints);
            const std::array<float, 4> turn{0.4082483F * s, 0.8164966F * s, 0.4082483F * s, std::cos(halfTurn)};
            p["rotation"].setData(turn.data(), 16);
            const std::array<float, 4> slide{1.5F, -0.75F, 0.25F, 0.0F};
            p["translation"].setData(slide.data(), 16);
            // Above what the format holds -- ten bits a quaternion component,
            // halves for the log sizes and the gradients -- and far below the
            // tenth to a third the blend alone is off by across the ramp.
            p["tolerance"].setData(0.03F);
            p["mode"].setData(one.mode);
            p["frameAngle"].setData(one.frameAngle);
        };
        {
            gpu::CommandBatch batch(device);
            make->dispatch(batch, {kCount, 1, 1}, bind);
            joints->dispatch(batch, {kJoints, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        {
            gpu::CommandBatch batch(device);
            scene::SplatSkinInput input;
            input.rest = &rest;
            input.influences = &influences;
            input.perSplat = kPerSplat;
            input.weightGradients = one.withGradients ? &gradients : nullptr;
            input.skinningXforms = &xforms;
            REQUIRE(skinner->skin(batch, input, positions, shape));
            REQUIRE(batch.submit(true));
        }
        {
            const uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            REQUIRE(counts.write(device, 0, sizeof(zero), zero));
            gpu::CommandBatch batch(device);
            compare->dispatch(batch, {kCount, 1, 1}, bind);
            REQUIRE(batch.submit(true));
        }
        uint32_t seen[2] = {0, 0};
        REQUIRE(counts.read(device, 0, sizeof(seen), seen));
        std::printf("  %-24s %u compared, %u off the posed covariance by more than 3%%\n", one.name, seen[0],
                    seen[1]);
        CHECK(seen[0] == kCount);
        if (one.withGradients) {
            CHECK(seen[1] == 0);
        } else {
            // The ramp is a half of the strip and nearly all of it is off.
            CHECK(seen[1] > kCount / 4);
        }
    }
}

// THE BASIS, AGAINST ITSELF.
//
// `sh.slang` writes the harmonics twice -- `evaluateRest` for a renderer
// adding a cloud's view dependent colour, `shBasisValue` for a bake or a
// transfer projecting onto the same functions -- and says the two must agree
// term for term. Nothing checked it until a transfer became the second thing
// to project, and a sign or an order wrong in one of them is a cloud that
// decodes to something other than what was baked into it, which neither side
// can see on its own.
TEST_CASE("a bake's basis is the one a renderer reads, term for term", "[scene][gpu][sh]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr uint32_t kTerms = 15;      // degree 3, without the constant one
    constexpr uint32_t kRowWords = 23;   // 15 coefficients, rgb, two halves a word
    constexpr uint32_t kDirections = 64;

    gpu::BufferDesc desc;
    desc.bytes = size_t{kTerms} * kRowWords * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "sh.rows";
    auto rows = gpu::Buffer::create(gpu->library->device(), desc);
    REQUIRE(rows);
    gpu::Buffer stats = test::uintBuffer(gpu->library->device(), 8, "sh.stats");
    gpu::BufferDesc worstDesc;
    worstDesc.bytes = 8 * sizeof(float);
    worstDesc.elementBytes = sizeof(float);
    worstDesc.label = "sh.worst";
    const std::array<float, 8> zeros{};
    auto worst = gpu::Buffer::create(gpu->library->device(), worstDesc, zeros.data());
    REQUIRE(worst);

    auto write = gpu::ComputeKernel::create(*gpu->library, "athenea/test/sh_basis_check", "shBasisWrite");
    if (!write) FAIL(write.error().toString());
    auto check = gpu::ComputeKernel::create(*gpu->library, "athenea/test/sh_basis_check", "shBasisCheck");
    if (!check) FAIL(check.error().toString());
    const auto bind = [&](rhi::ShaderCursor cursor, bool reading) {
        if (reading) {
            cursor["sh"].setBinding(rows->rhi());
        } else {
            cursor["rows"].setBinding(rows->rhi());
        }
        cursor["stats"].setBinding(stats.rhi());
        cursor["worst"].setBinding(worst->rhi());
        cursor["params"]["terms"].setData(kTerms);
        cursor["params"]["rowWords"].setData(kRowWords);
        cursor["params"]["directions"].setData(kDirections);
        cursor["params"]["tolerance"].setData(1.0e-4F);
    };
    {
        gpu::CommandBatch batch(gpu->library->device());
        write->dispatch(batch, {kTerms, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, false); });
        REQUIRE(batch.submit(true));
    }
    {
        gpu::CommandBatch batch(gpu->library->device());
        check->dispatch(batch, {kTerms, 1, 1}, [&](rhi::ShaderCursor c) { bind(c, true); });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 8> counts{};
    std::array<float, 8> readings{};
    REQUIRE(stats.read(gpu->library->device(), 0, sizeof(counts), counts.data()));
    REQUIRE(worst->read(gpu->library->device(), 0, sizeof(readings), readings.data()));
    std::printf("  the basis: %u readings, %u beyond; the largest was %.4f and the worst reading is "
                "%.6f off\n", counts[1], counts[0], double(counts[3]) * 1.0e-6,
                double(counts[2]) * 1.0e-6);
    CHECK(counts[1] == kTerms * kDirections);
    CHECK(counts[0] == 0);
    CHECK(counts[3] > 100000u);   // a basis of order one was actually compared
}
