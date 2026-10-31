// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER IN A .athc (SSF, phase 0): blobs and anchors made on the
// device go into the file and come back word for word, whole and through a
// stream; a file without one reads as it always did; a solid header that is
// not this reader's is refused with its reason. What is read back are
// counters, never the layer.
#include "../gpu/GpuTest.h"
#include "../render/SplatFixtures.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/scene/GpuClouds.h"

using namespace athenea;

namespace {

constexpr uint32_t kBlobs = 5003;     // not a multiple of a workgroup
constexpr uint32_t kAnchors = 301;
constexpr uint32_t kSeed = 0x55F0u;
// FileHeader: version at 4, flags at 76; the solid header at 136 (after the
// 104-byte header and the 32-byte extra header's place).
constexpr uint64_t kFlagsAt = 76;
constexpr uint64_t kSolidAt = 136;
constexpr uint32_t kSolidBit = 1u << 7;

gpu::ComputeKernel ssfKernel(test::Gpu& gpu, const char* entry) {
    auto made = gpu::ComputeKernel::create(*gpu.library, "athenea/test/ssf_check", entry);
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

gpu::Buffer deviceWords(gpu::Device& device, uint64_t count, uint32_t element, const char* label) {
    gpu::BufferDesc desc;
    desc.bytes = std::max<uint64_t>(count, 1) * element;
    desc.elementBytes = element;
    desc.label = label;
    auto made = gpu::Buffer::create(device, desc);
    REQUIRE(made);
    return std::move(*made);
}

/// A layer from ssf_check's hash: every field varied.
lod::SolidLayer generated(test::Gpu& gpu) {
    lod::SolidLayer solid;
    solid.blobs = kBlobs;
    solid.anchors = kAnchors;
    solid.positions = deviceWords(*gpu.device, kBlobs, 16, "test.solid.positions");
    solid.shape = deviceWords(*gpu.device, uint64_t{kBlobs} * 4, 4, "test.solid.shape");
    solid.anchorWords = deviceWords(*gpu.device, uint64_t{kAnchors} * 4, 4, "test.solid.anchors");
    static gpu::ComputeKernel kGenerate = ssfKernel(gpu, "ssfGenerate");
    gpu::Buffer unused = test::uintBuffer(*gpu.device, 4, "test.solid.wrong");
    gpu::CommandBatch batch(*gpu.device);
    kGenerate.dispatch(batch, {std::max(kBlobs, kAnchors), 1, 1}, [&](rhi::ShaderCursor c) {
        c["blobPositions"].setBinding(solid.positions.rhi());
        c["blobShape"].setBinding(solid.shape.rhi());
        c["anchors"].setBinding(solid.anchorWords.rhi());
        c["wrong"].setBinding(unused.rhi());
        c["params"]["blobs"].setData(kBlobs);
        c["params"]["anchors"].setData(kAnchors);
        c["params"]["seed"].setData(kSeed);
    });
    REQUIRE(batch.submit(true));
    return solid;
}

/// ssfPackCheck's counters on a layer: words a repack changes, blobs and
/// anchors off what was generated, blobs read.
std::array<uint32_t, 4> packCheck(test::Gpu& gpu, const lod::SolidLayer& solid) {
    static gpu::ComputeKernel kCheck = ssfKernel(gpu, "ssfPackCheck");
    gpu::Buffer wrong = test::uintBuffer(*gpu.device, 4, "test.solid.wrong");
    gpu::CommandBatch batch(*gpu.device);
    kCheck.dispatch(batch, {std::max(solid.blobs, solid.anchors), 1, 1}, [&](rhi::ShaderCursor c) {
        c["blobPositions"].setBinding(solid.positions.rhi());
        c["blobShape"].setBinding(solid.shape.rhi());
        c["anchors"].setBinding(solid.anchorWords.rhi());
        c["wrong"].setBinding(wrong.rhi());
        c["params"]["blobs"].setData(solid.blobs);
        c["params"]["anchors"].setData(solid.anchors);
        c["params"]["seed"].setData(kSeed);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 4> counts{};
    REQUIRE(wrong.read(*gpu.device, 0, sizeof(counts), counts.data()));
    return counts;
}

/// Words that differ between two layers, all three arrays.
uint64_t wordsApart(test::Gpu& gpu, const lod::SolidLayer& a, const lod::SolidLayer& b) {
    REQUIRE(a.blobs == b.blobs);
    REQUIRE(a.anchors == b.anchors);
    uint64_t off = 0;
    for (const auto& [x, y, words] : {std::tuple{&a.positions, &b.positions, a.blobs * 4},
                                      std::tuple{&a.shape, &b.shape, a.blobs * 4},
                                      std::tuple{&a.anchorWords, &b.anchorWords, a.anchors * 4}}) {
        auto different = render::countDifferent(*gpu.library, *x, *y, words);
        REQUIRE(different);
        off += *different;
    }
    return off;
}

lod::LodCloud builtCloud(test::Gpu& gpu) {
    auto loader = scene::CloudLoader::create(*gpu.library);
    auto builder = lod::LodBuilder::create(*gpu.library);
    if (!loader) FAIL(loader.error().toString());
    if (!builder) FAIL(builder.error().toString());
    test::CloudBuilder built = test::randomCloud(8000, 61, 0.003F, 0.05F);
    auto cloud = loader->upload(built.raw, 3);
    REQUIRE(cloud);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto lod = builder->build(*cloud, chunked);
    if (!lod) FAIL(lod.error().toString());
    return std::move(*lod);
}

uint32_t wordAt(const std::filesystem::path& path, uint64_t at) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(at));
    uint32_t w = 0;
    in.read(reinterpret_cast<char*>(&w), 4);
    REQUIRE(in);
    return w;
}

void patchWord(const std::filesystem::path& path, uint64_t at, uint32_t w) {
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    io.seekp(static_cast<std::streamoff>(at));
    io.write(reinterpret_cast<const char*>(&w), 4);
    REQUIRE(io);
}

}   // namespace

TEST_CASE("a solid layer's words pack and unpack to themselves", "[lod][gpu][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const lod::SolidLayer solid = generated(*gpu);
    const std::array<uint32_t, 4> counts = packCheck(*gpu, solid);
    std::printf("  %u blobs, %u anchors: %u words changed by a repack, %u blobs and %u anchors off\n", counts[3],
                kAnchors, counts[0], counts[1], counts[2]);
    CHECK(counts[3] == kBlobs);
    CHECK(counts[0] == 0);
    CHECK(counts[1] == 0);
    CHECK(counts[2] == 0);
}

TEST_CASE("a solid layer goes through a .athc word for word, whole and streamed", "[lod][gpu][athc][ssf]") {
    ATHENEA_REQUIRE_GPU(gpu);
    lod::LodCloud lod = builtCloud(*gpu);
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "athenea_test_athc_solid";
    std::filesystem::create_directories(dir);
    const std::filesystem::path plain = dir / "plain.athc";
    const std::filesystem::path solidPath = dir / "solid.athc";

    // Without a layer: no bit 7, and it reads with none, as before.
    REQUIRE(lod::writeAthc(*gpu->device, lod, plain));
    CHECK((wordAt(plain, kFlagsAt) & kSolidBit) == 0);
    {
        auto read = lod::readAthc(*gpu->device, plain);
        if (!read) FAIL(read.error().toString());
        CHECK_FALSE(read->solid.present());
        CHECK(read->count == lod.count);
    }

    lod.solid = generated(*gpu);
    REQUIRE(lod::writeAthc(*gpu->device, lod, solidPath));
    CHECK(std::filesystem::file_size(solidPath) % 4096 == 0);
    CHECK(wordAt(solidPath, 4) == 2);   // version 2: flags are set
    CHECK((wordAt(solidPath, kFlagsAt) & kSolidBit) != 0);

    SECTION("read whole") {
        auto read = lod::readAthc(*gpu->device, solidPath);
        if (!read) FAIL(read.error().toString());
        REQUIRE(read->solid.present());
        CHECK(wordsApart(*gpu, lod.solid, read->solid) == 0);
        const std::array<uint32_t, 4> counts = packCheck(*gpu, read->solid);
        CHECK(counts[0] == 0);
        CHECK(counts[1] == 0);
        CHECK(counts[2] == 0);
        // The residual layer is the cloud's gaussians, unchanged by the layer.
        CHECK(read->count == lod.count);
        CHECK(read->levels.size() == lod.levels.size());
        auto different = render::countDifferent(*gpu->library, lod.splats.shape, read->splats.shape, lod.count * 4);
        REQUIRE(different);
        CHECK(*different == 0);
    }
    SECTION("streamed") {
        auto pool = lod::StreamingPool::open(*gpu->device, solidPath);
        if (!pool) FAIL(pool.error().toString());
        REQUIRE((*pool)->cloud().solid.present());
        CHECK(wordsApart(*gpu, lod.solid, (*pool)->cloud().solid) == 0);
    }
    SECTION("a solid header this reader does not read is refused, saying why") {
        const uint32_t version = wordAt(solidPath, kSolidAt);
        REQUIRE(version == 1);
        patchWord(solidPath, kSolidAt, 2);
        auto later = lod::readAthc(*gpu->device, solidPath);
        REQUIRE_FALSE(later);
        CHECK(later.error().toString().find("solid layer version 2, this reads 1") != std::string::npos);
        patchWord(solidPath, kSolidAt, 1);
        patchWord(solidPath, kSolidAt + 4, 0x7FFFFFFFu);   // blobs past the end
        auto past = lod::readAthc(*gpu->device, solidPath);
        REQUIRE_FALSE(past);
        CHECK(past.error().toString().find("solid layer past the end") != std::string::npos);
    }
    std::filesystem::remove_all(dir);
}

// What a reader from before bit 7 does with the file: the same as with any
// bit it does not know (the bit 3 case in test_lod.cpp). Here a bit past 7,
// so this reader is that older one for it.
TEST_CASE("a .athc with a flag bit past the solid layer's is refused, naming it", "[lod][athc][ssf]") {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "athenea_test_athc_bit8";
    std::filesystem::create_directories(dir);
    const std::filesystem::path in = dir / "future.athc";
    const std::filesystem::path out = dir / "copied.athc";
    std::vector<char> page(4096, 0);
    std::memcpy(page.data(), "ATHC", 4);
    const uint32_t version = 2;
    const uint32_t flags = kSolidBit | (1u << 8);
    std::memcpy(page.data() + 4, &version, 4);
    std::memcpy(page.data() + kFlagsAt, &flags, 4);
    {
        std::ofstream o(in, std::ios::binary | std::ios::trunc);
        o.write(page.data(), static_cast<std::streamsize>(page.size()));
        REQUIRE(o);
    }
    auto refused = lod::migrateLrtc(in, out);
    REQUIRE_FALSE(refused);
    const std::string message = refused.error().toString();
    INFO(message);
    CHECK(message.find("unknown flag bits 8;") != std::string::npos);
    CHECK(message.find("7 (solid layer)") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(out));
    std::filesystem::remove_all(dir);
}
