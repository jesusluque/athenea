// Copyright (c) 2026 jesus luque.
//
// A MERGE TREE READ AS ITS LEAVES (usd::uploadParticleField, athenea convert):
// a field marked `lodTree` whose gaussians standing for one leaf (`lodSize`
// 1) are the cloud, and whose merges -- here written far off in x -- are not
// kept. The leaves' transfer comes with them. Checked on the device; the
// count and two counters are what is read back.
#include "../gpu/GpuTest.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"

using namespace athenea;
namespace fs = std::filesystem;

TEST_CASE("a merge tree is read as its leaves", "[usd][gpu][tree]") {
    ATHENEA_REQUIRE_GPU(gpu);
    // Six leaves on a line at x 0..0.5, then two merges of three at x 100.
    constexpr int kLeaves = 6, kMerges = 2, kCount = kLeaves + kMerges;
    std::ostringstream out;
    out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Z\"\n)\n"
           "def Xform \"World\"\n{\n    def ParticleField3DGaussianSplat \"Splats\"\n    {\n"
           "        point3f[] positions = [";
    for (int k = 0; k < kCount; ++k) {
        out << (k ? ", " : "") << "(" << (k < kLeaves ? 0.1 * k : 100.0) << ", 0, 0)";
    }
    out << "]\n        quatf[] orientations = [";
    for (int k = 0; k < kCount; ++k) out << (k ? ", " : "") << "(1, 0, 0, 0)";
    out << "]\n        float3[] scales = [";
    for (int k = 0; k < kCount; ++k) out << (k ? ", " : "") << "(0.05, 0.05, 0.001)";
    out << "]\n        float[] opacities = [";
    for (int k = 0; k < kCount; ++k) out << (k ? ", " : "") << "0.9";
    out << "]\n        int radiance:sphericalHarmonicsDegree = 0\n"
           "        float3[] radiance:sphericalHarmonicsCoefficients = [";
    for (int k = 0; k < kCount; ++k) out << (k ? ", " : "") << "(0.5, 0.5, 0.5)";
    out << "]\n        float[] primvars:athenea:splat:transferDirect = [";
    for (int k = 0; k < kCount * 16; ++k) out << (k ? ", " : "") << (k % 16 == 0 ? 1 : 0);
    out << "] ( elementSize = 16\n interpolation = \"vertex\" )\n"
           "        bool primvars:athenea:splat:lodTree = 1\n"
           "        int[] primvars:athenea:splat:lodSize = [1, 1, 1, 1, 1, 1, 3, 3]\n"
           "    }\n}\n";
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "tree";
    fs::create_directories(dir);
    const fs::path path = dir / "tree.usda";
    {
        std::ofstream file(path);
        file << out.str();
    }
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = usd::uploadParticleField(*gpu->library, *loader, path);
    if (!splats) FAIL(splats.error().toString());
    CHECK(splats->count == kLeaves);
    CHECK(splats->transfer.valid());
    CHECK(splats->transfer.bytes() >= uint64_t{kLeaves} * splats->transferWords * 4);
    gpu::Buffer wrong = test::uintBuffer(*gpu->device, 2, "test.wrong");
    gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/tree_leaves_check");
    gpu::CommandBatch batch(*gpu->device);
    check.dispatch(batch, {splats->count, 1, 1}, [&](rhi::ShaderCursor c) {
        c["positions"].setBinding(splats->positions.rhi());
        c["wrong"].setBinding(wrong.rhi());
        c["params"]["count"].setData(splats->count);
        c["params"]["limit"].setData(50.0F);
    });
    REQUIRE(batch.submit(true));
    std::array<uint32_t, 2> counts{};
    REQUIRE(wrong.read(*gpu->device, 0, sizeof(counts), counts.data()));
    CHECK(counts[1] == kLeaves);
    CHECK(counts[0] == 0);
}
