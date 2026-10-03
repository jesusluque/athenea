// Copyright (c) 2026 jesus luque.
//
// colour::ColourCompiler: the functions it compiles, run on the device and
// checked there against another implementation (the inverse, the matrix the
// chromaticities give, the same space reached through another config); and
// ColourNames, which is bookkeeping and is checked as such.
#include "../gpu/GpuTest.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

#include "athenea/colour/ColourCompiler.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

/// What a check kernel leaves: how many colours it compared, and the worst
/// relative difference max_c |a - b| / max(|b|, 1e-3) among them.
struct Agreement {
    uint32_t colours = 0;
    float    worst = 0.0F;
};

/// A kernel generated around two expressions of `x`, the colour a thread
/// makes from its index, compared in the thread: `imports` are the modules
/// the expressions call. Positive floats order as their bits, so the worst
/// difference is an atomic max of a uint.
gpu::ComputeKernel twoWays(test::Gpu& gpu, const std::string& imports, const std::string& colour,
                           const std::string& a, const std::string& b) {
    const std::string source = imports +
                               "RWStructuredBuffer<uint> counts;   // colours compared, worst (float bits)\n"
                               "uniform uint count;\n"
                               "[shader(\"compute\")]\n"
                               "[numthreads(64, 1, 1)]\n"
                               "void colourTwoWays(uint3 tid: SV_DispatchThreadID) {\n"
                               "    if (tid.x >= count) {\n"
                               "        return;\n"
                               "    }\n"
                               "    const float v = float(tid.x) / float(count - 1);\n"
                               "    const float4 x = " + colour + ";\n"
                               "    const float4 first = " + a + ";\n"
                               "    const float4 second = " + b + ";\n"
                               "    const float3 d = abs(first.rgb - second.rgb) / max(abs(second.rgb), 1e-3);\n"
                               "    const float worst = max(max(d.x, d.y), d.z);\n"
                               "    InterlockedAdd(counts[0], 1u);\n"
                               "    InterlockedMax(counts[1], isnan(worst) ? 0x7F800000u : asuint(worst));\n"
                               "}\n";
    char name[48];
    std::snprintf(name, sizeof(name), "athenea_test_colour_%016llx",
                  static_cast<unsigned long long>(colour::fnv1a(source)));
    auto program = gpu.library->loadSource(name, "module " + std::string(name) + ";\n" + source, {"colourTwoWays"});
    if (!program) FAIL(program.error().toString());
    auto made = gpu::ComputeKernel::create(*gpu.library, name, "colourTwoWays");
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

Agreement runTwoWays(test::Gpu& gpu, const gpu::ComputeKernel& kernel, uint32_t count,
                     std::initializer_list<const colour::ColourFunction*> functions) {
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 2, "colour.counts");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            for (const colour::ColourFunction* function : functions) {
                function->bind(cursor);
            }
            cursor["counts"].setBinding(counts.rhi());
            cursor["count"].setData(count);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t read[2] = {};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(read), read));
    Agreement out;
    out.colours = read[0];
    std::memcpy(&out.worst, &read[1], sizeof(float));
    return out;
}

}   // namespace

TEST_CASE("colour names resolve to the config's spaces, data to raw, and the file decides when they do not say",
          "[colour][names]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto compiler = colour::ColourCompiler::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    const colour::ColourNames& names = (*compiler)->names();
    const auto kind = [&](const char* name, colour::FileEncoding file = colour::FileEncoding::Linear) {
        return names.resolve(name, file).kind;
    };
    // The file decides.
    CHECK(kind("", colour::FileEncoding::Srgb8) == colour::SpaceKind::Srgb);
    CHECK(kind("auto", colour::FileEncoding::Srgb8) == colour::SpaceKind::Srgb);
    CHECK(kind("") == colour::SpaceKind::Working);
    // Data, by any of its names.
    for (const char* data : {"raw", "Raw", "data", "Non-Color", "Utility - Raw", "none", "identity"}) {
        CHECK(kind(data) == colour::SpaceKind::Raw);
    }
    // MaterialX's, UsdUVTexture's and USD's names for the two the engine
    // reads without a function.
    for (const char* srgb : {"srgb_texture", "sRGB", "srgb_rec709_scene", "sRGB - Texture"}) {
        CHECK(kind(srgb) == colour::SpaceKind::Srgb);
    }
    for (const char* linear : {"lin_rec709", "lin_rec709_scene", "linear", "Linear Rec.709 (sRGB)"}) {
        CHECK(kind(linear) == colour::SpaceKind::Working);
    }
    if (!colour::ocioBuilt()) {
        CHECK(kind("acescg") == colour::SpaceKind::Unknown);
        return;
    }
    // The rest, by alias, by role and by USD's token, to the config's name.
    for (const char* ap1 : {"acescg", "lin_ap1", "lin_ap1_scene", "ACES - ACEScg", "scene_linear"}) {
        const colour::ResolvedSpace resolved = names.resolve(ap1, colour::FileEncoding::Linear);
        CHECK(resolved.kind == colour::SpaceKind::Other);
        CHECK(resolved.name == "ACEScg");
        CHECK(!resolved.builtin);
    }
    CHECK(names.resolve("g22_rec709", colour::FileEncoding::Srgb8).name == "Gamma 2.2 Encoded Rec.709");
    CHECK(names.resolve("g24_rec709_scene", colour::FileEncoding::Srgb8).name == "Gamma 2.4 Encoded Rec.709");
    CHECK(kind("no such space") == colour::SpaceKind::Unknown);
}

TEST_CASE("sRGB to linear and back, compiled by OpenColorIO, is the identity over a 4096-value ramp",
          "[colour][ocio]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto compiler = colour::ColourCompiler::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    const std::string working = (*compiler)->names().working();
    const std::string srgb = (*compiler)->names().resolve("srgb_texture", colour::FileEncoding::Linear).name;
    auto decode = (*compiler)->function(srgb, working);
    if (!decode) FAIL(decode.error().toString());
    std::printf("  %s\n", decode->description().c_str());
    // Compiled once: the same pair is the same module.
    auto again = (*compiler)->function(srgb, working);
    REQUIRE(again);
    CHECK(again->module() == decode->module());
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO: the built-in decode has no inverse to check it against");
    }
    auto encode = (*compiler)->function(working, srgb);
    if (!encode) FAIL(encode.error().toString());
    CHECK(encode->module() != decode->module());
    const uint32_t count = 4096;
    const gpu::ComputeKernel kernel =
        twoWays(*gpu, "import " + decode->module() + ";\nimport " + encode->module() + ";\n",
                "float4(v, 0.5 * v, 1.0 - v, 1.0)", encode->entry() + "(" + decode->entry() + "(x))", "x");
    const Agreement agreement = runTwoWays(*gpu, kernel, count, {&*decode, &*encode});
    std::printf("  round trip of %u values: worst relative %.3e\n", agreement.colours, double(agreement.worst));
    CHECK(agreement.colours == count);
    CHECK(agreement.worst < 1e-5F);
}

TEST_CASE("linear Rec.709 to ACEScg, compiled by OpenColorIO, is the matrix the chromaticities give",
          "[colour][ocio]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    auto compiler = colour::ColourCompiler::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    auto toAp1 = (*compiler)->function((*compiler)->names().working(), "ACEScg");
    if (!toAp1) FAIL(toAp1.error().toString());
    std::printf("  %s\n", toAp1->description().c_str());
    // Known colours: the primaries, white, a grey, and a walk through the
    // cube in between; aces2.slang's rgbToRgb is the reference's matrix from
    // the two sets of primaries, Bradford-adapted from D65 to the ACES white.
    const uint32_t count = 512;
    const std::string colour =
        "tid.x == 0 ? float4(1, 0, 0, 1) : tid.x == 1 ? float4(0, 1, 0, 1) : tid.x == 2 ? float4(0, 0, 1, 1)"
        " : tid.x == 3 ? float4(1, 1, 1, 1) : tid.x == 4 ? float4(0.18, 0.18, 0.18, 1)"
        " : float4(frac(v * 7.0), frac(v * 13.0), frac(v * 3.0), 1.0) * 4.0";
    const gpu::ComputeKernel kernel =
        twoWays(*gpu, "import athenea.technique.aces2;\nimport " + toAp1->module() + ";\n", colour,
                toAp1->entry() + "(x)", "float4(mul(x.rgb, rgbToRgb(kRec709, kAP1)), x.a)");
    const Agreement agreement = runTwoWays(*gpu, kernel, count, {&*toAp1});
    std::printf("  %u colours: worst relative %.3e\n", agreement.colours, double(agreement.worst));
    CHECK(agreement.colours == count);
    CHECK(agreement.worst < 1e-4F);
}

TEST_CASE("a colour space the config in use lacks is reached through the studio config",
          "[colour][ocio]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    // A studio's config of three spaces, without ACEScg.
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "colour";
    fs::create_directories(dir);
    const fs::path path = dir / "small.ocio";
    {
        std::ofstream out(path);
        out << "ocio_profile_version: 2\n"
               "name: small\n"
               "roles:\n"
               "  default: linear\n"
               "  scene_linear: linear\n"
               "  aces_interchange: aces\n"
               "  data: raw\n"
               "displays:\n"
               "  sRGB:\n"
               "    - !<View> {name: Raw, colorspace: raw}\n"
               "colorspaces:\n"
               "  - !<ColorSpace>\n"
               "    name: aces\n"
               "  - !<ColorSpace>\n"
               "    name: linear\n"
               "    aliases: [lin_rec709_scene]\n"
               "    from_scene_reference: !<MatrixTransform> {matrix: [2.52168618674388, -1.13413098823972, "
               "-0.387555198504164, 0, -0.276479914229922, 1.37271908766826, -0.096239173438334, 0, "
               "-0.0153780649660342, -0.152975335867399, 1.16835340083343, 0, 0, 0, 0, 1]}\n"
               "  - !<ColorSpace>\n"
               "    name: raw\n"
               "    isdata: true\n";
    }
    auto small = colour::ColourCompiler::create(*gpu->library, path.string());
    if (!small) FAIL(small.error().toString());
    auto studio = colour::ColourCompiler::create(*gpu->library);
    if (!studio) FAIL(studio.error().toString());
    CHECK((*small)->names().working() == "linear");
    const colour::ResolvedSpace ap1 = (*small)->names().resolve("acescg", colour::FileEncoding::Linear);
    CHECK(ap1.kind == colour::SpaceKind::Other);
    CHECK(ap1.builtin);
    CHECK((*small)->names().resolve("raw", colour::FileEncoding::Linear).kind == colour::SpaceKind::Raw);
    auto across = (*small)->function(ap1.name, (*small)->names().working());
    if (!across) FAIL(across.error().toString());
    auto within = (*studio)->function("ACEScg", (*studio)->names().working());
    if (!within) FAIL(within.error().toString());
    std::printf("  %s\n  %s\n", across->description().c_str(), within->description().c_str());
    const uint32_t count = 256;
    const gpu::ComputeKernel kernel =
        twoWays(*gpu, "import " + across->module() + ";\nimport " + within->module() + ";\n",
                "float4(frac(v * 7.0), frac(v * 13.0), frac(v * 3.0), 1.0) * 2.0", across->entry() + "(x)",
                within->entry() + "(x)");
    const Agreement agreement = runTwoWays(*gpu, kernel, count, {&*across, &*within});
    std::printf("  ACEScg from the studio config into a config without it: worst relative %.3e\n",
                double(agreement.worst));
    CHECK(agreement.colours == count);
    CHECK(agreement.worst < 1e-5F);
}
