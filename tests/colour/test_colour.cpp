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
#include <array>
#include <optional>
#include <vector>

#include "athenea/colour/ColourCompiler.h"
#include "athenea/colour/WorkingSpace.h"
#include "athenea/gpu/WorkingSpace.h"

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

/// Phase 2a: the kernels read Rec.709's numbers from the working-space
/// record, and the record of lin_rec709_scene is those numbers. The kernel
/// holds the record the device binds against the literals, word by word,
/// and every function that reads it against the expression it replaced --
/// blackbody, a capture's decode, the edit's grade, a luminance, the XYZ
/// matrix -- by their bits, over 4096 colours and temperatures.
TEST_CASE("the working-space record of lin_rec709_scene holds the constants the kernels held, to the bit",
          "[colour][working_space]") {
    ATHENEA_REQUIRE_GPU(gpu);
    CHECK(gpu->device->workingSpace() == gpu::linRec709Scene());
    auto kernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/working_space_check", "workingSpaceCheck");
    if (!kernel) FAIL(kernel.error().toString());
    const uint32_t count = 4096;
    gpu::Buffer counts = test::uintBuffer(*gpu->device, 7, "counts");
    {
        gpu::CommandBatch batch(*gpu->device);
        kernel->dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["counts"].setBinding(counts.rhi());
            cursor["count"].setData(count);
        });
        REQUIRE(batch.submit(true));
    }
    uint32_t c[7] = {};
    REQUIRE(counts.read(*gpu->device, 0, sizeof(c), c));
    std::printf("  record words off %u; differing: blackbody %u, cloudLight %u, editGrade %u, luminance %u, "
                "matrices %u; threads %u\n", c[0], c[1], c[2], c[3], c[4], c[5], c[6]);
    CHECK(c[0] == 0);
    CHECK(c[1] == 0);
    CHECK(c[2] == 0);
    CHECK(c[3] == 0);
    CHECK(c[4] == 0);
    CHECK(c[5] == 0);
    CHECK(c[6] == count);
}

TEST_CASE("a working space is named as any colour space, and one no 3x3 reaches is refused",
          "[colour][working_space][names]") {
    ATHENEA_REQUIRE_GPU(gpu);
    for (const char* rec709 : {"", "auto", "lin_rec709_scene", "lin_rec709", "Linear Rec.709 (sRGB)"}) {
        auto chosen = colour::chooseWorkingSpace(*gpu->library, colour::kStudioConfig, rec709);
        INFO(rec709);
        REQUIRE(chosen);
        CHECK(chosen->compiler->names().workingIsRec709());
        CHECK(chosen->record == gpu::linRec709Scene());
    }
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build: only linear Rec.709 is rendered in");
    }
    // ACEScg by four of its names: the same record, built on the device.
    std::optional<gpu::WorkingSpace> ap1;
    for (const char* name : {"acescg", "lin_ap1_scene", "ACEScg", "role:scene_linear"}) {
        auto chosen = colour::chooseWorkingSpace(*gpu->library, colour::kStudioConfig, name);
        INFO(name);
        REQUIRE(chosen);
        CHECK(!chosen->compiler->names().workingIsRec709());
        CHECK(chosen->record.flags[0] == 0u);
        if (!ap1) {
            ap1 = chosen->record;
        } else {
            CHECK(chosen->record == *ap1);
        }
    }
    // Data, a curve, a name no config knows: refused, with a sentence.
    for (const char* refused : {"raw", "srgb_texture", "no such space", "ACEScct"}) {
        auto chosen = colour::chooseWorkingSpace(*gpu->library, colour::kStudioConfig, refused);
        INFO(refused);
        REQUIRE(!chosen);
        std::printf("  %s: %s\n", refused, chosen.error().message().c_str());
    }
}

/// Phase 2b: the record of a working space that is not Rec.709, built on the
/// device from OCIO's two functions, against the reference's matrices
/// (aces2.slang): bringing decoded sRGB in is Rec.709 to AP1, Bradford
/// adapted; taking it back out is the inverse; the luminance is AP1's Y row.
/// The kernels read it as they read Rec.709's table: the device is given the
/// record, and a generated kernel calls the record's own functions.
TEST_CASE("the record of ACEScg holds Rec.709 to AP1 as the reference's chromaticities give it",
          "[colour][working_space][ocio]") {
    ATHENEA_REQUIRE_GPU(gpu);
    if (!colour::ocioBuilt()) {
        SKIP("no OpenColorIO in this build");
    }
    auto chosen = colour::chooseWorkingSpace(*gpu->library, colour::kStudioConfig, "acescg");
    if (!chosen) FAIL(chosen.error().toString());
    const gpu::WorkingSpace before = gpu->device->workingSpace();
    gpu->device->setWorkingSpace(chosen->record);
    const uint32_t count = 512;
    const std::string colour =
        "tid.x == 0 ? float4(1, 0, 0, 1) : tid.x == 1 ? float4(0, 1, 0, 1) : tid.x == 2 ? float4(0, 0, 1, 1)"
        " : tid.x == 3 ? float4(1, 1, 1, 1) : float4(frac(v * 7.0), frac(v * 13.0), frac(v * 3.0), 1.0) * 4.0";
    const std::string imports = "import athenea.technique.aces2;\nimport athenea.common.working_space;\n";
    const Agreement in = runTwoWays(
        *gpu, twoWays(*gpu, imports, colour, "float4(workingFromSrgbLinear(x.rgb), x.a)",
                      "float4(mul(x.rgb, rgbToRgb(kRec709, kAP1)), x.a)"),
        count, {});
    const Agreement out = runTwoWays(
        *gpu, twoWays(*gpu, imports, colour, "float4(workingToSrgbLinear(x.rgb), x.a)",
                      "float4(mul(x.rgb, rgbToRgb(kAP1, kRec709)), x.a)"),
        count, {});
    const Agreement trip = runTwoWays(
        *gpu, twoWays(*gpu, imports, colour, "float4(workingToSrgbLinear(workingFromSrgbLinear(x.rgb)), x.a)", "x"),
        count, {});
    // A grey's luminance is the grey; a colour's, AP1's Y row (white D60, the
    // reference's; the record's is Rec.709's D65 Y row brought through the
    // config's Bradford, so the two agree to the adaptation's rounding).
    const Agreement luminance = runTwoWays(
        *gpu, twoWays(*gpu, imports, colour, "float4(float3(workingLuminance(x.rgb)), 1.0)",
                      "float4(float3(dot(x.rgb, float3(rgbToXyz(kAP1, 1.0)[0].y, rgbToXyz(kAP1, 1.0)[1].y, "
                      "rgbToXyz(kAP1, 1.0)[2].y))), 1.0)"),
        count, {});
    gpu->device->setWorkingSpace(before);
    std::printf("  ACEScg's record: in %.3e, out %.3e, round trip %.3e, luminance %.3e (worst relative); "
                "weights %.7f %.7f %.7f\n",
                double(in.worst), double(out.worst), double(trip.worst), double(luminance.worst),
                double(chosen->record.luminance[0]), double(chosen->record.luminance[1]),
                double(chosen->record.luminance[2]));
    CHECK(in.colours == count);
    CHECK(in.worst < 1e-4F);
    CHECK(out.worst < 1e-4F);
    CHECK(trip.worst < 1e-5F);
    CHECK(luminance.worst < 2e-3F);
}

namespace {

/// A buffer of floats as they are given.
gpu::Buffer floatsOnDevice(test::Gpu& gpu, const std::vector<float>& values, const char* label) {
    auto made = gpu::Buffer::fromSpan<float>(*gpu.device, values, label);
    REQUIRE(made);
    return std::move(*made);
}

/// `converted` against `original` with `function` applied in the check
/// kernel itself: every colour word the same bits as the function gives
/// (`halves`: the half it rounds to), every other word untouched. Counts
/// [colour words that differ, other words that differ, colours checked].
std::array<uint32_t, 3> checkWords(test::Gpu& gpu, const colour::ColourFunction& function, const gpu::Buffer& converted,
                                   const gpu::Buffer& original, uint32_t words, uint32_t stride, uint32_t first,
                                   bool halves) {
    const std::string f = function.entry();
    const std::string source =
        "import " + function.module() + ";\n"
        "StructuredBuffer<uint> converted;\nStructuredBuffer<uint> original;\nRWStructuredBuffer<uint> counts;\n"
        "uniform uint words;\nuniform uint stride;\nuniform uint first;\nuniform uint halves;\n"
        "[shader(\"compute\")]\n[numthreads(64, 1, 1)]\n"
        "void checkWords(uint3 tid: SV_DispatchThreadID) {\n"
        "    const uint w = tid.x;\n"
        "    if (w >= words) {\n        return;\n    }\n"
        "    const uint k = w < first ? 0xFFFFFFFFu : (w - first) % stride;\n"
        "    const uint colourWords = halves != 0 ? 2u : 3u;\n"
        "    if (k >= colourWords) {\n"
        "        InterlockedAdd(counts[1], converted[w] != original[w] ? 1u : 0u);\n        return;\n    }\n"
        "    const uint base = w - k;\n"
        "    float3 c;\n"
        "    if (halves != 0) {\n"
        "        c = float3(f16tof32(original[base] & 0xFFFFu), f16tof32(original[base] >> 16), "
        "f16tof32(original[base + 1] & 0xFFFFu));\n"
        "    } else {\n"
        "        c = float3(asfloat(original[base]), asfloat(original[base + 1]), asfloat(original[base + 2]));\n"
        "    }\n"
        "    const float3 e = " + f + "(float4(c, 1.0)).rgb;\n"
        "    uint expected;\n"
        "    if (halves != 0) {\n"
        "        expected = k == 0 ? (f32tof16(e.r) | (f32tof16(e.g) << 16)) "
        ": (f32tof16(e.b) | (original[base + 1] & 0xFFFF0000u));\n"
        "    } else {\n"
        "        expected = asuint(e[k]);\n"
        "    }\n"
        "    InterlockedAdd(counts[0], converted[w] != expected ? 1u : 0u);\n"
        "    InterlockedAdd(counts[2], k == 0 ? 1u : 0u);\n"
        "}\n";
    char name[48];
    std::snprintf(name, sizeof(name), "athenea_test_words_%016llx",
                  static_cast<unsigned long long>(colour::fnv1a(source)));
    auto program = gpu.library->loadSource(name, "module " + std::string(name) + ";\n" + source, {"checkWords"});
    if (!program) FAIL(program.error().toString());
    auto kernel = gpu::ComputeKernel::create(*gpu.library, name, "checkWords");
    if (!kernel) FAIL(kernel.error().toString());
    gpu::Buffer counts = test::uintBuffer(*gpu.device, 3, "words.counts");
    {
        gpu::CommandBatch batch(*gpu.device);
        kernel->dispatch(batch, {words, 1, 1}, [&](rhi::ShaderCursor cursor) {
            function.bind(cursor);
            cursor["converted"].setBinding(converted.rhi());
            cursor["original"].setBinding(original.rhi());
            cursor["counts"].setBinding(counts.rhi());
            cursor["words"].setData(words);
            cursor["stride"].setData(stride);
            cursor["first"].setData(first);
            cursor["halves"].setData(uint32_t{halves ? 1u : 0u});
        });
        REQUIRE(batch.submit(true));
    }
    std::array<uint32_t, 3> c{};
    REQUIRE(counts.read(*gpu.device, 0, sizeof(c), c.data()));
    return c;
}

}   // namespace

/// Phase 3: a colour that is not a texture -- a material's constant, a
/// light's colour, a displayColor -- goes to the device as authored and is
/// converted where it lies, by convertColourWords around the compiled
/// function. Each layout it is used with, strided floats (a light record, a
/// mesh's float4 primvar), four halves (a point's colour) and a list (a
/// material's row), gives every colour word the very bits the function
/// gives when a check kernel calls it, and leaves every other word alone. A
/// colour in the working space, or unsaid under Rec.709, is not dispatched.
TEST_CASE("colour constants are brought into the working space as the compiled function brings them, to the bit",
          "[colour][working_space][words]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto compiler = colour::ColourCompiler::create(*gpu->library);
    if (!compiler) FAIL(compiler.error().toString());
    colour::ColourCompiler& c = **compiler;
    // Unsaid and Rec.709 by name: nothing to do, nothing done.
    CHECK(!c.converts(""));
    CHECK(!c.converts("lin_rec709_scene"));
    CHECK(!c.converts("raw"));
    CHECK(c.converts("srgb_texture"));
    const uint32_t colours = 1000;
    const uint32_t stride = 5;   // three colour words, two of something else
    std::vector<float> values(size_t{colours} * stride + 1);
    for (size_t k = 0; k < values.size(); ++k) {
        values[k] = static_cast<float>((k * 2654435761u) % 4096) / 1024.0F - 0.5F;
    }
    const gpu::Buffer original = floatsOnDevice(*gpu, values, "words.original");
    const auto convert = [&](const char* space, const colour::ColourWordsRun& run, const gpu::Buffer& into) {
        gpu::CommandBatch batch(*gpu->device);
        auto done = c.convertWords(batch, into, space, run);
        REQUIRE(done);
        REQUIRE(batch.submit(true));
        return *done;
    };
    {
        // An unsaid space dispatches nothing, so the words stay the words.
        const gpu::Buffer same = floatsOnDevice(*gpu, values, "words.same");
        CHECK(!convert("", {1, colours, stride}, same));
        CHECK(!convert("lin_rec709", {1, colours, stride}, same));
    }
    const std::vector<const char*> spaces = colour::ocioBuilt() ? std::vector<const char*>{"acescg", "srgb_texture"}
                                                                : std::vector<const char*>{"srgb_texture"};
    for (const char* space : spaces) {
        INFO(space);
        const std::string resolved = c.names().resolve(space, colour::FileEncoding::Linear).name;
        auto function = c.function(resolved, c.names().working());
        if (!function) FAIL(function.error().toString());
        // Strided floats, after a word that is not a colour.
        const gpu::Buffer strided = floatsOnDevice(*gpu, values, "words.strided");
        CHECK(convert(space, {1, colours, stride}, strided));
        const auto s = checkWords(*gpu, *function, strided, original, static_cast<uint32_t>(values.size()), stride, 1,
                                  false);
        // Four halves a colour, two words.
        const gpu::Buffer halved = floatsOnDevice(*gpu, values, "words.halves");
        colour::ColourWordsRun halves{0, colours, 2, true};
        CHECK(convert(space, halves, halved));
        const auto h = checkWords(*gpu, *function, halved, original, colours * 2, 2, 0, true);
        // A list: every fifth word's colour, named one by one.
        std::vector<uint32_t> at;
        for (uint32_t k = 0; k < colours; ++k) {
            at.push_back(1 + k * stride);
        }
        auto list = gpu::Buffer::fromSpan<uint32_t>(*gpu->device, at, "words.list");
        REQUIRE(list);
        const gpu::Buffer listed = floatsOnDevice(*gpu, values, "words.listed");
        colour::ColourWordsRun run;
        run.count = colours;
        run.list = &*list;
        CHECK(convert(space, run, listed));
        const auto l = checkWords(*gpu, *function, listed, original, static_cast<uint32_t>(values.size()), stride, 1,
                                  false);
        std::printf("  %s: strided %u colour words off, %u others; halves %u, %u; listed %u, %u (%u colours each)\n",
                    space, s[0], s[1], h[0], h[1], l[0], l[1], s[2]);
        CHECK(s[2] == colours);
        CHECK(h[2] == colours);
        CHECK(s[0] == 0);
        CHECK(s[1] == 0);
        CHECK(h[0] == 0);
        CHECK(h[1] == 0);
        CHECK(l[0] == 0);
        CHECK(l[1] == 0);
    }
}
