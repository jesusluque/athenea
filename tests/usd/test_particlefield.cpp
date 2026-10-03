// Copyright (c) 2026 jesus luque.
//
// THE PARTICLEFIELD AS OPENUSD DEFINES IT. What the schema and its reference
// code (usdVol's schema.usda, the PLY and SPZ converters and hdParticleField
// in extras/imaging/examples) say a field holds, read and written as they say
// it: the harmonics' degree where it is not authored, arrays shorter than the
// particles, the space its colours are in (UsdColorSpaceAPI), and a material
// bound to it as the place a gaussian's surface is said.
//
// The cases without [gpu] check bookkeeping only -- which arrays are handed
// on, what a name means, what a network says -- and read no data back. The
// [gpu] ones compare frames on the device (compareImages).
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>

#include <pxr/base/gf/colorSpace.h>
#include <pxr/base/plug/registry.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/usd/colorSpaceAPI.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdVol/particleField3DGaussianSplat.h>

#include "athenea/io/RawSplats.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/PrimData.h"
#include "athenea/usd/StageRenderer.h"

using namespace athenea;
namespace fs = std::filesystem;
PXR_NAMESPACE_USING_DIRECTIVE

namespace {

[[maybe_unused]] const bool kPluginsRegistered = [] {
    PlugRegistry::GetInstance().RegisterPlugins(fs::path(ATHENEA_HYDRA_PLUGIN_DIR).string());
    return true;
}();

fs::path scratch(const std::string& name) {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "particlefield";
    fs::create_directories(dir);
    return dir / name;
}

/// `count` gaussians of degree 1, as test_usd's `cloud` makes them.
io::RawSplats cloud(uint32_t count) {
    io::RawSplats raw;
    raw.source = "particlefield synthetic";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 23;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
    uint64_t state = 1717;
    const auto next = [&] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 40) & 0xFFFFFF) / 16777216.0F;
    };
    for (uint32_t i = 0; i < count; ++i) {
        raw.records.resize(raw.records.size() + 23, 0.0F);
        float* r = raw.records.data() + raw.records.size() - 23;
        r[0] = next() * 4 - 2; r[1] = next() * 3 - 1.5F; r[2] = next() * 4 - 2;
        r[3] = next() * 6 - 3;
        r[4] = std::log(0.02F + next() * 0.2F); r[5] = std::log(0.02F + next() * 0.2F);
        r[6] = std::log(0.01F + next() * 0.1F);
        r[7] = next() - 0.5F; r[8] = next() - 0.5F; r[9] = next() - 0.5F; r[10] = next() - 0.5F;
        for (int k = 11; k < 23; ++k) {
            r[k] = next() * 2 - 1;
        }
        raw.count += 1;
    }
    return raw;
}

/// A shot over the field at `cloudFile`'s /World/Splats.
fs::path shotOf(const std::string& name, const std::string& cloudFile) {
    const fs::path shot = scratch(name);
    std::ofstream out(shot);
    out << "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
           "def Xform \"World\"\n{\n"
           "    def \"Cloud\" ( references = @./" << cloudFile << "@</World/Splats> )\n    {\n    }\n"
           "    def Camera \"Shot\"\n    {\n"
           "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
           "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
           "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
           "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n}\n";
    return shot;
}

/// The shot's frame, on the device.
gpu::Buffer frameOf(test::Gpu* gpu, const fs::path& shot, uint32_t w, uint32_t h) {
    auto renderer = usd::StageRenderer::open(shot);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render("/World/Shot", 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto made = gpu::Buffer::create(*gpu->device, desc, image->rgba.data());
    if (!made) FAIL(made.error().toString());
    return std::move(*made);
}

/// A copy of the exported stage at `from`, edited by `edit` and saved at `to`.
void copyEdited(const fs::path& from, const fs::path& to, const std::function<void(UsdPrim)>& edit) {
    fs::remove(to);
    SdfLayerRefPtr layer = SdfLayer::FindOrOpen(from.string());
    REQUIRE(layer);
    REQUIRE(layer->Export(to.string()));
    UsdStageRefPtr stage = UsdStage::Open(to.string());
    REQUIRE(stage);
    edit(stage->GetPrimAtPath(SdfPath("/World/Splats")));
    REQUIRE(stage->GetRootLayer()->Save());
}

usd::ParticleFieldArrays fieldOf(size_t n, size_t coefficientsPerParticle) {
    usd::ParticleFieldArrays a;
    a.positions = VtValue(VtVec3fArray(n, GfVec3f(0.0F)));
    a.shCoefficients = VtValue(VtVec3fArray(n * coefficientsPerParticle, GfVec3f(0.0F)));
    return a;
}

}   // namespace

TEST_CASE("a colour space's name says whether a cloud's colours are light", "[usd][particlefield]") {
    CHECK(usd::cloudColoursLinear("lin_rec709_scene") == std::optional<bool>(true));
    CHECK(usd::cloudColoursLinear(GfColorSpaceNames->LinearRec709.GetString()) == std::optional<bool>(true));
    CHECK(usd::cloudColoursLinear("lin_rec709_display") == std::optional<bool>(true));
    CHECK(usd::cloudColoursLinear("srgb_rec709_scene") == std::optional<bool>(false));
    CHECK(usd::cloudColoursLinear(GfColorSpaceNames->SRGBRec709.GetString()) == std::optional<bool>(false));
    CHECK(usd::cloudColoursLinear("sRGB") == std::optional<bool>(false));
    CHECK_FALSE(usd::cloudColoursLinear("lin_ap1_scene").has_value());
    CHECK_FALSE(usd::cloudColoursLinear("g22_rec709_scene").has_value());
    CHECK_FALSE(usd::cloudColoursLinear("").has_value());
}

TEST_CASE("a ParticleField's colours are in the space USD resolves for its harmonics, unless athenea's primvar says",
          "[usd][particlefield]") {
    UsdStageRefPtr stage = UsdStage::CreateInMemory();
    UsdGeomXform world = UsdGeomXform::Define(stage, SdfPath("/World"));
    auto field = UsdVolParticleField3DGaussianSplat::Define(stage, SdfPath("/World/Splats"));
    field.CreateRadianceSphericalHarmonicsCoefficientsAttr(VtValue(VtVec3fArray(1, GfVec3f(0.0F))));
    const UsdPrim prim = field.GetPrim();

    // Nothing said: a capture's, which the caller takes for sRGB.
    CHECK_FALSE(usd::fieldColoursLinear(prim).has_value());

    // An ancestor's colorSpace:name stands over everything below it.
    UsdColorSpaceAPI::Apply(world.GetPrim()).CreateColorSpaceNameAttr(VtValue(GfColorSpaceNames->LinearRec709));
    CHECK(usd::fieldColoursLinear(prim) == std::optional<bool>(true));

    // The prim's own stands over the ancestor's.
    UsdColorSpaceAPI::Apply(prim).CreateColorSpaceNameAttr(VtValue(GfColorSpaceNames->SRGBRec709));
    CHECK(usd::fieldColoursLinear(prim) == std::optional<bool>(false));

    // The attribute's own stands over the prim's.
    field.GetRadianceSphericalHarmonicsCoefficientsAttr().SetColorSpace(GfColorSpaceNames->LinearRec709);
    CHECK(usd::fieldColoursLinear(prim) == std::optional<bool>(true));

    // And athenea's primvar, where authored, over all of them.
    prim.CreateAttribute(TfToken("primvars:athenea:splat:linear"), SdfValueTypeNames->Bool, true).Set(false);
    CHECK(usd::fieldColoursLinear(prim) == std::optional<bool>(false));
}

TEST_CASE("an unauthored harmonics degree is the schema's 3, or the whole degree a shorter array holds",
          "[usd][particlefield]") {
    constexpr size_t kN = 10;
    SECTION("sixteen a particle, unauthored: degree 3") {
        const usd::ParticleFieldArrays a = fieldOf(kN, 16);
        const scene::SplatStreams s = usd::splatStreams(a, "unauthored");
        CHECK(s.coefficients == 16);
        CHECK_FALSE(s.sh.empty());
    }
    SECTION("four a particle, unauthored: degree 1, not sixteen out of step") {
        const usd::ParticleFieldArrays a = fieldOf(kN, 4);
        const scene::SplatStreams s = usd::splatStreams(a, "unauthored");
        CHECK(s.coefficients == 4);
        CHECK_FALSE(s.sh.empty());
    }
    SECTION("one a particle, unauthored: degree 0") {
        const usd::ParticleFieldArrays a = fieldOf(kN, 1);
        CHECK(usd::splatStreams(a, "unauthored").coefficients == 1);
    }
    SECTION("authored 0 over sixteen a particle: the first of each is not the DC, the array is truncated") {
        usd::ParticleFieldArrays a = fieldOf(kN, 16);
        a.shDegree = 0;
        const scene::SplatStreams s = usd::splatStreams(a, "authored");
        CHECK(s.coefficients == 1);
        CHECK_FALSE(s.sh.empty());
    }
    SECTION("authored 3 over four a particle: too short, ignored, the schema's grey") {
        usd::ParticleFieldArrays a = fieldOf(kN, 4);
        a.shDegree = 3;
        const scene::SplatStreams s = usd::splatStreams(a, "authored");
        CHECK(s.coefficients == 16);
        CHECK(s.sh.empty());
    }
    SECTION("fewer coefficients than particles, unauthored: none") {
        usd::ParticleFieldArrays a = fieldOf(kN, 1);
        a.shCoefficients = VtValue(VtVec3fArray(kN - 1, GfVec3f(0.0F)));
        CHECK(usd::splatStreams(a, "short").sh.empty());
    }
}

TEST_CASE("an orientation, scale or opacity array shorter than the particles is ignored, not the cloud",
          "[usd][particlefield]") {
    constexpr size_t kN = 10;
    usd::ParticleFieldArrays a = fieldOf(kN, 1);
    a.orientations = VtValue(VtQuatfArray(kN - 1, GfQuatf(1.0F)));
    a.scales = VtValue(VtVec3fArray(kN - 2, GfVec3f(1.0F)));
    a.opacities = VtValue(VtFloatArray(kN - 3, 1.0F));
    const scene::SplatStreams s = usd::splatStreams(a, "short");
    CHECK(s.count == kN);
    CHECK(s.rotations.empty());
    CHECK(s.scales.empty());
    CHECK(s.opacities.empty());
    // Long enough, they stay: a longer array is truncated, not dropped.
    a.orientations = VtValue(VtQuatfArray(kN + 1, GfQuatf(1.0F)));
    a.scales = VtValue(VtVec3fArray(kN, GfVec3f(1.0F)));
    a.opacities = VtValue(VtFloatArray(kN, 1.0F));
    const scene::SplatStreams t = usd::splatStreams(a, "whole");
    CHECK_FALSE(t.rotations.empty());
    CHECK_FALSE(t.scales.empty());
    CHECK_FALSE(t.opacities.empty());
}

TEST_CASE("an exported cloud says its colour space in USD's own words", "[usd][gpu][particlefield]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(200);
    for (const bool linear : {false, true}) {
        const fs::path path = scratch(linear ? "space-linear.usda" : "space-srgb.usda");
        fs::remove(path);
        usd::ExportOptions options;
        options.addCamera = false;
        options.linear = linear;
        REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, path, options));
        UsdStageRefPtr stage = UsdStage::Open(path.string());
        REQUIRE(stage);
        UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Splats"));
        REQUIRE(prim.HasAPI<UsdColorSpaceAPI>());
        const UsdAttribute harmonics =
            prim.GetAttribute(UsdVolTokens->radianceSphericalHarmonicsCoefficients);
        CHECK(UsdColorSpaceAPI::ComputeColorSpaceName(harmonics) ==
              (linear ? GfColorSpaceNames->LinearRec709 : GfColorSpaceNames->SRGBRec709));
        // Read back by the field's own declaration alone, the primvar gone.
        prim.RemoveProperty(TfToken("primvars:athenea:splat:linear"));
        CHECK(usd::fieldColoursLinear(prim) == std::optional<bool>(linear));
    }
}

// THE STANDARD'S DECLARATIONS DRAW AS ATHENEA'S OWN. The same exported cloud
// three ways: its degree unauthored (the schema's fallback, which the array
// it holds bears out), its colour space said only by UsdColorSpaceAPI, and
// as written. Each draws the frame the written one does; a control said to
// be in the other space does not.
TEST_CASE("a field that leaves its degree unauthored, or says its space only in UsdColorSpaceAPI, draws as written",
          "[usd][gpu][particlefield]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path written = scratch("as-written.usdc");
    fs::remove(written);
    usd::ExportOptions options;
    options.addCamera = false;
    options.linear = true;
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, written, options));
    copyEdited(written, scratch("degree-unauthored.usdc"), [](UsdPrim prim) {
        REQUIRE(prim.GetAttribute(UsdVolTokens->radianceSphericalHarmonicsDegree).Clear());
    });
    copyEdited(written, scratch("space-by-api.usdc"), [](UsdPrim prim) {
        REQUIRE(prim.RemoveProperty(TfToken("primvars:athenea:splat:linear")));
    });
    copyEdited(written, scratch("space-other.usdc"), [](UsdPrim prim) {
        REQUIRE(prim.RemoveProperty(TfToken("primvars:athenea:splat:linear")));
        UsdColorSpaceAPI(prim).GetColorSpaceNameAttr().Set(GfColorSpaceNames->SRGBRec709);
    });
    constexpr uint32_t w = 240;
    constexpr uint32_t h = 180;
    const gpu::Buffer base = frameOf(gpu, shotOf("as-written-shot.usda", "as-written.usdc"), w, h);
    const gpu::Buffer degree = frameOf(gpu, shotOf("degree-unauthored-shot.usda", "degree-unauthored.usdc"), w, h);
    const gpu::Buffer space = frameOf(gpu, shotOf("space-by-api-shot.usda", "space-by-api.usdc"), w, h);
    const gpu::Buffer other = frameOf(gpu, shotOf("space-other-shot.usda", "space-other.usdc"), w, h);
    auto byDegree = render::compareImages(*gpu->library, base, degree, w, h);
    auto bySpace = render::compareImages(*gpu->library, base, space, w, h);
    auto control = render::compareImages(*gpu->library, base, other, w, h);
    REQUIRE(byDegree);
    REQUIRE(bySpace);
    REQUIRE(control);
    std::printf("  degree unauthored: max %u; space by UsdColorSpaceAPI: max %u; said sRGB: p99 %u, max %u\n",
                byDegree->max, bySpace->max, control->p99, control->max);
    CHECK(byDegree->max == 0);
    CHECK(bySpace->max == 0);
    CHECK(control->max > 2);
}
