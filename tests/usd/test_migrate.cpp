// Copyright (c) 2026 jesus luque.
//
// lucabRTrender's files, migrated: a stage written with the Lrt* schemas and
// `lrt:` properties, a .lrtc, and a package holding both, each drawn after
// `usd::migrate` exactly as the same thing authored under athenea's names.
#include "../gpu/GpuTest.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <pxr/base/plug/registry.h>
#include <pxr/usd/sdf/zipFile.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usd/stage.h>

#include "athenea/lod/Athc.h"
#include "athenea/lod/Lod.h"
#include "athenea/render/ReferenceRenderer.h"
#include "athenea/render/TileRasterizer.h"
#include "athenea/scene/GpuClouds.h"
#include "athenea/usd/Export.h"
#include "athenea/usd/Migrate.h"
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
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "migrate";
    fs::create_directories((dir / name).parent_path());
    return dir / name;
}

io::RawSplats cloud(uint32_t count) {
    io::RawSplats raw;
    raw.source = "migrate synthetic";
    io::SplatEncoding& e = raw.encoding;
    e.floatsPerRecord = 23;
    e.x = 0; e.y = 1; e.z = 2; e.opacity = 3; e.scale0 = 4; e.scale1 = 5; e.scale2 = 6;
    e.rotW = 7; e.rotX = 8; e.rotY = 9; e.rotZ = 10; e.dc0 = 11; e.dc1 = 12; e.dc2 = 13;
    e.restBase = 14; e.restPerColour = 3; e.restColourOuter = 1;
    uint64_t state = 7;
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

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

/// `text` with every occurrence of `from` replaced: the same stage authored
/// under the other names.
std::string replaced(std::string text, const std::string& from, const std::string& to) {
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
    return text;
}
std::string underAthenea(const std::string& lrt) {
    return replaced(replaced(replaced(replaced(lrt, "Lrt", "Athenea"), ":lrt:", ":athenea:"), "\"lrt:", "\"athenea:"),
                    " lrt:", " athenea:");
}

gpu::Buffer drawn(gpu::Device& device, const fs::path& stage, uint32_t w, uint32_t h, const std::string& camera) {
    auto renderer = usd::StageRenderer::open(stage);
    if (!renderer) FAIL(renderer.error().toString());
    auto image = (*renderer)->render(camera, 0.0, w, h);
    if (!image) FAIL(image.error().toString());
    gpu::BufferDesc desc;
    desc.bytes = image->rgba.size() * sizeof(float);
    desc.elementBytes = 16;
    auto buffer = gpu::Buffer::create(device, desc, image->rgba.data());
    REQUIRE(buffer);
    return std::move(*buffer);
}

const char* kCamera =
    "    def Camera \"Shot\"\n    {\n"
    "        float2 clippingRange = (0.1, 1000)\n        float focalLength = 30\n"
    "        float horizontalAperture = 24.576\n        float verticalAperture = 18.432\n"
    "        double3 xformOp:translate = (0.4, 0.2, 7)\n"
    "        uniform token[] xformOpOrder = [\"xformOp:translate\"]\n    }\n";

}   // namespace

TEST_CASE("a stage written with lucabRTrender's names draws, migrated, as the same stage authored under athenea's",
          "[usd][gpu][migrate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    const io::RawSplats raw = cloud(2500);
    const fs::path cloudPath = scratch("src/edit-cloud.usdc");
    fs::remove(cloudPath);
    REQUIRE(usd::writeParticleFieldStage(*gpu->library, raw, cloudPath, {.addCamera = false}));

    // Every kind of name the rename touched: an applied schema, constant
    // primvars (one time sampled, one with metadata), a connection and a
    // relationship naming a renamed property, a render setting, the renderer
    // a pass names, and customData keys.
    const std::string old =
        "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n"
        "    customLayerData = { string \"lrt:note\" = \"kept\" }\n)\n"
        "def Xform \"World\"\n{\n"
        "    def Xform \"Group\" (\n        prepend apiSchemas = [\"LrtSplatEditAPI\"]\n"
        "        customData = { int \"lrt:version\" = 1 }\n    )\n    {\n"
        "        bool primvars:lrt:edit:active = 1 ( interpolation = \"constant\" )\n"
        "        token primvars:lrt:edit:shape = \"sphere\" ( interpolation = \"constant\" )\n"
        "        token primvars:lrt:edit:mode = \"grade\" ( interpolation = \"constant\" )\n"
        "        float3 primvars:lrt:edit:centre = (0.5, 0, 0) ( interpolation = \"constant\" )\n"
        "        float3 primvars:lrt:edit:size = (1.5, 0, 0) ( interpolation = \"constant\" )\n"
        "        color3f primvars:lrt:edit:tint = (1, 0.3, 0.1) (\n"
        "            interpolation = \"constant\"\n            doc = \"the grade's tint\"\n"
        "            customData = { string \"lrt:origin\" = \"openFXplayer\" }\n        )\n"
        "        float primvars:lrt:edit:saturation.timeSamples = { 0: 0.5, 10: 0.9 }\n"
        "        float primvars:lrt:edit:opacity = 0.6 ( interpolation = \"constant\" )\n"
        "        bool primvars:lrt:edit:invert = 1 ( interpolation = \"constant\" )\n"
        "        custom float driver\n"
        "        custom float driver.connect = </World/Group.primvars:lrt:edit:opacity>\n"
        "        rel lrt:edit:follows = </World/Group.primvars:lrt:edit:tint>\n"
        "        def \"Cloud\" ( references = @./edit-cloud.usdc@</World/Splats> )\n        {\n"
        "            double3 xformOp:rotateXYZ = (0, 35, 0)\n"
        "            uniform token[] xformOpOrder = [\"xformOp:rotateXYZ\"]\n        }\n    }\n" +
        std::string(kCamera) +
        "    def Scope \"Render\"\n    {\n"
        "        def RenderSettings \"Settings\"\n        {\n"
        "            int lrt:lightSamples = 1\n        }\n"
        "        def Scope \"Pass\"\n        {\n"
        "            uniform token hydra:rendererName = \"HdLrtRendererPlugin\"\n        }\n    }\n}\n";
    const fs::path oldPath = scratch("src/shot.usda");
    const fs::path nativePath = scratch("src/native.usda");
    writeText(oldPath, old);
    writeText(nativePath, underAthenea(old));

    // Migrated beside a different directory: the cloud it references is not
    // copied, so its relative path is anchored to the original.
    const fs::path migrated = scratch("out/shot.usda");
    auto report = usd::migrate(oldPath, migrated);
    if (!report) FAIL(report.error().toString());
    for (const usd::MigrateChange& c : report->changes) {
        std::printf("  %-9s %s %s -> %s\n", usd::toString(c.kind), c.where.c_str(), c.from.c_str(), c.to.c_str());
    }
    using Kind = usd::MigrateChange::Kind;
    CHECK(report->count(Kind::Schema) == 1);
    CHECK(report->count(Kind::Property) == 11);   // nine primvars, the relationship, the setting
    CHECK(report->count(Kind::Target) == 2);
    CHECK(report->count(Kind::Value) == 1);
    CHECK(report->count(Kind::Metadata) == 3);
    CHECK(report->count(Kind::Anchored) == 1);
    CHECK(report->count(Kind::Warning) == 0);

    {
        UsdStageRefPtr stage = UsdStage::Open(migrated.string());
        REQUIRE(stage);
        const UsdPrim group = stage->GetPrimAtPath(SdfPath("/World/Group"));
        CHECK(group.HasAPI(TfToken("AtheneaSplatEditAPI")));
        CHECK_FALSE(group.HasProperty(TfToken("primvars:lrt:edit:tint")));
        GfVec3f tint;
        CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:tint")).Get(&tint));
        CHECK(tint == GfVec3f(1.0F, 0.3F, 0.1F));
        CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:tint")).GetDocumentation() == "the grade's tint");
        // Flat keys holding a colon (a ByKey lookup would read the colon as nesting).
        const VtDictionary tintData = group.GetAttribute(TfToken("primvars:athenea:edit:tint")).GetCustomData();
        CHECK(tintData.count("athenea:origin") == 1);
        CHECK(tintData.count("lrt:origin") == 0);
        CHECK(group.GetCustomData().count("athenea:version") == 1);
        CHECK(group.GetAttribute(TfToken("primvars:athenea:edit:saturation")).GetNumTimeSamples() == 2);
        SdfPathVector sources;
        group.GetAttribute(TfToken("driver")).GetConnections(&sources);
        CHECK(sources == SdfPathVector{SdfPath("/World/Group.primvars:athenea:edit:opacity")});
        SdfPathVector targets;
        group.GetRelationship(TfToken("athenea:edit:follows")).GetTargets(&targets);
        CHECK(targets == SdfPathVector{SdfPath("/World/Group.primvars:athenea:edit:tint")});
        int samples = 0;
        CHECK(stage->GetPrimAtPath(SdfPath("/World/Render/Settings"))
                  .GetAttribute(TfToken("athenea:lightSamples"))
                  .Get(&samples));
        CHECK(samples == 1);
        TfToken renderer;
        stage->GetPrimAtPath(SdfPath("/World/Render/Pass")).GetAttribute(TfToken("hydra:rendererName")).Get(&renderer);
        CHECK(renderer == TfToken("HdAtheneaRendererPlugin"));
        CHECK(stage->GetRootLayer()->GetCustomLayerData().count("athenea:note") == 1);
    }

    const uint32_t w = 240, h = 180;
    gpu::Buffer migratedFrame = drawn(*gpu->device, migrated, w, h, "/World/Shot");
    gpu::Buffer nativeFrame = drawn(*gpu->device, nativePath, w, h, "/World/Shot");
    gpu::Buffer oldFrame = drawn(*gpu->device, oldPath, w, h, "/World/Shot");
    auto same = render::compareImages(*gpu->library, migratedFrame, nativeFrame, w, h);
    REQUIRE(same);
    CHECK(same->max == 0);
    CHECK(same->over2 == 0);
    // Unmigrated, the edit is not read at all: the test sees it arrive.
    auto unread = render::compareImages(*gpu->library, oldFrame, nativeFrame, w, h);
    REQUIRE(unread);
    CHECK(unread->over2 > unread->pixels / 20);

    // Again over the copy: nothing left to rename, and it draws the same.
    const fs::path twice = scratch("out/shot-again.usda");
    auto again = usd::migrate(migrated, twice);
    if (!again) FAIL(again.error().toString());
    CHECK(again->renames() == 0);
    CHECK(again->count(Kind::Warning) == 0);
    gpu::Buffer twiceFrame = drawn(*gpu->device, twice, w, h, "/World/Shot");
    auto stable = render::compareImages(*gpu->library, twiceFrame, nativeFrame, w, h);
    REQUIRE(stable);
    CHECK(stable->max == 0);

    // The old stage and its cloud in a package: migrated as a package, and
    // drawn as the native stage.
    const fs::path oldPackage = scratch("src/shot.usdz");
    fs::remove(oldPackage);
    {
        SdfZipFileWriter writer = SdfZipFileWriter::CreateNew(oldPackage.string());
        REQUIRE(writer);
        CHECK_FALSE(writer.AddFile(oldPath.string(), "shot.usda").empty());
        CHECK_FALSE(writer.AddFile(cloudPath.string(), "edit-cloud.usdc").empty());
        REQUIRE(writer.Save());
    }
    const fs::path newPackage = scratch("out/shot.usdz");
    auto packaged = usd::migrate(oldPackage, newPackage);
    if (!packaged) FAIL(packaged.error().toString());
    CHECK(packaged->count(Kind::Schema) == 1);
    CHECK(packaged->count(Kind::Anchored) == 0);   // inside a package everything stays relative
    gpu::Buffer packageFrame = drawn(*gpu->device, newPackage, w, h, "/World/Shot");
    auto inPackage = render::compareImages(*gpu->library, packageFrame, nativeFrame, w, h);
    REQUIRE(inPackage);
    CHECK(inPackage->max == 0);
    CHECK(inPackage->over2 == 0);

    // The input is never written over.
    CHECK_FALSE(usd::migrate(oldPath, oldPath));
    CHECK_FALSE(usd::migrate(oldPath, scratch("src/../src/shot.usda")));
}

TEST_CASE("a .lrtc migrates to a .athc that reads, cuts and streams through USD as the one athenea writes",
          "[usd][gpu][lod][migrate]") {
    ATHENEA_REQUIRE_GPU(gpu);
    auto loader = scene::CloudLoader::create(*gpu->library);
    REQUIRE(loader);
    auto splats = loader->upload(cloud(20000));
    REQUIRE(splats);
    auto builder = lod::LodBuilder::create(*gpu->library);
    REQUIRE(builder);
    lod::LodBuildSettings chunked;
    chunked.chunkSplats = 1000;
    auto built = builder->build(*splats, chunked);
    REQUIRE(built);
    const fs::path native = scratch("lod/native.athc");
    REQUIRE(lod::writeAthc(*gpu->device, *built, native));

    // The file lucabRTrender wrote for this cloud: version 1 of the layout
    // under the magic LRTC, its last header word the padding it zeroed (here
    // the flags word, which must say no normals for the two to agree).
    std::string bytes;
    {
        std::ifstream in(native, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    REQUIRE(bytes.size() > 4096);
    uint32_t flags = 0;
    std::memcpy(&flags, bytes.data() + 76, 4);
    REQUIRE(flags == 0);
    const uint32_t version1 = 1;
    std::memcpy(bytes.data(), "LRTC", 4);
    std::memcpy(bytes.data() + 4, &version1, 4);
    const fs::path oldCloud = scratch("lod/src/cloud.lrtc");
    writeText(oldCloud, bytes);
    CHECK_FALSE(lod::readAthc(*gpu->device, oldCloud));   // unreadable as it is

    const fs::path converted = scratch("lod/converted.athc");
    auto ok = lod::migrateLrtc(oldCloud, converted);
    if (!ok) FAIL(ok.error().toString());
    CHECK(*ok);
    auto copied = lod::migrateLrtc(converted, scratch("lod/copied.athc"));   // an .athc: copied, no conversion
    REQUIRE(copied);
    CHECK_FALSE(*copied);
    CHECK_FALSE(lod::migrateLrtc(oldCloud, oldCloud));

    // Read whole, cut for one view and drawn: the same frame as the native file.
    const uint32_t w = 240, h = 180;
    render::Camera camera;
    camera.cameraToWorld = aofx::xform::translation({0.3, 0.4, 4.5});
    camera.lens.focal = 30.0;
    camera.lens.nearZ = 0.1;
    camera.lens.farZ = 1000.0;
    render::RenderSettings settings;
    settings.width = w;
    settings.height = h;
    const render::Projection projection = render::projectionFor(camera, w, h);
    auto rasterizer = render::TileRasterizer::create(*gpu->library);
    REQUIRE(rasterizer);
    auto cutter = lod::CutSelector::create(*gpu->library);
    REQUIRE(cutter);
    const auto cutAndDraw = [&](const fs::path& file, render::RenderTargets& out) {
        auto read = lod::readAthc(*gpu->device, file);
        if (!read) FAIL(read.error().toString());
        std::vector<lod::CutStats> stats;
        auto selected = cutter->select(projection, std::vector<lod::LodInstance>{{&*read}}, 12.0F, &stats);
        REQUIRE(selected);
        CHECK(stats.front().merged > 0);
        CHECK(stats.front().splats > 0);
        REQUIRE(rasterizer->render(projection, *selected, settings, out));
    };
    render::RenderTargets fromNative, fromConverted;
    cutAndDraw(native, fromNative);
    cutAndDraw(converted, fromConverted);
    auto cut = render::compareImages(*gpu->library, fromConverted.colour, fromNative.colour, w, h);
    REQUIRE(cut);
    CHECK(cut->max == 0);
    CHECK(cut->over2 == 0);

    // Through a stage: an old one naming the .lrtc, migrated with --recursive
    // (the .lrtc converted beside the copy), against the one athenea would
    // write naming its own file; streamed with a budget, so chunks load.
    const std::string old =
        "#usda 1.0\n(\n    defaultPrim = \"World\"\n    upAxis = \"Y\"\n)\n"
        "def Xform \"World\"\n{\n"
        "    def ParticleField3DGaussianSplat \"Cloud\" (\n"
        "        prepend apiSchemas = [\"LrtStreamedAssetAPI\"]\n    )\n    {\n"
        "        asset primvars:lrt:asset = @./cloud.lrtc@ ( interpolation = \"constant\" )\n"
        "        float primvars:lrt:lod:threshold = 12 ( interpolation = \"constant\" )\n"
        "        int64 primvars:lrt:stream:budget = 8000 ( interpolation = \"constant\" )\n"
        "    }\n" +
        std::string(kCamera) + "}\n";
    const fs::path oldStage = scratch("lod/src/shot.usda");
    writeText(oldStage, old);
    writeText(scratch("lod/native.usda"), replaced(underAthenea(old), "./cloud.lrtc", "./native.athc"));
    usd::MigrateOptions recursive;
    recursive.recursive = true;
    auto report = usd::migrate(oldStage, scratch("lod/out/shot.usda"), recursive);
    if (!report) FAIL(report.error().toString());
    CHECK(report->written.size() == 2);
    CHECK(fs::exists(scratch("lod/out/cloud.athc")));
    CHECK(report->count(usd::MigrateChange::Kind::Warning) == 0);
    {
        UsdStageRefPtr stage = UsdStage::Open(scratch("lod/out/shot.usda").string());
        REQUIRE(stage);
        const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Cloud"));
        CHECK(prim.HasAPI(TfToken("AtheneaStreamedAssetAPI")));
        SdfAssetPath asset;
        CHECK(prim.GetAttribute(TfToken("primvars:athenea:asset")).Get(&asset));
        CHECK(asset.GetAssetPath() == "./cloud.athc");
    }
    gpu::Buffer migratedFrame = drawn(*gpu->device, scratch("lod/out/shot.usda"), w, h, "/World/Shot");
    gpu::Buffer nativeFrame = drawn(*gpu->device, scratch("lod/native.usda"), w, h, "/World/Shot");
    auto through = render::compareImages(*gpu->library, migratedFrame, nativeFrame, w, h);
    REQUIRE(through);
    CHECK(through->max == 0);
    CHECK(through->over2 == 0);

    // The same stage and .lrtc in a package: the .lrtc converted and renamed
    // inside it, and the stage naming the new name. (Drawn it is not: the
    // engine maps a .athc as a file, and one inside a package is not one.)
    const fs::path oldPackage = scratch("lod/src/shot.usdz");
    fs::remove(oldPackage);
    {
        SdfZipFileWriter writer = SdfZipFileWriter::CreateNew(oldPackage.string());
        REQUIRE(writer);
        CHECK_FALSE(writer.AddFile(oldStage.string(), "shot.usda").empty());
        CHECK_FALSE(writer.AddFile(oldCloud.string(), "cloud.lrtc").empty());
        REQUIRE(writer.Save());
    }
    const fs::path newPackage = scratch("lod/out/shot.usdz");
    auto packaged = usd::migrate(oldPackage, newPackage);
    if (!packaged) FAIL(packaged.error().toString());
    CHECK(packaged->count(usd::MigrateChange::Kind::Warning) == 0);
    {
        SdfZipFile zip = SdfZipFile::Open(newPackage.string());
        REQUIRE(zip);
        std::vector<std::string> names(zip.begin(), zip.end());
        CHECK(names == std::vector<std::string>{"shot.usda", "cloud.athc"});
        UsdStageRefPtr stage = UsdStage::Open(newPackage.string());
        REQUIRE(stage);
        const UsdPrim prim = stage->GetPrimAtPath(SdfPath("/World/Cloud"));
        CHECK(prim.HasAPI(TfToken("AtheneaStreamedAssetAPI")));
        SdfAssetPath asset;
        CHECK(prim.GetAttribute(TfToken("primvars:athenea:asset")).Get(&asset));
        CHECK(asset.GetAssetPath() == "./cloud.athc");
    }
}
