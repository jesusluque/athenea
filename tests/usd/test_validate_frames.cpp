// Copyright (c) 2026 jesus luque.
//
// What a measured material's frame leaves out (apps/athenea/src/ValidateFrames.h),
// on a stage of two meshes bound to two materials, read through the same
// bindings reader the measure uses. Narrowing the measure to one material
// narrows what is measured, never what the frame switches off: a cloud that
// stands for the whole stage (`athenea flatten --validate`) draws without
// any of the stage's meshes, and a conversion of one material leaves the
// other's meshes out of it. Prim paths only: bookkeeping, no data.
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ValidateFrames.h"
#include "athenea/usd/MeshStage.h"

using namespace athenea;
namespace fs = std::filesystem;

namespace {

fs::path twoMaterialStage() {
    const fs::path dir = fs::temp_directory_path() / "athenea-tests" / "validate";
    fs::create_directories(dir);
    const fs::path path = dir / "two_materials.usda";
    std::ofstream(path) << R"(#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Wall" {}
        def Material "Panel" {}
    }
    def Mesh "wall" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, 0, 0), (1, 0, 0), (1, 0, 1), (-1, 0, 1)]
        rel material:binding = </World/Looks/Wall>
    }
    def Mesh "panel" (prepend apiSchemas = ["MaterialBindingAPI"])
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-0.5, -0.1, 0.2), (0.5, -0.1, 0.2), (0.5, -0.1, 0.8), (-0.5, -0.1, 0.8)]
        rel material:binding = </World/Looks/Panel>
    }
}
)";
    return path;
}

}   // namespace

TEST_CASE("a measure narrowed to one material still frames the whole stage", "[usd][validate]") {
    auto groups = usd::stageMaterialGroups(twoMaterialStage(), "", {}, 0.0);
    REQUIRE(groups);
    REQUIRE(groups->size() == 2);
    const std::vector<std::string> both{"/World/panel", "/World/wall"};
    const std::vector<std::string> narrowed{"Wall"};

    SECTION("one cloud for the whole stage switches off every mesh, measured or not") {
        const auto frames = cli::validateFrames(*groups, narrowed, {}, true);
        REQUIRE(frames.size() == 1);
        CHECK(frames[0].group.material == "/World/Looks/Wall");
        CHECK(frames[0].off == both);
    }
    SECTION("one material converted leaves the other's meshes out and in the frame") {
        const auto frames = cli::validateFrames(*groups, narrowed, {}, false);
        REQUIRE(frames.size() == 1);
        CHECK(frames[0].hidden == std::vector<std::string>{"/World/panel"});
        CHECK(frames[0].off == std::vector<std::string>{"/World/wall"});
    }
    SECTION("unnarrowed, every material is measured, each against the whole stage") {
        const auto frames = cli::validateFrames(*groups, {}, {}, true);
        REQUIRE(frames.size() == 2);
        for (const cli::ValidateFrame& frame : frames) {
            CHECK(frame.off == both);
        }
    }
}
