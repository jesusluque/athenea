// Copyright (c) 2026 jesus luque.
//
// The panels' description: bookkeeping, so checked on the CPU. What rows a
// front end is given, and what they say, with gaussians and without.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "athenea/ui/GaussianPanel.h"
#include "athenea/ui/ViewerPanels.h"

using namespace athenea;

namespace {

/// The control `id` of `panel`, or null.
const ui::Control* find(const ui::Panel& panel, const std::string& id) {
    for (const ui::Section& section : panel.sections) {
        for (const ui::Control& control : section.controls) {
            if (control.id == id) {
                return &control;
            }
        }
    }
    return nullptr;
}

const ui::Section* section(const ui::Panel& panel, const std::string& id) {
    for (const ui::Section& s : panel.sections) {
        if (s.id == id) {
            return &s;
        }
    }
    return nullptr;
}

size_t shownRows(const ui::Panel& panel) {
    size_t n = 0;
    for (const ui::Section& s : panel.sections) {
        for (const ui::Control& c : s.controls) {
            n += c.isShown() ? 1 : 0;
        }
    }
    return n;
}

ui::GaussianReport twoClouds() {
    ui::GaussianReport g;
    g.frame = 812;
    g.route = "raster";
    g.inStage = 1500000;
    g.submitted = 1234567;
    g.counted = true;
    g.countedFrame = 811;
    g.countedSlots = 1234567;
    g.visible = 812345;
    g.pairs = 2000000;
    g.maxTiles = 40;
    g.culled[2] = 400000;   // outside near/far
    g.culled[4] = 22222;    // too faint
    g.cloudBytes = 152000000;
    ui::GaussianCloud bird;
    bird.prim = "/World/Sparrow";
    bird.gaussians = 1234567;
    bird.drawn = true;
    bird.submitted = 1234567;
    bird.shDegree = 3;
    bird.relit = true;
    bird.normals = true;
    bird.pbr = true;
    bird.ior = 1.5F;
    bird.bytes = 150000000;
    bird.counted = true;
    bird.visible = 812345;
    bird.pairs = 2000000;
    ui::GaussianCloud hidden;
    hidden.prim = "/World/Hidden";
    hidden.gaussians = 265433;
    hidden.linear = true;
    hidden.bytes = 2000000;
    g.clouds = {bird, hidden};
    return g;
}

}   // namespace

TEST_CASE("numbers are written as a person reads them", "[ui]") {
    CHECK(ui::thousands(0) == "0");
    CHECK(ui::thousands(999) == "999");
    CHECK(ui::thousands(1000) == "1,000");
    CHECK(ui::thousands(1234567) == "1,234,567");
    CHECK(ui::thousands(12345678901ull) == "12,345,678,901");
    CHECK(ui::byteSize(512) == "512 B");
    CHECK(ui::byteSize(2048) == "2 KB");
    CHECK(ui::byteSize(152000000) == "145.0 MB");
    CHECK(ui::byteSize(3ull << 30) == "3.00 GB");
}

TEST_CASE("the Gaussians panel says so when the stage has none", "[ui]") {
    ui::GaussianReport report;
    bool timeStages = false;
    const ui::Panel panel = ui::gaussianPanel([&]() -> const ui::GaussianReport& { return report; }, timeStages);
    CHECK(panel.id == "gaussians");
    const ui::Control* none = find(panel, "gaussians.none");
    REQUIRE(none != nullptr);
    CHECK(none->isShown());
    CHECK(none->reading() == "no gaussians in this stage");
    // Nothing else: no counts, no times, no clouds.
    CHECK(shownRows(panel) == 1);
    for (const char* id : {"gaussians.counts", "gaussians.stages", "gaussians.memory", "gaussians.clouds"}) {
        const ui::Section* s = section(panel, id);
        REQUIRE(s != nullptr);
        CHECK_FALSE(s->isShown());
    }
}

TEST_CASE("the Gaussians panel reads its report every time a row is drawn", "[ui]") {
    ui::GaussianReport report;
    bool timeStages = false;
    const ui::Panel panel = ui::gaussianPanel([&]() -> const ui::GaussianReport& { return report; }, timeStages);
    // Built over an empty stage, then the stage gains clouds: the rows follow.
    report = twoClouds();
    CHECK_FALSE(find(panel, "gaussians.none")->isShown());
    CHECK(find(panel, "gaussians.frame")->reading() == "frame 812, rasterised");
    CHECK(find(panel, "gaussians.counted")->reading() == "frame 811, 1 behind");
    CHECK(find(panel, "gaussians.stage")->reading() == "1,500,000");
    CHECK(find(panel, "gaussians.visible")->reading() == "812,345 (65.8%)");
    CHECK(find(panel, "gaussians.pairs")->reading() == "2,000,000, 2.46 a visible gaussian");
    // A reason is a row only where it culled something.
    CHECK(find(panel, "cull.depth")->isShown());
    CHECK(find(panel, "cull.faint")->isShown());
    CHECK_FALSE(find(panel, "cull.offscreen")->isShown());
    // One row a cloud, and no more.
    CHECK(find(panel, "gaussians.cloud.0")->isShown());
    CHECK(find(panel, "gaussians.cloud.1")->isShown());
    CHECK_FALSE(find(panel, "gaussians.cloud.2")->isShown());
    const std::string bird = find(panel, "gaussians.cloud.0")->reading();
    CHECK(bird.find("/World/Sparrow") != std::string::npos);
    CHECK(bird.find("812,345 visible") != std::string::npos);
    CHECK(bird.find("SH degree 3, capture sRGB, relit, normals, PBR, ior 1.50") != std::string::npos);
    const std::string hidden = find(panel, "gaussians.cloud.1")->reading();
    CHECK(hidden.find("not drawn this frame") != std::string::npos);
    CHECK(hidden.find("linear") != std::string::npos);
    // Stage times are there only when they were measured, and the switch
    // that asks for them writes the front end's flag.
    CHECK_FALSE(find(panel, "gaussians.ms.Project")->isShown());
    find(panel, "gaussians.timeStages")->setFlag(true);
    CHECK(timeStages);
    report.stagesTimed = true;
    report.projectMs = 1.25;
    CHECK(find(panel, "gaussians.ms.Project")->reading() == "1.25 ms");
    // The traced route counts nothing it could cull, and says what it built.
    report.route = "rt";
    report.traced = true;
    report.rebuilt = true;
    report.traceRoute = "hardware";
    CHECK_FALSE(find(panel, "gaussians.visible")->isShown());
    CHECK_FALSE(find(panel, "gaussians.timeStages")->isShown());
    CHECK(find(panel, "gaussians.structures")->reading().rfind("rebuilt, hardware", 0) == 0);
}

TEST_CASE("the viewer's panels gain a Gaussians panel where a front end has the numbers", "[ui]") {
    ui::ViewerSettings settings;
    CHECK(ui::viewerPanels(settings, {}, {}).size() == 1);
    ui::GaussianReport report = twoClouds();
    ui::ViewerFacts facts;
    facts.gaussians = [&]() -> const ui::GaussianReport& { return report; };
    const std::vector<ui::Panel> panels = ui::viewerPanels(settings, facts, {});
    REQUIRE(panels.size() == 2);
    CHECK(panels[1].id == "gaussians");
    find(panels[1], "gaussians.timeStages")->setFlag(true);
    CHECK(settings.timeSplatStages);
}
