// Copyright (c) 2026 jesus luque.
#include "athenea/ui/GaussianPanel.h"

#include <cstdio>
#include <utility>
#include <vector>

namespace athenea::ui {
namespace {

std::string fixed(double value, int decimals) {
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

/// "812,345 (65.8%)": a count and its share of `of`.
std::string share(uint64_t value, uint64_t of) {
    std::string out = thousands(value);
    if (of > 0) {
        out += " (" + fixed(100.0 * double(value) / double(of), 1) + "%)";
    }
    return out;
}

Control reading(std::string id, std::string label, std::function<std::string()> what,
                std::function<bool()> shown = {}, std::string note = {}) {
    Control control;
    control.kind = Control::Kind::Reading;
    control.id = std::move(id);
    control.label = std::move(label);
    control.reading = std::move(what);
    control.shown = std::move(shown);
    control.note = std::move(note);
    return control;
}

/// Why a splat was culled, in render::SplatCounters::Cull's order.
struct Reason {
    const char* id;
    const char* label;
    const char* note;
};
constexpr Reason kReasons[7] = {
    {"cull.unprojected", "Not projected", "slots no splat projection wrote"},
    {"cull.edit", "Removed by an edit", "the prim's SplatEdit took them away"},
    {"cull.depth", "Outside near/far", "behind the eye, nearer than the near plane, or past the far one"},
    {"cull.degenerate", "No area", "a footprint the covariance gives no area"},
    {"cull.faint", "Too faint", "under 1/255 of opacity once spread over their footprint"},
    {"cull.offscreen", "Off screen", "their footprint lies outside the frame: the frustum's sides"},
    {"cull.notile", "Touch no tile", "in the frame, but too small or too thin for any tile to meet"},
};

}   // namespace

std::string thousands(uint64_t value) {
    std::string digits = std::to_string(value);
    std::string out;
    out.reserve(digits.size() + digits.size() / 3);
    const size_t lead = digits.size() % 3;
    for (size_t k = 0; k < digits.size(); ++k) {
        if (k != 0 && (k + 3 - lead) % 3 == 0) {
            out.push_back(',');
        }
        out.push_back(digits[k]);
    }
    return out;
}

std::string byteSize(uint64_t bytes) {
    if (bytes < 1024) {
        return std::to_string(bytes) + " B";
    }
    if (bytes < 1024ull * 1024) {
        return fixed(double(bytes) / 1024.0, 0) + " KB";
    }
    if (bytes < 1024ull * 1024 * 1024) {
        return fixed(double(bytes) / (1024.0 * 1024.0), 1) + " MB";
    }
    return fixed(double(bytes) / (1024.0 * 1024.0 * 1024.0), 2) + " GB";
}

std::string describeCloud(const GaussianCloud& cloud, const GaussianReport& report) {
    std::string out = cloud.prim;
    // How many, and what became of them.
    out += "\n" + thousands(cloud.gaussians) + " gaussians";
    if (!cloud.drawn) {
        out += ", not drawn this frame";
    } else {
        out += ", " + thousands(cloud.submitted) + " submitted";
        if (cloud.counted) {
            out += ", " + thousands(cloud.visible) + " visible, " + thousands(cloud.pairs) + " pairs (frame " +
                   std::to_string(report.countedFrame) + ")";
        }
    }
    // Its level of detail.
    if (!cloud.lod.empty()) {
        out += "\nLOD: " + cloud.lod;
        if (cloud.lodOwn + cloud.lodMerged > 0) {
            out += ", " + thousands(cloud.lodOwn) + " own + " + thousands(cloud.lodMerged) + " merged";
        }
    }
    if (cloud.streamed) {
        out += "\nStreamed: " + std::to_string(cloud.chunksResident) + " of " + std::to_string(cloud.chunks) +
               " chunks on the device, " + std::to_string(cloud.chunksWanted) + " wanted, " +
               std::to_string(cloud.chunksMissing) + " missing, " + std::to_string(cloud.chunksInFlight) +
               " loading";
    }
    // What it carries.
    std::vector<std::string> carries;
    carries.push_back("SH degree " + std::to_string(cloud.shDegree));
    carries.push_back(cloud.linear ? "linear" : "capture sRGB");
    if (cloud.relit) {
        carries.push_back("relit");
    }
    if (cloud.litBody) {
        carries.push_back("lit body");
    }
    if (cloud.transfer > 0) {
        carries.push_back("transfer " + std::to_string(cloud.transfer));
    }
    if (cloud.skinned) {
        carries.push_back("skinned");
    }
    if (cloud.normals) {
        carries.push_back("normals");
    }
    if (cloud.emission) {
        carries.push_back("emission");
    }
    if (cloud.pbr) {
        carries.push_back("PBR");
    }
    if (cloud.crypto) {
        carries.push_back("ids");
    }
    if (cloud.visibility) {
        carries.push_back("baked visibility");
    }
    if (cloud.ior > 0.0F) {
        carries.push_back("ior " + fixed(double(cloud.ior), 2));
    }
    out += "\n";
    for (size_t k = 0; k < carries.size(); ++k) {
        out += (k == 0 ? "" : ", ") + carries[k];
    }
    out += "\n" + byteSize(cloud.bytes) + " on the device";
    return out;
}

Panel gaussianPanel(GaussianSource report, bool& timeStages) {
    const GaussianSource r = std::move(report);
    const auto any = [r] { return !r().clouds.empty(); };
    const auto none = [r] { return r().clouds.empty(); };
    const auto counted = [r] { return !r().clouds.empty() && r().counted && r().route != "rt"; };
    const auto raster = [r] { return !r().clouds.empty() && r().route != "rt"; };
    const auto traced = [r] { return !r().clouds.empty() && r().traced && r().route == "rt"; };

    Panel panel;
    panel.id = "gaussians";
    panel.title = "Gaussians";

    // WHICH FRAME THESE ARE ABOUT. Two answers, because the counts the device
    // keeps are read without waiting and arrive a frame or two later.
    Section frame;
    frame.id = "gaussians.frame";
    frame.title = "Frame";
    frame.controls.push_back(reading("gaussians.none", "", [] { return std::string("no gaussians in this stage"); },
                                     none));
    frame.controls.push_back(reading(
        "gaussians.frame", "Frame",
        [r] {
            const GaussianReport& g = r();
            const std::string route = g.route == "rt"          ? "splats traced"
                                      : g.route == "rt+raster" ? "meshes traced, splats rasterised"
                                                               : "rasterised";
            return "frame " + std::to_string(g.frame) + ", " + route;
        },
        any));
    frame.controls.push_back(reading(
        "gaussians.counted", "Counted",
        [r] {
            const GaussianReport& g = r();
            if (!g.counted) {
                return std::string("nothing counted yet");
            }
            const uint64_t behind = g.frame >= g.countedFrame ? g.frame - g.countedFrame : 0;
            return "frame " + std::to_string(g.countedFrame) +
                   (behind == 0 ? std::string(", this one") : ", " + std::to_string(behind) + " behind");
        },
        raster, "counted on the device and read without waiting for it"));
    panel.sections.push_back(std::move(frame));

    // HOW MANY, from the stage down to what the blend walked.
    Section counts;
    counts.id = "gaussians.counts";
    counts.title = "How many";
    counts.controls.push_back(
        reading("gaussians.stage", "In the stage", [r] { return thousands(r().inStage); }, any,
                "every cloud's gaussians, drawn or not"));
    counts.controls.push_back(reading(
        "gaussians.submitted", "Submitted", [r] { return share(r().submitted, r().inStage); }, any,
        "handed to the renderer: after the level of detail, the hidden prims and the levels not chosen"));
    counts.controls.push_back(reading(
        "gaussians.visible", "Visible", [r] { return share(r().visible, r().countedSlots); }, counted,
        "kept by the projection: what the depth sort sorts"));
    counts.controls.push_back(reading(
        "gaussians.culled", "Culled",
        [r] {
            const GaussianReport& g = r();
            uint64_t culled = 0;
            for (uint32_t n : g.culled) {
                culled += n;
            }
            return share(culled, g.countedSlots);
        },
        counted));
    for (size_t k = 0; k < 7; ++k) {
        const Reason& reason = kReasons[k];
        counts.controls.push_back(reading(
            reason.id, std::string("  ") + reason.label, [r, k] { return thousands(r().culled[k]); },
            [r, k] { return !r().clouds.empty() && r().counted && r().route != "rt" && r().culled[k] > 0; }, reason.note));
    }
    counts.controls.push_back(reading(
        "gaussians.pairs", "Tile pairs",
        [r] {
            const GaussianReport& g = r();
            std::string out = thousands(g.pairs);
            if (g.visible > 0) {
                out += ", " + fixed(double(g.pairs) / double(g.visible), 2) + " a visible gaussian";
            }
            return out;
        },
        counted, "(tile, gaussian) pairs: what the tile sort sorts and the blend walks"));
    counts.controls.push_back(reading("gaussians.maxTiles", "Most tiles", [r] { return thousands(r().maxTiles); },
                                      counted, "the most tiles one gaussian touched"));
    counts.controls.push_back(reading(
        "gaussians.sorts", "Sort sizes",
        [r] { return "depth " + thousands(r().visible) + " keys, tiles " + thousands(r().pairs) + " keys"; },
        counted));
    counts.controls.push_back(reading(
        "gaussians.traced", "Traced", [r] { return thousands(r().tracedSplats); }, traced,
        "the ray tracer draws every gaussian it is given: it culls nothing to count"));
    panel.sections.push_back(std::move(counts));

    // WHAT EACH STAGE COST. Timing a stage means waiting for it, so the times
    // are only there when asked for, and the switch says what it costs.
    Section stages;
    stages.id = "gaussians.stages";
    stages.title = "GPU stages";
    {
        Control time;
        time.kind = Control::Kind::Toggle;
        time.id = "gaussians.timeStages";
        time.label = "Time each stage";
        time.note = "each stage waits for the device: the frame is slower while this is on";
        bool* flag = &timeStages;
        time.flag = [flag] { return *flag; };
        time.setFlag = [flag](bool set) { *flag = set; };
        time.shown = raster;
        stages.controls.push_back(std::move(time));
    }
    const auto timed = [r] { return !r().clouds.empty() && r().route != "rt" && r().stagesTimed; };
    const std::pair<const char*, double GaussianReport::*> kStages[] = {
        {"Project", &GaussianReport::projectMs},     {"Counts", &GaussianReport::countsMs},
        {"Depth sort", &GaussianReport::depthSortMs}, {"Emit", &GaussianReport::emitMs},
        {"Tile sort", &GaussianReport::tileSortMs},   {"Blend", &GaussianReport::blendMs},
        {"Total", &GaussianReport::totalMs},
    };
    for (const auto& [label, member] : kStages) {
        const auto field = member;
        stages.controls.push_back(reading(std::string("gaussians.ms.") + label, label,
                                          [r, field] { return fixed(r().*field, 2) + " ms"; }, timed));
    }
    stages.controls.push_back(reading(
        "gaussians.structures", "Structures",
        [r] {
            const GaussianReport& g = r();
            std::string out = std::string(g.rebuilt ? "rebuilt" : "kept or refitted") + ", " + g.traceRoute;
            if (g.tracedChunks > 0) {
                out += ", " + thousands(g.tracedChunks) + " chunks";
            }
            return out;
        },
        traced));
    stages.controls.push_back(reading("gaussians.build", "Build", [r] { return fixed(r().buildMs, 2) + " ms"; },
                                      traced, "structures, and colours for this eye"));
    stages.controls.push_back(
        reading("gaussians.trace", "Trace", [r] { return fixed(r().traceMs, 2) + " ms"; }, traced));
    stages.controls.push_back(
        reading("gaussians.tracedTotal", "Total", [r] { return fixed(r().tracedMs, 2) + " ms"; }, traced));
    panel.sections.push_back(std::move(stages));

    Section memory;
    memory.id = "gaussians.memory";
    memory.title = "Device memory";
    memory.controls.push_back(reading("gaussians.cloudBytes", "Clouds", [r] { return byteSize(r().cloudBytes); },
                                      any, "every cloud's arrays, a posed copy and its skeleton included"));
    memory.controls.push_back(reading(
        "gaussians.poolBytes", "Levels of detail", [r] { return byteSize(r().poolBytes); },
        [r] { return r().poolBytes > 0; }, "the assets cuts are taken from, and the streaming stores"));
    panel.sections.push_back(std::move(memory));

    // ONE ROW A CLOUD, as many as the stage has up to the panel's limit:
    // which prim, how many, what it carries, what it holds.
    Section clouds;
    clouds.id = "gaussians.clouds";
    clouds.title = "Clouds";
    for (size_t k = 0; k < kGaussianPanelClouds; ++k) {
        clouds.controls.push_back(reading(
            "gaussians.cloud." + std::to_string(k), "",
            [r, k] { return k < r().clouds.size() ? describeCloud(r().clouds[k], r()) : std::string(); },
            [r, k] { return k < r().clouds.size(); }));
    }
    clouds.controls.push_back(reading(
        "gaussians.more", "",
        [r] {
            return std::to_string(r().clouds.size() - kGaussianPanelClouds) + " more clouds, in the totals above";
        },
        [r] { return r().clouds.size() > kGaussianPanelClouds; }));
    panel.sections.push_back(std::move(clouds));
    return panel;
}

}   // namespace athenea::ui
