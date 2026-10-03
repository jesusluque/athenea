// Copyright (c) 2026 jesus luque.
//
// The viewer's own panels, as a description any front end can draw.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "athenea/ui/Controls.h"
#include "athenea/ui/GaussianPanel.h"

namespace athenea::ui {

/// Everything the panels edit. A front end owns one of these and hands it to
/// `viewerPanels`; the controls read and write it, and the renderer is told
/// about the change by whoever is driving the frame.
struct ViewerSettings {
    // What is drawn, and how what a pixel sees is found.
    std::string technique = "raster";
    std::string visibility = "automatic";
    std::string aov = "color";

    // The camera the gestures move.
    double focal = 35.0;          ///< mm
    double fStop = 0.0;           ///< 0 is a pinhole
    double focusDistance = 0.0;   ///< 0 follows the orbit's distance

    // Light.
    bool   defaultLights = true;
    bool   cloudShadows = false;
    double shadowDensity = 1.0;

    // The path tracer.
    int  pathSamples = 1;
    int  pathBounces = 4;
    bool denoise = true;

    // What reaches the screen.
    std::string viewTransform = "agx";
    std::string displayEncoding = "srgb";
    double      exposure = 0.0;       ///< stops
    double      renderScale = 1.0;
    double      shutter = 0.0;        ///< frames

    // Time.
    double time = 0.0;
    bool   playing = false;
    /// One time code a drawn frame, rather than the stage's rate by the
    /// clock: every frame of a clip is seen however long each takes to draw.
    bool   everyFrame = false;

    // The Gaussians panel.
    bool   timeSplatStages = false;   ///< time each rasteriser stage, each waiting for the device
};

/// A variant set of the stage: USD's "pick one of these". The engine does not
/// know what any of them mean -- an animation clip, a level of detail, a shirt
/// -- so the panels offer every set a stage brings.
struct VariantSet {
    std::string              prim;
    std::string              name;
    std::vector<std::string> variants;
    std::string              selected;
};

/// A dome light of the stage: the one light whose whole character is a file,
/// so swapping the file relights the frame with nothing else touched.
struct Dome {
    std::string prim;
    std::string name;
    std::string texture;    ///< empty: the stage's own opinion (a colour, or its file)
    float       rotation = 0.0F;   ///< degrees about the up axis
};

/// What the panels show but do not edit: what the frame and the stage know.
/// Each is a function because a panel is built once and read every frame.
struct ViewerFacts {
    std::function<bool()>        stageHasLights;
    std::function<bool()>        ocioAvailable;
    std::function<bool()>        extendedRange;
    /// The time codes the stage's animation occupies now: the selected
    /// clip's, not the longest one the root layer declares.
    std::function<std::pair<double, double>()> timeRange;
    std::function<std::vector<VariantSet>()>   variantSets;
    std::function<std::vector<Dome>()>         domes;
    /// The images a dome may be given (.hdr, .exr): read once, when the panels
    /// are built -- a panel that lists a folder every frame stutters.
    std::function<std::vector<std::string>()>  skies;
    std::function<std::string()> device;        ///< "Metal on Apple A16"
    std::function<std::string()> frameTiming;   ///< "1179 x 2556, draw 8.2 ms"
    std::function<std::string()> sceneCounts;   ///< "1.2 M gaussians, 3 lights"
    std::function<std::string()> status;        ///< the last error, or empty
    std::function<std::string()> paths;         ///< "128 paths a pixel"
    /// The gaussians on screen (GaussianPanel.h): set, the panels gain a
    /// Gaussians panel that reads it.
    GaussianSource               gaussians;
};

/// What a button does. Framing and playback are the front end's to carry out,
/// since only it knows about the camera it moves and the clock it runs.
struct ViewerActions {
    std::function<void()> frameAll;
    std::function<void()> stepBack;
    std::function<void()> stepForward;
    std::function<void()> toStart;
    std::function<void()> reloadStage;
    std::function<void(const std::string& prim, const std::string& set, const std::string& variant)>
        chooseVariant;
    /// An empty texture puts the stage's own opinion back.
    std::function<void(const std::string& prim, const std::string& texture)> setDomeTexture;
    std::function<void(const std::string& prim, float degrees)>              setDomeRotation;
    /// Where more skies are: `athenea view --hdri`, as a front end asks for it.
    std::function<void()>                                                     chooseSkyFolder;
};

/// The panels, over `settings`, which must outlive them. The variant sets are
/// read once, here: a front end builds the panels again when it opens a stage.
/// View first; Gaussians after it where `facts.gaussians` is set.
[[nodiscard]] std::vector<Panel> viewerPanels(ViewerSettings& settings, ViewerFacts facts,
                                              ViewerActions actions);

}   // namespace athenea::ui
