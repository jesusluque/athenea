// Copyright (c) 2026 jesus luque.
//
// athenea view: a window onto a USD stage through the engine's Hydra delegate.
// Frames stay on the device -- the display transform writes the window's
// surface texture, Dear ImGui draws its panels over it -- and nothing comes
// back but a picked pixel's ids.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::view {

struct ViewOptions {
    std::filesystem::path stage;
    std::string           camera;                   ///< a camera prim; empty, a free camera framing the stage
    std::string           technique = "raster";     ///< "raster" or "rt"
    /// A prim whose matte is pulled on its own, white on black, from the
    /// first frame: the Isolate button without the click. Needs a matte, so
    /// it turns the Cryptomatte output on by itself.
    std::string           isolate;
    /// Which output the window starts on, as the Output combo names one:
    /// "color", "depth", "primId", "Neye", "cryptomatte", "CryptoObject00"...
    /// Empty is the colour, which is where the window always started.
    std::string           aov;
    std::string           visibility = "automatic";   ///< mesh visibility: automatic, raster, rays, bvh
    uint32_t              width = 1600;
    uint32_t              height = 900;
    uint32_t              frames = 0;               ///< stop after this many; 0, when the window closes
    uint32_t              lightSamples = 1;         ///< samples per light per pixel
    bool                  chooseLights = false;     ///< one light a sample, by power
    /// The path traced technique. A window keeps gathering paths while the
    /// camera is still and starts again when it moves; `pathTotal` is when
    /// the frame counts as converged, which is when a denoise runs.
    uint32_t              pathSamples = 1;          ///< paths a pixel each frame
    uint32_t              pathBounces = 4;          ///< bounces after the first hit
    uint32_t              pathTotal = 64;           ///< paths a pixel it counts as converged at
    bool                  denoise = false;          ///< denoise once converged
    /// A sky and a sun in the session layer when the stage authors no lights
    /// (StageRenderer::setDefaultLights). Without them a stage like the chess
    /// set or Kitchen_set is lit from the eye, and the path tracer has
    /// nothing to show that the raster does not.
    bool                  defaultLights = true;
    /// Extended dynamic range: a float surface in linear P3, ACES 2.0 with
    /// the screen's peak as its peak. On a standard display the same as off.
    bool                  edr = false;
    /// An OpenColorIO view to start with: any of these set compiles the
    /// config's display and view into the display kernel (empty config:
    /// OCIO's built-in studio config; empty display or view: its defaults).
    std::string           ocioConfig;
    std::string           ocioDisplay;
    std::string           ocioView;
    bool                  visible = true;
    /// The viewer's own camera's diaphragm: depth of field at `focus`, 0 a
    /// pinhole. A stage camera brings its own. The panel has both.
    double                fStop = 0.0;
    double                focus = 0.0;
    /// Variant selections to make before the first frame, each as USD writes
    /// one in a path: `/World{clip=air_fly_A0}`. The panel offers whatever
    /// variant sets the stage carries.
    std::vector<std::string> variants;
    /// WHERE THE PANEL'S LIST OF SKIES COMES FROM: directories of `.hdr` and
    /// `.exr` images it offers for the stage's dome lights.
    ///
    /// Left empty, the folder of whatever image each dome already carries is
    /// offered, so a stage whose sky sits in a library of them needs no
    /// option at all. A dome that is only a colour brings no folder, and then
    /// this is the only way to give it one.
    std::vector<std::filesystem::path> hdriPaths;
    /// Start with the timeline playing, as the Play button does.
    bool                  play = false;
    /// Play a time code a drawn frame rather than by the wall clock, so that
    /// every frame of the timeline is drawn however long each one takes. The
    /// panel has it as a tick beside Play.
    bool                  everyFrame = false;
    /// How long the shutter is open, in frames, for a camera the viewer makes
    /// for itself (a stage camera's own is taken instead). It is what a
    /// cloud's motion blur is integrated over: 0.5 is a 180 degree shutter,
    /// 0 is none. The panel has it as a slider.
    float                 shutter = 0.0F;
    /// Where the last frame goes as it was shown, panels included: an EXR of
    /// display-encoded values. Empty, nowhere.
    std::filesystem::path snapshot;
    /// WHERE EVERY FRAME GOES, as it was shown and with the panels on it: a
    /// directory, one PNG a frame named `frame_00000.png` and up. What
    /// `--snapshot` writes once, this writes for the whole run, which is what
    /// a capture of the viewer playing a timeline is. Empty, nowhere.
    ///
    /// A PNG and not an EXR: a run is hundreds of frames and a retina window
    /// is eighty megabytes of float each, while what a capture wants is what
    /// the display transform already encoded.
    ///
    /// It costs a readback and a file a frame, so a captured run does not
    /// time like a run that is only drawn.
    std::filesystem::path capture;
};

struct ViewStats {
    uint32_t frames = 0;
    uint64_t snapshotLitPixels = 0;   ///< of the snapshot: pixels brighter than the background
    double   medianDrawMs = 0.0;    ///< Hydra and the engine
    double   medianFrameMs = 0.0;   ///< the whole loop, events to present
    double   lastTime = 0.0;        ///< the USD time the last frame drew
    uint32_t distinctTimes = 0;     ///< how many different times the frames drew
};

[[nodiscard]] Result<ViewStats> runViewer(const ViewOptions& options);

}   // namespace athenea::view
