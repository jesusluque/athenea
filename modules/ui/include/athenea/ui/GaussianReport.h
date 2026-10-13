// Copyright (c) 2026 jesus luque.
//
// WHAT THE GAUSSIANS ON SCREEN ARE, AND WHAT THE FRAME DID WITH THEM.
//
// Two kinds of number, kept apart because they are known at different times:
//
//  - what the frame was handed: the stage's clouds, what each carries, what
//    the level of detail kept of it, what it holds on the device. Bookkeeping
//    the engine already has, exact for `frame`.
//  - what the device counted: how many it kept and why it culled the rest,
//    the tile pairs, each cloud's share. Copied out without waiting
//    (gpu::AsyncReadback), so they belong to `countedFrame`, which may be a
//    frame or two behind `frame` -- and a panel says which.
//
// The engine fills it (athenea::usd::GaussianStats is this) and the Gaussians
// panel reads it (`gaussianPanel`): numbers only, so the panels still know
// nothing of a device or a stage.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace athenea::ui {

/// One cloud of the stage (a ParticleField prim, or a level of one).
struct GaussianCloud {
    std::string prim;
    uint32_t    gaussians = 0;    ///< what it holds: the cloud, or the whole asset a cut is taken from
    bool        drawn = false;    ///< in this frame at all: visible, and its level the one chosen
    uint32_t    submitted = 0;    ///< handed to the renderer this frame, after the level of detail
    /// The level of detail, in words: "" where it has none; "level 2 of 4 in
    /// 'bird'" for a variant level; "cut" for a cut of an asset.
    std::string lod;
    uint32_t    lodOwn = 0;       ///< of `submitted`, the cloud's own splats (a cut that reported it)
    uint32_t    lodMerged = 0;    ///< and the merged gaussians standing in for groups
    bool        streamed = false; ///< a `.athc` read chunk by chunk
    uint32_t    chunks = 0;
    uint32_t    chunksResident = 0;
    uint32_t    chunksWanted = 0; ///< chunks this view asked for
    uint32_t    chunksMissing = 0;
    uint32_t    chunksInFlight = 0;

    // What it carries: static facts of the data.
    uint32_t shDegree = 0;
    bool     linear = false;      ///< its colours are linear light, not a capture's sRGB
    bool     relit = false;       ///< AtheneaSplatLightingAPI
    bool     litBody = false;
    uint32_t transfer = 0;        ///< transfer values a gaussian (0, 9, 10 zonal, 36)
    bool     skinned = false;
    bool     normals = false;
    bool     emission = false;
    bool     pbr = false;
    bool     crypto = false;
    bool     visibility = false;  ///< a baked per-part visibility
    float    ior = 0.0F;

    uint64_t bytes = 0;           ///< device memory its arrays hold, a posed copy included

    // As the device counted it, in `GaussianReport::countedFrame`.
    bool     counted = false;
    uint32_t visible = 0;
    uint32_t pairs = 0;
};

struct GaussianReport {
    uint64_t    frame = 0;        ///< the engine's count of frames drawn, this one included
    /// "raster"; "rt" (the splats traced); "rt+raster" (meshes traced, the
    /// splats rasterised over them). Empty before a frame.
    std::string route;
    uint64_t    inStage = 0;      ///< gaussians in every cloud of the stage, drawn or not
    uint64_t    submitted = 0;    ///< handed to the renderer this frame

    // The rasteriser's stages, in milliseconds. Only measured when
    // `stagesTimed`: each stage then waits for the device, which slows the
    // frame, so it is asked for (athenea::usd::StageRenderer::setTimeSplatStages).
    bool   stagesTimed = false;
    double projectMs = 0, countsMs = 0, depthSortMs = 0, emitMs = 0, tileSortMs = 0, blendMs = 0, totalMs = 0;

    // The ray tracer, when it drew the splats.
    bool        traced = false;
    bool        rebuilt = false;  ///< structures built this frame, not refitted or kept
    double      buildMs = 0, traceMs = 0, tracedMs = 0;
    uint32_t    tracedSplats = 0;
    uint32_t    tracedChunks = 0;
    std::string traceRoute;       ///< "hardware" or "compute BVH"

    // As the device counted it.
    bool     counted = false;
    uint64_t countedFrame = 0;
    uint32_t countedSlots = 0;    ///< gaussians that frame submitted
    uint32_t visible = 0;         ///< what the depth sort sorted
    uint32_t pairs = 0;           ///< what the tile sort sorted
    uint32_t maxTiles = 0;
    /// Culled, by reason, as render::SplatCounters::Cull numbers them.
    std::array<uint32_t, 7> culled{};

    // Device memory.
    uint64_t cloudBytes = 0;      ///< every cloud's arrays
    uint64_t poolBytes = 0;       ///< what the levels of detail and the streaming stores hold

    std::vector<GaussianCloud> clouds;
};

}   // namespace athenea::ui
