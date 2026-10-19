// Copyright (c) 2026 jesus luque.
//
// The engine in a process of its own, as the OpenFX plugin talks to it.
//
// WHY ANOTHER PROCESS. A compositor is a process with its own OpenUSD, its own
// TBB and its own idea of the GPU. The engine brings one of each, and two of
// any of them in one process is a crash waiting for the first symbol they
// share. So the plugin links none of the engine: it starts `athenea-mcp`,
// speaks the JSON-RPC that server already speaks to everybody, and reads each
// frame back from a file the server writes -- the numbers alone, no picture.
// A device lost or a kernel that faults takes the server down, not Nuke, and
// the next frame starts it again.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace athenea::ofx {

/// A camera as a compositor holds one.
struct ViewCamera {
    std::array<double, 16> cameraToWorld{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};   ///< row by row
    double focal = 50.0;        ///< mm
    double haperture = 24.576;  ///< mm
    double vaperture = 0.0;     ///< mm; 0 takes the image's aspect
    double nearZ = 0.1;
    double farZ = 10000.0;
};

/// One frame asked for.
struct FrameRequest {
    std::filesystem::path cloud;   ///< the .usdc/.usda holding the gaussians
    ViewCamera            camera;
    double                time = 0.0;   ///< USD time code
    uint32_t              width = 0;
    uint32_t              height = 0;
    /// Layers beyond colour and depth, each four floats a pixel: "P" and
    /// "CryptoObject00".."CryptoObject02".
    std::vector<std::string> layers;
    /// The sky every dome of the stage carries: empty is the stage's own.
    std::string           sky;
    double                skyRotation = 0.0;   ///< degrees about the stage's up axis
    bool                  skyVisible = true;   ///< false: a transparent background, the sky still lighting
};

/// One frame as the server drew it: bottom row first, linear, premultiplied.
struct Frame {
    uint32_t           width = 0;
    uint32_t           height = 0;
    std::vector<float> rgba;    ///< four floats a pixel
    std::vector<float> depth;   ///< one a pixel: view z, 0 where nothing was drawn
    std::map<std::string, std::vector<float>> layers;   ///< as asked, four floats a pixel each
    std::string        manifest;   ///< the Cryptomatte's names and ids (JSON), where a matte was asked for
};

class RenderServer {
public:
    /// The server binary: $ATHENEA_MCP if set, else `athenea-mcp` beside the
    /// plugin's bundle (Contents/Resources/bin), else on the PATH.
    explicit RenderServer(std::filesystem::path binary = {});
    ~RenderServer();
    RenderServer(const RenderServer&) = delete;
    RenderServer& operator=(const RenderServer&) = delete;

    /// Renders `request`; on failure returns false and says why in `error`.
    /// Starts the server the first time, and again after it died.
    bool render(const FrameRequest& request, Frame& frame, std::string& error);

    /// Where the server binary is looked for when none is given.
    [[nodiscard]] static std::filesystem::path defaultBinary();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::ofx
