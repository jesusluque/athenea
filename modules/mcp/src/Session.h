// Copyright (c) 2026 jesus luque.
//
// What the tools act on and what survives between calls: the stage that is
// open, the settings asked for, and the last frame's size -- so `pick` can
// answer about the frame the caller just rendered.
#pragma once

#include <atomic>
#include <map>
#include <optional>
#include <memory>
#include <string>

#include "athenea/bridge/CameraQueue.h"
#include "athenea/bridge/ClientPages.h"
#include "athenea/bridge/Frames.h"
#include "athenea/core/Platform.h"
#include "athenea/lod/SplatStream.h"

#include "FrameLoop.h"

#include <filesystem>
#include <memory>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/usd/StageRenderer.h"

namespace athenea::mcp {

/// A live stream a producer in another process feeds through this server
/// (stream_open): the ring it writes page bytes into, and the queue the
/// stage's AtheneaLiveSourceAPI prim reads (lod::PageQueue::named).
struct LiveStream {
    platform::SharedMemory          ring;
    std::shared_ptr<lod::PageQueue> queue;
    uint32_t                        chunks = 0;
    std::shared_ptr<void>           claim;   ///< the queue held for this session; let go with the last copy
    /// Records the producer's device writes (stream_open's devicePages):
    /// made and dropped on the frame thread, read by commits on any.
    std::shared_ptr<bridge::ClientPages> devicePages;
    /// Takes the producer out of the process's registry when the stream goes.
    std::shared_ptr<void>                devicePagesKey;
};

/// A subscriber's frames (render_subscribe). Frame thread only.
struct Subscription {
    std::unique_ptr<bridge::FramePublisher> publisher;
    std::vector<bridge::AovSpec>            aovs;
    uint32_t                                width = 0, height = 0;
    std::string                             camera;   ///< a stage camera when no camera_set came
    std::optional<render::Camera>           framing;  ///< or a framing camera, for a stage with none
    double                                  time = 0.0;
    bool                                    timebaseTai = false;
    uint32_t                                maxInFlight = 2;
    uint64_t                                frame = 0;      ///< the last frame published
    // What the frames did.
    uint64_t ticks = 0, drawn = 0, late = 0, skipped = 0, failed = 0;
    uint64_t waitsLast = 0, waitsSteadyMax = 0, waitsSteadyFrames = 0, waitsSteadyTotal = 0;
    uint64_t stalledSinceNs = 0;
    std::string lastError;
};

struct Session {
    /// THE FRAME THREAD: every tool that reaches the device runs on it
    /// (Server::callTool), and a subscription's frames are drawn on it.
    /// Made on first use; gone before the renderer.
    std::unique_ptr<FrameLoop>          loop;
    /// Hands this session's fences (and Vulkan memory) to other processes.
    std::unique_ptr<platform::HandleServer> handles;
    /// camera_set's cameras, read by the frames at their instants.
    std::shared_ptr<bridge::CameraQueue> cameras = std::make_shared<bridge::CameraQueue>();
    std::unique_ptr<Subscription>       subscription;
    /// The instant the frame being drawn stands for: what the live clock
    /// answers, so the epoch and the camera are chosen at the same instant.
    std::shared_ptr<std::atomic<uint64_t>> liveInstant = std::make_shared<std::atomic<uint64_t>>(0);

    /// The streams opened here, by name. The server's thread only enqueues
    /// onto them: the stage's render thread is the one that reaches the device.
    std::map<std::string, LiveStream>   streams;
    std::unique_ptr<usd::StageRenderer> renderer;
    std::filesystem::path               stage;
    uint32_t                            lastWidth = 0;
    uint32_t                            lastHeight = 0;
    double                              lastTime = 0.0;
    std::string                         lastCamera;

    [[nodiscard]] bool open() const noexcept { return renderer != nullptr; }

    /// The frame thread, started on first use. Any thread.
    FrameLoop& frameLoop() {
        if (!loop) loop = std::make_unique<FrameLoop>();
        return *loop;
    }
    ~Session() {
        // Device objects go on the frame thread that made them.
        if (loop) {
            loop->run([this] {
                (void)loop->setTicker(nullptr, {});
                subscription.reset();
                for (auto& [name, stream] : streams) {
                    stream.devicePagesKey.reset();
                    stream.devicePages.reset();
                }
                renderer.reset();
                return nlohmann::json();
            });
            loop.reset();
        }
    }
};

}   // namespace athenea::mcp
