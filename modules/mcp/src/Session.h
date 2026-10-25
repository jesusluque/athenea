// Copyright (c) 2026 jesus luque.
//
// What the tools act on and what survives between calls: the stage that is
// open, the settings asked for, and the last frame's size -- so `pick` can
// answer about the frame the caller just rendered.
#pragma once

#include <map>
#include <memory>
#include <string>

#include "athenea/core/Platform.h"
#include "athenea/lod/SplatStream.h"

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
};

struct Session {
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
};

}   // namespace athenea::mcp
