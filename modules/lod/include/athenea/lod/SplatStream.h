// Copyright (c) 2026 jesus luque.
//
// LIVE SPLAT STREAMS. A cloud whose chunks change while it is drawn: pages
// pushed by producers -- a network receiver, another process, a file being
// played -- in epochs, each epoch a set of changes that only make sense
// together (a frame of a capture, an edit). The frame never draws half an
// epoch: it draws the last one whose every page is on the device.
//
//   PageUpdate   one change: chunk `chunk` of stream `stream` takes these
//                bytes (add, replace) or leaves (remove), in epoch `epoch`,
//                which has `pages` pages in all
//   PageQueue    many producers push, one consumer drains: a mutex and a
//                vector a stream, nothing a producer waits on but the push
//   LiveSource   a ChunkSource over one stream of a queue: it gathers pages
//                into epochs and hands a StreamingPool the complete ones, in
//                order; the pool loads them into slots beside the epoch
//                shown and shows the new one when every page is placed
//
// THE THREADS (ChunkSource.h). Producers call PageQueue::push and
// LiveSource::check, nothing else. The pool's loader threads call
// LiveSource::fetch. The frame thread -- the pool's -- calls everything that
// reaches the device, and LiveSource::poll through the pool's update.
//
// A page's bytes are the .athc chunk block (ChunkSource.h) of the stream's
// format, with 1 to `chunkSplats` splats. An epoch's pages are all pushed
// before any page of a later epoch of the same stream: when a later epoch is
// complete while an earlier one is not, the earlier one is dropped (its pages
// never all came) and counted.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/lod/ChunkSource.h"

namespace athenea::lod {

struct PageUpdate {
    using Op = ChunkChange::Op;
    uint32_t               stream = 0;
    uint64_t               epoch = 0;     ///< increasing within a stream, from 1
    uint32_t               chunk = 0;
    Op                     op = Op::Add;
    std::vector<std::byte> bytes;         ///< the chunk block; empty for a remove
    scene::Bounds          bounds;        ///< the page's splats', as the producer knows them
    uint64_t               ptpTime = 0;   ///< the epoch's PTP time, nanoseconds
    uint32_t               pages = 0;     ///< pages in epoch `epoch` (every page of it says the same)
};

class PageQueue {
public:
    /// Any thread: a producer's one call. Every reader of the stream gets the
    /// page (two prims naming one stream both draw it). A stream no reader
    /// reads keeps only its newest complete epoch and what came after it.
    void push(PageUpdate page);

    /// A reader of `stream` (a LiveSource): it is handed what the stream kept
    /// unread, and from now on every page pushed. Any thread.
    [[nodiscard]] uint64_t subscribe(uint32_t stream);
    void unsubscribe(uint64_t reader);
    /// The reader's pages pushed since its last call, in push order. Any thread.
    void read(uint64_t reader, std::vector<PageUpdate>& out);
    /// What `stream` kept unread, taken. Any thread.
    void drain(uint32_t stream, std::vector<PageUpdate>& out);

    /// THE QUEUE A NAME STANDS FOR, made on first use and shared by every
    /// caller of the same name for the process's life: how a producer and a
    /// stage that names the stream (AtheneaLiveSourceAPI) meet without
    /// knowing each other. Any thread.
    [[nodiscard]] static std::shared_ptr<PageQueue> named(const std::string& name);

    /// One producer at a time may feed a named stream from outside the
    /// process (stream_open): true when `owner` now holds it -- or held it
    /// already. Any thread.
    [[nodiscard]] bool claim(uint64_t owner);
    void release(uint64_t owner);

    /// What the producer asked a frame to show (stream_open's `sync`), for a
    /// stage that does not say: 0 unsaid, 1 the latest epoch, 2 by the clock.
    /// Any thread.
    void setSyncHint(uint8_t hint) noexcept { syncHint_.store(hint); }
    [[nodiscard]] uint8_t syncHint() const noexcept { return syncHint_.load(); }

private:
    struct Reader {
        uint32_t                stream = 0;
        std::vector<PageUpdate> pages;
    };
    std::atomic<uint8_t>                                  syncHint_{0};
    std::mutex                                            mutex_;
    std::unordered_map<uint64_t, Reader>                  readers_;
    uint64_t                                              nextReader_ = 1;
    std::unordered_map<uint32_t, std::vector<PageUpdate>> unread_;
    uint64_t                                              owner_ = 0;
};

struct LiveSettings {
    uint32_t      stream = 0;
    uint32_t      chunks = 0;    ///< chunk ids 0 .. chunks - 1
    ChunkFormat   format;
    scene::Bounds bounds;        ///< what the stream's splats stay within, for the cloud
    std::string   name = "live";
};

class LiveSource final : public ChunkSource {
public:
    /// Any thread.
    [[nodiscard]] static Result<std::unique_ptr<LiveSource>> create(std::shared_ptr<PageQueue> queue,
                                                                    const LiveSettings& settings);
    ~LiveSource() override;

    /// Any thread: whether `page` is a page of a stream of `settings`' kind.
    /// A producer may ask before pushing; poll refuses what fails it.
    [[nodiscard]] static Result<void> check(const PageUpdate& page, const LiveSettings& settings);

    [[nodiscard]] const ChunkFormat& format() const noexcept override;
    [[nodiscard]] uint32_t chunkCount() const noexcept override;
    [[nodiscard]] scene::Bounds bounds(uint32_t chunk) const override;
    [[nodiscard]] std::string name() const override;
    [[nodiscard]] bool live() const noexcept override { return true; }
    /// Frame thread, once: no levels, no store; the counts and the bounds.
    [[nodiscard]] Result<LodCloud> openCloud(gpu::Device& device) override;
    /// Pages are pushed, never asked for: an empty request.
    [[nodiscard]] ChunkRequest request(uint32_t chunk) const override;
    /// Loader threads: the page's bytes, once; then they are freed.
    [[nodiscard]] Result<void> fetch(const ChunkRequest& request, std::span<std::byte> staging) override;
    /// Frame thread: the page's bytes freed unfetched (its epoch was dropped).
    void discard(const ChunkRequest& request) override;
    /// Frame thread: the pages pushed since the last call, gathered; every
    /// epoch now complete, in order.
    void poll(std::vector<ChunkChange>& out) override;

    struct Stats {
        uint64_t pages = 0;     ///< accepted
        uint64_t refused = 0;   ///< failed check()
        uint64_t late = 0;      ///< for an epoch already handed over or dropped
        uint64_t dropped = 0;   ///< epochs given up: a later one completed first
        uint64_t epochs = 0;    ///< handed to the pool
    };
    /// Any thread.
    [[nodiscard]] Stats stats() const;

private:
    LiveSource();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::lod
