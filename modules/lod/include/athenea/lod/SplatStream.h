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
    /// Any thread: a producer's one call.
    void push(PageUpdate page);
    /// The consumer of `stream` (its LiveSource, on the frame thread): every
    /// page of it pushed so far, in push order, appended to `out`.
    void drain(uint32_t stream, std::vector<PageUpdate>& out);

private:
    std::mutex                                            mutex_;
    std::unordered_map<uint32_t, std::vector<PageUpdate>> pending_;
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
