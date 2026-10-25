// Copyright (c) 2026 jesus luque.
//
// .athc: a cloud with its levels of detail, as a file a stream reads chunk by
// chunk. Everything in it was computed on the device; the CPU copies bytes.
//
//   page 0        header
//   page 1..      level table, chunk table
//   aligned       the finest level's group starts
//   aligned       each level: positions, shape, SH, cells (per group)[, normals][, emission]
//   aligned       each chunk: positions, shape, SH, finest group (per splat)[, normals][, emission]
//
// Version 2 adds the header's `flags`; bit 0 says every block ends with the
// shading normals (one word an element, GpuSplats::normals), bit 1 that the
// colours are linear light (GpuSplats::linear) rather than a capture's sRGB,
// and bit 2 that the emitted radiance follows them (one RGB9E5 word an
// element, GpuSplats::emission). Version 1 files, which have none, are still read.
// A flag bit the reader does not know is refused: it may add a block.
//
// Every block starts on a 4096-byte page, so a chunk maps and faults in alone.
// Little-endian, the device's own packing (athenea/common/packing.slang).
//
//   AthcSource     a .athc as a ChunkSource: its format, its chunks' bytes
//   StreamingPool  the merged levels on the device from the start; the
//                  splats' chunks loaded by worker threads when a cut wants
//                  them (or pushed by a live source, SplatStream.h), placed
//                  in a store of fixed size, most wanted first, and dropped
//                  when it is full and they are wanted least
//
// Threads (ChunkSource.h): the pool's open, want and update run on the frame
// thread -- the thread that opened it -- and are refused on any other; its
// loader threads only copy bytes into staging the frame thread mapped.
#pragma once

#include <filesystem>
#include <memory>
#include <span>

#include "athenea/lod/ChunkSource.h"
#include "athenea/lod/Lod.h"

namespace athenea::lod {

/// Writes a cloud whose chunks are all on the device.
[[nodiscard]] Result<void> writeAthc(gpu::Device& device, const LodCloud& cloud, const std::filesystem::path& path);

/// Reads the whole file onto the device: every chunk there, chunk c in slot c.
[[nodiscard]] Result<LodCloud> readAthc(gpu::Device& device, const std::filesystem::path& path);

[[nodiscard]] bool isAthc(const std::filesystem::path& path);

/// A .lrtc lucabRTrender wrote, as the .athc it is under this name: the same
/// layout (version 1) with another magic, so the header is rewritten and the
/// payload copied as it is. True when `in` was a .lrtc; an .athc is copied
/// unchanged and returns false. `out` may not be `in`.
[[nodiscard]] Result<bool> migrateLrtc(const std::filesystem::path& in, const std::filesystem::path& out);

/// A .athc file's chunks: its header parsed, its pages mapped, nothing on the
/// device until openCloud.
class AthcSource final : public ChunkSource {
public:
    /// Any thread: reads the header and maps the file.
    [[nodiscard]] static Result<std::unique_ptr<AthcSource>> open(const std::filesystem::path& path);
    ~AthcSource() override;

    [[nodiscard]] const ChunkFormat& format() const noexcept override;
    [[nodiscard]] uint32_t chunkCount() const noexcept override;
    [[nodiscard]] scene::Bounds bounds(uint32_t chunk) const override;
    [[nodiscard]] std::string name() const override;
    /// Frame thread, once.
    [[nodiscard]] Result<LodCloud> openCloud(gpu::Device& device) override;
    [[nodiscard]] ChunkRequest request(uint32_t chunk) const override;
    /// Loader threads (or any): a copy off the mapping, so the page faults
    /// are the caller's and not the frame's.
    [[nodiscard]] Result<void> fetch(const ChunkRequest& request, std::span<std::byte> staging) override;

private:
    AthcSource();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// WHICH EPOCH A LIVE STREAM'S FRAME SHOWS. Latest: the newest complete one,
/// as soon as it is on the device. Clock: the newest complete one whose
/// `ptpTime` is not after the instant the frame renders (StreamingPool::
/// setClock); later ones are held, complete and unfetched, until their
/// instant comes -- at most `heldEpochs` of them, the oldest dropped past it.
enum class LiveSync : uint8_t { Latest, Clock };

struct StreamingSettings {
    /// The store's size in splats, rounded down to whole chunks (at least one;
    /// a file's at most its chunks, a live stream's at most twice them, since
    /// an epoch is built beside the one shown).
    uint64_t budgetSplats = uint64_t{4} << 20;
    uint32_t loaders = 2;
    /// The most chunk bytes an update places; a chunk past it waits for the
    /// next. 0: no limit. At least one chunk a frame is placed, however big.
    uint64_t bytesPerFrame = 0;
    /// Chunks the loaders may hold in staging at once (each a whole chunk's
    /// bytes, in one mapped upload buffer). 0: 2 * loaders + 4.
    uint32_t stagingChunks = 0;
    /// The kernels a source of SPZ pages (ChunkFormat::Payload::Spz) is
    /// decoded with; needed for that payload only.
    gpu::ShaderLibrary* library = nullptr;
    /// A live source: which epoch a frame shows, and how many complete
    /// epochs wait for their instant at most (LiveSync::Clock).
    LiveSync sync = LiveSync::Latest;
    uint32_t heldEpochs = 8;
};

class StreamingPool {
public:
    /// Frame thread: the thread that calls this is the pool's frame thread.
    [[nodiscard]] static Result<std::unique_ptr<StreamingPool>> open(gpu::Device& device,
                                                                     const std::filesystem::path& path,
                                                                     const StreamingSettings& settings = {});
    /// The same over any source; the pool owns it.
    [[nodiscard]] static Result<std::unique_ptr<StreamingPool>> open(gpu::Device& device,
                                                                     std::unique_ptr<ChunkSource> source,
                                                                     const StreamingSettings& settings = {});
    /// Frame thread.
    ~StreamingPool();

    /// Frame thread. Cut this, as any cloud; it changes only in update(). A
    /// live source's shows one whole epoch, the last one complete.
    [[nodiscard]] const LodCloud& cloud() const noexcept;

    /// Frame thread: the instant, PTP nanoseconds, the next update's frame
    /// shows (LiveSync::Clock). An epoch already shown stays shown when the
    /// clock goes back: an epoch is a change on the one before, and is not
    /// undone; the epochs after it wait for the clock to come round again.
    void setClock(uint64_t ptpNs);

    /// Frame thread. What a cut of this frame wants (CutStats::needs), once
    /// per instance. A live source's chunks are all wanted: ignored.
    void want(std::span<const uint32_t> needs);

    /// Ends the frame. Wanted chunks not on the device are queued, most wanted
    /// first, as many as there is a place for: a free slot, a chunk not wanted
    /// now (least recently wanted first), or a chunk wanted less than half as
    /// much -- the half keeps two chunks from trading places frame after
    /// frame. Chunks loaded since the last update are placed the same way.
    /// `wait` blocks until the queued ones have loaded and places them too: an
    /// offline render's way to a complete frame. Returns the chunks placed.
    ///
    /// A live source: the epochs that became complete are loaded, in order,
    /// into slots the shown epoch does not use; when every page of one is
    /// placed, it is shown (cloud() changes) and the slots it no longer uses
    /// are freed. An epoch with no room beside the one shown is dropped
    /// (Status::droppedNoRoom, a warning) and the next is tried; so is one
    /// with a page a loader could not read, a corrupt SPZ page
    /// (Status::droppedCorrupt). `wait`
    /// blocks for the loads of the epoch being built.
    ///
    /// Either way, at most `bytesPerFrame` chunk bytes are placed. Frame
    /// thread only: a call from any other is refused.
    [[nodiscard]] Result<uint32_t> update(bool wait = false);

    struct Status {
        uint32_t slots = 0;      ///< chunks the store holds
        uint32_t resident = 0;   ///< chunks on the device
        uint32_t missing = 0;    ///< wanted at the last update and not on the device
        uint32_t inFlight = 0;   ///< queued or loading
        uint64_t loads = 0;      ///< chunks placed since open
        uint64_t evictions = 0;
        uint32_t deferred = 0;   ///< loaded, held back by bytesPerFrame at the last update
        uint64_t placedBytes = 0;   ///< chunk bytes the last update placed
        // A live source:
        uint64_t epoch = 0;      ///< the epoch shown (0: none yet)
        uint64_t ptpTime = 0;    ///< its PTP time
        uint32_t pendingEpochs = 0;   ///< complete, not yet shown
        uint64_t epochs = 0;     ///< epochs shown since open
        uint64_t droppedNoRoom = 0;   ///< epochs dropped: no room beside the one shown
        uint64_t droppedCorrupt = 0;  ///< epochs dropped: a page a loader could not read
        uint64_t droppedHeld = 0;     ///< epochs dropped: more than heldEpochs waiting for their instant
    };
    /// Any thread.
    [[nodiscard]] Status status() const;

    /// Frame thread: the bounds of the chunk as shown (a live page's own; a
    /// file's whole cloud).
    [[nodiscard]] scene::Bounds chunkBounds(uint32_t chunk) const;

private:
    StreamingPool();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace athenea::lod
