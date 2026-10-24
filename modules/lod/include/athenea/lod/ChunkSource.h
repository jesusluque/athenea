// Copyright (c) 2026 jesus luque.
//
// WHERE A STREAMED CLOUD'S CHUNKS COME FROM. A StreamingPool keeps a store of
// chunk slots on the device; a ChunkSource is what fills them: a .athc file
// (AthcSource, Athc.h), whose chunks never change and are loaded when a cut
// wants them, or a live stream (LiveSource, SplatStream.h), whose chunks are
// pushed in epochs and all held on the device.
//
// THE THREADS. Three kinds, and what each may do:
//   frame thread   the one that opened the pool: the only one that calls the
//                  device (uploads, copies, submits). openCloud and poll run
//                  here.
//   loader threads the pool's own: fetch() copies a chunk's bytes into a
//                  staging range the frame thread mapped once at open. No
//                  device call, ever.
//   producers      anyone else (network, IPC, files): they only enqueue
//                  (PageQueue::push), and never see the device or the pool.
//
// A chunk's bytes are the .athc chunk block (or, for a live stream of SPZ
// pages, an SPZ file: ChunkFormat::Payload): n positions (float4), n shapes
// (4 uint), n * shWords SH words, n finest-group words, then -- where the
// format says -- n normals, n emissions and each extra's n * words. n is at
// most `chunkSplats`.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/lod/Lod.h"

namespace athenea::gpu {
class Device;
}

namespace athenea::lod {

/// What every chunk of a source holds, element by element.
struct ChunkFormat {
    /// WHAT A CHUNK'S BYTES ARE. Athc: the .athc chunk block below, copied
    /// into the store as it is. Spz: an SPZ file (any version the reader
    /// takes), decompressed by a loader thread into staging and turned into
    /// the store's layout by the device (spz_records, then the validate and
    /// decode every format takes) -- a live page only.
    enum class Payload : uint8_t { Athc, Spz };
    Payload  payload = Payload::Athc;
    uint32_t chunkSplats = 0;     ///< the most splats a chunk holds
    uint32_t restPerColour = 0;   ///< 0, 3, 8, 15
    uint32_t shWords = 0;
    bool     normals = false;     ///< a normal word an element
    bool     emission = false;    ///< an RGB9E5 word an element
    bool     linear = false;      ///< colours in linear light (GpuSplats::linear)
    uint32_t pbrWords = 0;        ///< 1 where the cloud carries a material
    uint32_t lobesWords = 0;      ///< 3 where it carries the material's layers
    uint32_t transferCount = 0;
    uint32_t transferWords = 0;
    uint32_t shadowWords = 0;

    /// Staging an SPZ chunk takes: a header, then its six streams (the most
    /// a splat's take, degree 4: 9 + 1 + 3 + 3 + 4 + 72 bytes), each on a word.
    static constexpr uint64_t kSpzHeaderBytes = 128;
    static constexpr uint64_t kSpzSplatBytes = 92;

    /// Bytes one splat takes in a chunk block (Athc), or at most in staging (Spz).
    [[nodiscard]] uint64_t elementBytes() const noexcept {
        if (payload == Payload::Spz) {
            return kSpzSplatBytes;
        }
        return 16 + 16 + 4 * uint64_t{shWords} + 4 + (normals ? 4 : 0) + (emission ? 4 : 0) +
               4 * uint64_t{pbrWords + lobesWords + transferWords + shadowWords};
    }
    /// The most a chunk takes in staging.
    [[nodiscard]] uint64_t chunkBytes() const noexcept {
        return payload == Payload::Spz ? kSpzHeaderBytes + kSpzSplatBytes * chunkSplats + 6 * 4
                                       : elementBytes() * chunkSplats;
    }
    bool operator==(const ChunkFormat&) const = default;
};

/// One load: a chunk, and which of its versions (a live page's ticket; 0 for
/// a file, whose chunks have one).
struct ChunkRequest {
    uint32_t chunk = 0;
    uint64_t ticket = 0;
    uint32_t splats = 0;   ///< splats the version holds
    uint64_t bytes = 0;    ///< splats * elementBytes
};

/// A change a live source pushes: chunk `chunk` takes the version `request`
/// (add, replace) or leaves (remove), as part of epoch `epoch`.
struct ChunkChange {
    enum class Op : uint8_t { Add, Replace, Remove };
    uint64_t      epoch = 0;
    Op            op = Op::Add;
    ChunkRequest  request;
    scene::Bounds bounds;
    uint64_t      ptpTime = 0;   ///< the epoch's PTP time, in nanoseconds
};

class ChunkSource {
public:
    virtual ~ChunkSource() = default;

    /// Any thread: fixed from construction.
    [[nodiscard]] virtual const ChunkFormat& format() const noexcept = 0;
    /// Any thread: chunk ids are 0 .. chunkCount() - 1, fixed from construction.
    [[nodiscard]] virtual uint32_t chunkCount() const noexcept = 0;
    /// Any thread: the bounds of the chunk's latest version the source handed
    /// out. A file keeps none per chunk: the whole cloud's.
    [[nodiscard]] virtual scene::Bounds bounds(uint32_t chunk) const = 0;
    /// Any thread: a name for messages (a file's name, a stream's).
    [[nodiscard]] virtual std::string name() const = 0;
    /// Whether chunks are pushed in epochs (poll) rather than loaded on demand.
    [[nodiscard]] virtual bool live() const noexcept { return false; }

    /// FRAME THREAD, once, at open: the cloud without its store -- the merged
    /// levels and the finest groups' starts on the device, the counts and the
    /// bounds. A live source's has no levels: its chunks are drawn as they are.
    [[nodiscard]] virtual Result<LodCloud> openCloud(gpu::Device& device) = 0;

    /// The request that loads chunk `chunk` on demand. Any thread. A live
    /// source answers only through poll.
    [[nodiscard]] virtual ChunkRequest request(uint32_t chunk) const = 0;

    /// LOADER THREADS: the bytes of `request` into `staging`, which holds
    /// exactly `request.bytes`. No device call. Several loaders call it at
    /// once, for different requests.
    [[nodiscard]] virtual Result<void> fetch(const ChunkRequest& request, std::span<std::byte> staging) = 0;

    /// FRAME THREAD: `request` will never be fetched (its epoch was dropped);
    /// a source that holds its bytes frees them.
    virtual void discard(const ChunkRequest& /*request*/) {}

    /// FRAME THREAD, each update: the changes of every epoch that became
    /// complete since the last call, in epoch order, an epoch's changes
    /// together. A file pushes none.
    virtual void poll(std::vector<ChunkChange>& /*out*/) {}
};

}   // namespace athenea::lod
