// Copyright (c) 2026 jesus luque.
//
// LIVE PAGES THE DEVICE DECODES FROM WHAT THEIR PRODUCER HOLDS. A page of a
// live stream of SPZ pages (ChunkFormat::Payload::Spz) is either an SPZ file
// or, when its bytes start with DevicePageHeader::kMagic, a small header and
// the producer's own payload: the splats are in something the producer has
// already put where the device can read it -- a video frame the hardware
// decoder wrote (video::VideoStream) -- and the producer, not the pool, knows
// how to turn it into the store's layout.
//
//   DevicePageHeader  the page's first bytes, staged as they are: which
//                     producer (`source`), which of its frames, which splats
//   DevicePages       the producer's side: place() records the kernels that
//                     write a page's splats into the store; retired() says
//                     which of the batches that did so the device has run
//
// THE THREADS (ChunkSource.h). publish and withdraw: any thread. place and
// retired: the pool's frame thread, the only one that calls the device. A
// producer holds what its pages read until every pool that placed them has
// retired them -- or the page's epoch was dropped, which no call says: a
// producer keeps a few frames and lets the oldest go.
#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include "athenea/core/Result.h"

namespace athenea::gpu {
class Buffer;
class CommandBatch;
}   // namespace athenea::gpu

namespace athenea::scene {
struct GpuSplats;
}

namespace athenea::lod {

struct DevicePageHeader {
    static constexpr uint32_t kMagic = 0x50565441;   // "ATVP"
    uint32_t magic = kMagic;
    uint32_t version = 1;
    uint64_t source = 0;        ///< DevicePages::publish's key
    uint64_t frame = 0;         ///< the producer's ticket for what the page reads
    uint32_t chunk = 0;
    uint32_t first = 0;         ///< the page's first splat in its frame
    uint32_t count = 0;         ///< splats, 1 to chunkSplats
    uint32_t antialiased = 0;   ///< the splats were fitted to be drawn antialiased
    uint32_t payloadBytes = 0;  ///< the producer's bytes after the header
    uint32_t pad[5] = {};
};
static_assert(sizeof(DevicePageHeader) == 64);

class DevicePages {
public:
    virtual ~DevicePages() = default;

    /// FRAME THREAD: records into `batch` what writes the page's `count`
    /// splats into `store` from splat `at` (positions, shape and `keepRest`
    /// rest coefficients a colour). Its payload is in `staging` from byte
    /// `payloadOffset` (on a 16-byte boundary). `pool` and `fence` name the
    /// batch: the value the pool's fence takes when the device has run it.
    [[nodiscard]] virtual Result<void> place(gpu::CommandBatch& batch, const DevicePageHeader& page,
                                             const gpu::Buffer& staging, uint64_t payloadOffset,
                                             scene::GpuSplats& store, uint64_t at, uint32_t keepRest,
                                             const void* pool, uint64_t fence) = 0;
    /// FRAME THREAD: `pool`'s fence has reached `completed`.
    virtual void retired(const void* pool, uint64_t completed) = 0;

    /// A producer offers itself to every pool of the process: the key its
    /// pages carry (DevicePageHeader::source). Any thread.
    [[nodiscard]] static uint64_t publish(std::shared_ptr<DevicePages> pages);
    static void withdraw(uint64_t key);
    /// Null when no producer has that key (any more). Any thread.
    [[nodiscard]] static std::shared_ptr<DevicePages> find(uint64_t key);
    /// Whether `bytes` start as a device page does.
    [[nodiscard]] static bool isDevicePage(std::span<const std::byte> bytes) noexcept;
};

}   // namespace athenea::lod
