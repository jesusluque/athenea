// Copyright (c) 2026 jesus luque.
//
// LIVE PAGES ANOTHER PROCESS WROTE WITH ITS DEVICE (stream_open's
// `devicePages`). Beside the SPZ ring a producer may ask for memory both
// devices see: it writes gaussians there as float records -- a 3DGS PLY
// vertex's fourteen floats (x y z, opacity as a logit, three log scales, the
// rotation w x y z, the DC colour) and its rest coefficients, all red then
// green then blue -- with its own device, raises a fence it shares with the
// engine to a value of its choosing, and commits pages naming the records
// and that value. A commit queues device pages (lod::DevicePages): when the
// pool places one, the engine's device first waits for the fence to reach the
// page's value -- a wait on the device, never the host -- and then validates
// and decodes the records into the store as any cloud's are.
//
// When may the producer write over records again: the region's header holds
// `consumed`, the newest commit ticket whose pages every pool placing them
// has retired (or whose epoch a newer one replaced): a ticket is the commit's
// epoch. Written by the engine's host as the pools retire them.
//
//   memory        Metal: shared memory, a header page then the records, one
//                 buffer over the records' pages on either side. Elsewhere: not done.
//   the fence     `written`: the engine makes it, the producer imports and
//                 raises it (platform::HandleServer, the answer's token).
//
// THREADS. create: the frame thread (it makes buffers). page, info: any
// thread (the server's). place, retired: the pool's frame thread (lod's
// DevicePages contract).
#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "athenea/core/Platform.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/SharedFence.h"
#include "athenea/lod/DevicePages.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class Device;
class ShaderLibrary;
}   // namespace athenea::gpu

namespace athenea::bridge {

inline constexpr uint32_t kPagesMagic = 0x50435441;   // "ATCP"

/// The region's first page.
struct PagesHeader {
    uint32_t magic = kPagesMagic;
    uint32_t version = 1;
    uint32_t floatsPerRecord = 0;
    uint32_t degree = 0;
    uint64_t records = 0;    ///< records the region holds
    uint64_t consumed = 0;   ///< the newest ticket every pool is done with (engine's host)
    uint64_t placed = 0;     ///< pages placed
    uint64_t pad[3] = {};
};

/// A device page's payload after lod::DevicePageHeader.
struct ClientPagePayload {
    uint64_t fenceValue = 0;   ///< the device waits for `written` to reach it
    uint64_t pad = 0;
};

/// The records' layout, as the decode reads it: a 3DGS PLY vertex's.
[[nodiscard]] io::SplatEncoding clientRecordEncoding(uint32_t degree);

class ClientPages final : public lod::DevicePages {
public:
    struct Info {
        std::string memory;        ///< shared memory: a PagesHeader page, then the records
        uint64_t    bytes = 0;
        uint64_t    records = 0;
        uint32_t    floatsPerRecord = 0;
        uint32_t    degree = 0;
        std::string server;
        uint64_t    token = 0;     ///< the `written` fence
    };

    /// FRAME THREAD: room for `records` gaussians of `degree` (0..3).
    [[nodiscard]] static Result<std::shared_ptr<ClientPages>> create(gpu::Device& device, gpu::ShaderLibrary& library,
                                                                     platform::HandleServer& server, uint64_t records,
                                                                     uint32_t degree);
    ~ClientPages() override;

    /// Any thread.
    [[nodiscard]] const Info& info() const noexcept { return info_; }
    [[nodiscard]] uint64_t key() const noexcept { return key_; }
    /// ANY THREAD: the bytes of a page that places records `first`..`first +
    /// count` once `written` reaches `fenceValue` -- what a commit pushes as
    /// the page. InvalidArgument for records outside the region.
    [[nodiscard]] Result<std::vector<std::byte>> page(uint64_t ticket, uint32_t chunk, uint64_t first, uint32_t count,
                                                      uint64_t fenceValue, bool antialiased);

    /// ANY THREAD: `pages` device pages of ticket `ticket` were queued (a
    /// commit that went through), so `consumed` waits for all of them.
    void committed(uint64_t ticket, uint32_t pages);

    /// Pool's frame thread (lod::DevicePages).
    [[nodiscard]] Result<void> place(gpu::CommandBatch& batch, const lod::DevicePageHeader& page,
                                     const gpu::Buffer& staging, uint64_t payloadOffset, scene::GpuSplats& store,
                                     uint64_t at, uint32_t keepRest, const void* pool, uint64_t fence) override;
    void retired(const void* pool, uint64_t completed) override;

    struct Stats {
        uint64_t placed = 0, waits = 0, consumed = 0;
    };
    [[nodiscard]] Stats stats() const;

private:
    ClientPages() = default;
    struct Use {
        const void* pool;
        uint64_t    fence;
        uint64_t    ticket;
        size_t      scratch;
    };
    gpu::Device*                       device_ = nullptr;
    platform::HandleServer*            server_ = nullptr;
    platform::SharedMemory             memory_;
    PagesHeader*                       header_ = nullptr;
    gpu::Buffer                        records_;
    gpu::SharedFence                   written_;
    std::unique_ptr<scene::CloudLoader> loader_;
    std::vector<scene::CloudLoader::DecodeScratch> scratch_;
    std::vector<bool>                  scratchBusy_;
    std::vector<uint32_t>              scratchCapacity_;
    std::vector<gpu::Buffer>           scratchRecords_;   ///< a page's records, copied where the decode reads
    std::map<uint64_t, uint64_t>       fences_;   ///< ticket -> the fence value its pages wait for
    std::vector<Use>                   uses_;
    uint64_t                           key_ = 0;
    std::map<uint64_t, std::pair<uint32_t, uint32_t>> tickets_;   ///< ticket -> (pages committed, retired)
    mutable std::mutex                 mutex_;
    std::atomic<uint64_t>              placed_{0}, waits_{0};
    Info                               info_;
};

/// THE PRODUCER'S SIDE (another process): the records' memory as a buffer
/// on its own device, and the `written` fence it raises. Made and used on
/// the producer's device thread; `consumed` from any.
class ClientPagesView {
public:
    [[nodiscard]] static Result<ClientPagesView> open(gpu::Device& device, const ClientPages::Info& info);
    [[nodiscard]] const gpu::Buffer& records() const noexcept { return records_; }
    [[nodiscard]] gpu::SharedFence& written() noexcept { return written_; }
    /// The newest commit ticket (epoch) the engine is done with.
    [[nodiscard]] uint64_t consumed() const;

private:
    platform::SharedMemory memory_;
    gpu::Buffer            records_;
    gpu::SharedFence       written_;
};

}   // namespace athenea::bridge
