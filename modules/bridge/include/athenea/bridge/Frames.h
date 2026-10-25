// Copyright (c) 2026 jesus luque.
//
// A FRAME HANDED TO ANOTHER PROCESS AS IT LIES ON THE DEVICE (render_subscribe).
//
// The engine draws a frame, a kernel writes the AOVs a subscriber named into
// one slot of a ring of N, and a fence the subscriber shares rises to the
// frame's number when the device has written them. The subscriber reads the
// slot where it lies -- its own device over the same memory -- and says it is
// done with a frame by raising a second fence (or a word, for a reader with
// no device). The engine reuses a slot only once the subscriber is done with
// the frame it held: the device waits for that (fence mode), or the frame is
// not published (poll mode). Nobody's host waits for anybody's device.
//
//   the ring      shared memory (platform::SharedMemory): a RingHeader, the
//                 AOVs' layout, and a record a frame -- which slot, the
//                 instant the frame stands for, the epoch it showed, the
//                 camera it was drawn from. Written by the engine's host when
//                 it records the frame, read by the subscriber's host after
//                 the frame's fence has risen.
//   the slots     Metal: a second shared memory region, each AOV of each
//                 slot on whole pages, so either side makes a buffer over its
//                 pages (gpu::Buffer::overSharedPages) and reads it in place.
//                 Vulkan: one exportable device buffer (an opaque descriptor),
//                 the AOVs at offsets. CUDA: not done (a device pointer's IPC
//                 handle is where it goes).
//   the fences    `frames` (the engine raises it to F once frame F is in its
//                 slot) and `release` (the subscriber raises it to F once it
//                 is done with every frame up to F), handed out by a
//                 platform::HandleServer under a token the answer carries.
//
// THREADS. FramePublisher: create, publish, observe -- the frame thread,
// which is the device's one caller (refused elsewhere, FrameThread);
// stats, info, unblock -- any thread. FrameSubscriber: open, release --
// the subscriber's own device thread; completed, record, waitFrame -- any.
#pragma once

#include <atomic>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "athenea/core/FrameThread.h"
#include "athenea/core/Platform.h"
#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/SharedFence.h"
#include "athenea/gpu/Texture.h"

namespace athenea::gpu {
class CommandBatch;
class Device;
class ShaderLibrary;
}   // namespace athenea::gpu

namespace athenea::technique {
struct DisplaySource;
class DisplayTransform;
}   // namespace athenea::technique

namespace athenea::bridge {

/// How an AOV lies in a slot.
enum class Format : uint32_t {
    R32F = 1,      ///< one float a pixel (depth, alpha)
    R16F = 2,      ///< one half a pixel
    RGBA32F = 3,   ///< four floats (colour, normals, a Cryptomatte layer)
    RGBA16F = 4,   ///< four halves
    R32U = 5,      ///< one 32-bit id (primId, instanceId, elementId)
    RGBA8 = 6,     ///< the colour as shown: the display transform, sRGB, top row first
};
[[nodiscard]] const char* formatName(Format format) noexcept;
[[nodiscard]] std::optional<Format> parseFormat(const std::string& name) noexcept;
[[nodiscard]] uint32_t bytesPerPixel(Format format) noexcept;

/// An AOV a subscriber asks for: "color", "alpha", "depth", "normal",
/// "Neye", "primId", "instanceId", "elementId", "CryptoObject00".."02".
struct AovSpec {
    std::string aov;
    Format      format = Format::RGBA32F;
};
/// Whether `format` is one `aov` can be given in; the default where `format`
/// is empty. InvalidArgument names what it takes.
[[nodiscard]] Result<Format> formatFor(const std::string& aov, const std::string& format);

inline constexpr uint32_t kMaxAovs = 8;
inline constexpr uint32_t kRecords = 64;   ///< frames' records kept, a ring
inline constexpr uint32_t kRingMagic = 0x52465441;   // "ATFR"

/// A frame's record. `frame` is written last (release) and read first
/// (acquire); a reader checks it again after the rest.
struct RingRecord {
    uint64_t frame = 0;          ///< 0: none yet
    uint64_t slot = 0;
    uint64_t instantNs = 0;      ///< the instant the frame stands for (the live clock's timebase)
    uint64_t epoch = 0;          ///< the live epoch it showed (0 none)
    uint64_t epochPtpNs = 0;     ///< that epoch's own time
    uint64_t recordedNs = 0;     ///< system clock when the engine recorded it
    uint64_t cameraSamples = 0;  ///< 0 no queued camera, 1 one held, 2 two interpolated
    uint64_t pad0 = 0;
    double   cameraToWorld[16] = {};   ///< row by row, looking down -z
    double   focal = 0, haperture = 0, vaperture = 0, windowX = 0, windowY = 0, nearZ = 0, farZ = 0, time = 0;
};
static_assert(sizeof(RingRecord) == 256);

struct RingAov {
    char     name[32] = {};
    uint32_t format = 0;
    uint32_t bytesPerPixel = 0;
    uint64_t offset = 0;   ///< bytes from the slot's start (Metal: whole pages)
    uint64_t bytes = 0;
};

struct RingHeader {
    uint32_t   magic = kRingMagic;
    uint32_t   version = 1;
    uint32_t   backend = 0;     ///< gpu::Backend
    uint32_t   slots = 0;
    uint32_t   width = 0;
    uint32_t   height = 0;
    uint32_t   aovCount = 0;
    uint32_t   fenceSync = 0;   ///< 1: the device waits on `release`; 0: poll `released`
    uint64_t   slotBytes = 0;
    uint64_t   dataBytes = 0;
    uint64_t   published = 0;   ///< the newest frame recorded (engine's host)
    uint64_t   completed = 0;   ///< the newest frame seen done (engine's host, a frame late at most)
    uint64_t   released = 0;    ///< poll mode: the newest frame the subscriber is done with (its host)
    uint64_t   stateOffset = 0; ///< a slot's FrameState, bytes from the slot's start

    RingAov    aovs[kMaxAovs];
    RingRecord records[kRecords];
};

/// WHAT THE DEVICE SAYS OF THE FRAME IN A SLOT, written by the device with
/// the AOVs: the splat rasteriser's totals and the pairs its buffers held
/// (a frame whose pairs passed them was drawn short, once, while they grew),
/// and 1 in `written`. Zeros for a frame of no splats.
struct FrameState {
    uint32_t visible = 0;
    uint32_t pairs = 0;
    uint32_t pairCapacity = 0;
    uint32_t written = 0;
    [[nodiscard]] bool whole() const noexcept { return pairs <= pairCapacity; }
};

// The layout a reader without this header (the Python example) reads by offset.
static_assert(sizeof(RingAov) == 56);
static_assert(offsetof(RingHeader, published) == 48 && offsetof(RingHeader, completed) == 56 &&
              offsetof(RingHeader, released) == 64 && offsetof(RingHeader, aovs) == 80 &&
              offsetof(RingHeader, records) == 528);

/// WHAT A SUBSCRIBER NEEDS TO OPEN THE FRAMES: the MCP answer's JSON.
struct SubscriptionInfo {
    std::string ring;           ///< shared memory: the RingHeader
    uint64_t    ringBytes = 0;
    std::string data;           ///< Metal: shared memory holding the slots
    uint64_t    dataBytes = 0;
    std::string server;         ///< platform::HandleServer
    uint64_t    token = 0;      ///< the handles: frames fence, release fence (and the data's descriptor)
    std::string backend;
    uint32_t    slots = 0, width = 0, height = 0;
    bool        fenceSync = true;
    uint64_t    slotBytes = 0;
    uint64_t    stateOffset = 0;    ///< a slot's FrameState, bytes from its start
    struct Aov {
        std::string name;
        Format      format = Format::RGBA32F;
        uint64_t    offset = 0, bytes = 0;
    };
    std::vector<Aov> aovs;

    [[nodiscard]] std::string toJson() const;
    [[nodiscard]] static Result<SubscriptionInfo> fromJson(const std::string& text);
};

/// A frame's AOV as the engine holds it (technique::DisplaySource's words).
struct AovSource {
    const gpu::Buffer* buffer = nullptr;
    uint32_t           stride = 1;    ///< words a pixel
    uint32_t           offset = 0;    ///< the AOV's first word
    /// For RGBA8, the source the display transform reads (colour).
    const technique::DisplaySource* shown = nullptr;
};

/// The splat totals a frame in flight left on the device, and the pairs its
/// buffers held (usd::StageRenderer::splatFrameTotals).
struct FrameTotals {
    const gpu::Buffer* totals = nullptr;
    uint32_t           pairCapacity = 0;
};

struct PublisherDesc {
    uint32_t             width = 0, height = 0;
    uint32_t             slots = 3;
    std::vector<AovSpec> aovs;
    bool                 fenceSync = true;
};

class FramePublisher {
public:
    struct Stats {
        uint64_t published = 0;   ///< frames written into a slot
        uint64_t skipped = 0;     ///< poll mode: frames not published, their slot not yet released
        uint64_t completed = 0;   ///< the newest frame the device has written
        uint64_t unblocked = 0;   ///< times a stalled subscriber's release was raised for it
    };

    /// FRAME THREAD: the ring, the slots and the fences, offered on `server`.
    [[nodiscard]] static Result<std::unique_ptr<FramePublisher>> create(gpu::Device& device,
                                                                        gpu::ShaderLibrary& library,
                                                                        platform::HandleServer& server,
                                                                        const PublisherDesc& desc);
    ~FramePublisher();

    /// Any thread.
    [[nodiscard]] const SubscriptionInfo& info() const noexcept { return info_; }
    [[nodiscard]] Stats stats() const;

    /// FRAME THREAD: frame `frame` (from 1, rising) written into its slot
    /// from `sources` (one an AOV, in the order asked), its record kept, and
    /// the frames fence raised to it once the device has run that -- all
    /// after the work already on the queue (the frame's draw). Fence mode:
    /// the device first waits for the subscriber to release the slot's last
    /// frame. Poll mode: false, and nothing recorded, where it has not.
    /// Never waits on the host.
    [[nodiscard]] Result<bool> publish(uint64_t frame, std::span<const AovSource> sources, const RingRecord& record,
                                       const FrameTotals& totals = {});
    /// FRAME THREAD: the ring's `completed` word brought up to the frames
    /// fence (read, never waited for). Answers the frames recorded and not
    /// yet written: what a frame loop holds its pace by.
    uint64_t observe();
    /// Any thread: raises the release fence to the last frame recorded, for
    /// a subscriber that has gone or stopped releasing: a device waiting on
    /// it goes on. Counted.
    void unblock();

private:
    FramePublisher() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SubscriptionInfo      info_;
};

class FrameSubscriber {
public:
    /// THE SUBSCRIBER'S DEVICE THREAD: the ring mapped, the slots made
    /// buffers on `device` over the same memory, the fences imported.
    [[nodiscard]] static Result<std::unique_ptr<FrameSubscriber>> open(gpu::Device& device,
                                                                      const SubscriptionInfo& info);
    ~FrameSubscriber();

    /// The newest frame whose AOVs are written: the frames fence (fence
    /// mode) or the ring's `completed` word. Any thread, never waits.
    [[nodiscard]] uint64_t completed() const;
    /// The record of `frame`, if the ring still holds it. Any thread.
    [[nodiscard]] std::optional<RingRecord> record(uint64_t frame) const;
    /// THE SUBSCRIBER'S HOST WAITS (it may; the engine never does) for a
    /// frame at least `atLeast`, polling the fence. Any thread.
    [[nodiscard]] Result<uint64_t> waitFrame(uint64_t atLeast, std::chrono::milliseconds timeout) const;
    /// AOV `index` of slot `slot`: the buffer, and where in it the AOV
    /// starts (0 on Metal, where each is a buffer of its own). Any thread.
    [[nodiscard]] const gpu::Buffer& aov(uint32_t slot, uint32_t index) const;
    [[nodiscard]] uint64_t aovOffset(uint32_t slot, uint32_t index) const;
    [[nodiscard]] const SubscriptionInfo& info() const noexcept { return info_; }
    [[nodiscard]] const RingHeader& header() const noexcept;
    /// What the device wrote of the frame in `slot` beside its AOVs. Read
    /// after the frame's fence has risen. Any thread.
    [[nodiscard]] FrameState state(uint32_t slot) const;

    /// THE SUBSCRIBER'S DEVICE THREAD: done with every frame up to `frame`
    /// once `batch` (its reads) has run -- submitted here, and the release
    /// raised by the device after it (fence mode), or the word written now
    /// (poll mode: call after the reads are done).
    [[nodiscard]] Result<void> release(gpu::CommandBatch& batch, uint64_t frame);

private:
    FrameSubscriber() = default;
    struct Impl;
    std::unique_ptr<Impl> impl_;
    SubscriptionInfo      info_;
};

}   // namespace athenea::bridge
