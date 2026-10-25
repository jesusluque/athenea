// Copyright (c) 2026 jesus luque.
//
// A `.athv` VIDEO PLAYED INTO A LIVE STREAM. Each video frame is one epoch of
// the stream a stage names (AtheneaLiveSourceAPI, lod/SplatStream.h): the
// producer reads the frame's record, has the hardware decoder decode it, and
// pushes a device page a chunk (lod/DevicePages.h) -- a header, the frame's
// ranges and the chunk's position high parts. The pool's frame thread then
// asks this stream to place each page: the decoded luma plane is adopted as
// a texture where the decoder left it, and video_planes_decode.slang writes
// the chunk's splats into the store. Nothing is copied off the decoder's
// buffer and nothing is computed on the processor.
//
// THE THREADS. push: one producer thread at a time (the reader and the
// decoder are its own). place, retired: the pool's frame thread. A decoded
// frame is held until every pool that placed its pages has retired them;
// one whose epoch was dropped goes when 16 newer frames have come.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "athenea/core/Result.h"
#include "athenea/lod/DevicePages.h"
#include "athenea/video/Athv.h"
#include "athenea/video/Codec.h"
#include "athenea/video/Planes.h"

namespace athenea::gpu {
class ShaderLibrary;
}

namespace athenea::lod {
class PageQueue;
}

namespace athenea::video {

class VideoStream final : public lod::DevicePages {
public:
    /// The file opened, its decoder and the device's texture cache made, the
    /// kernels compiled, and the stream published to the process's pools.
    /// The thread that will draw (it compiles kernels on `library`).
    [[nodiscard]] static Result<std::shared_ptr<VideoStream>> open(const std::filesystem::path& path,
                                                                  gpu::ShaderLibrary& library);
    ~VideoStream() override;
    /// No more pages placed: the stream withdrawn from the pools.
    void close();

    [[nodiscard]] const AthvHeader& header() const noexcept { return reader_->header(); }
    /// Source frames: an epoch each, of one picture or of several slices.
    [[nodiscard]] uint32_t frames() const noexcept { return reader_->sourceFrames(); }
    [[nodiscard]] uint64_t fileBytes() const noexcept { return reader_->bytes(); }
    [[nodiscard]] double time(uint32_t k) const;
    /// What frame `k`'s pictures take in the file.
    [[nodiscard]] uint64_t frameBytes(uint32_t k) const;
    /// Chunks the largest frame has: the pages of every epoch.
    [[nodiscard]] uint32_t chunks() const noexcept;

    /// PRODUCER: frame `k`'s pictures decoded and pushed as epoch `epoch` of
    /// `stream`, `pages` pages in all (chunks past the frame's own are
    /// removed).
    [[nodiscard]] Result<void> push(lod::PageQueue& queue, uint32_t stream, uint32_t k, uint64_t epoch,
                                    uint64_t ptpTime, uint32_t pages);

    struct Stats {
        std::vector<double> readMs;     ///< a frame's records read and their high parts inflated
        std::vector<double> decodeMs;   ///< the hardware decode of a frame's pictures
        uint64_t            placed = 0; ///< pages placed
        uint64_t            held = 0;   ///< frames held now
        bool                hardware = false;   ///< the decoder runs on the hardware block
    };
    [[nodiscard]] Stats stats() const;

    [[nodiscard]] Result<void> place(gpu::CommandBatch& batch, const lod::DevicePageHeader& page,
                                     const gpu::Buffer& staging, uint64_t payloadOffset, scene::GpuSplats& store,
                                     uint64_t at, uint32_t keepRest, const void* pool, uint64_t fence) override;
    void retired(const void* pool, uint64_t completed) override;

private:
    VideoStream() = default;
    struct Held {
        std::shared_ptr<DecodedFrame>              frame;
        LumaTexture                                luma;     ///< made by the frame thread when first placed
        PlaneLayout                                layout;
        uint32_t                                   pagesLeft = 0;
        std::vector<std::pair<const void*, uint64_t>> uses;  ///< (pool, fence) not yet retired
    };
    void evictLocked();

    std::unique_ptr<AthvReader>    reader_;
    std::unique_ptr<VideoDecoder>  decoder_;
    std::unique_ptr<LumaTextures>  textures_;
    std::unique_ptr<PlanesCodec>   codec_;
    uint64_t                       key_ = 0;
    mutable std::mutex             mutex_;
    std::map<uint64_t, Held>       held_;
    uint64_t                       nextTicket_ = 1;
    Stats                          stats_;
};

}   // namespace athenea::video
