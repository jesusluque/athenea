// Copyright (c) 2026 jesus luque.
//
// THE HARDWARE VIDEO CODEC, AS INPUT AND OUTPUT. The planes (Planes.h) go
// through a platform's own encoder and decoder, which are fixed-function
// blocks, like a file read: nothing here computes on the values. Only luma
// carries data; chroma is held at mid grey, so 4:2:0 -- the one sampling
// every decoder takes -- loses nothing.
//
//   VideoEncoder   luma samples in, access units out (the producer's side)
//
// WHERE EACH PLATFORM STANDS. macOS and iOS: VideoToolbox
// (src/VideoToolbox.mm, the only file here that calls a framework). Others
// get Unsupported from the factory.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "athenea/core/Result.h"

namespace athenea::video {

struct EncoderSettings {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bits = 8;           ///< 8: HEVC Main; 10: Main 10
    double   rate = 30.0;
    /// 0..1, the encoder's constant quality; ignored when `megabits` is set.
    double   quality = 0.9;
    /// An average bit rate instead of a quality, Mbit/s (0: by quality).
    double   megabits = 0.0;
    /// Frames from one key frame to the next (1: every frame a key frame).
    uint32_t keyInterval = 30;
    bool     realTime = false;
};

/// One frame out of the encoder, as `.athv` keeps it (Athv.h).
struct EncodedUnit {
    std::vector<uint8_t>              unit;            ///< length-prefixed NAL units
    std::vector<std::vector<uint8_t>> parameterSets;   ///< on a key frame
    bool                              keyframe = false;
    uint64_t                          index = 0;       ///< the frame it encodes
};

class VideoEncoder {
public:
    virtual ~VideoEncoder() = default;
    /// Frame `index`: `luma` is width x height samples, top row first (bytes,
    /// or 16-bit words with the 10 bits high as P010 holds them). Returns its
    /// unit: the encoder finishes each frame before it takes the next.
    [[nodiscard]] virtual Result<std::vector<EncodedUnit>> encode(std::span<const std::byte> luma,
                                                                  uint64_t index) = 0;
    /// What is left, once every frame has been given.
    [[nodiscard]] virtual Result<std::vector<EncodedUnit>> finish() = 0;
    /// Whether the encoder runs on the hardware block.
    [[nodiscard]] virtual bool hardware() const noexcept = 0;
};

[[nodiscard]] Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& settings);

}   // namespace athenea::video
