// Copyright (c) 2026 jesus luque.
//
// THE HARDWARE VIDEO CODEC, AS INPUT AND OUTPUT. The planes (Planes.h) go
// through a platform's own encoder and decoder, which are fixed-function
// blocks, like a file read: nothing here computes on the values. Only luma
// carries data; chroma is held at mid grey, so 4:2:0 -- the one sampling
// every decoder takes -- loses nothing.
//
//   VideoEncoder   luma samples in, access units out (the producer's side)
//   VideoDecoder   access units in, frames out where the device can read
//                  them: a DecodedFrame is the decoder's own buffer
//   LumaTextures   a decoded frame's luma plane as a texture of the engine's
//                  device, without a copy (frame thread)
//
// WHERE EACH PLATFORM STANDS. macOS and iOS: VideoToolbox, the frame an
// IOSurface the Metal device reads through CVMetalTextureCache
// (src/VideoToolbox.mm, the only file here that calls a framework). Others
// get Unsupported from the factories: NVDEC (CUDA: the frame in device
// memory, wrapped the way gpu_host wraps buffers) and WebCodecs (a
// VideoFrame copied plane by plane -- GPUExternalTexture converts to RGB,
// which would alter the values) are where they go.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {
class Device;
}

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
    /// or 16-bit words with the 10 bits high as P010 holds them); `chroma`
    /// the second plane as NV12 holds it ((Cb, Cr) byte pairs, width x
    /// height / 2 bytes, 8 bits only), or empty for mid grey. Returns its
    /// unit: the encoder finishes each frame before it takes the next.
    [[nodiscard]] virtual Result<std::vector<EncodedUnit>> encode(std::span<const std::byte> luma,
                                                                  std::span<const std::byte> chroma,
                                                                  uint64_t index) = 0;
    /// What is left, once every frame has been given.
    [[nodiscard]] virtual Result<std::vector<EncodedUnit>> finish() = 0;
    /// Whether the encoder runs on the hardware block.
    [[nodiscard]] virtual bool hardware() const noexcept = 0;
};

[[nodiscard]] Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& settings);

/// A frame the decoder wrote, in its own memory: held as long as this is.
class DecodedFrame {
public:
    virtual ~DecodedFrame() = default;
    [[nodiscard]] virtual uint32_t width() const noexcept = 0;
    [[nodiscard]] virtual uint32_t height() const noexcept = 0;
    [[nodiscard]] virtual uint32_t bits() const noexcept = 0;
};

class VideoDecoder {
public:
    virtual ~VideoDecoder() = default;
    /// One access unit; `parameterSets` with a key frame (a decoder starts
    /// at one). The frame decoded, in the decoder's memory. Any one thread.
    [[nodiscard]] virtual Result<std::shared_ptr<DecodedFrame>> decode(
        std::span<const uint8_t> unit, const std::vector<std::vector<uint8_t>>& parameterSets, bool keyframe) = 0;
    [[nodiscard]] virtual bool hardware() const noexcept = 0;
};

/// `codec` the stream's FourCC ('hvc1'); `bits` 8 or 10.
[[nodiscard]] Result<std::unique_ptr<VideoDecoder>> createHardwareDecoder(uint32_t codec, uint32_t bits);

/// A decoded frame's luma plane as a texture of `device` (R8Unorm, R16Unorm
/// at 10 bits), sharing the decoder's memory; `keep` holds what the texture
/// reads alive.
struct LumaTexture {
    rhi::ComPtr<rhi::ITexture> texture;
    /// The second plane, (Cb, Cr) a 2 x 2 block, as RG8Unorm (8 bits only;
    /// null at 10).
    rhi::ComPtr<rhi::ITexture> chroma;
    std::shared_ptr<void>      keep;
};

class LumaTextures {
public:
    virtual ~LumaTextures() = default;
    /// FRAME THREAD (it makes a texture of the device).
    [[nodiscard]] virtual Result<LumaTexture> wrap(const DecodedFrame& frame) = 0;
};

[[nodiscard]] Result<std::unique_ptr<LumaTextures>> createLumaTextures(gpu::Device& device);

}   // namespace athenea::video
