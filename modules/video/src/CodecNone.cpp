// Copyright (c) 2026 jesus luque.
//
// Codec.h's decoder where no hardware decoder is wired yet (the encoder is
// NVENC, src/Nvenc.cpp). NVDEC (CUDA: cuvidCreateDecoder
// on the engine's context, the frame mapped with cuvidMapVideoFrame and its
// luma adopted as a texture of the CUDA device) and WebCodecs (VideoDecoder,
// VideoFrame.copyTo plane by plane into a buffer the planes decode reads) go
// here; the planes, the container and the live path above them do not change.
#include "athenea/video/Codec.h"

namespace athenea::video {

Result<std::unique_ptr<VideoDecoder>> createHardwareDecoder(uint32_t /*codec*/, uint32_t /*bits*/) {
    return Error(ErrorCode::Unsupported, "video: no hardware decoder on this platform yet (NVDEC is next)");
}

Result<std::unique_ptr<LumaTextures>> createLumaTextures(gpu::Device& /*device*/) {
    return Error(ErrorCode::Unsupported, "video: decoded frames as textures need a hardware decoder");
}

}   // namespace athenea::video
