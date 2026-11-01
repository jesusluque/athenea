// Copyright (c) 2026 jesus luque.
//
// Codec.h where a platform has no hardware codec wired: every factory says
// Unsupported. Apple builds VideoToolbox (src/VideoToolbox.mm), Linux and
// Windows NVENC and NVDEC (src/Nvenc.cpp, src/Nvdec.cpp); WebCodecs
// (VideoDecoder, VideoFrame.copyTo plane by plane into a buffer the planes
// decode reads) would go here, and the planes, the container and the live
// path above them would not change.
#include "athenea/video/Codec.h"

namespace athenea::video {

Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& /*settings*/) {
    return Error(ErrorCode::Unsupported, "video: no hardware encoder on this platform");
}

Result<std::unique_ptr<VideoDecoder>> createHardwareDecoder(uint32_t /*codec*/, uint32_t /*bits*/,
                                                            gpu::Device* /*device*/) {
    return Error(ErrorCode::Unsupported, "video: no hardware decoder on this platform");
}

Result<std::unique_ptr<LumaTextures>> createLumaTextures(gpu::Device& /*device*/) {
    return Error(ErrorCode::Unsupported, "video: decoded frames as textures need a hardware decoder");
}

}   // namespace athenea::video
