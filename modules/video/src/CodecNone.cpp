// Copyright (c) 2026 jesus luque.
//
// Codec.h where no hardware encoder is wired yet (NVENC on CUDA is where one
// goes); the planes and the container above it do not change.
#include "athenea/video/Codec.h"

namespace athenea::video {

Result<std::unique_ptr<VideoEncoder>> createHardwareEncoder(const EncoderSettings& /*settings*/) {
    return Error(ErrorCode::Unsupported, "video: no hardware encoder on this platform yet");
}

}   // namespace athenea::video
