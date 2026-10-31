// Copyright (c) 2026 jesus luque.
//
// A .athv's frames decoded in order into a store of splats on the device --
// video-decode's loop, for another command that wants the gaussians a video
// carries (vdb2splat --from-athv). Frames go in order: a geometry layer's
// frames are predicted from the ones before.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/scene/GpuClouds.h"

namespace athenea::gpu {
class Device;
class ShaderLibrary;
}   // namespace athenea::gpu

namespace athenea::cli {

class AthvFrames {
public:
    [[nodiscard]] static Result<std::unique_ptr<AthvFrames>> open(gpu::Device& device, gpu::ShaderLibrary& library,
                                                                  const std::string& path);
    virtual ~AthvFrames() = default;
    /// The next source frame, every picture of it, into `store()`; its splats.
    [[nodiscard]] virtual Result<uint32_t> next() = 0;
    [[nodiscard]] virtual const scene::GpuSplats& store() const = 0;
    [[nodiscard]] virtual uint32_t restPerColour() const = 0;
    [[nodiscard]] virtual uint32_t sourceFrames() const = 0;
    /// The file's bytes over its frames, and the median of the hardware
    /// decode a picture so far (ms).
    [[nodiscard]] virtual double bytesPerFrame() const = 0;
    [[nodiscard]] virtual double decodeMs() const = 0;
};

}   // namespace athenea::cli
