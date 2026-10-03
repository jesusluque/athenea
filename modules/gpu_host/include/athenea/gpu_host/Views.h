// Copyright (c) 2026 jesus luque.
//
// The crossing between the two runtimes, as free functions.
//
// gpu_host::Context is this engine's own arrangement: its thread, its pool,
// its image storage. A program that has all three already -- the compositor
// this engine is built into does -- wants none of that and exactly this: gpe
// adopting the device slang-rhi opened, a slang-rhi view of a gpe buffer, a
// gpe handle on a slang-rhi buffer. So the three live here, on the bare
// objects, and Context's methods of the same names call them. One
// implementation of the crossing, whichever context is around it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "athenea/core/Result.h"
#include "athenea/gpu/Buffer.h"
#include "athenea/gpu/Device.h"

namespace gpe {
class Device;
class PooledDevice;
}   // namespace gpe

namespace athenea::gpu_host {

/// gpe on slang-rhi's device -- the same MTLDevice and command queue, or the
/// same CUDA context and stream -- or null where gpe has no backend for it
/// (Vulkan). The caller keeps `device` alive for as long as what this returns.
[[nodiscard]] std::unique_ptr<gpe::Device> adoptForCompute(gpu::Device& device);

/// A slang-rhi view of a gpe buffer: the same memory, bound by name. `bytes`
/// and `elementBytes` are what the shader will see; a view of part of a
/// buffer is a view of its first bytes.
[[nodiscard]] Result<gpu::Buffer> renderView(gpu::Device& device, gpe::PooledDevice& compute,
                                             uint64_t gpeBuffer, size_t bytes,
                                             uint32_t elementBytes, std::string label = {});

/// A gpe handle on a slang-rhi buffer. gpe claims the memory, never owns it:
/// `compute.release(handle)` before the buffer goes, or gpe holds a pointer to
/// freed memory and the next kernel that reads it draws whatever is there.
[[nodiscard]] Result<uint64_t> computeView(gpe::PooledDevice& compute, const gpu::Buffer& buffer);

}   // namespace athenea::gpu_host
