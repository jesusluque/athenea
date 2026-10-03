// Copyright (c) 2026 jesus luque.
#pragma once

#include <cstddef>
#include <cstdint>
#include <source_location>
#include <span>
#include <string>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {

class Device;

enum class Memory {
    Device,     ///< GPU-only. The default for everything a kernel writes.
    Upload,     ///< CPU writes, GPU reads.
    Readback,   ///< GPU writes (by copy), CPU reads.
};

struct BufferDesc {
    uint64_t    bytes = 0;
    /// Bytes per element for a structured buffer. Zero for raw bytes.
    uint32_t    elementBytes = 0;
    Memory      memory = Memory::Device;
    /// Add vertex/index/indirect/AS-input usage on top of storage + copy.
    rhi::BufferUsage extraUsage = rhi::BufferUsage::None;
    std::string label;
};

/// A device buffer. Owns its slang-rhi object; cheap to copy (ref-counted).
class Buffer {
public:
    Buffer() = default;

    [[nodiscard]] static Result<Buffer> create(Device& device, const BufferDesc& desc,
                                               const void* initial = nullptr);

    /// Wraps a buffer somebody else allocated on the same device -- a gpe
    /// buffer, found by its MTLBuffer* or CUDA pointer. The owner keeps it.
    [[nodiscard]] static Result<Buffer> wrap(Device& device, rhi::NativeHandle handle,
                                             const BufferDesc& desc);

    template <typename T>
    [[nodiscard]] static Result<Buffer> fromSpan(Device& device, std::span<const T> data,
                                                 std::string label,
                                                 Memory memory = Memory::Device) {
        BufferDesc desc;
        desc.bytes = data.size_bytes();
        desc.elementBytes = sizeof(T);
        desc.memory = memory;
        desc.label = std::move(label);
        return create(device, desc, data.data());
    }

    [[nodiscard]] bool valid() const noexcept { return buffer_ != nullptr; }
    [[nodiscard]] uint64_t bytes() const noexcept { return bytes_; }
    [[nodiscard]] uint32_t elementBytes() const noexcept { return elementBytes_; }
    [[nodiscard]] uint64_t count() const noexcept {
        return elementBytes_ != 0 ? bytes_ / elementBytes_ : bytes_;
    }
    [[nodiscard]] rhi::IBuffer* rhi() const noexcept { return buffer_.get(); }
    [[nodiscard]] rhi::NativeHandle native() const;

    /// Synchronous readback, for the CLI's final image and for tests' metric
    /// counters. Never on a frame path: see gpu::AsyncReadback. One host
    /// wait, counted at `where`; several buffers wanted at once are
    /// gpu::readBuffers, which waits once for all of them.
    [[nodiscard]] Result<void> read(Device& device, uint64_t offset, uint64_t bytes, void* into,
                                    std::source_location where = std::source_location::current()) const;

    template <typename T>
    [[nodiscard]] Result<std::vector<T>> readAll(Device& device,
                                                 std::source_location where = std::source_location::current()) const {
        std::vector<T> out(bytes_ / sizeof(T));
        ATHENEA_TRY(read(device, 0, out.size() * sizeof(T), out.data(), where));
        return out;
    }

    /// Synchronous upload into a device buffer.
    [[nodiscard]] Result<void> write(Device& device, uint64_t offset, uint64_t bytes,
                                     const void* from);

private:
    rhi::ComPtr<rhi::IBuffer> buffer_;
    uint64_t                  bytes_ = 0;
    uint32_t                  elementBytes_ = 0;
};

/// One region of a device buffer for gpu::readBuffers to bring back.
struct ReadRequest {
    const Buffer* buffer = nullptr;
    uint64_t      offset = 0;
    uint64_t      bytes = 0;
    void*         into = nullptr;
};

/// Several readbacks for the price of one host wait: each region is copied
/// into one readback buffer, the copies submitted together and waited for
/// once, and the buffer mapped. What a frame's image and depth come back by.
[[nodiscard]] Result<void> readBuffers(Device& device, std::span<const ReadRequest> requests,
                                       std::source_location where = std::source_location::current());

}   // namespace athenea::gpu
