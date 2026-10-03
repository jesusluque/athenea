// Copyright (c) 2026 jesus luque.
//
// The render device: slang-rhi, one per process.
//
// WHY SLANG-RHI AND NOT GPE FOR RENDERING
//
// gpe is a compute engine with kernels compiled at build time and bound by
// position. That is right for image effects and wrong for a renderer that
// needs a rasterisation pipeline for points, acceleration structures for ray
// tracing, and bindings a person cannot get wrong by counting. slang-rhi gives
// all three on Metal, CUDA (with OptiX) and Vulkan, from one Slang source
// compiled for whichever device the process opened.
//
// gpe is not replaced: it adopts this device (see athenea::gpu_host::RhiBridge),
// so the two share memory and nothing crosses between them.
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <slang-rhi.h>

#include "athenea/core/Result.h"

namespace athenea::gpu {

enum class Backend { Metal, CUDA, Vulkan, D3D12 };

[[nodiscard]] const char* toString(Backend) noexcept;

struct DeviceDesc {
    /// Empty means the platform's preference: Metal on Apple, CUDA then Vulkan
    /// on Linux, D3D12 then Vulkan on Windows. There is no CPU entry and there
    /// never will be.
    std::vector<Backend> backends;
    /// slang-rhi's own validation layer. Slow; for tests and debugging.
    bool validation = false;
    /// Extra directories searched for `import`ed Slang modules, after the
    /// engine's own shader directory.
    std::vector<std::filesystem::path> shaderPaths;
    /// Compiled shaders kept on disk between runs. Empty: $ATHENEA_SHADER_CACHE,
    /// else athenea/shaders in the platform's cache directory.
    std::filesystem::path shaderCache;
    bool                  useShaderCache = true;
};

/// What the persistent shader cache did since the device opened.
struct ShaderCacheStats {
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t writes = 0;
};

/// What the device can do, asked once at creation.
struct Caps {
    std::string apiName;
    std::string adapterName;
    bool rasterization = false;
    bool rayTracing = false;       ///< ray tracing pipelines (shader tables)
    bool rayQuery = false;         ///< inline RayQuery in compute
    bool accelerationStructure = false;
    bool timestampQuery = false;
    bool half = false;
    bool unifiedMemory = false;
    /// SV_VertexID and SV_InstanceID already count from a draw's start
    /// locations (Metal's vertex_id and instance_id). Elsewhere Slang
    /// subtracts them, and a shader adds SV_Start*Location back.
    bool drawIdsIncludeStart = false;
    /// A float4 stored through an RWTexture2D arrives as the texture's
    /// format. CUDA's surface store writes the float's own bytes instead
    /// (tests/gpu/test_textures.cpp measures it), so where this is false a
    /// kernel packs the texels into a buffer and the buffer is copied into
    /// the texture.
    bool convertingStores = false;
    uint32_t optixVersion = 0;
};

/// The native objects behind the device, for adopting it elsewhere (gpe).
struct NativeHandles {
    rhi::NativeHandle device;    ///< MTLDevice | CUcontext | VkDevice
    rhi::NativeHandle queue;     ///< MTLCommandQueue | CUstream | VkQueue
};

class Device {
public:
    [[nodiscard]] static Result<std::shared_ptr<Device>> create(const DeviceDesc& desc = {});

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    [[nodiscard]] Backend backend() const noexcept { return backend_; }
    [[nodiscard]] const Caps& caps() const noexcept { return caps_; }
    [[nodiscard]] NativeHandles native() const;
    /// All zero when the device runs without a shader cache.
    [[nodiscard]] ShaderCacheStats shaderCacheStats() const;

    [[nodiscard]] rhi::IDevice* rhi() const noexcept { return device_.get(); }
    [[nodiscard]] rhi::ICommandQueue* queue() const noexcept { return queue_.get(); }
    [[nodiscard]] slang::ISession* slangSession() const noexcept { return session_.get(); }

    /// The directories Slang searches, in order, engine shaders first.
    [[nodiscard]] const std::vector<std::string>& shaderSearchPaths() const noexcept {
        return searchPaths_;
    }

    /// Blocks until everything submitted has run. A command buffer that
    /// failed meanwhile is not lost: the next CommandBatch::submit reports it
    /// (takeQueueError).
    void waitIdle();

    /// waitIdle, and what was freed actually given back: on Metal slang-rhi
    /// keeps a freed buffer in the device's residency set -- still resident,
    /// still counted -- until the next submit commits the set, so an empty
    /// one is submitted and waited for. What a caller that has just let go
    /// of memory to make room does before it asks again.
    void releaseFreed();

    /// THE DEVICE'S MEMORY BUDGET: how many bytes this process may keep on
    /// the device before it starts to hurt. $ATHENEA_GPU_BUDGET in MiB where
    /// it is set, else Metal's recommended working set
    /// (`recommendedMaxWorkingSetSize`); `setMemoryBudget` replaces either.
    /// 0 where nothing says (CUDA and Vulkan without the variable), and then
    /// nothing is held to one.
    [[nodiscard]] uint64_t memoryBudget() const noexcept { return memoryBudget_.load(); }
    /// What the device holds for this process now (Metal's
    /// `currentAllocatedSize`); 0 where the backend does not say.
    [[nodiscard]] uint64_t memoryInUse() const;
    /// The budget less what is in use, and on unified memory no more than
    /// systemHeadroom; UINT64_MAX where neither says.
    [[nodiscard]] uint64_t memoryAvailable() const;
    /// On unified memory (Apple silicon), the physical memory the system has
    /// free now less 1.5 GiB kept for the rest of the machine: the budget
    /// above is the device's, this is the machine's, and an allocation must
    /// fit both. UINT64_MAX on a discrete device or where it cannot be read.
    [[nodiscard]] uint64_t systemHeadroom() const;
    /// A budget of the caller's: a host that shares the device, or a test
    /// that wants to run out. 0 lifts it.
    void setMemoryBudget(uint64_t bytes) noexcept { memoryBudget_.store(bytes); }
    /// OutOfMemory when `bytes` more would take the device past its budget:
    /// asked before an allocation, so a request that cannot fit fails as a
    /// Result rather than as a command buffer the device refuses later.
    [[nodiscard]] Result<void> admit(uint64_t bytes, std::string_view what) const;

    /// A slang-rhi result from the queue as a Result: SLANG_E_OUT_OF_MEMORY
    /// (a command buffer the device ran out of memory on, with the patch in
    /// cmake/patches) is OutOfMemory, any other failure DeviceFailure.
    [[nodiscard]] static Result<void> queueResult(SlangResult result, std::string_view what);
    /// A failure waitIdle met and could not return, once.
    [[nodiscard]] Result<void> takeQueueError();

    /// Called before this device submits anything to its queue.
    ///
    /// For a second runtime sharing the queue (gpe, see gpu_host::Context)
    /// that holds work back in batches: it hands its batch over here, so what
    /// the renderer submits next runs after it -- the order the calls were
    /// made in, with nothing for a caller to remember.
    void setBeforeSubmit(std::function<void()> hook) { beforeSubmit_ = std::move(hook); }
    void beforeSubmit() const {
        if (beforeSubmit_) {
            beforeSubmit_();
        }
    }

private:
    Device() = default;

    Backend                         backend_ = Backend::Metal;
    Caps                            caps_;
    std::vector<std::string>        searchPaths_;
    rhi::ComPtr<rhi::IPersistentCache> shaderCache_;
    rhi::ComPtr<rhi::IDevice>       device_;
    rhi::ComPtr<rhi::ICommandQueue> queue_;
    rhi::ComPtr<slang::ISession>    session_;
    std::function<void()>           beforeSubmit_;
    std::atomic<uint64_t>           memoryBudget_{0};
    std::mutex                      queueErrorGuard_;
    std::optional<Error>            queueError_;
};

/// Where the engine's own .slang files are: $ATHENEA_SHADER_DIR, then a
/// `shaders` directory found from the image this code is linked into
/// (platform::moduleDir), then from the executable, then the build tree.
/// The image comes first so a plugin loaded by another program (hdAthenea
/// in usdview or Blender) finds its own files and not the host's.
[[nodiscard]] std::filesystem::path shaderDirectory();

/// The first `shaders` directory holding the engine's tree (`athenea/`) at
/// one of `starts` or up to three directories above it, in the order given:
/// <exe>/../shaders for a program in bin/, and
/// <plugin>/hdAthenea/../../../shaders for the plugin in a build tree.
/// Empty if there is none.
[[nodiscard]] std::filesystem::path findShaderDirectory(std::span<const std::filesystem::path> starts);

}   // namespace athenea::gpu
