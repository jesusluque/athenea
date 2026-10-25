// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/GpuTimer.h"

#include <chrono>

#include <slang-rhi.h>

#include "athenea/core/Platform.h"
#include "athenea/gpu/CommandBatch.h"

namespace athenea::gpu {
namespace {

void SLANG_MCALL keepCommandBuffer(const rhi::ExecuteCallbackContext* context, void* user, const void*, rhi::Size) {
    auto** slot = static_cast<void**>(user);
    if (context != nullptr && context->nativeHandle.type == rhi::NativeHandleType::MTLCommandBuffer) {
        void* buffer = reinterpret_cast<void*>(context->nativeHandle.value);
        platform::retainObject(buffer);
        platform::releaseObject(*slot);
        *slot = buffer;
    }
}

}   // namespace

GpuTimer::~GpuTimer() {
    platform::releaseObject(commandBuffer_);
}

Result<void> GpuTimer::run(CommandBatch& batch) {
    platform::releaseObject(commandBuffer_);
    commandBuffer_ = nullptr;
    rhi::ExecuteCallbackDesc desc;
    desc.callback = keepCommandBuffer;
    desc.userObject = &commandBuffer_;
    batch.encoder()->executeCallback(desc);
    batch.markDirty();
    const auto start = std::chrono::steady_clock::now();
    ATHENEA_TRY(batch.submit(true));
    const double host = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const double device = platform::metalCommandBufferMilliseconds(commandBuffer_);
    fromDevice_ = device >= 0.0;
    milliseconds_ = fromDevice_ ? device : host;
    return ok();
}

}   // namespace athenea::gpu
