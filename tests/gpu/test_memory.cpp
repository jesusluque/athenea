// Copyright (c) 2026 jesus luque.
//
// RUNNING OUT OF DEVICE MEMORY IS AN ERROR, NOT AN ABORT.
//
// `athenea view` on a 5.9 M gaussian cloud, with other jobs on the GPU, died
// inside slang-rhi: a command buffer the device ran out of memory on hit an
// assertion in its completion handler. With the patch in cmake/patches it is
// a result the next submit returns, and an allocation the device's budget
// cannot hold is refused before it is made. These are the two ends a test
// can reach without starving the other jobs on a shared GPU: the budget, and
// the result codes the queue's failures become. The engine's side -- what it
// gives up and that it keeps drawing -- is tests/usd/test_usd.cpp ("[memory]").
#include "GpuTest.h"

#include <cstdio>

using namespace athenea;

namespace {

/// Puts the device's budget back however the test leaves.
struct BudgetKept {
    gpu::Device& device;
    uint64_t     budget;
    explicit BudgetKept(gpu::Device& d) : device(d), budget(d.memoryBudget()) {}
    ~BudgetKept() { device.setMemoryBudget(budget); }
};

}   // namespace

TEST_CASE("an allocation past the device's memory budget is an OutOfMemory result", "[gpu][memory]") {
    ATHENEA_REQUIRE_GPU(gpu);
    gpu::Device& device = *gpu->device;
    const BudgetKept kept(device);
    std::printf("  budget %llu MiB, %llu MiB in use, system headroom %llu MiB\n",
                static_cast<unsigned long long>(device.memoryBudget() >> 20),
                static_cast<unsigned long long>(device.memoryInUse() >> 20),
                static_cast<unsigned long long>(device.systemHeadroom() >> 20));

    // Four MiB more than the device holds now, and a buffer of sixteen.
    device.setMemoryBudget(device.memoryInUse() + (uint64_t{4} << 20));
    gpu::BufferDesc desc;
    desc.bytes = uint64_t{16} << 20;
    desc.elementBytes = 4;
    desc.label = "test.tooBig";
    auto refused = gpu::Buffer::create(device, desc);
    REQUIRE_FALSE(refused);
    CHECK(refused.error().code() == ErrorCode::OutOfMemory);
    CHECK(refused.error().message().find("test.tooBig") != std::string::npos);
    std::printf("  %s\n", refused.error().toString().c_str());
    CHECK(device.memoryAvailable() <= (uint64_t{4} << 20));

    // What fits still fits, and the device still runs work.
    desc.bytes = uint64_t{1} << 20;
    desc.label = "test.fits";
    auto fits = gpu::Buffer::create(device, desc);
    REQUIRE(fits);
    gpu::CommandBatch batch(device);
    batch.encoder()->clearBuffer(fits->rhi(), 0, desc.bytes);
    batch.markDirty();
    REQUIRE(batch.submit(true));

    // With no budget at all, a size no machine has is still refused: by the
    // system's headroom on unified memory, by the device itself elsewhere (a
    // null buffer from Metal is SLANG_E_OUT_OF_MEMORY with the patch). A
    // result either way, not an abort, and nothing is allocated.
    device.setMemoryBudget(0);
    CHECK(device.memoryAvailable() == device.systemHeadroom());
    desc.bytes = uint64_t{1} << 50;
    desc.label = "test.absurd";
    auto absurd = gpu::Buffer::create(device, desc);
    REQUIRE_FALSE(absurd);
    CHECK(absurd.error().code() == ErrorCode::OutOfMemory);
}

TEST_CASE("a queue's failure is OutOfMemory where the device ran out, DeviceFailure otherwise", "[gpu][memory]") {
    // What a failed command buffer becomes on the way out of CommandBatch::submit.
    auto oom = gpu::Device::queueResult(SLANG_E_OUT_OF_MEMORY, "a frame");
    REQUIRE_FALSE(oom);
    CHECK(oom.error().code() == ErrorCode::OutOfMemory);
    CHECK(oom.error().message() == "the GPU ran out of memory running a frame");
    auto other = gpu::Device::queueResult(SLANG_FAIL, "a frame");
    REQUIRE_FALSE(other);
    CHECK(other.error().code() == ErrorCode::DeviceFailure);
    CHECK(gpu::Device::queueResult(SLANG_OK, "a frame"));
}
