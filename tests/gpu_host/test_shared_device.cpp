// Copyright (c) 2026 jesus luque.
//
// One device, two runtimes, no copies.
//
// gpe writes a buffer with its `iota` kernel; slang-rhi, through a view of the
// same memory, adds to it; gpe reads the sum. And the other way: a slang-rhi
// buffer adopted by gpe. If any of those crossed by copy, the numbers would
// still be right -- so the test also checks the objects are the same objects.
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "gpe/args.h"
#include "gpe/pool.h"
#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/ComputeKernel.h"
#include "athenea/gpu/ShaderLibrary.h"
#include "athenea/gpu_host/Context.h"

namespace {

struct IotaUniforms {
    uint32_t count;
    uint32_t first;
};
struct AddUniforms {
    uint32_t count;
    uint32_t amount;
};

constexpr uint32_t kCount = 1000;
constexpr uint32_t kSlots = 1024;

std::unique_ptr<athenea::gpu_host::Context> makeContext() {
    auto context = athenea::gpu_host::Context::create();
    if (!context) {
        return nullptr;
    }
    return std::move(*context);
}

}   // namespace

TEST_CASE("gpe and slang-rhi write the same buffer, in order", "[gpu_host][gpu]") {
    auto context = makeContext();
    if (context == nullptr || context->compute() == nullptr) {
        SKIP("no GPU device with a gpe backend");
    }
    gpe::PooledDevice& compute = *context->compute();
    athenea::gpu::ShaderLibrary library(context->deviceShared());

    REQUIRE(context->run([&] {
        const gpe::BufferId values = compute.alloc(kSlots * sizeof(uint32_t));
        REQUIRE(values != gpe::kInvalidBuffer);
        const gpe::KernelId iota = compute.load("iota");
        REQUIRE(iota != gpe::kInvalidKernel);

        gpe::Args args;
        args.buffer(values, sizeof(uint32_t)).uniforms(IotaUniforms{kCount, 7});
        compute.dispatch(iota, gpe::Grid{kCount, 1, 1}, args.data(), args.size());

        auto view = context->renderView(values, kSlots * sizeof(uint32_t), sizeof(uint32_t),
                                        "shared");
        REQUIRE(view);
        CHECK(view->native().value == compute.backendBuffer(values));

        auto add = athenea::gpu::ComputeKernel::create(library, "athenea/test/add", "addMain");
        if (!add) {
            FAIL(add.error().toString());
        }
        athenea::gpu::CommandBatch batch(context->device());
        add->dispatch(batch, {kCount, 1, 1}, [&](rhi::ShaderCursor cursor) {
            cursor["values"].setBinding(view->rhi());
            cursor["params"]["count"].setData(kCount);
            cursor["params"]["amount"].setData(uint32_t{1000});
        });
        REQUIRE(batch.submit(true));

        std::vector<uint32_t> out(kSlots, 0);
        compute.download(out.data(), values, out.size() * sizeof(uint32_t));
        bool summed = true;
        for (uint32_t i = 0; i < kCount; ++i) {
            summed = summed && out[i] == 1007 + i;
        }
        CHECK(summed);

        compute.release(values);
    }));
}

TEST_CASE("a slang-rhi buffer adopted by gpe is the same memory", "[gpu_host][gpu]") {
    auto context = makeContext();
    if (context == nullptr || context->compute() == nullptr) {
        SKIP("no GPU device with a gpe backend");
    }
    gpe::PooledDevice& compute = *context->compute();

    // The same allocation off the GPU thread and on it: gpe makes its adopted
    // CUDA context current for its own work, and a buffer slang-rhi allocates
    // while that is so goes to whichever context the thread is left in.
    {
        std::vector<uint32_t> zeros(kSlots, 0);
        auto outside = athenea::gpu::Buffer::fromSpan<uint32_t>(context->device(), zeros, "off the GPU thread");
        if (!outside) FAIL(outside.error().toString());
    }
    REQUIRE(context->run([&] {
        std::vector<uint32_t> zeros(kSlots, 0);
        auto buffer = athenea::gpu::Buffer::fromSpan<uint32_t>(context->device(), zeros, "rhi-owned");
        if (!buffer) FAIL(buffer.error().toString());
        auto adopted = context->computeView(*buffer);
        if (!adopted) FAIL(adopted.error().toString());
        CHECK(compute.backendBuffer(*adopted) == buffer->native().value);

        gpe::Args args;
        args.buffer(*adopted, sizeof(uint32_t)).uniforms(IotaUniforms{kCount, 3});
        compute.dispatch(compute.load("iota"), gpe::Grid{kCount, 1, 1}, args.data(), args.size());
        compute.sync();

        auto read = buffer->readAll<uint32_t>(context->device());
        REQUIRE(read);
        CHECK((*read)[0] == 3);
        CHECK((*read)[kCount - 1] == 3 + kCount - 1);
        CHECK((*read)[kCount] == 0);
        compute.release(*adopted);
    }));
}
