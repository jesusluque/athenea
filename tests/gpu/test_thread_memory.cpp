// Copyright (c) 2026 jesus luque.
//
// WHAT A THREAD'S MEMORY BEARS (docs/decisions.md, 18 Nov). A kernel that
// keeps more in a thread's memory than the device gives a thread is built by
// Metal without a word, and its threads read back what others wrote: the path
// tracer's frames differed from run to run for it. Measured here with nothing
// but an array: filled from a buffer, read while it stands, compared with the
// buffer again -- the size from which threads find it changed is the
// device's, and what every kernel that shades has to stay under.
#include "GpuTest.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace athenea;

TEST_CASE("a kernel's thread keeps an array of sixteen kilobytes as it wrote it", "[gpu][threadmemory]") {
    ATHENEA_REQUIRE_GPU(gpu);
    constexpr uint32_t kThreads = 512;   // a side: 1024 groups of 256
    constexpr uint32_t kWords = 1u << 16;
    std::vector<uint32_t> words(kWords);
    uint32_t state = 0x9E3779B9u;
    for (uint32_t& w : words) {
        state = state * 1664525u + 1013904223u;   // bookkeeping: a fill, not data
        w = state;
    }
    gpu::BufferDesc desc;
    desc.bytes = words.size() * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "salts";
    auto salts = gpu::Buffer::create(*gpu->device, desc, words.data());
    REQUIRE(salts);
    struct Size {
        const char* entry;
        uint32_t    kilobytes;
    };
    // Past the first the device does not bear the rest are still run: what
    // they print is the measure.
    // The "Two": two arrays of that size in two functions, one done with
    // before the other is filled -- whether they share their memory.
    constexpr Size kSizes[] = {{"keeps1k", 1},      {"keeps4k", 4},      {"keeps8k", 8},      {"keeps16k", 16},
                               {"keeps24k", 24},    {"keeps32k", 32},    {"keeps48k", 48},    {"keeps64k", 64},
                               {"keeps128k", 128},  {"keeps256k", 256},  {"keepsTwo4k", 4},   {"keepsTwo8k", 8},
                               {"keepsTwo16k", 16}, {"keepsTwo24k", 24}, {"keepsTwo32k", 32}, {"keepsTwo64k", 64}};
    // CUDA's kernel that keeps 24 KB never comes back (docs/decisions.md, 22-23
    // Nov), so there the sizes past the claim are named and not run.
    const bool cuda = gpu->device->backend() == gpu::Backend::CUDA;
    for (const Size& size : kSizes) {
        if (cuda && (size.kilobytes > 16 || std::string(size.entry).find("Two") != std::string::npos)) {
            std::printf("  %s %3u KB: not run on CUDA\n", size.entry, size.kilobytes);
            continue;
        }
        auto kernel = gpu::ComputeKernel::create(*gpu->library, "athenea/test/thread_memory", size.entry);
        if (!kernel) {
            std::printf("  %s %3u KB: no pipeline: %s\n", size.entry, size.kilobytes, kernel.error().toString().c_str());
            CHECK(size.kilobytes > 16);
            continue;
        }
        // Three times: what a thread reads back of another's is not the same
        // at every run.
        uint32_t worst[4] = {0, 0, 0, 0};
        for (int run = 0; run < 3; ++run) {
            gpu::Buffer counts = test::uintBuffer(*gpu->device, 4, "counts");
            gpu::CommandBatch batch(*gpu->device);
            kernel->dispatch(batch, {kThreads, kThreads, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["counts"].setBinding(counts.rhi());
                cursor["salts"].setBinding(salts->rhi());
                cursor["memory"]["threads"].setData(kThreads);
                cursor["memory"]["words"].setData(kWords);
            });
            REQUIRE(batch.submit(true));
            uint32_t got[4] = {0, 0, 0, 0};
            REQUIRE(counts.read(*gpu->device, 0, sizeof(got), got));
            REQUIRE(got[0] == kThreads * kThreads);
            if (got[1] >= worst[1]) {
                worst[0] = got[0], worst[1] = got[1], worst[2] = got[2], worst[3] = got[3];
            }
        }
        const bool two = std::string(size.entry).find("Two") != std::string::npos;
        std::printf("  %s%3u KB: %u of %u threads read their array back changed (%u words)\n", two ? "twice " : "",
                    size.kilobytes, worst[1], worst[0], worst[2]);
        if (size.kilobytes <= 16 && !two) {
            CHECK(worst[1] == 0);
        }
    }
}
