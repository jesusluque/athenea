// Copyright (c) 2026 jesus luque.
#include "athenea/gpu/algo/PrefixSum.h"

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::gpu {
namespace {

void setParams(rhi::ShaderCursor cursor, uint32_t count, uint32_t chunks, bool counted) {
    cursor["params"]["count"].setData(count);
    cursor["params"]["chunkSize"].setData(PrefixSum::chunkFor(count));
    cursor["params"]["chunkCount"].setData(chunks);
    cursor["params"]["counted"].setData(uint32_t{counted ? 1u : 0u});
}

}   // namespace

Result<PrefixSum> PrefixSum::create(ShaderLibrary& library) {
    PrefixSum sum;
    sum.device_ = &library.device();
    auto totals = ComputeKernel::create(library, "athenea/algo/prefix_chunk_totals", "prefixChunkTotals");
    if (!totals) return std::move(totals).error();
    auto starts = ComputeKernel::create(library, "athenea/algo/prefix_chunk_starts", "prefixChunkStarts");
    if (!starts) return std::move(starts).error();
    auto local = ComputeKernel::create(library, "athenea/algo/prefix_local", "prefixLocal");
    if (!local) return std::move(local).error();
    sum.totals_ = std::move(*totals);
    sum.starts_ = std::move(*starts);
    sum.local_ = std::move(*local);
    return sum;
}

Result<void> PrefixSum::reserve(uint32_t chunks) {
    if (chunks <= capacity_) {
        return ok();
    }
    // Geometric, so a frame that grows by a splat does not reallocate.
    uint32_t grown = capacity_ == 0 ? 16 : capacity_;
    while (grown < chunks) {
        grown *= 2;
    }
    BufferDesc desc;
    desc.bytes = uint64_t{grown} * sizeof(uint32_t);
    desc.elementBytes = sizeof(uint32_t);
    desc.label = "prefix.chunkTotals";
    auto totals = Buffer::create(*device_, desc);
    if (!totals) return std::move(totals).error();
    desc.label = "prefix.chunkStarts";
    auto starts = Buffer::create(*device_, desc);
    if (!starts) return std::move(starts).error();
    chunkTotals_ = *totals;
    chunkStarts_ = *starts;
    capacity_ = grown;
    return ok();
}

Result<void> PrefixSum::apply(CommandBatch& batch, const Buffer& input, const Buffer& output,
                              const Buffer& total, uint32_t count) {
    return run(batch, input, output, total, nullptr, count);
}

Result<void> PrefixSum::apply(CommandBatch& batch, const Buffer& input, const Buffer& output,
                              const Buffer& total, const Buffer& count, uint32_t maxCount) {
    if (count.count() < 1) {
        return Error(ErrorCode::InvalidArgument, "a prefix sum's count buffer is empty");
    }
    return run(batch, input, output, total, &count, maxCount);
}

Result<void> PrefixSum::run(CommandBatch& batch, const Buffer& input, const Buffer& output,
                            const Buffer& total, const Buffer* countFrom, uint32_t count) {
    if (input.rhi() == output.rhi()) {
        return Error(ErrorCode::InvalidArgument, "prefix sum input and output must differ");
    }
    if (input.count() < count || output.count() < count || total.count() < 1) {
        return Error(ErrorCode::InvalidArgument, "prefix sum buffers are too small");
    }
    const uint32_t chunk = chunkFor(count);
    const uint32_t chunks = count == 0 ? 1 : (count + chunk - 1) / chunk;
    ATHENEA_TRY(reserve(chunks));

    const bool counted = countFrom != nullptr;
    // Bound either way, as every name a shader declares: the input stands in.
    rhi::IBuffer* countBinding = counted ? countFrom->rhi() : input.rhi();
    totals_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["countFrom"].setBinding(countBinding);
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        setParams(cursor, count, chunks, counted);
    });
    starts_.dispatch(batch, {1, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["chunkTotals"].setBinding(chunkTotals_.rhi());
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["total"].setBinding(total.rhi());
        setParams(cursor, count, chunks, counted);
    });
    local_.dispatch(batch, {chunks, 1, 1}, [&](rhi::ShaderCursor cursor) {
        cursor["input"].setBinding(input.rhi());
        cursor["countFrom"].setBinding(countBinding);
        cursor["chunkStarts"].setBinding(chunkStarts_.rhi());
        cursor["output"].setBinding(output.rhi());
        setParams(cursor, count, chunks, counted);
    });
    return ok();
}

}   // namespace athenea::gpu
