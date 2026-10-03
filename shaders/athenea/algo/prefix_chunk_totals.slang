// Copyright (c) 2026 jesus luque.
//
// Prefix sum, step 1 of 3: the sum of each chunk of the input.
//
// WHY CHUNKS AND NOT A BLELLOCH TREE
//
// A tree scan needs log2(N) dispatches each way and a power-of-two buffer.
// A chunked scan needs three dispatches whatever N is: one invocation per
// chunk walks its chunk in order (4096 elements is a short loop for a GPU
// thread), one invocation walks the chunk totals, and one invocation per chunk
// walks its chunk again writing offsets. No shared group memory, no atomics --
// nothing Metal, CUDA, Vulkan and D3D disagree about.
module prefix_chunk_totals;

struct PrefixParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint counted;   // 1: the count is countFrom[0], at most `count`
};

StructuredBuffer<uint>       input;
RWStructuredBuffer<uint>     chunkTotals;
// A device-sized sum's count, written by an earlier pass of the same frame:
// `params.count` is then only the most it can be. Bound either way.
StructuredBuffer<uint>       countFrom;
ConstantBuffer<PrefixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void prefixChunkTotals(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint count = params.counted != 0 ? min(countFrom[0], params.count) : params.count;
    const uint end = min(begin + params.chunkSize, count);
    uint sum = 0;
    for (uint i = begin; i < end; ++i) {
        sum += input[i];
    }
    chunkTotals[c] = sum;
}
