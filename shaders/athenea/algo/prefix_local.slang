// Copyright (c) 2026 jesus luque.
//
// Prefix sum, step 3 of 3: exclusive offsets, chunk by chunk.
// output[i] = input[0] + ... + input[i-1].
module prefix_local;

struct PrefixParams {
    uint count;
    uint chunkSize;
    uint chunkCount;
    uint counted;   // 1: the count is countFrom[0], at most `count`
};

StructuredBuffer<uint>       input;
StructuredBuffer<uint>       chunkStarts;
RWStructuredBuffer<uint>     output;
// A device-sized sum's count, written by an earlier pass of the same frame:
// `params.count` is then only the most it can be. Bound either way.
StructuredBuffer<uint>       countFrom;
ConstantBuffer<PrefixParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void prefixLocal(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunkCount) {
        return;
    }
    const uint begin = c * params.chunkSize;
    const uint count = params.counted != 0 ? min(countFrom[0], params.count) : params.count;
    const uint end = min(begin + params.chunkSize, count);
    uint running = chunkStarts[c];
    for (uint i = begin; i < end; ++i) {
        const uint value = input[i];
        output[i] = running;
        running += value;
    }
}
