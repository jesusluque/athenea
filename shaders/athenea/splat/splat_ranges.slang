// Copyright (c) 2026 jesus luque.
// Where each tile's run of sorted pairs starts and ends. Each boundary is
// written by exactly one invocation.
module splat_ranges;
import frame;

StructuredBuffer<uint>      pairTiles;
RWStructuredBuffer<uint>    ranges;
StructuredBuffer<uint>      totalPairs;   // how many pairs: the emit's prefix sum
ConstantBuffer<FrameParams> params;       // count: the pairs' capacity, past which none was written

[shader("compute")]
[numthreads(256, 1, 1)]
void splatRanges(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    const uint count = min(params.count, totalPairs[0]);
    if (i >= count) {
        return;
    }
    const uint t = pairTiles[i];
    if (i == 0 || pairTiles[i - 1] != t) {
        ranges[t * 2] = i;
    }
    if (i + 1 == count || pairTiles[i + 1] != t) {
        ranges[t * 2 + 1] = i + 1;
    }
}
