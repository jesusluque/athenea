// Copyright (c) 2026 jesus luque.
// Tiles touched, in depth order: the input to the pairs' prefix sum.
module splat_gather_counts;
import frame;

StructuredBuffer<uint>      order;
StructuredBuffer<uint>      tilesTouched;
RWStructuredBuffer<uint>    sortedCounts;
StructuredBuffer<uint>      visibleTotal;   // how many are visible: the projection's prefix sum
ConstantBuffer<FrameParams> params;         // count: the most there can be

[shader("compute")]
[numthreads(256, 1, 1)]
void splatGatherCounts(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= min(params.count, visibleTotal[0])) {
        return;
    }
    sortedCounts[k] = tilesTouched[order[k]];
}
