// Copyright (c) 2026 jesus luque.
// The tiles every splat touches, summed: the frame's pairs. A frame needs the
// total alone (how much room the pairs take), which a prefix sum over every
// splat gave with an offset a splat nobody read; here a thread sums a run of
// 64 and adds it to the total, cleared before.
module splat_touched_total;
import frame;

StructuredBuffer<uint>      tilesTouched;
RWStructuredBuffer<uint>    total;
ConstantBuffer<FrameParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void splatTouchedTotal(uint3 tid: SV_DispatchThreadID) {
    const uint first = tid.x * 64u;
    if (first >= params.count) {
        return;
    }
    const uint last = min(first + 64u, params.count);
    uint sum = 0u;
    for (uint k = first; k < last; ++k) {
        sum += tilesTouched[k];
    }
    if (sum > 0u) {
        InterlockedAdd(total[0], sum);
    }
}
