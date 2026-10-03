// Copyright (c) 2026 jesus luque.
//
// A split bake's halves and its allotment, checked where they are.
//
// splitIndirectSeen: how many gaussians found indirect light at all (their
// guides' first entry over `threshold`) -- a split that put everything in
// the direct half would pass the sum test and still be wrong.
//
// allotSynth: the sums of a first pass, made up -- every gaussian's mean
// luminance 0.5 over 64 paths of two steps each, the first half of them with
// a variance of `low` and the second with `high` -- so the allotment can be
// asked for on numbers whose answer is known.
//
// allotHalves: the passes allotted to each half, summed. counts[0] the first
// half's, counts[1] the second's.
module bake_split_check;

struct SplitCheckParams {
    uint  count;
    float threshold;
    float low;
    float high;
};

StructuredBuffer<float4>         guides;
RWStructuredBuffer<float4>       sums;    // 5 a gaussian: a bake of one coefficient
StructuredBuffer<uint>           allot;
RWStructuredBuffer<uint>         counts;
ConstantBuffer<SplitCheckParams> check;

[shader("compute")]
[numthreads(64, 1, 1)]
void splitIndirectSeen(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= check.count) {
        return;
    }
    if (guides[k].x > check.threshold) {
        InterlockedAdd(counts[0], 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void allotSynth(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= check.count) {
        return;
    }
    const float n = 64.0;
    const float mean = 0.5;
    const float variance = k < check.count / 2 ? check.low : check.high;
    const float second = mean * mean + variance * (n - 1.0) / n;
    const uint at = k * 5;
    sums[at] = float4(mean / 0.28209479, 0.0, 0.0, 0.0) * n;
    sums[at + 1] = float4(0.0);
    sums[at + 2] = float4(1.0, 1.0, 1.0, n);   // brightest, opacity summed
    sums[at + 3] = float4(n * mean, n * second, 0.0, 0.0);
    sums[at + 4] = float4(0.0, 2.0 * n, n, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void allotHalves(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= check.count) {
        return;
    }
    InterlockedAdd(counts[k < check.count / 2 ? 0 : 1], allot[k]);
}
