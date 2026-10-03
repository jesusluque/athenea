// Copyright (c) 2026 jesus luque.
//
// Checks for mesh2splat's host side, which now runs on the device.
//
// m2sSpanCheck: every ray's offset is `1e-4` of the diagonal of the box its
// records stand in, the box the test built them in (`params.low`,
// `params.high`). counts[0]: rays checked; counts[1]: rays off by more than
// `tolerance` of the offset.
//
// m2sAnswerCompare: two answers, float4 for float4. counts[0]: entries
// compared; counts[1]: entries apart by more than `tolerance` in any
// component; counts[2]: the largest difference, as its float bits (a
// non-negative float orders as its bits do).
module mesh2splat_host_check;

struct CheckParams {
    float4 low;
    float4 high;
    uint   count;
    float  tolerance;
    uint   pad0;
    uint   pad1;
};

StructuredBuffer<float4>    a;
StructuredBuffer<float4>    b;
RWStructuredBuffer<uint>    counts;
ConstantBuffer<CheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sSpanCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float expected = length(params.high.xyz - params.low.xyz) * 1.0e-4;
    const float got = a[k * 3].w;
    InterlockedAdd(counts[0], 1u);
    if (!(abs(got - expected) <= params.tolerance * expected)) {
        InterlockedAdd(counts[1], 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sAnswerCompare(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float4 d = abs(a[k] - b[k]);
    const float worst = max(max(d.x, d.y), max(d.z, d.w));
    InterlockedAdd(counts[0], 1u);
    if (!(worst <= params.tolerance)) {
        InterlockedAdd(counts[1], 1u);
    }
    InterlockedMax(counts[2], asuint(isfinite(worst) ? worst : 3.0e38));
}
