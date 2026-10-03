// Copyright (c) 2026 jesus luque.
//
// THE LIGHT A GAUSSIAN GIVES OFF, CHECKED ON THE DEVICE (GpuSplats::emission).
//
// What a test reads back is counters: how many emissions were compared and
// how many were not what they should be. The emissions themselves -- what a
// cloud was given, what a file gave back, what a merge made of several --
// are compared here.
module splat_emission_check;
import athenea.common.packing;

struct EmissionCheckParams {
    uint   count;
    float  tolerance;   ///< relative to the brightest channel: RGB9E5 steps by 1/512 of it
    uint   pad0;
    uint   pad1;
    float4 low;         ///< `emissionWithin`: the box every merge must stay inside
    float4 high;
};

/// The five emissions a test cloud is given, record `i` taking `i % 5`:
/// nothing, a warm lamp, a blue one past one, a bright one far past it, and
/// a dim grey. tests/usd/test_usd.cpp writes the same table into its records.
public float3 tableEmission(uint i) {
    const uint k = i % 5u;
    if (k == 0u) return float3(0.0, 0.0, 0.0);
    if (k == 1u) return float3(1.5, 0.75, 0.25);
    if (k == 2u) return float3(0.02, 0.5, 4.0);
    if (k == 3u) return float3(100.0, 3.0, 0.0);
    return float3(0.3, 0.3, 0.3);
}

StructuredBuffer<uint>     emissionA;
StructuredBuffer<uint>     emissionB;
RWStructuredBuffer<uint>   counts;   ///< [compared, wrong, ...]
ConstantBuffer<EmissionCheckParams> params;

bool near(float3 got, float3 want) {
    const float scale = max(max(want.x, want.y), max(want.z, 1.0e-6));
    return all(abs(got - want) <= params.tolerance * scale);
}

/// `emissionA` against the table, and against `emissionB` (another route to
/// the same cloud, word for word): [0] compared, [1] off the table, [2]
/// words apart from B.
[shader("compute")]
[numthreads(64, 1, 1)]
void emissionAgainstTable(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    if (!near(unpackRgb9e5(emissionA[i]), tableEmission(i))) {
        InterlockedAdd(counts[1], 1u);
    }
    if (emissionA[i] != emissionB[i]) {
        InterlockedAdd(counts[2], 1u);
    }
}

/// Merged emissions: each channel a weighted mean of what was merged, so
/// inside the box the merged ones span. [0] compared, [1] outside.
[shader("compute")]
[numthreads(64, 1, 1)]
void emissionWithin(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3 e = unpackRgb9e5(emissionA[i]);
    const float3 slack = params.tolerance * max(params.high.xyz, float3(1.0e-6));
    if (any(e < params.low.xyz - slack) || any(e > params.high.xyz + slack)) {
        InterlockedAdd(counts[1], 1u);
    }
}
