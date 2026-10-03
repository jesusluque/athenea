// Copyright (c) 2026 jesus luque.
//
// What a decimation carried, checked against where each kept gaussian
// stands: a floor whose left half is one prim, one joint and one pattern of
// bits and whose right half is another, with a float attribute that is each
// record's own x. Counters of violations, which must come back zero.
module decimate_check;

struct DecimateCheckParams {
    uint  count;
    float tolerance;
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float4>          positions;   // the kept cloud's
StructuredBuffer<uint>            meanX;       // one float a kept gaussian
StructuredBuffer<uint>            ids;
StructuredBuffer<uint>            joints;      // four a kept gaussian
StructuredBuffer<uint>            weights;
StructuredBuffer<uint>            bits;
RWStructuredBuffer<uint>          counts;      // [mean, id, rig, bits, merged]
ConstantBuffer<DecimateCheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void decimateCheck(uint3 tid: SV_DispatchThreadID) {
    const uint o = tid.x;
    if (o >= params.count) {
        return;
    }
    const float3 p = positions[o].xyz;
    const bool left = p.x < 0.0;
    // A float's mean is the weighted mean of its records' x: the merge's own
    // centre, which the moments weigh alike.
    if (abs(asfloat(meanX[o]) - p.x) > params.tolerance) {
        InterlockedAdd(counts[0], 1u);
    }
    if (ids[o] != (left ? 7u : 9u)) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint joint = left ? 1u : 2u;
    if (joints[o * 4] != joint || abs(asfloat(weights[o * 4]) - 1.0) > 1.0e-5 ||
        asfloat(weights[o * 4 + 1]) != 0.0 || asfloat(weights[o * 4 + 2]) != 0.0 ||
        asfloat(weights[o * 4 + 3]) != 0.0) {
        InterlockedAdd(counts[2], 1u);
    }
    if (bits[o * 2] != (left ? 0xF0F0F0F0u : 0x0F0F0F0Fu) || bits[o * 2 + 1] != (left ? 1u : 2u)) {
        InterlockedAdd(counts[3], 1u);
    }
}
