// Copyright (c) 2026 jesus luque.
//
// Per particle: the box around its cut ellipsoid and the Morton code of its
// centre within the cloud's bounds, for the radix sort that orders the leaves.
module bvh_leaves;
import athenea.common.packing;
import rt_common;
import bvh_common;
import athenea.common.morton;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
RWStructuredBuffer<float4>  leafBoxes;   // per particle: lo, hi (cloud-local)
RWStructuredBuffer<uint>    mortonKeys;
RWStructuredBuffer<uint>    mortonValues;
ConstantBuffer<BvhParams>   params;


[shader("compute")]
[numthreads(256, 1, 1)]
void bvhLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    const uint row = i * 4;
    float3 boxLo, boxHi;
    rtParticleBox(p, shape[row], shape[row + 1], shape[row + 2], boxLo, boxHi);
    leafBoxes[i * 2] = float4(boxLo, 0.0);
    leafBoxes[i * 2 + 1] = float4(boxHi, 0.0);

    const float3 lo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ);
    const float3 hi = float3(params.boundsHiX, params.boundsHiY, params.boundsHiZ);
    mortonKeys[i] = morton30((p.xyz - lo) / max(hi - lo, float3(1e-20)));
    mortonValues[i] = i;
}
