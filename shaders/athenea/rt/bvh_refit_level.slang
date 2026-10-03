// Copyright (c) 2026 jesus luque.
//
// One height of a posed cloud's tree: each node the union of its children,
// a leaf's box made from the posed particle (bvh_levels.slang).
module bvh_refit_level;
import athenea.common.packing;
import rt_common;
import bvh_levels;

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      children;
StructuredBuffer<uint>      leaves;
StructuredBuffer<uint>      levelNodes;
RWStructuredBuffer<float4>  boxes;
ConstantBuffer<LevelParams> params;

void childBox(uint child, out float3 lo, out float3 hi) {
    if ((child & kLeafBit) != 0) {
        const uint particle = leaves[params.leafBase + (child & ~kLeafBit)];
        const uint row = particle * 4;
        rtParticleBox(positions[particle], shape[row], shape[row + 1], shape[row + 2], lo, hi);
    } else {
        lo = boxes[(params.nodeBase + child) * 2].xyz;
        hi = boxes[(params.nodeBase + child) * 2 + 1].xyz;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhRefitLevel(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.levelCount) {
        return;
    }
    const uint u = levelNodes[params.nodeBase + params.levelStart + tid.x];
    const uint node = params.nodeBase + u;
    float3 aLo, aHi, bLo, bHi;
    childBox(children[node * 2], aLo, aHi);
    childBox(children[node * 2 + 1], bLo, bHi);
    boxes[node * 2] = float4(min(aLo, bLo), 0.0);
    boxes[node * 2 + 1] = float4(max(aHi, bHi), 0.0);
}
