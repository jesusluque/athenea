// Copyright (c) 2026 jesus luque.
//
// WHAT A MERGE MAY NOT MIX, CHECKED (lod_class.slang): `lodClassFill` gives a
// cloud's splats normals facing apart -- +z for even splats, -z for odd ones,
// a panel's two faces at the same places -- and `lodClassCheck` counts the
// merged groups whose normal leans less than 0.9 along z, which an average of
// both faces would.
module lod_class_check;
import athenea.common.packing;

struct ClassCheckParams {
    uint count;
};

RWStructuredBuffer<uint> normals;
RWStructuredBuffer<uint> wrong;   ///< [0] groups mixed, [1] groups read
ConstantBuffer<ClassCheckParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodClassFill(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    normals[i] = packNormal((i & 1u) == 0u ? float3(0.0, 0.0, 1.0) : float3(0.0, 0.0, -1.0));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lodClassCheck(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    if (abs(unpackNormal(normals[g]).z) < 0.9) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[1], 1u);
}
