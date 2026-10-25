// Copyright (c) 2026 jesus luque.
//
// WHAT A MERGE MAY NOT MIX, as a class a splat (the high key the levels of
// detail sort by, above the Morton code): a group of a level holds one class.
//   0..5   the axis its shading normal leans most along, and the sign: a
//          panel's two faces, whose normals face apart, are never one group,
//          where one group's averaged normal turned a sharp reflection into
//          rings across the Corvette's doors
//   6      glass and sheets (lodKeptApart), which the cut never draws merged
// A cloud without shading normals leans by its shortest axis, whose sign is
// arbitrary: there the classes split by axis alone.
module lod_class;
import athenea.common.packing;
import lod_common;

StructuredBuffer<uint>    shape;     // 4 a splat (packing.slang)
StructuredBuffer<uint>    normals;   // a splat, where params.normals
StructuredBuffer<uint>    pbr;       // a splat, where params.pbrs
RWStructuredBuffer<uint>  classes;
ConstantBuffer<LodParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodClass(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    if (params.pbrs != 0 && lodKeptApart(pbr[i])) {
        classes[i] = 6u;
        return;
    }
    float3 n;
    if (params.normals != 0) {
        n = unpackNormal(normals[i]);
    } else {
        const uint row = i * 4;
        const float3 scale = float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2]));
        const float4 q = decodeQuaternion(shape[row]);
        const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
        // The rotation's column for that axis (packing's quaternion, x y z w).
        const float x = q.x, y = q.y, z = q.z, w = q.w;
        const float3 c0 = float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
        const float3 c1 = float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
        const float3 c2 = float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
        n = shortest == 0u ? c0 : (shortest == 1u ? c1 : c2);
        n = abs(n);   // no sign to trust
    }
    const float3 a = abs(n);
    const uint axis = a.x >= a.y ? (a.x >= a.z ? 0u : 2u) : (a.y >= a.z ? 1u : 2u);
    classes[i] = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);
}
