// Copyright (c) 2026 jesus luque.
//
// Where a level's cells begin in the sorted splats: 1 where the code's top 3r
// bits differ from the splat before, or (`params.classes`) its class does.
module lod_boundaries;
import lod_common;

StructuredBuffer<uint>    keys;
StructuredBuffer<uint>    classes;   // sorted with `keys`, where params.classes
RWStructuredBuffer<uint>  boundary;
ConstantBuffer<LodParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void lodBoundaries(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint shift = levelShift(params.level);
    const bool apart = params.classes != 0 && i > 0 && classes[i] != classes[i - 1];
    boundary[i] = (i == 0 || (keys[i] >> shift) != (keys[i - 1] >> shift) || apart) ? 1u : 0u;
}
