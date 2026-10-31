// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER'S NEIGHBOUR GRID AND ITS FIELD AT POINTS (lod::SolidField;
// ssf_field.slang for the method). Everything the grid is -- its bounds, its
// level, the sorted codes -- is made here; nothing is read back.
//   ssfGridClear   the bounds' accumulators to their empty values
//   ssfGridBounds  every blob's centre widened by its support radius into
//                  them (floats as ordered words, atomics)
//   ssfGridSetup   one thread: the bounds, the extent, the level into gridInfo
//   ssfGridKeys    each blob's Morton code and index, for the sort
//   ssfFieldEval   (grad F, F) at each point
module ssf_grid;
import athenea.common.morton;
import ssf_field;

struct SsfGridParams {
    uint  count;   ///< blobs, or points for ssfFieldEval
    float tau;
    uint  pad0;
    uint  pad1;
};

RWStructuredBuffer<uint>   gridBounds;   ///< [0..2] ordered min, [3..5] ordered max, [6] largest radius
RWStructuredBuffer<float>  gridOut;      ///< gridInfo, written by ssfGridSetup
RWStructuredBuffer<uint>   keys;
RWStructuredBuffer<uint>   values;
StructuredBuffer<float4>   points;
RWStructuredBuffer<float4> field;        ///< grad F, F
ConstantBuffer<SsfGridParams> params;

/// A float as a word whose unsigned order is the float's.
uint ordered(float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

float unordered(uint w) {
    return asfloat((w & 0x80000000u) != 0 ? (w & 0x7FFFFFFFu) : ~w);
}

[shader("compute")]
[numthreads(8, 1, 1)]
void ssfGridClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < 3) {
        gridBounds[tid.x] = 0xFFFFFFFFu;
    } else if (tid.x < 7) {
        gridBounds[tid.x] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfGridBounds(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 c = blobPositions[i].xyz;
    const float r = supportRadius(i);
    [unroll] for (uint k = 0; k < 3; ++k) {
        InterlockedMin(gridBounds[k], ordered(c[k] - r));
        InterlockedMax(gridBounds[3 + k], ordered(c[k] + r));
    }
    InterlockedMax(gridBounds[6], asuint(max(r, 0.0)));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void ssfGridSetup(uint3 tid: SV_DispatchThreadID) {
    const float3 lo = float3(unordered(gridBounds[0]), unordered(gridBounds[1]), unordered(gridBounds[2]));
    const float3 hi = float3(unordered(gridBounds[3]), unordered(gridBounds[4]), unordered(gridBounds[5]));
    const float radius = asfloat(gridBounds[6]);
    // A little past the largest edge, so a centre on the far side is inside.
    const float extent = max(max(hi.x - lo.x, max(hi.y - lo.y, hi.z - lo.z)) * 1.0001, 1.0e-20);
    // The finest level whose cells are at least the largest support wide.
    uint level = 0;
    while (level < 10 && extent / float(2u << level) >= radius) {
        ++level;
    }
    gridOut[kGridLoX] = lo.x;
    gridOut[kGridLoY] = lo.y;
    gridOut[kGridLoZ] = lo.z;
    gridOut[kGridExtent] = extent;
    gridOut[kGridLevel] = float(level);
    gridOut[kGridRadius] = radius;
    gridOut[kGridCount] = float(params.count);
    gridOut[7] = 0.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfGridKeys(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 lo = float3(gridOut[kGridLoX], gridOut[kGridLoY], gridOut[kGridLoZ]);
    keys[i] = morton30((blobPositions[i].xyz - lo) / gridOut[kGridExtent]);
    values[i] = i;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfFieldEval(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    field[i] = fieldAt(points[i].xyz, params.tau);
}
