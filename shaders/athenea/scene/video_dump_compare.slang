// Copyright (c) 2026 jesus luque.
//
// A RECEIVER'S DECODE AGAINST THIS ONE (athenea video-decode --dump-compare):
// another decoder's gaussians of a frame, a row a splat in the frame's order
// (x y z, opacity, three log scales, base colour rgb), against the store the
// planes kernel wrote for the same frame. One thread a row; the device keeps
// the largest and the summed difference of each group, the rows past a
// tolerance, and this decode's own opacity and largest log scale -- counters
// the host reads back and nothing else.
//
//   counters  per group g (0 position, 1 opacity, 2 log scale, 3 colour):
//             [g * 3] largest |difference| (a float's bits: positive floats
//             order as their bits), [g * 3 + 1] the sum in units of 1e-6 m,
//             1e-4, 1e-3, 1e-4, [g * 3 + 2] rows past the tolerance; then
//             [12] the largest opacity's bits, [13] the opacity sum in 1e-4,
//             [14 ..] a histogram of each splat's largest log scale, 256 bins
//             over [-16, 0)
module video_dump_compare;
import athenea.common.packing;

struct DumpParams {
    uint  count;
    uint  stride;      // floats a row
    float positionTolerance;
    float opacityTolerance;
    float scaleTolerance;
    float colourTolerance;
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<float>    rows;
RWStructuredBuffer<uint>   counters;
ConstantBuffer<DumpParams> params;

static const float kUnits[4] = { 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-4 };

void add(uint g, float difference, float tolerance) {
    uint before;
    InterlockedMax(counters[g * 3], asuint(difference));
    InterlockedAdd(counters[g * 3 + 1], uint(min(difference / kUnits[g], 16383.0)), before);
    if (difference > tolerance) {
        InterlockedAdd(counters[g * 3 + 2], 1u, before);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoDumpCompare(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint r = i * params.stride;
    const float4 p = positions[i];
    const uint w1 = shape[i * 4 + 1], w2 = shape[i * 4 + 2], w3 = shape[i * 4 + 3];
    const float3 ls = float3(lowHalf(w1), highHalf(w1), lowHalf(w2));
    const float3 base = float3(highHalf(w2), lowHalf(w3), highHalf(w3));
    const float3 dp = abs(p.xyz - float3(rows[r], rows[r + 1], rows[r + 2]));
    add(0, max(dp.x, max(dp.y, dp.z)), params.positionTolerance);
    add(1, abs(p.w - rows[r + 3]), params.opacityTolerance);
    const float3 dl = abs(ls - float3(rows[r + 4], rows[r + 5], rows[r + 6]));
    add(2, max(dl.x, max(dl.y, dl.z)), params.scaleTolerance);
    const float3 dc = abs(base - float3(rows[r + 7], rows[r + 8], rows[r + 9]));
    add(3, max(dc.x, max(dc.y, dc.z)), params.colourTolerance);
    uint before;
    InterlockedMax(counters[12], asuint(max(p.w, 0.0)));
    InterlockedAdd(counters[13], uint(clamp(p.w, 0.0, 5.0) * 1.0e4), before);
    const float widest = max(ls.x, max(ls.y, ls.z));
    const uint bin = uint(clamp((widest + 16.0) * 16.0, 0.0, 255.0));
    InterlockedAdd(counters[14 + bin], 1u, before);
}

// --- what the decoder left: a histogram of plane 0 ---------------------------

Texture2D<float>           lumaPlane;
RWStructuredBuffer<uint>   histogram;   // 256 bins of the 8-bit sample
ConstantBuffer<DumpParams> lumaParams;  // count: the picture's width; stride: its height

[shader("compute")]
[numthreads(256, 1, 1)]
void videoLumaHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint w = lumaParams.count, h = lumaParams.stride;
    if (tid.x >= w * h) {
        return;
    }
    const float f = lumaPlane.Load(int3(int(tid.x % w), int(tid.x / w), 0));
    uint before;
    InterlockedAdd(histogram[min(uint(round(f * 255.0)), 255u)], 1u, before);
}
