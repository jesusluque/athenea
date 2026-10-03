// Copyright (c) 2026 jesus luque.
//
// The mean colour of a rectangle of pixels, in fixed point: what a test reads
// back when the question is "what colour is the shadow", one number a channel.
module patch_mean;

struct PatchParams {
    uint  width;
    uint  x0;
    uint  y0;
    uint  w;
    uint  h;
    float scale;
    uint  withAlpha;   ///< 1: the alpha too, into sums[4], which the caller then made room for
    uint  pad1;
};

StructuredBuffer<float4>     frame;
RWStructuredBuffer<uint>     sums;   // r, g, b, count
ConstantBuffer<PatchParams>  params;

[shader("compute")]
[numthreads(8, 8, 1)]
void patchMean(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.w || tid.y >= params.h) {
        return;
    }
    const float4 c = frame[(params.y0 + tid.y) * params.width + params.x0 + tid.x];
    InterlockedAdd(sums[0], uint(max(c.x, 0.0) * params.scale));
    InterlockedAdd(sums[1], uint(max(c.y, 0.0) * params.scale));
    InterlockedAdd(sums[2], uint(max(c.z, 0.0) * params.scale));
    InterlockedAdd(sums[3], 1);
    if (params.withAlpha != 0) {
        InterlockedAdd(sums[4], uint(max(c.w, 0.0) * params.scale));
    }
}
