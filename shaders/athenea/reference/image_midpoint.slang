// Copyright (c) 2026 jesus luque.
//
// THE MIDPOINT OF TWO FRAMES, against which the frame between them is
// measured for flicker (`render::compareFlicker`). A smooth motion is nearly
// linear over three frames, so the frame between is nearly the mean of its
// neighbours and its second difference is small; a shadow that flickers is
// not, and its second difference is what it flickers by.
module image_midpoint;

struct MidpointParams {
    uint pixels;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float4>       before;
StructuredBuffer<float4>       after;
RWStructuredBuffer<float4>     midpoint;
ConstantBuffer<MidpointParams> midpointParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void imageMidpoint(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= midpointParams.pixels) {
        return;
    }
    midpoint[at] = 0.5 * (before[at] + after[at]);
}
