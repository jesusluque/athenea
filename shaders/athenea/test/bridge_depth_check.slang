// Copyright (c) 2026 jesus luque.
//
// Two depth AOVs (one float a pixel) compared on the device
// (tests/mcp/test_bridge.cpp): counts[0] pixels where one drew and the other
// did not (0 is nothing), counts[1] pixels both drew whose depths part by
// more than `relative` of the reference's; counts[2] and [3] the pixels
// each drew, [4] and [5] the largest depth each holds and [6] and [7] the
// smallest above 0 (a positive float's bits order as the float does).
module bridge_depth_check;

struct DepthParams {
    uint  pixels;
    float relative;
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float>      a;
StructuredBuffer<float>      b;
RWStructuredBuffer<uint>     counts;
ConstantBuffer<DepthParams>  params;

[shader("compute")]
[numthreads(256, 1, 1)]
void bridgeDepthCheck(uint3 id: SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= params.pixels) {
        return;
    }
    const float x = a[i];
    const float y = b[i];
    const bool drewX = x > 0.0;
    const bool drewY = y > 0.0;
    if (drewX) {
        InterlockedAdd(counts[2], 1u);
        InterlockedMax(counts[4], asuint(x));
        InterlockedMin(counts[6], asuint(x));
    }
    if (drewY) {
        InterlockedAdd(counts[3], 1u);
        InterlockedMax(counts[5], asuint(y));
        InterlockedMin(counts[7], asuint(y));
    }
    if (drewX != drewY) {
        InterlockedAdd(counts[0], 1u);
    } else if (drewX && abs(x - y) > params.relative * abs(y)) {
        InterlockedAdd(counts[1], 1u);
    }
}
