// Copyright (c) 2026 jesus luque.
//
// How many pixels of an image are neither one thing nor the other.
//
// An edge drawn without antialiasing is a staircase: every pixel is either
// the background or the surface, and none is part of both. One drawn with it
// has a band of pixels holding the fraction of the pixel each side covers.
// Counting that band is therefore counting the antialiasing, and it is a
// counter, which is all a test may read back.
module edge_count;

struct EdgeParams {
    uint  pixels;
    float low;     ///< below this it is the one thing
    float high;    ///< above this it is the other
    uint  pad0;
};

StructuredBuffer<float4>      frame;
RWStructuredBuffer<uint>      counts;   // 0: between, 1: below, 2: above
ConstantBuffer<EdgeParams>    params;

[shader("compute")]
[numthreads(256, 1, 1)]
void edgeCount(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.pixels) {
        return;
    }
    const float3 c = frame[at].xyz;
    const float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    if (lum < params.low) {
        InterlockedAdd(counts[1], 1);
    } else if (lum > params.high) {
        InterlockedAdd(counts[2], 1);
    } else {
        InterlockedAdd(counts[0], 1);
    }
}
