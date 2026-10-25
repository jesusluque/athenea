// Copyright (c) 2026 jesus luque.
//
// TEMPORAL NOISE OF A DECODE (athenea video-flicker): the error of a frame
// drawn from the video against the same frame drawn from its source, and how
// much that error changes from one frame to the next -- per pixel, the
// largest channel of |(video_k - source_k) - (video_k-1 - source_k-1)| in
// sRGB code values (0..255, the linear frames encoded as a display shows
// them). What stays put between frames is not flicker, however large; what
// changes is. One thread a pixel of `rect` (x0, y0, x1, y1 in the image's
// own rows: the engine's, bottom first); a histogram of 256 bins of a code
// and the sum in 1/16 codes, counters the host reads back.
module video_flicker;

struct FlickerParams {
    uint count;   // the rect's pixels
    uint width;   // the image's
    uint x0;
    uint y0;
    uint rectWidth;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float4>      videoNow;
StructuredBuffer<float4>      sourceNow;
StructuredBuffer<float4>      videoBefore;
StructuredBuffer<float4>      sourceBefore;
RWStructuredBuffer<uint>      counters;   // [0..255] histogram (a code a bin, the last open), [256] sum (1/16 code)
ConstantBuffer<FlickerParams> params;

float3 codes(float4 linear) {
    const float3 c = saturate(linear.rgb);
    const float3 s = select(c <= 0.0031308, 12.92 * c, 1.055 * pow(c, float3(1.0 / 2.4)) - 0.055);
    return s * 255.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoFlicker(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint i = (params.y0 + tid.x / params.rectWidth) * params.width + params.x0 + tid.x % params.rectWidth;
    const float3 now = codes(videoNow[i]) - codes(sourceNow[i]);
    const float3 before = codes(videoBefore[i]) - codes(sourceBefore[i]);
    const float3 d = abs(now - before);
    const float f = max(d.x, max(d.y, d.z));
    uint ignored;
    InterlockedAdd(counters[min(uint(f), 255u)], 1u, ignored);
    InterlockedAdd(counters[256], uint(min(f * 16.0, 4000.0)), ignored);
}
