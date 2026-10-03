// Copyright (c) 2026 jesus luque.
//
// What a test needs to compare a one-pass blur against the truth: the mean of
// several frames, and the energy of one. Both on the device, since the rule
// here is that no test does arithmetic on data with the CPU.
module image_average;

struct AverageParams {
    uint  pixels;
    float weight;    ///< accumulate: what share of the mean this frame is
    float scale;     ///< energy: fixed point, since there are no float atomics
    uint  pad0;
};

RWStructuredBuffer<float4> accum;
StructuredBuffer<float4>   frame;
RWStructuredBuffer<uint>   total;
ConstantBuffer<AverageParams> params;

/// `accum += frame * weight`. Called once a sub-frame with weight 1/N.
[shader("compute")]
[numthreads(256, 1, 1)]
void imageAccumulate(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.pixels) {
        return;
    }
    accum[at] += frame[at] * params.weight;
}

/// The premultiplied luminance of a whole image, in fixed point.
///
/// This is what a blurred splat must keep: convolving by a normalised measure
/// moves light about without making or destroying any, so the sum over the
/// image is the assertion that the opacity paid for the spreading exactly.
[shader("compute")]
[numthreads(256, 1, 1)]
void imageEnergy(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.pixels) {
        return;
    }
    const float4 c = frame[at];
    const float lum = dot(c.xyz, float3(0.2126, 0.7152, 0.0722));
    InterlockedAdd(total[0], uint(max(lum, 0.0) * params.scale));
}
