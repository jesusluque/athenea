// Copyright (c) 2026 jesus luque.
//
// What splats blended in linear light must give, as images a test compares a
// render against (`compareImages`), built here rather than on the CPU.
//
// `linearBlend`: two layers of constant colour and opacity, one in front of
// the other, over nothing: front a with alpha af, back b with alpha ab. The
// pixel is af a + (1 - af) ab b, premultiplied, and its alpha
// 1 - (1 - af)(1 - ab) -- the over operator in linear light, which is what a
// cloud of two splats that each cover the frame must render to.
//
// `srgbFinish`: the pipeline clouds were drawn with before they were blended
// in linear light, applied to an image that blended a cloud's sRGB values as
// they are: the finished splat contribution, unpremultiplied, taken through
// the sRGB curve and multiplied back. Over a transparent background that is
// exactly what the blend wrote, so a test can measure how far a capture's
// look moved.
module linear_blend;
import athenea.common.color;

struct LinearBlendParams {
    float4 front;    // rgb, alpha
    float4 back;     // rgb, alpha
    uint   width;
    uint   height;
    uint   pad0;
    uint   pad1;
};

RWStructuredBuffer<float4>       analytic;
ConstantBuffer<LinearBlendParams> blendParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void linearBlend(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= blendParams.width * blendParams.height) {
        return;
    }
    const float af = blendParams.front.w;
    const float ab = blendParams.back.w;
    analytic[at] = float4(af * blendParams.front.rgb + (1.0 - af) * ab * blendParams.back.rgb,
                          1.0 - (1.0 - af) * (1.0 - ab));
}

struct FinishParams {
    uint pixels;
    uint pad0;
    uint pad1;
    uint pad2;
};

RWStructuredBuffer<float4>  finished;
ConstantBuffer<FinishParams> finishParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void srgbFinish(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= finishParams.pixels) {
        return;
    }
    const float4 c = finished[at];
    finished[at] = c.a > 1e-6 ? float4(srgbToLinear(c.rgb / c.a) * c.a, c.a) : c;
}
