// Copyright (c) 2026 jesus luque.
//
// LIGHT IS LINEAR, CHECKED ON THE DEVICE (docs/decisions.md, light groups).
//
// `lightLayersCheck`: three frames of one size combined pixel by pixel,
// ka a + kb b + kc c in rgb, the alpha `a`'s -- a frame's coverage does not
// depend on its lamps. What a test compares against another frame with
// `compareImages`: image(A) + image(B) - image(off) against image(A and B),
// 2 image(A) - image(off) against image(A at level 2).
module light_layers_check;

struct CombineParams {
    uint  pixels;
    float ka;
    float kb;
    float kc;
};

StructuredBuffer<float4>      a;
StructuredBuffer<float4>      b;
StructuredBuffer<float4>      c;
RWStructuredBuffer<float4>    combined;
ConstantBuffer<CombineParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void lightLayersCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.pixels) {
        return;
    }
    const float4 x = a[i];
    combined[i] = float4(params.ka * x.rgb + params.kb * b[i].rgb + params.kc * c[i].rgb, x.w);
}
