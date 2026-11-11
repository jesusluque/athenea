// Copyright (c) 2026 jesus luque.
//
// What `athenea lift` needs round a lift (splat_lift.slang): a mask from a
// rectangle of the frame or from a picture someone painted, and a frame keyed
// by a matte.
module splat_lift_mask;

struct LiftMaskParams {
    uint  width;
    uint  height;
    uint  pad0; uint pad1;
    // The rectangle, in fractions of the frame from its bottom left corner:
    // left, bottom, right, top.
    float left; float bottom; float right; float top;
};

StructuredBuffer<float4>         image;    // a painted mask, or the frame to key: bottom row first
StructuredBuffer<float>          matte;
RWStructuredBuffer<float>        mask;
RWStructuredBuffer<float4>       keyed;
ConstantBuffer<LiftMaskParams>   params;

[shader("compute")]
[numthreads(16, 16, 1)]
void liftMaskRect(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const float2 at = (float2(tid.xy) + 0.5) / float2(float(params.width), float(params.height));
    const bool inside = at.x >= params.left && at.x <= params.right && at.y >= params.bottom && at.y <= params.top;
    mask[tid.y * params.width + tid.x] = inside ? 1.0 : 0.0;
}

/// A painted mask is its red channel: a grey picture, white where the mask is.
[shader("compute")]
[numthreads(16, 16, 1)]
void liftMaskImage(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    mask[at] = saturate(image[at].r);
}

/// The frame with the matte as its alpha: premultiplied, so what is left of a
/// pixel is the share of it the matte holds.
[shader("compute")]
[numthreads(16, 16, 1)]
void liftKey(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float4 pixel = image[at];
    const float held = min(saturate(matte[at]), pixel.a);
    keyed[at] = pixel.a > 1.0e-6 ? float4(pixel.rgb * (held / pixel.a), held) : float4(0.0);
}
