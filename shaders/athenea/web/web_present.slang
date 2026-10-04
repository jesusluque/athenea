// Copyright (c) 2026 jesus luque.
//
// THE PICTURE ON THE PAGE: the blend's linear, bottom-row-first colour,
// exposed and encoded as sRGB into the canvas's own texture (rgba8unorm,
// configured with STORAGE_BINDING), top row first as a page shows it. One
// storage buffer, one storage texture.
module web_present;
import athenea.common.color;

public struct PresentParams {
    public uint  width;
    public uint  height;
    public float exposure;   // a multiplier, 1 = as blended
    public uint  pad;
};

StructuredBuffer<float4>          colour;
[format("rgba8")]
WTexture2D<float4>                target;
ConstantBuffer<PresentParams>     present;

[shader("compute")]
[numthreads(16, 16, 1)]
void webPresent(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= present.width || tid.y >= present.height) {
        return;
    }
    const float4 c = colour[(present.height - 1 - tid.y) * present.width + tid.x];
    const float3 shown = linearToSrgb(saturate(c.rgb * present.exposure));
    target.Store(tid.xy, float4(shown, saturate(c.a)));
}
