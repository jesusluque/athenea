// Copyright (c) 2026 jesus luque.
//
// THE PICTURE ON THE PAGE: the blend's bottom-row-first colour, exposed and,
// where it was blended in linear light, encoded as sRGB into the canvas's own texture (rgba8unorm,
// configured with STORAGE_BINDING), top row first as a page shows it. One
// storage buffer, one storage texture.
module web_present;
import athenea.common.color;

public struct PresentParams {
    public uint  width;
    public uint  height;
    public float exposure;   // a multiplier, 1 = as blended
    /// 1: the blend was in linear light and is encoded to sRGB here. 0: it was
    /// in display sRGB already -- standard 3DGS, which blends a capture's
    /// colours as they were trained -- and is written as it is.
    public uint  encode;
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
    const float3 exposed = saturate(c.rgb * present.exposure);
    const float3 shown = present.encode != 0 ? linearToSrgb(exposed) : exposed;
    target.Store(tid.xy, float4(shown, saturate(c.a)));
}
