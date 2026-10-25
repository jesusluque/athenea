// Copyright (c) 2026 jesus luque.
//
// THE FRAME AS IT IS SHOWN, into a slot another process reads (format
// rgba8, bridge::FramePublisher): the display transform's texture, top row
// first as a picture is, a word a pixel (r in the low byte).
module bridge_shown;

Texture2D<float4>            shown;
RWStructuredBuffer<uint>     dst;

struct ShownParams {
    uint width;
    uint height;
    uint dstWord;
    uint pad;
};
ConstantBuffer<ShownParams>  params;

[shader("compute")]
[numthreads(16, 16, 1)]
void bridgePackShown(uint3 id: SV_DispatchThreadID) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    const float4 c = saturate(shown.Load(int3(int(id.x), int(id.y), 0)));
    const uint4 b = uint4(round(c * 255.0));
    dst[params.dstWord + id.y * params.width + id.x] = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}
