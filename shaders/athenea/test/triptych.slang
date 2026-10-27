// Copyright (c) 2026 jesus luque.
//
// Two linear frames and their difference side by side, as 8-bit sRGB for a
// person to look at: a | b | |a - b| times `gain`. Written for the comparison
// images the mesh lights' tests save; nothing is measured here.
module triptych;

StructuredBuffer<float4>   a;
StructuredBuffer<float4>   b;
RWStructuredBuffer<uint>   out;        // (3 width) x height, rgba8, top row first
uniform uint               width;
uniform uint               height;
uniform float              exposure;   // what both frames are multiplied by
uniform float              gain;       // and the difference

float encode(float v) {
    const float c = saturate(v);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

uint pack(float3 c) {
    const uint3 q = uint3(float3(encode(c.r), encode(c.g), encode(c.b)) * 255.0 + 0.5);
    return q.r | (q.g << 8) | (q.b << 16) | (255u << 24);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void triptych(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }
    // The frames are bottom row first, as Hydra's buffers; a picture is not.
    const uint at = (height - 1u - tid.y) * width + tid.x;
    const float3 ca = a[at].rgb * exposure;
    const float3 cb = b[at].rgb * exposure;
    const uint row = tid.y * width * 3u;
    out[row + tid.x] = pack(ca);
    out[row + width + tid.x] = pack(cb);
    out[row + 2u * width + tid.x] = pack(abs(ca - cb) * gain);
}
