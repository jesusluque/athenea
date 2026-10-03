// Copyright (c) 2026 jesus luque.
//
// A glowing medium against the two frames that bracket it. Without the
// volume a pixel reads B; with an absorbing volume it reads A = B T, T the
// transmittance along the pixel's ray; with the same volume glowing L_e it
// reads E = B T + L_e (1 - T). So, pixel by pixel through the volume,
//
//     E - A + L_e A / B = L_e
//
// and the mean of the left side over the pixels the volume covers is what
// this sums, in fixed point, for a test to hold to L_e.
module volume_emission_check;

struct EmissionCheck {
    uint   pixels;
    float  minBackground;   // a pixel counts where B is at least this and A below B
    float  scale;
    uint   pad0;
    float3 emission;
    float  pad1;
};

StructuredBuffer<float4>     without;   // B
StructuredBuffer<float4>     absorbing; // A
StructuredBuffer<float4>     glowing;   // E
RWStructuredBuffer<uint>     sums;      // r, g, b of the left side, count
ConstantBuffer<EmissionCheck> check;

[shader("compute")]
[numthreads(256, 1, 1)]
void volumeEmissionCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.pixels) {
        return;
    }
    const float3 b = without[i].rgb;
    const float3 a = absorbing[i].rgb;
    const float3 e = glowing[i].rgb;
    if (b.y < check.minBackground || a.y >= b.y * 0.98) {
        return;
    }
    const float3 left = e - a + check.emission * (a / max(b, float3(1e-6)));
    InterlockedAdd(sums[0], uint(max(left.x, 0.0) * check.scale));
    InterlockedAdd(sums[1], uint(max(left.y, 0.0) * check.scale));
    InterlockedAdd(sums[2], uint(max(left.z, 0.0) * check.scale));
    InterlockedAdd(sums[3], 1);
}
