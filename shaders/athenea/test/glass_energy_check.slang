// Copyright (c) 2026 jesus luque.
//
// WHAT GLASS SENDS ON IS WHAT IT DID NOT REFLECT.
//
// A transmitting gaussian's body is the light that came through it, and the
// reflection is added on top of that whatever the surface does. At full
// strength the two are more light than ever arrived, and a glass ball came
// back half again as bright as the mesh beside it with a rim two and a half
// times too hot. What passes is `1 - ggxEnvDielectric`, the same complement
// `lobes.slang` writes for a transmitting lobe, and what this holds is the
// sum: reflected plus transmitted, over roughness and angle, never above one.
module glass_energy_check;
import athenea.splat.splat_relight;
import athenea.common.packing;

struct GlassEnergyParams {
    uint  steps;
    float tolerance;
    uint  pad0; uint pad1;
};

RWStructuredBuffer<uint>  stats;   ///< 8 wide
RWStructuredBuffer<float> worst;   ///< 8 wide
ConstantBuffer<GlassEnergyParams> params;

/// stats[0] points where the two together are more than what arrived
/// stats[1] points taken
/// stats[2] points where nothing passed at all away from grazing
/// worst[0] the largest sum; worst[1] where its roughness was
[shader("compute")]
[numthreads(1, 1, 1)]
void glassEnergyCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float most = 0.0;
    for (uint i = 0; i < params.steps; ++i) {
        const float roughness = (float(i) + 0.5) / float(params.steps);
        for (uint j = 0; j < params.steps; ++j) {
            const float nDotV = (float(j) + 0.5) / float(params.steps);
            SplatSurface s;
            s.lobes = plainLobes();
            s.thinWalled = false;
            s.hasShadowBits = false;
            s.cellWords = 0u;
            s.cellBase = 0u;
            s.albedo = float3(1.0);
            s.metallic = 0.0;
            s.roughness = roughness;
            s.lit = false;
            const float3 reflected = splatPolish(s, nDotV);
            const float3 passed = saturate(float3(1.0) - ggxEnvDielectric(roughness, nDotV));
            const float sum = max(max(reflected.x + passed.x, reflected.y + passed.y),
                                  reflected.z + passed.z);
            if (sum > most) {
                most = sum;
                worst[1] = roughness;
                worst[2] = nDotV;
            }
            if (sum > 1.0 + params.tolerance) {
                InterlockedAdd(stats[0], 1u);
            }
            // Away from the horizon a dielectric passes nearly everything: a
            // complement that closed up would be a glass nobody sees through.
            if (nDotV > 0.3 && passed.x < 0.5) {
                InterlockedAdd(stats[2], 1u);
            }
            InterlockedAdd(stats[1], 1u);
        }
    }
    worst[0] = most;
}
