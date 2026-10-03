// Copyright (c) 2026 jesus luque.
//
// <sheen_bsdf>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_sheen_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N, int mode,
                    inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
        return;
    const float3 V = closureData.V;
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float dirAlbedo = mode == 0 ? imageworksSheenDirectionalAlbedo(NdotV, roughness)
                                      : zeltnerSheenDirectionalAlbedo(NdotV, clamp(roughness, 0.01, 1.0));
    Lobe lobe = atheneaLobe(kLobeSheen);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    lobe.flags = mode == 0 ? 0u : kFlagZeltner;
    bsdf = atheneaBsdfOf(atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(1.0 - dirAlbedo * weight));
}
