// Copyright (c) 2026 jesus luque.
//
// <burley_diffuse_bsdf>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_burley_diffuse_bsdf(ClosureData closureData, float weight, float3 color, float roughness, float3 N,
                             inout BSDF bsdf)
{
    bsdf.throughput = float3(0.0);
    if (weight < M_FLOAT_EPS)
        return;
    Lobe lobe = atheneaLobe(kLobeBurley);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    bsdf = atheneaBsdfOf(atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0)), float3(weight), float3(0.0));
}
