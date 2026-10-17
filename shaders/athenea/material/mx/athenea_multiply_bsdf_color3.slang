// Copyright (c) 2026 jesus luque.
//
// <multiply> by a colour. The colour is also kept on each lobe it reaches
// (lobeTint): standard_surface's coat colour is this multiply over the base,
// and a weight alone mixes it with every scalar the graph put there. Only
// in a kernel that asks for the marks (kAtheneaMarkLobes).
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_multiply_bsdf_color3(ClosureData closureData, BSDF in1, float3 in2, out BSDF result)
{
    result = in1;
    const float3 tint = clamp(in2, 0.0, 1.0);
    for (uint k = 0; k < kBuildLobes; ++k)
    {
        result.weights[k] = in1.weights[k] * tint;
        if (kAtheneaMarkLobes && k < gAtheneaLobeCount && any(in1.weights[k] != float3(0.0)) &&
            gAtheneaLobes[k].kind != kLobeHair)
            setLobeTint(gAtheneaLobes[k], lobeTint(gAtheneaLobes[k]) * tint);
    }
}
