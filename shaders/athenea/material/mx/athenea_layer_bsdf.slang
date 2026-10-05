// Copyright (c) 2026 jesus luque.
//
// <layer>: the base beneath what the top lets through. Each lobe of the base
// counts one layer more over it (lobeLayersOver), which is how a conversion
// tells a coat from the specular under it (material_sample.slang).
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_layer_bsdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result)
{
    result = athenea_bsdf_zero();
    for (uint k = 0; k < kBuildLobes; ++k)
    {
        result.weights[k] = top.weights[k] + base.weights[k] * top.throughput;
        if (k < gAtheneaLobeCount && any(base.weights[k] != float3(0.0)) && gAtheneaLobes[k].kind != kLobeHair)
            gAtheneaLobes[k].pad3 += 1.0;
    }
    result.throughput = top.throughput * base.throughput;
}
