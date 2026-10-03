// Copyright (c) 2026 jesus luque.
//
// <mix>
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_mix_bsdf(ClosureData closureData, BSDF fg, BSDF bg, float mixValue, out BSDF result)
{
    result = athenea_bsdf_zero();
    for (uint k = 0; k < kBuildLobes; ++k)
        result.weights[k] = lerp(bg.weights[k], fg.weights[k], mixValue);
    result.throughput = lerp(bg.throughput, fg.throughput, mixValue);
}
