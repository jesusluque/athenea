// Copyright (c) 2026 jesus luque.
//
// <layer>: the base beneath what the top lets through. Each lobe of the base
// counts one layer more over it (lobeLayersOver), which is how a conversion
// tells a coat from the specular under it (material_sample.slang), in a
// kernel that asks for the marks (kAtheneaMarkLobes).
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

// A DIELECTRIC TRANSMISSION IN THE BASE PAYS ITS OWN FRESNEL (kFlagOwnFresnel,
// lobes.slang): the reflection layered straight over it -- a dielectric that
// reflects at the same index, OpenPBR's and standard_surface's specular --
// passes it whole, once; a layer further out (a coat) charges it as any.
void athenea_layer_bsdf(ClosureData closureData, BSDF top, BSDF base, out BSDF result)
{
    result = athenea_bsdf_zero();
    // The index of the dielectric reflection the top holds, if it holds one.
    float overIor = -1.0;
    for (uint j = 0; j < kBuildLobes; ++j)
    {
        if (j < gAtheneaLobeCount && any(top.weights[j] != float3(0.0)) && gAtheneaLobes[j].kind == kLobeDielectric &&
            gAtheneaLobes[j].scatter == kScatterReflect)
            overIor = gAtheneaLobes[j].ior;
    }
    for (uint k = 0; k < kBuildLobes; ++k)
    {
        const bool spared = overIor > 0.0 && k < gAtheneaLobeCount && any(base.weights[k] != float3(0.0)) &&
                            gAtheneaLobes[k].kind == kLobeDielectric &&
                            gAtheneaLobes[k].scatter == kScatterTransmit &&
                            (gAtheneaLobes[k].flags & kFlagOwnFresnel) == 0 &&
                            abs(gAtheneaLobes[k].ior - overIor) <= 1.0e-4 * max(overIor, 1.0);
        if (spared)
            gAtheneaLobes[k].flags |= kFlagOwnFresnel;
        result.weights[k] = top.weights[k] + base.weights[k] * (spared ? float3(1.0) : top.throughput);
        if (kAtheneaMarkLobes && k < gAtheneaLobeCount && any(base.weights[k] != float3(0.0)) &&
            gAtheneaLobes[k].kind != kLobeHair)
            gAtheneaLobes[k].pad3 += 1.0;
    }
    result.throughput = top.throughput * base.throughput;
}
