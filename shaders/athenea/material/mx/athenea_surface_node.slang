// Copyright (c) 2026 jesus luque.
//
// What the nodes that build a whole surface's lobes share
// (athenea_open_pbr_bsdf, athenea_standard_surface_bsdf, athenea_gltf_pbr_bsdf): MaterialX's graph
// for such a surface passes a BSDF value from closure node to closure node,
// kBuildLobes wide each, and one node pushes the same lobes with the same
// weights instead. These are the closure nodes' own steps taken one lobe at a
// time -- a push where the weight pushes one (athenea_dielectric_bsdf and the
// others), a layer's effect on one lobe of its base (athenea_layer_bsdf), a
// colour multiply's (athenea_multiply_bsdf_color3) -- so a change to one of
// those nodes is a change here, and the tests that hold each node to its
// graph ("[openpbr]", "[standardsurface]", "[gltfpbr]") say when the two have parted.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

// What a reflecting dielectric_bsdf of weight one takes from what lies under
// it (athenea_dielectric_bsdf's directional albedo, a thin wall's included).
float3 athenea_node_dielectric_albedo(float3 V, float ior, float2 roughness, float3 N)
{
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    const float avgAlpha = averageAlpha(safeAlpha);
    const float R0 = iorToF0(ior);
    const float R = fresnelDielectric(NdotV, ior);
    const float F0 = gAtheneaThinWalled ? 2.0 * R0 / (1.0 + R0) : R0;
    const float3 F = float3(gAtheneaThinWalled ? 2.0 * R / (1.0 + R) : R);
    const float3 comp = ggxEnergyCompensation(NdotV, avgAlpha, F);
    return ggxDirectionalAlbedo(NdotV, avgAlpha, float3(F0), float3(1.0)) * comp;
}

// A dielectric_bsdf's lobe pushed where its weight pushes one.
uint athenea_node_dielectric(float weight, float3 tint, float ior, float2 roughness, float3 N, float3 X, uint scatter)
{
    if (weight < M_FLOAT_EPS)
        return kBuildLobes;
    Lobe lobe = atheneaLobe(kLobeDielectric);
    lobe.colour0 = max(tint, float3(0.0));
    lobe.ior = ior;
    lobe.alpha = roughness;
    lobe.scatter = scatter;
    return atheneaPushLobe(lobe, N, X);
}

// A generalized_schlick_bsdf's (exponent 5), reflecting.
uint athenea_node_schlick(float weight, float3 color0, float3 color82, float3 color90, float2 roughness, float3 N,
                          float3 X)
{
    if (weight < M_FLOAT_EPS)
        return kBuildLobes;
    Lobe lobe = atheneaLobe(kLobeSchlick);
    lobe.colour0 = color0;
    lobe.colour1 = color90;
    lobe.colour2 = color82;
    lobe.exponent = 5.0;
    lobe.alpha = roughness;
    lobe.scatter = kScatterReflect;
    return atheneaPushLobe(lobe, N, X);
}

// What a reflecting generalized_schlick_bsdf of weight one takes from what
// lies under it: its directional albedo, averaged
// (athenea_generalized_schlick_bsdf).
float athenea_node_schlick_albedo(float3 V, float3 color0, float3 color82, float3 color90, float exponent,
                                  float2 roughness, float3 N)
{
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float avgAlpha = averageAlpha(clamp(roughness, M_FLOAT_EPS, 1.0));
    const float3 F = fresnelHoffmanSchlick(NdotV, max(color0, 0.0), max(color82, 0.0), max(color90, 0.0), exponent);
    const float3 comp = ggxEnergyCompensation(NdotV, avgAlpha, F);
    const float3 dirAlbedo = ggxDirectionalAlbedo(NdotV, avgAlpha, max(color0, 0.0), max(color90, 0.0)) * comp;
    return dot(dirAlbedo, float3(1.0 / 3.0));
}

// A conductor_bsdf's.
uint athenea_node_conductor(float weight, float3 ior, float3 extinction, float2 roughness, float3 N, float3 X)
{
    if (weight < M_FLOAT_EPS)
        return kBuildLobes;
    Lobe lobe = atheneaLobe(kLobeConductor);
    lobe.colour0 = ior;
    lobe.colour1 = extinction;
    lobe.alpha = roughness;
    return atheneaPushLobe(lobe, N, X);
}

// A diffuse or translucent lobe.
uint athenea_node_diffuse(uint kind, float weight, float3 color, float roughness, uint flags, float3 N)
{
    if (weight < M_FLOAT_EPS)
        return kBuildLobes;
    Lobe lobe = atheneaLobe(kind);
    lobe.colour0 = color;
    lobe.roughness = roughness;
    lobe.flags = flags;
    return atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0));
}

// athenea_layer_bsdf, for the base's lobe `index` weighing `weight` under a top
// that reflects as a dielectric of index `overIor` (negative: it does not)
// and lets `throughput` by: the lobe's weight in the layer. A dielectric
// transmission under a reflection at its own index passes whole and is
// flagged as paying its own Fresnel; every lobe of the base that weighs
// something counts a layer more over it.
float3 athenea_node_under(uint index, float3 weight, float overIor, float3 throughput)
{
    if (index >= kBuildLobes)
        return float3(0.0);
    const bool weighs = any(weight != float3(0.0));
    const bool spared = overIor > 0.0 && weighs && gAtheneaLobes[index].kind == kLobeDielectric &&
                        gAtheneaLobes[index].scatter == kScatterTransmit &&
                        (gAtheneaLobes[index].flags & kFlagOwnFresnel) == 0 &&
                        abs(gAtheneaLobes[index].ior - overIor) <= 1.0e-4 * max(overIor, 1.0);
    if (spared)
        gAtheneaLobes[index].flags |= kFlagOwnFresnel;
    if (kAtheneaMarkLobes && weighs)
        gAtheneaLobes[index].pad3 += 1.0;
    return weight * (spared ? float3(1.0) : throughput);
}

// athenea_multiply_bsdf_color3 for one lobe: its weight by `tint` (already
// clamped to [0, 1]), and the tint kept on the lobe for a conversion.
float3 athenea_node_tinted(uint index, float3 weight, float3 tint)
{
    if (index >= kBuildLobes)
        return float3(0.0);
    if (kAtheneaMarkLobes && any(weight != float3(0.0)))
        setLobeTint(gAtheneaLobes[index], lobeTint(gAtheneaLobes[index]) * tint);
    return weight * tint;
}
