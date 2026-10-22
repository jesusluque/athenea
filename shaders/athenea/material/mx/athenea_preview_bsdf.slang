// Copyright (c) 2026 jesus luque.
//
// <athenea_preview_bsdf>: UsdPreviewSurface's lobes in one node
// (athenea_usd_preview_surface.mtlx), as its graph of generalized_schlick,
// oren_nayar, conductor, layer, multiply and add built them: the same lobes
// pushed in the same order, the same weights, the same layer marks. The graph
// passed a BSDF value from node to node -- a dozen of them, kBuildLobes wide,
// in thread memory -- and evaluated at every pixel (a texture on the diffuse
// colour) that was most of 35 ms at 1080p on an RTX 3070; here there is one.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

// A reflecting generalized_schlick_bsdf (color82 white, exponent 5) as
// athenea_generalized_schlick_bsdf pushes it: its index, kBuildLobes where its
// weight pushed nothing, and the throughput it leaves what lies under it.
float3 athenea_preview_schlick(float3 V, float weight, float3 color0, float3 color90, float2 roughness, float3 N,
                               float3 X, out uint index)
{
    index = kBuildLobes;
    if (weight < M_FLOAT_EPS)
        return float3(1.0);
    const float3 n = dot(N, V) < 0.0 ? -N : N;
    const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
    const float avgAlpha = averageAlpha(clamp(roughness, M_FLOAT_EPS, 1.0));
    const float3 F = fresnelHoffmanSchlick(NdotV, max(color0, 0.0), float3(1.0), max(color90, 0.0), 5.0);
    const float3 comp = ggxEnergyCompensation(NdotV, avgAlpha, F);
    const float3 dirAlbedo = ggxDirectionalAlbedo(NdotV, avgAlpha, max(color0, 0.0), max(color90, 0.0)) * comp;
    const float average = dot(dirAlbedo, float3(1.0 / 3.0));
    Lobe lobe = atheneaLobe(kLobeSchlick);
    lobe.colour0 = color0;
    lobe.colour1 = color90;
    lobe.colour2 = float3(1.0);
    lobe.exponent = 5.0;
    lobe.alpha = roughness;
    lobe.scatter = kScatterReflect;
    index = atheneaPushLobe(lobe, N, X);
    return float3(1.0 - average * weight);
}

// One layer more over pushed lobe `index` where it weighs something
// (athenea_layer_bsdf's mark, read by a conversion: lobeLayersOver).
void athenea_preview_mark(uint index, float3 weight)
{
    if (kAtheneaMarkLobes && index < gAtheneaLobeCount && any(weight != float3(0.0)))
        gAtheneaLobes[index].pad3 += 1.0;
}

void athenea_preview_bsdf(ClosureData closureData, float3 diffuseColor, float useSpecularWorkflow,
                          float3 specularColor, float metallic, float2 roughness, float clearcoat,
                          float2 clearcoatRoughness, float ior, float3 metalIor, float3 metalExtinction, float3 N,
                          float3 X, out BSDF result)
{
    const float3 V = closureData.V;
    const float R = (1.0 - ior) / (1.0 + ior);
    const float R_sq = R * R;
    const float3 specularColorMetallic = lerp(float3(1.0), diffuseColor, metallic);
    const float3 F0 = lerp(specularColorMetallic * R_sq, specularColorMetallic, metallic);
    const float diffuseWeight = lerp(1.0 - metallic, 1.0, useSpecularWorkflow);

    // Pushed as the graph's nodes were: the coat, the specular workflow's
    // specular, the diffuse (one lobe, which both workflows weigh), the
    // metal, the metalness workflow's specular.
    uint coat;
    const float3 coatThroughput =
        athenea_preview_schlick(V, clearcoat, float3(R_sq), float3(1.0), clearcoatRoughness, N, X, coat);
    uint specular1;
    const float3 throughput1 = athenea_preview_schlick(V, 1.0, specularColor, float3(1.0), roughness, N, X, specular1);
    uint diffuse = kBuildLobes;
    if (diffuseWeight >= M_FLOAT_EPS)
    {
        Lobe lobe = atheneaLobe(kLobeOrenNayar);
        lobe.colour0 = diffuseColor;
        lobe.roughness = 0.0;
        diffuse = atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0));
    }
    uint metal = kBuildLobes;
    if (metallic >= M_FLOAT_EPS)
    {
        Lobe lobe = atheneaLobe(kLobeConductor);
        lobe.colour0 = metalIor;
        lobe.colour1 = metalExtinction;
        lobe.alpha = roughness;
        metal = atheneaPushLobe(lobe, N, X);
    }
    uint specular2;
    const float3 throughput2 =
        athenea_preview_schlick(V, 1.0, F0, specularColorMetallic, roughness, N, X, specular2);

    // The weights the graph's combinators made of them: the specular
    // workflow (its specular over the diffuse) and the metalness workflow
    // (the metal, and its specular over the diffuse, mixed by metallic) mixed
    // by the workflow, under the coat. A multiply clamps its factor to [0, 1].
    const float specularWorkflow = saturate(useSpecularWorkflow);
    const float metalnessWorkflow = saturate(1.0 - useSpecularWorkflow);
    const float dielectric = saturate(1.0 - metallic);
    const float3 diffuseUnder =
        diffuseWeight * (throughput1 * specularWorkflow + throughput2 * (dielectric * metalnessWorkflow));
    const float3 specular1Weight = float3(specularWorkflow);
    const float3 metalWeight = float3(metallic * metalnessWorkflow);
    const float3 specular2Weight = float3(dielectric * metalnessWorkflow);

    // The layer marks: the diffuse lies under each workflow's specular, and
    // everything the workflows weigh lies under the coat.
    athenea_preview_mark(diffuse, float3(diffuseWeight));
    athenea_preview_mark(diffuse, float3(diffuseWeight));
    athenea_preview_mark(specular1, specular1Weight);
    athenea_preview_mark(diffuse, diffuseUnder);
    athenea_preview_mark(metal, metalWeight);
    athenea_preview_mark(specular2, specular2Weight);

    result = athenea_bsdf_zero();
    if (coat < kBuildLobes)
        result.weights[coat] = float3(clearcoat);
    if (specular1 < kBuildLobes)
        result.weights[specular1] = specular1Weight * coatThroughput;
    if (diffuse < kBuildLobes)
        result.weights[diffuse] = diffuseUnder * coatThroughput;
    if (metal < kBuildLobes)
        result.weights[metal] = metalWeight * coatThroughput;
    if (specular2 < kBuildLobes)
        result.weights[specular2] = specular2Weight * coatThroughput;
    // What the surface constructor reads is the weights; the throughput past
    // the coat is the coat's, over nothing that lets light through.
    result.throughput = float3(0.0);
}
