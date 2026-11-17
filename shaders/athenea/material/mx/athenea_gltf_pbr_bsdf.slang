// Copyright (c) 2026 jesus luque.
//
// <athenea_gltf_pbr_bsdf>: glTF PBR's lobes in one node
// (athenea_gltf_pbr.mtlx), as MaterialX's graph of dielectric,
// generalized_schlick, sheen and oren_nayar under layer and mix built them:
// the same eight lobes pushed in the same order, the same weights, the same
// marks (a layer over, a transmission that pays its own Fresnel under a
// clearcoat at its index). The graph passed a BSDF value from node to node,
// kBuildLobes wide each, in thread memory, and a path evaluates its material
// at every vertex and at every face of a glass it crosses; here there is one
// value, the one the surface reads.
//
// Each lobe's weight is what the graph's nodes made of it, multiplied in the
// graph's order, so the two routes give the same floats. A mix whose weight
// is connected reaches the generator as two weighted sides added
// (MaterialX's premultipliedBsdfAdd): that is why the iridescence's two
// reflections and two metals are each pushed, alike (the engine's lobes take
// nothing from a thin film), with weights that sum.
#include "athenea/material/mx/athenea_surface_node.slang"

// The lobes, in the order they are pushed.
static const uint kGltfPbrClearcoat = 0;
static const uint kGltfPbrSheen = 1;
static const uint kGltfPbrMetalFilm = 2;
static const uint kGltfPbrMetal = 3;
static const uint kGltfPbrReflectionFilm = 4;
static const uint kGltfPbrReflection = 5;
static const uint kGltfPbrTransmission = 6;
static const uint kGltfPbrDiffuse = 7;
static const uint kGltfPbrLobes = 8;

// The clearcoat's index: the graph's constant.
static const float kGltfPbrClearcoatIor = 1.5;

void athenea_gltf_pbr_bsdf(ClosureData closureData, float clearcoat, float2 clearcoat_roughness, float sheen,
                           float3 sheen_color, float sheen_roughness, float iridescence, float3 base_color,
                           float metallic, float3 dielectric_f0, float3 dielectric_f90, float2 roughness,
                           float transmission, float ior, float3 N, float3 coatN, float3 X, float3 coatX,
                           out BSDF result)
{
    const float3 V = closureData.V;

    // The weights the generator folds into the two sides of a mix.
    const float filmWeight = 1.0 * iridescence;
    const float bareWeight = 1.0 * (1.0 - iridescence);
    const float transmissionWeight = 1.0 * transmission;
    const float diffuseWeight = 1.0 * (1.0 - transmission);

    uint index[kGltfPbrLobes];
    float3 weight[kGltfPbrLobes];

    // ---- the lobes, pushed as the graph's nodes were ----------------------

    // The clearcoat.
    index[kGltfPbrClearcoat] = athenea_node_dielectric(clearcoat, float3(1.0), kGltfPbrClearcoatIor,
                                                       clearcoat_roughness, coatN, coatX, kScatterReflect);
    float3 coatThroughput = float3(1.0);
    if (!(clearcoat < M_FLOAT_EPS))
        coatThroughput =
            1.0 - athenea_node_dielectric_albedo(V, kGltfPbrClearcoatIor, clearcoat_roughness, coatN) * clearcoat;

    // The sheen: Conty and Kulla's.
    index[kGltfPbrSheen] = kBuildLobes;
    float3 sheenThroughput = float3(1.0);
    if (!(sheen < M_FLOAT_EPS))
    {
        const float3 n = dot(N, V) < 0.0 ? -N : N;
        const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
        const float dirAlbedo = imageworksSheenDirectionalAlbedo(NdotV, sheen_roughness);
        Lobe lobe = atheneaLobe(kLobeSheen);
        lobe.colour0 = sheen_color;
        lobe.roughness = sheen_roughness;
        lobe.flags = 0u;
        index[kGltfPbrSheen] = atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0));
        sheenThroughput = float3(1.0 - dirAlbedo * sheen);
    }

    // The metal, under the iridescence and bare.
    index[kGltfPbrMetalFilm] =
        athenea_node_schlick(filmWeight, base_color, float3(1.0), float3(1.0), roughness, N, X);
    index[kGltfPbrMetal] = athenea_node_schlick(bareWeight, base_color, float3(1.0), float3(1.0), roughness, N, X);

    // The dielectric's reflection, under the iridescence and bare, and what
    // the two together let by: an add of throughputs, max(a + b - 1, 0).
    index[kGltfPbrReflectionFilm] =
        athenea_node_schlick(filmWeight, dielectric_f0, float3(1.0), dielectric_f90, roughness, N, X);
    index[kGltfPbrReflection] =
        athenea_node_schlick(bareWeight, dielectric_f0, float3(1.0), dielectric_f90, roughness, N, X);
    float3 reflectionFilmThroughput = float3(1.0);
    float3 reflectionBareThroughput = float3(1.0);
    {
        const bool film = !(filmWeight < M_FLOAT_EPS);
        const bool bare = !(bareWeight < M_FLOAT_EPS);
        if (film || bare)
        {
            const float average =
                athenea_node_schlick_albedo(V, dielectric_f0, float3(1.0), dielectric_f90, 5.0, roughness, N);
            if (film)
                reflectionFilmThroughput = float3(1.0 - average * filmWeight);
            if (bare)
                reflectionBareThroughput = float3(1.0 - average * bareWeight);
        }
    }
    const float3 reflectionThroughput = max(reflectionFilmThroughput + reflectionBareThroughput - 1.0, 0.0);

    // The transmission and the diffuse base, both the base colour's.
    index[kGltfPbrTransmission] =
        athenea_node_dielectric(transmissionWeight, base_color, ior, roughness, N, X, kScatterTransmit);
    index[kGltfPbrDiffuse] = athenea_node_diffuse(kLobeOrenNayar, diffuseWeight, base_color, 0.0, 0u, N);

    // ---- the weights the graph's combinators made of them -------------------

    weight[kGltfPbrClearcoat] = float3(clearcoat);
    weight[kGltfPbrSheen] = float3(sheen);
    weight[kGltfPbrMetalFilm] = float3(filmWeight);
    weight[kGltfPbrMetal] = float3(bareWeight);
    weight[kGltfPbrReflectionFilm] = float3(filmWeight);
    weight[kGltfPbrReflection] = float3(bareWeight);
    weight[kGltfPbrTransmission] = float3(transmissionWeight);
    weight[kGltfPbrDiffuse] = float3(diffuseWeight);
    // A lobe that was not pushed weighs nothing.
    for (uint k = 0; k < kGltfPbrLobes; ++k)
    {
        if (index[k] >= kBuildLobes)
            weight[k] = float3(0.0);
    }

    // A multiply clamps its factor to [0, 1].
    const float metal = clamp(metallic, 0.0, 1.0);
    const float dielectric = clamp(1.0 - metallic, 0.0, 1.0);

    // The metal, by the metalness.
    weight[kGltfPbrMetalFilm] = weight[kGltfPbrMetalFilm] * metal;
    weight[kGltfPbrMetal] = weight[kGltfPbrMetal] * metal;
    // The transmission and the diffuse under the reflection: a Schlick, so
    // nothing is spared its Fresnel here.
    for (uint k = kGltfPbrTransmission; k < kGltfPbrLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], -1.0, reflectionThroughput);
    // The dielectric, by what is not metal.
    for (uint k = kGltfPbrReflectionFilm; k < kGltfPbrLobes; ++k)
        weight[k] = weight[k] * dielectric;
    // Under the sheen, and all of it under the clearcoat.
    for (uint k = kGltfPbrMetalFilm; k < kGltfPbrLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], -1.0, sheenThroughput);
    const float coatIor = any(weight[kGltfPbrClearcoat] != float3(0.0)) ? kGltfPbrClearcoatIor : -1.0;
    for (uint k = kGltfPbrSheen; k < kGltfPbrLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], coatIor, coatThroughput);

    result = athenea_bsdf_zero();
    for (uint k = 0; k < kGltfPbrLobes; ++k)
    {
        if (index[k] < kBuildLobes)
            result.weights[index[k]] = weight[k];
    }
    // What the surface constructor reads is the weights; the throughput past
    // the clearcoat is its own, over nothing that lets light through.
    result.throughput = float3(0.0);
}
