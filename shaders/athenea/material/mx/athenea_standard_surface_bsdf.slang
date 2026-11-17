// Copyright (c) 2026 jesus luque.
//
// <athenea_standard_surface_bsdf>: Autodesk Standard Surface's lobes in one
// node (athenea_standard_surface.mtlx), as MaterialX's graph of dielectric,
// conductor, sheen, translucent, subsurface and oren_nayar under layer, mix
// and multiply built them: the same eight lobes pushed in the same order,
// the same weights, the same marks (a layer over, the coat's tint, a
// transmission that pays its own Fresnel). The graph passed a BSDF value
// from node to node -- 29 of them, kBuildLobes wide, in thread memory -- and
// a path evaluates its material at every vertex and at every face of a glass
// it crosses; here there is one value, the one the surface reads.
//
// Each lobe's weight is what the graph's nodes made of it, multiplied in the
// graph's order, so the two routes give the same floats. A mix whose weight
// is connected reaches the generator as two weighted sides added
// (MaterialX's premultipliedBsdfAdd): the foreground's own weight by the
// mix, the background multiplied by its complement.
#include "athenea/material/mx/athenea_surface_node.slang"

// The lobes, in the order they are pushed.
static const uint kStandardSurfaceCoat = 0;
static const uint kStandardSurfaceMetal = 1;
static const uint kStandardSurfaceSpecular = 2;
static const uint kStandardSurfaceTransmission = 3;
static const uint kStandardSurfaceSheen = 4;
static const uint kStandardSurfaceTranslucent = 5;
static const uint kStandardSurfaceSubsurface = 6;
static const uint kStandardSurfaceDiffuse = 7;
static const uint kStandardSurfaceLobes = 8;

void athenea_standard_surface_bsdf(ClosureData closureData, float coat, float coat_ior, float2 coat_roughness,
                                   float metalness, float3 metal_ior, float3 metal_extinction, float specular,
                                   float3 specular_color, float specular_ior, float2 roughness, float transmission,
                                   float3 transmission_color, float2 transmission_roughness, float sheen,
                                   float3 sheen_color, float sheen_roughness, float subsurface,
                                   float3 subsurface_color, float thin_walled, float base, float3 base_color,
                                   float diffuse_roughness, float3 coat_attenuation, float3 N, float3 coatN, float3 X,
                                   float3 coatX, out BSDF result)
{
    const float3 V = closureData.V;

    // The weights the generator folds into the two sides of a mix.
    const float metalWeight = 1.0 * metalness;
    const float transmissionWeight = 1.0 * transmission;
    const float translucentWeight = 1.0 * thin_walled;
    const float subsurfaceWeight = 1.0 * (1.0 - thin_walled);
    const float diffuseWeight = base * (1.0 - subsurface);

    uint index[kStandardSurfaceLobes];
    float3 weight[kStandardSurfaceLobes];

    // ---- the lobes, pushed as the graph's nodes were ----------------------

    // The coat.
    index[kStandardSurfaceCoat] =
        athenea_node_dielectric(coat, float3(1.0), coat_ior, coat_roughness, coatN, coatX, kScatterReflect);
    float3 coatThroughput = float3(1.0);
    if (!(coat < M_FLOAT_EPS))
        coatThroughput = 1.0 - athenea_node_dielectric_albedo(V, coat_ior, coat_roughness, coatN) * coat;

    // The metal: a conductor of the artistic index.
    index[kStandardSurfaceMetal] = athenea_node_conductor(metalWeight, metal_ior, metal_extinction, roughness, N, X);

    // The dielectric's reflection, and what it lets by.
    index[kStandardSurfaceSpecular] =
        athenea_node_dielectric(specular, specular_color, specular_ior, roughness, N, X, kScatterReflect);
    float3 specularThroughput = float3(1.0);
    if (!(specular < M_FLOAT_EPS))
        specularThroughput = 1.0 - athenea_node_dielectric_albedo(V, specular_ior, roughness, N) * specular;

    // The transmission, at its own roughness.
    index[kStandardSurfaceTransmission] = athenea_node_dielectric(transmissionWeight, transmission_color, specular_ior,
                                                                  transmission_roughness, N, X, kScatterTransmit);

    // The sheen: Conty and Kulla's.
    index[kStandardSurfaceSheen] = kBuildLobes;
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
        index[kStandardSurfaceSheen] = atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0));
        sheenThroughput = float3(1.0 - dirAlbedo * sheen);
    }

    // The subsurface, a thin wall's (translucent) or a solid's (a diffuse,
    // until a path walks inside); the diffuse base.
    index[kStandardSurfaceTranslucent] =
        athenea_node_diffuse(kLobeTranslucent, translucentWeight, subsurface_color, 0.0, 0u, N);
    index[kStandardSurfaceSubsurface] =
        athenea_node_diffuse(kLobeOrenNayar, subsurfaceWeight, subsurface_color, 0.0, 0u, N);
    index[kStandardSurfaceDiffuse] =
        athenea_node_diffuse(kLobeOrenNayar, diffuseWeight, base_color, diffuse_roughness, 0u, N);

    // ---- the weights the graph's combinators made of them -------------------

    weight[kStandardSurfaceCoat] = float3(coat);
    weight[kStandardSurfaceMetal] = float3(metalWeight);
    weight[kStandardSurfaceSpecular] = float3(specular);
    weight[kStandardSurfaceTransmission] = float3(transmissionWeight);
    weight[kStandardSurfaceSheen] = float3(sheen);
    weight[kStandardSurfaceTranslucent] = float3(translucentWeight);
    weight[kStandardSurfaceSubsurface] = float3(subsurfaceWeight);
    weight[kStandardSurfaceDiffuse] = float3(diffuseWeight);
    // A lobe that was not pushed weighs nothing.
    for (uint k = 0; k < kStandardSurfaceLobes; ++k)
    {
        if (index[k] >= kBuildLobes)
            weight[k] = float3(0.0);
    }

    // A multiply clamps its factor to [0, 1].
    const float subsurfaces = clamp(subsurface, 0.0, 1.0);
    const float opaque = clamp(1.0 - transmission, 0.0, 1.0);
    const float dielectric = clamp(1.0 - metalness, 0.0, 1.0);
    const float3 attenuation = clamp(coat_attenuation, 0.0, 1.0);

    // The subsurface chosen, by the subsurface weight, beside the diffuse.
    weight[kStandardSurfaceTranslucent] = weight[kStandardSurfaceTranslucent] * subsurfaces;
    weight[kStandardSurfaceSubsurface] = weight[kStandardSurfaceSubsurface] * subsurfaces;
    // Under the sheen.
    for (uint k = kStandardSurfaceTranslucent; k < kStandardSurfaceLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], -1.0, sheenThroughput);
    // The sheen and what is under it, by what does not transmit, beside the
    // transmission.
    for (uint k = kStandardSurfaceSheen; k < kStandardSurfaceLobes; ++k)
        weight[k] = weight[k] * opaque;
    // Under the dielectric's reflection.
    const float specularIor = any(weight[kStandardSurfaceSpecular] != float3(0.0)) ? specular_ior : -1.0;
    for (uint k = kStandardSurfaceTransmission; k < kStandardSurfaceLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], specularIor, specularThroughput);
    // Mixed with the metal by the metalness.
    for (uint k = kStandardSurfaceSpecular; k < kStandardSurfaceLobes; ++k)
        weight[k] = weight[k] * dielectric;
    // Tinted by the coat, and under it.
    for (uint k = kStandardSurfaceMetal; k < kStandardSurfaceLobes; ++k)
        weight[k] = athenea_node_tinted(index[k], weight[k], attenuation);
    const float coatIor = any(weight[kStandardSurfaceCoat] != float3(0.0)) ? coat_ior : -1.0;
    for (uint k = kStandardSurfaceMetal; k < kStandardSurfaceLobes; ++k)
        weight[k] = athenea_node_under(index[k], weight[k], coatIor, coatThroughput);

    result = athenea_bsdf_zero();
    for (uint k = 0; k < kStandardSurfaceLobes; ++k)
    {
        if (index[k] < kBuildLobes)
            result.weights[index[k]] = weight[k];
    }
    // What the surface constructor reads is the weights; the throughput past
    // the coat is its own, over nothing that lets light through.
    result.throughput = float3(0.0);
}
