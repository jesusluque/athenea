// Copyright (c) 2026 jesus luque.
//
// <athenea_open_pbr_bsdf>: OpenPBR Surface's lobes in one node
// (athenea_open_pbr_surface.mtlx), as MaterialX's graph of sheen, dielectric,
// generalized_schlick, oren_nayar, translucent and subsurface under layer,
// mix, multiply and add built them: the same eleven lobes pushed in the same
// order, the same weights, the same marks (a layer over, a tint, a
// transmission that pays its own Fresnel). The graph passed a BSDF value
// from node to node -- 45 of them, kBuildLobes wide, in thread memory -- and
// a path evaluates its material at every vertex and at every face of a glass
// it crosses; here there is one value, the one the surface reads.
//
// Each lobe's weight is what the graph's nodes made of it, multiplied in the
// graph's order, so the two routes give the same floats. A mix whose weight
// is connected reaches the generator as two weighted sides added
// (MaterialX's premultipliedBsdfAdd): that is why the thin film's two
// reflections and two metals are each pushed, alike, with weights that sum.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

// The lobes, in the order they are pushed.
static const uint kOpenPbrFuzz = 0;
static const uint kOpenPbrCoat = 1;
static const uint kOpenPbrMetalFilm = 2;
static const uint kOpenPbrMetal = 3;
static const uint kOpenPbrReflectionFilm = 4;
static const uint kOpenPbrReflection = 5;
static const uint kOpenPbrTransmission = 6;
static const uint kOpenPbrThinDiffuse = 7;
static const uint kOpenPbrThinTranslucent = 8;
static const uint kOpenPbrSubsurface = 9;
static const uint kOpenPbrDiffuse = 10;
static const uint kOpenPbrLobes = 11;

// What a reflecting dielectric_bsdf of weight one takes from what lies under
// it (athenea_dielectric_bsdf's directional albedo, a thin wall's included).
float3 athenea_open_pbr_dielectric_albedo(float3 V, float ior, float2 roughness, float3 N)
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
uint athenea_open_pbr_dielectric(float weight, float3 tint, float ior, float2 roughness, float3 N, float3 X,
                                 uint scatter)
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

// A generalized_schlick_bsdf's (color90 white, exponent 5).
uint athenea_open_pbr_schlick(float weight, float3 color0, float3 color82, float2 roughness, float3 N, float3 X)
{
    if (weight < M_FLOAT_EPS)
        return kBuildLobes;
    Lobe lobe = atheneaLobe(kLobeSchlick);
    lobe.colour0 = color0;
    lobe.colour1 = float3(1.0);
    lobe.colour2 = color82;
    lobe.exponent = 5.0;
    lobe.alpha = roughness;
    lobe.scatter = kScatterReflect;
    return atheneaPushLobe(lobe, N, X);
}

// A diffuse or translucent lobe.
uint athenea_open_pbr_diffuse(uint kind, float weight, float3 color, float roughness, uint flags, float3 N)
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
float3 athenea_open_pbr_under(uint index, float3 weight, float overIor, float3 throughput)
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
float3 athenea_open_pbr_tinted(uint index, float3 weight, float3 tint)
{
    if (index >= kBuildLobes)
        return float3(0.0);
    if (kAtheneaMarkLobes && any(weight != float3(0.0)))
        setLobeTint(gAtheneaLobes[index], lobeTint(gAtheneaLobes[index]) * tint);
    return weight * tint;
}

void athenea_open_pbr_bsdf(ClosureData closureData, float fuzz_weight, float3 fuzz_color, float fuzz_roughness,
                           float coat_weight, float coat_ior, float2 coat_roughness, float specular_weight,
                           float thin_film_weight, float3 metal_reflectivity, float3 specular_color, float2 roughness,
                           float ior, float3 transmission_tint, float transmission_weight, float3 subsurface_color,
                           float3 subsurface_brdf_factor, float3 subsurface_btdf_factor, float subsurface_weight,
                           float thin_walled, float base_weight, float3 base_color, float base_diffuse_roughness,
                           float base_metalness, float3 base_darkening, float3 coat_attenuation, float3 N,
                           float3 coatN, float3 X, float3 coatX, out BSDF result)
{
    const float3 V = closureData.V;

    // The weights the generator folds into the two sides of a mix.
    const float metalFilmWeight = specular_weight * thin_film_weight;
    const float metalWeight = specular_weight * (1.0 - thin_film_weight);
    const float reflectionFilmWeight = 1.0 * thin_film_weight;
    const float reflectionWeight = 1.0 * (1.0 - thin_film_weight);
    const float subsurfaceLobeWeight = 1.0 * (1.0 - thin_walled);
    const float diffuseWeight = base_weight * (1.0 - subsurface_weight);

    uint index[kOpenPbrLobes];
    float3 weight[kOpenPbrLobes];

    // ---- the lobes, pushed as the graph's nodes were ----------------------

    // The fuzz: Zeltner's sheen.
    index[kOpenPbrFuzz] = kBuildLobes;
    float3 fuzzThroughput = float3(1.0);
    if (!(fuzz_weight < M_FLOAT_EPS))
    {
        const float3 n = dot(N, V) < 0.0 ? -N : N;
        const float NdotV = clamp(dot(n, V), M_FLOAT_EPS, 1.0);
        const float dirAlbedo = zeltnerSheenDirectionalAlbedo(NdotV, clamp(fuzz_roughness, 0.01, 1.0));
        Lobe lobe = atheneaLobe(kLobeSheen);
        lobe.colour0 = fuzz_color;
        lobe.roughness = fuzz_roughness;
        lobe.flags = kFlagZeltner;
        index[kOpenPbrFuzz] = atheneaPushLobe(lobe, N, float3(1.0, 0.0, 0.0));
        fuzzThroughput = float3(1.0 - dirAlbedo * fuzz_weight);
    }

    // The coat.
    index[kOpenPbrCoat] =
        athenea_open_pbr_dielectric(coat_weight, float3(1.0), coat_ior, coat_roughness, coatN, coatX, kScatterReflect);
    float3 coatThroughput = float3(1.0);
    if (!(coat_weight < M_FLOAT_EPS))
        coatThroughput = 1.0 - athenea_open_pbr_dielectric_albedo(V, coat_ior, coat_roughness, coatN) * coat_weight;

    // The metal, under a thin film and bare.
    index[kOpenPbrMetalFilm] =
        athenea_open_pbr_schlick(metalFilmWeight, metal_reflectivity, specular_color, roughness, N, X);
    index[kOpenPbrMetal] = athenea_open_pbr_schlick(metalWeight, metal_reflectivity, specular_color, roughness, N, X);

    // The dielectric's reflection, under a thin film and bare, and what the
    // two together let by: an add of throughputs, max(a + b - 1, 0).
    index[kOpenPbrReflectionFilm] =
        athenea_open_pbr_dielectric(reflectionFilmWeight, specular_color, ior, roughness, N, X, kScatterReflect);
    index[kOpenPbrReflection] =
        athenea_open_pbr_dielectric(reflectionWeight, specular_color, ior, roughness, N, X, kScatterReflect);
    float3 reflectionFilmThroughput = float3(1.0);
    float3 reflectionBareThroughput = float3(1.0);
    {
        const bool film = !(reflectionFilmWeight < M_FLOAT_EPS);
        const bool bare = !(reflectionWeight < M_FLOAT_EPS);
        if (film || bare)
        {
            const float3 dirAlbedo = athenea_open_pbr_dielectric_albedo(V, ior, roughness, N);
            if (film)
                reflectionFilmThroughput = 1.0 - dirAlbedo * reflectionFilmWeight;
            if (bare)
                reflectionBareThroughput = 1.0 - dirAlbedo * reflectionWeight;
        }
    }
    const float3 reflectionThroughput = max(reflectionFilmThroughput + reflectionBareThroughput - 1.0, 0.0);

    // The transmission; the thin wall's subsurface, reflected and let
    // through; the subsurface of a solid; the diffuse base.
    index[kOpenPbrTransmission] =
        athenea_open_pbr_dielectric(1.0, transmission_tint, ior, roughness, N, X, kScatterTransmit);
    index[kOpenPbrThinDiffuse] =
        athenea_open_pbr_diffuse(kLobeOrenNayar, 1.0, subsurface_color, base_diffuse_roughness, 0u, N);
    index[kOpenPbrThinTranslucent] = athenea_open_pbr_diffuse(kLobeTranslucent, 1.0, subsurface_color, 0.0, 0u, N);
    index[kOpenPbrSubsurface] =
        athenea_open_pbr_diffuse(kLobeOrenNayar, subsurfaceLobeWeight, subsurface_color, 0.0, 0u, N);
    index[kOpenPbrDiffuse] = athenea_open_pbr_diffuse(kLobeOrenNayar, diffuseWeight, base_color,
                                                      base_diffuse_roughness, kFlagEnergyCompensation, N);

    // ---- the weights the graph's combinators made of them -------------------

    // A multiply clamps its factor to [0, 1].
    const float metalness = clamp(base_metalness, 0.0, 1.0);
    const float dielectric = clamp(1.0 - base_metalness, 0.0, 1.0);
    const float transmits = clamp(transmission_weight, 0.0, 1.0);
    const float opaque = clamp(1.0 - transmission_weight, 0.0, 1.0);
    const float subsurface = clamp(subsurface_weight, 0.0, 1.0);
    const float thin = clamp(thin_walled, 0.0, 1.0);
    const float3 darkening = clamp(base_darkening, 0.0, 1.0);
    const float3 attenuation = clamp(coat_attenuation, 0.0, 1.0);

    weight[kOpenPbrFuzz] = float3(fuzz_weight);
    weight[kOpenPbrCoat] = float3(coat_weight);
    weight[kOpenPbrMetalFilm] = float3(metalFilmWeight) * metalness;
    weight[kOpenPbrMetal] = float3(metalWeight) * metalness;
    weight[kOpenPbrReflectionFilm] = float3(reflectionFilmWeight);
    weight[kOpenPbrReflection] = float3(reflectionWeight);
    // The transmission, by the transmission weight.
    weight[kOpenPbrTransmission] = float3(1.0) * transmits;
    // The thin wall's subsurface: its two halves each by its factor (a
    // tint), mixed half and half, where the wall is thin; the solid's where
    // it is not; both by the subsurface weight, beside the diffuse, by what
    // does not transmit.
    weight[kOpenPbrThinDiffuse] =
        athenea_open_pbr_tinted(index[kOpenPbrThinDiffuse], float3(1.0), clamp(subsurface_brdf_factor, 0.0, 1.0));
    weight[kOpenPbrThinTranslucent] =
        athenea_open_pbr_tinted(index[kOpenPbrThinTranslucent], float3(1.0), clamp(subsurface_btdf_factor, 0.0, 1.0));
    weight[kOpenPbrThinDiffuse] = lerp(float3(0.0), weight[kOpenPbrThinDiffuse], 0.5) * thin * subsurface * opaque;
    weight[kOpenPbrThinTranslucent] =
        lerp(weight[kOpenPbrThinTranslucent], float3(0.0), 0.5) * thin * subsurface * opaque;
    weight[kOpenPbrSubsurface] = float3(subsurfaceLobeWeight) * subsurface * opaque;
    weight[kOpenPbrDiffuse] = float3(diffuseWeight) * opaque;
    // A lobe that was not pushed weighs nothing.
    for (uint k = 0; k < kOpenPbrLobes; ++k)
    {
        if (index[k] >= kBuildLobes)
            weight[k] = float3(0.0);
    }

    // The dielectric base: the substrate under the reflection.
    float reflectionIor = -1.0;
    if (any(weight[kOpenPbrReflectionFilm] != float3(0.0)) || any(weight[kOpenPbrReflection] != float3(0.0)))
        reflectionIor = ior;
    for (uint k = kOpenPbrTransmission; k < kOpenPbrLobes; ++k)
        weight[k] = athenea_open_pbr_under(index[k], weight[k], reflectionIor, reflectionThroughput);
    // Mixed with the metal by the metalness.
    for (uint k = kOpenPbrReflectionFilm; k < kOpenPbrLobes; ++k)
        weight[k] = weight[k] * dielectric;
    // Darkened by the coat, then tinted by it.
    for (uint k = kOpenPbrMetalFilm; k < kOpenPbrLobes; ++k)
        weight[k] = athenea_open_pbr_tinted(index[k], weight[k], darkening);
    for (uint k = kOpenPbrMetalFilm; k < kOpenPbrLobes; ++k)
        weight[k] = athenea_open_pbr_tinted(index[k], weight[k], attenuation);
    // Under the coat, and all of it under the fuzz.
    const float coatIor = any(weight[kOpenPbrCoat] != float3(0.0)) ? coat_ior : -1.0;
    for (uint k = kOpenPbrMetalFilm; k < kOpenPbrLobes; ++k)
        weight[k] = athenea_open_pbr_under(index[k], weight[k], coatIor, coatThroughput);
    for (uint k = kOpenPbrCoat; k < kOpenPbrLobes; ++k)
        weight[k] = athenea_open_pbr_under(index[k], weight[k], -1.0, fuzzThroughput);

    result = athenea_bsdf_zero();
    for (uint k = 0; k < kOpenPbrLobes; ++k)
    {
        if (index[k] < kBuildLobes)
            result.weights[index[k]] = weight[k];
    }
    // What the surface constructor reads is the weights; the throughput past
    // the fuzz and the coat is theirs, over nothing that lets light through.
    result.throughput = float3(0.0);
}
