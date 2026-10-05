// Copyright (c) 2026 jesus luque.
//
// A MATERIAL READ BACK INTO A GAUSSIAN'S FEW NUMBERS, where the material was
// evaluated: the lobe stack a compiled MaterialX graph built at a point,
// reduced to what a converted gaussian carries -- its base colour, its
// metalness, the roughness of its specular and of its coat, the light it
// gives off.
//
// The conversion reads constants and plain maps off the stage on the
// processor (usd::MeshStage); this is for what that cannot carry, an input a
// graph computes -- a procedural brick, a marble's noise, a hexagonal tiling,
// a map tiled four times -- and a coat colour, which tints everything under
// the coat and has no number of its own on a gaussian. The bake's path
// tracer evaluates the material at each gaussian's first vertex anyway; it
// hands that stack here, a sample at a time, and the mean is the gaussian's.
//
// What tells the lobes apart:
//   - a diffuse lobe's colour is the base colour, before any weight: its
//     weight holds the mixing and layering, and its colour what the graph
//     fed it, times the colours `multiply` nodes put on it (lobeTint);
//   - a metal is a conductor, whose reflectivity at normal incidence is the
//     base colour it was made from (MaterialX's artistic index), or a Schlick
//     reflecting more than a fifth at normal incidence, or any Schlick where
//     the conversion was told the material is metal (the bake's rule,
//     bakeBody);
//   - a coat is a reflecting dielectric lying under fewer layers than the
//     metal (lobeLayersOver), which sits beside the specular under the coat;
//     with no metal, the shallower of two reflecting dielectrics at different
//     depths. A lone dielectric is the specular. (standard_surface: the coat
//     under no layer, metal and specular under one, the diffuse under three;
//     OpenPBR: the fuzz over all of it, so one more each.)
module material_sample;
__exported import athenea.material.lobes;

static const float3 kSampleLuminance = float3(0.2126, 0.7152, 0.0722);

/// One evaluation, reduced. `albedoSeen` and the like are 1 where the stack
/// had what the field is read from, 0 where it did not (the caller keeps its
/// own value there).
public struct MaterialSample {
    public float3 albedo;          ///< the base colour, times the tints over it
    public float  albedoSeen;
    public float  diffuseShare;    ///< 1: the colour is a diffuse lobe's, before its weight; 0: a metal's reflectivity
    public float  metallic;        ///< the metal's share of the base's weight
    public float  roughness;       ///< the specular's, or the metal's
    public float  roughnessSeen;
    public float  coatRoughness;
    public float  coatSeen;
    public float3 emission;
};

float sampleHeft(float3 weight) { return dot(max(weight, float3(0.0)), kSampleLuminance) + 1.0e-6; }

/// MaterialX's roughness from a GGX lobe's two alphas (alpha = roughness^2,
/// split by the anisotropy; their geometric mean keeps it).
float roughnessOfAlpha(float2 alpha) { return sqrt(sqrt(max(alpha.x * alpha.y, 0.0))); }

/// A conductor's reflectivity at normal incidence from its index and extinction.
float3 conductorReflectivity(float3 n, float3 k) {
    const float3 k2 = k * k;
    return ((n - 1.0) * (n - 1.0) + k2) / max((n + 1.0) * (n + 1.0) + k2, float3(1.0e-6));
}

bool sampleIsDiffuse(Lobe lobe) { return lobe.kind == kLobeOrenNayar || lobe.kind == kLobeBurley; }

bool sampleIsMetal(Lobe lobe, bool saidMetal, bool conductor) {
    if (lobe.kind == kLobeConductor) {
        return true;
    }
    const float f0 = max(max(lobe.colour0.x, lobe.colour0.y), lobe.colour0.z);
    return lobe.kind == kLobeSchlick && (f0 > 0.2 || (saidMetal && !conductor));
}

bool sampleReflects(Lobe lobe) {
    return (lobe.kind == kLobeDielectric || lobe.kind == kLobeSchlick) && lobe.scatter != kScatterTransmit;
}

/// The stack at one point, reduced. `saidMetal`: the conversion read the
/// material as metal at all (the bake's `bakeMetalness`).
public MaterialSample materialSampleOf(LobeStack stack, bool saidMetal) {
    MaterialSample out;
    out.albedo = float3(0.0);
    out.albedoSeen = 0.0;
    out.diffuseShare = 0.0;
    out.metallic = 0.0;
    out.roughness = 0.0;
    out.roughnessSeen = 0.0;
    out.coatRoughness = 0.0;
    out.coatSeen = 0.0;
    out.emission = stack.emission;
    bool conductor = false;
    for (uint k = 0; k < stack.count; ++k) {
        conductor = conductor || stack.lobes[k].kind == kLobeConductor;
    }
    float3 diffuse = float3(0.0);
    float  diffuseHeft = 0.0;
    float3 metal = float3(0.0);
    float  metalHeft = 0.0;
    float  metalLayers = 0.0;
    float  metalRoughness = 0.0;
    float  deepest = -1.0;    // the deepest reflecting dielectric's layers
    float  shallowest = 1.0e9;
    for (uint k = 0; k < stack.count; ++k) {
        const Lobe lobe = stack.lobes[k];
        if (lobe.kind == kLobeHair) {
            continue;
        }
        const float heft = sampleHeft(lobe.weight);
        if (sampleIsDiffuse(lobe)) {
            diffuse += lobe.colour0 * lobeTint(lobe) * heft;
            diffuseHeft += heft;
        } else if (sampleIsMetal(lobe, saidMetal, conductor)) {
            const float3 reflectivity =
                lobe.kind == kLobeConductor ? conductorReflectivity(lobe.colour0, lobe.colour1) : lobe.colour0;
            metal += reflectivity * lobeTint(lobe) * heft;
            metalHeft += heft;
            metalLayers += lobeLayersOver(lobe) * heft;
            metalRoughness += roughnessOfAlpha(lobe.alpha) * heft;
        } else if (sampleReflects(lobe)) {
            deepest = max(deepest, lobeLayersOver(lobe));
            shallowest = min(shallowest, lobeLayersOver(lobe));
        }
    }
    if (diffuseHeft > 1.0e-5) {
        out.albedo = diffuse / diffuseHeft;
        out.albedoSeen = 1.0;
        out.diffuseShare = 1.0;
    } else if (metalHeft > 1.0e-5) {
        out.albedo = metal / metalHeft;
        out.albedoSeen = 1.0;
    }
    if (diffuseHeft + metalHeft > 2.0e-5) {
        out.metallic = metalHeft / (diffuseHeft + metalHeft);
    }
    // THE BASE'S DEPTH: the metal's; or, with none, the deepest reflecting
    // dielectric's where two stand at different depths (the shallower is a
    // coat); or a lone dielectric's own, which is then the specular.
    const float baseLayers = metalHeft > 1.0e-5 ? metalLayers / metalHeft
                             : deepest > shallowest + 0.5 ? deepest
                                                          : shallowest;
    float specularHeft = 0.0;
    float specularRoughness = 0.0;
    float coatHeft = 0.0;
    float coatRoughness = 0.0;
    for (uint k = 0; k < stack.count; ++k) {
        const Lobe lobe = stack.lobes[k];
        if (lobe.kind == kLobeHair || sampleIsDiffuse(lobe) || sampleIsMetal(lobe, saidMetal, conductor) ||
            !sampleReflects(lobe)) {
            continue;
        }
        const float heft = sampleHeft(lobe.weight);
        if (lobeLayersOver(lobe) < baseLayers - 0.5) {
            coatHeft += heft;
            coatRoughness += roughnessOfAlpha(lobe.alpha) * heft;
        } else {
            specularHeft += heft;
            specularRoughness += roughnessOfAlpha(lobe.alpha) * heft;
        }
    }
    if (metalHeft > 1.0e-5) {
        out.roughness = metalRoughness / metalHeft;
        out.roughnessSeen = 1.0;
    } else if (specularHeft > 1.0e-5) {
        out.roughness = specularRoughness / specularHeft;
        out.roughnessSeen = 1.0;
    }
    if (coatHeft > 1.0e-5) {
        out.coatRoughness = coatRoughness / coatHeft;
        out.coatSeen = 1.0;
    }
    return out;
}

/// THE SUMS A BAKE KEEPS OVER ITS SAMPLES, and the three planes they become:
/// (albedo, its diffuse share), (roughness, metallic, coat roughness, the
/// coat's share of samples), (emission, the samples that saw a surface).
public struct MaterialSums {
    public float4 albedo;      ///< rgb summed where seen, w the count
    public float  diffuse;     ///< diffuse shares summed where the albedo was seen
    public float2 roughness;   ///< sum, count
    public float2 coat;        ///< sum, count
    public float  metallic;    ///< summed over every sample
    public float3 emission;
    public float  samples;
};

public MaterialSums materialSumsZero() {
    MaterialSums s;
    s.albedo = float4(0.0);
    s.diffuse = 0.0;
    s.roughness = float2(0.0);
    s.coat = float2(0.0);
    s.metallic = 0.0;
    s.emission = float3(0.0);
    s.samples = 0.0;
    return s;
}

public void materialSumsAdd(inout MaterialSums sums, MaterialSample m) {
    sums.albedo += float4(m.albedo * m.albedoSeen, m.albedoSeen);
    sums.diffuse += m.diffuseShare * m.albedoSeen;
    sums.roughness += float2(m.roughness * m.roughnessSeen, m.roughnessSeen);
    sums.coat += float2(m.coatRoughness * m.coatSeen, m.coatSeen);
    sums.metallic += m.metallic;
    sums.emission += m.emission;
    sums.samples += 1.0;
}

/// The planes, as means: (albedo, diffuse share), (roughness or -1, metallic,
/// coat roughness or -1, 0), (emission, samples). -1: never seen.
public void materialSumsPlanes(MaterialSums sums, out float4 p0, out float4 p1, out float4 p2) {
    const float n = max(sums.samples, 1.0);
    p0 = sums.albedo.w > 0.0 ? float4(sums.albedo.xyz / sums.albedo.w, sums.diffuse / sums.albedo.w)
                             : float4(-1.0, -1.0, -1.0, 0.0);
    p1 = float4(sums.roughness.y > 0.0 ? sums.roughness.x / sums.roughness.y : -1.0, sums.metallic / n,
                sums.coat.y > 0.0 ? sums.coat.x / sums.coat.y : -1.0, 0.0);
    p2 = float4(sums.emission / n, sums.samples);
}
