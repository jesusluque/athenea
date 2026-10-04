// Copyright (c) 2026 jesus luque.
//
// WHAT A MATERIAL LAYERS OVER ITS BASE, CHECKED ON THE DEVICE
// (`GpuSplats::lobes`, splat_relight.slang's `SplatLayers`).
//
// Counters come back, nothing else. Two questions:
//
//   lobesPlainCheck   a gaussian with the plain lobes -- every cloud that
//                     carries none -- is reflected exactly as before they
//                     existed; and a coat over anything gives back no more
//                     light than it takes
//   lobesRoundTrip    twelve values packed into three words come back
//                     within a byte's step, every one of them
module lobes_check;
import athenea.splat.splat_relight;
import athenea.common.packing;
import athenea.material.microfacet;

struct LobesCheckParams {
    uint  steps;       ///< a side of the grid of roughness, N.V and metalness
    uint  count;       ///< values packed in lobesRoundTrip
    float tolerance;   ///< relative, for what differs only by rounding
    uint  pad;
};

RWStructuredBuffer<uint>        counts;
ConstantBuffer<LobesCheckParams> params;

/// The specular lobe as splat_relight.slang wrote it before the layers:
/// D, Smith's height-correlated V and Schlick's F, with F0 0.04 for a
/// dielectric and the albedo for a metal.
float3 specularBefore(float3 n, float3 wo, float3 wi, float3 albedo, float metallic, float roughness, bool lit) {
    const float3 h = normalize(wo + wi);
    const float nDotV = max(dot(n, wo), 1.0e-4);
    const float nDotL = max(dot(n, wi), 0.0);
    const float nDotH = max(dot(n, h), 0.0);
    const float vDotH = max(dot(wo, h), 0.0);
    if (nDotL <= 0.0) {
        return float3(0.0);
    }
    const float alpha = max(roughness * roughness, 1.0e-3);
    const float a2 = alpha * alpha;
    const float d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    const float distribution = a2 / max(3.14159265358979 * d * d, 1.0e-9);
    const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    const float visibility = 0.5 / max(lambdaV + lambdaL, 1.0e-9);
    const float3 f0 = lit ? float3(0.04 * (1.0 - metallic)) : lerp(float3(0.04), albedo, metallic);
    const float3 fresnel = f0 + (float3(1.0) - f0) * pow(1.0 - vDotH, 5.0);
    return distribution * visibility * fresnel;
}

bool near(float3 a, float3 b) {
    const float3 scale = max(max(abs(a), abs(b)), float3(1.0e-6));
    return all(abs(a - b) <= params.tolerance * scale);
}

/// [0] compared, [1] an environment reflection not exactly the one before,
/// [2] layers that are not nothing, [3] a light's lobe off the one before,
/// [4] the plain lobes not themselves through a pack, [5] a coat that gave
/// back more than it took.
[shader("compute")]
[numthreads(64, 1, 1)]
void lobesPlainCheck(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.steps;
    if (tid.x >= n * n * n) {
        return;
    }
    const float roughness = (float(tid.x % n) + 0.5) / float(n);
    const float nDotV = (float((tid.x / n) % n) + 0.5) / float(n);
    const float metallic = float(tid.x / (n * n)) / float(max(n - 1, 1u));
    const float3 albedo = float3(0.9, 0.5, 0.05) * (0.2 + 0.8 * roughness);
    InterlockedAdd(counts[0], 1u);

    const SplatLobes plain = unpackLobes(packLobes(plainLobes()));
    SplatSurface s;
    s.normal = float3(0.0, 0.0, 1.0);
    s.albedo = albedo;
    s.metallic = metallic;
    s.schlickMetal = false;
    s.roughness = roughness;
    s.lit = false;
    s.lobes = plain;

    // The polish under a sky, whole: exactly the function every cloud read.
    // To a millionth: the same arithmetic, which a compiler may still fuse
    // differently once the function has a branch for a Schlick metal (a
    // bit-for-bit test read 693 of 4096 off by one ulp then).
    const float3 now = environmentBrdfOf(s, nDotV);
    const float3 before = environmentBrdf(albedo, metallic, roughness, nDotV);
    if (any(abs(now - before) > 1.0e-6 * max(abs(before), float3(1.0e-6))) ||
        any(ggxEnvDielectricAt(roughness, nDotV, plain.specularIor) != ggxEnvDielectric(roughness, nDotV))) {
        InterlockedAdd(counts[1], 1u);
    }
    // No coat and no sheen: nothing taken, nothing given.
    const SplatLayers layers = splatLayers(s, nDotV);
    if (any(layers.under != float3(1.0)) || any(layers.belowCoat != float3(1.0)) || any(layers.coat != float3(0.0)) ||
        any(layers.sheen != float3(0.0))) {
        InterlockedAdd(counts[2], 1u);
    }
    // A light's lobe, an albedo and a baked body alike, against the formula
    // it replaced.
    const float3 wo = float3(sqrt(max(1.0 - nDotV * nDotV, 0.0)), 0.0, nDotV);
    const float3 wi = normalize(float3(-0.3, 0.4, 0.8));
    bool off = !near(specularLobe(s, wo, wi), specularBefore(s.normal, wo, wi, albedo, metallic, roughness, false));
    s.lit = true;
    off = off || !near(specularLobe(s, wo, wi), specularBefore(s.normal, wo, wi, albedo, metallic, roughness, true));
    if (off) {
        InterlockedAdd(counts[3], 1u);
    }
    const SplatLobes p = plainLobes();
    if (plain.specularWeight != p.specularWeight || any(plain.specularColour != p.specularColour) ||
        plain.specularIor != p.specularIor || plain.coatWeight != p.coatWeight || plain.coatIor != p.coatIor ||
        plain.coatDarkening != p.coatDarkening ||
        any(plain.sheenColour != p.sheenColour)) {
        InterlockedAdd(counts[4], 1u);
    }
    // A COAT CREATES NO LIGHT. Under a sky of one, what reaches the base and
    // comes back is at most what came back without the coat, and the coat
    // adds back only what it took: `under T + coat <= max(T, 1)`.
    s.lit = false;
    const float3 without = albedo * (1.0 - metallic) + environmentBrdfOf(s, nDotV);
    s.lobes.coatWeight = 1.0;
    s.lobes.coatRoughness = 1.0 - roughness;
    s.lobes.coatIor = 1.2 + 1.5 * metallic;
    const SplatLayers coated = splatLayers(s, nDotV);
    const float3 with = coated.under * without + coated.coat;
    if (any(with > max(without, float3(1.0)) * (1.0 + params.tolerance))) {
        InterlockedAdd(counts[5], 1u);
    }
}

/// A value in 0..1 from an index, the same every run.
float hashed(uint i, uint k) {
    uint x = i * 747796405u + k * 2891336453u + 12345u;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    x = (x >> 22u) ^ x;
    return float(x & 0xFFFFu) / 65535.0;
}

/// [0] compared, [1] a weight, colour or roughness more than half a byte
/// off, [2] an index more than half its step off, or a darkening not on
/// where it was asked for at half or more.
[shader("compute")]
[numthreads(64, 1, 1)]
void lobesRoundTrip(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    SplatLobes l;
    l.specularWeight = hashed(i, 0);
    l.specularColour = float3(hashed(i, 1), hashed(i, 2), hashed(i, 3));
    l.specularIor = 1.0 + 1.99 * hashed(i, 4);
    l.coatWeight = hashed(i, 5);
    l.coatRoughness = hashed(i, 6);
    l.coatIor = 1.0 + 1.98 * hashed(i, 7);
    l.coatDarkening = hashed(i, 12);
    l.sheenColour = float3(hashed(i, 8), hashed(i, 9), hashed(i, 10));
    l.sheenRoughness = hashed(i, 11);
    const SplatLobes b = unpackLobes(packLobes(l));
    InterlockedAdd(counts[0], 1u);
    const float byte = 0.5 / 255.0 + 1.0e-6;
    const float step = 0.5 / 128.0 + 1.0e-6;
    const float coatStep = 0.5 / 64.0 + 1.0e-6;   // seven bits: the eighth is the darkening
    if (abs(b.specularWeight - l.specularWeight) > byte || any(abs(b.specularColour - l.specularColour) > byte) ||
        abs(b.coatWeight - l.coatWeight) > byte || abs(b.coatRoughness - l.coatRoughness) > byte ||
        any(abs(b.sheenColour - l.sheenColour) > byte) || abs(b.sheenRoughness - l.sheenRoughness) > byte) {
        InterlockedAdd(counts[1], 1u);
    }
    if (abs(b.specularIor - l.specularIor) > step || abs(b.coatIor - l.coatIor) > coatStep ||
        b.coatDarkening != (l.coatDarkening >= 0.5 ? 1.0 : 0.0)) {
        InterlockedAdd(counts[2], 1u);
    }
}

/// A SCHLICK METAL (task TX: OpenPBR's and glTF's) against the conductor of an
/// artistic index (standard_surface's), over roughness and N.V, a car paint's
/// dark base with a white edge: the Schlick never reflects more than the
/// conductor, which is what read a paint matte and bright; head on and smooth
/// it is its colour; and a white metal is white either way.
/// [0] points, [1] a Schlick above the conductor, [2] head on not its colour,
/// [3] a white metal not white, [4] out of [0, 1]
[shader("compute")]
[numthreads(64, 1, 1)]
void lobesSchlickCheck(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.steps;
    if (tid.x >= n * n) {
        return;
    }
    const float roughness = (float(tid.x % n) + 0.5) / float(n);
    const float nDotV = (float(tid.x / n) + 0.5) / float(n);
    InterlockedAdd(counts[0], 1u);
    SplatSurface s;
    s.normal = float3(0.0, 0.0, 1.0);
    s.albedo = float3(0.047, 0.06, 0.047);
    s.metallic = 1.0;
    s.roughness = roughness;
    s.lit = false;
    s.lobes = plainLobes();
    s.schlickMetal = true;
    const float3 schlick = environmentBrdfOf(s, nDotV);
    s.schlickMetal = false;
    const float3 conductor = environmentBrdfOf(s, nDotV);
    // Between thirty and seventy-five degrees, where the two part (at the
    // pole they agree, and right at grazing the conductor's fit and the
    // Schlick's cross).
    if (nDotV > 0.25 && nDotV < 0.87 && any(schlick > conductor + params.tolerance)) {
        InterlockedAdd(counts[1], 1u);
    }
    if (any(schlick < float3(0.0)) || any(schlick > float3(1.0 + params.tolerance))) {
        InterlockedAdd(counts[4], 1u);
    }
    s.schlickMetal = true;
    s.roughness = 0.0;
    const float3 headOn = environmentBrdfOf(s, 1.0);
    if (tid.x == 0 && any(abs(headOn - s.albedo) > float3(0.01))) {
        InterlockedAdd(counts[2], 1u);
    }
    s.albedo = float3(1.0);
    s.roughness = roughness;
    if (any(abs(environmentBrdfOf(s, nDotV) - float3(1.0)) > float3(0.05))) {
        InterlockedAdd(counts[3], 1u);
    }
}
