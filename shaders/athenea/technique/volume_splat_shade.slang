// Copyright (c) 2026 jesus luque.
//
// WHAT EACH GAUSSIAN OF A MEDIUM SENDS TOWARDS THE CAMERA, baked once a frame
// (proposal 105 §3, step 2). The rasteriser then draws the gaussian with the
// exact optical depth of each pixel's ray and this radiance
// (splat/splat_volume.slang): emission-absorption over baked in-scattering,
// as the TX clouds bake their transfer -- not a path traced medium, which is
// what it is measured against.
//
// The radiance of gaussian i, per unit of extinction, at its centre x:
//
//   L_i = a_i S_i + (1 - a_i) (E + e B(T_i))
//
//   S_i = sum_lights  E_l sum_{n < N} b^n p_{g c^n}(w_l, w_o) T_l(x)^(a^n)
//       + sum_domes   L_d sum_{n < N} b^n Tbar_d(x)^(a^n)
//
// a_i the albedo (the medium's times the gaussian's), E the medium's emission,
// e B(T) the black body at the gaussian's temperature (volume_blackbody.slang),
// E_l the irradiance a light gives a point facing it, p Henyey-Greenstein
// with the medium's g and the eye's direction w_o at the centre, T_l the
// transmittance from the light to x the cloud shadow map reads at x (every
// cloud of the frame and this medium itself, the medium's own Fourier terms
// resolving x's depth inside it: 105 §4) -- for a distant light the mean of
// three readings along the light through the gaussian (Gauss-Hermite), since
// the mean of a transmittance over a wide gaussian is more than its value at
// the centre -- Tbar_d the mean of the map's dome directions
// (the zenith and a ring forty degrees up). n = 0 is single scattering; the
// octaves after it are Wrenninge's approximation of multiple scattering
// (extinction a^n towards the light, energy b^n, anisotropy c^n), which keeps
// a dense bright smoke from going dark the way single scattering alone does.
module volume_splat_shade;
import athenea.common.packing;
import athenea.light.lights;
import athenea.technique.splat_shadow_read;

public struct VolumeShadeParams {
    public uint   count;
    public uint   lightCount;
    public uint   hasMap;
    public uint   octaves;
    public float3 albedo;
    public float  anisotropy;
    public float3 emission;
    public float  emissionScale;
    public float  temperatureScale;
    public float  temperatureOffset;
    public uint   categoriesLo;
    public uint   categoriesHi;
    public float  w00, w01, w02, w03;
    public float  w10, w11, w12, w13;
    public float  w20, w21, w22, w23;
    public float3 eye;
    public float  octaveExtinction;   ///< a
    public float  octaveEnergy;       ///< b
    public float  octaveAnisotropy;   ///< c
    public uint   lutSize;
    public float  lutMaxKelvin;
};

StructuredBuffer<float4>           positions;
StructuredBuffer<uint>             shape;           // 4 words a gaussian, as GpuSplats packs them
StructuredBuffer<uint>             volume;          // two words a gaussian (scene/volume_pack.slang)
StructuredBuffer<LightRecord>      lights;
StructuredBuffer<uint>             volumeShadowMap; // technique::SplatShadowMap::map(), or a word of nothing
StructuredBuffer<float4>           blackbodyTable;     // volume_blackbody.slang's table
RWStructuredBuffer<uint>           radianceOut;     // rgb halves, two words a gaussian
ConstantBuffer<VolumeShadeParams>  shadeParams;

/// The map's own buffer, read as the pass's gaussian receivers read it.
struct VolumeMapSource : IShadowMapSource {
    static float headerReal(uint word) { return asfloat(volumeShadowMap[word]); }
    static uint headerCount(uint word) { return volumeShadowMap[word]; }
    static float texel(ShadowMapFrame f, uint light, int x, int y, uint which) {
        const uint word = f.offset + (uint(y) * f.resolution + uint(x)) * shadowStrideOf(f.coefficients, f.medium) + which;
        const uint raw = volumeShadowMap[word];
        const bool mediumWord = which >= f.coefficients + 2;
        const uint term = mediumWord ? which - (f.coefficients + 2) : which;
        return !mediumWord && which >= f.coefficients ? shadowOrderedFloat(raw)
               : term == 0                            ? float(raw) / kShadowFixedScale
                                                      : float(int(raw)) / kShadowFixedScale;
    }
    static uint levels(ShadowMapFrame f) { return 1u; }
    static float transmittanceAt(ShadowMapFrame f, uint light, uint level, int x, int y) {
        return exp(-max(texel(f, light, x, y, 0), 0.0));
    }
};

float hg(float cosTheta, float g) {
    const float denom = 1.0 + g * g - 2.0 * g * cosTheta;
    return (1.0 - g * g) / (4.0 * 3.14159265358979 * denom * sqrt(max(denom, 1.0e-8)));
}

/// The black body at `kelvin`, linear between the table's entries.
float3 blackbodyAt(float kelvin) {
    const VolumeShadeParams p = shadeParams;
    if (p.lutSize < 2 || !(kelvin > 0.0)) {
        return float3(0.0);
    }
    const float at = clamp(kelvin / p.lutMaxKelvin, 0.0, 1.0) * float(p.lutSize - 1);
    const uint lo = min(uint(at), p.lutSize - 2);
    return lerp(blackbodyTable[lo].rgb, blackbodyTable[lo + 1].rgb, at - float(lo));
}

/// The octaves' sum for one direction: what the phase and the transmittance
/// give at each, weighed by its energy. The transmittance is three readings
/// along the light through the gaussian (`behind`, `centre`, `ahead`, Gauss-
/// Hermite's nodes at -sqrt 3, 0 and sqrt 3 deviations, weights 1/6, 2/3,
/// 1/6): the mean of a transmittance over the gaussian is more than its value
/// at the centre (it is convex in the depth), and a wide gaussian read at its
/// centre alone went dark.
float octaveSum3(float behind, float centre, float ahead, float cosTheta, bool isotropic) {
    const VolumeShadeParams p = shadeParams;
    float sum = 0.0;
    float energy = 1.0;
    float extinction = 1.0;
    float g = p.anisotropy;
    for (uint n = 0; n < max(p.octaves, 1u); ++n) {
        const float phase = isotropic ? 1.0 : hg(cosTheta, g);
        const float t = (pow(max(behind, 0.0), extinction) + 4.0 * pow(max(centre, 0.0), extinction) +
                         pow(max(ahead, 0.0), extinction)) / 6.0;
        sum += energy * phase * t;
        energy *= p.octaveEnergy;
        extinction *= p.octaveExtinction;
        g *= p.octaveAnisotropy;
    }
    return sum;
}

/// What reaches a point inside the medium from light `slot`: the read every
/// receiver does -- the surfaces' transmittance with their lit side, times
/// the medium's own at the point's depth (splat_shadow_read's shadowTap).
float octaveSum(float transmittance, float cosTheta, bool isotropic) {
    return octaveSum3(transmittance, transmittance, transmittance, cosTheta, isotropic);
}

float shadowFor(uint slot, float3 x) {
    if (shadeParams.hasMap == 0 || slot >= kShadowSlots) {
        return 1.0;
    }
    return shadowMapRead<VolumeMapSource>(shadowFrameOf<VolumeMapSource>(slot), slot, x);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void volumeSplatShade(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    const VolumeShadeParams p = shadeParams;
    if (i >= p.count) {
        return;
    }
    const float3 local = positions[i].xyz;
    const float3 x = float3(dot(float3(p.w00, p.w01, p.w02), local) + p.w03,
                            dot(float3(p.w10, p.w11, p.w12), local) + p.w13,
                            dot(float3(p.w20, p.w21, p.w22), local) + p.w23);
    const uint2 medium = uint2(volume[i * 2], volume[i * 2 + 1]);
    const float3 own = float3(float(medium.y & 255u), float((medium.y >> 8) & 255u), float((medium.y >> 16) & 255u)) /
                       255.0;
    const float3 albedo = saturate(own * p.albedo);
    const float3 wo = normalize(p.eye - x);
    // The gaussian's deviations in the world: the object's linear part times
    // R S (its columns the axes, scaled).
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float4 q = decodeQuaternion(shape[row]);
    const float3x3 rot = float3x3(
        1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.w * q.z), 2.0 * (q.x * q.z + q.w * q.y),
        2.0 * (q.x * q.y + q.w * q.z), 1.0 - 2.0 * (q.x * q.x + q.z * q.z), 2.0 * (q.y * q.z - q.w * q.x),
        2.0 * (q.x * q.z - q.w * q.y), 2.0 * (q.y * q.z + q.w * q.x), 1.0 - 2.0 * (q.x * q.x + q.y * q.y));
    const float3x3 objectLinear = float3x3(p.w00, p.w01, p.w02, p.w10, p.w11, p.w12, p.w20, p.w21, p.w22);
    const float3x3 deviations = mul(objectLinear, float3x3(rot[0] * scale, rot[1] * scale, rot[2] * scale));

    // The dome directions the map holds, read once: every dome shares them.
    float domeMean = 1.0;
    if (p.hasMap != 0) {
        float sum = 0.0;
        uint taken = 0;
        for (uint k = 0; k < kShadowSlots; ++k) {
            if (VolumeMapSource.headerCount(k * kShadowFrameWords + 19) == 2u) {
                sum += shadowFor(k, x);
                taken += 1;
            }
        }
        domeMean = taken > 0 ? sum / float(taken) : 1.0;
    }

    float3 scattered = float3(0.0);
    for (uint k = 0; k < p.lightCount; ++k) {
        const LightRecord l = lights[k];
        if (!lightLinked(l.lightCategory, p.categoriesLo, p.categoriesHi)) {
            continue;
        }
        const bool shadows = (l.flags & kLightShadow) != 0;
        if (l.kind == kLightDome) {
            // A dome's colour (a textured one's mean is not read here), from
            // every direction, isotropic: the phase integrates to one.
            scattered += lightEmission(l) * octaveSum(shadows ? domeMean : 1.0, 1.0, true);
            continue;
        }
        if (l.kind == kLightDistant) {
            const float3 wi = lightAxisZ(l);
            const float3 irradiance = lightEmission(l) * (distantIsDelta(l) ? 1.0 : distantSizeFactor(l));
            // The gaussian's deviation along the light: |M^T wi| for M the
            // world's linear part times R S.
            const float step = 1.7320508 * length(mul(wi, deviations));
            const float t = shadows ? shadowFor(k, x) : 1.0;
            const float behind = shadows ? shadowFor(k, x - wi * step) : 1.0;
            const float ahead = shadows ? shadowFor(k, x + wi * step) : 1.0;
            scattered += irradiance * octaveSum3(behind, t, ahead, dot(wi, -wo), false);
            continue;
        }
        // A light that stands somewhere: four stratified samples of its
        // surface, each its radiance over its density.
        const float t = shadows ? shadowFor(k, x) : 1.0;
        float3 sum = float3(0.0);
        for (uint s = 0; s < 4; ++s) {
            const float2 u = float2((float(s & 1u) + 0.5) * 0.5, (float(s >> 1u) + 0.5) * 0.5);
            const LightSample ls = sampleLight(l, x, wo, u);
            if (!ls.valid) {
                continue;
            }
            const float weight = ls.delta || !(ls.pdf > 0.0) ? 1.0 : 1.0 / ls.pdf;
            sum += ls.radiance * weight * octaveSum(t, dot(ls.wi, -wo), false);
        }
        scattered += 0.25 * sum;
    }
    const float temperature = lowHalf(medium.x >> 16);
    float3 glow = p.emission;
    if (p.emissionScale > 0.0 && temperature > 0.0) {
        glow += p.emissionScale * blackbodyAt(p.temperatureOffset + p.temperatureScale * temperature);
    }
    const float3 radiance = max(albedo * scattered + (float3(1.0) - albedo) * glow, float3(0.0));
    radianceOut[i * 2] = packHalves(radiance.x, radiance.y);
    radianceOut[i * 2 + 1] = packHalves(radiance.z, 0.0);
}
