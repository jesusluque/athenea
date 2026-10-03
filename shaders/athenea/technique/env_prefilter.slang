// Copyright (c) 2026 jesus luque.
//
// A DOME'S SKY CONVOLVED WITH A GGX LOBE, ONE LEVEL A ROUGHNESS.
//
// Nine harmonics hold what a diffuse surface takes from a sky; a reflection
// takes the sky itself, and how much of it depends on how rough the surface
// is. So the image is convolved once per roughness and kept in an octahedral
// map, which a kernel with no texture table can read out of a buffer
// (athenea/light/environment.slang has the layout and the reader).
//
// The lobe is sampled as the surface lobes sample it -- `ggxSampleVisibleNormal`
// from the material module -- with the normal, the view and the reflected
// direction taken as one, which is the approximation every prefiltered
// environment makes and the reason a grazing reflection stretches in a real
// renderer and does not here.
//
// A sample reads a level of the source's own chain rather than its finest:
// one texel of a 4k sky, seen through a wide lobe with sixty-four samples, is
// a firefly that never averages out. The level is chosen by how much solid
// angle a sample stands for against the solid angle of a texel.
module env_prefilter;
import athenea.light.lights_image;
import athenea.light.environment;
import athenea.material.microfacet;

struct EnvPrefilterParams {
    uint  domes;
    uint  level;       ///< which roughness level is being written
    uint  samples;     ///< directions a texel takes; 1 at level 0
    uint  sourceLevel; ///< the coarsest source level a sample may read
    uint  baseSide;    ///< the octahedral side of level 0, this sky's own
    float roughness;   ///< of this level
    float pad0; float pad1;
};

StructuredBuffer<LightRecord>    lights;
// `LightTable::bind` sets this beside the records; a name it writes must
// exist here even where the count is not what this kernel walks.
uniform uint                  lightCount;
StructuredBuffer<uint>           domeLights;
RWStructuredBuffer<uint>         texelsOut;
ConstantBuffer<EnvPrefilterParams> params;

static const float kPrefilterPi = 3.14159265358979;

/// A texel of the source, along `dir`, at `level`.
float3 skyAt(LightRecord light, float3 dir, float level) {
    if (light.texture == 0xFFFFFFFFu) {
        return float3(1.0);
    }
    const TextureRecord record = gTextures.records[light.texture];
    return gTextures.textures[record.slot]
        .SampleLevel(gTextures.samplers[light.sampler], domeUv(light, dir), level)
        .rgb;
}

/// An orthonormal frame about `z`. (Duff et al., the same construction the
/// light module uses, repeated so this kernel imports no more than it reads.)
void frameAbout(float3 z, out float3 x, out float3 y) {
    const float sign = z.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (sign + z.z);
    const float b = z.x * z.y * a;
    x = float3(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x);
    y = float3(b, sign + z.y * z.y * a, -z.y);
}

/// Hammersley, for a sequence with no state.
float2 hammersley(uint i, uint n) {
    uint bits = i;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555u) << 1) | ((bits & 0xAAAAAAAAu) >> 1);
    bits = ((bits & 0x33333333u) << 2) | ((bits & 0xCCCCCCCCu) >> 2);
    bits = ((bits & 0x0F0F0F0Fu) << 4) | ((bits & 0xF0F0F0F0u) >> 4);
    bits = ((bits & 0x00FF00FFu) << 8) | ((bits & 0xFF00FF00u) >> 8);
    return float2(float(i) / float(max(n, 1u)), float(bits) * 2.3283064365386963e-10);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void envPrefilter(uint3 tid: SV_DispatchThreadID) {
    const uint side = envSideOf(params.level, params.baseSide);
    const uint perDome = side * side;
    const uint at = tid.x;
    const uint dome = at / perDome;
    const uint texel = at % perDome;
    if (dome >= params.domes) {
        return;
    }
    const LightRecord light = lights[domeLights[dome]];
    const uint2 uv = uint2(texel % side, texel / side);
    const float3 normal = octDecode((float2(uv) + 0.5) / float(side));

    // How wide a texel of the source is, in solid angle: every level's choice
    // of which level of the source to read is measured against it.
    float sourceTexels = 1.0;
    if (light.texture != 0xFFFFFFFFu) {
        const TextureRecord record = gTextures.records[light.texture];
        sourceTexels = float(uint(record.width) * uint(record.height));
    }
    const float texelSolidAngle = 4.0 * kPrefilterPi / max(sourceTexels, 1.0);

    float3 sum = float3(0.0);
    if (params.level == 0 || params.samples <= 1) {
        // A MIRROR TAKES THE SKY AS IT IS -- at the level its own texels can
        // hold. Where the map is as wide as the sky this is level 0 and
        // nothing is lost; where the sky is wider than the map is allowed to
        // be, reading level 0 would alias what a texel cannot resolve, so the
        // same solid-angle rule the rough levels use picks the level here.
        const float mapSolidAngle = 4.0 * kPrefilterPi / float(side * side);
        const float level = clamp(0.5 * log2(mapSolidAngle / texelSolidAngle), 0.0,
                                  float(params.sourceLevel));
        sum = skyAt(light, normal, level);
    } else {
        const float alpha = max(params.roughness * params.roughness, 1.0e-3);
        const float2 alpha2 = float2(alpha, alpha);
        float3 tangent, bitangent;
        frameAbout(normal, tangent, bitangent);
        float weight = 0.0;
        for (uint s = 0; s < params.samples; ++s) {
            const float2 u = hammersley(s, params.samples);
            // The view is the normal: what every prefiltered environment
            // assumes, since a map indexed by a direction has no view to keep.
            const float3 h = ggxSampleVisibleNormal(float3(0.0, 0.0, 1.0), alpha2, u);
            const float3 hWorld = normalize(tangent * h.x + bitangent * h.y + normal * h.z);
            const float3 wi = reflect(-normal, hWorld);
            const float nDotL = dot(normal, wi);
            if (nDotL <= 0.0) {
                continue;
            }
            const float pdf = max(ggxVisibleNormalPdf(float3(0.0, 0.0, 1.0), h, alpha2) /
                                      (4.0 * max(h.z, 1.0e-6)),
                                  1.0e-9);
            const float sampleSolidAngle = 1.0 / (float(params.samples) * pdf);
            const float level = clamp(0.5 * log2(sampleSolidAngle / texelSolidAngle), 0.0,
                                      float(params.sourceLevel));
            sum += skyAt(light, wi, level) * nDotL;
            weight += nDotL;
        }
        sum = weight > 0.0 ? sum / weight : skyAt(light, normal, float(params.sourceLevel));
    }
    sum *= lightEmission(light);

    const uint word = envWordAt(dome, params.level, uv, params.baseSide);
    texelsOut[word] = (f32tof16(sum.g) << 16) | f32tof16(sum.r);
    texelsOut[word + 1] = f32tof16(sum.b);
}
