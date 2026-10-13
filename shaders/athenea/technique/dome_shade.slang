// Copyright (c) 2026 jesus luque.
//
// A DOME ON A MESH, READ PREFILTERED: NO SAMPLE, NO RAY, NO GRAIN (task PLAY-G).
//
// One dome sample a pixel with a shadow ray is what made the Corvette's
// ground grain (std/mean 0.82 at 1080p) and cost 395 ms, every further sample
// a hundred more. A surface whose lobes all reflect is lit by the dome in
// closed form instead:
//
//   the diffuse lobes   by the irradiance the sky's nine harmonics give at the
//                       normal, plus the sun `env_sun` took out of them
//   the glossy lobes    by the dome's own image along the mirror direction,
//                       read at the mip whose texels are as wide as the lobe
//
// each weighed by its directional albedo -- the split sum. The shadow is the
// cloud's, from the map's dome directions, each read softened by the width of
// sky it stands for (`shadowMapReadSoft`). What it cannot see is a mesh
// shadowing another mesh from the dome: there is no ray.
//
// A KERNEL OF ITS OWN, after the shading kernel. Written inside it the read
// overflowed what Metal keeps of a thread -- rows of garbage across the
// Corvette's ground, and a pipeline that would not compile without the cloud
// map -- so the shading kernel only leaves what the lobe stack says of a
// pixel (the albedos, the roughness, the normal, where it stands) in three
// textures, and this kernel, which holds no lobe stack at all, adds the domes.
module dome_shade;
import athenea.light.lights_image;
import athenea.light.environment;
import athenea.technique.camera;
import athenea.technique.splat_shadow_read;
import athenea.common.octahedral;

struct DomeShadeParams {
    uint  clouds;      ///< 1: the cloud shadow map is bound
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

struct DomeViewToWorld {
    float4 row0;
    float4 row1;
    float4 row2;
};

StructuredBuffer<LightRecord>       lights;
uniform uint                        lightCount;
ConstantBuffer<CameraParams>        camera;
ConstantBuffer<DomeViewToWorld>     toWorld;
ConstantBuffer<DomeShadeParams>     params;
/// What the shading kernel left of a pixel whose domes it did not light:
/// (diffuse albedo, coverage), (glossy albedo, GGX alpha) and (where it
/// stands, its normal octahedral in 16 bits a coordinate). Coverage 0: the
/// shading kernel lit the domes itself, or nothing is there.
Texture2D<float4>                   domeDiffuse;
Texture2D<float4>                   domeGlossy;
Texture2D<float4>                   domePoint;
/// Environment::meshView: twelve texels a dome (env_mesh.slang).
Texture2D<float4>                   domeLighting;
Texture2DArray<float>               cloudShadow;
Texture2DArray<float>               cloudShadowChain;
RWStructuredBuffer<float4>          colour;
RWStructuredBuffer<float4>          groupColour;   // groupCount planes, a pixel each
uniform uint                        groupCount;

static const uint  kDomeRows = 4;      // kEnvironmentDomes
static const uint  kNoDomeRow = 0xFFFFFFFFu;
/// How wide a region of the sky one of the map's six dome directions stands
/// for: the tangent of half the angle between two of them, about thirty
/// degrees. It is the source width the cloud's shadow is softened by.
static const float kDomeSpread = 0.58;
static const float kDomePi = 3.14159265358979;

struct DomeCloudShadow : IShadowMapSource {
    static float headerReal(uint word) { return cloudShadow.Load(shadowHeaderTexel(word)); }
    static uint headerCount(uint word) { return uint(cloudShadow.Load(shadowHeaderTexel(word))); }
    static float texel(ShadowMapFrame f, uint light, int x, int y, uint which) {
        return cloudShadow.Load(int4(x, y, int(shadowLayerOf(light, f.coefficients, which)), 0));
    }
    static uint levels(ShadowMapFrame f) { return shadowChainLevels(f.resolution); }
    static float transmittanceAt(ShadowMapFrame f, uint light, uint level, int x, int y) {
        return cloudShadowChain.Load(int4(x, y, int(light), int(level)));
    }
};

float3 domeApply(float4 row0, float4 row1, float4 row2, float3 x, float w) {
    return float3(dot(row0.xyz, x) + row0.w * w, dot(row1.xyz, x) + row1.w * w, dot(row2.xyz, x) + row2.w * w);
}

/// The row prepared for light k, or kNoDomeRow.
uint domeRowOf(uint k) {
    for (uint r = 0; r < kDomeRows; ++r) {
        if (uint(domeLighting.Load(int3(11, int(r), 0)).x + 0.5) == k + 1u) {
            return r;
        }
    }
    return kNoDomeRow;
}

/// The sky's radiance along d, from the row's nine coefficients.
float3 domeShRadiance(uint row, float3 d) {
    float3 sum = float3(0.0);
    for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
        sum += domeLighting.Load(int3(int(k), int(row), 0)).rgb * envBasisValue(k, d);
    }
    return sum;
}

/// The irradiance at a surface facing n, from the same coefficients.
float3 domeShIrradiance(uint row, float3 n) {
    float3 sum = float3(0.0);
    for (uint k = 0; k < kEnvIrradianceCoefficients; ++k) {
        sum += domeLighting.Load(int3(int(k), int(row), 0)).rgb * (envBandWeight(k) * envBasisValue(k, n));
    }
    return max(sum, float3(0.0));
}

/// The sky along `dir` as a lobe of GGX roughness `alpha` sees it: the
/// dome's own image at the mip whose texels are as wide as the lobe
/// (4 pi alpha^2 of solid angle against 4 pi / (w h) a texel).
float3 domeSkyBlurred(LightRecord l, float3 dir, float alpha) {
    if (l.texture == 0xFFFFFFFFu) {
        return lightEmission(l);
    }
    const TextureRecord record = gTextures.records[l.texture];
    if ((record.flags & 1u) == 0u) {
        return lightEmission(l);
    }
    const float side = sqrt(max(record.width * record.height, 1.0));
    const float top = ceil(log2(max(max(record.width, record.height), 1.0)));
    const float level = alpha > 1.0e-3 ? clamp(log2(alpha * side), 0.0, top) : 0.0;
    return lightEmission(l) *
           gTextures.textures[record.slot].SampleLevel(gTextures.samplers[l.sampler], domeUv(l, dir), level).rgb;
}

/// Slot k's map at p, softened by a source `spread` wide.
float domeCloudThrough(float3 p, float3 dx, float3 dy, uint k, float spread) {
    if (params.clouds == 0 || k >= kShadowSlots) {
        return 1.0;
    }
    return shadowMapReadSoft<DomeCloudShadow>(shadowFrameOf<DomeCloudShadow>(k), k, p, dx, dy, spread);
}

/// The way slot k looks toward the sky, where it is one of a dome's
/// directions; nothing otherwise.
float3 domeSlotDirection(uint k) {
    const uint at = k * kShadowFrameWords;
    if (DomeCloudShadow.headerCount(at + 19) != 2u) {
        return float3(0.0);
    }
    return -float3(DomeCloudShadow.headerReal(at + 12), DomeCloudShadow.headerReal(at + 13),
                   DomeCloudShadow.headerReal(at + 14));
}

uint domeSlotNearest(float3 wi) {
    return params.clouds != 0 ? shadowDomeSlotOf<DomeCloudShadow>(wi) : kShadowSlots;
}

/// How much of the dome's sky over a surface facing n the clouds let
/// through: each of the map's dome directions above the surface, read soft,
/// weighed by its cosine and by how bright the sky is there.
float domeSkyThrough(uint row, float3 p, float3 dx, float3 dy, float3 n) {
    if (params.clouds == 0) {
        return 1.0;
    }
    float through = 0.0;
    float total = 0.0;
    for (uint k = 0; k < kShadowSlots; ++k) {
        const float3 towards = domeSlotDirection(k);
        const float facing = dot(n, towards);
        if (!(facing > 0.0)) {
            continue;
        }
        const float bright = max(dot(domeShRadiance(row, towards), float3(0.2126, 0.7152, 0.0722)), 1.0e-6);
        const float w = facing * bright;
        through += w * domeCloudThrough(p, dx, dy, k, kDomeSpread);
        total += w;
    }
    return total > 0.0 ? through / total : 1.0;
}

/// Where the ray through a pixel's neighbour meets the plane at p.
float3 domeOnPlane(float3 origin, float3 direction, float3 p, float3 n) {
    const float facing = dot(direction, n);
    if (!(abs(facing) > 1.0e-6)) {
        return p;
    }
    const float t = dot(p - origin, n) / facing;
    return t > 0.0 ? origin + direction * t : p;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void domeShade(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const int3 texel = int3(int(tid.x), int(tid.y), 0);
    const float4 diffuse = domeDiffuse.Load(texel);
    const float coverage = diffuse.w;
    if (!(coverage > 0.0)) {
        return;
    }
    const float4 glossy = domeGlossy.Load(texel);
    const float4 at = domePoint.Load(texel);
    const float3 p = at.xyz;
    const uint packed = asuint(at.w);
    const float3 n = octDecode(float2(float(packed & 0xFFFFu), float(packed >> 16u)) / 65534.0);
    // The eye, and how far a pixel spans on the surface: the footprint the
    // cloud map is read at, by ray differentials as the shading kernel does.
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    const float3 look = normalize(domeApply(toWorld.row0, toWorld.row1, toWorld.row2, direction, 0.0));
    const float3 wo = -look;
    viewRay(camera, float2(tid.xy) + float2(1.5, 0.5), origin, direction);
    const float3 dx = domeOnPlane(domeApply(toWorld.row0, toWorld.row1, toWorld.row2, origin, 1.0),
                                  domeApply(toWorld.row0, toWorld.row1, toWorld.row2, direction, 0.0), p, n) - p;
    viewRay(camera, float2(tid.xy) + float2(0.5, 1.5), origin, direction);
    const float3 dy = domeOnPlane(domeApply(toWorld.row0, toWorld.row1, toWorld.row2, origin, 1.0),
                                  domeApply(toWorld.row0, toWorld.row1, toWorld.row2, direction, 0.0), p, n) - p;
    const uint pixels = camera.width * camera.height;
    const uint index = tid.y * camera.width + tid.x;
    float3 sum = float3(0.0);
    for (uint k = 0; k < lightCount; ++k) {
        const LightRecord light = lights[k];
        if (light.kind != kLightDome) {
            continue;
        }
        const uint row = domeRowOf(k);
        if (row == kNoDomeRow) {
            // A dome past the fourth was not prepared: its image's mean, the
            // one thing that needs no preparing, for every lobe.
            const float3 mean = domeSkyBlurred(light, n, 1.0e6);
            const float3 flat = (diffuse.rgb + glossy.rgb) * mean * coverage;
            sum += flat;
            if (light.group != 0 && light.group <= groupCount && light.group <= 8) {
                groupColour[(light.group - 1) * pixels + index] += float4(flat, 0.0);
            }
            continue;
        }
        // Light linking was the shading kernel's to ask: it marks a pixel
        // only where every dome it reads prefiltered reaches it (diffuse.w).
        const bool casts = (light.flags & kLightShadow) != 0;
        float3 arrived = float3(0.0);
        if (any(diffuse.rgb > float3(0.0))) {
            float3 e = domeShIrradiance(row, n) * (casts ? domeSkyThrough(row, p, dx, dy, n) : 1.0);
            const float4 sunAt = domeLighting.Load(int3(9, int(row), 0));
            if (sunAt.w > 0.0) {
                const float facing = max(dot(n, sunAt.xyz), 0.0);
                if (facing > 0.0) {
                    const float through =
                        casts ? domeCloudThrough(p, dx, dy, domeSlotNearest(sunAt.xyz), kDomeSpread) : 1.0;
                    e += domeLighting.Load(int3(10, int(row), 0)).rgb * (facing * through);
                }
            }
            arrived += diffuse.rgb * e * (1.0 / kDomePi);
        }
        if (any(glossy.rgb > float3(0.0))) {
            const float3 mirror = reflect(-wo, n);
            const float alpha = glossy.w;
            const float through =
                casts ? domeCloudThrough(p, dx, dy, domeSlotNearest(mirror), max(alpha, 0.05)) : 1.0;
            arrived += glossy.rgb * domeSkyBlurred(light, mirror, alpha) * through;
        }
        arrived *= coverage;
        sum += arrived;
        if (light.group != 0 && light.group <= groupCount && light.group <= 8) {
            groupColour[(light.group - 1) * pixels + index] += float4(arrived, 0.0);
        }
    }
    colour[index] += float4(sum, 0.0);
}
