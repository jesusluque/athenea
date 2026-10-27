// Copyright (c) 2026 jesus luque.
#include "athenea/technique/MaterialShading.h"

#include "athenea/core/Log.h"

#include <algorithm>
#include <array>
#include <chrono>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::technique {

namespace {

const char* kKernelPrelude = R"(
import athenea.common.octahedral;
import athenea.light.lights_image;
import athenea.light.light_bvh;
import athenea.light.mesh_lights;
import athenea.light.ltc;
import athenea.rt.bvh_common;

struct LightingParams {
    uint  samples;
    uint  chooseLights;
    uint  shadowCutouts;   // 1: a material of the frame cuts, and a shadow ray asks what it meets
    uint  meshSamples;     // samples a pixel through the mesh lights' tree (MeshLights); 0: none
};

Texture2D<uint4>              visibility;   // (instance + 1, triangle); row 0 on top
RWStructuredBuffer<float4>    colour;
RWStructuredBuffer<float>     depth;
ConstantBuffer<CameraParams>  camera;
StructuredBuffer<LightRecord> lights;
uniform uint                  lightNodeBase;
uniform uint                  lightTreeNodes;
uniform uint                  lightUnboundedCount;
uniform uint                  lightCount;
uniform uint                  meshLightWords;   // the mesh lights' region in iesValues; 0: none
ConstantBuffer<LightingParams> lighting;

/// Two numbers for the i-th sample of light k at a pixel. Stratification and
/// a frame's worth of decorrelation are the path tracer's (M6); here the
/// samples only have to be spread.
uint pcgHashShade(uint input) {
    const uint state = input * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Pixel, light, sample and dimension folded through the hash in turn -- the
/// path tracer's construction. The second number used to be one LCG step of
/// the first, which ties the pair to a lattice.
float2 sampleAt(uint2 pixel, uint light, uint index) {
    uint key = pcgHashShade(pixel.y * 65536u + pixel.x);
    key = pcgHashShade(key + light);
    key = pcgHashShade(key + index);
    const uint a = pcgHashShade(key + 0u);
    const uint b = pcgHashShade(key + 1u);
    return float2(float(a >> 8) * (1.0 / 16777216.0), float(b >> 8) * (1.0 / 16777216.0));
}

/// Lobe samples a pixel draws toward the lights at infinity: one a light
/// sample, up to this many.
static const uint kLobeLookups = 32;
/// Shadow bits a pixel's light samples take: the lobe samples' come after.
uint lightSampleBits() {
    if (lightCount == 0u) {
        return 0u;
    }
    const uint samples = max(lighting.samples, 1u);
    return lighting.chooseLights != 0 ? samples : lightCount * samples;
}
uint lobeLookups() {
    return lightCount != 0 ? min(max(lighting.samples, 1u), kLobeLookups) : 0u;
}
/// THE MESH LIGHTS' SAMPLES (technique::MeshLights): this many a pixel, each a
/// cluster through the tree, a triangle by its share and a point on it. Their
/// shadow bits come after the light samples' and the lobe samples', whatever
/// number of lobe samples a pixel drew.
uint meshSampleCount() {
    return meshClusterCount(iesValues, meshLightWords) > 0u ? lighting.meshSamples : 0u;
}
uint meshSampleBit(uint i) {
    return lightSampleBits() + lobeLookups() + i;
}
/// THE MESH LIGHTS IN CLOSED FORM (meshAnalytic): the tile's lightcut
/// (mesh_light_cut.slang), the LTC table and the filtered shadow ratio.
uniform uint              meshAnalytic;
Texture2D<uint>           meshCut;
uniform uint              meshCutK;
Texture2D<float4>         ltcTable;
Texture2D<float>          meshShadow;
uniform uint              meshShadowOn;
static const uint kMeshRatioSamples = 2u;

/// The table's texel for a GGX alpha and the cosine to the eye, bilinear
/// between the four around it.
float4 ltcAt(float alpha, float cosTheta) {
    const uint size = 64u;
    const float2 uv = ltcCoordinates(alpha, cosTheta, size) * float(size) - 0.5;
    const int2 i0 = clamp(int2(floor(uv)), int2(0), int2(int(size) - 1));
    const int2 i1 = min(i0 + 1, int2(int(size) - 1));
    const float2 f = saturate(uv - float2(i0));
    const float4 a = lerp(ltcTable.Load(int3(i0.x, i0.y, 0)), ltcTable.Load(int3(i1.x, i0.y, 0)), f.x);
    const float4 b = lerp(ltcTable.Load(int3(i0.x, i1.y, 0)), ltcTable.Load(int3(i1.x, i1.y, 0)), f.x);
    return lerp(a, b, f.y);
}

/// What one lobe returns of a rectangle of unit radiance, its weight in it:
/// a diffuse lobe by the rectangle's form factor, a GGX one by the LTC
/// fitted to it times its albedo in reflection, Zeltner's sheen by its own
/// LTC. Transmission and hair are left out (the plan's: they stay sampled).
float3 lobeRectangle(Lobe lobe, float3 toEye, float3 p0, float3 p1, float3 p2, float3 p3, bool twoSided,
                     bool diffuseOnly) {
    float3 n = lobe.normal;
    if (dot(n, toEye) < 0.0) {
        n = -n;
    }
    const float3x3 frame = ltcFrame(n, toEye);
    const float cosEye = saturate(dot(n, toEye));
    switch (lobe.kind) {
    case kLobeOrenNayar:
    case kLobeBurley:
        return lobe.weight * lobeAlbedo(lobe, toEye) * ltcQuad(frame, p0, p1, p2, p3, twoSided) / kPi;
    case kLobeTranslucent:
        return lobe.weight * lobe.colour0 * ltcQuad(ltcFrame(-n, toEye), p0, p1, p2, p3, twoSided) / kPi;
    case kLobeSheen: {
        if ((lobe.flags & kFlagZeltner) == 0u || diffuseOnly) {
            return lobe.weight * lobeAlbedo(lobe, toEye) * ltcQuad(frame, p0, p1, p2, p3, twoSided) / kPi;
        }
        const float r = clamp(lobe.roughness, 0.01, 1.0);
        const float aInv = zeltnerLtcAInv(cosEye, r);
        const float bInv = zeltnerLtcBInv(cosEye, r);
        const float3x3 m = mul(ltcInverse(float4(aInv, bInv, aInv, 0.0)), frame);
        return lobe.weight * lobeAlbedo(lobe, toEye) * ltcQuad(m, p0, p1, p2, p3, twoSided) / kPi;
    }
    case kLobeConductor:
    case kLobeSchlick:
    case kLobeDielectric: {
        if (diffuseOnly || lobe.scatter == kScatterTransmit) {
            return float3(0.0);
        }
        Lobe reflecting = lobe;
        reflecting.scatter = kScatterReflect;
        const float alpha = clamp(averageAlpha(max(lobe.alpha, float2(1.0e-4))), 4.0e-4, 1.0);
        const float3x3 m = mul(ltcInverse(ltcAt(alpha, max(cosEye, 1.0e-3))), frame);
        return lobe.weight * lobeAlbedo(reflecting, toEye) * ltcQuad(m, p0, p1, p2, p3, twoSided) / kPi;
    }
    default:
        return float3(0.0);
    }
}

/// The mesh lights at a pixel in closed form: each entry of its tile's cut.
/// A cluster is integrated over its rectangle; a node of the tree that
/// stands for many is a Lambertian point at its box's middle, its power that
/// of everything under it, its colour and facing its most important
/// cluster's, and only the diffuse lobes see it. Unshadowed: the caller
/// multiplies by the ratio.
float3 meshClosedForm(uint2 tid, float3 p, float3 n, float3 toEye) {
    float3 sum = float3(0.0);
    const uint base = meshWord(iesValues, kMeshNodeBase);
    const uint2 tile = tid / 16u;
    for (uint e = 0u; e < meshCutK; ++e) {
        const uint entry = meshCut.Load(int3(int(tile.x * meshCutK + e), int(tile.y), 0));
        if (entry == kMeshNone) {
            break;
        }
        const LightNode top = lightNodeAt(iesValues, base, entry);
        LightNode node = top;
        for (uint step = 0u; step < 64u && node.links.w == 0u; ++step) {
            const LightNode a = lightNodeAt(iesValues, base, node.links.x);
            const LightNode b = lightNodeAt(iesValues, base, node.links.y);
            node = lightImportance(a, p, n) + 1.0e-6 * a.lo.w >= lightImportance(b, p, n) + 1.0e-6 * b.lo.w ? a : b;
        }
        const MeshCluster m = meshCluster(iesValues, node.links.x & ~kLeafBit);
        if (!(m.power > 0.0)) {
            continue;
        }
        const bool twoSided = (m.flags & kLightTwoSided) != 0u;
        if (top.links.w == 0u) {
            // A node: E = P cos(emitter) cos(receiver) / d^2, P its rgb power.
            const float3 centre = 0.5 * (top.lo.xyz + top.hi.xyz);
            const float3 offset = centre - p;
            const float radius2 = 0.25 * dot(top.hi.xyz - top.lo.xyz, top.hi.xyz - top.lo.xyz);
            const float d2 = max(dot(offset, offset), radius2);
            const float distance = sqrt(max(dot(offset, offset), 1.0e-12));
            const float3 wi = offset / distance;
            // The node is a box, not a point: the receiver's cosine is taken
            // with the angle the box subtends allowed for (Conty and Kulla's
            // bound), so a node straddling a surface's horizon still lights it.
            const float spread = asin(saturate(sqrt(radius2) / distance));
            const float facing = dot(m.normal, -wi);
            const float emitter = twoSided ? abs(facing) : max(facing, 0.0);
            const float3 power = m.radiance * (4.0 * m.halfX * m.halfY) * (top.lo.w / m.power);
            float3 albedo = float3(0.0);
            for (uint k = 0u; k < gAtheneaResult.count; ++k) {
                const Lobe lobe = gAtheneaResult.lobes[k];
                if (lobe.kind == kLobeOrenNayar || lobe.kind == kLobeBurley ||
                    (lobe.kind == kLobeSheen && (lobe.flags & kFlagZeltner) == 0u)) {
                    const float receiver = dot(lobe.normal, wi) * (dot(lobe.normal, toEye) < 0.0 ? -1.0 : 1.0);
                    albedo += lobe.weight * lobeAlbedo(lobe, toEye) *
                              cos(min(max(acos(clamp(receiver, -1.0, 1.0)) - spread, 0.0), 0.5 * kPi));
                }
            }
            sum += albedo / kPi * power * emitter / d2;
            continue;
        }
        const float3 x = m.axisX * m.halfX;
        const float3 y = m.axisY * m.halfY;
        const float3 q = m.centre - p;
        const float3 p0 = q - x - y;
        const float3 p1 = q + x - y;
        const float3 p2 = q + x + y;
        const float3 p3 = q - x + y;
        float3 response = float3(0.0);
        // The stack where the material left it, read in place.
        for (uint k = 0u; k < gAtheneaResult.count; ++k) {
            response += lobeRectangle(gAtheneaResult.lobes[k], toEye, p0, p1, p2, p3, twoSided, false);
        }
        sum += response * m.radiance;
    }
    return sum;
}

/// How far a shadow ray toward a point on an emitter goes: short of the
/// emitter itself by more than `occluded` moves the ray's origin (along the
/// normal and along the ray, 1e-3 of the scale each), or the emitting
/// triangle is its own occluder and every sample is in shadow.
float meshShadowReach(float3 p, float distance) {
    return max(distance - 5.0e-3 * max(1.0, length(p)), 0.0);
}

/// The numbers mesh light sample i draws with: a pick and a point.
float meshPick(uint2 pixel, uint i) { return sampleAt(pixel, 0xA000u, i).x; }
float3 meshPoint(uint2 pixel, uint i) {
    const float2 a = sampleAt(pixel, 0xA001u, i);
    const float2 b = sampleAt(pixel, 0xA002u, i);
    return float3(a, b.x);
}

/// TRANSPARENT OPACITY (UsdPreviewSurface 2.6, opacityMode transparent),
/// drawn as the path tracer draws it: the visibility pass kept this sample
/// with probability p = max(opacity, 1/20) (materialCuts), so the specular
/// and the emission are worth 1/p and the diffuse opacity/p -- in
/// expectation the whole specular, opacity of the diffuse, and what is
/// behind where the lot passed. Drawn as coverage instead, a feather card's
/// clear texels reflected nothing here and a glass sheet's worth under the
/// tracer, and the two routes drew two different birds. Applied to the
/// stack evaluateMaterial left, in every kernel that reads it.
void weighTransparent(MaterialRecord m) {
    if ((m.flags & kMaterialTransparent) == 0 || !(gAtheneaResult.opacity < 1.0)) {
        return;
    }
    const float o = max(gAtheneaResult.opacity, 0.0);
    const float kept = max(o, kTransparentKeep);
    for (uint k = 0; k < gAtheneaResult.count; ++k) {
        const uint kind = gAtheneaResult.lobes[k].kind;
        const bool diffuse = kind == kLobeOrenNayar || kind == kLobeBurley || kind == kLobeTranslucent;
        gAtheneaResult.lobes[k].weight *= (diffuse ? o : 1.0) / kept;
    }
    gAtheneaResult.emission /= kept;
}

/// WHETHER EVERY LOBE OF THE STACK ONLY REFLECTS, as evaluateMaterial left
/// it: what a dome read prefiltered can answer (kDomePrefilter). Read in
/// place, never copied -- the stack is a kilobyte of thread memory.
bool stackReflectsOnly() {
    for (uint k = 0; k < gAtheneaResult.count; ++k) {
        const uint kind = gAtheneaResult.lobes[k].kind;
        if (kind == kLobeDielectric || kind == kLobeSchlick) {
            if (gAtheneaResult.lobes[k].scatter != kScatterReflect) {
                return false;
            }
        } else if (kind != kLobeOrenNayar && kind != kLobeBurley && kind != kLobeSheen && kind != kLobeConductor) {
            return false;
        }
    }
    return true;
}
/// Whether every dome of the frame reaches an instance of these categories:
/// a pixel reads its domes prefiltered only then, since the kernel that adds
/// them (dome_shade.slang) does not ask.
bool domesReach(uint lo, uint hi) {
    for (uint k = 0; k < lightCount; ++k) {
        if (lights[k].kind == kLightDome && !lightLinked(lights[k].lightCategory, lo, hi)) {
            return false;
        }
    }
    return true;
}
/// Whether a distant light is among the frame's: the one light at infinity
/// a lobe sample still meets where the domes are read prefiltered.
bool anyDistantLight() {
    for (uint k = 0; k < lightCount; ++k) {
        if (lights[k].kind == kLightDistant) {
            return true;
        }
    }
    return false;
}

/// A lobe sample's direction, as the lobe kernel writes it for the shadow
/// kernel: octahedral, 16 bits a coordinate; all ones for no sample.
static const uint kNoLobe = 0xFFFFFFFFu;
uint packLobe(float3 wi) {
    const float2 e = saturate(octEncode(wi));
    return uint(e.x * 65534.0 + 0.5) | (uint(e.y * 65534.0 + 0.5) << 16u);
}
float3 unpackLobe(uint word) {
    const float2 e = float2(float(word & 0xFFFFu), float(word >> 16u)) / 65534.0;
    return octDecode(e);
}
)";

/// Where the device traces rays, a light is occluded by anything between the
/// shading point and the sample. Cutouts do not open a shadow yet: a sample
/// cut out of visibility still stops a shadow ray.
/// The light groups: declared only in the kernel of a frame that has them,
/// so a frame without compiles exactly the kernel it always did.
const char* kGroups = R"(
static const bool kLightGroups = true;
RWStructuredBuffer<float4>    groupColour;   // groupCount planes, a pixel each
uniform uint                  groupCount;
void writeGroups(uint at, uint pixels, float3 groups[8], float opacity) {
    for (uint g = 0; g < groupCount && g < 8; ++g) {
        groupColour[g * pixels + at] = float4(groups[g] * opacity, opacity);
    }
}
)";

const char* kNoGroups = R"(
static const bool kLightGroups = false;
void writeGroups(uint at, uint pixels, float3 groups[8], float opacity) {}
)";

const char* kShadowRay = R"(
uniform RaytracingAccelerationStructure shadowScene;

/// Whether the instance a shadow ray found casts this light's shadow: shadow
/// linking, resolved the same way light linking is, by one bit.
bool castsShadow(uint category, uint instance) {
    return lightLinked(category, instances[instance].categoriesLo, instances[instance].categoriesHi);
}

/// WHETHER A SHADOW RAY GOES ON THROUGH A SURFACE IT MET: a cut-out's
/// opacity is coverage, a shadow is too, and the ray passes where the
/// surface is not there. Asked of the material's opacity alone
/// (`evaluateOpacity`), which leaves the lobe stack this kernel is reading
/// as it was. A face marked invisible passes whatever it wears.
static uint gShadowDraws = 0;

bool shadowPassesThrough(uint instance, uint primitive, float2 barycentrics, float3 direction) {
    const float3 d = direction;
    const float3 viewDirection = float3(toWorld.row0.x * d.x + toWorld.row1.x * d.y + toWorld.row2.x * d.z,
                                        toWorld.row0.y * d.x + toWorld.row1.y * d.y + toWorld.row2.y * d.z,
                                        toWorld.row0.z * d.x + toWorld.row1.z * d.y + toWorld.row2.z * d.z);
    const float3 weights = float3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    const Surface s = surfaceFromWeights(uint4(instance + 1, primitive, 0, 0), weights, viewDirection);
    if ((triangleSubsets[s.mesh.firstTriangle + s.triangle] & 0x80000000u) != 0) {
        return true;
    }
    const MaterialRecord m = materials[materialRowOf(s)];
    if ((m.flags & (kMaterialCutout | kMaterialTransparent)) == 0) {
        return false;
    }
    // No footprint from the camera: this point is not the pixel's, and the
    // quad a footprint is read across is not all here -- only some of a
    // quad's threads ask, and asking across it wrote rows of black.
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, 0u, 0u, s, lookup.time, false);
    const float opacity = evaluateOpacity(m.function, inputs, m.blob);
    uint key = pcgHashShade(instance * 9781u + primitive);
    key = pcgHashShade(key + asuint(barycentrics.x));
    key = pcgHashShade(key + asuint(barycentrics.y) + gShadowDraws++);
    return float(key >> 8) * (1.0 / 16777216.0) >= opacity;
}

/// The nearest triangle along a shadow ray, in a function of its own so
/// the query is gone before anything asks the material about it: a query
/// live across a material's call wrote rows of garbage in this kernel.
struct ShadowHit {
    bool   found;
    uint   instance;
    uint   primitive;
    float2 barycentrics;
    float  t;
};

ShadowHit nearestShadowHit(RayDesc ray) {
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(shadowScene, RAY_FLAG_FORCE_OPAQUE, 0xFF, ray);
    query.Proceed();
    ShadowHit hit;
    hit.found = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    hit.instance = hit.found ? query.CommittedInstanceID() : 0u;
    hit.primitive = hit.found ? query.CommittedPrimitiveIndex() : 0u;
    hit.barycentrics = hit.found ? query.CommittedTriangleBarycentrics() : float2(0.0);
    hit.t = hit.found ? query.CommittedRayT() : 0.0;
    return hit;
}

bool occluded(float3 p, float3 n, float3 wi, float distance, uint shadowCategory) {
    // Off the surface it sits on, by a distance that grows with the scene, so
    // a grazing ray does not find the triangle it started from. Along the ray
    // as well as the normal, since a normal only says which side the surface
    // faces, not which side the light is on. What this costs is contact: an
    // occluder within the offset is not seen.
    const float scale = max(1.0, length(p));
    const float3 away = dot(n, wi) < 0.0 ? -n : n;
    RayDesc ray;
    ray.Origin = p + (away + wi) * (1.0e-3 * scale);
    ray.Direction = wi;
    ray.TMin = 1.0e-3 * scale;
    ray.TMax = max(distance - ray.TMin, 0.0);
    if (ray.TMax <= ray.TMin) {
        return false;
    }
    if (shadowCategory == kLightUnlinked && lighting.shadowCutouts == 0) {
        // Nothing to ask of the occluder: the first hit is the answer.
        RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> query;
        query.TraceRayInline(shadowScene, RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF,
                             ray);
        query.Proceed();
        return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    }
    // Linked, or with cut-outs about: walk on past whatever does not cast
    // this light's shadow, and past what a cut-out's opacity removed.
    for (uint step = 0; step < 64; ++step) {
        const ShadowHit hit = nearestShadowHit(ray);
        if (!hit.found) {
            return false;
        }
        const bool casts = shadowCategory == kLightUnlinked || castsShadow(shadowCategory, hit.instance);
        if (casts && (lighting.shadowCutouts == 0 ||
                      !shadowPassesThrough(hit.instance, hit.primitive, hit.barycentrics, wi))) {
            return true;
        }
        const float t = hit.t;
        ray.TMin = t + max(1.0e-4, t * 1.0e-5);
        if (ray.TMin >= ray.TMax) {
            return false;
        }
    }
    return false;
}
)";

/// THE CLOUD'S SHADOW ON A MESH, WITHOUT A RAY.
///
/// A ray asks what stands in the way; the map was told, once a frame, by
/// everything that does. So a floor under a splat cloud is darkened by a
/// lookup -- which is the only kind of cloud shadow a device without ray
/// tracing can have, and the only one the raster route has ever had.
const char* kShadowMap = R"(
import athenea.technique.splat_shadow_read;
static const bool kCloudShadows = true;
// As textures, not buffers: this kernel is at Metal's limit of thirty-one
// buffers, and texture slots it still has.
Texture2DArray<float> cloudShadow;
Texture2DArray<float> cloudShadowChain;   // exp(-total) a layer a light, with its mips

// The textures as a source splat_shadow_read understands: the same read the
// pass's probe and the gaussian receivers do through its buffer.
struct CloudShadowTexture : IShadowMapSource {
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

/// Where the rays through the next pixel over and the next pixel up meet the
/// plane of the surface at `p`: how far a pixel spans there, which is the
/// footprint the map's transmittance chain is read at. Ray differentials, as
/// materialInputsAt takes a texture's -- not the quad's, since the light loops
/// below are not uniform across it. Grazing, the span is left at nothing and
/// the read is level zero's.
float3 cloudOnPlane(float3 origin, float3 direction, float3 p, float3 n) {
    const float facing = dot(direction, n);
    if (!(abs(facing) > 1.0e-6)) {
        return p;
    }
    const float t = dot(p - origin, n) / facing;
    return t > 0.0 ? origin + direction * t : p;
}

void cloudPixelSpan(CameraParams c, ViewToWorld w, uint2 px, float3 p, float3 n, out float3 dx, out float3 dy) {
    float3 origin;
    float3 direction;
    viewRay(c, float2(px) + float2(1.5, 0.5), origin, direction);
    dx = cloudOnPlane(applyRows(w, origin, 1.0), applyRows(w, direction, 0.0), p, n) - p;
    viewRay(c, float2(px) + float2(0.5, 1.5), origin, direction);
    dy = cloudOnPlane(applyRows(w, origin, 1.0), applyRows(w, direction, 0.0), p, n) - p;
}

float cloudTransmittance(float3 p, float3 dx, float3 dy, uint light, bool casts) {
    if (!casts || light >= kShadowSlots) {
        return 1.0;
    }
    return shadowMapRead<CloudShadowTexture>(shadowFrameOf<CloudShadowTexture>(light), light, p, dx, dy);
}
/// A light along a sample's direction: a dome by the map of its directions
/// nearest to it (task TX), any other light by its own.
float cloudTransmittanceAlong(float3 p, float3 dx, float3 dy, uint light, LightRecord record, float3 wi, bool casts) {
    if (!casts) {
        return 1.0;
    }
    const uint slot = record.kind == kLightDome ? shadowDomeSlotOf<CloudShadowTexture>(wi) : light;
    return cloudTransmittance(p, dx, dy, slot, true);
}
)";

/// A DOME, READ PREFILTERED RATHER THAN SAMPLED (task PLAY-G).
///
/// A pixel whose lobes all reflect, and which every dome reaches, leaves its
/// domes out of both kinds of sample and writes what dome_shade.slang needs
/// to light it in closed form instead: the diffuse lobes' directional albedo
/// with the coverage, the glossy lobes' with their GGX alpha, and where it
/// stands with its normal. That kernel holds no lobe stack: done here, the
/// read overflowed what Metal keeps of a thread (rows of garbage across the
/// Corvette's ground, a pipeline that would not compile without the cloud
/// map). Any other pixel writes a coverage of 0 and samples its domes.
const char* kDomePrefilter = R"(
static const bool kDomePrefiltered = true;
RWTexture2D<float4> domeDiffuseOut;
RWTexture2D<float4> domeGlossyOut;
RWTexture2D<float4> domePointOut;
void writeDomeSurface(uint2 tid, bool closed, float3 wo, float3 p, float coverage) {
    if (!closed || !(coverage > 0.0)) {
        domeDiffuseOut[tid] = float4(0.0);
        return;
    }
    float3 diffuse = float3(0.0);
    float3 glossy = float3(0.0);
    float alphaSum = 0.0;
    float glossySum = 0.0;
    float3 normal = float3(0.0);
    for (uint k = 0; k < gAtheneaResult.count; ++k) {
        const uint kind = gAtheneaResult.lobes[k].kind;
        const float3 albedo = gAtheneaResult.lobes[k].weight * lobeAlbedo(gAtheneaResult.lobes[k], wo);
        if (k == 0) {
            normal = gAtheneaResult.lobes[k].normal;
        }
        if (kind == kLobeOrenNayar || kind == kLobeBurley || kind == kLobeSheen) {
            diffuse += albedo;
        } else {
            const float2 a = gAtheneaResult.lobes[k].alpha;
            const float share = max(dot(albedo, float3(0.2126, 0.7152, 0.0722)), 0.0);
            glossy += albedo;
            alphaSum += share * sqrt(max(a.x * a.y, 0.0));
            glossySum += share;
        }
    }
    const float2 e = saturate(octEncode(normalize(normal)));
    const uint packed = uint(e.x * 65534.0 + 0.5) | (uint(e.y * 65534.0 + 0.5) << 16u);
    domeDiffuseOut[tid] = float4(diffuse, coverage);
    domeGlossyOut[tid] = float4(glossy, glossySum > 0.0 ? alphaSum / glossySum : 0.0);
    domePointOut[tid] = float4(p, asfloat(packed));
}

/// A MATERIAL THE SAME EVERYWHERE, UNDER THE DOMES ALONE, is shaded by a
/// kernel of its own from the table tabulateMaterials made (kTabledBody):
/// this kernel only steps aside for its pixels. One test, no evaluation --
/// written here, the table's read pushed this kernel past what Metal would
/// compile (an internal compiler error under a UsdPreviewSurface ground).
Texture2D<float4> materialTable;
uniform uint      materialTableRows;
bool tabled(Surface s) {
    if (materialTableRows == 0) {
        return false;
    }
    const uint row = materialRowOf(s);
    // Not a mesh light nor an emission group's: the table holds a row's
    // emission, not an instance's.
    return (s.instance.flags & 0xFEu) == 0 && row < materialTableRows && (materials[row].flags & kMaterialUniform) != 0 &&
           materialTable.Load(int3(2, int(row), 0)).w > 0.5 &&
           domesReach(s.instance.categoriesLo, s.instance.categoriesHi);
}
)";

/// What the shading kernel reads of shadeEvaluated's (kEvaluatedBody): the
/// pixels it shaded, 1 in the mask, which this kernel steps aside for.
/// `evaluatedRan` is 0 on a frame it did not run, and the mask is not read.
const char* kEvaluatedMask = R"(
Texture2D<uint> evaluatedMask;
uniform uint    evaluatedRan;
bool evaluated(uint2 tid) {
    return evaluatedRan != 0 && evaluatedMask.Load(int3(int(tid.x), int(tid.y), 0)) != 0;
}
)";

const char* kNoDomePrefilter = R"(
static const bool kDomePrefiltered = false;
void writeDomeSurface(uint2 tid, bool closed, float3 wo, float3 p, float coverage) {}
bool tabled(Surface s) { return false; }
bool evaluated(uint2 tid) { return false; }
)";

/// A MATERIAL THAT ONLY REFLECTS, UNDER THE DOMES ALONE, but not the same
/// everywhere -- a texture, a primvar -- so the table cannot stand in for
/// it: evaluated at the pixel, and its albedos written for dome_shade.slang,
/// with none of the lights' machinery. Under the domes alone the shading
/// kernel does no more for such a pixel (domeClosed: no light sample, no lobe
/// sample, its emission and coverage), but it carries every light's code in
/// thread memory: a UsdPreviewSurface with a texture on its diffuse colour
/// took 35 ms at 1080p there on an RTX 3070, against 1.3 for the table.
/// Every pixel it reaches writes the mask, 1 where it shaded and 0 where it
/// left the pixel to the shading kernel (a lobe that transmits, a dome
/// linked away), so the mask needs no clearing.
const char* kEvaluatedBody = R"(
RWTexture2D<uint> evaluatedOut;

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeEvaluated(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    // In quad order, as the shading kernel: bump's derivatives.
    const uint2 tid = atheneaQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        evaluatedOut[tid] = 0;
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    if (tabled(s) || (s.instance.flags & 2u) != 0) {
        evaluatedOut[tid] = 0;
        return;   // shadeTabled's or shadeUnlit's
    }
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, inputs, m.blob);
    weighTransparent(m);
    if (!stackReflectsOnly() || !domesReach(s.instance.categoriesLo, s.instance.categoriesHi)) {
        evaluatedOut[tid] = 0;
        return;   // the shading kernel's
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint pixels = camera.width * camera.height;
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    const float coverage =
        (m.flags & (kMaterialCutout | kMaterialTransparent)) != 0 ? 1.0 : gAtheneaResult.opacity;
    const float3 emitted =
        meshLightEmission(iesValues, meshLightWords, s.instance.flags, inputs.inside, gAtheneaResult.emission);
    colour[at] = float4(emitted * coverage, coverage);
    writeDomeSurface(tid, true, toEye, inputs.positionWorld, coverage);
    depth[at] = s.depth;
    float3 groups[8];
    for (uint g = 0; g < 8; ++g) groups[g] = float3(0.0);
    writeGroups(at, pixels, groups, coverage);
    evaluatedOut[tid] = 1;
}
)";

/// A MATERIAL THE SAME EVERYWHERE, UNDER THE DOMES ALONE: what its lobes
/// return depends on the angle to the eye and nothing else, so it was
/// tabulated once this frame (tabulateMaterials, kTableBins angles) and a
/// pixel reads the table instead of evaluating the graph -- which for the
/// Corvette's OpenPBR ground was 150 ms at 1080p. It writes what the shading
/// kernel would have (the emission, the coverage, no light group's share)
/// and what dome_shade.slang adds the domes from. The frame's other lights,
/// were there any, would need the lobe stack itself: `materialTableRows` is
/// 0 unless the domes are all there is.
const char* kTabledBody = R"(
Texture2D<float4>          materialTable;
uniform uint               materialTableRows;
RWTexture2D<float4>        domeDiffuseOut;
RWTexture2D<float4>        domeGlossyOut;
RWTexture2D<float4>        domePointOut;
RWStructuredBuffer<float4> groupColour;   // groupCount planes, a pixel each
uniform uint               groupCount;
static const uint          kTableBins = 32;

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeTabled(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height || materialTableRows == 0) {
        return;
    }
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const uint row = materialRowOf(s);
    if (row >= materialTableRows || (materials[row].flags & kMaterialUniform) == 0 ||
        materialTable.Load(int3(2, int(row), 0)).w < 0.5 ||
        !domesReach(s.instance.categoriesLo, s.instance.categoriesHi)) {
        return;   // the shading kernel's
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint pixels = camera.width * camera.height;
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time, false);
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    const float bin = clamp(dot(inputs.normalWorld, toEye) * float(kTableBins) - 0.5, 0.0, float(kTableBins - 1));
    const uint lo = uint(bin);
    const uint hi = min(lo + 1u, kTableBins - 1u);
    const float f = bin - float(lo);
    const float4 d = lerp(materialTable.Load(int3(int(lo * 3), int(row), 0)),
                          materialTable.Load(int3(int(hi * 3), int(row), 0)), f);
    const float4 g = lerp(materialTable.Load(int3(int(lo * 3 + 1), int(row), 0)),
                          materialTable.Load(int3(int(hi * 3 + 1), int(row), 0)), f);
    const float3 emission = lerp(materialTable.Load(int3(int(lo * 3 + 2), int(row), 0)).rgb,
                                 materialTable.Load(int3(int(hi * 3 + 2), int(row), 0)).rgb, f);
    const float coverage = d.w;
    colour[at] = float4(emission * coverage, coverage);
    depth[at] = s.depth;
    for (uint k = 0; k < groupCount && k < 8; ++k) {
        groupColour[k * pixels + at] = float4(0.0, 0.0, 0.0, coverage);
    }
    if (!(coverage > 0.0)) {
        domeDiffuseOut[tid.xy] = float4(0.0);
        return;
    }
    const float2 e = saturate(octEncode(inputs.normalWorld));
    const uint packed = uint(e.x * 65534.0 + 0.5) | (uint(e.y * 65534.0 + 0.5) << 16u);
    domeDiffuseOut[tid.xy] = float4(d.rgb, coverage);
    domeGlossyOut[tid.xy] = g;
    domePointOut[tid.xy] = float4(inputs.positionWorld, asfloat(packed));
}
)";

/// AN UNLIT SURFACE (`primvars:athenea:unlit`, instance flag bit 1): drawn
/// and hidden behind as any mesh, and shown as its material's first texture
/// -- the row's in `unlitRows`, at the `unlitSlot` primvar (st_lightmap)
/// where the mesh has one, else at `stSlot` -- with no light, no lobe and no
/// dome: a surface whose light is baked into its texture. A row with no
/// texture shows the instance's displayColor. Every other kernel steps aside
/// for these pixels; the domes' G-buffer is cleared under them. In a frame
/// whose every mesh is unlit (`unlitAlone`) the other kernels do not run at
/// all, and this one writes what the shading kernel would have where no
/// unlit surface is: nothing, at depth 0.
const char* kUnlitBody = R"(
StructuredBuffer<uint> unlitRows;   // 2 a material row: the texture's id and sampler; ~0 none
uniform uint           unlitRowCount;   // rows in it: a row past them reads no texture
uniform uint           unlitSlot;
uniform uint           stSlot;
uniform uint           unlitAlone;

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeUnlit(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint pixels = camera.width * camera.height;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    Surface s;
    bool unlit = false;
    if (seen.x != 0) {
        s = surfaceAt(camera, tid.x, tid.y, seen);
        unlit = (s.instance.flags & 2u) != 0;
    }
    if (!unlit) {
        if (unlitAlone != 0) {
            colour[at] = float4(0.0);
            depth[at] = 0.0;
            float3 none[8];
            for (uint g = 0; g < 8; ++g) none[g] = float3(0.0);
            writeGroups(at, pixels, none, 0.0);
            writeDomeSurface(tid.xy, false, float3(0.0), float3(0.0), 0.0);
        }
        return;
    }
    const uint row = materialRowOf(s);
    float3 shown = s.instance.colour.rgb;
    const uint id = row < unlitRowCount ? unlitRows[row * 2] : 0xFFFFFFFFu;
    if (id != 0xFFFFFFFFu) {
        uint record = unlitSlot != kNoPrimvar ? primvarRecordOf(s.mesh, unlitSlot) : kNoPrimvar;
        if (record == kNoPrimvar && stSlot != kNoPrimvar) {
            record = primvarRecordOf(s.mesh, stSlot);
        }
        // Read at the level the pixel's footprint asks for: from level 0
        // alone, fine baked detail seen from afar aliases into speckle.
        float2 dx;
        float2 dy;
        const float2 uv = primvarFootprint(camera, tid.x, tid.y, s, record, dx, dy);
        bool found;
        const float4 texel = sampleTexture(gTextures, id, unlitRows[row * 2 + 1], uv, dx, dy, found);
        if (found) {
            shown = texel.rgb;
        }
    }
    colour[at] = float4(shown, 1.0);
    depth[at] = s.depth;
    writeDomeSurface(tid.xy, false, float3(0.0), float3(0.0), 0.0);
    float3 groups[8];
    for (uint g = 0; g < 8; ++g) groups[g] = float3(0.0);
    writeGroups(at, pixels, groups, 1.0);
}
)";

/// SCREEN-SPACE REFLECTIONS (`athenea:screenSpaceReflections`), after the
/// shading and the domes: a pixel that reflects marches its mirror direction
/// through the mesh layer's depth, in view space, and adds what the layer
/// holds where the march goes behind a surface, weighed by how much it
/// reflects and faded by its roughness, by the screen's edge and by the
/// distance. What reflects: an unlit surface by its constant primvars
/// `athenea:reflectance` (F0, Schlick's) and `athenea:roughness`; a shaded
/// one by the glossy albedo and alpha the prefiltered domes' G-buffer holds,
/// where the frame has it. The march reads a copy of the colour (`source`),
/// so no pixel reads what another is writing.
const char* kReflectionsBody = R"(
StructuredBuffer<float4>  source;          // the shaded colour, before the reflections
Texture2D<float4>         domeDiffuseIn;   // coverage in w: the G-buffer is this pixel's
Texture2D<float4>         domeGlossyIn;    // glossy albedo, alpha
uniform uint              hasGlossy;
uniform uint              reflectanceSlot;
uniform uint              roughnessSlot;

static const uint  kMarchSteps = 48;
static const uint  kRefineSteps = 6;
static const float kMaxDistance = 12.0;   // view units
static const float kThickness = 0.25;     // how deep behind a surface a hit still counts, view units

float constantPrimvar(Surface s, uint slot, float fallback) {
    if (slot == kNoPrimvar) {
        return fallback;
    }
    const uint record = primvarRecordOf(s.mesh, slot);
    return record == kNoPrimvar ? fallback
                                : primvarAt(record, s.mesh, s.triangle, s.points, s.weights, float4(fallback)).x;
}

/// Where view point `q` lands on the screen, in pixels (row 0 on top).
float2 toPixel(float3 q) {
    return float2(q.x / q.z * camera.focalX + camera.centreX, q.y / q.z * camera.focalY + camera.centreY);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeReflections(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height || camera.orthographic != 0) {
        return;
    }
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (seen.x == 0) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    float3 tint;
    float roughness;
    float3 n = s.normal;
    const float3 v = normalize(s.viewPosition);
    if (dot(n, v) > 0.0) {
        n = -n;
    }
    const float cosine = saturate(dot(-v, n));
    if ((s.instance.flags & 2u) != 0) {
        const float f0 = saturate(constantPrimvar(s, reflectanceSlot, 0.0));
        roughness = saturate(constantPrimvar(s, roughnessSlot, 0.3));
        // Schlick's; nothing where the surface says it reflects nothing.
        tint = float3(f0 > 0.0 ? f0 + (1.0 - f0) * pow(1.0 - cosine, 5.0) : 0.0);
    } else if (hasGlossy != 0 && domeDiffuseIn.Load(int3(int(tid.x), int(tid.y), 0)).w > 0.0) {
        const float4 glossy = domeGlossyIn.Load(int3(int(tid.x), int(tid.y), 0));
        tint = glossy.rgb;
        roughness = sqrt(max(glossy.w, 0.0));
    } else {
        return;
    }
    // Rough surfaces blur what they reflect past what a march can show.
    const float sharpness = saturate(1.0 - roughness / 0.6);
    if (!(max(max(tint.x, tint.y), tint.z) * sharpness > 1.0e-3)) {
        return;
    }
    const float3 r = reflect(v, n);
    const float3 origin = s.viewPosition + n * (0.002 * s.depth);
    float previous = 0.0;
    float hitT = -1.0;
    for (uint i = 1; i <= kMarchSteps; ++i) {
        const float u = float(i) / float(kMarchSteps);
        const float t = kMaxDistance * u * u;
        const float3 q = origin + r * t;
        if (q.z <= camera.nearZ) {
            break;
        }
        const float2 p = toPixel(q);
        if (any(p < 0.0) || p.x >= float(camera.width) || p.y >= float(camera.height)) {
            break;
        }
        const float surface = depth[uint(p.y) * camera.width + uint(p.x)];
        if (surface > 0.0 && q.z > surface && q.z - surface < kThickness) {
            // Between the last step in front and this one behind.
            float lo = previous;
            float hi = t;
            for (uint k = 0; k < kRefineSteps; ++k) {
                const float mid = 0.5 * (lo + hi);
                const float3 m = origin + r * mid;
                const float2 mp = toPixel(m);
                const float ms = depth[uint(clamp(mp.y, 0.0, float(camera.height - 1))) * camera.width +
                                       uint(clamp(mp.x, 0.0, float(camera.width - 1)))];
                if (ms > 0.0 && m.z > ms) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hitT = hi;
            break;
        }
        previous = t;
    }
    if (hitT < 0.0) {
        return;
    }
    const float2 hp = toPixel(origin + r * hitT);
    const uint2 hit = uint2(clamp(hp, float2(0.0), float2(camera.width - 1, camera.height - 1)));
    const float4 seenThere = source[hit.y * camera.width + hit.x];
    // Faded where the screen ends (what lies past it is not known) and with
    // the distance marched.
    const float2 edge = min(hp, float2(camera.width, camera.height) - hp) / (0.1 * float2(camera.width, camera.height));
    const float fade = saturate(min(edge.x, edge.y)) * saturate(1.0 - hitT / kMaxDistance);
    const float4 here = source[at];
    colour[at] = float4(here.rgb + tint * sharpness * fade * seenThere.rgb, here.a);
}
)";

/// The table shadeTabled reads: a row a material, kTableBins angles to
/// the eye, three texels each -- (diffuse albedo, opacity), (glossy albedo,
/// alpha), (emission, whether the domes answer it) -- for every material
/// flagged kMaterialUniform, evaluated on a flat patch facing up with the eye
/// at that angle.
const char* kTableBody = R"(
RWTexture2D<float4> materialTableOut;
uniform uint        materialTableRows;
static const uint   kTableBins = 32;

[shader("compute")]
[numthreads(64, 1, 1)]
void tabulateMaterials(uint3 tid: SV_DispatchThreadID) {
    const uint row = tid.x / kTableBins;
    const uint bin = tid.x % kTableBins;
    if (row >= materialTableRows) {
        return;
    }
    const MaterialRecord m = materials[row];
    if ((m.flags & kMaterialUniform) == 0) {
        materialTableOut[uint2(bin * 3 + 2, row)] = float4(0.0);
        return;
    }
    const float c = (float(bin) + 0.5) / float(kTableBins);
    const float3 wo = float3(sqrt(max(1.0 - c * c, 0.0)), 0.0, c);
    MaterialInputs inputs;
    inputs.positionWorld = float3(0.0);
    inputs.normalWorld = float3(0.0, 0.0, 1.0);
    inputs.tangentWorld = float3(1.0, 0.0, 0.0);
    inputs.bitangentWorld = float3(0.0, 1.0, 0.0);
    inputs.positionObject = float3(0.0);
    inputs.normalObject = float3(0.0, 0.0, 1.0);
    inputs.tangentObject = float3(1.0, 0.0, 0.0);
    inputs.bitangentObject = float3(0.0, 1.0, 0.0);
    inputs.viewPosition = wo;
    inputs.uvDx = float2(0.0);
    inputs.uvDy = float2(0.0);
    inputs.frame = 0.0;
    inputs.time = lookup.time;
    inputs.inside = false;
    inputs.triangle = 0;
    inputs.points = uint3(0);
    inputs.weights = float3(1.0, 0.0, 0.0);
    inputs.displayColor = float4(0.18, 0.18, 0.18, 1.0);
    inputs.worldFromObject0 = float4(1.0, 0.0, 0.0, 0.0);
    inputs.worldFromObject1 = float4(0.0, 1.0, 0.0, 0.0);
    inputs.worldFromObject2 = float4(0.0, 0.0, 1.0, 0.0);
    evaluateMaterial(m.function, inputs, m.blob);
    float3 diffuse = float3(0.0);
    float3 glossy = float3(0.0);
    float alphaSum = 0.0;
    float glossySum = 0.0;
    for (uint k = 0; k < gAtheneaResult.count; ++k) {
        const uint kind = gAtheneaResult.lobes[k].kind;
        const float3 albedo = gAtheneaResult.lobes[k].weight * lobeAlbedo(gAtheneaResult.lobes[k], wo);
        if (kind == kLobeOrenNayar || kind == kLobeBurley || kind == kLobeSheen) {
            diffuse += albedo;
        } else {
            const float2 a = gAtheneaResult.lobes[k].alpha;
            const float share = max(dot(albedo, float3(0.2126, 0.7152, 0.0722)), 0.0);
            glossy += albedo;
            alphaSum += share * sqrt(max(a.x * a.y, 0.0));
            glossySum += share;
        }
    }
    materialTableOut[uint2(bin * 3, row)] = float4(diffuse, gAtheneaResult.opacity);
    materialTableOut[uint2(bin * 3 + 1, row)] = float4(glossy, glossySum > 0.0 ? alphaSum / glossySum : 0.0);
    materialTableOut[uint2(bin * 3 + 2, row)] = float4(gAtheneaResult.emission, stackReflectsOnly() ? 1.0 : 0.0);
}
)";

const char* kNoShadowMap = R"(
static const bool kCloudShadows = false;
void cloudPixelSpan(CameraParams c, ViewToWorld w, uint2 px, float3 p, float3 n, out float3 dx, out float3 dy) {
    dx = float3(0.0);
    dy = float3(0.0);
}
float cloudTransmittance(float3 p, float3 dx, float3 dy, uint light, bool casts) { return 1.0; }
float cloudTransmittanceAlong(float3 p, float3 dx, float3 dy, uint light, LightRecord record, float3 wi, bool casts) {
    return 1.0;
}
)";

/// THE SHADOW RAYS ARE TRACED BY A KERNEL OF THEIR OWN.
///
/// The shading kernel evaluates a material into a lobe stack in thread
/// memory, and on Metal (Apple M5 Pro) a ray query in the same kernel
/// corrupted it: rows of garbage in blocks of half a threadgroup. The notes
/// below record the workarounds that each kept one material clean; the
/// sparrow broke all of them -- a frame that merely holds a cut-out material
/// (an unbound one will do) striped a grey floor under a sun. So the kernels
/// are two. `traceShadows` rebuilds the pixel's surface, draws the light
/// samples the shading kernel will (the same hash, the same order), traces
/// each one's shadow ray -- walking through cut-outs by their opacity, which
/// it asks of the occluder's material -- and writes one bit a sample. The
/// shading kernel reads the bit and holds no intersector at all.
const char* kNoShadowBits = R"(
bool occludedSample(uint at, uint bit) {
    return false;
}
)";

const char* kShadowBits = R"(
/// traceShadows' answers: `shadowWords` words a pixel, bit b set where the
/// shading loop's sample b is in shadow. No words, no shadows (a frame with
/// nothing to trace against).
StructuredBuffer<uint> shadowBits;
uniform uint           shadowWords;
bool occludedSample(uint at, uint bit) {
    if (bit >= shadowWords * 32u) {
        return false;
    }
    return ((shadowBits[at * shadowWords + (bit >> 5u)] >> (bit & 31u)) & 1u) != 0u;
}
)";

/// The lobe samples' directions, drawn where the material is evaluated and
/// no ray is traced: the same samples the shading kernel draws, in its order,
/// for traceShadows to trace.
const char* kLobeBody = R"(
RWStructuredBuffer<uint> lobeDirsOut;   // lobeLookups() a pixel, then a word a pixel: dome closed
uniform uint             domesClosed;   // 1: the frame reads its domes prefiltered

[shader("compute")]
[numthreads(16, 16, 1)]
void drawLobes(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    // In quad order, as the shading kernel: bump reads its quad.
    const uint2 tid = atheneaQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    const uint lobes = lobeLookups();
    if (seen.x == 0 || lobes == 0) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, inputs, m.blob);
    weighTransparent(m);
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    // WHERE THE DOMES ARE READ PREFILTERED (kDomePrefilter) a stack that only
    // reflects takes no dome sample at all: marked after the directions, a
    // word a pixel, so traceShadows traces no ray for it. Nor a lobe sample,
    // where no distant light is left for one to meet.
    const bool closed = domesClosed != 0 && stackReflectsOnly() &&
                        domesReach(s.instance.categoriesLo, s.instance.categoriesHi);
    lobeDirsOut[camera.width * camera.height * lobes + at] = closed ? 1u : 0u;
    if (closed && !anyDistantLight()) {
        return;
    }
    for (uint j = 0; j < lobes; ++j) {
        const float2 ua = sampleAt(tid, lightCount + 1, j);
        const float2 ub = sampleAt(tid, lightCount + 2, j);
        const LobeSample ms = stackSample(gAtheneaResult, toEye, float3(ua, ub.x));
        lobeDirsOut[at * lobes + j] = ms.valid && any(ms.weight > float3(0.0)) ? packLobe(ms.wi) : kNoLobe;
    }
}
)";

const char* kTraceBody = R"(
RWStructuredBuffer<uint> shadowBitsOut;
uniform uint             shadowWords;
RWStructuredBuffer<float4> meshRatioOut;   // a pixel: visible, unshadowed, view z, normal (mesh_light_cut.slang)
uniform uint             meshRatio;        // 1: the closed form's shadow ratio is measured here
StructuredBuffer<uint>   lobeDirs;      // drawLobes' directions, lobeLookups() a pixel
uniform uint             lobesDrawn;    // 0: no lobe directions this frame
uniform uint             domesClosed;   // 1: drawLobes marked the pixels whose domes are read prefiltered

void putShadowBit(uint at, inout uint word, inout uint bit, bool blocked) {
    if (blocked) {
        word |= 1u << (bit & 31u);
    }
    ++bit;
    if ((bit & 31u) == 0u) {
        shadowBitsOut[at * shadowWords + (bit >> 5u) - 1u] = word;
        word = 0u;
    }
}

/// The shading kernel's light samples, in its order, each a bit: whether its
/// shadow ray is blocked. A sample the shading loop skips before its shadow
/// (not linked, not valid) still has its bit, so the numbering is the loop's.
[shader("compute")]
[numthreads(16, 16, 1)]
void traceShadows(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    const uint2 tid = atheneaQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    if (meshRatio != 0u) {
        meshRatioOut[tid.y * camera.width + tid.x] = float4(0.0);
    }
    if (seen.x == 0 || (lightCount == 0 && meshSampleCount() == 0u && meshRatio == 0u)) {
        return;   // shading reads no bit there
    }
    const uint at = tid.y * camera.width + tid.x;
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time, false);
    const float3 p = inputs.positionWorld;
    const float3 n = inputs.normalWorld;
    const uint samples = max(lighting.samples, 1u);
    // A pixel drawLobes marked reads its domes prefiltered: no ray for them,
    // and no lobe sample unless a distant light is there to meet.
    const uint lobesHere = lobesDrawn != 0 ? lobeLookups() : 0u;
    const bool closed = domesClosed != 0 && lobesHere != 0 &&
                        lobeDirs[camera.width * camera.height * lobesHere + at] != 0u;
    uint word = 0u;
    uint bit = 0u;
    if (lightCount == 0u) {
        // Mesh lights alone: their bits are the first.
    } else if (lighting.chooseLights != 0) {
        for (uint i = 0; i < samples; ++i) {
            bool blocked = false;
            const float2 u = sampleAt(tid, 0, i);
            const float pick = sampleAt(tid, 1, i).x;
            const LightChoice choice =
                lighting.chooseLights == 2
                    ? chooseLightAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount, p, n, pick)
                    : chooseLight(lights, lightCount, pick);
            if (choice.valid) {
                const LightRecord light = lights[choice.index];
                if ((light.flags & kLightShadow) != 0 && !(closed && light.kind == kLightDome) &&
                    lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                    const LightSample ls = sampleLightImaged(light, p, n, u);
                    blocked = ls.valid && occluded(p, n, ls.wi, ls.distance, light.shadowCategory);
                }
            }
            putShadowBit(at, word, bit, blocked);
        }
    } else {
        for (uint k = 0; k < lightCount; ++k) {
            const LightRecord light = lights[k];
            const bool traced = (light.flags & kLightShadow) != 0 && !(closed && light.kind == kLightDome) &&
                                lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi);
            for (uint i = 0; i < samples; ++i) {
                bool blocked = false;
                if (traced) {
                    const LightSample ls = sampleLightImaged(light, p, n, sampleAt(tid, k, i));
                    blocked = ls.valid && occluded(p, n, ls.wi, ls.distance, light.shadowCategory);
                }
                putShadowBit(at, word, bit, blocked);
            }
        }
    }
    // Then each lobe sample's, toward whatever lies at infinity along it.
    const uint lobes = closed && !anyDistantLight() ? 0u : lobesHere;
    for (uint j = 0; j < lobes; ++j) {
        const uint packed = lobeDirs[at * lobes + j];
        const bool blocked =
            packed != kNoLobe && occluded(p, n, unpackLobe(packed), 1.0e30, kLightUnlinked);
        putShadowBit(at, word, bit, blocked);
    }
    // THE CLOSED FORM'S SHADOW, as a ratio: a few samples through the tree,
    // what was visible of what was there, weighed by luminance and cosine,
    // for meshShadowFilter to filter before dividing.
    if (meshRatio != 0u && meshClusterCount(iesValues, meshLightWords) > 0u) {
        float visible = 0.0;
        float total = 0.0;
        for (uint i = 0u; i < kMeshRatioSamples; ++i) {
            const LightChoice choice = chooseMeshCluster(iesValues, meshLightWords, p, n, meshPick(tid, i));
            if (!choice.valid) {
                continue;
            }
            const MeshLightSample ms =
                sampleMeshCluster(iesValues, meshLightWords, choice.index, p, meshPoint(tid, i));
            if (!ms.valid) {
                continue;
            }
            const float w = dot(ms.radiance, float3(0.2126, 0.7152, 0.0722)) * max(dot(n, ms.wi), 0.0) /
                            (ms.pdf * choice.probability);
            total += w;
            if (!occluded(p, n, ms.wi, meshShadowReach(p, ms.distance), kLightUnlinked)) {
                visible += w;
            }
        }
        const float2 e = saturate(octEncode(n));
        const uint packed = uint(e.x * 65535.0 + 0.5) | (uint(e.y * 65535.0 + 0.5) << 16u);
        meshRatioOut[at] = float4(visible, total, s.depth, asfloat(packed));
    }
    // Then the mesh lights' samples, after every lobe bit the frame numbers
    // (a pixel that drew fewer leaves the rest clear): the same choice and
    // point the shading kernel draws, shadowed short of the emitter itself.
    const uint meshes = meshSampleCount();
    if (meshes > 0u) {
        while (bit < meshSampleBit(0u)) {
            putShadowBit(at, word, bit, false);
        }
        for (uint i = 0u; i < meshes; ++i) {
            bool blocked = false;
            const LightChoice choice = chooseMeshCluster(iesValues, meshLightWords, p, n, meshPick(tid, i));
            if (choice.valid) {
                const MeshLightSample ms =
                    sampleMeshCluster(iesValues, meshLightWords, choice.index, p, meshPoint(tid, i));
                blocked = ms.valid && occluded(p, n, ms.wi, meshShadowReach(p, ms.distance), kLightUnlinked);
            }
            putShadowBit(at, word, bit, blocked);
        }
    }
    if ((bit & 31u) != 0u) {
        shadowBitsOut[at * shadowWords + (bit >> 5u)] = word;
    }
}
)";

const char* kKernelBody = R"(
/// Lights at infinity -- a dome, a distant light -- are the ones the raster
/// can also meet along a lobe's own sample, since nothing but occlusion stands
/// between: with the same shadow ray, or none where the light casts none, the
/// two strategies see the same light and are weighed against each other. A
/// light whose shadow links leave some occluders out keeps light sampling
/// alone, since a lobe sample would have to ask the same of every occluder.
bool atInfinity(LightRecord l) {
    return l.kind == kLightDome || l.kind == kLightDistant;
}
bool rasterWeighs(LightRecord l) {
    return atInfinity(l) && ((l.flags & kLightShadow) == 0 || l.shadowCategory == kLightUnlinked);
}
/// Light k's probability of being chosen, as the frame chooses: one for every
/// light at every pixel, its power share, or the light tree's.
float rasterChoice(uint k, float3 p, float3 n) {
    if (lighting.chooseLights == 2) {
        return lightPdfChoiceAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount, lights, k, p, n);
    }
    if (lighting.chooseLights == 1) {
        const float power = lights[lightCount - 1].cumulative;
        if (!(power > 0.0)) {
            return 0.0;
        }
        const float before = k == 0 ? 0.0 : lights[k - 1].cumulative;
        return max(lights[k].cumulative - before, 1.0e-9) / power;
    }
    return 1.0;
}
/// The power heuristic between n_a samples at density a and n_b at density b.
float rasterMis(float na, float a, float nb, float b) {
    const float x = na * a;
    const float y = nb * b;
    return x * x / max(x * x + y * y, 1.0e-30);
}
/// The density the lobe stack draws `wi` with, for the light samples' and
/// the lobe samples' weights. Once the trace shared this kernel, asking for
/// it near a shadow ray wrote garbage, and a Phong proxy about the mirror
/// direction stood in -- zero for any lobe broader than a GGX alpha of about
/// 0.35, so a broad lobe's light samples kept the whole weight and a glossy
/// feather at a grazing angle took a dome sample at 300 times the sky. The
/// rays are traceShadows' now; this kernel weighs as the path tracer does.
float lobeDensity(float3 toEye, float3 wi) {
    return stackPdf(gAtheneaResult, toEye, wi);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void shadeMaterials(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    // In quad order, so the nodes that want screen derivatives (bump) find
    // their neighbours in the thread's quad.
    const uint2 tid = atheneaQuadPixel(group.xy, index);
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const uint4 seen = visibility.Load(int3(int(tid.x), int(camera.height - 1 - tid.y), 0));
    const uint pixels = camera.width * camera.height;
    float3 groups[8];
    for (uint g = 0; g < 8; ++g) groups[g] = float3(0.0);
    if (seen.x == 0) {
        colour[at] = float4(0.0);
        depth[at] = 0.0;
        writeGroups(at, pixels, groups, 0.0);
        writeDomeSurface(tid, false, float3(0.0), float3(0.0), 0.0);
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    if (tabled(s) || evaluated(tid) || (s.instance.flags & 2u) != 0) {
        return;   // shadeTabled's, shadeEvaluated's or shadeUnlit's
    }
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time);
    const MaterialRecord m = materials[materialRowOf(s)];
    evaluateMaterial(m.function, inputs, m.blob);
    weighTransparent(m);
    // The lobe stack is read where the material left it, in thread memory.
    // This kernel holds no intersector: with one beside the stack it wrote
    // rows of garbage in blocks of half a threadgroup on an Apple M5 Pro,
    // whatever was copied or not (kNoShadowBits has the history).
#define stack gAtheneaResult
    const float3 toEye = normalize(inputs.viewPosition - inputs.positionWorld);
    // A MeshLightAPI prim's own emission, and an emission group's level.
    float3 radiance = meshLightEmission(iesValues, meshLightWords, s.instance.flags, inputs.inside, stack.emission);
    // What light sampling cannot reach, or reaches badly, is met along the
    // lobes' own samples instead: a dome is sampled over the hemisphere above
    // the normal, so light through the surface (glass) never meets it; a
    // delta lobe (smooth glass, a mirror) answers no light sample; a glossy
    // metal's narrow lobe is almost never found by a dome's samples. Without
    // this, the chess set's glass pawn heads and polished rims drew black.
    //
    // Each lobe sample is shadowed as a light sample is: drawLobes drew the
    // same directions and traceShadows traced them (kNoShadowBits). While the
    // trace shared this kernel it could not be -- a reflection saw the sky
    // whether or not something stood in the way, and a feathered belly's
    // underside, under the bird, read 0.26 where the tracer reads 0.16.
    // Weighed against light sampling by the power heuristic over both true
    // densities, at every lobe: broad lobes too (lobeDensity says why).
    const uint lightSampleCount = max(lighting.samples, 1u);
    // A dome this stack lets be read prefiltered is left out of both kinds of
    // sample below and added in closed form by dome_shade.slang (kDomePrefilter).
    const bool domeClosed = kDomePrefiltered && stackReflectsOnly() &&
                            domesReach(s.instance.categoriesLo, s.instance.categoriesHi);
    // Nor is a lobe sample drawn where only domes are left at infinity for it
    // to meet: drawLobes and traceShadows skip the same pixels.
    const uint lobeCount = domeClosed && !anyDistantLight() ? 0u : lobeLookups();
    // How far this pixel spans on the surface: the footprint a cloud's shadow
    // map is read with (cloudPixelSpan). Nothing where no cloud casts.
    float3 cloudDx;
    float3 cloudDy;
    cloudPixelSpan(camera, toWorld, tid, inputs.positionWorld, inputs.normalWorld, cloudDx, cloudDy);
    if (lobeCount != 0) {
        const float3 n0 = inputs.normalWorld;
        const float eyeSide = dot(toEye, n0);
        float3 lobeSum = float3(0.0);
        for (uint j = 0; j < lobeCount; ++j) {
            const float2 ua = sampleAt(tid, lightCount + 1, j);
            const float2 ub = sampleAt(tid, lightCount + 2, j);
            const LobeSample ms = stackSample(stack, toEye, float3(ua, ub.x));
            if (!ms.valid || !any(ms.weight > float3(0.0))) {
                continue;
            }
            const bool through = dot(ms.wi, n0) * eyeSide < 0.0;
            for (uint k = 0; k < lightCount; ++k) {
                const LightRecord light = lights[k];
                if (!atInfinity(light) ||
                    !lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                    continue;
                }
                if (domeClosed && light.kind == kLightDome) {
                    continue;
                }
                const bool weighs = rasterWeighs(light);
                // Light sampling has a light it weighs, and a delta lobe never;
                // through the surface only the dome, which light sampling does
                // not reach there, and a distant light through a delta lobe.
                if (through) {
                    if (light.kind != kLightDome && !ms.delta) {
                        continue;
                    }
                } else if (!weighs && !ms.delta) {
                    continue;
                }
                const LightHit lh = lightHitImaged(light, inputs.positionWorld, ms.wi);
                if (!lh.valid) {
                    continue;
                }
                // The lobe sample's own shadow ray, traced by traceShadows
                // along the direction drawLobes drew. A light whose shadow
                // links leave occluders out is not asked (that ray was traced
                // against everything).
                if ((light.flags & kLightShadow) != 0 && light.shadowCategory == kLightUnlinked &&
                    occludedSample(at, lightSampleBits() + j)) {
                    continue;
                }
                float weight = 1.0;
                if (!ms.delta && !through) {
                    const float lightDensity = rasterChoice(k, inputs.positionWorld, n0) *
                                               lightPdfImaged(light, inputs.positionWorld, n0, ms.wi);
                    weight = rasterMis(float(lobeCount), ms.pdf, float(lightSampleCount), lightDensity);
                }
                // And the clouds in the way, along the lobe's own direction:
                // a dome by the map of its nearest direction (task TX).
                const float cloudThrough = cloudTransmittanceAlong(inputs.positionWorld, cloudDx, cloudDy, k, light,
                                                                   ms.wi, (light.flags & kLightShadow) != 0);
                const float3 arrived = weight * ms.weight * lh.radiance * cloudThrough;
                lobeSum += arrived;
                if (kLightGroups && light.group != 0 && light.group <= 8) {
                    groups[light.group - 1] += arrived / float(lobeCount);
                }
            }
        }
        radiance += lobeSum / float(lobeCount);
    }
)"
R"(    const uint meshSamples = meshSampleCount();
    if (meshAnalytic != 0u && meshClusterCount(iesValues, meshLightWords) > 0u) {
        // In closed form, shadowed by the filtered ratio of a few rays.
        const float shadow = meshShadowOn != 0u ? meshShadow.Load(int3(int(tid.x), int(tid.y), 0)) : 1.0;
        radiance += shadow * meshClosedForm(tid, inputs.positionWorld, inputs.normalWorld, toEye);
    } else if (meshSamples > 0u) {
        // THE MESH LIGHTS, SAMPLED: a cluster through the tree by its
        // importance here, a triangle by its share, a point on it -- the
        // path tracer's next event estimate, without its MIS (no lobe sample
        // here meets an emitter). Shadowed by traceShadows' bits.
        float3 sum = float3(0.0);
        for (uint i = 0u; i < meshSamples; ++i) {
            const LightChoice choice = chooseMeshCluster(iesValues, meshLightWords, inputs.positionWorld,
                                                         inputs.normalWorld, meshPick(tid, i));
            if (!choice.valid) {
                continue;
            }
            const MeshLightSample ms = sampleMeshCluster(iesValues, meshLightWords, choice.index,
                                                         inputs.positionWorld, meshPoint(tid, i));
            if (!ms.valid) {
                continue;
            }
            const float3 f = stackEval(stack, toEye, ms.wi);
            if (!any(f > float3(0.0)) || occludedSample(at, meshSampleBit(i))) {
                continue;
            }
            sum += f * ms.radiance / (ms.pdf * choice.probability);
        }
        radiance += sum / float(meshSamples);
    }
    if (lightCount == 0) {
        // The headlight: unit radiance from the eye, so a white Lambert
        // surface facing it shows 1. What meshes were lit by before there
        // were lights -- and still are, unless mesh lights light them.
        if (meshClusterCount(iesValues, meshLightWords) == 0u) {
            radiance += kPi * stackEval(stack, toEye, toEye);
        }
    } else if (lighting.chooseLights != 0) {
        // One light a sample, in proportion to its power: the cost of a pixel
        // stops growing with the number of lights, and the density it was
        // chosen with is divided back out.
        const uint samples = max(lighting.samples, 1u);
        float3 sum = float3(0.0);
        for (uint i = 0; i < samples; ++i) {
            const float2 u = sampleAt(tid, 0, i);
            const float pick = sampleAt(tid, 1, i).x;
            const LightChoice choice =
                lighting.chooseLights == 2
                    ? chooseLightAny(iesValues, lightNodeBase, lightTreeNodes, lightUnboundedCount,
                                     inputs.positionWorld, inputs.normalWorld, pick)
                    : chooseLight(lights, lightCount, pick);
            if (!choice.valid) {
                continue;
            }
            const LightRecord light = lights[choice.index];
            if (!lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                continue;
            }
            if (domeClosed && light.kind == kLightDome) {
                continue;   // dome_shade.slang adds it
            }
            const LightSample ls = sampleLightImaged(light, inputs.positionWorld, inputs.normalWorld, u);
            if (!ls.valid) {
                continue;
            }
            const float3 f = stackEval(stack, toEye, ls.wi);
            if (!any(f > float3(0.0))) {
                continue;
            }
            // Weighed against the lobe samples by the stack's own density
            // (lobeDensity), as the path tracer weighs.
            float weight = 1.0;
            if (!ls.delta && rasterWeighs(light) && !(domeClosed && light.kind == kLightDome)) {
                weight = rasterMis(float(samples), ls.pdf * choice.probability, float(lobeCount),
                                   lobeDensity(toEye, ls.wi));
            }
            if ((light.flags & kLightShadow) != 0 && occludedSample(at, i)) {
                continue;
            }
            // And what the clouds between this point and the light stopped,
            // tinted and faded as the light's ShadowAPI says.
            const float3 cloudThrough = shadowTint(
                light,
                cloudTransmittanceAlong(inputs.positionWorld, cloudDx, cloudDy, choice.index, light, ls.wi,
                                        (light.flags & kLightShadow) != 0),
                ls.distance);
            const float3 arrived = cloudThrough * weight * f * ls.radiance / (ls.pdf * choice.probability);
            sum += arrived;
            if (kLightGroups && light.group != 0 && light.group <= 8) {
                groups[light.group - 1] += arrived / float(samples);
            }
        }
        radiance += sum / float(samples);
    } else {
        const uint samples = max(lighting.samples, 1u);
        for (uint k = 0; k < lightCount; ++k) {
            const LightRecord light = lights[k];
            // Light linking: a light reaches only the categories its
            // collection resolved to, and one with no collection reaches all.
            if (!lightLinked(light.lightCategory, s.instance.categoriesLo, s.instance.categoriesHi)) {
                continue;
            }
            if (domeClosed && light.kind == kLightDome) {
                continue;   // dome_shade.slang adds it
            }
            const bool shadow = (light.flags & kLightShadow) != 0;
            float3 sum = float3(0.0);
            for (uint i = 0; i < samples; ++i) {
                const LightSample ls =
                    sampleLightImaged(light, inputs.positionWorld, inputs.normalWorld, sampleAt(tid, k, i));
                if (!ls.valid) {
                    continue;
                }
                // stackEval carries the cosine, so the estimator is the
                // response times the arriving radiance over the density.
                const float3 f = stackEval(stack, toEye, ls.wi);
                if (!any(f > float3(0.0))) {
                    continue;
                }
                // Weighed against the lobe samples, as above.
                float weight = 1.0;
                if (!ls.delta && rasterWeighs(light) && !(domeClosed && light.kind == kLightDome)) {
                    weight = rasterMis(float(samples), ls.pdf, float(lobeCount), lobeDensity(toEye, ls.wi));
                }
                if (shadow && occludedSample(at, k * samples + i)) {
                    continue;
                }
                sum += shadowTint(light,
                                  cloudTransmittanceAlong(inputs.positionWorld, cloudDx, cloudDy, k, light, ls.wi, shadow),
                                  ls.distance) * weight *
                       f * ls.radiance / ls.pdf;
            }
            radiance += sum / float(samples);
            if (kLightGroups && light.group != 0 && light.group <= 8) {
                groups[light.group - 1] += sum / float(samples);
            }
        }
    }
    // A cutout's sample survived its lot in the visibility pass: it is there
    // whole. Any other opacity is still blended, as displayOpacity is.
    const float coverage = (m.flags & (kMaterialCutout | kMaterialTransparent)) != 0 ? 1.0 : stack.opacity;
    colour[at] = float4(radiance * coverage, coverage);
    // The domes left out above, for dome_shade.slang to add in closed form.
    writeDomeSurface(tid, domeClosed, toEye, inputs.positionWorld, coverage);
    depth[at] = s.depth;
    writeGroups(at, pixels, groups, coverage);
}
)";

/// kLobeLookups in the kernels' prelude.
constexpr uint64_t kLobeLookups = 32;

}   // namespace

Result<void> MaterialShading::prepareClosedForm() {
    if (!ltc_.has_value()) {
        auto made = light::LtcTable::create(*library_);
        if (!made) return std::move(made).error();
        ltc_.emplace(std::move(*made));
    }
    if (!meshCut_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/light/mesh_light_cut", "meshLightCut");
        if (!made) return std::move(made).error();
        meshCut_.emplace(std::move(*made));
    }
    if (!meshFilter_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/light/mesh_light_cut", "meshShadowFilter");
        if (!made) return std::move(made).error();
        meshFilter_.emplace(std::move(*made));
    }
    return ok();
}

Result<void> MaterialShading::ensureMeshTextures(uint32_t cutWidth, uint32_t cutHeight, uint32_t width,
                                                 uint32_t height) {
    const auto texture = [&](gpu::Texture& into, rhi::ComPtr<rhi::ITextureView>& view, uint32_t w, uint32_t h,
                             rhi::Format format, const char* label) -> Result<void> {
        if (into.valid() && into.width() >= w && into.height() >= h && view) {
            return ok();
        }
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.width = std::max(w, into.valid() ? into.width() : 1u);
        desc.height = std::max(h, into.valid() ? into.height() : 1u);
        desc.format = format;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = label;
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        into = std::move(*made);
        auto v = into.view(0);
        if (!v) return std::move(v).error();
        view = std::move(*v);
        return ok();
    };
    ATHENEA_TRY(texture(meshCutTexture_, meshCutView_, cutWidth, cutHeight, rhi::Format::R32Uint, "meshLights.cut"));
    ATHENEA_TRY(texture(meshShadowTexture_, meshShadowView_, width, height, rhi::Format::R32Float,
                        "meshLights.shadow"));
    const uint64_t bytes = uint64_t{width} * height * 16;
    if (!meshRatio_.valid() || meshRatio_.bytes() < bytes) {
        gpu::BufferDesc desc;
        desc.bytes = std::max<uint64_t>(bytes, 16);
        desc.elementBytes = 16;
        desc.label = "meshLights.ratio";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        meshRatio_ = std::move(*made);
    }
    return ok();
}

Result<MaterialShading> MaterialShading::create(gpu::ShaderLibrary& library) {
    MaterialShading shading;
    shading.library_ = &library;
    shading.device_ = &library.device();
    return shading;
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs) {
    return setPrograms(programs, groups_);
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs, bool groups) {
    return setPrograms(programs, groups, clouds_);
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs, bool groups, bool clouds) {
    return setPrograms(programs, groups, clouds, domes_);
}

Result<void> MaterialShading::setPrograms(const MaterialPrograms& programs, bool groups, bool clouds, bool domes) {
    domes = domes && !domesRefused_;
    if (programs.module() == module_ && groups == groups_ && clouds == clouds_ && domes == domes_ &&
        kernel_.has_value()) {
        return ok();
    }
    const auto make = [&](bool shadows) -> Result<gpu::ComputeKernel> {
        // A frame with no cloud casting anything compiles exactly the kernel it
        // always did: the map is not declared, so it is not bound either.
        const std::string name = programs.module() + (shadows ? "_shade_shadowed" : "_shade") +
                                 (groups ? "_groups" : "") + (clouds ? "_cloudshadow" : "") +
                                 (domes ? "_domes" : "");
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                                   (groups ? kGroups : kNoGroups) + (shadows ? kShadowBits : kNoShadowBits) +
                                   (clouds ? kShadowMap : kNoShadowMap) +
                                   (domes ? std::string(kDomePrefilter) + kEvaluatedMask : kNoDomePrefilter) +
                                   kKernelBody;
        auto program = library_->loadSource(name, source, {"shadeMaterials"});
        if (!program) return std::move(program).error();
        return gpu::ComputeKernel::create(*library_, name, "shadeMaterials");
    };
    // The shadow rays' own kernel, which depends on the materials alone
    // (a cut-out occluder's opacity is asked of them).
    const bool canTrace = device_->caps().rayQuery && device_->caps().accelerationStructure;
    if (canTrace && !shadowsRefused_ && (!trace_.has_value() || traceModule_ != programs.module())) {
        const std::string name = programs.module() + "_shadow_rays";
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude + kShadowRay + kTraceBody;
        auto traced = [&]() -> Result<gpu::ComputeKernel> {
            auto program = library_->loadSource(name, source, {"traceShadows"});
            if (!program) return std::move(program).error();
            return gpu::ComputeKernel::create(*library_, name, "traceShadows");
        }();
        if (!traced) {
            // A device that can trace rays may still not hold this kernel
            // ("Compute function exceeds available stack space" at pipeline
            // creation, an iPad's GPU, when the lobe stack and the
            // intersector shared one kernel). Drawn without shadow rays is a
            // frame; failing is none. Said once, and kept to.
            log::warn("material shading: {}; shading without shadow rays on this device",
                      traced.error().toString());
            shadowsRefused_ = true;
            trace_.reset();
        } else {
            trace_.emplace(std::move(*traced));
            traceModule_ = programs.module();
        }
        // And the kernel that draws the lobe samples' directions for it: the
        // material, no ray.
        const std::string lobeName = programs.module() + "_lobe_dirs";
        const std::string lobeSource = "import " + programs.module() + ";\n" + kKernelPrelude + kLobeBody;
        auto program = library_->loadSource(lobeName, lobeSource, {"drawLobes"});
        if (!program) return std::move(program).error();
        auto drawn = gpu::ComputeKernel::create(*library_, lobeName, "drawLobes");
        if (!drawn) return std::move(drawn).error();
        lobes_.emplace(std::move(*drawn));
    }
    // The table a material the same everywhere is shaded from under the
    // domes alone (kTableBody): the materials, no light, no ray.
    if (domes && (!table_.has_value() || tableModule_ != programs.module())) {
        const std::string tableName = programs.module() + "_dome_table";
        const std::string tableSource = "import " + programs.module() + ";\n" + kKernelPrelude + kTableBody;
        auto program = library_->loadSource(tableName, tableSource, {"tabulateMaterials"});
        if (!program) return std::move(program).error();
        auto made = gpu::ComputeKernel::create(*library_, tableName, "tabulateMaterials");
        if (!made) return std::move(made).error();
        table_.emplace(std::move(*made));
        const std::string tabledName = programs.module() + "_dome_tabled";
        const std::string tabledSource = "import " + programs.module() + ";\n" + kKernelPrelude + kTabledBody;
        auto tabledProgram = library_->loadSource(tabledName, tabledSource, {"shadeTabled"});
        if (!tabledProgram) return std::move(tabledProgram).error();
        auto tabled = gpu::ComputeKernel::create(*library_, tabledName, "shadeTabled");
        if (!tabled) return std::move(tabled).error();
        tabled_.emplace(std::move(*tabled));
        tableModule_ = programs.module();
    }
    const bool shadows = canTrace && !shadowsRefused_;
    auto kernel = make(shadows);
    if (!kernel && domes) {
        // A device that will not make the kernel with the domes' G-buffer
        // (an internal compiler error, as a UsdPreviewSurface ground once
        // drew from Metal) still draws a frame: with the domes sampled, as
        // before there was a prefiltered read. Said once, and kept to.
        log::warn("material shading: {}; the domes are sampled on this device", kernel.error().toString());
        domesRefused_ = true;
        domes = false;
        kernel = make(shadows);
    }
    if (!kernel) return std::move(kernel).error();
    // The reflections' kernel (kReflectionsBody).
    {
        const std::string name = programs.module() + "_reflections";
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude + kReflectionsBody;
        auto program = library_->loadSource(name, source, {"shadeReflections"});
        if (!program) return std::move(program).error();
        auto made = gpu::ComputeKernel::create(*library_, name, "shadeReflections");
        if (!made) return std::move(made).error();
        reflections_.emplace(std::move(*made));
    }
    // The unlit surfaces' kernel, of the same variant (kUnlitBody).
    {
        const std::string name = programs.module() + "_unlit" + (groups ? "_groups" : "") + (domes ? "_domes" : "");
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                                   (groups ? kGroups : kNoGroups) +
                                   (domes ? std::string(kDomePrefilter) : std::string(kNoDomePrefilter)) + kUnlitBody;
        auto program = library_->loadSource(name, source, {"shadeUnlit"});
        if (!program) return std::move(program).error();
        auto made = gpu::ComputeKernel::create(*library_, name, "shadeUnlit");
        if (!made) return std::move(made).error();
        unlit_.emplace(std::move(*made));
    }
    // The pixels that only reflect under the domes alone, evaluated without
    // the lights (kEvaluatedBody). A device that will not make it shades
    // them in the shading kernel, as before; said once a set of materials.
    evaluated_.reset();
    if (domes) {
        const std::string name = programs.module() + "_dome_evaluated" + (groups ? "_groups" : "");
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                                   (groups ? kGroups : kNoGroups) + kDomePrefilter + kEvaluatedBody;
        auto made = [&]() -> Result<gpu::ComputeKernel> {
            auto program = library_->loadSource(name, source, {"shadeEvaluated"});
            if (!program) return std::move(program).error();
            return gpu::ComputeKernel::create(*library_, name, "shadeEvaluated");
        }();
        if (made) {
            evaluated_.emplace(std::move(*made));
        } else {
            log::warn("material shading: {}; every pixel in the shading kernel on this device",
                      made.error().toString());
        }
    }
    kernel_.emplace(std::move(*kernel));
    shadowed_ = shadows;
    module_ = programs.module();
    groups_ = groups;
    clouds_ = clouds;
    domes_ = domes;
    return ok();
}

Result<void> MaterialShading::shade(gpu::CommandBatch& batch, const VisibilityTargets& targets,
                                    const render::Projection& projection, const MaterialFrame& frame,
                                    render::RenderTargets& out) {
    if (!kernel_.has_value()) {
        return Error(ErrorCode::InvalidArgument, "material shading: no materials set");
    }
    const bool groups = frame.groups.count > 0;
    if (groups && (frame.groups.colour == nullptr || !frame.groups.colour->valid())) {
        return Error(ErrorCode::InvalidArgument, "material shading: light groups asked for without their buffer");
    }
    const bool clouds = frame.cloudShadow != nullptr;
    const bool domes = frame.domeLighting != nullptr && !domesRefused_;
    if ((groups != groups_ || clouds != clouds_ || domes != domes_) && frame.programs != nullptr) {
        ATHENEA_TRY(setPrograms(*frame.programs, groups, clouds, domes));
    }
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
    // What the shading kernel leaves the dome kernel, a texel a pixel.
    if (domes && (!domeDiffuse_.valid() || domeDiffuse_.width() != targets.width ||
                  domeDiffuse_.height() != targets.height)) {
        const auto make = [&](gpu::Texture& into, rhi::ComPtr<rhi::ITextureView>& view,
                              const char* label) -> Result<void> {
            gpu::TextureDesc desc;
            desc.type = rhi::TextureType::Texture2D;
            desc.width = targets.width;
            desc.height = targets.height;
            desc.format = rhi::Format::RGBA32Float;
            desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
            desc.label = label;
            auto made = gpu::Texture::create(*device_, desc);
            if (!made) return std::move(made).error();
            into = std::move(*made);
            auto madeView = into.view(0);
            if (!madeView) return std::move(madeView).error();
            view = std::move(*madeView);
            return ok();
        };
        ATHENEA_TRY(make(domeDiffuse_, domeDiffuseView_, "materials.domeDiffuse"));
        ATHENEA_TRY(make(domeGlossy_, domeGlossyView_, "materials.domeGlossy"));
        ATHENEA_TRY(make(domePoint_, domePointView_, "materials.domePoint"));
    }
    if (domes && !domeNoMap_.valid()) {
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2DArray;
        desc.width = 1;
        desc.height = 1;
        desc.arrayLength = 1;
        desc.format = rhi::Format::R32Float;
        desc.usage = rhi::TextureUsage::ShaderResource;
        desc.label = "materials.domeNoMap";
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        domeNoMap_ = std::move(*made);
        auto view = domeNoMap_.view(0);
        if (!view) return std::move(view).error();
        domeNoMapView_ = std::move(*view);
    }
    if (domes && !domeKernel_.has_value()) {
        auto made = gpu::ComputeKernel::create(*library_, "athenea/technique/dome_shade", "domeShade");
        if (!made) return std::move(made).error();
        domeKernel_.emplace(std::move(*made));
    }
    // ATHENEA_STAGES: each kernel submitted, waited for and timed.
    times_ = {};
    auto stageStart = std::chrono::steady_clock::now();
    const auto stage = [&](double& into) -> Result<void> {
        if (!timeStages_) {
            return ok();
        }
        ATHENEA_TRY(batch.submit(true));
        const auto now = std::chrono::steady_clock::now();
        into = std::chrono::duration<double, std::milli>(now - stageStart).count();
        stageStart = now;
        return ok();
    };
    if (timeStages_) {
        ATHENEA_TRY(batch.submit(true));   // what was recorded before is not ours
        stageStart = std::chrono::steady_clock::now();
    }
    if (out.width != targets.width || out.height != targets.height || !out.colour.valid()) {
        gpu::BufferDesc colour;
        colour.bytes = pixels * 16;
        colour.elementBytes = 16;
        colour.label = "materials.colour";
        auto madeColour = gpu::Buffer::create(*device_, colour);
        if (!madeColour) return std::move(madeColour).error();
        gpu::BufferDesc depth;
        depth.bytes = pixels * 4;
        depth.elementBytes = 4;
        depth.label = "materials.depth";
        auto madeDepth = gpu::Buffer::create(*device_, depth);
        if (!madeDepth) return std::move(madeDepth).error();
        out.colour = std::move(*madeColour);
        out.depth = std::move(*madeDepth);
        out.width = targets.width;
        out.height = targets.height;
    }
    auto ids = targets.ids.view(0);
    if (!ids) return std::move(ids).error();
    const uint32_t lightChoice = frame.lights == nullptr ? 0u
                                 : frame.chooseLights    ? (frame.lightBvh && frame.lights->hasBvh() ? 2u : 1u)
                                                         : 0u;
    // The mesh lights' samples a pixel, where the frame has clusters.
    const uint32_t meshSamples = frame.meshLights ? frame.meshLightSamples : 0u;
    const auto bindLights = [&](rhi::ShaderCursor cursor) {
        if (frame.lights != nullptr) {
            frame.lights->bind(cursor);
            cursor["lighting"]["samples"].setData(frame.samples);
            cursor["lighting"]["chooseLights"].setData(lightChoice);
            // A shadow ray walks on through what a cut-out's opacity removed,
            // by a lot, as the path tracer's does: asked in traceShadows,
            // which holds no lobe stack (kNoShadowBits says why that matters).
            cursor["lighting"]["shadowCutouts"].setData(uint32_t{frame.cutouts ? 1u : 0u});
            cursor["lighting"]["meshSamples"].setData(meshSamples);
        }
    };
    // The shadow rays first, one bit a light sample, in a kernel of their own.
    // Then a bit a lobe sample (lobeLookups() in the kernels), whose
    // directions drawLobes writes first.
    uint32_t shadowWords = 0;
    uint32_t lobes = 0;
    const uint32_t lightCount = frame.lights != nullptr ? frame.lights->count() : 0u;
    // A frame whose every mesh is unlit is shadeUnlit's alone: no lobe, no
    // shadow ray, no shading kernel, no dome (kUnlitBody, `unlitAlone`).
    const bool alone = !frame.lit && frame.unlitRows != nullptr && unlit_.has_value();
    const bool tracing = !(domes && frame.domesOnly) && !alone;
    // The mesh lights in closed form: where the frame has clusters and draws
    // no sample of them. The tiles' cut, and the shadow's ratio where rays are.
    const bool closedForm = frame.meshLights && frame.meshLightSamples == 0 && frame.lights != nullptr &&
                            ltc_.has_value() && meshCut_.has_value() && meshFilter_.has_value();
    constexpr uint32_t kCutK = 16;
    const uint32_t tilesX = (targets.width + 15) / 16;
    const uint32_t tilesY = (targets.height + 15) / 16;
    ATHENEA_TRY(ensureMeshTextures(closedForm ? tilesX * kCutK : 1, closedForm ? tilesY : 1,
                                   closedForm ? targets.width : 1, closedForm ? targets.height : 1));
    if (tracing && shadowed_ && trace_.has_value() && lobes_.has_value() && frame.shadows != nullptr &&
        (lightCount > 0 || meshSamples > 0 || closedForm)) {
        const uint64_t samples = std::max(frame.samples, 1u);
        lobes = lightCount > 0 ? static_cast<uint32_t>(std::min<uint64_t>(samples, kLobeLookups)) : 0u;
        const uint64_t lightBits = lightCount == 0 ? 0 : frame.chooseLights ? samples : samples * lightCount;
        const uint64_t bits = lightBits + lobes + meshSamples;
        shadowWords = static_cast<uint32_t>(std::max<uint64_t>((bits + 31) / 32, 1));
    }
    // The directions, and then a word a pixel: whether drawLobes found its
    // domes read prefiltered (kDomePrefilter).
    const uint64_t lobeBytes = std::max<uint64_t>(pixels * (lobes + 1) * 4, 4);
    if (lobes != 0 && (!lobeDirs_.valid() || lobeDirs_.bytes() < lobeBytes)) {
        gpu::BufferDesc desc;
        desc.bytes = lobeBytes;
        desc.elementBytes = 4;
        desc.label = "materials.lobeDirs";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        lobeDirs_ = std::move(*made);
    }
    const uint64_t shadowBytes = std::max<uint64_t>(pixels * shadowWords * 4, 4);
    if (shadowed_ && (!shadowBits_.valid() || shadowBits_.bytes() < shadowBytes)) {
        gpu::BufferDesc desc;
        desc.bytes = shadowBytes;
        desc.elementBytes = 4;
        desc.label = "materials.shadowBits";
        auto made = gpu::Buffer::create(*device_, desc);
        if (!made) return std::move(made).error();
        shadowBits_ = std::move(*made);
    }
    if (lobes != 0) {
        lobes_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["visibility"].setBinding((*ids).get());
            cursor["lobeDirsOut"].setBinding(lobeDirs_.rhi());
            cursor["domesClosed"].setData(uint32_t{domes ? 1u : 0u});
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
        ATHENEA_TRY(stage(times_.lobes));
    }
    if (shadowWords != 0) {
        trace_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["shadowScene"].setBinding(frame.shadows);
            cursor["visibility"].setBinding((*ids).get());
            cursor["shadowBitsOut"].setBinding(shadowBits_.rhi());
            cursor["shadowWords"].setData(shadowWords);
            cursor["lobeDirs"].setBinding(lobes != 0 ? lobeDirs_.rhi() : shadowBits_.rhi());
            cursor["lobesDrawn"].setData(uint32_t{lobes != 0 ? 1u : 0u});
            cursor["domesClosed"].setData(uint32_t{domes ? 1u : 0u});
            cursor["meshRatioOut"].setBinding(meshRatio_.rhi());
            cursor["meshRatio"].setData(uint32_t{closedForm ? 1u : 0u});
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
        ATHENEA_TRY(stage(times_.shadows));
    }
    const bool meshShadowed = closedForm && shadowWords != 0;
    if (closedForm) {
        meshCut_->dispatch(batch, {tilesX, tilesY, 1}, [&](rhi::ShaderCursor cursor) {
            // The scene a surface is rebuilt from and the view to world: not
            // the material frame, whose tables this kernel does not declare.
            bindScene(cursor, *frame.scene);
            const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
            cursor["toWorld"]["row0"].setData(toWorld.data(), sizeof(float) * 4);
            cursor["toWorld"]["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            cursor["toWorld"]["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            cursor["meshRegion"].setBinding(frame.lights->words().rhi());
            cursor["meshRegionWords"].setData(frame.lights->meshWords());
            cursor["visibility"].setBinding((*ids).get());
            cursor["cutOut"].setBinding(meshCutView_.get());
            cursor["cutK"].setData(kCutK);
            cursor["tilesX"].setData(tilesX);
            cursor["tilesY"].setData(tilesY);
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
        if (meshShadowed) {
            meshFilter_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
                cursor["ratio"].setBinding(meshRatio_.rhi());
                cursor["shadowOut"].setBinding(meshShadowView_.get());
                cursor["radius"].setData(uint32_t{4});
                setCamera(cursor["camera"], projection, targets.width, targets.height);
            });
        }
    }
    // A material the same everywhere, under the domes alone, is tabulated
    // here, a row a material, and its pixels shaded from it (shadeTabled).
    const uint32_t tableRows = domes && frame.domesOnly && !alone && table_.has_value() && tabled_.has_value()
                                   ? static_cast<uint32_t>(frame.records->count())
                                   : 0u;
    if (domes && (!tableTexture_.valid() || tableTexture_.height() < std::max(tableRows, 1u))) {
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.width = 32 * 3;   // kTableBins angles, three texels each
        desc.height = std::max(tableRows, 1u);
        desc.format = rhi::Format::RGBA32Float;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = "materials.domeTable";
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        tableTexture_ = std::move(*made);
        auto view = tableTexture_.view(0);
        if (!view) return std::move(view).error();
        tableView_ = std::move(*view);
    }
    if (tableRows != 0) {
        table_->dispatch(batch, {tableRows * 32, 1, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            cursor["materialTableOut"].setBinding(tableView_.get());
            cursor["materialTableRows"].setData(tableRows);
        });
    }
    // Then the pixels that only reflect but are not the same everywhere
    // (shadeEvaluated), which the shading kernel skips by the mask.
    if (domes && (evaluatedMask_.width() != targets.width || evaluatedMask_.height() != targets.height)) {
        gpu::TextureDesc desc;
        desc.type = rhi::TextureType::Texture2D;
        desc.width = targets.width;
        desc.height = targets.height;
        desc.format = rhi::Format::R32Uint;
        desc.usage = rhi::TextureUsage::UnorderedAccess | rhi::TextureUsage::ShaderResource;
        desc.label = "materials.evaluatedMask";
        auto made = gpu::Texture::create(*device_, desc);
        if (!made) return std::move(made).error();
        evaluatedMask_ = std::move(*made);
        auto view = evaluatedMask_.view(0);
        if (!view) return std::move(view).error();
        evaluatedMaskView_ = std::move(*view);
    }
    // Not run where every material is the table's: over a whole frame it
    // still reads every pixel's visibility and writes the mask.
    const bool evaluating = tableRows != 0 && evaluated_.has_value() && frame.varying;
    if (evaluating) {
        evaluated_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["visibility"].setBinding((*ids).get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            cursor["materialTable"].setBinding(tableView_.get());
            cursor["materialTableRows"].setData(tableRows);
            cursor["domeDiffuseOut"].setBinding(domeDiffuseView_.get());
            cursor["domeGlossyOut"].setBinding(domeGlossyView_.get());
            cursor["domePointOut"].setBinding(domePointView_.get());
            cursor["evaluatedOut"].setBinding(evaluatedMaskView_.get());
            if (groups) {
                cursor["groupColour"].setBinding(frame.groups.colour->rhi());
                cursor["groupCount"].setData(std::min(frame.groups.count, kMaxLightGroups));
            }
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
    }
    if (!alone) {
        kernel_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            // The lights are these two kernels' alone.
            bindLights(cursor);
            if (shadowed_) {
                cursor["shadowBits"].setBinding(shadowBits_.rhi());
                cursor["shadowWords"].setData(shadowWords);
            }
            if (clouds) {
                cursor["cloudShadow"].setBinding(frame.cloudShadow);
                cursor["cloudShadowChain"].setBinding(frame.cloudShadowChain);
            }
            if (domes) {
                cursor["domeDiffuseOut"].setBinding(domeDiffuseView_.get());
                cursor["domeGlossyOut"].setBinding(domeGlossyView_.get());
                cursor["domePointOut"].setBinding(domePointView_.get());
                cursor["materialTable"].setBinding(tableView_.get());
                cursor["materialTableRows"].setData(tableRows);
                cursor["evaluatedMask"].setBinding(evaluatedMaskView_.get());
                cursor["evaluatedRan"].setData(uint32_t{evaluating ? 1u : 0u});
            }
            cursor["visibility"].setBinding((*ids).get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            if (groups) {
                cursor["groupColour"].setBinding(frame.groups.colour->rhi());
                cursor["groupCount"].setData(std::min(frame.groups.count, kMaxLightGroups));
            }
            cursor["meshAnalytic"].setData(uint32_t{closedForm ? 1u : 0u});
            cursor["meshCut"].setBinding(meshCutView_.get());
            cursor["meshCutK"].setData(closedForm ? kCutK : 0u);
            cursor["ltcTable"].setBinding(ltc_.has_value() ? ltc_->view() : meshShadowView_.get());
            cursor["meshShadow"].setBinding(meshShadowView_.get());
            cursor["meshShadowOn"].setData(uint32_t{meshShadowed ? 1u : 0u});
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
    }
    // The unlit surfaces (kUnlitBody), where the frame has any.
    if (frame.unlitRows != nullptr && unlit_.has_value()) {
        unlit_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["visibility"].setBinding((*ids).get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            cursor["unlitRows"].setBinding(frame.unlitRows->rhi());
            cursor["unlitRowCount"].setData(static_cast<uint32_t>(frame.unlitRows->bytes() / 8));
            cursor["unlitSlot"].setData(frame.unlitSlot);
            cursor["stSlot"].setData(frame.stSlot);
            cursor["unlitAlone"].setData(uint32_t{alone ? 1u : 0u});
            if (domes) {
                cursor["domeDiffuseOut"].setBinding(domeDiffuseView_.get());
                cursor["domeGlossyOut"].setBinding(domeGlossyView_.get());
                cursor["domePointOut"].setBinding(domePointView_.get());
                cursor["materialTable"].setBinding(tableView_.get());
                cursor["materialTableRows"].setData(uint32_t{0});
            }
            if (groups) {
                cursor["groupColour"].setBinding(frame.groups.colour->rhi());
                cursor["groupCount"].setData(std::min(frame.groups.count, kMaxLightGroups));
            }
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
    }
    // The pixels the shading kernel stepped aside for: a material the same
    // everywhere, shaded from the table (kTabledBody).
    if (tableRows != 0) {
        tabled_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["visibility"].setBinding((*ids).get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            cursor["materialTable"].setBinding(tableView_.get());
            cursor["materialTableRows"].setData(tableRows);
            cursor["domeDiffuseOut"].setBinding(domeDiffuseView_.get());
            cursor["domeGlossyOut"].setBinding(domeGlossyView_.get());
            cursor["domePointOut"].setBinding(domePointView_.get());
            cursor["groupColour"].setBinding(groups ? frame.groups.colour->rhi() : out.colour.rhi());
            cursor["groupCount"].setData(groups ? std::min(frame.groups.count, kMaxLightGroups) : 0u);
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
    }
    ATHENEA_TRY(stage(times_.shade));
    // THE DOMES, PREFILTERED: added over what the shading kernel wrote, at
    // the pixels it left them to (dome_shade.slang).
    if (domes && frame.lights != nullptr && !alone) {
        const std::array<float, 12> toWorld = aofx::xform::inverseAffine(projection.worldToView).rows3x4();
        domeKernel_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            frame.lights->bind(cursor);
            frame.textures->bind(cursor["gTextures"]);
            cursor["toWorld"]["row0"].setData(toWorld.data(), sizeof(float) * 4);
            cursor["toWorld"]["row1"].setData(toWorld.data() + 4, sizeof(float) * 4);
            cursor["toWorld"]["row2"].setData(toWorld.data() + 8, sizeof(float) * 4);
            cursor["params"]["clouds"].setData(uint32_t{clouds ? 1u : 0u});
            cursor["domeDiffuse"].setBinding(domeDiffuseView_.get());
            cursor["domeGlossy"].setBinding(domeGlossyView_.get());
            cursor["domePoint"].setBinding(domePointView_.get());
            cursor["domeLighting"].setBinding(frame.domeLighting);
            // Without a map the kernel never reads one; a texel stands in.
            cursor["cloudShadow"].setBinding(clouds ? frame.cloudShadow : domeNoMapView_.get());
            cursor["cloudShadowChain"].setBinding(clouds ? frame.cloudShadowChain : domeNoMapView_.get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["groupColour"].setBinding(groups ? frame.groups.colour->rhi() : out.colour.rhi());
            cursor["groupCount"].setData(groups ? std::min(frame.groups.count, kMaxLightGroups) : 0u);
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
        ATHENEA_TRY(stage(times_.domes));
    }
    // THE REFLECTIONS (kReflectionsBody), from a copy of what is shaded.
    if (frame.reflections && reflections_.has_value()) {
        const uint64_t bytes = pixels * 16;
        if (!reflectionSource_.valid() || reflectionSource_.bytes() < bytes) {
            gpu::BufferDesc desc;
            desc.bytes = bytes;
            desc.elementBytes = 16;
            desc.label = "materials.reflectionSource";
            auto made = gpu::Buffer::create(*device_, desc);
            if (!made) return std::move(made).error();
            reflectionSource_ = std::move(*made);
        }
        // What the domes added is in the copy: they are done first.
        ATHENEA_TRY(batch.submit(true));
        batch.encoder()->copyBuffer(reflectionSource_.rhi(), 0, out.colour.rhi(), 0, bytes);
        batch.markDirty();
        ATHENEA_TRY(batch.submit(true));
        reflections_->dispatch(batch, {targets.width, targets.height, 1}, [&](rhi::ShaderCursor cursor) {
            bindMaterialFrame(cursor, frame, projection);
            bindLights(cursor);
            cursor["visibility"].setBinding((*ids).get());
            cursor["colour"].setBinding(out.colour.rhi());
            cursor["depth"].setBinding(out.depth.rhi());
            cursor["source"].setBinding(reflectionSource_.rhi());
            const bool glossy = domes && domeDiffuseView_ && domeGlossyView_;
            cursor["domeDiffuseIn"].setBinding(glossy ? domeDiffuseView_.get() : domeNoMapView_.get());
            cursor["domeGlossyIn"].setBinding(glossy ? domeGlossyView_.get() : domeNoMapView_.get());
            cursor["hasGlossy"].setData(uint32_t{glossy ? 1u : 0u});
            cursor["reflectanceSlot"].setData(frame.reflectanceSlot);
            cursor["roughnessSlot"].setData(frame.roughnessSlot);
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
        ATHENEA_TRY(stage(times_.reflections));
    }
    return ok();
}

}   // namespace athenea::technique
