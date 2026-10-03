// Copyright (c) 2026 jesus luque.
#include "athenea/technique/MaterialShading.h"

#include "athenea/core/Log.h"

#include <algorithm>

#include "athenea/gpu/CommandBatch.h"
#include "athenea/gpu/Device.h"
#include "athenea/gpu/ShaderLibrary.h"

namespace athenea::technique {

namespace {

const char* kKernelPrelude = R"(
import athenea.common.octahedral;
import athenea.light.lights_image;
import athenea.light.light_bvh;

struct LightingParams {
    uint  samples;
    uint  chooseLights;
    uint  shadowCutouts;   // 1: a material of the frame cuts, and a shadow ray asks what it meets
    uint  pad1;
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
    const uint samples = max(lighting.samples, 1u);
    return lighting.chooseLights != 0 ? samples : lightCount * samples;
}
uint lobeLookups() {
    return lightCount != 0 ? min(max(lighting.samples, 1u), kLobeLookups) : 0u;
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
)";

const char* kNoShadowMap = R"(
static const bool kCloudShadows = false;
void cloudPixelSpan(CameraParams c, ViewToWorld w, uint2 px, float3 p, float3 n, out float3 dx, out float3 dy) {
    dx = float3(0.0);
    dy = float3(0.0);
}
float cloudTransmittance(float3 p, float3 dx, float3 dy, uint light, bool casts) { return 1.0; }
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
RWStructuredBuffer<uint> lobeDirsOut;

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
StructuredBuffer<uint>   lobeDirs;      // drawLobes' directions, lobeLookups() a pixel
uniform uint             lobesDrawn;    // 0: no lobe directions this frame

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
    if (seen.x == 0 || lightCount == 0) {
        return;   // shading reads no bit there
    }
    const uint at = tid.y * camera.width + tid.x;
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
    const MaterialInputs inputs = materialInputsAt(camera, toWorld, tid.x, tid.y, s, lookup.time, false);
    const float3 p = inputs.positionWorld;
    const float3 n = inputs.normalWorld;
    const uint samples = max(lighting.samples, 1u);
    uint word = 0u;
    uint bit = 0u;
    if (lighting.chooseLights != 0) {
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
                if ((light.flags & kLightShadow) != 0 &&
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
            const bool traced = (light.flags & kLightShadow) != 0 &&
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
    const uint lobes = lobesDrawn != 0 ? lobeLookups() : 0u;
    for (uint j = 0; j < lobes; ++j) {
        const uint packed = lobeDirs[at * lobes + j];
        const bool blocked =
            packed != kNoLobe && occluded(p, n, unpackLobe(packed), 1.0e30, kLightUnlinked);
        putShadowBit(at, word, bit, blocked);
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
        return;
    }
    const Surface s = surfaceAt(camera, tid.x, tid.y, seen);
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
    float3 radiance = stack.emission;
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
    const uint lobeCount = lobeLookups();
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
                const float3 arrived = weight * ms.weight * lh.radiance;
                lobeSum += arrived;
                if (kLightGroups && light.group != 0 && light.group <= 8) {
                    groups[light.group - 1] += arrived / float(lobeCount);
                }
            }
        }
        radiance += lobeSum / float(lobeCount);
    }
    if (lightCount == 0) {
        // The headlight: unit radiance from the eye, so a white Lambert
        // surface facing it shows 1. What meshes were lit by before there
        // were lights.
        radiance += kPi * stackEval(stack, toEye, toEye);
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
            if (!ls.delta && rasterWeighs(light)) {
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
                cloudTransmittance(inputs.positionWorld, cloudDx, cloudDy, choice.index, (light.flags & kLightShadow) != 0),
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
                if (!ls.delta && rasterWeighs(light)) {
                    weight = rasterMis(float(samples), ls.pdf, float(lobeCount), lobeDensity(toEye, ls.wi));
                }
                if (shadow && occludedSample(at, k * samples + i)) {
                    continue;
                }
                sum += shadowTint(light, cloudTransmittance(inputs.positionWorld, cloudDx, cloudDy, k, shadow), ls.distance) * weight *
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
    depth[at] = s.depth;
    writeGroups(at, pixels, groups, coverage);
}
)";

/// kLobeLookups in the kernels' prelude.
constexpr uint64_t kLobeLookups = 32;

}   // namespace

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
    if (programs.module() == module_ && groups == groups_ && clouds == clouds_ && kernel_.has_value()) {
        return ok();
    }
    const auto make = [&](bool shadows) -> Result<gpu::ComputeKernel> {
        // A frame with no cloud casting anything compiles exactly the kernel it
        // always did: the map is not declared, so it is not bound either.
        const std::string name = programs.module() + (shadows ? "_shade_shadowed" : "_shade") +
                                 (groups ? "_groups" : "") + (clouds ? "_cloudshadow" : "");
        const std::string source = "import " + programs.module() + ";\n" + kKernelPrelude +
                                   (groups ? kGroups : kNoGroups) + (shadows ? kShadowBits : kNoShadowBits) +
                                   (clouds ? kShadowMap : kNoShadowMap) + kKernelBody;
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
    const bool shadows = canTrace && !shadowsRefused_;
    auto kernel = make(shadows);
    if (!kernel) return std::move(kernel).error();
    kernel_.emplace(std::move(*kernel));
    shadowed_ = shadows;
    module_ = programs.module();
    groups_ = groups;
    clouds_ = clouds;
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
    if ((groups != groups_ || clouds != clouds_) && frame.programs != nullptr) {
        ATHENEA_TRY(setPrograms(*frame.programs, groups, clouds));
    }
    const uint64_t pixels = uint64_t{targets.width} * targets.height;
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
    const auto bindLights = [&](rhi::ShaderCursor cursor) {
        if (frame.lights != nullptr) {
            frame.lights->bind(cursor);
            cursor["lighting"]["samples"].setData(frame.samples);
            cursor["lighting"]["chooseLights"].setData(lightChoice);
            // A shadow ray walks on through what a cut-out's opacity removed,
            // by a lot, as the path tracer's does: asked in traceShadows,
            // which holds no lobe stack (kNoShadowBits says why that matters).
            cursor["lighting"]["shadowCutouts"].setData(uint32_t{frame.cutouts ? 1u : 0u});
        }
    };
    // The shadow rays first, one bit a light sample, in a kernel of their own.
    // Then a bit a lobe sample (lobeLookups() in the kernels), whose
    // directions drawLobes writes first.
    uint32_t shadowWords = 0;
    uint32_t lobes = 0;
    const uint32_t lightCount = frame.lights != nullptr ? frame.lights->count() : 0u;
    if (shadowed_ && trace_.has_value() && lobes_.has_value() && frame.shadows != nullptr && lightCount > 0) {
        const uint64_t samples = std::max(frame.samples, 1u);
        lobes = static_cast<uint32_t>(std::min<uint64_t>(samples, kLobeLookups));
        const uint64_t bits = (frame.chooseLights ? samples : samples * lightCount) + lobes;
        shadowWords = static_cast<uint32_t>((bits + 31) / 32);
    }
    const uint64_t lobeBytes = std::max<uint64_t>(pixels * lobes * 4, 4);
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
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
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
            setCamera(cursor["camera"], projection, targets.width, targets.height);
        });
    }
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
        cursor["visibility"].setBinding((*ids).get());
        cursor["colour"].setBinding(out.colour.rhi());
        cursor["depth"].setBinding(out.depth.rhi());
        if (groups) {
            cursor["groupColour"].setBinding(frame.groups.colour->rhi());
            cursor["groupCount"].setData(std::min(frame.groups.count, kMaxLightGroups));
        }
        setCamera(cursor["camera"], projection, targets.width, targets.height);
    });
    return ok();
}

}   // namespace athenea::technique
