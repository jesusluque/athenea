// Copyright (c) 2026 jesus luque.
//
// WHAT A CLOUD STOPS, SEEN FROM THE LIGHT, WITHOUT A RAY AND WITHOUT A SORT.
//
// The transmittance along a ray through a cloud is a product over the
// gaussians it meets, and a product does not care what order its factors come
// in. So a pass from the light needs no depth sort and no per-pixel list: each
// gaussian adds its optical depth to the texels its footprint covers, and what
// the texel holds at the end is the whole cloud's.
//
// With sigma_i(x) = -ln(1 - alpha_i(x)) for the alpha gaussian i has at texel
// x, the accumulated
//
//     a0(x) = sum_i sigma_i(x)      gives     T(x) = exp(-a0(x))
//
// which is EXACT for a receiver standing behind the whole cloud -- a ground
// plane under a bird -- and is the zeroth term of the Fourier expansion
// (Jansen & Bavoil, I3D 2010) that a receiver inside the cloud needs. One code
// path, `coefficients` says how many terms; one is the total.
//
// This is the pass. What reads the map is next door in splat_shadow_read,
// which declares no parameter of its own: a shading kernel importing this one
// would inherit every buffer below, and it is at Metal's limit already.
module splat_shadow_map;

import athenea.common.packing;
import athenea.light.lights;
import athenea.splat.frame;   // cutoffPower and footprintExtent: one footprint, one meaning
import athenea.technique.splat_shadow_read;

// --- the pass ------------------------------------------------------------

public struct ShadowMapParams {
    public uint  count;        ///< gaussians in this instance
    public uint  light;        ///< which slot is being written
    public uint  words;        ///< the whole buffer, for the clear
    public uint  resolution;
    public uint  coefficients;
    public float margin;       ///< how much wider than the caster's box the map is
    public uint  lightCount;
    public uint  base;         ///< the factors receiver: this instance's first slot
    /// What the optical depth is multiplied by as it is accumulated: the
    /// shadow's density, as a compositor means it. 1 is what the cloud's own
    /// opacity says and is the only physical value; less lets light through a
    /// solid thing, more darkens a thin one.
    public float density;
    public uint  level;        ///< the transmittance chain: which level a dispatch writes
    /// The caster's box in its own space, and its object-to-world rows: the
    /// box is a bound the host holds from the upload; where it lands is the
    /// device's business (`shadowMapBox`).
    public float3 boxMin;
    public float  pad1;
    public float3 boxMax;
    public float  pad2;
    public float  w00, w01, w02, w03;
    public float  w10, w11, w12, w13;
    public float  w20, w21, w22, w23;
    /// The caster's box as it was bound, in its own space: what a skinned
    /// cloud's map is sized from, since the posed box breathes with the
    /// wings and a map sized from it changed its texels every frame.
    public float3 restMin;
    /// How far behind the nearest caster a receiver still counts as on it,
    /// in world units; 0 is two per cent of the slab, a slab that does not
    /// change with the pose.
    public float  selfBias;
    public float3 restMax;
    public float  pad4;
    /// THE DOME'S DIRECTIONS (docs/decisions.md, task TX: a cloud's shadow on
    /// a mesh under a sky): how many slots after the lights' are maps along
    /// directions of the first shadow-casting dome among them -- its zenith
    /// and a ring forty degrees up. 0: none.
    public uint  domeSlots;
    public uint  slots;        ///< the lights' and the dome's together
    public uint  pad5;
    public uint  pad6;
};

public ConstantBuffer<ShadowMapParams> shadowParams;
public RWStructuredBuffer<uint>        shadowMap;
public StructuredBuffer<LightRecord>   shadowLights;
public StructuredBuffer<float4>        shadowPositions;   ///< posed, the cloud's own space
public StructuredBuffer<uint>          shadowShape;       ///< 4 words a gaussian, as GpuSplats packs them

/// The pass's own buffer as a source the read understands: the gaussian
/// receivers and the probe read it through this, a floor through the
/// texture, and the two through the same `shadowMapRead`.
struct ShadowBufferSource : IShadowMapSource {
    static float headerReal(uint word) { return asfloat(shadowMap[word]); }
    static uint headerCount(uint word) { return shadowMap[word]; }
    static float texel(ShadowMapFrame f, uint light, int x, int y, uint which) {
        const uint word = f.offset + (uint(y) * f.resolution + uint(x)) * shadowStrideOf(f.coefficients) + which;
        const uint raw = shadowMap[word];
        // The total is unsigned, every Fourier term is signed, and the last
        // word of a texel is a depth stored as an ordered key.
        return which == f.coefficients ? shadowOrderedFloat(raw)
               : which == 0            ? float(raw) / kShadowFixedScale
                                       : float(int(raw)) / kShadowFixedScale;
    }
    // The buffer has no chain: a receiver read from it is a point.
    static uint levels(ShadowMapFrame f) { return 1u; }
    static float transmittanceAt(ShadowMapFrame f, uint light, uint level, int x, int y) {
        return exp(-max(texel(f, light, x, y, 0), 0.0));
    }
};

float3 toWorld(float3 p) {
    const ShadowMapParams s = shadowParams;
    return float3(dot(float3(s.w00, s.w01, s.w02), p) + s.w03,
                  dot(float3(s.w10, s.w11, s.w12), p) + s.w13,
                  dot(float3(s.w20, s.w21, s.w22), p) + s.w23);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void shadowMapClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= shadowParams.words) {
        return;
    }
    // Two kinds of word start at the largest key there is, because both are
    // reduced with a minimum: the three the casters' box is gathered into,
    // and the last word of every texel, which holds the depth of the nearest
    // thing that cast into it. The box's maxima and the rest radius start at
    // the smallest, and everything else at nothing, which is what a sum
    // wants.
    const uint header = shadowHeaderWords();
    bool smallest = tid.x >= kShadowBoxAt && tid.x < kShadowBoxAt + 3;
    const bool largest = tid.x >= kShadowBoxAt + 3 && tid.x < kShadowBoxAt + kShadowBoxWords;
    if (tid.x >= header) {
        const uint stride = shadowStrideOf(shadowParams.coefficients);
        smallest = (tid.x - header) % stride == shadowParams.coefficients;
    }
    // Finite sentinels, not the extreme keys: those decode to a NaN, and a
    // comparison against a NaN is not something a device compiling with fast
    // arithmetic will answer the way IEEE says. Measured: an untouched texel
    // read as lit and a whole corner of the cloud stopped casting.
    shadowMap[tid.x] = smallest  ? shadowOrderedKey(1.0e30)
                       : largest ? shadowOrderedKey(-1.0e30)
                                 : 0u;
}

/// Eight threads a caster: the corners of its box in world space, reduced
/// into the six words every frame is then centred from, and the corners of
/// its rest box, whose farthest from the rest box's centre is the radius the
/// frame is sized from. The boxes are bounds the host already holds; where
/// their corners land is not something the host works out.
[shader("compute")]
[numthreads(8, 1, 1)]
void shadowMapBox(uint3 tid: SV_DispatchThreadID) {
    const uint corner = tid.x;
    if (corner >= 8) {
        return;
    }
    const ShadowMapParams s = shadowParams;
    const float3 p = float3((corner & 1u) != 0u ? s.boxMax.x : s.boxMin.x,
                            (corner & 2u) != 0u ? s.boxMax.y : s.boxMin.y,
                            (corner & 4u) != 0u ? s.boxMax.z : s.boxMin.z);
    const float3 world = toWorld(p);
    for (uint k = 0; k < 3; ++k) {
        InterlockedMin(shadowMap[kShadowBoxAt + k], shadowOrderedKey(world[k]));
        InterlockedMax(shadowMap[kShadowBoxAt + 3 + k], shadowOrderedKey(world[k]));
    }
    // The rest sphere: the half diagonal of the bind pose's box as the object
    // transform carries it, which no pose changes.
    const float3 rest = float3((corner & 1u) != 0u ? s.restMax.x : s.restMin.x,
                               (corner & 2u) != 0u ? s.restMax.y : s.restMin.y,
                               (corner & 4u) != 0u ? s.restMax.z : s.restMin.z);
    const float radius = length(toWorld(rest) - toWorld(0.5 * (s.restMin + s.restMax)));
    InterlockedMax(shadowMap[kShadowBoxAt + 6], shadowOrderedKey(radius));
}

/// One thread a light: the map's frame, from the light's own axes and the box
/// the caster occupies. A distant light gets an orthographic frame -- it is
/// the only kind whose rays are parallel -- and a light that stands somewhere
/// gets one about the direction from it to the box, which is exact for a
/// light far enough away and is what a spot is.
///
/// A FRAME THAT DOES NOT BREATHE WITH THE WINGS. The map used to be framed on
/// the posed box: every flap changed its widest extent, so the texel's size
/// and phase changed under a body that had not moved, and the edge of its
/// shadow shimmered; the slab changed with it, and the bias and the Fourier
/// frequencies with the slab. Now
///
/// - the extent is the casters' REST sphere (or the posed box where that is
///   wider), times the margin, rounded UP to a quarter octave -- so it
///   changes only when the casters' reach crosses a step, never with a pose
///   inside it;
/// - the centre is the posed box's, so the map follows a bird that flies,
///   but SNAPPED to the texel grid in u and v and to a quarter of the slab in
///   depth: a point that does not move lands at the same place in its texel,
///   and at the same depth, whatever else does;
/// - the slab is the extent's, and the bias is a length in the world.
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowMapFrame(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= kShadowSlots) {
        return;
    }
    const uint at = k * kShadowFrameWords;
    // A slot after the lights' is one of the dome's directions, if a dome
    // that casts shadows is among the lights.
    const bool domeDirection = k >= shadowParams.lightCount && k < shadowParams.lightCount + shadowParams.domeSlots;
    uint domeAt = 0xFFFFFFFFu;
    if (domeDirection) {
        for (uint d = 0; d < shadowParams.lightCount && domeAt == 0xFFFFFFFFu; ++d) {
            const LightRecord candidate = shadowLights[d];
            if (candidate.kind == kLightDome && (candidate.flags & kLightShadow) != 0) {
                domeAt = d;
            }
        }
    }
    if (k >= shadowParams.lightCount && domeAt == 0xFFFFFFFFu) {
        shadowMap[at + 19] = 0;
        return;
    }
    const LightRecord light = shadowLights[domeDirection ? domeAt : k];
    const ShadowMapParams s = shadowParams;
    const float3 boxMin = float3(shadowOrderedFloat(shadowMap[kShadowBoxAt + 0]), shadowOrderedFloat(shadowMap[kShadowBoxAt + 1]),
                                 shadowOrderedFloat(shadowMap[kShadowBoxAt + 2]));
    const float3 boxMax = float3(shadowOrderedFloat(shadowMap[kShadowBoxAt + 3]), shadowOrderedFloat(shadowMap[kShadowBoxAt + 4]),
                                 shadowOrderedFloat(shadowMap[kShadowBoxAt + 5]));
    const float restRadius = shadowOrderedFloat(shadowMap[kShadowBoxAt + 6]);
    if (!all(boxMax >= boxMin)) {
        shadowMap[at + 19] = 0;   // nothing cast anything
        return;
    }
    const float3 centre = 0.5 * (boxMin + boxMax);
    const float3 half3 = max(0.5 * (boxMax - boxMin), float3(1.0e-6));
    // Which way the light travels. UsdLux shines down its own -Z.
    float3 axisZ = -lightAxisZ(light);
    if (domeDirection) {
        // The dome's pole is its own +Y, carried into the world by its rows
        // (a Z-up stage turns it there). Slot 0 of the dome's is the zenith,
        // the rest a ring forty degrees above the horizon.
        const float3 pole = normalize(float3(light.row0.y, light.row1.y, light.row2.y));
        const uint j = k - shadowParams.lightCount;
        float3 towards = pole;
        if (j > 0u) {
            const float3 helper = abs(pole.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
            const float3 t = normalize(cross(helper, pole));
            const float3 b = cross(pole, t);
            const float phi = 2.0 * kShadowPi * float(j - 1u) / float(max(shadowParams.domeSlots - 1u, 1u));
            const float elevation = 0.6981317;   // forty degrees
            towards = normalize(pole * sin(elevation) + (t * cos(phi) + b * sin(phi)) * cos(elevation));
        }
        axisZ = -towards;
    } else if (light.kind != kLightDistant) {
        const float3 origin = lightOrigin(light);
        const float3 toBox = centre - origin;
        const float reach = length(toBox);
        if (!(reach > 1.0e-6)) {
            shadowMap[at + 19] = 0;
            return;
        }
        axisZ = toBox / reach;
    }
    if (!domeDirection && (light.kind == kLightDome || (light.flags & kLightShadow) == 0)) {
        // A dome has no direction to build a map about, and a light that casts
        // no shadow casts none here either.
        shadowMap[at + 19] = 0;
        return;
    }
    // An orthonormal frame about it.
    const float3 up = abs(axisZ.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 axisX = normalize(cross(up, axisZ));
    const float3 axisY = cross(axisZ, axisX);
    // The posed box's extent along each axis: its half diagonal projected,
    // which is the support function of a box and so exact for any rotation.
    // It is never more than the posed box's half diagonal, so a pose that
    // stays inside the rest sphere never reaches past it.
    const float extentX = abs(half3.x * axisX.x) + abs(half3.y * axisX.y) + abs(half3.z * axisX.z);
    const float extentY = abs(half3.x * axisY.x) + abs(half3.y * axisY.y) + abs(half3.z * axisY.z);
    const float extentZ = abs(half3.x * axisZ.x) + abs(half3.y * axisZ.y) + abs(half3.z * axisZ.z);
    // A skinned cloud is posed away from the box it was bound in, so the map
    // covers more than the bind pose: a wing that leaves the map stops
    // casting, and that is the one artefact worth paying a margin to avoid.
    const float reach = max(max(max(extentX, extentY), extentZ), max(restRadius, 1.0e-6)) * (1.0 + s.margin);
    const float widest = exp2(ceil(4.0 * log2(reach)) * 0.25);
    const float texels = float(s.resolution) / (2.0 * widest);
    const float slab = 2.0 * widest;
    // The centre on the world's texel grid: a whole number of texels, so a
    // point keeps its place in its texel when the centre moves.
    const float gridU = round(dot(axisX, centre) * texels);
    const float gridV = round(dot(axisY, centre) * texels);
    // In depth the quantum is a quarter of the slab: coarse, because a shift
    // of the depth origin moves every Fourier term's phase and the
    // reconstruction with it, and a quarter still keeps every caster inside
    // (its reach is at most 1 / (1 + margin) of the half slab, and the snap
    // moves the centre by an eighth of the slab at most).
    const float steps = round(dot(axisZ, centre) / slab * 4.0);
    const float4 rowU = float4(axisX * texels, 0.5 * float(s.resolution) - gridU);
    const float4 rowV = float4(axisY * texels, 0.5 * float(s.resolution) - gridV);
    const float4 rowZ = float4(axisZ / slab, 0.5 - steps / 4.0);
    const float bias = (s.selfBias > 0.0 ? s.selfBias : 0.02 * slab) / slab;
    shadowMap[at + 0] = asuint(rowU.x); shadowMap[at + 1] = asuint(rowU.y);
    shadowMap[at + 2] = asuint(rowU.z); shadowMap[at + 3] = asuint(rowU.w);
    shadowMap[at + 4] = asuint(rowV.x); shadowMap[at + 5] = asuint(rowV.y);
    shadowMap[at + 6] = asuint(rowV.z); shadowMap[at + 7] = asuint(rowV.w);
    shadowMap[at + 8] = asuint(rowZ.x); shadowMap[at + 9] = asuint(rowZ.y);
    shadowMap[at + 10] = asuint(rowZ.z); shadowMap[at + 11] = asuint(rowZ.w);
    shadowMap[at + 12] = asuint(axisZ.x); shadowMap[at + 13] = asuint(axisZ.y);
    shadowMap[at + 14] = asuint(axisZ.z);
    shadowMap[at + 15] = asuint(texels);
    shadowMap[at + 16] = s.resolution;
    shadowMap[at + 17] = s.coefficients;
    shadowMap[at + 18] = shadowHeaderWords() + k * s.resolution * s.resolution * shadowStrideOf(s.coefficients);
    // 2 marks a dome's direction, which a reader picks by the way it looks.
    shadowMap[at + 19] = domeDirection ? 2u : 1u;
    shadowMap[at + 20] = asuint(bias);
}

/// One thread a gaussian: its optical depth into every texel its footprint
/// covers. Nothing is sorted and nothing is read back -- a sum is a sum.
[shader("compute")]
[numthreads(256, 1, 1)]
void shadowMapSplat(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= shadowParams.count) {
        return;
    }
    const ShadowMapFrame f = shadowFrameOf<ShadowBufferSource>(shadowParams.light);
    if (f.valid == 0) {
        return;
    }
    const float4 posOpacity = shadowPositions[i];
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shadowShape[row + 1]), highHalf(shadowShape[row + 1]),
                                    lowHalf(shadowShape[row + 2])));
    const float alpha = posOpacity.w;
    if (!(alpha > 1.0 / 255.0)) {
        return;
    }
    const float3 world = toWorld(posOpacity.xyz);
    const float u = dot(f.rowU.xyz, world) + f.rowU.w;
    const float v = dot(f.rowV.xyz, world) + f.rowV.w;
    const float z = clamp(dot(f.rowZ.xyz, world) + f.rowZ.w, 0.0, 1.0);

    // The 2D covariance in the map, which is the 3D one squashed along the
    // light's axis: the same R S S^T R^T the projection builds, carried by the
    // two map axes. Orthographic, so the Jacobian is those two rows and
    // nothing depends on depth.
    const float4 q = decodeQuaternion(shadowShape[row]);
    const float x = q.x, y = q.y, zq = q.z, w = q.w;
    const float3x3 r = float3x3(
        1.0 - 2.0 * (y * y + zq * zq), 2.0 * (x * y - w * zq), 2.0 * (x * zq + w * y),
        2.0 * (x * y + w * zq), 1.0 - 2.0 * (x * x + zq * zq), 2.0 * (y * zq - w * x),
        2.0 * (x * zq - w * y), 2.0 * (y * zq + w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);
    // Into world space: the object's linear part, rows as the params hold it.
    const ShadowMapParams s = shadowParams;
    const float3x3 objectLinear = float3x3(s.w00, s.w01, s.w02, s.w10, s.w11, s.w12, s.w20, s.w21, s.w22);
    const float3x3 mw = mul(objectLinear, m);
    const float3 ax = mul(f.rowU.xyz, mw);
    const float3 ay = mul(f.rowV.xyz, mw);
    float a = dot(ax, ax);
    float b = dot(ax, ay);
    float c = dot(ay, ay);
    // A third of a texel of dilation, as the raster dilates a pixel: a
    // gaussian smaller than a texel must still darken the one it sits in.
    a += 0.3;
    c += 0.3;
    const float det = a * c - b * b;
    if (!(det > 1.0e-12)) {
        return;
    }
    const float3 conic = float3(c / det, -b / det, a / det);
    const float cutoff = cutoffPower(alpha);
    const float2 extent = footprintExtent(conic, cutoff);
    const int loX = max(int(floor(u - extent.x)), 0);
    const int hiX = min(int(ceil(u + extent.x)), int(f.resolution) - 1);
    const int loY = max(int(floor(v - extent.y)), 0);
    const int hiY = min(int(ceil(v + extent.y)), int(f.resolution) - 1);
    if (loX > hiX || loY > hiY) {
        return;
    }
    // Fourier terms, where they were asked for: the same sum, weighed.
    const uint pairs = (f.coefficients - 1) / 2;
    for (int ty = loY; ty <= hiY; ++ty) {
        for (int tx = loX; tx <= hiX; ++tx) {
            const float dx = (float(tx) + 0.5) - u;
            const float dy = (float(ty) + 0.5) - v;
            const float power = 0.5 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            if (power < 0.0 || power > cutoff) {
                continue;
            }
            const float texelAlpha = min(alpha * exp(-power), 0.999);
            if (!(texelAlpha > 1.0 / 255.0)) {
                continue;
            }
            // Optical depth, which is what adds: transmittances multiply, and
            // their logarithms are what a sum may hold.
            const float sigma = -log(1.0 - texelAlpha) * shadowParams.density;
            const uint stride = shadowStrideOf(f.coefficients);
            const uint at = f.offset + (uint(ty) * f.resolution + uint(tx)) * stride;
            InterlockedAdd(shadowMap[at], uint(sigma * kShadowFixedScale));
            // And how near the light the nearest caster in this texel is: what
            // says where the lit side of a surface-like cloud is.
            InterlockedMin(shadowMap[at + f.coefficients], shadowOrderedKey(z));
            for (uint k = 1; k <= pairs; ++k) {
                const float angle = 2.0 * kShadowPi * float(k) * z;
                // Signed, added as two's complement: an unsigned add wraps
                // exactly the way a signed one does.
                InterlockedAdd(shadowMap[at + k * 2 - 1], uint(int(sigma * cos(angle) * kShadowFixedScale)));
                InterlockedAdd(shadowMap[at + k * 2], uint(int(sigma * sin(angle) * kShadowFixedScale)));
            }
        }
    }
}

/// One number a map amounts to: the mean transmittance over its texels, in
/// fixed point since there are no float atomics. What a test checks against a
/// cloud of known optical depth, and what ATHENEA_SHADOW_DEBUG prints.
public RWStructuredBuffer<uint> shadowTotal;

[shader("compute")]
[numthreads(256, 1, 1)]
void shadowMapMean(uint3 tid: SV_DispatchThreadID) {
    const uint texel = tid.x;
    const ShadowMapFrame f = shadowFrameOf<ShadowBufferSource>(shadowParams.light);
    if (f.valid == 0 || texel >= f.resolution * f.resolution) {
        return;
    }
    const float optical = float(shadowMap[f.offset + texel * shadowStrideOf(f.coefficients)]) / kShadowFixedScale;
    InterlockedAdd(shadowTotal[0], uint(exp(-optical) * 65536.0));
}

/// What the map answers at points somebody names: one thread a point, the
/// same read a receiver does. For tests, and for ATHENEA_SHADOW_DEBUG.
public StructuredBuffer<float4>  shadowProbePoints;
public RWStructuredBuffer<float> shadowProbeOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void shadowMapProbe(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= shadowParams.count) {
        return;
    }
    // Through the writable binding, since this runs in the same pass, and
    // through the same read a floor and a gaussian do.
    const uint light = shadowParams.light;
    shadowProbeOut[at] = shadowMapRead<ShadowBufferSource>(shadowFrameOf<ShadowBufferSource>(light), light,
                                                           shadowProbePoints[at].xyz);
}

// --- the map as a texture, for the kernels with no binding slot left ------
//
// A shading kernel is at Metal's limit of thirty-one buffers before this
// arrives, so the map reaches it as a texture instead: one array layer a
// coefficient a light, and the frames in layer zero's first rows. Nothing is
// sampled -- the reads are integer loads -- but a texture costs a texture
// slot, and of those there are plenty.

public RWTexture2DArray<float> shadowOut;
/// THE TRANSMITTANCE CHAIN: exp(-total) a layer a light, and its mips. A
/// floor far from the map covers several texels a pixel, and what it should
/// read is the mean of the transmittances under it -- not exp of the mean
/// optical depth, which is darker, and not one texel, which aliases. Level
/// zero is written with the rest of the texture; each level after it is the
/// mean of four of the one before (`shadowMapChain`).
public RWTexture2DArray<float> shadowChainOut;
public RWTexture2DArray<float> shadowChainSource;

/// Layer zero, sixteen words a row: light k's frame, word by word, as floats.
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowMapHeader(uint3 tid: SV_DispatchThreadID) {
    const uint word = tid.x;
    if (word >= kShadowFrameWords * kShadowSlots) {
        return;
    }
    // The numeric fields keep their value; the four counts are small enough
    // that a float holds them exactly.
    const uint raw = shadowMap[word];
    const float value = shadowHeaderIsReal(word) ? asfloat(raw) : float(raw);
    shadowOut[uint3(shadowHeaderTexel(word).xyz)] = value;
}

/// Every texel of every coefficient: fixed point out, float in.
[shader("compute")]
[numthreads(8, 8, 1)]
void shadowMapResolve(uint3 tid: SV_DispatchThreadID) {
    const uint resolution = shadowParams.resolution;
    if (tid.x >= resolution || tid.y >= resolution) {
        return;
    }
    const uint coefficients = shadowParams.coefficients;
    const uint slice = tid.z;
    if (slice >= shadowParams.slots * shadowStrideOf(coefficients)) {
        return;
    }
    const uint stride = shadowStrideOf(coefficients);
    const uint light = slice / stride;
    const uint which = slice % stride;
    const uint offset = shadowHeaderWords() + light * resolution * resolution * stride;
    const uint word = offset + (tid.y * resolution + tid.x) * stride + which;
    // The total is unsigned, every Fourier term is signed, and the last word
    // of a texel is a depth stored as an ordered key.
    const float value = which == coefficients ? shadowOrderedFloat(shadowMap[word])
                        : which == 0          ? float(shadowMap[word]) / kShadowFixedScale
                                              : float(int(shadowMap[word])) / kShadowFixedScale;
    shadowOut[uint3(tid.x, tid.y, shadowLayerOf(light, coefficients, which))] = value;
    if (which == 0) {
        shadowChainOut[uint3(tid.x, tid.y, light)] = exp(-max(value, 0.0));
    }
}

/// One level of the chain from the one before: the mean of the four texels
/// it covers, clamped where a level of odd size has no fourth. `level` is the
/// one written; the source is bound as its own view.
[shader("compute")]
[numthreads(8, 8, 1)]
void shadowMapChain(uint3 tid: SV_DispatchThreadID) {
    const uint level = shadowParams.level;
    const uint size = max(shadowParams.resolution >> level, 1u);
    const uint below = max(shadowParams.resolution >> (level - 1), 1u);
    if (tid.x >= size || tid.y >= size || tid.z >= shadowParams.slots) {
        return;
    }
    const uint x0 = min(tid.x * 2, below - 1);
    const uint y0 = min(tid.y * 2, below - 1);
    const uint x1 = min(tid.x * 2 + 1, below - 1);
    const uint y1 = min(tid.y * 2 + 1, below - 1);
    const float sum = shadowChainSource[uint3(x0, y0, tid.z)] + shadowChainSource[uint3(x1, y0, tid.z)] +
                      shadowChainSource[uint3(x0, y1, tid.z)] + shadowChainSource[uint3(x1, y1, tid.z)];
    shadowChainOut[uint3(tid.x, tid.y, tid.z)] = 0.25 * sum;
}

// --- the other receiver: the gaussians themselves ------------------------
//
// A mesh behind the cloud reads the total and is exactly right. A gaussian
// standing *inside* it is not: what reaches it is what the cloud stopped
// between the light and its own depth, which is what the Fourier terms
// reconstruct. So the same map, read at the gaussian's own z, is a cloud
// shadowing itself -- and one cloud shadowing another, since every caster of
// the frame added into the same texels.
//
// What it writes is what the relight already reads: `factors[(base + i) *
// lights + k]`, the same slot a baked visibility field or a shadow ray fills.

public RWStructuredBuffer<float> shadowFactors;

[shader("compute")]
[numthreads(256, 1, 1)]
void shadowMapFactors(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= shadowParams.count) {
        return;
    }
    const uint slot = (shadowParams.base + i) * shadowParams.lightCount;
    const float3 world = toWorld(shadowPositions[i].xyz);
    for (uint k = 0; k < shadowParams.lightCount && k < kShadowSlots; ++k) {
        // On the lit side of the first surface -- which is where most of a
        // surface-like cloud's gaussians are -- nothing is in the way, and
        // the reconstruction is not asked a question it cannot answer: the
        // read's own test, the same a floor's is.
        shadowFactors[slot + k] = shadowMapRead<ShadowBufferSource>(shadowFrameOf<ShadowBufferSource>(k), k, world);
    }
}

/// Every factor back to "nothing is in the way", before any of them is
/// written: a cloud with no field and no map must not read what the last
/// frame left there.
[shader("compute")]
[numthreads(256, 1, 1)]
void shadowFactorsClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < shadowParams.words) {
        shadowFactors[tid.x] = 1.0;
    }
}
