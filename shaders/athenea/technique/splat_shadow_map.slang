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

/// The frame as the pass itself reads it, through its writable binding.
ShadowMapFrame shadowFrameOfRW(RWStructuredBuffer<uint> map, uint light) {
    const uint at = light * kShadowFrameWords;
    ShadowMapFrame f;
    f.rowU = float4(asfloat(map[at + 0]), asfloat(map[at + 1]), asfloat(map[at + 2]), asfloat(map[at + 3]));
    f.rowV = float4(asfloat(map[at + 4]), asfloat(map[at + 5]), asfloat(map[at + 6]), asfloat(map[at + 7]));
    f.rowZ = float4(asfloat(map[at + 8]), asfloat(map[at + 9]), asfloat(map[at + 10]), asfloat(map[at + 11]));
    f.axisZ = float3(asfloat(map[at + 12]), asfloat(map[at + 13]), asfloat(map[at + 14]));
    f.texelsPerUnit = asfloat(map[at + 15]);
    f.resolution = map[at + 16];
    f.coefficients = map[at + 17];
    f.offset = map[at + 18];
    f.valid = map[at + 19];
    return f;
}

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
    public float pad3;
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
};

public ConstantBuffer<ShadowMapParams> shadowParams;
public RWStructuredBuffer<uint>        shadowMap;
public StructuredBuffer<LightRecord>   shadowLights;
public StructuredBuffer<float4>        shadowPositions;   ///< posed, the cloud's own space
public StructuredBuffer<uint>          shadowShape;       ///< 4 words a gaussian, as GpuSplats packs them

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
    // thing that cast into it. Everything else starts at nothing, which is
    // what a sum wants.
    const uint header = shadowHeaderWords();
    bool smallest = tid.x >= kShadowBoxAt && tid.x < kShadowBoxAt + 3;
    const bool largest = tid.x >= kShadowBoxAt + 3 && tid.x < kShadowBoxAt + 6;
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
/// into the six words every frame is then built from. The box is a bound the
/// host already holds; where its corners land is not something the host works
/// out.
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
}

/// One thread a light: the map's frame, from the light's own axes and the box
/// the caster occupies. A distant light gets an orthographic frame -- it is
/// the only kind whose rays are parallel -- and a light that stands somewhere
/// gets one about the direction from it to the box, which is exact for a
/// light far enough away and is what a spot is.
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowMapFrame(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= kShadowSlots) {
        return;
    }
    const uint at = k * kShadowFrameWords;
    if (k >= shadowParams.lightCount) {
        shadowMap[at + 19] = 0;
        return;
    }
    const LightRecord light = shadowLights[k];
    const ShadowMapParams s = shadowParams;
    const float3 boxMin = float3(shadowOrderedFloat(shadowMap[kShadowBoxAt + 0]), shadowOrderedFloat(shadowMap[kShadowBoxAt + 1]),
                                 shadowOrderedFloat(shadowMap[kShadowBoxAt + 2]));
    const float3 boxMax = float3(shadowOrderedFloat(shadowMap[kShadowBoxAt + 3]), shadowOrderedFloat(shadowMap[kShadowBoxAt + 4]),
                                 shadowOrderedFloat(shadowMap[kShadowBoxAt + 5]));
    if (!all(boxMax >= boxMin)) {
        shadowMap[at + 19] = 0;   // nothing cast anything
        return;
    }
    const float3 centre = 0.5 * (boxMin + boxMax);
    // A skinned cloud is posed away from the box it was bound in, so the map
    // covers more than the bind pose: a wing that leaves the map stops
    // casting, and that is the one artefact worth paying a margin to avoid.
    const float3 half3 = max(0.5 * (boxMax - boxMin) * (1.0 + s.margin), float3(1.0e-6));
    // Which way the light travels. UsdLux shines down its own -Z.
    float3 axisZ = -lightAxisZ(light);
    if (light.kind != kLightDistant) {
        const float3 origin = lightOrigin(light);
        const float3 toBox = centre - origin;
        const float reach = length(toBox);
        if (!(reach > 1.0e-6)) {
            shadowMap[at + 19] = 0;
            return;
        }
        axisZ = toBox / reach;
    }
    if (light.kind == kLightDome || (light.flags & kLightShadow) == 0) {
        // A dome has no direction to build a map about, and a light that casts
        // no shadow casts none here either.
        shadowMap[at + 19] = 0;
        return;
    }
    // An orthonormal frame about it.
    const float3 up = abs(axisZ.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    const float3 axisX = normalize(cross(up, axisZ));
    const float3 axisY = cross(axisZ, axisX);
    // The box's extent along each axis: its half diagonal projected, which is
    // the support function of a box and so exact for any rotation.
    const float extentX = abs(half3.x * axisX.x) + abs(half3.y * axisX.y) + abs(half3.z * axisX.z);
    const float extentY = abs(half3.x * axisY.x) + abs(half3.y * axisY.y) + abs(half3.z * axisY.z);
    const float extentZ = abs(half3.x * axisZ.x) + abs(half3.y * axisZ.y) + abs(half3.z * axisZ.z);
    const float widest = max(max(extentX, extentY), 1.0e-6);
    const float texels = float(s.resolution) / (2.0 * widest);
    const float centreU = dot(axisX, centre);
    const float centreV = dot(axisY, centre);
    const float centreZ = dot(axisZ, centre);
    const float slab = max(2.0 * extentZ, 1.0e-6);
    const float4 rowU = float4(axisX * texels, 0.5 * float(s.resolution) - centreU * texels);
    const float4 rowV = float4(axisY * texels, 0.5 * float(s.resolution) - centreV * texels);
    const float4 rowZ = float4(axisZ / slab, 0.5 - centreZ / slab);
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
    shadowMap[at + 19] = 1;
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
    const ShadowMapFrame f = shadowFrameOfRW(shadowMap, shadowParams.light);
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
    const ShadowMapFrame f = shadowFrameOfRW(shadowMap, shadowParams.light);
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
    const float3 p = shadowProbePoints[at].xyz;
    // Through the writable binding, since this runs in the same pass.
    float answer = 1.0;
    const ShadowMapFrame f = shadowFrameOfRW(shadowMap, shadowParams.light);
    if (f.valid != 0) {
        const float u = dot(f.rowU.xyz, p) + f.rowU.w;
        const float v = dot(f.rowV.xyz, p) + f.rowV.w;
        const float z = dot(f.rowZ.xyz, p) + f.rowZ.w;
        if (u >= 0.0 && v >= 0.0 && u < float(f.resolution) && v < float(f.resolution) && z > 0.0) {
            const uint texel = uint(v) * f.resolution + uint(u);
            const uint word = f.offset + texel * shadowStrideOf(f.coefficients);
            if (z <= shadowOrderedFloat(shadowMap[word + f.coefficients]) + kShadowSelfBias) {
                shadowProbeOut[at] = 1.0;
                return;
            }
            const float total = float(shadowMap[word]) / kShadowFixedScale;
            float optical = total;
            if (f.coefficients > 1 && z < 1.0) {
                float sum = total * z;
                const uint pairs = (f.coefficients - 1) / 2;
                for (uint k = 1; k <= pairs; ++k) {
                    const float a = float(int(shadowMap[word + k * 2 - 1])) / kShadowFixedScale;
                    const float b = float(int(shadowMap[word + k * 2])) / kShadowFixedScale;
                    const float angle = 2.0 * kShadowPi * float(k) * z;
                    sum += (a * sin(angle) + b * (1.0 - cos(angle))) / (kShadowPi * float(k));
                }
                optical = clamp(sum, 0.0, max(total, 0.0));
            }
            answer = exp(-max(optical, 0.0));
        }
    }
    shadowProbeOut[at] = answer;
}

// --- the map as a texture, for the kernels with no binding slot left ------
//
// A shading kernel is at Metal's limit of thirty-one buffers before this
// arrives, so the map reaches it as a texture instead: one array layer a
// coefficient a light, and the frames in layer zero's first row. Nothing is
// sampled -- the reads are integer loads -- but a texture costs a texture
// slot, and of those there are plenty.

public RWTexture2DArray<float> shadowOut;

/// Layer zero, first row: light k's frame, word by word, as floats.
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowMapHeader(uint3 tid: SV_DispatchThreadID) {
    const uint word = tid.x;
    if (word >= kShadowFrameWords * kShadowSlots) {
        return;
    }
    const uint light = word / kShadowFrameWords;
    const uint at = word % kShadowFrameWords;
    // The numeric fields keep their value; the four counts are small enough
    // that a float holds them exactly.
    const uint raw = shadowMap[word];
    const float value = at < 16 ? asfloat(raw) : float(raw);
    shadowOut[uint3(word, 0, 0)] = value;
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
    if (slice >= shadowParams.lightCount * shadowStrideOf(coefficients)) {
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
    shadowOut[uint3(tid.x, tid.y, slice + 1)] = value;
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
        const ShadowMapFrame f = shadowFrameOfRW(shadowMap, k);
        float answer = 1.0;
        if (f.valid != 0) {
            const float u = dot(f.rowU.xyz, world) + f.rowU.w;
            const float v = dot(f.rowV.xyz, world) + f.rowV.w;
            const float z = clamp(dot(f.rowZ.xyz, world) + f.rowZ.w, 0.0, 1.0);
            if (u >= 0.0 && v >= 0.0 && u < float(f.resolution) && v < float(f.resolution)) {
                const uint word = f.offset + (uint(v) * f.resolution + uint(u)) * shadowStrideOf(f.coefficients);
                const float nearest = shadowOrderedFloat(shadowMap[word + f.coefficients]);
                const float total = float(shadowMap[word]) / kShadowFixedScale;
                float optical = total;
                if (f.coefficients > 1) {
                    float sum = total * z;
                    const uint pairs = (f.coefficients - 1) / 2;
                    for (uint c = 1; c <= pairs; ++c) {
                        const float a = float(int(shadowMap[word + c * 2 - 1])) / kShadowFixedScale;
                        const float b = float(int(shadowMap[word + c * 2])) / kShadowFixedScale;
                        sum += shadowOpticalOf(total, a, b, c, z);
                    }
                    optical = clamp(sum, 0.0, max(total, 0.0));
                }
                // On the lit side of the first surface -- which is where most
                // of a surface-like cloud's gaussians are -- nothing is in
                // the way, and the reconstruction is not asked a question it
                // cannot answer.
                answer = z <= nearest + kShadowSelfBias ? 1.0 : exp(-max(optical, 0.0));
            }
        }
        shadowFactors[slot + k] = answer;
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
