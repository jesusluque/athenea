// Copyright (c) 2026 jesus luque.
//
// READING A CLOUD'S SHADOW MAP: the pure half, with no parameter of its own.
//
// A shading kernel is at Metal's limit of thirty-one buffers before it hears
// about clouds at all, and importing a module brings that module's global
// parameters with it -- so everything the pass declares lives next door in
// splat_shadow_map.slang, and what a receiver needs is here: functions that
// take the map they are to read as an argument, and the layout both agree on.
module splat_shadow_read;

public static const float kShadowFixedScale = 4096.0;
public static const uint kShadowFrameWords = 20;
public static const uint kShadowSlots = 8;
public static const uint kShadowBoxAt = kShadowFrameWords * kShadowSlots;

public uint shadowHeaderWords() { return kShadowBoxAt + 6; }

/// A texel keeps its coefficients and then one word more: the depth of the
/// nearest thing that cast into it, as an ordered key.
///
/// WHY A DEPTH AS WELL AS A TRANSMITTANCE. A Fourier series truncated to a
/// few terms cannot hold a step, and a splat surface is a step: reconstructed
/// at its own depth a gaussian reads about half of its own optical depth and
/// the whole cloud turns black (measured -- the sparrow went to soot). The
/// depth of the first caster is what says "this is the lit side": nearer than
/// it, by a bias, nothing is in the way, and the reconstruction is left to do
/// what it is good at, which is everything behind that.
public uint shadowStrideOf(uint coefficients) { return coefficients + 1; }

/// How much of the slab a receiver may be inside the first surface and still
/// count as standing on it.
public static const float kShadowSelfBias = 0.02;

public static const float kShadowPi = 3.14159265358979;

/// A light's map: how a world point becomes a texel and a depth, and where
/// its texels are. Written by `shadowMapFrame`, read by everything after.
public struct ShadowMapFrame {
    public float4 rowU;        ///< u = dot(rowU.xyz, p) + rowU.w, in texels
    public float4 rowV;
    public float4 rowZ;        ///< depth across the caster's slab, 0 at the light's side, 1 beyond
    public float3 axisZ;       ///< the direction light travels, world space
    public float  texelsPerUnit;   ///< how many texels a world unit is
    public uint   resolution;
    public uint   coefficients;
    public uint   offset;      ///< first word of this light's texels
    public uint   valid;       ///< 0: this light has no map, and nothing shadows
};

/// A float as a uint that sorts the way the float does: the sign bit flipped
/// for a positive one, every bit for a negative. What lets an atomic min over
/// uints be a min over floats, which is the only kind a device is sure to
/// have.
public uint shadowOrderedKey(float f) {
    const uint bits = asuint(f);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

public float shadowOrderedFloat(uint key) {
    return asfloat((key & 0x80000000u) != 0u ? (key & 0x7FFFFFFFu) : ~key);
}

/// FOURIER OPACITY MAPPING, the depth-resolved half of the same map.
///
/// The density along the slab is written as a Fourier series, so its integral
/// from the light to depth z is closed form. With
///
///     a_k = sum_i sigma_i cos(2 pi k z_i),  b_k = sum_i sigma_i sin(2 pi k z_i)
///
/// the optical depth to z is
///
///     a_0 z + sum_k (1 / (pi k)) [ a_k sin(2 pi k z) + b_k (1 - cos(2 pi k z)) ]
///
/// which at z = 1 is a_0, the total: the cheap variant really is this one's
/// first term. Ringing where the cloud is opaque is the known artefact, and
/// why the reconstruction is clamped to the total it must not exceed.
public float shadowOpticalOf(float total, float a, float b, uint k, float z) {
    const float angle = 2.0 * kShadowPi * float(k) * z;
    return (a * sin(angle) + b * (1.0 - cos(angle))) / (kShadowPi * float(k));
}

/// THE READ a shading kernel does, from the texture the pass resolved into:
/// one array layer a coefficient a light, the frames in layer zero's first
/// row. Nothing is sampled -- these are integer loads.
public float shadowMapTransmittanceTex(Texture2DArray<float> tex, uint light, float3 p) {
    if (light >= kShadowSlots) {
        return 1.0;
    }
    const uint at = light * kShadowFrameWords;
    const float4 rowU = float4(tex[uint3(at + 0, 0, 0)], tex[uint3(at + 1, 0, 0)],
                               tex[uint3(at + 2, 0, 0)], tex[uint3(at + 3, 0, 0)]);
    const float4 rowV = float4(tex[uint3(at + 4, 0, 0)], tex[uint3(at + 5, 0, 0)],
                               tex[uint3(at + 6, 0, 0)], tex[uint3(at + 7, 0, 0)]);
    const float4 rowZ = float4(tex[uint3(at + 8, 0, 0)], tex[uint3(at + 9, 0, 0)],
                               tex[uint3(at + 10, 0, 0)], tex[uint3(at + 11, 0, 0)]);
    const uint resolution = uint(tex[uint3(at + 16, 0, 0)]);
    const uint coefficients = uint(tex[uint3(at + 17, 0, 0)]);
    const uint valid = uint(tex[uint3(at + 19, 0, 0)]);
    if (valid == 0 || resolution == 0) {
        return 1.0;
    }
    const float u = dot(rowU.xyz, p) + rowU.w;
    const float v = dot(rowV.xyz, p) + rowV.w;
    if (u < 0.0 || v < 0.0 || u >= float(resolution) || v >= float(resolution)) {
        return 1.0;   // beside the cloud: nothing of it is in the way
    }
    const float z = dot(rowZ.xyz, p) + rowZ.w;
    if (z <= 0.0) {
        return 1.0;   // nearer the light than anything that casts
    }
    const uint stride = shadowStrideOf(coefficients);
    const uint3 texel = uint3(uint(u), uint(v), light * stride + 1);
    // Nearer than the first thing that cast into this texel: lit.
    if (z <= tex[uint3(texel.xy, light * stride + coefficients + 1)] + kShadowSelfBias) {
        return 1.0;
    }
    const float total = tex[texel];
    float optical = total;
    if (coefficients > 1 && z < 1.0) {
        float sum = total * z;
        const uint pairs = (coefficients - 1) / 2;
        for (uint k = 1; k <= pairs; ++k) {
            sum += shadowOpticalOf(total, tex[uint3(texel.xy, texel.z + k * 2 - 1)],
                                   tex[uint3(texel.xy, texel.z + k * 2)], k, z);
        }
        optical = clamp(sum, 0.0, max(total, 0.0));
    }
    return exp(-max(optical, 0.0));
}
