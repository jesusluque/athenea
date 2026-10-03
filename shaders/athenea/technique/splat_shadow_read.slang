// Copyright (c) 2026 jesus luque.
//
// READING A CLOUD'S SHADOW MAP: the pure half, with no parameter of its own.
//
// A shading kernel is at Metal's limit of thirty-one buffers before it hears
// about clouds at all, and importing a module brings that module's global
// parameters with it -- so everything the pass declares lives next door in
// splat_shadow_map.slang, and what a receiver needs is here: the layout both
// agree on, and the read itself, written once over whatever holds the map
// (`IShadowMapSource`). The pass's own buffer is one source (the gaussian
// receivers, the probe), the shading kernel's texture is the other; both read
// the header and the texels through the same function, so a probe answers
// what a floor reads.
//
// EVERY READ IS FILTERED BY HAND. Four integer loads and bilinear weights
// computed here, never a sampler: CUDA's hardware filter carries its weights
// in nine bits and Metal's in more, and a shadow that differs between the two
// machines by the filter is a shadow nobody can measure. And the filter is
// over transmittances, not over what the texels hold: each tap is tested
// against its own nearest caster and turned into exp(-tau) before the four
// are blended -- percentage-closer filtering, since depths and Fourier terms
// do not average into anything meaningful.
module splat_shadow_read;

public static const float kShadowFixedScale = 4096.0;
/// Words a light slot's frame takes in the header (`ShadowMapFrame`).
public static const uint kShadowFrameWords = 21;
public static const uint kShadowSlots = 8;
public static const uint kShadowBoxAt = kShadowFrameWords * kShadowSlots;
/// After the frames: the casters' posed box (three minima, three maxima) and
/// the widest of their rest spheres, each reduced with an atomic.
public static const uint kShadowBoxWords = 7;
/// How many header words a row of the texture's layer zero holds: sixteen,
/// which the smallest map there is (16 texels a side) has room for. The
/// header used to be one row, which a map narrower than its 160 words cut.
public static const uint kShadowHeaderRow = 16;

public uint shadowHeaderWords() { return kShadowBoxAt + kShadowBoxWords; }

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

/// The texture layer of light `light`'s value `which` (0 the total, 1.. the
/// Fourier terms, `coefficients` the nearest depth). Layer zero is the
/// header's.
public uint shadowLayerOf(uint light, uint coefficients, uint which) {
    return 1 + light * shadowStrideOf(coefficients) + which;
}

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
    public uint   offset;      ///< first word of this light's texels, in the buffer
    public uint   valid;       ///< 0: this light has no map, and nothing shadows
    /// How far behind the nearest caster a receiver still stands on it, in
    /// the slab's units: a length in the world (`ShadowMapJob::selfBias`)
    /// over a slab that does not change with the pose, so the same length
    /// every frame. It was two per cent of a slab that breathed with the
    /// wings.
    public float  bias;
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

/// Which header words are numbers rather than counts: what the texture's
/// header rows store as the float itself rather than as a count converted.
public bool shadowHeaderIsReal(uint word) {
    const uint at = word % kShadowFrameWords;
    return at < 16 || at == 20;
}

/// Header word `word`'s place in the texture: layer zero, sixteen a row.
public int4 shadowHeaderTexel(uint word) {
    return int4(int(word % kShadowHeaderRow), int(word / kShadowHeaderRow), 0, 0);
}

/// What holds a map: the pass's buffer, or the texture a shading kernel
/// reads. Static, since a source is a global of the kernel that imports this
/// module and not something a struct carries.
public interface IShadowMapSource {
    /// Header word `word`, as a number (the rows, the axis, the scale, the
    /// bias) and as a count (the resolution, the terms, the offset, valid).
    static float headerReal(uint word);
    static uint headerCount(uint word);
    /// Level zero of light `light`'s texel (x, y), decoded: `which` 0 is the
    /// total optical depth, 1.. the Fourier terms, `f.coefficients` the depth
    /// of the nearest caster. In range: the caller has checked.
    static float texel(ShadowMapFrame f, uint light, int x, int y, uint which);
};

/// The header, read in one place for every source.
public ShadowMapFrame shadowFrameOf<S : IShadowMapSource>(uint light) {
    const uint at = light * kShadowFrameWords;
    ShadowMapFrame f;
    f.rowU = float4(S.headerReal(at + 0), S.headerReal(at + 1), S.headerReal(at + 2), S.headerReal(at + 3));
    f.rowV = float4(S.headerReal(at + 4), S.headerReal(at + 5), S.headerReal(at + 6), S.headerReal(at + 7));
    f.rowZ = float4(S.headerReal(at + 8), S.headerReal(at + 9), S.headerReal(at + 10), S.headerReal(at + 11));
    f.axisZ = float3(S.headerReal(at + 12), S.headerReal(at + 13), S.headerReal(at + 14));
    f.texelsPerUnit = S.headerReal(at + 15);
    f.resolution = S.headerCount(at + 16);
    f.coefficients = S.headerCount(at + 17);
    f.offset = S.headerCount(at + 18);
    f.valid = S.headerCount(at + 19);
    f.bias = S.headerReal(at + 20);
    return f;
}

/// One tap: the transmittance a receiver at depth `z` reads through texel
/// (x, y). Outside the map nothing of the cloud is in the way.
public float shadowTap<S : IShadowMapSource>(ShadowMapFrame f, uint light, int x, int y, float z) {
    if (x < 0 || y < 0 || x >= int(f.resolution) || y >= int(f.resolution)) {
        return 1.0;
    }
    // Nearer than the first thing that cast into this texel: lit. An
    // untouched texel's nearest is 1e30, so it is lit too.
    if (z <= S.texel(f, light, x, y, f.coefficients) + f.bias) {
        return 1.0;
    }
    const float total = S.texel(f, light, x, y, 0);
    float optical = total;
    if (f.coefficients > 1 && z < 1.0) {
        float sum = total * max(z, 0.0);
        const uint pairs = (f.coefficients - 1) / 2;
        for (uint k = 1; k <= pairs; ++k) {
            sum += shadowOpticalOf(total, S.texel(f, light, x, y, k * 2 - 1), S.texel(f, light, x, y, k * 2), k,
                                   max(z, 0.0));
        }
        optical = clamp(sum, 0.0, max(total, 0.0));
    }
    return exp(-max(optical, 0.0));
}

/// Bilinear weights over the four texels about (u, v), whose centres are at
/// half-integers.
public void shadowBilinear(float u, float v, out int2 origin, out float2 weight) {
    const float x = u - 0.5;
    const float y = v - 0.5;
    const float x0 = floor(x);
    const float y0 = floor(y);
    origin = int2(int(x0), int(y0));
    weight = float2(x - x0, y - y0);
}

/// Level zero, percentage-closer: four taps, each tested and turned into a
/// transmittance, blended bilinearly.
public float shadowPcf<S : IShadowMapSource>(ShadowMapFrame f, uint light, float u, float v, float z) {
    int2 o;
    float2 w;
    shadowBilinear(u, v, o, w);
    const float t00 = shadowTap<S>(f, light, o.x, o.y, z);
    const float t10 = shadowTap<S>(f, light, o.x + 1, o.y, z);
    const float t01 = shadowTap<S>(f, light, o.x, o.y + 1, z);
    const float t11 = shadowTap<S>(f, light, o.x + 1, o.y + 1, z);
    return lerp(lerp(t00, t10, w.x), lerp(t01, t11, w.x), w.y);
}

/// THE READ, for every receiver: a floor's, a gaussian's, a probe's.
public float shadowMapRead<S : IShadowMapSource>(ShadowMapFrame f, uint light, float3 p) {
    if (f.valid == 0 || f.resolution == 0) {
        return 1.0;
    }
    const float u = dot(f.rowU.xyz, p) + f.rowU.w;
    const float v = dot(f.rowV.xyz, p) + f.rowV.w;
    const float r = float(f.resolution);
    // Within a texel of the border a tap or two still lands inside.
    if (!(u > -1.0 && v > -1.0 && u < r + 1.0 && v < r + 1.0)) {
        return 1.0;   // beside the cloud: nothing of it is in the way
    }
    const float z = dot(f.rowZ.xyz, p) + f.rowZ.w;
    if (z <= 0.0) {
        return 1.0;   // nearer the light than anything that casts
    }
    return shadowPcf<S>(f, light, u, v, z);
}
