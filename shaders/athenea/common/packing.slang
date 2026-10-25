// Copyright (c) 2026 jesus luque.
//
// How a Gaussian is stored on the GPU, in one place.
//
//   positions  float4   xyz, opacity (linear, 0..1)            16 bytes
//   shape      uint * 4                                         16 bytes
//     [0] rotation, smallest-three: 10+10+10 bits and the dropped index
//     [1] f16 log(scale.x) | f16 log(scale.y) << 16
//     [2] f16 log(scale.z) | f16 base colour r << 16
//     [3] f16 base colour g | f16 base colour b << 16
//   sh         uint * words   f16 rest coefficients, rgb per basis
//   normals    uint           optional: the shading normal, octahedral
//                             2 x 16 bits (`packNormal`), apart from the frame
//   emission   uint           optional: the radiance the surface gives off by
//                             itself, linear RGB9E5 (`packRgb9e5`)
//   lobes      uint * 3       optional: the layers over the base a material
//                             names -- specular weight and colour, coat,
//                             sheen -- a byte a value (`packLobes`)
//
// 32 bytes before harmonics, against openFXplayer's 64 float32. Why each
// field is what it is:
//   - position stays float32: fp16 would step two pixels at 4K close up
//     (openFXplayer measured it), and it is what a camera flies into.
//   - scale is stored as a logarithm in fp16: 0.1% relative precision at every
//     size, where fp16 of the scale itself loses thin splats below 1e-4.
//   - rotation in 10-bit smallest-three is within a tenth of a degree.
//   - colours are fp16: base colour can leave 0..1, and harmonics are signed.
module packing;

public static const float kSqrtHalf = 0.70710678;

/// Smallest-three quaternion (x, y, z, w) -> uint.
public uint encodeQuaternion(float4 q) {
    q = normalize(q);
    uint largest = 0;
    float best = abs(q.x);
    for (uint k = 1; k < 4; ++k) {
        if (abs(q[k]) > best) {
            best = abs(q[k]);
            largest = k;
        }
    }
    if (q[largest] < 0.0) {
        q = -q;
    }
    uint word = largest;
    uint shift = 22;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            continue;
        }
        const float unit = clamp(q[k] / kSqrtHalf * 0.5 + 0.5, 0.0, 1.0);
        word |= uint(round(unit * 1023.0)) << shift;
        shift -= 10;
    }
    return word;
}

public float4 decodeQuaternion(uint word) {
    const uint largest = word & 3u;
    float3 small = float3(float((word >> 22) & 1023u), float((word >> 12) & 1023u),
                          float((word >> 2) & 1023u));
    small = (small / 1023.0 * 2.0 - 1.0) * kSqrtHalf;
    const float rest = sqrt(max(0.0, 1.0 - dot(small, small)));
    float4 q = float4(0.0);
    uint next = 0;
    for (uint k = 0; k < 4; ++k) {
        if (k == largest) {
            q[k] = rest;
        } else {
            q[k] = small[next];
            next += 1;
        }
    }
    return q;
}

/// A rotation matrix whose columns are the axes, as (x, y, z, w).
///
/// It lives here beside the quaternion's own packing because two things now
/// need it: the LOD merge, which takes the eigenframe of a merged covariance,
/// and the splat skinner, which takes the frame a joint left a gaussian in.
public float4 quaternionOfAxes(float3x3 m) {
    const float trace = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (trace > 0.0) {
        const float s = 0.5 / sqrt(trace + 1.0);
        q = float4((m[2][1] - m[1][2]) * s, (m[0][2] - m[2][0]) * s, (m[1][0] - m[0][1]) * s, 0.25 / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]);
        q = float4(0.25 * s, (m[1][0] + m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s);
    } else if (m[1][1] > m[2][2]) {
        const float s = 2.0 * sqrt(1.0 - m[0][0] + m[1][1] - m[2][2]);
        q = float4((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[0][2] - m[2][0]) / s);
    } else {
        const float s = 2.0 * sqrt(1.0 - m[0][0] - m[1][1] + m[2][2]);
        q = float4((m[0][2] + m[2][0]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s);
    }
    return normalize(q);
}

/// The other way: the three axes a quaternion turns the basis into, as the
/// columns of a matrix.
public float3x3 axesOfQuaternion(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

/// A SHADING NORMAL IN ONE WORD (`GpuSplats::normals`): the octahedral map
/// (common/octahedral.slang's, written out here so packing imports nothing),
/// x in the low sixteen bits and y in the high, each a unorm. A step is
/// 2^-15 of the square, a few thousandths of a degree -- finer than the
/// 10-bit rotation the frame keeps. A zero vector packs as +z.
public uint packNormal(float3 n) {
    const float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    n = l1 > 1.0e-20 ? n / l1 : float3(0.0, 0.0, 1.0);
    float2 p = n.xy;
    if (n.z < 0.0) {
        p = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    const uint2 q = uint2(round(saturate(p * 0.5 + 0.5) * 65535.0));
    return q.x | (q.y << 16);
}

/// The other way: a unit vector.
public float3 unpackNormal(uint word) {
    const float2 f = float2(float(word & 0xFFFFu), float(word >> 16)) / 65535.0 * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

/// LIGHT IN ONE WORD (`GpuSplats::emission`): RGB9E5, three 9-bit mantissas
/// under one 5-bit exponent, as the shared-exponent texture format has it.
/// Unsigned and HDR -- up to 65408, down to 2^-24 -- with every channel
/// stepped by 1/512 of the brightest, which is finer than f16 on the channel
/// that is seen. Negative and not-a-number are nothing; past the top is the
/// top.
public uint packRgb9e5(float3 rgb) {
    const float kTop = 65408.0;   // (511 / 512) * 2^16
    const float3 c = float3(rgb.x > 0.0 ? min(rgb.x, kTop) : 0.0, rgb.y > 0.0 ? min(rgb.y, kTop) : 0.0,
                            rgb.z > 0.0 ? min(rgb.z, kTop) : 0.0);
    const float largest = max(c.x, max(c.y, c.z));
    if (!(largest >= 5.9604645e-8)) {   // half the smallest step there is
        return 0u;
    }
    // The exponent that puts the largest channel's mantissa below 512, and
    // one more where rounding takes it to 512.
    int e = max(-16, int(floor(log2(largest)))) + 16;
    float step = exp2(float(e - 24));
    if (uint(floor(largest / step + 0.5)) >= 512u) {
        e += 1;
        step *= 2.0;
    }
    const uint3 m = min(uint3(floor(c / step + 0.5)), uint3(511u));
    return m.x | (m.y << 9) | (m.z << 18) | (uint(e) << 27);
}

/// The other way: linear RGB.
public float3 unpackRgb9e5(uint word) {
    const float step = exp2(float(int(word >> 27) - 24));
    return float3(float(word & 511u), float((word >> 9) & 511u), float((word >> 18) & 511u)) * step;
}

/// A SHAPE OPERATOR IN ONE WORD (a `.athc`'s curvature, its flag bit 6; the
/// same word in `.athz`'s geometry layer): the three components (uu, uv, vv,
/// per unit of length) as `geometry_quantize` keeps them -- a sign and 7 bits
/// of log2 of the size -- each over the 16 octaves below the largest's log2,
/// which the fourth byte holds, rounded up, as a signed byte:
///   bits 0-7, 8-15, 16-23   uu, uv, vv: bit 7 the sign, bits 0-6 q, the size
///                           2^(top - 16 + q 16/127); q = 0 is flat
///   bits 24-31              top, two's complement, -64..63
/// A component under 16 octaves below the largest, or under 1e-6, is flat.
public static const float kCurvatureOctaves = 16.0;

public uint packCurvature(float3 k) {
    const float largest = max(abs(k.x), max(abs(k.y), abs(k.z)));
    if (!(largest > 1.0e-6)) {
        return 0u;
    }
    const int top = clamp(int(ceil(log2(largest))), -64, 63);
    const float floorLog = float(top) - kCurvatureOctaves;
    uint word = (uint(top) & 0xFFu) << 24;
    [unroll] for (uint c = 0; c < 3; ++c) {
        const float a = abs(k[c]);
        if (a > 1.0e-6 && log2(a) > floorLog) {
            const uint q = uint(clamp(round((log2(a) - floorLog) * (127.0 / kCurvatureOctaves)), 1.0, 127.0));
            word |= (q | (k[c] < 0.0 ? 0x80u : 0u)) << (8u * c);
        }
    }
    return word;
}

public float3 unpackCurvature(uint word) {
    const int top = int(word << 0u) >> 24;
    const float floorLog = float(top) - kCurvatureOctaves;
    float3 k = float3(0.0);
    [unroll] for (uint c = 0; c < 3; ++c) {
        const uint b = (word >> (8u * c)) & 0xFFu;
        const uint q = b & 127u;
        if (q != 0u) {
            const float size = exp2(floorLog + float(q) * (kCurvatureOctaves / 127.0));
            k[c] = (b & 0x80u) != 0u ? -size : size;
        }
    }
    return k;
}

/// WHAT A MATERIAL LAYERS OVER ITS BASE, PER GAUSSIAN (`GpuSplats::lobes`).
///
/// The base -- colour, metallic, roughness, transmission -- is `pbr`'s word.
/// What OpenPBR, standard_surface and UsdPreviewSurface put over it is here,
/// in the units their own inputs take, for the lobes `splat_relight` already
/// evaluates: no lobe is new, these are the weights and parameters of the
/// dielectric, conductor and sheen it has.
///
///   specularWeight  0..1   scales the dielectric reflection
///   specularColour  0..1   tints the dielectric reflection; a metal's edge
///   specularIor     1..3   the dielectric's index (0.04 head on at 1.5)
///   coatWeight      0..1   a clear dielectric layer over all of it
///   coatRoughness   0..1
///   coatIor         1..3
///   coatDarkening   0..1   OpenPBR's `coat_darkening`: the base under the
///                          coat darkened by the light the coat's inside
///                          sends back to it (on at 0.5 and above)
///   sheenColour     0..1   the sheen's colour times its weight
///   sheenRoughness  0..1
public struct SplatLobes {
    public float  specularWeight;
    public float3 specularColour;
    public float  specularIor;
    public float  coatWeight;
    public float  coatRoughness;
    public float  coatIor;
    public float  coatDarkening;
    public float3 sheenColour;
    public float  sheenRoughness;
};

/// What a gaussian with no lobes reflects with: the specular of every
/// cloud before these existed -- weight one, white, an index of 1.5 -- and
/// no coat and no sheen. Packed, it unpacks to exactly this (1.5 is a whole
/// step of the index's byte), so a cloud that carries it and one that does
/// not are the same picture.
public SplatLobes plainLobes() {
    SplatLobes l;
    l.specularWeight = 1.0;
    l.specularColour = float3(1.0);
    l.specularIor = 1.5;
    l.coatWeight = 0.0;
    l.coatRoughness = 0.0;
    l.coatIor = 1.5;
    l.coatDarkening = 0.0;
    l.sheenColour = float3(0.0);
    l.sheenRoughness = 0.0;
    return l;
}

uint lobeByte(float v) {
    return uint(saturate(v) * 255.0 + 0.5);
}

/// An index of refraction in a byte: `1 + byte / 128`, 1 to 2.99 in steps of
/// 0.008 -- 1.5 and 1.25 exactly, every glass and lacquer there is.
uint iorByte(float ior) {
    return uint(clamp((ior - 1.0) * 128.0 + 0.5, 0.0, 255.0));
}

float byteIor(uint b) {
    return 1.0 + float(b & 0xFFu) / 128.0;
}

/// The coat's index in seven bits, `1 + bits / 64` (1 to 2.98, 1.5 and 1.25
/// exactly), and whether the coat darkens what is under it in the eighth.
uint coatByte(float ior, float darkening) {
    return uint(clamp((ior - 1.0) * 64.0 + 0.5, 0.0, 127.0)) | (darkening >= 0.5 ? 128u : 0u);
}

/// Three words: (specular colour, weight), (coat weight, roughness, index,
/// the specular's index), (sheen colour, roughness).
public uint3 packLobes(SplatLobes l) {
    const uint w0 = lobeByte(l.specularColour.x) | (lobeByte(l.specularColour.y) << 8) |
                    (lobeByte(l.specularColour.z) << 16) | (lobeByte(l.specularWeight) << 24);
    const uint w1 = lobeByte(l.coatWeight) | (lobeByte(l.coatRoughness) << 8) |
                    (coatByte(l.coatIor, l.coatDarkening) << 16) |
                    (iorByte(l.specularIor) << 24);
    const uint w2 = lobeByte(l.sheenColour.x) | (lobeByte(l.sheenColour.y) << 8) |
                    (lobeByte(l.sheenColour.z) << 16) | (lobeByte(l.sheenRoughness) << 24);
    return uint3(w0, w1, w2);
}

float byteOf(uint word, uint at) {
    return float((word >> (at * 8)) & 0xFFu) / 255.0;
}

public SplatLobes unpackLobes(uint3 w) {
    SplatLobes l;
    l.specularColour = float3(byteOf(w.x, 0), byteOf(w.x, 1), byteOf(w.x, 2));
    l.specularWeight = byteOf(w.x, 3);
    l.coatWeight = byteOf(w.y, 0);
    l.coatRoughness = byteOf(w.y, 1);
    l.coatIor = 1.0 + float((w.y >> 16) & 0x7Fu) / 64.0;
    l.coatDarkening = ((w.y >> 23) & 1u) != 0u ? 1.0 : 0.0;
    l.specularIor = byteIor(w.y >> 24);
    l.sheenColour = float3(byteOf(w.z, 0), byteOf(w.z, 1), byteOf(w.z, 2));
    l.sheenRoughness = byteOf(w.z, 3);
    return l;
}

/// The thirteen floats a record keeps them in (`io::SplatEncoding::lobes`),
/// in this order: specular weight, colour (3), index; coat weight, roughness,
/// index; sheen colour (3), roughness; coat darkening.
public static const uint kLobeFields = 13;

public uint packHalves(float a, float b) {
    return (f32tof16(a) & 0xFFFFu) | ((f32tof16(b) & 0xFFFFu) << 16);
}

public float lowHalf(uint word) { return f16tof32(word & 0xFFFFu); }
public float highHalf(uint word) { return f16tof32(word >> 16); }

/// f16 can hold +-65504; anything past that becomes infinity, which a
/// covariance turns into NaN. Clamped, not trusted.
public float halfSafe(float v) { return clamp(v, -65000.0, 65000.0); }

/// A SURFEL TO A RAY (docs/decisions.md, surfels): its flat axis -- a scale of
/// exactly 0, stored as the f16 floor of its logarithm -- held to a ten
/// thousandth of its widest, so a ray tracer that takes a particle to the
/// unit sphere (rt_frames) can: its peak along a ray is then where the ray
/// crosses the disc's plane, and the response there the disc's own, which is
/// a 2D gaussian's ray-splat intersection. A gaussian with no flat axis is as
/// it was.
public static const float kSurfelThin = 1.0e-4;

public float3 surfelSafeScale(float3 s) {
    return max(s, float3(kSurfelThin * max(max(s.x, s.y), s.z)));
}

/// The area of a gaussian's two widest axes: what the levels of detail and a
/// decimation weigh a splat by (opacity times this). Not the product of all
/// three over the smallest, which a surfel's zero takes to nothing.
public float discArea(float3 s) {
    return max(max(s.x * s.y, s.y * s.z), s.x * s.z);
}

/// A FRAME WITH ITS THINNEST AXIS THIRD. A conversion writes a disc's
/// thickness as its third scale, and every reader takes the third axis for
/// the normal and the first two for its plane -- the relight, the curvature
/// and the zonal transfer read in them, sparkwebGPU likewise. Eigenvectors
/// come in whatever order the solver left them: the matx car paint merged
/// 10x had its normal first or second in 48910 of 55477 merges, its
/// curvature then projected on a plane holding the normal. So the columns
/// (the axes) and their scales are turned cyclically -- which keeps the
/// frame's handedness -- until the thinnest is third.
public void thinAxisThird(inout float3x3 axes, inout float3 scales) {
    const uint thin = scales.x <= scales.y ? (scales.x <= scales.z ? 0u : 2u) : (scales.y <= scales.z ? 1u : 2u);
    if (thin == 2u) {
        return;
    }
    const float3 c0 = float3(axes[0][0], axes[1][0], axes[2][0]);
    const float3 c1 = float3(axes[0][1], axes[1][1], axes[2][1]);
    const float3 c2 = float3(axes[0][2], axes[1][2], axes[2][2]);
    // thin 0: (1, 2, 0); thin 1: (2, 0, 1).
    const float3 n0 = thin == 0u ? c1 : c2;
    const float3 n1 = thin == 0u ? c2 : c0;
    const float3 n2 = thin == 0u ? c0 : c1;
    axes = float3x3(n0.x, n1.x, n2.x, n0.y, n1.y, n2.y, n0.z, n1.z, n2.z);
    scales = thin == 0u ? float3(scales.y, scales.z, scales.x) : float3(scales.z, scales.x, scales.y);
}

/// Whether a shape's scales hold a surfel's flat axis.
public bool isSurfel(float3 s) {
    return min(min(s.x, s.y), s.z) == 0.0;
}

/// A HARD SURFEL (docs/decisions.md, kernel hardness): a disc drawn as
/// o exp(-(|u|^p + |v|^p) / 2) in its own axes, p = 2 the gaussian. A merge
/// of a flat patch spans it as a box would, where a gaussian with the same
/// moments throws its tails a third past the patch -- the hairs a merge wider
/// than the camera's pixel leaves at a panel's edge. p travels in the f16
/// logarithm of the disc's flat axis, which every reader takes for a zero
/// scale whatever it holds below e^-100: -infinity (or the -65000 a merge
/// writes) is p = 2, -(1024 + 64 (p - 2)) is p, to a thirtieth up to 32.
/// What it stores as its scale is the moment-matched one (`hardnessVariance`
/// turns it into the box's), so a reader that knows nothing of p draws the
/// gaussian with the same moments. `athenea:splat:hardness` in a file.
public static const float kHardnessMax = 32.0;

public float surfelHardnessLog(float p) {
    return p <= 2.001 ? -65000.0 : -(1024.0 + 64.0 * (min(p, kHardnessMax) - 2.0));
}

public float surfelHardness(float flatLog) {
    return (flatLog > -60000.0 && flatLog <= -1024.0) ? 2.0 + (-flatLog - 1024.0) / 64.0 : 2.0;
}

/// ln Gamma(x) for x in (0, 3]: Stirling's series at x + 3, brought back by
/// the recurrence (1e-6 here).
public float lnGamma(float x) {
    const float z = x + 3.0;
    const float iz = 1.0 / z;
    const float iz2 = iz * iz;
    const float s = (z - 0.5) * log(z) - z + 0.91893853 + iz * (1.0 / 12.0 - iz2 * (1.0 / 360.0 - iz2 / 1260.0));
    return s - log(x * (x + 1.0) * (x + 2.0));
}

/// The variance of exp(-|x|^p / 2) along one axis, a unit gaussian's being 1:
/// 2^(2/p) Gamma(3/p) / Gamma(1/p). 1/3 for the unit box.
public float hardnessVariance(float p) {
    return exp(2.0 / p * 0.69314718 + lnGamma(3.0 / p) - lnGamma(1.0 / p));
}

/// Its kurtosis, m4 / m2^2: Gamma(5/p) Gamma(1/p) / Gamma(3/p)^2. 3 for the
/// gaussian, 1.8 for the box.
public float hardnessKurtosis(float p) {
    return exp(lnGamma(5.0 / p) + lnGamma(1.0 / p) - 2.0 * lnGamma(3.0 / p));
}

/// The p whose kurtosis is `k`, by bisection on 1/p (the kurtosis grows with
/// it), at most `most`; a cluster as peaked as a gaussian or more is one.
public float hardnessOfKurtosis(float k, float most) {
    if (!(k < 2.999)) {
        return 2.0;
    }
    if (hardnessKurtosis(most) >= k) {
        return most;
    }
    float lo = 1.0 / most;   // the hardest
    float hi = 0.5;          // the gaussian
    for (uint it = 0; it < 24; ++it) {
        const float mid = 0.5 * (lo + hi);
        if (hardnessKurtosis(1.0 / mid) > k) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return 2.0 / (lo + hi);
}

public static const float kSH0 = 0.28209479177387814;

/// WHAT A LAMP'S LAYER IS PACKED AT (GpuSplats::lightLayers): its light per
/// unit of the group's radiance times this, in RGB9E5. The light a lamp
/// leaves on a body a few metres away is 1e-5 of its radiance and less, where
/// RGB9E5's smallest exponent leaves a few steps; scaled, full precision
/// reaches down to 1e-9 and the top is 4 -- an emitter seen through clear
/// glass is 1.
public static const float kLightLayerScale = 16384.0;

/// Harmonic words per splat for `restPerColour` basis functions (3, 8, 15).
public uint shWordsFor(uint restPerColour) { return (restPerColour * 3 + 1) / 2; }

/// A texel written as the texture's own bytes, for a device whose store into
/// a texture does not convert (CUDA: surf2Dwrite writes the float4's bytes).
/// `kind`: 1 four 8-bit unorms in one word, 2 four halves in two, 3 four
/// floats in four. The words are the texture's rows packed, row 0 first,
/// which is what copyBufferToTexture takes.
public void atheneaPackTexel(RWStructuredBuffer<uint> packed, uint kind, uint index, float4 texel) {
    if (kind == 1) {
        const uint4 bytes = uint4(round(saturate(texel) * 255.0));
        packed[index] = bytes.x | (bytes.y << 8) | (bytes.z << 16) | (bytes.w << 24);
    } else if (kind == 2) {
        packed[index * 2] = (f32tof16(texel.x) & 0xFFFFu) | ((f32tof16(texel.y) & 0xFFFFu) << 16);
        packed[index * 2 + 1] = (f32tof16(texel.z) & 0xFFFFu) | ((f32tof16(texel.w) & 0xFFFFu) << 16);
    } else {
        packed[index * 4] = asuint(texel.x);
        packed[index * 4 + 1] = asuint(texel.y);
        packed[index * 4 + 2] = asuint(texel.z);
        packed[index * 4 + 3] = asuint(texel.w);
    }
}
