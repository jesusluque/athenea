// Copyright (c) 2026 jesus luque.
//
// A GAUSSIAN'S TRANSFER AS ZONAL LOBES IN ITS OWN FRAME (proposal 014 B).
//
// A transfer is how much of an environment reaches a gaussian -- visibility
// times the cosine, projected onto nine harmonics -- and it is baked in the
// world. A cloud a skeleton carries turns its gaussians every frame, and nine
// harmonics in the world do not turn with them: rotating them is a 9 x 9 a
// gaussian (or one per part, and then a gaussian on a border belongs to two).
// A zonal lobe does turn cheaply. It is a function symmetric about an axis,
// three numbers for bands 0 to 2, and rotating it is rotating the axis: one
// 3 x 3. So each gaussian keeps one or two lobes whose axes are written in its
// own frame, and the frame -- whatever pose the skeleton put it in -- takes
// them to the world (Relightable Full-Body Gaussian Codec Avatars, 2501.14726,
// keeps its transfer the same way and for the same reason).
//
// THE FIT. A lobe about the axis `a` with coefficients `z` is, as harmonics,
// `g_lm = z_l k_l Y_lm(a)`, `k_l = sqrt(4 pi / (2l + 1))`. For a given axis
// the best `z` is the projection, `z_l = k_l sum_m f_lm Y_lm(a)`, and by the
// addition theorem what it leaves is `|f|^2 - sum_l z_l^2`: the axis to take
// is the one whose projection keeps the most. That is searched for -- the
// band-1 direction, the gaussian's normal, sixty-four directions over a
// hemisphere (the energy is even in the axis), then a local refinement -- and
// with two lobes the second is fitted to what the first leaves, and the two
// are refitted against each other twice (Sloan, "Fast ZH approximation", in
// "Stupid Spherical Harmonics Tricks", 2008).
//
// THE SHADOW BITS GO WITH THEM. The sixty-four bits of which ways out are open
// were traced over an octahedral grid of the world; beside a zonal transfer
// they are laid out over a grid of the gaussian's frame instead, each cell of
// it taking the world's cell its centre falls in, so the sun is looked up in
// the frame too and a wing's feather keeps its hard shadow when it turns.
//
// The pictures, as the host lays them out (entries in rows of `width`, padded
// to `stride` pixels):
//   Transfer  4 entries a gaussian: harmonics 0-3, 4-7, (8, the two words of
//             bits as the floats they are the bits of, 1 where the bake found
//             a surface), and the gaussian's frame as a quaternion (x, y, z, w)
//             -- the frame it had when the bake traced it
//   Color     3 entries a gaussian: (axis 0 as the octahedral square's u, v,
//             z0, z1), (z2, axis 1 u, v, z0), (z1, z2, the two words of bits
//             in the gaussian's frame)
//
// What crosses back is a histogram of the fit's relative error against the
// nine harmonics, `|f - g| / |f|`, by quarter octave.
module splattransferzonal;

struct TzParams {
    uint count;
    uint lobes;        ///< 1 or 2
    uint inWidth;
    uint inStride;

    uint outWidth;
    uint outStride;
    uint rebin;        ///< 1: the shadow bits are laid out in the gaussian's frame
    uint pad0;
};

StructuredBuffer<float4>   transferIn;
RWStructuredBuffer<float4> target;
RWStructuredBuffer<uint>   counters;   // [0, 48) the error's histogram, 48 the gaussians fitted
ConstantBuffer<TzParams>   params;

static const uint  kBuckets = 48;
static const uint  kFitted = 48;
static const uint  kCounters = 49;
static const float kPi = 3.14159265358979;

uint pictureIndex(uint entry, uint width, uint stride) {
    return (entry / width) * stride + entry % width;
}

/// The real harmonics to degree 2 in the engine's order
/// (athenea/light/environment.slang's `envBasisValue`); repeated because a
/// bundle is compiled on its own, and the test holds the two together.
float basis(uint k, float3 d) {
    const float x = d.x, y = d.y, z = d.z;
    switch (k) {
    case 0: return 0.28209479177387814;
    case 1: return -0.4886025119029199 * y;
    case 2: return 0.4886025119029199 * z;
    case 3: return -0.4886025119029199 * x;
    case 4: return 1.0925484305920792 * x * y;
    case 5: return -1.0925484305920792 * y * z;
    case 6: return 0.31539156525252005 * (2.0 * z * z - x * x - y * y);
    case 7: return -1.0925484305920792 * x * z;
    case 8: return 0.5462742152960396 * (x * x - y * y);
    default: return 0.0;
    }
}

uint bandOf(uint k) {
    return k == 0 ? 0u : (k < 4 ? 1u : 2u);
}

float bandScale(uint band) {
    return band == 0 ? 3.5449077018110318 : (band == 1 ? 2.0466534158929770 : 1.5853309190424043);
}

/// The best zonal coefficients about `a`: the projection, band by band.
float3 zonalAbout(float f[9], float3 a) {
    float3 sum = float3(0.0);
    [unroll] for (uint k = 0; k < 9; ++k) {
        sum[bandOf(k)] += f[k] * basis(k, a);
    }
    return sum * float3(bandScale(0), bandScale(1), bandScale(2));
}

/// What a lobe about `a` keeps of `f`: the sum of its squared coefficients.
float kept(float f[9], float3 a) {
    const float3 z = zonalAbout(f, a);
    return dot(z, z);
}

void takeAway(inout float f[9], float3 a, float3 z) {
    [unroll] for (uint k = 0; k < 9; ++k) {
        f[k] -= z[bandOf(k)] * bandScale(bandOf(k)) * basis(k, a);
    }
}

/// The square [0, 1]^2 and back, as athenea/common/octahedral.slang has it.
float2 octWrap(float2 v) {
    return (1.0 - abs(v.yx)) * float2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
}

float2 octEncode(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-20);
    float2 p = n.z >= 0.0 ? n.xy : octWrap(n.xy);
    return p * 0.5 + 0.5;
}

float3 octDecode(float2 f) {
    f = f * 2.0 - 1.0;
    float3 n = float3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    const float t = saturate(-n.z);
    n.x += n.x >= 0.0 ? -t : t;
    n.y += n.y >= 0.0 ? -t : t;
    return normalize(n);
}

/// A rotation matrix whose columns are the axes, from (x, y, z, w)
/// (athenea/common/packing.slang's `axesOfQuaternion`).
float3x3 axesOf(float4 q) {
    q = dot(q, q) > 1.0e-20 ? normalize(q) : float4(0.0, 0.0, 0.0, 1.0);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

/// THE AXIS THAT KEEPS THE MOST of `f`: two hints, the band-1 direction,
/// sixty-four directions over a hemisphere (Fibonacci), then a walk that
/// halves its step whenever no neighbour keeps more.
float3 bestAxis(float f[9], float3 hint0, float3 hint1) {
    float3 best = normalize(hint0);
    float most = kept(f, best);
    const float3 second = normalize(hint1);
    if (kept(f, second) > most) {
        best = second;
        most = kept(f, second);
    }
    // Y_1 is (-y, z, -x) up to a constant, so the band's own direction is:
    const float3 linear = float3(-f[3], -f[1], f[2]);
    if (dot(linear, linear) > 1.0e-20) {
        const float3 d = normalize(linear);
        const float k = kept(f, d);
        if (k > most) {
            best = d;
            most = k;
        }
    }
    for (uint i = 0; i < 64; ++i) {
        const float z = 1.0 - (float(i) + 0.5) / 64.0;
        const float r = sqrt(max(1.0 - z * z, 0.0));
        const float phi = float(i) * 2.39996322972865332;
        const float3 d = float3(r * cos(phi), r * sin(phi), z);
        const float k = kept(f, d);
        if (k > most) {
            best = d;
            most = k;
        }
    }
    float step = 0.2;
    for (uint walk = 0; walk < 24 && step > 1.0e-3; ++walk) {
        const float3 helper = abs(best.z) < 0.9 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
        const float3 t = normalize(cross(helper, best));
        const float3 b = cross(best, t);
        bool moved = false;
        [unroll] for (uint n = 0; n < 4; ++n) {
            const float3 towards = (n & 1) != 0 ? ((n & 2) != 0 ? -b : b) : ((n & 2) != 0 ? -t : t);
            const float3 d = normalize(best + towards * step);
            const float k = kept(f, d);
            if (k > most) {
                best = d;
                most = k;
                moved = true;
            }
        }
        if (!moved) {
            step *= 0.5;
        }
    }
    return best;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tzClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < kCounters) {
        counters[tid.x] = 0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void tzFit(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const float4 e0 = transferIn[pictureIndex(g * 4, params.inWidth, params.inStride)];
    const float4 e1 = transferIn[pictureIndex(g * 4 + 1, params.inWidth, params.inStride)];
    const float4 e2 = transferIn[pictureIndex(g * 4 + 2, params.inWidth, params.inStride)];
    const float4 e3 = transferIn[pictureIndex(g * 4 + 3, params.inWidth, params.inStride)];
    float f[9] = { e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w, e2.x };
    const uint2 worldBits = uint2(asuint(e2.y), asuint(e2.z));
    const float3x3 frame = axesOf(e3);
    const float3 normal = float3(frame[0][2], frame[1][2], frame[2][2]);

    float whole = 0.0;
    [unroll] for (uint k = 0; k < 9; ++k) {
        whole += f[k] * f[k];
    }
    float3 axis0 = normal;
    float3 axis1 = normal;
    float3 z0 = float3(0.0);
    float3 z1 = float3(0.0);
    const bool found = e2.w > 0.5 && whole > 1.0e-20;
    if (found) {
        axis0 = bestAxis(f, normal, normal);
        z0 = zonalAbout(f, axis0);
        if (params.lobes > 1) {
            // The second lobe holds what the first leaves, then each is
            // refitted to what the other leaves.
            float left[9] = f;
            takeAway(left, axis0, z0);
            axis1 = bestAxis(left, -normal, axis0);
            z1 = zonalAbout(left, axis1);
            for (uint round = 0; round < 2; ++round) {
                float rest[9] = f;
                takeAway(rest, axis1, z1);
                axis0 = bestAxis(rest, axis0, normal);
                z0 = zonalAbout(rest, axis0);
                float other[9] = f;
                takeAway(other, axis0, z0);
                axis1 = bestAxis(other, axis1, -normal);
                z1 = zonalAbout(other, axis1);
            }
        }
        float error[9] = f;
        takeAway(error, axis0, z0);
        if (params.lobes > 1) {
            takeAway(error, axis1, z1);
        }
        float wrong = 0.0;
        [unroll] for (uint k = 0; k < 9; ++k) {
            wrong += error[k] * error[k];
        }
        const float relative = sqrt(wrong / whole);
        // By quarter octave from 2^-10: bucket 0 holds everything below.
        const int bucket = relative > 0.0 ? clamp(int(floor(4.0 * log2(relative))) + 40, 0, int(kBuckets) - 1) : 0;
        InterlockedAdd(counters[bucket], 1u);
        InterlockedAdd(counters[kFitted], 1u);
    }

    // The axes in the gaussian's frame: the frame's transpose.
    const float2 local0 = octEncode(mul(axis0, frame));
    const float2 local1 = octEncode(mul(axis1, frame));

    uint2 bits = worldBits;
    if (params.rebin != 0 && any(worldBits != uint2(0, 0))) {
        bits = uint2(0, 0);
        for (uint cell = 0; cell < 64; ++cell) {
            const float3 local = octDecode((float2(cell % 8, cell / 8) + 0.5) / 8.0);
            const float3 world = mul(frame, local);
            const uint2 at = uint2(clamp(int2(floor(octEncode(world) * 8.0)), int2(0), int2(7)));
            const uint index = at.y * 8 + at.x;
            const uint word = index < 32 ? worldBits.x : worldBits.y;
            if (((word >> (index & 31u)) & 1u) != 0) {
                if (cell < 32) {
                    bits.x |= 1u << cell;
                } else {
                    bits.y |= 1u << (cell - 32);
                }
            }
        }
    }

    target[pictureIndex(g * 3, params.outWidth, params.outStride)] = float4(local0, z0.x, z0.y);
    target[pictureIndex(g * 3 + 1, params.outWidth, params.outStride)] =
        float4(z0.z, local1, params.lobes > 1 ? z1.x : 0.0);
    target[pictureIndex(g * 3 + 2, params.outWidth, params.outStride)] =
        float4(params.lobes > 1 ? z1.y : 0.0, params.lobes > 1 ? z1.z : 0.0, asfloat(bits.x), asfloat(bits.y));
}
