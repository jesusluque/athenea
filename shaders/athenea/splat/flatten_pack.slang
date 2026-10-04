// Copyright (c) 2026 jesus luque.
//
// A FLATTENED CLOUD IN A FORMAT'S OWN NUMBERS (athenea flatten, task TXF):
// the records splatFlatten wrote (flatten_basis.slang's layout) as the floats
// a 3DGS PLY holds, the bytes of SPZ's six streams, or the attributes of a
// glTF KHR_gaussian_splatting primitive. The processor only writes headers
// and copies these out (athenea/io/SplatWriters.h).
//
// - `flattenBounds`, then `flattenBoundsResolve`: the box the positions fill
//   (glTF wants it on POSITION, and SPZ's fixed point is chosen from it), and
//   what each format will lose: sizes under its floor, SPZ harmonics past its
//   [-1, 1], gaussians whose higher bands were shrunk to stay positive.
// - `flattenPackPly`: 62 floats a gaussian -- x y z, nx ny nz, f_dc, f_rest
//   all red then green then blue, the opacity as a logit, ln sigma, w x y z.
// - `flattenPackSpz`: one SPZ stream, four bytes a thread, each byte worked
//   out from its gaussian as Niantic's packer works it out (third_party/spz,
//   `packGaussians`), so the result reads with any SPZ reader.
// - `flattenPackGlb`: the glTF attributes, each a block of its own.
module flatten_pack;
import flatten_basis;

struct PackParams {
    uint  count;            // gaussians in `records`
    uint  stream;           // SPZ: 0 positions, 1 alphas, 2 colours, 3 scales, 4 rotations, 5 harmonics
    uint  words;            // SPZ: words of the stream
    uint  fractionalBits;   // SPZ's fixed point
    uint  sh1Bits;          // SPZ: bits kept of band 1
    uint  shRestBits;       // SPZ: bits kept of bands 2 and 3
    float logScaleFloor;    // ln sigma no size goes below (PLY, glTF)
    uint  base;             // where this cloud's gaussians start in the output (PLY, glTF)
    uint  total;            // gaussians of the whole output (glTF's blocks)
};
ConstantBuffer<PackParams> params;
StructuredBuffer<float>   records;
RWStructuredBuffer<float> floats;   // PLY records or glTF blocks
RWStructuredBuffer<uint>  words;    // an SPZ stream
// [0..2] min, [3..5] max of x y z as ordered bits; [6] sizes under the floor;
// [7] gaussians with an SPZ harmonic past one; [8] gaussians shrunk;
// [9] sizes under SPZ's floor; then, resolved: [16..21] the box as float
// bits, [22] SPZ's fractional bits.
RWStructuredBuffer<uint>  stats;

static const float kSpzColourScale = 0.15;
static const float kSpzLogScaleFloor = -10.0;

uint orderedBits(float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

float fromOrdered(uint u) {
    return asfloat((u & 0x80000000u) != 0u ? (u & 0x7FFFFFFFu) : ~u);
}

float rec(uint g, uint k) {
    return records[g * kFlatStride + k];
}

float3 coefficient(uint g, uint k) {
    const uint at = g * kFlatStride + kFlatSh + k * 3;
    return float3(records[at], records[at + 1], records[at + 2]);
}

/// What SPZ's [-1, 1] makes of a gaussian's higher bands: one where they fit,
/// and otherwise the factor that brings the largest back to one -- the whole
/// set scaled, so the shape of the view dependence is kept and only its
/// strength is lost, rather than the largest clipped alone.
float spzBox(uint g) {
    float most = 0.0;
    for (uint k = 1; k < 16; ++k) {
        const float3 c = abs(coefficient(g, k));
        most = max(most, max(c.x, max(c.y, c.z)));
    }
    return most > 1.0 ? 1.0 / most : 1.0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenBounds(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    for (uint a = 0; a < 3; ++a) {
        const uint bits = orderedBits(rec(g, kFlatPosition + a));
        InterlockedMin(stats[a], bits);
        InterlockedMax(stats[3 + a], bits);
    }
    uint floored = 0, spzFloored = 0;
    for (uint a = 0; a < 3; ++a) {
        const float s = rec(g, kFlatLogScale + a);
        floored += s < params.logScaleFloor ? 1u : 0u;
        spzFloored += s < kSpzLogScaleFloor ? 1u : 0u;
    }
    if (floored != 0) {
        InterlockedAdd(stats[6], floored);
    }
    if (spzBox(g) < 1.0) {
        InterlockedAdd(stats[7], 1u);
    }
    if (rec(g, kFlatShrink) < 0.999) {
        InterlockedAdd(stats[8], 1u);
    }
    if (spzFloored != 0) {
        InterlockedAdd(stats[9], spzFloored);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void flattenBoundsResolve(uint3 tid: SV_DispatchThreadID) {
    float most = 1.0e-6;
    for (uint a = 0; a < 6; ++a) {
        const float v = fromOrdered(stats[a]);
        stats[16 + a] = asuint(v);
        most = max(most, abs(v));
    }
    // As many fractional bits as keep the farthest coordinate inside 24
    // signed bits, with one to spare; never more than 20 (a micrometre in
    // metres), never fewer than none.
    stats[22] = uint(clamp(int(floor(22.0 - log2(most))), 0, 20));
}

float logScaleOf(uint g, uint a) {
    return max(rec(g, kFlatLogScale + a), params.logScaleFloor);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackPly(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint out = (params.base + g) * 62;
    for (uint a = 0; a < 3; ++a) {
        floats[out + a] = rec(g, kFlatPosition + a);
        floats[out + 3 + a] = rec(g, kFlatNormal + a);
    }
    const float3 dc = coefficient(g, 0);
    floats[out + 6] = dc.x;
    floats[out + 7] = dc.y;
    floats[out + 8] = dc.z;
    for (uint k = 1; k < 16; ++k) {
        const float3 c = coefficient(g, k);
        floats[out + 9 + (k - 1)] = c.x;
        floats[out + 9 + 15 + (k - 1)] = c.y;
        floats[out + 9 + 30 + (k - 1)] = c.z;
    }
    const float alpha = clamp(rec(g, kFlatOpacity), 1.0e-6, 1.0 - 1.0e-6);
    floats[out + 54] = log(alpha / (1.0 - alpha));
    for (uint a = 0; a < 3; ++a) {
        floats[out + 55 + a] = logScaleOf(g, a);
    }
    for (uint a = 0; a < 4; ++a) {
        floats[out + 58 + a] = rec(g, kFlatRotation + a);
    }
}

uint toByte(float x) {
    return uint(clamp(round(x), 0.0, 255.0));
}

uint quantizeSh(float x, uint bucket) {
    int q = int(round(x * 128.0) + 128.0);
    q = (q + int(bucket) / 2) / int(bucket) * int(bucket);
    return uint(clamp(q, 0, 255));
}

/// SPZ's rotation, smallest three: two bits naming the largest of x y z w,
/// then the other three as a sign and nine bits of magnitude over sqrt(1/2),
/// the largest made positive -- Niantic's `packQuaternionSmallestThree`.
uint smallestThree(uint g) {
    const float4 wxyz = float4(rec(g, kFlatRotation), rec(g, kFlatRotation + 1), rec(g, kFlatRotation + 2),
                               rec(g, kFlatRotation + 3));
    const float4 q = normalize(float4(wxyz.y, wxyz.z, wxyz.w, wxyz.x));   // x y z w
    uint largest = 0;
    for (uint i = 1; i < 4; ++i) {
        if (abs(q[i]) > abs(q[largest])) {
            largest = i;
        }
    }
    const uint negate = q[largest] < 0.0 ? 1u : 0u;
    uint comp = largest;
    for (uint i = 0; i < 4; ++i) {
        if (i != largest) {
            const uint negbit = (q[i] < 0.0 ? 1u : 0u) ^ negate;
            const uint mag = uint(511.0 * (abs(q[i]) / 0.70710678118654752) + 0.5);
            comp = (comp << 10u) | (negbit << 9u) | min(mag, 511u);
        }
    }
    return comp;
}

static const uint kSpzBytes[6] = { 9, 1, 3, 3, 4, 45 };

uint spzByte(uint g, uint o) {
    switch (params.stream) {
    case 0: {
        const float scale = float(1u << params.fractionalBits);
        const int fixed = int(round(rec(g, kFlatPosition + o / 3) * scale));
        return uint(fixed >> (8 * int(o % 3))) & 0xFFu;
    }
    case 1:
        return toByte(rec(g, kFlatOpacity) * 255.0);
    case 2:
        return toByte(coefficient(g, 0)[o] * (kSpzColourScale * 255.0) + 0.5 * 255.0);
    case 3:
        return toByte((rec(g, kFlatLogScale + o) + 10.0) * 16.0);
    case 4:
        return (smallestThree(g) >> (8 * o)) & 0xFFu;
    default: {
        const uint k = o / 3 + 1;
        const uint bits = k < 4 ? params.sh1Bits : params.shRestBits;
        return quantizeSh(coefficient(g, k)[o % 3] * spzBox(g), 1u << (8u - bits));
    }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackSpz(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    if (w >= params.words) {
        return;
    }
    const uint per = kSpzBytes[min(params.stream, 5u)];
    const uint bytes = params.count * per;
    uint word = 0;
    for (uint n = 0; n < 4; ++n) {
        const uint b = w * 4 + n;
        if (b < bytes) {
            word |= spzByte(b / per, b % per) << (8 * n);
        }
    }
    words[w] = word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenPackGlb(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint n = params.total;
    const uint at = params.base + g;
    for (uint a = 0; a < 3; ++a) {
        floats[at * 3 + a] = rec(g, kFlatPosition + a);
    }
    // KHR_gaussian_splatting:ROTATION is x y z w, as every glTF rotation.
    floats[3 * n + at * 4] = rec(g, kFlatRotation + 1);
    floats[3 * n + at * 4 + 1] = rec(g, kFlatRotation + 2);
    floats[3 * n + at * 4 + 2] = rec(g, kFlatRotation + 3);
    floats[3 * n + at * 4 + 3] = rec(g, kFlatRotation);
    for (uint a = 0; a < 3; ++a) {
        floats[7 * n + at * 3 + a] = exp(logScaleOf(g, a));
    }
    floats[10 * n + at] = saturate(rec(g, kFlatOpacity));
    for (uint k = 0; k < 16; ++k) {
        const float3 c = coefficient(g, k);
        const uint block = 11 * n + k * 3 * n;
        floats[block + at * 3] = c.x;
        floats[block + at * 3 + 1] = c.y;
        floats[block + at * 3 + 2] = c.z;
    }
}
