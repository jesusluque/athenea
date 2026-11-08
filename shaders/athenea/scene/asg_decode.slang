// Copyright (c) 2026 jesus luque.
//
// AN .asg DECODED, A THREAD A GAUSSIAN (3Dspace's bitstream v1, proposal 002;
// its reference decoder is proto/asg_io.py's `unpack`). The gaussian's anchor
// is found by a binary search over the anchors' `first`; its leaf is read at
// `leaf_off + i * leaf_width` bits of the leaf section, fields from the least
// significant bit, and every field is a code in its anchor's range:
// value = lo + code / (2^bits - 1) * (hi - lo), the midpoint at 0 bits.
//
// Written: a float record (io::SplatEncoding as scene/Asg.cpp sets it: x y z,
// opacity, three linear scales, rotation w x y z, the DC coefficient, metallic,
// roughness, transmission, the shading normal), the anchor's Cryptomatte id,
// the material's thin-walled flag, the transfer's direct and indirect blocks
// (an anchor's mean plus the leaf's residual over its basis) and the open
// directions (the anchor's majority pattern XOR a dictionary entry, written
// with each byte's bits most significant first, as shadowBits primvars hold
// them).
module asg_decode;

struct AsgParams {
    uint  count;
    uint  anchors;
    uint  flags;
    uint  anchorSize;
    uint  offAnchors;
    uint  offLeaves;
    uint  offMaterials;
    uint  offShadowDict;
    uint4 widths0;        ///< tilt, theta, ls, aniso
    uint4 widths1;        ///< thick, op, col, rough
    uint4 widths2;        ///< shadow, tw, full_pos, full_rot
    uint4 widths3;        ///< tilt, theta, ls, aniso of a glossy anchor
    uint  blocks;
    uint  shadowBitsPer;
    uint  meanK;
    uint  majorityAt;
    uint4 dims;
    uint4 ks;
    uint4 meanClusterAt;
    uint4 meanAt;
    uint4 scaleAt;
    uint4 residualAt;
    uint4 meansTableAt;
    uint4 basesAt;
    uint  dictionaryAt;
    uint  stride;
};
ConstantBuffer<AsgParams> params;

ByteAddressBuffer          file;
RWStructuredBuffer<float>  records;
RWStructuredBuffer<uint>   crypto;
RWStructuredBuffer<int>    thinWalled;
RWStructuredBuffer<float>  direct;
RWStructuredBuffer<float>  indirect;
RWStructuredBuffer<uint>   shadowBits;

static const uint kOpacityLog = 4u, kMeansCpca = 16u, kShadowGlobal = 32u;
static const uint kFull = 1u, kGlossy = 2u, kOpaque = 4u, kTheta = 8u;
static const float kPi = 3.14159265358979;

uint byteAt(uint at) { return (file.Load(at & ~3u) >> ((at & 3u) * 8u)) & 0xFFu; }
uint halfAt(uint at) { return (file.Load(at & ~3u) >> ((at & 3u) * 8u)) & 0xFFFFu; }
float f16At(uint at) { return f16tof32(halfAt(at)); }
float f32At(uint at) { return asfloat(file.Load(at)); }

/// `width` bits (at most 16) starting at bit `bit` of the leaf section.
uint bitsAt(uint bit, uint width) {
    if (width == 0u) {
        return 0u;
    }
    const uint word = params.offLeaves + (bit >> 5u) * 4u;
    const uint shift = bit & 31u;
    uint v = file.Load(word) >> shift;
    if (shift + width > 32u) {
        v |= file.Load(word + 4u) << (32u - shift);
    }
    return v & ((1u << width) - 1u);
}

float dequant(uint code, float lo, float hi, uint bits) {
    if (bits == 0u) {
        return 0.5 * (lo + hi);
    }
    return lo + float(code) / float((1u << bits) - 1u) * (hi - lo);
}

/// smallest-three: three codes of `bits` (components over sqrt(1/2)), the
/// index of the largest, which is positive.
float4 smallestThree(uint3 codes, uint largest, uint bits) {
    const float r[3] = { dequant(codes.x, -1.0, 1.0, bits) * sqrt(0.5), dequant(codes.y, -1.0, 1.0, bits) * sqrt(0.5),
                         dequant(codes.z, -1.0, 1.0, bits) * sqrt(0.5) };
    float q[4];
    uint k = 0u;
    for (uint i = 0u; i < 4u; ++i) {
        if (i == largest) {
            q[i] = sqrt(max(0.0, 1.0 - r[0] * r[0] - r[1] * r[1] - r[2] * r[2]));
        } else {
            q[i] = r[k];
            ++k;
        }
    }
    return float4(q[0], q[1], q[2], q[3]);   // x y z w
}

/// Columns are the rotated axes.
float3x3 quatToMatrix(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
                    2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
                    2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y));
}

/// The rotation whose columns are m's (right-handed), as x y z w (Shepperd).
float4 matrixToQuat(float3x3 m) {
    const float t = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (t > 0.0) {
        const float s = sqrt(t + 1.0) * 2.0;
        q = float4((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s, 0.25 * s);
    } else if (m[0][0] >= m[1][1] && m[0][0] >= m[2][2]) {
        const float s = sqrt(max(1.0 + m[0][0] - m[1][1] - m[2][2], 1e-12)) * 2.0;
        q = float4(0.25 * s, (m[1][0] + m[0][1]) / s, (m[2][0] + m[0][2]) / s, (m[2][1] - m[1][2]) / s);
    } else if (m[1][1] >= m[2][2]) {
        const float s = sqrt(max(1.0 + m[1][1] - m[0][0] - m[2][2], 1e-12)) * 2.0;
        q = float4((m[1][0] + m[0][1]) / s, 0.25 * s, (m[2][1] + m[1][2]) / s, (m[0][2] - m[2][0]) / s);
    } else {
        const float s = sqrt(max(1.0 + m[2][2] - m[0][0] - m[1][1], 1e-12)) * 2.0;
        q = float4((m[2][0] + m[0][2]) / s, (m[2][1] + m[1][2]) / s, 0.25 * s, (m[1][0] - m[0][1]) / s);
    }
    return normalize(q);
}

float3 octDecode(float2 xy) {
    float3 n = float3(xy, 1.0 - abs(xy.x) - abs(xy.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(xy.yx)) * float2(xy.x >= 0.0 ? 1.0 : -1.0, xy.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

uint reverseByte(uint b) {
    b = ((b & 0xF0u) >> 4u) | ((b & 0x0Fu) << 4u);
    b = ((b & 0xCCu) >> 2u) | ((b & 0x33u) << 2u);
    b = ((b & 0xAAu) >> 1u) | ((b & 0x55u) << 1u);
    return b;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void asgDecode(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    // The anchor: the last whose first is at most g.
    uint lo = 0u, hi = params.anchors;
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if (file.Load(params.offAnchors + mid * params.anchorSize + 28u) <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const uint a = params.offAnchors + lo * params.anchorSize;
    const uint i = g - file.Load(a + 28u);
    const uint format = file.Load(a + 32u);
    const uint leafWidth = halfAt(a + 42u);
    // leaf_off is a u64 of bits; the section is under 4 GB, so its low word.
    uint bit = file.Load(a + 8u) + i * leafWidth;
    uint read = 0u;

    const bool glossy = (format & kGlossy) != 0u;
    const uint tilt = glossy ? params.widths3.x : params.widths0.x;
    const uint theta = glossy ? params.widths3.y : params.widths0.y;
    const uint lsBits = glossy ? params.widths3.z : params.widths0.z;
    const uint anisoBits = glossy ? params.widths3.w : params.widths0.w;

    // The anchor's frame (columns u, v, n) and centre.
    const uint lowFrame = file.Load(a), highFrame = file.Load(a + 4u);
    const float3x3 F = quatToMatrix(smallestThree(uint3(lowFrame & 0xFFFFu, lowFrame >> 16u, highFrame & 0xFFFFu),
                                                  (highFrame >> 16u) & 3u, 16u));
    const float3 centre = float3(f32At(a + 16u), f32At(a + 20u), f32At(a + 24u));
    const float3 boxLo = float3(f16At(a + 48u), f16At(a + 50u), f16At(a + 52u));
    const float3 boxHi = float3(f16At(a + 54u), f16At(a + 56u), f16At(a + 58u));

    float3 local;
    float3x3 axes;   // columns: largest, middle, smallest (the disc's normal)
    if ((format & kFull) != 0u) {
        const uint pb = params.widths2.z, rb = params.widths2.w;
        const uint x = bitsAt(bit, pb), y = bitsAt(bit + pb, pb), z = bitsAt(bit + 2u * pb, pb);
        bit += 3u * pb;
        local = float3(dequant(x, boxLo.x, boxHi.x, pb), dequant(y, boxLo.y, boxHi.y, pb), dequant(z, boxLo.z, boxHi.z, pb));
        const uint3 q = uint3(bitsAt(bit, rb), bitsAt(bit + rb, rb), bitsAt(bit + 2u * rb, rb));
        bit += 3u * rb;
        const uint largest = bitsAt(bit, 2u);
        bit += 2u;
        axes = quatToMatrix(smallestThree(q, largest, rb));
    } else {
        const uint ub = (format >> 4u) & 15u, vb = (format >> 8u) & 15u, hb = (format >> 12u) & 15u;
        const uint u = bitsAt(bit, ub), v = bitsAt(bit + ub, vb), h = bitsAt(bit + ub + vb, hb);
        bit += ub + vb + hb;
        local = float3(dequant(u, boxLo.x, boxHi.x, ub), dequant(v, boxLo.y, boxHi.y, vb), dequant(h, boxLo.z, boxHi.z, hb));
        const uint ox = bitsAt(bit, tilt), oy = bitsAt(bit + tilt, tilt);
        bit += 2u * tilt;
        const float3 n = octDecode(float2(dequant(ox, -1.0, 1.0, tilt), dequant(oy, -1.0, 1.0, tilt)));
        float angle = 0.0;
        if ((format & kTheta) != 0u) {
            angle = dequant(bitsAt(bit, theta), 0.0, kPi, theta);
            bit += theta;
        }
        // In the disc's plane: the anchor's u projected onto it, and n x that.
        const float3 reference = normalize(float3(1.0, 0.0, 0.0) - n * n.x);
        const float3 a1 = cos(angle) * reference + sin(angle) * cross(n, reference);
        axes = float3x3(a1.x, cross(n, a1).x, n.x, a1.y, cross(n, a1).y, n.y, a1.z, cross(n, a1).z, n.z);
    }
    const float3x3 world = mul(F, axes);

    const uint ls = bitsAt(bit, lsBits);
    bit += lsBits;
    const uint aniso = bitsAt(bit, anisoBits);
    bit += anisoBits;
    const uint thickBits = params.widths1.x;
    const uint thick = bitsAt(bit, thickBits);
    bit += thickBits;
    const uint opBits = (format & kOpaque) != 0u ? 0u : params.widths1.y;
    const uint op = bitsAt(bit, opBits);
    bit += opBits;
    const uint colBits = params.widths1.z;
    const uint3 rgb = uint3(bitsAt(bit, colBits), bitsAt(bit + colBits, colBits), bitsAt(bit + 2u * colBits, colBits));
    bit += 3u * colBits;
    const uint roughBits = params.widths1.w;
    const uint rough = bitsAt(bit, roughBits);
    bit += roughBits;
    const uint shadowBitsWidth = params.shadowBitsPer != 0u ? params.widths2.x : 0u;
    const uint shadow = bitsAt(bit, shadowBitsWidth);
    bit += shadowBitsWidth;

    const float sMax = exp(dequant(ls, f16At(a + 60u), f16At(a + 62u), lsBits));
    const float sMid = sMax * dequant(aniso, f16At(a + 64u), f16At(a + 66u), anisoBits);
    const float sMin = sMid * exp(dequant(thick, f16At(a + 68u), f16At(a + 70u), thickBits));
    float alpha = 1.0;
    if (opBits != 0u) {
        if ((params.flags & kOpacityLog) != 0u) {
            alpha = op == (1u << opBits) - 1u ? 1.0 : exp2(dequant(op, f16At(a + 46u), 0.0, opBits));
        } else {
            alpha = dequant(op, 0.0, 1.0, opBits);
        }
    }
    const float3 colLo = float3(f16At(a + 72u), f16At(a + 74u), f16At(a + 76u));
    const float3 colHi = float3(f16At(a + 78u), f16At(a + 80u), f16At(a + 82u));
    const uint material = params.offMaterials + halfAt(a + 44u) * 4u;

    const float3 p = centre + mul(F, local);
    const float4 q = matrixToQuat(world);
    const uint r = g * params.stride;
    records[r + 0u] = p.x;
    records[r + 1u] = p.y;
    records[r + 2u] = p.z;
    records[r + 3u] = alpha;
    records[r + 4u] = sMax;
    records[r + 5u] = sMid;
    records[r + 6u] = sMin;
    records[r + 7u] = q.w;
    records[r + 8u] = q.x;
    records[r + 9u] = q.y;
    records[r + 10u] = q.z;
    records[r + 11u] = dequant(rgb.x, colLo.x, colHi.x, colBits);
    records[r + 12u] = dequant(rgb.y, colLo.y, colHi.y, colBits);
    records[r + 13u] = dequant(rgb.z, colLo.z, colHi.z, colBits);
    records[r + 14u] = float(byteAt(material)) / 255.0;
    records[r + 15u] = dequant(rough, 0.0, 1.0, roughBits);
    records[r + 16u] = float(byteAt(material + 1u)) / 255.0;
    records[r + 17u] = world[0][2];
    records[r + 18u] = world[1][2];
    records[r + 19u] = world[2][2];
    crypto[g] = file.Load(a + 36u);
    thinWalled[g] = int(byteAt(material + 2u));

    // The transfer: each block's mean, plus the residual over the anchor's basis.
    const uint tw = params.widths2.y;
    for (uint b = 0u; b < params.blocks; ++b) {
        const uint D = params.dims[b], K = params.ks[b];
        float residualWeights[8];
        for (uint j = 0u; j < 8u; ++j) {
            residualWeights[j] = 0.0;
        }
        for (uint j = 0u; j < K && j < 8u; ++j) {
            const float s = f16At(a + params.scaleAt[b] + j * 2u);
            residualWeights[j] = dequant(bitsAt(bit, tw), -s, s, tw);
            bit += tw;
        }
        const uint basis = K != 0u ? params.basesAt[b] + byteAt(a + params.residualAt[b]) * K * D * 2u : 0u;
        const bool cpca = (params.flags & kMeansCpca) != 0u;
        const uint Km = params.meanK;
        const uint table = cpca ? params.meansTableAt[b] + byteAt(a + params.meanClusterAt[b]) * (D + Km * D + Km) * 2u : 0u;
        for (uint d = 0u; d < D; ++d) {
            float value;
            if (cpca) {
                value = f16At(table + d * 2u);
                for (uint k = 0u; k < Km; ++k) {
                    const float s = f16At(table + (D + Km * D + k) * 2u);
                    value += dequant(byteAt(a + params.meanClusterAt[b] + 1u + k), -s, s, 8u) * f16At(table + (D + k * D + d) * 2u);
                }
            } else {
                value = f16At(a + params.meanAt[b] + d * 2u);
            }
            for (uint j = 0u; j < K && j < 8u; ++j) {
                value += residualWeights[j] * f16At(basis + (j * D + d) * 2u);
            }
            if (b == 0u) {
                direct[g * D + d] = value;
            } else if (b == 1u) {
                indirect[g * D + d] = value;
            }
        }
    }

    // The open directions: majority XOR entry, each byte's bits reversed.
    if (params.shadowBitsPer != 0u) {
        const uint bytesPer = params.shadowBitsPer / 8u;
        const uint entry = (params.flags & kShadowGlobal) != 0u
                               ? params.offShadowDict + shadow * bytesPer
                               : a + params.dictionaryAt + shadow * bytesPer;
        for (uint w = 0u; w < bytesPer / 4u; ++w) {
            uint word = 0u;
            for (uint k = 0u; k < 4u; ++k) {
                const uint j = w * 4u + k;
                word |= reverseByte(byteAt(a + params.majorityAt + j) ^ byteAt(entry + j)) << (8u * k);
            }
            shadowBits[g * (bytesPer / 4u) + w] = word;
        }
    }
}
