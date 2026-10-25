// Copyright (c) 2026 jesus luque.
//
// WHAT A ROUND TRIP IS CHECKED WITH, and what an export is made from:
//
//   athzCompare  words of two arrays that differ, counted: 0 is a record
//                decoded bit for bit
//   athzHash     a hash of an array, its words tagged by where they stand
//                (so a swap changes it) and summed, so the threads may add
//                in any order: two arrays hash alike when they are alike
//   athzFlat     a cloud's records (GpuSplats) as the flattened record
//                FlattenPack packs into SPZ's streams (flatten_basis.slang's
//                layout): position, linear opacity, log scales, rotation
//                w x y z, the 3DGS DC and the rest rgb per basis -- turned from
//                the engine's frame (a 3DGS file's, right-down-front) to
//                SPZ's right-up-back, which its reader turns back
module athz_check;
import athenea.common.packing;
import athenea.splat.flatten_basis;

struct AthzCheckParams {
    uint words;           ///< athzCompare, athzHash: words compared or hashed
    uint offsetA;         ///< words skipped in `a`
    uint offsetB;         ///< and in `b`
    uint count;           ///< athzFlat: records
    uint shWords;
    uint restPerColour;
    uint toSrgb;          ///< athzFlat: a cloud in linear light, its DC written as sRGB
    uint pad0;
};
ConstantBuffer<AthzCheckParams> params;

ByteAddressBuffer a;
ByteAddressBuffer b;
RWStructuredBuffer<Atomic<uint>> counters;   ///< athzCompare: [0]; athzHash: [0] [1]

uint mix(uint h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzCompare(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.words) {
        return;
    }
    if (a.Load((params.offsetA + i) * 4u) != b.Load((params.offsetB + i) * 4u)) {
        counters[0].add(1u);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzHash(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.words) {
        return;
    }
    const uint w = a.Load((params.offsetA + i) * 4u);
    counters[0].add(mix(w ^ mix(i * 0x9E3779B9u + 1u)));
    counters[1].add(mix(w + mix(i ^ 0x7F4A7C15u)) * 2654435761u);
}

ByteAddressBuffer        positions;
ByteAddressBuffer        shape;
ByteAddressBuffer        sh;
RWStructuredBuffer<float> flat;

/// A rest coefficient's sign with y and z turned (splat_encoding's restFlipSign).
float flipSign(uint basis) {
    static const float kSigns[15] = { -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0 };
    return kSigns[min(basis, 14u)];
}

float srgbOf(float v) {
    v = max(v, 0.0);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzFlat(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = asfloat(positions.Load4(i * 16u));
    const uint4 s = shape.Load4(i * 16u);
    const float4 q = decodeQuaternion(s.x);   // x y z w
    float3 base = float3(highHalf(s.z), lowHalf(s.w), highHalf(s.w));
    if (params.toSrgb != 0u) {
        base = float3(srgbOf(base.x), srgbOf(base.y), srgbOf(base.z));
    }
    const uint at = i * kFlatStride;
    for (uint k = 0; k < kFlatStride; ++k) {
        flat[at + k] = 0.0;
    }
    flat[at + kFlatPosition + 0u] = p.x;
    flat[at + kFlatPosition + 1u] = -p.y;
    flat[at + kFlatPosition + 2u] = -p.z;
    flat[at + kFlatOpacity] = p.w;
    flat[at + kFlatLogScale + 0u] = lowHalf(s.y);
    flat[at + kFlatLogScale + 1u] = highHalf(s.y);
    flat[at + kFlatLogScale + 2u] = lowHalf(s.z);
    flat[at + kFlatRotation + 0u] = q.w;
    flat[at + kFlatRotation + 1u] = q.x;
    flat[at + kFlatRotation + 2u] = -q.y;
    flat[at + kFlatRotation + 3u] = -q.z;
    flat[at + kFlatSh + 0u] = (base.x - 0.5) / kSH0;
    flat[at + kFlatSh + 1u] = (base.y - 0.5) / kSH0;
    flat[at + kFlatSh + 2u] = (base.z - 0.5) / kSH0;
    const uint rest = min(params.restPerColour, kFlatCoefficients - 1u);
    for (uint j = 0; j < rest * 3u; ++j) {
        const uint word = sh.Load((i * params.shWords + j / 2u) * 4u);
        const float v = (j & 1u) != 0u ? highHalf(word) : lowHalf(word);
        flat[at + kFlatSh + 3u + j] = v * flipSign(j / 3u);
    }
    flat[at + kFlatShrink] = 1.0;
}
