// Copyright (c) 2026 jesus luque.
//
// What a volume needs beyond its grid, computed on the device: the index
// bounds of its leaves (the statistics NanoVDB would have computed on the
// host are off), each leaf's largest value -- the majorant delta tracking
// takes through that leaf -- and the grid's largest value overall. One
// thread a leaf; then, for an instanced volume, one an instance to compose
// its world-to-index; then one a record to write what the grid's walk found
// into it, in the same buffer of words the kernels read it from.
import athenea.volume.nanovdb;

struct PrepareParams {
    uint gridWord;      // the grid's word offset in `words`
    uint leafCount;
    uint leafBase;      // word offset in `words` of this grid's leaf maxima
    uint slot;          // which grid: its six bounds entries, its majorant word
    uint firstRecord;   // the records that read it (instances, finish)
    uint recordCount;
    uint pad0;
    uint pad1;
    float4 prototype0;  // the volume's world to index without its instance, 3x4 rows
    float4 prototype1;
    float4 prototype2;
};

StructuredBuffer<uint>       words;      // the volumes: header, records, grids, leaf maxima
RWStructuredBuffer<uint>     wordsOut;   // the same buffer, for what these kernels write into it (leaf maxima, records)
RWStructuredBuffer<int>      bounds;     // per grid: min xyz, max xyz (exclusive) in index space
RWStructuredBuffer<uint>     majorant;   // per grid: the largest leaf maximum, as float bits (non-negative floats order as uints)
ConstantBuffer<PrepareParams> params;
StructuredBuffer<float4>     instanceRows;   // instances: 3 rows each, the instance's transform before the prototype's

static const uint kHeaderWords = 8;
static const uint kRecordWords = 40;

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeBounds(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.leafCount) return;
    const NvGrid grid = nvGrid(params.gridWord);
    const int3 origin = nvLeafOrigin(words, nvLeaf(words, grid, i));
    const uint base = params.slot * 6u;
    InterlockedMin(bounds[base + 0], origin.x);
    InterlockedMin(bounds[base + 1], origin.y);
    InterlockedMin(bounds[base + 2], origin.z);
    InterlockedMax(bounds[base + 3], origin.x + 8);
    InterlockedMax(bounds[base + 4], origin.y + 8);
    InterlockedMax(bounds[base + 5], origin.z + 8);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeLeafMax(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.leafCount) return;
    const NvGrid grid = nvGrid(params.gridWord);
    const NvLeaf leaf = nvLeaf(words, grid, i);
    float largest = 0.0;
    for (uint n = 0; n < 512; ++n) {
        if (nvLeafActive(words, leaf, n)) {
            largest = max(largest, nvLeafValue(words, leaf, n));
        }
    }
    wordsOut[params.leafBase + i] = asuint(largest);
    InterlockedMax(majorant[params.slot], asuint(max(largest, 0.0)));
}

/// The inverse of an affine 3x4 (rows, translation in w).
float3x4 inverseAffine(float3x4 m) {
    const float3 a = m[0].xyz;
    const float3 b = m[1].xyz;
    const float3 c = m[2].xyz;
    const float3 r0 = cross(b, c);
    const float3 r1 = cross(c, a);
    const float3 r2 = cross(a, b);
    const float det = dot(a, r0);
    const float inv = abs(det) > 1e-30 ? 1.0 / det : 0.0;
    // The rows of the inverse 3x3 are the cofactor columns over the determinant.
    const float3 i0 = float3(r0.x, r1.x, r2.x) * inv;
    const float3 i1 = float3(r0.y, r1.y, r2.y) * inv;
    const float3 i2 = float3(r0.z, r1.z, r2.z) * inv;
    const float3 t = float3(m[0].w, m[1].w, m[2].w);
    return float3x4(float4(i0, -dot(i0, t)), float4(i1, -dot(i1, t)), float4(i2, -dot(i2, t)));
}

float4x4 full(float3x4 m) {
    return float4x4(m[0], m[1], m[2], float4(0.0, 0.0, 0.0, 1.0));
}

/// One instance of an instanced volume: world to index is the volume's own
/// (the prototype's, the host's) after the inverse of the instance's.
[shader("compute")]
[numthreads(64, 1, 1)]
void volumeInstances(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.recordCount) return;
    const float3x4 instance = float3x4(instanceRows[k * 3], instanceRows[k * 3 + 1], instanceRows[k * 3 + 2]);
    const float4x4 m = mul(full(float3x4(params.prototype0, params.prototype1, params.prototype2)),
                           full(inverseAffine(instance)));
    const uint b = kHeaderWords + (params.firstRecord + k) * kRecordWords;
    for (uint r = 0; r < 3; ++r) {
        for (uint c = 0; c < 4; ++c) {
            wordsOut[b + 4 + r * 4 + c] = asuint(m[r][c]);
        }
    }
}

/// A record's bounds and majorant, and where leaf 0 sits, once its grid's
/// leaves have been walked: one thread a record that reads the grid.
[shader("compute")]
[numthreads(64, 1, 1)]
void volumeFinish(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.recordCount) return;
    const uint b = kHeaderWords + (params.firstRecord + tid.x) * kRecordWords;
    const uint base = params.slot * 6u;
    const bool any = params.leafCount > 0;
    for (uint k = 0; k < 3; ++k) {
        wordsOut[b + 16 + k] = asuint(any ? float(bounds[base + k]) : 0.0);
        wordsOut[b + 19 + k] = asuint(any ? float(bounds[base + 3 + k]) : 0.0);
    }
    wordsOut[b + 27] = majorant[params.slot];
    wordsOut[b + 3] = any ? nvLeaf(words, nvGrid(params.gridWord), 0u).byteOffset / 4u : 0u;
}
