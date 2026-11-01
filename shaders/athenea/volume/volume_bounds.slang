// Copyright (c) 2026 jesus luque.
//
// Where the volumes stand in the world: each record's index box (its grid's
// leaves, volume_prepare's) through the inverse of its world-to-index, folded
// into one box -- what a viewer frames a stage of volumes alone by. One group
// of threads, each folding a stride of the records, then the group's halves
// folded together; a record whose grid has no leaf is left out.
StructuredBuffer<uint>     words;    // the volumes: header, records, grids (world::VolumeSet)
RWStructuredBuffer<float4> result;   // low corner (w: records counted), high corner

static const uint kHeaderWords = 8;
static const uint kRecordWords = 168;
static const uint kThreads = 256;

groupshared float3 sharedLo[kThreads];
groupshared float3 sharedHi[kThreads];
groupshared uint   sharedCount[kThreads];

float4 row(uint b, uint r) {
    const uint w = b + 4u + r * 4u;
    return float4(asfloat(words[w]), asfloat(words[w + 1u]), asfloat(words[w + 2u]), asfloat(words[w + 3u]));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void volumeWorldBounds(uint3 tid: SV_GroupThreadID) {
    const uint t = tid.x;
    const uint records = words[0];
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    uint counted = 0u;
    for (uint i = t; i < records; i += kThreads) {
        const uint b = kHeaderWords + i * kRecordWords;
        const float3 boxLo = float3(asfloat(words[b + 16u]), asfloat(words[b + 17u]), asfloat(words[b + 18u]));
        const float3 boxHi = float3(asfloat(words[b + 19u]), asfloat(words[b + 20u]), asfloat(words[b + 21u]));
        if (any(boxHi <= boxLo)) {
            continue;   // a grid of no leaf: nothing of it is drawn
        }
        // World to index is A x + t; index to world is A^-1 (x - t).
        const float4 r0 = row(b, 0u);
        const float4 r1 = row(b, 1u);
        const float4 r2 = row(b, 2u);
        const float3x3 a = float3x3(r0.xyz, r1.xyz, r2.xyz);
        const float det = determinant(a);
        if (!(abs(det) > 0.0)) {
            continue;   // a map that collapses the grid stands nowhere
        }
        // The inverse by its cofactors: row k of A^-1 is (column k's cross
        // products) / det.
        const float3 c0 = cross(r1.xyz, r2.xyz) / det;
        const float3 c1 = cross(r2.xyz, r0.xyz) / det;
        const float3 c2 = cross(r0.xyz, r1.xyz) / det;
        const float3x3 inv = transpose(float3x3(c0, c1, c2));
        const float3 shift = float3(r0.w, r1.w, r2.w);
        // A box through an affine map: the centre moves, the half extents
        // go through the absolute linear part.
        const float3 centre = mul(inv, (boxLo + boxHi) * 0.5 - shift);
        const float3 halfSize = (boxHi - boxLo) * 0.5;
        const float3 extent = float3(dot(abs(inv[0]), halfSize), dot(abs(inv[1]), halfSize), dot(abs(inv[2]), halfSize));
        lo = min(lo, centre - extent);
        hi = max(hi, centre + extent);
        counted += 1u;
    }
    sharedLo[t] = lo;
    sharedHi[t] = hi;
    sharedCount[t] = counted;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = kThreads / 2u; step > 0u; step /= 2u) {
        if (t < step) {
            sharedLo[t] = min(sharedLo[t], sharedLo[t + step]);
            sharedHi[t] = max(sharedHi[t], sharedHi[t + step]);
            sharedCount[t] += sharedCount[t + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0u) {
        result[0] = float4(sharedLo[0], float(sharedCount[0]));
        result[1] = float4(sharedHi[0], 0.0);
    }
}
