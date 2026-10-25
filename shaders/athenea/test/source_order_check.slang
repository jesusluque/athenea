// Copyright (c) 2026 jesus luque.
//
// WHERE ON ITS SOURCE A GAUSSIAN STANDS, CHECKED (tests/mesh2splat, "[source]"):
// a converted cloud's source coordinates as the file carries them, keyed as
// mesh2splat keys them (common/source_order.slang), and counted here.
//
//   srcKeys           each gaussian's key; one the file holds empty (opacity 0) last
//   srcOrderBreaks    counts[0]: places where a key is above the next one's
//   srcWordsDiffer    counts[0]: words of two arrays that differ
//   srcCoarseNear     counts[0]: coarse gaussians with st whose place in the
//                     fine order (where their key falls) holds a fine gaussian
//                     of another mesh or further than `tolerance` in st;
//                     counts[1] those looked at (a mesh without st is ordered
//                     by its triangles' cells, which a coarser grid numbers
//                     otherwise)
//   srcIdsFound       counts[0]: ids of the second set not among the first's
//   srcPositionsMatch counts[0]: gaussians whose distance from their set's
//                     first gaussian differs by more than `tolerance` from
//                     the other set's of the same index (the same up to the
//                     axes a file is written in)
module source_order_check;
import athenea.common.source_order;

struct CheckParams {
    uint  count;       ///< gaussians (the first set's)
    uint  other;       ///< the second set's
    uint  width;       ///< srcWordsDiffer: words a gaussian
    float tolerance;
};

StructuredBuffer<int>        mesh;
StructuredBuffer<int>        triangle;
StructuredBuffer<int2>       cell;
StructuredBuffer<float2>     uv;
StructuredBuffer<float>      opacity;     // one a gaussian, linear
RWStructuredBuffer<uint2>    keys;        // (lo, hi)
StructuredBuffer<uint2>      otherKeys;
StructuredBuffer<int>        otherMesh;
StructuredBuffer<float2>     otherUv;
StructuredBuffer<uint>       wordsA;
StructuredBuffer<uint>       wordsB;
StructuredBuffer<float4>     positionsA;
StructuredBuffer<float4>     positionsB;
RWStructuredBuffer<uint>     counts;
ConstantBuffer<CheckParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void srcKeys(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const int2 c = cell[i];
    keys[i] = opacity[i] > 0.0 ? sourceKey(uint(max(mesh[i], 0)), uv[i], uint(max(triangle[i], 0)),
                                           (uint(c.x) & 0xFFFFu) | (uint(c.y) << 16))
                               : kSourceLast;
}

bool above(uint2 a, uint2 b) {
    return a.y > b.y || (a.y == b.y && a.x > b.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void srcOrderBreaks(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i + 1 >= params.count) {
        return;
    }
    if (above(keys[i], keys[i + 1])) {
        InterlockedAdd(counts[0], 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void srcWordsDiffer(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count * params.width) {
        return;
    }
    if (wordsA[i] != wordsB[i]) {
        InterlockedAdd(counts[0], 1u);
    }
}

/// The first set is the coarse one, the second (`other*`) the fine one, in
/// its order: where a coarse key falls among the fine keys, by bisection.
[shader("compute")]
[numthreads(64, 1, 1)]
void srcCoarseNear(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || opacity[i] <= 0.0 || params.other == 0 || !sourceHasUv(uv[i])) {
        return;
    }
    const uint2 k = keys[i];
    uint lo = 0, hi = params.other;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (above(k, otherKeys[mid])) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    const uint j = min(lo, params.other - 1);
    InterlockedAdd(counts[1], 1u);
    if (!(otherMesh[j] == mesh[i] && length(otherUv[j] - uv[i]) <= params.tolerance)) {
        InterlockedAdd(counts[0], 1u);
    }
}

/// `wordsA` the first set's ids (two words a gaussian), `wordsB` the second's.
[shader("compute")]
[numthreads(64, 1, 1)]
void srcIdsFound(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.other) {
        return;
    }
    const uint2 id = uint2(wordsB[i * 2], wordsB[i * 2 + 1]);
    for (uint k = 0; k < params.count; ++k) {
        if (wordsA[k * 2] == id.x && wordsA[k * 2 + 1] == id.y) {
            return;
        }
    }
    InterlockedAdd(counts[0], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void srcPositionsMatch(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= min(params.count, params.other)) {
        return;
    }
    // Up to the axes a file is written in: each one's distance from the
    // set's first gaussian, which no turn or flip of the axes changes.
    const float a = length(positionsA[i].xyz - positionsA[0].xyz);
    const float b = length(positionsB[i].xyz - positionsB[0].xyz);
    if (abs(a - b) > params.tolerance) {
        InterlockedAdd(counts[0], 1u);
    }
}
