// Copyright (c) 2026 jesus luque.
//
// THE FRAME AFTER THE PROJECTION, SIZED ON THE GPU (081: no count crosses to
// the CPU inside a frame). webArgs reads the two totals the projection's
// prefix sums wrote -- splats visible, tile pairs wanted -- holds them to
// what the buffers have room for, and writes them where the later kernels
// read them (`counts`) and as the arguments of the indirect dispatches that
// follow (`args`). The gather, the emit and the ranges are splat_gather_counts,
// splat_emit and splat_ranges with their count read from `counts`, and their
// thread index folded into two dimensions, so a dispatch past 65535 groups
// is two-dimensional (082 section 1.4).
module web_tiles;
import athenea.splat.frame;

/// Groups a dimension of a folded dispatch: x is at most this, y the rest.
public static const uint kFoldGroups = 32768;

/// The linear thread of a folded dispatch of 256-thread groups.
uint folded(uint3 tid) {
    return tid.x + tid.y * (kFoldGroups * 256u);
}

public struct WebFrame {
    public uint visibleAt;   // the word of `totals` the visible splats' prefix sum ended in
    public uint pairsAt;     // and the pairs'
    public uint slots;       // what the projection had: visible splats are at most this
    public uint pairCap;     // pairs the buffers hold
};

/// `counts`, as webArgs leaves it.
public static const uint kVisible = 0;      // splats visible, sorted, emitted
public static const uint kPairs = 1;        // pairs drawn: at most pairCap
public static const uint kPairsWanted = 2;  // pairs the frame wanted
public static const uint kArgsVisible = 0;  // `args` words: x, y, z a dispatch
public static const uint kArgsPairs = 3;

StructuredBuffer<uint>    totals;
RWStructuredBuffer<uint>  countsOut;
RWStructuredBuffer<uint>  args;
ConstantBuffer<WebFrame>  web;

void foldInto(uint at, uint threads) {
    const uint groups = (threads + 255u) / 256u;
    const uint x = min(groups, kFoldGroups);
    args[at] = x;
    args[at + 1] = x == 0u ? 0u : (groups + x - 1u) / x;
    args[at + 2] = 1u;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void webArgs(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint visible = min(totals[web.visibleAt], web.slots);
    const uint wanted = totals[web.pairsAt];
    const uint pairs = min(wanted, web.pairCap);
    countsOut[kVisible] = visible;
    countsOut[kPairs] = pairs;
    countsOut[kPairsWanted] = wanted;
    foldInto(kArgsVisible, visible);
    foldInto(kArgsPairs, pairs);
}

StructuredBuffer<uint>      counts;
StructuredBuffer<uint>      order;
StructuredBuffer<uint>      tilesTouched;
RWStructuredBuffer<uint>    sortedCounts;
ConstantBuffer<FrameParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void webGather(uint3 tid: SV_DispatchThreadID) {
    const uint k = folded(tid);
    if (k >= counts[kVisible]) {
        return;
    }
    sortedCounts[k] = tilesTouched[order[k]];
}

StructuredBuffer<uint>       offsets;
StructuredBuffer<uint>       tileRects;
StructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>     pairTiles;
RWStructuredBuffer<uint>     pairSplats;

/// splat_emit's, with the pairs' room in `params.base`.
[shader("compute")]
[numthreads(256, 1, 1)]
void webEmit(uint3 tid: SV_DispatchThreadID) {
    const uint k = folded(tid);
    if (k >= counts[kVisible]) {
        return;
    }
    const uint s = order[k];
    const uint expected = tilesTouched[s];
    if (expected == 0) {
        return;
    }
    const uint first = offsets[k];
    const ProjRecord record = proj[s];
    const float2 centre = record.screen.xy;
    const float cutoff = record.screen.w;
    const float3 conic = record.conic.xyz;
    const uint4 rect = uint4(tileRects[s * 4], tileRects[s * 4 + 1], tileRects[s * 4 + 2], tileRects[s * 4 + 3]);
    uint written = 0;
    for (uint ty = rect.y; ty <= rect.w && written < expected; ++ty) {
        for (uint tx = rect.x; tx <= rect.z && written < expected; ++tx) {
            if (!tileTouched(params, centre, conic, cutoff, tx, ty)) {
                continue;
            }
            const uint at = first + written;
            if (at < params.base) {
                pairTiles[at] = ty * params.tilesX + tx;
                pairSplats[at] = s;
            }
            written += 1;
        }
    }
}

StructuredBuffer<uint>    sortedTiles;
RWStructuredBuffer<uint>  ranges;

/// splat_ranges', over the pairs the frame drew.
[shader("compute")]
[numthreads(256, 1, 1)]
void webRanges(uint3 tid: SV_DispatchThreadID) {
    const uint i = folded(tid);
    const uint count = counts[kPairs];
    if (i >= count) {
        return;
    }
    const uint t = sortedTiles[i];
    if (i == 0 || sortedTiles[i - 1] != t) {
        ranges[t * 2] = i;
    }
    if (i + 1 == count || sortedTiles[i + 1] != t) {
        ranges[t * 2 + 1] = i + 1;
    }
}
