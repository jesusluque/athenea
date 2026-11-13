// Copyright (c) 2026 jesus luque.
//
// A SCENE FOUND A LOOK AT A TIME. A render is one camera's look at the world
// at one time, and what it gives is not only a picture: it is which gaussians
// that camera saw. `seen` keeps, a gaussian, the look that first saw it (0:
// none has), and the scene as it is known at a time is the gaussians the looks
// up to it found. The next look then adds only what was hidden from the ones
// before -- behind a column, round a corner -- and says how much of what it
// saw the world already held.
//
// What a look saw is what the blend says each gaussian gave its frame (its
// contribution, in 1/4096 of a pixel): a gaussian counts once it gave
// `threshold` of that.
module scene_discover;
import athenea.common.packing;

struct DiscoverParams {
    uint count;       // gaussians in the piece; or the words a clear walks
    uint base;        // the piece's first slot in the frame
    uint stamp;       // the look: what a gaussian found by it is stamped with
    uint threshold;   // in 1/4096 of a pixel
    uint upTo;        // discoverMask, discoverCount: the last look that counts
    uint drop;        // discoverMask: 1, and `seenRead` is a record of what is gone: those are what is not drawn (with `upTo`, those gone by that look)
    // discoverGone: a gaussian wider than `coarse` times its depth along the view (the row `viewZ` gives
    // it; 0: not asked) is coarser than the look that flagged its pixels, and goes on a fifth of its
    // weight rather than on most of it.
    float coarse;
    // discoverGone: with `most` over nothing, a gaussian is stamped only where this share of its weight
    // fell on flagged pixels, whatever its width: the rule for one that is refined and not gone -- a
    // parent a finer look covered part of stays the surface's where the look did not reach.
    float most;
    float4 viewZ;
    // discoverCut: pixels a unit of x over z, the threshold in pixels, and the last look that counts
    // (`upTo`; 0: all).
    float focal; float threshold2; uint pad3; uint pad4;
};

StructuredBuffer<uint>     contribution;   // a uint a slot, as the blend summed it
RWStructuredBuffer<uint>   seen;           // a uint a gaussian
RWStructuredBuffer<uint>   counters;       // found now, seen again, known up to `upTo`
StructuredBuffer<uint>     seenRead;
StructuredBuffer<uint>     lifted;         // two a slot: weight where a frame saw through, weight (1/1024 of a pixel)
StructuredBuffer<float4>   positions;      // xyz, opacity
StructuredBuffer<uint>     shape;          // four words a gaussian, as GpuSplats packs them
RWStructuredBuffer<float>  sizes;          // discoverSizes: a neural cloud's latents, a gaussian's size as its colour
RWStructuredBuffer<float4> positionsOut;
RWStructuredBuffer<uint>   cleared;
StructuredBuffer<uint>     refinedRead;    // discoverCut: the look that laid finer gaussians over each, 0 while it is the finest
StructuredBuffer<float>    parentRead;     // discoverCut: the width of what each was laid over, 0 where over nothing
StructuredBuffer<uint>     harmonics;      // discoverGather: a cloud's, `stamp` words a gaussian
RWStructuredBuffer<uint>   shapeOut;
RWStructuredBuffer<uint>   harmonicsOut;
ConstantBuffer<DiscoverParams> params;

/// HOW WIDE A GAUSSIAN IS, as a look counts it: the middle of its three scales (a log). A disc made by a
/// look is as wide as the look's pixel across the way its surface falls from the eye, longer along it and
/// thin through it; its width is the pixel's, and that is the middle one. The largest is the stretch, and
/// by it every floor and every wall seen aslant read as coarser than the look that had just made them.
float discWidth(uint row) {
    const float a = lowHalf(shape[row + 1]);
    const float b = highHalf(shape[row + 1]);
    const float c = lowHalf(shape[row + 2]);
    return max(min(a, b), min(max(a, b), c));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void discoverClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        cleared[tid.x] = 0u;
    }
}

/// What this look saw: stamped where no look had seen it, counted either way.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverMark(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || contribution[params.base + i] < params.threshold) {
        return;
    }
    if (seen[i] == 0u) {
        seen[i] = params.stamp;
        InterlockedAdd(counters[0], 1u);
    } else {
        InterlockedAdd(counters[1], 1u);
    }
}

/// WHAT A FRAME SAW THROUGH IS GONE. A gaussian most of whose weight in the
/// scene's frame fell on pixels the new frame sees past is no longer there:
/// stamped with the look that says so, in a record of its own (`seen` here),
/// and drawn by nothing after.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverGone(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || seen[i] != 0u) {
        return;
    }
    const uint through = lifted[(params.base + i) * 2];
    const uint whole = lifted[(params.base + i) * 2 + 1];
    // A coarse gaussian lies under many of the look's pixels, some flagged and some not, and what is
    // laid over it is finer: it need not be mostly flagged to be in the way.
    bool wide = false;
    if (params.coarse > 0.0) {
        const uint row = i * 4;
        const float size = exp(discWidth(row));
        const float z = dot(params.viewZ.xyz, positions[i].xyz) + params.viewZ.w;
        wide = z > 0.0 && size > params.coarse * z;
    }
    const bool mostly = params.most > 0.0 ? float(through) >= params.most * float(whole)
                                          : (wide ? through * 5u > whole : through * 2u > whole);
    if (whole >= params.threshold && mostly) {
        seen[i] = params.stamp;
        InterlockedAdd(counters[0], 1u);
    }
}

/// HOW COARSE THE SCENE IS, AS A PICTURE. A gaussian's largest standard
/// deviation written where a neural cloud keeps its base colour (its features
/// zero, its network's weights zero): drawn by that cloud, a frame's colour
/// is the size of the gaussians each pixel is made of, blended as they are.
/// `upTo` is the latents a gaussian (the neural cloud's stride), `stamp` how
/// many of them are features.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverSizes(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float size = exp(discWidth(row));
    for (uint k = 0; k < params.upTo; ++k) {
        sizes[i * params.upTo + k] = k < params.stamp ? 0.0 : size;
    }
}

/// How many gaussians the looks up to `upTo` found.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverCount(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < params.count && seenRead[i] != 0u && seenRead[i] <= params.upTo) {
        InterlockedAdd(counters[2], 1u);
    }
}

/// The piece as it is known: a gaussian no look up to `upTo` found keeps its
/// place and has no opacity, so nothing draws it.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverMask(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    // A record of what is gone, with a last look: what went after it still stands.
    const bool known = params.drop != 0 ? (seenRead[i] == 0u || (params.upTo != 0u && seenRead[i] > params.upTo))
                                        : (seenRead[i] != 0u && seenRead[i] <= params.upTo);
    positionsOut[i] = float4(p.xyz, known ? p.w : 0.0);
}

/// THE SCENE CUT FOR A VIEW. A look makes a gaussian as wide as its pixel is at its distance, so a
/// gaussian's width is its level of detail, and a finer look does not end a coarser gaussian: it refines it
/// (`refinedRead`), and each finer one keeps the width of what it was laid over (`parentRead`). From there a
/// gaussian decides by itself whether it is the one to draw, with no tree to walk:
///
///   it stands at the look asked for            (`seenRead`, the record of what is gone);
///   what it was laid over would be too coarse  (its parent's width over `threshold2` pixels there), or it
///                                              was laid over nothing;
///   and it is fine enough itself, or nothing finer came (its own width under the threshold, or not refined).
///
/// A parent and its children then never both draw, and where no look came near the coarse gaussian is all
/// there is and is drawn however near the eye. One that is not drawn keeps its place and has no opacity.
bool cutDraws(uint i) {
    const float4 p = positions[i];
    const bool stands = seenRead[i] == 0u || (params.upTo != 0u && seenRead[i] > params.upTo);
    const bool finest = refinedRead[i] == 0u || (params.upTo != 0u && refinedRead[i] > params.upTo);
    if (!stands) {
        return false;
    }
    if (!(params.threshold2 > 0.0)) {
        // No threshold: the finest there is, everywhere.
        return finest;
    }
    const float z = max(dot(params.viewZ.xyz, p.xyz) + params.viewZ.w, 1.0e-6);
    const float pixels = params.focal / z;
    const float own = exp(discWidth(i * 4)) * pixels;
    const float over = parentRead[i] * pixels;
    // Nine tenths of the threshold on the parent's side: where the two tests fall between a parent and its
    // children, both are drawn rather than neither.
    const bool parentTooCoarse = parentRead[i] <= 0.0 || over >= 0.9 * params.threshold2;
    return parentTooCoarse && (finest || own < params.threshold2);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void discoverCut(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = positions[i];
    positionsOut[i] = float4(p.xyz, cutDraws(i) ? p.w : 0.0);
}

/// The same answer as a flag a gaussian (`cleared`), for the cut that is gathered: a gaussian that is not
/// drawn costs its projection all the same, and a view from far draws a few of very many.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverCutFlags(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < params.count) {
        cleared[i] = cutDraws(i) ? 1u : 0u;
    }
}

/// The gaussians flagged (`contribution`), each at its place among them (`lifted`, their prefix sum): its
/// position, its shape and its `stamp` words of harmonics, and nothing else of it -- a cloud to draw.
[shader("compute")]
[numthreads(64, 1, 1)]
void discoverGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || contribution[i] == 0u) {
        return;
    }
    const uint to = lifted[i];
    positionsOut[to] = positions[i];
    for (uint k = 0; k < 4u; ++k) {
        shapeOut[to * 4u + k] = shape[i * 4u + k];
    }
    for (uint w = 0; w < params.stamp; ++w) {
        harmonicsOut[to * params.stamp + w] = harmonics[i * params.stamp + w];
    }
}
