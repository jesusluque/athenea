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
    float coarse; uint pad2;
    float4 viewZ;
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
ConstantBuffer<DiscoverParams> params;

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
        const float size = exp(max(lowHalf(shape[row + 1]), max(highHalf(shape[row + 1]), lowHalf(shape[row + 2]))));
        const float z = dot(params.viewZ.xyz, positions[i].xyz) + params.viewZ.w;
        wide = z > 0.0 && size > params.coarse * z;
    }
    if (whole >= params.threshold && (wide ? through * 5u > whole : through * 2u > whole)) {
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
    const float largest = max(lowHalf(shape[row + 1]), max(highHalf(shape[row + 1]), lowHalf(shape[row + 2])));   // logs
    const float size = exp(largest);
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
