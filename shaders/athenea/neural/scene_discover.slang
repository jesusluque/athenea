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

struct DiscoverParams {
    uint count;       // gaussians in the piece; or the words a clear walks
    uint base;        // the piece's first slot in the frame
    uint stamp;       // the look: what a gaussian found by it is stamped with
    uint threshold;   // in 1/4096 of a pixel
    uint upTo;        // discoverMask, discoverCount: the last look that counts
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>     contribution;   // a uint a slot, as the blend summed it
RWStructuredBuffer<uint>   seen;           // a uint a gaussian
RWStructuredBuffer<uint>   counters;       // found now, seen again, known up to `upTo`
StructuredBuffer<uint>     seenRead;
StructuredBuffer<float4>   positions;      // xyz, opacity
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
    const bool known = seenRead[i] != 0u && seenRead[i] <= params.upTo;
    positionsOut[i] = float4(p.xyz, known ? p.w : 0.0);
}
