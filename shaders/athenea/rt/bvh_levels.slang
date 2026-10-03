// Copyright (c) 2026 jesus luque.
//
// REFITTING A CLOUD'S TREE A LEVEL AT A TIME.
//
// A skinned cloud is posed anew every frame with the same particles in the
// same order, so the tree bvh_hierarchy built stays a valid tree -- only its
// boxes are stale. Refitting them is one pass per height, bottom up: a node
// of height h (1 + the taller child's, a leaf being 0) reads children whose
// boxes the passes before it settled, so after the last height every box is
// exact, whatever the buffer held before. No pass is repeated "until nothing
// changes", which is what made the build's refit read a counter on the host.
//
//   at build     bvhHeights      the heights, iterated to a fixed point with
//                                the build's own refit passes
//                bvhLevelKeys    a key per node (its height) for the radix
//                                sort, and the count of each height
//   per pose     bvhRefitLevel   the nodes of one height: children's union,
//                                a leaf's box made from the posed particle
//
// `levelNodes[nodeBase + k]`, sorted by height, is the node (cloud-local)
// pass `levelStart + t` refits; the host keeps where each height starts.
module bvh_levels;
import athenea.common.packing;
import rt_common;
import bvh_common;

public static const uint kMaxLevels = 128;

public struct LevelParams {
    public uint count;        // particles in the cloud
    public uint nodeBase;     // first internal node in the combined buffers
    public uint leafBase;     // first leaf in the combined buffers
    public uint levelStart;   // bvhRefitLevel: this height's first entry in levelNodes
    public uint levelCount;   // bvhRefitLevel: nodes of this height
    public uint pad0; public uint pad1; public uint pad2;
};

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     children;
StructuredBuffer<uint>     leaves;
RWStructuredBuffer<uint>   heights;      // per internal node, cloud-local
RWStructuredBuffer<uint>   changed;      // bvhHeights: or-ed into the refit's flag
RWStructuredBuffer<uint>   levelKeys;    // bvhLevelKeys
RWStructuredBuffer<uint>   levelValues;  // bvhLevelKeys
RWStructuredBuffer<uint>   levelCounts;  // bvhLevelKeys: kMaxLevels counters
StructuredBuffer<uint>     levelNodes;   // bvhRefitLevel
RWStructuredBuffer<float4> boxes;        // bvhRefitLevel
ConstantBuffer<LevelParams> params;

uint childHeight(uint child) {
    return (child & kLeafBit) != 0 ? 0u : heights[child];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhHeights(uint3 tid: SV_DispatchThreadID) {
    const uint u = tid.x;
    if (u + 1 >= params.count) {
        return;
    }
    const uint node = params.nodeBase + u;
    // A cloud's children are cloud-local; its heights are too (one buffer a cloud).
    const uint h = 1 + max(childHeight(children[node * 2]), childHeight(children[node * 2 + 1]));
    if (h != heights[u]) {
        changed[u] = 1u;
    }
    heights[u] = h;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhLevelKeys(uint3 tid: SV_DispatchThreadID) {
    const uint u = tid.x;
    if (u + 1 >= params.count) {
        return;
    }
    const uint h = min(heights[u], kMaxLevels - 1);
    levelKeys[u] = h;
    levelValues[u] = u;
    InterlockedAdd(levelCounts[h], 1u);
}

void childBox(uint child, out float3 lo, out float3 hi) {
    if ((child & kLeafBit) != 0) {
        const uint particle = leaves[params.leafBase + (child & ~kLeafBit)];
        const uint row = particle * 4;
        rtParticleBox(positions[particle], shape[row], shape[row + 1], shape[row + 2], lo, hi);
    } else {
        lo = boxes[(params.nodeBase + child) * 2].xyz;
        hi = boxes[(params.nodeBase + child) * 2 + 1].xyz;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bvhRefitLevel(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.levelCount) {
        return;
    }
    const uint u = levelNodes[params.nodeBase + params.levelStart + tid.x];
    const uint node = params.nodeBase + u;
    float3 aLo, aHi, bLo, bHi;
    childBox(children[node * 2], aLo, aHi);
    childBox(children[node * 2 + 1], bLo, bHi);
    boxes[node * 2] = float4(min(aLo, bLo), 0.0);
    boxes[node * 2 + 1] = float4(max(aHi, bHi), 0.0);
}
