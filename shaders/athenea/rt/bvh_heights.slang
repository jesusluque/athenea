// Copyright (c) 2026 jesus luque.
//
// A node's height (1 + its taller child's, a leaf being 0), iterated with the
// build's refit passes until neither moves (bvh_levels.slang).
module bvh_heights;
import bvh_levels;

StructuredBuffer<uint>      children;
RWStructuredBuffer<uint>    heights;   // per internal node, cloud-local
RWStructuredBuffer<uint>    changed;   // set where a height moved; the refit's pass cleared or set it
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
    // Children are cloud-local, and so are the heights (one buffer a cloud).
    const uint h = 1 + max(childHeight(children[node * 2]), childHeight(children[node * 2 + 1]));
    if (h != heights[u]) {
        changed[u] = 1u;
    }
    heights[u] = h;
}
