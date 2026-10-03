// Copyright (c) 2026 jesus luque.
//
// A key per internal node, its height, for the radix sort that orders the
// refit's passes, and the count of each height (bvh_levels.slang).
module bvh_level_keys;
import bvh_levels;

StructuredBuffer<uint>      heights;
RWStructuredBuffer<uint>    levelKeys;
RWStructuredBuffer<uint>    levelValues;
RWStructuredBuffer<uint>    levelCounts;   // kMaxLevels counters
ConstantBuffer<LevelParams> params;

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
