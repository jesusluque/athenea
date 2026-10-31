// Copyright (c) 2026 jesus luque.
//
// A converted grid against what it was made of: the active voxels
// counted through the leaf masks (a count known by construction from the
// boxes the fixture filled), and each leaf's maximum recomputed through
// the accessor, voxel by voxel over the voxels a lookup in its cell reads,
// against the prepare kernel's.
import athenea.volume.nanovdb;

struct CheckParams {
    uint gridWord;
    uint leafCount;
    uint leafBase;
    uint pad0;
};

StructuredBuffer<uint>     words;   // the leaf maxima are in it too, at leafBase
RWStructuredBuffer<uint>   counts;   // [active voxels, leaf maxima off, leaves checked]
ConstantBuffer<CheckParams> check;

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeCountActive(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.leafCount) return;
    const NvLeaf leaf = nvLeaf(words, nvGrid(words, check.gridWord), i);
    uint active = 0;
    for (uint n = 0; n < 512; ++n) {
        active += nvLeafActive(words, leaf, n) ? 1u : 0u;
    }
    InterlockedAdd(counts[0], active);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeLeafMaxCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= check.leafCount) return;
    const NvGrid grid = nvGrid(words, check.gridWord);
    const int3 origin = nvLeafOrigin(words, nvLeaf(words, grid, i));
    // Again, the other way: every voxel a lookup in the leaf's cell reads,
    // [origin, origin + 8] on each axis, through the accessor, z outermost.
    float largest = 0.0;
    for (int z = 0; z <= 8; ++z) {
        for (int y = 0; y <= 8; ++y) {
            for (int x = 0; x <= 8; ++x) {
                largest = max(largest, nvScalar(nvValueAt(words, grid, origin + int3(x, y, z))));
            }
        }
    }
    InterlockedAdd(counts[2], 1u);
    if (largest != asfloat(words[check.leafBase + i])) {
        InterlockedAdd(counts[1], 1u);
    }
}

// --- Grid types: every voxel of a region about a box, read through the
// typed accessor, against the value the box was filled with (0 outside).

struct TypedParams {
    uint   gridWord;
    uint   type;       // the NanoVDB GridType code the header must hold
    int    lo;         // the region, [lo, hi)^3, in index space
    int    hi;
    float4 inside;     // the box's value: xyz, w components
    int3   boxMin;     // the box, [boxMin, boxMax)
    int    pad0;
    int3   boxMax;
    int    pad1;
};

ConstantBuffer<TypedParams> typed;

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeTypedCheck(uint3 tid: SV_DispatchThreadID) {
    const int side = typed.hi - typed.lo;
    if (tid.x >= uint(side * side * side)) return;
    const int3 ijk = int3(typed.lo + int(tid.x) % side, typed.lo + (int(tid.x) / side) % side,
                          typed.lo + int(tid.x) / (side * side));
    const NvGrid grid = nvGrid(words, typed.gridWord);
    const bool inBox = all(ijk >= typed.boxMin) && all(ijk < typed.boxMax);
    const float4 expected = inBox ? typed.inside : float4(0.0, 0.0, 0.0, typed.inside.w);
    const float4 v = nvValueAt(words, grid, ijk);
    InterlockedAdd(counts[2], 1u);
    if (any(v != expected)) InterlockedAdd(counts[0], 1u);
    if (tid.x == 0 && grid.type != typed.type) InterlockedAdd(counts[1], 1u);
}
