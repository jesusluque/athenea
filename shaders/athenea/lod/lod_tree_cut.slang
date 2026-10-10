// Copyright (c) 2026 jesus luque.
//
// A CUT THROUGH A MERGE TREE, for one camera (`athenea decimate --cut-budget`).
// `decimate --tree` writes every merge after the leaves, each with its
// parent (`lodParent`, itself at a root) and the run of leaves it stands for
// (`lodSize`, 1 a leaf). A node passes when it is a leaf or when it is no
// wider on the camera's screen than `threshold` pixels; the cut is every
// node that passes while none of its ancestors does -- so each leaf is drawn
// once, by itself or by the highest merge over it that is small enough on
// screen, whatever order the sizes come in down a branch. The host searches
// the threshold whose cut holds the budget. The tree's file order is not
// the store's (LodBuilder sorts the store by Morton code): `storeOrder`
// names the file gaussian each store splat is.
module lod_tree_cut;
import athenea.common.packing;
import lod_blob;

struct CutParams {
    uint   count;           ///< gaussians: leaves and merges
    float  threshold;       ///< pixels on the camera's screen a merge may be drawn at
    float  pixelsPerUnit;   ///< pixels a unit of length covers at a distance of one
    uint   maxDepth;        ///< how far up a branch is walked
    float4 eye;             ///< the camera's position
};

StructuredBuffer<uint>    storeOrder;   // store splat -> file gaussian
StructuredBuffer<float4>  positions;    // the store's
StructuredBuffer<uint>    shape;        // the store's, four words a splat
StructuredBuffer<uint>    parent;       // file gaussian -> its parent's file index (itself at a root)
StructuredBuffer<uint>    leaves;       // file gaussian -> how many leaves it stands for
RWStructuredBuffer<float> width;        // file gaussian -> its width on screen, pixels
RWStructuredBuffer<float> score;        // store splat -> 1 in the cut, 0 not
RWStructuredBuffer<uint>  counter;      // [in the cut]
ConstantBuffer<CutParams> params;

static const uint kRoot = 0xFFFFFFFFu;

/// Each gaussian's width on the camera's screen: twice its widest size over
/// its distance, in pixels.
[shader("compute")]
[numthreads(256, 1, 1)]
void treeCutWidth(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const float4 p = positions[s];
    const Blob b = blobOf(p, shape[s * 4], shape[s * 4 + 1], shape[s * 4 + 2], shape[s * 4 + 3]);
    const float distance = max(length(p.xyz - params.eye.xyz), 1.0e-6);
    width[storeOrder[s]] = 2.0 * max(b.scale.x, max(b.scale.y, b.scale.z)) / distance * params.pixelsPerUnit;
}

bool passes(uint i) {
    return leaves[i] <= 1u || width[i] <= params.threshold;
}

/// In the cut: it passes, and no ancestor does.
[shader("compute")]
[numthreads(256, 1, 1)]
void treeCutKeep(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    bool keep = passes(i);
    uint at = i;
    uint up = parent[i];
    // A root is its own parent (lod_pairs_final's treeOut); -1 is one too.
    for (uint depth = 0; keep && up != at && up != kRoot && up < params.count && depth < params.maxDepth; ++depth) {
        keep = !passes(up);
        at = up;
        up = parent[up];
    }
    score[s] = keep ? 1.0 : 0.0;
    if (keep) {
        InterlockedAdd(counter[0], 1u);
    }
}
