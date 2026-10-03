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
// One file a kernel (bvh_heights, bvh_level_keys, bvh_refit_level), since a
// name a module declares is bound whether its entry reads it or not.
module bvh_levels;
__exported import bvh_common;

public static const uint kMaxLevels = 128;

public struct LevelParams {
    public uint count;        // particles in the cloud
    public uint nodeBase;     // first internal node in the combined buffers
    public uint leafBase;     // first leaf in the combined buffers
    public uint levelStart;   // bvhRefitLevel: this height's first entry in levelNodes
    public uint levelCount;   // bvhRefitLevel: nodes of this height
    public uint pad0; public uint pad1; public uint pad2;
};
