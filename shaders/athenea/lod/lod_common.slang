// Copyright (c) 2026 jesus luque.
//
// LEVEL OF DETAIL, BUILT ON THE DEVICE
//
// Splats sorted by Morton code. The octree cell at level r holding a splat is
// its code's top 3r bits, so every cell's splats are one contiguous run of the
// sorted order: a level's cells ("groups") are where that prefix changes, found
// by a boundary pass and a prefix sum, never by walking a tree.
//
// Each group carries moments of the splats under it (Kerbl et al. 2024, "A
// Hierarchical 3D Gaussian Representation"): with w = opacity x the area of
// the splat's two longest axes,
//   W = sum w     S1 = sum w mu     S2 = sum w (Sigma + (mu - m)(mu - m)^T)     C = sum w colour
// with m = S1 / W the group's own mean: the second moment ABOUT THE GROUP, not
// about the world's origin. About the origin it was the difference of two
// numbers the size of mu mu^T, which a float holds to about 5e-7 m^2 two
// metres out -- the whole variance of a gaussian a millimetre wide, and a
// hundred times that of the needles a 5120 px bake lays on long thin
// triangles (0.9 x 0.04 mm): their merges came out as thin as the rounding
// made them, and a cut drew holes where they stood (docs/decisions.md).
// Moments add -- W, S1 and C by summing, S2 with each child's offset from the
// parent's mean (the parallel axis theorem, lod_merge_moments) -- so a level's
// groups are built from their children, from the finest merged level up, and
// each group becomes one Gaussian:
//   mu = S1 / W,  Sigma = S2 / W (eigen-decomposed into rotation and
//   scales),  colour = C / W,  opacity = min(W / its own two-axis area, 0.99).
//
// A group is drawn when its cell projects to at most `threshold` pixels and
// its parent's does not; a splat is drawn when the finest merged level's cell
// holding it does not. A child cell lies inside its parent, so its projected
// size (edge over nearest distance) is never larger: along any line down the
// tree the test flips once, and every place is drawn at exactly one level.
// Threshold 0 draws every splat and nothing merged: the cloud as it is.
//
// STREAMING
//
// The splats come in chunks -- runs of the Morton order -- that may or may not
// be on the device; the merged levels always are. A finest-level group whose
// cell wants its splats, but some of whose splats are in a chunk not loaded,
// is drawn merged instead (its own Gaussian: the nearest resident ancestor of
// those splats), and those chunks are asked for.
module lod_common;

public static const uint kLevels = 10;   // 30-bit codes

/// Moments per group: W, S1 (3), S2 (6: xx xy xz yy yz zz, about the group's
/// own mean), C base colour
/// (3), C rest (3 per basis kept, rgb per basis), and where the cloud keeps
/// shading normals N (3): the sum of w n, weighed as a colour is, whose
/// direction is the group's normal; and where it keeps an emission E (3):
/// the sum of w e, weighed as a colour is, whose mean is the group's.
public static const uint kMomentsHead = 13;

/// Where a group's emission moments start: after the harmonics', and after
/// the normals' where those are kept.
public uint emissionMoments(LodParams p) {
    return kMomentsHead + p.keep * 3 + (p.normals != 0 ? 3u : 0u);
}

public struct LodParams {
    public uint  count;        // splats, or groups at this level
    public uint  level;
    public uint  groups;       // groups at this level
    public uint  fineGroups;   // groups at the finer level this one is built from
    public uint  keep;         // rest basis functions per colour
    public uint  shWords;
    public uint  stride;       // moments per group
    public uint  base;         // gather: first output slot
    public float boundsLoX; public float boundsLoY; public float boundsLoZ;
    public float extent;      // the bounds' largest edge: cells are cubes
    // cut
    public float eyeX; public float eyeY; public float eyeZ;
    public float pixelsPerUnit;   // focal in pixels at distance 1 (perspective) or per unit (ortho), times the transform's scale
    public float threshold;       // pixels a drawn cell may span
    public uint  orthographic;
    public uint  coarsest;        // 1 when this is the coarsest stored level
    public uint  offset;          // chunks: the chunk's first splat within its buffers
    public uint  chunkSplats;     // splats per chunk
    public uint  chunks;
    public uint  splats;          // the whole cloud's splats
    /// 1: the cloud keeps shading normals (`GpuSplats::normals`), which
    /// follow the splats into Morton order and the cut, and whose weighted sum
    /// is three moments more after the harmonics', renormalised per group.
    public uint  normals;
    /// 1: the cloud keeps an emitted radiance (`GpuSplats::emission`), which
    /// follows the splats as the normals do, and whose weighted sum is three
    /// moments after theirs, divided by the weight per group.
    public uint  emission;
    /// 1: `classes` (sorted with the splats) splits the groups: what faces
    /// apart, and glass, are never one group (lod_class.slang).
    public uint  classes;
    /// 1: the cut reads each group's material (`pbr`): a group of glass or
    /// a sheet is never drawn merged, its splats are.
    public uint  pbrs;
    /// finalize: the most a merged Gaussian's opacity may be. Over 1 it is
    /// the harder kernel (splat/frame.slang's falloffAlpha), which keeps the
    /// mass of members that covered their cell more than once.
    public float mostOpacity;
};

/// A GLASS OR A SHEET, by its material word (splat_encoding's pbr): what
/// transmits more than it reflects, or a thin wall that transmits at all.
/// Kept out of merges: a merge of a pane is a lattice of holes.
public bool lodKeptApart(uint pbrWord) {
    const float transmission = float((pbrWord >> 16) & 0xFFu) / 255.0;
    const bool thin = ((pbrWord >> 24) & 1u) != 0u;
    return transmission > 0.5 || (thin && transmission > 0.0);
}

public uint levelShift(uint level) {
    return 3 * (kLevels - level);
}
