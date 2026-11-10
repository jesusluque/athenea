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
import athenea.common.morton;

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
    /// cut: the cut staggered. Each cell switches at its own threshold,
    /// `threshold * dissolve^(h - 1/2)` with h hashed from its code and level,
    /// so the edge between two levels is spread over a band of distances
    /// rather than sliding over the scene as one ring. 1 (or unset) is the
    /// cut as it was.
    public float dissolve;
    /// gather: the cut morphed (lod_gather). A gaussian whose parent's cell
    /// spans r times the parent's threshold (r > 1, or the parent would be
    /// drawn) is drawn a fraction t = (morph - r) / (morph - 1) of the way to
    /// its parent: at the switch it is the parent, so nothing pops. 0 (or
    /// anything up to 1) draws the gaussians as they are.
    public float morph;
    /// gather: where a gaussian's parent is. 0: none (the coarsest level, or
    /// the morph off); 1: a group, whose parent is the one of `parentGroups`
    /// at `level - 1` holding cell >> 3; 2: a splat, whose parent is its
    /// finest-level group (`groups`).
    public uint  morphParent;
    public uint  parentGroups;
    /// cut: 1 where a cell outside the view is not drawn (a budgeted cut,
    /// which counts what a frame shows): the view's four side planes in the
    /// cloud's space, n . x + d >= 0 inside, already widened by a margin.
    /// cut and gather: the splats thinned (lodThinFraction). While a
    /// finest-level group's cell spans under `thin` thresholds a fraction of
    /// its splats is drawn, down to a thirty-second where the group takes
    /// their place. Up to 1: every splat.
    public float  thin;
    /// cut, the finest level: 1 to write the splats each group draws into
    /// `splatCounts` (an appended gather places and counts them by it).
    public uint   countSplats;
    /// gather, appended: the most gaussians it may write past `base`, and
    /// the place in the cloud's Morton order of the run's first splat.
    public uint   appendLimit;
    public uint   mortonBase;
    public uint   frustumOn;
    public float4 frustumA;
    public float4 frustumB;
    public float4 frustumC;
    public float4 frustumD;
};

/// Whether a cell reaches into the view (LodParams::frustumOn): its box,
/// grown by an edge and a half for the gaussians that reach past it, against the
/// four side planes.
public bool lodVisible(LodParams p, uint cell, uint level) {
    if (p.frustumOn == 0) {
        return true;
    }
    const float edge = p.extent / float(1u << level);
    const float3 centre = float3(p.boundsLoX, p.boundsLoY, p.boundsLoZ) + (float3(cellIndex(cell)) + 0.5) * edge;
    const float reach = 2.0 * edge;   // half the edge, and an edge and a half more
    const float4 planes[4] = {p.frustumA, p.frustumB, p.frustumC, p.frustumD};
    for (uint k = 0; k < 4; ++k) {
        const float3 n = planes[k].xyz;
        if (dot(n, centre) + planes[k].w + (abs(n.x) + abs(n.y) + abs(n.z)) * reach < 0.0) {
            return false;
        }
    }
    return true;
}

/// The pixels a level-`level` cell spans at its nearest point to the eye.
public float lodProjectedEdge(LodParams p, uint cell, uint level) {
    const float edge = p.extent / float(1u << level);
    if (p.orthographic != 0) {
        return edge * p.pixelsPerUnit;
    }
    const float3 lo = float3(p.boundsLoX, p.boundsLoY, p.boundsLoZ) + float3(cellIndex(cell)) * edge;
    const float3 eye = float3(p.eyeX, p.eyeY, p.eyeZ);
    const float3 nearest = clamp(eye, lo, lo + edge);
    // Inside the cell: as near as it gets, so as big as it gets.
    const float distance = max(length(nearest - eye), 1e-6);
    return edge * p.pixelsPerUnit / distance;
}

/// A cell's own threshold (LodParams::dissolve). A group asks its parent with
/// the parent's own, so the cut stays a cut: nothing drawn twice or missed.
public float lodCellHash(uint cell, uint level) {
    uint v = cell * 747796405u + level * 2891336453u + 1u;
    v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    v = (v >> 22u) ^ v;
    return float(v) * (1.0 / 4294967296.0);
}

public float lodThresholdOf(LodParams p, uint cell, uint level) {
    const float k = max(p.dissolve, 1.0);
    return k <= 1.0 ? p.threshold : p.threshold * exp2(log2(k) * (lodCellHash(cell, level) - 0.5));
}

/// How far a gaussian is drawn toward its parent (LodParams::morph): 0 where
/// the parent's cell spans `morph` times its threshold or more, 1 at the
/// threshold, where the cut switches to the parent.
public float lodMorph(LodParams p, uint parentCell, uint parentLevel) {
    if (p.morph <= 1.0 || p.threshold <= 0.0) {
        return 0.0;
    }
    const float r = lodProjectedEdge(p, parentCell, parentLevel) / lodThresholdOf(p, parentCell, parentLevel);
    return saturate((p.morph - r) / (p.morph - 1.0));
}

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

/// THE SPLATS THINNED (LodParams::thin). The cut's last step is its largest:
/// one merged gaussian for the tens of splats of a finest-level group, and
/// with the eye moving the place where it happens shows as a line, and each
/// group's turn as a pop. Before it a group's splats are thinned: while its
/// cell spans r thresholds, r under `thin`, a fraction of them is drawn,
/// halving with each equal step of log r down to a thirty-second at the
/// switch (one or two of a group's tens). Those drawn take the mass of those
/// dropped: a little by width (the fourth root of the inverse fraction, at
/// most one and a half), the rest by opacity, up to 1 and never past it
/// (lod_gather): over 1 a gaussian is the harder kernel, drawn without what
/// a small one pays back, so a thinned splat whose opacity crossed 1 as the
/// fraction moved jumped, and those jumps were the flicker of the band.
///
/// Which go first is a splat's place in its group's run of the Morton order,
/// by the golden ratio (`lodThinTurn`): the fraction f of a group of n keeps
/// f n of them, one more or less, spread along the run -- the same every
/// frame, and a splat once dropped stays dropped farther away -- and the
/// run's first splat is never dropped, so no group is left with none.
public static const float kLodThinLeast = 1.0 / 32.0;

public float lodThinFraction(LodParams p, uint cell, uint level) {
    if (p.thin <= 1.0 || p.threshold <= 0.0) {
        return 1.0;
    }
    const float r = lodProjectedEdge(p, cell, level) / lodThresholdOf(p, cell, level);
    const float t = saturate(log(max(r, 1.0)) / log(p.thin));
    return t >= 1.0 ? 1.0 : exp2(log2(kLodThinLeast) * (1.0 - t));
}

/// How far toward its group's merged gaussian a thinned splat is drawn: the
/// band's last half, all the way at the switch.
public float lodThinToward(float fraction) {
    return fraction < 1.0 ? saturate(2.0 * log2(fraction) / log2(kLodThinLeast) - 1.0) : 0.0;
}

/// The turn of the splat at `morton` in the cloud's order, its group's run
/// beginning at `first`: under the fraction, it is drawn.
public float lodThinTurn(uint morton, uint first) {
    return float((morton - first) * 0x9E3779B9u) * (1.0 / 4294967296.0);
}

public bool lodThinKeeps(uint morton, uint first, float fraction) {
    return fraction >= 1.0 || lodThinTurn(morton, first) < fraction;
}

/// A kept splat does not go out at once: over the last `kLodThinFadeBand` of
/// the fraction before its turn it fades, so a band where the fraction moves
/// does not flicker with splats going on and off. What the fading ones lack
/// of the mass the rest make up (`kLodThinFadeMass`).
public static const float kLodThinFadeBand = 0.3;
public static const float kLodThinFadeMass = 1.0 / (1.0 - 0.5 * kLodThinFadeBand);

public float lodThinFade(uint morton, uint first, float fraction) {
    return fraction >= 1.0 ? 1.0 : saturate((fraction - lodThinTurn(morton, first)) / (kLodThinFadeBand * fraction));
}
