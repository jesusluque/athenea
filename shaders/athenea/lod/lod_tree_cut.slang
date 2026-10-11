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
    float  relaxedShare;    ///< a merge past the guards passes only this far under the threshold
    float  colourWeight;    ///< w_c: the colour spread's weight in the error (research 100 §3.2; 0: none)
    float  normalWeight;    ///< w_n: the normal cone's
    uint   hasNormals;
    float  guard;           ///< a merge past the guards: its own error times this ...
    float  floorSize;       ///< ... and at least this many times its size
    uint   cullBack;        ///< 1: a node whose whole normal cone faces away is not drawn (a closed one-sided part)
    uint   maxRun;          ///< treeCoverage: the longest run of leaves composited
    uint   fileCount;       ///< the file's gaussians, which the tree's indices count: past `count` where the
                            ///< loader dropped some (under 1/255)
    uint   pageCount;       ///< treePagesCount: the tree's pages
};

StructuredBuffer<uint>    storeOrder;   // store splat -> file gaussian
StructuredBuffer<float4>  positions;    // the store's
StructuredBuffer<uint>    shape;        // the store's, four words a splat
StructuredBuffer<uint>    parent;       // file gaussian -> its parent's file index (itself at a root)
StructuredBuffer<uint>    leaves;       // file gaussian -> how many leaves it stands for
StructuredBuffer<float>   cost;         // file gaussian -> its merge's cost, negative past the guards (lod_pairs)
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

/// A merge past the guards -- across a colour edge, glass with paint, a
/// fold -- is what a far view needs and what closes a grille's holes near:
/// it passes only `relaxedShare` as wide as the rest.
bool passes(uint i) {
    if (leaves[i] <= 1u) {
        return true;
    }
    return width[i] <= params.threshold * (cost[i] < 0.0 ? params.relaxedShare : 1.0);
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
    for (uint depth = 0; keep && up != at && up != kRoot && up < params.fileCount && depth < params.maxDepth; ++depth) {
        keep = !passes(up);
        at = up;
        up = parent[up];
    }
    score[s] = keep ? 1.0 : 0.0;
    if (keep) {
        InterlockedAdd(counter[0], 1u);
    }
}

// A CUT BY ERROR, NOT BY SIZE (research 100, after Nanite and 3D Tiles):
// each node's error in metres, eps(n) = max(r(n) |cost(n)|, eps(children))
// -- r the node's size, sqrt of its covariance's trace, cost its merge's
// (lodCost, whatever its sign) -- held monotonic up the tree, and the sphere
// round its leaves, rho(n) = max over children of |mu(n) - mu(c)| + rho(c)
// (a leaf's is twice its widest size). The error projected at the sphere's
// nearest point, P(n) = eps ppu / max(d - rho, near), and a node is in the
// cut when P(n) <= threshold < P(parent): no walk up the ancestors, one cut.
RWStructuredBuffer<uint>   errorBits;   // file gaussian -> eps as a float's bits (positive: ordered as uints)
RWStructuredBuffer<uint>   radiusBits;  // file gaussian -> rho, the same
RWStructuredBuffer<float4> centre;      // file gaussian -> its position
RWStructuredBuffer<uint>   changed;     // [nodes raised this round]
RWStructuredBuffer<uint>   spreadBits;  // file gaussian -> s_c: how far a leaf's colour strays from the node's
RWStructuredBuffer<uint>   coneBits;    // file gaussian -> theta: how far a leaf's normal turns from the node's
RWStructuredBuffer<float4> look;        // file gaussian -> colour (rgb) and roughness
RWStructuredBuffer<float4> facing;      // file gaussian -> normal
StructuredBuffer<uint>     storeNormals;   // the store's, packed (packing's unpackNormal)
StructuredBuffer<float>    roughness;      // file gaussian -> its roughness

[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorOwn(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    const float4 p = positions[s];
    const Blob b = blobOf(p, shape[s * 4], shape[s * 4 + 1], shape[s * 4 + 2], shape[s * 4 + 3]);
    const float size = sqrt(dot(b.scale, b.scale));
    centre[i] = float4(p.xyz, 0.0);
    const bool leaf = leaves[i] <= 1u;
    errorBits[i] = asuint(leaf ? 0.0 : size * abs(cost[i]));
    radiusBits[i] = asuint(2.0 * max(b.scale.x, max(b.scale.y, b.scale.z)));
    look[i] = float4(b.colour, roughness[i]);
    facing[i] = float4(params.hasNormals != 0 ? unpackNormal(storeNormals[s]) : b.axes[2], 0.0);
    spreadBits[i] = 0u;
    coneBits[i] = 0u;
}

/// One level of the bounds' climb (research 100 §3.2): a parent's colour
/// spread and normal cone cover each child's, plus how far the child's own
/// colour and normal stand from the parent's -- bounds on every leaf under it,
/// by the triangle inequality, which the merge's own cost (the last pair's)
/// does not give: a chain of merges each within the edge drifts across it.
[shader("compute")]
[numthreads(256, 1, 1)]
void treeBoundsUp(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.fileCount) {
        return;
    }
    const uint up = parent[i];
    if (up == i || up == kRoot || up >= params.fileCount) {
        return;
    }
    uint was;
    const float spread = asfloat(spreadBits[i]) + length(look[i].xyz - look[up].xyz);
    InterlockedMax(spreadBits[up], asuint(spread), was);
    if (asuint(spread) > was) {
        InterlockedAdd(changed[0], 1u);
    }
    const float turn = asfloat(coneBits[i]) + acos(clamp(dot(facing[i].xyz, facing[up].xyz), -1.0, 1.0));
    InterlockedMax(coneBits[up], asuint(turn), was);
    if (asuint(turn) > was) {
        InterlockedAdd(changed[0], 1u);
    }
    const float reach = length(centre[i].xyz - centre[up].xyz) + asfloat(radiusBits[i]);
    InterlockedMax(radiusBits[up], asuint(reach), was);
    if (asuint(reach) > was) {
        InterlockedAdd(changed[0], 1u);
    }
}

/// A node's own error with its bounds, once they hold: r (|cost| + w_c s_c +
/// w_n theta^2 / alpha), and a merge past the guards (cost negative) times
/// `guard` and at least `floorSize` r -- its cost does not measure what it
/// breaks (a glass's moire).
[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorBounded(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    if (leaves[i] <= 1u) {
        errorBits[i] = 0u;
        return;
    }
    const Blob b = blobOf(positions[s], shape[s * 4], shape[s * 4 + 1], shape[s * 4 + 2], shape[s * 4 + 3]);
    const float size = sqrt(dot(b.scale, b.scale));
    const float alpha = max(look[i].w, 0.02) * max(look[i].w, 0.02);
    const float theta = asfloat(coneBits[i]);
    float e = size * (abs(cost[i]) + params.colourWeight * asfloat(spreadBits[i]) +
                      params.normalWeight * theta * theta / alpha);
    if (cost[i] < 0.0) {
        e = max(params.guard * e, params.floorSize * size);
    }
    errorBits[i] = asuint(e);
}

/// One level of the error's climb alone (the sphere already holds).
[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorClimb(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.fileCount) {
        return;
    }
    const uint up = parent[i];
    if (up == i || up == kRoot || up >= params.fileCount) {
        return;
    }
    uint was;
    InterlockedMax(errorBits[up], errorBits[i], was);
    if (errorBits[i] > was) {
        InterlockedAdd(changed[0], 1u);
    }
}

/// One level of the climb: each node raises its parent's error and sphere to
/// cover its own. Run until nothing changes.
[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorUp(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.fileCount) {
        return;
    }
    const uint up = parent[i];
    if (up == i || up == kRoot || up >= params.fileCount) {
        return;
    }
    uint was;
    InterlockedMax(errorBits[up], errorBits[i], was);
    if (errorBits[i] > was) {
        InterlockedAdd(changed[0], 1u);
    }
    const float reach = length(centre[i].xyz - centre[up].xyz) + asfloat(radiusBits[i]);
    InterlockedMax(radiusBits[up], asuint(reach), was);
    if (asuint(reach) > was) {
        InterlockedAdd(changed[0], 1u);
    }
}

float projectedError(uint i) {
    const float d = length(centre[i].xyz - params.eye.xyz) - asfloat(radiusBits[i]);
    return asfloat(errorBits[i]) * params.pixelsPerUnit / max(d, 1.0e-3);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorKeep(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    const uint up = parent[i];
    const bool root = up == i || up == kRoot || up >= params.fileCount;
    bool keep = projectedError(i) <= params.threshold && (root || projectedError(up) > params.threshold);
    // ITS BACK, NOT DRAWN (research, on the grille at 393 px): the faces of a
    // closed one-sided part that look away are hidden by its front, but at a
    // pixel or less the front does not cover and the back adds black. A node
    // whose whole cone of normals faces away -- n.v < -sin(theta) -- goes;
    // every leaf under it faces away too.
    if (keep && params.cullBack != 0) {
        const float3 v = normalize(params.eye.xyz - centre[i].xyz);
        const float theta = asfloat(coneBits[i]);
        if (theta < 1.5707963 && dot(facing[i].xyz, v) < -sin(theta)) {
            keep = false;
        }
    }
    score[s] = keep ? 1.0 : 0.0;
    if (keep) {
        InterlockedAdd(counter[0], 1u);
    }
}

// WHAT A KEPT MERGE REALLY COVERS (research, the grille at 393 px): mass over
// the moment ellipse counts the leaves' overlapping tails twice and paints the
// merge darker than its leaves drew. For each node the cut keeps, its leaves
// are projected on its plane and composited on an 8 x 8 grid over its +-1.25
// sigma as the raster would, 1 - prod(1 - alpha G); the node's opacity becomes
// that coverage fitted under its own footprint (least squares). Runs longer than `maxRun`
// keep mass over area (far views, where it shows least).
RWStructuredBuffer<uint>   storeOf;     // file gaussian -> its store splat
StructuredBuffer<uint>     runStart;    // file gaussian -> its first leaf's file index (lodStart)
RWStructuredBuffer<float4> outPositions;   // the store's positions, w the opacity, rewritten

[shader("compute")]
[numthreads(256, 1, 1)]
void treeStoreOf(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s < params.count) {
        storeOf[storeOrder[s]] = s;
    }
}

float3x3 covarianceOf(const Blob b) {
    // A Blob's axes are the ROWS of `axes` (lod_blob); read as columns they
    // were the inverse rotation's, and every leaf and merge of a coverage
    // fit stood turned the wrong way -- a fender's merges at 0.59 of their
    // mass where research's fit of the same leaves gave 1.00.
    const float3 s2 = b.scale * b.scale;
    float3x3 c = float3x3(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    [unroll] for (uint k = 0; k < 3; ++k) {
        const float3 a = b.axes[k];
        c += s2[k] * float3x3(a.x * a, a.y * a, a.z * a);
    }
    return c;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void treeCoverage(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count || !(score[s] > 0.5)) {
        return;
    }
    const uint i = storeOrder[s];
    const uint run = leaves[i];
    if (run <= 1u || run > params.maxRun) {
        return;
    }
    const Blob n = blobOf(positions[s], shape[s * 4], shape[s * 4 + 1], shape[s * 4 + 2], shape[s * 4 + 3]);
    // The node's plane: its two widest axes.
    uint k0 = 0, k1 = 1, k2 = 2;
    if (n.scale.x <= n.scale.y && n.scale.x <= n.scale.z) { k0 = 1; k1 = 2; k2 = 0; }
    else if (n.scale.y <= n.scale.z) { k0 = 0; k1 = 2; k2 = 1; }
    const float3 u = n.axes[k0];
    const float3 v = n.axes[k1];
    const float su = max(n.scale[k0], 1.0e-9), sv = max(n.scale[k1], 1.0e-9);
    static const uint kGrid = 8;
    // Over its core, +-1.25 sigma: past it the node's fringe is where its
    // neighbours carry a surface on, and a fit there left a panel's merges at
    // 0.84 -- the background through each one's middle, a fender mottled.
    static const float kCore = 1.25;
    float through[64];
    for (uint c = 0; c < 64; ++c) {
        through[c] = 1.0;
    }
    const uint first = runStart[i];
    for (uint l = 0; l < run; ++l) {
        const uint ls = storeOf[first + l];
        const Blob f = blobOf(positions[ls], shape[ls * 4], shape[ls * 4 + 1], shape[ls * 4 + 2], shape[ls * 4 + 3]);
        const float3x3 c3 = covarianceOf(f);
        // Its covariance on the node's plane, and its centre there.
        const float3 cu = mul(c3, u), cv = mul(c3, v);
        const float a = dot(u, cu) + 1.0e-12, bb = dot(u, cv), d = dot(v, cv) + 1.0e-12;
        const float det = max(a * d - bb * bb, 1.0e-30);
        const float2 m = float2(dot(f.centre - n.centre, u), dot(f.centre - n.centre, v));
        const float alpha = min(f.opacity, 0.99);
        for (uint y = 0; y < kGrid; ++y) {
            for (uint x = 0; x < kGrid; ++x) {
                const float2 p = float2((float(x) + 0.5) / kGrid * 2.0 * kCore - kCore,
                                        (float(y) + 0.5) / kGrid * 2.0 * kCore - kCore) * float2(su, sv) - m;
                const float q = (d * p.x * p.x - 2.0 * bb * p.x * p.y + a * p.y * p.y) / det;
                through[y * kGrid + x] *= 1.0 - alpha * exp(-0.5 * q);
            }
        }
    }
    float sum = 0.0, weight = 0.0;
    for (uint y = 0; y < kGrid; ++y) {
        for (uint x = 0; x < kGrid; ++x) {
            const float2 p = float2((float(x) + 0.5) / kGrid * 2.0 * kCore - kCore,
                                    (float(y) + 0.5) / kGrid * 2.0 * kCore - kCore);
            const float g = exp(-0.5 * dot(p, p));
            sum += g * (1.0 - through[y * kGrid + x]);
            weight += g * g;
        }
    }
    // The opacity a whose a G is nearest the leaves' composite C in the least
    // squares: sum G C / sum G^2 (over sum G it came out half: a merge as
    // wide as one opaque leaf at 0.5). Leaves edge on to its plane -- a
    // grille's wires seen square on -- cover nothing, and the merge goes.
    const float covered = sum / max(weight, 1.0e-12);
    float4 out = positions[s];
    out.w = isfinite(covered) ? clamp(covered, 0.0, 0.99) : out.w;
    outPositions[s] = out;
}

// THE COVERAGE BACK IN THE FILE'S ORDER (`decimate --tree-coverage`): a
// tree's lodParent, lodStart and lodSize are file indices, so the tree is
// written as it came and only its merges' opacities change; a leaf's stays
// what the file says (NaN here: not written).
RWStructuredBuffer<float> fileOpacity;   // file gaussian -> its merge's coverage
static const float kTreeFloor = 2.0 / 255.0;   // a loader keeps 1/255 and over (splat_validate)

[shader("compute")]
[numthreads(256, 1, 1)]
void treeCoverageToFile(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    fileOpacity[i] = leaves[i] > 1u ? max(outPositions[s].w, kTreeFloor) : asfloat(0x7FC00000u);
}

/// Over the file's gaussians (`count` is theirs here): a merge the loader
/// dropped -- under 1/255, a mass tree's empty merge -- written at the floor,
/// so it is loaded next time and the tree under it stays one tree.
[shader("compute")]
[numthreads(256, 1, 1)]
void treeCoverageFloor(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i < params.fileCount && leaves[i] > 1u && isnan(fileOpacity[i])) {
        fileOpacity[i] = kTreeFloor;
    }
}
