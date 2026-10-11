// Copyright (c) 2026 jesus luque.
//
// WHAT A CUT KEEPS RESIDENT, PAGED (`decimate --cut-pages`): the pages a cut
// through a merge tree needs, laid out by level and by error, counted on the
// device. Apart from lod_tree_cut, whose buffers fill what Metal binds to one
// program; the parameters are its own, bound by the same names.
module lod_pages;

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

static const uint kRoot = 0xFFFFFFFFu;

StructuredBuffer<uint>    storeOrder;   // store splat -> file gaussian
StructuredBuffer<uint>    parent;       // file gaussian -> its parent's file index (itself at a root)
StructuredBuffer<uint>    leaves;       // file gaussian -> how many leaves it stands for
RWStructuredBuffer<float> score;        // store splat -> 1 in the cut, 0 not
RWStructuredBuffer<uint>  errorBits;    // file gaussian -> eps, climbed
RWStructuredBuffer<uint>  radiusBits;   // file gaussian -> rho
RWStructuredBuffer<float4> centre;      // file gaussian -> its position
StructuredBuffer<uint>    runStart;     // file gaussian -> its first leaf (lodStart)
ConstantBuffer<CutParams> params;

float projectedError(uint i) {
    const float d = length(centre[i].xyz - params.eye.xyz) - asfloat(radiusBits[i]);
    return asfloat(errorBits[i]) * params.pixelsPerUnit / max(d, 1.0e-3);
}

// WHAT A CUT KEEPS RESIDENT, paged by level (`decimate --cut-pages`): page 0
// the top of the piece, always resident; a page is needed where the cut keeps
// a node in it or under it, and the gaussians those pages hold are counted.
StructuredBuffer<uint>   pageOf;       // file gaussian -> its page
StructuredBuffer<uint>   pageSize;     // page -> its gaussians
StructuredBuffer<uint>   pageParent;   // page -> the page its roots' parents are in (0 at the top)
RWStructuredBuffer<uint> touched;      // page -> 1 where the cut keeps a node of it
RWStructuredBuffer<uint> pageCounts;   // [pages touched, gaussians in them]
RWStructuredBuffer<uint> keptSizes;    // 32 bins of log2(leaves) of the nodes the cut keeps

[shader("compute")]
[numthreads(256, 1, 1)]
void treePagesTouched(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s < params.count && score[s] > 0.5) {
        InterlockedAdd(keptSizes[min(firstbithigh(max(leaves[storeOrder[s]], 1u)), 31u)], 1u);
        // Its page and every page above it: the walk down reads them.
        uint p = pageOf[storeOrder[s]];
        for (uint depth = 0; depth < 64u && touched[p] == 0u; ++depth) {
            touched[p] = 1u;
            if (p == 0u) {
                break;
            }
            p = pageParent[p];
        }
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treePagesCount(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p < params.pageCount && (touched[p] != 0u || p == 0u)) {
        InterlockedAdd(pageCounts[0], 1u);
        InterlockedAdd(pageCounts[1], pageSize[p]);
    }
}

// PAGES BY ERROR (Nanite's order): a node is needed exactly when its parent's
// projected error passes the threshold -- the cut and everything above it --
// so a page holds nodes of one band of their parent's error (an octave), near
// one another. A view needs, where it looks, the bands over its threshold.
RWStructuredBuffer<uint> errorPageSize;     // key -> its gaussians
RWStructuredBuffer<uint> errorPageNeeded;   // key -> 1 where a node of it is needed

uint bandOf(uint i) {
    const uint up = parent[i];
    const bool root = up == i || up == kRoot || up >= params.fileCount;
    const float e = root ? 1.0e30 : asfloat(errorBits[up]);
    return e > 0.0 ? uint(clamp(log2(e) + 40.0, 0.0, 63.0)) : 0u;
}

// Within a band, the nodes in leaf order (where they stand) are cut into
// pages of `maxRun` -- every page that size, near ones together.
RWStructuredBuffer<uint> sortHi;       // store splat -> its band (then sorted)
RWStructuredBuffer<uint> sortLo;       // its first leaf
RWStructuredBuffer<uint> sortValues;   // its file gaussian
RWStructuredBuffer<uint> bandCount;    // band -> its nodes
StructuredBuffer<uint>   bandStart;    // band -> its first place in the sorted order
StructuredBuffer<uint>   bandPage;     // band -> its first page

[shader("compute")]
[numthreads(256, 1, 1)]
void treeBandKeys(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint i = storeOrder[s];
    const uint band = bandOf(i);
    // The key, in the pages' own order: the coarsest band first, then the
    // first leaf (where it stands), then the larger first. A parent's band is
    // never finer than its child's (the errors are climbed), its first leaf
    // never later, and where both tie it is the larger -- so a parent never
    // lands after its child (sparkwebgpu-ce found 92 on the hood when a band
    // ran its leaves backwards).
    const uint start = min(runStart[i], 0x1FFFFFFu);
    sortHi[s] = ((63u - band) << 19) | (start >> 6);
    sortLo[s] = ((start & 63u) << 26) | (0x3FFFFFFu - min(leaves[i], 0x3FFFFFFu));
    sortValues[s] = i;
    InterlockedAdd(bandCount[band], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeBandPages(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint i = sortValues[at];
    const uint page = at / max(params.maxRun, 1u);   // as treeBandTable lays them
    if (page >= params.pageCount) {
        return;
    }
    InterlockedAdd(errorPageSize[page], 1u);
    const uint up = parent[i];
    const bool root = up == i || up == kRoot || up >= params.fileCount;
    if (root || projectedError(up) > params.threshold) {
        errorPageNeeded[page] = 1u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeErrorPagesCount(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k < params.pageCount && errorPageSize[k] > 0u) {
        InterlockedAdd(pageCounts[2], 1u);
        InterlockedMax(pageCounts[3], errorPageSize[k]);
        if (errorPageNeeded[k] != 0u) {
            InterlockedAdd(pageCounts[0], 1u);
            InterlockedAdd(pageCounts[1], errorPageSize[k]);
        }
    }
}

// THE PAGE TABLE A TREE CARRIES (`decimate --tree-pages`): each node's page,
// and for each page what a player tests it by without reading its nodes --
// its band, its nodes and the first of them in page order, the box that
// holds every node's sphere (so the distance to it is the nearest any node
// stands), and the largest and smallest error among its nodes' parents (it is
// needed where the largest, projected from the box, passes the threshold).
// A node reads only its ancestors, in coarser bands: a page depends on none
// finer than itself.
RWStructuredBuffer<uint>  filePage;     // file gaussian -> its page
RWStructuredBuffer<uint>  pageInfo;     // page -> band, nodes, first (3 words)
RWStructuredBuffer<uint>  pageBox;      // page -> lo xyz, hi xyz, ordered as uints (6 words)
RWStructuredBuffer<uint>  pageError;    // page -> largest, smallest parent error (2 words)
RWStructuredBuffer<float> pageBounds;   // page -> lo xyz, hi xyz, largest, smallest (8 floats)

uint ordered(float f) {
    const uint b = asuint(f);
    return (b & 0x80000000u) != 0u ? ~b : b | 0x80000000u;
}

float unordered(uint u) {
    return asfloat((u & 0x80000000u) != 0u ? u & 0x7FFFFFFFu : ~u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeBandTable(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint i = sortValues[at];
    const uint band = 63u - (sortHi[at] >> 19);
    // The whole order, coarsest band first, cut into pages of `maxRun`: a
    // view needs the bands over its threshold, a prefix of this order where
    // it looks, so only the page at the edge is part needed -- and no coarse
    // band of a few nodes is a page of its own.
    const uint page = at / max(params.maxRun, 1u);
    if (page >= params.pageCount) {
        return;
    }
    filePage[i] = page;
    InterlockedMax(pageInfo[page * 3], band);
    InterlockedAdd(pageInfo[page * 3 + 1], 1u);
    InterlockedMin(pageInfo[page * 3 + 2], at);
    // The box holds the PARENTS' spheres: a node is needed when its parent's
    // error, projected from the parent's sphere, passes the threshold
    // (sparkwebgpu-ce); a root's own.
    const uint up = parent[i];
    const bool root = up == i || up == kRoot || up >= params.fileCount;
    const uint from = root ? i : up;
    const float3 c = centre[from].xyz;
    const float r = asfloat(radiusBits[from]);
    [unroll] for (uint k = 0; k < 3; ++k) {
        InterlockedMin(pageBox[page * 6 + k], ordered(c[k] - r));
        InterlockedMax(pageBox[page * 6 + 3 + k], ordered(c[k] + r));
    }
    const uint e = root ? asuint(3.0e38) : errorBits[up];
    InterlockedMax(pageError[page * 2], e);
    InterlockedMin(pageError[page * 2 + 1], e);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeBandTableOut(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= params.pageCount) {
        return;
    }
    [unroll] for (uint k = 0; k < 6; ++k) {
        pageBounds[p * 8 + k] = unordered(pageBox[p * 6 + k]);
    }
    pageBounds[p * 8 + 6] = asfloat(pageError[p * 2]);
    pageBounds[p * 8 + 7] = asfloat(pageError[p * 2 + 1]);
}
