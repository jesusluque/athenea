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
    sortHi[s] = band;
    sortLo[s] = runStart[i];
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
    const uint band = sortHi[at];
    const uint page = bandPage[band] + (at - bandStart[band]) / max(params.maxRun, 1u);
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
