// Copyright (c) 2026 jesus luque.
//
// FEWER GAUSSIANS BY PAIRS (docs/decisions.md, simplify): NanoGS (Xiong et
// al. 2026, arXiv 2603.16103) on the device, for a baked cloud that carries
// more than a colour.
//
// A pass: every node -- a store splat, or the merge of several -- asks its
// neighbours along the Morton order of the centres what a merge with each
// would cost, and keeps the cheapest (pairBest); two nodes that chose each
// other are a pair, and the cheapest pairs merge (pairMerge), at most half of
// the nodes a pass, until the count the ratio asks for.
//
// A node is moments, not a gaussian: its mass W (opacity times the area of
// its two widest axes, the levels of detail's own weight), W times its centre
// and W times its second moment, and W times what the cost compares. The
// moments of a merge are the sums of its children's, so the gaussian a node
// stands for is the moment matched one of everything under it, whatever the
// order of the merges -- and so is every attribute the cloud carries: each
// is merged once at the end from the records a node stands for
// (lod_attributes' lodMergeAttribute, mass weighted), which is what merging
// them pair by pair would have given.
//
// THE COST, for nodes i and j and the gaussian m their moments make:
//   - the shape: how much wider m is than what it replaces, scale free:
//     log I2(m) - (pi log I2(i) + (1 - pi) log I2(j)), I2 the sum of the
//     products of the covariance's eigenvalues by twos (the square of a
//     disc's area, read without the eigenvalues), pi = W_i / W. Runnalls'
//     bound on the KL divergence a merge loses, per unit of mass, with the
//     disc's area in place of the volume a surfel does not have.
//     Weighed by what the merge changes on the screen: nothing below half a
//     pixel, all of it from two (`pixelsPerUnit`, the conversion's camera). A
//     merge mass weighted alone takes the smallest gaussians first, and the
//     small details are what go.
//   - how thick m comes out against the thicker of the two: a surface's
//     gaussians are discs, and two discs on a curve merge into a lens.
//   - the colour, encoded as it is seen (sRGB), so a dark detail on a light
//     surface counts as it shows; past `edge` an edge, and no merge.
//   - the normal: 1 - cos, and for the highlight that moves with it
//     (1 - cos) / alpha^2 times what the surface reflects -- a mirror's
//     highlight moves with its normal, a rough surface's barely.
//   - metallic, roughness and transmission.
// No merge across a key (another prim, joint, sheet, thin wall, Schlick
// metal: lod_attributes' lodAttributeKey), between glass and what is not,
// between a mirror (alpha < 0.1) and what is not, or between normals more
// than `maxAngle` apart.
module lod_pairs;
import athenea.common.packing;
import athenea.common.morton;
import lod_blob;

public static const uint kNodeFloats = 24;
// 0 W | 1..3 W mu | 4..9 W (Sigma + mu mu^T): xx xy xz yy yz zz | 10..12 W srgb |
// 13..15 W normal | 16 W alpha^2 | 17 W metallic | 18 W transmission | 19 W specular |
// 20 key (bits) | 21 records | 22 W thickness ratio | 23 its node of the merge tree (bits)
public static const uint kWindow = 8;   // neighbours each side along a Morton order
public static const float kNoMerge = 1.0e30;

public struct PairParams {
    public uint  count;          // nodes (store splats on the first pass)
    public uint  shift;          // pairCodes: 0 or 1, which of two Morton orders
    public uint  keyed;          // 1: `keys` holds a key a record
    public uint  materialled;    // 1: `material` holds metallic, roughness, transmission a record
    public uint  normals;        // 1: the store carries shading normals
    public float threshold;      // pairMerge: the costliest pair merged this pass
    public float boundsLoX; public float boundsLoY; public float boundsLoZ;
    public float extent;
    public float boundsShiftX; public float boundsShiftY; public float boundsShiftZ;   // the second order's offset
    public float pixelsPerUnit;  // focal in pixels at distance 1; 0: no camera, every merge seen whole
    public float eyeX; public float eyeY; public float eyeZ;
    public float edge;           // colour distance (sRGB) that is an edge
    public float maxCos;         // normals less alike than this are not merged
    public float colourWeight;
    public float normalWeight;
    public float materialWeight;
    public float flatWeight;
    public float highlightCone;
    public uint  mergeGlass;
    public float maxAspect;      // a merge's plane longer than this against its width is refused     // 1: transmitting gaussians merge among themselves  // glossy normals further apart than this times sqrt(alpha) radians never merge
    public uint  tree;           // 1: record the merges as a tree
    public uint  leaves;         // the store's splats: the tree's first node after them
    public uint  pass;           // the pass a merge is made in; pairTreeOffsets: the pass to place
    public uint  histogramBins;
    public float histogramLo;    // log10 of the cheapest cost the histogram holds
    public float histogramStep;  // log10 a bin
};

ConstantBuffer<PairParams>  params;
StructuredBuffer<float4>    splatPositions;   // the store, Morton ordered
StructuredBuffer<uint>      splatShape;
StructuredBuffer<uint>      splatNormals;
StructuredBuffer<uint>      recordOf;         // per store splat: its record
StructuredBuffer<uint>      keys;             // per record
StructuredBuffer<float4>    material;         // per record
RWStructuredBuffer<float>   nodes;
RWStructuredBuffer<uint>    codes;            // pairCodes: the Morton code of each node's centre
RWStructuredBuffer<uint>    order;            // pairCodes: the node, to be sorted by code
StructuredBuffer<uint>      sorted;           // pairBest: nodes in Morton order
RWStructuredBuffer<uint>    best;             // per node: the partner chosen
RWStructuredBuffer<float>   bestCost;         // per node: what it costs
RWStructuredBuffer<uint>    histogram;        // pairHistogram: pairs a log10 cost bin
RWStructuredBuffer<uint>    alive;            // per node: 1 kept (absorbing its partner, or alone), 0 merged away
RWStructuredBuffer<uint>    absorbedBy;       // per node: the node it merged into, or itself
// THE MERGE TREE (`params.tree`): the leaves are the store splats (0..n-1),
// each merge a node after them, numbered as it is made.
RWStructuredBuffer<uint>    treeParent;       // per tree node: its parent, or itself for a root
RWStructuredBuffer<uint2>   treeChildren;     // per merge (tree node - n): its two children
RWStructuredBuffer<float>   treeCost;         // per tree node: what the merge that made it cost (0 a leaf)
RWStructuredBuffer<uint>    treePass;         // per merge: the pass it was made in
RWStructuredBuffer<uint>    treeMade;         // [0]: merges made so far

float srgbOf(float v) {
    v = max(v, 0.0);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

/// A STORE SPLAT AS A NODE: its moments, and what the cost compares.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairInit(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint row = s * 4;
    const Blob b = blobOf(splatPositions[s], splatShape[row], splatShape[row + 1], splatShape[row + 2], splatShape[row + 3]);
    const float w = max(b.opacity * discArea(b.scale), 1.0e-30);
    // Sigma = A diag(s^2) A^T, the axes A's columns.
    const float3 s2 = b.scale * b.scale;
    float3x3 sigma = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (uint k = 0; k < 3; ++k) {
        const float3 a = b.axes[k];   // lod_blob: its rows are the axes
        sigma += s2[k] * float3x3(a.x * a, a.y * a, a.z * a);
    }
    const float3 mu = b.centre;
    const uint r = recordOf[s];
    const float4 m = params.materialled != 0 ? material[r] : float4(0.0, 1.0, 0.0, 0.0);
    const float alpha = max(m.y * m.y, 1.0e-3);   // GGX alpha from roughness
    // The normal: the shading normal where the cloud keeps one, else the
    // gaussian's thinnest axis.
    const uint thin = b.scale.x <= b.scale.y ? (b.scale.x <= b.scale.z ? 0u : 2u) : (b.scale.y <= b.scale.z ? 1u : 2u);
    float3 n = b.axes[thin];
    if (params.normals != 0) {
        n = unpackNormal(splatNormals[s]);
    }
    const float3 sorted3 = float3(min(min(b.scale.x, b.scale.y), b.scale.z), 0.0, max(max(b.scale.x, b.scale.y), b.scale.z));
    const float middle = b.scale.x + b.scale.y + b.scale.z - sorted3.x - sorted3.z;
    // How much the surface reflects, for the highlight's term: a metal all of
    // it, a dielectric its Fresnel at normal incidence.
    const float specular = lerp(0.04, 1.0, saturate(m.x));
    const uint at = s * kNodeFloats;
    nodes[at] = w;
    nodes[at + 1] = w * mu.x;
    nodes[at + 2] = w * mu.y;
    nodes[at + 3] = w * mu.z;
    nodes[at + 4] = w * (sigma[0][0] + mu.x * mu.x);
    nodes[at + 5] = w * (sigma[0][1] + mu.x * mu.y);
    nodes[at + 6] = w * (sigma[0][2] + mu.x * mu.z);
    nodes[at + 7] = w * (sigma[1][1] + mu.y * mu.y);
    nodes[at + 8] = w * (sigma[1][2] + mu.y * mu.z);
    nodes[at + 9] = w * (sigma[2][2] + mu.z * mu.z);
    nodes[at + 10] = w * srgbOf(b.colour.x);
    nodes[at + 11] = w * srgbOf(b.colour.y);
    nodes[at + 12] = w * srgbOf(b.colour.z);
    nodes[at + 13] = w * n.x;
    nodes[at + 14] = w * n.y;
    nodes[at + 15] = w * n.z;
    nodes[at + 16] = w * alpha;
    nodes[at + 17] = w * m.x;
    nodes[at + 18] = w * m.z;
    nodes[at + 19] = w * specular;
    nodes[at + 20] = asfloat(params.keyed != 0 ? keys[r] : 0u);
    nodes[at + 21] = 1.0;
    nodes[at + 22] = w * sorted3.x / max(middle, 1.0e-20);
    nodes[at + 23] = asfloat(s);   // the tree's leaves are the store splats
}

/// What a node is, read from its moments.
struct Node {
    float    w;
    float3   mu;
    float3x3 sigma;
    float3   colour;
    float3   normalSum;
    float    alpha;
    float    metallic;
    float    transmission;
    float    specular;
    uint     key;
    float    thickness;
    float    raw[24];
};

Node nodeOf(uint i) {
    Node n;
    const uint at = i * kNodeFloats;
    for (uint k = 0; k < kNodeFloats; ++k) {
        n.raw[k] = nodes[at + k];
    }
    n.w = max(n.raw[0], 1.0e-30);
    const float inv = 1.0 / n.w;
    n.mu = float3(n.raw[1], n.raw[2], n.raw[3]) * inv;
    n.sigma = float3x3(n.raw[4], n.raw[5], n.raw[6], n.raw[5], n.raw[7], n.raw[8], n.raw[6], n.raw[8], n.raw[9]) * inv -
              float3x3(n.mu.x * n.mu, n.mu.y * n.mu, n.mu.z * n.mu);
    n.colour = float3(n.raw[10], n.raw[11], n.raw[12]) * inv;
    n.normalSum = float3(n.raw[13], n.raw[14], n.raw[15]) * inv;
    n.alpha = n.raw[16] * inv;
    n.metallic = n.raw[17] * inv;
    n.transmission = n.raw[18] * inv;
    n.specular = n.raw[19] * inv;
    n.key = asuint(n.raw[20]);
    n.thickness = n.raw[22] * inv;
    return n;
}

/// I2: the sum of the products of the eigenvalues by twos -- for a disc
/// (one eigenvalue nothing) the square of its area over pi^2, without the
/// eigenvalues.
float secondInvariant(float3x3 s) {
    return s[0][0] * s[1][1] + s[1][1] * s[2][2] + s[0][0] * s[2][2] - s[0][1] * s[0][1] - s[1][2] * s[1][2] -
           s[0][2] * s[0][2];
}

/// How long a node's plane is against its width: the square root of its two
/// wide eigenvalues' ratio.
float aspectOf(float3x3 s) {
    const float t = s[0][0] + s[1][1] + s[2][2];
    const float q = max(secondInvariant(s), 0.0);
    const float r = sqrt(max(t * t - 4.0 * q, 0.0));
    return sqrt((t + r) / max(t - r, 1.0e-30));
}

/// WHAT MERGING TWO NODES COSTS; kNoMerge where they may not merge.
float mergeCost(Node a, Node b) {
    if (a.key != b.key) {
        return kNoMerge;
    }
    if ((a.transmission > 0.5) != (b.transmission > 0.5)) {
        return kNoMerge;   // glass with what is not
    }
    // Glass with glass only when asked: a window is a sheet drawn as a layer
    // of the reflection it gives head on (splat_project's kSheetAlpha), and
    // the Corvette's windshield merged 10x came out a rainbow moire.
    if (params.mergeGlass == 0 && a.transmission > 0.5) {
        return kNoMerge;
    }
    if ((a.alpha < 0.01) != (b.alpha < 0.01)) {
        return kNoMerge;   // a mirror (roughness under 0.1) with what is not
    }
    const float la = length(a.normalSum);
    const float lb = length(b.normalSum);
    const float cosine = la > 1.0e-6 && lb > 1.0e-6 ? dot(a.normalSum, b.normalSum) / (la * lb) : 1.0;
    if (cosine < params.maxCos) {
        return kNoMerge;
    }
    // A HIGHLIGHT IS WHERE THE NORMAL POINTS IT. On a glossy surface (alpha
    // under 0.1) two normals further apart than the lobe is wide (an angle of
    // about sqrt(alpha)) put the highlight in two places, and their merge in
    // one between them: measured on the matx chrome ball merged 5x, every
    // merged disc caught a lamp or did not, a glitter over the whole sphere
    // (relMSE 0.59 against the full cloud, 0.02 at 2x). So they do not merge,
    // whatever the ratio: a mirror keeps the gaussians its highlight needs.
    const float gloss = min(a.alpha, b.alpha);
    if (gloss < 0.1 && 1.0 - cosine > 0.5 * params.highlightCone * params.highlightCone * gloss) {
        return kNoMerge;
    }
    const float3 dc = a.colour - b.colour;
    const float colour = dot(dc, dc);
    if (colour > params.edge * params.edge) {
        return kNoMerge;   // an edge in colour
    }
    const float w = a.w + b.w;
    const float pi = a.w / w;
    const float3 mu = pi * a.mu + (1.0 - pi) * b.mu;
    const float3 da = a.mu - mu;
    const float3 db = b.mu - mu;
    const float3x3 sigma = pi * (a.sigma + float3x3(da.x * da, da.y * da, da.z * da)) +
                           (1.0 - pi) * (b.sigma + float3x3(db.x * db, db.y * db, db.z * db));
    const float i2 = max(secondInvariant(sigma), 1.0e-36);
    const float ia = max(secondInvariant(a.sigma), 1.0e-36);
    const float ib = max(secondInvariant(b.sigma), 1.0e-36);
    const float shape = max(0.5 * (log(i2) - pi * log(ia) - (1.0 - pi) * log(ib)), 0.0);
    // How thick the merge is against its width: det / I2 is the smallest
    // eigenvalue, nearly, where it is small; I2 / trace the middle one.
    const float trace = sigma[0][0] + sigma[1][1] + sigma[2][2];
    const float det = max(determinant(sigma), 0.0);
    const float thick = sqrt(det * trace / (i2 * i2));
    const float flat = max(thick - max(a.thickness, b.thickness), 0.0);
    // NO NEEDLES. Merged along a strip -- a trim, a badge's edge, a crease --
    // a chain of discs moment-matches into a needle, and a needle widened
    // smears across what is beside it: the Corvette merged 10x streaked.
    // The aspect of the merge's plane (its two wide eigenvalues, from the
    // trace and I2 without solving: e = (t +- sqrt(t^2 - 4 I2)) / 2) may not
    // pass `maxAspect`, or its members' own if they were longer already.
    const float root = sqrt(max(trace * trace - 4.0 * i2, 0.0));
    const float aspect = sqrt((trace + root) / max(trace - root, 1.0e-30));
    if (aspect > max(params.maxAspect, 1.25 * max(aspectOf(a.sigma), aspectOf(b.sigma)))) {
        return kNoMerge;
    }
    // What the screen sees of it: the merged gaussian's width in pixels.
    float seen = 1.0;
    if (params.pixelsPerUnit > 0.0) {
        const float distance = max(length(mu - float3(params.eyeX, params.eyeY, params.eyeZ)), 1.0e-6);
        const float pixels = sqrt(max(trace, 0.0)) * params.pixelsPerUnit / distance;
        seen = smoothstep(0.5, 2.0, pixels);
    }
    const float alpha = 0.5 * (a.alpha + b.alpha);
    const float specular = 0.5 * (a.specular + b.specular);
    const float turn = 1.0 - cosine;
    const float normal = turn * (1.0 + specular / alpha);
    const float material = (a.metallic - b.metallic) * (a.metallic - b.metallic) +
                           (sqrt(a.alpha) - sqrt(b.alpha)) * (sqrt(a.alpha) - sqrt(b.alpha)) +
                           (a.transmission - b.transmission) * (a.transmission - b.transmission);
    return seen * (shape + params.flatWeight * flat * flat + params.normalWeight * normal) +
           params.colourWeight * colour + params.materialWeight * material;
}

/// The Morton code of a node's centre, in one of two orders: the second
/// shifted by half the cloud's finest cell, so that what one order puts far
/// apart across a cell's border the other puts side by side.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairCodes(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint at = i * kNodeFloats;
    const float w = max(nodes[at], 1.0e-30);
    float3 mu = float3(nodes[at + 1], nodes[at + 2], nodes[at + 3]) / w;
    if (params.shift != 0) {
        mu += float3(params.boundsShiftX, params.boundsShiftY, params.boundsShiftZ);
    }
    const float3 unit = saturate((mu - float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ)) /
                                 max(params.extent, 1.0e-30));
    codes[i] = morton30(unit);
    order[i] = i;
}

/// EACH NODE'S CHEAPEST NEIGHBOUR, `kWindow` either side along a Morton
/// order (`sorted`), against the one it already holds from the other order.
/// `first`: no choice yet.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairBest(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint i = sorted[at];
    const Node a = nodeOf(i);
    uint chosen = params.shift == 0 ? i : best[i];
    float cheapest = params.shift == 0 ? kNoMerge : bestCost[i];
    const uint lo = at >= kWindow ? at - kWindow : 0u;
    const uint hi = min(at + kWindow, params.count - 1u);
    for (uint k = lo; k <= hi; ++k) {
        if (k == at) {
            continue;
        }
        const uint j = sorted[k];
        const float c = mergeCost(a, nodeOf(j));
        // Ties go to the lower index, so two nodes see the same order.
        if (c < cheapest || (c == cheapest && c < kNoMerge && j < chosen)) {
            cheapest = c;
            chosen = j;
        }
    }
    best[i] = chosen;
    bestCost[i] = cheapest;
}

/// The costs of the pairs that chose each other, as a histogram of log10
/// cost: what the host reads to set the threshold that merges as many as
/// the pass may.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint j = best[i];
    if (j == i || j >= params.count || best[j] != i || i > j || !(bestCost[i] < kNoMerge)) {
        return;
    }
    const float l = log10(max(bestCost[i], 1.0e-30));
    const int bin = clamp(int(floor((l - params.histogramLo) / params.histogramStep)), 0, int(params.histogramBins) - 1);
    InterlockedAdd(histogram[bin], 1u);
}

/// THE PAIRS MERGE: the lower node takes the higher's moments, the higher is
/// gone. A node in no pair, or one past the threshold, stays as it is.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairMerge(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint j = best[i];
    const bool pair = j != i && j < params.count && best[j] == i && bestCost[i] <= params.threshold &&
                      bestCost[i] < kNoMerge;
    absorbedBy[i] = i;
    alive[i] = 1u;
    if (!pair) {
        return;
    }
    if (i > j) {
        absorbedBy[i] = j;
        alive[i] = 0u;
        return;
    }
    const uint a = i * kNodeFloats;
    const uint b = j * kNodeFloats;
    for (uint k = 0; k < kNodeFloats; ++k) {
        if (k == 20 || k == 23) {
            continue;   // the key (the same, or they were never a pair) and the tree's node
        }
        nodes[a + k] = nodes[a + k] + nodes[b + k];
    }
    if (params.tree != 0) {
        uint made;
        InterlockedAdd(treeMade[0], 1u, made);
        const uint node = params.leaves + made;
        const uint left = asuint(nodes[a + 23]);
        const uint right = asuint(nodes[b + 23]);
        treeParent[left] = node;
        treeParent[right] = node;
        treeParent[node] = node;
        treeChildren[made] = uint2(left, right);
        treeCost[node] = bestCost[i];
        treePass[made] = params.pass;
        nodes[a + 23] = asfloat(node);
    }
}

// THE TREE LAID OUT: its leaves in depth-first order, so that every node is a
// run of them -- the roots one after another (an exclusive scan of their
// sizes), and each merge's first child at its own start, the second after the
// first's leaves. A pass's merges are placed from the last pass back, their
// parents being later.
RWStructuredBuffer<uint>    treeStart;        // per tree node: its first leaf in depth-first order
StructuredBuffer<uint>      rootStart;        // per final node: the scan of the sizes (records, node float 21)

/// The final nodes' sizes, for the scan of the roots' starts (into `alive`).
[shader("compute")]
[numthreads(256, 1, 1)]
void pairRootSizes(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    alive[i] = uint(nodes[i * kNodeFloats + 21] + 0.5);
}

/// The final nodes, the roots: their starts.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairTreeRoots(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    treeStart[asuint(nodes[i * kNodeFloats + 23])] = rootStart[i];
}

/// One pass's merges: their children's starts. `params.count`: merges made.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairTreeOffsets(uint3 tid: SV_DispatchThreadID) {
    const uint made = tid.x;
    if (made >= params.count || treePass[made] != params.pass) {
        return;
    }
    const uint node = params.leaves + made;
    const uint2 children = treeChildren[made];
    const uint start = treeStart[node];
    treeStart[children.x] = start;
    treeStart[children.y] = start + treeSizeOf(children.x);
}

RWStructuredBuffer<uint>    treeSize;         // per tree node: its leaves

uint treeSizeOf(uint node) {
    return node < params.leaves ? 1u : treeSize[node];
}

/// Each merge's size, from its children's, a pass at a time from the first.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairTreeSizes(uint3 tid: SV_DispatchThreadID) {
    const uint made = tid.x;
    if (made >= params.count || treePass[made] != params.pass) {
        return;
    }
    const uint2 children = treeChildren[made];
    treeSize[params.leaves + made] = treeSizeOf(children.x) + treeSizeOf(children.y);
}

// THE PASS'S NODES COMPACTED, and every store splat told which it is now.
StructuredBuffer<uint>      slotOf;       // per node: its place among the kept (exclusive prefix sum of `alive`)
StructuredBuffer<float>     nodesIn;
RWStructuredBuffer<float>   nodesOut;
RWStructuredBuffer<uint>    clusterOf;    // per store splat: its node

[shader("compute")]
[numthreads(256, 1, 1)]
void pairCompact(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || alive[i] == 0u) {
        return;
    }
    const uint to = slotOf[i] * kNodeFloats;
    const uint from = i * kNodeFloats;
    for (uint k = 0; k < kNodeFloats; ++k) {
        nodesOut[to + k] = nodesIn[from + k];
    }
}

/// `params.count`: store splats here.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairRelabel(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    clusterOf[s] = slotOf[absorbedBy[clusterOf[s]]];
}
