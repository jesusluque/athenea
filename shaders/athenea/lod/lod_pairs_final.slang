// Copyright (c) 2026 jesus luque.
//
// WHAT THE PAIRS LEFT, as a cloud (lod_pairs.slang does the merging): every
// store splat knows its cluster; the clusters are made runs of a permuted
// store (count, scan, scatter), so that a cluster is what a decimation's kept
// gaussian is -- a run of store splats and the records they came from -- and
// lod_attributes merges every attribute the cloud carries over it unchanged.
// Then each cluster's gaussian from its members' moments, all of them at once
// rather than pair by pair (the same, the sums being sums).
//
// A SURFEL WHERE THE MERGE IS FLAT (docs/decisions.md, surfels): a cluster
// whose members are discs, and whose own thinnest axis is under
// `flatTolerance` of its middle one, is written with that axis exactly 0, the
// disc it is -- drawn by the ray through each pixel (splat_blend's
// surfelPower), and by Spark as 2DGS. A cluster of one is the splat as it was.
module lod_pairs_final;
import athenea.common.packing;
import lod_blob;

public struct PairFinalParams {
    public uint  count;          // store splats, or clusters
    public uint  shWords;        // words of rest harmonics a splat
    public uint  keep;           // rest basis functions a colour
    public uint  surfels;        // 1: a flat merge becomes a surfel
    public uint  normals;        // 1: the store carries shading normals
    public float flatTolerance;  // thinnest over middle axis under which a flat merge is a disc
    public uint  treeLeaves;     // pairTreeLayout: the store splats, the tree's leaves
};

ConstantBuffer<PairFinalParams> params;
StructuredBuffer<uint>      clusterOf;     // per store splat
RWStructuredBuffer<uint>    counts;        // per cluster
RWStructuredBuffer<uint>    starts;        // per cluster: exclusive scan of counts (the tree: written by pairTreeLayout)
RWStructuredBuffer<uint>    filled;        // per cluster
RWStructuredBuffer<uint>    members;       // per permuted slot: the store splat
StructuredBuffer<float4>    storePositions;
StructuredBuffer<uint>      storeShape;
StructuredBuffer<uint>      storeSh;
StructuredBuffer<uint>      storeNormals;
StructuredBuffer<uint>      storeRecordOf;
RWStructuredBuffer<float4>  permPositions;
RWStructuredBuffer<uint>    permShape;
RWStructuredBuffer<uint>    permRecordOf;
RWStructuredBuffer<uint2>   ranges;        // per cluster
RWStructuredBuffer<float4>  outPositions;
RWStructuredBuffer<uint>    outShape;
RWStructuredBuffer<uint>    outSh;
RWStructuredBuffer<uint>    outNormals;
RWStructuredBuffer<uint>    surfelCount;   // [0]: clusters written as surfels
RWStructuredBuffer<float>   outCoverage;   // per cluster: W over the merged area, not held under 0.99

// THE MERGE TREE AS A CLOUD: the leaves first, in depth-first order (a leaf's
// place is its start), then every merge in the order it was made (tree node
// t at t). Each output is a run of the leaves -- one, or a merge's -- so the
// same gather and finalize make it, and every attribute merges over it.
StructuredBuffer<uint>      treeStart;
StructuredBuffer<uint>      treeSize;
StructuredBuffer<uint>      treeParent;
StructuredBuffer<float>     treeCost;
RWStructuredBuffer<uint>    outParent;     // per output: its parent's output (a root its own)
RWStructuredBuffer<float>   outCost;       // per output: the merge's cost (0 a leaf)

uint treeOut(uint node, uint leaves) {
    return node < leaves ? treeStart[node] : node;
}

/// `params.count`: the tree's nodes; `params.treeLeaves`: its leaves.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairTreeLayout(uint3 tid: SV_DispatchThreadID) {
    const uint node = tid.x;
    if (node >= params.count) {
        return;
    }
    const uint leaves = params.treeLeaves;
    const uint out = treeOut(node, leaves);
    if (node < leaves) {
        members[out] = node;
        starts[out] = out;
        counts[out] = 1u;
    } else {
        starts[out] = treeStart[node];
        counts[out] = treeSize[node];
    }
    outParent[out] = treeOut(treeParent[node], leaves);
    outCost[out] = treeCost[node];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pairCount(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    InterlockedAdd(counts[clusterOf[s]], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pairScatter(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint c = clusterOf[s];
    uint at;
    InterlockedAdd(filled[c], 1u, at);
    members[starts[c] + at] = s;
}

/// The store in cluster order, and the record of each, for lod_attributes.
[shader("compute")]
[numthreads(256, 1, 1)]
void pairGather(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= params.count) {
        return;
    }
    const uint s = members[p];
    permPositions[p] = storePositions[s];
    for (uint k = 0; k < 4; ++k) {
        permShape[p * 4 + k] = storeShape[s * 4 + k];
    }
    permRecordOf[p] = storeRecordOf[s];
}

/// Cyclic Jacobi on a symmetric 3x3 (lod_finalize's): `a` ends diagonal and
/// `v`'s columns are the eigenvectors.
void jacobi(inout float3x3 a, out float3x3 v) {
    v = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    for (int sweep = 0; sweep < 6; ++sweep) {
        for (int pq = 0; pq < 3; ++pq) {
            const int p = pq == 2 ? 1 : 0;
            const int q = pq == 0 ? 1 : 2;
            const float apq = a[p][q];
            if (abs(apq) < 1e-30) {
                continue;
            }
            const float theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const float t = sign(theta + 1e-30) / (abs(theta) + sqrt(theta * theta + 1.0));
            const float c = 1.0 / sqrt(t * t + 1.0);
            const float s = t * c;
            float3x3 j = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
            j[p][p] = c;
            j[q][q] = c;
            j[p][q] = s;
            j[q][p] = -s;
            a = mul(transpose(j), mul(a, j));
            v = mul(v, j);
        }
    }
}

/// ONE CLUSTER'S GAUSSIAN: its members' mass-weighted moments (mass =
/// opacity times the area of the two widest axes), the eigenvectors and
/// square roots of the covariance, the opacity that keeps the mass
/// (W over the merged area, held under 0.99), and the mean of the colours and
/// harmonics. One member: written as it was, bit for bit.
[shader("compute")]
[numthreads(64, 1, 1)]
void pairFinalize(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.count) {
        return;
    }
    const uint first = starts[c];
    const uint n = counts[c];
    ranges[c] = uint2(first, first + n);
    const uint row = c * 4;
    if (n == 1) {
        const uint s = members[first];
        outPositions[c] = storePositions[s];
        for (uint k = 0; k < 4; ++k) {
            outShape[row + k] = storeShape[s * 4 + k];
        }
        for (uint k = 0; k < params.shWords; ++k) {
            outSh[c * params.shWords + k] = storeSh[s * params.shWords + k];
        }
        if (params.normals != 0) {
            outNormals[c] = storeNormals[s];
        }
        outCoverage[c] = storePositions[s].w;
        return;
    }
    float w = 0.0;
    float3 s1 = float3(0.0);
    float3x3 s2 = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    float3 colour = float3(0.0);
    float3 normal = float3(0.0);
    bool discs = true;
    for (uint p = first; p < first + n; ++p) {
        const uint s = members[p];
        const Blob b = blobOf(storePositions[s], storeShape[s * 4], storeShape[s * 4 + 1], storeShape[s * 4 + 2],
                              storeShape[s * 4 + 3]);
        const float wi = max(b.opacity * discArea(b.scale), 1.0e-30);
        const float3 sq = b.scale * b.scale;
        float3x3 sigma = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
        for (uint k = 0; k < 3; ++k) {
            const float3 a = b.axes[k];   // lod_blob: its rows are the axes
            sigma += sq[k] * float3x3(a.x * a, a.y * a, a.z * a);
        }
        w += wi;
        s1 += wi * b.centre;
        s2 += wi * (sigma + float3x3(b.centre.x * b.centre, b.centre.y * b.centre, b.centre.z * b.centre));
        colour += wi * b.colour;
        if (params.normals != 0) {
            normal += wi * unpackNormal(storeNormals[s]);
        }
        const float thinnest = min(min(b.scale.x, b.scale.y), b.scale.z);
        const float widest = max(max(b.scale.x, b.scale.y), b.scale.z);
        const float middle = b.scale.x + b.scale.y + b.scale.z - thinnest - widest;
        discs = discs && thinnest <= params.flatTolerance * middle;
    }
    const float3 mu = s1 / w;
    float3x3 cov = s2 / w - float3x3(mu.x * mu, mu.y * mu, mu.z * mu);
    float3x3 axes;
    jacobi(cov, axes);
    if (determinant(axes) < 0.0) {
        axes[0][2] = -axes[0][2];
        axes[1][2] = -axes[1][2];
        axes[2][2] = -axes[2][2];
    }
    float3 scale = sqrt(max(float3(cov[0][0], cov[1][1], cov[2][2]), float3(1e-14)));
    const uint thin = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    const float widest = max(max(scale.x, scale.y), scale.z);
    const float middle = scale.x + scale.y + scale.z - scale[thin] - widest;
    const float area = discArea(scale);
    // THE COVERAGE, AND THE OPACITY IT IS DRAWN WITH. W / area is what the
    // merge covers; past one -- discs that overlapped, merged into one no
    // wider -- the opacity is held at 0.99, and a renderer that blends that
    // reopens the gaps the overlap had closed (a merged shell seen through
    // from afar). So the coverage is written as it is beside it
    // (`athenea:splat:coverage`), for a reader that draws by it.
    const float coverage = w / max(area, 1e-30);
    const float opacity = min(coverage, 0.99);
    outCoverage[c] = coverage;
    // A disc of discs stays one.
    const bool surfel = params.surfels != 0 && discs && scale[thin] <= params.flatTolerance * middle;
    if (surfel) {
        scale[thin] = 0.0;
        InterlockedAdd(surfelCount[0], 1u);
    }
    const float3 base = colour / w;
    outPositions[c] = float4(mu, opacity);
    const float3 ls = float3(scale.x > 0.0 ? log(scale.x) : -65000.0, scale.y > 0.0 ? log(scale.y) : -65000.0,
                             scale.z > 0.0 ? log(scale.z) : -65000.0);
    outShape[row] = encodeQuaternion(quaternionOfAxes(axes));
    outShape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    outShape[row + 2] = packHalves(halfSafe(ls.z), halfSafe(base.x));
    outShape[row + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));
    if (params.normals != 0) {
        const float3 axis = float3(axes[0][thin], axes[1][thin], axes[2][thin]);
        outNormals[c] = packNormal(dot(normal, normal) > 1.0e-24 * w * w ? normalize(normal) : axis);
    }
    // The rest harmonics: their mass-weighted mean, a half at a time.
    for (uint k = 0; k < params.shWords; ++k) {
        float lo = 0.0;
        float hi = 0.0;
        for (uint p = first; p < first + n; ++p) {
            const uint s = members[p];
            const float4 pos = storePositions[s];
            const float3 sc = exp(float3(lowHalf(storeShape[s * 4 + 1]), highHalf(storeShape[s * 4 + 1]),
                                         lowHalf(storeShape[s * 4 + 2])));
            const float wi = max(pos.w * discArea(sc), 1.0e-30);
            const uint word = storeSh[s * params.shWords + k];
            lo += wi * lowHalf(word);
            hi += wi * highHalf(word);
        }
        outSh[c * params.shWords + k] = packHalves(halfSafe(lo / w), halfSafe(hi / w));
    }
}
