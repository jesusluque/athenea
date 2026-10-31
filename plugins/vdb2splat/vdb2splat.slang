// Copyright (c) 2026 jesus luque.
//
// A volume into volumetric gaussians (proposal 103, §3.3): the extinction
// field a sparse grid holds, fitted by a sum of gaussians mu_t(x) =
// sum_i sigma_i G_i(x), G_i of peak one -- the model whose transmittance has
// a closed form and does not care what order its factors come in.
//
// WHAT ARRIVES. Not the NanoVDB tree: the engine walks that and lays the
// leaves out as a picture (shaders/athenea/volume/vdb_splats.slang,
// vsPackLeaves), 513 entries a leaf -- its origin, then its 512 voxels in
// NanoVDB's order (x slowest, z fastest), each (density, temperature, Cd as
// rgba8 in a float's bits, active). So this effect needs no tree reader and
// a grid from anywhere is the same picture.
//
// WHAT LEAVES. Four entries a gaussian, in the grid's index space (a voxel
// ijk spans [ijk, ijk + 1)): (centre, sigma), (A00, A10, A11, A20), (A21,
// A22, temperature, Cd rgba8), (the half extents of its box at three
// deviations, 1). A is the lower Cholesky factor of the precision -- P = A
// A^T, G(x) = exp(-|A^T (x - mu)|^2 / 2) -- which is what an evaluation
// wants and what a covariance is not.
//
// HOW (the passes, in order; Vdb2Splat.cpp runs them):
//  1. Moments by block. Every leaf's 2^3, 4^3 and 8^3 blocks (sixty-four,
//     eight and one), then blocks of 16^3 up to `maxBlock` voxels, summed
//     from their eight children: mass, first and second moments, and the
//     temperature's count, sum and sum of squares. Raw moments add, which
//     is what lets a parent be its children's sum.
//  2. A block's error: what its moment gaussian alone leaves of the
//     density, sum (rho - rho^)^2 over its voxels, plus `emissionWeight`
//     times the temperature's spread in it (in density units). Decided by
//     error, not by density: a flame of thin smoke gains gaussians.
//  3. The cut: the blocks whose error is under tau and whose every
//     ancestor's is over it (a 2^3 block is kept whatever). The host
//     bisects tau on the count until the cut holds `initFraction` of the
//     budget.
//  4. Refinement (Adam) of every gaussian against the voxels, its gradient
//     gathered by the gaussian over the voxels in its box -- one group a
//     gaussian, no atomics on floats. The forward pass is a leaf's group
//     over the gaussians whose boxes meet the leaf. Every voxel of a leaf
//     counts, inactive ones at density zero.
//  5. Densify and prune, every `densifyEvery` iterations: a gaussian whose
//     mass is a negligible share goes; new ones are born at the voxels the
//     mixture leaves most under (the host bisects the residual threshold on
//     the count), into the free slots.
//  6. The attributes (temperature, Cd) last, with the geometry fixed: the
//     mixture's attribute at a voxel is sum c_i w_i / sum w_i, w_i = sigma_i
//     G_i, fitted by density-weighted least squares (linear in c), by Jacobi
//     steps.
//
// The order of the output is the slots', which no race decides.

struct V2sParams {
    uint  leafCount;
    uint  leafEntries;     // 513
    uint  capacity;        // the budget: slots in the state buffers
    uint  binned;          // 1: the forward pass walks each leaf's list (v2sBin*), 0: every slot

    int4  cellLo;          // the leaves' box in leaf cells (8 voxels): low corner
    int4  cellExtent;      // and its size

    uint  maxLevel;        // the coarsest block, 2^maxLevel voxels a side (3 to 7)
    uint  nodeCount;       // every block of the cut's search: 73 a leaf, then the coarse ones
    uint  level;           // the level a per-level pass is at
    uint  mode;            // what a shared pass does (see each)

    uint4 coarseBase;      // first coarse node of levels 4..7, in the coarse buffer
    uint4 coarseDims0;     // dims of level 4 (xyz), 5 in coarseDims1, ...
    uint4 coarseDims1;
    uint4 coarseDims2;
    uint4 coarseDims3;

    float tau;             // the cut's error threshold
    float emissionWeight;
    float threshold;       // densify: the residual a voxel must exceed
    float maxStd;          // the largest deviation a gaussian may take, voxels

    float minStd;
    float learningRate;    // this iteration's, decayed by the host's schedule
    uint  step;            // Adam's step, from 1
    float pruneMass;       // a gaussian below this share of the total mass goes

    uint  hasTemperature;
    uint  hasColour;
    uint  listBase;        // where in `list` a compaction writes
    uint  flagCount;       // how many flags a compaction reads

    uint  bornCount;       // how many voxels were chosen to be born at
    uint  freeBase;        // where in `list` the free slots are
    uint  outCount;        // gaussians written to the output
    uint  splitCount;      // gaussians split this round (v2sSplit)

    uint  warm;            // 1: the slots start from `previous` (a sequence's frame before), not from a cut
    uint  hasVelocity;     // 1: `velocity` holds the frame before's flow, voxels a frame: the slots are advected
    uint  keepSlots;       // 1: the output is every slot in its place, the dead ones zero (a sequence's)
    float dt;              // frames the advection covers
};

StructuredBuffer<float4>   leaves;     // 0: the grid, 513 entries a leaf
RWStructuredBuffer<float4> records;    // 1: evaluation records, 4 a slot; the output at the end
RWStructuredBuffer<uint>   counters;   // 2: see kCounter*
RWStructuredBuffer<uint>   leafMap;    // 3: leaf + 1 at each leaf cell of the box, 0 where none
RWStructuredBuffer<float4> nodes;      // 4: a leaf's 73 blocks, 4 entries each
RWStructuredBuffer<float4> coarse;     // 5: the blocks of 16^3 voxels and over, 4 entries each
RWStructuredBuffer<float>  partial;    // 6: per leaf: 4 coarse errors, then 4 loss sums
RWStructuredBuffer<uint>   flags;      // 7: one a node, a voxel or a slot: what a compaction selects
RWStructuredBuffer<float4> state;      // 8: a slot's parameters, 4 entries
RWStructuredBuffer<float4> adam;       // 9: a slot's moments, 8 entries (m: 4, v: 4)
RWStructuredBuffer<float4> grad;       // 10: a slot's gradient, 4 entries
RWStructuredBuffer<float4> model;      // 11: per voxel: (rho^, sum w T, sum w R, sum w G), (sum w B, 0, 0, 0)
RWStructuredBuffer<uint>   list;       // 12: compactions' output
RWStructuredBuffer<uint>   bins;       // 13: per leaf: its count, where its list starts, a cursor
RWStructuredBuffer<uint>   pairs;      // 14: the leaves' lists of slots, one after another
StructuredBuffer<float4>   previous;   // 15: the frame before's output, slot for slot (warm)
StructuredBuffer<float4>   velocity;   // 16: the flow at the leaves' voxels, as `leaves` is laid out
ConstantBuffer<V2sParams>  params;

static const uint kNodesPerLeaf = 73;
static const uint kCounterBounds = 0;    // 6: low xyz, high xyz, biased by 2^30
static const uint kCounterRhoMax = 6;    // float bits
static const uint kCounterTMax = 7;
static const uint kCounterCount = 8;     // a count a pass makes
static const uint kCounterErrMax = 9;    // float bits: the largest block error
static const uint kCounterTotal = 10;    // a compaction's total
static const uint kCounterSplitMax = 11; // float bits: the largest gaussian error (v2sSplitFlags)
static const uint kCounterStats = 12;    // 4 floats: sum r^2, sum rho^2, sum rho, relative L2
static const uint kBias = 1u << 30;
static const float kTwoPi32 = 15.7496099;   // (2 pi)^(3/2)
static const float kVarianceFloor = 0.01;
// A 3D gaussian stopped at 3.5 deviations keeps 0.99343 of its mass and
// 0.97489 of its variance (chi-square: P(chi2_3 < 12.25), and P(chi2_5 <
// 12.25) over it): what an EM step's moments, gathered inside the stop,
// are divided by so that a step does not shrink the gaussian it measures.
static const float kTruncMass = 0.993426;
static const float kTruncVariance = 0.974891;

// --- Small things ---------------------------------------------------------------

uint3 morton3(uint m) {
    uint3 c = uint3(0, 0, 0);
    for (uint k = 0; k < 3; ++k) {
        c.x |= ((m >> (3 * k + 2)) & 1u) << k;
        c.y |= ((m >> (3 * k + 1)) & 1u) << k;
        c.z |= ((m >> (3 * k + 0)) & 1u) << k;
    }
    return c;
}

uint voxelIndex(uint3 c) { return c.x * 64u + c.y * 8u + c.z; }

int3 leafOrigin(uint leaf) { return int3(leaves[leaf * params.leafEntries].xyz); }

float4 voxelOf(uint leaf, uint n) { return leaves[leaf * params.leafEntries + 1 + n]; }

/// Leaf at a leaf cell (box-relative), or -1.
int leafAtCell(int3 cell) {
    if (any(cell < 0) || any(cell >= params.cellExtent.xyz)) return -1;
    const uint at = (uint(cell.x) * uint(params.cellExtent.y) + uint(cell.y)) * uint(params.cellExtent.z) + uint(cell.z);
    return int(leafMap[at]) - 1;
}

int3 floorDiv8(int3 v) { return int3(floor(float3(v) / 8.0)); }

float3 unpackColour(float bits) {
    const uint u = asuint(bits);
    return float3(float(u & 255u), float((u >> 8) & 255u), float((u >> 16) & 255u)) / 255.0;
}

float packColour(float3 c) {
    const uint3 q = uint3(round(saturate(c) * 255.0));
    return asfloat(q.x | (q.y << 8) | (q.z << 16) | (255u << 24));
}

/// A gaussian as an evaluation wants it.
struct Gauss {
    float3 mu;
    float  sigma;
    float  a00, a10, a11, a20, a21, a22;
};

float quadForm(Gauss g, float3 x) {
    const float3 d = x - g.mu;
    const float u0 = g.a00 * d.x + g.a10 * d.y + g.a20 * d.z;
    const float u1 = g.a11 * d.y + g.a21 * d.z;
    const float u2 = g.a22 * d.z;
    return u0 * u0 + u1 * u1 + u2 * u2;
}

/// A gaussian stops at 3.5 deviations (G = 0.0022 there): the ellipsoid
/// that fits the box `kReach` deviations along each axis.
static const float kReach = 3.5;
static const float kReachQ = kReach * kReach;

float evalGauss(Gauss g, float3 x) {
    const float q = quadForm(g, x);
    return q < kReachQ ? g.sigma * exp(-0.5 * q) : 0.0;
}

/// The covariance's diagonal: Sigma = A^-T A^-1.
float3 stdOf(Gauss g) {
    const float b00 = 1.0 / g.a00, b11 = 1.0 / g.a11, b22 = 1.0 / g.a22;
    const float b10 = -g.a10 * b00 / g.a11;
    const float b21 = -g.a21 * b11 / g.a22;
    const float b20 = -(g.a20 * b00 + g.a21 * b10) / g.a22;
    return sqrt(float3(b00 * b00 + b10 * b10 + b20 * b20, b11 * b11 + b21 * b21, b22 * b22));
}

/// A gaussian from a block's raw moments (relative to `origin`, voxel
/// centres at +0.5): the mass kept, the covariance a voxel box's 1/12 wider.
Gauss gaussFromCovariance(float3 centre, float mass, float sxx, float sxy, float sxz, float syy, float syz, float szz);

Gauss gaussFromMoments(float4 n0, float4 n1, float4 n2, float3 origin, out bool valid) {
    const float m = n0.x;
    valid = m > 1e-20;
    const float inv = valid ? 1.0 / m : 0.0;
    const float3 mean = n0.yzw * inv;
    // Point samples at the voxel centres, which is what the mixture is
    // evaluated at: their spread is the gaussian's own, with a floor for a
    // block of one voxel.
    const float sxx = n1.x * inv - mean.x * mean.x + kVarianceFloor;
    const float sxy = n1.y * inv - mean.x * mean.y;
    const float sxz = n1.z * inv - mean.x * mean.z;
    const float syy = n1.w * inv - mean.y * mean.y + kVarianceFloor;
    const float syz = n2.x * inv - mean.y * mean.z;
    const float szz = n2.y * inv - mean.z * mean.z + kVarianceFloor;
    return gaussFromCovariance(origin + mean, m, sxx, sxy, sxz, syy, syz, szz);
}

/// A gaussian of `mass` (the integral of sigma G) at `centre` with this
/// covariance: its precision by the adjugate, then the precision's Cholesky
/// factor.
Gauss gaussFromCovariance(float3 centre, float mass, float sxx, float sxy, float sxz, float syy, float syz, float szz) {
    Gauss g;
    // The precision, by the adjugate, then its Cholesky factor.
    const float c00 = syy * szz - syz * syz;
    const float c01 = sxz * syz - sxy * szz;
    const float c02 = sxy * syz - sxz * syy;
    const float c11 = sxx * szz - sxz * sxz;
    const float c12 = sxy * sxz - sxx * syz;
    const float c22 = sxx * syy - sxy * sxy;
    const float det = max(sxx * c00 + sxy * c01 + sxz * c02, 1e-12);
    const float p00 = c00 / det, p01 = c01 / det, p02 = c02 / det;
    const float p11 = c11 / det, p12 = c12 / det, p22 = c22 / det;
    g.a00 = sqrt(max(p00, 1e-8));
    g.a10 = p01 / g.a00;
    g.a20 = p02 / g.a00;
    g.a11 = sqrt(max(p11 - g.a10 * g.a10, 1e-8));
    g.a21 = (p12 - g.a10 * g.a20) / g.a11;
    g.a22 = sqrt(max(p22 - g.a20 * g.a20 - g.a21 * g.a21, 1e-8));
    g.mu = centre;
    g.sigma = mass * g.a00 * g.a11 * g.a22 / kTwoPi32;   // integral sigma G = mass (a voxel's volume is one)
    return g;
}

/// Holds a factor to the sizes a gaussian may take: the diagonal between
/// 1 / maxStd and 1 / minStd, each off-diagonal within twice the geometric
/// mean of the two diagonals beside it (so the shear stays conditioned and
/// no deviation runs away). False where a value is not finite.
bool holdFactor(inout Gauss g) {
    const float lo = 1.0 / params.maxStd, hi = 1.0 / params.minStd;
    g.a00 = clamp(g.a00, lo, hi);
    g.a11 = clamp(g.a11, lo, hi);
    g.a22 = clamp(g.a22, lo, hi);
    const float b10 = 2.0 * sqrt(g.a00 * g.a11), b20 = 2.0 * sqrt(g.a00 * g.a22), b21 = 2.0 * sqrt(g.a11 * g.a22);
    g.a10 = clamp(g.a10, -b10, b10);
    g.a20 = clamp(g.a20, -b20, b20);
    g.a21 = clamp(g.a21, -b21, b21);
    const float sum = g.mu.x + g.mu.y + g.mu.z + g.sigma + g.a00 + g.a10 + g.a11 + g.a20 + g.a21 + g.a22;
    return !isnan(sum) && !isinf(sum);
}

Gauss gaussOfState(uint slot) {
    const float4 s0 = state[slot * 4 + 0];
    const float4 s1 = state[slot * 4 + 1];
    const float4 s2 = state[slot * 4 + 2];
    Gauss g;
    g.mu = s0.xyz;
    g.sigma = exp(s0.w);
    g.a00 = exp(s1.x);
    g.a10 = s1.y;
    g.a11 = exp(s1.z);
    g.a20 = s1.w;
    g.a21 = s2.x;
    g.a22 = exp(s2.y);
    return g;
}

bool aliveSlot(uint slot) { return slot < params.capacity && state[slot * 4 + 2].w > 0.5; }

void storeState(uint slot, Gauss g, float temperature, float3 colour) {
    state[slot * 4 + 0] = float4(g.mu, log(max(g.sigma, 1e-30)));
    state[slot * 4 + 1] = float4(log(g.a00), g.a10, log(g.a11), g.a20);
    state[slot * 4 + 2] = float4(g.a21, log(g.a22), temperature, 1.0);
    state[slot * 4 + 3] = float4(colour, 0.0);
    for (uint k = 0; k < 8; ++k) adam[slot * 8 + k] = float4(0.0);
}

Gauss gaussOfRecord(uint slot, out float3 halfExtent, out bool alive) {
    const float4 r0 = records[slot * 4 + 0];
    const float4 r1 = records[slot * 4 + 1];
    const float4 r2 = records[slot * 4 + 2];
    const float4 r3 = records[slot * 4 + 3];
    Gauss g;
    g.mu = r0.xyz;
    g.sigma = r0.w;
    g.a00 = r1.x; g.a10 = r1.y; g.a11 = r1.z; g.a20 = r1.w;
    g.a21 = r2.x; g.a22 = r2.y;
    halfExtent = r3.xyz;
    alive = r3.w > 0.5;
    return g;
}

// --- 0. Counters, bounds, the leaf map ---------------------------------------------

[shader("compute")]
[numthreads(1, 1, 1)]
void v2sClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    if (params.mode == 1) {
        counters[kCounterCount] = 0u;
        counters[kCounterSplitMax] = 0u;
        return;
    }
    for (uint k = 0; k < 3; ++k) {
        counters[kCounterBounds + k] = 0xFFFFFFFFu;
        counters[kCounterBounds + 3 + k] = 0u;
    }
    for (uint k = 6; k < 16; ++k) counters[k] = 0u;
}

/// One thread a leaf: the box in leaf cells, the largest density and temperature.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sBounds(uint3 tid: SV_DispatchThreadID) {
    const uint leaf = tid.x;
    if (leaf >= params.leafCount) return;
    const int3 cell = floorDiv8(leafOrigin(leaf));
    for (uint k = 0; k < 3; ++k) {
        InterlockedMin(counters[kCounterBounds + k], uint(cell[k] + int(kBias)));
        InterlockedMax(counters[kCounterBounds + 3 + k], uint(cell[k] + 1 + int(kBias)));
    }
    float rhoMax = 0.0, tMax = 0.0;
    for (uint n = 0; n < 512; ++n) {
        const float4 v = voxelOf(leaf, n);
        if (v.w > 0.5) {
            rhoMax = max(rhoMax, v.x);
            tMax = max(tMax, v.y);
        }
    }
    InterlockedMax(counters[kCounterRhoMax], asuint(max(rhoMax, 0.0)));
    InterlockedMax(counters[kCounterTMax], asuint(max(tMax, 0.0)));
}

/// One thread a cell of the box: cleared. `mode` 1: one thread a leaf, set.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sLeafMap(uint3 tid: SV_DispatchThreadID) {
    if (params.mode == 0) {
        const uint cells = uint(params.cellExtent.x * params.cellExtent.y * params.cellExtent.z);
        if (tid.x < cells) leafMap[tid.x] = 0u;
        return;
    }
    const uint leaf = tid.x;
    if (leaf >= params.leafCount) return;
    const int3 cell = floorDiv8(leafOrigin(leaf)) - params.cellLo.xyz;
    const uint at = (uint(cell.x) * uint(params.cellExtent.y) + uint(cell.y)) * uint(params.cellExtent.z) + uint(cell.z);
    leafMap[at] = leaf + 1;
}

// --- 1, 2. A leaf's blocks: moments and error -----------------------------------------

groupshared float gReduce[512];
groupshared float gNodeGauss[kNodesPerLeaf * 10];

/// Sums gReduce over runs of 8, 64 and 512 (the morton blocks of 2^3, 4^3,
/// 8^3) and hands each run's total to `sink` -- written by the threads
/// that hold the run's first entry.
void reduceLevels(uint t, uint leaf, uint slotInNode) {
    for (uint stride = 1; stride < 512; stride *= 2) {
        GroupMemoryBarrierWithGroupSync();
        for (uint h = 0; h < 2; ++h) {
            const uint i = t + h * 256;
            if ((i % (2 * stride)) == 0) gReduce[i] += gReduce[i + stride];
        }
        GroupMemoryBarrierWithGroupSync();
        const uint run = stride * 2;
        if (run == 8 || run == 64 || run == 512) {
            for (uint h = 0; h < 2; ++h) {
                const uint i = t + h * 256;
                if ((i % run) == 0) {
                    const uint node = run == 8 ? i / 8 : run == 64 ? 64 + i / 64 : 72;
                    const uint entry = (leaf * kNodesPerLeaf + node) * 4 + slotInNode / 4;
                    float4 e = nodes[entry];
                    e[slotInNode % 4] = gReduce[i];
                    nodes[entry] = e;
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

float nodeEmissionError(float4 n2, float4 n3) {
    const float count = n2.z;
    if (params.hasTemperature == 0 || count < 1.0) return 0.0;
    const float rhoMax = asfloat(counters[kCounterRhoMax]);
    const float tMax = max(asfloat(counters[kCounterTMax]), 1e-20);
    const float spread = max(n3.x - n2.w * n2.w / count, 0.0);
    return params.emissionWeight * spread * (rhoMax * rhoMax) / (tMax * tMax);
}

/// One group a leaf: its 73 blocks' moments, then each block's own error.
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sLeafNodes(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint leaf = gid.x;
    const uint t = gtid.x;
    if (leaf >= params.leafCount) return;
    float q[2][13];
    float rho[2];
    float3 pos[2];
    for (uint h = 0; h < 2; ++h) {
        const uint m = t + h * 256;
        const uint3 c = morton3(m);
        const float4 v = voxelOf(leaf, voxelIndex(c));
        const float r = v.w > 0.5 ? max(v.x, 0.0) : 0.0;
        const float a = v.w > 0.5 ? 1.0 : 0.0;
        const float3 x = float3(c) + 0.5;
        rho[h] = r;
        pos[h] = x;
        q[h][0] = r;
        q[h][1] = r * x.x; q[h][2] = r * x.y; q[h][3] = r * x.z;
        q[h][4] = r * x.x * x.x; q[h][5] = r * x.x * x.y; q[h][6] = r * x.x * x.z;
        q[h][7] = r * x.y * x.y; q[h][8] = r * x.y * x.z; q[h][9] = r * x.z * x.z;
        q[h][10] = a; q[h][11] = a * v.y; q[h][12] = a * v.y * v.y;
    }
    for (uint k = 0; k < 13; ++k) {
        gReduce[t] = q[0][k];
        gReduce[t + 256] = q[1][k];
        reduceLevels(t, leaf, k);
    }
    // Each block's gaussian, once.
    DeviceMemoryBarrierWithGroupSync();
    if (t < kNodesPerLeaf) {
        const uint e = (leaf * kNodesPerLeaf + t) * 4;
        bool valid;
        const Gauss g = gaussFromMoments(nodes[e], nodes[e + 1], nodes[e + 2], float3(0.0), valid);
        gNodeGauss[t * 10 + 0] = g.mu.x; gNodeGauss[t * 10 + 1] = g.mu.y; gNodeGauss[t * 10 + 2] = g.mu.z;
        gNodeGauss[t * 10 + 3] = valid ? g.sigma : 0.0;
        gNodeGauss[t * 10 + 4] = g.a00; gNodeGauss[t * 10 + 5] = g.a10; gNodeGauss[t * 10 + 6] = g.a11;
        gNodeGauss[t * 10 + 7] = g.a20; gNodeGauss[t * 10 + 8] = g.a21; gNodeGauss[t * 10 + 9] = g.a22;
    }
    GroupMemoryBarrierWithGroupSync();
    // The residual of each level's own block, reduced the same way, into slot 13.
    for (uint level = 0; level < 3; ++level) {
        for (uint h = 0; h < 2; ++h) {
            const uint m = t + h * 256;
            const uint node = level == 0 ? m / 8 : level == 1 ? 64 + m / 64 : 72;
            Gauss g;
            g.mu = float3(gNodeGauss[node * 10 + 0], gNodeGauss[node * 10 + 1], gNodeGauss[node * 10 + 2]);
            g.sigma = gNodeGauss[node * 10 + 3];
            g.a00 = gNodeGauss[node * 10 + 4]; g.a10 = gNodeGauss[node * 10 + 5]; g.a11 = gNodeGauss[node * 10 + 6];
            g.a20 = gNodeGauss[node * 10 + 7]; g.a21 = gNodeGauss[node * 10 + 8]; g.a22 = gNodeGauss[node * 10 + 9];
            const float d = rho[h] - evalGauss(g, pos[h]);
            gReduce[m] = d * d;
        }
        // Only the run of this level is wanted; reduceLevels writes all three,
        // so the levels below are written first and overwritten by nothing.
        for (uint stride = 1; stride < 512; stride *= 2) {
            GroupMemoryBarrierWithGroupSync();
            for (uint h = 0; h < 2; ++h) {
                const uint i = t + h * 256;
                if ((i % (2 * stride)) == 0) gReduce[i] += gReduce[i + stride];
            }
            GroupMemoryBarrierWithGroupSync();
            const uint run = stride * 2;
            const uint wanted = level == 0 ? 8u : level == 1 ? 64u : 512u;
            if (run == wanted) {
                for (uint h = 0; h < 2; ++h) {
                    const uint i = t + h * 256;
                    if ((i % run) == 0) {
                        const uint node = level == 0 ? i / 8 : level == 1 ? 64 + i / 64 : 72;
                        const uint e = (leaf * kNodesPerLeaf + node) * 4;
                        float4 n3 = nodes[e + 3];
                        n3.y = gReduce[i] + nodeEmissionError(nodes[e + 2], n3);
                        n3.z = 0.0;
                        n3.w = 0.0;
                        nodes[e + 3] = n3;
                        InterlockedMax(counters[kCounterErrMax], asuint(max(n3.y, 0.0)));
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// --- The coarse blocks: 16^3 voxels and over ------------------------------------------

uint3 coarseDims(uint level) {
    const uint k = level - 4;
    return k == 0 ? params.coarseDims0.xyz : k == 1 ? params.coarseDims1.xyz : k == 2 ? params.coarseDims2.xyz
                                                                                    : params.coarseDims3.xyz;
}

uint coarseIndex(uint level, uint3 c) {
    const uint3 d = coarseDims(level);
    return params.coarseBase[level - 4] + (c.x * d.y + c.y) * d.z + c.z;
}

/// A coarse block's voxel origin.
float3 coarseOrigin(uint level, uint3 c) {
    return float3(params.cellLo.xyz * 8 + int3(c) * int(1u << level));
}

/// Moments `n0..n2` relative to an origin `shift` below the new one's, as
/// relative to the new one: S1 + m o, S2 + o S1^T + S1 o^T + m o o^T.
void shiftMoments(inout float4 n0, inout float4 n1, inout float4 n2, float3 o) {
    const float m = n0.x;
    const float3 s1 = n0.yzw;
    n1.x += 2.0 * o.x * s1.x + m * o.x * o.x;
    n1.y += o.x * s1.y + o.y * s1.x + m * o.x * o.y;
    n1.z += o.x * s1.z + o.z * s1.x + m * o.x * o.z;
    n1.w += 2.0 * o.y * s1.y + m * o.y * o.y;
    n2.x += o.y * s1.z + o.z * s1.y + m * o.y * o.z;
    n2.y += 2.0 * o.z * s1.z + m * o.z * o.z;
    n0.yzw = s1 + m * o;
}

/// One thread a coarse block of `level`: its children's moments summed.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sCoarseMoments(uint3 tid: SV_DispatchThreadID) {
    const uint level = params.level;
    const uint3 d = coarseDims(level);
    const uint i = tid.x;
    if (i >= d.x * d.y * d.z) return;
    const uint3 c = uint3(i / (d.y * d.z), (i / d.z) % d.y, i % d.z);
    const float3 origin = coarseOrigin(level, c);
    float4 s0 = float4(0.0), s1 = float4(0.0), s2 = float4(0.0), s3 = float4(0.0);
    for (uint k = 0; k < 8; ++k) {
        const uint3 child = c * 2 + uint3((k >> 2) & 1, (k >> 1) & 1, k & 1);
        float4 n0, n1, n2, n3;
        float3 childOrigin;
        if (level == 4) {
            const int leaf = leafAtCell(int3(child));
            if (leaf < 0) continue;
            const uint e = (uint(leaf) * kNodesPerLeaf + 72) * 4;
            n0 = nodes[e]; n1 = nodes[e + 1]; n2 = nodes[e + 2]; n3 = nodes[e + 3];
            childOrigin = float3(leafOrigin(uint(leaf)));
        } else {
            const uint3 cd = coarseDims(level - 1);
            if (any(child >= cd)) continue;
            const uint e = coarseIndex(level - 1, child) * 4;
            n0 = coarse[e]; n1 = coarse[e + 1]; n2 = coarse[e + 2]; n3 = coarse[e + 3];
            childOrigin = coarseOrigin(level - 1, child);
        }
        shiftMoments(n0, n1, n2, childOrigin - origin);
        s0 += n0; s1 += n1; s2.xyzw += n2; s3.x += n3.x;
    }
    const uint e = coarseIndex(level, c) * 4;
    coarse[e] = s0; coarse[e + 1] = s1; coarse[e + 2] = s2; coarse[e + 3] = float4(s3.x, 0.0, 0.0, 0.0);
}

/// One group a leaf: what each coarse ancestor's gaussian alone leaves of
/// the leaf's density, into `partial` (4 a leaf, levels 4..7).
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sCoarseError(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint leaf = gid.x;
    const uint t = gtid.x;
    if (leaf >= params.leafCount) return;
    const int3 origin = leafOrigin(leaf);
    const int3 cell = floorDiv8(origin) - params.cellLo.xyz;
    for (uint level = 4; level <= params.maxLevel; ++level) {
        const uint3 c = uint3(cell) >> (level - 3);
        const uint e = coarseIndex(level, c) * 4;
        bool valid;
        const Gauss g = gaussFromMoments(coarse[e], coarse[e + 1], coarse[e + 2], coarseOrigin(level, c), valid);
        for (uint h = 0; h < 2; ++h) {
            const uint m = t + h * 256;
            const uint3 v = morton3(m);
            const float4 vox = voxelOf(leaf, voxelIndex(v));
            const float rho = vox.w > 0.5 ? max(vox.x, 0.0) : 0.0;
            const float d = rho - (valid ? evalGauss(g, float3(origin) + float3(v) + 0.5) : 0.0);
            gReduce[m] = d * d;
        }
        for (uint stride = 1; stride < 512; stride *= 2) {
            GroupMemoryBarrierWithGroupSync();
            for (uint h = 0; h < 2; ++h) {
                const uint i = t + h * 256;
                if ((i % (2 * stride)) == 0) gReduce[i] += gReduce[i + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) partial[leaf * 8 + (level - 4)] = gReduce[0];
        GroupMemoryBarrierWithGroupSync();
    }
}

/// One thread a coarse block of `level`: its leaves' errors summed, and the
/// temperature's spread added.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sCoarseErrorGather(uint3 tid: SV_DispatchThreadID) {
    const uint level = params.level;
    const uint3 d = coarseDims(level);
    const uint i = tid.x;
    if (i >= d.x * d.y * d.z) return;
    const uint3 c = uint3(i / (d.y * d.z), (i / d.z) % d.y, i % d.z);
    const uint span = 1u << (level - 3);
    float sum = 0.0;
    for (uint x = 0; x < span; ++x) {
        for (uint y = 0; y < span; ++y) {
            for (uint z = 0; z < span; ++z) {
                const int leaf = leafAtCell(int3(c * span + uint3(x, y, z)));
                if (leaf >= 0) sum += partial[uint(leaf) * 8 + (level - 4)];
            }
        }
    }
    const uint e = coarseIndex(level, c) * 4;
    float4 n3 = coarse[e + 3];
    n3.y = sum + nodeEmissionError(coarse[e + 2], n3);
    coarse[e + 3] = n3;
    if (coarse[e].x > 0.0) InterlockedMax(counters[kCounterErrMax], asuint(max(n3.y, 0.0)));
}

// --- 3. The cut ---------------------------------------------------------------------

/// A node of the search, by its flat number: level, and its four entries.
struct NodeRef {
    uint   level;      // 1..maxLevel
    uint   entry;      // first of its 4 entries
    bool   isCoarse;
    uint   leaf;       // for a leaf's own blocks
    uint   local;      // 0..72
    uint3  coarseCell;
    float3 origin;     // voxel origin its moments are relative to
};

NodeRef nodeRef(uint i) {
    NodeRef r;
    const uint leafNodes = params.leafCount * kNodesPerLeaf;
    if (i < leafNodes) {
        r.isCoarse = false;
        r.leaf = i / kNodesPerLeaf;
        r.local = i % kNodesPerLeaf;
        r.level = r.local < 64 ? 1 : r.local < 72 ? 2 : 3;
        r.entry = i * 4;
        r.origin = float3(leafOrigin(r.leaf));
        r.coarseCell = uint3(0, 0, 0);
        return r;
    }
    const uint j = i - leafNodes;
    r.isCoarse = true;
    r.leaf = 0;
    r.local = 0;
    r.level = params.maxLevel;
    for (uint level = 4; level <= params.maxLevel; ++level) {
        const uint3 d = coarseDims(level);
        const uint base = params.coarseBase[level - 4];
        if (j >= base && j < base + d.x * d.y * d.z) {
            r.level = level;
            const uint k = j - base;
            r.coarseCell = uint3(k / (d.y * d.z), (k / d.z) % d.y, k % d.z);
        }
    }
    r.entry = j * 4;
    r.origin = coarseOrigin(r.level, r.coarseCell);
    return r;
}

float4 nodeEntry(NodeRef r, uint k) { return r.isCoarse ? coarse[r.entry + k] : nodes[r.entry + k]; }

/// The error of the ancestor of node `r` at `level` (> r.level).
float ancestorError(NodeRef r, uint level) {
    if (!r.isCoarse) {
        if (level == 2) return nodes[(r.leaf * kNodesPerLeaf + 64 + r.local / 8) * 4 + 3].y;
        if (level == 3) return nodes[(r.leaf * kNodesPerLeaf + 72) * 4 + 3].y;
        const int3 cell = floorDiv8(leafOrigin(r.leaf)) - params.cellLo.xyz;
        const uint3 c = uint3(cell) >> (level - 3);
        return coarse[coarseIndex(level, c) * 4 + 3].y;
    }
    const uint3 c = r.coarseCell >> (level - r.level);
    return coarse[coarseIndex(level, c) * 4 + 3].y;
}

bool inCut(uint i) {
    const NodeRef r = nodeRef(i);
    const float4 n0 = nodeEntry(r, 0);
    if (!(n0.x > 0.0)) return false;
    const float err = nodeEntry(r, 3).y;
    if (err > params.tau && r.level > 1) return false;
    for (uint level = r.level + 1; level <= max(params.maxLevel, 3u); ++level) {
        if (ancestorError(r, level) <= params.tau) return false;
    }
    return true;
}

/// One thread a node: whether it is in the cut at `tau`, counted, and
/// flagged for the compaction.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sCutFlags(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.nodeCount) return;
    const bool kept = inCut(i);
    flags[i] = kept ? 1u : 0u;
    if (kept) InterlockedAdd(counters[kCounterCount], 1u);
}

// --- Compaction: the flagged, in order ------------------------------------------------

groupshared uint gScan[256];

/// One group: every flagged index of `flags[0, flagCount)`, in order, into
/// `list` from `listBase`, the total into the counters.
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sCompact(uint3 gtid: SV_GroupThreadID) {
    const uint t = gtid.x;
    const uint total = params.flagCount;
    const uint slice = (total + 255) / 256;
    const uint from = min(t * slice, total);
    const uint to = min(from + slice, total);
    uint sum = 0;
    for (uint i = from; i < to; ++i) sum += flags[i];
    gScan[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        uint running = 0;
        for (uint k = 0; k < 256; ++k) {
            const uint here = gScan[k];
            gScan[k] = running;
            running += here;
        }
        counters[kCounterTotal] = running;
    }
    GroupMemoryBarrierWithGroupSync();
    uint at = gScan[t];
    for (uint i = from; i < to; ++i) {
        if (flags[i] != 0) {
            list[params.listBase + at] = i;
            ++at;
        }
    }
}

/// One thread a node of the cut: its gaussian into slot k.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sInitEmit(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.bornCount) return;
    const NodeRef r = nodeRef(list[params.listBase + k]);
    bool valid;
    Gauss g = gaussFromMoments(nodeEntry(r, 0), nodeEntry(r, 1), nodeEntry(r, 2), r.origin, valid);
    const float4 n2 = nodeEntry(r, 2);
    const float temperature = n2.z > 0.0 ? n2.w / n2.z : 0.0;
    // Held to the deviations a gaussian may take.
    holdFactor(g);
    storeState(k, g, temperature, float3(1.0));
}

// --- 4. Refinement --------------------------------------------------------------------

/// One thread a slot: emptied, before the cut is laid down.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sReset(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    for (uint k = 0; k < 4; ++k) state[slot * 4 + k] = float4(0.0);
}

/// One thread a slot: the record an evaluation reads (and, at the end, the output).
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sPrepare(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    if (!aliveSlot(slot)) {
        records[slot * 4 + 3] = float4(0.0);
        records[slot * 4 + 0] = float4(0.0);
        return;
    }
    const Gauss g = gaussOfState(slot);
    const float4 s2 = state[slot * 4 + 2];
    const float4 s3 = state[slot * 4 + 3];
    const float3 extent = kReach * stdOf(g);
    const float sum = g.mu.x + g.mu.y + g.mu.z + g.sigma + extent.x + extent.y + extent.z;
    if (isnan(sum) || isinf(sum)) {
        // Not a gaussian any more: freed, never evaluated.
        float4 dead = s2;
        dead.w = 0.0;
        state[slot * 4 + 2] = dead;
        records[slot * 4 + 3] = float4(0.0);
        records[slot * 4 + 0] = float4(0.0);
        return;
    }
    records[slot * 4 + 0] = float4(g.mu, g.sigma);
    records[slot * 4 + 1] = float4(g.a00, g.a10, g.a11, g.a20);
    records[slot * 4 + 2] = float4(g.a21, g.a22, s2.z, packColour(s3.xyz));
    records[slot * 4 + 3] = float4(min(extent, float3(kReach * params.maxStd * 2.0)), 1.0);
}

groupshared float gChunk[256 * 16];
groupshared uint  gHit[256];

/// One group a leaf: the mixture at each of its voxels, from the gaussians
/// whose boxes meet it, and the attributes' weighted sums. `mode` 1 also
/// sums the loss into `partial` (from entry 4 of the leaf's 8).
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sForward(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint leaf = gid.x;
    const uint t = gtid.x;
    if (leaf >= params.leafCount) return;
    const float3 lo = float3(leafOrigin(leaf));
    const float3 hi = lo + 8.0;
    float3 x[2];
    float acc[2][5];
    for (uint h = 0; h < 2; ++h) {
        x[h] = lo + float3(morton3(t + h * 256)) + 0.5;
        for (uint k = 0; k < 5; ++k) acc[h][k] = 0.0;
    }
    // Binned, the leaf's own list (every gaussian whose box meets it, in
    // slot order); else every slot, culled by its box.
    const uint listed = params.binned != 0 ? bins[leaf] : params.capacity;
    const uint first = params.binned != 0 ? bins[params.leafCount + leaf] : 0u;
    for (uint base = 0; base < listed; base += 256) {
        const uint slot = base + t < listed ? (params.binned != 0 ? pairs[first + base + t] : base + t)
                                            : params.capacity;
        bool hit = false;
        if (slot < params.capacity) {
            const float4 r0 = records[slot * 4 + 0];
            const float4 r3 = records[slot * 4 + 3];
            hit = r3.w > 0.5 && all(r0.xyz + r3.xyz > lo) && all(r0.xyz - r3.xyz < hi);
            if (hit) {
                for (uint k = 0; k < 4; ++k) {
                    const float4 r = records[slot * 4 + k];
                    gChunk[t * 16 + k * 4 + 0] = r.x;
                    gChunk[t * 16 + k * 4 + 1] = r.y;
                    gChunk[t * 16 + k * 4 + 2] = r.z;
                    gChunk[t * 16 + k * 4 + 3] = r.w;
                }
            }
        }
        gHit[t] = hit ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint j = 0; j < 256; ++j) {
            if (gHit[j] == 0) continue;
            Gauss g;
            g.mu = float3(gChunk[j * 16 + 0], gChunk[j * 16 + 1], gChunk[j * 16 + 2]);
            g.sigma = gChunk[j * 16 + 3];
            g.a00 = gChunk[j * 16 + 4]; g.a10 = gChunk[j * 16 + 5]; g.a11 = gChunk[j * 16 + 6];
            g.a20 = gChunk[j * 16 + 7]; g.a21 = gChunk[j * 16 + 8]; g.a22 = gChunk[j * 16 + 9];
            const float temperature = gChunk[j * 16 + 10];
            const float3 colour = unpackColour(gChunk[j * 16 + 11]);
            for (uint h = 0; h < 2; ++h) {
                const float w = evalGauss(g, x[h]);
                acc[h][0] += w;
                acc[h][1] += w * temperature;
                acc[h][2] += w * colour.x;
                acc[h][3] += w * colour.y;
                acc[h][4] += w * colour.z;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    float loss[2];
    float energy[2];
    float mass[2];
    for (uint h = 0; h < 2; ++h) {
        const uint n = voxelIndex(morton3(t + h * 256));
        const uint v = leaf * 512 + n;
        model[v * 2 + 0] = float4(acc[h][0], acc[h][1], acc[h][2], acc[h][3]);
        model[v * 2 + 1] = float4(acc[h][4], 0.0, 0.0, 0.0);
        const float4 vox = voxelOf(leaf, n);
        const float rho = vox.w > 0.5 ? max(vox.x, 0.0) : 0.0;
        const float d = acc[h][0] - rho;
        // The loss over the active voxels: what the round trip is judged on.
        loss[h] = vox.w > 0.5 ? d * d : 0.0;
        energy[h] = rho * rho;
        mass[h] = rho;
    }
    if (params.mode == 1) {
        float sums[3];
        for (uint k = 0; k < 3; ++k) {
            gReduce[t] = k == 0 ? loss[0] : k == 1 ? energy[0] : mass[0];
            gReduce[t + 256] = k == 0 ? loss[1] : k == 1 ? energy[1] : mass[1];
            for (uint stride = 1; stride < 512; stride *= 2) {
                GroupMemoryBarrierWithGroupSync();
                for (uint h = 0; h < 2; ++h) {
                    const uint i = t + h * 256;
                    if ((i % (2 * stride)) == 0) gReduce[i] += gReduce[i + stride];
                }
            }
            GroupMemoryBarrierWithGroupSync();
            sums[k] = gReduce[0];
            GroupMemoryBarrierWithGroupSync();
        }
        if (t == 0) {
            partial[leaf * 8 + 4] = sums[0];
            partial[leaf * 8 + 5] = sums[1];
            partial[leaf * 8 + 6] = sums[2];
        }
    }
}

/// One group: the loss sums of every leaf, in order, and the relative L2
/// error sqrt(sum r^2 / sum rho^2), into the counters' stats.
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sLoss(uint3 gtid: SV_GroupThreadID) {
    const uint t = gtid.x;
    for (uint k = 0; k < 3; ++k) {
        float s = 0.0;
        for (uint leaf = t; leaf < params.leafCount; leaf += 256) s += partial[leaf * 8 + 4 + k];
        gReduce[t] = s;
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) {
            float total = 0.0;
            for (uint i = 0; i < 256; ++i) total += gReduce[i];
            counters[kCounterStats + k] = asuint(total);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) {
        const float r2 = asfloat(counters[kCounterStats + 0]);
        const float e2 = asfloat(counters[kCounterStats + 1]);
        counters[kCounterStats + 3] = asuint(e2 > 0.0 ? sqrt(r2 / e2) : 0.0);
    }
}

groupshared float gGrad[64 * 14];

/// One group a slot: its gradient over the voxels in its box. `mode` 0 the
/// geometry's (10 parameters, of the density loss sum (rho^ - rho)^2); 1
/// the attributes' (of sum rho (attr^ - attr)^2), with the Hessian's row sums;
/// 2 the raw moments, about its centre, of the density it is responsible
/// for (rho w / rho^): an EM step's.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sBackward(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint slot = gid.x;
    const uint t = gtid.x;
    if (slot >= params.capacity) return;
    float3 half;
    bool alive;
    const Gauss g = gaussOfRecord(slot, half, alive);
    float acc[14];
    for (uint k = 0; k < 14; ++k) acc[k] = 0.0;
    if (alive) {
        const float4 s1 = state[slot * 4 + 1];
        const float4 s2 = state[slot * 4 + 2];
        const float4 s3 = state[slot * 4 + 3];
        const int3 vlo = int3(floor(g.mu - half));
        const int3 vhi = int3(floor(g.mu + half));
        const int3 clo = max(floorDiv8(vlo) - params.cellLo.xyz, int3(0));
        const int3 chi = min(floorDiv8(vhi) - params.cellLo.xyz, params.cellExtent.xyz - 1);
        for (int cx = clo.x; cx <= chi.x; ++cx) {
            for (int cy = clo.y; cy <= chi.y; ++cy) {
                for (int cz = clo.z; cz <= chi.z; ++cz) {
                    const int leaf = leafAtCell(int3(cx, cy, cz));
                    if (leaf < 0) continue;
                    const int3 o = leafOrigin(uint(leaf));
                    for (uint k = t; k < 512; k += 64) {
                        const uint3 c = uint3(k >> 6, (k >> 3) & 7u, k & 7u);   // NanoVDB's own order
                        const int3 ijk = o + int3(c);
                        if (any(ijk < vlo) || any(ijk > vhi)) continue;
                        const float3 x = float3(ijk) + 0.5;
                        const float3 d = x - g.mu;
                        const float u0 = g.a00 * d.x + g.a10 * d.y + g.a20 * d.z;
                        const float u1 = g.a11 * d.y + g.a21 * d.z;
                        const float u2 = g.a22 * d.z;
                        const float q = u0 * u0 + u1 * u1 + u2 * u2;
                        if (q >= kReachQ) continue;
                        const float w = g.sigma * exp(-0.5 * q);
                        const uint v = uint(leaf) * 512 + k;
                        const float4 mv = model[v * 2 + 0];
                        const float4 vox = voxelOf(uint(leaf), k);
                        const float rho = vox.w > 0.5 ? max(vox.x, 0.0) : 0.0;
                        if (params.mode == 0) {
                            const float r2 = 2.0 * (mv.x - rho);
                            // d rho^/d log sigma = w; d/d mu = w P d = w A u;
                            // d/d A_ij = -w d_i u_j; a diagonal by its log, times A_jj.
                            acc[3] += r2 * w;
                            acc[0] += r2 * w * (g.a00 * u0);
                            acc[1] += r2 * w * (g.a10 * u0 + g.a11 * u1);
                            acc[2] += r2 * w * (g.a20 * u0 + g.a21 * u1 + g.a22 * u2);
                            acc[4] += r2 * (-w * d.x * u0) * g.a00;
                            acc[5] += r2 * (-w * d.y * u0);
                            acc[6] += r2 * (-w * d.y * u1) * g.a11;
                            acc[7] += r2 * (-w * d.z * u0);
                            acc[8] += r2 * (-w * d.z * u1);
                            acc[9] += r2 * (-w * d.z * u2) * g.a22;
                            // What the gaussian is answerable for of the residual:
                            // its share of the voxel times the square, for the
                            // densify round's split (v2sSplitFlags).
                            if (mv.x > 1e-20) acc[10] += (w / mv.x) * 0.25 * r2 * r2;
                        } else if (params.mode == 2) {
                            // EM: the voxel's density, shared by responsibility.
                            if (mv.x > 1e-20) {
                                const float c = rho * w / mv.x;
                                acc[0] += c;
                                acc[1] += c * d.x; acc[2] += c * d.y; acc[3] += c * d.z;
                                acc[4] += c * d.x * d.x; acc[5] += c * d.x * d.y; acc[6] += c * d.x * d.z;
                                acc[7] += c * d.y * d.y; acc[8] += c * d.y * d.z; acc[9] += c * d.z * d.z;
                            }
                        } else if (rho > 0.0 && mv.x > 1e-20) {
                            const float share = w / mv.x;
                            const float4 mv1 = model[v * 2 + 1];
                            const float3 targetColour = unpackColour(vox.z);
                            const float tHat = mv.y / mv.x;
                            const float3 cHat = float3(mv.z, mv.w, mv1.x) / mv.x;
                            acc[0] += 2.0 * rho * (tHat - vox.y) * share;
                            acc[1] += 2.0 * rho * (cHat.x - targetColour.x) * share;
                            acc[2] += 2.0 * rho * (cHat.y - targetColour.y) * share;
                            acc[3] += 2.0 * rho * (cHat.z - targetColour.z) * share;
                            // The Hessian's row sum, not its diagonal: the shares at
                            // a voxel add to one, so this bounds the coupling
                            // between overlapping gaussians (Gershgorin) and the
                            // step cannot overshoot. The diagonal alone did, by
                            // about the overlap, and drove temperatures to 0.
                            acc[4] += 2.0 * rho * share;
                        }
                    }
                }
            }
        }
    }
    for (uint k = 0; k < 14; ++k) gGrad[t * 14 + k] = acc[k];
    GroupMemoryBarrierWithGroupSync();
    // One thread an entry: four sums each, written whole (a thread a
    // component would race on the entry).
    if (t < 4) {
        float4 e = float4(0.0);
        for (uint c = 0; c < 4; ++c) {
            const uint k = t * 4 + c;
            if (k >= 14) continue;
            float sum = 0.0;
            for (uint i = 0; i < 64; ++i) sum += gGrad[i * 14 + k];
            e[c] = sum;
        }
        grad[slot * 4 + t] = e;
    }
}

/// One thread a slot: an Adam step on the geometry, held to the sizes a
/// gaussian may take.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sAdam(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (!aliveSlot(slot)) return;
    const float beta1 = 0.9, beta2 = 0.999;
    const float c1 = 1.0 - pow(beta1, float(params.step));
    const float c2 = 1.0 - pow(beta2, float(params.step));
    float4 p[3] = { state[slot * 4 + 0], state[slot * 4 + 1], state[slot * 4 + 2] };
    const float4 gr[3] = { grad[slot * 4 + 0], grad[slot * 4 + 1], grad[slot * 4 + 2] };
    // Each parameter's step, in its own unit: voxels for the centre, the
    // log for sigma and the diagonal, and the off-diagonal's in the
    // diagonal's scale.
    const float scaleOff = sqrt(exp(p[1].x + p[1].z));
    const float meanStd = exp(-(p[1].x + p[1].z + p[2].y) / 3.0);
    const float lr = params.learningRate;
    const float rate[10] = { lr * meanStd, lr * meanStd, lr * meanStd, lr, lr, lr * scaleOff, lr, lr * scaleOff,
                             lr * scaleOff, lr };
    for (uint k = 0; k < 10; ++k) {
        const uint e = k / 4, c = k % 4;
        const float gk = gr[e][c];
        float4 m = adam[slot * 8 + e];
        float4 v = adam[slot * 8 + 4 + e];
        m[c] = beta1 * m[c] + (1.0 - beta1) * gk;
        v[c] = beta2 * v[c] + (1.0 - beta2) * gk * gk;
        adam[slot * 8 + e] = m;
        adam[slot * 8 + 4 + e] = v;
        const float stepK = rate[k] * (m[c] / c1) / (sqrt(v[c] / c2) + 1e-12);
        // The parameter layout: p0 = (mu, log sigma), p1 = (l00, a10, l11, a20), p2 = (a21, l22, ...).
        const uint pe = k < 4 ? 0u : k < 8 ? 1u : 2u;
        const uint pc = k < 4 ? k : k < 8 ? k - 4 : k - 8;
        p[pe][pc] -= stepK;
    }
    const float lo = -log(params.maxStd), hi = -log(params.minStd);
    p[1].x = clamp(p[1].x, lo, hi);
    p[1].z = clamp(p[1].z, lo, hi);
    p[2].y = clamp(p[2].y, lo, hi);
    const float d0 = exp(p[1].x), d1 = exp(p[1].z), d2 = exp(p[2].y);
    p[1].y = clamp(p[1].y, -2.0 * sqrt(d0 * d1), 2.0 * sqrt(d0 * d1));
    p[1].w = clamp(p[1].w, -2.0 * sqrt(d0 * d2), 2.0 * sqrt(d0 * d2));
    p[2].x = clamp(p[2].x, -2.0 * sqrt(d1 * d2), 2.0 * sqrt(d1 * d2));
    const float rhoMax = asfloat(counters[kCounterRhoMax]);
    p[0].w = min(p[0].w, log(max(rhoMax * 4.0, 1e-30)));
    state[slot * 4 + 0] = p[0];
    state[slot * 4 + 1] = p[1];
    state[slot * 4 + 2] = p[2];
}

/// One thread a slot: an EM step -- the gaussian moved to the moments of
/// the density it is responsible for (the gather's mode 2), the stop's
/// share undone; its sizes held as Adam holds them.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sEm(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (!aliveSlot(slot)) return;
    const float4 g0 = grad[slot * 4 + 0];
    const float4 g1 = grad[slot * 4 + 1];
    const float4 g2 = grad[slot * 4 + 2];
    const float m = g0.x;
    if (!(m > 1e-20)) {
        float4 s2 = state[slot * 4 + 2];
        s2.w = 0.0;   // responsible for nothing: freed
        state[slot * 4 + 2] = s2;
        return;
    }
    const float3 shift = g0.yzw / m;
    const float k = 1.0 / kTruncVariance;
    const Gauss old = gaussOfState(slot);
    Gauss g = gaussFromCovariance(old.mu + shift, m / kTruncMass,
                                  (g1.x / m - shift.x * shift.x) * k + 1e-4, (g1.y / m - shift.x * shift.y) * k,
                                  (g1.z / m - shift.x * shift.z) * k, (g1.w / m - shift.y * shift.y) * k + 1e-4,
                                  (g2.x / m - shift.y * shift.z) * k, (g2.y / m - shift.z * shift.z) * k + 1e-4);
    if (!holdFactor(g)) return;   // the moments were not a gaussian's: kept as it was
    const float4 s2 = state[slot * 4 + 2];
    const float4 s3 = state[slot * 4 + 3];
    storeState(slot, g, s2.z, s3.xyz);
}

/// One thread a slot: a Jacobi step on the attributes.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sAttributes(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (!aliveSlot(slot)) return;
    const float4 g0 = grad[slot * 4 + 0];
    const float h = grad[slot * 4 + 1].x;
    if (!(h > 1e-20)) return;
    float4 s2 = state[slot * 4 + 2];
    float4 s3 = state[slot * 4 + 3];
    const float damp = 1.0;
    if (params.hasTemperature != 0) s2.z = max(s2.z - damp * g0.x / h, 0.0);
    if (params.hasColour != 0) s3.xyz = saturate(s3.xyz - damp * g0.yzw / h);
    state[slot * 4 + 2] = s2;
    state[slot * 4 + 3] = s3;
}

// --- Binning: each leaf's list of the gaussians whose boxes meet it ---------------

/// The leaf cells a slot's box covers, clamped to the grid's box; false
/// for a dead slot.
bool boxCells(uint slot, out int3 clo, out int3 chi) {
    clo = int3(0); chi = int3(-1);
    if (slot >= params.capacity) return false;
    const float4 r0 = records[slot * 4 + 0];
    const float4 r3 = records[slot * 4 + 3];
    if (r3.w < 0.5) return false;
    clo = max(floorDiv8(int3(floor(r0.xyz - r3.xyz))) - params.cellLo.xyz, int3(0));
    chi = min(floorDiv8(int3(floor(r0.xyz + r3.xyz))) - params.cellLo.xyz, params.cellExtent.xyz - 1);
    return true;
}

/// One thread a leaf: its cursor cleared, and (`mode` 0) its count.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sBinClear(uint3 tid: SV_DispatchThreadID) {
    const uint leaf = tid.x;
    if (leaf >= params.leafCount) return;
    if (params.mode == 0) bins[leaf] = 0u;
    bins[2 * params.leafCount + leaf] = 0u;
}

/// One thread a slot: `mode` 0 counts it into every leaf its box meets, 1
/// writes it into their lists.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sBinSlots(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    int3 clo, chi;
    if (!boxCells(slot, clo, chi)) return;
    for (int cx = clo.x; cx <= chi.x; ++cx) {
        for (int cy = clo.y; cy <= chi.y; ++cy) {
            for (int cz = clo.z; cz <= chi.z; ++cz) {
                const int leaf = leafAtCell(int3(cx, cy, cz));
                if (leaf < 0) continue;
                if (params.mode == 0) {
                    InterlockedAdd(bins[uint(leaf)], 1u);
                } else {
                    uint at;
                    InterlockedAdd(bins[2 * params.leafCount + uint(leaf)], 1u, at);
                    pairs[bins[params.leafCount + uint(leaf)] + at] = slot;
                }
            }
        }
    }
}

/// One group: where each leaf's list starts (an exclusive scan of the
/// counts, in leaf order), the total into the counters.
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sBinScan(uint3 gtid: SV_GroupThreadID) {
    const uint t = gtid.x;
    const uint total = params.leafCount;
    const uint slice = (total + 255) / 256;
    const uint from = min(t * slice, total);
    const uint to = min(from + slice, total);
    uint sum = 0;
    for (uint i = from; i < to; ++i) sum += bins[i];
    gScan[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        uint running = 0;
        for (uint k = 0; k < 256; ++k) {
            const uint here = gScan[k];
            gScan[k] = running;
            running += here;
        }
        counters[kCounterTotal] = running;
    }
    GroupMemoryBarrierWithGroupSync();
    uint at = gScan[t];
    for (uint i = from; i < to; ++i) {
        bins[total + i] = at;
        at += bins[i];
    }
}

groupshared uint gSort[2048];

/// One group a leaf: its list sorted by slot (bitonic, in group memory), so
/// a voxel adds its gaussians in an order no race decides. A list longer
/// than 2048 is left as the atomics wrote it.
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sBinSort(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint leaf = gid.x;
    const uint t = gtid.x;
    if (leaf >= params.leafCount) return;
    const uint n = bins[leaf];
    if (n < 2 || n > 2048) return;
    const uint first = bins[params.leafCount + leaf];
    uint size = 2;
    while (size < n) size *= 2;
    for (uint i = t; i < size; i += 256) gSort[i] = i < n ? pairs[first + i] : 0xFFFFFFFFu;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 2; k <= size; k *= 2) {
        for (uint j = k / 2; j > 0; j /= 2) {
            for (uint i = t; i < size; i += 256) {
                const uint l = i ^ j;
                if (l > i) {
                    const bool up = (i & k) == 0;
                    const uint a = gSort[i], b = gSort[l];
                    if ((a > b) == up) {
                        gSort[i] = b;
                        gSort[l] = a;
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
        }
    }
    for (uint i = t; i < n; i += 256) pairs[first + i] = gSort[i];
}

// --- A sequence: the frame before, carried by the flow --------------------------------

/// The flow at a point (voxels a frame), at the voxel under it; zero outside
/// the leaves.
float3 flowAt(float3 p) {
    const int3 ijk = int3(floor(p));
    const int leaf = leafAtCell(floorDiv8(ijk) - params.cellLo.xyz);
    if (leaf < 0) return float3(0.0);
    const uint3 c = uint3(ijk - leafOrigin(uint(leaf)));
    const float4 v = velocity[uint(leaf) * params.leafEntries + 1 + voxelIndex(c)];
    return v.w > 0.5 ? v.xyz : float3(0.0);
}

/// One thread a slot: the frame before's gaussian in its own slot -- carried
/// by the flow where there is one, its centre by a midpoint step (RK2) and
/// its covariance by the flow's gradient, F = I + dt grad u, Sigma' = F Sigma
/// F^T (SplashSplat's deformation), its mass kept -- or empty where the
/// slot was. A slot keeps its gaussian from frame to frame: that is what a
/// sequence's planes are indexed by.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sWarm(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    const float4 r0 = previous[slot * 4 + 0];
    const float4 r1 = previous[slot * 4 + 1];
    const float4 r2 = previous[slot * 4 + 2];
    const float4 r3 = previous[slot * 4 + 3];
    if (!(r3.w > 0.5)) {
        for (uint k = 0; k < 4; ++k) state[slot * 4 + k] = float4(0.0);
        return;
    }
    Gauss g;
    g.mu = r0.xyz; g.sigma = r0.w;
    g.a00 = r1.x; g.a10 = r1.y; g.a11 = r1.z; g.a20 = r1.w; g.a21 = r2.x; g.a22 = r2.y;
    if (params.hasVelocity != 0) {
        const float dt = params.dt;
        const float3 mid = g.mu + 0.5 * dt * flowAt(g.mu);
        const float3 moved = g.mu + dt * flowAt(mid);
        // The gradient by central differences a voxel apart; its columns
        // are the flow's derivative along each axis.
        float3x3 f = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
        for (uint k = 0; k < 3; ++k) {
            float3 e = float3(0.0);
            e[k] = 1.0;
            const float3 du = 0.5 * (flowAt(g.mu + e) - flowAt(g.mu - e));
            f[0][k] += dt * du.x; f[1][k] += dt * du.y; f[2][k] += dt * du.z;
        }
        const float b00 = 1.0 / g.a00, b11 = 1.0 / g.a11, b22 = 1.0 / g.a22;
        const float b10 = -g.a10 * b00 / g.a11;
        const float b21 = -g.a21 * b11 / g.a22;
        const float b20 = -(g.a20 * b00 + g.a21 * b10) / g.a22;
        const float3x3 b = float3x3(b00, 0, 0, b10, b11, 0, b20, b21, b22);
        const float3x3 sigma = mul(f, mul(mul(transpose(b), b), transpose(f)));
        const float mass = g.sigma * kTwoPi32 / (g.a00 * g.a11 * g.a22);
        Gauss n = gaussFromCovariance(moved, mass, sigma[0][0], sigma[0][1], sigma[0][2], sigma[1][1], sigma[1][2],
                                      sigma[2][2]);
        if (holdFactor(n)) g = n;
        else g.mu = moved;
    }
    holdFactor(g);
    storeState(slot, g, r2.z, unpackColour(r2.w));
}

// --- 5. Densify and prune ---------------------------------------------------------

/// One thread a slot: its residual error (the geometry gather's 11th sum),
/// `mode` 0 into the largest, 1 counted against `threshold`, 2 flagged.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sSplitFlags(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    const float err = aliveSlot(slot) ? grad[slot * 4 + 2].z : 0.0;
    if (params.mode == 0) {
        InterlockedMax(counters[kCounterSplitMax], asuint(max(err, 0.0)));
    } else if (params.mode == 1) {
        if (err > params.threshold) InterlockedAdd(counters[kCounterCount], 1u);
    } else {
        flags[slot] = err > params.threshold ? 1u : 0u;
    }
}

/// Eigen-decomposition of a symmetric 3x3 by Jacobi rotations (values on
/// the diagonal, vectors in the columns of `v`).
void jacobi3(inout float3x3 a, out float3x3 v) {
    v = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    for (uint sweep = 0; sweep < 12; ++sweep) {
        for (uint pq = 0; pq < 3; ++pq) {
            const uint p = pq == 2 ? 1u : 0u;
            const uint q = pq == 0 ? 1u : 2u;
            const float apq = a[p][q];
            if (abs(apq) < 1e-12 * (abs(a[p][p]) + abs(a[q][q])) + 1e-30) continue;
            const float theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const float t = sign(theta + 1e-30) / (abs(theta) + sqrt(theta * theta + 1.0));
            const float c = 1.0 / sqrt(t * t + 1.0);
            const float s = t * c;
            float3x3 j = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
            j[p][p] = c; j[q][q] = c; j[p][q] = s; j[q][p] = -s;
            a = mul(transpose(j), mul(a, j));
            v = mul(v, j);
        }
    }
}

/// One thread a split: the k-th worst gaussian becomes two, half its mass
/// each, a deviation apart along its longest axis (each at half a deviation
/// from the centre, that axis's variance down by a quarter of it, so the
/// pair keeps the one's second moment), the second into the k-th free slot
/// -- Gaussian Fluids' reseeding, by the residual.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sSplit(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.splitCount) return;
    const uint slot = list[params.listBase + k];
    const uint other = list[params.freeBase + k];
    const Gauss g = gaussOfState(slot);
    const float b00 = 1.0 / g.a00, b11 = 1.0 / g.a11, b22 = 1.0 / g.a22;
    const float b10 = -g.a10 * b00 / g.a11;
    const float b21 = -g.a21 * b11 / g.a22;
    const float b20 = -(g.a20 * b00 + g.a21 * b10) / g.a22;
    const float3x3 b = float3x3(b00, 0, 0, b10, b11, 0, b20, b21, b22);
    const float3x3 sigma = mul(transpose(b), b);
    float3x3 e = sigma;
    float3x3 axes;
    jacobi3(e, axes);
    uint major = 0;
    if (e[1][1] > e[major][major]) major = 1;
    if (e[2][2] > e[major][major]) major = 2;
    const float variance = max(e[major][major], 1e-8);
    const float3 axis = float3(axes[0][major], axes[1][major], axes[2][major]);
    const float a = 0.5 * sqrt(variance);
    const float3x3 shrunk = sigma - (a * a) * float3x3(axis.x * axis, axis.y * axis, axis.z * axis);
    const float mass = g.sigma * kTwoPi32 / (g.a00 * g.a11 * g.a22);
    const float4 s2 = state[slot * 4 + 2];
    const float4 s3 = state[slot * 4 + 3];
    for (uint h = 0; h < 2; ++h) {
        Gauss n = gaussFromCovariance(g.mu + (h == 0 ? a : -a) * axis, 0.5 * mass, shrunk[0][0], shrunk[0][1],
                                      shrunk[0][2], shrunk[1][1], shrunk[1][2], shrunk[2][2]);
        if (!holdFactor(n)) return;
        storeState(h == 0 ? slot : other, n, s2.z, s3.xyz);
    }
}

/// One thread a slot: a gaussian whose mass is a negligible share goes.
/// Then flags the free slots below the capacity for the compaction.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sPrune(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    bool free = !aliveSlot(slot);
    if (!free) {
        const Gauss g = gaussOfState(slot);
        const float mass = g.sigma * kTwoPi32 / (g.a00 * g.a11 * g.a22);
        const float total = asfloat(counters[kCounterStats + 2]);
        if (mass < params.pruneMass * total || !(g.sigma > 0.0)) {
            float4 s2 = state[slot * 4 + 2];
            s2.w = 0.0;
            state[slot * 4 + 2] = s2;
            free = true;
        }
    }
    flags[slot] = free ? 1u : 0u;
}

/// One thread a voxel: whether the mixture leaves it under by more than
/// `threshold` -- counted (`mode` 0) or flagged for the compaction (1).
[shader("compute")]
[numthreads(256, 1, 1)]
void v2sUnder(uint3 tid: SV_DispatchThreadID) {
    const uint v = tid.x;
    if (v >= params.leafCount * 512) return;
    const float4 vox = voxelOf(v / 512, v % 512);
    const float rho = vox.w > 0.5 ? vox.x : 0.0;
    const bool under = rho - model[v * 2].x > params.threshold;
    if (params.mode == 0) {
        if (under) InterlockedAdd(counters[kCounterCount], 1u);
    } else {
        flags[v] = under ? 1u : 0u;
    }
}

/// One thread a new gaussian: at the k-th chosen voxel, into the k-th free slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sSpawn(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.bornCount) return;
    const uint v = list[params.listBase + k];
    const uint slot = list[params.freeBase + k];
    const uint leaf = v / 512, n = v % 512;
    const uint3 c = uint3(n >> 6, (n >> 3) & 7u, n & 7u);
    const float4 vox = voxelOf(leaf, n);
    Gauss g;
    g.mu = float3(leafOrigin(leaf) + int3(c)) + 0.5;
    const float s = max(params.minStd, 1.0);
    g.a00 = 1.0 / s; g.a11 = 1.0 / s; g.a22 = 1.0 / s;
    g.a10 = 0.0; g.a20 = 0.0; g.a21 = 0.0;
    g.sigma = max(vox.x - model[v * 2].x, 1e-6);
    storeState(slot, g, vox.y, params.hasColour != 0 ? unpackColour(vox.z) : float3(1.0));
}

// --- The output ---------------------------------------------------------------------

/// One thread a slot: flags the living for the compaction.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sAliveFlags(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.capacity) return;
    flags[slot] = aliveSlot(slot) ? 1u : 0u;
}

/// One thread an output gaussian: `mode` 0 copies the k-th living record
/// aside (into `adam`), 1 lays it down at k.
[shader("compute")]
[numthreads(64, 1, 1)]
void v2sWrite(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.outCount) return;
    if (params.mode == 0) {
        const uint slot = list[params.listBase + k];
        for (uint e = 0; e < 4; ++e) adam[k * 8 + e] = records[slot * 4 + e];
    } else {
        for (uint e = 0; e < 4; ++e) records[k * 4 + e] = adam[k * 8 + e];
    }
}
