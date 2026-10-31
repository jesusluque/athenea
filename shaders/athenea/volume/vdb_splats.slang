// Copyright (c) 2026 jesus luque.
//
// Volumetric gaussians and the grids they come from and go back to
// (proposal 103; the fit itself is the Vdb2Splat effect, plugins/vdb2splat):
//
//  - vsPackLeaves: a NanoVDB grid's leaves as the picture the effect reads,
//    513 entries a leaf: (origin, 0), then its 512 voxels in NanoVDB's order,
//    (density, temperature, Cd as rgba8 in a float's bits, active). The
//    temperature and Cd are read at the density's voxels from their own grids.
//  - vsSplatToGrid: the way back (§3.4): the mixture sum sigma_i G_i
//    evaluated at the active voxels of a given topology and written into the
//    grid's own values, in place -- what the M9 path tracer then renders as
//    it renders any grid. The temperature too, where a temperature grid is
//    bound, as the density-weighted mean sum c_i w_i / sum w_i.
//  - vsFixture: a test's or a measurement's grid, filled on its topology.
//  - vsCompare / vsReduce: two grids of one topology against each other.
//  - vsRays / vsRayReduce: optical depth along random rays, the mixture's
//    closed form against ratio tracking through a grid (medium.slang).
//  - vsExport: the gaussians in the world, as a ParticleField's records.
//
// A gaussian record is four float4s in the grid's index space (a voxel ijk
// spans [ijk, ijk + 1)): (centre, sigma), (A00, A10, A11, A20), (A21, A22,
// temperature, Cd rgba8), (half extents of its box, alive). A is the lower
// Cholesky factor of the precision: G(x) = exp(-|A^T (x - mu)|^2 / 2),
// stopped at 3.5 deviations as the fit stops it.
import athenea.volume.medium;

#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

struct VsParams {
    uint  densityWord;      // the density grid's word offset in `words`
    uint  leafCount;
    uint  temperatureWord;  // 0xFFFFFFFF: none
    uint  colourWord;       // 0xFFFFFFFF: none (a vec3 grid)

    uint  count;            // gaussians
    uint  leafFirst;        // a slice of the leaves (splat-to-grid)
    uint  kind;             // vsFixture: 0 the temperature alone, 1 a smooth noise blob
    uint  rays;

    float4 centre;          // vsFixture: the blob's centre (index space), w its radius
    float4 indexToWorld0;   // rows of index to world (vsExport, vsRays)
    float4 indexToWorld1;
    float4 indexToWorld2;

    float densityScale;     // density to extinction a world unit (vsExport's opacity, vsRays)
    uint  samples;          // vsRays: ratio-tracking estimates a ray
    uint  seed;
    float tolerance;        // vsRays: what discretisation may cost a transmittance

    float4 motion;          // vsFixture kind 2: (angular speed about y, rad a frame; rise, voxels a frame; the
                            // frame; the voxel size). vsPackLeaves kind 3: w the scale from vel to voxels a frame
};

StructuredBuffer<uint>     words;      // grids (NanoVDB), and for vsRays the VolumeSet's words
RWStructuredBuffer<uint>   wordsOut;   // the same buffer, written
StructuredBuffer<uint>     other;      // vsCompare: the reference grid, same layout
StructuredBuffer<uint>     prevJudged; // vsFlicker: the judged grid the frame before, same layout
StructuredBuffer<uint>     prevOther;  // vsFlicker: the reference the frame before
StructuredBuffer<float4>   gaussiansB; // vsSlots: the frame's records, slot for slot with `gaussians` the frame before's
StructuredBuffer<float4>   gaussians;  // records, 4 a gaussian
RWStructuredBuffer<float4> leavesOut;  // vsPackLeaves: the picture
RWStructuredBuffer<float>  sums;       // per leaf or per ray partial sums, then the totals
RWStructuredBuffer<float>  exported;   // vsExport: 14 floats a gaussian, then 5 more
ConstantBuffer<VsParams>   params;

static const uint kNone = 0xFFFFFFFFu;
static const float kReach = 3.5;
static const float kReachQ = kReach * kReach;
static const uint kLeafEntries = 513;

// --- The tree -----------------------------------------------------------------------

pnanovdb_grid_handle_t gridAt(uint word) {
    pnanovdb_grid_handle_t g;
    g.address.byte_offset = word * 4u;
    return g;
}

pnanovdb_leaf_handle_t leafOf(uint gridWord, uint i, uint gridType) {
    const pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(words, gridAt(gridWord));
    const uint leafSize = PNANOVDB_GRID_TYPE_GET(gridType, leaf_size);
    pnanovdb_address_t at = pnanovdb_address_offset64(tree.address, pnanovdb_tree_get_node_offset_leaf(words, tree));
    at = pnanovdb_address_offset(at, i * leafSize);
    pnanovdb_leaf_handle_t leaf;
    leaf.address = at;
    return leaf;
}

int3 originOf(pnanovdb_leaf_handle_t leaf) {
    const pnanovdb_coord_t c = pnanovdb_leaf_get_bbox_min(words, leaf);
    return int3(c.x, c.y, c.z);
}

/// The byte address of voxel `ijk`'s value in another grid of the same
/// space, or none where that grid has no leaf there.
bool addressIn(uint gridWord, uint gridType, int3 ijk, out pnanovdb_address_t at) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, pnanovdb_tree_get_root(words, pnanovdb_grid_get_tree(words, gridAt(gridWord))));
    pnanovdb_coord_t c;
    c.x = ijk.x; c.y = ijk.y; c.z = ijk.z;
    at = pnanovdb_readaccessor_get_value_address(gridType, words, acc, c);
    return pnanovdb_readaccessor_get_dim(gridType, words, acc, c) == 1u;
}

float3 unpackColour(float bits) {
    const uint u = asuint(bits);
    return float3(float(u & 255u), float((u >> 8) & 255u), float((u >> 16) & 255u)) / 255.0;
}

/// A half's bits, in the low 16 of a word.
uint halfBits(float x) {
    const uint h = f32tof16(x);
    return h & 0xFFFFu;
}

float packColour(float3 c) {
    const uint3 q = uint3(round(saturate(c) * 255.0));
    return asfloat(q.x | (q.y << 8) | (q.z << 16) | (255u << 24));
}

// --- The leaves, as the effect reads them ---------------------------------------------

[shader("compute")]
[numthreads(256, 1, 1)]
void vsPackLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x / 512, n = tid.x % 512;
    if (i >= params.leafCount) return;
    const pnanovdb_leaf_handle_t leaf = leafOf(params.densityWord, i, PNANOVDB_GRID_TYPE_FLOAT);
    const int3 origin = originOf(leaf);
    if (n == 0) leavesOut[i * kLeafEntries] = float4(float3(origin), 0.0);
    const bool active = pnanovdb_leaf_get_value_mask(words, leaf, n);
    const float rho = pnanovdb_read_float(words, pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, words, leaf, n));
    const int3 ijk = origin + int3(int(n >> 6), int((n >> 3) & 7u), int(n & 7u));
    float temperature = 0.0;
    float3 colour = float3(1.0);
    pnanovdb_address_t at;
    if (params.temperatureWord != kNone && addressIn(params.temperatureWord, PNANOVDB_GRID_TYPE_FLOAT, ijk, at)) {
        temperature = pnanovdb_read_float(words, at);
    }
    if (params.kind == 3) {
        // THE VELOCITY, in voxels a frame (`motion.w` scales the grid's own
        // units), at the density's voxels: what a warm start advects by.
        float3 vel = float3(0.0);
        if (params.colourWord != kNone && addressIn(params.colourWord, PNANOVDB_GRID_TYPE_VEC3F, ijk, at)) {
            vel = float3(pnanovdb_read_float(words, at), pnanovdb_read_float(words, pnanovdb_address_offset(at, 4u)),
                         pnanovdb_read_float(words, pnanovdb_address_offset(at, 8u))) * params.motion.w;
        }
        leavesOut[i * kLeafEntries + 1 + n] = float4(vel, active ? 1.0 : 0.0);
        return;
    }
    if (params.colourWord != kNone && addressIn(params.colourWord, PNANOVDB_GRID_TYPE_VEC3F, ijk, at)) {
        colour = float3(pnanovdb_read_float(words, at), pnanovdb_read_float(words, pnanovdb_address_offset(at, 4u)),
                        pnanovdb_read_float(words, pnanovdb_address_offset(at, 8u)));
    }
    leavesOut[i * kLeafEntries + 1 + n] = float4(active ? rho : 0.0, temperature, packColour(colour), active ? 1.0 : 0.0);
}

// --- The way back ----------------------------------------------------------------------

groupshared float gChunk[256 * 12];
groupshared uint  gHit[256];

/// One group a leaf (of the slice from `leafFirst`): the mixture at each
/// active voxel, written as the voxel's value; and the temperature where a
/// temperature grid is bound and has the voxel.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsSplatToGrid(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint i = params.leafFirst + gid.x;
    const uint t = gtid.x;
    if (i >= params.leafCount) return;
    const pnanovdb_leaf_handle_t leaf = leafOf(params.densityWord, i, PNANOVDB_GRID_TYPE_FLOAT);
    const int3 origin = originOf(leaf);
    const float3 lo = float3(origin), hi = lo + 8.0;
    float rho[2] = { 0.0, 0.0 };
    float heat[2] = { 0.0, 0.0 };
    float3 x[2];
    for (uint h = 0; h < 2; ++h) {
        const uint n = t + h * 256;
        x[h] = lo + float3(float(n >> 6), float((n >> 3) & 7u), float(n & 7u)) + 0.5;
    }
    for (uint base = 0; base < params.count; base += 256) {
        const uint g = base + t;
        bool hit = false;
        if (g < params.count) {
            const float4 r0 = gaussians[g * 4 + 0];
            const float4 r3 = gaussians[g * 4 + 3];
            hit = r3.w > 0.5 && all(r0.xyz + r3.xyz > lo) && all(r0.xyz - r3.xyz < hi);
            if (hit) {
                const float4 r1 = gaussians[g * 4 + 1];
                const float4 r2 = gaussians[g * 4 + 2];
                gChunk[t * 12 + 0] = r0.x; gChunk[t * 12 + 1] = r0.y; gChunk[t * 12 + 2] = r0.z;
                gChunk[t * 12 + 3] = r0.w;
                gChunk[t * 12 + 4] = r1.x; gChunk[t * 12 + 5] = r1.y; gChunk[t * 12 + 6] = r1.z;
                gChunk[t * 12 + 7] = r1.w;
                gChunk[t * 12 + 8] = r2.x; gChunk[t * 12 + 9] = r2.y; gChunk[t * 12 + 10] = r2.z;
            }
        }
        gHit[t] = hit ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint j = 0; j < 256; ++j) {
            if (gHit[j] == 0) continue;
            const float3 mu = float3(gChunk[j * 12 + 0], gChunk[j * 12 + 1], gChunk[j * 12 + 2]);
            const float sigma = gChunk[j * 12 + 3];
            const float a00 = gChunk[j * 12 + 4], a10 = gChunk[j * 12 + 5], a11 = gChunk[j * 12 + 6];
            const float a20 = gChunk[j * 12 + 7], a21 = gChunk[j * 12 + 8], a22 = gChunk[j * 12 + 9];
            const float temperature = gChunk[j * 12 + 10];
            for (uint h = 0; h < 2; ++h) {
                const float3 d = x[h] - mu;
                const float u0 = a00 * d.x + a10 * d.y + a20 * d.z;
                const float u1 = a11 * d.y + a21 * d.z;
                const float u2 = a22 * d.z;
                const float q = u0 * u0 + u1 * u1 + u2 * u2;
                if (q < kReachQ) {
                    const float w = sigma * exp(-0.5 * q);
                    rho[h] += w;
                    heat[h] += w * temperature;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint h = 0; h < 2; ++h) {
        const uint n = t + h * 256;
        if (!pnanovdb_leaf_get_value_mask(words, leaf, n)) continue;
        const pnanovdb_address_t at = pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, words, leaf, n);
        wordsOut[at.byte_offset / 4u] = asuint(rho[h]);
        pnanovdb_address_t tAt;
        if (params.temperatureWord != kNone &&
            addressIn(params.temperatureWord, PNANOVDB_GRID_TYPE_FLOAT, origin + int3(int(n >> 6), int((n >> 3) & 7u),
                                                                                       int(n & 7u)), tAt)) {
            wordsOut[tAt.byte_offset / 4u] = asuint(rho[h] > 1e-20 ? heat[h] / rho[h] : 0.0);
        }
    }
}

// --- Fixtures -----------------------------------------------------------------------

uint fxHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

float lattice(int3 c, uint seed) {
    const uint h = fxHash(uint(c.x) * 73856093u ^ uint(c.y) * 19349663u ^ uint(c.z) * 83492791u ^ seed);
    return float(h >> 8) / 16777216.0 * 2.0 - 1.0;
}

/// Value noise, smooth (quintic): three octaves.
float smoothNoise(float3 p, uint seed) {
    float total = 0.0, amp = 0.5;
    for (uint o = 0; o < 3; ++o) {
        const int3 c = int3(floor(p));
        const float3 f = p - floor(p);
        const float3 s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        float v = 0.0;
        for (uint k = 0; k < 8; ++k) {
            const int3 o3 = int3(int(k >> 2) & 1, int(k >> 1) & 1, int(k) & 1);
            const float3 w = lerp(1.0 - s, s, float3(o3));
            v += w.x * w.y * w.z * lattice(c + o3, seed + o * 101u);
        }
        total += amp * v;
        amp *= 0.5;
        p *= 2.03;
    }
    return total;
}

/// One thread a voxel of the density grid: `kind` 1 writes a smooth noise
/// blob's density (a ball of radius `centre.w` whose edge the noise moves);
/// both kinds write a temperature where a temperature grid is bound -- hot
/// at the centre, 300 at the radius.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsFixture(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x / 512, n = tid.x % 512;
    if (i >= params.leafCount) return;
    const pnanovdb_leaf_handle_t leaf = leafOf(params.densityWord, i, PNANOVDB_GRID_TYPE_FLOAT);
    if (!pnanovdb_leaf_get_value_mask(words, leaf, n)) return;
    const int3 ijk = originOf(leaf) + int3(int(n >> 6), int((n >> 3) & 7u), int(n & 7u));
    float3 x = float3(ijk) + 0.5;
    if (params.kind == 2) {
        // A SEQUENCE'S FRAME: the blob carried by a known flow, a turn about
        // the vertical through its centre at `motion.x` a frame and a rise of
        // `motion.y` voxels a frame, so frame t is frame 0 at the point the
        // flow brought here, and the velocity is the flow's own.
        const float t = params.motion.z;
        const float3 c = params.centre.xyz + float3(0.0, params.motion.y * t, 0.0);
        const float3 d = x - c;
        pnanovdb_address_t vAt;
        if (params.colourWord != kNone && addressIn(params.colourWord, PNANOVDB_GRID_TYPE_VEC3F, ijk, vAt)) {
            const float3 u = (params.motion.x * float3(d.z, 0.0, -d.x) + float3(0.0, params.motion.y, 0.0)) *
                             params.motion.w;
            wordsOut[vAt.byte_offset / 4u] = asuint(u.x);
            wordsOut[vAt.byte_offset / 4u + 1u] = asuint(u.y);
            wordsOut[vAt.byte_offset / 4u + 2u] = asuint(u.z);
        }
        const float th = -params.motion.x * t;
        x = params.centre.xyz + float3(d.x * cos(th) + d.z * sin(th), d.y, -d.x * sin(th) + d.z * cos(th));
    }
    const float r = length(x - params.centre.xyz) / params.centre.w;
    if (params.kind == 1 || params.kind == 2) {
        const float edge = 1.0 - r + 0.35 * smoothNoise(x / (params.centre.w * 0.35), params.seed);
        const float density = saturate(edge * 3.0) * (0.6 + 0.4 * smoothNoise(x / (params.centre.w * 0.2), params.seed + 7u));
        const pnanovdb_address_t at = pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, words, leaf, n);
        wordsOut[at.byte_offset / 4u] = asuint(max(density, 0.0));
    }
    pnanovdb_address_t tAt;
    if (params.temperatureWord != kNone && addressIn(params.temperatureWord, PNANOVDB_GRID_TYPE_FLOAT, ijk, tAt)) {
        wordsOut[tAt.byte_offset / 4u] = asuint(300.0 + 1500.0 * saturate(1.0 - r) * saturate(1.0 - r));
    }
}

/// One thread a gaussian: a known mixture to recover -- `count` gaussians
/// of random centre inside [centre.w, centre.xyz - centre.w) (index space),
/// deviations between `tolerance` and `densityScale` voxels on a random
/// shear, and a peak between 0.3 and 1 -- written as records into
/// `exported`'s words (4 float4 a gaussian), whose evaluation is then the
/// grid to fit (vsSplatToGrid).
[shader("compute")]
[numthreads(64, 1, 1)]
void vsRandomMixture(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) return;
    uint s = fxHash(g * 2654435761u + params.seed);
    const float margin = params.centre.w;
    const float3 lo = float3(margin), hi = params.centre.xyz - margin;
    float u[10];
    for (uint k = 0; k < 10; ++k) {
        s = fxHash(s + 0x9e3779b9u);
        u[k] = float(s >> 8) / 16777216.0;
    }
    const float3 mu = lo + (hi - lo) * float3(u[0], u[1], u[2]);
    const float minStd = params.tolerance, maxStd = params.densityScale;
    const float3 sd = minStd * pow(maxStd / minStd, float3(u[3], u[4], u[5]));
    const float a00 = 1.0 / sd.x, a11 = 1.0 / sd.y, a22 = 1.0 / sd.z;
    // Off the diagonal, at most 0.6 of the geometric mean of the two
    // diagonals beside it: the factor stays well conditioned.
    const float a10 = (u[6] - 0.5) * 1.2 * sqrt(a00 * a11);
    const float a20 = (u[7] - 0.5) * 1.2 * sqrt(a00 * a22);
    const float a21 = (u[8] - 0.5) * 1.2 * sqrt(a11 * a22);
    const float sigma = 0.3 + 0.7 * u[9];
    // The box: the covariance's diagonal from B = A^-1.
    const float b00 = 1.0 / a00, b11 = 1.0 / a11, b22 = 1.0 / a22;
    const float b10 = -a10 * b00 / a11;
    const float b21 = -a21 * b11 / a22;
    const float b20 = -(a20 * b00 + a21 * b10) / a22;
    const float3 std = sqrt(float3(b00 * b00 + b10 * b10 + b20 * b20, b11 * b11 + b21 * b21, b22 * b22));
    const uint o = g * 16;
    exported[o + 0] = mu.x; exported[o + 1] = mu.y; exported[o + 2] = mu.z; exported[o + 3] = sigma;
    exported[o + 4] = a00; exported[o + 5] = a10; exported[o + 6] = a11; exported[o + 7] = a20;
    exported[o + 8] = a21; exported[o + 9] = a22; exported[o + 10] = 300.0 + 1200.0 * u[9];
    exported[o + 11] = packColour(float3(1.0));
    exported[o + 12] = kReach * std.x; exported[o + 13] = kReach * std.y; exported[o + 14] = kReach * std.z;
    exported[o + 15] = 1.0;
}

// --- Two grids against each other ------------------------------------------------------

groupshared float gSum[256 * 3];

/// One group a leaf: sum (a - b)^2, sum b^2 and the largest |a - b| over the
/// active voxels, `words` the grid judged and `other` the reference, into
/// `sums` (3 a leaf).
[shader("compute")]
[numthreads(256, 1, 1)]
void vsCompare(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint i = gid.x;
    const uint t = gtid.x;
    if (i >= params.leafCount) return;
    const pnanovdb_leaf_handle_t leaf = leafOf(params.densityWord, i, PNANOVDB_GRID_TYPE_FLOAT);
    float d2 = 0.0, b2 = 0.0, dm = 0.0;
    for (uint h = 0; h < 2; ++h) {
        const uint n = t + h * 256;
        if (!pnanovdb_leaf_get_value_mask(words, leaf, n)) continue;
        const pnanovdb_address_t at = pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, words, leaf, n);
        if (params.kind == 1) {
            // A temperature, weighed by the reference's density at the voxel
            // (the two buffers share their layout): what a density-weighted
            // attribute is, and all it can be where there is no density.
            pnanovdb_address_t tAt;
            const int3 ijk = originOf(leaf) + int3(int(n >> 6), int((n >> 3) & 7u), int(n & 7u));
            if (params.temperatureWord == kNone || !addressIn(params.temperatureWord, PNANOVDB_GRID_TYPE_FLOAT, ijk, tAt)) {
                continue;
            }
            const float rho = max(pnanovdb_read_float(other, at), 0.0);
            const float a = pnanovdb_read_float(words, tAt);
            const float b = pnanovdb_read_float(other, tAt);
            d2 += rho * (a - b) * (a - b);
            b2 += rho * b * b;
            if (rho > 0.0) dm = max(dm, abs(a - b));
            continue;
        }
        const float a = pnanovdb_read_float(words, at);
        const float b = pnanovdb_read_float(other, at);
        d2 += (a - b) * (a - b);
        b2 += b * b;
        dm = max(dm, abs(a - b));
    }
    gSum[t * 3 + 0] = d2;
    gSum[t * 3 + 1] = b2;
    gSum[t * 3 + 2] = dm;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        float s0 = 0.0, s1 = 0.0, s2 = 0.0;
        for (uint k = 0; k < 256; ++k) {
            s0 += gSum[k * 3 + 0];
            s1 += gSum[k * 3 + 1];
            s2 = max(s2, gSum[k * 3 + 2]);
        }
        sums[i * 3 + 0] = s0;
        sums[i * 3 + 1] = s1;
        sums[i * 3 + 2] = s2;
    }
}

/// One group: the per-item sums of `count` items of 3 (sum, sum, max) in
/// order, written after them: (total 0, total 1, max, sqrt(total 0 / total 1)).
[shader("compute")]
[numthreads(256, 1, 1)]
void vsReduce(uint3 gtid: SV_GroupThreadID) {
    const uint t = gtid.x;
    float s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (uint i = t; i < params.count; i += 256) {
        s0 += sums[i * 3 + 0];
        s1 += sums[i * 3 + 1];
        s2 = params.kind == 3 ? s2 + sums[i * 3 + 2] : max(s2, sums[i * 3 + 2]);
    }
    gSum[t * 3 + 0] = s0;
    gSum[t * 3 + 1] = s1;
    gSum[t * 3 + 2] = s2;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        float a = 0.0, b = 0.0, c = 0.0;
        for (uint k = 0; k < 256; ++k) {
            a += gSum[k * 3 + 0];
            b += gSum[k * 3 + 1];
            c = params.kind == 3 ? c + gSum[k * 3 + 2] : max(c, gSum[k * 3 + 2]);
        }
        const uint o = params.count * 3;
        sums[o + 0] = a;
        sums[o + 1] = b;
        sums[o + 2] = c;
        // kind 2: the mean of the first (rays); otherwise the relative L2 error.
        sums[o + 3] = params.kind == 2 ? a / float(max(params.count, 1u)) : b > 0.0 ? sqrt(a / b) : 0.0;
    }
}

// --- A sequence: slots and flicker ---------------------------------------------------

/// One thread a slot: alive in both frames, alive the frame before, and how
/// far its centre moved over its mean deviation where alive in both -- into
/// `sums` (3 a slot; vsReduce kind 3 sums all three).
[shader("compute")]
[numthreads(64, 1, 1)]
void vsSlots(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) return;
    const float4 a0 = gaussians[g * 4 + 0], a3 = gaussians[g * 4 + 3];
    const float4 b0 = gaussiansB[g * 4 + 0], b3 = gaussiansB[g * 4 + 3];
    const bool before = a3.w > 0.5, now = b3.w > 0.5;
    const float scale = max((a3.x + a3.y + a3.z) / (3.0 * kReach), 1e-6);
    sums[g * 3 + 0] = before && now ? 1.0 : 0.0;
    sums[g * 3 + 1] = before ? 1.0 : 0.0;
    sums[g * 3 + 2] = before && now ? length(b0.xyz - a0.xyz) / scale : 0.0;
}

/// One group a leaf: sum ((b1 - b0) - (o1 - o0))^2 and sum (o1 - o0)^2 over
/// the active voxels -- what the frame-to-frame change of the judged grid
/// (`words` now, `prevJudged` before) gets wrong of the reference's
/// (`other`, `prevOther`): flicker, relative to the motion itself.
[shader("compute")]
[numthreads(256, 1, 1)]
void vsFlicker(uint3 gid: SV_GroupID, uint3 gtid: SV_GroupThreadID) {
    const uint i = gid.x;
    const uint t = gtid.x;
    if (i >= params.leafCount) return;
    const pnanovdb_leaf_handle_t leaf = leafOf(params.densityWord, i, PNANOVDB_GRID_TYPE_FLOAT);
    float d2 = 0.0, o2 = 0.0;
    for (uint h = 0; h < 2; ++h) {
        const uint n = t + h * 256;
        if (!pnanovdb_leaf_get_value_mask(words, leaf, n)) continue;
        const pnanovdb_address_t at = pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, words, leaf, n);
        const float b1 = pnanovdb_read_float(words, at), b0 = pnanovdb_read_float(prevJudged, at);
        const float o1 = pnanovdb_read_float(other, at), o0 = pnanovdb_read_float(prevOther, at);
        const float e = (b1 - b0) - (o1 - o0);
        d2 += e * e;
        o2 += (o1 - o0) * (o1 - o0);
    }
    gSum[t * 3 + 0] = d2;
    gSum[t * 3 + 1] = o2;
    gSum[t * 3 + 2] = 0.0;
    GroupMemoryBarrierWithGroupSync();
    if (t == 0) {
        float s0 = 0.0, s1 = 0.0;
        for (uint k = 0; k < 256; ++k) {
            s0 += gSum[k * 3 + 0];
            s1 += gSum[k * 3 + 1];
        }
        sums[i * 3 + 0] = s0;
        sums[i * 3 + 1] = s1;
        sums[i * 3 + 2] = 0.0;
    }
}

// --- Optical depth along rays -------------------------------------------------------------

/// erf, Abramowitz and Stegun 7.1.26 (|error| < 1.5e-7).
float erfApprox(float x) {
    const float s = sign(x);
    const float a = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * a);
    const float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
                           0.254829592) * t * exp(-a * a);
    return s * y;
}

/// The integral of G along o + t d, t in [t0, t1], in closed form.
float lineIntegral(float3 mu, float a00, float a10, float a11, float a20, float a21, float a22, float3 o, float3 d,
                   float t0, float t1) {
    const float3 e = o - mu;
    const float3 ue = float3(a00 * e.x + a10 * e.y + a20 * e.z, a11 * e.y + a21 * e.z, a22 * e.z);
    const float3 ud = float3(a00 * d.x + a10 * d.y + a20 * d.z, a11 * d.y + a21 * d.z, a22 * d.z);
    const float a = dot(ud, ud);
    if (a < 1e-20) return 0.0;
    const float b = dot(ud, ue);
    const float c = dot(ue, ue);
    const float tc = -b / a;
    const float k = sqrt(0.5 * a);
    const float exponent = -0.5 * (c - b * b / a);
    if (exponent < -40.0) return 0.0;
    return exp(exponent) * sqrt(3.14159265358979 / (2.0 * a)) * (erfApprox(k * (t1 - tc)) - erfApprox(k * (t0 - tc)));
}

uint rayHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

float rand01(inout uint s) {
    s = s * 747796405u + 2891336453u;
    return float(rayHash(s) >> 8) / 16777216.0;
}

float3 indexToWorld(float3 p) {
    const float4 h = float4(p, 1.0);
    return float3(dot(params.indexToWorld0, h), dot(params.indexToWorld1, h), dot(params.indexToWorld2, h));
}

/// One thread a ray, between two random points of the volume's index box:
/// the mixture's transmittance in closed form against the mean of
/// `samples` ratio-tracking estimates through volume 0 of `words` (a
/// VolumeSet's). Per ray into `sums`: |T_mix - T_rt|, beyond (5 standard
/// errors + tolerance) or not, and the closed-form optical depth.
[shader("compute")]
[numthreads(64, 1, 1)]
void vsRays(uint3 tid: SV_DispatchThreadID) {
    const uint ray = params.leafFirst + tid.x;   // a slice of the rays from `leafFirst`
    if (ray >= params.rays) return;
    const VolumeRecord r = volumeRecord(words, 0);
    uint s = rayHash(ray * 9781u + params.seed);
    const float3 span = r.boundsMax - r.boundsMin;
    const float3 pa = r.boundsMin + span * float3(rand01(s), rand01(s), rand01(s));
    const float3 pb = r.boundsMin + span * float3(rand01(s), rand01(s), rand01(s));
    // The world segment, by its length: medium.slang's parameter is a
    // distance along a unit direction. The closed form runs in index space
    // with the same parameter, the direction pulled back unnormalised.
    const float3 wa = indexToWorld(pa);
    const float3 wb = indexToWorld(pb);
    const float len = length(wb - wa);
    if (!(len > 0.0)) {
        sums[ray * 3 + 0] = 0.0; sums[ray * 3 + 1] = 0.0; sums[ray * 3 + 2] = 0.0;
        return;
    }
    const float3 oi = pa;
    const float3 di = (pb - pa) / len;
    float tau = 0.0;
    for (uint g = 0; g < params.count; ++g) {
        const float4 r0 = gaussians[g * 4 + 0];
        const float4 r1 = gaussians[g * 4 + 1];
        const float4 r2 = gaussians[g * 4 + 2];
        tau += r0.w * lineIntegral(r0.xyz, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, oi, di, 0.0, len);
    }
    tau *= r.sigmaScale;
    const float tMix = exp(-tau);
    float mean = 0.0, square = 0.0;
    for (uint k = 0; k < params.samples; ++k) {
        uint2 rng = mediumStreamOf(ray, params.seed, k, 0u, 0u);   // a stream of its own an estimate
        const float est = mediumTransmittance(words, r, wa, (wb - wa) / len, 0.0, len, rng);
        mean += est;
        square += est * est;
    }
    mean /= float(params.samples);
    const float variance = max(square / float(params.samples) - mean * mean, 0.0);
    const float se = sqrt(variance / float(params.samples));
    const float diff = abs(tMix - mean);
    sums[ray * 3 + 0] = diff;
    sums[ray * 3 + 1] = diff > 5.0 * se + params.tolerance ? 1.0 : 0.0;
    sums[ray * 3 + 2] = tau;
}

// --- Into the world, as a ParticleField's records ----------------------------------------

/// Eigen-decomposition of a symmetric 3x3 by Jacobi rotations: the values,
/// and the vectors as the columns of `v`.
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

float4 quaternionOf(float3x3 m) {   // m's columns are the axes; w x y z
    const float tr = m[0][0] + m[1][1] + m[2][2];
    float4 q;
    if (tr > 0.0) {
        const float s = sqrt(tr + 1.0) * 2.0;
        q = float4(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s);
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        q = float4((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s);
    } else if (m[1][1] > m[2][2]) {
        const float s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        q = float4((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s);
    } else {
        const float s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        q = float4((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s);
    }
    return normalize(q);
}

/// One thread a gaussian: 14 floats a record -- position, opacity, three
/// deviations, rotation w x y z, the colour (Cd, linear) -- then 5 halves,
/// one to a word: sigma (density, its peak), Cd rgb, temperature. In the world: the
/// covariance M Sigma M^T, M the linear part of index to world. The
/// opacity is what a standard viewer draws: 1 - exp(-tau) through the
/// gaussian's middle along its mean deviation, tau = sigma densityScale
/// sqrt(2 pi) s.
[shader("compute")]
[numthreads(64, 1, 1)]
void vsExport(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) return;
    const float4 r0 = gaussians[g * 4 + 0];
    const float4 r1 = gaussians[g * 4 + 1];
    const float4 r2 = gaussians[g * 4 + 2];
    const float a00 = r1.x, a10 = r1.y, a11 = r1.z, a20 = r1.w, a21 = r2.x, a22 = r2.y;
    // B = A^-1, Sigma = B^T B (index space).
    const float b00 = 1.0 / a00, b11 = 1.0 / a11, b22 = 1.0 / a22;
    const float b10 = -a10 * b00 / a11;
    const float b21 = -a21 * b11 / a22;
    const float b20 = -(a20 * b00 + a21 * b10) / a22;
    const float3x3 b = float3x3(b00, 0, 0, b10, b11, 0, b20, b21, b22);
    const float3x3 sigmaIndex = mul(transpose(b), b);
    const float3x3 m = float3x3(params.indexToWorld0.xyz, params.indexToWorld1.xyz, params.indexToWorld2.xyz);
    float3x3 cov = mul(m, mul(sigmaIndex, transpose(m)));
    float3x3 axes;
    jacobi3(cov, axes);
    float3 scale = sqrt(max(float3(cov[0][0], cov[1][1], cov[2][2]), 1e-20));
    if (determinant(axes) < 0.0) {
        axes[0][2] = -axes[0][2]; axes[1][2] = -axes[1][2]; axes[2][2] = -axes[2][2];
    }
    const float4 q = quaternionOf(axes);
    const float3 position = indexToWorld(r0.xyz);
    const float meanStd = pow(scale.x * scale.y * scale.z, 1.0 / 3.0);
    // Never under 1/255: a loader drops a gaussian fainter than that (the
    // 3DGS rule), and the export's own decode did, writing a faint one's
    // shape as zeros beside its sigma -- 61% of a 50k smoke at density 1.
    const float opacity = max(1.0 - exp(-r0.w * params.densityScale * 2.50662827 * meanStd), 1.0 / 254.0);
    const float3 colour = unpackColour(r2.w);
    const uint o = g * 14;
    exported[o + 0] = position.x; exported[o + 1] = position.y; exported[o + 2] = position.z;
    exported[o + 3] = opacity;
    exported[o + 4] = scale.x; exported[o + 5] = scale.y; exported[o + 6] = scale.z;
    exported[o + 7] = q.x; exported[o + 8] = q.y; exported[o + 9] = q.z; exported[o + 10] = q.w;
    exported[o + 11] = colour.x; exported[o + 12] = colour.y; exported[o + 13] = colour.z;
    // Then five halves a gaussian, one to a word (Export's `volumeHalves`).
    const uint e = params.count * 14 + g * 5;
    exported[e + 0] = asfloat(halfBits(r0.w));
    exported[e + 1] = asfloat(halfBits(colour.x));
    exported[e + 2] = asfloat(halfBits(colour.y));
    exported[e + 3] = asfloat(halfBits(colour.z));
    exported[e + 4] = asfloat(halfBits(r2.z));
}
