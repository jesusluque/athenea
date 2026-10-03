// Copyright (c) 2026 jesus luque.
//
// THE SPLAT BAKE FILTER: a baked cloud's indirect light, averaged between
// neighbours on its surface.
//
// A bake traces a few hundred paths at each gaussian, and the light that
// arrives after a bounce is where most of their noise is: the direct light
// is next event estimation, nearly noiseless and the one that draws the
// shadows; the indirect is whatever the bounce happened to find. Neighbouring
// gaussians on one surface see nearly the same bounced light, so averaging it
// between them buys what more paths would, without the paths -- as long as the
// average stops where the surface, the prim or the light itself changes.
//
// The neighbours are found in a hash grid: each gaussian goes into the cell
// of a grid as fine as a gaussian and a half (the median footprint, measured
// here), and an iteration reads the twenty-seven cells around it at a stride
// of 2^iteration cells -- a-trous (Dammertz et al. 2010; Schied et al.'s SVGF,
// 2017, for the variance), over the cloud's points instead of a picture's
// pixels. A hash, not a sort: the gaussians stay in the order they came, and
// what costs a sort elsewhere costs a counter here.
//
// What a neighbour is weighed by:
//   - the B3 spline of a-trous, a quarter, a half, a quarter along each axis;
//   - where it stands on this one's tangent plane (a gaussian of the stride),
//     and how far off that plane (a gaussian of half a cell, so the far side
//     of a thin wall is not a neighbour);
//   - how nearly it faces the same way (cos^sigmaNormal);
//   - whether it came from the same prim (its Cryptomatte id), so marble and
//     gold do not mix;
//   - how far its light is from this one's in units of this one's noise
//     (exp(-|l_i - l_j| / (sigmaLuminance sqrt(var_i)))): a gaussian whose
//     light is certain keeps it, and a step in the light that is larger than
//     the noise is an edge, which stays where it is.
// The variance is carried through the iterations as SVGF carries it: the sum
// of the squared weights times the neighbours' variances, over the squared
// sum of the weights; the luminance it is compared with, as the light is.
//
// The pictures, as the host lays them out (entries in rows of `width`, padded
// to `stride` pixels):
//   Points    3 entries a gaussian: (position, footprint), (normal, Cryptomatte
//             id as float bits), (the light's mean luminance over the paths,
//             the variance of that mean, -, -)
//   Indirect  `coefficients` entries a gaussian: the indirect light's
//             harmonics, rgb, fitted (the direct light never comes here)
//   Color     the same, filtered
module splatbakefilter;

struct BfParams {
    uint  count;
    uint  coefficients;
    uint  pointsWidth;
    uint  pointsStride;

    uint  indirectWidth;
    uint  indirectStride;
    uint  dstWidth;
    uint  dstStride;

    uint  gridSize;       ///< cells of the hash, a power of two
    uint  slots;          ///< gaussians a cell holds; the rest are counted and left out
    uint  iteration;      ///< 0, 1, 2, ...: the stride is 2^iteration cells
    uint  iterations;

    float sigmaNormal;
    float sigmaLuminance;
    float cellScale;      ///< the cell, in median footprints
    uint  pad0;
};

StructuredBuffer<float4>   points;
StructuredBuffer<float4>   indirectIn;
RWStructuredBuffer<float4> target;
RWStructuredBuffer<uint>   grid;       // (slots + 1) a cell: how many, then which
RWStructuredBuffer<float4> ping;       // coefficients a gaussian
RWStructuredBuffer<float4> pong;
RWStructuredBuffer<float2> guide;      // two a gaussian, after an even iteration and after an odd one: (luminance, its variance)
RWStructuredBuffer<uint>   counters;   // [0, 512) footprints by quarter octave, 512 left out, 513 the cell's bits
ConstantBuffer<BfParams>   params;

static const uint  kBuckets = 512;
static const uint  kLeftOut = 512;
static const uint  kCell = 513;

uint pictureIndex(uint entry, uint width, uint stride) {
    return (entry / width) * stride + entry % width;
}

float4 pointEntry(uint k, uint e) {
    return points[pictureIndex(k * 3 + e, params.pointsWidth, params.pointsStride)];
}

/// Coefficient `c` of gaussian `k` from where this iteration reads.
float4 sourceAt(uint k, uint c) {
    const uint entry = k * params.coefficients + c;
    if (params.iteration == 0) {
        return indirectIn[pictureIndex(entry, params.indirectWidth, params.indirectStride)];
    }
    return (params.iteration & 1u) == 1u ? ping[entry] : pong[entry];
}

void writeAt(uint k, uint c, float4 value) {
    const uint entry = k * params.coefficients + c;
    if (params.iteration + 1 >= params.iterations) {
        target[pictureIndex(entry, params.dstWidth, params.dstStride)] = value;
    } else if ((params.iteration & 1u) == 0u) {
        ping[entry] = value;
    } else {
        pong[entry] = value;
    }
}

/// THE LUMINANCE AND ITS VARIANCE that an iteration weighs by: the host's
/// guides at first, then what the last iteration made of them. Not read off
/// the harmonics' constant term: a fit over the half of the sphere a surface
/// faces puts in its constant whatever the far half needs, and two neighbours
/// a few degrees apart have constants that differ by more than their light.
float2 guideAt(uint k) {
    if (params.iteration == 0) {
        const float4 g = pointEntry(k, 2);
        return float2(g.x, max(g.y, 0.0));
    }
    return guide[((params.iteration - 1) & 1u) * params.count + k];
}

int3 cellOf(float3 p, float cell) {
    return int3(floor(p / cell));
}

uint hashCell(int3 c) {
    const uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u);
    return h & (params.gridSize - 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bfClear(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k < params.gridSize) {
        grid[k * (params.slots + 1)] = 0;
    }
    if (k < kCell + 1) {
        counters[k] = 0;
    }
}

/// The footprints, by quarter octave: the cell is settled from their median,
/// which a few gaussians of a coarser level cannot move.
[shader("compute")]
[numthreads(64, 1, 1)]
void bfMeasure(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float footprint = pointEntry(k, 0).w;
    if (!(footprint > 0.0)) {
        return;
    }
    const int bucket = clamp(int(floor(4.0 * log2(footprint))) + 256, 0, int(kBuckets) - 1);
    InterlockedAdd(counters[bucket], 1u);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bfCell(uint3 tid: SV_DispatchThreadID) {
    uint total = 0;
    for (uint b = 0; b < kBuckets; ++b) {
        total += counters[b];
    }
    uint seen = 0;
    uint median = 256;
    for (uint b = 0; b < kBuckets; ++b) {
        seen += counters[b];
        if (total > 0 && seen * 2 >= total) {
            median = b;
            break;
        }
    }
    // The middle of the quarter octave; with no footprints at all, one unit.
    const float footprint = total > 0 ? exp2((float(median) - 256.0 + 0.5) / 4.0) : 1.0;
    counters[kCell] = asuint(footprint * params.cellScale);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bfInsert(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float cell = asfloat(counters[kCell]);
    const uint h = hashCell(cellOf(pointEntry(k, 0).xyz, cell));
    uint slot;
    InterlockedAdd(grid[h * (params.slots + 1)], 1u, slot);
    if (slot < params.slots) {
        grid[h * (params.slots + 1) + 1 + slot] = k;
    } else {
        InterlockedAdd(counters[kLeftOut], 1u);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bfIterate(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint count = min(params.coefficients, 16u);
    const float cell = asfloat(counters[kCell]);
    const float4 e0 = pointEntry(i, 0);
    const float4 e1 = pointEntry(i, 1);
    const float3 pi = e0.xyz;
    const float3 ni = normalize(e1.xyz);
    const uint idi = asuint(e1.w);
    const float2 gi = guideAt(i);
    const float li = gi.x;
    const float vi = gi.y;
    const uint step = 1u << params.iteration;
    const float spread = float(step) * cell;
    const float offPlane = 0.5 * cell;
    const float noise = params.sigmaLuminance * sqrt(vi) + 1.0e-6;
    const int3 home = cellOf(pi, cell);

    float3 sum[16];
    for (uint c = 0; c < 16; ++c) {
        sum[c] = float3(0.0);
    }
    float weights = 0.0;
    float weightedVariance = 0.0;
    float weightedLuminance = 0.0;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const float spline = (dx == 0 ? 0.5 : 0.25) * (dy == 0 ? 0.5 : 0.25) * (dz == 0 ? 0.5 : 0.25);
                const uint h = hashCell(home + int3(dx, dy, dz) * int(step));
                const uint held = min(grid[h * (params.slots + 1)], params.slots);
                for (uint s = 0; s < held; ++s) {
                    const uint j = grid[h * (params.slots + 1) + 1 + s];
                    const float4 f0 = pointEntry(j, 0);
                    const float4 f1 = pointEntry(j, 1);
                    if (asuint(f1.w) != idi) {
                        continue;
                    }
                    // A hash shares a cell between places: what stands in
                    // another one is weighed by where it really is.
                    const float3 d = f0.xyz - pi;
                    const float along = dot(d, ni);
                    const float across = max(dot(d, d) - along * along, 0.0);
                    float w = spline * exp(-across / (2.0 * spread * spread) - along * along / (2.0 * offPlane * offPlane));
                    w *= pow(saturate(dot(ni, normalize(f1.xyz))), params.sigmaNormal);
                    if (!(w > 1.0e-6)) {
                        continue;
                    }
                    const float2 gj = guideAt(j);
                    w *= exp(-abs(li - gj.x) / noise);
                    if (!(w > 1.0e-6)) {
                        continue;
                    }
                    for (uint c = 0; c < count; ++c) {
                        sum[c] += w * sourceAt(j, c).xyz;
                    }
                    weights += w;
                    weightedVariance += w * w * gj.y;
                    weightedLuminance += w * gj.x;
                }
            }
        }
    }
    if (!(weights > 0.0)) {
        // Not even itself (a cell full before it came): it keeps its own.
        for (uint c = 0; c < count; ++c) {
            writeAt(i, c, sourceAt(i, c));
        }
        guide[(params.iteration & 1u) * params.count + i] = gi;
        return;
    }
    for (uint c = 0; c < count; ++c) {
        writeAt(i, c, float4(sum[c] / weights, 0.0));
    }
    guide[(params.iteration & 1u) * params.count + i] =
        float2(weightedLuminance / weights, weightedVariance / (weights * weights));
}
