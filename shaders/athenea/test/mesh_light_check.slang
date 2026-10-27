// Copyright (c) 2026 jesus luque.
//
// Mesh lights against themselves on the device (tests/technique/test_mesh_lights.cpp):
// a planar cluster's samples binned over its own plane against the area
// density its triangles' shares say; each sample's reported density against
// the one meshClusterPdf finds for its direction; the tree's choice against
// the probability it reports.
module mesh_light_check;
import athenea.light.lights;
import athenea.light.light_bvh;
import athenea.light.mesh_lights;

struct MeshCheckParams {
    uint   bins;        // a side: bins x bins over the cluster's rectangle
    uint   samples;
    uint   cluster;
    uint   words;       // the region's
    float4 point;       // where the shading point is
    float4 normal;
};

StructuredBuffer<float>         region;
RWStructuredBuffer<uint>        binOf;       // per sample: its bin, bins^2 for none
RWStructuredBuffer<uint>        flags;       // per sample: bit 0 pdf differs, bit 1 choice differs
RWStructuredBuffer<float>       observed;
RWStructuredBuffer<float>       expected;
RWStructuredBuffer<float>       result;      // [chi2, dof, drawn fraction, pdf misses, choice misses, integral]
ConstantBuffer<MeshCheckParams> params;

float2 sampleAt2(uint i, uint salt) {
    uint h = (i * 2654435761u) ^ (salt * 0x9E3779B9u);
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const uint g = h * 1664525u + 1013904223u;
    return float2(float(h >> 8) * (1.0 / 16777216.0), float(g >> 8) * (1.0 / 16777216.0));
}

/// Where a point lies on the cluster's rectangle, as a bin.
uint binAt(MeshCluster m, float3 at) {
    const float3 d = at - m.centre;
    const float x = dot(d, m.axisX) / (2.0 * m.halfX) + 0.5;
    const float y = dot(d, m.axisY) / (2.0 * m.halfY) + 0.5;
    if (x < 0.0 || x >= 1.0 || y < 0.0 || y >= 1.0) {
        return params.bins * params.bins;
    }
    return min(uint(x * float(params.bins)), params.bins - 1u) * params.bins +
           min(uint(y * float(params.bins)), params.bins - 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshDraw(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.samples) {
        return;
    }
    const float2 a = sampleAt2(i, 1u);
    const float2 b = sampleAt2(i, 2u);
    const MeshLightSample s = sampleMeshCluster(region, params.words, params.cluster, params.point.xyz,
                                                float3(a, b.x));
    uint f = 0u;
    uint bin = params.bins * params.bins;
    if (s.valid) {
        bin = binAt(meshCluster(region, params.cluster), s.position);
        const float pdf = meshClusterPdf(region, params.cluster, params.point.xyz, s.wi);
        f |= abs(pdf - s.pdf) > 1.0e-3 * s.pdf ? 1u : 0u;
    }
    // The tree: what it chose, against the probability it says for that.
    const LightChoice c = chooseMeshCluster(region, params.words, params.point.xyz, params.normal.xyz, b.y);
    if (c.valid) {
        const float again = meshClusterChoicePdf(region, params.words, c.index, params.point.xyz, params.normal.xyz);
        f |= abs(again - c.probability) > 1.0e-4 * max(c.probability, 1.0e-6) ? 2u : 0u;
    }
    binOf[i] = bin;
    flags[i] = f;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void meshTally(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.bins * params.bins;
    for (uint k = 0u; k <= n; ++k) {
        observed[k] = 0.0;
    }
    uint pdfMisses = 0u;
    uint choiceMisses = 0u;
    for (uint i = 0u; i < params.samples; ++i) {
        observed[binOf[i]] += 1.0;
        pdfMisses += flags[i] & 1u;
        choiceMisses += (flags[i] >> 1) & 1u;
    }
    result[3] = float(pdfMisses);
    result[4] = float(choiceMisses);
}

/// The area density the cluster's shares say, integrated over each bin by
/// sub-cells: a triangle's share over its area wherever a sub-cell's centre
/// lies in it.
[shader("compute")]
[numthreads(64, 1, 1)]
void meshExpect(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.bins * params.bins;
    const uint b = tid.x;
    if (b >= n) {
        return;
    }
    const MeshCluster m = meshCluster(region, params.cluster);
    const float total = region[meshTriangleAt(region, m.first + m.count - 1u) + 3];
    const uint sub = 8u;
    const float cellX = 2.0 * m.halfX / float(params.bins * sub);
    const float cellY = 2.0 * m.halfY / float(params.bins * sub);
    const uint bx = b / params.bins;
    const uint by = b % params.bins;
    float sum = 0.0;
    for (uint j = 0u; j < sub; ++j) {
        for (uint k = 0u; k < sub; ++k) {
            const float x = -m.halfX + (float(bx * sub + j) + 0.5) * cellX;
            const float y = -m.halfY + (float(by * sub + k) + 0.5) * cellY;
            const float3 q = m.centre + m.axisX * x + m.axisY * y;
            for (uint t = 0u; t < m.count; ++t) {
                const MeshTriangle tri = meshTriangle(region, m.first + t);
                const float3 nrm = cross(tri.b - tri.a, tri.c - tri.a);
                const float3 e0 = cross(tri.b - tri.a, q - tri.a);
                const float3 e1 = cross(tri.c - tri.b, q - tri.b);
                const float3 e2 = cross(tri.a - tri.c, q - tri.c);
                if (dot(e0, nrm) >= 0.0 && dot(e1, nrm) >= 0.0 && dot(e2, nrm) >= 0.0) {
                    sum += tri.power / total / (0.5 * length(nrm)) * cellX * cellY;
                    break;
                }
            }
        }
    }
    expected[b] = sum * float(params.samples);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void meshStatistic(uint3 tid: SV_DispatchThreadID) {
    const uint n = params.bins * params.bins;
    float statistic = 0.0;
    float dof = 0.0;
    float pooledObserved = 0.0;
    float pooledExpected = 0.0;
    float drawn = 0.0;
    float integral = 0.0;
    for (uint b = 0u; b < n; ++b) {
        drawn += observed[b];
        integral += expected[b];
        if (expected[b] < 5.0) {
            pooledObserved += observed[b];
            pooledExpected += expected[b];
            continue;
        }
        statistic += (observed[b] - expected[b]) * (observed[b] - expected[b]) / expected[b];
        dof += 1.0;
    }
    if (pooledExpected >= 5.0) {
        statistic += (pooledObserved - pooledExpected) * (pooledObserved - pooledExpected) / pooledExpected;
        dof += 1.0;
    }
    result[0] = statistic;
    result[1] = max(dof - 1.0, 1.0);
    result[2] = drawn / float(params.samples);
    result[5] = integral / float(params.samples);
}

// ---- two frames, one a multiple of the other -----------------------------------

StructuredBuffer<float4>   frameA;
StructuredBuffer<float4>   frameB;
RWStructuredBuffer<uint>   scaledCounts;   // [pixels where scale a and b part beyond tolerance, pixels lit]
uniform uint               scaledPixels;
uniform float              scale;
uniform float              scaledTolerance;

/// Whether `scale` times frame a is frame b, pixel by pixel, to a relative
/// tolerance: what a level does to its group's light, being linear.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshScaled(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= scaledPixels) {
        return;
    }
    const float3 a = frameA[p].rgb * scale;
    const float3 b = frameB[p].rgb;
    if (any(b > float3(1.0e-6))) {
        InterlockedAdd(scaledCounts[1], 1u);
    }
    const float3 gap = abs(a - b) / max(abs(b), float3(1.0e-3));
    if (any(gap > float3(scaledTolerance))) {
        InterlockedAdd(scaledCounts[0], 1u);
    }
}

// ---- the groups' baked layers, added -------------------------------------------

struct LayerSumParams {
    uint  count;       // points
    uint  groupsA;     // float4 a point in layersA, added
    uint  groupsB;     // float4 a point in layersB, added; 0: layersB is a bake's constant term
    float tolerance;   // relative, of the larger of the reference and `floor`
    float floor;
    float lit;         // a reference brighter than this counts as lit
};

StructuredBuffer<float4>       layersA;
StructuredBuffer<float4>       layersB;
RWStructuredBuffer<uint>       layerCounts;   // [points apart, points lit, worst gap as float bits, red over green]
ConstantBuffer<LayerSumParams> layerSum;

/// Whether the sum of one bake's groups is another's (their groups added, or
/// a bake's colour, `0.5 + kSH0 * c0` as a cloud keeps its constant term),
/// point by point, to a tolerance: light being linear in its emitters.
[shader("compute")]
[numthreads(64, 1, 1)]
void meshLayerSum(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= layerSum.count) {
        return;
    }
    float3 a = float3(0.0);
    for (uint g = 0u; g < layerSum.groupsA; ++g) {
        a += layersA[i * layerSum.groupsA + g].rgb;
    }
    float3 b = float3(0.0);
    if (layerSum.groupsB == 0u) {
        b = 0.5 + 0.28209479177387814 * layersB[i].rgb;
    } else {
        for (uint g = 0u; g < layerSum.groupsB; ++g) {
            b += layersB[i * layerSum.groupsB + g].rgb;
        }
    }
    const float scale = max(max(b.r, max(b.g, b.b)), layerSum.floor);
    const float3 gap = abs(a - b) / scale;
    const float worst = max(gap.r, max(gap.g, gap.b));
    if (!(worst <= layerSum.tolerance)) {
        InterlockedAdd(layerCounts[0], 1u);
    }
    if (max(b.r, max(b.g, b.b)) > layerSum.lit) {
        InterlockedAdd(layerCounts[1], 1u);
    }
    InterlockedMax(layerCounts[2], asuint(isfinite(worst) ? worst : 3.0e38));
    if (a.r > 1.5 * a.g) {
        InterlockedAdd(layerCounts[3], 1u);
    }
}

// ---- two frames compared where a mask says ---------------------------------------

StructuredBuffer<float4>       maskedA;
StructuredBuffer<float4>       maskedB;
StructuredBuffer<float4>       maskFrame;    // where its red is over a half
RWStructuredBuffer<float>      maskedOut;    // [sum of relative squared error, pixels, a's rgb sum, b's rgb sum]
uniform uint                   maskedPixels;

/// The relative squared error (a - b)^2 / (b^2 + 0.01), its channels' mean,
/// summed over the masked pixels with the pixels counted and each frame's
/// colour summed: one thread, a frame's worth of pixels.
[shader("compute")]
[numthreads(1, 1, 1)]
void meshMaskedError(uint3 tid: SV_DispatchThreadID) {
    float error = 0.0;
    float count = 0.0;
    float3 sumA = float3(0.0);
    float3 sumB = float3(0.0);
    for (uint p = 0u; p < maskedPixels; ++p) {
        if (!(maskFrame[p].r > 0.5)) {
            continue;
        }
        const float3 a = maskedA[p].rgb;
        const float3 b = maskedB[p].rgb;
        const float3 e = (a - b) * (a - b) / (b * b + float3(0.01));
        error += (e.r + e.g + e.b) / 3.0;
        count += 1.0;
        sumA += a;
        sumB += b;
    }
    maskedOut[0] = error;
    maskedOut[1] = count;
    maskedOut[2] = sumA.r; maskedOut[3] = sumA.g; maskedOut[4] = sumA.b;
    maskedOut[5] = sumB.r; maskedOut[6] = sumB.g; maskedOut[7] = sumB.b;
}
