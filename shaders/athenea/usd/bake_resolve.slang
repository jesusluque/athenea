// Copyright (c) 2026 jesus luque.
//
// A BAKE KEPT AS SUMS, AND WHAT IS DONE WITH THEM BEFORE THEY ARE A FIT
// (`StageRenderer::bakeSplitOnDevice`, `StageRenderer::resolveBake`).
//
// The path tracer, asked for a split bake (PathParams.bakeSplit), writes sums
// and no fit: a gaussian's direct light against each harmonic, its indirect
// light against each, the brightest sample with the opacity summed, the
// luminance's moments and what the paths cost. Kept as sums they add: a
// second pass of paths at the gaussians that need it is added on, and the fit
// is made once, over every path a gaussian took. A gaussian's entries lie
// together, `2 * coefficients + 3` float4:
//
//   [0, C)        sum of direct light times each basis function (rgb)
//   [C, 2C)       the same of the indirect light
//   2C            brightest sample (rgb), opacity summed (w)
//   2C + 1        sum lumD, sum lumD^2, sum lumI, sum lumI^2
//   2C + 2        sum lumD * lumI, steps the paths took, paths, 0
//
// The kernels here, in the order a bake runs them:
//
//   bakeSumsAdd    a pass's planes into the sums, at a gaussian's own place
//                  (and, for a pass over some of them, at the place `slots`
//                  says); added to what is there, the brightest maxed.
//   bakeWeigh      how much more each gaussian is worth: sqrt(relative
//                  variance / cost), MARS's allocation with the gaussian as
//                  the cell (arXiv 2410.20429), and a workgroup's sum of it.
//   bakeWeighTotal the workgroups' sums summed.
//   bakeAllot      the extra passes each gaussian gets, in proportion.
//   bakeSelect     the gaussians a pass is for, and their rays, packed.
//   bakeSplitFit   each half fitted on its own (the fit is linear, so the
//                  two add to the fit of the whole), and what a filter
//                  wants to know: the indirect luminance and the variance
//                  of its mean, and the same of the whole light.
//   bakeCombine    direct plus indirect, bounded and made whole as the
//                  tracer's own bake is (bakeEncode), in the layout
//                  `bakePointsOnDevice` answers in.
module bake_resolve;
import athenea.common.sh;
import athenea.common.bake_fit;

struct ResolveParams {
    uint  count;          ///< gaussians (or, adding a pass, the pass's points)
    uint  coefficients;   ///< harmonics fitted, 1 to 16
    uint  first;          ///< bakeSumsAdd: the pass's first gaussian
    uint  plane;          ///< bakeSumsAdd: float4 a plane of the pass's grid
    uint  indexed;        ///< bakeSumsAdd: the pass's point k is gaussian slots[first + k]
    uint  accumulate;     ///< bakeSumsAdd: 0 the first pass writes, 1 adds
    uint  pass;           ///< bakeSelect: gaussians allotted at least this many extra passes
    uint  maxPasses;      ///< bakeAllot: no gaussian gets more
    float budgetPasses;   ///< bakeAllot: extra passes to share out, over all of them
    float meanFloor;      ///< bakeWeigh: a mean below this is weighed as this, so black is not infinitely noisy
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float4>    planes;     // a pass's output, a plane an entry
RWStructuredBuffer<float4>  sums;       // 2C + 3 a gaussian
StructuredBuffer<float4>    rays;       // 3 a gaussian: point, surface normal (w footprint), own facing (w 1)
RWStructuredBuffer<float4>  passRays;   // bakeSelect: the pass's rays, packed
RWStructuredBuffer<uint>    slots;      // which gaussian a pass's point is
RWStructuredBuffer<float>   weights;    // a gaussian's worth
RWStructuredBuffer<float>   partials;   // a workgroup's sum of it, then [0] the total
RWStructuredBuffer<uint>    allot;      // a gaussian's extra passes
RWStructuredBuffer<uint>    counter;    // [0] points selected
RWStructuredBuffer<float4>  direct;     // C a gaussian: the direct half, fitted
RWStructuredBuffer<float4>  indirect;   // C a gaussian: the indirect half, fitted
RWStructuredBuffer<float4>  guides;     // 1 a gaussian: indirect luminance and its mean's variance, then the whole light's
RWStructuredBuffer<float4>  answer;     // C a gaussian: the bake, as bakePointsOnDevice gives it
ConstantBuffer<ResolveParams> params;

uint entriesOf(uint c) {
    return 2 * c + 3;
}

/// Which way gaussian `k` faces: its own where it said one, else the surface's
/// (the path tracer's bakeFacing, the normal its fit was made about).
float3 facingOf(uint k) {
    const float4 own = rays[k * 3 + 2];
    return own.w > 0.5 ? normalize(own.xyz) : normalize(rays[k * 3 + 1].xyz);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeSumsAdd(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint entries = entriesOf(params.coefficients);
    const uint dest = params.indexed != 0 ? slots[params.first + k] : params.first + k;
    const uint into = dest * entries;
    const uint brightestAt = 2 * params.coefficients;
    for (uint e = 0; e < entries; ++e) {
        const float4 value = planes[e * params.plane + k];
        if (params.accumulate == 0) {
            sums[into + e] = value;
        } else if (e == brightestAt) {
            const float4 kept = sums[into + e];
            sums[into + e] = float4(max(kept.xyz, value.xyz), kept.w + value.w);
        } else {
            sums[into + e] = sums[into + e] + value;
        }
    }
}

groupshared float gWeigh[256];

/// HOW MUCH A GAUSSIAN IS WORTH ANOTHER PATH.
///
/// The error that matters is relative -- what the eye and relMSE see -- so a
/// gaussian's spread is taken over its mean squared. With a budget of paths
/// to share among cells of variance v and cost c, the allocation that
/// minimises the summed variance of the means is n proportional to
/// sqrt(v / c) (MARS, arXiv 2410.20429, section 4, with the gaussian as the
/// cell and its paths as the one technique). The cost is the steps its
/// paths took: a gaussian whose paths bounce about a glass costs more than
/// one that sees the sky at once.
[shader("compute")]
[numthreads(256, 1, 1)]
void bakeWeigh(uint3 tid: SV_DispatchThreadID, uint3 lid: SV_GroupThreadID, uint3 gid: SV_GroupID) {
    const uint k = tid.x;
    float w = 0.0;
    if (k < params.count) {
        const uint entries = entriesOf(params.coefficients);
        const uint at = k * entries + 2 * params.coefficients;
        const float4 moments = sums[at + 1];
        const float4 tally = sums[at + 2];
        const float n = tally.z;
        if (n > 1.0 && sums[at].w > 0.0) {
            const float mean = (moments.x + moments.z) / n;
            const float second = (moments.y + moments.w + 2.0 * tally.x) / n;
            const float variance = max(second - mean * mean, 0.0) * n / (n - 1.0);
            const float scale = max(mean, params.meanFloor);
            const float cost = max(tally.y / n, 1.0);
            w = sqrt(variance / (scale * scale) / cost);
        }
        weights[k] = w;
    }
    gWeigh[lid.x] = w;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid.x < stride) {
            gWeigh[lid.x] += gWeigh[lid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x == 0) {
        partials[1 + gid.x] = gWeigh[0];
    }
}

/// The workgroups' sums into partials[0]: one workgroup, each thread a
/// stride of them, then the tree.
[shader("compute")]
[numthreads(256, 1, 1)]
void bakeWeighTotal(uint3 lid: SV_GroupThreadID) {
    const uint groups = (params.count + 255) / 256;
    float sum = 0.0;
    for (uint g = lid.x; g < groups; g += 256) {
        sum += partials[1 + g];
    }
    gWeigh[lid.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lid.x < stride) {
            gWeigh[lid.x] += gWeigh[lid.x + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid.x == 0) {
        partials[0] = gWeigh[0];
    }
}

uint hashOf(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

/// THE EXTRA PASSES A GAUSSIAN GETS: its share of the budget, in whole
/// passes, rounded up or down at random with the fraction as the odds -- so
/// the budget is spent as asked on average, and a gaussian whose share is a
/// tenth of a pass gets one one time in ten rather than never.
[shader("compute")]
[numthreads(64, 1, 1)]
void bakeAllot(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float total = partials[0];
    const float share = total > 0.0 ? weights[k] / total * params.budgetPasses : 0.0;
    const float odds = float(hashOf(k * 9781u + 6271u) >> 8) / 16777216.0;
    allot[k] = min(uint(floor(share + odds)), params.maxPasses);
}

/// The gaussians pass `pass` is for -- every one allotted at least that many
/// -- packed, with their rays, so the tracer runs over them alone.
[shader("compute")]
[numthreads(64, 1, 1)]
void bakeSelect(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count || allot[k] < params.pass) {
        return;
    }
    uint at;
    InterlockedAdd(counter[0], 1u, at);
    slots[at] = k;
    passRays[at * 3] = rays[k * 3];
    passRays[at * 3 + 1] = rays[k * 3 + 1];
    passRays[at * 3 + 2] = rays[k * 3 + 2];
}

/// Each half fitted over every path the gaussian took (bakeFit: the half of
/// the sphere it faces), and the indirect half's luminance with the variance
/// of its mean -- what a filter between neighbours weighs by.
[shader("compute")]
[numthreads(64, 1, 1)]
void bakeSplitFit(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint count = min(params.coefficients, 16u);
    const uint at = k * entriesOf(count);
    const float4 tally = sums[at + 2 * count + 2];
    const float4 moments = sums[at + 2 * count + 1];
    const float n = max(tally.z, 1.0);
    uint evens[6];
    uint odds[10];
    uint ne = 0;
    uint no = 0;
    for (uint c = 0; c < count; ++c) {
        if ((bakeBand(c) & 1u) == 0u) {
            evens[ne++] = c;
        } else {
            odds[no++] = c;
        }
    }
    float e[6][10];
    bakeGram(facingOf(k), e, evens, ne, odds, no);
    float3 d[16];
    float3 i[16];
    for (uint c = 0; c < 16; ++c) {
        d[c] = c < count ? sums[at + c].xyz : float3(0.0);
        i[c] = c < count ? sums[at + count + c].xyz : float3(0.0);
    }
    bakeFit(d, e, evens, ne, odds, no, kBakeMeasure / n);
    bakeFit(i, e, evens, ne, odds, no, kBakeMeasure / n);
    for (uint c = 0; c < count; ++c) {
        direct[k * count + c] = float4(d[c], 0.0);
        indirect[k * count + c] = float4(i[c], 0.0);
    }
    const float mean = moments.z / n;
    const float variance = max(moments.w / n - mean * mean, 0.0) / max(n - 1.0, 1.0);
    const float whole = (moments.x + moments.z) / n;
    const float wholeSquares = (moments.y + moments.w + 2.0 * tally.x) / n;
    const float wholeVariance = max(wholeSquares - whole * whole, 0.0) / max(n - 1.0, 1.0);
    guides[k] = float4(mean, variance, whole, wholeVariance);
}

/// The two halves summed, bounded by the brightest sample and made whole
/// (bakeEncode), the constant shifted to where a cloud keeps its DC: what the
/// tracer's own bake writes, in the layout bakePointsOnDevice answers in.
[shader("compute")]
[numthreads(64, 1, 1)]
void bakeCombine(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint count = min(params.coefficients, 16u);
    const uint at = k * entriesOf(count);
    const float4 brightest = sums[at + 2 * count];
    const float n = max(sums[at + 2 * count + 2].z, 1.0);
    float3 c[16];
    for (uint j = 0; j < 16; ++j) {
        c[j] = j < count ? direct[k * count + j].xyz + indirect[k * count + j].xyz : float3(0.0);
    }
    bakeEncode(c, count, facingOf(k), brightest.xyz);
    for (uint j = 0; j < count; ++j) {
        float3 value = c[j];
        if (j == 0) {
            value -= float3(kShDcOffset);
        }
        answer[k * count + j] = float4(value, brightest.w / n);
    }
}
