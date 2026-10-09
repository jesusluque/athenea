// Copyright (c) 2026 jesus luque.
//
// What a merge by pairs left (lod::Decimator::decimatePairs), checked against
// the records each kept gaussian stands for: counters of violations, which
// must come back zero, and of surfels.
//   [0] mass not kept: W, the members' opacity times area summed, against the
//       kept one's coverage times its area (its opacity is the members' mean,
//       drawn as they overlapped; the coverage is its mass)
//   [6] an opacity off its members' area-weighted mean (held at 0.99)
//   [1] a shape out of range: opacity outside (0, 1], a centre or scale not
//       finite, a negative scale, more than one axis at 0
//   [2] a covariance not symmetric positive semidefinite
//   [3] a merge across ids
//   [4] a merged float (the records' x) outside its members' range
//   [5] surfels (one scale exactly 0)
module pairs_check;
import athenea.common.packing;

struct PairsCheckParams {
    uint  count;
    float tolerance;
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float4>  positions;        // the kept cloud's
StructuredBuffer<uint>    shape;
StructuredBuffer<uint2>   ranges;
StructuredBuffer<float4>  storePositions;   // the store in cluster order
StructuredBuffer<uint>    storeShape;
StructuredBuffer<uint>    recordOf;         // per permuted store splat
StructuredBuffer<uint>    ids;              // per record
StructuredBuffer<uint>    xs;               // per record
StructuredBuffer<uint>    meanX;            // per kept gaussian
StructuredBuffer<float>   coverage;         // per kept gaussian
RWStructuredBuffer<uint>  counts;
ConstantBuffer<PairsCheckParams> params;

float3 scalesOf(uint row1, uint row2) {
    return exp(float3(lowHalf(row1), highHalf(row1), lowHalf(row2)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pairsCheck(uint3 tid: SV_DispatchThreadID) {
    const uint o = tid.x;
    if (o >= params.count) {
        return;
    }
    const float4 p = positions[o];
    const float3 s = scalesOf(shape[o * 4 + 1], shape[o * 4 + 2]);
    const uint zeros = uint(s.x == 0.0) + uint(s.y == 0.0) + uint(s.z == 0.0);
    if (!(p.w > 0.0 && p.w <= 1.0) || !all(isfinite(p.xyz)) || !all(isfinite(s)) || any(s < 0.0) || zeros > 1u) {
        InterlockedAdd(counts[1], 1u);
    }
    if (zeros == 1u) {
        InterlockedAdd(counts[5], 1u);
    }
    // Sigma = R diag(s^2) R^T: symmetric, and every principal minor at or
    // above nothing (up to rounding).
    const float3x3 r = axesOfQuaternion(decodeQuaternion(shape[o * 4]));
    float3x3 sigma = float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (uint k = 0; k < 3; ++k) {
        const float3 a = float3(r[0][k], r[1][k], r[2][k]);
        sigma += s[k] * s[k] * float3x3(a.x * a, a.y * a, a.z * a);
    }
    const float trace = sigma[0][0] + sigma[1][1] + sigma[2][2];
    const float eps = 1.0e-5 * trace * trace;
    const bool symmetric = abs(sigma[0][1] - sigma[1][0]) <= 1.0e-5 * trace &&
                           abs(sigma[0][2] - sigma[2][0]) <= 1.0e-5 * trace &&
                           abs(sigma[1][2] - sigma[2][1]) <= 1.0e-5 * trace;
    const bool semidefinite = sigma[0][0] >= 0.0 && sigma[1][1] >= 0.0 && sigma[2][2] >= 0.0 &&
                              sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[0][1] >= -eps &&
                              sigma[1][1] * sigma[2][2] - sigma[1][2] * sigma[1][2] >= -eps &&
                              sigma[0][0] * sigma[2][2] - sigma[0][2] * sigma[0][2] >= -eps &&
                              determinant(sigma) >= -eps * trace;
    if (!symmetric || !semidefinite) {
        InterlockedAdd(counts[2], 1u);
    }
    const uint2 run = ranges[o];
    float w = 0.0;
    float areas = 0.0;
    float alphas = 0.0;
    float lo = 1.0e30;
    float hi = -1.0e30;
    const uint id = ids[recordOf[run.x]];
    bool crossed = false;
    for (uint m = run.x; m < run.y; ++m) {
        const float3 ms = scalesOf(storeShape[m * 4 + 1], storeShape[m * 4 + 2]);
        w += storePositions[m].w * discArea(ms);
        areas += discArea(ms);
        alphas += storePositions[m].w * discArea(ms);
        const float x = asfloat(xs[recordOf[m]]);
        lo = min(lo, x);
        hi = max(hi, x);
        crossed = crossed || ids[recordOf[m]] != id;
    }
    // Two scales as f16 logarithms: a step of 1/256 near e^-4, so 2% of area.
    const float kept = coverage[o] * discArea(s);
    if (run.y - run.x > 1u && abs(kept - w) > 2.0e-2 * w) {
        InterlockedAdd(counts[0], 1u);
    }
    const float mean = min(alphas / max(areas, 1.0e-30), 0.99);
    if (run.y - run.x > 1u && abs(p.w - mean) > 1.0e-3) {
        InterlockedAdd(counts[6], 1u);
    }
    if (crossed) {
        InterlockedAdd(counts[3], 1u);
    }
    const float mx = asfloat(meanX[o]);
    if (mx < lo - params.tolerance || mx > hi + params.tolerance) {
        InterlockedAdd(counts[4], 1u);
    }
}
