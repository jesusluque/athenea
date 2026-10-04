// Copyright (c) 2026 jesus luque.
//
// WHAT `athenea flatten` IS CHECKED WITH (tests/render/test_flatten.cpp).
//
// `flattenCheckFit`: degree 3 harmonics known in the file's frame, a
// gaussian's frame drawn at random, what they show along the N local
// directions fed through the fit's matrix and turned into the file's frame
// (flatten_basis.slang) -- they have to come back as they went in, for a
// surface's weighting and a thin wall's. A wrong sign, order or direction in
// the basis, the matrix or the turn shows as an error of the order of the
// coefficients themselves.
//
// `flattenCheckRecords`: flattened records of made-up gaussians, for the
// packers to write and the readers to read back.
module flatten_check;
import athenea.splat.flatten_basis;

struct CheckParams {
    uint  cases;
    uint  directions;
    uint  cls;          // 0: a surface, 1: a thin wall
    float tolerance;
    uint  count;        // flattenCheckRecords
};
ConstantBuffer<CheckParams> params;
StructuredBuffer<float>     basisRows;
RWStructuredBuffer<uint>    stats;     // [0] cases over tolerance, [1] worst error in millionths
RWStructuredBuffer<float>   records;

float hashFloat(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0;
}

float rand(uint seed, uint k) {
    return hashFloat(seed * 977u + k * 7919u + 13u);
}

float3 randomDirection(uint seed, uint k) {
    const float z = rand(seed, k) * 2.0 - 1.0;
    const float phi = rand(seed, k + 1) * 6.283185307179586;
    const float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void flattenCheckFit(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.cases) {
        return;
    }
    float3 truth[16];
    for (uint k = 0; k < 16; ++k) {
        truth[k] = float3(rand(c, 3 * k), rand(c, 3 * k + 1), rand(c, 3 * k + 2)) - 0.5;
    }
    // The gaussian's frame in the file's.
    const float3 n = randomDirection(c, 100);
    float3 t, b;
    flatFrame(n, t, b);
    float3 local[16];
    for (uint k = 0; k < 16; ++k) {
        local[k] = float3(0.0);
    }
    const uint dirs = params.directions;
    for (uint k = 0; k < dirs; ++k) {
        const float3 l = fibonacciDirection(k, dirs);
        const float3 v = -(t * l.x + b * l.y + n * l.z);   // from the eye, in the file's frame
        float y[16];
        flatBasis(v, y);
        float3 g = float3(0.0);
        for (uint j = 0; j < 16; ++j) {
            g += truth[j] * y[j];
        }
        const uint at = (params.cls * dirs + k) * 16;
        for (uint j = 0; j < 16; ++j) {
            local[j] += basisRows[at + j] * g;
        }
    }
    float3 turned[16];
    flatRotate(local, float3x3(t, b, n), turned);
    float worst = 0.0;
    for (uint k = 0; k < 16; ++k) {
        const float3 e = abs(turned[k] - truth[k]);
        worst = max(worst, max(e.x, max(e.y, e.z)));
    }
    if (!(worst <= params.tolerance)) {
        InterlockedAdd(stats[0], 1u);
    }
    InterlockedMax(stats[1], uint(min(worst, 4000.0) * 1.0e6));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void flattenCheckRecords(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint at = g * kFlatStride;
    for (uint k = 0; k < kFlatStride; ++k) {
        records[at + k] = 0.0;
    }
    for (uint a = 0; a < 3; ++a) {
        records[at + kFlatPosition + a] = rand(g, a) * 2.0 - 1.0;
        records[at + kFlatLogScale + a] = log(0.02 + 0.1 * rand(g, 3 + a));
    }
    records[at + kFlatOpacity] = 0.3 + 0.65 * rand(g, 6);
    const float4 q = normalize(float4(rand(g, 7), rand(g, 8), rand(g, 9), rand(g, 10)) - 0.5);
    for (uint a = 0; a < 4; ++a) {
        records[at + kFlatRotation + a] = q[a];
    }
    for (uint k = 0; k < 16; ++k) {
        for (uint ch = 0; ch < 3; ++ch) {
            const float r = rand(g, 11 + k * 3 + ch);
            records[at + kFlatSh + k * 3 + ch] = k == 0 ? r * 2.5 - 1.0 : (r - 0.5) * 1.2;
        }
    }
    records[at + kFlatNormal + 2] = 1.0;
    records[at + kFlatShrink] = 1.0;
}
