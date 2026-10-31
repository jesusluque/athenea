// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER'S FIELD, CHECKED ON THE DEVICE (SSF phase 1;
// lod/ssf_field.slang). Synthetic layers and the points to probe them are
// made here too; a test reads back counters only.
//   ssfSphere, ssfCube, ssfOneBlob   layers: blobs on a sphere (Fibonacci),
//                                    on a cube's faces, one blob alone
//   ssfPointsNear                    a point within each blob's support, in turn
//   ssfGradCheck     (a) the analytic gradient against central differences,
//                    where the stencil crosses no support's edge
//   ssfGridCheck     the grid's field against the sum over every blob
//   ssfIsolatedCheck (b) one blob's zero set is its ellipsoid scaled by
//                    sqrt(1 - (tau / w)^(1/3)); inside it F > 0, between it and
//                    the support F < 0, past the support (d) F = -tau exactly
//   ssfOutsideCheck  (d) where no blob reaches, F is -tau and grad F zero, exactly
//   ssfSminCheck     (c) the smooth minimum: symmetric bit for bit, the plain
//                    minimum where |a - b| >= k and at k = 0, its derivative
//                    the differences' and continuous where |a - b| = k
module ssf_field_check;
import athenea.common.packing;
import athenea.lod.ssf_field;

struct SsfFieldCheckParams {
    uint  count;       ///< blobs for a generator, points or samples for a check
    uint  blobs;       ///< the layer's, for ssfGridCheck and ssfPointsNear
    uint  side;        ///< ssfCube: blobs along a face's edge
    uint  seed;
    float size;        ///< sphere radius, cube half-edge
    float radius;      ///< a blob's tangent half-axes
    float thickness;   ///< c, the anchor's
    float tau;
    float step;        ///< central differences' step
    float gradFloor;   ///< the gradient a relative error is taken against at least
    float k;           ///< ssfSminCheck's
    float weight;      ///< ssfOneBlob's
    float4 centre;     ///< ssfOneBlob's centre (xyz) and theta (w)
    float4 normal;     ///< ssfOneBlob's normal (xyz) and second tangent half-axis (w)
};

RWStructuredBuffer<float4> outPositions;
RWStructuredBuffer<uint>   outShape;
RWStructuredBuffer<uint>   outNormals;
RWStructuredBuffer<uint>   outAnchors;
RWStructuredBuffer<float4> probes;
RWStructuredBuffer<uint>   wrong;   ///< [0] failed, [1] tested, [2] worst relative error x 1e6, [3] a second count
ConstantBuffer<SsfFieldCheckParams> params;

static const float kPi = 3.14159265358979;

uint pcg(uint v) {
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float unit(uint i, uint field) {
    return float(pcg(pcg(i * 16u + field) ^ params.seed) >> 8) * (1.0 / 16777216.0);
}

void writeBlob(uint i, float3 centre, float3 n, float2 radii, float theta, float weight) {
    SolidBlob b;
    b.position = centre;
    b.weight = weight;
    b.radii = radii;
    b.theta = theta;
    b.colour = float3(0.5);
    b.anchor = 0;
    float4 position;
    uint4  shape;
    packSolidBlob(b, position, shape);
    outPositions[i] = position;
    [unroll] for (uint k = 0; k < 4; ++k) {
        outShape[i * 4 + k] = shape[k];
    }
    outNormals[i] = packNormal(normalize(n));
    if (i == 0) {
        SolidAnchor a;
        a.thickness = params.thickness;
        a.smoothK = 0.0;
        a.appearance = kNoAppearance;
        a.flags = 0;
        const uint4 w = packAnchor(a);
        [unroll] for (uint k = 0; k < 4; ++k) {
            outAnchors[k] = w[k];
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfSphere(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    // Fibonacci: even area per blob.
    const float z = 1.0 - (2.0 * float(i) + 1.0) / float(params.count);
    const float phi = float(i) * kPi * (3.0 - sqrt(5.0));
    const float rho = sqrt(max(1.0 - z * z, 0.0));
    const float3 n = float3(rho * cos(phi), rho * sin(phi), z);
    writeBlob(i, n * params.size, n, float2(params.radius), kPi * unit(i, 0), 1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfCube(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint perFace = params.side * params.side;
    const uint face = i / perFace;
    const uint at = i % perFace;
    const float2 uv = (float2(at % params.side, at / params.side) + 0.5) / float(params.side) * 2.0 - 1.0;
    const uint axis = face / 2;
    const float sign = (face & 1u) != 0 ? -1.0 : 1.0;
    float3 n = float3(0.0);
    n[axis] = sign;
    float3 p = float3(0.0);
    p[axis] = sign;
    p[(axis + 1) % 3] = uv.x;
    p[(axis + 2) % 3] = uv.y;
    writeBlob(i, p * params.size, n, float2(params.radius), kPi * unit(i, 0), 1.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void ssfOneBlob(uint3 tid: SV_DispatchThreadID) {
    writeBlob(0, params.centre.xyz, params.normal.xyz, float2(params.radius, params.normal.w), params.centre.w,
              params.weight);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfPointsNear(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const FieldBlob f = fieldBlob(j % params.blobs);
    const float3 offset = float3(unit(j, 1), unit(j, 2), unit(j, 3)) * 2.0 - 1.0;
    probes[j] = float4(f.centre + offset * f.radius, 0.0);
}

/// One axis's derivative by the fourth-order central stencil,
/// (8 (f(h) - f(-h)) - (f(2h) - f(-2h))) / 12h: the second-order one truncates
/// by h^2 F(3) / 6, a few 1e-3 of the gradient with c = 0.05 at any step
/// float32's rounding allows; this one by h^4 F(5) / 30, under 1e-5.
float difference(float3 x, uint axis, float h) {
    float3 p1 = x, m1 = x, p2 = x, m2 = x;
    p1[axis] += h;
    m1[axis] -= h;
    p2[axis] += 2.0 * h;
    m2[axis] -= 2.0 * h;
    const float near = fieldAt(p1, params.tau).w - fieldAt(m1, params.tau).w;
    const float far = fieldAt(p2, params.tau).w - fieldAt(m2, params.tau).w;
    // The step the floats took: a quarter of the far span.
    const float step = 0.25 * (p2[axis] - m2[axis]);
    return (8.0 * near - far) / (12.0 * step);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfGradCheck(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const float3 x = probes[j].xyz;
    // A stencil that crosses a support's edge (where k is C2 only) is no
    // oracle: such a probe is counted apart, not tested.
    if (supportMargin(x) <= 2.0 * params.step * 1.5) {
        InterlockedAdd(wrong[3], 1u << 16);
        return;
    }
    const float4 an = fieldAt(x, params.tau);
    const float3 fd = float3(difference(x, 0, params.step), difference(x, 1, params.step),
                             difference(x, 2, params.step));
    const float relative = length(fd - an.xyz) / max(length(an.xyz), params.gradFloor);
    if (!(relative <= 1.0e-3)) {
        InterlockedAdd(wrong[0], 1u);
    }
    if (length(an.xyz) >= params.gradFloor) {
        InterlockedAdd(wrong[3], 1u);   // low 16 bits: steep probes (under 65536 probes)
    }
    InterlockedMax(wrong[2], uint(min(relative, 4000.0) * 1.0e6));
    InterlockedAdd(wrong[1], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfGridCheck(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const float3 x = probes[j].xyz;
    const float4 g = fieldAt(x, params.tau);
    const float4 e = fieldEverywhere(x, params.tau, params.blobs);
    // The same terms in another order: a float's rounding of each, no more.
    const float scale = 1.0 + abs(e.w) + params.tau;
    const float relative = max(abs(g.w - e.w) / scale, length(g.xyz - e.xyz) / max(length(e.xyz), params.gradFloor));
    if (!(relative <= 1.0e-5)) {
        InterlockedAdd(wrong[0], 1u);
    }
    if (e.w > -params.tau) {
        InterlockedAdd(wrong[3], 1u);   // points some blob reached
    }
    InterlockedMax(wrong[2], uint(min(relative, 4000.0) * 1.0e6));
    InterlockedAdd(wrong[1], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfIsolatedCheck(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const FieldBlob f = fieldBlob(0);
    // An even direction on the sphere, and that direction on the ellipsoid.
    const float z = unit(j, 4) * 2.0 - 1.0;
    const float phi = unit(j, 5) * 2.0 * kPi;
    const float rho = sqrt(max(1.0 - z * z, 0.0));
    const float3 dir = float3(rho * cos(phi), rho * sin(phi), z);
    const float3 axes = 1.0 / f.inverseAxes;
    const float3 onAxes = dir.x * axes.x * f.t1 + dir.y * axes.y * f.t2 + dir.z * axes.z * f.n;
    const float s = sqrt(1.0 - pow(params.tau / f.weight, 1.0 / 3.0));
    uint failed = 0;
    // On the zero set: F within float rounding of zero, against the weight.
    if (!(abs(fieldAt(f.centre + onAxes * s, params.tau).w) <= 2.0e-4 * f.weight)) {
        failed |= 1u;
    }
    if (!(fieldAt(f.centre + onAxes * (0.5 * s), params.tau).w > 0.0)) {
        failed |= 2u;
    }
    if (!(fieldAt(f.centre + onAxes * (0.5 * (s + 1.0)), params.tau).w < 0.0)) {
        failed |= 4u;
    }
    const float4 past = fieldAt(f.centre + onAxes * 1.01, params.tau);
    if (past.w != -params.tau || any(past.xyz != 0.0)) {
        failed |= 8u;
    }
    if (failed != 0) {
        InterlockedAdd(wrong[0], 1u);
        InterlockedOr(wrong[3], failed);
    }
    InterlockedAdd(wrong[1], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfOutsideCheck(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    // Half inside, near the centre of a closed layer of size `size` (the
    // sphere's or the cube's, where no blob reaches), half outside its bounds.
    const float3 r = float3(unit(j, 6), unit(j, 7), unit(j, 8)) * 2.0 - 1.0;
    const float3 x = (j & 1u) != 0 ? r * (0.5 * params.size) : (normalize(r + 1.0e-3) * (3.0 + r.x) * params.size);
    const float4 field = fieldAt(x, params.tau);
    if (field.w != -params.tau || any(field.xyz != 0.0)) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[1], 1u);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfSminCheck(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const float k = params.k;
    // Values of the order of k, so a step of k / 1000 is many of a float's.
    const float a = (unit(j, 9) * 2.0 - 1.0) * 4.0 * max(k, 1.0e-3);
    // Gaps over [-3k, 3k], and a sample on each side of |a - b| = k.
    const float t = (j % 4u) == 0 ? (k + 1.0e-4 * k) * ((j & 4u) != 0 ? 1.0 : -1.0)
                  : (j % 4u) == 1 ? (k - 1.0e-4 * k) * ((j & 4u) != 0 ? 1.0 : -1.0)
                                  : (unit(j, 10) * 6.0 - 3.0) * max(k, 1.0e-3);
    const float b = a + t;
    float dA, dB, dAs;
    const float m = smin(a, b, k, dA);
    const float mSwapped = smin(b, a, k, dB);
    uint failed = 0;
    if (asuint(m) != asuint(mSwapped)) {
        failed |= 1u;   // not symmetric bit for bit
    }
    if ((k <= 0.0 || abs(a - b) >= k) && m != min(a, b)) {
        failed |= 2u;   // not the plain minimum where it must be
    }
    if (k > 0.0) {
        // The derivative against central differences.
        const float h = 1.0e-3 * k;
        float unused;
        const float ap = a + h, am = a - h;
        const float fd = (smin(ap, b, k, unused) - smin(am, b, k, unused)) / (ap - am);
        if (!(abs(fd - dA) <= 2.0e-3)) {
            failed |= 4u;
        }
        // C1: the derivative a hair either side of |a - b| = k.
        const float sideA = b - k * (1.0 + 1.0e-4), sideB = b - k * (1.0 - 1.0e-4);
        float d0, d1;
        smin(sideA, b, k, d0);
        smin(sideB, b, k, d1);
        if (!(abs(d0 - d1) <= 1.0e-3)) {
            failed |= 8u;
        }
        // And the value: within k/4 of the minimum, never above it.
        if (!(m <= min(a, b) && m >= min(a, b) - 0.25 * k * (1.0 + 1.0e-5))) {
            failed |= 16u;
        }
    }
    if (failed != 0) {
        InterlockedAdd(wrong[0], 1u);
        InterlockedOr(wrong[3], failed);
    }
    InterlockedAdd(wrong[1], 1u);
}
