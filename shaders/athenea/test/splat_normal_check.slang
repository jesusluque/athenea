// Copyright (c) 2026 jesus luque.
//
// A GAUSSIAN'S SHADING NORMAL, CHECKED ON THE DEVICE (GpuSplats::normals).
//
// What a test reads back is counters: how many normals were compared and how
// many were not what they should be. The normals themselves -- what a cloud
// was given, what a file gave back, what a skeleton turned, what a merge
// made of several -- are compared here.
module splat_normal_check;
import athenea.common.packing;

struct NormalCheckParams {
    uint   count;
    uint   perSplat;
    uint   joints;
    uint   width;       ///< floats a value, for `normalFloatsUnit`
    float4 n1;          ///< `normalsInArc`: the two directions a merge may lie between
    float4 n2;
    float4 row0;        ///< the joint transform, rows of a 3x4: linear part and translation in w
    float4 row1;
    float4 row2;
    float  tolerance;
    uint   pad0;
    uint   pad1;
    uint   pad2;
};

/// The five normals a test cloud is given, record `i` taking `i % 5`: unit
/// vectors with exact decimal components, one of them facing away from +z.
/// tests/usd/test_usd.cpp writes the same table into its records.
float3 tableNormal(uint i) {
    const uint k = i % 5u;
    if (k == 0u) return float3(0.0, 0.0, 1.0);
    if (k == 1u) return float3(0.6, 0.0, 0.8);
    if (k == 2u) return float3(0.0, -0.6, 0.8);
    if (k == 3u) return float3(-0.48, 0.6, 0.64);
    return float3(0.0, 0.8, -0.6);
}

StructuredBuffer<uint>     normalsA;
StructuredBuffer<uint>     normalsB;
StructuredBuffer<float>    floats;
RWStructuredBuffer<uint>   counts;   ///< [compared, wrong, wrong side, ...]
ConstantBuffer<NormalCheckParams> params;

/// `normalsA` against the table, and against `normalsB` (another route to
/// the same cloud): [0] compared, [1] off the table, [2] apart from B.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalsAgainstTable(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3 a = unpackNormal(normalsA[i]);
    if (dot(a, tableNormal(i)) < 1.0 - params.tolerance) {
        InterlockedAdd(counts[1], 1u);
    }
    if (dot(a, unpackNormal(normalsB[i])) < 1.0 - params.tolerance) {
        InterlockedAdd(counts[2], 1u);
    }
}

/// Merged normals between two directions: each must be a unit vector in the
/// plane the two span and on the short arc between them, which is where any
/// weighted mean of the two, made unit, lies. [0] compared, [1] outside.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalsInArc(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3 n = unpackNormal(normalsA[i]);
    const float3 a = params.n1.xyz;
    const float3 b = params.n2.xyz;
    const float across = dot(a, b);
    const bool inPlane = abs(dot(n, normalize(cross(a, b)))) <= params.tolerance;
    const bool between = dot(n, a) >= across - params.tolerance && dot(n, b) >= across - params.tolerance;
    if (!inPlane || !between) {
        InterlockedAdd(counts[1], 1u);
    }
}

/// `width` floats a value, each value meant to be a unit vector (a merged
/// `primvars:athenea:splat:normal`): [0] compared, [1] not unit.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalFloatsUnit(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float length2 = 0.0;
    for (uint k = 0; k < params.width; ++k) {
        const float v = floats[i * params.width + k];
        length2 += v * v;
    }
    InterlockedAdd(counts[0], 1u);
    if (abs(sqrt(length2) - 1.0) > params.tolerance) {
        InterlockedAdd(counts[1], 1u);
    }
}

// --- skinning ------------------------------------------------------------------

RWStructuredBuffer<float4> restPositions;
RWStructuredBuffer<uint>   restShape;
RWStructuredBuffer<uint>   restNormals;
RWStructuredBuffer<float2> influences;
RWStructuredBuffer<float4> xforms;
StructuredBuffer<uint>     shape;     ///< what the skinner wrote
StructuredBuffer<uint>     normals;

/// A cloud of discs on a helix, each turned its own way, and each given a
/// shading normal 0.4 radians off its disc's axis towards its first axis --
/// as a normal map tilts one.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalSkinMake(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const float t = float(at) / max(float(params.count), 1.0);
    const float angle = t * 17.0;
    restPositions[at] = float4(cos(angle) * (1.0 + t), t * 4.0 - 2.0, sin(angle) * (1.0 + t), 0.9);
    const float3 axis = normalize(float3(0.3 + t, 1.0 - t, 0.2 + t * t));
    const float half = angle * 0.5;
    const float4 q = float4(axis * sin(half), cos(half));
    const uint row = at * 4;
    restShape[row] = encodeQuaternion(q);
    restShape[row + 1] = packHalves(log(0.05 + 0.2 * t), log(0.04 + 0.3 * (1.0 - t)));
    restShape[row + 2] = packHalves(log(1.0e-5), 0.5);
    restShape[row + 3] = packHalves(0.5, 0.5);
    // From the frame as it was packed, so the tilt is measured from the axis
    // the skinner reads and not from the one before quantisation.
    const float3x3 frame = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3 u = float3(frame[0][0], frame[1][0], frame[2][0]);
    const float3 n = float3(frame[0][2], frame[1][2], frame[2][2]);
    restNormals[at] = packNormal(normalize(n * cos(0.4) + u * sin(0.4)));
    for (uint k = 0; k < params.perSplat; ++k) {
        influences[at * params.perSplat + k] = float2(0.0, k == 0 ? 1.0 : 0.0);
    }
}

/// Every joint the same transform, rows of a 4x4.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalSkinJoints(uint3 tid: SV_DispatchThreadID) {
    const uint joint = tid.x;
    if (joint >= params.joints) {
        return;
    }
    xforms[joint * 4 + 0] = params.row0;
    xforms[joint * 4 + 1] = params.row1;
    xforms[joint * 4 + 2] = params.row2;
    xforms[joint * 4 + 3] = float4(0.0, 0.0, 0.0, 1.0);
}

/// What the skinner did to the normals: a normal goes by the inverse
/// transpose of the joint's linear part, so [1] counts those that did not;
/// and it must stay on the side of the disc it was on, so [2] counts those
/// whose side of the posed disc is not their side of the rest one.
[shader("compute")]
[numthreads(64, 1, 1)]
void normalSkinCompare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3x3 m = float3x3(params.row0.xyz, params.row1.xyz, params.row2.xyz);
    const float3 rest = unpackNormal(restNormals[at]);
    const float3 want = normalize(mul(transpose(inverse3(m)), rest));
    const float3 got = unpackNormal(normals[at]);
    if (dot(got, want) < 1.0 - params.tolerance) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint row = at * 4;
    const float3x3 was = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3x3 is = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float before = dot(rest, float3(was[0][2], was[1][2], was[2][2]));
    const float after = dot(got, float3(is[0][2], is[1][2], is[2][2]));
    if ((before > 0.0) != (after > 0.0)) {
        InterlockedAdd(counts[2], 1u);
    }
}

float3x3 inverse3(float3x3 m) {
    const float3 a = m[0], b = m[1], c = m[2];
    const float3 r0 = cross(b, c), r1 = cross(c, a), r2 = cross(a, b);
    const float det = dot(a, r0);
    // Columns of the inverse are the cross products over the determinant.
    return transpose(float3x3(r0, r1, r2)) / det;
}
