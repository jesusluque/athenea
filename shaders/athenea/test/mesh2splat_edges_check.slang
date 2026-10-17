// Copyright (c) 2026 jesus luque.
//
// A gaussian reaches no further than its surface goes (m2sHoldEdges): the
// unit quad of mesh2splat_check with its four outer edges marked hard -- each
// triangle's own as bits, the other's as lines -- and its diagonal not, and
// every gaussian the effect wrote checked here. Counters only cross back.
module mesh2splat_edges_check;

struct EdgeCheckParams {
    uint  meshWidth;      ///< the Edges picture: the mesh picture's shape
    uint  meshStride;
    uint  recordPixels;
    uint  splats;

    uint  dstWidth;
    uint  dstStride;
    float sigma;          ///< a gaussian's size with nothing held: sigma over the resolution
    float reach;          ///< the effect's edgeReach

    float tolerance;
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

RWStructuredBuffer<float4> edges;     ///< written by `m2sQuadEdges`
StructuredBuffer<float4>   records;   ///< what the effect wrote
RWStructuredBuffer<uint>   counts;    ///< [past an edge, held inside, lost along an edge, held at all]
ConstantBuffer<EdgeCheckParams> params;

uint edgeAt(uint entry) {
    return (entry / params.meshWidth) * params.meshStride + entry % params.meshWidth;
}

uint recordAt(uint slot, uint component) {
    const uint entry = slot * params.recordPixels + component;
    return (entry / params.dstWidth) * params.dstStride + entry % params.dstWidth;
}

/// mesh2splat_check's quad is (0 1 2) and (0 2 3): the first triangle's
/// own edges 0 (y = 0) and 1 (x = 1) are outer, its 2 the diagonal; the
/// second's 0 is the diagonal, its 1 (y = 1) and 2 (x = 0) outer. Each is
/// told the other's outer edges as lines, since they share both ends of the
/// diagonal (mesh2splat_edges).
[shader("compute")]
[numthreads(1, 1, 1)]
void m2sQuadEdges(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    edges[edgeAt(0)] = float4(2.0, 3.0, 0.0, 0.0);
    edges[edgeAt(1)] = float4(1.0, 0.0, 0.0, 0.0);     // x = 0, inside +x
    edges[edgeAt(2)] = float4(0.0, -1.0, 0.0, -1.0);   // y = 1, inside -y
    edges[edgeAt(6)] = float4(2.0, 6.0, 0.0, 0.0);
    edges[edgeAt(7)] = float4(0.0, 1.0, 0.0, 0.0);     // y = 0, inside +y
    edges[edgeAt(8)] = float4(-1.0, 0.0, 0.0, -1.0);   // x = 1, inside -x
}

float3 turn(float4 q, float3 v) {
    const float3 u = float3(q.y, q.z, q.w);
    return v * (q.x * q.x - dot(u, u)) + u * (2.0 * dot(u, v)) + cross(u, v) * (2.0 * q.x);
}

/// The size of a gaussian along a direction in its plane.
float sizeAlong(float4 q, float3 scale, float3 direction) {
    const float a = dot(turn(q, float3(1.0, 0.0, 0.0)), direction) * scale.x;
    const float b = dot(turn(q, float3(0.0, 1.0, 0.0)), direction) * scale.y;
    const float c = dot(turn(q, float3(0.0, 0.0, 1.0)), direction) * scale.z;
    return sqrt(a * a + b * b + c * c);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCheckEdges(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= params.splats) {
        return;
    }
    const float3 p = records[recordAt(splat, 0)].xyz;
    const float3 scale = records[recordAt(splat, 1)].xyz;
    const float4 q = records[recordAt(splat, 2)];
    const float sx = sizeAlong(q, scale, float3(1.0, 0.0, 0.0));
    const float sy = sizeAlong(q, scale, float3(0.0, 1.0, 0.0));
    const float toX = min(p.x, 1.0 - p.x);   // to the nearer of the edges square to x
    const float toY = min(p.y, 1.0 - p.y);
    const float slack = params.tolerance * params.sigma;
    // No reach past an outer edge (a twentieth is the floor a centre on it keeps).
    if (sx * params.reach > max(toX, 0.05 * params.sigma * params.reach) + slack ||
        sy * params.reach > max(toY, 0.05 * params.sigma * params.reach) + slack) {
        InterlockedAdd(counts[0], 1u);
    }
    const bool farX = toX >= params.sigma * params.reach;
    const bool farY = toY >= params.sigma * params.reach;
    // Away from every outer edge, nothing held -- the diagonal is no edge.
    if (farX && farY && (abs(sx - params.sigma) > slack || abs(sy - params.sigma) > slack)) {
        InterlockedAdd(counts[1], 1u);
    }
    // Near one edge only, its size along that edge kept.
    if ((farX && !farY && abs(sx - params.sigma) > slack) || (farY && !farX && abs(sy - params.sigma) > slack)) {
        InterlockedAdd(counts[2], 1u);
    }
    if (!farX || !farY) {
        InterlockedAdd(counts[3], 1u);
    }
}
