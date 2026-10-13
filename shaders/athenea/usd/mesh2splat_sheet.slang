// Copyright (c) 2026 jesus luque.
//
// WHETHER A GLASS IS A SHEET, from the triangles a piece was packed into
// (the conversion's own picture: per triangle its three corners and their
// normals, in the world). Two measures, both on the device:
//
// - its edges, keyed by where the two points stand, sorted, and their runs
//   counted: an edge one triangle uses is open (a windscreen modelled as one
//   surface), one more than two use is no solid's;
// - its thickness as twice its volume over its area (2V/A, research 055):
//   a slab's two faces a few millimetres apart give their gap, a ball two
//   thirds of its radius.
//
// Converter::classifySheets reads the three counts and the two sums back.
module mesh2splat_sheet;

struct SheetParams {
    uint triangles;   ///< the piece's
    uint count;       ///< edges: three a triangle
    uint width;       ///< entries a row of the picture
    uint stride;      ///< pixels a row of it, padded
};

StructuredBuffer<float4>     picture;    // six float4 a triangle: (p, u) (n, v) three times
RWStructuredBuffer<uint>     keysLo;     // an edge's smaller point
RWStructuredBuffer<uint>     keysHi;     // and its larger
RWStructuredBuffer<uint>     values;     // the sort's companions, unused
RWStructuredBuffer<uint>     counts;     // 0 edges, 1 open (one use), 2 more than two uses
RWStructuredBuffer<float2>   measures;   // a triangle each: signed volume to the origin, area
RWStructuredBuffer<float>    sums;       // 0 the volume, 1 the area
ConstantBuffer<SheetParams>  sheet;

float3 cornerOf(uint t, uint c) {
    const uint e = t * 6 + c * 2;
    return picture[(e / sheet.width) * sheet.stride + e % sheet.width].xyz;
}

/// A point by where it stands, not by its index: a sphere whose seam repeats
/// its points for the texture coordinates is closed all the same. The bits
/// of the three coordinates, mixed (murmur's finaliser).
uint pointKey(float3 p) {
    uint h = 0x9E3779B9u;
    [unroll] for (uint k = 0; k < 3; ++k) {
        uint v = asuint(p[k] == 0.0 ? 0.0 : p[k]);   // -0 is 0
        v *= 0xCC9E2D51u;
        v = (v << 15) | (v >> 17);
        v *= 0x1B873593u;
        h ^= v;
        h = (h << 13) | (h >> 19);
        h = h * 5u + 0xE6546B64u;
    }
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sSheetEdges(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= sheet.triangles) {
        return;
    }
    const float3 p0 = cornerOf(t, 0), p1 = cornerOf(t, 1), p2 = cornerOf(t, 2);
    const uint3 from = uint3(pointKey(p0), pointKey(p1), pointKey(p2));
    const uint3 to = from.yzx;
    [unroll] for (uint e = 0; e < 3; ++e) {
        // An edge from a point to itself (a collapsed corner at a sphere's
        // pole) is not an edge: kept out of the counts.
        const bool collapsed = from[e] == to[e];
        keysLo[t * 3 + e] = collapsed ? 0xFFFFFFFFu : min(from[e], to[e]);
        keysHi[t * 3 + e] = collapsed ? 0xFFFFFFFFu : max(from[e], to[e]);
        values[t * 3 + e] = 0u;
    }
    // The tetrahedron the triangle makes with the origin, signed, and its area.
    measures[t] = float2(dot(p0, cross(p1, p2)) / 6.0, 0.5 * length(cross(p1 - p0, p2 - p0)));
}

/// After the sort: each run of one edge, counted where it starts.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sSheetRuns(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= sheet.count) {
        return;
    }
    const bool starts = p == 0 || keysLo[p] != keysLo[p - 1] || keysHi[p] != keysHi[p - 1];
    if (!starts || (keysLo[p] == 0xFFFFFFFFu && keysHi[p] == 0xFFFFFFFFu)) {
        return;
    }
    uint length = 1;
    while (p + length < sheet.count && length < 8u && keysLo[p + length] == keysLo[p] &&
           keysHi[p + length] == keysHi[p]) {
        ++length;
    }
    InterlockedAdd(counts[0], 1u);
    if (length == 1u) {
        InterlockedAdd(counts[1], 1u);
    } else if (length > 2u) {
        InterlockedAdd(counts[2], 1u);
    }
}

/// The two sums, by one group: each thread a strided share, then the group's.
groupshared float2 sPartial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sSheetSum(uint3 gid: SV_GroupThreadID) {
    const uint lane = gid.x;
    float2 sum = float2(0.0);
    for (uint t = lane; t < sheet.triangles; t += 256u) {
        sum += measures[t];
    }
    sPartial[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 128u; step > 0u; step >>= 1) {
        if (lane < step) {
            sPartial[lane] += sPartial[lane + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0u) {
        sums[0] = sPartial[0].x;
        sums[1] = sPartial[0].y;
    }
}
