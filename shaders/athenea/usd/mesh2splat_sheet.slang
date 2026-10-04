// Copyright (c) 2026 jesus luque.
//
// WHETHER A GLASS IS A SHEET: the edges of a piece's triangles, by where the
// points they join stand, sorted, and their runs counted. An edge one triangle uses is
// open -- the rim of a windscreen modelled as one surface -- and an edge
// more than two use is not a solid's either; a closed solid uses every edge
// twice. mesh2splat reads a transmitting piece with enough open edges as
// thin-walled, with no word from the material (Converter::classifySheets).
module mesh2splat_sheet;

struct SheetParams {
    uint triangles;   ///< the piece's
    uint listed;      ///< 1: the piece's triangles are `list`'s, a subset's; 0: the mesh's, in order
    uint count;       ///< edges: three a triangle
    uint pad0;
};

StructuredBuffer<uint>       indices;    // the mesh's: three points a triangle
StructuredBuffer<float4>     positions;  // the mesh's points
StructuredBuffer<uint>       list;       // a subset's triangles, where `listed`
RWStructuredBuffer<uint>     keysLo;     // an edge's smaller point
RWStructuredBuffer<uint>     keysHi;     // and its larger
RWStructuredBuffer<uint>     values;     // the sort's companions, unused
RWStructuredBuffer<uint>     counts;     // 0 edges, 1 open (one use), 2 more than two uses
ConstantBuffer<SheetParams>  sheet;

/// A point by where it stands, not by its index: a sphere whose seam repeats
/// its points for the texture coordinates is closed all the same. The bits
/// of the three coordinates, mixed (murmur's finaliser).
uint pointKey(uint index) {
    const float3 p = positions[index].xyz;
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
    const uint k = tid.x;
    if (k >= sheet.triangles) {
        return;
    }
    const uint t = sheet.listed != 0 ? list[k] : k;
    const uint a = pointKey(indices[t * 3]), b = pointKey(indices[t * 3 + 1]), c = pointKey(indices[t * 3 + 2]);
    const uint3 from = uint3(a, b, c);
    const uint3 to = uint3(b, c, a);
    [unroll] for (uint e = 0; e < 3; ++e) {
        // An edge from a point to itself (a collapsed corner at a sphere's
        // pole) is not an edge: kept out of the counts.
        const bool collapsed = from[e] == to[e];
        keysLo[k * 3 + e] = collapsed ? 0xFFFFFFFFu : min(from[e], to[e]);
        keysHi[k * 3 + e] = collapsed ? 0xFFFFFFFFu : max(from[e], to[e]);
        values[k * 3 + e] = 0u;
    }
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
