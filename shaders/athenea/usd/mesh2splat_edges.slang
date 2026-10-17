// Copyright (c) 2026 jesus luque.
//
// WHERE A SURFACE STOPS OR FOLDS, for the gaussians that stand near it.
//
// A gaussian is a cell wide, and a cell is a pixel of the camera that will
// look (`--cell-from-camera`). A part narrower than that -- the flange a
// car's hood folds down into its gap, a few millimetres deep -- still gets
// gaussians a cell wide, standing in the flange's plane: they reach out of
// the gap and over the paint beside it, in the dark the gap is baked in. On
// the Corvette every panel edge came out a thick black wavy line where the
// path tracer draws a thin one. The same reach carries paint over the edge
// of a panel into the gap.
//
// An edge is hard where the mesh ends there (one triangle uses it: the part
// stops, or another mesh begins), where the two triangles that share it fold
// by more than the crease angle, or where more than two share it. Each hard
// edge is a line with an inside: the plane through it, square to its
// triangle, facing that triangle's third corner -- `(n, c)` with the inside
// `dot(n, x) >= c`. A triangle knows its own hard edges, and each line is
// told as well to every other triangle that touches either of its ends: a
// gaussian near the corner of a triangle can stand beside a hard edge that
// belongs to the next one. The conversion keeps a gaussian's reach across
// each line to its distance from it (mesh2splat.slang, m2sHoldEdges).
//
// Edges and corners are keyed by where their points stand, over all of a
// mesh's pieces, so a boundary between two of its materials on one smooth
// surface is not an edge: both triangles are there. Both lists are sorted;
// each run of the edges is one edge, and an end's triangles are a run of the
// corners, found by a binary search. Last, each piece's lines are written
// six entries a triangle into a buffer of its own, and copied, when the
// piece is converted, into a picture of its mesh picture's shape -- the
// Edges clip: in a triangle's first entry how many lines it was told and its
// own edges' bits, in the next five the lines.
module mesh2splat_edges;

static const uint kLines = 5;     // a triangle's lines: the entries of its record after the first
static const uint kNone = 0xFFFFFFFFu;

struct EdgeParams {
    uint  triangles;   ///< the piece's (keys, write), or the mesh's (clear)
    uint  first;       ///< the piece's first triangle among its mesh's
    uint  count;       ///< the mesh's edges, which is also its corners: three a triangle
    uint  width;       ///< entries a row of the picture
    uint  stride;      ///< pixels a row of it, padded
    float creaseCos;   ///< two triangles turning further than this cosine fold
    uint  pad0;
    uint  pad1;
};

StructuredBuffer<float4>    picture;       // six float4 a triangle: (p, u) (n, v) three times
RWStructuredBuffer<uint>    keysLo;        // an edge's smaller point
RWStructuredBuffer<uint>    keysHi;        // and its larger
RWStructuredBuffer<uint>    values;        // the mesh's triangle * 4 + the edge
RWStructuredBuffer<uint>    cornerKeys;    // a corner's point
RWStructuredBuffer<uint>    cornerValues;  // the mesh's triangle * 4 + the corner
RWStructuredBuffer<float4>  points;        // the mesh's corners, three a triangle
RWStructuredBuffer<uint>    flags;         // a mesh triangle each: its own hard edges, bit e for corner e to e + 1
RWStructuredBuffer<uint>    lineCount;     // a mesh triangle each: the lines told to it (may pass kLines)
RWStructuredBuffer<float4>  lines;         // kLines a mesh triangle: (n, c)
RWStructuredBuffer<float4>  edgeData;      // the piece's six entries a triangle, packed (m2sEdgeWrite)
RWStructuredBuffer<uint>    openEdges;     // one: the mesh's edges not shared by exactly two triangles
RWStructuredBuffer<float4>  edgePicture;   // the piece's Edges clip, a picture's rows (m2sEdgeCopy)
ConstantBuffer<EdgeParams>  edge;

uint entryOf(uint index) {
    return (index / edge.width) * edge.stride + index % edge.width;
}

float3 cornerOf(uint t, uint c) {
    return picture[entryOf(t * 6 + c * 2)].xyz;
}

/// A point by where it stands, not by its index: a seam that repeats its
/// points for the texture coordinates is one surface all the same. The bits
/// of the three coordinates, mixed (murmur's finaliser), as mesh2splat_sheet
/// keys them.
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

/// One piece's triangles: their three edges and three corners keyed, their
/// corners kept for the lines, and no line told to them yet.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEdgeKeys(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= edge.triangles) {
        return;
    }
    const uint g = edge.first + t;
    const float3 p[3] = { cornerOf(t, 0), cornerOf(t, 1), cornerOf(t, 2) };
    const uint3 from = uint3(pointKey(p[0]), pointKey(p[1]), pointKey(p[2]));
    const uint3 to = from.yzx;
    [unroll] for (uint e = 0; e < 3; ++e) {
        // An edge from a point to itself (a collapsed corner) is no edge.
        const bool collapsed = from[e] == to[e];
        keysLo[g * 3 + e] = collapsed ? kNone : min(from[e], to[e]);
        keysHi[g * 3 + e] = collapsed ? kNone : max(from[e], to[e]);
        values[g * 3 + e] = g * 4 + e;
        cornerKeys[g * 3 + e] = from[e];
        cornerValues[g * 3 + e] = g * 4 + e;
        points[g * 3 + e] = float4(p[e], 0.0);
    }
    lineCount[g] = 0u;
    flags[g] = 0u;
}

float3 faceOf(uint g) {
    const float3 a = points[g * 3].xyz;
    const float3 crossed = cross(points[g * 3 + 1].xyz - a, points[g * 3 + 2].xyz - a);
    const float area2 = length(crossed);
    return area2 > 1.0e-30 ? crossed / area2 : float3(0.0);
}

/// The first of the sorted corners whose point is `key`.
uint firstCorner(uint key) {
    uint low = 0;
    uint high = edge.count;
    while (low < high) {
        const uint middle = (low + high) / 2;
        if (cornerKeys[middle] < key) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

bool touches(uint g, uint key) {
    return pointKey(points[g * 3].xyz) == key || pointKey(points[g * 3 + 1].xyz) == key ||
           pointKey(points[g * 3 + 2].xyz) == key;
}

/// A line to every triangle with a corner at `key` but its own, which keeps
/// its own edges as bits, and those `skip` -- the other end's key -- already
/// reached.
void tellEnd(uint key, uint skip, uint owner, float4 line) {
    for (uint k = firstCorner(key), seen = 0; k < edge.count && cornerKeys[k] == key && seen < 32u; ++k, ++seen) {
        const uint g = cornerValues[k] / 4;
        if (g == owner || (skip != kNone && touches(g, skip))) {
            continue;
        }
        uint slot;
        InterlockedAdd(lineCount[g], 1u, slot);
        if (slot < kLines) {
            lines[g * kLines + slot] = line;
        }
    }
}

/// After both sorts: each run of one edge, at its start. One triangle on it
/// is an open edge; two are a fold where their planes turn further than the
/// crease; more than two is no surface's. Each triangle on a hard edge makes
/// it a line facing its own inside, told to the triangles at both its ends.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEdgeRuns(uint3 tid: SV_DispatchThreadID) {
    const uint p = tid.x;
    if (p >= edge.count) {
        return;
    }
    const bool starts = p == 0 || keysLo[p] != keysLo[p - 1] || keysHi[p] != keysHi[p - 1];
    if (!starts || (keysLo[p] == kNone && keysHi[p] == kNone)) {
        return;
    }
    uint length = 1;
    while (p + length < edge.count && length < 8u && keysLo[p + length] == keysLo[p] &&
           keysHi[p + length] == keysHi[p]) {
        ++length;
    }
    bool hard = length != 2u;
    if (hard) {
        InterlockedAdd(openEdges[0], 1u);   // not a closed surface: its back may show (m2s's `closed`)
    }
    if (length == 2u) {
        const float3 a = faceOf(values[p] / 4);
        const float3 b = faceOf(values[p + 1] / 4);
        // A degenerate neighbour says nothing about a fold.
        hard = dot(a, a) > 0.5 && dot(b, b) > 0.5 && dot(a, b) < edge.creaseCos;
    }
    if (!hard) {
        return;
    }
    for (uint k = 0; k < length; ++k) {
        const uint g = values[p + k] / 4;
        const uint e = values[p + k] % 4;
        const float3 from = points[g * 3 + e].xyz;
        const float3 along = points[g * 3 + (e + 1) % 3].xyz - from;
        const float3 toOther = points[g * 3 + (e + 2) % 3].xyz - from;
        if (!(dot(along, along) > 1.0e-30)) {
            continue;
        }
        const float3 inward = toOther - along * (dot(toOther, along) / dot(along, along));
        if (!(dot(inward, inward) > 1.0e-30)) {
            continue;
        }
        InterlockedOr(flags[g], 1u << e);
        const float3 n = normalize(inward);
        const float4 line = float4(n, dot(n, from));
        tellEnd(keysLo[p], kNone, g, line);
        tellEnd(keysHi[p], keysLo[p], g, line);
    }
}

/// One piece's lines into its six entries a triangle: in the first how
/// many it was told (at most five) and its own hard edges' bits, the lines
/// in the next five. Which five, where more were told, is whichever came
/// first; the conversion applies them in an order of their own.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEdgeWrite(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= edge.triangles) {
        return;
    }
    const uint g = edge.first + t;
    const uint count = min(lineCount[g], kLines);
    edgeData[t * 6] = float4(float(count), float(flags[g]), 0.0, 0.0);
    for (uint k = 0; k < kLines; ++k) {
        edgeData[t * 6 + 1 + k] = k < count ? lines[g * kLines + k] : float4(0.0);
    }
}

/// A piece's entries into its Edges picture, just before the piece is
/// converted: the pictures come from the device's image pool, which would
/// not hold the whole stage's at once beside its meshes.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEdgeCopy(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= edge.triangles * 6) {
        return;
    }
    edgePicture[entryOf(i)] = edgeData[i];
}
