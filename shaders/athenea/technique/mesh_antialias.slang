// Copyright (c) 2026 jesus luque.
//
// EDGES RESOLVED FROM THE NEIGHBOURHOOD'S TRIANGLES (athenea:meshAntialias),
// over a mesh layer already shaded at one sample a pixel. No second
// visibility pass and no second shading: a pixel one of whose four sides saw
// another (instance, triangle) takes the triangles its 3x3 neighbourhood saw
// as the candidates for what its sub-samples see, intersects each sample's
// ray with every one of them (the nearest it is inside wins, as a depth test
// would), and takes for that sample the colour already shaded at the pixel
// that saw the winner. A sample none of them covers sees what a neighbour
// that saw nothing shows, the background, or else the pixel's own colour. The layer
// is the mean of the samples.
//
// What the neighbours saw is all it knows: a triangle no pixel centre around
// it saw -- a wire thinner than a pixel -- is not found, as it is not drawn
// at one sample. A candidate that covers this pixel's own centre nearer than
// what the centre saw was removed there (a cutout's hole, a hidden instance)
// and is not trusted, so a hole is not filled. Two triangles of one instance
// in one plane are one surface: the sample keeps the pixel's own colour, so
// a flat textured wall is not blurred along the diagonals it was split by.
// Only the ids and the shaded colours are read, so every visibility route
// resolves alike; the ids themselves are not changed.
module mesh_antialias;
import surface;

struct AntialiasParams {
    uint samples;   ///< 4, 8 or 16
    uint pad0;
    uint pad1;
    uint pad2;
};

Texture2D<uint4>                visibility;   // (instance + 1, triangle); row 0 on top
StructuredBuffer<float4>        source;       // the shaded layer, bottom row first (a copy)
RWStructuredBuffer<float4>      colour;       // the layer, written where an edge crosses the pixel
ConstantBuffer<CameraParams>    camera;
ConstantBuffer<AntialiasParams> antialias;

/// The standard multisample positions, in sixteenths of a pixel from its
/// centre: 4 from 0, 8 from 4, 16 from 12.
static const int2 kSamples[28] = {
    int2(-2, -6), int2(6, -2), int2(-6, 2), int2(2, 6),
    int2(1, -3), int2(-1, 3), int2(5, 1), int2(-3, -5), int2(-5, 5), int2(-7, -1), int2(3, 7), int2(7, -7),
    int2(1, 1), int2(-1, -3), int2(-3, 2), int2(4, -1), int2(-5, -2), int2(2, 5), int2(5, 3), int2(3, -5),
    int2(-2, 6), int2(0, -7), int2(-4, -6), int2(-6, 4), int2(-8, 0), int2(7, -4), int2(6, 7), int2(-7, -8)
};

/// The pixel first, then its four sides, then its corners: a candidate
/// found twice keeps the nearer pixel's colour.
static const int2 kAround[9] = {
    int2(0, 0), int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
    int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)
};

/// More distinct triangles than this around one pixel are rare; the rest are
/// left out.
static const uint kCandidates = 6;

/// The ray's distance to the triangle (a, a + e1, a + e2), or -1 where it
/// passes outside it by more than `slack` in barycentrics.
float coveredAt(float3 o, float3 d, float3 a, float3 e1, float3 e2, float slack) {
    const float3 p = cross(d, e2);
    const float det = dot(e1, p);
    if (abs(det) < 1e-20) {
        return -1.0;
    }
    const float inverse = 1.0 / det;
    const float3 s = o - a;
    const float u = dot(s, p) * inverse;
    if (u < -slack || u > 1.0 + slack) {
        return -1.0;
    }
    const float3 q = cross(s, e1);
    const float v = dot(d, q) * inverse;
    if (v < -slack || u + v > 1.0 + slack) {
        return -1.0;
    }
    return dot(e2, q) * inverse;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void meshAntialias(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const int w = int(camera.width);
    const int h = int(camera.height);
    const int2 pixel = int2(tid.xy);   // bottom row first
    uint2 seen[9];
    bool inside[9];
    bool edge = false;
    // The pixel and its sides decide whether an edge is here at all; the
    // corners are read only then (a pixel whose sides all agree with it is
    // left as it is).
    for (uint k = 0; k < 9; ++k) {
        const int2 q = pixel + kAround[k];
        inside[k] = q.x >= 0 && q.y >= 0 && q.x < w && q.y < h;
        seen[k] = uint2(0);
        if (k < 5 && inside[k]) {
            seen[k] = visibility.Load(int3(q.x, h - 1 - q.y, 0)).xy;
            edge = edge || any(seen[k] != seen[0]);
        }
    }
    if (!edge) {
        return;   // the colour already there is the pixel's
    }
    for (uint k = 5; k < 9; ++k) {
        if (inside[k]) {
            const int2 q = pixel + kAround[k];
            seen[k] = visibility.Load(int3(q.x, h - 1 - q.y, 0)).xy;
        }
    }

    // The distinct triangles, each with the first pixel that saw it, and the
    // first pixel that saw nothing.
    uint2  ids[kCandidates];
    uint   from[kCandidates];
    float3 corner[kCandidates];
    float3 e1[kCandidates];
    float3 e2[kCandidates];
    bool   trusted[kCandidates];
    for (uint c = 0; c < kCandidates; ++c) {
        ids[c] = uint2(0);
        from[c] = 0;
        corner[c] = float3(0.0);
        e1[c] = float3(0.0);
        e2[c] = float3(0.0);
        trusted[c] = false;
    }
    uint count = 0;
    int background = -1;
    for (uint k = 0; k < 9; ++k) {
        if (!inside[k]) {
            continue;
        }
        if (seen[k].x == 0) {
            if (background < 0) {
                background = int(k);
            }
            continue;
        }
        bool known = false;
        for (uint c = 0; c < count; ++c) {
            known = known || all(ids[c] == seen[k]);
        }
        if (!known && count < kCandidates) {
            ids[count] = seen[k];
            from[count] = k;
            ++count;
        }
    }
    for (uint c = 0; c < count; ++c) {
        const InstanceRecord r = instances[ids[c].x - 1];
        const MeshRecord mesh = meshes[r.mesh];
        const uint t = mesh.firstTriangle + ids[c].y;
        const uint3 points = uint3(indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]);
        const uint base = mesh.firstPoint + r.pointsOffset;
        const float3 a = rowsApply(r.view0, r.view1, r.view2, positions[base + points.x].xyz, 1.0);
        const float3 b = rowsApply(r.view0, r.view1, r.view2, positions[base + points.y].xyz, 1.0);
        const float3 cc = rowsApply(r.view0, r.view1, r.view2, positions[base + points.z].xyz, 1.0);
        corner[c] = a;
        e1[c] = b - a;
        e2[c] = cc - a;
    }

    // What the centre saw, and which candidates it did not see though they
    // lie over it, nearer: removed there, so not to be drawn here.
    float3 origin;
    float3 direction;
    viewRay(camera, float2(pixel) + 0.5, origin, direction);
    const bool own = seen[0].x != 0;   // candidate 0 is then the pixel's own
    const float ownT = own ? coveredAt(origin, direction, corner[0], e1[0], e2[0], 1e30) : 3.4e38;
    for (uint c = 0; c < count; ++c) {
        const float t = c == 0 && own ? -1.0 : coveredAt(origin, direction, corner[c], e1[c], e2[c], -1e-4);
        trusted[c] = !(t > 0.0 && t < ownT * (1.0 - 1e-4));
    }
    // The pixel's own plane, for the triangles that continue it.
    float3 ownNormal = float3(0.0);
    if (own) {
        ownNormal = normalize(cross(e1[0], e2[0]));
    }

    const uint at = uint(pixel.y) * camera.width + uint(pixel.x);
    const float4 mine = source[at];
    const uint samples = antialias.samples;
    const uint first = samples == 4 ? 0u : samples == 8 ? 4u : 12u;
    float4 sum = float4(0.0);
    for (uint s = 0; s < samples; ++s) {
        const float2 offset = float2(kSamples[first + s]) / 16.0;
        viewRay(camera, float2(pixel) + 0.5 + offset, origin, direction);
        int best = -1;
        float bestT = 3.4e38;
        for (uint c = 0; c < count; ++c) {
            if (!trusted[c]) {
                continue;
            }
            const float t = coveredAt(origin, direction, corner[c], e1[c], e2[c], 1e-5);
            if (t > 0.0 && t < bestT) {
                bestT = t;
                best = int(c);
            }
        }
        float4 value = mine;
        if (best < 0) {
            if (background >= 0) {
                const int2 q = pixel + kAround[background];
                value = source[uint(q.y) * camera.width + uint(q.x)];
            }
        } else if (best > 0 || !own) {
            const uint c = uint(best);
            const bool sameSurface =
                own && ids[c].x == seen[0].x && abs(dot(normalize(cross(e1[c], e2[c])), ownNormal)) > 0.9999 &&
                abs(dot(corner[c] - corner[0], ownNormal)) <= 1e-5 * max(length(corner[0]), 1e-3);
            if (!sameSurface) {
                const int2 q = pixel + kAround[from[c]];
                value = source[uint(q.y) * camera.width + uint(q.x)];
            }
        }
        sum += value;
    }
    colour[at] = sum / float(samples);
}
