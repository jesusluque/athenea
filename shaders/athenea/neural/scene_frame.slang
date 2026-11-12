// Copyright (c) 2026 jesus luque.
//
// GAUSSIANS MADE FROM A FRAME. A render -- any renderer's: its colour and its
// depth from a camera that is known -- is one look at the world, and what it
// leaves in the scene is a gaussian a pixel: where the pixel's surface is, a
// disc lying in it as wide as the pixel is there, the pixel's colour. The
// scene is then made of what was looked at, with no geometry converted.
//
// A later look adds only what the scene does not hold: a pixel whose surface
// the scene, drawn from the same camera, already has at that depth makes
// nothing. What is left is what the earlier looks could not see.
//
// A gaussian is written as a 3DGS PLY keeps one (io/SplatWriters.h): 62
// floats, the harmonics above the first left at zero.
module scene_frame;

public static const uint kFrameRecord = 62;

struct FrameParams {
    uint  width;
    uint  height;
    float focalX; float focalY;      // pixels per unit of x/z
    float centreX; float centreY;
    float tolerance;                 // a pixel's surface is held where the scene's depth is within this share of its own
    float spread;                    // a disc's standard deviation, in pixels of its own depth
    float r0x; float r0y; float r0z; float r0w;   // view to world, three rows
    float r1x; float r1y; float r1z; float r1w;
    float r2x; float r2y; float r2z; float r2w;
    uint  hasKnown;
    /// REPAINTED: a surface the scene holds at the frame's depth whose colour is no longer the frame's --
    /// by more than this share of the brighter of the two -- is not held: what is there is taken out and
    /// made again from the frame. 0: a surface is held by its depth alone.
    float repaint;
    uint pad1; uint pad2;
};

StructuredBuffer<float4>   colour;       // the frame: linear light, premultiplied
StructuredBuffer<float>    depth;        // its view depth, 0 or less where nothing is
StructuredBuffer<float4>   knownColour;  // the scene as it is known, from the same camera
StructuredBuffer<float>    knownDepth;
RWStructuredBuffer<uint>   valid;        // a pixel: it makes a gaussian
StructuredBuffer<uint>     validRead;
StructuredBuffer<uint>     dest;         // its place among those that do
RWStructuredBuffer<float>  records;
RWStructuredBuffer<uint>   counters;     // pixels with a surface, held by the scene, seen through, repainted
RWStructuredBuffer<float>  seenThrough;  // a pixel: the frame sees past what the scene has there
ConstantBuffer<FrameParams> params;

float3 viewPoint(uint x, uint y, float z) {
    return float3((float(x) + 0.5 - params.centreX) / params.focalX * z,
                  (float(y) + 0.5 - params.centreY) / params.focalY * z, z);
}

float3 toWorld(float3 v) {
    return float3(dot(float3(params.r0x, params.r0y, params.r0z), v) + params.r0w,
                  dot(float3(params.r1x, params.r1y, params.r1z), v) + params.r1w,
                  dot(float3(params.r2x, params.r2y, params.r2z), v) + params.r2w);
}

float3 turnToWorld(float3 v) {
    return float3(dot(float3(params.r0x, params.r0y, params.r0z), v), dot(float3(params.r1x, params.r1y, params.r1z), v),
                  dot(float3(params.r2x, params.r2y, params.r2z), v));
}

/// Whether the scene's colour at a pixel is no longer the frame's.
bool repainted(uint at) {
    if (!(params.repaint > 0.0)) {
        return false;
    }
    const float4 held = knownColour[at];
    const float3 was = held.rgb / max(held.a, 1.0e-3);   // the scene's colour where it covers
    const float3 now = colour[at].rgb;
    const float3 off = abs(now - was) / (max(now, was) + 0.02);
    return max(off.x, max(off.y, off.z)) > params.repaint;
}

float depthAt(int x, int y) {
    if (x < 0 || y < 0 || x >= int(params.width) || y >= int(params.height)) {
        return 0.0;
    }
    return depth[uint(y) * params.width + uint(x)];
}

/// Which pixels make a gaussian: those with a surface the scene does not hold.
[shader("compute")]
[numthreads(16, 16, 1)]
void frameValid(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float z = depth[at];
    bool makes = z > 0.0 && isfinite(z) && colour[at].a > 0.5;
    if (makes) {
        InterlockedAdd(counters[0], 1u);
        if (params.hasKnown != 0) {
            const float held = knownDepth[at];
            if (knownColour[at].a > 0.5 && abs(held - z) <= params.tolerance * z && !repainted(at)) {
                makes = false;
                InterlockedAdd(counters[1], 1u);
            }
        }
    }
    valid[at] = makes ? 1u : 0u;
}

/// WHERE THE FRAME SEES THROUGH THE SCENE. The scene, drawn through the
/// frame's camera, has a surface nearer than the frame's own: the frame looked
/// past where something used to be. What stood there has moved or gone, and
/// the gaussians that made it are to be taken out (scene_discover.slang's
/// discoverGone, through the blend's lift).
[shader("compute")]
[numthreads(16, 16, 1)]
void frameSeenThrough(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float z = depth[at];
    const bool past = z > 0.0 && isfinite(z) && colour[at].a > 0.5 && knownColour[at].a > 0.5 &&
                      knownDepth[at] < z * (1.0 - params.tolerance);
    // A frame with nothing at a pixel where the scene has something sees through it too.
    const bool empty = !(z > 0.0) && colour[at].a < 0.5 && knownColour[at].a > 0.5;
    // And a surface that is where it was and is another colour is taken out to be made again.
    const bool again = z > 0.0 && colour[at].a > 0.5 && knownColour[at].a > 0.5 &&
                       abs(knownDepth[at] - z) <= params.tolerance * z && repainted(at);
    seenThrough[at] = (past || empty || again) ? 1.0 : 0.0;
    if (past || empty) {
        InterlockedAdd(counters[2], 1u);
    }
    if (again) {
        InterlockedAdd(counters[3], 1u);
    }
}

/// A unit quaternion (w x y z) of the frame whose columns are `a`, `b`, `c`.
float4 quaternionOf(float3 a, float3 b, float3 c) {
    const float trace = a.x + b.y + c.z;
    float4 q;
    if (trace > 0.0) {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4(0.25 * s, (b.z - c.y) / s, (c.x - a.z) / s, (a.y - b.x) / s);
    } else if (a.x > b.y && a.x > c.z) {
        const float s = sqrt(1.0 + a.x - b.y - c.z) * 2.0;
        q = float4((b.z - c.y) / s, 0.25 * s, (b.x + a.y) / s, (c.x + a.z) / s);
    } else if (b.y > c.z) {
        const float s = sqrt(1.0 + b.y - a.x - c.z) * 2.0;
        q = float4((c.x - a.z) / s, (b.x + a.y) / s, 0.25 * s, (c.y + b.z) / s);
    } else {
        const float s = sqrt(1.0 + c.z - a.x - b.y) * 2.0;
        q = float4((a.y - b.x) / s, (c.x + a.z) / s, (c.y + b.z) / s, 0.25 * s);
    }
    return normalize(q);
}

float encoded(float linear) {
    const float v = max(linear, 0.0);
    return v <= 0.0031308 ? 12.92 * v : 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

/// A pixel's gaussian, at its place among those made.
[shader("compute")]
[numthreads(16, 16, 1)]
void framePack(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    if (validRead[at] == 0u) {
        return;
    }
    const int x = int(tid.x);
    const int y = int(tid.y);
    const float z = depth[at];
    const float3 here = viewPoint(tid.x, tid.y, z);
    // The surface's normal from its neighbours' depths, where the surface goes on to them; a pixel at an
    // edge takes the side that does, and one with neither faces the eye.
    const float zl = depthAt(x - 1, y), zr = depthAt(x + 1, y), zd = depthAt(x, y - 1), zu = depthAt(x, y + 1);
    const float near = 0.05 * z;
    float3 alongX = float3(0.0);
    float3 alongY = float3(0.0);
    if (zr > 0.0 && abs(zr - z) < near) {
        alongX = viewPoint(tid.x + 1, tid.y, zr) - here;
    } else if (zl > 0.0 && abs(zl - z) < near) {
        alongX = here - viewPoint(tid.x - 1, tid.y, zl);
    }
    if (zu > 0.0 && abs(zu - z) < near) {
        alongY = viewPoint(tid.x, tid.y + 1, zu) - here;
    } else if (zd > 0.0 && abs(zd - z) < near) {
        alongY = here - viewPoint(tid.x, tid.y - 1, zd);
    }
    float3 normal = cross(alongX, alongY);
    const float3 towardsEye = -normalize(here);
    if (dot(normal, normal) < 1.0e-20) {
        normal = towardsEye;
    }
    normal = normalize(normal);
    if (dot(normal, towardsEye) < 0.0) {
        normal = -normal;
    }
    // A disc in the surface: two axes square to the normal, the third the normal.
    const float3 other = abs(normal.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    const float3 first = normalize(cross(other, normal));
    const float3 second = cross(normal, first);
    // As wide as the pixel is at its depth, and wider where the surface turns from the eye, so that its
    // neighbours' discs still meet; a tenth as thick.
    const float pixel = z / params.focalX;
    const float facing = max(dot(normal, towardsEye), 0.25);
    // A pixel at the surface's rim -- a neighbour with nothing in it, or with something far behind or in
    // front -- gets a disc half as wide: a full one would paint what lies beyond the rim.
    const bool rim = !(zl > 0.0 && abs(zl - z) < near) || !(zr > 0.0 && abs(zr - z) < near) ||
                     !(zd > 0.0 && abs(zd - z) < near) || !(zu > 0.0 && abs(zu - z) < near);
    const float sigma = params.spread * (rim ? 0.5 : 1.0) * pixel / sqrt(facing);
    const float4 q = quaternionOf(turnToWorld(first), turnToWorld(second), turnToWorld(normal));
    const float3 world = toWorld(here);
    const float3 c = colour[at].rgb;
    const uint row = dest[at] * kFrameRecord;
    for (uint k = 0; k < kFrameRecord; ++k) {
        records[row + k] = 0.0;
    }
    records[row + 0] = world.x;
    records[row + 1] = world.y;
    records[row + 2] = world.z;
    // The first harmonic, as a capture keeps a colour: display-encoded, about a half, over Y00.
    records[row + 6] = (encoded(c.r) - 0.5) / 0.28209479;
    records[row + 7] = (encoded(c.g) - 0.5) / 0.28209479;
    records[row + 8] = (encoded(c.b) - 0.5) / 0.28209479;
    records[row + 54] = 6.0;   // an opacity's logit: all but opaque
    records[row + 55] = log(sigma);
    records[row + 56] = log(sigma);
    records[row + 57] = log(sigma * 0.1);
    records[row + 58] = q.x;
    records[row + 59] = q.y;
    records[row + 60] = q.z;
    records[row + 61] = q.w;
}
