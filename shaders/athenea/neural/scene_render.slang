// Copyright (c) 2026 jesus luque.
//
// A RENDER WHOSE RESULT IS GAUSSIANS. scene_frame.slang makes a gaussian a
// pixel of any renderer's picture, from its colour and its depth alone. This
// is the same look taken by the engine's own path tracer, which knows more
// of each pixel than a picture says (technique::PathAux): which way the
// surface faces, what it is made of, and -- where it lets the eye through --
// what stands behind it and what the glass took on the way.
//
// So a gaussian is made of its pixel's material. An opaque surface gives a
// disc in the surface carrying its metal's share and its roughness. A surface
// that lets the eye through gives two: the glass itself, with its
// transmission, its tint, its index and whether it is a sheet -- what the
// engine's gaussians refract by -- and, where the bent ray ends, the thing
// seen through it with the light it sent before the glass took its share.
// The picture of the room is not painted on the pane.
//
// What the pane is drawn by where nothing refracts -- the rasteriser, which
// blends -- is an opacity and a colour that give the look's own pixel back
// over what stands behind: as opaque as the glass's darkest channel takes,
// and coloured by what the other channels keep over that (renderPack).
//
// A gaussian is as wide as its pixel is at its distance: twice as far, twice
// as wide. And a look is taken at a footprint of its own -- a cell of
// `footprint` pixels a side, one gaussian a cell -- so that a render still
// gathering its paths leaves a coarse scene at once and finer ones over it
// as it settles (the scene's coarser gaussians there are taken out by
// scene_frame.slang's frameSeenThrough, as a nearer look's are).
//
// A record is kRenderRecord floats, laid out as io::SplatEncoding says of
// them (render::Discovery::renderEncoding).
module scene_render;

public static const uint kRenderRecord = 30;

// A record's fields.
static const uint kAtPosition = 0;       // x y z
static const uint kAtOpacity = 3;        // a logit
static const uint kAtScale = 4;          // three logs
static const uint kAtRotation = 7;       // w x y z
static const uint kAtColour = 11;        // the first harmonic, as a capture keeps a colour
static const uint kAtNormal = 14;        // x y z, world
static const uint kAtMetallic = 17;
static const uint kAtRoughness = 18;
static const uint kAtTransmission = 19;
static const uint kAtRow = 20;           // the material's row + 1 in the frame that made it; 0 for what was seen through glass
static const uint kAtIndex = 21;         // its index of refraction
static const uint kAtSheet = 22;         // 1: a sheet with no inside (or nothing that bends)
static const uint kAtTint = 23;          // r g b: what the surface lets through one crossing, before Fresnel
static const uint kAtParent = 26;        // the width of the coarser gaussians it was laid over, 0 where over nothing: its level's parent
static const uint kAtParentColour = 27;  // and their colour, as a base colour is kept (display-encoded): what it is moved towards in the cut's band

// technique::PathAux's planes.
static const uint kPlaneNormal = 1;
static const uint kPlaneMade = 2;
static const uint kPlaneRow = 3;
static const uint kPlaneBehind = 4;
static const uint kPlanePassed = 5;
static const uint kPlaneTint = 6;
static const uint kPlaneFlat = 7;

struct RenderLookParams {
    uint  width;
    uint  height;
    float focalX; float focalY;      // pixels per unit of x/z
    float centreX; float centreY;
    float tolerance;                 // a surface is held where the scene's depth is within this share of its own
    float spread;                    // a disc's standard deviation, in footprints at its own depth
    float r0x; float r0y; float r0z; float r0w;   // view to world, three rows
    float r1x; float r1y; float r1z; float r1w;
    float r2x; float r2y; float r2z; float r2w;
    float w0x; float w0y; float w0z; float w0w;   // world to view, three rows
    float w1x; float w1y; float w1z; float w1w;
    float w2x; float w2y; float w2z; float w2w;
    uint  hasKnown;
    float repaint;                   // scene_frame.slang's
    float refine;
    uint  footprint;                 // pixels a side of the cell one gaussian is made of
    float glassOpacity;              // the least a pane covers, however clear
    float turned;                    // the least a disc's facing is taken as: it is stretched by no more than 1 / this
    uint  hasFirst;                  // knownFirst is bound
    /// PANES APART: 1, a surface that lets the eye through is made as a pane and what stands behind it.
    /// 0: it is made as anything else is, a disc carrying its pixel -- what a look at coarse cells does,
    /// since the finer look that follows takes a disc out by its depth and a pane has none the scene's
    /// depth stops at.
    uint  panes;
    uint  hasBefore;                 // beforeSize and beforeDepth are bound
    /// A LEVEL OF A FRAME'S PYRAMID: 1, a gaussian laid over nothing the scene held is given the level
    /// above its own for a parent -- cells twice as wide, which the same frame makes -- so that the cut
    /// has something coarser to draw of a surface found from near. The top level is made with 0.
    uint  pyramid;
    uint pad2; uint pad3;
};

StructuredBuffer<float4>   colour;       // the frame: linear light, premultiplied
StructuredBuffer<float>    depth;        // its view depth, 0 or less where nothing is
StructuredBuffer<float4>   aux;          // the path tracer's planes, each width x height
StructuredBuffer<float4>   knownColour;  // the scene as it is known, from the same camera
StructuredBuffer<float>    knownDepth;   // where it is all but opaque
StructuredBuffer<float>    knownFirst;   // where it begins: a pane's own depth
StructuredBuffer<float4>   knownSize;    // the scene drawn with each gaussian's size as its colour
StructuredBuffer<float4>   beforeSize;   // the same before this look took anything out: what a gaussian made now is laid over
StructuredBuffer<float>    beforeDepth;
StructuredBuffer<float4>   beforeColour; // and its colour there
RWStructuredBuffer<uint>   made;         // a cell: how many gaussians it makes (none, one, two)
StructuredBuffer<uint>     madeRead;
RWStructuredBuffer<uint>   which;        // a cell: 1 its surface's, 2 what stands behind it
StructuredBuffer<uint>     whichRead;
StructuredBuffer<uint>     dest;         // its first's place among those made
RWStructuredBuffer<float>  records;
RWStructuredBuffer<uint>   counters;     // cells with a surface, held by the scene, that let the eye through, made behind one
RWStructuredBuffer<float>  seenThrough;  // renderSparePanes: scene_frame.slang's mask of where the frame sees through the scene
ConstantBuffer<RenderLookParams> params;

float3 viewPoint(float x, float y, float z) {
    return float3((x - params.centreX) / params.focalX * z, (y - params.centreY) / params.focalY * z, z);
}

float3 toWorld(float3 v) {
    return float3(dot(float3(params.r0x, params.r0y, params.r0z), v) + params.r0w,
                  dot(float3(params.r1x, params.r1y, params.r1z), v) + params.r1w,
                  dot(float3(params.r2x, params.r2y, params.r2z), v) + params.r2w);
}

float3 toView(float3 p) {
    return float3(dot(float3(params.w0x, params.w0y, params.w0z), p) + params.w0w,
                  dot(float3(params.w1x, params.w1y, params.w1z), p) + params.w1w,
                  dot(float3(params.w2x, params.w2y, params.w2z), p) + params.w2w);
}

float4 plane(uint which, uint at) { return aux[which * params.width * params.height + at]; }

/// The pixel at the middle of a cell, which stands for it.
uint2 middleOf(uint2 cell) {
    const uint half = params.footprint / 2u;
    return uint2(min(cell.x * params.footprint + half, params.width - 1u),
                 min(cell.y * params.footprint + half, params.height - 1u));
}

bool repainted(uint at) {
    if (!(params.repaint > 0.0)) {
        return false;
    }
    // Over a neighbourhood, not a pixel: the scene is gaussians a pixel wide and the frame is paths, and
    // at the edge of every square of a chequered floor one pixel of the two differed by more than any
    // share worth asking -- a tenth of a still frame was made again every frame, over what was there.
    // Five by five pixels of each, their means: a thing that moved is wider than that, and noise is not.
    float3 was = float3(0.0);
    float3 now = float3(0.0);
    float cover = 0.0;
    float taken = 0.0;
    const int x0 = int(at % params.width);
    const int y0 = int(at / params.width);
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            const int x = clamp(x0 + dx, 0, int(params.width) - 1);
            const int y = clamp(y0 + dy, 0, int(params.height) - 1);
            const uint p = uint(y) * params.width + uint(x);
            const float4 held = knownColour[p];
            was += held.rgb;
            cover += held.a;
            now += colour[p].rgb;
            taken += 1.0;
        }
    }
    was /= max(cover, 1.0e-3);   // the scene's colour where it covers
    now /= taken;
    const float3 off = abs(now - was) / (max(now, was) + 0.02);
    return max(off.x, max(off.y, off.z)) > params.repaint;
}

/// How wide this look's disc is at depth `z`, a standard deviation.
float discAt(float z) { return params.spread * float(params.footprint) * z / params.focalX; }

bool coarser(uint at, float z) {
    if (!(params.refine > 0.0)) {
        return false;
    }
    const float4 held = knownSize[at];
    const float has = held.r / max(held.a, 1.0e-3);
    return has > params.refine * discAt(z);
}

/// Whether the scene already holds a surface at view depth `z` through pixel `at`.
bool holds(uint at, float z, bool pane) {
    if (params.hasKnown == 0) {
        return false;
    }
    // A pane is thin of opacity: the scene's depth goes past it to what is all but opaque, and the pane
    // is held where the scene begins.
    const float has = pane && params.hasFirst != 0 ? knownFirst[at] : knownDepth[at];
    const bool there = pane ? has > 0.0 : knownColour[at].a > 0.5;
    return there && abs(has - z) <= params.tolerance * z;
}

/// Which cells make gaussians, and how many.
[shader("compute")]
[numthreads(16, 16, 1)]
void renderValid(uint3 tid: SV_DispatchThreadID) {
    const uint cellsX = (params.width + params.footprint - 1u) / params.footprint;
    const uint cellsY = (params.height + params.footprint - 1u) / params.footprint;
    if (tid.x >= cellsX || tid.y >= cellsY) {
        return;
    }
    const uint cell = tid.y * cellsX + tid.x;
    const uint2 mid = middleOf(tid.xy);
    const uint at = mid.y * params.width + mid.x;
    const float z = depth[at];
    uint kinds = 0u;
    if (z > 0.0 && isfinite(z) && colour[at].a > 0.5) {
        InterlockedAdd(counters[0], 1u);
        const float through = plane(kPlaneMade, at).z;
        const bool pane = params.panes != 0 && through > 0.02;
        // A SHEET IS HELD WHERE THE SCENE BEGINS, and is not made again: its depth is not what the scene's
        // depth says. A solid of glass is a surface like any other -- its gaussians are opaque, the scene's
        // depth is theirs -- and a nearer look lays finer ones over it; only its colour says nothing of
        // whether it changed, since what glass shows is the eye's.
        // (A lens: glass that is no sheet and has an index to bend by. Whatever else lets the eye through
        // is held as a sheet is.)
        const float4 rowAt = plane(kPlaneRow, at);
        const bool solid = pane && rowAt.y <= 0.5 && rowAt.z > 1.0;
        const bool sheet = pane && !solid;
        if (holds(at, z, sheet) && (sheet || ((solid || !repainted(at)) && !coarser(at, z)))) {
            InterlockedAdd(counters[1], 1u);
        } else {
            kinds |= 1u;
        }
        if (pane) {
            InterlockedAdd(counters[2], 1u);
            const float4 behind = plane(kPlaneBehind, at);
            // Where the glass keeps less than half of some channel -- dark glass, or glass met so nearly
            // edge on that most of the pixel is what it reflects -- the pixel says too little of the thing
            // behind to make it from: it stays a picture on the pane.
            const float3 kept = plane(kPlanePassed, at).rgb;
            // And a lens's pixel is of where its bent ray ended only as that one eye saw it, through two
            // faces and with what they reflected: made from it, the thing behind was shards of the wrong
            // colour on what the lens hid. Only a sheet's pixel makes what stands behind it.
            if (behind.w > 0.5 && plane(kPlaneRow, at).y > 0.5 && min(kept.x, min(kept.y, kept.z)) >= 0.5) {
                // What stands behind is held where the scene, through the pixel its place falls on,
                // already has something at its depth.
                const float3 v = toView(behind.xyz);
                bool has = false;
                if (v.z > 0.0 && params.hasKnown != 0) {
                    const int px = int(floor(v.x / v.z * params.focalX + params.centreX));
                    const int py = int(floor(v.y / v.z * params.focalY + params.centreY));
                    if (px >= 0 && py >= 0 && px < int(params.width) && py < int(params.height)) {
                        const uint there = uint(py) * params.width + uint(px);
                        has = knownColour[there].a > 0.5 && abs(knownDepth[there] - v.z) <= params.tolerance * v.z;
                    }
                }
                if (v.z > 0.0 && !has) {
                    kinds |= 2u;
                    InterlockedAdd(counters[3], 1u);
                }
            }
        }
    }
    which[cell] = kinds;
    made[cell] = (kinds & 1u) + ((kinds >> 1) & 1u);
}

/// NOTHING IS TAKEN OUT THROUGH GLASS. Where the frame's surface lets the eye through, the scene's depth
/// is not that surface's -- it goes past a pane to what stands behind -- and what the mask would carry to
/// the gaussians there is the pane's and everything seen through it: a look from a second side took the
/// floor behind a pane out with it. Those pixels are struck from the mask; a pane and what is behind it
/// are held as they were found.
[shader("compute")]
[numthreads(16, 16, 1)]
void renderSparePanes(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    // A sheet's pixels, and anything that lets the eye through with no index: a lens is where the scene's depth says it is.
    const float4 rowAt = plane(kPlaneRow, at);
    if (plane(kPlaneMade, at).z > 0.02 && !(rowAt.y <= 0.5 && rowAt.z > 1.0)) {
        seenThrough[at] = 0.0;
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

/// The opacity each of a look's discs has for the surface they make together to cover `cover` of what is
/// behind it: the discs overlap -- a pixel lies under 2 pi spread^2 of them, weighed -- and each lets
/// through its share of the whole.
float discOpacity(float cover) {
    const float under = 6.2831853 * params.spread * params.spread;
    return 1.0 - exp(log(max(1.0 - cover, 1.0e-4)) / max(under, 1.0));
}

float logit(float a) {
    const float c = clamp(a, 0.003, 0.9975);
    return log(c / (1.0 - c));
}

/// A disc at `world` lying square to `normal` (world, unit): `along` wide across the way the surface
/// turns from `eyeward` (world, unit, towards the eye) and `across` wide the other way.
void writeDisc(uint row, float3 world, float3 normal, float3 eyeward, float across, float along, float3 light,
               float opacity) {
    // The way the surface falls away from the eye, in the surface: where a pixel's footprint is stretched.
    float3 fall = eyeward - normal * dot(eyeward, normal);
    if (dot(fall, fall) < 1.0e-8) {
        const float3 other = abs(normal.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        fall = cross(other, normal);
    }
    const float3 first = normalize(fall);
    const float3 second = cross(normal, first);
    const float4 q = quaternionOf(first, second, normal);
    for (uint k = 0; k < kRenderRecord; ++k) {
        records[row + k] = 0.0;
    }
    records[row + kAtPosition + 0] = world.x;
    records[row + kAtPosition + 1] = world.y;
    records[row + kAtPosition + 2] = world.z;
    records[row + kAtOpacity] = logit(opacity);
    records[row + kAtScale + 0] = log(along);
    records[row + kAtScale + 1] = log(across);
    records[row + kAtScale + 2] = log(across * 0.1);
    records[row + kAtRotation + 0] = q.x;
    records[row + kAtRotation + 1] = q.y;
    records[row + kAtRotation + 2] = q.z;
    records[row + kAtRotation + 3] = q.w;
    records[row + kAtColour + 0] = (encoded(light.r) - 0.5) / 0.28209479;
    records[row + kAtColour + 1] = (encoded(light.g) - 0.5) / 0.28209479;
    records[row + kAtColour + 2] = (encoded(light.b) - 0.5) / 0.28209479;
    records[row + kAtNormal + 0] = normal.x;
    records[row + kAtNormal + 1] = normal.y;
    records[row + kAtNormal + 2] = normal.z;
    records[row + kAtRoughness] = 1.0;
    records[row + kAtIndex] = 1.0;
    records[row + kAtSheet] = 1.0;
}

/// A cell's gaussians, at their places among those made.
[shader("compute")]
[numthreads(16, 16, 1)]
void renderPack(uint3 tid: SV_DispatchThreadID) {
    const uint cellsX = (params.width + params.footprint - 1u) / params.footprint;
    const uint cellsY = (params.height + params.footprint - 1u) / params.footprint;
    if (tid.x >= cellsX || tid.y >= cellsY) {
        return;
    }
    const uint cell = tid.y * cellsX + tid.x;
    const uint kinds = whichRead[cell];
    if (kinds == 0u) {
        return;
    }
    const uint2 mid = middleOf(tid.xy);
    const uint at = mid.y * params.width + mid.x;
    const float z = depth[at];
    const float4 rowHere = plane(kPlaneRow, at);
    // THE CELL'S LIGHT: its pixels' that show the same surface as its middle -- the same material at
    // about the same depth -- and no other's. A cell on a silhouette then carries one side's colour and
    // not the two mixed, and a pixel a silhouette crosses, which the renderer itself mixed, is left out
    // where the cell has others.
    float3 sum = float3(0.0);
    float  weight = 0.0;
    float3 inner = float3(0.0);
    float  innerWeight = 0.0;
    const int f = int(params.footprint);
    const int x0 = int(tid.x) * f;
    const int y0 = int(tid.y) * f;
    for (int dy = 0; dy < f; ++dy) {
        for (int dx = 0; dx < f; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            if (x >= int(params.width) || y >= int(params.height)) {
                continue;
            }
            const uint p = uint(y) * params.width + uint(x);
            const float zp = depth[p];
            if (!(zp > 0.0) || abs(zp - z) > 0.05 * z || plane(kPlaneRow, p).x != rowHere.x) {
                continue;
            }
            sum += colour[p].rgb;
            weight += 1.0;
            // Inside the surface: its four neighbours show it too.
            bool edge = false;
            for (int n = 0; n < 4; ++n) {
                const int nx = x + (n == 0 ? -1 : n == 1 ? 1 : 0);
                const int ny = y + (n == 2 ? -1 : n == 3 ? 1 : 0);
                if (nx < 0 || ny < 0 || nx >= int(params.width) || ny >= int(params.height)) {
                    continue;
                }
                const uint q = uint(ny) * params.width + uint(nx);
                const float zq = depth[q];
                edge = edge || !(zq > 0.0) || abs(zq - zp) > 0.05 * zp || plane(kPlaneRow, q).x != rowHere.x;
            }
            if (!edge) {
                inner += colour[p].rgb;
                innerWeight += 1.0;
            }
        }
    }
    float3 light = innerWeight > 0.0 ? inner / innerWeight : sum / max(weight, 1.0);
    if (innerWeight == 0.0 && f == 1) {
        // A lone pixel on a silhouette: the colour of the neighbour that is inside the same surface,
        // where there is one.
        for (int n = 0; n < 4; ++n) {
            const int nx = int(mid.x) + (n == 0 ? -1 : n == 1 ? 1 : 0);
            const int ny = int(mid.y) + (n == 2 ? -1 : n == 3 ? 1 : 0);
            if (nx < 0 || ny < 0 || nx >= int(params.width) || ny >= int(params.height)) {
                continue;
            }
            const uint q = uint(ny) * params.width + uint(nx);
            const float zq = depth[q];
            if (!(zq > 0.0) || abs(zq - z) > 0.05 * z || plane(kPlaneRow, q).x != rowHere.x) {
                continue;
            }
            // Its own far neighbour, on the line away from here, shows the surface as well.
            const int fx = nx + (nx - int(mid.x));
            const int fy = ny + (ny - int(mid.y));
            if (fx < 0 || fy < 0 || fx >= int(params.width) || fy >= int(params.height)) {
                continue;
            }
            const uint far = uint(fy) * params.width + uint(fx);
            if (depth[far] > 0.0 && abs(depth[far] - zq) <= 0.05 * zq && plane(kPlaneRow, far).x == rowHere.x) {
                light = colour[q].rgb;
                break;
            }
        }
    }
    const float3 here = viewPoint(float(mid.x) + 0.5, float(mid.y) + 0.5, z);
    const float3 world = toWorld(here);
    const float3 eye = float3(params.r0w, params.r1w, params.r2w);
    const float3 eyeward = normalize(eye - world);
    // THE DISC LIES IN THE TRIANGLE, and what the gaussian keeps as its normal is the shading one: a
    // normal map turns the light, not the surface, and discs laid by it stand at odds with their
    // neighbours and leave the surface open from any other eye.
    float3 normal = plane(kPlaneFlat, at).xyz;
    float3 shading = plane(kPlaneNormal, at).xyz;
    if (dot(normal, normal) < 0.25) {
        normal = dot(shading, shading) < 0.25 ? eyeward : shading;
    }
    normal = normalize(normal);
    if (dot(normal, eyeward) < 0.0) {
        normal = -normal;
    }
    shading = dot(shading, shading) < 0.25 ? normal : normalize(shading);
    // AS WIDE AS ITS FOOTPRINT IS AT ITS DISTANCE, and along the way the surface falls from the eye as
    // long as the footprint lies on it: across over the facing, not both ways over its root.
    const float across = discAt(z);
    const float along = across / max(dot(normal, eyeward), params.turned);
    uint row = dest[cell] * kRenderRecord;
    const float4 madeHere = plane(kPlaneMade, at);
    const bool pane = params.panes != 0 && madeHere.z > 0.02;
    const float through = pane ? madeHere.z : 0.0;
    if ((kinds & 1u) != 0u) {
        float3 own = light;
        float  opacity = 0.9975;
        // A SOLID OF GLASS IS A LENS, and what it shows is where the bent rays end: nothing straight behind
        // it gives that back by blending. It keeps its picture, as seen from this look, and says what it is
        // made of for whoever refracts. A sheet bends nothing, and is made apart from what it shows.
        const float3 keptHere = plane(kPlanePassed, at).rgb;
        const bool shows = plane(kPlaneBehind, at).w > 0.5 && min(keptHere.x, min(keptHere.y, keptHere.z)) >= 0.5;
        if (pane && rowHere.y > 0.5 && shows) {
            // A PANE OVER WHAT IT SHOWS. The look's pixel is the thing behind times what the glass kept,
            // channel by channel (and whatever the glass reflected). Blending cannot multiply; it covers.
            // So the pane covers as much as the glass's darkest channel took, and is coloured by what the
            // others kept over that: over the gaussian behind, which carries the light before the glass,
            // the two give the pixel back.
            const float3 kept = clamp(plane(kPlanePassed, at).rgb, float3(0.5), float3(1.0));
            const float least = min(kept.x, min(kept.y, kept.z));
            const float cover = clamp(1.0 - least, params.glassOpacity, 0.98);
            own = max(light * (float3(1.0) - (1.0 - cover) / kept) / cover, float3(0.0));
            opacity = discOpacity(cover);
        }
        // A pane with nothing behind it to make -- the sky, or glass too dark to say -- keeps its picture.
        writeDisc(row, world, normal, eyeward, across, along, own, opacity);
        records[row + kAtNormal + 0] = shading.x;
        records[row + kAtNormal + 1] = shading.y;
        records[row + kAtNormal + 2] = shading.z;
        // WHAT IT IS LAID OVER. Where the scene, before this look took anything out, held this surface
        // with coarser gaussians, their width is this one's parent's: the level above it, which is drawn in
        // its place from where this one is too fine to matter (scene_discover.slang's discoverCut).
        bool laidOver = false;
        if (params.hasBefore != 0u) {
            const float4 was = beforeSize[at];
            const float wide = was.r / max(was.a, 1.0e-3);
            if (was.a > 0.5 && abs(beforeDepth[at] - z) <= params.tolerance * z && wide > 1.2 * across) {
                records[row + kAtParent] = wide;
                const float4 held = beforeColour[at];
                const float3 parent = held.rgb / max(held.a, 1.0e-3);
                records[row + kAtParentColour + 0] = encoded(parent.r);
                records[row + kAtParentColour + 1] = encoded(parent.g);
                records[row + kAtParentColour + 2] = encoded(parent.b);
                laidOver = true;
            }
        }
        if (!laidOver && params.pyramid != 0u) {
            // The level above is made where its own cell's middle shows a surface the scene does not hold
            // (renderValid, of the look at twice the footprint): only then is there a parent to name, and
            // only if that middle shows this same surface.
            const uint twice = params.footprint * 2u;
            const uint2 above = uint2(min((mid.x / twice) * twice + params.footprint, params.width - 1u),
                                      min((mid.y / twice) * twice + params.footprint, params.height - 1u));
            const uint at2 = above.y * params.width + above.x;
            const float z2 = depth[at2];
            const bool there = z2 > 0.0 && colour[at2].a > 0.5 && abs(z2 - z) <= 0.05 * z &&
                               plane(kPlaneRow, at2).x == rowHere.x;
            const bool heldThere = params.hasKnown != 0 && knownColour[at2].a > 0.5 &&
                                   abs(knownDepth[at2] - z2) <= params.tolerance * z2;
            if (there && !heldThere) {
                records[row + kAtParent] = 2.0 * across;
                // The level above is this frame's own mean over a cell twice as wide: this gaussian's
                // colour stands for it here.
                records[row + kAtParentColour + 0] = encoded(own.r);
                records[row + kAtParentColour + 1] = encoded(own.g);
                records[row + kAtParentColour + 2] = encoded(own.b);
            }
        }
        const float3 tint = plane(kPlaneTint, at).rgb;
        records[row + kAtTint + 0] = tint.r;
        records[row + kAtTint + 1] = tint.g;
        records[row + kAtTint + 2] = tint.b;
        records[row + kAtMetallic] = madeHere.x;
        records[row + kAtRoughness] = madeHere.y;
        // What the surface transmits is the material's, whatever this look makes of glass: a coarse look
        // that makes no panes still leaves gaussians of glass, and a level of detail drawn from them is
        // glass to whoever shades it (it was written 0 there, and a lens two levels up was a matt picture).
        records[row + kAtTransmission] = madeHere.z > 0.02 ? madeHere.z : 0.0;
        records[row + kAtRow] = rowHere.x;
        records[row + kAtIndex] = rowHere.z;
        records[row + kAtSheet] = rowHere.y;
        row += kRenderRecord;
    }
    if ((kinds & 2u) != 0u) {
        const float4 behind = plane(kPlaneBehind, at);
        const float3 passed = clamp(plane(kPlanePassed, at).rgb, float3(0.5), float3(1.0));
        const float3 back = normalize(world - behind.xyz);
        // The way round: from the eye to the pane, and on to what is behind it.
        const float way = length(here) + length(behind.xyz - world);
        const float wide = params.spread * float(params.footprint) * way / params.focalX;
        writeDisc(row, behind.xyz, back, back, wide, wide, light / passed, 0.9975);
    }
}
