// Copyright (c) 2026 jesus luque.
//
// A port of Electronic Arts' mesh2splat conversion, from its OpenGL pipeline
// to one compute kernel. The algorithm is theirs:
//
//   Copyright (c) 2024 Electronic Arts Inc. All rights reserved.
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are
//   met:
//   1. Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//   2. Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//   3. Neither the name of Electronic Arts, Inc. ("EA") nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//   THIS SOFTWARE IS PROVIDED BY ELECTRONIC ARTS AND ITS CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED.
//
//   https://github.com/electronicarts/mesh2splat
//
// WHAT THEIR PIPELINE DOES, AND WHAT THIS ONE DOES INSTEAD
//
// Theirs is a geometry shader and a fragment shader. The geometry shader takes
// a triangle, picks the axis its normal points along most, projects it onto
// that plane, and works out the Jacobian of the map from texture coordinates
// to space -- `J = V (UV)^-1`. From the Jacobian come the local frame and two
// of the three sizes: a gaussian is as wide as a texel is, measured in the
// world. The fragment shader then runs once per texel the projected triangle
// covers, samples the maps there, and appends a gaussian to a buffer with an
// atomic counter.
//
// There is no rasteriser here, so the fragments are walked: one thread a
// triangle, over the cells of the projection grid its bounding box covers.
// That is the same set of samples the rasteriser would have produced, at the
// same density -- `resolution` cells across the model's longest side is their
// render target's resolution -- and it keeps the atomic append, which is what
// makes the order of the output nobody's business.
//
// WHAT IS NOT THEIRS: transmission. Their fragment shader samples albedo,
// normal and metallic-roughness. A glass material (`transmission = 1`) would
// come out of it as an opaque white gaussian, so `transmission` and
// `transmissionColour` are taken here as a material constant, and the opacity
// and colour of every gaussian on that mesh answer to them. A gaussian tints;
// it does not refract, and the two are not the same thing.

/// Must match `Mesh2SplatUniforms` in Mesh2Splat.cpp exactly.
struct M2sParams {
    uint  triangles;
    uint  meshWidth;      // the mesh image: 6 float4 an entry, row major
    uint  meshStride;
    uint  recordPixels;   // float4 a splat: 4, or 6 with the PBR channels

    uint  dstWidth;       // where the records go
    uint  dstHeight;
    uint  dstStride;
    uint  maxSplats;

    float4 boundsMin;     // the box the density is measured over: the model's, or the mesh's own
    float4 boundsExtent;  // its size, and w its longest side (read where cellByLongest is set)

    uint  resolution;     // cells across that box
    uint  maxCells;       // most cells one triangle may walk
    float sigmaX;         // 0.65 in their shader
    float sigmaY;

    float flatness;       // the third size: their log(1e-7)
    float opacity;
    float transmission;   // ours: 1 - this is what a gaussian keeps
    uint  useNormalMap;

    float4 materialColour;    // multiplied into the albedo; w unused
    float4 transmissionColour;

    uint  countWidth;     // the per-triangle cell counts, one uint a triangle
    float metallic;       // the material's own, multiplied into the map where there is one
    float roughness;
    uint  countStride;

    uint  hasAlbedo;
    uint  albedoWidth;
    uint  albedoHeight;
    uint  albedoStride;

    uint  hasNormal;
    uint  normalWidth;
    uint  normalHeight;
    uint  normalStride;

    uint  hasMetallicRoughness;
    uint  mrWidth;
    uint  mrHeight;
    uint  mrStride;

    uint  hasInfluences;  // the joints a corner is carried by, same shape as the mesh
    uint  opacityChannel; // 0 none, else the cut-out's channel: 1 r, 2 g, 3 b, 4 a
    uint  opacityWidth;
    uint  opacityHeight;

    uint  opacityStride;
    float opacityCut;
    uint  hasUv2;         // the Texcoord2 picture is there: the mesh's second set of coordinates
    uint  albedoUv2;      // which maps read by it (1) rather than by the mesh picture's own (0)

    uint  normalUv2;
    uint  mrUv2;
    uint  opacityUv2;
    uint  firstTriangle;  // triangles before this are not this run's: a mesh converted in slices

    // THE CELL, BOUNDED IN THE WORLD. A cell is `range / resolution` of the
    // world; these hold it between two sizes in world units (0: no bound),
    // and `cellByLongest` measures every triangle against the box's longest
    // side rather than the wider of its projection's two ranges, so a mesh
    // has one cell whichever way its triangles face.
    float cellMin;
    float cellMax;
    uint  cellByLongest;
    float glassOpacity;   // what a fully transmitting material still stops

    // A HEIGHT ALONG THE NORMAL (m2sRelief says what it does).
    uint  displaceChannel;   // 0: no map, the bias alone; else 1 r, 2 g, 3 b, 4 a
    uint  displaceWidth;
    uint  displaceHeight;
    uint  displaceStride;

    float displaceScale;     // texel * scale + bias, in world units
    float displaceBias;
    uint  displaceRefine;    // 0: no displacement; else the most gaussians a cell along each axis
    uint  displaceUv2;

    // FEWER GAUSSIANS WHERE THE SURFACE IS THE SAME (m2sWalkBlocks).
    float simplify;          // 0: every cell; else how far a block's cells may differ and still be one
    uint  simplifyLevels;    // the largest block is 2^this cells a side
    uint  simplifyPad0;
    uint  simplifyPad1;
};

StructuredBuffer<float4>   mesh;        // per triangle: (position, u) and (normal, v), three corners
StructuredBuffer<float4>   albedo;
StructuredBuffer<float4>   normalMap;
StructuredBuffer<float4>   metallicRoughness;
StructuredBuffer<float4>   opacityMask; // a cut-out, read on one channel
StructuredBuffer<float4>   influences;  // 2 entries a corner: (joint, weight) twice
RWStructuredBuffer<uint>   counters;    // [wanted, written, degenerate, dropped, done: first triangle the budget cut]
RWStructuredBuffer<uint>   cellCounts;  // a triangle's covered cells, then where they start
RWStructuredBuffer<float4> records;     // the output plane
StructuredBuffer<float4>   texcoord2;   // the mesh picture's shape: (u2, v2) a corner, where there is a second set
StructuredBuffer<float4>   heightMap;   // the displacement, read on one channel
ConstantBuffer<M2sParams>  params;

/// The counters start a render, and a buffer the host handed over holds
/// whatever it held before. Cleared in a dispatch of its own so that no thread
/// of the one that counts has to race the clear.
[shader("compute")]
[numthreads(1, 1, 1)]
void m2sClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    counters[0] = 0;
    counters[1] = 0;
    counters[2] = 0;
    counters[3] = 0;
    counters[4] = params.triangles;
    counters[5] = 0;   // cells the relief would have split finer than displaceRefine allows   // every triangle fits until the scan says which did not
}

/// Where one component of a record lands. A record is several entries long
/// and the row it starts on may not have room for all of them, so each is
/// addressed on its own rather than assuming they sit together.
uint recordAt(uint slot, uint component) {
    const uint entry = slot * params.recordPixels + component;
    const uint x = entry % params.dstWidth;
    const uint y = entry / params.dstWidth;
    return y * params.dstStride + x;
}

/// A triangle's entry in the cell-count picture. Four to a pixel, addressed
/// the way every other picture here is.
uint countAt(uint triangle) {
    const uint pixel = triangle / 4;
    const uint x = pixel % max(params.countWidth, 1u);
    const uint y = pixel / max(params.countWidth, 1u);
    return (y * params.countStride + x) * 4 + (triangle % 4);
}

float4 meshEntry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return mesh[y * params.meshStride + x];
}

/// The joints picture is the mesh picture's shape, entry for entry, which is
/// what saves it a second set of dimensions.
float4 influenceEntry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return influences[y * params.meshStride + x];
}

/// The second set of texture coordinates, the mesh picture's shape as well.
float2 texcoord2Entry(uint index) {
    const uint x = index % params.meshWidth;
    const uint y = index / params.meshWidth;
    return texcoord2[y * params.meshStride + x].xy;
}

/// THE JOINTS A GAUSSIAN IS CARRIED BY: the barycentric blend of the three
/// corners', which is what interpolating the skin means.
///
/// Twelve pairs go in and four come out, because four is what the cloud
/// carries: a joint already there gains the weight, an empty slot takes it,
/// and otherwise it displaces the lightest. The four are renormalised, so a
/// gaussian is carried entirely however its triangle was authored.
void blendInfluences(uint triangle, float3 bary, out float4 joints, out float4 weights) {
    joints = float4(0.0);
    weights = float4(0.0);
    for (uint corner = 0; corner < 3; ++corner) {
        const uint at = triangle * 6 + corner * 2;
        const float4 low = influenceEntry(at);
        const float4 high = influenceEntry(at + 1);
        float2 pairs[4];
        pairs[0] = low.xy;
        pairs[1] = low.zw;
        pairs[2] = high.xy;
        pairs[3] = high.zw;
        for (uint i = 0; i < 4; ++i) {
            const float weight = pairs[i].y * bary[corner];
            if (!(weight > 0.0)) {
                continue;
            }
            const float joint = pairs[i].x;
            bool placed = false;
            for (uint slot = 0; slot < 4; ++slot) {
                if (weights[slot] > 0.0 && joints[slot] == joint) {
                    weights[slot] += weight;
                    placed = true;
                    break;
                }
            }
            if (placed) {
                continue;
            }
            uint lightest = 0;
            for (uint slot = 1; slot < 4; ++slot) {
                if (weights[slot] < weights[lightest]) {
                    lightest = slot;
                }
            }
            if (weight > weights[lightest]) {
                joints[lightest] = joint;
                weights[lightest] = weight;
            }
        }
    }
    const float sum = weights.x + weights.y + weights.z + weights.w;
    weights = sum > 0.0 ? weights / sum : float4(1.0, 0.0, 0.0, 0.0);
}

/// A texture, wrapped, sampled as the four texels around the point. Row 0 is
/// v = 0: the host writes these images bottom row first, as every image this
/// engine carries is.
float4 sampleWrapped(StructuredBuffer<float4> texture, uint width, uint height, uint stride, float2 uv) {
    if (width == 0 || height == 0) {
        return float4(1.0, 1.0, 1.0, 1.0);
    }
    const float2 wrapped = uv - floor(uv);
    const float x = wrapped.x * float(width) - 0.5;
    const float y = wrapped.y * float(height) - 0.5;
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const uint x1 = uint((x0 + 1 + int(width)) % int(width));
    const uint y1 = uint((y0 + 1 + int(height)) % int(height));
    const uint xa = uint((x0 + int(width)) % int(width));
    const uint ya = uint((y0 + int(height)) % int(height));
    const float4 c00 = texture[ya * stride + xa];
    const float4 c10 = texture[ya * stride + x1];
    const float4 c01 = texture[y1 * stride + xa];
    const float4 c11 = texture[y1 * stride + x1];
    return lerp(lerp(c00, c10, fx), lerp(c01, c11, fx), fy);
}

/// The rotation of a frame, as a quaternion in w, x, y, z -- the order the
/// records carry and the engine's decode expects.
float4 quaternionOfFrame(float3 x, float3 y, float3 z) {
    const float trace = x.x + y.y + z.z;
    float4 q;
    if (trace > 0.0) {
        const float s = sqrt(trace + 1.0) * 2.0;
        q = float4(0.25 * s, (y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s);
    } else if (x.x > y.y && x.x > z.z) {
        const float s = sqrt(1.0 + x.x - y.y - z.z) * 2.0;
        q = float4((y.z - z.y) / s, 0.25 * s, (y.x + x.y) / s, (z.x + x.z) / s);
    } else if (y.y > z.z) {
        const float s = sqrt(1.0 + y.y - x.x - z.z) * 2.0;
        q = float4((z.x - x.z) / s, (y.x + x.y) / s, 0.25 * s, (z.y + y.z) / s);
    } else {
        const float s = sqrt(1.0 + z.z - x.x - y.y) * 2.0;
        q = float4((x.y - y.x) / s, (z.x + x.z) / s, (z.y + y.z) / s, 0.25 * s);
    }
    const float length2 = dot(q, q);
    return length2 > 1.0e-20 ? q / sqrt(length2) : float4(1.0, 0.0, 0.0, 0.0);
}

/// Which pair of axes a triangle is walked in: the two the normal points
/// along least, which is their triplanar choice written as a lookup.
uint2 projectionAxes(float3 normal) {
    const float3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) {
        return uint2(1, 2);   // x dominant: walk y and z
    }
    if (a.y >= a.z) {
        return uint2(0, 2);
    }
    return uint2(0, 1);
}

float axisOf(float3 v, uint axis) {
    return axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
}

/// A point in the plane it is drawn in: relative to the model's box, over the
/// wider of that plane's two ranges, which is their `orthogonalUvs`.
float2 orthogonalUv(float3 p, uint2 axes, float range) {
    return float2((axisOf(p, axes.x) - axisOf(params.boundsMin.xyz, axes.x)) / range,
                  (axisOf(p, axes.y) - axisOf(params.boundsMin.xyz, axes.y)) / range);
}
/// EVERYTHING A TRIANGLE DECIDES FOR EVERY GAUSSIAN THAT STANDS ON IT,
/// worked out in one place so that the pass that counts and the pass that
/// writes cannot disagree about a single bit.
///
/// Why there are two passes at all: the slot used to be taken with
/// `InterlockedAdd`, and this file said so -- "the order of the output is
/// nobody's business". Two runs over the same mesh gave the same gaussians in
/// a different order, so "gaussian `i` is the same gaussian" was false before
/// any animation was in the picture. Counting first and then writing at
/// `where this triangle starts + k` makes the order the mesh's own, which is
/// what lets a gaussian be followed from one frame to the next -- and it is
/// also what makes a budget too small keep the *first* splats rather than
/// whichever ones won a race.
struct M2sTriangle {
    bool   valid;          ///< a plane, and a Jacobian over its projection
    float4 p0, n0, p1, n1, p2, n2;
    float2 t0, t1, t2;     ///< the second set of texture coordinates, zero without one
    float3 a, b, c;
    float3 geometric;
    float3 tangent;
    float3 scale;
    float2 oa;
    float2 edge1, edge2;
    float  inverseTwice;
    float  cell;
    int2   first;
    uint2  span;
    uint   cells;          ///< walked, after the per-triangle bound
    uint   beyond;         ///< what that bound left out
};

M2sTriangle m2sTriangleOf(uint triangle) {
    M2sTriangle t;
    t.valid = false;
    t.beyond = 0;
    t.cells = 0;
    t.span = uint2(1, 1);
    t.first = int2(0, 0);
    t.cell = 1.0;
    t.oa = float2(0.0);
    t.edge1 = float2(0.0);
    t.edge2 = float2(0.0);
    t.inverseTwice = 0.0;
    t.geometric = float3(0.0, 0.0, 1.0);
    t.tangent = float3(1.0, 0.0, 0.0);
    t.scale = float3(0.0);

    const uint base = triangle * 6;
    t.p0 = meshEntry(base + 0);
    t.n0 = meshEntry(base + 1);
    t.p1 = meshEntry(base + 2);
    t.n1 = meshEntry(base + 3);
    t.p2 = meshEntry(base + 4);
    t.n2 = meshEntry(base + 5);
    t.t0 = float2(0.0);
    t.t1 = float2(0.0);
    t.t2 = float2(0.0);
    if (params.hasUv2 != 0) {
        t.t0 = texcoord2Entry(base + 0);
        t.t1 = texcoord2Entry(base + 2);
        t.t2 = texcoord2Entry(base + 4);
    }

    t.a = t.p0.xyz;
    t.b = t.p1.xyz;
    t.c = t.p2.xyz;
    const float3 e1 = t.b - t.a;
    const float3 e2 = t.c - t.a;
    const float3 crossed = cross(e1, e2);
    const float area2 = length(crossed);
    if (!(area2 > 1.0e-20)) {
        return t;   // no plane, no frame, nothing to place
    }
    t.geometric = crossed / area2;

    // THE PLANE THE TRIANGLE IS DRAWN IN, which is their triplanar choice:
    // the two axes its normal points along least, the position in them taken
    // relative to the model's box and divided by the wider of the two ranges.
    // That is their `orthogonalUvs`, and it is what their render target
    // rasterises -- so a cell of it is `range / resolution` of the world.
    const uint2 axes = projectionAxes(t.geometric);
    const float rangeU = axisOf(params.boundsExtent.xyz, axes.x);
    const float rangeV = axisOf(params.boundsExtent.xyz, axes.y);
    // Their range is the wider of the plane's two, which makes a cell depend
    // on which way the triangle faces: on a car 0.77 wide, 1.92 long and
    // 0.60 tall, a panel facing forward walked a cell 2.5 times finer than
    // the hood. `cellByLongest` measures every triangle against the box's
    // longest side instead, so a mesh has one cell.
    const float range = params.cellByLongest != 0 ? max(params.boundsExtent.w, 1.0e-20)
                                                  : max(max(rangeU, rangeV), 1.0e-20);
    t.oa = orthogonalUv(t.a, axes, range);
    const float2 ob = orthogonalUv(t.b, axes, range);
    const float2 oc = orthogonalUv(t.c, axes, range);

    // The Jacobian of THAT map to space -- `J = V (O)^-1`, their
    // `computeUv3DJacobian` over `orthogonalUvs` and not over the texture's
    // coordinates. Its columns are how far a step of one unit of the
    // projection moves in the world; a step of one cell is that over the
    // resolution, and a gaussian is `sigma` of one cell.
    const float2 duv1 = ob - t.oa;
    const float2 duv2 = oc - t.oa;
    const float determinant = duv1.x * duv2.y - duv2.x * duv1.y;
    if (!(abs(determinant) > 1.0e-24)) {
        return t;   // edge on to its own projection: no cell of it is covered
    }
    const float inverse = 1.0 / determinant;
    const float3 ju = (e1 * duv2.y - e2 * duv1.y) * inverse;
    const float3 jv = (e2 * duv1.x - e1 * duv2.x) * inverse;
    float perCell = 1.0 / float(max(params.resolution, 1u));
    // Bounded in the world: a cell finer than `cellMin` or coarser than
    // `cellMax` is brought to the bound. Only where a bound bites is perCell
    // recomputed, so an unbounded run keeps `1 / resolution` exactly and its
    // cell centres where they were.
    {
        const float cellWorld = range * perCell;
        const float bounded = min(max(cellWorld, params.cellMin > 0.0 ? params.cellMin : 0.0),
                                  params.cellMax > 0.0 ? params.cellMax : 3.4e38);
        if (bounded != cellWorld) {
            perCell = bounded / range;
        }
    }
    // A GAUSSIAN CANNOT BE BIGGER THAN THE TRIANGLE IT STANDS ON. The
    // Jacobian's columns are `1 / determinant`, and a sliver seen almost edge
    // on by its own projection has a determinant just above the floor above,
    // so `ju` and `jv` run away: on the bmw27 car 1859 such triangles drew
    // white spikes metres long across the frame. The triangle's longest edge
    // is the bound that is always true, whatever the projection did.
    const float longestEdge =
        sqrt(max(dot(e1, e1), max(dot(e2, e2), dot(t.c - t.b, t.c - t.b))));
    const float alongU = min(length(ju) * params.sigmaX * perCell, longestEdge);
    const float alongV = min(length(jv) * params.sigmaY * perCell, longestEdge);
    // THE THIRD SIZE IS A FRACTION OF THE OTHER TWO, not a length.
    //
    // mesh2splat writes 1e-7 there, a number in the model's own units, while
    // the two sizes across the surface are in cells -- so how thin a gaussian
    // is depends on how big the model happens to be. The chess pawn is 66 mm
    // across and traced fine; the Khronos fox is a hundred units long, which
    // makes the same 1e-7 fifteen hundred times more extreme, and the ray
    // tracer -- which integrates density along the ray rather than projecting
    // an ellipse as the rasteriser does -- saw a ghost where the rasteriser
    // saw a fox. A fraction is the same number for every model.
    t.scale = float3(alongU, alongV, min(alongU, alongV) * params.flatness);

    // THE FRAME every gaussian of this triangle shares, and theirs: along the
    // triangle's longest edge, with the surface normal as the short axis and
    // the third made square to both. Not along the Jacobian's columns -- the
    // gaussian is a disc in the triangle's plane however that plane was
    // parametrised.
    float3 longest = e1;
    const float3 e3 = t.c - t.b;
    if (dot(e2, e2) > dot(e1, e1) && dot(e2, e2) > dot(e3, e3)) {
        longest = e2;
    } else if (dot(e3, e3) > dot(e1, e1) && dot(e3, e3) > dot(e2, e2)) {
        longest = e3;
    }
    t.tangent = normalize(longest);

    // The barycentric solve, in the plane the triangle is walked in.
    t.edge1 = ob - t.oa;
    t.edge2 = oc - t.oa;
    const float twice = t.edge1.x * t.edge2.y - t.edge2.x * t.edge1.y;
    t.inverseTwice = 1.0 / twice;
    const float2 lowest = min(t.oa, min(ob, oc));
    const float2 highest = max(t.oa, max(ob, oc));
    t.cell = perCell;
    t.first = int2(floor(lowest / t.cell));
    const int2 last = int2(floor(highest / t.cell));
    t.span = uint2(max(last.x - t.first.x + 1, 1), max(last.y - t.first.y + 1, 1));
    const uint wanted = t.span.x * t.span.y;
    t.cells = min(wanted, max(params.maxCells, 1u));
    t.beyond = wanted > params.maxCells ? wanted - params.maxCells : 0u;
    t.valid = true;
    return t;
}

/// The k-th cell of a triangle's walk: where it is in barycentric
/// coordinates, and whether its centre is inside the triangle at all.
bool m2sCellInside(M2sTriangle t, uint k, out float3 bary) {
    const uint cx = k % t.span.x;
    const uint cy = k / t.span.x;
    const float2 at = (float2(float(t.first.x + int(cx)), float(t.first.y + int(cy))) + 0.5) * t.cell;
    const float2 relative = at - t.oa;
    const float beta = (relative.x * t.edge2.y - t.edge2.x * relative.y) * t.inverseTwice;
    const float gamma = (t.edge1.x * relative.y - relative.x * t.edge1.y) * t.inverseTwice;
    const float alpha = 1.0 - beta - gamma;
    bary = float3(alpha, beta, gamma);
    return !(alpha < 0.0 || beta < 0.0 || gamma < 0.0);
}

/// IS THERE A SURFACE AT THIS CELL AT ALL?
///
/// A feather is a flat card whose shape lives entirely in a mask -- the
/// sparrow's primaries are an atlas of cut-out silhouettes in the alpha of
/// the map that also holds their normals. Converted without it, every
/// feather is the whole rectangle: opaque, straight-edged, and eating the
/// budget the real feather needed. Below the cut there is no surface, so no
/// gaussian is counted and none is written -- and the count and the emit must
/// agree on that or the slots a triangle was given do not match the slots it
/// fills.
/// The uv a cell's centre stands at, which is the same barycentric blend the
/// emit does -- they must see the same texel.
float2 m2sUvAt(const M2sTriangle t, float3 bary) {
    const float2 uv0 = float2(t.p0.w, t.n0.w);
    const float2 uv1 = float2(t.p1.w, t.n1.w);
    const float2 uv2 = float2(t.p2.w, t.n2.w);
    return uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;
}

/// The same in the second set.
float2 m2sUv2At(const M2sTriangle t, float3 bary) {
    return t.t0 * bary.x + t.t1 * bary.y + t.t2 * bary.z;
}

/// What the cut-out map reads here: one where there is no map.
float m2sMaskAt(const M2sTriangle t, float3 bary) {
    if (params.opacityChannel == 0) {
        return 1.0;
    }
    const float2 uv = params.opacityUv2 != 0 ? m2sUv2At(t, bary) : m2sUvAt(t, bary);
    const float4 texel = sampleWrapped(opacityMask, params.opacityWidth, params.opacityHeight,
                                       params.opacityStride, uv);
    return params.opacityChannel == 1 ? texel.x
         : params.opacityChannel == 2 ? texel.y
         : params.opacityChannel == 3 ? texel.z
                                      : texel.w;
}

bool m2sCovered(const M2sTriangle t, float3 bary) {
    return params.opacityChannel == 0 || m2sMaskAt(t, bary) >= params.opacityCut;
}

/// THE RELIEF.
///
/// A displaced surface stands `height` off the mesh along its normal, and
/// that is all a gaussian needs to follow it: it is a point, so moving it
/// costs nothing, where a mesh would have to be cut into triangles finer than
/// the relief before it could move at all.
///
/// What moving it does not do is keep the surface covered. A cell of the
/// projection is a square of the flat surface; raised, it is a patch of the
/// relief, and where the relief is steep that patch is several times longer
/// than the square was. One gaussian sized for the square leaves the rest of
/// it bare. So each cell measures how far the relief stretches it along each
/// of its two axes -- the raised surface's tangent against the flat one's --
/// and is split into that many gaussians along each, each sized for its share
/// of the stretched patch and turned to the relief's own plane. That is what
/// tessellating the relief would buy a mesh, for the cost of a few more
/// points and no triangle.
float m2sHeightAt(const M2sTriangle t, float3 bary) {
    float height = 0.0;
    if (params.displaceChannel != 0) {
        const float2 uv = params.displaceUv2 != 0 ? m2sUv2At(t, bary) : m2sUvAt(t, bary);
        const float4 texel = sampleWrapped(heightMap, params.displaceWidth, params.displaceHeight,
                                           params.displaceStride, uv);
        height = params.displaceChannel == 1 ? texel.x
               : params.displaceChannel == 2 ? texel.y
               : params.displaceChannel == 3 ? texel.z
                                             : texel.w;
    }
    return height * params.displaceScale + params.displaceBias;
}

/// A point of the projection in the triangle's barycentric coordinates,
/// unclamped: a sub-cell may stand just past the edge its cell's centre is
/// inside of, and the plane goes on.
float3 m2sBaryOf(const M2sTriangle t, float2 at) {
    const float2 relative = at - t.oa;
    const float beta = (relative.x * t.edge2.y - t.edge2.x * relative.y) * t.inverseTwice;
    const float gamma = (t.edge1.x * relative.y - relative.x * t.edge1.y) * t.inverseTwice;
    return float3(1.0 - beta - gamma, beta, gamma);
}

/// The same, brought onto the triangle: what a joint's weight is read at,
/// since a weight is not something to extrapolate. A point already on it is
/// left exactly as it is.
float3 m2sOnTriangle(float3 bary) {
    if (all(bary >= float3(0.0))) {
        return bary;
    }
    const float3 kept = max(bary, float3(0.0));
    const float sum = kept.x + kept.y + kept.z;
    return sum > 0.0 ? kept / sum : float3(1.0 / 3.0);
}

float3 m2sFlatAt(const M2sTriangle t, float3 bary) {
    return t.a * bary.x + t.b * bary.y + t.c * bary.z;
}

float3 m2sNormalAt(const M2sTriangle t, float3 bary) {
    const float3 blended = t.n0.xyz * bary.x + t.n1.xyz * bary.y + t.n2.xyz * bary.z;
    return dot(blended, blended) > 1.0e-12 ? normalize(blended) : t.geometric;
}

/// Where the relief puts a point of the projection. Everything is read at
/// the point itself, off the triangle too: the triangle's plane, its normals
/// and its texture coordinates go on past its edge, and a sub-cell half a
/// cell beyond it is on the same relief as its neighbour inside -- clamped
/// back, it read the height of a point it was not at, and its slope with it.
float3 m2sRaisedAt(const M2sTriangle t, float2 at) {
    const float3 bary = m2sBaryOf(t, at);
    return m2sFlatAt(t, bary) + m2sNormalAt(t, bary) * m2sHeightAt(t, bary);
}

/// How the relief moves under one cell's step along each axis of the
/// projection -- the raised surface's two tangents, a cell long -- and how
/// many times longer that is than the flat surface's.
struct M2sRelief {
    float3 alongU;
    float3 alongV;
    float2 stretch;
};

M2sRelief m2sReliefAt(const M2sTriangle t, float2 at, float2 half) {
    M2sRelief r;
    const float2 du = float2(half.x, 0.0);
    const float2 dv = float2(0.0, half.y);
    r.alongU = (m2sRaisedAt(t, at + du) - m2sRaisedAt(t, at - du)) * (t.cell / (2.0 * half.x));
    r.alongV = (m2sRaisedAt(t, at + dv) - m2sRaisedAt(t, at - dv)) * (t.cell / (2.0 * half.y));
    const float3 flatU = (m2sFlatAt(t, m2sBaryOf(t, at + du)) - m2sFlatAt(t, m2sBaryOf(t, at - du))) *
                         (t.cell / (2.0 * half.x));
    const float3 flatV = (m2sFlatAt(t, m2sBaryOf(t, at + dv)) - m2sFlatAt(t, m2sBaryOf(t, at - dv))) *
                         (t.cell / (2.0 * half.y));
    r.stretch = float2(length(r.alongU) / max(length(flatU), 1.0e-20),
                       length(r.alongV) / max(length(flatV), 1.0e-20));
    return r;
}

/// The centre of the k-th cell of a triangle's walk, in the projection.
float2 m2sCellCentre(const M2sTriangle t, uint k) {
    const uint cx = k % t.span.x;
    const uint cy = k / t.span.x;
    return (float2(float(t.first.x + int(cx)), float(t.first.y + int(cy))) + 0.5) * t.cell;
}

/// Into how many gaussians along each axis the relief splits a cell. A
/// stretch of a quarter more is left alone -- the gaussians overlap by more
/// than that -- and a split finer than `displaceRefine` is capped, and
/// counted.
uint2 m2sSplit(const M2sTriangle t, float2 centre, out bool capped) {
    const M2sRelief r = m2sReliefAt(t, centre, float2(t.cell * 0.5));
    const float2 wanted = ceil(r.stretch * 0.8);
    const float most = float(params.displaceRefine);
    capped = wanted.x > most || wanted.y > most;
    return uint2(clamp(wanted, float2(1.0), float2(most)));
}

/// How many gaussians this triangle stands, one thread a triangle.
[shader("compute")]
[numthreads(64, 1, 1)]
void m2sCount(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= params.triangles) {
        return;
    }
    if (triangle < params.firstTriangle) {
        cellCounts[countAt(triangle)] = 0;   // an earlier slice's
        return;
    }
    const M2sTriangle t = m2sTriangleOf(triangle);
    if (!t.valid) {
        InterlockedAdd(counters[2], 1u);
        cellCounts[countAt(triangle)] = 0;
        return;
    }
    if (t.beyond > 0) {
        InterlockedAdd(counters[3], t.beyond);
    }
    uint inside = 0;
    uint capped = 0;
    uint unused = 0;
    if (params.simplify > 0.0) {
        inside = m2sWalkBlocks(triangle, t, false, unused, 0u, capped);
    } else for (uint k = 0; k < t.cells; ++k) {
        float3 bary;
        if (m2sCellInside(t, k, bary) && m2sCovered(t, bary)) {
            if (params.displaceRefine != 0) {
                bool cut;
                const uint2 split = m2sSplit(t, m2sCellCentre(t, k), cut);
                inside += split.x * split.y;
                capped += cut ? 1u : 0u;
            } else {
                ++inside;
            }
        }
    }
    if (capped > 0) {
        InterlockedAdd(counters[5], capped);
    }
    // A TRIANGLE SMALLER THAN A CELL STILL EXISTS.
    //
    // The density is a grid over the model and a gaussian stands where a cell
    // centre falls inside a triangle, which is what rasterising it amounts
    // to -- and a triangle smaller than a cell need contain no centre at all.
    // Rasterised, it vanishes, and that is what a rasteriser is for. Convert-
    // ed, it takes its geometry with it: the Buick's grille, its chrome and
    // its window rubbers are tens of thousands of triangles finer than a
    // centimetre, and a mesh of 80137 of them came out as 34011 gaussians
    // with the detail gone. So one that caught nothing gets a single gaussian
    // at its middle. It is the least a triangle can be worth and it is not
    // nothing.
    if (inside == 0) {
        inside = 1;
    }
    cellCounts[countAt(triangle)] = inside;
}

/// Where each triangle's gaussians start: the exclusive running sum of the
/// counts, written back over them.
///
/// One workgroup, three phases: each thread adds up a slice, thread zero runs
/// the two hundred and fifty six slice totals into a running sum, and each
/// thread walks its slice again laying its own running sum down. A serial
/// scan over the whole mesh in one thread was the other option and is what
/// this avoids: the pawn has 42892 triangles and the fox will have more.
static const uint kScanThreads = 256;
groupshared uint gScanPartial[kScanThreads];

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sScan(uint3 gid: SV_GroupThreadID) {
    const uint thread = gid.x;
    const uint total = params.triangles;
    const uint slice = (total + kScanThreads - 1) / kScanThreads;
    const uint from = min(thread * slice, total);
    const uint to = min(from + slice, total);

    uint sum = 0;
    for (uint i = from; i < to; ++i) {
        sum += cellCounts[countAt(i)];
    }
    gScanPartial[thread] = sum;
    GroupMemoryBarrierWithGroupSync();

    if (thread == 0) {
        uint running = 0;
        for (uint t = 0; t < kScanThreads; ++t) {
            const uint here = gScanPartial[t];
            gScanPartial[t] = running;
            running += here;
        }
        counters[0] = running;                              // wanted
        counters[1] = min(running, params.maxSplats);       // written
    }
    GroupMemoryBarrierWithGroupSync();

    uint running = gScanPartial[thread];
    for (uint i = from; i < to; ++i) {
        const uint here = cellCounts[countAt(i)];
        cellCounts[countAt(i)] = running;
        // The first triangle the budget cuts into: the slots are a prefix
        // in the mesh's order, so every triangle after it is cut too, and
        // the host's next slice starts here.
        // ... and a triangle the budget cuts into writes nothing: the slice
        // that starts at it writes it whole, so `written` is where it starts.
        if (here > 0 && running + here > params.maxSplats) {
            InterlockedMin(counters[4], i);
            InterlockedMin(counters[1], running);
        }
        running += here;
    }
}

/// ONE GAUSSIAN, WRITTEN: what the maps say at `bary`, turned to `normal`
/// with its first axis along `tangent`, `scale` wide. `false` where the frame
/// collapsed and nothing was written. A displaced one carries three entries
/// more: the point of the flat surface under it and how high it stands, the
/// flat surface's normal, and the relief's.
bool m2sWrite(uint triangle, const M2sTriangle t, uint slot, float3 bary, float3 position, float3 normal,
              float3 tangent, float3 scale, float mask, float3 flat, float3 flatNormal, float height) {
    const float2 uv = m2sUvAt(t, bary);
    const float2 uv2 = m2sUv2At(t, bary);
    float3 colour = params.materialColour.xyz;
    if (params.hasAlbedo != 0) {
        const float4 texel = sampleWrapped(albedo, params.albedoWidth, params.albedoHeight,
                                           params.albedoStride, params.albedoUv2 != 0 ? uv2 : uv);
        colour *= texel.xyz;
    }
    // The material's own values, and the map where there is one: glTF
    // multiplies the two and so does this, which is also what leaves a
    // material with no map at the values it authored. Theirs defaulted to
    // (0.1, 0.5) and ignored the material, so every conversion came out
    // of the same plastic.
    float metallic = params.metallic;
    float roughness = params.roughness;
    if (params.hasMetallicRoughness != 0) {
        // Blue is metallic, green is roughness: their packing.
        const float4 texel = sampleWrapped(metallicRoughness, params.mrWidth, params.mrHeight,
                                           params.mrStride, params.mrUv2 != 0 ? uv2 : uv);
        metallic *= texel.z;
        roughness *= texel.y;
    }
    float3 shading = normal;
    if (params.hasNormal != 0) {
        const float4 texel = sampleWrapped(normalMap, params.normalWidth, params.normalHeight,
                                           params.normalStride, params.normalUv2 != 0 ? uv2 : uv);
        const float3 fromMap = normalize(texel.xyz * 2.0 - 1.0);
        const float3 bitangent = normalize(cross(normal, tangent));
        const float3 straight = normalize(cross(bitangent, normal));
        shading = normalize(straight * fromMap.x + bitangent * fromMap.y + normal * fromMap.z);
    }

    // Ours, not theirs: transmission. Their conversion has no channel for
    // it, so glass came out as opaque white. What goes into the record is
    // what the material says -- its opacity, its colour taken towards the
    // transmission colour, and the transmission itself in a channel of
    // its own. What a renderer makes of that is the renderer's: a
    // translucent material is not a transparent one, and lowering the
    // opacity here would say it was.
    // What it transmits is taken off its opacity by as much as the
    // caller asked for: at one a glass gaussian is as opaque as any
    // other, which is what this did before the parameter existed, and
    // low is a ball you can see the room through. The material's colour
    // still goes towards the transmission colour either way.
    const float keptOpacity =
        params.opacity * mask * lerp(1.0, params.glassOpacity, params.transmission);
    colour = lerp(colour, colour * params.transmissionColour.xyz, params.transmission);

    const float3 shortAxis = params.useNormalMap != 0 ? shading : normal;
    float3 alongU = tangent - shortAxis * dot(shortAxis, tangent);
    if (!(dot(alongU, alongU) > 1.0e-20)) {
        return false;   // the frame collapsed: no rotation to write
    }
    alongU = normalize(alongU);
    const float3 alongV = cross(shortAxis, alongU);
    const float4 rotation = quaternionOfFrame(alongU, alongV, shortAxis);
    records[recordAt(slot, 0)] = float4(position, keptOpacity);
    records[recordAt(slot, 1)] = float4(scale, 0.0);
    records[recordAt(slot, 2)] = rotation;
    records[recordAt(slot, 3)] = float4(colour, 0.0);
    // The entries before the relief's three: 4, 6 or 8.
    const uint own = params.displaceRefine != 0 ? params.recordPixels - 3 : params.recordPixels;
    if (own >= 6) {
        records[recordAt(slot, 4)] = float4(shading, metallic);
        records[recordAt(slot, 5)] = float4(roughness, params.transmission, uv.x, uv.y);
    }
    if (own >= 8 && params.hasInfluences != 0) {
        float4 joints;
        float4 weights;
        blendInfluences(triangle, m2sOnTriangle(bary), joints, weights);
        records[recordAt(slot, 6)] = float4(joints.x, weights.x, joints.y, weights.y);
        records[recordAt(slot, 7)] = float4(joints.z, weights.z, joints.w, weights.w);
    }
    if (params.displaceRefine != 0) {
        const uint extra = own;
        records[recordAt(slot, extra)] = float4(flat, height);
        records[recordAt(slot, extra + 1)] = float4(flatNormal, 0.0);
        records[recordAt(slot, extra + 2)] = float4(normal, 0.0);
    }
    return true;
}

/// ONE CELL'S GAUSSIANS: one on the flat surface, or as many as the relief
/// splits it into, each stood on it. `centre` is the cell's centre in the
/// projection and `bary` the point it stands for; `caught` is false for the
/// one gaussian a triangle that caught no cell gets at its middle. `false`
/// once the slots run out.
bool m2sCellEmit(uint triangle, const M2sTriangle t, float2 centre, float3 bary, float mask, bool caught,
                 inout uint slot, uint end) {
    if (params.displaceRefine == 0) {
        if (slot >= end) {
            return false;
        }
        const float3 position = m2sFlatAt(t, bary);
        float3 normal = normalize(t.n0.xyz * bary.x + t.n1.xyz * bary.y + t.n2.xyz * bary.z);
        if (!(dot(normal, normal) > 0.5)) {
            normal = t.geometric;
        }
        if (m2sWrite(triangle, t, slot, bary, position, normal, t.tangent, t.scale, mask, position,
                     normal, 0.0)) {
            ++slot;
        }
        return true;
    }

    // RAISED: the cell split as far as the relief stretches it (the
    // count asked the same question at the same point), each share stood
    // on the relief and turned to its plane.
    bool cut;
    const uint2 split = caught ? m2sSplit(t, centre, cut) : uint2(1, 1);
    for (uint j = 0; j < split.y; ++j) {
        for (uint i = 0; i < split.x; ++i) {
            if (slot >= end) {
                return false;
            }
            const float2 offset = (float2(float(i), float(j)) + 0.5) / float2(split) - 0.5;
            const float2 at = centre + offset * t.cell;
            const float3 plane = m2sBaryOf(t, at);
            const float3 flat = m2sFlatAt(t, plane);
            const float3 flatNormal = m2sNormalAt(t, plane);
            const float height = m2sHeightAt(t, plane);
            const M2sRelief r = m2sReliefAt(t, at, t.cell * 0.5 / float2(split));
            float3 normal = cross(r.alongU, r.alongV);
            normal = dot(normal, normal) > 1.0e-30 ? normalize(normal) : flatNormal;
            if (dot(normal, flatNormal) < 0.0) {
                normal = -normal;
            }
            float3 tangent = r.alongU - normal * dot(normal, r.alongU);
            tangent = dot(tangent, tangent) > 1.0e-30 ? normalize(tangent) : t.tangent;
            // Its share of the stretched patch: the flat cell's size, as
            // long as the relief made it, over the gaussians it went to.
            // Where the split was capped, no longer than an uncapped
            // share: a pole of the texture coordinates stretches without
            // bound, and sized by that, a gaussian there was a spike
            // metres long across the frame. A gap is the lesser wrong.
            const float2 across =
                t.scale.xy * min(r.stretch, float2(split) * 1.25) / float2(split);
            const float3 scale = float3(across, min(across.x, across.y) * params.flatness);
            if (m2sWrite(triangle, t, slot, plane, flat + flatNormal * height, normal, tangent, scale,
                         mask, flat, flatNormal, height)) {
                ++slot;
            }
        }
    }
    return true;
}

/// FEWER GAUSSIANS WHERE THE SURFACE IS THE SAME.
///
/// A cell is a gaussian wherever the surface is, and most of a surface is
/// the same from one cell to the next: a painted panel, a flat cobble top, a
/// wall. What a gaussian carries there is carried as well by one gaussian of
/// four cells, or of sixty-four. So a triangle is walked as a tree of blocks
/// aligned to the cell grid -- 2^simplifyLevels cells a side at the top, down
/// to one -- and a block becomes one gaussian of its own size when it lies
/// wholly inside the triangle and, over every cell centre in it, what the
/// maps say moves by no more than `simplify`: the colour, metallic and
/// roughness, the cut-out (which must also stand above the cut everywhere),
/// the shading normal with its normal map, and the relief's normal (with no
/// cell of it stretched enough to split). Otherwise it goes down to its four
/// children, and a single cell is what it always was.
///
/// A block's gaussian is `size` cells wide where a cell's is one, so they
/// overlap their neighbours as a cell's do and a surface covered by either is
/// covered as much. The count and the emit walk the same tree through this
/// one function, so they ask the same questions in the same order.
static const uint kBlockStack = 16;

/// The cell's place in the triangle's walk, where it has one.
bool m2sCellIndex(const M2sTriangle t, int2 cell, out uint k) {
    const int2 at = cell - t.first;
    k = 0;
    if (at.x < 0 || at.y < 0 || uint(at.x) >= t.span.x || uint(at.y) >= t.span.y) {
        return false;
    }
    k = uint(at.y) * t.span.x + uint(at.x);
    return k < t.cells;
}

/// What the maps say at a point of the projection, as far as merging asks.
struct M2sLook {
    float3 colour;
    float2 mr;
    float  mask;
    float3 normal;   // the shading normal: interpolated, then the normal map's
    float3 relief;   // the relief's normal, the flat one without a relief
    float  stretch;  // the relief's larger stretch at one cell
};

M2sLook m2sLookAt(const M2sTriangle t, float2 at) {
    M2sLook l;
    const float3 bary = m2sBaryOf(t, at);
    const float2 uv = m2sUvAt(t, bary);
    const float2 uv2 = m2sUv2At(t, bary);
    l.colour = params.materialColour.xyz;
    if (params.hasAlbedo != 0) {
        l.colour *= sampleWrapped(albedo, params.albedoWidth, params.albedoHeight, params.albedoStride,
                                  params.albedoUv2 != 0 ? uv2 : uv).xyz;
    }
    l.mr = float2(params.metallic, params.roughness);
    if (params.hasMetallicRoughness != 0) {
        const float4 texel = sampleWrapped(metallicRoughness, params.mrWidth, params.mrHeight, params.mrStride,
                                           params.mrUv2 != 0 ? uv2 : uv);
        l.mr *= float2(texel.z, texel.y);
    }
    l.mask = m2sMaskAt(t, bary);
    const float3 flat = m2sNormalAt(t, bary);
    l.normal = flat;
    if (params.hasNormal != 0) {
        const float4 texel = sampleWrapped(normalMap, params.normalWidth, params.normalHeight,
                                           params.normalStride, params.normalUv2 != 0 ? uv2 : uv);
        const float3 fromMap = normalize(texel.xyz * 2.0 - 1.0);
        const float3 bitangent = normalize(cross(flat, t.tangent));
        const float3 straight = normalize(cross(bitangent, flat));
        l.normal = normalize(straight * fromMap.x + bitangent * fromMap.y + flat * fromMap.z);
    }
    l.relief = flat;
    l.stretch = 1.0;
    if (params.displaceRefine != 0) {
        const M2sRelief r = m2sReliefAt(t, at, float2(t.cell * 0.5));
        const float3 raised = cross(r.alongU, r.alongV);
        l.relief = dot(raised, raised) > 1.0e-30 ? normalize(raised) : flat;
        if (dot(l.relief, flat) < 0.0) {
            l.relief = -l.relief;
        }
        l.stretch = max(r.stretch.x, r.stretch.y);
    }
    return l;
}

/// Whether a block of `size` cells a side at `origin` is one gaussian.
bool m2sBlockIsOne(const M2sTriangle t, int2 origin, uint size) {
    uint first;
    uint last;
    if (!m2sCellIndex(t, origin, first) || !m2sCellIndex(t, origin + int2(int(size) - 1), last)) {
        return false;   // part of it is past the walk's bound
    }
    // Wholly inside, and so is all its gaussian reaches -- a block past each
    // side: the triangle's plane and coordinates go on past its edge, but the
    // surface may not, and at a mesh's border a merged block's tail hung in
    // the air beyond it. All four corners of that, with a margin, so that a
    // corner on the edge two triangles share is nobody's to merge.
    const float margin = 1.0e-4;
    for (uint c = 0; c < 4; ++c) {
        const int2 corner = origin - int2(int(size)) + int2(int((c & 1u) * size * 3), int((c >> 1) * size * 3));
        const float3 bary = m2sBaryOf(t, float2(corner) * t.cell);
        if (bary.x < margin || bary.y < margin || bary.z < margin) {
            return false;
        }
    }
    const float tolerance = params.simplify;
    float3 colourLow = float3(3.4e38);
    float3 colourHigh = float3(-3.4e38);
    float2 mrLow = float2(3.4e38);
    float2 mrHigh = float2(-3.4e38);
    float maskLow = 3.4e38;
    float maskHigh = -3.4e38;
    float3 normalFirst = float3(0.0);
    float3 reliefFirst = float3(0.0);
    // OVER WHAT THE GAUSSIAN REACHES, NOT ONLY OVER THE BLOCK. A block's
    // gaussian is as wide as the block, as a cell's is as wide as a cell, and
    // its tail reaches a block past each side: judged on the block alone, a
    // cobble's flat top merged right up to the mortar and the mortar came out
    // blurred a block wide. So the cells a block beyond each side must agree
    // as well -- read on the triangle's own plane and coordinates, which go on
    // past its edge. The block itself is read first: it is where a block most
    // often fails, and the cheaper place to find out.
    const int reach = int(size);
    for (uint pass = 0; pass < 2; ++pass) {
    const int from = pass == 0 ? 0 : -reach;
    const int to = pass == 0 ? int(size) : int(size) + reach;
    for (int j = from; j < to; ++j) {
        for (int i = from; i < to; ++i) {
            const bool inBlock = i >= 0 && j >= 0 && i < int(size) && j < int(size);
            if (pass == 1 && inBlock) {
                continue;   // read in the first pass
            }
            const M2sLook l = m2sLookAt(t, (float2(origin + int2(i, j)) + 0.5) * t.cell);
            if (params.opacityChannel != 0 && !(l.mask >= params.opacityCut)) {
                return false;   // a hole in it: the cells say where
            }
            if (l.stretch * 0.8 > 1.0) {
                return false;   // a cell the relief would split
            }
            if (pass == 0 && i == 0 && j == 0) {
                normalFirst = l.normal;
                reliefFirst = l.relief;
            } else if (length(l.normal - normalFirst) > tolerance || length(l.relief - reliefFirst) > tolerance) {
                return false;
            }
            colourLow = min(colourLow, l.colour);
            colourHigh = max(colourHigh, l.colour);
            mrLow = min(mrLow, l.mr);
            mrHigh = max(mrHigh, l.mr);
            maskLow = min(maskLow, l.mask);
            maskHigh = max(maskHigh, l.mask);
            if (any(colourHigh - colourLow > tolerance) || any(mrHigh - mrLow > tolerance) ||
                maskHigh - maskLow > tolerance) {
                return false;
            }
        }
    }
    }
    return true;
}

/// One block's gaussian: at its centre, `size` cells wide, stood on the relief
/// where there is one.
bool m2sBlockEmit(uint triangle, const M2sTriangle t, int2 origin, uint size, inout uint slot, uint end) {
    if (slot >= end) {
        return false;
    }
    const float2 centre = (float2(origin) + float(size) * 0.5) * t.cell;
    const float3 bary = m2sBaryOf(t, centre);
    const float mask = m2sMaskAt(t, bary);
    const float3 flat = m2sFlatAt(t, bary);
    const float3 flatNormal = m2sNormalAt(t, bary);
    float3 scale = float3(t.scale.xy * float(size), 0.0);
    if (params.displaceRefine == 0) {
        scale.z = min(scale.x, scale.y) * params.flatness;
        if (m2sWrite(triangle, t, slot, bary, flat, flatNormal, t.tangent, scale, mask, flat, flatNormal, 0.0)) {
            ++slot;
        }
        return true;
    }
    const float height = m2sHeightAt(t, bary);
    const M2sRelief r = m2sReliefAt(t, centre, float2(t.cell * float(size) * 0.5));
    float3 normal = cross(r.alongU, r.alongV);
    normal = dot(normal, normal) > 1.0e-30 ? normalize(normal) : flatNormal;
    if (dot(normal, flatNormal) < 0.0) {
        normal = -normal;
    }
    float3 tangent = r.alongU - normal * dot(normal, r.alongU);
    tangent = dot(tangent, tangent) > 1.0e-30 ? normalize(tangent) : t.tangent;
    scale.xy *= min(r.stretch, float2(1.25));
    scale.z = min(scale.x, scale.y) * params.flatness;
    if (m2sWrite(triangle, t, slot, bary, flat + flatNormal * height, normal, tangent, scale, mask, flat,
                 flatNormal, height)) {
        ++slot;
    }
    return true;
}

/// The walk, for the count (`write` false: how many) and for the emit (the
/// same, written from `slot` up to `end`).
uint m2sWalkBlocks(uint triangle, const M2sTriangle t, bool write, inout uint slot, uint end, inout uint capped) {
    const uint levels = min(params.simplifyLevels, 5u);   // the stack holds 1 + 3 * 5
    const uint top = 1u << levels;
    const int2 low = int2(floor(float2(t.first) / float(top)));
    const int2 high = int2(floor(float2(t.first + int2(t.span) - 1) / float(top)));
    uint count = 0;
    int2 stackOrigin[kBlockStack];
    uint stackLevel[kBlockStack];
    for (int by = low.y; by <= high.y; ++by) {
        for (int bx = low.x; bx <= high.x; ++bx) {
            uint depth = 0;
            stackOrigin[0] = int2(bx, by) * int(top);
            stackLevel[0] = levels;
            depth = 1;
            while (depth > 0) {
                --depth;
                const int2 origin = stackOrigin[depth];
                const uint level = stackLevel[depth];
                if (level == 0) {
                    uint k;
                    float3 bary;
                    if (!m2sCellIndex(t, origin, k) || !m2sCellInside(t, k, bary) || !m2sCovered(t, bary)) {
                        continue;
                    }
                    if (params.displaceRefine != 0) {
                        bool cut;
                        const uint2 split = m2sSplit(t, m2sCellCentre(t, k), cut);
                        count += split.x * split.y;
                        capped += cut ? 1u : 0u;
                    } else {
                        count += 1;
                    }
                    if (write && !m2sCellEmit(triangle, t, m2sCellCentre(t, k), bary, m2sMaskAt(t, bary), true,
                                              slot, end)) {
                        return count;
                    }
                    continue;
                }
                const uint size = 1u << level;
                if (m2sBlockIsOne(t, origin, size)) {
                    count += 1;
                    if (write && !m2sBlockEmit(triangle, t, origin, size, slot, end)) {
                        return count;
                    }
                    continue;
                }
                // Its four children, the first on top so the walk is in order.
                const int half = int(size / 2);
                for (int c = 3; c >= 0; --c) {
                    stackOrigin[depth] = origin + int2((c & 1) * half, (c >> 1) * half);
                    stackLevel[depth] = level - 1;
                    ++depth;
                }
            }
        }
    }
    return count;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sEmit(uint3 tid: SV_DispatchThreadID) {
    const uint triangle = tid.x;
    if (triangle >= params.triangles) {
        return;
    }
    if (triangle < params.firstTriangle || triangle >= counters[4]) {
        return;   // an earlier slice's, or the budget cut into it: the next slice's
    }
    const M2sTriangle t = m2sTriangleOf(triangle);
    if (!t.valid) {
        return;
    }

    // Where this triangle's gaussians go, which the scan settled. A budget
    // too small then keeps the first of them, in the mesh's own order, and
    // not whichever ones won a race.
    uint slot = cellCounts[countAt(triangle)];
    // AND WHERE THEY END, which is where the next triangle's start: this
    // kernel writes the slots the count gave it, no more and no fewer.
    //
    // The two passes ask the same questions of the same functions, and they
    // are still two kernels, which the compiler contracts and fuses each its
    // own way. A cell centre exactly on the edge two triangles share -- a
    // quad's diagonal on a grid that halves it -- came out inside for the
    // count and outside for the emit once the emit changed shape: three
    // slots given, two written, and the third a gaussian of zero size at the
    // origin. Where the relief splits a cell, `ceil` of a stretch is the same
    // kind of edge. So the emit stops at its last slot, and a slot it has
    // nothing for is written as a gaussian that stops nothing.
    const uint end = min(triangle + 1 < params.triangles ? cellCounts[countAt(triangle + 1)] : counters[0],
                         params.maxSplats);

    // The tree of blocks, where the caller asked for fewer gaussians; a
    // triangle it finds nothing in falls through to the one at its middle.
    if (params.simplify > 0.0) {
        uint none = 0;
        uint probe = 0;
        if (m2sWalkBlocks(triangle, t, false, probe, 0u, none) > 0) {
            m2sWalkBlocks(triangle, t, true, slot, end, none);
            const float3 middle = float3(1.0 / 3.0);
            while (slot < end) {
                m2sWrite(triangle, t, slot, middle, m2sFlatAt(t, middle), m2sNormalAt(t, middle), t.tangent,
                         t.scale, 0.0, m2sFlatAt(t, middle), m2sNormalAt(t, middle), 0.0);
                ++slot;
            }
            return;
        }
    }

    // Whether any cell caught it at all; one that caught none is worth a
    // gaussian at its middle (m2sCount says why).
    bool caught = false;
    for (uint k = 0; k < t.cells; ++k) {
        float3 bary;
        if (m2sCellInside(t, k, bary) && m2sCovered(t, bary)) {
            caught = true;
            break;
        }
    }
    const uint walk = caught ? t.cells : 1u;
    for (uint k = 0; k < walk; ++k) {
        float3 bary = float3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0);
        if (caught && !m2sCellInside(t, k, bary)) {
            continue;   // the cell's centre is outside this triangle
        }
        // THE CUT-OUT'S VALUE IS THE GAUSSIAN'S OPACITY, not a yes or a no.
        // A feather's barb is a texel of alpha 0.3, and a gaussian of
        // opacity 0.3 is what blends the way the card did; cut at a half,
        // every barb was either a whole gaussian or nothing.
        const float mask = caught ? m2sMaskAt(t, bary) : 1.0;
        if (caught && !(mask >= params.opacityCut)) {
            continue;   // the cut-out says there is no surface here
        }
        const float2 centre = caught ? m2sCellCentre(t, k) : t.oa + t.edge1 * bary.y + t.edge2 * bary.z;
        if (!m2sCellEmit(triangle, t, centre, bary, mask, caught, slot, end)) {
            return;
        }
    }
    // What the count gave and nothing here filled: at the triangle's middle,
    // its own size, and clear.
    const float3 middle = float3(1.0 / 3.0);
    while (slot < end) {
        m2sWrite(triangle, t, slot, middle, m2sFlatAt(t, middle), m2sNormalAt(t, middle), t.tangent, t.scale,
                 0.0, m2sFlatAt(t, middle), m2sNormalAt(t, middle), 0.0);
        ++slot;
    }
}
