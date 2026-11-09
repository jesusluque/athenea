// Copyright (c) 2026 jesus luque.
//
// The selected Gaussians of one level (or the splats) copied into this frame's
// cloud, compacted: element i goes to base + dest[i] when selected[i].
//
// THE CUT MORPHED (LodParams::morph). The cut switches a parent for its
// children at one threshold, and the switch shows as a pop: the parent is
// one gaussian where its children were many. Each child is drawn instead a
// fraction t of the way to its parent, t rising to 1 as the parent's cell
// shrinks to its threshold: position, log-scales, rotation (nlerp, sign
// aligned), base colour and harmonics, normal and emission. Its opacity
// keeps its mass, opacity x two-axis area (the weight its parent was merged
// with, lod_common), so at t = 1 the children stack where the parent stands
// with the parent's mass between them, and when the cut switches the image
// does not. The parent read is its own: a glass or sheet parent (never drawn
// merged) is not morphed to.
module lod_gather;
import athenea.common.packing;
import lod_common;

StructuredBuffer<uint>     selected;
StructuredBuffer<uint>     dest;
StructuredBuffer<float4>   srcPositions;
StructuredBuffer<uint>     srcShape;
StructuredBuffer<uint>     srcSh;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
RWStructuredBuffer<uint>   sh;
// The shading normals, where `params.normals` says the cloud keeps them.
StructuredBuffer<uint>     srcNormals;
RWStructuredBuffer<uint>   normals;
// The radiance given off, where `params.emission` says the cloud keeps it.
StructuredBuffer<uint>     srcEmission;
RWStructuredBuffer<uint>   emission;
// The morph's parents (params.morphParent): this level's cells, the parent
// level's cells and gaussians, its materials where params.pbrs, and each
// splat's finest-level group.
StructuredBuffer<uint>     cells;
StructuredBuffer<uint>     parentCells;
StructuredBuffer<float4>   parentPositions;
StructuredBuffer<uint>     parentShape;
StructuredBuffer<uint>     parentSh;
StructuredBuffer<uint>     parentNormals;
StructuredBuffer<uint>     parentEmission;
StructuredBuffer<uint>     parentPbr;
StructuredBuffer<uint>     groups;
ConstantBuffer<LodParams>  params;

/// The parent level's group holding `cell` (its code at the parent's level):
/// a level's groups are in code order (lod_groups).
uint parentOf(uint cell) {
    uint lo = 0;
    uint hi = params.parentGroups;
    while (lo < hi) {
        const uint middle = (lo + hi) / 2;
        if (parentCells[middle] < cell) {
            lo = middle + 1;
        } else {
            hi = middle;
        }
    }
    return lo;
}

float3 shapeScale(uint row, StructuredBuffer<uint> from) {
    return float3(lowHalf(from[row + 1]), highHalf(from[row + 1]), lowHalf(from[row + 2]));
}

float3 shapeBase(uint row, StructuredBuffer<uint> from) {
    return float3(highHalf(from[row + 2]), lowHalf(from[row + 3]), highHalf(from[row + 3]));
}

uint lerpHalves(uint a, uint b, float t) {
    return packHalves(lerp(lowHalf(a), lowHalf(b), t), lerp(highHalf(a), highHalf(b), t));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lodGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || selected[i] == 0) {
        return;
    }
    const uint j = params.base + dest[i];
    const uint k = params.offset + i;   // chunks share their cloud's buffers from `offset`

    // The parent and how far toward it.
    float t = 0.0;
    uint q = 0;
    if (params.morphParent == 1u) {
        const uint cell = cells[k] >> 3;
        q = parentOf(cell);
        t = q < params.parentGroups && parentCells[q] == cell ? lodMorph(params, cell, params.level - 1) : 0.0;
    } else if (params.morphParent == 2u) {
        q = groups[k];
        t = lodMorph(params, parentCells[q], params.level);
    }
    if (t > 0.0 && params.pbrs != 0 && lodKeptApart(parentPbr[q])) {
        t = 0.0;
    }

    if (t <= 0.0) {
        positions[j] = srcPositions[k];
        for (uint w = 0; w < 4; ++w) {
            shape[j * 4 + w] = srcShape[k * 4 + w];
        }
        for (uint w = 0; w < params.shWords; ++w) {
            sh[j * params.shWords + w] = srcSh[k * params.shWords + w];
        }
        if (params.normals != 0) {
            normals[j] = srcNormals[k];
        }
        if (params.emission != 0) {
            emission[j] = srcEmission[k];
        }
        return;
    }

    const float4 own = srcPositions[k];
    const float4 parent = parentPositions[q];
    const uint row = k * 4;
    const uint parentRow = q * 4;
    const float3 ownLog = shapeScale(row, srcShape);
    const float3 parentLog = shapeScale(parentRow, parentShape);
    const float3 logScale = lerp(ownLog, parentLog, t);
    float4 rotation = decodeQuaternion(srcShape[row]);
    float4 parentRotation = decodeQuaternion(parentShape[parentRow]);
    parentRotation = dot(rotation, parentRotation) < 0.0 ? -parentRotation : parentRotation;
    rotation = normalize(lerp(rotation, parentRotation, t));
    const float3 base = lerp(shapeBase(row, srcShape), shapeBase(parentRow, parentShape), t);
    // Its mass kept: at t = 1 the children's opacities sum to the parent's.
    const float mass = own.w * discArea(exp(ownLog));
    const float opacity = min(mass / max(discArea(exp(logScale)), 1e-30), max(own.w, parent.w));

    positions[j] = float4(lerp(own.xyz, parent.xyz, t), opacity);
    shape[j * 4] = encodeQuaternion(rotation);
    shape[j * 4 + 1] = packHalves(halfSafe(logScale.x), halfSafe(logScale.y));
    shape[j * 4 + 2] = packHalves(halfSafe(logScale.z), halfSafe(base.x));
    shape[j * 4 + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));
    for (uint w = 0; w < params.shWords; ++w) {
        sh[j * params.shWords + w] = lerpHalves(srcSh[k * params.shWords + w], parentSh[q * params.shWords + w], t);
    }
    if (params.normals != 0) {
        const float3 n = lerp(unpackNormal(srcNormals[k]), unpackNormal(parentNormals[q]), t);
        normals[j] = dot(n, n) > 1e-12 ? packNormal(normalize(n)) : srcNormals[k];
    }
    if (params.emission != 0) {
        emission[j] = packRgb9e5(lerp(unpackRgb9e5(srcEmission[k]), unpackRgb9e5(parentEmission[q]), t));
    }
}
