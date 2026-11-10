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

StructuredBuffer<uint>     state;        // the finest level's per-group state (lod_cut): bit 1, its splats are drawn
StructuredBuffer<uint>     splatBases;   // per finest-level group: the splats drawn by the groups before it
StructuredBuffer<uint>     starts;       // per finest-level group: its first splat in the Morton order

/// Gaussian k of the source written as gaussian j of the frame's cloud.
/// `thinMore` over 1 is a thinned splat (lod_common): it carries that many
/// times its mass -- a little by width, the rest by opacity up to 1 -- and is
/// drawn `thinToward` of the way to its group's merged gaussian, its opacity
/// times `thinFade` (it is going out).
void emit(uint j, uint k, float thinMore = 1.0, float thinToward = 0.0, float thinFade = 1.0) {
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
    const float wider = min(sqrt(sqrt(thinMore)), 1.5);
    if (thinMore > 1.0) {
        // A thinned splat's own way to its group, whatever the morph says.
        q = groups[k];
        t = thinToward;
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
        if (thinMore > 1.0) {
            const float opacity = positions[j].w;
            positions[j].w = min(opacity * thinMore / (wider * wider), max(opacity, 1.0)) * thinFade;
            const float add = log(wider);
            const uint a = shape[j * 4 + 1];
            const uint b = shape[j * 4 + 2];
            shape[j * 4 + 1] = packHalves(halfSafe(lowHalf(a) + add), halfSafe(highHalf(a) + add));
            shape[j * 4 + 2] = packHalves(halfSafe(lowHalf(b) + add), highHalf(b));
        }
        return;
    }

    const float4 own = srcPositions[k];
    const float4 parent = parentPositions[q];
    const uint row = k * 4;
    const uint parentRow = q * 4;
    const float3 ownLog = shapeScale(row, srcShape);
    const float3 parentLog = shapeScale(parentRow, parentShape);
    const float3 logScale = lerp(ownLog + log(wider), parentLog, t);
    float4 rotation = decodeQuaternion(srcShape[row]);
    float4 parentRotation = decodeQuaternion(parentShape[parentRow]);
    parentRotation = dot(rotation, parentRotation) < 0.0 ? -parentRotation : parentRotation;
    rotation = normalize(lerp(rotation, parentRotation, t));
    const float3 base = lerp(shapeBase(row, srcShape), shapeBase(parentRow, parentShape), t);
    // Its mass kept: at t = 1 the children's opacities sum to the parent's.
    const float mass = own.w * thinMore * discArea(exp(ownLog));
    const float plain = thinMore > 1.0 ? min(own.w * thinMore / (wider * wider), max(own.w, 1.0)) : own.w;
    const float opacity = min(mass / max(discArea(exp(logScale)), 1e-30), max(plain, parent.w));

    positions[j] = float4(lerp(own.xyz, parent.xyz, t), opacity * thinFade);
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

[shader("compute")]
[numthreads(256, 1, 1)]
void lodGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || selected[i] == 0) {
        return;
    }
    emit(params.base + dest[i], params.offset + i);   // chunks share their cloud's buffers from `offset`
}

/// THE SPLATS APPENDED. A cut made every frame (a budgeted one, for a moving
/// eye) spent most of its time on its splats: a selection, a prefix sum over
/// every splat in the store and a copy of the selection to tell the next cut
/// by. Here the finest level's groups say how many splats each draws, a
/// prefix sum over the groups (a few million, not the splats) says where
/// each group's begin, and a splat asks its group's state and takes its own
/// place: its group's, plus how far into the group it is in the Morton
/// order. One pass over the splats, and the order they land in is the
/// store's, the same every frame.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodGatherAppend(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint k = params.offset + i;
    const uint g = groups[k];
    if ((state[g] & 2u) == 0u) {
        return;
    }
    // The splats thinned (lod_common): one the fraction drops is not drawn,
    // and one it keeps stands after the kept ones before it in its group.
    const uint morton = params.mortonBase + i;
    const bool whole = params.pbrs != 0 && lodKeptApart(parentPbr[g]);
    const float fraction = whole ? 1.0 : lodThinFraction(params, parentCells[g], params.level);
    uint rank = morton - starts[g];
    if (fraction < 1.0) {
        if (!lodThinKeeps(morton, fraction)) {
            return;
        }
        rank = 0u;
        for (uint m = starts[g]; m < morton; ++m) {
            rank += lodThinKeeps(m, fraction) ? 1u : 0u;
        }
    }
    const uint at = splatBases[g] + rank;
    if (at >= params.appendLimit) {
        return;
    }
    // Toward its group over the band's last half: what is left of a group at
    // the switch is the merged gaussian that takes its place.
    const float left = fraction < 1.0 ? 1.0 - log2(fraction) / log2(kLodThinLeast) : 1.0;
    emit(params.base + at, k, fraction < 1.0 ? kLodThinFadeMass / fraction : 1.0,
         fraction < 1.0 ? saturate(1.0 - 2.0 * left) : 0.0, lodThinFade(morton, fraction));
}

/// THE SPLATS BY THEIR GROUPS. Where the store is the cloud's Morton order
/// (every chunk in its own slot: splat m of the order is splat m of the
/// store), a thread a finest-level group writes what the appended gather
/// would, splat by splat of its run: the fraction is asked once a group, and
/// a splat's place is a count kept along the run, where the appended gather
/// counts the kept ones before each splat anew. Thinned, that pass over every
/// splat of a town cost 19 to 38 ms.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodGatherRuns(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count || (state[g] & 2u) == 0u) {
        return;
    }
    const bool whole = params.pbrs != 0 && lodKeptApart(parentPbr[g]);
    const float fraction = whole ? 1.0 : lodThinFraction(params, parentCells[g], params.level);
    const float left = fraction < 1.0 ? 1.0 - log2(fraction) / log2(kLodThinLeast) : 1.0;
    const float more = fraction < 1.0 ? kLodThinFadeMass / fraction : 1.0;
    const float toward = fraction < 1.0 ? saturate(1.0 - 2.0 * left) : 0.0;
    const uint last = g + 1 < params.count ? starts[g + 1] : params.splats;
    uint at = splatBases[g];
    for (uint m = starts[g]; m < last; ++m) {
        if (!lodThinKeeps(m, fraction)) {
            continue;
        }
        if (at < params.appendLimit) {
            emit(params.base + at, m, more, toward, lodThinFade(m, fraction));
        }
        ++at;
    }
}
