// Copyright (c) 2026 jesus luque.
//
// Which groups of a level (or which splats) this frame draws. See lod_common.
module lod_cut;
import athenea.common.morton;
import lod_common;

StructuredBuffer<uint>    cells;      // per group: its cell code at `level`; per splat: its code
StructuredBuffer<uint>    pbr;        // per group, where params.pbrs: its material (lodKeptApart)
RWStructuredBuffer<uint>  selected;
ConstantBuffer<LodParams> params;     // count; level (of `cells`); threshold; eye; coarsest

/// The pixels a level-`level` cell `cell` spans at its nearest point to the eye.
float projectedEdge(uint cell, uint level) {
    const float edge = params.extent / float(1u << level);
    if (params.orthographic != 0) {
        return edge * params.pixelsPerUnit;
    }
    const float3 lo = float3(params.boundsLoX, params.boundsLoY, params.boundsLoZ) + float3(cellIndex(cell)) * edge;
    const float3 eye = float3(params.eyeX, params.eyeY, params.eyeZ);
    const float3 nearest = clamp(eye, lo, lo + edge);
    // Inside the cell: as near as it gets, so as big as it gets.
    const float distance = max(length(nearest - eye), 1e-6);
    return edge * params.pixelsPerUnit / distance;
}

bool small(uint cell, uint level) {
    return params.threshold > 0.0 && projectedEdge(cell, level) <= params.threshold;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lodCutGroups(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint cell = cells[g];
    // Glass and sheets are never drawn merged: their groups hold nothing
    // else (lod_class), and their parents are glass too.
    const bool kept = params.pbrs != 0 && lodKeptApart(pbr[g]);
    const bool here = !kept && small(cell, params.level);
    const bool above = !kept && params.coarsest == 0 && small(cell >> 3, params.level - 1);
    selected[g] = (here && !above) ? 1u : 0u;
}

// The finest merged level decides for the splats under it. Its per-group
// state: bit 0 when the cell wants its splats (it is not small), bit 1 when
// they are drawn -- wanted, and every chunk holding them on the device.
StructuredBuffer<uint>   starts;           // finest level: each group's first splat
StructuredBuffer<uint>   residentChunks;   // per chunk: 1 when on the device
RWStructuredBuffer<uint> state;            // finest level, per group
StructuredBuffer<uint>   groups;           // per splat in the store: its finest-level group

static const uint kWanted = 1;
static const uint kDrawn = 2;
static const uint kKept = 4;   // glass or a sheet: nothing drawn in its place while it is away

/// The finest merged level: drawn as any level is, and also wherever its cell
/// wants splats that are not all on the device yet.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodCutFinest(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint cell = cells[g];
    const bool kept = params.pbrs != 0 && lodKeptApart(pbr[g]);
    const bool here = !kept && small(cell, params.level);
    const bool above = !kept && params.coarsest == 0 && small(cell >> 3, params.level - 1);
    bool resident = true;
    if (!here) {
        const uint end = g + 1 < params.count ? starts[g + 1] : params.splats;
        const uint last = (end - 1) / params.chunkSplats;
        for (uint c = starts[g] / params.chunkSplats; c <= last; ++c) {
            resident = resident && residentChunks[c] != 0;
        }
    }
    // Glass and sheets not on the device yet draw nothing, not their merge:
    // a merge of a glass's splats is no glass (lodKeptApart), and the Corvette's
    // windscreen drew black blocks over the cabin until its chunks came.
    selected[g] = ((here && !above) || (!here && !resident && !kept)) ? 1u : 0u;
    state[g] = (here ? 0u : kWanted) | ((!here && resident) ? kDrawn : 0u) | (kept ? kKept : 0u);
}

/// Splats of a run of the store, from `offset`: drawn when their group says.
[shader("compute")]
[numthreads(256, 1, 1)]
void lodCutSplats(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    selected[i] = (state[groups[params.offset + i]] & kDrawn) != 0 ? 1u : 0u;
}

/// Per chunk: 0 when no finest-level group with splats in it wants them, else
/// how badly -- the largest cell edge among those that do, in 1/16 pixels.
[shader("compute")]
[numthreads(64, 1, 1)]
void lodChunkNeeds(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.chunks) {
        return;
    }
    const uint lo = c * params.chunkSplats;
    const uint hi = lo + params.chunkSplats;
    // The group holding the chunk's first splat: the last starting at or before it.
    uint first = 0;
    uint last = params.count;
    while (last - first > 1) {
        const uint middle = (first + last) / 2;
        if (starts[middle] <= lo) {
            first = middle;
        } else {
            last = middle;
        }
    }
    float largest = 0.0;
    for (uint g = first; g < params.count && starts[g] < hi; ++g) {
        if ((state[g] & kWanted) != 0) {
            // Glass draws nothing until it comes (lodCutFinest): it is wanted
            // as a cell four times the threshold is, ahead of what a merge
            // stands in for, where it is not wanted more by its own size.
            const float edge = projectedEdge(cells[g], params.level);
            largest = max(largest, (state[g] & kKept) != 0 ? max(edge, 4.0 * params.threshold) : edge);
        }
    }
    selected[c] = largest > 0.0 ? uint(clamp(largest * 16.0, 1.0, 4294967040.0)) : 0u;
}
