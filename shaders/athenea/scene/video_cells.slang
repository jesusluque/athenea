// Copyright (c) 2026 jesus luque.
//
// A STABLE ORDER BY SPACE (athenea video-encode --cells): frames whose gaussians
// share no identity (each a fit of its own, a volume's Eulerian grid) laid out so
// that index i is the same place every frame. The sequence's box is cut into
// cubic cells; each cell owns a fixed run of slots, as many as the most
// gaussians it holds in any frame; within a cell a frame's gaussians take its
// slots in Morton order of their place in the cell, and the cell's slots past
// them are dead (a speck of no extinction and no opacity at the cell's middle).
// The host does the bookkeeping: the box from the bounds, each cell's most and
// its slots' start, a frame's first sorted index of each cell.
//
//   videoCellBounds   the frame's box, as ordered uints (atomic min / max)
//   videoCellKeys     a splat's key, the cell over its Morton code in the cell
//                     (subBits an axis, up to 10), and one count more for its cell
//   videoCellDead     every slot a dead speck at its cell's middle
//   videoCellScatter  the sorted splats into their slots
//   videoPadSlots     a frame's splats kept in their slots, the slots past its
//                     count dead specks at the origin (`--slots`)
module video_cells;
import athenea.common.packing;

struct CellParams {
    uint   count;      ///< splats of the frame (keys, scatter) or slots (dead)
    uint   shWords;
    uint   cellsX;
    uint   cellsY;
    uint   cellsZ;
    uint   cells;
    float  cellSize;
    float  deadLogSigma;   ///< what a dead slot's first rest coefficient holds
    float3 boxMin;
    uint   subBits;        ///< Morton bits an axis within a cell: (32 - the cell's bits) / 3
};

RWStructuredBuffer<uint>    cellBounds;    // 6: min xyz, max xyz as ordered uints
RWStructuredBuffer<uint>    cellCounts;    // a frame's gaussians a cell
RWStructuredBuffer<uint>    cellKeys;      // a splat's key
RWStructuredBuffer<uint>    cellValues;    // a splat's index (sorted with its key)
StructuredBuffer<uint>      cellFirst;     // the frame's first sorted index of each cell
StructuredBuffer<uint>      cellBase;      // each cell's first slot
StructuredBuffer<uint>      cellOfSlot;    // each slot's cell
StructuredBuffer<float4>    inPositions;
StructuredBuffer<uint>      inShape;
StructuredBuffer<uint>      inSh;
RWStructuredBuffer<float4>  outPositions;
RWStructuredBuffer<uint>    outShape;
RWStructuredBuffer<uint>    outSh;
ConstantBuffer<CellParams>  cellParams;

/// A dead slot's log deviation, as vdb2splat --cloud writes one (1e-6): a
/// speck, so the many a frame may hold at one place add no optical depth.
static const float kDeadLogScale = -13.815511;

uint orderedOf(float f) {
    const uint u = asuint(f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

uint spread10(uint v) {
    uint x = v & 1023u;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoCellBounds(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= cellParams.count) return;
    const float3 p = inPositions[i].xyz;
    for (uint a = 0; a < 3; ++a) {
        InterlockedMin(cellBounds[a], orderedOf(p[a]));
        InterlockedMax(cellBounds[3 + a], orderedOf(p[a]));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoCellKeys(uint3 tid: SV_DispatchThreadID) {
    const CellParams p = cellParams;
    const uint i = tid.x;
    if (i >= p.count) return;
    const float3 at = (inPositions[i].xyz - p.boxMin) / p.cellSize;
    const uint3 dims = uint3(p.cellsX, p.cellsY, p.cellsZ);
    const uint3 c = min(uint3(max(at, float3(0.0))), dims - 1u);
    const uint cell = (c.z * p.cellsY + c.y) * p.cellsX + c.x;
    const uint side = 1u << p.subBits;
    const uint3 sub = min(uint3(saturate(at - float3(c)) * float(side)), uint3(side - 1u));
    cellKeys[i] = (cell << (3u * p.subBits)) | spread10(sub.x) | (spread10(sub.y) << 1) | (spread10(sub.z) << 2);
    cellValues[i] = i;
    InterlockedAdd(cellCounts[cell], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoCellDead(uint3 tid: SV_DispatchThreadID) {
    const CellParams p = cellParams;
    const uint t = tid.x;
    if (t >= p.count) return;
    const uint cell = cellOfSlot[t];
    const uint x = cell % p.cellsX;
    const uint y = (cell / p.cellsX) % p.cellsY;
    const uint z = cell / (p.cellsX * p.cellsY);
    outPositions[t] = float4(p.boxMin + (float3(x, y, z) + 0.5) * p.cellSize, 0.0);
    outShape[t * 4 + 0] = encodeQuaternion(float4(0.0, 0.0, 0.0, 1.0));
    outShape[t * 4 + 1] = packHalves(kDeadLogScale, kDeadLogScale);
    outShape[t * 4 + 2] = packHalves(kDeadLogScale, 0.5);
    outShape[t * 4 + 3] = packHalves(0.5, 0.5);
    for (uint w = 0; w < p.shWords; ++w) {
        outSh[t * p.shWords + w] = w == 0 ? packHalves(p.deadLogSigma, 0.0) : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoCellScatter(uint3 tid: SV_DispatchThreadID) {
    const CellParams p = cellParams;
    const uint j = tid.x;
    if (j >= p.count) return;
    const uint cell = cellKeys[j] >> (3u * p.subBits);
    const uint s = cellValues[j];
    const uint d = cellBase[cell] + (j - cellFirst[cell]);
    outPositions[d] = inPositions[s];
    for (uint w = 0; w < 4; ++w) outShape[d * 4 + w] = inShape[s * 4 + w];
    for (uint w = 0; w < p.shWords; ++w) outSh[d * p.shWords + w] = inSh[s * p.shWords + w];
}

/// `--slots`: slot t < count is the frame's splat t; every slot past it, up
/// to `cells` (the slots a frame), a dead speck at the origin -- where a dead
/// slot of `vdb2splat --cloud` stands -- with ln sigma `deadLogSigma`.
[shader("compute")]
[numthreads(256, 1, 1)]
void videoPadSlots(uint3 tid: SV_DispatchThreadID) {
    const CellParams p = cellParams;
    const uint t = tid.x;
    if (t >= p.cells) return;
    if (t < p.count) {
        outPositions[t] = inPositions[t];
        for (uint w = 0; w < 4; ++w) outShape[t * 4 + w] = inShape[t * 4 + w];
        for (uint w = 0; w < p.shWords; ++w) outSh[t * p.shWords + w] = inSh[t * p.shWords + w];
        return;
    }
    outPositions[t] = float4(0.0, 0.0, 0.0, 0.0);
    outShape[t * 4 + 0] = encodeQuaternion(float4(0.0, 0.0, 0.0, 1.0));
    outShape[t * 4 + 1] = packHalves(kDeadLogScale, kDeadLogScale);
    outShape[t * 4 + 2] = packHalves(kDeadLogScale, 0.5);
    outShape[t * 4 + 3] = packHalves(0.5, 0.5);
    for (uint w = 0; w < p.shWords; ++w) {
        outSh[t * p.shWords + w] = w == 0 ? packHalves(p.deadLogSigma, 0.0) : 0u;
    }
}
