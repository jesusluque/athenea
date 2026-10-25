// Copyright (c) 2026 jesus luque.
//
// A FLATTENED CLOUD CUT INTO CHUNKS THAT LIE TOGETHER (athenea flatten
// --chunk-splats): the pages of a live stream, one chunk a page.
//
// - `flatChunkKeys`: each gaussian's Morton code -- ten bits an axis of its
//   position in the box the cloud fills -- and its own index, for the radix
//   sort (algo/radix_*.slang) that orders them.
// - `flatChunkGather`: the records in that order. A chunk is then a run of
//   `chunkSplats` of them, which the processor copies out on the device.
module flatten_chunks;
import flatten_basis;

struct ChunkParams {
    uint   count;    // gaussians in `records`
    float3 lo;       // the box's corner
    float3 scale;    // 1023 over the box's extent, per axis (0 on a flat axis)
};
ConstantBuffer<ChunkParams> params;
StructuredBuffer<float>     records;
RWStructuredBuffer<float>   sorted;
RWStructuredBuffer<uint>    keys;
RWStructuredBuffer<uint>    order;

/// Ten bits spread to every third bit.
uint spread(uint v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flatChunkKeys(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint at = g * kFlatStride + kFlatPosition;
    const float3 p = float3(records[at], records[at + 1], records[at + 2]);
    const uint3 cell = uint3(clamp((p - params.lo) * params.scale, float3(0.0), float3(1023.0)));
    keys[g] = spread(cell.x) | (spread(cell.y) << 1) | (spread(cell.z) << 2);
    order[g] = g;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void flatChunkGather(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint from = order[g] * kFlatStride;
    const uint to = g * kFlatStride;
    for (uint k = 0; k < kFlatStride; ++k) {
        sorted[to + k] = records[from + k];
    }
}
