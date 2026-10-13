// Copyright (c) 2026 jesus luque.
//
// A CLOUD PRUNED BY WHAT THE FRAMES SAY (`athenea decimate --drop-hidden`,
// `--prune-error`). The tile rasteriser drew the store (its splats in Morton
// order) from an orbit of cameras and counted, at each splat's slot, what it
// gave each frame (`RenderSettings::contribution`): the weight it blended
// with, or the squared error taking it away would leave. Each frame's count
// is folded into a float score here; a histogram of the scores lets the host
// pick the threshold that keeps the wanted number; and the store is
// compacted in its own order -- positions, shapes, harmonics, normals and
// the order that names each one's splat of the cloud, through which
// everything else it carries is read.
module lod_prune;

static const uint kBins = 256;
static const float kLogLo = -12.0;              // log10 of the smallest score binned (bin 0 also holds 0)
static const float kLogStep = 18.0 / 256.0;

struct PruneParams {
    uint  count;       // store splats
    float threshold;   // a score under this is dropped
    uint  shWords;
    uint  normals;     // 1: the store carries shading normals
    float scale;       // a frame's count to the score's unit
    uint  wide;        // 1: two words a splat (low, high), the high one 2^32 of the low
};

RWStructuredBuffer<uint>   frameCount;     // per store splat (two words if `wide`): one frame's count, cleared as it is folded
RWStructuredBuffer<float>  score;          // per store splat: summed over the frames
RWStructuredBuffer<uint>   histogram;      // kBins
RWStructuredBuffer<uint>   keep;           // per store splat: 1 kept, 0 dropped
StructuredBuffer<uint>     offsets;        // the exclusive prefix sum of `keep`
StructuredBuffer<uint>     storeOrder;
StructuredBuffer<float4>   inPositions;
StructuredBuffer<uint>     inShape;
StructuredBuffer<uint>     inSh;
StructuredBuffer<uint>     inNormals;
RWStructuredBuffer<float4> outPositions;
RWStructuredBuffer<uint>   outShape;
RWStructuredBuffer<uint>   outSh;
RWStructuredBuffer<uint>   outNormals;
RWStructuredBuffer<uint>   outOrder;
ConstantBuffer<PruneParams> params;

uint binOf(float v) {
    if (!(v > 0.0)) {
        return 0u;
    }
    return uint(clamp((log10(v) - kLogLo) / kLogStep, 0.0, float(kBins - 1u)));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pruneFold(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    if (params.wide != 0u) {
        score[s] += (float(frameCount[s * 2]) + float(frameCount[s * 2 + 1]) * 4294967296.0) * params.scale;
        frameCount[s * 2] = 0u;
        frameCount[s * 2 + 1] = 0u;
    } else {
        score[s] += float(frameCount[s]) * params.scale;
        frameCount[s] = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pruneHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    InterlockedAdd(histogram[binOf(score[s])], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pruneKeep(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    keep[s] = score[s] >= params.threshold ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void pruneScatter(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count || keep[s] == 0u) {
        return;
    }
    const uint d = offsets[s];
    outPositions[d] = inPositions[s];
    for (uint k = 0; k < 4; ++k) {
        outShape[d * 4 + k] = inShape[s * 4 + k];
    }
    for (uint k = 0; k < params.shWords; ++k) {
        outSh[d * params.shWords + k] = inSh[s * params.shWords + k];
    }
    if (params.normals != 0) {
        outNormals[d] = inNormals[s];
    }
    outOrder[d] = storeOrder[s];
}
