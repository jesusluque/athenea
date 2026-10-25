// Copyright (c) 2026 jesus luque.
//
// A frame's packed cloud into video planes (video_planes_common.slang):
//   videoPlanesRangePartial  each attribute's min and max over a run of 1024
//                            splats (one thread a run and attribute)
//   videoPlanesRangeFinish   ... over the runs: the frame's ranges, 2 floats
//                            an attribute
//   videoPlanesRangeWiden    a group of pictures' ranges: the frame's, or
//                            their union with what the group had (`first` 0)
//   videoPlanesMorton        the splats' 30-bit Morton keys over the
//                            position ranges, for the radix sort
//   videoPlanesEncode        the mosaic, one word (four 8-bit samples or two
//                            16-bit P010 ones) a thread, the splats read
//                            through the sorted order
//   videoPlanesBlocks        with `positionBlock`: each block's box, a
//                            thread a block, at the head of its chunk's side
//                            payload (before the mosaic and the high parts,
//                            which are taken over it)
//   videoPlanesChroma        with `chroma`: the second plane, (Cb, Cr) a 2 x 2
//                            block -- a colour Y tile's Co and Cg, the four
//                            splats' mean (or weighed by opacity x area);
//                            mid grey under every other tile
//   videoPlanesHigh          the position high parts as the side payload:
//                            chunk by chunk, after the chunk's boxes, axis by
//                            axis, `chunkSplats` entries each (bytes, or
//                            16-bit words at 10 bits)
module video_planes_encode;
import athenea.common.packing;
import athenea.common.morton;
import video_planes_common;

StructuredBuffer<float4>         positions;
StructuredBuffer<uint>           shape;
StructuredBuffer<uint>           sh;
StructuredBuffer<uint>           order;      // Morton rank -> source splat; a slice's from rank `first`
RWStructuredBuffer<float>        partial;    // runs x attributes x 2
RWStructuredBuffer<float>        ranges;     // attributes x 2
StructuredBuffer<float>          frameRanges;
RWStructuredBuffer<uint>         keys;
RWStructuredBuffer<uint>         values;
RWStructuredBuffer<uint>         mosaic;     // width x height samples
RWStructuredBuffer<uint>         chromaMosaic;   // width / 2 x height / 2 (Cb, Cr) byte pairs: NV12's second plane
RWStructuredBuffer<uint>         high;       // the side payload, `highChunkBytes` a chunk
ConstantBuffer<VideoPlanesParams> params;

static const uint kRun = 1024;

float valueOf(uint s, uint a) {
    return attributeOf(positions, shape, sh, params.sourceShWords, params.sourceRest, s, a, params.chroma);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void videoPlanesRangePartial(uint3 tid: SV_DispatchThreadID) {
    const uint runs = (params.count + kRun - 1) / kRun;
    const uint run = tid.x / params.attributes, a = tid.x % params.attributes;
    if (run >= runs) {
        return;
    }
    float lo = 3.4e38, hi = -3.4e38;
    const uint end = min(params.count, (run + 1) * kRun);
    for (uint i = run * kRun; i < end; ++i) {
        const float v = valueOf(i, a);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    partial[(run * params.attributes + a) * 2] = lo;
    partial[(run * params.attributes + a) * 2 + 1] = hi;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void videoPlanesRangeFinish(uint3 tid: SV_DispatchThreadID) {
    const uint a = tid.x;
    if (a >= params.attributes) {
        return;
    }
    const uint runs = (params.count + kRun - 1) / kRun;
    float lo = 3.4e38, hi = -3.4e38;
    for (uint r = 0; r < runs; ++r) {
        lo = min(lo, partial[(r * params.attributes + a) * 2]);
        hi = max(hi, partial[(r * params.attributes + a) * 2 + 1]);
    }
    if (!(hi >= lo)) {   // no splats
        lo = 0.0;
        hi = 0.0;
    }
    ranges[a * 2] = lo;
    ranges[a * 2 + 1] = hi;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void videoPlanesRangeWiden(uint3 tid: SV_DispatchThreadID) {
    const uint a = tid.x;
    if (a >= params.attributes) {
        return;
    }
    const float lo = frameRanges[a * 2], hi = frameRanges[a * 2 + 1];
    if (params.first != 0) {
        ranges[a * 2] = lo;
        ranges[a * 2 + 1] = hi;
    } else {
        ranges[a * 2] = min(ranges[a * 2], lo);
        ranges[a * 2 + 1] = max(ranges[a * 2 + 1], hi);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesMorton(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 lo = float3(ranges[0], ranges[2], ranges[4]);
    const float3 hi = float3(ranges[1], ranges[3], ranges[5]);
    keys[i] = morton30((positions[i].xyz - lo) / max(hi - lo, float3(1e-20)));
    values[i] = i;
}

/// The range position axis `axis` of picture splat `i` is quantised over:
/// the frame's, or its block's box (videoPlanesBlocks wrote it).
float2 positionRange(uint i, uint axis) {
    if (params.positionBlock == 0) {
        return float2(ranges[axis * 2], ranges[axis * 2 + 1]);
    }
    const uint word = high[(i / params.chunkSplats) * (params.highChunkBytes / 4u) + blockWordOf(params, i, axis)];
    return float2(lowHalf(word), highHalf(word));
}

uint positionOf(uint s, uint i, uint axis) {
    const float2 r = positionRange(i, axis);
    return quantise(positions[s][axis], r.x, r.y, positionLevels(params));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void videoPlanesBlocks(uint3 tid: SV_DispatchThreadID) {
    const uint perChunk = params.chunkSplats / params.positionBlock;
    const uint chunks = (params.count + params.chunkSplats - 1) / params.chunkSplats;
    const uint chunk = tid.x / perChunk, b = tid.x % perChunk;
    if (chunk >= chunks) {
        return;
    }
    const uint i0 = chunk * params.chunkSplats + b * params.positionBlock;
    const uint i1 = min(i0 + params.positionBlock, params.count);
    float3 lo = float3(3.4e38), hi = float3(-3.4e38);
    for (uint i = i0; i < i1; ++i) {
        const float3 p = positions[order[params.first + i]].xyz;
        lo = min(lo, p);
        hi = max(hi, p);
    }
    if (i1 <= i0) {   // a block past the last splat
        lo = float3(0.0);
        hi = float3(0.0);
    }
    const uint base = chunk * (params.highChunkBytes / 4u) + b * 3u;
    for (uint axis = 0; axis < 3; ++axis) {
        high[base + axis] = halfBelow(lo[axis]) | (halfAbove(hi[axis]) << 16u);
    }
}

/// The sample plane `plane` holds for Morton rank `i`.
uint sampleOf(uint plane, uint i) {
    i = min(i, params.count - 1);   // past the last splat: the last again, which costs the codec nothing
    const uint s = order[params.first + i];
    if (plane < 3 || (params.hiInVideo != 0 && plane < 6)) {
        const uint axis = plane % 3;
        const uint q = positionOf(s, i, axis);
        const uint2 parts = splitPosition(params, q);
        const bool isHigh = params.hiInVideo != 0 && plane < 3;
        return isHigh ? parts.x : parts.y;
    }
    const uint a = planeAttribute(params, plane);
    return quantise(valueOf(s, a), ranges[a * 2], ranges[a * 2 + 1], sampleLevels(params));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesEncode(uint3 tid: SV_DispatchThreadID) {
    const uint perWord = params.bits > 8 ? 2u : 4u;
    const uint wordsPerRow = params.width / perWord;
    const uint y = tid.x / wordsPerRow;
    if (y >= params.height) {
        return;
    }
    const uint x0 = (tid.x % wordsPerRow) * perWord;
    uint word = 0;
    for (uint k = 0; k < perWord; ++k) {
        const uint2 at = planeAndSplatOf(params, uint2(x0 + k, y));
        uint v = 0;
        if (at.x < params.planes) {
            v = sampleOf(at.x, at.y);
        } else {
            v = 1u << (params.bits - 1);   // a spare tile: mid grey
        }
        word |= params.bits > 8 ? (v << params.sampleShift) << (k * 16u) : v << (k * 8u);
    }
    mosaic[tid.x] = word;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesHigh(uint3 tid: SV_DispatchThreadID) {
    const uint perWord = params.bits > 8 ? 2u : 4u;
    const uint chunks = (params.count + params.chunkSplats - 1) / params.chunkSplats;
    const uint perChunk = 3u * params.chunkSplats;
    const uint e0 = tid.x * perWord;
    if (e0 >= chunks * perChunk) {
        return;
    }
    // A word holds entries of one chunk (chunkSplats is a multiple of 4).
    const uint chunk = e0 / perChunk;
    uint word = 0;
    for (uint k = 0; k < perWord; ++k) {
        const uint e = e0 + k;
        const uint axis = (e % perChunk) / params.chunkSplats;
        const uint i = chunk * params.chunkSplats + e % params.chunkSplats;
        uint v = 0;
        if (i < params.count) {
            v = positionOf(order[params.first + i], i, axis) >> params.bits;
        }
        word |= v << (k * (32u / perWord));
    }
    const uint bytes = (e0 % perChunk) * (params.bits > 8 ? 2u : 1u);
    high[chunk * (params.highChunkBytes / 4u) + (blockBytes(params) + bytes) / 4u] = word;
}

/// What a splat weighs in its block's chroma: 1, or its opacity times the
/// area of its two largest axes (`chromaWeight` 1).
float chromaWeightOf(uint s) {
    if (params.chromaWeight == 0) {
        return 1.0;
    }
    const float3 ls = float3(valueOf(s, 4), valueOf(s, 5), valueOf(s, 6));
    const float smallest = min(ls.x, min(ls.y, ls.z));
    return max(valueOf(s, 3), 1.0e-4) * exp(ls.x + ls.y + ls.z - smallest);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesChroma(uint3 tid: SV_DispatchThreadID) {
    // Two chroma samples a word: Cb, Cr, Cb, Cr.
    const uint wordsPerRow = params.width / 4u;
    const uint cy = tid.x / wordsPerRow;
    if (cy >= params.height / 2u) {
        return;
    }
    uint word = 0;
    for (uint k = 0; k < 2; ++k) {
        const uint cx = (tid.x % wordsPerRow) * 2u + k;
        const uint2 at = planeAndSplatOf(params, uint2(cx * 2u, cy * 2u));   // the block's (0, 0): splat 4m
        const uint q = at.x < params.planes ? colourQuantityOf(params, at.x) : ~0u;
        uint cb = 128u, cr = 128u;
        if (q != ~0u && at.y < params.count) {
            const uint a = kFirstColour + 3u * q;
            float2 sum = float2(0.0, 0.0);
            float weight = 0.0;
            // The block's splats; those past the last are not splats.
            for (uint m = 0; m < 4; ++m) {
                const uint i = at.y + m;
                if (i >= params.count) {
                    break;
                }
                const uint s = order[params.first + i];
                const float w = chromaWeightOf(s);
                sum += w * float2(valueOf(s, a + 1u), valueOf(s, a + 2u));
                weight += w;
            }
            const float2 mean = sum / max(weight, 1.0e-30);
            cb = quantise(mean.x, ranges[(a + 1u) * 2], ranges[(a + 1u) * 2 + 1], sampleLevels(params));
            cr = quantise(mean.y, ranges[(a + 2u) * 2], ranges[(a + 2u) * 2 + 1], sampleLevels(params));
        }
        word |= (cb | (cr << 8u)) << (k * 16u);
    }
    chromaMosaic[tid.x] = word;
}
