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
//   videoPlanesHigh          the position high parts as the side payload:
//                            chunk by chunk, axis by axis, `chunkSplats`
//                            entries each (bytes, or 16-bit words at 10 bits)
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
RWStructuredBuffer<uint>         high;       // the side payload
ConstantBuffer<VideoPlanesParams> params;

static const uint kRun = 1024;

float valueOf(uint s, uint a) {
    return attributeOf(positions, shape, sh, params.sourceShWords, params.sourceRest, s, a);
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

/// The sample plane `plane` holds for Morton rank `i`.
uint sampleOf(uint plane, uint i) {
    i = min(i, params.count - 1);   // past the last splat: the last again, which costs the codec nothing
    const uint s = order[params.first + i];
    if (plane < 3 || (params.hiInVideo != 0 && plane < 6)) {
        const uint axis = plane % 3;
        const uint q = quantise(positions[s][axis], ranges[axis * 2], ranges[axis * 2 + 1], positionLevels(params));
        const uint2 parts = splitPosition(params, q);
        const bool isHigh = params.hiInVideo != 0 && plane < 3;
        return isHigh ? parts.x : parts.y;
    }
    const uint a = plane - (params.hiInVideo != 0 ? 6u : 3u) + kFirstAttributePlane;
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
    const uint entries = chunks * 3u * params.chunkSplats;
    const uint e0 = tid.x * perWord;
    if (e0 >= entries) {
        return;
    }
    uint word = 0;
    for (uint k = 0; k < perWord; ++k) {
        const uint e = e0 + k;
        const uint chunk = e / (3u * params.chunkSplats);
        const uint axis = (e / params.chunkSplats) % 3u;
        const uint i = chunk * params.chunkSplats + e % params.chunkSplats;
        uint v = 0;
        if (i < params.count) {
            const uint s = order[params.first + i];
            const uint q = quantise(positions[s][axis], ranges[axis * 2], ranges[axis * 2 + 1],
                                    positionLevels(params));
            v = q >> params.bits;
        }
        word |= v << (k * (32u / perWord));
    }
    high[tid.x] = word;
}
