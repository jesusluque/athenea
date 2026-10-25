// Copyright (c) 2026 jesus luque.
//
// Video planes back into packed splats (video_planes_common.slang): the luma
// plane of a frame the hardware decoder wrote, read where it lies (a texture
// over the decoder's own buffer), and a page of the frame's side data -- its
// ranges, and the position high parts when they are not planes -- into the
// store's layout (common/packing.slang). One thread a splat of the page.
//
// `page` is the staging a live pool copied the page into, or any buffer that
// holds the same: `rangesOffset` and `hiOffset` are bytes into it.
//
//   videoPlanesDecode   splats first .. first + pageCount of the frame into
//                       store splats at .. at + pageCount
//   videoPlanesCompare  a decoded cloud against its source, attribute by
//                       attribute, in steps of the attribute's quantisation:
//                       the largest, the sum (in 1/16 steps) and, for the
//                       position, how many splats' high parts differ
module video_planes_decode;
import athenea.common.packing;
import video_planes_common;

Texture2D<float>                  luma;
StructuredBuffer<uint>            page;
RWStructuredBuffer<float4>        outPositions;
RWStructuredBuffer<uint>          outShape;
RWStructuredBuffer<uint>          outSh;
ConstantBuffer<VideoPlanesParams> params;

uint sampleAt(uint plane, uint i) {
    const uint2 xy = pixelOf(params, plane, i);
    const float f = luma.Load(int3(int2(xy), 0));
    return uint(round(f * float(params.sampleScale))) >> params.sampleShift;
}

float rangeOf(uint a, uint end) {
    return asfloat(page[(params.rangesOffset >> 2) + a * 2 + end]);
}

/// The high part of axis `axis` of frame splat `i`, from the side payload:
/// chunk by chunk from the page's own, axis by axis, `chunkSplats` entries.
uint highOf(uint axis, uint i) {
    const uint chunk = i / params.chunkSplats - params.first / params.chunkSplats;
    const uint e = (chunk * 3u + axis) * params.chunkSplats + i % params.chunkSplats;
    if (params.bits > 8) {
        const uint word = page[(params.hiOffset >> 2) + e / 2u];
        return (word >> ((e & 1u) * 16u)) & 0xFFFFu;
    }
    const uint word = page[(params.hiOffset >> 2) + e / 4u];
    return (word >> ((e & 3u) * 8u)) & 0xFFu;
}

float attributeAt(uint a, uint i) {
    return dequantise(sampleAt(attributePlane(params, a), i), rangeOf(a, 0), rangeOf(a, 1), sampleLevels(params));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesDecode(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.pageCount) {
        return;
    }
    const uint i = params.first + tid.x;
    const uint j = params.at + tid.x;
    float3 p;
    for (uint axis = 0; axis < 3; ++axis) {
        const uint lo = sampleAt(lowPlane(params, axis), i);
        const uint hi = params.hiInVideo != 0 ? sampleAt(axis, i) : highOf(axis, i);
        p[axis] = dequantise(joinPosition(params, hi, lo), rangeOf(axis, 0), rangeOf(axis, 1), positionLevels(params));
    }
    const float opacity = clamp(attributeAt(3, i), 0.0, 5.0);
    outPositions[j] = float4(p, opacity);

    const float3 ls = float3(attributeAt(4, i), attributeAt(5, i), attributeAt(6, i));
    float4 q = float4(attributeAt(7, i), attributeAt(8, i), attributeAt(9, i), attributeAt(10, i));
    const float qLength = length(q);
    q = qLength > 1e-8 ? q / qLength : float4(0, 0, 0, 1);
    const float3 base = float3(attributeAt(11, i), attributeAt(12, i), attributeAt(13, i));
    const uint row = j * 4;
    outShape[row + 0] = encodeQuaternion(q);
    outShape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    outShape[row + 2] = packHalves(halfSafe(ls.z), halfSafe(base.x));
    outShape[row + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));

    if (params.keep == 0) {
        return;
    }
    const uint shRow = j * params.shWords;
    uint word = 0;
    uint filled = 0;
    uint w = 0;
    for (uint k = 0; k < params.keep * 3u; ++k) {
        const float v = k / 3u < params.restPerColour ? attributeAt(14 + k, i) : 0.0;
        word |= (f32tof16(halfSafe(v)) & 0xFFFFu) << (filled * 16u);
        filled += 1;
        if (filled == 2) {
            outSh[shRow + w] = word;
            w += 1;
            word = 0;
            filled = 0;
        }
    }
    if (filled == 1) {
        outSh[shRow + w] = word;
    }
}

// --- the round trip, measured -----------------------------------------------

StructuredBuffer<float4>  decodedPositions;
StructuredBuffer<uint>    decodedShape;
StructuredBuffer<uint>    decodedSh;
StructuredBuffer<float4>  sourcePositions;
StructuredBuffer<uint>    sourceShape;
StructuredBuffer<uint>    sourceSh;
StructuredBuffer<uint>    order;          // Morton rank -> source splat
StructuredBuffer<float>   frameRanges;    // attributes x 2
RWStructuredBuffer<uint>  errors;         // attributes x 4: max (1/16 steps), sum, its carries, high parts that differ

static const uint kCompareRun = 64;

[shader("compute")]
[numthreads(64, 1, 1)]
void videoPlanesCompare(uint3 tid: SV_DispatchThreadID) {
    const uint runs = (params.count + kCompareRun - 1) / kCompareRun;
    const uint run = tid.x / params.attributes, a = tid.x % params.attributes;
    if (run >= runs) {
        return;
    }
    const float lo = frameRanges[a * 2], hi = frameRanges[a * 2 + 1];
    const float levels = float(a < 3 ? positionLevels(params) : sampleLevels(params));
    const float step = hi > lo ? (hi - lo) / levels : 1.0;
    uint largest = 0, sum = 0, far = 0;
    const uint end = min(params.count, (run + 1) * kCompareRun);
    for (uint i = run * kCompareRun; i < end; ++i) {
        const float d = attributeOf(decodedPositions, decodedShape, decodedSh, params.shWords, params.restPerColour, i, a);
        const float s = attributeOf(sourcePositions, sourceShape, sourceSh, params.sourceShWords, params.sourceRest,
                                    order[params.first + i], a);
        const float steps = abs(d - s) / step;
        const uint sixteenths = uint(min(steps * 16.0, 4.0e9));
        largest = max(largest, sixteenths);
        sum += min(sixteenths, 1u << 20);
        if (a < 3) {
            const uint qd = quantise(d, lo, hi, positionLevels(params));
            const uint qs = quantise(s, lo, hi, positionLevels(params));
            far += (qd >> params.bits) != (qs >> params.bits) ? 1u : 0u;
        }
    }
    InterlockedMax(errors[a * 4], largest);
    uint before;
    InterlockedAdd(errors[a * 4 + 1], sum, before);
    if (before + sum < before) {
        InterlockedAdd(errors[a * 4 + 2], 1u);
    }
    InterlockedAdd(errors[a * 4 + 3], far);
}
