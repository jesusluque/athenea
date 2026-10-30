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
// The decoded picture's second plane, (Cb, Cr) a 2 x 2 block (RG8Unorm):
// with `chroma`, a colour Y tile's Co and Cg. Bound to anything when not.
Texture2D<float2>                 chromaPlane;
StructuredBuffer<uint>            page;
RWStructuredBuffer<float4>        outPositions;
RWStructuredBuffer<uint>          outShape;
RWStructuredBuffer<uint>          outSh;
ConstantBuffer<VideoPlanesParams> params;
// The geometry layer's state (`geometry` 1, video_geometry.slang's
// videoGeometryApply): four words a frame splat. Bound to `page` when unused.
StructuredBuffer<uint>            geometryState;

uint sampleAt(uint plane, uint i) {
    const uint2 xy = pixelOf(params, plane, i);
    const float f = luma.Load(int3(int2(xy), 0));
    return uint(round(f * float(params.sampleScale))) >> params.sampleShift;
}

float rangeOf(uint a, uint end) {
    return asfloat(page[(params.rangesOffset >> 2) + a * 2 + end]);
}

/// Where chunk `chunk` (of the page's own) starts in the side payload, in words.
uint chunkWord(uint chunk) {
    return (params.hiOffset >> 2) + chunk * (params.highChunkBytes / 4u);
}

/// The high part of axis `axis` of picture splat `i`, from the side payload:
/// chunk by chunk from `chunk` (the page's own), after the chunk's boxes,
/// axis by axis, `chunkSplats` entries.
uint highAt(uint chunk, uint axis, uint i) {
    if (params.positionHighBits == 0) {
        return 0u;
    }
    const uint e = axis * params.chunkSplats + i % params.chunkSplats;
    const uint base = chunkWord(chunk) + highsAt(params) / 4u;
    if (params.bits > 8) {
        const uint word = page[base + e / 2u];
        return (word >> ((e & 1u) * 16u)) & 0xFFFFu;
    }
    const uint word = page[base + e / 4u];
    return (word >> ((e & 3u) * 8u)) & 0xFFu;
}

/// The range axis `axis` of picture splat `i` was quantised over.
float2 positionRangeAt(uint chunk, uint axis, uint i) {
    if (params.positionBlock == 0) {
        return float2(rangeOf(axis, 0), rangeOf(axis, 1));
    }
    const uint word = page[chunkWord(chunk) + blockWordOf(params, i, axis)];
    return float2(lowHalf(word), highHalf(word));
}

/// The high bits picture splat `i`'s block carries (adaptive precision), or
/// everyone's.
uint highBitsAt(uint chunk, uint i) {
    if (!adaptive(params)) {
        return params.positionHighBits;
    }
    const uint b = (i % params.chunkSplats) / params.positionBlock;
    const uint word = page[chunkWord(chunk) + (blockBytes(params) >> 2) + b / 4u];
    return (word >> ((b & 3u) * 8u)) & 0xFFu;
}

uint highOf(uint axis, uint i) {
    return highAt(i / params.chunkSplats - params.first / params.chunkSplats, axis, i);
}

float attributeAt(uint a, uint i) {
    return dequantise(sampleAt(attributePlane(params, a), i), rangeOf(a, 0), rangeOf(a, 1), sampleLevels(params));
}

/// Colour quantity q (0 the base colour, 1 + b harmonic basis b) of splat
/// `i` as RGB: three luma planes, or (`chroma`) its Y plane and that tile's
/// Co and Cg at the splat's 2 x 2 block.
float3 colourAt(uint q, uint i) {
    const uint a = kFirstColour + 3u * q;
    if (params.chroma == 0) {
        return float3(attributeAt(a, i), attributeAt(a + 1u, i), attributeAt(a + 2u, i));
    }
    const uint2 xy = pixelOf(params, attributePlane(params, a), i);
    const float2 c = chromaPlane.Load(int3(int2(xy / 2u), 0));
    const uint cb = uint(round(c.x * 255.0)), cr = uint(round(c.y * 255.0));
    const float3 ycocg = float3(attributeAt(a, i),
                                dequantise(cb, rangeOf(a + 1u, 0), rangeOf(a + 1u, 1), sampleLevels(params)),
                                dequantise(cr, rangeOf(a + 2u, 0), rangeOf(a + 2u, 1), sampleLevels(params)));
    return rgbOfYcocg(ycocg);
}

void writeRest(uint i, uint j) {
    if (params.keep == 0) {
        return;
    }
    const uint shRow = j * params.shWords;
    uint word = 0;
    uint filled = 0;
    uint w = 0;
    float3 basis = float3(0.0, 0.0, 0.0);
    for (uint k = 0; k < params.keep * 3u; ++k) {
        if (k % 3u == 0) {
            basis = k / 3u < params.restPerColour ? colourAt(1u + k / 3u, i) : float3(0.0, 0.0, 0.0);
            if (params.occlusion != 0 && k == 6) {
                // A relightable stream's occlusion, where its source had it:
                // band 1's x basis, grey.
                basis = float3(attributeAt(params.occlusion, i));
            }
        }
        const float v = basis[k % 3u];
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

[shader("compute")]
[numthreads(256, 1, 1)]
void videoPlanesDecode(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.pageCount) {
        return;
    }
    const uint i = params.first + tid.x;
    const uint j = params.at + tid.x;
    if (params.geometry != 0) {
        // Position, scale and rotation from the exact layer: their bits over
        // the sequence's ranges, the rotation the store's own word.
        const uint g = (params.geometryAt + i) * 4u;
        const uint4 w = uint4(geometryState[g], geometryState[g + 1], geometryState[g + 2], geometryState[g + 3]);
        float3 gp, gl;
        for (uint axis = 0; axis < 3; ++axis) {
            gp[axis] = dequantise(geometryField(w, axis), rangeOf(axis, 0), rangeOf(axis, 1),
                                  (1u << params.geometryPositionBits) - 1u);
            gl[axis] = dequantise(geometryField(w, 3u + axis), rangeOf(4u + axis, 0), rangeOf(4u + axis, 1),
                                  (1u << params.geometryScaleBits) - 1u);
        }
        outPositions[j] = float4(gp, clamp(attributeAt(3, i), 0.0, 5.0));
        const float3 gbase = colourAt(0, i);
        outShape[j * 4 + 0] = w.w;
        outShape[j * 4 + 1] = packHalves(halfSafe(gl.x), halfSafe(gl.y));
        outShape[j * 4 + 2] = packHalves(halfSafe(gl.z), halfSafe(gbase.x));
        outShape[j * 4 + 3] = packHalves(halfSafe(gbase.y), halfSafe(gbase.z));
        writeRest(i, j);
        return;
    }
    float3 p;
    for (uint axis = 0; axis < 3; ++axis) {
        const uint lo = sampleAt(lowPlane(params, axis), i);
        const uint hi = params.hiInVideo != 0 ? sampleAt(axis, i) : highOf(axis, i);
        const uint chunk = i / params.chunkSplats - params.first / params.chunkSplats;
        const float2 r = positionRangeAt(chunk, axis, i);
        p[axis] = dequantise(joinPosition(params, hi, lo), r.x, r.y, levelsWith(params, highBitsAt(chunk, i)));
    }
    const float opacity = clamp(attributeAt(3, i), 0.0, 5.0);
    outPositions[j] = float4(p, opacity);

    const float3 ls = float3(attributeAt(4, i), attributeAt(5, i), attributeAt(6, i));
    float4 q = float4(attributeAt(7, i), attributeAt(8, i), attributeAt(9, i), attributeAt(10, i));
    const float qLength = length(q);
    q = qLength > 1e-8 ? q / qLength : float4(0, 0, 0, 1);
    const float3 base = colourAt(0, i);
    const uint row = j * 4;
    outShape[row + 0] = encodeQuaternion(q);
    outShape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    outShape[row + 2] = packHalves(halfSafe(ls.z), halfSafe(base.x));
    outShape[row + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));

    writeRest(i, j);
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
    const float levels = float(a < 3 ? referenceLevels(params) : sampleLevels(params));
    const float step = hi > lo ? (hi - lo) / levels : 1.0;
    uint largest = 0, sum = 0, far = 0;
    const uint end = min(params.count, (run + 1) * kCompareRun);
    for (uint i = run * kCompareRun; i < end; ++i) {
        const float d = attributeOf(decodedPositions, decodedShape, decodedSh, params.shWords,
                                    params.occlusion != 0 ? 1u : params.restPerColour, i, a, params.chroma,
                                    params.occlusion);
        const float s = attributeOf(sourcePositions, sourceShape, sourceSh, params.sourceShWords, params.sourceRest,
                                    order[params.first + i], a, params.chroma, params.occlusion);
        const float steps = abs(d - s) / step;
        const uint sixteenths = uint(min(steps * 16.0, 4.0e9));
        largest = max(largest, sixteenths);
        sum += min(sixteenths, 1u << 20);
        if (a < 3 && params.geometry == 0) {
            // Both quantised over the range the position was sent over (the
            // block's box, read from the side payload as the decode read
            // it): the decoded high part against the source's.
            const float2 r = params.hiInVideo == 0 ? positionRangeAt(i / params.chunkSplats, a, i) : float2(lo, hi);
            const uint levels = params.hiInVideo == 0 ? levelsWith(params, highBitsAt(i / params.chunkSplats, i))
                                                      : positionLevels(params);
            const uint qd = quantise(d, r.x, r.y, levels);
            const uint qs = quantise(s, r.x, r.y, levels);
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
