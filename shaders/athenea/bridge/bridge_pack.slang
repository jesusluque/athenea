// Copyright (c) 2026 jesus luque.
//
// A FRAME'S AOV INTO A SLOT ANOTHER PROCESS READS (bridge::FramePublisher):
// the engine's buffer as it lies -- `stride` words a pixel, the AOV from word
// `offset` of each, `components` of them -- written into the slot in the
// format the subscriber asked for, bottom row first as the engine renders.
// A thread an output word:
//   format 0  float32 (or a 32-bit id), the words as they are
//   format 1  float16, two halves a word (the last word of an odd count
//             carries a zero half)
module bridge_pack;

struct PackParams {
    uint pixels;
    uint stride;       ///< source words a pixel
    uint offset;       ///< the AOV's first word in a pixel
    uint components;   ///< 1 or 4
    uint format;       ///< 0 words, 1 halves
    uint dstWord;      ///< first word written in `dst`
    uint words;        ///< words written
    uint pad;
};

StructuredBuffer<uint>       src;
RWStructuredBuffer<uint>     dst;
ConstantBuffer<PackParams>   params;

uint sourceWord(uint value) {
    const uint pixel = value / params.components;
    const uint c = value - pixel * params.components;
    return src[pixel * params.stride + params.offset + c];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void bridgePack(uint3 id: SV_DispatchThreadID) {
    const uint w = id.x;
    if (w >= params.words) {
        return;
    }
    const uint values = params.pixels * params.components;
    if (params.format == 0) {
        dst[params.dstWord + w] = sourceWord(w);
        return;
    }
    const uint first = 2 * w;
    const uint lo = f32tof16(asfloat(sourceWord(first)));
    const uint hi = first + 1 < values ? f32tof16(asfloat(sourceWord(first + 1))) : 0;
    dst[params.dstWord + w] = lo | (hi << 16);
}
