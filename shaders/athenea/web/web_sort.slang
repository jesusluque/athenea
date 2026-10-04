// Copyright (c) 2026 jesus luque.
//
// THE WEB'S SORT AND PREFIX SUM, SIZED ON THE GPU (proposal 082; 081's rule
// that no count crosses to the CPU inside a frame). The native chunked radix
// sort and prefix sum (algo/radix_*.slang, algo/prefix_*.slang), with two
// changes:
//
//   - the count is read from `counts` (word `countAt`, held to `capacity`),
//     where an earlier kernel of the same frame wrote it, not from a uniform
//     the CPU filled. The chunk size is fixed by the capacity, so the CPU can
//     dispatch one thread a chunk of the capacity without knowing the count:
//     a chunk past the count returns at once;
//   - the digit is kDigitBits wide, a WGSL override: 8 (256 digits, the
//     native sort's) or 4 (16 digits, twice the passes, a sixteenth of the
//     histograms -- 082's T1 variant).
//
// No group memory, no subgroup operation, no look-back: what every browser
// and every GPU runs (082 section 1). Stable, as the native sort is.
module web_sort;

[SpecializationConstant]
const uint kDigitBits = 8;

public struct WebCount {
    public uint countAt;     // the word of `counts` that holds the count
    public uint capacity;    // the most it may be: what the buffers hold
    public uint chunkSize;   // elements a chunk, fixed by the capacity
    public uint shift;       // the radix pass's first bit
};

StructuredBuffer<uint>    counts;
ConstantBuffer<WebCount>  sorting;

uint digits() {
    return 1u << kDigitBits;
}

uint countNow() {
    return min(counts[sorting.countAt], sorting.capacity);
}

uint chunksNow() {
    return (countNow() + sorting.chunkSize - 1) / sorting.chunkSize;
}

uint digitOf(uint key) {
    return (key >> sorting.shift) & (digits() - 1u);
}

StructuredBuffer<uint>    keys;
RWStructuredBuffer<uint>  histogram;     // chunks * digits
RWStructuredBuffer<uint>  digitTotals;   // digits
RWStructuredBuffer<uint>  chunkStarts;   // chunks * digits

[shader("compute")]
[numthreads(64, 1, 1)]
void webRadixHistogram(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= chunksNow()) {
        return;
    }
    const uint row = c * digits();
    for (uint d = 0; d < digits(); ++d) {
        histogram[row + d] = 0;
    }
    const uint begin = c * sorting.chunkSize;
    const uint end = min(begin + sorting.chunkSize, countNow());
    for (uint i = begin; i < end; ++i) {
        histogram[row + digitOf(keys[i])] += 1;
    }
}

StructuredBuffer<uint>    counted;   // the histogram, read

[shader("compute")]
[numthreads(64, 1, 1)]
void webRadixTotals(uint3 tid: SV_DispatchThreadID) {
    const uint d = tid.x;
    if (d >= digits()) {
        return;
    }
    const uint chunks = chunksNow();
    uint sum = 0;
    for (uint c = 0; c < chunks; ++c) {
        sum += counted[c * digits() + d];
    }
    digitTotals[d] = sum;
}

StructuredBuffer<uint>    totalsOfDigits;

[shader("compute")]
[numthreads(64, 1, 1)]
void webRadixStarts(uint3 tid: SV_DispatchThreadID) {
    const uint d = tid.x;
    if (d >= digits()) {
        return;
    }
    uint running = 0;
    for (uint below = 0; below < d; ++below) {
        running += totalsOfDigits[below];
    }
    const uint chunks = chunksNow();
    for (uint c = 0; c < chunks; ++c) {
        chunkStarts[c * digits() + d] = running;
        running += counted[c * digits() + d];
    }
}

StructuredBuffer<uint>    srcValues;
RWStructuredBuffer<uint>  dstKeys;
RWStructuredBuffer<uint>  dstValues;

[shader("compute")]
[numthreads(64, 1, 1)]
void webRadixScatter(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= chunksNow()) {
        return;
    }
    const uint row = c * digits();
    const uint begin = c * sorting.chunkSize;
    const uint end = min(begin + sorting.chunkSize, countNow());
    for (uint i = begin; i < end; ++i) {
        const uint key = keys[i];
        const uint slot = row + digitOf(key);
        const uint at = chunkStarts[slot];
        dstKeys[at] = key;
        dstValues[at] = srcValues[i];
        chunkStarts[slot] = at + 1;   // after the pair is placed (radix_scatter.slang says why)
    }
}

// The prefix sum, the same way: output[i] = input[0..i), `total` the sum.
StructuredBuffer<uint>    input;
RWStructuredBuffer<uint>  output;
RWStructuredBuffer<uint>  partTotals;   // a chunk's sum
RWStructuredBuffer<uint>  partStarts;   // a chunk's first output
RWStructuredBuffer<uint>  total;

[shader("compute")]
[numthreads(64, 1, 1)]
void webPrefixTotals(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= chunksNow()) {
        return;
    }
    const uint begin = c * sorting.chunkSize;
    const uint end = min(begin + sorting.chunkSize, countNow());
    uint sum = 0;
    for (uint i = begin; i < end; ++i) {
        sum += input[i];
    }
    partTotals[c] = sum;
}

StructuredBuffer<uint>    partSums;

[shader("compute")]
[numthreads(1, 1, 1)]
void webPrefixStarts(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint chunks = chunksNow();
    uint running = 0;
    for (uint c = 0; c < chunks; ++c) {
        partStarts[c] = running;
        running += partSums[c];
    }
    total[0] = running;
}

StructuredBuffer<uint>    startsOfParts;

[shader("compute")]
[numthreads(64, 1, 1)]
void webPrefixLocal(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= chunksNow()) {
        return;
    }
    const uint begin = c * sorting.chunkSize;
    const uint end = min(begin + sorting.chunkSize, countNow());
    uint running = startsOfParts[c];
    for (uint i = begin; i < end; ++i) {
        const uint value = input[i];
        output[i] = running;
        running += value;
    }
}
