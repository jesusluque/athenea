// Copyright (c) 2026 jesus luque.
//
// THE SAME SORT, A BLOCK AT A TIME.
//
// radix_common.slang's passes walk a chunk an invocation, in order: portable,
// stable, no group memory -- and slow where it matters, since a chunk's keys
// are read by one thread one after another and every histogram lives in
// device memory. A frame of six million splats spent a third of its time
// sorting. This is the classic reduce-then-scan form of the same LSD pass:
//
//   1. radixBlockHistogram -- a group of 256 counts a tile of 1024 keys in
//                             group memory and writes its 256 counts
//                             digit-major (digit * blocks + block)
//   2. an exclusive prefix sum over those counts (gpu::PrefixSum), which is
//      where every tile's share of every digit starts
//   3. radixBlockScatter   -- the group sorts its tile by the digit, stably,
//                             in group memory (eight one-bit splits), and
//                             writes each pair to its start plus its rank
//
// Stable: within a tile the splits keep order, and tiles take their shares in
// tile order, as chunks did. Only group memory and barriers -- no atomics on
// device memory, no wave operations -- so it asks of a backend what any
// compute kernel with a tile of shared memory does.
module radix_block;
import radix_common;

static const uint kBlockThreads = 256;
static const uint kPerThread = 4;
static const uint kTile = kBlockThreads * kPerThread;

public struct BlockParams {
    public uint count;    // pairs
    public uint blocks;   // tiles, ceil(count / 1024)
    public uint shift;    // this pass's digit
    public uint wide;     // 1: keys are 64 bits
};

StructuredBuffer<uint>      keysLo;
StructuredBuffer<uint>      keysHi;
RWStructuredBuffer<uint>    blockHistogram;   // 256 * blocks, digit-major
StructuredBuffer<uint>      blockOffsets;     // its exclusive prefix sum
StructuredBuffer<uint>      srcValues;
RWStructuredBuffer<uint>    dstKeysLo;
RWStructuredBuffer<uint>    dstKeysHi;
RWStructuredBuffer<uint>    dstValues;
ConstantBuffer<BlockParams> blockParams;

groupshared uint sCount[kDigits];
groupshared uint sStart[kDigits];
groupshared uint sScan[kBlockThreads];
groupshared uint sLo[kTile];
groupshared uint sHi[kTile];
groupshared uint sVal[kTile];
groupshared uint tLo[kTile];
groupshared uint tHi[kTile];
groupshared uint tVal[kTile];

/// This thread's exclusive prefix of `value` over the group, and the group's
/// total: Hillis-Steele over 256 entries, eight steps.
uint groupExclusive(uint value, uint lane, out uint total) {
    sScan[lane] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < kBlockThreads; offset <<= 1) {
        const uint add = lane >= offset ? sScan[lane - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sScan[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    total = sScan[kBlockThreads - 1];
    const uint inclusive = sScan[lane];
    GroupMemoryBarrierWithGroupSync();
    return inclusive - value;
}

uint digitHere(uint lo, uint hi) {
    return digitOf(lo, hi, blockParams.shift, blockParams.wide);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void radixBlockHistogram(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID) {
    const uint lane = thread.x;
    const uint block = group.x;
    sCount[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    const uint base = block * kTile;
    for (uint k = 0; k < kPerThread; ++k) {
        const uint i = base + k * kBlockThreads + lane;   // neighbouring threads, neighbouring keys
        if (i < blockParams.count) {
            const uint hi = blockParams.wide != 0 ? keysHi[i] : 0u;
            InterlockedAdd(sCount[digitHere(keysLo[i], hi)], 1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    blockHistogram[lane * blockParams.blocks + block] = sCount[lane];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void radixBlockScatter(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID) {
    const uint lane = thread.x;
    const uint block = group.x;
    const uint base = block * kTile;
    const uint valid = min(kTile, blockParams.count - base);

    // The tile, in its order. What lies past the end takes the last digit,
    // so the splits leave it after every real pair and nothing writes it.
    sCount[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 0; k < kPerThread; ++k) {
        const uint local = k * kBlockThreads + lane;
        if (local < valid) {
            const uint lo = keysLo[base + local];
            const uint hi = blockParams.wide != 0 ? keysHi[base + local] : 0u;
            sLo[local] = lo;
            sHi[local] = hi;
            sVal[local] = srcValues[base + local];
            InterlockedAdd(sCount[digitHere(lo, hi)], 1u);
        } else {
            sLo[local] = 0xFFFFFFFFu;
            sHi[local] = 0xFFFFFFFFu;
            sVal[local] = 0u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Where each digit starts in the sorted tile.
    uint unused;
    sStart[lane] = groupExclusive(sCount[lane], lane, unused);

    // Sorted by the digit, a bit at a time: each split puts the zeros first
    // and keeps order within each side. This thread owns four neighbours.
    for (uint bit = 0; bit < 8; ++bit) {
        uint zeros = 0;
        for (uint k = 0; k < kPerThread; ++k) {
            const uint at = lane * kPerThread + k;
            zeros += ((digitHere(sLo[at], sHi[at]) >> bit) & 1u) == 0u ? 1u : 0u;
        }
        uint totalZeros;
        const uint before = groupExclusive(zeros, lane, totalZeros);
        uint zerosSoFar = before;
        for (uint k = 0; k < kPerThread; ++k) {
            const uint at = lane * kPerThread + k;
            const bool zero = ((digitHere(sLo[at], sHi[at]) >> bit) & 1u) == 0u;
            const uint to = zero ? zerosSoFar : totalZeros + (at - zerosSoFar);
            zerosSoFar += zero ? 1u : 0u;
            tLo[to] = sLo[at];
            tHi[to] = sHi[at];
            tVal[to] = sVal[at];
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint k = 0; k < kPerThread; ++k) {
            const uint at = lane * kPerThread + k;
            sLo[at] = tLo[at];
            sHi[at] = tHi[at];
            sVal[at] = tVal[at];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Each pair to where its tile's share of its digit starts, plus its rank
    // within that share.
    for (uint k = 0; k < kPerThread; ++k) {
        const uint at = k * kBlockThreads + lane;
        if (at < valid) {
            const uint lo = sLo[at];
            const uint hi = sHi[at];
            const uint d = digitHere(lo, hi);
            const uint to = blockOffsets[d * blockParams.blocks + block] + (at - sStart[d]);
            dstKeysLo[to] = lo;
            if (blockParams.wide != 0) {
                dstKeysHi[to] = hi;
            }
            dstValues[to] = sVal[at];
        }
    }
}
