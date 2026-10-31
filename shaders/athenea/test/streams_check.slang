// Copyright (c) 2026 jesus luque.
//
// The medium's random streams of one pixel, as the path tracer draws them:
// every (path, bounce, purpose) stream's states over its first draws,
// written out to be sorted on the device and counted where two are equal --
// two streams that pass through one state draw the same numbers from it on.
// The legacy construction (one LCG word seeded from a 24-bit float, which
// the medium used before) is written the same way, for the comparison.
import athenea.volume.medium;

struct StreamsParams {
    uint paths;
    uint bounces;
    uint steps;      // draws a stream
    uint legacy;     // 1: the old single-word LCG
    uint pixel;
    uint seed;
};

ConstantBuffer<StreamsParams> streams;
RWStructuredBuffer<uint>      keysLo;
RWStructuredBuffer<uint>      keysHi;
RWStructuredBuffer<uint>      counts;   // [equal neighbours after the sort]

static const uint kPurposes[4] = {21u, 23u, 25u, 27u};

uint pcgHash(uint input) {
    const uint state = input * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void streamStates(uint3 tid: SV_DispatchThreadID) {
    const uint n = streams.paths * streams.bounces * 4u;
    if (tid.x >= n) return;
    const uint purpose = kPurposes[tid.x % 4u];
    const uint bounce = (tid.x / 4u) % streams.bounces;
    const uint pathIndex = tid.x / (4u * streams.bounces);
    const uint base = tid.x * streams.steps;
    if (streams.legacy != 0u) {
        // random(pixel, sample, bounce, which) as a float, its bits the seed.
        uint key = pcgHash(streams.pixel);
        key = pcgHash(key + streams.seed);
        key = pcgHash(key + pathIndex);
        key = pcgHash(key + bounce);
        key = pcgHash(key + purpose);
        uint state = asuint(float(key >> 8) * (1.0 / 16777216.0)) ^ 0x5bd1e995u;
        for (uint k = 0; k < streams.steps; ++k) {
            state = state * 747796405u + 2891336453u;
            keysLo[base + k] = state;
            keysHi[base + k] = 0u;
        }
        return;
    }
    uint2 state = mediumStreamOf(streams.pixel, streams.seed, pathIndex, bounce, purpose);
    for (uint k = 0; k < streams.steps; ++k) {
        keysLo[base + k] = state.y;
        keysHi[base + k] = state.x;
        mediumRandom(state);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void streamEquals(uint3 tid: SV_DispatchThreadID) {
    const uint n = streams.paths * streams.bounces * 4u * streams.steps;
    if (tid.x == 0 || tid.x >= n) return;
    if (keysLo[tid.x] == keysLo[tid.x - 1] && keysHi[tid.x] == keysHi[tid.x - 1]) InterlockedAdd(counts[0], 1u);
}
