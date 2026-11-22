// Copyright (c) 2026 jesus luque.
//
// What a thread's memory bears on the device at hand: an array a kernel
// keeps, written whole from a buffer, read while it stands and compared with
// the buffer again at the end. Past what the device gives a thread Metal
// builds the pipeline and says nothing, and threads read back what others
// wrote (docs/decisions.md, 18 Nov, "What a thread's memory bears").
module thread_memory;

struct MemoryParams {
    uint threads;   // a side of the square dispatched: threads * threads run
    uint words;     // the buffer's length
};

RWStructuredBuffer<uint>     counts;   // [threads that ran, threads whose arrays changed, words that changed]
StructuredBuffer<uint>       salts;    // what the arrays are filled with: nothing the compiler knows
ConstantBuffer<MemoryParams> memory;

/// An array of N words filled, read where only the buffer says and compared
/// with the buffer again: how many of its words came back changed.
uint changedOf<let N : int>(uint id, uint salt) {
    uint held[N];
    for (uint i = 0; i < N; ++i) {
        held[i] = salts[(id * 7u + salt + i) % memory.words] ^ id;
    }
    // Read where only the buffer says, so the array is memory a thread
    // addresses and not values the compiler follows.
    uint wandered = 0u;
    uint at = id % N;
    for (uint r = 0; r < 64u; ++r) {
        wandered ^= held[at];
        at = (at + salts[(id + r) % memory.words]) % N;
    }
    uint changed = 0u;
    for (uint i = 0; i < N; ++i) {
        if (held[i] != (salts[(id * 7u + salt + i) % memory.words] ^ id)) {
            ++changed;
        }
    }
    // What the wandering read, kept alive.
    return changed + (wandered == 0xFFFFFFFFu && id == 0xFFFFFFFFu ? 1u : 0u);
}

void count(uint changed) {
    uint was;
    InterlockedAdd(counts[0], 1u, was);
    if (changed != 0u) {
        InterlockedAdd(counts[1], 1u, was);
        InterlockedAdd(counts[2], changed, was);
    }
}

void keeps<let N : int>(uint2 tid) {
    if (tid.x >= memory.threads || tid.y >= memory.threads) {
        return;
    }
    const uint id = tid.y * memory.threads + tid.x;
    count(changedOf<N>(id, 0u));
}

/// Two arrays of N words, one after the other, each in a function of its
/// own and done with before the next is filled: whether the device's
/// compiler gives them the same memory or one each -- whether what a kernel
/// bears is its deepest chain of calls or every function it reaches.
uint firstOf<let N : int>(uint id) { return changedOf<N>(id, 0u); }
uint secondOf<let N : int>(uint id) { return changedOf<N>(id, 12345u); }

void keepsTwo<let N : int>(uint2 tid) {
    if (tid.x >= memory.threads || tid.y >= memory.threads) {
        return;
    }
    const uint id = tid.y * memory.threads + tid.x;
    uint changed = 0u;
    // Which comes first is the buffer's to say, so neither is folded away.
    if ((salts[id % memory.words] & 1u) != 0u) {
        changed += firstOf<N>(id);
        changed += secondOf<N>(id);
    } else {
        changed += secondOf<N>(id);
        changed += firstOf<N>(id);
    }
    count(changed);
}

// The array's size in bytes names the entry: four bytes a word.
[shader("compute")] [numthreads(16, 16, 1)] void keeps1k(uint3 tid: SV_DispatchThreadID) { keeps<256>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps4k(uint3 tid: SV_DispatchThreadID) { keeps<1024>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps8k(uint3 tid: SV_DispatchThreadID) { keeps<2048>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps16k(uint3 tid: SV_DispatchThreadID) { keeps<4096>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps24k(uint3 tid: SV_DispatchThreadID) { keeps<6144>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps32k(uint3 tid: SV_DispatchThreadID) { keeps<8192>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps48k(uint3 tid: SV_DispatchThreadID) { keeps<12288>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps64k(uint3 tid: SV_DispatchThreadID) { keeps<16384>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps128k(uint3 tid: SV_DispatchThreadID) { keeps<32768>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keeps256k(uint3 tid: SV_DispatchThreadID) { keeps<65536>(tid.xy); }

// Two arrays of that size, one after the other.
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo4k(uint3 tid: SV_DispatchThreadID) { keepsTwo<1024>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo8k(uint3 tid: SV_DispatchThreadID) { keepsTwo<2048>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo16k(uint3 tid: SV_DispatchThreadID) { keepsTwo<4096>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo24k(uint3 tid: SV_DispatchThreadID) { keepsTwo<6144>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo32k(uint3 tid: SV_DispatchThreadID) { keepsTwo<8192>(tid.xy); }
[shader("compute")] [numthreads(16, 16, 1)] void keepsTwo64k(uint3 tid: SV_DispatchThreadID) { keepsTwo<16384>(tid.xy); }
