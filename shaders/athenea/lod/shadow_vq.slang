// Copyright (c) 2026 jesus luque.
//
// A GAUSSIAN'S OPEN DIRECTIONS FROM A CODEBOOK (`athenea decimate
// --shadow-codebook`; research 096). A transfer's shadow bits are 256 a
// gaussian (a 16 x 16 octahedral grid, one bit a cell open to the sky): 32
// bytes, more than the whole of the rest of a compressed gaussian. Here they
// are clustered by Hamming distance -- k-means where a centroid is its
// members' majority, bit by bit -- and each gaussian's bits replaced by its
// centroid's, which is what a file holding one index a gaussian and the
// codebook would draw. The codebook starts from gaussians spread through the
// cloud's order.
module shadow_vq;

struct VqParams {
    uint count;       ///< gaussians
    uint clusters;    ///< C
    uint words;       ///< 32-bit words a gaussian's bits take (8)
    uint pad;
};

RWStructuredBuffer<uint> bits;        // words a gaussian
RWStructuredBuffer<uint> centroids;   // words a cluster
RWStructuredBuffer<uint> assigned;    // a gaussian's cluster
RWStructuredBuffer<uint> ones;        // clusters x words x 32: members with the bit set
RWStructuredBuffer<uint> members;     // a cluster's
RWStructuredBuffer<uint> distances;   // 257 bins: how many gaussians end that many bits from their centroid
ConstantBuffer<VqParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void vqSeed(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.clusters) {
        return;
    }
    const uint g = uint((uint64_t(c) * params.count) / max(params.clusters, 1u));
    for (uint w = 0; w < params.words; ++w) {
        centroids[c * params.words + w] = bits[g * params.words + w];
    }
}

/// The nearest centroid, and its members' bits counted.
[shader("compute")]
[numthreads(64, 1, 1)]
void vqAssign(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    uint own[8];
    for (uint w = 0; w < 8u; ++w) {
        own[w] = w < params.words ? bits[g * params.words + w] : 0u;
    }
    uint best = 0xFFFFFFFFu;
    uint chosen = 0u;
    for (uint c = 0; c < params.clusters; ++c) {
        uint d = 0u;
        for (uint w = 0; w < params.words; ++w) {
            d += countbits(own[w] ^ centroids[c * params.words + w]);
        }
        if (d < best) {
            best = d;
            chosen = c;
        }
    }
    assigned[g] = chosen;
    InterlockedAdd(members[chosen], 1u);
    for (uint w = 0; w < params.words; ++w) {
        uint word = own[w];
        while (word != 0u) {
            const uint b = firstbitlow(word);
            word &= word - 1u;
            InterlockedAdd(ones[(chosen * params.words + w) * 32u + b], 1u);
        }
    }
}

/// Each centroid its members' majority, bit by bit; the counts cleared for the
/// next round. A cluster nobody chose keeps what it had.
[shader("compute")]
[numthreads(64, 1, 1)]
void vqUpdate(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;   // cluster * words + word
    if (at >= params.clusters * params.words) {
        return;
    }
    const uint c = at / params.words;
    const uint n = members[c];
    if (n > 0u) {
        uint word = 0u;
        for (uint b = 0; b < 32u; ++b) {
            if (ones[at * 32u + b] * 2u > n) {
                word |= 1u << b;
            }
        }
        centroids[at] = word;
    }
    for (uint b = 0; b < 32u; ++b) {
        ones[at * 32u + b] = 0u;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void vqClearMembers(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.clusters) {
        members[tid.x] = 0u;
    }
}

/// Every gaussian its centroid's bits, and how far they were from its own.
[shader("compute")]
[numthreads(64, 1, 1)]
void vqApply(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint c = assigned[g];
    uint d = 0u;
    for (uint w = 0; w < params.words; ++w) {
        const uint was = bits[g * params.words + w];
        const uint now = centroids[c * params.words + w];
        d += countbits(was ^ now);
        bits[g * params.words + w] = now;
    }
    InterlockedAdd(distances[min(d, 256u)], 1u);
}
