// Copyright (c) 2026 jesus luque.
//
// A Cryptomatte, checked against the frame it was written beside: every rank
// names an id the scene has, the ranks descend, no id is named twice, and what
// the ranks cover adds up to the pixel's own alpha. The specification's
// promise, in other words, measured on the device that made it.
module crypto_check;

struct CryptoParams {
    uint  width;
    uint  height;
    uint  allowedCount;   // ids in `allowed`, 0 included: a pixel nothing covered
    float tolerance;      // how far the coverages may be from the pixel's alpha
    uint  pad0; uint pad1; uint pad2; uint pad3;
};

StructuredBuffer<float4>     crypto;    // 3 layers: crypto[layer * pixels + at]
StructuredBuffer<float4>     colour;    // the frame the matte was written beside
StructuredBuffer<uint>       allowed;   // the ids the scene can name
RWStructuredBuffer<uint>     stats;     // 8 wide, one row: see below
ConstantBuffer<CryptoParams> params;

static const uint kLayers = 3;

bool isAllowed(uint id) {
    for (uint k = 0; k < params.allowedCount; ++k) {
        if (allowed[k] == id) {
            return true;
        }
    }
    return false;
}

/// stats[0] ranks naming an id the scene does not hold
/// stats[1] pixels whose ranks do not descend
/// stats[2] pixels naming one id twice
/// stats[3] pixels whose coverage misses the pixel's alpha by more than the tolerance
/// stats[4] pixels with any coverage at all
/// stats[5] pixels naming allowed[0], stats[6] allowed[1], stats[7] both
[shader("compute")]
[numthreads(16, 16, 1)]
void cryptoCheck(uint3 tid: SV_DispatchThreadID) {
    const uint x = tid.x;
    const uint y = tid.y;
    if (x >= params.width || y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = y * params.width + x;

    uint ids[kLayers * 2];
    float covs[kLayers * 2];
    for (uint layer = 0; layer < kLayers; ++layer) {
        const float4 row = crypto[layer * pixels + at];
        ids[layer * 2] = bit_cast<uint>(row.x);
        covs[layer * 2] = row.y;
        ids[layer * 2 + 1] = bit_cast<uint>(row.z);
        covs[layer * 2 + 1] = row.w;
    }

    float sum = 0.0;
    float previous = 3.4e38;
    bool descends = true;
    bool twice = false;
    for (uint k = 0; k < kLayers * 2; ++k) {
        if (covs[k] <= 0.0) {
            continue;
        }
        sum += covs[k];
        if (covs[k] > previous) {
            descends = false;
        }
        previous = covs[k];
        if (!isAllowed(ids[k])) {
            InterlockedAdd(stats[0], 1u);
        }
        for (uint j = 0; j < k; ++j) {
            if (covs[j] > 0.0 && ids[j] == ids[k]) {
                twice = true;
            }
        }
    }
    if (!descends) {
        InterlockedAdd(stats[1], 1u);
    }
    if (twice) {
        InterlockedAdd(stats[2], 1u);
    }
    if (abs(sum - colour[at].w) > params.tolerance) {
        InterlockedAdd(stats[3], 1u);
    }
    if (sum > 0.0) {
        InterlockedAdd(stats[4], 1u);
    }
    bool first = false;
    bool second = false;
    for (uint k = 0; k < kLayers * 2; ++k) {
        if (covs[k] <= 0.0) {
            continue;
        }
        first = first || (params.allowedCount > 0 && ids[k] == allowed[0]);
        second = second || (params.allowedCount > 1 && ids[k] == allowed[1]);
    }
    if (first) {
        InterlockedAdd(stats[5], 1u);
    }
    if (second) {
        InterlockedAdd(stats[6], 1u);
    }
    if (first && second) {
        InterlockedAdd(stats[7], 1u);
    }
}
