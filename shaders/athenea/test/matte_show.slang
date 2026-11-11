// Copyright (c) 2026 jesus luque.
//
// A matte and the coverage it is measured against (matte_check.slang), as two
// grey pictures for a person to look at: a debug dump, nothing is checked here.
module matte_show;

struct MatteShowParams {
    uint width;
    uint height;
    uint id;      // the prim's Cryptomatte id
    uint pad0;
};

StructuredBuffer<float4>          crypto;
StructuredBuffer<float>           matte;
RWStructuredBuffer<float4>        shownMatte;
RWStructuredBuffer<float4>        shownTruth;
ConstantBuffer<MatteShowParams>   params;

[shader("compute")]
[numthreads(16, 16, 1)]
void matteShow(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    // The pictures are written top row first, as a file is read.
    const uint at = tid.y * params.width + tid.x;
    const uint to = (params.height - 1 - tid.y) * params.width + tid.x;
    float truth = 0.0;
    for (uint layer = 0; layer < 3; ++layer) {
        const float4 row = crypto[layer * pixels + at];
        truth += bit_cast<uint>(row.x) == params.id ? row.y : 0.0;
        truth += bit_cast<uint>(row.z) == params.id ? row.w : 0.0;
    }
    // Squared, since the dump encodes with a gamma and a matte is not light.
    shownMatte[to] = float4(float3(pow(saturate(matte[at]), 2.2)), 1.0);
    shownTruth[to] = float4(float3(pow(saturate(truth), 2.2)), 1.0);
}
