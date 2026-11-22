// Copyright (c) 2026 jesus luque.
//
// A mask lifted to the gaussians (athenea/splat/splat_lift.slang), checked
// against what the Cryptomatte knows: a prim's own coverage is the matte a
// lift of that prim should draw, from whichever camera.
//
// Two entry points over one frame. `matteMask` makes the mask a person would
// have painted on it -- a pixel is in where the prim is most of what covers
// it -- and `matteCheck` measures a matte against the prim's coverage.
module matte_check;

struct MatteCheckParams {
    uint  width;
    uint  height;
    uint  id;           // the prim's Cryptomatte id
    float tolerance;    // how far a pixel's matte may be from the prim's coverage
};

StructuredBuffer<float4>           crypto;   // 3 layers: crypto[layer * pixels + at]
StructuredBuffer<float4>           colour;
StructuredBuffer<float>            matte;
RWStructuredBuffer<float>          mask;
RWStructuredBuffer<uint>           stats;    // 8 wide, one row: see matteCheck
ConstantBuffer<MatteCheckParams>   params;

float coverageOf(uint at, uint pixels) {
    float sum = 0.0;
    for (uint layer = 0; layer < 3; ++layer) {
        const float4 row = crypto[layer * pixels + at];
        if (bit_cast<uint>(row.x) == params.id) {
            sum += row.y;
        }
        if (bit_cast<uint>(row.z) == params.id) {
            sum += row.w;
        }
    }
    return sum;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void matteMask(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    mask[at] = coverageOf(at, pixels) > 0.5 * colour[at].w ? 1.0 : 0.0;
}

/// stats[0] pixels anything covers (alpha above 0.02)
/// stats[1] of those, pixels whose matte is further than the tolerance from the prim's coverage
/// stats[2] the sum of |matte - coverage| over them, in 1/1024 of a pixel
/// stats[3] the prim's coverage summed over them, in 1/1024 of a pixel
/// stats[4] the matte summed over them, in 1/1024 of a pixel
[shader("compute")]
[numthreads(16, 16, 1)]
void matteCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    if (!(colour[at].w > 0.02)) {
        return;
    }
    const float truth = coverageOf(at, pixels);
    const float ours = matte[at];
    InterlockedAdd(stats[0], 1u);
    if (abs(ours - truth) > params.tolerance) {
        InterlockedAdd(stats[1], 1u);
    }
    InterlockedAdd(stats[2], uint(abs(ours - truth) * 1024.0 + 0.5));
    InterlockedAdd(stats[3], uint(truth * 1024.0 + 0.5));
    InterlockedAdd(stats[4], uint(ours * 1024.0 + 0.5));
}
