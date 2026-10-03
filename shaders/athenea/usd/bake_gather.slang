// Copyright (c) 2026 jesus luque.
//
// A bake's answer, a point's entries together (`StageRenderer::bakePoints`).
//
// The path tracer writes a bake as a plane an entry over its grid -- every
// point's constant term, then every point's first harmonic, and so on -- and
// what the caller wants is a point at a time, in the order its records are
// in. This lays one pass's planes out so, at `first` of the whole answer: a
// bake taken in passes fills one buffer, and nothing crosses to the
// processor to be rearranged there.
module bake_gather;

struct GatherParams {
    uint count;     ///< points in this pass
    uint first;     ///< the first of them in the whole answer
    uint entries;   ///< float4 a point: the coefficients, and two more for a transfer
    uint plane;     ///< float4 a plane of the pass's grid
};

StructuredBuffer<float4>     planes;
RWStructuredBuffer<float4>   answer;
ConstantBuffer<GatherParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void bakeGather(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint into = (params.first + k) * params.entries;
    for (uint c = 0; c < params.entries; ++c) {
        answer[into + c] = planes[c * params.plane + k];
    }
}
