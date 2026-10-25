// Copyright (c) 2026 jesus luque.
//
// What a gaussian contributes, for a frame cut to fewer before it goes as
// video (athenea video-encode --reduce prune / mix): its opacity times the
// area of its two widest axes -- a footprint seen from any side, no camera
// asked for. One thread a store splat; the score goes to
// lod::Decimator::scoreHistogram and prune, which keep the highest.
module video_reduce;
import athenea.common.packing;

struct ReduceParams {
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<float4>     positions;   // xyz, opacity
StructuredBuffer<uint>       shape;       // packing's four words
RWStructuredBuffer<float>    score;
ConstantBuffer<ReduceParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void videoReduceScore(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= params.count) {
        return;
    }
    const uint w1 = shape[s * 4 + 1], w2 = shape[s * 4 + 2];
    const float3 ls = clamp(float3(lowHalf(w1), highHalf(w1), lowHalf(w2)), float3(-30.0), float3(10.0));
    const float smallest = min(ls.x, min(ls.y, ls.z));
    score[s] = max(positions[s].w, 0.0) * exp(ls.x + ls.y + ls.z - smallest);
}
