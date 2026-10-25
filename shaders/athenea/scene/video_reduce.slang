// Copyright (c) 2026 jesus luque.
//
// What a gaussian contributes, for a frame cut to fewer before it goes as
// video (athenea video-encode --reduce prune / mix), no camera asked for:
//   kind 0  its opacity times the area of its two widest axes -- a footprint
//           seen from any side
//   kind 1  its opacity times sqrt(that area) times its widest over its
//           middle axis: a thin gaussian (a feather's barb, an edge) counts
//           for its length, so edges are not the first to go One thread a store splat; the score goes to
// lod::Decimator::scoreHistogram and prune, which keep the highest.
module video_reduce;
import athenea.common.packing;

struct ReduceParams {
    uint count;
    uint kind;
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
    const float widest = max(ls.x, max(ls.y, ls.z));
    const float middle = ls.x + ls.y + ls.z - smallest - widest;
    const float opacity = max(positions[s].w, 0.0);
    score[s] = params.kind == 0 ? opacity * exp(widest + middle) : opacity * exp(1.5 * widest - 0.5 * middle);
}

