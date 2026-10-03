// Copyright (c) 2026 jesus luque.
//
// How far off the surface a bake's ray starts, from the cloud's own box.
//
// Far enough that the ray does not begin inside the triangle it is about to
// hit, and near enough that nothing else fits in between: a ten-thousandth of
// the diagonal of the box every gaussian stands in. A fraction of the MODEL,
// and of nothing else -- taken from the scene's unit instead, the chess
// pawn's rays began a millimetre off a 66 mm model, inside the glass ball,
// and the gold ring under it baked grey (docs/decisions.md).
//
// The box was folded on the processor over records read back for it; it is
// now two passes here -- a low and a high corner a chunk of records, which
// `athenea/scene/bounds_reduce` folds -- and the offset is written into every
// ray by the device that reads it.
module mesh2splat_span;

struct SpanParams {
    uint count;       ///< gaussians in the cloud
    uint perRecord;   ///< floats a record
    uint chunkSize;   ///< records a chunk
    uint chunkCount;
};

StructuredBuffer<float>    records;
RWStructuredBuffer<float4> extents;   // chunkCount * 2: low, high
StructuredBuffer<float4>   box;       // 2: the folded low and high corners
RWStructuredBuffer<float4> rays;      // three a gaussian; the first's w is the offset
ConstantBuffer<SpanParams> span;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sRecordChunks(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= span.chunkCount) {
        return;
    }
    const uint begin = c * span.chunkSize;
    const uint end = min(begin + span.chunkSize, span.count);
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    for (uint i = begin; i < end; ++i) {
        const uint at = i * span.perRecord;
        const float3 p = float3(records[at], records[at + 1], records[at + 2]);
        lo = min(lo, p);
        hi = max(hi, p);
    }
    extents[c * 2] = float4(lo, 0.0);
    extents[c * 2 + 1] = float4(hi, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sRaySpan(uint3 tid: SV_DispatchThreadID) {
    const uint splat = tid.x;
    if (splat >= span.count) {
        return;
    }
    const float diagonal = length(box[1].xyz - box[0].xyz);
    const float step = (diagonal > 0.0 ? diagonal : 1.0) * 1.0e-4;
    const float4 start = rays[splat * 3];
    rays[splat * 3] = float4(start.xyz, step);
}
