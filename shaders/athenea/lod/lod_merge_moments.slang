// Copyright (c) 2026 jesus luque.
//
// A coarser level's moments: each group's children at the finer level,
// summed. A group's children are the finer groups of its splats -- contiguous,
// from the finer group of its first splat to that of the next group's first.
// The second moments are each about their own group's mean (lod_common), so a
// child's is carried to the parent's mean by the parallel axis theorem: its
// own, plus its weight times the outer product of its mean's offset.
module lod_merge_moments;
import lod_common;

StructuredBuffer<uint>     starts;       // this level, per group
StructuredBuffer<uint>     fineGroup;    // finer level, per splat
StructuredBuffer<float>    fineMoments;
RWStructuredBuffer<float>  moments;
ConstantBuffer<LodParams>  params;       // count: splats; groups; fineGroups

[shader("compute")]
[numthreads(64, 1, 1)]
void lodMergeMoments(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.groups) {
        return;
    }
    const uint first = fineGroup[starts[g]];
    const uint end = g + 1 < params.groups ? fineGroup[starts[g + 1]] : params.fineGroups;
    const uint at = g * params.stride;
    for (uint k = 0; k < params.stride; ++k) {
        if (k >= 4 && k <= 9) {
            continue;   // the second moments, below
        }
        float sum = 0.0;
        for (uint h = first; h < end; ++h) {
            sum += fineMoments[h * params.stride + k];
        }
        moments[at + k] = sum;
    }
    const float total = moments[at];
    const float3 mean = total > 0.0 ? float3(moments[at + 1], moments[at + 2], moments[at + 3]) / total
                                    : float3(0.0);
    float s[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    for (uint h = first; h < end; ++h) {
        const uint from = h * params.stride;
        const float w = fineMoments[from];
        const float3 d = w > 0.0 ? float3(fineMoments[from + 1], fineMoments[from + 2], fineMoments[from + 3]) / w - mean
                                 : float3(0.0);
        s[0] += fineMoments[from + 4] + w * d.x * d.x;
        s[1] += fineMoments[from + 5] + w * d.x * d.y;
        s[2] += fineMoments[from + 6] + w * d.x * d.z;
        s[3] += fineMoments[from + 7] + w * d.y * d.y;
        s[4] += fineMoments[from + 8] + w * d.y * d.z;
        s[5] += fineMoments[from + 9] + w * d.z * d.z;
    }
    for (uint k = 0; k < 6; ++k) {
        moments[at + 4 + k] = s[k];
    }
}
