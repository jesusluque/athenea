// Copyright (c) 2026 jesus luque.
//
// A HARD MERGE SAYS ITS p ONE WAY (test_lod's hard surfels): the hardness a
// decimation writes beside each gaussian (`athenea:splat:hardness`) is the
// one the flat axis of its shape carries (packing's hard surfel), within
// [2, the most asked]; and a flat merge of eight or more discs of an even
// floor comes out harder than a gaussian (two by two, overlapping as these
// do, is still kurtosis 2.66, p 2.6: hardly a box yet).
//   counts[0] hardness and shape disagree   [1] out of range
//   counts[2] surfels of eight or more       [3] of them harder than 2.5
module hardness_check;
import athenea.common.packing;

struct HardnessCheckParams {
    uint  count;
    float most;
};

StructuredBuffer<uint>   shape;
StructuredBuffer<float>  hardness;
StructuredBuffer<uint2>  ranges;
RWStructuredBuffer<uint> counts;
ConstantBuffer<HardnessCheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void hardnessCheck(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= params.count) {
        return;
    }
    const float3 logs = float3(lowHalf(shape[c * 4 + 1]), highHalf(shape[c * 4 + 1]), lowHalf(shape[c * 4 + 2]));
    const float flat = min(min(logs.x, logs.y), logs.z);
    const float p = hardness[c];
    if (abs(p - surfelHardness(flat)) > 0.05) {
        InterlockedAdd(counts[0], 1u);
    }
    if (!(p >= 2.0 && p <= params.most + 1.0e-3)) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint members = ranges[c].y - ranges[c].x;
    if (members >= 8u && isSurfel(exp(logs))) {
        InterlockedAdd(counts[2], 1u);
        if (p > 2.5) {
            InterlockedAdd(counts[3], 1u);
        }
    }
}
