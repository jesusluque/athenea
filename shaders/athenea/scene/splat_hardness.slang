// Copyright (c) 2026 jesus luque.
//
// A HARD SURFEL'S p INTO ITS RECORD (`athenea:splat:hardness`, packing's hard
// surfel), after streams.slang laid the records out: one float a splat, 2 a
// gaussian, written as the negative of itself in place of the surfel's zero
// scale, which splat_encoding's decodeLogScale turns into the logarithm that
// carries it. A module of its own: streams binds every buffer Metal gives a
// kernel.
module splat_hardness;
import athenea.common.packing;

struct HardnessParams {
    uint count;    // records in this slice
    uint first;    // the slice's first element of the stream
    uint stride;   // floats a record
    uint half;     // 1: the stream is float16
};

StructuredBuffer<uint>       hardness;
RWStructuredBuffer<float>    records;
ConstantBuffer<HardnessParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatHardness(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const float p = params.half != 0 ? ((e & 1u) == 0 ? lowHalf(hardness[e / 2]) : highHalf(hardness[e / 2]))
                                     : asfloat(hardness[e]);
    if (!(p > 2.0)) {
        return;
    }
    const uint r = i * params.stride;
    for (uint k = 0; k < 3; ++k) {
        if (records[r + 4 + k] == 0.0) {
            records[r + 4 + k] = -p;
        }
    }
}
