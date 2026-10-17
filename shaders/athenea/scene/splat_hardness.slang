// Copyright (c) 2026 jesus luque.
//
// A HARD SURFEL'S p INTO ITS RECORD (`athenea:splat:hardness`, packing's hard
// surfel), after streams.slang laid the records out: one float a splat, 2 a
// gaussian, written as the negative of itself in place of the surfel's zero
// scale, which splat_encoding's decodeLogScale turns into the logarithm that
// carries it. A module of its own: streams binds every buffer Metal gives a
// kernel.
//
// THE MATERIAL'S OWN ROUGHNESS on the pbr word too
// (`primvars:athenea:splat:materialRoughness`, splat_encoding's
// kMaterialRoughnessMark): eight times one more than its 5-bit step, added to
// the transmission streams wrote with its thin wall's and Schlick's marks.
module splat_hardness;
import athenea.common.packing;

struct HardnessParams {
    uint count;    // records in this slice
    uint first;    // the slice's first element of the stream
    uint stride;   // floats a record
    uint half;     // 1: the stream is float16
    uint pbrBase;  // the record's metallic, roughness, transmission (splatMaterialRoughness)
};

StructuredBuffer<uint>       hardness;
StructuredBuffer<uint>       materialRoughness;
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

static const float kMaterialRoughnessMark = 8.0;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatMaterialRoughness(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint e = params.first + i;
    const float v = params.half != 0
                        ? ((e & 1u) == 0 ? lowHalf(materialRoughness[e / 2]) : highHalf(materialRoughness[e / 2]))
                        : asfloat(materialRoughness[e]);
    records[i * params.stride + params.pbrBase + 2] += kMaterialRoughnessMark * (1.0 + round(saturate(v) * 31.0));
}
