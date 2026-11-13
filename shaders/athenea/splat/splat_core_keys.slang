// Copyright (c) 2026 jesus luque.
//
// One axis of a cloud's positions as keys the radix sort orders as the floats
// they are: the sign bit flipped for a positive value, every bit for a
// negative one. render::coreBounds sorts them and reads the two at the share
// it keeps, so a capture's few splats thousands of units out do not frame
// the shot. A position that is no number sorts past every finite one.
module splat_core_keys;

StructuredBuffer<float4>   positions;
RWStructuredBuffer<uint>   keys;
RWStructuredBuffer<uint>   values;

struct CoreKeyParams {
    uint count;
    uint axis;
};
ConstantBuffer<CoreKeyParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatCoreKeys(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float v = positions[i][params.axis];
    const uint bits = asuint(v);
    keys[i] = isnan(v) ? 0xFFFFFFFFu : ((bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u);
    values[i] = i;
}
