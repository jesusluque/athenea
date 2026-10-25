// Copyright (c) 2026 jesus luque.
//
// A PRODUCER'S DEVICE WRITING GAUSSIANS (tests/mcp/test_bridge.cpp): `count`
// float records of degree 0 from `first`, a 3DGS PLY vertex's fourteen floats
// (x y z, opacity logit, three log scales, rotation w x y z, DC), drawn from
// a hash of the seed and the index -- what another process puts into the
// pages it shares with the engine before it raises their fence.
module bridge_records;

struct RecordParams {
    uint count;
    uint first;
    uint seed;
    uint stride;
};

RWStructuredBuffer<float>     records;
ConstantBuffer<RecordParams>  params;

uint hash(uint v) {
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    return v;
}

float unit(uint i, uint k) {
    return float(hash(i * 977u + k * 131u + params.seed * 7919u) & 0xFFFFFFu) / 16777215.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bridgeTestRecords(uint3 id: SV_DispatchThreadID) {
    const uint i = id.x;
    if (i >= params.count) {
        return;
    }
    const uint at = (params.first + i) * params.stride;
    records[at + 0] = lerp(-1.6, 1.6, unit(i, 0));
    records[at + 1] = lerp(-1.2, 1.2, unit(i, 1));
    records[at + 2] = lerp(-1.0, 1.0, unit(i, 2));
    records[at + 3] = lerp(-1.0, 3.0, unit(i, 3));   // opacity, a logit
    const float size = log(lerp(0.03, 0.12, unit(i, 4)));
    records[at + 4] = size;
    records[at + 5] = size + log(lerp(0.3, 1.0, unit(i, 5)));
    records[at + 6] = size + log(lerp(0.2, 1.0, unit(i, 6)));
    float4 q = float4(unit(i, 7), unit(i, 8), unit(i, 9), unit(i, 10)) * 2.0 - 1.0;
    q = length(q) > 1e-3 ? normalize(q) : float4(1, 0, 0, 0);
    records[at + 7] = q.x;
    records[at + 8] = q.y;
    records[at + 9] = q.z;
    records[at + 10] = q.w;
    records[at + 11] = lerp(-1.0, 1.5, unit(i, 11));
    records[at + 12] = lerp(-1.0, 1.5, unit(i, 12));
    records[at + 13] = lerp(-1.0, 1.5, unit(i, 13));
}
