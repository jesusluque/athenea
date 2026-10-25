// Copyright (c) 2026 jesus luque.
//
// A KIND'S ELEMENTS INTERLEAVED for the encoder: each element's words of
// every array one after another (athz_common.slang's order), so the encoder
// binds one buffer for its records and one for their parents -- Metal gives a
// kernel 31 buffers.
module athz_interleave;
import athz_common;

struct InterleaveParams {
    uint shWords;
    uint normals;
    uint emission;
    uint pbrWords;
    uint lobesWords;
    uint transferWords;
    uint shadowWords;
    uint words;
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};
ConstantBuffer<InterleaveParams> params;

ByteAddressBuffer srcPositions;
ByteAddressBuffer srcShape;
ByteAddressBuffer srcSh;
ByteAddressBuffer srcTail;
ByteAddressBuffer srcNormals;
ByteAddressBuffer srcEmission;
ByteAddressBuffer srcPbr;
ByteAddressBuffer srcLobes;
ByteAddressBuffer srcTransfer;
ByteAddressBuffer srcShadow;
RWByteAddressBuffer records;

[shader("compute")]
[numthreads(256, 1, 1)]
void athzInterleave(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= params.count) {
        return;
    }
    AthzLayout l;
    l.shWords = params.shWords;
    l.normals = params.normals;
    l.emission = params.emission;
    l.pbrWords = params.pbrWords;
    l.lobesWords = params.lobesWords;
    l.transferWords = params.transferWords;
    l.shadowWords = params.shadowWords;
    for (uint w = 0; w < params.words; ++w) {
        const uint2 f = athzField(l, w);
        const uint at = athzAddress(l, f, e);
        uint v = 0u;
        switch (f.x) {
        case kFieldPositions: v = srcPositions.Load(at); break;
        case kFieldShape: v = srcShape.Load(at); break;
        case kFieldSh: v = srcSh.Load(at); break;
        case kFieldTail: v = srcTail.Load(at); break;
        case kFieldNormals: v = srcNormals.Load(at); break;
        case kFieldEmission: v = srcEmission.Load(at); break;
        case kFieldPbr: v = srcPbr.Load(at); break;
        case kFieldLobes: v = srcLobes.Load(at); break;
        case kFieldTransfer: v = srcTransfer.Load(at); break;
        default: v = srcShadow.Load(at); break;
        }
        records.Store((e * params.words + w) * 4u, v);
    }
}
