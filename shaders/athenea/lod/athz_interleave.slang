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
    uint curvatureWords;
    uint influenceWords;
    uint gradientWords;
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
ByteAddressBuffer srcCurvature;
ByteAddressBuffer srcInfluences;
ByteAddressBuffer srcGradients;
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
    l.curvatureWords = params.curvatureWords;
    l.influenceWords = params.influenceWords;
    l.gradientWords = params.gradientWords;
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
        case kFieldShadow: v = srcShadow.Load(at); break;
        case kFieldCurvature: v = srcCurvature.Load(at); break;
        case kFieldInfluences: v = srcInfluences.Load(at); break;
        default: v = srcGradients.Load(at); break;
        }
        records.Store((e * params.words + w) * 4u, v);
    }
}

// --- layer K's arrays ----------------------------------------------------------

struct GatherParams {
    uint count;       ///< elements written
    uint words;       ///< words an element
    uint useOrigin;   ///< 1: `origin` maps a decoded splat to its file record
    uint pad;
};
ConstantBuffer<GatherParams> gather;
ByteAddressBuffer   gatherFrom;   ///< words a file record
ByteAddressBuffer   order;        ///< a store splat's splat of the decoded cloud (LodCloud::order)
ByteAddressBuffer   origin;       ///< a decoded splat's file record (GpuSplats::origin)
RWByteAddressBuffer gatherTo;

/// A file's per-gaussian words in the store's (Morton) order.
[shader("compute")]
[numthreads(256, 1, 1)]
void athzGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= gather.count) {
        return;
    }
    uint r = order.Load(i * 4u);
    if (gather.useOrigin != 0u) {
        r = origin.Load(r * 4u);
    }
    for (uint w = 0; w < gather.words; ++w) {
        gatherTo.Store((i * gather.words + w) * 4u, gatherFrom.Load((r * gather.words + w) * 4u));
    }
}

/// THE JOINTS A FRAME, IN HALVES: each joint's matrix as the skinner takes
/// it (sixteen floats, USD's GfMatrix transposed: the translation in the last
/// column, the last row 0 0 0 1) kept as its first three rows, twelve halves
/// in six words; and back, frame `frame` of them into floats.
struct HalvesParams {
    uint matrices;   ///< joints * frames
    uint joints;
    uint frame;
    uint pad;
};
ConstantBuffer<HalvesParams> halves;
ByteAddressBuffer   matricesIn;
RWByteAddressBuffer halvesOut;
ByteAddressBuffer   halvesIn;
RWByteAddressBuffer matricesOut;

[shader("compute")]
[numthreads(64, 1, 1)]
void athzToHalves(uint3 tid: SV_DispatchThreadID) {
    const uint m = tid.x;
    if (m >= halves.matrices) {
        return;
    }
    uint h[12];
    for (uint k = 0; k < 12u; ++k) {   // rows 0 to 2
        h[k] = f32tof16(asfloat(matricesIn.Load((m * 16u + k) * 4u)));
    }
    for (uint k = 0; k < 6u; ++k) {
        halvesOut.Store((m * 6u + k) * 4u, h[2u * k] | (h[2u * k + 1u] << 16));
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void athzFromHalves(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= halves.joints) {
        return;
    }
    const uint m = halves.frame * halves.joints + j;
    for (uint k = 0; k < 16u; ++k) {
        float v = k == 15u ? 1.0 : 0.0;   // the last row: 0 0 0 1
        if (k < 12u) {
            const uint word = halvesIn.Load((m * 6u + k / 2u) * 4u);
            v = f16tof32((k & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
        }
        matricesOut.Store((j * 16u + k) * 4u, asuint(v));
    }
}
