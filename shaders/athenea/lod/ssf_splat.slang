// Copyright (c) 2026 jesus luque.
//
// MODE A (SSF phase 2, lod::solidToSplats): a solid layer drawn by the
// rasteriser as it is, each blob a flat gaussian again.
//   ssfToSplat  a blob's centre, its tangent radii over the support's ratio,
//               a thin axis of `thinRatio` of the smaller, its frame turned
//               by theta about its normal, its colour; the opacity of the
//               gaussian it came from
//   ssfGather   what each output gaussian takes from the one it came from: the
//               harmonics, the shading normal, and -- for a residual, which is
//               that gaussian -- its position and shape too
module ssf_splat;
import athenea.common.packing;
import ssf_field;

struct SsfSplatParams {
    uint  count;
    uint  first;        ///< the output's first gaussian this dispatch writes
    uint  shWords;
    uint  normals;      ///< 1: the source carries shading normals (srcNormals)
    uint  geometry;     ///< ssfGather: 1 copies positions and shape too
    float radiusScale;
    float thinRatio;
    uint  pad0;
};

StructuredBuffer<float4>   srcPositions;
StructuredBuffer<uint>     srcShape;
StructuredBuffer<uint>     srcSh;
StructuredBuffer<uint>     srcNormals;
StructuredBuffer<uint>     sources;        ///< per output gaussian of the dispatch: its source
RWStructuredBuffer<float4> dstPositions;
RWStructuredBuffer<uint>   dstShape;
RWStructuredBuffer<uint>   dstSh;
RWStructuredBuffer<uint>   dstNormals;
ConstantBuffer<SsfSplatParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfToSplat(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const FieldBlob f = fieldBlob(i);
    const uint4 words = uint4(blobShape[i * 4], blobShape[i * 4 + 1], blobShape[i * 4 + 2], blobShape[i * 4 + 3]);
    const SolidBlob b = unpackSolidBlob(blobPositions[i], words);
    float3 t2 = f.t2;
    if (dot(cross(f.t1, t2), f.n) < 0.0) {
        t2 = -t2;   // a rotation, not a reflection
    }
    const float3x3 axes = float3x3(f.t1.x, t2.x, f.n.x, f.t1.y, t2.y, f.n.y, f.t1.z, t2.z, f.n.z);
    const float3 scale = float3(b.radii / params.radiusScale, params.thinRatio * b.radii.y / params.radiusScale);
    const uint o = params.first + i;
    dstPositions[o] = float4(b.position, srcPositions[sources[i]].w);
    dstShape[o * 4] = encodeQuaternion(quaternionOfAxes(axes));
    dstShape[o * 4 + 1] = packHalves(halfSafe(log(scale.x)), halfSafe(log(scale.y)));
    dstShape[o * 4 + 2] = packHalves(halfSafe(log(scale.z)), halfSafe(b.colour.r));
    dstShape[o * 4 + 3] = packHalves(halfSafe(b.colour.g), halfSafe(b.colour.b));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint s = sources[i];
    const uint o = params.first + i;
    for (uint k = 0; k < params.shWords; ++k) {
        dstSh[o * params.shWords + k] = srcSh[s * params.shWords + k];
    }
    if (params.normals != 0) {
        dstNormals[o] = srcNormals[s];
    }
    if (params.geometry != 0) {
        dstPositions[o] = srcPositions[s];
        [unroll] for (uint k = 0; k < 4; ++k) {
            dstShape[o * 4 + k] = srcShape[s * 4 + k];
        }
    }
}
