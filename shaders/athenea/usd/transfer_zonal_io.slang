// Copyright (c) 2026 jesus luque.
//
// A transfer bake into the picture the splat transfer zonal fit reads, its
// answer back, and -- for a cloud a skeleton carries -- the bake's rays put
// where the pose stands (`athenea mesh2splat --transfer`,
// plugins/splattransferzonal).
//
// zonalPoseRays: a skinned cloud is built in the bind pose, and the stage the
// bake traces is posed at `--time`. Each ray is carried by its gaussian's own
// motion from the one to the other -- the skinner's posed point, and the turn
// from the rest frame to the posed one -- so it stands on the surface the
// tracer holds. Dropped records (validation keeps what can be drawn) are left
// alone: their position words read as the fill, a NaN.
//
// zonalPack: four entries a gaussian -- harmonics 0-3, 4-7, (8, the bits as
// the floats they are the bits of, found), the gaussian's frame as a
// quaternion (x, y, z, w), the posed one where a skeleton carries it.
//
// zonalUnpack: the fit's three entries a gaussian into what the file keeps:
// ten floats a gaussian (two lobes of axis u, v, z0, z1, z2) and two words of
// bits, now in the gaussian's frame.
module transfer_zonal_io;
import athenea.common.packing;

struct ZonalIoParams {
    uint count;    ///< records
    uint width;    ///< entries a row of the picture
    uint stride;   ///< pixels a row of it, padded
    uint pad0;
};

StructuredBuffer<float>       direct;          // 9 a record, the bake's direct half
StructuredBuffer<int>         bitsIn;          // 2 a record, as traced
StructuredBuffer<uint>        frames;          // 4 words a record: a gaussian's shape, word 0 its rotation
StructuredBuffer<uint>        restPositions;   // 4 words a record (float4), NaN where dropped
StructuredBuffer<uint>        posedPositions;  // 4 words a record (float4)
StructuredBuffer<uint>        restShape;       // 4 words a record
RWStructuredBuffer<float4>    rays;            // 3 a record (bakePointsOnDevice's layout)
RWStructuredBuffer<float4>    picture;
RWStructuredBuffer<float>     zonal;           // 10 a record
RWStructuredBuffer<int>       bitsOut;         // 2 a record, in the gaussian's frame
ConstantBuffer<ZonalIoParams> io;

uint at(uint entry) {
    return (entry / io.width) * io.stride + entry % io.width;
}

float3 positionOf(StructuredBuffer<uint> words, uint r) {
    return float3(asfloat(words[r * 4]), asfloat(words[r * 4 + 1]), asfloat(words[r * 4 + 2]));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void zonalPoseRays(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= io.count) {
        return;
    }
    const float3 rest = positionOf(restPositions, r);
    if (any(isnan(rest))) {
        return;
    }
    const float3 posed = positionOf(posedPositions, r);
    const float3x3 restAxes = axesOfQuaternion(decodeQuaternion(restShape[r * 4]));
    const float3x3 posedAxes = axesOfQuaternion(decodeQuaternion(frames[r * 4]));
    // The turn that takes the rest frame to the posed one.
    const float3x3 turn = mul(posedAxes, transpose(restAxes));
    const float4 start = rays[r * 3];
    const float4 facing = rays[r * 3 + 1];
    const float4 relief = rays[r * 3 + 2];
    rays[r * 3] = float4(posed + mul(turn, start.xyz - rest), start.w);
    rays[r * 3 + 1] = float4(mul(turn, facing.xyz), facing.w);
    rays[r * 3 + 2] = float4(mul(turn, relief.xyz), relief.w);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void zonalPack(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= io.count) {
        return;
    }
    const uint first = r * 9;
    picture[at(r * 4)] = float4(direct[first], direct[first + 1], direct[first + 2], direct[first + 3]);
    picture[at(r * 4 + 1)] = float4(direct[first + 4], direct[first + 5], direct[first + 6], direct[first + 7]);
    // m2sTransferInto left a gaussian the bake found nothing under at zero;
    // the fit leaves those alone.
    float whole = 0.0;
    for (uint k = 0; k < 9; ++k) {
        whole += abs(direct[first + k]);
    }
    picture[at(r * 4 + 2)] =
        float4(direct[first + 8], asfloat(bitsIn[r * 2]), asfloat(bitsIn[r * 2 + 1]), whole > 0.0 ? 1.0 : 0.0);
    picture[at(r * 4 + 3)] = decodeQuaternion(frames[r * 4]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void zonalUnpack(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r >= io.count) {
        return;
    }
    const float4 e0 = picture[at(r * 3)];
    const float4 e1 = picture[at(r * 3 + 1)];
    const float4 e2 = picture[at(r * 3 + 2)];
    const uint first = r * 10;
    zonal[first] = e0.x;
    zonal[first + 1] = e0.y;
    zonal[first + 2] = e0.z;
    zonal[first + 3] = e0.w;
    zonal[first + 4] = e1.x;
    zonal[first + 5] = e1.y;
    zonal[first + 6] = e1.z;
    zonal[first + 7] = e1.w;
    zonal[first + 8] = e2.x;
    zonal[first + 9] = e2.y;
    bitsOut[r * 2] = asint(e2.z);
    bitsOut[r * 2 + 1] = asint(e2.w);
}
