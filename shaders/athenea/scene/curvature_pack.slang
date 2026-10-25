// Copyright (c) 2026 jesus luque.
//
// A CLOUD'S SHAPE OPERATORS INTO ONE WORD A GAUSSIAN (common/packing.slang's
// packCurvature): what a `.athc` and the levels of detail carry.
module curvature_pack;
import athenea.common.packing;

struct CurvaturePackParams {
    uint count;
};

StructuredBuffer<float>                curvatureIn;    // 3 a gaussian
RWStructuredBuffer<uint>               curvatureOut;   // 1 a gaussian
ConstantBuffer<CurvaturePackParams>    params;

[shader("compute")]
[numthreads(64, 1, 1)]
void curvaturePack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    curvatureOut[i] = packCurvature(float3(curvatureIn[i * 3], curvatureIn[i * 3 + 1], curvatureIn[i * 3 + 2]));
}
