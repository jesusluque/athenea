// Copyright (c) 2026 jesus luque.
//
// `athenea flatten --facing`: one face of the cloud seen from the facing
// point, its depth copied into the cube the flatten reads
// (splat_project.slang's facingSeen).
module flatten_facing;

struct FacingGatherParams {
    uint pixels;   // a face's
    uint face;
};

StructuredBuffer<float>   faceDepth;
RWStructuredBuffer<float> cube;
ConstantBuffer<FacingGatherParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void flattenFacingGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.pixels) {
        return;
    }
    cube[params.face * params.pixels + i] = faceDepth[i];
}
