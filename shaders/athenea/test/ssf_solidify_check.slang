// Copyright (c) 2026 jesus luque.
//
// A SOLIDIFIED SPHERE, CHECKED ON THE DEVICE (SSF phase 2): of a cloud whose
// first `surface` gaussians lie flat on the unit sphere and the rest do not,
//   ssfSourcesCheck  how many of a list of sources are surface gaussians
//   ssfSphereCheck   how many blobs stand off the sphere by more than `off`,
//                    and how many normals lean off the radius by more than
//                    `lean` (as 1 - cos)
module ssf_solidify_check;
import athenea.common.packing;

struct SsfSolidifyCheckParams {
    uint  count;
    uint  surface;
    float off;
    float lean;
};

StructuredBuffer<uint>   sources;
StructuredBuffer<float4> blobPositions;
StructuredBuffer<uint>   blobNormals;
RWStructuredBuffer<uint> wrong;   ///< ssfSourcesCheck: [0] surface, [1] read; ssfSphereCheck: [0] off, [1] lean, [2] read, [3] worst off x 1e6
ConstantBuffer<SsfSolidifyCheckParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfSourcesCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    if (sources[tid.x] < params.surface) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[1], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfSphereCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const float3 p = blobPositions[tid.x].xyz;
    const float off = abs(length(p) - 1.0);
    if (!(off <= params.off)) {
        InterlockedAdd(wrong[0], 1u);
    }
    const float3 n = unpackNormal(blobNormals[tid.x]);
    if (!(1.0 - dot(n, normalize(p)) <= params.lean)) {
        InterlockedAdd(wrong[1], 1u);
    }
    InterlockedAdd(wrong[2], 1u);
    InterlockedMax(wrong[3], uint(min(off, 4000.0) * 1.0e6));
}
