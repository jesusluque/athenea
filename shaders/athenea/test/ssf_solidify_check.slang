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

/// A THIN PLATE: the first `surface` gaussians its top face at z = +off
/// (normal +z), the rest its bottom at -off (normal -z). How many blobs left
/// their face by more than a tenth of the gap, and how many inside the plate
/// (|x|, |y| < 0.4) lean off their face's normal by more than `lean`.
[shader("compute")]
[numthreads(256, 1, 1)]
void ssfPlateCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const bool top = sources[tid.x] < params.surface;
    const float3 p = blobPositions[tid.x].xyz;
    const float face = top ? params.off : -params.off;
    if (!(abs(p.z - face) <= 0.1 * params.off)) {
        InterlockedAdd(wrong[0], 1u);
    }
    if (abs(p.x) < 0.4 && abs(p.y) < 0.4) {
        const float3 n = unpackNormal(blobNormals[tid.x]);
        if (!(1.0 - n.z * (top ? 1.0 : -1.0) <= params.lean)) {
            InterlockedAdd(wrong[1], 1u);
        }
        InterlockedAdd(wrong[3], 1u);
    }
    InterlockedAdd(wrong[2], 1u);
}
