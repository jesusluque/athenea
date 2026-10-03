// Copyright (c) 2026 jesus luque.
//
// The triangles of one GeomSubset, as a list in the mesh's own order.
//
// `geom::MeshBuilder` says which subset every triangle is in
// (`triangleSubsets`: 0 for none, k + 1 for the k-th, the top bit for a face
// that is not drawn). A mesh whose subsets bind different materials is
// converted a subset at a time, each with its own; what that takes is the
// list of the subset's triangles -- a flag a triangle, a prefix sum
// (`gpu::PrefixSum`) and a scatter, so the list keeps the mesh's order and a
// conversion of it is the same array whatever thread ran first.
module mesh2splat_subset;

struct SubsetParams {
    uint triangles;   ///< of the mesh
    uint wanted;      ///< the subset's index plus one; 0: the triangles no subset claims
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>       triangleSubsets;
RWStructuredBuffer<uint>     flags;
StructuredBuffer<uint>       offsets;
RWStructuredBuffer<uint>     list;
ConstantBuffer<SubsetParams> subset;

static const uint kHidden = 0x80000000u;

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sSubsetFlags(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= subset.triangles) {
        return;
    }
    flags[t] = (triangleSubsets[t] & ~kHidden) == subset.wanted ? 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void m2sSubsetScatter(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= subset.triangles) {
        return;
    }
    if (flags[t] != 0) {
        list[offsets[t]] = t;
    }
}
