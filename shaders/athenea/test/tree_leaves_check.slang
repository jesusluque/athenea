// Copyright (c) 2026 jesus luque.
//
// HOW MANY GAUSSIANS STAND PAST `params.limit` IN X: what a test of a merge
// tree's leaves reads back (its merges are written there, far from them).
module tree_leaves_check;

struct TreeCheckParams {
    uint  count;
    float limit;
};

StructuredBuffer<float4> positions;
RWStructuredBuffer<uint> wrong;   ///< [0] past the limit, [1] read
ConstantBuffer<TreeCheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void treeLeavesCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    if (positions[i].x > params.limit) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[1], 1u);
}
