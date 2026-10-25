// Copyright (c) 2026 jesus luque.
//
// A MERGE TREE'S LEAVES (`primvars:athenea:splat:lodTree`): the gaussians that
// stand for one leaf (`lodSize` 1) are the cloud the tree was built from; the
// merges over them are not drawn with it. `treeLeaves` marks them, and
// `treeGather` copies a buffer's words of the marked ones into place.
module tree_leaves;

struct TreeParams {
    uint count;   ///< gaussians
    uint words;   ///< words a gaussian (treeGather)
};

StructuredBuffer<uint>    lodSize;    ///< a gaussian, as the splats were kept
RWStructuredBuffer<uint>  selected;   ///< treeLeaves writes, treeGather reads
StructuredBuffer<uint>    dest;       ///< the prefix sum of `selected`
StructuredBuffer<uint>    source;
RWStructuredBuffer<uint>  target;
ConstantBuffer<TreeParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void treeLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    selected[i] = lodSize[i] == 1u ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void treeGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || selected[i] == 0) {
        return;
    }
    const uint j = dest[i];
    for (uint w = 0; w < params.words; ++w) {
        target[j * params.words + w] = source[i * params.words + w];
    }
}
