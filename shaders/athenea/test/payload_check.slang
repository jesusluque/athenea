// Copyright (c) 2026 jesus luque.
//
// A CUT'S PAYLOAD READ THROUGH ITS INDEX AGAINST THE SAME CUT'S PAYLOAD
// GATHERED (lod::joinPayload, GpuSplats::payloadIndex): how many words differ.
// What the test reads back.
module payload_check;

struct PayloadCheckParams {
    uint count;   ///< gaussians in the cut
    uint words;   ///< words a gaussian
};

StructuredBuffer<uint>   gathered;   ///< count * words, the cut's own
StructuredBuffer<uint>   joined;     ///< the joined payload
StructuredBuffer<uint>   index;      ///< count, where each gaussian's is in `joined`
RWStructuredBuffer<uint> wrong;      ///< [0] words off, [1] words read
ConstantBuffer<PayloadCheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void payloadCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint at = index[i];
    for (uint w = 0; w < params.words; ++w) {
        if (gathered[i * params.words + w] != joined[at * params.words + w]) {
            InterlockedAdd(wrong[0], 1u);
        }
        InterlockedAdd(wrong[1], 1u);
    }
}
