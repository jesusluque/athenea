// Copyright (c) 2026 jesus luque.
//
// What a lift (athenea/splat/splat_lift.slang) told the gaussians, counted:
// the slots below `firstOutside` are the prim the mask was of, the rest are
// not, and a gaussian a camera saw should have been told which it is.
module lift_check;

struct LiftCheckParams {
    uint slots;
    uint firstOutside;
    uint pad0; uint pad1;
};

StructuredBuffer<uint>            lift;    // two a slot: weight * mask, weight
RWStructuredBuffer<uint>          stats;   // 8 wide, one row
ConstantBuffer<LiftCheckParams>   params;

/// stats[0] gaussians of the prim a lifted camera saw, stats[1] of those told they are mostly outside
/// stats[2] gaussians not of the prim a lifted camera saw, stats[3] of those told they are mostly inside
/// stats[4] gaussians no lifted camera saw
[shader("compute")]
[numthreads(64, 1, 1)]
void liftCheck(uint3 tid: SV_DispatchThreadID) {
    const uint slot = tid.x;
    if (slot >= params.slots) {
        return;
    }
    const uint inside = lift[slot * 2];
    const uint seen = lift[slot * 2 + 1];
    if (seen == 0u) {
        InterlockedAdd(stats[4], 1u);
        return;
    }
    const bool mostly = inside * 2u > seen;
    if (slot < params.firstOutside) {
        InterlockedAdd(stats[0], 1u);
        if (!mostly) {
            InterlockedAdd(stats[1], 1u);
        }
    } else {
        InterlockedAdd(stats[2], 1u);
        if (mostly) {
            InterlockedAdd(stats[3], 1u);
        }
    }
}
