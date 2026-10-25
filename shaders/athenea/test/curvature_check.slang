// Copyright (c) 2026 jesus luque.
//
// A SHAPE OPERATOR THROUGH ITS PACKED WORD AND BACK (common/packing.slang's
// packCurvature): how many components come back off by more than half a step
// (16/127 of an octave, 4.5%), with the wrong sign, or not flat where they lie
// more than 16 octaves under the largest. What the test reads back.
module curvature_check;
import athenea.common.packing;

struct CurvatureCheckParams {
    uint count;   ///< gaussians, three floats each
};

StructuredBuffer<float>  shapes;
RWStructuredBuffer<uint> wrong;   ///< [0] off by more than half a step, [1] sign, [2] not flat, [3] read
ConstantBuffer<CurvatureCheckParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void curvatureCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float3 k = float3(shapes[i * 3], shapes[i * 3 + 1], shapes[i * 3 + 2]);
    const float3 back = unpackCurvature(packCurvature(k));
    const float largest = max(abs(k.x), max(abs(k.y), abs(k.z)));
    const float floorSize = exp2(ceil(log2(largest)) - kCurvatureOctaves);
    [unroll] for (uint c = 0; c < 3; ++c) {
        const float a = abs(k[c]);
        if (a > 1.0e-6 && a > floorSize * 1.05) {
            if (abs(abs(back[c]) / a - 1.0) > 0.046) {
                InterlockedAdd(wrong[0], 1u);
            }
            if ((back[c] < 0.0) != (k[c] < 0.0)) {
                InterlockedAdd(wrong[1], 1u);
            }
        } else if (a < floorSize * 0.95 && back[c] != 0.0) {
            InterlockedAdd(wrong[2], 1u);
        }
        InterlockedAdd(wrong[3], 1u);
    }
}

/// AND A CARRIED ARRAY OF THEM: how many words are neither `reference`'s nor
/// zero (a merged group of the levels of detail is flat).
StructuredBuffer<uint> words;
StructuredBuffer<uint> reference;

[shader("compute")]
[numthreads(64, 1, 1)]
void curvatureWordsCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    if (words[i] != 0u && words[i] != reference[i]) {
        InterlockedAdd(wrong[0], 1u);
    }
    InterlockedAdd(wrong[3], 1u);
}
