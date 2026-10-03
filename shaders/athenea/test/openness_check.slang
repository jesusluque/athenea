// Copyright (c) 2026 jesus luque.
//
// THE OPENING A REFLECTION IS ALLOWED, WHERE NO RAY MEASURES IT.
//
// A gaussian's reflection was added with nothing in front of it, and the
// pawn's marble disc -- whose stem stands over it -- read a tenth bright
// against the mesh beside it. What closes it is the transfer the cloud
// already carries: its constant term is the visible fraction of the sky, and
// Lagarde and de Rousiers' fit turns that into how much of it a lobe of a
// given sharpness may see.
//
// The fit is an approximation and nothing here pretends otherwise. What is
// checked is what it must never get wrong: an open hemisphere lets the whole
// reflection through, a closed one lets none, more opening is never less
// reflection, and nothing ever leaves more than it was given.
module openness_check;
import athenea.splat.splat_relight;

struct OpennessParams {
    uint  steps;       ///< the grid is steps x steps x steps
    float tolerance;
    uint  pad0; uint pad1;
};

RWStructuredBuffer<uint>  stats;   ///< 8 wide
RWStructuredBuffer<float> worst;   ///< 8 wide
ConstantBuffer<OpennessParams> params;

/// stats[0] where an open sky did not pass the whole reflection
/// stats[1] where a closed one passed any
/// stats[2] where more opening gave less reflection
/// stats[3] where the answer left [0, 1]
/// stats[4] readings taken
/// worst[0] the worst deviation at an open sky
[shader("compute")]
[numthreads(1, 1, 1)]
void opennessCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float worstOpen = 0.0;
    for (uint i = 0; i < params.steps; ++i) {
        const float roughness = (float(i) + 0.5) / float(params.steps);
        for (uint j = 0; j < params.steps; ++j) {
            const float nDotV = (float(j) + 0.5) / float(params.steps);
            const float open = specularOpenness(nDotV, 1.0, roughness);
            const float shut = specularOpenness(nDotV, 0.0, roughness);
            worstOpen = max(worstOpen, abs(open - 1.0));
            if (abs(open - 1.0) > params.tolerance) {
                InterlockedAdd(stats[0], 1u);
            }
            if (shut > params.tolerance) {
                InterlockedAdd(stats[1], 1u);
            }
            float previous = -1.0;
            for (uint k = 0; k <= params.steps; ++k) {
                const float openness = float(k) / float(params.steps);
                const float value = specularOpenness(nDotV, openness, roughness);
                if (value < previous - params.tolerance) {
                    InterlockedAdd(stats[2], 1u);
                }
                if (value < -params.tolerance || value > 1.0 + params.tolerance) {
                    InterlockedAdd(stats[3], 1u);
                }
                previous = value;
                InterlockedAdd(stats[4], 1u);
            }
        }
    }
    worst[0] = worstOpen;
}
