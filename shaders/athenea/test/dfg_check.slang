// Copyright (c) 2026 jesus luque.
//
// THE SAME SURFACE, ASKED THE SAME QUESTION, BY THE TWO SIDES OF THE ENGINE.
//
// How much of an environment a GGX lobe sends back -- the DFG term of the
// split sum -- is fitted twice in this tree: Lazarov's fit in
// `splat_relight.slang`, which is what a gaussian reflects with, and
// MaterialX's `ggxDirectionalAlbedo` in `microfacet.slang`, which is what a
// surface reflects with. A cloud and the mesh beside it under one sky have to
// answer alike, and nothing held them to each other until this: the two are
// parameterised differently (one by roughness, the other by alpha) and a fit
// is only as good as where it was fitted.
module dfg_check;
import athenea.splat.splat_relight;
import athenea.material.microfacet;

struct DfgCheckParams {
    uint  steps;        ///< the grid is steps x steps over roughness and N.V
    float f0;           ///< the reflectivity both are asked about
    float tolerance;    ///< relative, on the larger of the two
    uint  pad0;
};

RWStructuredBuffer<uint>  stats;   ///< 8 wide
RWStructuredBuffer<float> worst;   ///< 8 wide
ConstantBuffer<DfgCheckParams> params;

/// stats[0] points beyond the tolerance; stats[1] points compared
/// stats[2] the worst relative difference, scaled by 1e6
/// worst[0..1] the two readings where it was worst; worst[2..3] where that was
[shader("compute")]
[numthreads(1, 1, 1)]
void dfgCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float worstRelative = 0.0;
    for (uint i = 0; i < params.steps; ++i) {
        // Away from the two ends: a fit is worst where it was least
        // constrained, and neither grazing nor a mirror is what a frame
        // spends its pixels on.
        const float roughness = (float(i) + 0.5) / float(params.steps);
        for (uint j = 0; j < params.steps; ++j) {
            const float nDotV = (float(j) + 0.5) / float(params.steps);
            // The cloud's, for a surface that is not metal at all: its
            // dielectric half is all there is.
            const float3 splat = environmentBrdf(float3(params.f0), 0.0, roughness, nDotV);
            // The surface's, as `lobes.slang` answers it for `kLobeDielectric`:
            // the fit at `alpha`, scaled by what multiple scattering adds.
            const float alpha = max(roughness * roughness, 1.0e-4);
            const float3 F = float3(fresnelDielectric(nDotV, 1.5));
            const float3 surface = ggxDirectionalAlbedo(nDotV, alpha, float3(0.04), float3(1.0)) *
                                   ggxEnergyCompensation(nDotV, alpha, F);
            const float a = splat.x;
            const float b = surface.x;
            const float relative = abs(a - b) / max(max(a, b), 1.0e-4);
            if (relative > worstRelative) {
                worstRelative = relative;
                worst[0] = a;
                worst[1] = b;
                worst[2] = roughness;
                worst[3] = nDotV;
            }
            if (relative > params.tolerance) {
                InterlockedAdd(stats[0], 1u);
            }
            InterlockedAdd(stats[1], 1u);
        }
    }
    InterlockedMax(stats[2], uint(worstRelative * 1.0e6));
}
