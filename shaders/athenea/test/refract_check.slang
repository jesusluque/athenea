// Copyright (c) 2026 jesus luque.
//
// SNELL, WHERE A GAUSSIAN BENDS THE SKY.
//
// A cloud has no surface to refract at, so what `splat_relight` does is bend
// the direction its transmitted half looks along, once, at the gaussian's own
// normal. That bend is the only physics in it and it is worth checking on its
// own: for an index `n`, a direction meeting the normal at `theta_i` leaves at
// `theta_t` with `sin(theta_t) = sin(theta_i) / n`, on the far side and in the
// plane the two directions share.
//
// Below the critical angle there is nothing to check -- entering a denser
// medium there is no critical angle at all -- and an index of zero must give
// back the straight direction, which is what a cloud did before an index
// existed.
module refract_check;
import athenea.splat.splat_relight;
import athenea.common.packing;

struct RefractCheckParams {
    uint  angles;       ///< incidences tried between 0 and 89 degrees
    float ior;          ///< the index they are bent by
    float tolerance;    ///< absolute, on a sine
    uint  pad0;
};

RWStructuredBuffer<uint>  stats;   ///< 8 wide
RWStructuredBuffer<float> worst;   ///< 8 wide
ConstantBuffer<RefractCheckParams> params;

/// stats[0] angles whose sine is off; stats[1] angles tried
/// stats[2] angles that came back on the near side, which none may
/// stats[3] angles that left the plane of incidence
/// stats[4] 1 where an index of zero did not give the straight direction
/// worst[0] the worst deviation of the sine; worst[1] the first angle's
[shader("compute")]
[numthreads(1, 1, 1)]
void refractCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    SplatSurface s;
    s.lobes = plainLobes();
    s.thinWalled = false;
    s.hasShadowBits = false;
    s.cellWords = 0u;
    s.cellBase = 0u;
    s.normal = float3(0.0, 0.0, 1.0);
    s.ior = params.ior;
    float worstOff = 0.0;
    for (uint k = 0; k < params.angles; ++k) {
        // The view, at `theta_i` from the normal in the x-z plane: `wo` points
        // from the surface towards the eye, which is how a frame hands it over.
        const float theta = (float(k) + 0.5) / float(max(params.angles, 1u)) * 1.5533430;   // < 89 degrees
        const float3 wo = float3(sin(theta), 0.0, cos(theta));
        const float3 bent = bentThrough(s, wo);
        const float sineIn = sin(theta);
        const float sineOut = length(float2(bent.x, bent.y));
        const float want = sineIn / max(params.ior, 1.0);
        const float off = abs(sineOut - want);
        worstOff = max(worstOff, off);
        if (off > params.tolerance) {
            InterlockedAdd(stats[0], 1u);
        }
        if (!(bent.z < 0.0)) {
            InterlockedAdd(stats[2], 1u);   // it has to carry on through
        }
        if (abs(bent.y) > params.tolerance) {
            InterlockedAdd(stats[3], 1u);   // and stay in the plane it came in
        }
        InterlockedAdd(stats[1], 1u);
        if (k == 0) {
            worst[1] = sineOut;
            worst[2] = want;
        }
    }
    worst[0] = worstOff;
    // No index: the straight direction, which is what this answered before.
    SplatSurface none = s;
    none.ior = 0.0;
    const float3 straight = bentThrough(none, float3(0.3, 0.0, 0.954));
    if (length(straight + float3(0.3, 0.0, 0.954)) > params.tolerance) {
        InterlockedAdd(stats[4], 1u);
    }
}
