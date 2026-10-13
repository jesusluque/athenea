// Copyright (c) 2026 jesus luque.
//
// A BAKED TRANSFER, AGAINST THE ONE CASE THAT HAS A CLOSED FORM.
//
// A point with nothing around it transfers the whole hemisphere its normal
// faces, and the projection of a clamped cosine lobe onto this basis is known
// exactly: its bands are pi, 2pi/3 and pi/4, so
//
//     T_k = A_l * Y_k(n) / pi
//
// which for a normal along +z is 0.28209 at the constant term, 0.32573 at the
// z term of the first band and 0.15770 at the 3z^2-1 term of the second, and
// zero everywhere else. A transfer that is wrong in its measure, in its
// stratification or in the weight it gives the cosine reads something else
// here, and nothing about the light it was baked under can hide it: a
// transfer does not gather light at all.
//
// The second thing checked is the one the whole idea rests on. A white
// Lambert surface under a sky of constant radiance L reflects exactly L, so
// the transfer dotted with that sky's own harmonics must come back as one.
module transfer_check;
import athenea.common.sh;
import athenea.common.color;

struct TransferCheckParams {
    uint  count;         ///< points
    uint  entries;       ///< float4 a point: coefficients, the coverage, then the open directions
    uint  coefficients;  ///< of those, how many are coefficients
    float tolerance;     ///< absolute, on a transfer of order one

    float3 normal;       ///< the surface every point stands on
    float skyRadiance;   ///< the constant sky the dot product is taken against
};

StructuredBuffer<float4>     baked;
RWStructuredBuffer<uint>     stats;   ///< 8 wide
RWStructuredBuffer<float>    worst;   ///< 8 wide
ConstantBuffer<TransferCheckParams> params;

/// The bands of the clamped cosine lobe (Ramamoorthi and Hanrahan).
// The clamped cosine's band 3 is zero, as every odd band past the first is.
float bandWeight(uint k) {
    return k == 0 ? 3.14159265358979 : (k < 4 ? 2.0943951023932 : (k < 9 ? 0.78539816339745 : 0.0));
}

/// stats[0] points whose direct transfer is beyond the tolerance
/// stats[1] points checked
/// stats[2] points whose sky product is beyond the tolerance
/// stats[3] points the bake found nothing under
/// worst[0] the worst deviation of a coefficient
/// worst[1] the worst deviation of the sky product
/// worst[2..4] the first point's first three direct coefficients
/// worst[5] the first point's sky product
[shader("compute")]
[numthreads(64, 1, 1)]
void transferCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint base = k * params.entries;
    // A point the bake found no surface under is not a point with no
    // transfer: it is counted apart.
    // The coverage follows the coefficients, whatever follows it.
    if (!(baked[base + params.coefficients].w > 0.0)) {
        InterlockedAdd(stats[3], 1u);
        return;
    }
    const float kPi = 3.14159265358979;
    float worstCoefficient = 0.0;
    // A constant sky of radiance L projects onto the first harmonic alone,
    // with `L * 4 pi * Y_0`; every other coefficient of it is zero.
    float product = 0.0;
    for (uint c = 0; c < params.coefficients; ++c) {
        const float direct = baked[base + c].w;
        const float want = bandWeight(c) * shBasisValue(c, params.normal) / kPi;
        worstCoefficient = max(worstCoefficient, abs(direct - want));
    }
    // The dot product, written plainly: only the constant term of a constant
    // sky is not zero, and its coefficient is `L * 4 pi * Y_0`.
    product = baked[base].w * params.skyRadiance * 4.0 * kPi * 0.28209479177387814;
    if (k == 0) {
        worst[2] = baked[base].w;
        worst[3] = params.coefficients > 2 ? baked[base + 2].w : 0.0;
        worst[4] = params.coefficients > 6 ? baked[base + 6].w : 0.0;
        worst[5] = product;
    }
    if (worstCoefficient > params.tolerance) {
        InterlockedAdd(stats[0], 1u);
    }
    if (abs(product - params.skyRadiance) > params.tolerance) {
        InterlockedAdd(stats[2], 1u);
    }
    InterlockedAdd(stats[1], 1u);
    InterlockedMax(stats[4], uint(worstCoefficient * 1.0e6));
    InterlockedMax(stats[5], uint(abs(product - params.skyRadiance) * 1.0e6));
}


/// A TRANSFER AGAINST A BAKE OF THE SAME PLACE.
///
/// The two keep different things -- one the light that arrived, the other how
/// much of an environment would -- so what can be compared is their product
/// with a sky. Under a sky of constant radiance L, whose only coefficient is
/// `L * 4 pi * Y0`, a surface of albedo `a` reads `a * L` either way: the
/// bake because that is what it measured, the transfer because that is what
/// it is for.
///
/// The bake's value is read exactly as a cloud reads it: the constant term is
/// stored shifted (`colour = 0.5 + kSH0 * dc`), in linear light.
struct TransferBakeParams {
    uint  count;
    uint  entries;       ///< float4 a point in `baked`; the coverage is the one before last
    float albedo;        ///< of the surface both were measured on
    float skyRadiance;

    float tolerance;     ///< relative
    uint  withIndirect;  ///< 1: the indirect half is added as well
    uint  pad0; uint pad1;
};

StructuredBuffer<float4>     radianceBaked;   ///< one float4 a point: the fitted constant term
ConstantBuffer<TransferBakeParams> bakeParams;

/// stats[0] points where the two disagree beyond the tolerance
/// stats[1] points compared
/// worst[0] the worst relative difference; worst[1] the transfer's reading,
/// worst[2] the bake's, at the first point compared
[shader("compute")]
[numthreads(64, 1, 1)]
void transferAgainstBake(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= bakeParams.count) {
        return;
    }
    const uint base = k * bakeParams.entries;
    if (!(baked[base + bakeParams.entries - 2].w > 0.0) || !(radianceBaked[k].w > 0.0)) {
        return;   // a point one of them found nothing under
    }
    const float kPi = 3.14159265358979;
    const float skyFirst = bakeParams.skyRadiance * 4.0 * kPi * 0.28209479177387814;
    // Only the first coefficient of a constant sky is not zero, so the dot
    // product is one term of each half.
    float3 recombined = float3(baked[base].w * skyFirst);
    if (bakeParams.withIndirect != 0) {
        recombined += baked[base].rgb * skyFirst;
    }
    recombined *= bakeParams.albedo;
    // The bake, read as a cloud reads it: linear light.
    const float3 shifted = float3(0.5) + float3(0.28209479177387814) * radianceBaked[k].rgb;
    const float3 fromBake = max(shifted, float3(0.0));
    const float a = (recombined.r + recombined.g + recombined.b) / 3.0;
    const float b = (fromBake.r + fromBake.g + fromBake.b) / 3.0;
    const float relative = abs(a - b) / max(b, 1.0e-6);
    if (k == 0) {
        worst[1] = a;
        worst[2] = b;
    }
    if (relative > bakeParams.tolerance) {
        InterlockedAdd(stats[0], 1u);
    }
    InterlockedAdd(stats[1], 1u);
    InterlockedMax(stats[6], uint(relative * 1.0e6));
}
