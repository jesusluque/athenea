// Copyright (c) 2026 jesus luque.
//
// WHAT THE CLOSED DIRECTIONS SHOW, TWO WAYS (task TX, step 2).
//
// `fieldCouplingCheck`: the reflected field and its coupling to a sky, on
// values written here. A field that is a constant `c` reads `c` along every
// direction at every roughness; the coupling of any indirect half to a white
// sky of radiance one is one, and to a sky twice as bright is two, whatever
// the indirect half is -- the frame is linear in the sky, and the field was
// baked under the white one.
//
// `fieldBakeCheck`: a bake's field against a closed form. A point on a black
// floor beside a grey wall of albedo `a`, under a white sky: the wall, a
// vertical surface whose lower half of hemisphere is the black floor, sends
// back `a / 2`, and that is what arrives from it; straight up is open sky,
// where the field holds nothing.
module field_check;
import athenea.splat.splat_relight;

struct FieldParams {
    uint  count;        ///< points (fieldBakeCheck)
    uint  entries;      ///< float4 a point
    uint  firstField;   ///< the first of the field's sixteen
    uint  steps;        ///< roughness and direction steps (fieldCouplingCheck)

    float wallRadiance; ///< what the wall sends back
    float tolerance;    ///< relative, for the wall
    float openFloor;    ///< the most the field may read straight up
    float roughness;    ///< the lobe the bake's field is read at

    float3 towardsWall;
    float  pad0;
};

RWStructuredBuffer<uint>   halves;   ///< a fake gaussian's transfer, f16 pairs
StructuredBuffer<uint>     transfer; ///< the same, read back
StructuredBuffer<float4>   baked;
RWStructuredBuffer<float4> skies;    ///< two skies of kEnvCoefficients each, written here
StructuredBuffer<float4>   envSh;    ///< the same, read back
RWStructuredBuffer<uint>   stats;    ///< 8 wide
RWStructuredBuffer<float>  worst;    ///< 8 wide
ConstantBuffer<FieldParams> params;

/// One thread: a gaussian of 84 values (degree 2, both halves, the field)
/// whose field is the constant 0.3 0.2 0.1 and whose indirect half is
/// anything; and two skies, white and twice white.
[shader("compute")]
[numthreads(1, 1, 1)]
void fieldFill(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float values[84];
    for (uint k = 0; k < 84u; ++k) {
        values[k] = 0.0;
    }
    for (uint k = 0; k < 9u; ++k) {
        values[k] = 0.1 * float(k + 1);                 // direct, anything
    }
    for (uint k = 0; k < 27u; ++k) {
        values[9 + k] = 0.05 + 0.01 * float(k % 7u);    // indirect, anything with a constant term
    }
    const float y0 = 3.5449077018110318;                // sqrt(4 pi): a constant onto Y_0
    values[36] = 0.3 * y0;
    values[37] = 0.2 * y0;
    values[38] = 0.1 * y0;
    for (uint w = 0; w < 42u; ++w) {
        halves[w] = (f32tof16(values[w * 2 + 1]) << 16) | f32tof16(values[w * 2]);
    }
    for (uint k = 0; k < kEnvCoefficients; ++k) {
        skies[k] = float4(k == 0 ? float3(y0) : float3(0.0), 0.0);
        skies[kEnvCoefficients + k] = float4(k == 0 ? float3(2.0 * y0) : float3(0.0), 0.0);
    }
}

/// stats[0] field readings off the constant
/// stats[1] couplings to the white sky off one
/// stats[2] couplings to the doubled sky off two
/// stats[3] readings taken
/// worst[0] the worst field reading, worst[1] the worst coupling
[shader("compute")]
[numthreads(1, 1, 1)]
void fieldCouplingCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    SplatSurface s;
    s.normal = float3(0.0, 0.0, 1.0);
    s.transferBase = 0;
    s.transferCount = 84;
    const float3 want = float3(0.3, 0.2, 0.1);
    float worstField = 0.0;
    float worstCoupling = 0.0;
    const float3 white1 = splatFieldCoupling(s, transfer, envSh, 0u, float4(0.0), float3(0.0));
    const float3 white2 = splatFieldCoupling(s, transfer, envSh, 1u, float4(0.0), float3(0.0));
    worstCoupling = max(worstCoupling, max(length(white1 - float3(1.0)), length(white2 - float3(2.0))));
    if (any(abs(white1 - float3(1.0)) > float3(1.0e-2))) {
        InterlockedAdd(stats[1], 1u);
    }
    if (any(abs(white2 - float3(2.0)) > float3(2.0e-2))) {
        InterlockedAdd(stats[2], 1u);
    }
    for (uint i = 0; i < params.steps; ++i) {
        const float roughness = float(i) / float(params.steps);
        for (uint j = 0; j < params.steps; ++j) {
            const float theta = 3.14159265 * (float(j) + 0.5) / float(params.steps);
            const float phi = 2.39996323 * float(j);
            const float3 d = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            const float3 got = splatFieldAlong(s, transfer, d, roughness);
            worstField = max(worstField, length(got - want));
            if (any(abs(got - want) > float3(2.0e-3))) {   // the field is f16
                InterlockedAdd(stats[0], 1u);
            }
            InterlockedAdd(stats[3], 1u);
        }
    }
    worst[0] = worstField;
    worst[1] = worstCoupling;
}

/// stats[0] points whose field toward the wall is off its radiance
/// stats[1] points whose field straight up reads more than `openFloor`
/// stats[2] points judged
/// stats[4] the worst relative error toward the wall, stats[5] the most read
/// up, both as the bits of a float (positive, so they order as uints)
[shader("compute")]
[numthreads(64, 1, 1)]
void fieldBakeCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    if (!(baked[k * params.entries + params.firstField - 3u].w > 0.0)) {   // the coverage
        return;
    }
    float3 wall = float3(0.0);
    float3 up = float3(0.0);
    const float alpha = params.roughness * params.roughness;
    for (uint j = 0; j < 16u; ++j) {
        const float3 c = baked[k * params.entries + params.firstField + j].rgb;
        const float band = j < 1u ? 0.0 : (j < 4u ? 2.0 : (j < 9u ? 6.0 : 12.0));
        const float g = exp(-band * alpha * alpha);
        wall += c * g * envBasisValue(j, normalize(params.towardsWall));
        up += c * g * envBasisValue(j, float3(0.0, 0.0, 1.0));
    }
    const float error = abs(dot(wall, float3(1.0 / 3.0)) - params.wallRadiance) / params.wallRadiance;
    if (error > params.tolerance) {
        InterlockedAdd(stats[0], 1u);
    }
    const float upRead = dot(up, float3(1.0 / 3.0));
    if (upRead > params.openFloor) {
        InterlockedAdd(stats[1], 1u);
    }
    InterlockedAdd(stats[2], 1u);
    uint dummy;
    InterlockedMax(stats[4], asuint(error), dummy);
    InterlockedMax(stats[5], asuint(max(upRead, 0.0)), dummy);
}
