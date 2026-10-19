// Copyright (c) 2026 jesus luque.
//
// THE SKY'S HARMONICS IN A CLOUD'S OWN FRAME, for a cloud the stage turned.
//
// A transfer (all but a zonal one) was measured in the frame of the file it
// was baked in: its harmonics, its reflected field and its open cells are
// directions of the cloud, not of the world. Under a prim that turns the
// cloud (a Y-up asset in a Z-up scene, a part rotated into place), reading
// them against the world's sky read a different sky into every gaussian, and
// a glass head that shows the room through its field came out in fragments.
// The directions a gaussian looks along are turned into the cloud's frame in
// the shader; the sky's sixteen coefficients are turned once a cloud here.
//
// `turned` is what the projection reads: the frame's sky as it is first
// (`domes * 16` entries, for what is evaluated in the world), then the same
// domes in the cloud's frame. A coefficient k of the turned sky is
//     c'_k = integral of Y_k(w) L(R w) dw,
// L the world's sky and R the cloud's rotation into the world, by a product
// rule exact for the degree-6 products of two degree-3 functions: four
// Gauss-Legendre nodes in z (exact to degree 7) times eight even steps in
// phi (exact to |m| = 7). One thread an output coefficient of one dome.
module sky_turn;

import athenea.light.environment;

struct SkyTurnParams {
    uint   domes;
    uint   pad0; uint pad1; uint pad2;
    float4 rotation0;   // R row by row (the cloud's axes in the world, as columns): world = R cloud
    float4 rotation1;
    float4 rotation2;
};

StructuredBuffer<float4>   sky;
RWStructuredBuffer<float4> turned;
ConstantBuffer<SkyTurnParams> params;

static const float kNodeZ[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
static const float kNodeW[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};

[shader("compute")]
[numthreads(64, 1, 1)]
void skyTurn(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint coefficients = kEnvCoefficients;
    if (at >= params.domes * coefficients * 2u) {
        return;
    }
    if (at < params.domes * coefficients) {
        turned[at] = sky[at];   // the world's, as it was
        return;
    }
    const uint local = at - params.domes * coefficients;
    const uint dome = local / coefficients;
    const uint k = local % coefficients;
    float4 sum = float4(0.0);
    for (uint i = 0; i < 4u; ++i) {
        const float z = kNodeZ[i];
        const float rho = sqrt(max(1.0 - z * z, 0.0));
        for (uint j = 0; j < 8u; ++j) {
            const float phi = float(j) * (2.0 * 3.14159265358979 / 8.0);
            const float3 w = float3(rho * cos(phi), rho * sin(phi), z);
            const float3 inWorld =
                float3(dot(params.rotation0.xyz, w), dot(params.rotation1.xyz, w), dot(params.rotation2.xyz, w));
            float4 radiance = float4(0.0);
            for (uint c = 0; c < coefficients; ++c) {
                radiance += sky[dome * coefficients + c] * envBasisValue(c, inWorld);
            }
            sum += radiance * (envBasisValue(k, w) * kNodeW[i] * (2.0 * 3.14159265358979 / 8.0));
        }
    }
    turned[at] = sum;
}
