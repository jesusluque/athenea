// Copyright (c) 2026 jesus luque.
//
// A DOME'S SKY AS A MESH'S SHADING KERNEL CAN TAKE IT: twelve texels a dome.
//
// The raster mesh shading kernel is at Metal's limit of thirty-one buffers,
// so the prepared sky (`Environment`'s harmonics and sun, which are buffers)
// reaches it as a small texture instead, the way the cloud shadow map does.
// A row a dome:
//
//   texels 0..8   the sky's radiance coefficients, degree 2 (rgb): what a
//                 diffuse surface's irradiance is read from, the sun left out
//   texel 9       the sun: its direction, and its solid angle (0: none)
//   texel 10      the sun's irradiance at normal incidence, and its cone
//   texel 11      the light the row belongs to, plus one (0: an empty row)
//
// Nothing is computed: the values are copied as `env_project` and `env_sun`
// wrote them, once, when the sky changes.
module env_mesh;
import athenea.light.environment;

public static const uint kEnvMeshTexels = 12;

struct EnvMeshParams {
    uint domes;       ///< rows written; the rest are cleared
    uint rows;        ///< the texture's height
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>       sh;           ///< kEnvCoefficients a dome
StructuredBuffer<float4>       sun;          ///< two a dome
StructuredBuffer<uint>         domeLights;   ///< slice -> its light
RWTexture2D<float4>            meshOut;
ConstantBuffer<EnvMeshParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void envMeshPack(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint row = at / kEnvMeshTexels;
    const uint texel = at % kEnvMeshTexels;
    if (row >= params.rows) {
        return;
    }
    float4 value = float4(0.0);
    if (row < params.domes) {
        if (texel < kEnvIrradianceCoefficients) {
            value = float4(sh[row * kEnvCoefficients + texel].rgb, 0.0);
        } else if (texel == 9) {
            value = sun[row * 2];
        } else if (texel == 10) {
            value = sun[row * 2 + 1];
        } else if (texel == 11) {
            value = float4(float(domeLights[row] + 1u), 0.0, 0.0, 0.0);
        }
    }
    meshOut[uint2(texel, row)] = value;
}
