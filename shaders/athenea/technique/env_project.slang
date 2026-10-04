// Copyright (c) 2026 jesus luque.
//
// A DOME'S SKY ONTO NINE COEFFICIENTS.
//
// What a diffuse surface takes from an environment is its irradiance, and the
// irradiance of any environment is held to about a percent by nine harmonics
// (Ramamoorthi and Hanrahan 2001) -- so a kernel that cannot sample the sky
// can still be lit by it, which is the whole point: the splat projection
// lives below `material` in the module order and has no texture table.
//
// The projection is a sum over the texels of one level of the image's own mip
// chain, each weighed by the solid angle it covers. A lat-long texel covers
// `2 pi^2 sin(theta) du dv`, which is the same measure the dome's importance
// sampling uses, so the two agree by construction.
//
// One thread a coefficient a dome: nine threads walking a few thousand texels
// each, once, when the sky changes. A reduction would be faster and there is
// nothing here to make faster.
module env_project;
import athenea.light.lights_image;
import athenea.light.environment;
import env_sun;

struct EnvProjectParams {
    uint domes;        ///< how many domes are prepared
    uint level;        ///< the mip level of the source read
    uint pad0;
    uint pad1;
};

StructuredBuffer<LightRecord> lights;
// `LightTable::bind` sets this beside the records; a name it writes must
// exist here even where the count is not what this kernel walks.
uniform uint                  lightCount;
StructuredBuffer<uint>        domeLights;   ///< slice -> the light it belongs to
/// What `env_sun` took out: two float4 a dome, the second of which is the
/// disc's irradiance and the first its direction and solid angle. A solid
/// angle of 0 means nothing was taken and this projects the whole sky.
StructuredBuffer<float4>      sun;
RWStructuredBuffer<float4>    shOut;        ///< kEnvCoefficients a dome, rgb used
ConstantBuffer<EnvProjectParams> params;

static const float kTwoPiSquared = 2.0 * 3.14159265358979 * 3.14159265358979;

[shader("compute")]
[numthreads(64, 1, 1)]
void envProject(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    const uint dome = at / kEnvCoefficients;
    const uint k = at % kEnvCoefficients;
    if (dome >= params.domes) {
        return;
    }
    const LightRecord light = lights[domeLights[dome]];
    const float3 emission = lightEmission(light);
    float3 sum = float3(0.0);
    if (light.texture != 0xFFFFFFFFu) {
        const TextureRecord record = gTextures.records[light.texture];
        const uint level = params.level;
        const uint w = max(uint(record.width) >> level, 1u);
        const uint h = max(uint(record.height) >> level, 1u);
        // The measure of one texel, with the sine that a lat-long owes the
        // poles: without it the top and bottom rows, which are a needle of
        // sky each, would count as much as the horizon.
        const float area = kTwoPiSquared / float(w * h);
        // WHAT THE SUN TOOK WITH IT. The disc is lit as a light now, so the
        // harmonics must not hold it too: the texels `env_sun` took for it are
        // skipped -- the same texels, decided by the same `sunRegionOf` at the
        // same level -- and what is left is a residual sky, which is the
        // smooth thing nine numbers were always able to hold. Skipped rather
        // than subtracted, because a texel of a sun is a thousand times its
        // neighbour and the difference of two such numbers is noise.
        const bool cutSun = sun[dome * 2].w > 0.0;
        const SunRegion region = sunRegionOf(light, level);
        for (uint y = 0; y < h; ++y) {
            const float v = (float(y) + 0.5) / float(h);
            const float sinTheta = sin((1.0 - v) * 3.14159265358979);
            if (sinTheta <= 0.0) {
                continue;
            }
            for (uint x = 0; x < w; ++x) {
                const float u = (float(x) + 0.5) / float(w);
                const float3 dir = domeDirection(light, float2(u, v));
                const float3 rgb = gTextures.textures[record.slot].Load(int3(int(x), int(y), int(level))).rgb;
                if (cutSun && sunHolds(region, dir, rgb)) {
                    continue;
                }
                sum += rgb * (envBasisValue(k, dir) * sinTheta * area);
            }
        }
        sum *= emission;
    } else if (k == 0) {
        // A dome with no image is its colour in every direction, and a
        // constant projects onto the first harmonic alone. Written in closed
        // form so that an imageless dome reads exactly what it read before
        // this kernel existed: irradiance pi * L at every normal.
        sum = emission * (4.0 * 3.14159265358979 * envBasisValue(0, float3(0.0, 1.0, 0.0)));
    }
    shOut[dome * kEnvCoefficients + k] = float4(sum, 0.0);
}
