// Copyright (c) 2026 jesus luque.
//
// A DOME'S SKY WITHOUT ITS SUN, for the prefilter to read (task TX, s90-s91).
//
// The split sum -- the sky prefiltered with the eye along the normal, times
// the lobe's directional albedo -- is right for a sky that changes slowly
// across a lobe and a quarter short for a compact sun. So the prefiltered map
// is the sky without its sun, and the reflections take the sun analytically
// (splat_relight's sunReflectPdf). Taking the sun out of each prefilter read
// by a model of the mip footprint left 60% of it in the map (s91: a cone of
// the footprint's solid angle against the disc is not what a bilinear read of
// two box-filtered levels holds). Here it is taken out where it is, texel by
// texel: a lat-long of the source at a level no wider than `width`, its sun
// texels -- the ones `env_sun` and `env_project` cut, as bright as the
// threshold the sun was found at and within `kSunWidest` of it -- set to
// nothing, as the harmonics have them. Its own mips (TextureStore's lat-long
// weighting) are then what every read of the prefilter samples.
module env_residual;
import athenea.light.lights_image;
import athenea.light.environment;

/// As `env_sun` cuts the sun (its kSunGrey and kSunWidest): repeated rather
/// than imported, since that module's own resources would come with it.
static const float3 kResidualGrey = float3(0.2126, 0.7152, 0.0722);
static const float  kResidualWidest = 0.175;   // radians, ten degrees

struct EnvResidualParams {
    uint dome;
    uint sourceLevel;   ///< the source's level this lat-long is a copy of
    uint width;
    uint height;
};

StructuredBuffer<LightRecord>    lights;
// `LightTable::bind` sets this beside the records; a name it writes must
// exist here even where the count is not what this kernel walks.
uniform uint                     lightCount;
StructuredBuffer<uint>           domeLights;
StructuredBuffer<float4>         sun;        ///< env_sun's: (direction, half angle), (irradiance, threshold)
RWTexture2D<float4>              residualOut;
ConstantBuffer<EnvResidualParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void envResidual(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const LightRecord light = lights[domeLights[params.dome]];
    if (light.texture == 0xFFFFFFFFu) {
        residualOut[tid.xy] = float4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    const TextureRecord record = gTextures.records[light.texture];
    const float3 rgb =
        gTextures.textures[record.slot].Load(int3(int(tid.x), int(tid.y), int(params.sourceLevel))).rgb;
    const float4 disc = sun[params.dome * 2];
    bool sunlit = false;
    if (disc.w > 0.0) {
        const float2 uv = (float2(tid.xy) + 0.5) / float2(float(params.width), float(params.height));
        const float3 dir = domeDirection(light, uv);
        sunlit = dot(rgb, kResidualGrey) >= sun[params.dome * 2 + 1].w && dot(dir, disc.xyz) >= cos(kResidualWidest);
    }
    residualOut[tid.xy] = float4(sunlit ? float3(0.0) : rgb, 1.0);
}
