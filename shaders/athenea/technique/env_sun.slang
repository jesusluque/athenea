// Copyright (c) 2026 jesus luque.
//
// THE SUN, TAKEN OUT OF THE SKY IT IS IN.
//
// Nine harmonics hold the irradiance of any environment to about a percent --
// as long as the environment has no sun in it. A half-degree disc a thousand
// times brighter than the sky around it is not a degree-2 function: projected,
// it smears into a lobe sixty degrees wide, and a pawn under a clear sky gets
// no terminator, no highlight and no shadow, only a soft lean towards the
// bright side. The percent is over the whole sphere; the error where it
// matters is total.
//
// So the disc is found and taken out. What is left is a residual sky, which
// really is smooth and really is held by nine numbers, and the disc comes back
// as what it always was: a directional light, with a cosine that is exact and
// a lobe that knows where the eye is.
//
// ONE THREAD A DOME, walking an octahedral grid of `kSunCells` a side twice.
// Four thousand samples a pass and at most four domes, run once when the sky
// changes: a reduction would need three dispatches and a barrier between each,
// and there is nothing here worth that. The grid rather than the image is what
// keeps the cost off the sky's resolution -- a 4k dome and an 8k one search
// the same 4096 directions.
module env_sun;
import athenea.light.lights_image;
import athenea.light.environment;

/// 4096 cells, about 1.8 degrees each: a sun of half a degree lands in one,
/// and an overcast lands in none.
public static const uint  kSunCells = 64;
/// Below this many times the sky's own mean the brightest cell is not a
/// source at all: an overcast has a brightest cell too.
public static const float kSunPeakRatio = 8.0;

/// HOW WIDE THE SOURCE IS, MEASURED RATHER THAN ASSUMED.
///
/// A fixed cone was tried at seven degrees and it is the wrong tool in both
/// directions. A captured sun is clipped and bloomed -- `kloppenheim_06_puresky`
/// is still four fifths as bright seven degrees off its centre -- so a fixed
/// cone cuts a slice out of the middle of it, which is the one thing that
/// must not happen: a ring left in the harmonics and a hard terminator where
/// a soft one belongs. Concentration was tried as the test and is the wrong
/// question too, since that same real sun puts only a fifth of its energy
/// inside three degrees while a small hard softbox puts nearly all of it
/// there.
///
/// So the source's own profile is measured. Rings of `kSunRing` about the
/// brightest cell, out to `kSunRings` of them, and the cone ends at the first
/// ring that has fallen back into the sky -- below `kSunEdge` of the middle's
/// brightness, or within `kSunSky` of the sky's own mean, whichever comes
/// first. A source still going at the last ring is wider than a degree-2
/// harmonic has trouble with and is left where it is.
/// The rings are measured against the sky *behind* the source, not against
/// zero: a low sun sits in a bright horizon, and `sunset_fairway`'s rings
/// never fall while the horizon holds them up. The outermost ring is that
/// background, and every ring has it taken off before the profile is read.
///
/// And nothing wider than `kSunWidest` is taken at all. Past ten degrees a
/// source is something nine coefficients hold well enough, and pulling it out
/// risks more than it buys -- `urban_alley_01` offered a twenty-four degree
/// wall, which is a sky, not a sun.
public static const float kSunRing   = 0.0349;   // radians, two degrees
public static const uint  kSunRings  = 16;       // out to thirty-two
public static const float kSunEdge   = 0.10;
public static const float kSunWidest = 0.175;    // radians, ten degrees

struct EnvSunParams {
    uint domes;
    uint level;        ///< the mip level the search reads
    uint projectLevel; ///< the level `env_project` sums over: the disc is measured there
    uint pad0;
};

StructuredBuffer<LightRecord> lights;
// `LightTable::bind` sets this beside the records; a name it writes must
// exist here even where the count is not what this kernel walks.
uniform uint                  lightCount;
StructuredBuffer<uint>        domeLights;
/// Two float4 a dome: (direction.xyz, the disc's solid angle) and
/// (irradiance.rgb, the sky's mean luminance). A solid angle of 0 says this
/// dome has no sun, and then nothing is taken out of its harmonics.
RWStructuredBuffer<float4>    sunOut;
ConstantBuffer<EnvSunParams>  params;

static const float kSunPi = 3.14159265358979;

/// THE SOLID ANGLE ONE CELL ACTUALLY COVERS.
///
/// The octahedral map is not equal-area: a cell on a diagonal covers about
/// twice what a cell at a pole does. Taking every cell as `4 pi / cells`
/// gathered a seven degree cone as 0.11 steradians where it is 0.047, and the
/// sun's irradiance -- which is a sum of radiance times this -- came out more
/// than twice what it should be. So it is measured, by the Jacobian of the
/// decode: the area the unit square's own cell maps to.
float cellSolidAngle(float2 uv) {
    const float h = 1.0e-3;
    const float3 du = octDecode(uv + float2(h, 0.0)) - octDecode(uv - float2(h, 0.0));
    const float3 dv = octDecode(uv + float2(0.0, h)) - octDecode(uv - float2(0.0, h));
    return length(cross(du, dv)) / (4.0 * h * h * float(kSunCells * kSunCells));
}

/// The sky along `dir`, at the level the caller chose.
float3 skyAlong(LightRecord light, float3 dir, uint level) {
    const TextureRecord record = gTextures.records[light.texture];
    return gTextures.textures[record.slot]
        .SampleLevel(gTextures.samplers[light.sampler], domeUv(light, dir), float(level))
        .rgb;
}

[shader("compute")]
[numthreads(4, 1, 1)]
void envSun(uint3 tid: SV_DispatchThreadID) {
    const uint dome = tid.x;
    if (dome >= params.domes) {
        return;
    }
    const LightRecord light = lights[domeLights[dome]];
    if (light.texture == 0xFFFFFFFFu) {
        // A dome with no image is a constant, and a constant has no sun.
        sunOut[dome * 2] = float4(0.0, 1.0, 0.0, 0.0);
        sunOut[dome * 2 + 1] = float4(0.0);
        return;
    }
    const uint cells = kSunCells * kSunCells;
    const float3 grey = float3(0.2126, 0.7152, 0.0722);
    const float3 emission = lightEmission(light);

    // The brightest cell, and the sky's mean luminance beside it -- the mean
    // over the sphere, so each cell counts for the solid angle it covers.
    float best = -1.0;
    uint bestCell = 0;
    float skySum = 0.0;
    float sphere = 0.0;
    for (uint c = 0; c < cells; ++c) {
        const float2 uv = (float2(float(c % kSunCells), float(c / kSunCells)) + 0.5) / float(kSunCells);
        const float area = cellSolidAngle(uv);
        const float luminance = dot(skyAlong(light, octDecode(uv), params.level) * emission, grey);
        skySum += luminance * area;
        sphere += area;
        if (luminance > best) {
            best = luminance;
            bestCell = c;
        }
    }
    const float skyMean = skySum / max(sphere, 1.0e-9);
    const float2 bestUv = (float2(float(bestCell % kSunCells), float(bestCell / kSunCells)) + 0.5) /
                          float(kSunCells);
    const float3 axis = octDecode(bestUv);

    // The source's profile: one ring of two degrees at a time about the
    // brightest cell, each ring's mean radiance and what it carries.
    float ringLum[kSunRings];
    float3 ringRgb[kSunRings];
    float ringArea[kSunRings];
    float3 ringDir[kSunRings];
    [unroll] for (uint r = 0; r < kSunRings; ++r) {
        ringLum[r] = 0.0;
        ringRgb[r] = float3(0.0);
        ringArea[r] = 0.0;
        ringDir[r] = float3(0.0);
    }
    const float reach = cos(kSunRing * float(kSunRings));
    for (uint c = 0; c < cells; ++c) {
        const float2 uv = (float2(float(c % kSunCells), float(c / kSunCells)) + 0.5) / float(kSunCells);
        const float3 dir = octDecode(uv);
        const float along = dot(dir, axis);
        if (along < reach) {
            continue;
        }
        const uint ring = min(uint(acos(clamp(along, -1.0, 1.0)) / kSunRing), kSunRings - 1);
        const float area = cellSolidAngle(uv);
        const float3 rgb = skyAlong(light, dir, params.level) * emission;
        const float lum = dot(rgb, grey);
        ringLum[ring] += lum * area;
        ringRgb[ring] += rgb * area;
        ringArea[ring] += area;
        ringDir[ring] += dir * lum;
    }
    // Where it ends. The outermost ring is the background the source sits in,
    // taken off every ring before the profile is read; the middle ring is the
    // source at its brightest above that background, and the cone ends at the
    // first ring that has fallen to a tenth of it.
    const float background = ringLum[kSunRings - 1] / max(ringArea[kSunRings - 1], 1.0e-9);
    const float middle = ringLum[0] / max(ringArea[0], 1.0e-9) - background;
    uint edge = kSunRings;
    for (uint r = 1; r < kSunRings; ++r) {
        const float mean = ringLum[r] / max(ringArea[r], 1.0e-9) - background;
        if (!(middle > 0.0) || mean < kSunEdge * middle) {
            edge = r;
            break;
        }
    }
    float3 irradiance = float3(0.0);
    float3 centre = float3(0.0);
    float cone = 0.0;
    for (uint r = 0; r < min(edge, kSunRings); ++r) {
        irradiance += ringRgb[r];
        centre += ringDir[r];
        cone += ringArea[r];
    }
    const float floor = max(skyMean, 1.0e-9);
    // A source still going at ten degrees is a sky, not a sun.
    const bool ends = float(edge) * kSunRing <= kSunWidest;
    const bool isSun = ends && best > kSunPeakRatio * floor && dot(centre, centre) > 0.0;
    const float3 direction = isSun ? normalize(centre) : axis;

    // WHAT THE DISC IS WORTH, IN THE PROJECTION'S OWN QUADRATURE.
    //
    // The rings above find the disc and how wide it is, and that is all they
    // are good for: a grid of 1.8 degree cells sampling a 1024 wide sky reads
    // a half degree sun at its peak and multiplies it by the whole cell,
    // which is eight times the energy that sun carries. The harmonics, mean-
    // while, lose exactly the lat-long texels `env_project` skips. Add back
    // one and remove the other and a grey ball under a chapel went from four
    // per cent bright to fourteen.
    //
    // So the irradiance is summed here the way the projection sums, over the
    // same texels of the same level with the same measure -- `2 pi^2 sin(theta)
    // du dv` -- so that what leaves the coefficients is what arrives as a
    // light, to the last texel.
    float3 delivered = float3(0.0);
    if (isSun) {
        const TextureRecord record = gTextures.records[light.texture];
        const uint w = max(uint(record.width) >> params.projectLevel, 1u);
        const uint h = max(uint(record.height) >> params.projectLevel, 1u);
        const float area = 2.0 * kSunPi * kSunPi / float(w * h);
        const float cosCone = cos(float(edge) * kSunRing);
        for (uint y = 0; y < h; ++y) {
            const float v = (float(y) + 0.5) / float(h);
            const float sinTheta = sin((1.0 - v) * kSunPi);
            if (sinTheta <= 0.0) {
                continue;
            }
            for (uint x = 0; x < w; ++x) {
                const float u = (float(x) + 0.5) / float(w);
                const float3 dir = domeDirection(light, float2(u, v));
                if (dot(dir, direction) < cosCone) {
                    continue;
                }
                const float3 rgb =
                    gTextures.textures[record.slot]
                        .Load(int3(int(x), int(y), int(params.projectLevel)))
                        .rgb;
                delivered += rgb * (sinTheta * area);
            }
        }
        delivered *= emission;
    }
    sunOut[dome * 2] = float4(direction, isSun ? cone : 0.0);
    // How wide the source turned out to be, in degrees, rides back in the
    // second row's w: it is how the log says what was found, or why a sky was
    // left alone.
    sunOut[dome * 2 + 1] = float4(delivered, float(edge) * kSunRing);
}
