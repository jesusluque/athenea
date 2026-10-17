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
// ONE THREAD A DOME, walking the texels the harmonics are projected from
// (`projectLevel`, at most 256 wide) twice: once for the sky's median and its
// brightest texel, once to sum the texels that are the sun. Run once when the
// sky changes. The sun is those texels -- brighter than kSunOverMedian times
// the median and kSunOfPeak of the peak, within kSunWidest of the peak --
// which `env_project` skips by the same test, so the two cannot disagree.
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
/// Two float4 a dome: (direction.xyz, the half angle of the cone its texels
/// fill) and (irradiance.rgb, the luminance its texels were cut at). A half
/// angle of 0 says this dome has no sun, and then nothing is taken out of its
/// harmonics.
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


public static const float3 kSunGrey = float3(0.2126, 0.7152, 0.0722);
/// A texel is the sun where it is this many times the sky's median...
public static const float kSunOverMedian = 16.0;
/// ...and at least this share of the peak, so the aureole stays in the sky.
public static const float kSunOfPeak = 0.02;
/// And a sky whose peak is not this many times its median has no sun.
public static const float kSunPeakOverMedian = 32.0;

/// WHERE A DOME'S SUN IS, as `env_sun` and `env_project` both decide it: the
/// texels of `level` brighter than a threshold set by the sky's median (a
/// histogram of log luminance over the projection's measure, robust to the
/// sun itself, which the mean is not), within kSunWidest of the brightest.
public struct SunRegion {
    public bool   found;
    public float3 axis;        ///< the brightest texel's direction
    public float  threshold;   ///< luminance at and above which a texel is the sun
};

public SunRegion sunRegionOf(LightRecord light, uint level) {
    SunRegion region;
    region.found = false;
    region.axis = float3(0.0, 1.0, 0.0);
    region.threshold = 0.0;
    if (light.texture == 0xFFFFFFFFu) {
        return region;
    }
    const TextureRecord record = gTextures.records[light.texture];
    const uint w = max(uint(record.width) >> level, 1u);
    const uint h = max(uint(record.height) >> level, 1u);
    // log2 luminance from 2^-20 to 2^20 in half stops.
    float histogram[80];
    for (uint b = 0; b < 80; ++b) {
        histogram[b] = 0.0;
    }
    float total = 0.0;
    float peak = -1.0;
    uint2 peakAt = uint2(0);
    for (uint y = 0; y < h; ++y) {
        const float v = (float(y) + 0.5) / float(h);
        const float weight = sin((1.0 - v) * kSunPi);
        if (weight <= 0.0) {
            continue;
        }
        for (uint x = 0; x < w; ++x) {
            const float lum =
                dot(gTextures.textures[record.slot].Load(int3(int(x), int(y), int(level))).rgb, kSunGrey);
            const uint bin = uint(clamp((log2(max(lum, 1.0e-9)) + 20.0) * 2.0, 0.0, 79.0));
            histogram[bin] += weight;
            total += weight;
            if (lum > peak) {
                peak = lum;
                peakAt = uint2(x, y);
            }
        }
    }
    float below = 0.0;
    float median = 0.0;
    for (uint b = 0; b < 80; ++b) {
        below += histogram[b];
        if (below >= 0.5 * total) {
            median = exp2(float(b) * 0.5 - 20.0 + 0.25);
            break;
        }
    }
    region.axis = domeDirection(light, (float2(peakAt) + 0.5) / float2(float(w), float(h)));
    region.threshold = max(kSunOverMedian * median, kSunOfPeak * peak);
    region.found = peak > kSunPeakOverMedian * max(median, 1.0e-9);
    return region;
}

/// Whether a texel along `dir` of luminance `rgb` is the sun of `region`.
public bool sunHolds(SunRegion region, float3 dir, float3 rgb) {
    return region.found && dot(rgb, kSunGrey) >= region.threshold && dot(dir, region.axis) >= cos(kSunWidest);
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
    // THE SUN AS THE TEXELS THAT ARE IT (task PLAY-G): found, summed and
    // taken out in one quadrature, the projection's own -- the texels of
    // `projectLevel` with `2 pi^2 sin(theta) du dv` each -- so that what
    // leaves the harmonics is exactly what arrives as the light, whatever the
    // sun's size. The rings this replaced measured the disc on one grid and
    // summed it on another, and a sun sharper than their 2 degree step (goegap,
    // peaking at 1.2e5) came out at 1% of its energy: 0.07 of irradiance
    // taken out where the sun delivers about six, and the rest smeared through
    // nine coefficients -- no shadow edge for a cloud or a catcher to cast.
    const float3 emission = lightEmission(light);
    const SunRegion region = sunRegionOf(light, params.projectLevel);
    float3 delivered = float3(0.0);
    float3 centre = float3(0.0);
    float solid = 0.0;
    float weight = 0.0;
    if (region.found) {
        const TextureRecord record = gTextures.records[light.texture];
        const uint w = max(uint(record.width) >> params.projectLevel, 1u);
        const uint h = max(uint(record.height) >> params.projectLevel, 1u);
        const float area = 2.0 * kSunPi * kSunPi / float(w * h);
        for (uint y = 0; y < h; ++y) {
            const float v = (float(y) + 0.5) / float(h);
            const float sinTheta = sin((1.0 - v) * kSunPi);
            if (sinTheta <= 0.0) {
                continue;
            }
            for (uint x = 0; x < w; ++x) {
                const float u = (float(x) + 0.5) / float(w);
                const float3 dir = domeDirection(light, float2(u, v));
                const float3 rgb =
                    gTextures.textures[record.slot].Load(int3(int(x), int(y), int(params.projectLevel))).rgb;
                if (!sunHolds(region, dir, rgb)) {
                    continue;
                }
                const float omega = sinTheta * area;
                delivered += rgb * omega;
                centre += dir * (dot(rgb, kSunGrey) * omega);
                weight += dot(rgb, kSunGrey) * omega;
                solid += omega;
            }
        }
    }
    // The cone it stands in: the solid angle's own, as a disc would.
    const float discCone = solid > 0.0 ? acos(saturate(1.0 - solid / (2.0 * kSunPi))) : 0.0;
    // A SUN IS A DISC, not a scatter. Its texels' light-weighted spread about
    // their centre -- <theta^2> = 2 (1 - |mean direction|) -- is what a disc
    // of that solid angle has, cone^2 / 2. A shop's lamps that pass the
    // threshold within ten degrees of each other are no sun: taken out and
    // given back as one disc at their centroid, they put one highlight where a
    // chrome strip mirrors several, and the Corvette's chrome went 0.527 to
    // 0.581 against the path traced frame (s94). A region spread past twice a
    // disc's mean square angle stays in the sky (no sun is taken out).
    const float spread = weight > 0.0 ? 2.0 * max(1.0 - length(centre) / weight, 0.0) : 0.0;
    const bool compact = spread <= 2.0 * 0.5 * discCone * discCone + 1.0e-6;
    const bool isSun = region.found && solid > 0.0 && dot(centre, centre) > 0.0 && compact;
    const float3 direction = isSun ? normalize(centre) : region.axis;
    const float cone = isSun ? discCone : 0.0;
    sunOut[dome * 2] = float4(direction, isSun ? max(cone, 1.0e-4) : 0.0);
    // The irradiance, and the threshold the region was cut at -- which is
    // what `env_project` cuts the same texels out of the harmonics with.
    sunOut[dome * 2 + 1] = float4(delivered * emission, region.threshold);
}
