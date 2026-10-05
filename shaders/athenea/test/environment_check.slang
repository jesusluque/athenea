// Copyright (c) 2026 jesus luque.
//
// THE PREPARED SKY, AGAINST WHAT IT MUST BE.
//
// Two things are checked here, and nothing crosses to the processor but
// counters:
//
//   the harmonics   a sky of constant radiance L puts `pi * L` of irradiance
//                   on every normal, whichever way it faces. That is the
//                   closed form a uniform dome has, and it is what the nine
//                   coefficients must give back -- if they do not, either the
//                   projection's measure or the cosine lobe's bands are wrong.
//   the levels      the prefiltered map read along a direction, against what
//                   the caller says is there. A sky that is white above the
//                   horizon and black below reads one at the pole and nothing
//                   at the other, and its roughest level reads the mean of the
//                   two, which is what a convolution is.
module environment_check;
import athenea.light.environment;

struct EnvCheckParams {
    uint  dome;
    uint  directions;    ///< how many normals the irradiance is checked at
    float expected;      ///< the irradiance every normal must read (0: not checked)
    float tolerance;     ///< relative

    float upExpected;    ///< level 0 along +y (-1: not checked)
    float downExpected;  ///< level 0 along -y (-1: not checked)
    float roughExpected; ///< the roughest level, any direction (-1: not checked)
    uint  baseSide;      ///< the octahedral side of level 0, as Environment chose it
};

StructuredBuffer<uint>       envTexels;
StructuredBuffer<float4>     envSh;
RWStructuredBuffer<uint>     stats;    ///< 8 wide: see below
RWStructuredBuffer<float>    worst;    ///< 8 wide: the readings, for a printf
ConstantBuffer<EnvCheckParams> params;

/// Directions spread over the sphere without a pole to crowd: the Fibonacci
/// spiral, which is what a test wants when the thing under test is itself
/// about directions.
float3 spiralDirection(uint i, uint n) {
    const float golden = 2.39996322972865332;
    const float z = 1.0 - 2.0 * (float(i) + 0.5) / float(max(n, 1u));
    const float r = sqrt(max(1.0 - z * z, 0.0));
    const float phi = golden * float(i);
    return float3(r * cos(phi), r * sin(phi), z);
}

/// stats[0] normals whose irradiance is beyond the tolerance
/// stats[1] normals checked
/// stats[2] 1 where +y does not read what it should
/// stats[3] 1 where -y does not read what it should
/// stats[4] 1 where the roughest level does not read what it should
/// worst[0] the worst relative error of the irradiance
/// worst[1..3] the irradiance at the first direction, rgb
/// worst[4] +y at level 0, worst[5] -y at level 0, worst[6] the roughest level
[shader("compute")]
[numthreads(1, 1, 1)]
void environmentCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    float worstRelative = 0.0;
    if (params.expected > 0.0) {
        for (uint i = 0; i < params.directions; ++i) {
            const float3 n = spiralDirection(i, params.directions);
            const float3 e = envIrradiance(envSh, params.dome, n);
            if (i == 0) {
                worst[1] = e.r;
                worst[2] = e.g;
                worst[3] = e.b;
            }
            const float mean = (e.r + e.g + e.b) / 3.0;
            const float relative = abs(mean - params.expected) / max(params.expected, 1.0e-6);
            worstRelative = max(worstRelative, relative);
            if (relative > params.tolerance) {
                InterlockedAdd(stats[0], 1u);
            }
            InterlockedAdd(stats[1], 1u);
        }
    }
    worst[0] = worstRelative;

    const float3 up = envRadiance(envTexels, params.dome, float3(0.0, 1.0, 0.0), 0.0, params.baseSide);
    const float3 down = envRadiance(envTexels, params.dome, float3(0.0, -1.0, 0.0), 0.0, params.baseSide);
    const float3 rough = envRadiance(envTexels, params.dome, float3(0.0, 1.0, 0.0), 1.0, params.baseSide);
    worst[4] = (up.r + up.g + up.b) / 3.0;
    worst[5] = (down.r + down.g + down.b) / 3.0;
    worst[6] = (rough.r + rough.g + rough.b) / 3.0;
    if (params.upExpected >= 0.0 && abs(worst[4] - params.upExpected) > params.tolerance) {
        InterlockedAdd(stats[2], 1u);
    }
    if (params.downExpected >= 0.0 && abs(worst[5] - params.downExpected) > params.tolerance) {
        InterlockedAdd(stats[3], 1u);
    }
    if (params.roughExpected >= 0.0 && abs(worst[6] - params.roughExpected) > params.tolerance) {
        InterlockedAdd(stats[4], 1u);
    }
}


/// A ROUGH LEVEL HOLDS A COMPACT BRIGHT SOURCE AS ONE LOBE: the texels of
/// `level` brighter than a fiftieth of the level's brightest and brighter than
/// all eight neighbours. One where the prefilter converged; a scatter of them
/// where its samples hit the source in a pattern (s81).
/// stats[0] such peaks, stats[1] texels scanned; worst[0] the brightest.
struct EnvPeaksParams {
    uint dome;
    uint level;
    uint baseSide;
    uint pad0;
};
ConstantBuffer<EnvPeaksParams> peaks;

float peakLum(uint2 uv) {
    return dot(envTexelAt(envTexels, peaks.dome, peaks.level, uv, peaks.baseSide), float3(0.2126, 0.7152, 0.0722));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void environmentPeaks(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint side = envSideOf(peaks.level, peaks.baseSide);
    float brightest = 0.0;
    for (uint y = 0; y < side; ++y) {
        for (uint x = 0; x < side; ++x) {
            brightest = max(brightest, peakLum(uint2(x, y)));
        }
    }
    uint found = 0;
    for (uint y = 1; y + 1 < side; ++y) {
        for (uint x = 1; x + 1 < side; ++x) {
            const float v = peakLum(uint2(x, y));
            if (v < brightest / 50.0) {
                continue;
            }
            bool top = true;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if ((dx != 0 || dy != 0) && peakLum(uint2(int(x) + dx, int(y) + dy)) >= v) {
                        top = false;
                    }
                }
            }
            found += top ? 1u : 0u;
        }
    }
    stats[0] = found;
    stats[1] = side * side;
    worst[0] = brightest;
}
