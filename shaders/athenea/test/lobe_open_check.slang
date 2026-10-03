// Copyright (c) 2026 jesus luque.
//
// HOW MUCH OF A REFLECTION'S LOBE SEES THE SKY, OUT OF A TX TRANSFER'S BITS.
//
// `splatLobeOpen` asks the 256 (or 1024) bits of a gaussian which of the
// directions its reflection gathers from are open, where the first transfer
// had one number for the whole hemisphere. Three skies of bits have answers
// that need no reference: all of it open (one at every roughness), none of it
// (zero), and the upper half open over a closed lower one -- a ground -- where
// a lobe pointing up sees all of it, one pointing down none, and tilting it
// from the one to the other never opens it again.
module lobe_open_check;
import athenea.splat.splat_relight;

struct LobeOpenParams {
    uint  side;        ///< 16 or 32 cells a side
    uint  steps;       ///< roughness and elevation steps
    float tolerance;
    uint  pad0;
};

RWStructuredBuffer<uint>  fill;     ///< three skies of `side * side / 32` words
StructuredBuffer<uint>    bits;     ///< the same, read back
RWStructuredBuffer<uint>  stats;    ///< 8 wide
RWStructuredBuffer<float> worst;    ///< 8 wide
ConstantBuffer<LobeOpenParams> params;

/// One thread a word: sky 0 open, sky 1 closed, sky 2 open above y = 0.
[shader("compute")]
[numthreads(64, 1, 1)]
void lobeOpenFill(uint3 tid: SV_DispatchThreadID) {
    const uint words = params.side * params.side / 32u;
    if (tid.x >= words * 3u) {
        return;
    }
    const uint sky = tid.x / words;
    const uint word = tid.x % words;
    uint value = 0u;
    for (uint b = 0; b < 32u; ++b) {
        const uint cell = word * 32u + b;
        const float3 d = octDecode((float2(float(cell % params.side), float(cell / params.side)) + 0.5) /
                                   float(params.side));
        const bool open = sky == 0u ? true : (sky == 1u ? false : d.y > 0.0);
        value |= open ? (1u << b) : 0u;
    }
    fill[tid.x] = value;
}

SplatSurface surfaceOn(uint sky) {
    SplatSurface s;
    s.normal = float3(0.0, 1.0, 0.0);
    s.cellWords = params.side * params.side / 32u;
    s.cellBase = sky * s.cellWords;
    return s;
}

/// stats[0] all-open readings that are not one
/// stats[1] all-closed readings that are not zero
/// stats[2] a ground: straight up not open, or straight down not closed
/// stats[3] a ground: tilting down opened it again
/// stats[4] readings out of [0, 1]
/// stats[5] readings taken
/// worst[0] the largest step back up while tilting down
[shader("compute")]
[numthreads(1, 1, 1)]
void lobeOpenCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float3 none = float3(0.0);
    float rise = 0.0;
    for (uint i = 0; i < params.steps; ++i) {
        const float roughness = float(i) / float(params.steps);
        float previous = 2.0;
        for (uint j = 0; j <= params.steps * 4u; ++j) {
            // From straight up to straight down, through a slant off the axes.
            const float elevation = 1.5707963 - 3.14159265 * float(j) / float(params.steps * 4u);
            const float3 m = normalize(float3(cos(elevation) * 0.8, sin(elevation), cos(elevation) * 0.6));
            const float open = splatLobeOpen(surfaceOn(0u), bits, m, roughness, none);
            const float shut = splatLobeOpen(surfaceOn(1u), bits, m, roughness, none);
            const float ground = splatLobeOpen(surfaceOn(2u), bits, m, roughness, none);
            if (abs(open - 1.0) > params.tolerance) {
                InterlockedAdd(stats[0], 1u);
            }
            if (shut > params.tolerance) {
                InterlockedAdd(stats[1], 1u);
            }
            if ((j == 0u && ground < 1.0 - params.tolerance && roughness <= 0.5) ||
                (j == params.steps * 4u && ground > params.tolerance && roughness <= 0.5)) {
                InterlockedAdd(stats[2], 1u);
            }
            if (ground > previous + 0.05) {
                InterlockedAdd(stats[3], 1u);
            }
            rise = max(rise, ground - previous);
            previous = ground;
            if (min(min(open, shut), ground) < 0.0 || max(max(open, shut), ground) > 1.0) {
                InterlockedAdd(stats[4], 1u);
            }
            InterlockedAdd(stats[5], 1u);
        }
    }
    worst[0] = rise;
}

/// A LIGHT'S SHADOW OUT OF THE SAME BITS (step 4): over the ground sky, a
/// light 45 degrees up is open whatever its size, one 45 degrees down is
/// closed, and one on the horizon is part open, the more evenly the wider it
/// is; a sphere of radius one ten away subtends atan(0.1).
/// stats[0] up not open, stats[1] down not closed, stats[2] the horizon not
/// between, stats[3] the angle off, stats[4] readings
[shader("compute")]
[numthreads(1, 1, 1)]
void lightOpenCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const float3 none = float3(0.0);
    for (uint i = 0; i < params.steps; ++i) {
        const float spread = 0.6 * float(i) / float(params.steps);
        const float3 up = normalize(float3(0.7, 0.7, 0.1));
        const float3 down = normalize(float3(0.7, -0.7, 0.1));
        const float3 level = normalize(float3(0.6, 0.0, 0.8));
        if (splatConeOpen(surfaceOn(2u), bits, up, spread, none) < 1.0 - params.tolerance) {
            InterlockedAdd(stats[0], 1u);
        }
        if (splatConeOpen(surfaceOn(2u), bits, down, spread, none) > params.tolerance) {
            InterlockedAdd(stats[1], 1u);
        }
        const float h = splatConeOpen(surfaceOn(2u), bits, level, spread, none);
        if (spread > 0.2 && (h < 0.2 || h > 0.8)) {
            InterlockedAdd(stats[2], 1u);
        }
        InterlockedAdd(stats[4], 1u);
    }
    LightRecord sphere;
    sphere.kind = kLightSphere;
    sphere.sizeX = 1.0;
    sphere.sizeY = 0.0;
    if (abs(lightHalfAngle(sphere, 10.0) - atan(0.1)) > 1.0e-5) {
        InterlockedAdd(stats[3], 1u);
    }
    LightRecord sun;
    sun.kind = kLightDistant;
    sun.sizeX = 0.0093;
    if (abs(lightHalfAngle(sun, 1.0e30) - 0.00465) > 1.0e-6) {
        InterlockedAdd(stats[3], 1u);
    }
}
