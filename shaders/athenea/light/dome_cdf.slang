// Copyright (c) 2026 jesus luque.
//
// A DOME'S IMAGE AS A DISTRIBUTION: the exact two-level CDF a lat-long is
// importance sampled with, built on the device from the image itself.
//
// After Cycles (kernel/light/background.h and the background map it builds,
// Apache-2.0, Blender Foundation): a marginal over the rows and, within each
// row, a conditional over its columns, of the luminance times sin(theta) --
// the sine because a lat-long texel covers 2 pi^2 sin(theta) du dv of solid
// angle, so a texel near a pole is worth less sky than one at the horizon.
// Sampling searches the marginal for a row and that row's conditional for a
// column (lights_image.slang); the density is read back from the same two
// tables, so what MIS weighs by is exactly what the samples were drawn with.
//
// Layout, in floats from `cdfBase` of the lights' values buffer (iesValues):
//
//     [0] width   [1] height   [2] mip level   [3] total
//     [4 .. 4 + height]                     the marginal, height + 1 sums
//     [5 + height + j (width + 1) ...]      row j's conditional, width + 1 sums
//
// Every table is a running sum, unnormalised, starting at zero: a cell's
// weight is the difference of two neighbours, and a search compares a
// number drawn against the last entry times u. One thread a row walks its
// texels in order, and one thread then walks the rows' totals: a sequential
// float sum is exact enough at a thousand terms and needs no second pass,
// where gpu/algo's PrefixSum scans uint32 and would need the luminance in
// fixed point.
//
// The table is built from a mip level at most kDomeCdfWidth wide (LightTable
// picks it): a 4K HDRI's full resolution costs 32 MB of sums for a sun the
// 1024-wide level already isolates to a few texels, which is the resolution
// Cycles builds its background map at by default.
module dome_cdf;
import athenea.material.material_inputs;

RWStructuredBuffer<float> cdfValues;
uniform uint              cdfBase;
uniform uint              cdfTexture;   // the dome's id in the texture table
uniform uint              cdfLevel;     // the mip level the table is built from
uniform uint              cdfWidth;     // that level's size
uniform uint              cdfHeight;

static const float3 kCdfLuminance = float3(0.2126, 0.7152, 0.0722);
static const float  kCdfPi = 3.14159265358979;

/// A texel's luminance, read exactly; zero where an HDRI holds a negative,
/// an infinity or a NaN, which would otherwise poison every sum after it.
float cdfTexel(uint slot, int x, int y) {
    const float3 rgb = gTextures.textures[slot].Load(int3(x, y, int(cdfLevel))).rgb;
    const float l = dot(rgb, kCdfLuminance);
    return (l >= 0.0 && l < 1.0e30) ? l : 0.0;
}

/// Row j's conditional: its texels' luminance times the row's sin(theta),
/// summed in order. Rows count up from the bottom (v = 0), as the texture's
/// do (texture_decode.slang), so row j spans v in [j, j + 1) / height.
[shader("compute")]
[numthreads(64, 1, 1)]
void domeCdfRows(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= cdfHeight) {
        return;
    }
    const TextureRecord record = gTextures.records[cdfTexture];
    const float sinTheta = sin((float(j) + 0.5) / float(cdfHeight) * kCdfPi);
    const uint row = cdfBase + 5 + cdfHeight + j * (cdfWidth + 1);
    float running = 0.0;
    cdfValues[row] = 0.0;
    for (uint i = 0; i < cdfWidth; ++i) {
        running += cdfTexel(record.slot, int(i), int(j)) * sinTheta;
        cdfValues[row + i + 1] = running;
    }
}

/// The marginal over the rows' totals, and the header.
[shader("compute")]
[numthreads(1, 1, 1)]
void domeCdfMarginal(uint3 tid: SV_DispatchThreadID) {
    const uint marginal = cdfBase + 4;
    float running = 0.0;
    cdfValues[marginal] = 0.0;
    for (uint j = 0; j < cdfHeight; ++j) {
        running += cdfValues[cdfBase + 5 + cdfHeight + j * (cdfWidth + 1) + cdfWidth];
        cdfValues[marginal + j + 1] = running;
    }
    cdfValues[cdfBase] = float(cdfWidth);
    cdfValues[cdfBase + 1] = float(cdfHeight);
    cdfValues[cdfBase + 2] = float(cdfLevel);
    cdfValues[cdfBase + 3] = running;
}
