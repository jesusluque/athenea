// Copyright (c) 2026 jesus luque.
//
// THE OTHER WAY TO DECODE: ONCE A PIXEL. Where neural_shade.slang turns each
// gaussian's latent into a colour before the blend, here the blend sums
// features (splat_blend's `features`) and one perceptron runs on what a pixel
// was left with: its features over its coverage, and the harmonics of its
// ray. What this file is for today is what that costs -- the walk carrying 8,
// 16 or 32 channels, and a network a pixel -- against a network a gaussian;
// nothing fits it yet, so what it draws is a network that has learnt nothing.
module neural_pixel;
import athenea.common.packing;

/// A gaussian's features for the blend: the sines and cosines of its latent's
/// numbers, at one frequency for 16 channels, the sines alone for 8, and a
/// second frequency for 32.
struct PixelParams {
    uint  count;          // gaussians, for neuralPixelFeatures
    uint  base;           // the cloud's first slot
    uint  words;          // feature words a slot and a pixel: 4, 8 or 16
    uint  latentStride;   // numbers a gaussian in `latent`
    uint  width;
    uint  height;
    float focalX; float focalY;
    float centreX; float centreY;
    uint  pad0; uint pad1;
};

static const uint kPixelHidden = 32;
static const uint kPixelInMost = 16 + 32;
/// (16 + 32) x 32, 32 x 32, 3 x 32: laid out for the widest input, read for the one in use.
static const uint kPixelLayer1 = kPixelHidden * kPixelInMost;
static const uint kPixelLayer2 = kPixelLayer1 + kPixelHidden * kPixelHidden;
public static const uint kPixelWeights = kPixelLayer2 + 3 * kPixelHidden;

StructuredBuffer<float>         latent;
StructuredBuffer<uint>          visible;
RWStructuredBuffer<uint>        slotFeatures;
StructuredBuffer<uint>          features;
StructuredBuffer<float>         weights;
StructuredBuffer<float4>        colour;
RWStructuredBuffer<float4>      decoded;
ConstantBuffer<PixelParams>     params;

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralPixelFeatures(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    if (visible[slot] == 0u) {
        return;
    }
    for (uint w = 0; w < params.words; ++w) {
        const float z = latent[i * params.latentStride + (w % 8u)] * (w >= 8u ? 2.0 : 1.0);
        // Eight channels are four words of two numbers each: the sines of the eight.
        slotFeatures[slot * params.words + w] =
            params.words == 4u ? packHalves(sin(latent[i * params.latentStride + w * 2u]),
                                            sin(latent[i * params.latentStride + w * 2u + 1u]))
                               : packHalves(sin(z), cos(z));
    }
}

groupshared float sPixelWeights[kPixelWeights];

[shader("compute")]
[numthreads(16, 16, 1)]
void neuralPixelDecode(uint3 tid: SV_DispatchThreadID, uint3 lane: SV_GroupThreadID) {
    for (uint w = lane.y * 16u + lane.x; w < kPixelWeights; w += 256u) {
        sPixelWeights[w] = weights[w];
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    const float4 blended = colour[at];
    if (!(blended.a > 1.0e-4)) {
        decoded[at] = blended;
        return;
    }
    float x[kPixelInMost];
    // The pixel's ray, in the camera's own space: a direction every pixel has its own of.
    const float3 d = normalize(float3((float(tid.x) + 0.5 - params.centreX) / params.focalX,
                                      (float(tid.y) + 0.5 - params.centreY) / params.focalY, 1.0));
    const float xx = d.x * d.x, yy = d.y * d.y, zz = d.z * d.z;
    x[0] = 0.2820947918;
    x[1] = -0.4886025119 * d.y;
    x[2] = 0.4886025119 * d.z;
    x[3] = -0.4886025119 * d.x;
    x[4] = 1.0925484306 * d.x * d.y;
    x[5] = -1.0925484306 * d.y * d.z;
    x[6] = 0.3153915653 * (2.0 * zz - xx - yy);
    x[7] = -1.0925484306 * d.x * d.z;
    x[8] = 0.5462742153 * (xx - yy);
    x[9] = -0.5900435899 * d.y * (3.0 * xx - yy);
    x[10] = 2.8906114426 * d.x * d.y * d.z;
    x[11] = -0.4570457995 * d.y * (4.0 * zz - xx - yy);
    x[12] = 0.3731763326 * d.z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    x[13] = -0.4570457995 * d.x * (4.0 * zz - xx - yy);
    x[14] = 1.4453057213 * d.z * (xx - yy);
    x[15] = -0.5900435899 * d.x * (xx - 3.0 * yy);
    const uint inputs = 16u + params.words * 2u;
    for (uint c = 0; c < params.words; ++c) {
        const uint word = features[c * pixels + at];
        x[16 + c * 2] = lowHalf(word) / blended.a;
        x[16 + c * 2 + 1] = highHalf(word) / blended.a;
    }
    float h0[kPixelHidden];
    for (uint r = 0; r < kPixelHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < inputs; ++c) {
            s += sPixelWeights[r * kPixelInMost + c] * x[c];
        }
        h0[r] = max(s, 0.0);
    }
    float h1[kPixelHidden];
    for (uint r = 0; r < kPixelHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kPixelHidden; ++c) {
            s += sPixelWeights[kPixelLayer1 + r * kPixelHidden + c] * h0[c];
        }
        h1[r] = max(s, 0.0);
    }
    float3 o = float3(0.0);
    for (uint r = 0; r < 3; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kPixelHidden; ++c) {
            s += sPixelWeights[kPixelLayer2 + r * kPixelHidden + c] * h1[c];
        }
        o[r] = s;
    }
    decoded[at] = float4(max(blended.rgb + blended.a * tanh(o), float3(0.0)), blended.a);
}
