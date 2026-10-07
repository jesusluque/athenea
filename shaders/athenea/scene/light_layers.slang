// Copyright (c) 2026 jesus luque.
//
// THE LAMPS' LAYERS OF A CLOUD (GpuSplats::lightLayers, docs/decisions.md:
// light groups).
//
// A group of lamps -- a car's low beam, its tail lamps -- is baked once per
// unit of its radiance: for every gaussian, the rgb light it sends out from
// that group alone. A frame adds each layer times the group's weight, so a
// lamp switched or dimmed is a number and nothing is baked again.
//
// `lightLayerPack`: one layer as the stage holds it -- three floats or halves
// a gaussian, `primvars:athenea:splat:lightLayer:<group>` -- into column
// `layer` of the cloud's words, RGB9E5, `layers` words a record.
//
// `lightLayerFromCloud`: what `athenea lights merge` does to a layer cloud.
// The same conversion, on the same geometry, with one group lit: its
// gaussians must be the base's, in the base's order, and its constant term
// is the group's light. Counts the positions that are not the base's and the
// gaussians that carry any light, and writes the light as floats: a lamp's
// light per unit of its radiance on a car body metres away is 1e-5 and less,
// where a half has a handful of bits left.
module light_layers;
import athenea.common.packing;
import athenea.common.color;

/// Value `index` of a float array: float32 words, or float16 halves packed two
/// to a little-endian word, first in the low bits (streams.slang's).
float streamValue(StructuredBuffer<uint> words, bool half, uint index) {
    if (half) {
        const uint word = words[index / 2];
        return (index & 1u) == 0 ? lowHalf(word) : highHalf(word);
    }
    return bit_cast<float>(words[index]);
}

struct PackParams {
    uint count;    ///< records
    uint layers;   ///< words a record
    uint layer;    ///< the column this layer goes to
    uint half;     ///< 1: the stream is float16
};

StructuredBuffer<uint>      layerStream;   // rgb a record
RWStructuredBuffer<uint>    packed;        // `layers` words a record
ConstantBuffer<PackParams>  packParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void lightLayerPack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= packParams.count) {
        return;
    }
    const bool half = packParams.half != 0;
    const float3 rgb = float3(streamValue(layerStream, half, i * 3), streamValue(layerStream, half, i * 3 + 1),
                              streamValue(layerStream, half, i * 3 + 2));
    // Negative and not-a-number are no light (packRgb9e5); scaled up first, so
    // the faint light a lamp leaves metres away keeps its mantissa
    // (kLightLayerScale).
    packed[i * packParams.layers + packParams.layer] = packRgb9e5(rgb * kLightLayerScale);
}

struct MergeParams {
    uint  count;
    uint  baseHalf;       ///< 1: the base's positions are float16
    uint  layerHalf;      ///< 1: the layer's positions are float16
    uint  shHalf;         ///< 1: the layer's harmonics are float16
    uint  coefficients;   ///< the layer's harmonics a gaussian, DC first
    uint  linearCloud;    ///< 1: the layer holds light; 0: a capture's sRGB
    float tolerance;      ///< how far a position may be from the base's, relative
    float floor;          ///< light at or below this is no light, for the count
};

StructuredBuffer<uint>      basePositions;    // xyz a gaussian
StructuredBuffer<uint>      layerPositions;   // xyz a gaussian
StructuredBuffer<uint>      layerSh;          // rgb a coefficient, `coefficients` a gaussian
RWStructuredBuffer<float>   layerLight;       // rgb a gaussian
RWStructuredBuffer<uint>    counters;         // [0] positions not the base's, [1] gaussians lit
ConstantBuffer<MergeParams> mergeParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void lightLayerFromCloud(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= mergeParams.count) {
        return;
    }
    const bool baseHalf = mergeParams.baseHalf != 0;
    const bool layerHalf = mergeParams.layerHalf != 0;
    bool same = true;
    [unroll] for (uint k = 0; k < 3; ++k) {
        const float a = streamValue(basePositions, baseHalf, i * 3 + k);
        const float b = streamValue(layerPositions, layerHalf, i * 3 + k);
        same = same && abs(a - b) <= mergeParams.tolerance * max(1.0, abs(a));
    }
    if (!same) {
        InterlockedAdd(counters[0], 1u);
    }
    // The constant term as 3DGS keeps it: colour = 0.5 + kSH0 dc, in light
    // where the cloud says so (common/color.slang's cloudLight). The rest of
    // the harmonics -- how the group's light changes with the eye -- is not
    // kept: a layer is view-independent (docs/decisions.md, light groups).
    const uint at = i * mergeParams.coefficients * 3;
    const bool shHalf = mergeParams.shHalf != 0;
    const float3 dc = float3(streamValue(layerSh, shHalf, at), streamValue(layerSh, shHalf, at + 1),
                             streamValue(layerSh, shHalf, at + 2));
    const float3 light = cloudLight(float3(0.5) + kSH0 * dc, mergeParams.linearCloud != 0);
    if (max(light.x, max(light.y, light.z)) > mergeParams.floor) {
        InterlockedAdd(counters[1], 1u);
    }
    layerLight[i * 3] = light.x;
    layerLight[i * 3 + 1] = light.y;
    layerLight[i * 3 + 2] = light.z;
}
