// Copyright (c) 2026 jesus luque.
//
// The kernels round a neural appearance (neural_appearance.slang): where a
// fit starts from, and the colours of a frame's gaussians.
module neural_shade;
import neural_appearance;
import athenea.common.color;
import athenea.common.packing;
import athenea.splat.frame;

struct NeuralShadeParams {
    uint  count;         // gaussians in the cloud, or weights for neuralInitWeights
    uint  base;          // the cloud's first slot in the frame
    uint  linearCloud;   // neuralInitLatent: the cloud's colours are light already
    uint  seed;
    float eyeX; float eyeY; float eyeZ;   // the eye, in the cloud's own space
    float spread;        // neuralInitWeights: the range of a weight
    // LIT: 1, and the frame's lights -- `lightCount` of them, each where it comes from (unit, towards it,
    // in the cloud's space) and its radiance. The answer is linear in the light, so several are several
    // evaluations summed.
    uint  lit; uint lightCount; uint pad0; uint pad1;
    float4 lightTowards[kNeuralLights];
    float4 lightRadiance[kNeuralLights];
    // What reaches each gaussian from the light: `shadowFactors` holds this many floats a slot, the
    // frame's first light's first. 0: all of it.
    uint  shadowLights;
    uint  split;        // two networks: the matt part and what the view adds (neural_appearance.slang)
    uint pad3; uint pad4;
};

StructuredBuffer<float4>           positions;
StructuredBuffer<uint>             shape;
StructuredBuffer<uint>             visible;     // one a slot, as the projection left it
StructuredBuffer<float>            latent;
StructuredBuffer<float>            weights;
StructuredBuffer<float>            shadowFactors;
RWStructuredBuffer<float>          made;        // the latent or the weights being started
RWStructuredBuffer<ProjRecord>     proj;
ConstantBuffer<NeuralShadeParams>  params;

/// Where a gaussian starts: no features, and its base the light its own
/// colour is (the harmonics' first term).
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralInitLatent(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float3 own = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const float3 light = cloudLight(own, params.linearCloud != 0);
    for (uint k = 0; k < kNeuralFeatures; ++k) {
        made[i * kNeuralStride + k] = 0.0;
    }
    made[i * kNeuralStride + kNeuralFeatures] = light.x;
    made[i * kNeuralStride + kNeuralFeatures + 1] = light.y;
    made[i * kNeuralStride + kNeuralFeatures + 2] = light.z;
}

/// Where the network starts: its two first layers at random in -spread ..
/// spread, its last at zero, so that it adds nothing until it has learnt to.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralInitWeights(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    made[i] = (i % kNeuralWeights) < kNeuralLayer2 ? (hash01(i * 2654435761u + params.seed) * 2.0 - 1.0) * params.spread : 0.0;
}

/// The network's weights in the group's memory: read from the buffer once a
/// group of 64 gaussians, each thread its share, rather than 960 times a
/// gaussian (neural_appearance.slang's INeuralWeights).
groupshared float sWeights[kNeuralWeights * 2];
struct NeuralWeightsHere : INeuralWeights {
    float at(uint index) { return sWeights[index]; }
}

/// The colours of one cloud's gaussians in this frame, written over what the
/// projection put in their records.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralShade(uint3 tid: SV_DispatchThreadID, uint3 lane: SV_GroupThreadID) {
    const uint held2 = params.split != 0 ? kNeuralWeights * 2 : kNeuralWeights;
    for (uint w = lane.x; w < held2; w += 64) {
        sWeights[w] = weights[w];
    }
    GroupMemoryBarrierWithGroupSync();
    const NeuralWeightsHere held;
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    if (visible[slot] == 0u) {
        return;
    }
    const float3 eye = float3(params.eyeX, params.eyeY, params.eyeZ);
    const float3 direction = normalize(positions[i].xyz - eye);
    const bool lit = params.lit != 0;
    const bool split = params.split != 0;
    NeuralWeightsFrom<NeuralWeightsHere> second;
    second.inner = held;
    second.offset = kNeuralWeights;
    if (!lit) {
        float3 own = neuralForward(latent, i, held, direction, false, float3(0.0), 1.0, split ? kNeuralMatt : kNeuralWhole).colour;
        if (split) {
            own += neuralForward(latent, i, second, direction, false, float3(0.0), 1.0, kNeuralShine).colour;
        }
        proj[slot].colour.xyz = own;
        return;
    }
    float3 colour = float3(0.0);
    const uint lights = min(params.lightCount, kNeuralLights);
    for (uint l = 0; l < lights; ++l) {
        // What reaches the gaussian from this light: a light the frame's shadows did not cover reaches it whole.
        const float reached = l < params.shadowLights ? shadowFactors[slot * params.shadowLights + l] : 1.0;
        const float3 light = neuralLocalLight(shape, i, positions[i].xyz, eye, params.lightTowards[l].xyz);
        float3 answer = neuralForward(latent, i, held, direction, true, light, reached, split ? kNeuralMatt : kNeuralWhole).colour;
        if (split) {
            answer += neuralForward(latent, i, second, direction, true, light, reached, kNeuralShine).colour;
        }
        colour += answer * params.lightRadiance[l].rgb;
    }
    proj[slot].colour.xyz = colour;
}
