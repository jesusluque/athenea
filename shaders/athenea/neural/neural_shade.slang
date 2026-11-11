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
    // LIT: 1, and the one light of the frame -- where it comes from (unit, towards it, in the cloud's
    // space) and its radiance.
    uint  lit; float lightX; float lightY; float lightZ;
    float radianceR; float radianceG; float radianceB; uint pad0;
};

StructuredBuffer<float4>           positions;
StructuredBuffer<uint>             shape;
StructuredBuffer<uint>             visible;     // one a slot, as the projection left it
StructuredBuffer<float>            latent;
StructuredBuffer<float>            weights;
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
    made[i] = i < kNeuralLayer2 ? (hash01(i * 2654435761u + params.seed) * 2.0 - 1.0) * params.spread : 0.0;
}

/// The colours of one cloud's gaussians in this frame, written over what the
/// projection put in their records.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralShade(uint3 tid: SV_DispatchThreadID) {
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
    const float3 light = lit ? neuralLocalLight(shape, i, positions[i].xyz, eye,
                                                float3(params.lightX, params.lightY, params.lightZ))
                             : float3(0.0);
    const NeuralPass pass = neuralForward(latent, i, weights, direction, lit, light);
    proj[slot].colour.xyz =
        lit ? pass.colour * float3(params.radianceR, params.radianceG, params.radianceB) : pass.colour;
}
