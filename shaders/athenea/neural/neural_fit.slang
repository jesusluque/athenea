// Copyright (c) 2026 jesus luque.
//
// A NEURAL APPEARANCE FITTED TO PICTURES, on the device (neural_appearance.slang).
//
// A step is one camera. The frame drawn from the latents is compared with the
// frame it should be (`neuralFitError`: the loss's derivative at each pixel).
// The blend carries that back to each gaussian's colour (its contributionMode
// 4: the weights are the same weights, the geometry is not being fitted).
// `neuralFitBackward` then walks the network backwards once a gaussian in
// view, keeping what each layer held and each layer's error;
// `neuralFitWeights` sums a weight's gradient over the gaussians -- one thread
// a weight, reading, so no thread adds into another's sum and no float is
// added atomically, which Metal cannot do -- and `neuralFitAdam` moves the
// latents and the weights.
module neural_fit;
import neural_appearance;
import athenea.splat.frame;
import athenea.common.color;

/// What `neuralFitBackward` keeps a gaussian for the weights' sums: whether
/// it was in view, the network's input and its two hidden layers, and the
/// three layers' errors.
/// (Two of these a gaussian: see neuralFitBackward.)
public static const uint kFitRecord = 1 + kNeuralIn + kNeuralHidden * 2 + kNeuralOut + kNeuralHidden * 2;
static const uint kRecX = 1;
static const uint kRecH0 = kRecX + kNeuralIn;
static const uint kRecH1 = kRecH0 + kNeuralHidden;
static const uint kRecDOut = kRecH1 + kNeuralHidden;
static const uint kRecD1 = kRecDOut + kNeuralOut;
static const uint kRecD0 = kRecD1 + kNeuralHidden;

struct NeuralFitParams {
    uint  width;
    uint  height;
    uint  count;        // gaussians; or the numbers neuralFitAdam and neuralFitClear walk
    uint  base;         // the cloud's first slot in the frame
    float eyeX; float eyeY; float eyeZ;
    uint  stride;       // neuralFitAdam: numbers a row (kNeuralStride, or 1 for the weights)
    uint  firstOfSecond;   // neuralFitAdam: the index in a row from which `rateSecond` is the rate
    float rate;
    float rateSecond;
    float beta1; float beta2; float epsilon;
    float correction1;  // 1 - beta1^t
    float correction2;  // 1 - beta2^t
    uint  sparse;       // neuralFitAdam: a number whose gradient is exactly zero was not seen, and stays
    // neuralFitWeights: the gaussians a step sums over are one in `every`, those whose index is `phase`
    // past a multiple of it -- a different share each step, and the sum scaled to the whole.
    uint  every;
    uint  phase;
    // LIT (neural_appearance.slang): 1, and the frame's light as neural_shade has it.
    uint  lit;
    float lightX; float lightY; float lightZ; uint pad0;
    float radianceR; float radianceG; float radianceB;
    uint  shadowLights;   // floats a slot in `shadowFactors`, the frame's light's first; 0: none
    // SMOOTH IN THE VIEW. A gaussian's colour seen from a direction `viewStep` (radians) off the
    // frame's own is held to the colour seen from it, `viewSmooth` times as hard as a pixel of the
    // picture is held to its picture: what keeps a network from turning with the view faster than
    // anything it was shown does, which is the shimmer a camera going by sees. 0: not held.
    float viewSmooth;
    float viewStep;
    uint  step;           // the fit's step, so that the direction off is another each time
    uint  pad2;
};

StructuredBuffer<float4>           student;      // the frame drawn from the latents
StructuredBuffer<float4>           teacher;      // the frame it should be
RWStructuredBuffer<float4>         pixelGrad;    // the loss's derivative at each pixel
RWStructuredBuffer<uint>           loss;         // the squared error summed, in 1/1024, a word a row of the frame
StructuredBuffer<uint>             colourGrad;   // three signed words a slot, as the blend summed them
StructuredBuffer<float4>           positions;
StructuredBuffer<uint>             shape;
StructuredBuffer<float>            shadowFactors;
StructuredBuffer<float>            latent;
StructuredBuffer<float>            weights;
RWStructuredBuffer<float>          records;      // kFitRecord a gaussian
RWStructuredBuffer<float>          gradient;     // what a kernel writes: the latents' or the weights'
StructuredBuffer<float>            recordsRead;
RWStructuredBuffer<float>          value;        // neuralFitAdam: what moves
RWStructuredBuffer<float>          moment1;
RWStructuredBuffer<float>          moment2;
StructuredBuffer<float>            gradientRead;
RWStructuredBuffer<uint>           cleared;
ConstantBuffer<NeuralFitParams>    params;

/// The error, and its derivative at each pixel. The error is the squared
/// difference over the squared level of what the pixel should be (plus a
/// hundredth): the relMSE the engine measures images by. A plain squared
/// difference in linear light is all highlights -- on a capture of the pawn it
/// fell to a ninth while the dark marble, a thirtieth of the glass's level and
/// most of the picture, drifted thirty code values off.
[shader("compute")]
[numthreads(16, 16, 1)]
void neuralFitError(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float3 wanted = teacher[at].rgb;
    const float3 d = student[at].rgb - wanted;
    const float level = dot(wanted, wanted) / 3.0 + 1.0e-2;
    pixelGrad[at] = float4(clamp(2.0 * d / level, float3(-50.0), float3(50.0)), 0.0);
    InterlockedAdd(loss[tid.y], uint(min(dot(d, d) / (3.0 * level), 64.0) * 1024.0 + 0.5));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralFitClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        cleared[tid.x] = 0u;
    }
}

/// What the weights' sums read of one evaluation, at `row`.
void storeRecord(uint row, NeuralPass pass, NeuralBack back) {
    records[row] = 1.0;
    for (uint k = 0; k < kNeuralIn; ++k) {
        records[row + kRecX + k] = pass.x[k];
    }
    for (uint k = 0; k < kNeuralHidden; ++k) {
        records[row + kRecH0 + k] = pass.h0[k];
        records[row + kRecH1 + k] = pass.h1[k];
        records[row + kRecD1 + k] = back.d1[k];
        records[row + kRecD0 + k] = back.d0[k];
    }
    for (uint k = 0; k < kNeuralOut; ++k) {
        records[row + kRecDOut + k] = back.dOut[k];
    }
}

/// One gaussian's backward pass: its own derivatives into `gradient`
/// (kNeuralStride of them), and what the weights' sums read into `records`.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralFitBackward(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    const int3 summed = int3(int(colourGrad[slot * 4]), int(colourGrad[slot * 4 + 1]), int(colourGrad[slot * 4 + 2]));
    const float weighed = float(colourGrad[slot * 4 + 3]) / kColourGradScale;
    // A gaussian the frame did not draw was asked nothing.
    const bool seen = any(summed != int3(0)) || weighed > 0.0;
    records[i * kFitRecord * 2 + kFitRecord] = 0.0;
    if (!seen) {
        records[i * kFitRecord * 2] = 0.0;
        for (uint k = 0; k < kNeuralStride; ++k) {
            gradient[i * kNeuralStride + k] = 0.0;
        }
        return;
    }
    const bool lit = params.lit != 0;
    // The frame drew the answer times the light's radiance: so is its
    // derivative. What of the light reached the gaussian is the pass's own.
    const float reached = lit && params.shadowLights != 0 ? shadowFactors[slot * params.shadowLights] : 1.0;
    const float3 dColour = float3(summed) / kColourGradScale *
                           (lit ? float3(params.radianceR, params.radianceG, params.radianceB) : float3(1.0));
    const float3 eye = float3(params.eyeX, params.eyeY, params.eyeZ);
    const float3 direction = normalize(positions[i].xyz - eye);
    const float3 light = lit ? neuralLocalLight(shape, i, positions[i].xyz, eye,
                                                float3(params.lightX, params.lightY, params.lightZ))
                             : float3(0.0);
    NeuralWeightBuffer held;
    held.buffer = weights;
    const NeuralPass pass = neuralForward(latent, i, held, direction, lit, light, reached);
    float3 asked = dColour;
    NeuralBack off;
    const bool smooth = params.viewSmooth > 0.0 && weighed > 0.0;
    if (smooth) {
        // The same gaussian seen a step off, in a direction square to the
        // view that a hash of the gaussian and the step picks.
        const float3 other = abs(direction.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
        const float3 u = normalize(cross(direction, other));
        const float3 v = cross(direction, u);
        const float turn = 6.2831853 * hash01(i * 2654435761u + params.step * 40503u);
        const float3 aside = normalize(direction + params.viewStep * (cos(turn) * u + sin(turn) * v));
        const NeuralPass beside = neuralForward(latent, i, held, aside, lit, light, reached);
        // Held as a pixel is held: the difference over the level, weighed by
        // how much of the frame the gaussian was.
        const float level = dot(pass.colour, pass.colour) / 3.0 + 1.0e-2;
        const float3 pull = params.viewSmooth * weighed * 2.0 * (pass.colour - beside.colour) / level;
        asked += pull;
        off = neuralBackward(beside, held, -pull);
        storeRecord(i * kFitRecord * 2 + kFitRecord, beside, off);
    }
    const NeuralBack back = neuralBackward(pass, held, asked);
    for (uint k = 0; k < kNeuralStride; ++k) {
        gradient[i * kNeuralStride + k] = back.dLatent[k] + (smooth ? off.dLatent[k] : 0.0);
    }
    storeRecord(i * kFitRecord * 2, pass, back);
}

/// One weight's gradient: its layer's error at its row times its layer's
/// input at its column, summed over the gaussians in view.
[shader("compute")]
[numthreads(16, 1, 1)]
void neuralFitWeights(uint3 tid: SV_DispatchThreadID) {
    const uint w = tid.x;
    if (w >= kNeuralWeights) {
        return;
    }
    uint error;
    uint input;
    if (w < kNeuralLayer1) {
        error = kRecD0 + w / kNeuralIn;
        input = kRecX + w % kNeuralIn;
    } else if (w < kNeuralLayer2) {
        error = kRecD1 + (w - kNeuralLayer1) / kNeuralHidden;
        input = kRecH0 + (w - kNeuralLayer1) % kNeuralHidden;
    } else {
        error = kRecDOut + (w - kNeuralLayer2) / kNeuralHidden;
        input = kRecH1 + (w - kNeuralLayer2) % kNeuralHidden;
    }
    float sum = 0.0;
    for (uint i = params.phase; i < params.count; i += params.every) {
        // Two evaluations a gaussian: the frame's, and the one a step off where the fit holds the view smooth.
        for (uint which = 0; which < 2; ++which) {
            const uint row = (i * 2 + which) * kFitRecord;
            if (recordsRead[row] != 0.0) {
                sum += recordsRead[row + error] * recordsRead[row + input];
            }
        }
    }
    gradient[w] = sum * float(params.every);
}

/// Adam, a number a thread.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralFitAdam(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float g = gradientRead[i];
    if (params.sparse != 0 && g == 0.0) {
        return;
    }
    const float m = params.beta1 * moment1[i] + (1.0 - params.beta1) * g;
    const float v = params.beta2 * moment2[i] + (1.0 - params.beta2) * g * g;
    moment1[i] = m;
    moment2[i] = v;
    const float rate = (i % params.stride) >= params.firstOfSecond ? params.rateSecond : params.rate;
    value[i] -= rate * (m / params.correction1) / (sqrt(v / params.correction2) + params.epsilon);
}
