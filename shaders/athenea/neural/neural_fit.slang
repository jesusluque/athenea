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

/// What `neuralFitBackward` keeps a gaussian for the weights' sums: whether
/// it was in view, the network's input and its two hidden layers, and the
/// three layers' errors.
public static const uint kFitRecord = 1 + kNeuralIn + kNeuralHidden * 2 + 3 + kNeuralHidden * 2;
static const uint kRecX = 1;
static const uint kRecH0 = kRecX + kNeuralIn;
static const uint kRecH1 = kRecH0 + kNeuralHidden;
static const uint kRecDOut = kRecH1 + kNeuralHidden;
static const uint kRecD1 = kRecDOut + 3;
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
    uint  pad0; uint pad1; uint pad2;
};

StructuredBuffer<float4>           student;      // the frame drawn from the latents
StructuredBuffer<float4>           teacher;      // the frame it should be
RWStructuredBuffer<float4>         pixelGrad;    // the loss's derivative at each pixel
RWStructuredBuffer<uint>           loss;         // the squared error summed, in 1/1024, a word a row of the frame
StructuredBuffer<uint>             colourGrad;   // three signed words a slot, as the blend summed them
StructuredBuffer<float4>           positions;
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

/// The squared error, and its derivative at each pixel: twice the difference.
[shader("compute")]
[numthreads(16, 16, 1)]
void neuralFitError(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const float3 d = student[at].rgb - teacher[at].rgb;
    pixelGrad[at] = float4(2.0 * d, 0.0);
    InterlockedAdd(loss[tid.y], uint(min(dot(d, d), 4.0) * 1024.0 + 0.5));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralFitClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        cleared[tid.x] = 0u;
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
    const int3 summed = int3(int(colourGrad[slot * 3]), int(colourGrad[slot * 3 + 1]), int(colourGrad[slot * 3 + 2]));
    // A gaussian the frame did not draw was asked nothing.
    const bool seen = any(summed != int3(0));
    if (!seen) {
        records[i * kFitRecord] = 0.0;
        for (uint k = 0; k < kNeuralStride; ++k) {
            gradient[i * kNeuralStride + k] = 0.0;
        }
        return;
    }
    const float3 dColour = float3(summed) / kColourGradScale;
    const float3 direction = normalize(positions[i].xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    const NeuralPass pass = neuralForward(latent, i, weights, direction);
    const NeuralBack back = neuralBackward(pass, weights, dColour);
    for (uint k = 0; k < kNeuralStride; ++k) {
        gradient[i * kNeuralStride + k] = back.dLatent[k];
    }
    const uint row = i * kFitRecord;
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
    for (uint k = 0; k < 3; ++k) {
        records[row + kRecDOut + k] = back.dOut[k];
    }
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
    for (uint i = 0; i < params.count; ++i) {
        const uint row = i * kFitRecord;
        if (recordsRead[row] != 0.0) {
            sum += recordsRead[row + error] * recordsRead[row + input];
        }
    }
    gradient[w] = sum;
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
