// Copyright (c) 2026 jesus luque.
//
// WHETHER A SMALL NETWORK RUNS AND LEARNS IN A KERNEL HERE. A 16 -> 32 -> 32 -> 3
// perceptron a pixel, its weights in a buffer: the forward pass as a deferred
// decoder would run it, and the backward pass Slang's automatic differentiation
// writes -- the gradient of a pixel's squared error with respect to its
// features, and with respect to every weight summed over the pixels. A weight
// is read through `weight`, whose derivative adds into `weightGrad`: integers,
// since Metal has no atomic floats. The check is the oldest one: a derivative
// against the difference of two evaluations a step apart.
module neural_mlp;

static const uint kIn = 16;
static const uint kHidden = 32;
static const uint kOut = 3;
static const uint kL0 = 0;                                  // kHidden x kIn weights, then kHidden biases
static const uint kL1 = kHidden * kIn + kHidden;
static const uint kL2 = kL1 + kHidden * kHidden + kHidden;
static const uint kWeights = kL2 + kOut * kHidden + kOut;

StructuredBuffer<float>     weights;
RWStructuredBuffer<int>     weightGrad;    // fixed point, summed over pixels
StructuredBuffer<float>     features;      // kIn a pixel
StructuredBuffer<float4>    target;
RWStructuredBuffer<float4>  colour;
RWStructuredBuffer<float>   featureGrad;   // kIn a pixel
RWStructuredBuffer<uint>    loss;          // fixed point
RWStructuredBuffer<float>   made;          // what `neuralMlpInit` fills: weights, features or a target's floats
RWStructuredBuffer<uint>    stats;         // 8 wide, one row: see neuralMlpCheck
struct P { uint width; uint height; float gradScale; float step; };
ConstantBuffer<P> params;

void weightBack(uint i, float d) {
    InterlockedAdd(weightGrad[i], int(d * params.gradScale));
}
[BackwardDerivative(weightBack)]
float weight(uint i) { return weights[i]; }

[Differentiable]
float3 decode(float x[kIn]) {
    float h0[kHidden];
    for (uint r = 0; r < kHidden; ++r) {
        float s = weight(kL0 + kHidden * kIn + r);
        for (uint c = 0; c < kIn; ++c) {
            s += weight(kL0 + r * kIn + c) * x[c];
        }
        h0[r] = max(s, 0.0);
    }
    float h1[kHidden];
    for (uint r = 0; r < kHidden; ++r) {
        float s = weight(kL1 + kHidden * kHidden + r);
        for (uint c = 0; c < kHidden; ++c) {
            s += weight(kL1 + r * kHidden + c) * h0[c];
        }
        h1[r] = max(s, 0.0);
    }
    float3 o;
    for (uint r = 0; r < kOut; ++r) {
        float s = weight(kL2 + kOut * kHidden + r);
        for (uint c = 0; c < kHidden; ++c) {
            s += weight(kL2 + r * kHidden + c) * h1[c];
        }
        o[r] = s;
    }
    return o;
}

[Differentiable]
float pixelLoss(float x[kIn], no_diff float3 wanted) {
    const float3 d = decode(x) - wanted;
    return dot(d, d);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void neuralMlpForward(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint at = tid.y * params.width + tid.x;
    float x[kIn];
    for (uint c = 0; c < kIn; ++c) x[c] = features[at * kIn + c];
    colour[at] = float4(decode(x), 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void neuralMlpBackward(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint at = tid.y * params.width + tid.x;
    float x[kIn];
    for (uint c = 0; c < kIn; ++c) x[c] = features[at * kIn + c];
    DifferentialPair<float[kIn]> dx = diffPair(x);
    bwd_diff(pixelLoss)(dx, target[at].rgb, 1.0);
    for (uint c = 0; c < kIn; ++c) featureGrad[at * kIn + c] = dx.d[c];
    InterlockedAdd(loss[0], uint(pixelLoss(x, target[at].rgb) * 1024.0));
}

/// The same derivatives written out: the forward pass keeping what each layer
/// held, then the error carried back layer by layer.
[shader("compute")]
[numthreads(16, 16, 1)]
void neuralMlpBackwardByHand(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint at = tid.y * params.width + tid.x;
    float x[kIn];
    for (uint c = 0; c < kIn; ++c) x[c] = features[at * kIn + c];
    float h0[kHidden];
    for (uint r = 0; r < kHidden; ++r) {
        float s = weights[kL0 + kHidden * kIn + r];
        for (uint c = 0; c < kIn; ++c) s += weights[kL0 + r * kIn + c] * x[c];
        h0[r] = max(s, 0.0);
    }
    float h1[kHidden];
    for (uint r = 0; r < kHidden; ++r) {
        float s = weights[kL1 + kHidden * kHidden + r];
        for (uint c = 0; c < kHidden; ++c) s += weights[kL1 + r * kHidden + c] * h0[c];
        h1[r] = max(s, 0.0);
    }
    float3 o;
    for (uint r = 0; r < kOut; ++r) {
        float s = weights[kL2 + kOut * kHidden + r];
        for (uint c = 0; c < kHidden; ++c) s += weights[kL2 + r * kHidden + c] * h1[c];
        o[r] = s;
    }
    const float3 dOut = 2.0 * (o - target[at].rgb);
    float d1[kHidden];
    for (uint c = 0; c < kHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kOut; ++r) s += weights[kL2 + r * kHidden + c] * dOut[r];
        d1[c] = h1[c] > 0.0 ? s : 0.0;
    }
    float d0[kHidden];
    for (uint c = 0; c < kHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kHidden; ++r) s += weights[kL1 + r * kHidden + c] * d1[r];
        d0[c] = h0[c] > 0.0 ? s : 0.0;
    }
    for (uint c = 0; c < kIn; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kHidden; ++r) s += weights[kL0 + r * kIn + c] * d0[r];
        featureGrad[at * kIn + c] = s;
    }
}

float hashed(uint v) {
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return float(v & 0xFFFFFFu) / 16777216.0;
}

/// Fills `made` with numbers in -gradScale .. gradScale, a hash of the index.
[shader("compute")]
[numthreads(64, 1, 1)]
void neuralMlpInit(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width) return;
    made[tid.x] = (hashed(tid.x + params.height * 7919u) * 2.0 - 1.0) * params.gradScale;
}

/// stats[0] pixels checked, stats[1] of those whose derivative with respect to one feature (the pixel's index
/// modulo kIn) is further from the central difference than a hundredth of the larger of the two and 1e-3,
/// stats[2] pixels whose loss is not finite.
[shader("compute")]
[numthreads(16, 16, 1)]
void neuralMlpCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) return;
    const uint at = tid.y * params.width + tid.x;
    float x[kIn];
    for (uint c = 0; c < kIn; ++c) x[c] = features[at * kIn + c];
    const uint which = at % kIn;
    const float3 wanted = target[at].rgb;
    const float here = x[which];
    x[which] = here + params.step;
    const float up = pixelLoss(x, wanted);
    x[which] = here - params.step;
    const float down = pixelLoss(x, wanted);
    const float difference = (up - down) / (2.0 * params.step);
    const float derivative = featureGrad[at * kIn + which];
    InterlockedAdd(stats[0], 1u);
    if (abs(difference - derivative) > max(0.01 * max(abs(difference), abs(derivative)), 1.0e-3)) {
        InterlockedAdd(stats[1], 1u);
    }
    InterlockedAdd(stats[3], uint(min(abs(difference), 1000.0) * 16.0));
    InterlockedAdd(stats[4], uint(min(abs(derivative), 1000.0) * 16.0));
    InterlockedAdd(stats[5], uint(min(up, 1000.0) * 16.0));
    if (isnan(up) || isinf(up)) {
        InterlockedAdd(stats[2], 1u);
    }
}
