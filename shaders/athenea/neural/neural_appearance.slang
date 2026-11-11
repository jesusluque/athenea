// Copyright (c) 2026 jesus luque.
//
// A GAUSSIAN'S COLOUR FROM A LATENT AND A SHARED NETWORK, in place of its
// harmonics. Each gaussian carries kNeuralFeatures numbers and a base colour;
// one small perceptron, the same for the whole cloud, turns the features and
// the direction the gaussian is seen from into what is added to the base:
//
//     colour = max(base + tanh(network(harmonics(direction), sin(z), cos(z))), 0)
//
// after Hahlbohm et al., "Compact Neural Appearance Models for Efficient
// Gaussian Splatting" (arXiv 2609.05255): the direction enters as its
// spherical harmonics of degrees 0 to 3 (16 numbers), the features as their
// sines and cosines (16), two hidden layers of 16 with ReLUs, no biases --
// 816 weights a cloud, 11 numbers a gaussian where harmonics of degree 3 are
// 48. With the features and the last layer at zero the colour is the base,
// which is where a fit starts from.
//
// The backward pass is written out here, not left to Slang's automatic
// differentiation: of a network written with arrays in loops, Slang
// 2026.14.1's is not the derivative (docs/decisions.md).
module neural_appearance;

public static const uint kNeuralFeatures = 8;
/// A gaussian's numbers: its features, then its base colour (linear light).
public static const uint kNeuralStride = kNeuralFeatures + 3;
public static const uint kNeuralIn = 16 + kNeuralFeatures * 2;
public static const uint kNeuralHidden = 16;
/// The weights, row-major, a layer after another: hidden x in, hidden x hidden, 3 x hidden.
public static const uint kNeuralLayer0 = 0;
public static const uint kNeuralLayer1 = kNeuralHidden * kNeuralIn;
public static const uint kNeuralLayer2 = kNeuralLayer1 + kNeuralHidden * kNeuralHidden;
public static const uint kNeuralWeights = kNeuralLayer2 + 3 * kNeuralHidden;

/// What one evaluation held, for the backward pass to walk.
public struct NeuralPass {
    public float x[kNeuralIn];
    public float h0[kNeuralHidden];
    public float h1[kNeuralHidden];
    public float3 squashed;    // tanh of the network's output
    public float3 colour;      // max(base + squashed, 0)
};

/// The real spherical harmonics of degrees 0 to 3 along a unit direction.
void harmonics(float3 d, out float sh[16]) {
    const float x = d.x, y = d.y, z = d.z;
    const float xx = x * x, yy = y * y, zz = z * z;
    sh[0] = 0.2820947918;
    sh[1] = -0.4886025119 * y;
    sh[2] = 0.4886025119 * z;
    sh[3] = -0.4886025119 * x;
    sh[4] = 1.0925484306 * x * y;
    sh[5] = -1.0925484306 * y * z;
    sh[6] = 0.3153915653 * (2.0 * zz - xx - yy);
    sh[7] = -1.0925484306 * x * z;
    sh[8] = 0.5462742153 * (xx - yy);
    sh[9] = -0.5900435899 * y * (3.0 * xx - yy);
    sh[10] = 2.8906114426 * x * y * z;
    sh[11] = -0.4570457995 * y * (4.0 * zz - xx - yy);
    sh[12] = 0.3731763326 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    sh[13] = -0.4570457995 * x * (4.0 * zz - xx - yy);
    sh[14] = 1.4453057213 * z * (xx - yy);
    sh[15] = -0.5900435899 * x * (xx - 3.0 * yy);
}

/// A gaussian's colour seen along `direction` (unit, from the eye to it, in
/// the cloud's own space). `latent` holds kNeuralStride numbers a gaussian.
public NeuralPass neuralForward(StructuredBuffer<float> latent, uint gaussian, StructuredBuffer<float> weights,
                                float3 direction) {
    NeuralPass pass;
    float sh[16];
    harmonics(direction, sh);
    for (uint k = 0; k < 16; ++k) {
        pass.x[k] = sh[k];
    }
    const uint row = gaussian * kNeuralStride;
    for (uint k = 0; k < kNeuralFeatures; ++k) {
        const float z = latent[row + k];
        pass.x[16 + k] = sin(z);
        pass.x[16 + kNeuralFeatures + k] = cos(z);
    }
    for (uint r = 0; r < kNeuralHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralIn; ++c) {
            s += weights[kNeuralLayer0 + r * kNeuralIn + c] * pass.x[c];
        }
        pass.h0[r] = max(s, 0.0);
    }
    for (uint r = 0; r < kNeuralHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralHidden; ++c) {
            s += weights[kNeuralLayer1 + r * kNeuralHidden + c] * pass.h0[c];
        }
        pass.h1[r] = max(s, 0.0);
    }
    float3 o = float3(0.0);
    for (uint r = 0; r < 3; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralHidden; ++c) {
            s += weights[kNeuralLayer2 + r * kNeuralHidden + c] * pass.h1[c];
        }
        o[r] = s;
    }
    pass.squashed = tanh(o);
    const float3 base = float3(latent[row + kNeuralFeatures], latent[row + kNeuralFeatures + 1],
                               latent[row + kNeuralFeatures + 2]);
    pass.colour = max(base + pass.squashed, float3(0.0));
    return pass;
}

/// The derivatives of a loss whose derivative with respect to the colour is
/// `dColour`: what each layer's error is, for whoever sums the weights'
/// gradients (weight[r][c] of a layer: its error at r times its input at c),
/// and the gaussian's own -- its features' and its base's.
public struct NeuralBack {
    public float3 dOut;                    // layer 2's error
    public float  d1[kNeuralHidden];       // layer 1's
    public float  d0[kNeuralHidden];       // layer 0's
    public float  dLatent[kNeuralStride];  // the features', then the base's
};

public NeuralBack neuralBackward(NeuralPass pass, StructuredBuffer<float> weights, float3 dColour) {
    NeuralBack back;
    // Through the clamp at zero, then the base (as it is) and the tanh.
    const float3 through = float3(pass.colour.x > 0.0 ? dColour.x : 0.0, pass.colour.y > 0.0 ? dColour.y : 0.0,
                                  pass.colour.z > 0.0 ? dColour.z : 0.0);
    back.dOut = through * (float3(1.0) - pass.squashed * pass.squashed);
    for (uint c = 0; c < kNeuralHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < 3; ++r) {
            s += weights[kNeuralLayer2 + r * kNeuralHidden + c] * back.dOut[r];
        }
        back.d1[c] = pass.h1[c] > 0.0 ? s : 0.0;
    }
    for (uint c = 0; c < kNeuralHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kNeuralHidden; ++r) {
            s += weights[kNeuralLayer1 + r * kNeuralHidden + c] * back.d1[r];
        }
        back.d0[c] = pass.h0[c] > 0.0 ? s : 0.0;
    }
    for (uint k = 0; k < kNeuralFeatures; ++k) {
        float dSin = 0.0;
        float dCos = 0.0;
        for (uint r = 0; r < kNeuralHidden; ++r) {
            dSin += weights[kNeuralLayer0 + r * kNeuralIn + 16 + k] * back.d0[r];
            dCos += weights[kNeuralLayer0 + r * kNeuralIn + 16 + kNeuralFeatures + k] * back.d0[r];
        }
        // x = sin z and cos z, which the pass kept.
        back.dLatent[k] = dSin * pass.x[16 + kNeuralFeatures + k] - dCos * pass.x[16 + k];
    }
    back.dLatent[kNeuralFeatures] = through.x;
    back.dLatent[kNeuralFeatures + 1] = through.y;
    back.dLatent[kNeuralFeatures + 2] = through.z;
    return back;
}
