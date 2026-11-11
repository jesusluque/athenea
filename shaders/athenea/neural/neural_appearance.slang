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
// 11 numbers a gaussian where harmonics of degree 3 are 48. With the features
// and the last layer at zero the colour is the base, which is where a fit
// starts from.
//
// LIT, the same gaussian answers a light instead of holding one: the network
// also takes the direction the light comes from (its harmonics of degrees 0
// to 2, nine numbers), and what it gives is how the base -- now what the
// gaussian is made of, not what it shone with -- answers a unit of light from
// there,
//
//     colour = radiance * base * (reached * direct + bounced)
//     direct = softplus(first three + c), softplus(c) = 1;  bounced = softplus(last three - 3)
//
// which is linear in the light as light is: two lights are two evaluations
// summed. `reached` is how much of the light arrives at the gaussian, which
// the engine says (a shadow ray, a shadow map; 1 where nobody does), and it
// multiplies only the first answer: what a floor throws back into a shadow,
// what comes round a corner, is light of that light that did not have to
// arrive, and a product with `reached` alone put those places out (the neck
// under the pawn's head, black where the path tracer has it pink). The same
// weights' layout serves both; unlit, the nine are zero and the last three
// outputs are not read.
//
// The light's direction is given in the gaussian's own frame
// (`neuralLocalLight`): its third coordinate is along the gaussian's normal --
// its shortest axis, turned towards the eye as the engine's relighting turns
// it -- so that how a surface answers a light from in front of it, behind it
// or along it is one thing for the network to learn and not one a gaussian.
// Given in the cloud's space, the network had each gaussian's facing to find
// in eight features, and learnt the mean of all lights instead.
//
// The backward pass is written out here, not left to Slang's automatic
// differentiation: of a network written with arrays in loops, Slang
// 2026.14.1's is not the derivative (docs/decisions.md).
module neural_appearance;
import athenea.common.packing;

public static const uint kNeuralFeatures = 8;
/// The lights a frame can draw a lit cloud under.
public static const uint kNeuralLights = 8;
/// A gaussian's numbers: its features, then its base colour (linear light).
public static const uint kNeuralStride = kNeuralFeatures + 3;
/// The view's harmonics, the features' sines and cosines, the light's harmonics.
public static const uint kNeuralLightAt = 16 + kNeuralFeatures * 2;
public static const uint kNeuralIn = kNeuralLightAt + 9;
/// softplus(kNeuralSoft) = 1: a lit gaussian that has learnt nothing answers with its base.
public static const float kNeuralSoft = 0.5413248546;
public static const uint kNeuralHidden = 16;
/// The weights, row-major, a layer after another: hidden x in, hidden x hidden, 3 x hidden.
public static const uint kNeuralLayer0 = 0;
public static const uint kNeuralLayer1 = kNeuralHidden * kNeuralIn;
public static const uint kNeuralLayer2 = kNeuralLayer1 + kNeuralHidden * kNeuralHidden;
/// What the network gives: three numbers for the light that arrives, three for the light that does
/// not have to (the second three are a lit cloud's alone).
public static const uint kNeuralOut = 6;
public static const uint kNeuralWeights = kNeuralLayer2 + kNeuralOut * kNeuralHidden;
/// softplus(-kNeuralDim) is a twentieth: where a lit gaussian's second answer starts.
public static const float kNeuralDim = 3.0;

/// WHERE THE NETWORK'S WEIGHTS ARE READ FROM. A frame reads all of them once
/// a gaussian in view, and reading them from the buffer each time is most of
/// what the network costs (seven milliseconds of a frame on an RTX 3070 for
/// 120 thousand gaussians); a kernel that shades may keep them in its group's
/// memory instead, read once a group. So the passes take whichever.
public interface INeuralWeights {
    float at(uint index);
}

/// TWO NETWORKS FOR ONE CLOUD (a split cloud). What a gaussian looks like has
/// a part that is the same from everywhere -- what it is made of, and the
/// light that falls on it -- and a part that turns with the eye: a highlight,
/// a reflection. One network that is given the view cannot keep them apart,
/// so nothing it says can be kept from one camera to the next. A split cloud
/// has two of the same shape: the first is never told the view (`kNeuralMatt`)
/// and its answer is the colour as one network's is; the second is
/// (`kNeuralShine`) and its answer is what the view adds. The first's answer
/// is then a gaussian's until the gaussian or its light changes.
public static const uint kNeuralWhole = 0;
public static const uint kNeuralMatt = 1;
public static const uint kNeuralShine = 2;

/// A network's weights that start `offset` into another's: a split cloud's second.
public struct NeuralWeightsFrom<W : INeuralWeights> : INeuralWeights {
    public W    inner;
    public uint offset;
    public float at(uint index) { return inner.at(offset + index); }
}

/// The weights where they live: the cloud's buffer.
public struct NeuralWeightBuffer : INeuralWeights {
    public StructuredBuffer<float> buffer;
    public float at(uint index) { return buffer[index]; }
}

/// What one evaluation held, for the backward pass to walk.
public struct NeuralPass {
    public float x[kNeuralIn];
    public float h0[kNeuralHidden];
    public float h1[kNeuralHidden];
    public float3 colour;      // before the light's radiance, where lit
    public float3 perOutput;   // the colour's derivative with respect to the network's first three outputs
    public float3 perSecond;   // with respect to its last three (zero where not lit)
    public float3 perBase;     // and with respect to the base
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

/// Where a light comes from, in a gaussian's own frame: along its two longer
/// axes, then along its normal (its shortest axis, towards `eye`). `light` is
/// a unit vector towards the light in the cloud's space, as `eye` and the
/// cloud's positions are.
/// `faceEye` false: the normal as the gaussian's own axes have it, whichever
/// side that is -- what a split cloud's matt part is told, since where the
/// eye is cannot enter what is kept from camera to camera; which side is out
/// is then the gaussian's features' to say.
public float3 neuralLocalLight(StructuredBuffer<uint> shape, uint gaussian, float3 position, float3 eye,
                               float3 light, bool faceEye = true) {
    const uint row = gaussian * 4;
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float3 scale = float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2]));   // logs
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    const uint a = (shortest + 1u) % 3u;
    const uint b = (shortest + 2u) % 3u;
    float3 normal = float3(axes[0][shortest], axes[1][shortest], axes[2][shortest]);
    if (faceEye && dot(normal, eye - position) < 0.0) {
        normal = -normal;
    }
    return float3(dot(light, float3(axes[0][a], axes[1][a], axes[2][a])),
                  dot(light, float3(axes[0][b], axes[1][b], axes[2][b])), dot(light, normal));
}

/// A gaussian's colour seen along `direction` (unit, from the eye to it, in
/// the cloud's own space). `latent` holds kNeuralStride numbers a gaussian.
/// `lit`: the colour is the answer to a unit of light arriving from `light`
/// (unit, towards the light, in the gaussian's own frame: `neuralLocalLight`)
/// rather than the gaussian's own.
public NeuralPass neuralForward<W : INeuralWeights>(StructuredBuffer<float> latent, uint gaussian, W weights,
                                                    float3 direction, bool lit, float3 light, float reached,
                                                    uint part = kNeuralWhole) {
    NeuralPass pass;
    float sh[16];
    harmonics(direction, sh);
    for (uint k = 0; k < 16; ++k) {
        // The matt part is not told where it is seen from.
        pass.x[k] = part == kNeuralMatt ? 0.0 : sh[k];
    }
    const uint row = gaussian * kNeuralStride;
    for (uint k = 0; k < kNeuralFeatures; ++k) {
        const float z = latent[row + k];
        pass.x[16 + k] = sin(z);
        pass.x[16 + kNeuralFeatures + k] = cos(z);
    }
    float towards[16];
    harmonics(lit ? light : float3(0.0, 0.0, 1.0), towards);
    for (uint k = 0; k < 9; ++k) {
        pass.x[kNeuralLightAt + k] = lit ? towards[k] : 0.0;
    }
    for (uint r = 0; r < kNeuralHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralIn; ++c) {
            s += weights.at(kNeuralLayer0 + r * kNeuralIn + c) * pass.x[c];
        }
        pass.h0[r] = max(s, 0.0);
    }
    for (uint r = 0; r < kNeuralHidden; ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralHidden; ++c) {
            s += weights.at(kNeuralLayer1 + r * kNeuralHidden + c) * pass.h0[c];
        }
        pass.h1[r] = max(s, 0.0);
    }
    float3 o = float3(0.0);
    float3 second = float3(0.0);
    for (uint r = 0; r < (lit ? kNeuralOut : 3u); ++r) {
        float s = 0.0;
        for (uint c = 0; c < kNeuralHidden; ++c) {
            s += weights.at(kNeuralLayer2 + r * kNeuralHidden + c) * pass.h1[c];
        }
        if (r < 3) {
            o[r] = s;
        } else {
            second[r - 3] = s;
        }
    }
    const float3 base = float3(latent[row + kNeuralFeatures], latent[row + kNeuralFeatures + 1],
                               latent[row + kNeuralFeatures + 2]);
    if (part == kNeuralShine) {
        // What the view adds, and nothing else: its first three outputs alone. Lit, light that arrived
        // and left towards the eye, so never less than none and starting at a twentieth; unlit, a
        // difference either way from what the matt part said.
        pass.perSecond = float3(0.0);
        if (lit) {
            const float3 dimmed = o - kNeuralDim;
            const float3 shine = max(dimmed, float3(0.0)) + log(1.0 + exp(-abs(dimmed)));
            pass.colour = base * reached * shine;
            pass.perOutput = base * reached / (1.0 + exp(-dimmed));
            pass.perBase = reached * shine;
        } else {
            const float3 squashed = tanh(o);
            pass.colour = squashed;
            pass.perOutput = float3(1.0) - squashed * squashed;
            pass.perBase = float3(0.0);
        }
    } else if (lit) {
        const float3 shifted = o + kNeuralSoft;
        // softplus, written so that a large argument does not overflow.
        const float3 direct = max(shifted, float3(0.0)) + log(1.0 + exp(-abs(shifted)));
        const float3 dimmed = second - kNeuralDim;
        const float3 bounced = max(dimmed, float3(0.0)) + log(1.0 + exp(-abs(dimmed)));
        const float3 answer = reached * direct + bounced;
        pass.colour = base * answer;
        pass.perOutput = base * reached / (1.0 + exp(-shifted));
        pass.perSecond = base / (1.0 + exp(-dimmed));
        pass.perBase = answer;
    } else {
        const float3 squashed = tanh(o);
        pass.colour = max(base + squashed, float3(0.0));
        // Through the clamp at zero, then the base as it is and the tanh.
        const float3 open = float3(pass.colour.x > 0.0 ? 1.0 : 0.0, pass.colour.y > 0.0 ? 1.0 : 0.0,
                                   pass.colour.z > 0.0 ? 1.0 : 0.0);
        pass.perOutput = open * (float3(1.0) - squashed * squashed);
        pass.perSecond = float3(0.0);
        pass.perBase = open;
    }
    return pass;
}

/// The derivatives of a loss whose derivative with respect to the colour is
/// `dColour`: what each layer's error is, for whoever sums the weights'
/// gradients (weight[r][c] of a layer: its error at r times its input at c),
/// and the gaussian's own -- its features' and its base's.
public struct NeuralBack {
    public float  dOut[kNeuralOut];        // layer 2's error
    public float  d1[kNeuralHidden];       // layer 1's
    public float  d0[kNeuralHidden];       // layer 0's
    public float  dLatent[kNeuralStride];  // the features', then the base's
};

public NeuralBack neuralBackward<W : INeuralWeights>(NeuralPass pass, W weights, float3 dColour) {
    NeuralBack back;
    const float3 first = dColour * pass.perOutput;
    const float3 last = dColour * pass.perSecond;
    for (uint r = 0; r < 3; ++r) {
        back.dOut[r] = first[r];
        back.dOut[3 + r] = last[r];
    }
    const float3 through = dColour * pass.perBase;
    for (uint c = 0; c < kNeuralHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kNeuralOut; ++r) {
            s += weights.at(kNeuralLayer2 + r * kNeuralHidden + c) * back.dOut[r];
        }
        back.d1[c] = pass.h1[c] > 0.0 ? s : 0.0;
    }
    for (uint c = 0; c < kNeuralHidden; ++c) {
        float s = 0.0;
        for (uint r = 0; r < kNeuralHidden; ++r) {
            s += weights.at(kNeuralLayer1 + r * kNeuralHidden + c) * back.d1[r];
        }
        back.d0[c] = pass.h0[c] > 0.0 ? s : 0.0;
    }
    for (uint k = 0; k < kNeuralFeatures; ++k) {
        float dSin = 0.0;
        float dCos = 0.0;
        for (uint r = 0; r < kNeuralHidden; ++r) {
            dSin += weights.at(kNeuralLayer0 + r * kNeuralIn + 16 + k) * back.d0[r];
            dCos += weights.at(kNeuralLayer0 + r * kNeuralIn + 16 + kNeuralFeatures + k) * back.d0[r];
        }
        // x = sin z and cos z, which the pass kept.
        back.dLatent[k] = dSin * pass.x[16 + kNeuralFeatures + k] - dCos * pass.x[16 + k];

    }
    back.dLatent[kNeuralFeatures] = through.x;
    back.dLatent[kNeuralFeatures + 1] = through.y;
    back.dLatent[kNeuralFeatures + 2] = through.z;
    return back;
}
