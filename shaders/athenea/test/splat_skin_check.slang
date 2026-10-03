// Copyright (c) 2026 jesus luque.
//
// The splat skinner's inputs and its answer, both made and checked on the
// device. What it pins is the two invariants a skinned cloud stands on: a
// skeleton at rest leaves every gaussian where it was, and a skeleton that
// moves rigidly carries every gaussian rigidly -- its position by the
// transform, its frame turned with it, and its size not at all.
module splat_skin_check;
import athenea.common.packing;

struct SkinCheckParams {
    uint   count;
    uint   perSplat;
    uint   joints;
    uint   pad0;
    float4 rotation;      ///< the rigid turn, as (x, y, z, w)
    float4 translation;   ///< and the slide, in xyz
    float  tolerance;
    uint   mode;          ///< the strip's: 0 bent across a ramp, 1 rigid, 2 a sheared joint
    float  frameAngle;    ///< the strip's: its gaussians' first axis, from x about z
    uint   pad3;
};

RWStructuredBuffer<float4> restPositions;
RWStructuredBuffer<uint>   restShape;
RWStructuredBuffer<float2> influences;
RWStructuredBuffer<float4> xforms;
StructuredBuffer<float4>   positions;    ///< what the skinner wrote
StructuredBuffer<uint>     shape;
RWStructuredBuffer<uint>   counts;       ///< [compared, moved, resized, turned]
ConstantBuffer<SkinCheckParams> params;

float3x3 turnOf(float4 q) {
    return axesOfQuaternion(q);
}

/// A cloud of gaussians on a helix, each turned differently and each a
/// different size, so that nothing in the check can pass by accident.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinMake(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const float t = float(at) / max(float(params.count), 1.0);
    const float angle = t * 17.0;
    restPositions[at] = float4(cos(angle) * (1.0 + t), t * 4.0 - 2.0, sin(angle) * (1.0 + t), 0.5 + 0.5 * t);

    // A rotation of its own: about an axis that wanders with the index.
    const float3 axis = normalize(float3(0.3 + t, 1.0 - t, 0.2 + t * t));
    const float half = angle * 0.5;
    const float4 q = float4(axis * sin(half), cos(half));
    const uint row = at * 4;
    restShape[row] = encodeQuaternion(q);
    restShape[row + 1] = packHalves(log(0.05 + 0.2 * t), log(0.03 + 0.3 * (1.0 - t)));
    restShape[row + 2] = packHalves(log(1.0e-5), 0.4 + 0.2 * t);
    restShape[row + 3] = packHalves(0.3, 0.6 - 0.2 * t);

    // Every gaussian on the one joint, with all of the weight: what the check
    // is about is the transform, not the blend.
    for (uint k = 0; k < params.perSplat; ++k) {
        influences[at * params.perSplat + k] = float2(0.0, k == 0 ? 1.0 : 0.0);
    }
}

/// Every joint given the same rigid transform, as rows of a 4x4 with the
/// translation in the last column -- the layout the kernel multiplies.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinJoints(uint3 tid: SV_DispatchThreadID) {
    const uint joint = tid.x;
    if (joint >= params.joints) {
        return;
    }
    const float3x3 turn = turnOf(params.rotation);
    xforms[joint * 4 + 0] = float4(turn[0][0], turn[0][1], turn[0][2], params.translation.x);
    xforms[joint * 4 + 1] = float4(turn[1][0], turn[1][1], turn[1][2], params.translation.y);
    xforms[joint * 4 + 2] = float4(turn[2][0], turn[2][1], turn[2][2], params.translation.z);
    xforms[joint * 4 + 3] = float4(0.0, 0.0, 0.0, 1.0);
}

/// What the skinner answered, against what a rigid transform must give.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinCompare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3x3 turn = turnOf(params.rotation);
    const float4 rest = restPositions[at];
    const float3 want = mul(turn, rest.xyz) + params.translation.xyz;
    const float4 got = positions[at];
    if (length(got.xyz - want) > params.tolerance || abs(got.w - rest.w) > 1.0e-6) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint row = at * 4;
    // The two sizes in the plane are the ones that move; the third and the
    // colour are copied, so they are checked as words.
    const float2 wantScale = float2(lowHalf(restShape[row + 1]), highHalf(restShape[row + 1]));
    const float2 gotScale = float2(lowHalf(shape[row + 1]), highHalf(shape[row + 1]));
    if (abs(exp(gotScale.x) - exp(wantScale.x)) > params.tolerance ||
        abs(exp(gotScale.y) - exp(wantScale.y)) > params.tolerance ||
        shape[row + 2] != restShape[row + 2] || shape[row + 3] != restShape[row + 3]) {
        InterlockedAdd(counts[2], 1u);
    }
    // THE FRAME, compared as axes and not as a quaternion: `q` and `-q` are
    // the same turn, and the axes have no such ambiguity.
    const float3x3 was = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3x3 is = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float3x3 wanted = mul(turn, was);
    float worst = 0.0;
    for (uint c = 0; c < 3; ++c) {
        const float3 a = float3(is[0][c], is[1][c], is[2][c]);
        const float3 b = float3(wanted[0][c], wanted[1][c], wanted[2][c]);
        worst = max(worst, length(a - b));
    }
    if (worst > params.tolerance) {
        InterlockedAdd(counts[3], 1u);
    }
}

// WHAT THE SHUTTER MOVED, WRITTEN BESIDE THE POSE.
//
// The displacement the skinner writes is the difference between the two poses
// it was handed, in the cloud's own space -- so with a pure slide on every
// joint it must be exactly that slide, and with the same pose twice it must be
// nothing at all. `translation` is what is expected; `tolerance` is half's,
// since the words are two halves each.
RWStructuredBuffer<uint> motion;   ///< what the skinner wrote

[shader("compute")]
[numthreads(64, 1, 1)]
void splatSkinMotion(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const float3 step = float3(lowHalf(motion[at * 2]), highHalf(motion[at * 2]),
                               lowHalf(motion[at * 2 + 1]));
    InterlockedAdd(counts[0], 1u);
    if (any(abs(step - params.translation.xyz) > params.tolerance)) {
        InterlockedAdd(counts[1], 1u);
    }
}

// THE WHOLE JACOBIAN, ON A STRIP.
//
// A strip of gaussians `x in [-1, 1]`, `y in [-0.2, 0.2]` in z = 0, carried by
// two joints whose weights ramp linearly across `x in [-0.5, 0.5]`
// (`w1 = x + 0.5`, so `grad w1 = (1, 0, 0)` there and zero outside). Joint 0
// stays; joint 1 turns a quarter about y through the origin. The posed map is
// `p' = (1 - w) p + w R p`, whose Jacobian is
//
//     J = (1 - w) I + w R + (R p - p) grad w^T,
//
// and a gaussian whose axes `E` and sizes `S` were its rest covariance must be
// posed as `J E S^2 E^T J^T`. The blend of the joints' linear parts alone --
// the first two terms -- is a third short of it at the middle of the ramp.
// Made, posed and compared on the device; one counter comes back.
//
// Modes: 0 that bend; 1 a control, both joints the same rigid turn and slide
// (the gradient term cancels, before as after); 2 a joint that shears, the
// weights a constant half each, where squaring the axes up threw the shear
// away. The gaussians' first axis is turned `frameAngle` from x in the plane,
// so a bend shears them as well.
RWStructuredBuffer<uint> gradients;   ///< three words of two halves a gaussian

static const uint kStripAcross = 256;

float stripWeight(float x) {
    return params.mode == 2 ? 0.5 : clamp(x + 0.5, 0.0, 1.0);
}

float3 stripGradient(float x) {
    return params.mode != 2 && abs(x) < 0.5 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void splatStripMake(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint rows = max(params.count / kStripAcross, 1u);
    const float x = -1.0 + 2.0 * (float(at % kStripAcross) + 0.5) / float(kStripAcross);
    const float y = -0.2 + 0.4 * (float(at / kStripAcross) + 0.5) / float(rows);
    restPositions[at] = float4(x, y, 0.0, 1.0);
    const float halfAngle = params.frameAngle * 0.5;
    const float4 q = float4(0.0, 0.0, sin(halfAngle), cos(halfAngle));
    const uint row = at * 4;
    restShape[row] = encodeQuaternion(q);
    restShape[row + 1] = packHalves(log(0.02), log(0.01));
    restShape[row + 2] = packHalves(log(1.0e-3), 0.5);
    restShape[row + 3] = packHalves(0.5, 0.5);

    const float w = stripWeight(x);
    influences[at * params.perSplat + 0] = float2(0.0, 1.0 - w);
    influences[at * params.perSplat + 1] = float2(1.0, w);
    for (uint k = 2; k < params.perSplat; ++k) {
        influences[at * params.perSplat + k] = float2(0.0, 0.0);
    }
    // The gradients in the gaussian's own rest axes, as a conversion keeps
    // them: joint 1's, joint 0's minus it, nothing for the empty third.
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float3 restU = float3(axes[0][0], axes[1][0], axes[2][0]);
    const float3 restV = float3(axes[0][1], axes[1][1], axes[2][1]);
    const float3 grad = stripGradient(x);
    const float2 g1 = float2(dot(grad, restU), dot(grad, restV));
    gradients[at * 3 + 0] = packHalves(-g1.x, -g1.y);
    gradients[at * 3 + 1] = packHalves(g1.x, g1.y);
    gradients[at * 3 + 2] = packHalves(0.0, 0.0);
}

/// Joint `j`'s transform in the strip's mode, rows of a 4x4 as the kernel
/// multiplies them.
float3x4 stripJoint(uint joint) {
    if (params.mode == 1) {
        const float3x3 turn = turnOf(params.rotation);
        return float3x4(float4(turn[0], params.translation.x), float4(turn[1], params.translation.y),
                        float4(turn[2], params.translation.z));
    }
    if (joint == 0) {
        return float3x4(float4(1.0, 0.0, 0.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(0.0, 0.0, 1.0, 0.0));
    }
    if (params.mode == 2) {
        return float3x4(float4(1.0, 0.8, 0.0, 0.1), float4(0.0, 1.2, 0.0, 0.0), float4(0.0, 0.3, 1.0, 0.0));
    }
    // A quarter turn about y: x goes to -z, z to x.
    return float3x4(float4(0.0, 0.0, 1.0, 0.0), float4(0.0, 1.0, 0.0, 0.0), float4(-1.0, 0.0, 0.0, 0.0));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void splatStripJoints(uint3 tid: SV_DispatchThreadID) {
    const uint joint = tid.x;
    if (joint >= params.joints) {
        return;
    }
    const float3x4 m = stripJoint(min(joint, 1u));
    xforms[joint * 4 + 0] = m[0];
    xforms[joint * 4 + 1] = m[1];
    xforms[joint * 4 + 2] = m[2];
    xforms[joint * 4 + 3] = float4(0.0, 0.0, 0.0, 1.0);
}

/// The posed covariance the skinner wrote, against `J E S^2 E^T J^T`.
/// counts: [compared, off by more than `tolerance` in relative Frobenius].
[shader("compute")]
[numthreads(64, 1, 1)]
void splatStripCompare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float3 p = restPositions[at].xyz;
    const float w = stripWeight(p.x);
    const float3x4 x0 = stripJoint(0);
    const float3x4 x1 = stripJoint(1);
    const float3x3 a0 = float3x3(x0[0].xyz, x0[1].xyz, x0[2].xyz);
    const float3x3 a1 = float3x3(x1[0].xyz, x1[1].xyz, x1[2].xyz);
    const float3 apart = mul(x1, float4(p, 1.0)) - mul(x0, float4(p, 1.0));
    const float3 grad = stripGradient(p.x);
    // `apart grad^T`: built by rows as `grad_i * apart`, which is its
    // transpose, and turned back.
    const float3x3 elastic = float3x3(apart * grad.x, apart * grad.y, apart * grad.z);
    const float3x3 jacobian = a0 * (1.0 - w) + a1 * w + transpose(elastic);

    const uint row = at * 4;
    const float3x3 was = axesOfQuaternion(decodeQuaternion(restShape[row]));
    const float2 restSize = exp(float2(lowHalf(restShape[row + 1]), highHalf(restShape[row + 1])));
    const float3 wantU = mul(jacobian, float3(was[0][0], was[1][0], was[2][0])) * restSize.x;
    const float3 wantV = mul(jacobian, float3(was[0][1], was[1][1], was[2][1])) * restSize.y;
    const float3x3 is = axesOfQuaternion(decodeQuaternion(shape[row]));
    const float2 size = exp(float2(lowHalf(shape[row + 1]), highHalf(shape[row + 1])));
    const float3 gotU = float3(is[0][0], is[1][0], is[2][0]) * size.x;
    const float3 gotV = float3(is[0][1], is[1][1], is[2][1]) * size.y;
    float difference = 0.0;
    float norm = 0.0;
    for (uint r = 0; r < 3; ++r) {
        for (uint c = 0; c < 3; ++c) {
            const float want = wantU[r] * wantU[c] + wantV[r] * wantV[c];
            const float got = gotU[r] * gotU[c] + gotV[r] * gotV[c];
            difference += (got - want) * (got - want);
            norm += want * want;
        }
    }
    if (!(sqrt(difference) <= params.tolerance * sqrt(norm))) {
        InterlockedAdd(counts[1], 1u);
    }
}
