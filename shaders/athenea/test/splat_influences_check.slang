// Copyright (c) 2026 jesus luque.
//
// A rig's influences packed on the device, against the pairs the host used
// to make. Everything is made here and compared here: the file's two arrays
// (joint indices and weights, a record each or one set for all), the record
// each kept gaussian came from, the pairs the host's loop wrote -- the same
// loop, as a kernel -- and, at the end, what the packer wrote against them.
// Only counters go back.
module splat_influences_check;
import athenea.common.packing;
import athenea.common.influences;

struct InfluenceCheckParams {
    uint  count;        ///< gaussians kept
    uint  declared;     ///< records in the file
    uint  perSplat;
    uint  held;         ///< entries in each array: declared x perSplat, or perSplat if constant
    uint  joints;
    uint  badRecord;    ///< the record whose `badSlot` names a joint the skeleton has not (~0: none)
    uint  badSlot;
    uint  layout;       ///< what `packed` holds: kInfluencePacked or kInfluencePairs
    float positionTolerance;
    float sizeTolerance;
    float axisTolerance;
    uint  pad0;
};

RWStructuredBuffer<int>    jointIndices;
RWStructuredBuffer<float>  jointWeights;
RWStructuredBuffer<uint>   origin;
RWStructuredBuffer<float2> pairsByRecord;   ///< the host's old loop, a record each
StructuredBuffer<float2>   reference;       ///< those, packed to the kept gaussians (CloudLoader::keptOnly)
StructuredBuffer<uint>     packed;          ///< what the packer wrote, in `layout`
RWStructuredBuffer<uint>   counts;
ConstantBuffer<InfluenceCheckParams> params;

uint hashOf(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float unitOf(uint x) {
    return float(hashOf(x) & 0xFFFFFFu) / 16777216.0;
}

/// The file's arrays, one thread a set of `perSplat`: joints anywhere in the
/// skeleton, weights of every size that add up to one, a pad (-1, weight 0)
/// in the last slot of every third set, and one joint past the skeleton.
[shader("compute")]
[numthreads(64, 1, 1)]
void infMake(uint3 tid: SV_DispatchThreadID) {
    const uint set = tid.x;
    if (set * params.perSplat >= params.held) {
        return;
    }
    const bool padded = set % 3u == 0u && params.perSplat > 1u;
    const uint weighted = padded ? params.perSplat - 1u : params.perSplat;
    float total = 0.0;
    for (uint k = 0; k < weighted; ++k) {
        // Some weights far under a step, some near a whole.
        const float u = unitOf(set * 31u + k * 7u + 1u);
        total += k == 0u ? 0.2 + u : u * u * u;
    }
    for (uint k = 0; k < params.perSplat; ++k) {
        const uint e = set * params.perSplat + k;
        int joint = int(hashOf(e * 13u + 5u) % max(params.joints, 1u));
        float weight = 0.0;
        if (k < weighted) {
            const float u = unitOf(set * 31u + k * 7u + 1u);
            weight = (k == 0u ? 0.2 + u : u * u * u) / total;
        } else {
            joint = -1;
        }
        if (set == params.badRecord && k == params.badSlot) {
            joint = int(params.joints + 3u);
            weight = 0.0;
        }
        jointIndices[e] = joint;
        jointWeights[e] = weight;
    }
}

/// Validation's choice, as a decode would make one: every other record and
/// a half, so a gaussian's record is never its own index.
[shader("compute")]
[numthreads(64, 1, 1)]
void infOrigin(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    origin[j] = min(2u * j + (j & 1u), params.declared - 1u);
}

/// THE HOST'S OLD LOOP: `pairs[k] = (max(index[k % held], 0), weight[k % held])`
/// over every record of the file, before keptOnly packed them to the kept.
[shader("compute")]
[numthreads(64, 1, 1)]
void infOldPairs(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    if (e >= params.declared * params.perSplat) {
        return;
    }
    const uint from = e % max(params.held, 1u);
    pairsByRecord[e] = float2(float(max(jointIndices[from], 0)), jointWeights[from]);
}

/// What the packer wrote, against the old pairs.
///   [0] influences compared
///   [1] a joint that is not the old one
///   [2] a weight more than a step from the old one (packed), or not the same
///       bits (pairs)
///   [3] gaussians whose packed weights do not add up to the old ones', to
///       half a step
///   [4] slots naming a joint past the skeleton
///   [5] of those, slots that hold anything but nothing
[shader("compute")]
[numthreads(64, 1, 1)]
void infCompare(uint3 tid: SV_DispatchThreadID) {
    const uint j = tid.x;
    if (j >= params.count) {
        return;
    }
    const uint first = j * params.perSplat;
    float oldSum = 0.0;
    float steps = 0.0;
    for (uint k = 0; k < params.perSplat; ++k) {
        InterlockedAdd(counts[0], 1u);
        const float2 was = reference[first + k];
        const float2 is = influenceAt(packed, params.layout, first, k);
        if (uint(was.x) >= params.joints) {
            InterlockedAdd(counts[4], 1u);
            if (is.x != 0.0 || is.y != 0.0) {
                InterlockedAdd(counts[5], 1u);
            }
            continue;
        }
        oldSum += was.y;
        if (is.x != was.x) {
            InterlockedAdd(counts[1], 1u);
        }
        if (params.layout == kInfluencePacked) {
            const float held = float(packed[first + k] >> 16);
            steps += held;
            if (abs(held - was.y * kWeightSteps) > 1.0 + 1.0e-3) {
                InterlockedAdd(counts[2], 1u);
            }
        } else if (asuint(is.y) != asuint(was.y)) {
            InterlockedAdd(counts[2], 1u);
        }
    }
    if (params.layout == kInfluencePacked && abs(steps - oldSum * kWeightSteps) > 0.5 + 1.0e-2) {
        InterlockedAdd(counts[3], 1u);
    }
}

// ----------------------------------------------------------------------------
// The same cloud skinned from the two layouts.

RWStructuredBuffer<float4> restPositions;
RWStructuredBuffer<uint>   restShape;
RWStructuredBuffer<float4> xforms;
RWStructuredBuffer<uint>   gradients;      ///< perSplat - 1 words a gaussian
StructuredBuffer<float4>   positionsA;     ///< skinned from the pairs
StructuredBuffer<uint>     shapeA;
StructuredBuffer<float4>   positionsB;     ///< skinned from the packed words
StructuredBuffer<uint>     shapeB;

/// A cloud on a helix some four units across, each gaussian turned and sized
/// its own way, and its weights' gradients across it.
[shader("compute")]
[numthreads(64, 1, 1)]
void infRest(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const float t = float(at) / max(float(params.count), 1.0);
    const float angle = t * 17.0;
    restPositions[at] = float4(cos(angle) * (1.0 + t), t * 4.0 - 2.0, sin(angle) * (1.0 + t), 0.5 + 0.5 * t);
    const float3 axis = normalize(float3(0.3 + t, 1.0 - t, 0.2 + t * t));
    const float half = angle * 0.5;
    const uint row = at * 4;
    restShape[row] = encodeQuaternion(float4(axis * sin(half), cos(half)));
    restShape[row + 1] = packHalves(log(0.05 + 0.2 * t), log(0.03 + 0.3 * (1.0 - t)));
    restShape[row + 2] = packHalves(log(1.0e-5), 0.4 + 0.2 * t);
    restShape[row + 3] = packHalves(0.3, 0.6 - 0.2 * t);
    const uint words = params.perSplat > 0u ? params.perSplat - 1u : 0u;
    for (uint k = 0; k < words; ++k) {
        gradients[at * words + k] = packHalves(unitOf(at * 11u + k) * 2.0 - 1.0, unitOf(at * 17u + k + 3u) * 2.0 - 1.0);
    }
}

/// Every joint its own turn and slide, rows of a 4x4 with the translation in
/// the last column, as the skinner takes them.
[shader("compute")]
[numthreads(64, 1, 1)]
void infJoints(uint3 tid: SV_DispatchThreadID) {
    const uint joint = tid.x;
    if (joint >= params.joints) {
        return;
    }
    const float3 axis = normalize(float3(unitOf(joint * 3u) - 0.5, unitOf(joint * 3u + 1u) - 0.5, 0.5));
    const float half = 0.6 * unitOf(joint * 3u + 2u);
    const float3x3 turn = axesOfQuaternion(float4(axis * sin(half), cos(half)));
    const float3 slide = float3(unitOf(joint + 40u), unitOf(joint + 41u), unitOf(joint + 42u)) - 0.5;
    xforms[joint * 4 + 0] = float4(turn[0][0], turn[0][1], turn[0][2], slide.x);
    xforms[joint * 4 + 1] = float4(turn[1][0], turn[1][1], turn[1][2], slide.y);
    xforms[joint * 4 + 2] = float4(turn[2][0], turn[2][1], turn[2][2], slide.z);
    xforms[joint * 4 + 3] = float4(0.0, 0.0, 0.0, 1.0);
}

/// The two posed clouds against each other.
///   [0] gaussians compared
///   [1] positions apart by more than `positionTolerance`, or opacity changed
///   [2] in-plane sizes apart by more than `sizeTolerance`, relatively, or the
///       copied words not the same
///   [3] frames whose axes are apart by more than `axisTolerance`
///   [6] the largest position difference, as a float's bits
[shader("compute")]
[numthreads(64, 1, 1)]
void infSkinCompare(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    InterlockedAdd(counts[0], 1u);
    const float4 a = positionsA[at];
    const float4 b = positionsB[at];
    const float apart = length(a.xyz - b.xyz);
    InterlockedMax(counts[6], asuint(apart));
    if (!(apart <= params.positionTolerance) || a.w != b.w) {
        InterlockedAdd(counts[1], 1u);
    }
    const uint row = at * 4;
    const float2 sizeA = exp(float2(lowHalf(shapeA[row + 1]), highHalf(shapeA[row + 1])));
    const float2 sizeB = exp(float2(lowHalf(shapeB[row + 1]), highHalf(shapeB[row + 1])));
    const float2 relative = abs(sizeA - sizeB) / max(sizeA, float2(1.0e-20, 1.0e-20));
    if (!(max(relative.x, relative.y) <= params.sizeTolerance) || shapeA[row + 2] != shapeB[row + 2] ||
        shapeA[row + 3] != shapeB[row + 3]) {
        InterlockedAdd(counts[2], 1u);
    }
    const float3x3 axesA = axesOfQuaternion(decodeQuaternion(shapeA[row]));
    const float3x3 axesB = axesOfQuaternion(decodeQuaternion(shapeB[row]));
    float worst = 0.0;
    for (uint c = 0; c < 3; ++c) {
        worst = max(worst, length(float3(axesA[0][c], axesA[1][c], axesA[2][c]) -
                                  float3(axesB[0][c], axesB[1][c], axesB[2][c])));
    }
    if (!(worst <= params.axisTolerance)) {
        InterlockedAdd(counts[3], 1u);
    }
}
