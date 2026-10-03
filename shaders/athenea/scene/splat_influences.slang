// Copyright (c) 2026 jesus luque.
//
// A RIG'S JOINTS AND WEIGHTS, AS THE FILE HOLDS THEM, INTO WHAT THE SKINNER
// READS. One thread a kept gaussian.
//
// UsdSkel keeps a gaussian's influences in two arrays, `jointIndices` (int)
// and `jointWeights` (float), `perSplat` each a record of the file -- or one
// set for every record, where the binding is constant. They come up as the
// file holds them, and this does in one pass what the host did in two: it
// pairs each joint with its weight and packs them to the gaussians validation
// kept (`origin`, as splat_kept does for anything else a record carries).
// The host used to interleave them into a float2 each -- 137 MB on the
// sparrow, on the host and again on the device -- and now copies bytes.
//
// Written in either layout of common/influences.slang. The packed one rounds
// the weights' running sum, not each weight, so the steps a gaussian holds add
// up to what its weights did, to one step.
//
// What is wrong in a file is fixed here and counted, never read past:
//   - a negative joint is joint 0 (UsdSkel's pad value is -1 with weight 0);
//   - a joint the skeleton does not have holds nothing (weight 0, joint 0)
//     and is counted in `counters[0]` -- it used to index past the
//     skeleton's transforms;
//   - a weight outside [0, 1], or not a number, is counted in `counters[1]`;
//     the packed layout holds it clamped, the pairs one as it was.
module splat_influences;
import athenea.common.influences;

struct InfluencePackParams {
    uint count;      ///< gaussians kept
    uint perSplat;   ///< influences a gaussian
    uint held;       ///< entries in each of the two arrays: count x perSplat, or perSplat if constant
    uint joints;     ///< joints the skeleton has
    uint layout;     ///< kInfluencePairs or kInfluencePacked
    uint pad0;
    uint pad1;
    uint pad2;
};

StructuredBuffer<uint>    origin;          // the record each kept gaussian came from
StructuredBuffer<int>     jointIndices;    // as the file holds them
StructuredBuffer<float>   jointWeights;
RWStructuredBuffer<uint>  influences;      // influenceWords(layout) x perSplat a kept gaussian
RWStructuredBuffer<uint>  counters;        // [joints out of range, weights out of range]
ConstantBuffer<InfluencePackParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatInfluencesPack(uint3 tid: SV_DispatchThreadID) {
    const uint at = tid.x;
    if (at >= params.count) {
        return;
    }
    const uint record = origin[at];
    const uint first = at * params.perSplat;
    float sum = 0.0;
    uint stepsSoFar = 0u;
    for (uint k = 0; k < params.perSplat; ++k) {
        // The same index the host's loop took (`k % held` over the flat
        // array), which is what makes a constant binding one set for all.
        const uint from = (record * params.perSplat + k) % max(params.held, 1u);
        uint joint = uint(max(jointIndices[from], 0));
        float weight = jointWeights[from];
        if (!(weight >= 0.0 && weight <= 1.0)) {
            InterlockedAdd(counters[1], 1u);
        }
        if (joint >= params.joints) {
            InterlockedAdd(counters[0], 1u);
            joint = 0u;
            weight = 0.0;
        }
        if (params.layout == kInfluencePacked) {
            // The running sum, rounded, less what the earlier influences took:
            // each weight within a step of its own, and the whole within half
            // a step of the file's.
            sum += (weight >= 0.0) ? weight : 0.0;
            const uint upTo = uint(floor(saturate(sum) * kWeightSteps + 0.5));
            const uint steps = upTo > stepsSoFar ? upTo - stepsSoFar : 0u;
            stepsSoFar = max(upTo, stepsSoFar);
            influences[first + k] = packInfluence(joint, steps);
        } else {
            influences[(first + k) * 2] = asuint(float(joint));
            influences[(first + k) * 2 + 1] = asuint(weight);
        }
    }
}
