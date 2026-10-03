// Copyright (c) 2026 jesus luque.
//
// HOW A GAUSSIAN KEEPS THE JOINTS THAT MOVE IT, on the device.
//
// Two layouts, read through one buffer of uint words:
//
//   packed  1 word an influence: joint u16 (low) | weight unorm16 (high).
//           What a cloud is kept in when its skeleton has no more than
//           65536 joints -- every one this engine has seen. 16 bytes a
//           gaussian for four, half of what the pairs cost.
//   pairs   2 words an influence: float joint, float weight (a float2). What
//           a skeleton too large for 16 bits keeps, and what a caller that
//           made its own pairs (the tests, the bake's own buffers) hands over.
//
// The weight's sixteen bits are a step of 1.5e-5. `splatInfluencesPack`
// rounds the running sum of a gaussian's weights rather than each weight, so
// the steps add up to the sum the file held, to one step: a gaussian whose
// weights sum to one keeps a sum of exactly one, and is not pulled toward the
// skeleton's origin by its rounding (see decisions.md, «A rig's joints,
// packed on the device»).
//
// A u16 in a structured buffer is not portable between Metal and CUDA, which
// is why this is shifts on a uint and not `uint16_t`.
module influences;

public static const uint kInfluencePairs = 0u;
public static const uint kInfluencePacked = 1u;
public static const float kWeightSteps = 65535.0;
/// The largest joint a packed word can name, plus one.
public static const uint kPackedJoints = 65536u;

/// Words an influence takes in `layout`.
public uint influenceWords(uint layout) {
    return layout == kInfluencePacked ? 1u : 2u;
}

/// A packed influence: `steps` is the weight in 1/65535ths, already rounded.
public uint packInfluence(uint joint, uint steps) {
    return (joint & 0xFFFFu) | (min(steps, 0xFFFFu) << 16);
}

/// (joint, weight) of influence `k` of a gaussian whose influences start at
/// influence `first`, in either layout.
public float2 influenceAt(StructuredBuffer<uint> words, uint layout, uint first, uint k) {
    const uint at = first + k;
    if (layout == kInfluencePacked) {
        const uint word = words[at];
        return float2(float(word & 0xFFFFu), float(word >> 16) / kWeightSteps);
    }
    return float2(asfloat(words[at * 2]), asfloat(words[at * 2 + 1]));
}
