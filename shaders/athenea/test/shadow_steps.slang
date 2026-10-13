// Copyright (c) 2026 jesus luque.
//
// How a series of values steps: the number of neighbours that differ by more
// than a threshold, and the largest difference, in fixed point. What a test of
// a receiver walking through a shadow counts, rather than reading the series
// back and differencing it on the CPU.
module shadow_steps;

struct StepParams {
    uint  count;       ///< values in the series
    float threshold;   ///< a difference above this is a jump
    float scale;       ///< fixed point of the largest difference
    uint  pad0;
};

StructuredBuffer<float>     series;
RWStructuredBuffer<uint>    steps;   // jumps, largest difference * scale
ConstantBuffer<StepParams>  stepParams;

[shader("compute")]
[numthreads(64, 1, 1)]
void shadowSteps(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i + 1 >= stepParams.count) {
        return;
    }
    const float jump = abs(series[i + 1] - series[i]);
    if (jump > stepParams.threshold) {
        InterlockedAdd(steps[0], 1u);
    }
    InterlockedMax(steps[1], uint(min(jump, 1.0e4) * stepParams.scale));
}
