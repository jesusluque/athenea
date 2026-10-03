// Copyright (c) 2026 jesus luque.
//
// What a compute kernel's screen derivatives are worth: a field that is
// linear in the pixel, differenced across the thread's quad
// (material_inputs' atheneaDdx/atheneaDdy), against the gradient it was made with.
module quad_derivatives;
import athenea.material.material_inputs;

struct FieldParams {
    float ax;
    float ay;
    uint  width;
    uint  height;
};

RWStructuredBuffer<uint>    counts;   // [threads whose derivatives differ]
ConstantBuffer<FieldParams> field;

[shader("compute")]
[numthreads(16, 16, 1)]
void quadDerivatives(uint3 group: SV_GroupID, uint index: SV_GroupIndex) {
    const uint2 tid = atheneaQuadPixel(group.xy, index);
    // Every thread of a quad reaches the differences, wanted or not.
    const float v = field.ax * float(tid.x) + field.ay * float(tid.y);
    const float dx = atheneaDdx(v);
    const float dy = atheneaDdy(v);
    if (tid.x >= field.width || tid.y >= field.height) {
        return;
    }
    if (abs(dx - field.ax) > 1.0e-5 || abs(dy - field.ay) > 1.0e-5) {
        uint was;
        InterlockedAdd(counts[0], 1u, was);
    }
}

