// Copyright (c) 2026 jesus luque.
//
// A MASK CARRIED FROM THE SCREEN TO THE GAUSSIANS, AND BACK TO A SCREEN.
//
// A frame's trace (splat_trace.slang) says which gaussians each pixel is made
// of and by how much. So a mask painted over that frame -- a key, a
// segmentation, a rectangle someone dragged -- is carried to the gaussians as
// a weighted mean: a gaussian is in the mask as far as the pixels it weighed
// in are,
//
//     s = sum over its pixels of (weight * mask) / sum over its pixels of weight
//
// (the soft form of FlashSplat's assignment, arXiv 2409.08270, and LUDVIG's
// lift, 2410.14462). Both sums are kept, not the quotient, so a second camera
// adds its own to the same gaussians: `lift` is two uints a slot, (weight *
// mask, weight), in 1 / kLiftScale of a pixel -- integers, since a device may
// have no atomic floats. A gaussian nobody saw has a weight of zero and no
// opinion.
//
// Back to a screen, from any camera's trace of the same instances: a pixel is
// in the matte as far as the gaussians it is made of are. What the trace's
// list had no room for (`rest`) and the gaussians no camera lifted are taken
// to be in the mask as far as the pixel's known ones are, which is what keeps
// a soft edge from thinning; what closed the walk from behind (`under`) is
// not a gaussian and is left out.
//
// A slot is a place in a frame: the two frames must be of the same instances
// in the same order, which is what makes a slot the same gaussian in both.
module splat_lift;
import splat_trace;

public static const float kLiftScale = 1024.0;

struct LiftParams {
    uint  width;
    uint  height;
    uint  slots;       // the frame's slots: `lift` holds two uints for each
    uint  hard;        // 1: a gaussian is in the mask or not, by `threshold`
    float threshold;
    uint  pad0; uint pad1; uint pad2;
};

StructuredBuffer<uint>       trace;
StructuredBuffer<float>      mask;     // one a pixel, 0 .. 1, bottom row first as the frame is
RWStructuredBuffer<uint>     lift;     // two a slot: weight * mask, weight
RWStructuredBuffer<float>    matte;    // one a pixel
ConstantBuffer<LiftParams>   params;

[shader("compute")]
[numthreads(16, 16, 1)]
void splatLift(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    const TraceRead read = readTrace(trace, at, pixels);
    const float m = saturate(mask[at]);
    for (uint k = 0; k < kTraceSlots; ++k) {
        const uint slot = read.slots[k];
        if (slot >= params.slots) {
            continue;
        }
        const uint weight = uint(read.weights[k] * kLiftScale + 0.5);
        if (weight == 0u) {
            continue;
        }
        InterlockedAdd(lift[slot * 2 + 1], weight);
        const uint inside = uint(read.weights[k] * m * kLiftScale + 0.5);
        if (inside != 0u) {
            InterlockedAdd(lift[slot * 2], inside);
        }
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void splatMatte(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    const TraceRead read = readTrace(trace, at, pixels);
    float listed = 0.0;
    float known = 0.0;
    float inside = 0.0;
    for (uint k = 0; k < kTraceSlots; ++k) {
        const uint slot = read.slots[k];
        if (slot >= params.slots) {
            continue;
        }
        listed += read.weights[k];
        const uint seen = lift[slot * 2 + 1];
        if (seen == 0u) {
            continue;
        }
        float s = float(lift[slot * 2]) / float(seen);
        if (params.hard != 0) {
            s = s > params.threshold ? 1.0 : 0.0;
        }
        known += read.weights[k];
        inside += read.weights[k] * s;
    }
    const float share = known > 0.0 ? inside / known : 0.0;
    matte[at] = saturate(inside + (listed - known + read.rest) * share);
}
