// Copyright (c) 2026 jesus luque.
//
// A trace (athenea/splat/splat_trace.slang), checked against the frame it was
// written beside and against the reference's trace of the same frame: every
// rank names a slot the frame has, the ranks descend, no gaussian is named
// twice, what the ranks hold plus what they could not plus what closed the
// walk is the pixel's own alpha -- and it is the trace the exact walk keeps.
module trace_check;
import athenea.splat.splat_trace;

struct TraceCheckParams {
    uint  width;
    uint  height;
    uint  slots;          // the frame's slots: a rank names one below this
    float tolerance;      // how far the weights may be from the pixel's alpha, and from the reference's
    uint  pad0; uint pad1; uint pad2; uint pad3;
};

StructuredBuffer<uint>            trace;       // the rasteriser's
StructuredBuffer<uint>            truth;       // the reference's
StructuredBuffer<float4>          colour;      // the frame the trace was written beside (background alpha 0)
RWStructuredBuffer<uint>          stats;       // 8 wide, one row: see below
ConstantBuffer<TraceCheckParams>  params;

/// stats[0] ranks naming a slot the frame does not have, or standing at no depth
/// stats[1] pixels whose ranks do not descend
/// stats[2] pixels naming one gaussian twice
/// stats[3] pixels whose weights, rest and under miss the pixel's alpha by more than the tolerance
/// stats[4] pixels naming any gaussian at all
/// stats[5] pixels whose list is full and still left something out (rest > tolerance)
/// stats[6] ranks that are not the reference's: another gaussian with another weight, or the same
///          rank weighing otherwise. (Two gaussians a hair apart in weight may swap ranks between
///          two walks that sum in a different order; that is not a difference.)
/// stats[7] pixels whose rest or under is not the reference's
[shader("compute")]
[numthreads(16, 16, 1)]
void traceCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint at = tid.y * params.width + tid.x;
    const TraceRead ours = readTrace(trace, at, pixels);
    const TraceRead theirs = readTrace(truth, at, pixels);

    float sum = ours.rest + ours.under;
    float previous = 3.4e38;
    bool descends = true;
    bool twice = false;
    uint named = 0;
    for (uint k = 0; k < kTraceSlots; ++k) {
        if (ours.slots[k] == kTraceNone) {
            continue;
        }
        named += 1;
        sum += ours.weights[k];
        if (ours.weights[k] > previous) {
            descends = false;
        }
        previous = ours.weights[k];
        if (ours.slots[k] >= params.slots || !(ours.depths[k] > 0.0)) {
            InterlockedAdd(stats[0], 1u);
        }
        for (uint j = 0; j < k; ++j) {
            if (ours.slots[j] == ours.slots[k]) {
                twice = true;
            }
        }
        const bool sameWeight = abs(ours.weights[k] - theirs.weights[k]) <= params.tolerance;
        if (!sameWeight || (ours.slots[k] != theirs.slots[k] && ours.weights[k] > 4.0 * params.tolerance &&
                            !(k > 0 && abs(ours.weights[k] - ours.weights[k - 1]) <= params.tolerance) &&
                            !(k + 1 < kTraceSlots && abs(ours.weights[k] - ours.weights[k + 1]) <= params.tolerance))) {
            InterlockedAdd(stats[6], 1u);
        }
    }
    if (!descends) {
        InterlockedAdd(stats[1], 1u);
    }
    if (twice) {
        InterlockedAdd(stats[2], 1u);
    }
    if (abs(sum - colour[at].w) > params.tolerance) {
        InterlockedAdd(stats[3], 1u);
    }
    if (named > 0) {
        InterlockedAdd(stats[4], 1u);
    }
    if (named == kTraceSlots && ours.rest > params.tolerance) {
        InterlockedAdd(stats[5], 1u);
    }
    if (abs(ours.rest - theirs.rest) > 2.0 * params.tolerance || abs(ours.under - theirs.under) > params.tolerance) {
        InterlockedAdd(stats[7], 1u);
    }
}
