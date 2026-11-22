// Copyright (c) 2026 jesus luque.
//
// WHICH GAUSSIANS A PIXEL IS MADE OF. Beside the colour, a frame that asks
// keeps for each pixel the kTraceSlots gaussians that weighed most in it:
// which one (its projected slot, the index every per-slot buffer of the frame
// is read by), the weight it blended with (transmittance times alpha) and the
// view depth it stood at. What the list could not hold is not lost: it is the
// pixel's `rest`, and what closed the walk from behind (the layer under the
// splats, an opaque disc) is its `under`. So
//
//     sum of the weights + rest + under = the pixel's coverage
//
// whatever the pixel is made of, which is what lets a mask painted on the
// screen be carried to the gaussians (and back from them to another camera's
// screen) without the edges of a soft thing -- hair, a defocused rim, where a
// pixel is fifty faint gaussians and not four -- losing the weight the list
// has no room for.
//
// Cryptomatte's list (splat_blend) is the same walk kept per prim; this one is
// per gaussian, so nothing is summed under one name and a gaussian is in a
// pixel's walk once.
//
// The layout, plane after plane as the matte is (`trace[word * pixels + at]`):
//   words 0 .. K-1     the slots, heaviest first; kTraceNone where a rank is empty
//   words K .. 2K-1    the view depths, as the floats their bits are
//   then K/2 words     the weights as halves, two a word
//   the last word      (rest, under) as halves
// A pixel nothing reached writes its first word alone (kTraceEmpty), and is
// read as empty whatever the others hold. (A pixel's words kept together
// rather than a plane each was tried: no faster on an RTX 3070 and a
// millisecond slower at 3840 x 2160 on Metal.)
module splat_trace;
import athenea.common.packing;

public static const uint kTraceSlots = 4;
public static const uint kTraceWords = kTraceSlots * 2 + kTraceSlots / 2 + 1;
public static const uint kTraceNone = 0xFFFFFFFFu;
/// A pixel's first word where nothing reached it: the pixel's only word written.
public static const uint kTraceEmpty = 0xFFFFFFFEu;

public struct TraceList {
    public uint  slots[kTraceSlots];
    public float weights[kTraceSlots];
    /// Everything the walk blended, in the list or not.
    public float total;
    /// The lightest rank and its weight, kept so that the many gaussians too
    /// light for the list -- most of a walk, once four are in -- cost one
    /// comparison each and no search.
    public uint  lightest;
    public float floor;

    [[mutating]] public void clear() {
        for (uint k = 0; k < kTraceSlots; ++k) {
            slots[k] = kTraceNone;
            weights[k] = 0.0;
        }
        total = 0.0;
        lightest = 0;
        floor = 0.0;
    }

    /// A gaussian of the walk. A free rank is one whose weight is still zero,
    /// so it is the lightest there is; a full list gives up its lightest only
    /// to something heavier, and what it gives up stays in `total`, which is
    /// where `rest` comes from.
    /// Its depth is not kept as the walk goes -- the walk is the engine's
    /// hottest loop and every value it carries is a register a pixel -- but
    /// read where the list is written, from the record its slot names.
    [[mutating]] public void add(uint slot, float weight) {
        if (!(weight > 0.0)) {
            return;
        }
        total += weight;
        if (!(weight > floor)) {
            return;
        }
        enter(slot, weight);
    }

    /// The same for a caller that has already added the weight to `total`
    /// and seen it is heavier than `floor`.
    [[mutating]] public void enter(uint slot, float weight) {
        slots[lightest] = slot;
        weights[lightest] = weight;
        lightest = 0;
        for (uint k = 1; k < kTraceSlots; ++k) {
            if (weights[k] < weights[lightest]) {
                lightest = k;
            }
        }
        floor = weights[lightest];
    }

    /// Ranks in descending weight; `under` is what closed the walk took, and
    /// `depths` the view depth of each rank's gaussian, in the list's own
    /// order (the caller reads them where it keeps its records).
    public void write(RWStructuredBuffer<uint> trace, uint at, uint pixels, float under, float depths[kTraceSlots]) {
        if (!(total > 0.0) && !(under > 0.0)) {
            trace[at] = kTraceEmpty;
            return;
        }
        uint order[kTraceSlots];
        for (uint k = 0; k < kTraceSlots; ++k) {
            order[k] = k;
        }
        for (uint i = 1; i < kTraceSlots; ++i) {
            const uint pick = order[i];
            uint j = i;
            while (j > 0 && weights[order[j - 1]] < weights[pick]) {
                order[j] = order[j - 1];
                --j;
            }
            order[j] = pick;
        }
        float held = 0.0;
        for (uint k = 0; k < kTraceSlots; ++k) {
            const uint from = order[k];
            const bool some = weights[from] > 0.0;
            trace[k * pixels + at] = some ? slots[from] : kTraceNone;
            trace[(kTraceSlots + k) * pixels + at] = some ? asuint(depths[from]) : 0u;
            held += weights[from];
        }
        for (uint pair = 0; pair < kTraceSlots / 2; ++pair) {
            trace[(kTraceSlots * 2 + pair) * pixels + at] =
                packHalves(weights[order[pair * 2]], weights[order[pair * 2 + 1]]);
        }
        trace[(kTraceWords - 1) * pixels + at] = packHalves(max(total - held, 0.0), max(under, 0.0));
    }
};

/// A pixel's trace, read back.
public struct TraceRead {
    public uint  slots[kTraceSlots];
    public float weights[kTraceSlots];
    public float depths[kTraceSlots];
    public float rest;
    public float under;
};

public TraceRead readTrace(StructuredBuffer<uint> trace, uint at, uint pixels) {
    TraceRead r;
    // Nothing reached the pixel and nothing closed its walk: only its first word was written.
    if (trace[at] == kTraceEmpty) {
        for (uint k = 0; k < kTraceSlots; ++k) {
            r.slots[k] = kTraceNone;
            r.depths[k] = 0.0;
            r.weights[k] = 0.0;
        }
        r.rest = 0.0;
        r.under = 0.0;
        return r;
    }
    const uint last = trace[(kTraceWords - 1) * pixels + at];
    for (uint k = 0; k < kTraceSlots; ++k) {
        r.slots[k] = trace[k * pixels + at];
        r.depths[k] = asfloat(trace[(kTraceSlots + k) * pixels + at]);
        const uint word = trace[(kTraceSlots * 2 + k / 2) * pixels + at];
        r.weights[k] = (k & 1u) != 0u ? highHalf(word) : lowHalf(word);
    }
    r.rest = lowHalf(last);
    r.under = highHalf(last);
    return r;
}
