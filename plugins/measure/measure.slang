// Copyright (c) 2026 jesus luque.
//
// What two pictures hold and how far apart they are, measured where they
// already are.
//
// The arithmetic is athenea's own comparison (`render::imageStats`,
// `compareHdr`, `compareImages`, in shaders/athenea/reference), written here
// a second time only because a bundle carries its kernels compiled and binds
// them by order: the same sums in the same order, the same Kahan carries, the
// same bins. That is what lets `athenea compare` print, through this effect,
// the very digits it printed before, and what the tests hold it to.
//
// FOUR PASSES
//
//   measureRows     a thread a row: the sum and the largest value of each
//                   channel inside the window, for both pictures; with a
//                   reference, the row's two histograms (8-bit code values,
//                   and the relative difference in eighths of an octave) and
//                   its relative squared error.
//   measureReduce   a thread a bin adds the rows' histograms; one more thread
//                   adds the rows' sums and maxes.
//   measureFinish   one thread walks the 256 bins of each histogram for the
//                   largest, the count over the threshold and the 99th
//                   percentile, divides what has to be divided and writes the
//                   numbers the effect reads back.
//   measureHeatmap  a thread a pixel of the output.
//
// The histograms live in float4 scratch pictures declared as uint, 512 to a
// row: 0..255 the 8-bit bins, 256..511 the relative ones.

struct MeasureParams {
    uint  width;          // Source's extent; Reference has the same
    uint  height;
    uint  srcStride;
    uint  refStride;
    uint  hasReference;
    uint  x0;             // the window the means are taken over, already clamped
    uint  y0;
    uint  x1;
    uint  y1;
    uint  rowsStride;     // float4s a row of `rows`
    uint  histStride;     // uints a row of `histogram`
    uint  threshold;      // "over": code-value differences above this
    uint  mode;           // 0 Source, 1 Difference, 2 Relative, 3 Code values
    float gain;
    uint  dstWidth;
    uint  dstHeight;
    uint  dstStride;
    int   dstOffsetX;     // the output's origin in Source's pixels
    int   dstOffsetY;
    uint  pad0;
};

StructuredBuffer<float4>    src;
StructuredBuffer<float4>    ref;
RWStructuredBuffer<float4>  rows;        // per row: srcSum, srcMax, refSum, refMax, (rowError, -, -, -)
RWStructuredBuffer<uint>    histogram;   // per row: 512 bins
RWStructuredBuffer<uint>    totals;      // 512 bins, then 4 float4s: srcSum, srcMax, refSum, refMax, then the error
RWStructuredBuffer<uint>    results;     // what the effect reads back; see measureFinish
RWStructuredBuffer<float4>  dst;
ConstantBuffer<MeasureParams> params;

static const uint kBins = 256u;

float encode(float linear) {
    const float c = saturate(linear);
    return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1.0 / 2.4) - 0.055;
}

uint code(float linear) { return uint(round(encode(linear) * 255.0)); }

/// The largest channel difference in 8-bit sRGB code values, alpha linear.
uint codeDifference(float4 pa, float4 pb) {
    uint worst = 0;
    for (uint c = 0; c < 3; ++c) {
        // Premultiplied is what composites, so that is what is compared.
        const int d = int(code(pa[c])) - int(code(pb[c]));
        worst = max(worst, uint(abs(d)));
    }
    return max(worst, uint(abs(int(round(saturate(pa.w) * 255.0)) - int(round(saturate(pb.w) * 255.0)))));
}

/// The largest channel's |a - b| / max(|b|, 1e-3).
float relativeDifference(float4 pa, float4 pb) {
    float worst = 0.0;
    for (uint c = 0; c < 4; ++c) {
        const float d = abs(pa[c] - pb[c]);
        worst = max(worst, d / max(abs(pb[c]), 1e-3));
    }
    return worst;
}

void kahan(inout float4 sum, inout float4 carry, float4 term) {
    const float4 t = sum + term;
    carry += select(abs(sum) >= abs(term), (sum - t) + term, (term - t) + sum);
    sum = t;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void measureRows(uint3 tid: SV_DispatchThreadID) {
    const uint y = tid.x;
    if (y >= params.height) {
        return;
    }
    const bool inside = y >= params.y0 && y < params.y1;

    // The window's sums and largest values, as image_stats has them.
    float4 sum = 0.0;
    float4 carry = 0.0;
    float4 peak = -1e30;
    if (inside) {
        for (uint x = params.x0; x < params.x1; ++x) {
            const float4 p = src[y * params.srcStride + x];
            kahan(sum, carry, p);
            peak = max(peak, p);
        }
    }
    const uint at = y * params.rowsStride;
    rows[at + 0] = sum + carry;
    rows[at + 1] = peak;
    if (params.hasReference == 0u) {
        return;
    }
    sum = 0.0;
    carry = 0.0;
    peak = -1e30;
    if (inside) {
        for (uint x = params.x0; x < params.x1; ++x) {
            const float4 p = ref[y * params.refStride + x];
            kahan(sum, carry, p);
            peak = max(peak, p);
        }
    }
    rows[at + 2] = sum + carry;
    rows[at + 3] = peak;

    // The differences are over the whole picture: they are distributions,
    // and the window is for the means.
    const uint row = y * params.histStride;
    for (uint k = 0; k < 2u * kBins; ++k) {
        histogram[row + k] = 0;
    }
    float error = 0.0;
    float errorCarry = 0.0;   // Kahan: a row of small terms added to a large sum
    for (uint x = 0; x < params.width; ++x) {
        const float4 pa = src[y * params.srcStride + x];
        const float4 pb = ref[y * params.refStride + x];
        histogram[row + codeDifference(pa, pb)] += 1;

        float worst = 0.0;
        float squared = 0.0;
        for (uint c = 0; c < 4; ++c) {
            const float d = abs(pa[c] - pb[c]);
            worst = max(worst, d / max(abs(pb[c]), 1e-3));
            squared += d * d / (pb[c] * pb[c] + 1e-2);
        }
        const uint bin = worst <= 0.0 ? 0u : uint(clamp(floor((log2(worst) + 16.0) * 8.0) + 1.0, 1.0, 255.0));
        histogram[row + kBins + bin] += 1;
        const float term = squared * 0.25;
        const float t = error + term;
        errorCarry += abs(error) >= abs(term) ? (error - t) + term : (term - t) + error;
        error = t;
    }
    rows[at + 4] = float4(error + errorCarry, 0.0, 0.0, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void measureReduce(uint3 tid: SV_DispatchThreadID) {
    const uint bin = tid.x;
    if (bin < 2u * kBins) {
        if (params.hasReference == 0u) {
            return;
        }
        uint sum = 0;
        for (uint r = 0; r < params.height; ++r) {
            sum += histogram[r * params.histStride + bin];
        }
        totals[bin] = sum;
        return;
    }
    if (bin != 2u * kBins) {
        return;
    }
    // The rows' sums and maxes, in the order image_stats adds them.
    const uint images = params.hasReference != 0u ? 2u : 1u;
    for (uint i = 0; i < images; ++i) {
        float4 sum = 0.0;
        float4 carry = 0.0;
        float4 peak = -1e30;
        for (uint y = 0; y < params.height; ++y) {
            kahan(sum, carry, rows[y * params.rowsStride + 2u * i]);
            peak = max(peak, rows[y * params.rowsStride + 2u * i + 1u]);
        }
        const float4 total = sum + carry;
        const uint at = 2u * kBins + 8u * i;
        for (uint c = 0; c < 4; ++c) {
            totals[at + c] = asuint(total[c]);
            totals[at + 4u + c] = asuint(peak[c]);
        }
    }
    if (params.hasReference == 0u) {
        return;
    }
    float error = 0.0;
    float carry = 0.0;
    for (uint y = 0; y < params.height; ++y) {
        const float term = rows[y * params.rowsStride + 4u].x;
        const float t = error + term;
        carry += abs(error) >= abs(term) ? (error - t) + term : (term - t) + error;
        error = t;
    }
    totals[2u * kBins + 16u] = asuint(error + carry);
}

/// The top of relative bin k: bin k > 0 holds [2^((k-1)/8 - 16), 2^(k/8 - 16)).
/// From a table of the eight eighths of an octave rather than exp2, so it is
/// the correctly rounded float of the bound and not an approximation of it.
float binTop(uint bin) {
    static const float eighths[8] = {1.0, 1.0905077326652577, 1.189207115002721, 1.2968395546510096,
                                     1.4142135623730951, 1.5422108254079407, 1.6817928305074290,
                                     1.8340080864093424};
    if (bin == 0u) {
        return 0.0;
    }
    return ldexp(eighths[bin % 8u], int(bin / 8u) - 16);
}

struct Walked {
    uint pixels;
    uint top;    // the highest bin holding anything
    uint over;   // pixels in bins above the threshold
    uint p99;    // the bin where 99% of the pixels have been counted
};

Walked walk(uint first, uint threshold) {
    Walked w;
    w.pixels = 0;
    w.top = 0;
    w.over = 0;
    w.p99 = 0;
    for (uint k = 0; k < kBins; ++k) {
        const uint n = totals[first + k];
        w.pixels += n;
        if (n > 0) {
            w.top = k;
        }
        if (k > threshold) {
            w.over += n;
        }
    }
    const uint p99At = w.pixels - w.pixels / 100u;
    uint running = 0;
    for (uint k = 0; k < kBins; ++k) {
        running += totals[first + k];
        if (running >= p99At) {
            w.p99 = k;
            break;
        }
    }
    return w;
}

// results, 33 words:
//    0..3  Source's mean        4..7  its max      8..11 its sum
//   12..15 Reference's mean    16..19 its max     20..23 its sum
//   24 relMSE  25 the relative squared error's sum  26 p99 relative  27 max relative
//   28 8-bit p99  29 8-bit max  30 over the threshold  31 pixels compared  32 pixels in the window
// Floats as their bits; counts and code values as uints.
[shader("compute")]
[numthreads(1, 1, 1)]
void measureFinish(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    const uint windowPixels = (params.x1 - params.x0) * (params.y1 - params.y0);
    const uint images = params.hasReference != 0u ? 2u : 1u;
    for (uint i = 0; i < 2u; ++i) {
        for (uint c = 0; c < 4; ++c) {
            const float sum = i < images ? asfloat(totals[2u * kBins + 8u * i + c]) : 0.0;
            const float peak = i < images ? asfloat(totals[2u * kBins + 8u * i + 4u + c]) : 0.0;
            results[12u * i + c] = asuint(sum / float(windowPixels));
            results[12u * i + 4u + c] = asuint(peak);
            results[12u * i + 8u + c] = asuint(sum);
        }
    }
    results[32] = windowPixels;
    if (params.hasReference == 0u) {
        for (uint k = 24; k < 32; ++k) {
            results[k] = 0;
        }
        return;
    }
    const Walked codes = walk(0u, params.threshold);
    const Walked relative = walk(kBins, kBins);
    const float error = asfloat(totals[2u * kBins + 16u]);
    results[24] = asuint(relative.pixels > 0 ? error / float(relative.pixels) : 0.0);
    results[25] = asuint(error);
    results[26] = asuint(binTop(relative.p99));
    results[27] = asuint(binTop(relative.top));
    results[28] = codes.p99;
    results[29] = codes.top;
    results[30] = codes.over;
    results[31] = codes.pixels;
}

/// Black, blue, cyan, green, yellow, red: a ramp that reads at a glance and
/// keeps zero black.
float3 ramp(float t) {
    static const float3 stops[6] = {float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 1.0), float3(0.0, 1.0, 1.0),
                                    float3(0.0, 1.0, 0.0), float3(1.0, 1.0, 0.0), float3(1.0, 0.0, 0.0)};
    const float s = saturate(t) * 5.0;
    const uint k = min(uint(s), 4u);
    return lerp(stops[k], stops[k + 1u], s - float(k));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void measureHeatmap(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.dstWidth || tid.y >= params.dstHeight) {
        return;
    }
    const int sx = int(tid.x) + params.dstOffsetX;
    const int sy = int(tid.y) + params.dstOffsetY;
    float4 out = 0.0;
    if (sx >= 0 && sy >= 0 && sx < int(params.width) && sy < int(params.height)) {
        const float4 pa = src[uint(sy) * params.srcStride + uint(sx)];
        out = pa;
        if (params.hasReference != 0u && params.mode != 0u) {
            const float4 pb = ref[uint(sy) * params.refStride + uint(sx)];
            if (params.mode == 1u) {
                out = float4(abs(pa.xyz - pb.xyz) * params.gain, 1.0);
            } else if (params.mode == 2u) {
                out = float4(ramp(relativeDifference(pa, pb) * params.gain), 1.0);
            } else {
                // Only what the threshold lets through is coloured, 32 code
                // values reaching red at a gain of 1; the rest stays black,
                // so a clean frame is a black frame.
                const uint d = codeDifference(pa, pb);
                out = float4(d > params.threshold ? ramp(float(d) / 32.0 * params.gain) : float3(0.0), 1.0);
            }
        }
    }
    dst[tid.y * params.dstStride + tid.x] = out;
}
