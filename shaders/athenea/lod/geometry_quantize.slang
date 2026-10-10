// Copyright (c) 2026 jesus luque.
//
// A CLOUD'S GEOMETRY AT THE BITS A PHONE'S CHUNK GIVES IT, and back
// (`athenea decimate --quantize`; research 097's S0 stream, 23 bytes): in
// chunks of `chunk` gaussians in the file's order, each chunk's box and
// ranges first, then every gaussian
//   position   3 x 16 bits in the chunk's box
//   scales     3 x 8 bits of their logarithm in the chunk's range
//              (a surfel's zero scale kept as it is)
//   rotation   32 bits: the three smallest components, 10 bits each, and
//              which was largest
//   opacity    8 bits
//   colour     3 x 8 bits of the base colour's coefficient in the chunk's range
//   normal     16 bits, octahedral
//   curvature  3 x 8 bits signed, over the chunk's largest
//   material   an index (not touched here: a table of the stage's materials)
// and written back as what a reader decodes, so a frame of the output shows
// what the layout costs. The records are the raw floats a cloud is read into
// (io::RawSplats): a ParticleField's scales and opacity as they are, or a
// PLY's logarithms and logit (`linearValues`).
module geometry_quantize;

struct QuantizeParams {
    uint count;
    uint chunk;
    uint stride;          ///< floats a record
    uint hasNormal;

    uint x, y, z, opacity;
    uint s0, s1, s2, rw;
    uint rx, ry, rz, dc0;
    uint dc1, dc2, normal, hasCurvature;
    uint linearValues;    ///< 1: scales and opacity as they are (a USD ParticleField's); 0: logarithms and a logit
    uint pad0, pad1, pad2;
};

RWStructuredBuffer<float>  records;
RWStructuredBuffer<float>  curvature;   // 3 a gaussian, where `hasCurvature`
RWStructuredBuffer<float4> ranges;      // a chunk's: 0 low position, 1 high, 2 low log scale / dc, 3 high
ConstantBuffer<QuantizeParams> params;

static const float kSurfel = -30.0;   // a log scale under this is a surfel's zero, kept

float field(uint g, uint at) { return records[g * params.stride + at]; }

/// A scale's logarithm, whichever way the record holds it (a zero is a
/// surfel's, below kSurfel).
float logScale(uint g, uint at) {
    const float v = records[g * params.stride + at];
    return params.linearValues != 0 ? (v > 1.0e-13 ? log(v) : -1.0e30) : v;
}
void setField(uint g, uint at, float v) { records[g * params.stride + at] = v; }

/// Each chunk's box, its log scales' and colour coefficients' ranges, and its
/// largest curvature.
[shader("compute")]
[numthreads(64, 1, 1)]
void quantizeRanges(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    const uint first = c * params.chunk;
    if (first >= params.count) {
        return;
    }
    const uint last = min(first + params.chunk, params.count);
    float3 lo = float3(3.4e38), hi = float3(-3.4e38);
    float sLo = 3.4e38, sHi = -3.4e38, dLo = 3.4e38, dHi = -3.4e38, kMax = 0.0;
    for (uint g = first; g < last; ++g) {
        const float3 p = float3(field(g, params.x), field(g, params.y), field(g, params.z));
        lo = min(lo, p);
        hi = max(hi, p);
        const float3 s = float3(logScale(g, params.s0), logScale(g, params.s1), logScale(g, params.s2));
        [unroll] for (uint k = 0; k < 3; ++k) {
            if (s[k] > kSurfel && isfinite(s[k])) {
                sLo = min(sLo, s[k]);
                sHi = max(sHi, s[k]);
            }
        }
        const float3 d = float3(field(g, params.dc0), field(g, params.dc1), field(g, params.dc2));
        dLo = min(dLo, min(d.x, min(d.y, d.z)));
        dHi = max(dHi, max(d.x, max(d.y, d.z)));
        if (params.hasCurvature != 0) {
            kMax = max(kMax, max(abs(curvature[g * 3]), max(abs(curvature[g * 3 + 1]), abs(curvature[g * 3 + 2]))));
        }
    }
    ranges[c * 4] = float4(lo, 0.0);
    ranges[c * 4 + 1] = float4(hi, 0.0);
    ranges[c * 4 + 2] = float4(sLo, dLo, kMax, 0.0);
    ranges[c * 4 + 3] = float4(sHi, dHi, 0.0, 0.0);
}

float quantize(float v, float lo, float hi, float steps) {
    const float span = hi - lo;
    if (!(span > 0.0)) {
        return lo;
    }
    return lo + round(saturate((v - lo) / span) * steps) / steps * span;
}

float2 octEncode16(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1.0e-20);
    float2 e = n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return round(saturate(e * 0.5 + 0.5) * 255.0) / 255.0 * 2.0 - 1.0;
}

float3 octDecode16(float2 e) {
    float3 n = float3(e, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        n.xy = (1.0 - abs(n.yx)) * float2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    }
    return normalize(n);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void quantizeRecords(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= params.count) {
        return;
    }
    const uint c = g / params.chunk;
    const float4 lo = ranges[c * 4], hi = ranges[c * 4 + 1], a = ranges[c * 4 + 2], b = ranges[c * 4 + 3];
    // Position: 16 bits an axis in the chunk's box.
    setField(g, params.x, quantize(field(g, params.x), lo.x, hi.x, 65535.0));
    setField(g, params.y, quantize(field(g, params.y), lo.y, hi.y, 65535.0));
    setField(g, params.z, quantize(field(g, params.z), lo.z, hi.z, 65535.0));
    // Scales: 8 bits of the logarithm in the chunk's range; a surfel's zero stays.
    const uint sAt[3] = { params.s0, params.s1, params.s2 };
    [unroll] for (uint k = 0; k < 3; ++k) {
        const float s = logScale(g, sAt[k]);
        if (s > kSurfel && isfinite(s)) {
            const float back = quantize(s, a.x, b.x, 255.0);
            setField(g, sAt[k], params.linearValues != 0 ? exp(back) : back);
        }
    }
    // Rotation: smallest three, 10 bits each over [-1/sqrt2, 1/sqrt2].
    float4 q = float4(field(g, params.rw), field(g, params.rx), field(g, params.ry), field(g, params.rz));
    q /= max(length(q), 1.0e-20);
    uint largest = 0;
    [unroll] for (uint k = 1; k < 4; ++k) {
        if (abs(q[k]) > abs(q[largest])) {
            largest = k;
        }
    }
    if (q[largest] < 0.0) {
        q = -q;
    }
    float sum = 0.0;
    [unroll] for (uint k = 0; k < 4; ++k) {
        if (k != largest) {
            q[k] = quantize(q[k], -0.70710678, 0.70710678, 1023.0);
            sum += q[k] * q[k];
        }
    }
    q[largest] = sqrt(max(1.0 - sum, 0.0));
    setField(g, params.rw, q.x);
    setField(g, params.rx, q.y);
    setField(g, params.ry, q.z);
    setField(g, params.rz, q.w);
    // Opacity: 8 bits of what it covers.
    if (params.linearValues != 0) {
        const float o = field(g, params.opacity);
        if (o <= 1.0) {
            setField(g, params.opacity, round(saturate(o) * 255.0) / 255.0);
        }
    } else {
        const float logit = field(g, params.opacity);
        const float covers = round(saturate(1.0 / (1.0 + exp(-logit))) * 255.0) / 255.0;
        const float held = clamp(covers, 0.5 / 255.0, 1.0 - 0.5 / 255.0);
        setField(g, params.opacity, log(held / (1.0 - held)));
    }
    // Colour: 8 bits of each coefficient in the chunk's range.
    setField(g, params.dc0, quantize(field(g, params.dc0), a.y, b.y, 255.0));
    setField(g, params.dc1, quantize(field(g, params.dc1), a.y, b.y, 255.0));
    setField(g, params.dc2, quantize(field(g, params.dc2), a.y, b.y, 255.0));
    // Normal: 16 bits, octahedral.
    if (params.hasNormal != 0) {
        const float3 n = float3(field(g, params.normal), field(g, params.normal + 1), field(g, params.normal + 2));
        if (dot(n, n) > 1.0e-20) {
            const float3 back = octDecode16(octEncode16(n));
            setField(g, params.normal, back.x);
            setField(g, params.normal + 1, back.y);
            setField(g, params.normal + 2, back.z);
        }
    }
    // Curvature: 8 bits signed over the chunk's largest.
    if (params.hasCurvature != 0) {
        [unroll] for (uint k = 0; k < 3; ++k) {
            curvature[g * 3 + k] = quantize(curvature[g * 3 + k], -a.z, a.z, 254.0);
        }
    }
}
