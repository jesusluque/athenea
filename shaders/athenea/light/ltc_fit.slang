// Copyright (c) 2026 jesus luque.
//
// THE LTC TABLE, FITTED ON THE DEVICE (light::LtcTable): for each GGX
// roughness and angle to the eye, the linearly transformed cosine closest to
// the lobe -- GGX with Smith's height-correlated masking, no Fresnel, times
// the cosine, normalised by its albedo -- by Heitz, Dupuy, Hill and Neubelt's
// method (2016): the transform is the lobe's mean direction's frame times
// [[m11, 0, m13], [0, m22, 0], [0, 0, 1]], and Nelder-Mead minimises the
// integral of the cubed difference, sampled from both distributions and
// weighed by their summed densities. The published table is theirs to
// license; this is a fit of our own of the same thing.
//
// One dispatch a roughness row, from the roughest down, a thread an angle,
// each starting from the row above's answer at its angle: the same
// continuation their fit uses. The result is the inverse transform's four
// free entries, normalised by its last, at texel (roughness, angle).
module ltc_fit;

static const float kFitPi = 3.14159265358979;

RWTexture2D<float4>        tableOut; // size x size, x the roughness, y the angle: (a, b, c, d) of the inverse
RWStructuredBuffer<float4> state;   // size x size: (m11, m22, m13, the error) at the end of each fit
uniform uint               size;
uniform uint               row;     // the roughness row this dispatch fits
uniform uint               samples; // a side of the stratified grid each error is measured on

float ggxD(float alpha, float cosH) {
    const float a2 = alpha * alpha;
    const float d = cosH * cosH * (a2 - 1.0) + 1.0;
    return a2 / (kFitPi * d * d);
}

float lambdaGgx(float alpha, float cosT) {
    const float c2 = cosT * cosT;
    const float t2 = max(1.0 - c2, 0.0) / max(c2, 1.0e-12);
    return 0.5 * (-1.0 + sqrt(1.0 + alpha * alpha * t2));
}

/// The lobe times the cosine, and the density of drawing L by the
/// half-vector's distribution.
float brdfEval(float3 v, float3 l, float alpha, out float pdf) {
    pdf = 0.0;
    if (l.z <= 0.0 || v.z <= 0.0) {
        return 0.0;
    }
    const float3 h = normalize(v + l);
    const float d = ggxD(alpha, h.z);
    const float g2 = 1.0 / (1.0 + lambdaGgx(alpha, v.z) + lambdaGgx(alpha, l.z));
    pdf = d * h.z / (4.0 * max(dot(v, h), 1.0e-9));
    return d * g2 / (4.0 * v.z * l.z) * l.z;
}

float3 brdfSample(float3 v, float alpha, float2 u) {
    const float phi = 2.0 * kFitPi * u.y;
    const float cos2 = (1.0 - u.x) / max(1.0 + (alpha * alpha - 1.0) * u.x, 1.0e-12);
    const float cosH = sqrt(saturate(cos2));
    const float sinH = sqrt(max(1.0 - cosH * cosH, 0.0));
    const float3 h = float3(sinH * cos(phi), sinH * sin(phi), cosH);
    return reflect(-v, h);
}

struct Ltc {
    float3x3 m;
    float3x3 inverse;
    float    det;   // of the inverse
    float    amplitude;
};

float3x3 inverse3(float3x3 m, out float det) {
    const float3 a = m[0];
    const float3 b = m[1];
    const float3 c = m[2];
    const float3 r0 = cross(b, c);
    const float3 r1 = cross(c, a);
    const float3 r2 = cross(a, b);
    det = dot(a, r0);
    const float inv = 1.0 / det;
    // The inverse's columns are r0, r1, r2 over det: as rows, the transpose.
    return transpose(float3x3(r0 * inv, r1 * inv, r2 * inv));
}

Ltc makeLtc(float3 mean, float m11, float m22, float m13, float amplitude) {
    // The frame of the lobe's mean direction, in the xz plane.
    const float3 z = mean;
    const float3 x = normalize(float3(z.z, 0.0, -z.x));
    const float3 y = float3(0.0, 1.0, 0.0);
    const float3x3 basis = transpose(float3x3(x, y, z));   // columns x, y, z
    const float3x3 shape = float3x3(float3(m11, 0.0, m13), float3(0.0, m22, 0.0), float3(0.0, 0.0, 1.0));
    Ltc l;
    l.m = mul(basis, shape);
    float det;
    l.inverse = inverse3(l.m, det);
    l.det = abs(1.0 / det);
    l.amplitude = amplitude;
    return l;
}

float ltcEval(Ltc l, float3 w) {
    const float3 o = mul(l.inverse, w);
    const float len = length(o);
    const float3 n = o / len;
    const float d = max(n.z, 0.0) / kFitPi;
    return l.amplitude * d * l.det / (len * len * len);
}

float3 ltcSample(Ltc l, float2 u) {
    const float r = sqrt(u.x);
    const float phi = 2.0 * kFitPi * u.y;
    const float3 c = float3(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
    return normalize(mul(l.m, c));
}

float fitError(Ltc l, float3 v, float alpha) {
    float error = 0.0;
    const uint n = samples;
    for (uint j = 0u; j < n; ++j) {
        for (uint i = 0u; i < n; ++i) {
            const float2 u = float2((float(i) + 0.5) / float(n), (float(j) + 0.5) / float(n));
            {
                const float3 w = ltcSample(l, u);
                float pdfBrdf;
                const float b = brdfEval(v, w, alpha, pdfBrdf);
                const float e = ltcEval(l, w);
                const float pdfLtc = e / l.amplitude;
                const float d = abs(b - e);
                error += d * d * d / max(pdfLtc + pdfBrdf, 1.0e-9);
            }
            {
                const float3 w = brdfSample(v, alpha, u);
                float pdfBrdf;
                const float b = brdfEval(v, w, alpha, pdfBrdf);
                const float e = ltcEval(l, w);
                const float pdfLtc = e / l.amplitude;
                const float d = abs(b - e);
                error += d * d * d / max(pdfLtc + pdfBrdf, 1.0e-9);
            }
        }
    }
    return error / float(n * n);
}

/// The lobe's albedo and mean direction, by sampling it.
void lobeMoments(float3 v, float alpha, out float albedo, out float3 mean) {
    albedo = 0.0;
    mean = float3(0.0);
    const uint n = 64u;
    for (uint j = 0u; j < n; ++j) {
        for (uint i = 0u; i < n; ++i) {
            const float2 u = float2((float(i) + 0.5) / float(n), (float(j) + 0.5) / float(n));
            const float3 w = brdfSample(v, alpha, u);
            float pdf;
            const float b = brdfEval(v, w, alpha, pdf);
            if (pdf > 0.0) {
                albedo += b / pdf;
                mean += w * (b / pdf);
            }
        }
    }
    albedo /= float(n * n);
    mean.y = 0.0;
    mean = dot(mean, mean) > 0.0 ? normalize(mean) : float3(0.0, 0.0, 1.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ltcFit(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= size) {
        return;
    }
    // The table's axes: roughness r (alpha = r^2) and sqrt(1 - cos theta).
    const float r = max(float(row) / float(size - 1u), 0.02);
    const float alpha = r * r;
    const float s = float(t) / float(size - 1u);
    const float cosTheta = clamp(1.0 - s * s, 1.0e-3, 0.9999);
    const float3 v = float3(sqrt(1.0 - cosTheta * cosTheta), 0.0, cosTheta);
    float albedo;
    float3 mean;
    lobeMoments(v, alpha, albedo, mean);
    albedo = max(albedo, 1.0e-6);
    // The start: the row above's fit at this angle, or an isotropic one.
    float3 x0;
    if (row + 1u < size) {
        x0 = state[(row + 1u) * size + t].xyz;
    } else {
        x0 = float3(alpha, alpha, 0.0);
    }
    if (t == 0u || cosTheta > 0.999) {
        mean = float3(0.0, 0.0, 1.0);
    }
    // Nelder-Mead over (m11, m22, m13).
    float3 p[4];
    float f[4];
    p[0] = x0;
    p[1] = x0 + float3(0.05, 0.0, 0.0);
    p[2] = x0 + float3(0.0, 0.05, 0.0);
    p[3] = x0 + float3(0.0, 0.0, 0.05);
    for (uint k = 0u; k < 4u; ++k) {
        p[k].xy = max(p[k].xy, float2(1.0e-4));
        f[k] = fitError(makeLtc(mean, p[k].x, p[k].y, p[k].z, albedo), v, alpha);
    }
    for (uint iteration = 0u; iteration < 120u; ++iteration) {
        // Order: best first.
        for (uint a = 0u; a < 3u; ++a) {
            for (uint b = 0u; b < 3u - a; ++b) {
                if (f[b + 1u] < f[b]) {
                    const float tf = f[b]; f[b] = f[b + 1u]; f[b + 1u] = tf;
                    const float3 tp = p[b]; p[b] = p[b + 1u]; p[b + 1u] = tp;
                }
            }
        }
        if (abs(f[3] - f[0]) <= 1.0e-6 * max(abs(f[0]), 1.0e-9) && iteration > 20u) {
            break;
        }
        const float3 centroid = (p[0] + p[1] + p[2]) / 3.0;
        float3 reflected = centroid + (centroid - p[3]);
        reflected.xy = max(reflected.xy, float2(1.0e-4));
        const float fr = fitError(makeLtc(mean, reflected.x, reflected.y, reflected.z, albedo), v, alpha);
        if (fr < f[0]) {
            float3 expanded = centroid + 2.0 * (centroid - p[3]);
            expanded.xy = max(expanded.xy, float2(1.0e-4));
            const float fe = fitError(makeLtc(mean, expanded.x, expanded.y, expanded.z, albedo), v, alpha);
            if (fe < fr) {
                p[3] = expanded;
                f[3] = fe;
            } else {
                p[3] = reflected;
                f[3] = fr;
            }
        } else if (fr < f[2]) {
            p[3] = reflected;
            f[3] = fr;
        } else {
            float3 contracted = centroid + 0.5 * (p[3] - centroid);
            contracted.xy = max(contracted.xy, float2(1.0e-4));
            const float fc = fitError(makeLtc(mean, contracted.x, contracted.y, contracted.z, albedo), v, alpha);
            if (fc < f[3]) {
                p[3] = contracted;
                f[3] = fc;
            } else {
                for (uint k = 1u; k < 4u; ++k) {
                    p[k] = p[0] + 0.5 * (p[k] - p[0]);
                    p[k].xy = max(p[k].xy, float2(1.0e-4));
                    f[k] = fitError(makeLtc(mean, p[k].x, p[k].y, p[k].z, albedo), v, alpha);
                }
            }
        }
    }
    uint best = 0u;
    for (uint k = 1u; k < 4u; ++k) {
        best = f[k] < f[best] ? k : best;
    }
    state[row * size + t] = float4(p[best], f[best]);
    const Ltc l = makeLtc(mean, p[best].x, p[best].y, p[best].z, albedo);
    // The inverse, in the frame with the eye in the xz plane, normalised so
    // its corner is 1: its four free entries.
    const float3x3 inv = l.inverse;
    const float e = inv[2][2];
    tableOut[uint2(row, t)] = float4(inv[0][0], inv[0][2], inv[1][1], inv[2][0]) / e;
}
