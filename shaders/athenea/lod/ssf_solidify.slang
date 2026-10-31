// Copyright (c) 2026 jesus luque.
//
// A TRAINED CLOUD SOLIDIFIED (SSF phase 2, lod::Solidifier): the gaussians
// that describe an opaque surface become blobs of the summed field
// (ssf_field.slang); the rest stay as they are, the residual layer.
//
//   ssfConvert     every gaussian as a candidate blob: its two larger scales
//                  the tangent radii (times the support's ratio), its thin axis
//                  the normal (signed by the shading normal where there is
//                  one), theta the larger axis's turn in the normal's frame,
//                  its DC the colour; a candidate if opaque enough, flat
//                  enough and not glass (lodKeptApart). Sums the log of the
//                  candidates' middle radius for the anchor.
//   ssfAnchor      one thread: the anchor's thickness from that mean
//   ssfCompact     the flagged into a dense list (from an exclusive prefix
//                  sum of the flags), and the rest into another: which
//                  gaussian each blob, and each residual, came from
//   ssfCoherence   a blob whose neighbours' normals do not agree with its own
//                  (hair, foliage, a corner's noise) is sent back
//   ssfWeights     each blob's weight scaled so the field at its centre is C:
//                  a partition of unity over the sheet, so the zero set lies
//                  at the same offset from it everywhere
//   ssfRidge       each centre moved along its normal to the field's ridge
//                  (where the derivative along the normal vanishes): the
//                  sheet a noisy capture's surface scatters about
//   ssfNormals     each normal the field's gradient on the outer zero set,
//                  signed as it was
//   ssfMerge       of two blobs nearer than a fraction of their radius whose
//                  normals agree, the later is dropped
//
// WHY A RIDGE AND NOT THE ZERO SET. A sheet of blobs centred on a surface
// has its field's maximum on the sheet, so F = 0 is two sheets at +-delta,
// delta = c sqrt(1 - (tau / C)^(1/3)) where the field on the sheet is C. At
// the centres the gradient vanishes, and "move to the zero set" is undefined;
// the ridge is what is well defined. And C must be the same everywhere, or
// delta follows the sampling's density: hence the weights.
module ssf_solidify;
import athenea.common.packing;
import athenea.common.morton;
import lod_common;
import ssf_field;

struct SsfSolidifyParams {
    uint  count;          ///< gaussians, or blobs
    uint  normals;        ///< 1: srcNormals holds shading normals
    uint  pbrs;           ///< 1: srcPbr holds material words
    uint  pad0;
    float opacityMin;     ///< a candidate is at least this opaque
    float flatMax;        ///< and its thin scale at most this of its middle one
    float coherenceMin;   ///< weighted mean |n_i . n_j| a blob keeps
    float radiusScale;    ///< a blob's support over its gaussian's scale
    float thicknessRatio; ///< the anchor's c over the mean middle radius
    float tau;
    float level;          ///< C, the field a blob's centre is weighted to
    float mergeFraction;  ///< centres nearer than this of the smaller radius merge
};

StructuredBuffer<float4>   srcPositions;   ///< xyz, opacity
StructuredBuffer<uint>     srcShape;       ///< 4 words a gaussian
StructuredBuffer<uint>     srcNormals;
StructuredBuffer<uint>     srcPbr;
StructuredBuffer<uint>     flags;          ///< 1 a candidate (or a kept blob)
StructuredBuffer<uint>     offsets;        ///< exclusive prefix sum of flags
StructuredBuffer<uint>     sourceIn;       ///< per blob: the gaussian it came from
RWStructuredBuffer<float4> outPositions;
RWStructuredBuffer<uint>   outShape;
RWStructuredBuffer<uint>   outNormals;
RWStructuredBuffer<uint>   outFlags;
RWStructuredBuffer<uint>   outAnchor;      ///< 4 words
RWStructuredBuffer<uint>   solidSource;    ///< per blob: the gaussian it came from
RWStructuredBuffer<uint>   residualSource; ///< per residual: the gaussian it is
RWStructuredBuffer<uint>   stats;          ///< [0] sum of (log2 middle radius + 32) * 16, [1] candidates
ConstantBuffer<SsfSolidifyParams> params;

static const float kPi = 3.14159265358979;

float3 axisOf(float3x3 m, uint k) {
    return float3(m[0][k], m[1][k], m[2][k]);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void ssfConvert(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(srcShape[row + 1]), highHalf(srcShape[row + 1]),
                                    lowHalf(srcShape[row + 2])));
    const float3x3 axes = axesOfQuaternion(decodeQuaternion(srcShape[row]));
    // Larger, middle, thin.
    uint big = 0, thin = 0;
    [unroll] for (uint k = 1; k < 3; ++k) {
        if (scale[k] > scale[big]) big = k;
        if (scale[k] < scale[thin]) thin = k;
    }
    const uint mid = big == thin ? (big + 1) % 3 : 3 - big - thin;
    float3 n = normalize(axisOf(axes, thin));
    if (params.normals != 0 && dot(n, unpackNormal(srcNormals[i])) < 0.0) {
        n = -n;
    }
    float3 u, v;
    frameOf(n, u, v);
    const float3 t = axisOf(axes, big);
    float theta = atan2(dot(t, v), dot(t, u));
    theta = theta < 0.0 ? theta + kPi : theta;
    SolidBlob b;
    b.position = srcPositions[i].xyz;
    b.weight = 1.0;
    b.radii = params.radiusScale * float2(scale[big], scale[mid]);
    b.theta = theta;
    b.colour = float3(highHalf(srcShape[row + 2]), lowHalf(srcShape[row + 3]), highHalf(srcShape[row + 3]));
    b.anchor = 0;
    float4 position;
    uint4  shape;
    packSolidBlob(b, position, shape);
    outPositions[i] = position;
    [unroll] for (uint k = 0; k < 4; ++k) {
        outShape[row + k] = shape[k];
    }
    outNormals[i] = packNormal(n);
    const bool glass = params.pbrs != 0 && lodKeptApart(srcPbr[i]);
    const bool candidate = srcPositions[i].w >= params.opacityMin &&
                           scale[thin] <= params.flatMax * scale[mid] && !glass;
    outFlags[i] = candidate ? 1u : 0u;
    if (candidate) {
        const float l = clamp(log2(b.radii.y), -31.0, 31.0);
        InterlockedAdd(stats[0], uint(round((l + 32.0) * 16.0)));
        InterlockedAdd(stats[1], 1u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void ssfAnchor(uint3 tid: SV_DispatchThreadID) {
    const float mean = stats[1] > 0 ? float(stats[0]) / float(stats[1]) / 16.0 - 32.0 : 0.0;
    SolidAnchor a;
    a.thickness = params.thicknessRatio * exp2(mean);
    a.smoothK = 0.0;
    a.appearance = kNoAppearance;
    a.flags = 0;
    const uint4 w = packAnchor(a);
    [unroll] for (uint k = 0; k < 4; ++k) {
        outAnchor[k] = w[k];
    }
}

/// From `count` flags and their exclusive prefix sum: the flagged's sources
/// in order, and the others'. With sourceIn bound to an identity of the
/// gaussians this is the first split; over blobs it composes.
[shader("compute")]
[numthreads(256, 1, 1)]
void ssfCompact(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    if (flags[i] != 0) {
        solidSource[offsets[i]] = sourceIn[i];
    } else {
        residualSource[i - offsets[i]] = sourceIn[i];
    }
}

/// Visits every blob j whose centre lies in the 27 cells around x.
#define SSF_NEIGHBOURS(x, BODY)                                                                      \
    {                                                                                                \
        const uint _count = uint(gridInfo[kGridCount]);                                              \
        const uint _level = uint(gridInfo[kGridLevel]);                                              \
        const float3 _lo = float3(gridInfo[kGridLoX], gridInfo[kGridLoY], gridInfo[kGridLoZ]);       \
        const float3 _at = floor(((x) - _lo) / gridInfo[kGridExtent] * float(1u << _level));         \
        const int _side = int(1u << _level);                                                         \
        const uint _shift = 3 * (10 - _level);                                                       \
        for (int _dz = -1; _dz <= 1; ++_dz)                                                          \
            for (int _dy = -1; _dy <= 1; ++_dy)                                                      \
                for (int _dx = -1; _dx <= 1; ++_dx) {                                                \
                    const int3 _cell = int3(_at) + int3(_dx, _dy, _dz);                              \
                    if (any(_cell < 0) || any(_cell >= _side)) continue;                             \
                    const uint _code = cellCode(uint3(_cell)) << _shift;                             \
                    const uint _end = _level == 0 ? _count : gridLowerBound(_code + (1u << _shift), _count); \
                    for (uint _j = gridLowerBound(_code, _count); _j < _end; ++_j) {                 \
                        const uint j = gridOrder[_j];                                                \
                        BODY                                                                         \
                    }                                                                                \
                }                                                                                    \
    }

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfCoherence(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const FieldBlob a = fieldBlob(i);
    float agree = 0.0, total = 0.0;
    uint near = 0;
    SSF_NEIGHBOURS(a.centre, {
        if (j != i) {
            const float3 d = blobPositions[j].xyz - a.centre;
            const float3 p = float3(dot(d, a.t1), dot(d, a.t2), dot(d, a.n)) * a.inverseAxes;
            const float r2 = dot(p, p);
            if (r2 < 1.0) {
                const float s = 1.0 - r2;
                const float w = s * s * s;
                agree += w * abs(dot(a.n, unpackNormal(blobNormals[j])));
                total += w;
                ++near;
            }
        }
    })
    // Alone, or among disagreeing neighbours: a residual (in doubt, residual).
    outFlags[i] = near >= 2 && agree >= params.coherenceMin * total ? 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfWeights(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 p = blobPositions[i];
    const float sum = fieldAt(p.xyz, 0.0).w;   // C now, tau aside
    const float scaled = sum > 0.0 ? clamp(params.level / sum, 0.1, 10.0) : 1.0;
    outPositions[i] = float4(p.xyz, p.w * scaled);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfRidge(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const FieldBlob f = fieldBlob(i);
    const float c = 1.0 / f.inverseAxes.z;
    // Newton on g(t) = dF/dn along the normal: the second derivative by a
    // difference of gradients a tenth of c apart.
    const float h = 0.1 * c;
    const float g = dot(fieldAt(f.centre, params.tau).xyz, f.n);
    const float gp = dot(fieldAt(f.centre + h * f.n, params.tau).xyz, f.n);
    const float gm = dot(fieldAt(f.centre - h * f.n, params.tau).xyz, f.n);
    const float curvature = (gp - gm) / (2.0 * h);
    float step = curvature < 0.0 ? -g / curvature : 0.0;   // a maximum: F curves down
    step = clamp(step, -0.5 * c, 0.5 * c);
    outPositions[i] = float4(f.centre + step * f.n, f.weight);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfNormals(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const FieldBlob f = fieldBlob(i);
    const float c = 1.0 / f.inverseAxes.z;
    // The outer zero set, where the field's gradient is the surface's normal.
    const float delta = c * sqrt(max(1.0 - pow(params.tau / params.level, 1.0 / 3.0), 0.0));
    const float3 g = fieldAt(f.centre + delta * f.n, params.tau).xyz;
    const float len = length(g);
    float3 n = f.n;
    if (len > 0.0) {
        n = -g / len;   // F falls outward
        n = dot(n, f.n) < 0.0 ? -n : n;
    }
    outNormals[i] = packNormal(n);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void ssfMerge(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const FieldBlob a = fieldBlob(i);
    const float own = min(1.0 / a.inverseAxes.x, 1.0 / a.inverseAxes.y);
    bool kept = true;
    SSF_NEIGHBOURS(a.centre, {
        if (j < i && kept) {
            const FieldBlob b = fieldBlob(j);
            const float reach = params.mergeFraction * min(own, min(1.0 / b.inverseAxes.x, 1.0 / b.inverseAxes.y));
            if (length(b.centre - a.centre) < reach && abs(dot(a.n, b.n)) > 0.95) {
                kept = false;
            }
        }
    })
    outFlags[i] = kept ? 1u : 0u;
}

StructuredBuffer<float4> pickPositions;
StructuredBuffer<uint>   pickShape;
StructuredBuffer<uint>   pickNormals;
StructuredBuffer<uint>   pickIndex;     ///< per output blob: the input blob it is
StructuredBuffer<uint>   pickSource;    ///< per input blob: its gaussian

/// 0, 1, 2, ...: the first split's sources are the gaussians themselves.
[shader("compute")]
[numthreads(256, 1, 1)]
void ssfIota(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        solidSource[tid.x] = tid.x;
    }
}

/// The blobs a list names, into a dense layer, and with them their sources.
[shader("compute")]
[numthreads(256, 1, 1)]
void ssfPickBlobs(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint from = pickIndex[i];
    outPositions[i] = pickPositions[from];
    [unroll] for (uint k = 0; k < 4; ++k) {
        outShape[i * 4 + k] = pickShape[from * 4 + k];
    }
    outNormals[i] = pickNormals[from];
    residualSource[i] = pickSource[from];
}
