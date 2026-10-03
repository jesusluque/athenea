// Copyright (c) 2026 jesus luque.
//
// What a bake's coefficients say the surface sends in one direction, against
// what it must send. It decodes them the way a cloud does -- the constant
// where 3DGS keeps it, the rest against the basis in the direction the eye
// looks -- so a fit that is right in the coefficients and wrong in the sum
// cannot pass.
module bake_check;
import athenea.common.sh;
import athenea.common.packing;
import athenea.common.color;

struct BakeCheckParams {
    uint   count;        // points
    uint   planes;       // coefficients each
    uint   pad0;
    uint   pad1;
    float4 direction;    // xyz: from the eye to the point; w: the relative tolerance
    float4 expected;     // xyz: the light it sends, linear, as a cloud is blended
};

StructuredBuffer<float4>        coefficients;
RWStructuredBuffer<uint>        counts;   // [checked, beyond the tolerance]
RWStructuredBuffer<float>       worst;    // [worst relative difference, then the first point's rgb]
ConstantBuffer<BakeCheckParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void bakeCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint  beyond = 0;
    float worstRelative = 0.0;
    for (uint p = 0; p < params.count; ++p) {
        const uint base = p * params.planes;
        float3 got = float3(0.5) + kSH0 * coefficients[base].xyz;
        for (uint k = 1; k < params.planes; ++k) {
            got += coefficients[base + k].xyz * shBasisValue(k, params.direction.xyz);
        }
        const float3 want = params.expected.xyz;
        const float scale = max(max(want.r, want.g), max(want.b, 1.0e-6));
        const float relative = max(max(abs(got.r - want.r), abs(got.g - want.g)), abs(got.b - want.b)) / scale;
        worstRelative = max(worstRelative, relative);
        beyond += relative > params.direction.w ? 1u : 0u;
        if (p == 0) {
            worst[1] = got.r;
            worst[2] = got.g;
            worst[3] = got.b;
        }
    }
    counts[0] = params.count;
    counts[1] = beyond;
    worst[0] = worstRelative;
}

// THE SERIES IS BOUNDED FROM EVERY SIDE. A bake is fitted over the half of
// the sphere the surface faces and read by a frame from any direction: edge
// on, and from behind for a disc. Per point, the series is evaluated over the
// whole sphere, read the way a cloud reads it, and the largest light it
// gives anywhere is kept. A fit that extrapolates past what any path saw --
// read on the far half it never fitted, or swung past the brightest sample --
// shows up as light no light in the scene sends.
struct BakeBoundParams {
    uint  count;    // points
    uint  planes;   // coefficients each
    float bound;    // the most light any direction may decode to
    uint  pad0;
};

ConstantBuffer<BakeBoundParams> boundParams;
RWStructuredBuffer<uint>        boundCounts;   // [points, beyond the bound, not a number]
RWStructuredBuffer<float>       brightest;     // [largest light decoded anywhere]

[shader("compute")]
[numthreads(1, 1, 1)]
void bakeBound(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) {
        return;
    }
    uint  beyond = 0;
    uint  invalid = 0;
    float largest = 0.0;
    const uint kElevations = 64;
    const uint kAzimuths = 32;
    for (uint p = 0; p < boundParams.count; ++p) {
        const uint base = p * boundParams.planes;
        float pointLargest = 0.0;
        bool  pointInvalid = false;
        for (uint zi = 0; zi < kElevations; ++zi) {
            const float z = 1.0 - 2.0 * (float(zi) + 0.5) / float(kElevations);
            const float r = sqrt(max(1.0 - z * z, 0.0));
            for (uint ai = 0; ai < kAzimuths; ++ai) {
                const float phi = 6.28318530718 * (float(ai) + 0.5) / float(kAzimuths);
                const float3 d = float3(r * cos(phi), r * sin(phi), z);
                float3 series = float3(0.5) + kSH0 * coefficients[base].xyz;
                for (uint k = 1; k < boundParams.planes; ++k) {
                    series += coefficients[base + k].xyz * shBasisValue(k, d);
                }
                // Light already: a bake is linear, as every cloud is blended.
                const float3 light = max(series, float3(0.0));
                pointInvalid = pointInvalid || any(isnan(light)) || any(isinf(light));
                pointLargest = max(pointLargest, max(light.r, max(light.g, light.b)));
            }
        }
        invalid += pointInvalid ? 1u : 0u;
        beyond += pointLargest > boundParams.bound ? 1u : 0u;
        largest = max(largest, pointLargest);
    }
    boundCounts[0] = boundParams.count;
    boundCounts[1] = beyond;
    boundCounts[2] = invalid;
    brightest[0] = largest;
}
