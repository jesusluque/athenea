// Copyright (c) 2026 jesus luque.
//
// A noisy bake on a plane, for the splat bake filter (plugins/splatbakefilter),
// and what the filter made of it.
//
// bfSynth lays gaussians on a grid of the plane z = 0, a unit apart and a unit
// wide, every one of the same prim and facing +z. Their indirect light is a
// step -- `low` left of the middle column, `high` from it on -- with a
// uniform noise of half-width `noise` added from a hash, and each says the
// variance of that noise: a bake of few paths over a lit edge.
//
// bfCheck, one thread, reads the filter's answer against the step without
// its noise: the mean squared error before and after, over the gaussians at
// least `margin` columns from the edge and from the borders, and the mean of
// the two columns either side of the edge, which a filter that moved the edge
// would pull towards each other.
module bake_filter_check;

struct SynthParams {
    uint  across;      // gaussians a row
    uint  rows;
    uint  pointsWidth; // the pictures' rows, in entries, and their strides
    uint  pointsStride;
    uint  lightWidth;
    uint  lightStride;
    uint  margin;
    uint  pad0;
    float low;
    float high;
    float noise;
    float pad1;
};

RWStructuredBuffer<float4>  points;
RWStructuredBuffer<float4>  light;
StructuredBuffer<float4>    filtered;
RWStructuredBuffer<float>   results;   // [error before, error after, low column, high column, kept]
ConstantBuffer<SynthParams> synth;

static const float kY0 = 0.28209479177387814;

uint at(uint entry, uint width, uint stride) {
    return (entry / width) * stride + entry % width;
}

float truthAt(uint x) {
    return x < synth.across / 2 ? synth.low : synth.high;
}

float noiseAt(uint k) {
    uint h = k * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return (float(h >> 8) / 16777216.0 * 2.0 - 1.0) * synth.noise;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void bfSynth(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= synth.across * synth.rows) {
        return;
    }
    const uint x = k % synth.across;
    const uint y = k / synth.across;
    const float value = truthAt(x) + noiseAt(k);
    points[at(k * 3, synth.pointsWidth, synth.pointsStride)] = float4(float(x), float(y), 0.0, 1.0);
    points[at(k * 3 + 1, synth.pointsWidth, synth.pointsStride)] = float4(0.0, 0.0, 1.0, asfloat(7u));
    points[at(k * 3 + 2, synth.pointsWidth, synth.pointsStride)] =
        float4(value, synth.noise * synth.noise / 3.0, 0.0, 0.0);
    // Grey light, so its luminance is the value: a constant term `value / Y0`.
    light[at(k, synth.lightWidth, synth.lightStride)] = float4(float3(value / kY0), 0.0);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void bfCheck(uint3 tid: SV_DispatchThreadID) {
    float before = 0.0;
    float after = 0.0;
    uint  counted = 0;
    float lowColumn = 0.0;
    float highColumn = 0.0;
    const uint middle = synth.across / 2;
    for (uint k = 0; k < synth.across * synth.rows; ++k) {
        const uint x = k % synth.across;
        const uint y = k / synth.across;
        const float truth = truthAt(x);
        const float got = filtered[at(k, synth.lightWidth, synth.lightStride)].x * kY0;
        const bool rowInside = y >= synth.margin && y + synth.margin < synth.rows;
        if (rowInside && x + 1 == middle) {
            lowColumn += got;
        }
        if (rowInside && x == middle) {
            highColumn += got;
        }
        const uint fromEdge = x < middle ? middle - x : x - middle + 1;
        if (!rowInside || fromEdge < synth.margin || x < synth.margin || x + synth.margin >= synth.across) {
            continue;
        }
        const float was = truth + noiseAt(k) - truth;
        before += was * was;
        after += (got - truth) * (got - truth);
        counted += 1;
    }
    const float rows = float(synth.rows - 2 * synth.margin);
    results[0] = before / float(max(counted, 1u));
    results[1] = after / float(max(counted, 1u));
    results[2] = lowColumn / rows;
    results[3] = highColumn / rows;
    results[4] = float(counted);
}
