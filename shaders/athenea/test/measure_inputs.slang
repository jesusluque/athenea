// Copyright (c) 2026 jesus luque.
//
// Two pictures for the Measure effect's test, written where the effect reads
// them: a source with radiance above 1, below 0 and exactly 0 in places, and
// a reference that agrees with it in some pixels, differs by a code value or
// two in others and by a great deal in the rest -- so every bin the
// comparison walks, the 8-bit ones and the relative ones, has something in it.
module measure_inputs;

struct InputsParams {
    uint width;
    uint height;
    uint stride;
    uint seed;
};

RWStructuredBuffer<float4>   source;
RWStructuredBuffer<float4>   reference;
ConstantBuffer<InputsParams> params;

uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

float unit(uint x) { return float(hash(x) >> 8) / 16777216.0; }

[shader("compute")]
[numthreads(16, 16, 1)]
void measureInputs(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint key = (tid.y * params.width + tid.x) * 8u + params.seed * 0x9e3779b9u;
    float4 s = float4(unit(key), unit(key + 1u), unit(key + 2u), unit(key + 3u));
    // A tenth of the pixels bright, a twentieth negative, a twentieth black.
    const float kind = unit(key + 4u);
    if (kind < 0.1) {
        s.xyz *= 8.0;
    } else if (kind < 0.15) {
        s.xyz = -s.xyz;
    } else if (kind < 0.2) {
        s = 0.0;
    }
    float4 r = s;
    const float change = unit(key + 5u);
    if (change < 0.3) {
        // the same pixel
    } else if (change < 0.6) {
        r.xyz += 0.004 * (unit(key + 6u) - 0.5);
    } else if (change < 0.9) {
        r = r * (0.5 + unit(key + 6u));
    } else {
        r = float4(unit(key + 6u) * 4.0, unit(key + 7u), 0.0, unit(key + 7u));
    }
    source[tid.y * params.stride + tid.x] = s;
    reference[tid.y * params.stride + tid.x] = r;
}
