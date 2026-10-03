// Copyright (c) 2026 jesus luque.
//
// What one linear image holds, for measurements that have to keep their sign:
// a white furnace that must return at most 1, the coverage of a converted
// plane, the light a surface receives. Per row, the sum of each channel and
// the largest value of each, then the rows summed (Kahan) and maxed. A pixel
// counts only inside the window [x0, x1) x [y0, y1).
module image_stats;

struct StatsParams {
    uint width;
    uint height;
    uint x0;
    uint y0;
    uint x1;
    uint y1;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>    image;
RWStructuredBuffer<float4>  rowSum;   // height
RWStructuredBuffer<float4>  rowMax;   // height
RWStructuredBuffer<float4>  stats;    // [0] sum, [1] max
ConstantBuffer<StatsParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void imageRowStats(uint3 tid: SV_DispatchThreadID) {
    const uint y = tid.x;
    if (y >= params.height) {
        return;
    }
    float4 sum = 0.0;
    float4 carry = 0.0;
    float4 peak = -1e30;
    if (y >= params.y0 && y < params.y1) {
        for (uint x = params.x0; x < params.x1; ++x) {
            const float4 p = image[y * params.width + x];
            const float4 t = sum + p;
            carry += select(abs(sum) >= abs(p), (sum - t) + p, (p - t) + sum);
            sum = t;
            peak = max(peak, p);
        }
    }
    rowSum[y] = sum + carry;
    rowMax[y] = peak;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void imageStatsReduce(uint3 tid: SV_DispatchThreadID) {
    float4 sum = 0.0;
    float4 carry = 0.0;
    float4 peak = -1e30;
    for (uint y = 0; y < params.height; ++y) {
        const float4 term = rowSum[y];
        const float4 t = sum + term;
        carry += select(abs(sum) >= abs(term), (sum - t) + term, (term - t) + sum);
        sum = t;
        peak = max(peak, rowMax[y]);
    }
    stats[0] = sum + carry;
    stats[1] = peak;
}
