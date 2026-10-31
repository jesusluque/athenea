// Copyright (c) 2026 jesus luque.
//
// Two frames of the same view, pixel by pixel: where one shows nothing
// (opacity 0) and the other a pixel more than half covered. Two estimates of
// one pixel's coverage at 64 paths cannot disagree that much by chance.
struct CoverageParams {
    uint pixels;
};

ConstantBuffer<CoverageParams> coverage;
StructuredBuffer<float4>       a;
StructuredBuffer<float4>       b;
RWStructuredBuffer<uint>       counts;   // [holes in a, holes in b, pixels over half covered in a]

[shader("compute")]
[numthreads(64, 1, 1)]
void coverageHoles(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= coverage.pixels) return;
    if (a[i].a == 0.0 && b[i].a > 0.5) InterlockedAdd(counts[0], 1u);
    if (b[i].a == 0.0 && a[i].a > 0.5) InterlockedAdd(counts[1], 1u);
    if (a[i].a > 0.5) InterlockedAdd(counts[2], 1u);
}
