// Copyright (c) 2026 jesus luque.
//
// An image made smaller by a whole factor, each pixel the mean of the block
// it covers: what a frame rendered at four times the resolution becomes when
// it is the reference for a frame that has to filter what it cannot resolve.
module box_reduce;

struct BoxParams {
    uint width;    ///< of the reduced image
    uint height;
    uint factor;   ///< the source is width * factor wide
    uint pad0;
};

StructuredBuffer<float4>   source;
RWStructuredBuffer<float4> reduced;
ConstantBuffer<BoxParams>  boxParams;

[shader("compute")]
[numthreads(8, 8, 1)]
void boxReduce(uint3 tid: SV_DispatchThreadID) {
    const BoxParams p = boxParams;
    if (tid.x >= p.width || tid.y >= p.height) {
        return;
    }
    const uint wide = p.width * p.factor;
    float4 sum = float4(0.0);
    for (uint y = 0; y < p.factor; ++y) {
        for (uint x = 0; x < p.factor; ++x) {
            sum += source[(tid.y * p.factor + y) * wide + tid.x * p.factor + x];
        }
    }
    reduced[tid.y * p.width + tid.x] = sum / float(p.factor * p.factor);
}
