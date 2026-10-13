// Copyright (c) 2026 jesus luque.
//
// A TX transfer's noisy halves into the pictures the splat bake filter reads,
// and its answer back (`athenea mesh2splat --transfer`, plugins/splatbakefilter;
// docs/decisions.md, task TX, "The transfer's grain").
//
// What is filtered is what bounced: the indirect half's rgb (the w of the
// same entries is the direct half, which holds the shadows and is left as it
// is) and the reflected field. Both are what a path saw after meeting the
// scene, the noisiest thing a transfer keeps, and neighbours on one surface
// of one prim see nearly the same of it. The filter's own guide -- how far a
// neighbour's light is from this one's in units of its noise -- has no noise
// to read here, so it is handed a variance that never stops it: the weights
// are where a neighbour stands, which way it faces and which prim it came
// from, as the light bake's are.
//
// The answer is laid as `bakePointsOnDevice` lays a transfer: `entries`
// float4 a gaussian, the coefficients first, then the coverage and the cells,
// and the field's sixteen last.
module transfer_filter_io;

struct TransferFilterIo {
    uint count;
    uint coefficients;  ///< of the indirect half: 9 or 16
    uint entries;       ///< float4 a gaussian in the answer
    uint fieldFirst;    ///< where the field's sixteen start, or 0 with none
    uint width;         ///< entries a row of the picture
    uint stride;        ///< pixels a row of it, padded
    uint perRecord;     ///< floats a record
    uint size;          ///< the record's first size
    /// Which half goes through: 0 the indirect half's coefficients, 1 the
    /// field's sixteen. Apart, since the filter takes at most sixteen
    /// entries a gaussian.
    uint part;
    /// AND WHICH OF ITS VALUES: `chunk` of them from `first`. A picture of a
    /// whole half of a car is half a gigabyte, more than the device pool
    /// serves; the filter weighs every value of a gaussian alike, so a few at
    /// a time is the same answer.
    uint first;
    uint chunk;
    /// The answer's first gaussian in the cloud: the rays, records and ids
    /// are the whole cloud's, the answer a slice of it (Converter::transfer).
    uint base;
};

StructuredBuffer<float4>       rays;     // 3 a gaussian (bakePointsOnDevice's layout)
StructuredBuffer<float>        records;  // the cloud's own, `perRecord` floats a gaussian
StructuredBuffer<uint>         ids;      // a gaussian: its prim's Cryptomatte id
RWStructuredBuffer<float4>     answer;   // the transfer, `entries` a gaussian
RWStructuredBuffer<float4>     picture;
ConstantBuffer<TransferFilterIo> io;

uint at(uint entry) {
    return (entry / io.width) * io.stride + entry % io.width;
}

/// Values a gaussian handed over: the indirect half's, or the field's.
uint perGaussian() {
    return io.chunk;
}

/// Where value j of the chunk of gaussian k is in the answer.
uint answerAt(uint k, uint j) {
    return k * io.entries + (io.part == 0u ? 0u : io.fieldFirst) + io.first + j;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transferFilterPoints(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= io.count) {
        return;
    }
    const uint g = io.base + k;
    const float4 point = rays[g * 3];
    const float spread = rays[g * 3 + 1].w;
    const float footprint = spread > 0.0 ? spread : records[g * io.perRecord + io.size];
    picture[at(k * 3)] = float4(point.xyz, footprint);
    const float4 own = rays[g * 3 + 2];
    const float3 facing = own.w > 0.5 ? normalize(own.xyz) : normalize(rays[g * 3 + 1].xyz);
    picture[at(k * 3 + 1)] = float4(facing, asfloat(ids[k]));
    // The luminance of what bounced, and a variance no step in it exceeds.
    const float3 bounced = answer[k * io.entries + (io.part == 0u ? 0u : io.fieldFirst)].rgb;
    picture[at(k * 3 + 2)] = float4(dot(bounced, float3(0.2126, 0.7152, 0.0722)), 1.0e12, 0.0, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transferFilterIn(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    const uint n = perGaussian();
    if (e >= io.count * n) {
        return;
    }
    picture[at(e)] = float4(answer[answerAt(e / n, e % n)].rgb, 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transferFilterOut(uint3 tid: SV_DispatchThreadID) {
    const uint e = tid.x;
    const uint n = perGaussian();
    if (e >= io.count * n) {
        return;
    }
    const uint to = answerAt(e / n, e % n);
    // The w of an indirect coefficient is the direct half's: kept.
    answer[to] = float4(picture[at(e)].rgb, answer[to].w);
}
