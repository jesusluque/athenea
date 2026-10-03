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

/// Values a gaussian handed over: the indirect half's, and the field's.
uint perGaussian() {
    return io.coefficients + (io.fieldFirst != 0u ? 16u : 0u);
}

/// Where value j of gaussian k is in the answer.
uint answerAt(uint k, uint j) {
    return k * io.entries + (j < io.coefficients ? j : io.fieldFirst + (j - io.coefficients));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transferFilterPoints(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= io.count) {
        return;
    }
    const float4 point = rays[k * 3];
    const float spread = rays[k * 3 + 1].w;
    const float footprint = spread > 0.0 ? spread : records[k * io.perRecord + io.size];
    picture[at(k * 3)] = float4(point.xyz, footprint);
    const float4 own = rays[k * 3 + 2];
    const float3 facing = own.w > 0.5 ? normalize(own.xyz) : normalize(rays[k * 3 + 1].xyz);
    picture[at(k * 3 + 1)] = float4(facing, asfloat(ids[k]));
    // The luminance of what bounced, and a variance no step in it exceeds.
    const float3 bounced = answer[k * io.entries].rgb;
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
