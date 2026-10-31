// Copyright (c) 2026 jesus luque.
//
// THE SAME GAUSSIANS EVERY FRAME (athenea video-encode --reduce keep): the
// first frame's scores (video_reduce.slang) choose which records stay; every
// frame keeps those same records, in their order, so a gaussian keeps its
// index through the sequence -- a stable order a player can smooth over.
//   videoKeepFlag     1 where a record's score reaches the threshold
//   videoKeepScatter  the flagged records, packed in order (keepAt: the flags'
//                     exclusive prefix sum)
module video_keep;


struct KeepParams {
    uint  count;
    uint  shWords;
    float threshold;
    uint  pad0;
};

StructuredBuffer<float>          keepScore;
RWStructuredBuffer<uint>         keepFlag;
StructuredBuffer<uint>           keepAt;      // exclusive prefix sum of keepFlag
StructuredBuffer<float4>         inPositions;
StructuredBuffer<uint>           inShape;
StructuredBuffer<uint>           inSh;
RWStructuredBuffer<float4>       outPositions;
RWStructuredBuffer<uint>         outShape;
RWStructuredBuffer<uint>         outSh;
ConstantBuffer<KeepParams>       keepParams;

[shader("compute")]
[numthreads(256, 1, 1)]
void videoKeepFlag(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= keepParams.count) {
        return;
    }
    keepFlag[tid.x] = keepScore[tid.x] >= keepParams.threshold ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoKeepScatter(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= keepParams.count || keepFlag[s] == 0u) {
        return;
    }
    const uint d = keepAt[s];
    outPositions[d] = inPositions[s];
    for (uint w = 0; w < 4; ++w) {
        outShape[d * 4 + w] = inShape[s * 4 + w];
    }
    for (uint w = 0; w < keepParams.shWords; ++w) {
        outSh[d * keepParams.shWords + w] = inSh[s * keepParams.shWords + w];
    }
}
