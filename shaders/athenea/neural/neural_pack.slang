// Copyright (c) 2026 jesus luque.
//
// A neural cloud's latents between the floats a fit works on and the halves a
// file carries (io/NeuralFile.h): two halves a word, low first, the last word
// of a gaussian padded where its numbers are odd.
module neural_pack;
import neural_appearance;
import athenea.common.packing;

public static const uint kNeuralWords = (kNeuralStride + 1) / 2;

struct NeuralPackParams {
    uint count;   // gaussians
    uint pad0; uint pad1; uint pad2;
};

StructuredBuffer<float>           floats;
RWStructuredBuffer<uint>          packed;
StructuredBuffer<uint>            packedRead;
RWStructuredBuffer<float>         unpacked;
ConstantBuffer<NeuralPackParams>  params;

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralPack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint w = 0; w < kNeuralWords; ++w) {
        const uint a = w * 2;
        const float low = halfSafe(floats[i * kNeuralStride + a]);
        const float high = a + 1 < kNeuralStride ? halfSafe(floats[i * kNeuralStride + a + 1]) : 0.0;
        packed[i * kNeuralWords + w] = packHalves(low, high);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralUnpack(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    for (uint k = 0; k < kNeuralStride; ++k) {
        const uint word = packedRead[i * kNeuralWords + k / 2];
        unpacked[i * kNeuralStride + k] = (k & 1u) != 0u ? highHalf(word) : lowHalf(word);
    }
}
