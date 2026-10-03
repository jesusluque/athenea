// Copyright (c) 2026 jesus luque.
//
// A LINEAR CLOUD'S COLOURS INTO THE WORKING SPACE, ONCE, AS IT IS DECODED.
//
// A cloud that holds light (`GpuSplats::linear`) says which space it holds it
// in (`GpuSplats::colourSpace`; lin_rec709_scene where it does not). Where
// that is not the working space, the 3x3 between them (colour_matrix.slang,
// built on the device from the function OCIO compiles) is applied here to
// everything that is a colour: the base colour the shape keeps, every rest
// coefficient of the harmonics, the radiance it gives off, and the indirect
// half of a transfer. Evaluating harmonics is linear in their coefficients,
// so the matrix applied to each coefficient is the matrix applied to every
// colour the cloud shows -- exact, but for the halves the result is stored
// back in. A capture is not converted here: its sRGB curve comes after the
// harmonics are evaluated, so its primaries are brought in a splat at a time
// (common/color.slang, `cloudLight`).
module cloud_to_working;
import athenea.common.packing;

public struct CloudToWorkingParams {
    public uint first;           // the first splat converted (a chunk's place in a store)
    public uint count;
    public uint shWords;
    public uint rest;            // rest basis functions per colour
    public uint emission;        // 1: `emission` holds one RGB9E5 word a splat
    public uint transferWords;   // words a splat of `transfer`; 0: none
    public uint transferCount;   // values a splat: 36 where the indirect half is there
};

StructuredBuffer<float4>             colourRows;   // colour_matrix.slang's rows: the cloud's space to the working space
RWStructuredBuffer<uint>             shape;
RWStructuredBuffer<uint>             sh;
RWStructuredBuffer<uint>             emissionWords;
RWStructuredBuffer<uint>             transfer;
ConstantBuffer<CloudToWorkingParams> params;

float3 toWorking(float3 c) {
    return float3(dot(colourRows[0].xyz, c), dot(colourRows[1].xyz, c), dot(colourRows[2].xyz, c));
}

float halfAt(RWStructuredBuffer<uint> words, uint base, uint k) {
    const uint word = words[base + k / 2];
    return (k & 1u) == 0u ? lowHalf(word) : highHalf(word);
}

void setHalf(RWStructuredBuffer<uint> words, uint base, uint k, float v) {
    const uint at = base + k / 2;
    const uint bits = f32tof16(v) & 0xFFFFu;
    words[at] = (k & 1u) == 0u ? (words[at] & 0xFFFF0000u) | bits : (words[at] & 0xFFFFu) | (bits << 16);
}

/// Three halves from half index `k` of the run at word `base`, converted.
void convertHalves(RWStructuredBuffer<uint> words, uint base, uint k) {
    const float3 c = float3(halfAt(words, base, k), halfAt(words, base, k + 1), halfAt(words, base, k + 2));
    const float3 w = toWorking(c);
    setHalf(words, base, k, w.x);
    setHalf(words, base, k + 1, w.y);
    setHalf(words, base, k + 2, w.z);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void cloudToWorking(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint i = params.first + tid.x;
    // The base colour: shape[2]'s high half, shape[3]'s two (packing.slang).
    const uint row = i * 4;
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const float3 b = toWorking(base);
    shape[row + 2] = (shape[row + 2] & 0xFFFFu) | ((f32tof16(b.x) & 0xFFFFu) << 16);
    shape[row + 3] = packHalves(b.y, b.z);
    // The rest coefficients, rgb a basis function.
    for (uint k = 0; k < params.rest; ++k) {
        convertHalves(sh, i * params.shWords, k * 3);
    }
    if (params.emission != 0) {
        emissionWords[i] = packRgb9e5(max(toWorking(unpackRgb9e5(emissionWords[i])), float3(0.0)));
    }
    // A transfer's indirect half: nine rgb triples after the nine scalars of
    // the direct one (the direct half is a fraction of the sky, no colour).
    if (params.transferWords != 0 && params.transferCount >= 36) {
        for (uint j = 0; j < 9; ++j) {
            convertHalves(transfer, i * params.transferWords, 9 + j * 3);
        }
    }
}
