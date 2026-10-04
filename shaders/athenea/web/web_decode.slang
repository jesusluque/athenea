// Copyright (c) 2026 jesus luque.
//
// THE WEB'S DECODE, PRESET T1: a slice of a 3DGS PLY's records, as the file
// holds them (float32, every property one float), into the base arena B0 --
// positions, shape, sh -- in common/packing.slang's layout. The fields'
// places and encodings are DecodeParams, filled from the PLY header by the
// page (bookkeeping: names to indices), exactly as io::readSplatPly fills
// them natively.
//
// What splat_decode.slang does for a capture, without what a capture never
// carries (the material, the transfer, the matte, the normal): four storage
// bindings, where splatDecode's fourteen are over the web's eight. Nothing
// is dropped either: a record that cannot be drawn is culled by the
// projection (a NaN fails its depth test, a zero scale its determinant), so
// the slot order is the file's and no compaction pass is needed.
module web_decode;
import athenea.common.packing;
import athenea.scene.splat_encoding;

StructuredBuffer<float>      raw;
RWStructuredBuffer<float4>   positions;
RWStructuredBuffer<uint>     shape;
RWStructuredBuffer<uint>     sh;
ConstantBuffer<DecodeParams> params;   // base: the slice's first splat

[shader("compute")]
[numthreads(256, 1, 1)]
void webDecode(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint s = params.stride;
    const uint j = params.base + i;

    const float a = decodeOpacity(fieldOf(raw, i, s, params.opacity), params.opacityMode);
    positions[j] = float4(decodePosition(raw, i, params), saturate(a));

    const float4 q = decodeQuaternion(raw, i, params);
    const float qLength = length(q);
    const uint rotation = encodeQuaternion(qLength > 1e-8 ? q : float4(0, 0, 0, 1));
    const float3 ls = float3(decodeLogScale(fieldOf(raw, i, s, params.scale0), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale1), params.scaleMode),
                             decodeLogScale(fieldOf(raw, i, s, params.scale2), params.scaleMode));
    const float3 base = float3(decodeColour(fieldOf(raw, i, s, params.dc0), params.colourMode),
                               decodeColour(fieldOf(raw, i, s, params.dc1), params.colourMode),
                               decodeColour(fieldOf(raw, i, s, params.dc2), params.colourMode));
    const uint row = j * 4;
    shape[row + 0] = rotation;
    shape[row + 1] = packHalves(halfSafe(ls.x), halfSafe(ls.y));
    shape[row + 2] = packHalves(halfSafe(ls.z), halfSafe(base.x));
    shape[row + 3] = packHalves(halfSafe(base.y), halfSafe(base.z));

    // Rest coefficients, rgb per basis, two halves a word (splatDecode).
    const uint shRow = j * params.shWords;
    uint half = 0;
    uint word = 0;
    uint filled = 0;
    for (uint basis = 0; basis < params.keepPerColour; ++basis) {
        for (uint colour = 0; colour < 3; ++colour) {
            const uint field = params.restColourOuter != 0 ? params.restBase + colour * params.filePerColour + basis
                                                           : params.restBase + basis * 3 + colour;
            float value = decodeRest(fieldOf(raw, i, s, field), params.restMode);
            if (params.flipYZ != 0) {
                value *= restFlipSign(basis);
            }
            word |= (f32tof16(halfSafe(value)) & 0xFFFFu) << (filled * 16);
            filled += 1;
            if (filled == 2) {
                sh[shRow + half] = word;
                half += 1;
                word = 0;
                filled = 0;
            }
        }
    }
    if (filled == 1) {
        sh[shRow + half] = word;
    }
}
