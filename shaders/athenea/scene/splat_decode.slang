// Copyright (c) 2026 jesus luque.
//
// Raw records -> packed splats (see common/packing.slang for the layout),
// written compacted: record i goes to base + dest[i] when valid[i].
module splat_decode;
import athenea.common.packing;
import splat_encoding;

StructuredBuffer<float>      raw;
StructuredBuffer<uint>       valid;
StructuredBuffer<uint>       dest;
RWStructuredBuffer<float4>   positions;
RWStructuredBuffer<uint>     shape;
RWStructuredBuffer<uint>     sh;
// What the splat reflects with, where the file says: one word a splat. Always
// bound -- `params.metallic` is what says whether there is anything to write.
RWStructuredBuffer<uint>     pbr;
// The Cryptomatte id a splat inherited, one word a splat; `params.cryptoObject`
// says whether the file carries one. Always bound.
RWStructuredBuffer<uint>     crypto;
// How much of an environment reaches this gaussian, f16 pairs: nine scalars
// and, where the file carries them, nine rgb triples after. Always bound;
// `params.transferCount` is what says whether anything is written.
RWStructuredBuffer<uint>     transfer;
// Which ways out are open, two words a splat; `params.shadowBits` says whether
// the file carries them. Always bound.
RWStructuredBuffer<uint>     shadowBits;
// WHICH RECORD EACH KEPT SPLAT CAME FROM. Validation drops what cannot be
// drawn, so a splat's index is not its record's; anything else the file keeps
// a gaussian -- a rig's influences, a part of a visibility -- is read through
// this (scene::keptOnly).
RWStructuredBuffer<uint>     origin;
ConstantBuffer<DecodeParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void splatDecode(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || valid[i] == 0) {
        return;
    }
    const uint s = params.stride;
    const uint j = params.base + dest[i];
    origin[j] = params.recordBase + i;

    const float a = decodeOpacity(fieldOf(raw, i, s, params.opacity), params.opacityMode);
    positions[j] = float4(decodePosition(raw, i, params), saturate(a));

    // Packed as (x, y, z, w). Normalised on encode: a trainer never
    // constrained it, and an unnormalised quaternion scales as well as rotates.
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

    if (params.metallic != kNoField || params.roughness != kNoField || params.transmission != kNoField) {
        const float metallic = params.metallic != kNoField ? fieldOf(raw, i, s, params.metallic) : 0.0;
        const float roughness = params.roughness != kNoField ? fieldOf(raw, i, s, params.roughness) : 1.0;
        const float through = params.transmission != kNoField ? fieldOf(raw, i, s, params.transmission) : 0.0;
        pbr[j] = packPbr(metallic, roughness, through);
    }
    if (params.cryptoObject != kNoField) {
        crypto[j] = bit_cast<uint>(fieldOf(raw, i, s, params.cryptoObject));
    }
    if (params.shadowBits != kNoField) {
        shadowBits[j * 2] = bit_cast<uint>(fieldOf(raw, i, s, params.shadowBits));
        shadowBits[j * 2 + 1] = bit_cast<uint>(fieldOf(raw, i, s, params.shadowBits + 1));
    }
    // A transfer is written as it stands: it is a number the frame multiplies
    // a sky by, not a colour to decode, and f16 is generous for a quantity
    // whose own bound is one.
    if (params.transferCount > 0 && params.transferBase != kNoField) {
        for (uint w = 0; w < params.transferWords; ++w) {
            const uint first = w * 2;
            const float lo = first < params.transferCount
                                 ? fieldOf(raw, i, s, params.transferBase + first) : 0.0;
            const float hi = first + 1 < params.transferCount
                                 ? fieldOf(raw, i, s, params.transferBase + first + 1) : 0.0;
            transfer[j * params.transferWords + w] = (f32tof16(hi) << 16) | f32tof16(lo);
        }
    }

    if (params.keepPerColour == 0) {
        return;
    }
    // Rest coefficients, rgb per basis. A 3DGS PLY stores them all red, then
    // all green, then all blue; the file's own per-colour count says where
    // green starts even when fewer are kept.
    const uint shRow = j * params.shWords;
    uint half = 0;
    uint word = 0;
    uint filled = 0;
    for (uint basis = 0; basis < params.keepPerColour; ++basis) {
        for (uint colour = 0; colour < 3; ++colour) {
            const uint field = params.restColourOuter != 0
                                   ? params.restBase + colour * params.filePerColour + basis
                                   : params.restBase + basis * 3 + colour;
            float value = decodeRest(fieldOf(raw, i, s, field), params.restMode);
            if (params.flipYZ != 0) {
                value *= restFlipSign(basis);
            }
            const uint h = f32tof16(halfSafe(value)) & 0xFFFFu;
            word |= h << (filled * 16);
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
