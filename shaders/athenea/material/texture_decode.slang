// Copyright (c) 2026 jesus luque.
//
// A decoded image's bytes, as a file reader left them (top row first, 1 to 4
// channels of 8-bit, 16-bit or float components), into the first level of an
// RGBA texture with row 0 at the bottom -- where texture coordinate v = 0 is,
// as MaterialX and USD have it.
//
// A texture in a colour space other than the working one or raw is decoded
// by a generated module (TextureStore, `athenea_texdec_<hash>`) whose entry
// is this one's with the compiled colour function (colour::ColourCompiler)
// between `decodeTexel` and `storeTexel`; alpha is never transformed.
module texture_decode;
import athenea.common.packing;

struct DecodeParams {
    uint width;
    uint height;
    uint channels;     // 1 grey, 2 grey and alpha, 3 rgb, 4 rgba
    uint component;    // 0 unorm8, 1 unorm16, 2 float16, 3 float32
    uint packedKind;   // 0: written to the texture; else atheneaPackTexel's kind
    uint rowPixels;    // pixels a source row; 0: the width
    uint bottomFirst;  // 1: the source's rows are bottom first already (a host's plane)
};

StructuredBuffer<uint>       bytes;   // packed little-endian, rows top first
RWTexture2D<float4>          level;
RWStructuredBuffer<uint>     packed;   // where the store does not convert
ConstantBuffer<DecodeParams> params;

uint byteAt(uint index) {
    return (bytes[index >> 2] >> ((index & 3u) * 8u)) & 0xFFu;
}

float componentAt(uint index) {
    if (params.component == 0) {
        return float(byteAt(index)) / 255.0;
    }
    if (params.component == 1) {
        return float(byteAt(index * 2) | (byteAt(index * 2 + 1) << 8)) / 65535.0;
    }
    if (params.component == 2) {
        return f16tof32(byteAt(index * 2) | (byteAt(index * 2 + 1) << 8));
    }
    const uint bits = byteAt(index * 4) | (byteAt(index * 4 + 1) << 8) | (byteAt(index * 4 + 2) << 16) |
                      (byteAt(index * 4 + 3) << 24);
    return asfloat(bits);
}

/// The texel this thread decodes, as the file holds it; false outside the image.
public bool decodeTexel(uint3 tid, out float4 texel) {
    texel = float4(0.0, 0.0, 0.0, 1.0);
    if (tid.x >= params.width || tid.y >= params.height) {
        return false;
    }
    const uint fileRow = params.bottomFirst != 0 ? tid.y : params.height - 1 - tid.y;
    const uint rowPixels = params.rowPixels != 0 ? params.rowPixels : params.width;
    const uint first = (fileRow * rowPixels + tid.x) * params.channels;
    if (params.channels <= 2) {
        texel.rgb = float3(componentAt(first));
        if (params.channels == 2) {
            texel.a = componentAt(first + 1);
        }
    } else {
        texel.r = componentAt(first);
        texel.g = componentAt(first + 1);
        texel.b = componentAt(first + 2);
        if (params.channels == 4) {
            texel.a = componentAt(first + 3);
        }
    }
    return true;
}

/// Into level 0, or packed where the store does not convert.
public void storeTexel(uint3 tid, float4 texel) {
    if (params.packedKind != 0) {
        atheneaPackTexel(packed, params.packedKind, tid.y * params.width + tid.x, texel);
    } else {
        level[tid.xy] = texel;
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void textureDecode(uint3 tid: SV_DispatchThreadID) {
    float4 texel;
    if (!decodeTexel(tid, texel)) {
        return;
    }
    storeTexel(tid, texel);
}
