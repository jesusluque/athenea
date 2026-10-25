// Copyright (c) 2026 jesus luque.
//
// WHAT athz CODES, AND HOW (docs/decisions.md, ".athz"; modules/lod/Athz.h).
//
// The unit coded is the record a pool slot holds, as a .athc chunk block
// lays it out: an element's words are positions (4), shape (4), harmonics
// (shWords), the tail word (a merged group's cell code, a splat's finest
// group), then where the cloud carries them the normal, the emission, the
// material (pbr, lobes) and the transfer (its words, its open directions).
// Each byte of an element is a PLANE: byte b of every element of a run of
// `runGaussians` is one stream of symbols 0..255, and the asset keeps, per
// plane, how it is coded:
//
//   encoding 0  raw bit-pack: (byte - base) in `width` bits, 0..8; a plane
//               that never changes takes no bits at all
//   encoding 1  rANS order 0 on the plane's own frequencies, 2^tableLog of
//               them, shared by every run of the asset
//
// THE rANS (integer only, so every backend decodes the same bits): a 32-bit
// state in [L, 2^16 L) with L = 2^16, renormalised by 16-bit words.
//   encode s (freq f, cum c):  if x >= f << (32 - n): emit x & 0xFFFF, x >>= 16
//                              x = ((x / f) << n) + x % f + c
//   decode:                    slot = x & (2^n - 1); s, f, c from the table
//                              x = f (x >> n) + slot - c; if x < L: x = x << 16 | word
// 32 lanes a run, interleaved: symbol g of a plane is lane g % 32's, so every
// lane meets a plane's end at once and the plane's table is loaded once for
// all of them. The lanes share one stream of words: at each step the lanes
// that renormalise take consecutive words, in lane order, from where the
// last step left -- a mask in workgroup memory, its bits counted (no subgroup
// operation). The encoder walks the steps backwards and writes the stream
// from its end, so the decoder reads it forwards. A run is decoded when every
// lane's state is back to L and every word was read: anything else is a
// corrupt run.
//
// A LAYER (G geometry, A appearance, T transfer, K temporal) is a set of
// planes with a stream of its own, so a reader that does not want one never
// reads it. Byte b's layer is the asset's choice (Athz.cpp's `layerOfByte`).
module athz_common;

public static const uint kAthzLanes = 32;
public static const uint kAthzLow = 65536;   ///< L: a state's least value
public static const uint kAthzLayers = 4;
public static const uint kAthzNoTable = 0xFFFFFFFFu;

/// A plane's description, a uint4 (`planes`): x = encoding | width << 8 |
/// base << 16 | layer << 24; y = its rANS table, or kAthzNoTable; z = where
/// its raw words start in a run's raw part, in words; w = what the encoder
/// estimated it costs (bytes over the asset, encoding 1) -- read by nothing.
public uint athzEncoding(uint4 plane) { return plane.x & 0xFFu; }
public uint athzWidth(uint4 plane) { return (plane.x >> 8) & 0xFFu; }
public uint athzBase(uint4 plane) { return (plane.x >> 16) & 0xFFu; }
public uint athzLayer(uint4 plane) { return plane.x >> 24; }

/// Raw words a plane takes in a run of `run` symbols at `width` bits.
public uint athzRawWords(uint run, uint width) { return (run * width + 31u) / 32u; }

/// The record's arrays, in the order an element lays them out.
public static const uint kFieldPositions = 0;
public static const uint kFieldShape = 1;
public static const uint kFieldSh = 2;
public static const uint kFieldTail = 3;
public static const uint kFieldNormals = 4;
public static const uint kFieldEmission = 5;
public static const uint kFieldPbr = 6;
public static const uint kFieldLobes = 7;
public static const uint kFieldTransfer = 8;
public static const uint kFieldShadow = 9;

/// What an element carries: words of each array (0 where absent).
public struct AthzLayout {
    public uint shWords;
    public uint normals;
    public uint emission;
    public uint pbrWords;
    public uint lobesWords;
    public uint transferWords;
    public uint shadowWords;
};

public uint athzFieldWords(AthzLayout l, uint field) {
    switch (field) {
    case kFieldPositions: return 4u;
    case kFieldShape: return 4u;
    case kFieldSh: return l.shWords;
    case kFieldTail: return 1u;
    case kFieldNormals: return l.normals;
    case kFieldEmission: return l.emission;
    case kFieldPbr: return l.pbrWords;
    case kFieldLobes: return l.lobesWords;
    case kFieldTransfer: return l.transferWords;
    default: return l.shadowWords;
    }
}

/// Which array word `w` of an element is in (x), and which of that array's
/// words an element (y).
public uint2 athzField(AthzLayout l, uint w) {
    for (uint f = 0; f < 10u; ++f) {
        const uint n = athzFieldWords(l, f);
        if (w < n) {
            return uint2(f, w);
        }
        w -= n;
    }
    return uint2(kFieldShadow, 0u);
}

/// The byte address of word `y` of array element `e` in its array.
public uint athzAddress(AthzLayout l, uint2 field, uint e) {
    return (e * athzFieldWords(l, field.x) + field.y) * 4u;
}
