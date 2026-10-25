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
public static const uint kFieldCurvature = 10;

/// What an element carries: words of each array (0 where absent).
public struct AthzLayout {
    public uint shWords;
    public uint normals;
    public uint emission;
    public uint pbrWords;
    public uint lobesWords;
    public uint transferWords;
    public uint shadowWords;
    public uint curvatureWords;
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
    case kFieldShadow: return l.shadowWords;
    default: return l.curvatureWords;
    }
}

/// Which array word `w` of an element is in (x), and which of that array's
/// words an element (y).
public uint2 athzField(AthzLayout l, uint w) {
    for (uint f = 0; f < 11u; ++f) {
        const uint n = athzFieldWords(l, f);
        if (w < n) {
            return uint2(f, w);
        }
        w -= n;
    }
    return uint2(kFieldCurvature, 0u);
}

/// The byte address of word `y` of array element `e` in its array.
public uint athzAddress(AthzLayout l, uint2 field, uint e) {
    return (e * athzFieldWords(l, field.x) + field.y) * 4u;
}

// --- transforms (E1b) ---------------------------------------------------------
//
// A WORD'S TRANSFORM, chosen per word of an element and per kind for the asset
// (the plane entry's w), applied to the whole word before it is split into its
// four byte planes. All integer, all exactly undone:
//   0 none
//   1 delta: the difference from the run's previous element, zigzagged (a run's
//     first element from 0) -- undone by a prefix sum along the run
//   2 delta16: the same on each 16-bit half alone (two f16, a packed pair)
//   3 xor: the exclusive or with the previous element (bit masks)
//   4 parent: the difference from the parent group's same word, zigzagged --
//     a splat's finest merged group (its tail word), a merged group's group
//     at the level above (its cell's parent); the coarsest level's from 0
//   5 parent16: the same, half by half
//   6 xorParent: the exclusive or with the parent's word
//   7 within: each 16-bit half against the coefficient before it in the
//     element -- the low half against the previous word's high half, the
//     high half against its own low half (a transfer's r, g, b of a basis;
//     a pair of scales) -- the previous word being the one coded before it
//   8 parentWithin: the same on the differences from the parent's words
// The tail word itself takes 0 to 3: it is what finds the parent.
public static const uint kXfNone = 0;
public static const uint kXfDelta = 1;
public static const uint kXfDelta16 = 2;
public static const uint kXfXor = 3;
public static const uint kXfParent = 4;
public static const uint kXfParent16 = 5;
public static const uint kXfXorParent = 6;
public static const uint kXfWithin = 7;
public static const uint kXfParentWithin = 8;
public static const uint kXfCount = 9;

/// transform 7's residual of `v` after the word `prev`, and back.
public uint athzWithin(uint v, uint prev) {
    const uint lo = v & 0xFFFFu, hi = v >> 16;
    return zigzag16((lo - (prev >> 16)) & 0xFFFFu) | (zigzag16((hi - lo) & 0xFFFFu) << 16);
}
public uint athzUnwithin(uint r, uint prev) {
    const uint lo = (unzigzag16(r & 0xFFFFu) + (prev >> 16)) & 0xFFFFu;
    const uint hi = (unzigzag16(r >> 16) + lo) & 0xFFFFu;
    return lo | (hi << 16);
}

/// The word coded just before word `w` (athzPlaneAt's order); the tail word
/// has none.
public uint athzPreviousWord(uint w, uint tailWord) {
    if (w == 0u) {
        return tailWord;
    }
    if (w == tailWord + 1u) {
        return tailWord > 0u ? tailWord - 1u : tailWord;
    }
    return w - 1u;
}

public uint athzTransform(uint4 plane) { return plane.w & 0xFFu; }

public uint zigzag(uint d) { return (d << 1) ^ (0u - (d >> 31)); }
public uint unzigzag(uint z) { return (z >> 1) ^ (0u - (z & 1u)); }
uint zigzag16(uint d) { return ((d << 1) ^ (0u - ((d >> 15) & 1u))) & 0xFFFFu; }
uint unzigzag16(uint z) { return ((z >> 1) ^ (0u - (z & 1u))) & 0xFFFFu; }
public uint sub16x2(uint a, uint b) { return ((a - b) & 0xFFFFu) | ((((a >> 16) - (b >> 16)) & 0xFFFFu) << 16); }
public uint add16x2(uint a, uint b) { return ((a + b) & 0xFFFFu) | ((((a >> 16) + (b >> 16)) & 0xFFFFu) << 16); }
public uint zigzag16x2(uint d) { return zigzag16(d & 0xFFFFu) | (zigzag16(d >> 16) << 16); }
public uint unzigzag16x2(uint z) { return unzigzag16(z & 0xFFFFu) | (unzigzag16(z >> 16) << 16); }

/// The residual a word is coded as, against what predicts it (`p`: the
/// previous element's word for 1 to 3, the parent's for 4 to 6).
public uint athzForward(uint t, uint v, uint p) {
    switch (t) {
    case kXfDelta:
    case kXfParent: return zigzag(v - p);
    case kXfDelta16:
    case kXfParent16: return zigzag16x2(sub16x2(v, p));
    case kXfXor:
    case kXfXorParent: return v ^ p;
    default: return v;
    }
}

/// A residual's step (what a run's prefix adds up, 1 to 3) or difference (4 to 6).
public uint athzStep(uint t, uint r) {
    switch (t) {
    case kXfDelta:
    case kXfParent: return unzigzag(r);
    case kXfDelta16:
    case kXfParent16: return unzigzag16x2(r);
    default: return r;
    }
}

/// How two steps combine (the prefix's operation; the parent's undoing).
public uint athzCombine(uint t, uint a, uint b) {
    switch (t) {
    case kXfDelta:
    case kXfParent: return a + b;
    case kXfDelta16:
    case kXfParent16: return add16x2(a, b);
    default: return a ^ b;
    }
}

/// THE ORDER PLANES ARE CODED IN: the tail word's four first (it finds the
/// parent), then every other word's, in the element's order.
public uint athzPlaneAt(uint i, uint tailWord) {
    if (i < 4u) {
        return tailWord * 4u + i;
    }
    uint w = (i - 4u) / 4u;
    if (w >= tailWord) {
        w += 1u;
    }
    return w * 4u + (i & 3u);
}
