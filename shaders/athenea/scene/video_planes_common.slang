// Copyright (c) 2026 jesus luque.
//
// A CLOUD AS VIDEO PLANES (modules/video/include/athenea/video/Planes.h): each
// attribute of a frame's splats one luma tile of a mosaic, the splats in
// Morton order, every value quantised over the frame's own range.
//
//   attributes  0-2 position xyz, 3 opacity, 4-6 log scale, 7-10 rotation
//               xyzw (w >= 0), 11-13 base colour, 14.. rest coefficients
//               rgb per basis (k = basis * 3 + colour)
//   planes      position as `bits` + `positionHighBits` (2 x `bits` unless
//               asked): the high part (exact: a side payload, or a plane of
//               its own), the low part folded (odd high parts store it
//               mirrored, so the plane has no saw teeth); then attributes
//               3.. one plane each
//   chroma      (`chroma` 1) every colour quantity -- the base colour, and
//               each harmonic basis -- as YCoCg (Co = R - B, t = B + Co / 2,
//               Cg = G - t, Y = t + Cg / 2): attribute 11 + 3q is quantity
//               q's Y, + 1 its Co, + 2 its Cg. Y has a luma plane of its
//               own; Co and Cg are the Cb and Cr of that plane's tile, one
//               sample a 2 x 2 block -- which is four consecutive splats of
//               the Morton order (the Z order puts splats 4k .. 4k + 3 in
//               one quad), so a chroma sample is the mean of four
//               neighbours. Planes: 3 position, 8 opacity / scale /
//               rotation, then 1 + restPerColour colour Y planes
//   positions   over the frame's range, or with `positionBlock` over the box
//               of their run of that many splats of the Morton order (six
//               halves, rounded outward, at the head of each chunk's side
//               payload: block b's axis a is word b * 3 + a, lo | hi << 16)
//   a tile      512 wide, splat i in 64 x 64 blocks of 4096 Morton-ordered
//               splats, eight blocks a row, Z order inside a block: a chunk
//               of 32768 splats is one band of 64 rows
module video_planes_common;
import athenea.common.packing;

public struct VideoPlanesParams {
    public uint count;          ///< splats in the frame
    public uint bits;           ///< 8 or 10 a sample
    public uint restPerColour;  ///< rest coefficients a colour in the planes
    public uint attributes;     ///< 14 + 3 * restPerColour
    public uint planes;
    public uint hiInVideo;      ///< 1: the high parts are planes 0-2, else a side payload
    public uint columns;        ///< tiles a mosaic row
    public uint tileHeight;
    public uint width;
    public uint height;
    public uint chunkSplats;
    public uint first;          ///< decode: the page's first splat; encode, compare: the slice's first Morton rank
    public uint pageCount;      ///< decode: the page's splats
    public uint at;             ///< decode: the store splat the page starts at
    public uint keep;           ///< decode: rest coefficients a colour the store takes
    public uint shWords;        ///< decode: the store's words a splat
    public uint rangesOffset;   ///< bytes: the frame's ranges in `page`
    public uint hiOffset;       ///< bytes: the high parts in `page`
    public uint sampleScale;    ///< 255 (8 bit) or 65535 (10 bit in 16)
    public uint sampleShift;    ///< 0 or 6 (P010 keeps 10 bits high)
    public uint sourceRest;     ///< encode: the source cloud's rest a colour
    public uint sourceShWords;  ///< encode, compare: the source cloud's words a splat
    public uint positionBlock;     ///< 0: positions over the frame's range; else splats a block of their own box
    public uint positionHighBits;  ///< a position's bits past `bits`: its exact high part (0: none)
    public uint highChunkBytes;    ///< a chunk's side payload: its blocks' boxes, then its high parts
    public uint chroma;            ///< 1: colours as YCoCg, Y in luma and Co, Cg in the tile's own chroma
    public uint chromaWeight;      ///< encode: a 2 x 2 block's chroma 0 the plain mean, 1 weighed by opacity x area
    public uint pad4;
    public uint pad5;
    public uint pad6;
};

public static const uint kTileWidth = 512;
public static const uint kBlock = 64;
public static const uint kFirstAttributePlane = 3;   // attributes 3.. after the position planes
public static const float kLogScaleFloor = -20.0;
public static const float kLogScaleTop = 20.0;
public static const uint kFirstColour = 11;          // the base colour's attributes, then the rest's

uint spread2(uint v) {
    v &= 0x3Fu;
    v = (v | (v << 4)) & 0x0F0Fu;
    v = (v | (v << 2)) & 0x3333u;
    v = (v | (v << 1)) & 0x5555u;
    return v;
}

uint compact2(uint v) {
    v &= 0x5555u;
    v = (v | (v >> 1)) & 0x3333u;
    v = (v | (v >> 2)) & 0x0F0Fu;
    v = (v | (v >> 4)) & 0x00FFu;
    return v;
}

/// Where splat `i` of plane `plane` lies in the mosaic.
public uint2 pixelOf(VideoPlanesParams p, uint plane, uint i) {
    const uint tileX = plane % p.columns, tileY = plane / p.columns;
    const uint block = i >> 12, j = i & 4095u;
    const uint x = tileX * kTileWidth + (block % 8u) * kBlock + compact2(j);
    const uint y = tileY * p.tileHeight + (block / 8u) * kBlock + compact2(j >> 1);
    return uint2(x, y);
}

/// The other way: the plane and the splat a pixel holds (a plane past the
/// last for the mosaic's spare tiles).
public uint2 planeAndSplatOf(VideoPlanesParams p, uint2 pixel) {
    const uint tileX = pixel.x / kTileWidth, tileY = pixel.y / p.tileHeight;
    const uint lx = pixel.x % kTileWidth, ly = pixel.y % p.tileHeight;
    const uint block = (ly / kBlock) * 8u + lx / kBlock;
    const uint j = spread2(lx % kBlock) | (spread2(ly % kBlock) << 1);
    return uint2(tileY * p.columns + tileX, block * 4096u + j);
}

/// The plane of position axis `axis`'s low part, of its high part (when the
/// high parts are planes), and of attribute `a` >= 3.
public uint lowPlane(VideoPlanesParams p, uint axis) { return p.hiInVideo != 0 ? 3u + axis : axis; }
/// The luma plane attribute `a` >= 3 lies in -- with `chroma`, a colour
/// attribute's quantity's Y plane, whose tile's chroma holds its Co and Cg.
public uint attributePlane(VideoPlanesParams p, uint a) {
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    if (p.chroma != 0 && a >= kFirstColour) {
        return first + 8u + (a - kFirstColour) / 3u;
    }
    return first + (a - kFirstAttributePlane);
}
/// The attribute a luma plane past the position's holds (with `chroma`, a
/// colour quantity's Y), and with `chroma` the quantity (or ~0u).
public uint planeAttribute(VideoPlanesParams p, uint plane) {
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    if (p.chroma != 0 && plane >= first + 8u) {
        return kFirstColour + 3u * (plane - first - 8u);
    }
    return plane - first + kFirstAttributePlane;
}
public uint colourQuantityOf(VideoPlanesParams p, uint plane) {
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    return p.chroma != 0 && plane >= first + 8u ? plane - first - 8u : ~0u;
}

/// YCoCg as lifting steps, invertible exactly in real arithmetic (YCoCg-R's
/// form, on floats: the quantisation after it is the only rounding).
public float3 ycocgOf(float3 c) {
    const float co = c.x - c.z;
    const float t = c.z + 0.5 * co;
    const float cg = c.y - t;
    return float3(t + 0.5 * cg, co, cg);
}
public float3 rgbOfYcocg(float3 y) {
    const float t = y.x - 0.5 * y.z;
    const float g = y.z + t;
    const float b = t - 0.5 * y.y;
    return float3(b + y.y, g, b);
}

public uint sampleLevels(VideoPlanesParams p) { return (1u << p.bits) - 1u; }
public uint positionLevels(VideoPlanesParams p) { return (1u << (p.bits + p.positionHighBits)) - 1u; }
/// What a position's error is measured in, whatever it was sent as: steps
/// of 2 x `bits` over the frame's range.
public uint referenceLevels(VideoPlanesParams p) { return (1u << (2u * p.bits)) - 1u; }
/// Bytes of a chunk's side payload before its high parts: its blocks' boxes.
public uint blockBytes(VideoPlanesParams p) {
    return p.positionBlock == 0 ? 0u : (p.chunkSplats / p.positionBlock) * 12u;
}
/// A block's box word (lo | hi << 16, halves) of axis `axis` for picture
/// splat `i`, as a word index into a chunk's side payload.
public uint blockWordOf(VideoPlanesParams p, uint i, uint axis) {
    return ((i % p.chunkSplats) / p.positionBlock) * 3u + axis;
}
/// The half at or below `v`, and at or above: a box rounded outward, so it
/// holds what it was taken over.
public uint halfBelow(float v) {
    uint h = f32tof16(v) & 0xFFFFu;
    if (f16tof32(h) > v) {
        h = (h & 0x8000u) != 0 ? h + 1u : (h == 0u ? 0x8001u : h - 1u);
    }
    return h;
}
public uint halfAbove(float v) {
    uint h = f32tof16(v) & 0xFFFFu;
    if (f16tof32(h) < v) {
        h = (h & 0x8000u) != 0 ? (h == 0x8000u ? 1u : h - 1u) : h + 1u;
    }
    return h;
}

/// Attribute `a` of splat `s` of a packed cloud (packing.slang's layout),
/// its colours as they are (`chroma` 0) or as YCoCg.
public float attributeOf(StructuredBuffer<float4> positions, StructuredBuffer<uint> shape,
                         StructuredBuffer<uint> sh, uint shWords, uint rest, uint s, uint a, uint chroma) {
    if (chroma != 0 && a >= kFirstColour) {
        const uint q = (a - kFirstColour) / 3u;
        const uint first = kFirstColour + 3u * q;   // r, g, b of quantity q
        const float3 c = float3(plainAttributeOf(positions, shape, sh, shWords, rest, s, first),
                                plainAttributeOf(positions, shape, sh, shWords, rest, s, first + 1u),
                                plainAttributeOf(positions, shape, sh, shWords, rest, s, first + 2u));
        return ycocgOf(c)[(a - kFirstColour) % 3u];
    }
    return plainAttributeOf(positions, shape, sh, shWords, rest, s, a);
}

float plainAttributeOf(StructuredBuffer<float4> positions, StructuredBuffer<uint> shape,
                       StructuredBuffer<uint> sh, uint shWords, uint rest, uint s, uint a) {
    if (a < 3) {
        return positions[s][a];
    }
    if (a == 3) {
        return positions[s].w;
    }
    if (a < 7) {
        const uint w1 = shape[s * 4 + 1], w2 = shape[s * 4 + 2];
        const float v = a == 4 ? lowHalf(w1) : (a == 5 ? highHalf(w1) : lowHalf(w2));
        return clamp(v, kLogScaleFloor, kLogScaleTop);
    }
    if (a < 11) {
        float4 q = decodeQuaternion(shape[s * 4]);
        if (q.w < 0.0) {
            q = -q;
        }
        return q[a - 7];
    }
    if (a < 14) {
        const uint w2 = shape[s * 4 + 2], w3 = shape[s * 4 + 3];
        return a == 11 ? highHalf(w2) : (a == 12 ? lowHalf(w3) : highHalf(w3));
    }
    const uint k = a - 14;
    if (k / 3u >= rest) {
        return 0.0;
    }
    const uint word = sh[s * shWords + k / 2u];
    return (k & 1u) != 0 ? highHalf(word) : lowHalf(word);
}

/// `v` over [lo, hi] in `levels` steps.
public uint quantise(float v, float lo, float hi, uint levels) {
    if (!(hi > lo)) {
        return 0u;
    }
    return min(uint(round(saturate((v - lo) / (hi - lo)) * float(levels))), levels);
}

public float dequantise(uint q, float lo, float hi, uint levels) {
    return hi > lo ? lo + (hi - lo) * (float(min(q, levels)) / float(levels)) : lo;
}

/// A position's quantised value as its high and low parts, the low one
/// folded on odd high parts.
public uint2 splitPosition(VideoPlanesParams p, uint q) {
    const uint mask = sampleLevels(p);
    const uint hi = q >> p.bits;
    const uint lo = q & mask;
    return uint2(hi, (hi & 1u) != 0 ? mask - lo : lo);
}

public uint joinPosition(VideoPlanesParams p, uint hi, uint lo) {
    const uint mask = sampleLevels(p);
    lo = min(lo, mask);
    return (hi << p.bits) | ((hi & 1u) != 0 ? mask - lo : lo);
}
