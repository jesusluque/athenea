// Copyright (c) 2026 jesus luque.
//
// A CLOUD AS VIDEO PLANES (modules/video/include/athenea/video/Planes.h): each
// attribute of a frame's splats one luma tile of a mosaic, the splats in
// Morton order, every value quantised over the frame's own range.
//
//   attributes  0-2 position xyz, 3 opacity, 4-6 log scale, 7-10 rotation
//               xyzw (w >= 0), 11-13 base colour, 14.. rest coefficients
//               rgb per basis (k = basis * 3 + colour)
//   planes      position as 2 x `bits`: the high part (exact: a side
//               payload, or a plane of its own), the low part folded (odd
//               high parts store it mirrored, so the plane has no saw
//               teeth); then attributes 3.. one plane each
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
    public uint pad1;
    public uint pad2;
};

public static const uint kTileWidth = 512;
public static const uint kBlock = 64;
public static const uint kFirstAttributePlane = 3;   // attributes 3.. after the position planes
public static const float kLogScaleFloor = -20.0;
public static const float kLogScaleTop = 20.0;

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
public uint attributePlane(VideoPlanesParams p, uint a) {
    return (p.hiInVideo != 0 ? 6u : 3u) + (a - kFirstAttributePlane);
}

public uint sampleLevels(VideoPlanesParams p) { return (1u << p.bits) - 1u; }
public uint positionLevels(VideoPlanesParams p) { return (1u << (2u * p.bits)) - 1u; }

/// Attribute `a` of splat `s` of a packed cloud (packing.slang's layout).
public float attributeOf(StructuredBuffer<float4> positions, StructuredBuffer<uint> shape,
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
