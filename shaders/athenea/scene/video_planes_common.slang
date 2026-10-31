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
    public float positionTarget;   ///< > 0: a block's high bits from its box (adaptive): what an 8-bit step covers
    public uint geometry;          ///< 1: position, scale and rotation in the exact geometry layer, not in planes
    public uint pad6;              ///< videoPlanesBoxUnion: 1 on a group's first frame
    public uint geometryAt;        ///< decode: the picture's first splat in its frame (the geometry state's index)
    public uint geometryPositionBits;  ///< the layer's bits a position axis (10-16)
    public uint geometryScaleBits;     ///< the layer's bits a log scale (6-10)
    public uint geometryFold;          ///< bit 31: geometry in the video; the fold K of position (0-7), log scale (8-15), rotation (16-23)
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
/// high parts are planes), and of attribute `a` >= 3. With the GEOMETRY LAYER
/// (`geometry` 1) position, scale and rotation have no plane: plane 0 is the
/// opacity and the colour planes follow it.
public uint lowPlane(VideoPlanesParams p, uint axis) { return p.hiInVideo != 0 ? 3u + axis : axis; }
/// The luma plane attribute `a` >= 3 lies in -- with `chroma`, a colour
/// attribute's quantity's Y plane, whose tile's chroma holds its Co and Cg.
public uint attributePlane(VideoPlanesParams p, uint a) {
    if (geometryVideo(p)) {
        if (a < kFirstColour) {
            return a == 3 ? opacityPlane(p) : coarsePlane(a);
        }
        return opacityPlane(p) + 1u + (p.chroma != 0 ? (a - kFirstColour) / 3u : a - kFirstColour);
    }
    if (p.geometry != 0) {
        if (a < kFirstColour) {
            return 0u;   // the opacity, the one attribute before the colours with a plane
        }
        return 1u + (p.chroma != 0 ? (a - kFirstColour) / 3u : a - kFirstColour);
    }
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    if (p.chroma != 0 && a >= kFirstColour) {
        return first + 8u + (a - kFirstColour) / 3u;
    }
    return first + (a - kFirstAttributePlane);
}
/// The attribute a luma plane past the position's holds (with `chroma`, a
/// colour quantity's Y), and with `chroma` the quantity (or ~0u).
public uint planeAttribute(VideoPlanesParams p, uint plane) {
    if (geometryVideo(p)) {
        if (plane < kGeometryAttributes) {
            return geometryAttribute(plane);
        }
        if (plane < opacityPlane(p)) {
            return foldedAttribute(p, plane - kGeometryAttributes).x;
        }
        if (plane == opacityPlane(p)) {
            return 3u;
        }
        const uint c = plane - opacityPlane(p) - 1u;
        return kFirstColour + (p.chroma != 0 ? 3u * c : c);
    }
    if (p.geometry != 0) {
        if (plane == 0) {
            return 3u;
        }
        return kFirstColour + (p.chroma != 0 ? 3u * (plane - 1u) : plane - 1u);
    }
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    if (p.chroma != 0 && plane >= first + 8u) {
        return kFirstColour + 3u * (plane - first - 8u);
    }
    return plane - first + kFirstAttributePlane;
}
public uint colourQuantityOf(VideoPlanesParams p, uint plane) {
    if (geometryVideo(p)) {
        return p.chroma != 0 && plane > opacityPlane(p) ? plane - opacityPlane(p) - 1u : ~0u;
    }
    if (p.geometry != 0) {
        return p.chroma != 0 && plane >= 1u ? plane - 1u : ~0u;
    }
    const uint first = p.hiInVideo != 0 ? 6u : 3u;
    return p.chroma != 0 && plane >= first + 8u ? plane - first - 8u : ~0u;
}

// --- GEOMETRY IN THE VIDEO (`geometryFold` bit 31) ---------------------------
//
// Nine geometry attributes -- position xyz (0-2), log scale xyz (4-6), and
// the rotation's x, y, z (7-9) in its CANONICAL form: of the four quaternions
// that turn the ellipsoid onto itself (q, and q times a half turn about each
// of its own axes: q.i, q.j, q.k) the one whose w is largest, w >= 0 -- so w
// is at least 1/2, rebuilt from the other three, and x, y, z lie in
// [-0.87, 0.87]. Each over its range in the record (attribute 10's range is
// written and not used):
//   coarse  C = round(t * 255), t = (v - lo) / (hi - lo) in [0, 1]
//   fine    (a folded attribute, fold K > 0) U = round(t * N), N =
//           floor(255 * 255 / K) levels, as two triangle waves of period
//           510 a quarter period apart (QUADRATURE): A = wave(U), B =
//           wave(U + 128), wave(u) = m <= 255 ? m : 510 - m, m = u mod 510
//   decode  the wave that is on a slope (A in 64..191, else B -- one of the
//           two always is) gives U up to its period, and the coarse estimate
//           C * N / 255 picks the period: the right one while the coarse
//           sample is within 64 levels, K / 4 of its codes. The fine sample's
//           own error is then the error, a level of N.
// A wave has no saw teeth, so a smooth attribute stays a smooth image; a
// single wave would be ambiguous near its turns, where U and its mirror are
// close and a coarse error picks the wrong one.

public static const uint kGeometryAttributes = 9;
public static const uint kFoldPeriod = 510;
public static const uint kQuadrature = 128;

public bool geometryVideo(VideoPlanesParams p) { return (p.geometryFold >> 31) != 0; }
/// Geometry attribute g (0-8) is attribute 0-2 or 4-9.
public uint geometryAttribute(uint g) { return g < 3 ? g : g + 1u; }
public uint geometryIndexOf(uint a) { return a < 3 ? a : a - 1u; }
public uint foldOf(VideoPlanesParams p, uint a) {
    const uint shift = a < 3 ? 0u : (a < 7 ? 8u : 16u);
    return (p.geometryFold >> shift) & 0xFFu;
}
/// Folded geometry attributes before geometry attribute g.
public uint foldedBefore(VideoPlanesParams p, uint g) {
    uint n = 0;
    for (uint k = 0; k < g && k < kGeometryAttributes; ++k) {
        n += foldOf(p, geometryAttribute(k)) != 0 ? 1u : 0u;
    }
    return n;
}
public uint coarsePlane(uint a) { return geometryIndexOf(a); }
/// Wave 0 (A) or 1 (B) of folded attribute `a`.
public uint finePlane(VideoPlanesParams p, uint a, uint wave) {
    return kGeometryAttributes + 2u * foldedBefore(p, geometryIndexOf(a)) + wave;
}
public uint opacityPlane(VideoPlanesParams p) {
    return kGeometryAttributes + 2u * foldedBefore(p, kGeometryAttributes);
}
/// The attribute and the wave of fine plane `f` (from the first fine plane).
public uint2 foldedAttribute(VideoPlanesParams p, uint f) {
    uint n = 0;
    for (uint k = 0; k < kGeometryAttributes; ++k) {
        const uint a = geometryAttribute(k);
        if (foldOf(p, a) != 0) {
            if (n == f / 2u) {
                return uint2(a, f % 2u);
            }
            n += 1u;
        }
    }
    return uint2(0u, 0u);
}
public bool isFinePlane(VideoPlanesParams p, uint plane) {
    return geometryVideo(p) && plane >= kGeometryAttributes && plane < opacityPlane(p);
}
public uint foldLevels(uint fold) { return (255u * 255u) / max(fold, 1u); }
public uint foldWave(uint u) {
    const uint m = u % kFoldPeriod;
    return m <= 255u ? m : kFoldPeriod - m;
}
/// U from the coarse sample and wave A's and B's.
public uint unfold(uint coarse, uint waveA, uint waveB, uint fold) {
    const uint n = foldLevels(fold);
    const bool useA = waveA >= 64u && waveA <= 191u;
    const int shift = useA ? 0 : int(kQuadrature);
    const int f = int(min(useA ? waveA : waveB, 255u));
    const float estimate = float(coarse) * float(n) / 255.0 + float(shift);
    const int j0 = int(floor(estimate / float(kFoldPeriod)));
    float best = 3.4e38;
    int found = 0;
    for (int j = j0 - 1; j <= j0 + 1; ++j) {
        const int a = j * int(kFoldPeriod) + f;
        const int b = j * int(kFoldPeriod) + int(kFoldPeriod) - f;
        if (abs(float(a) - estimate) < best) {
            best = abs(float(a) - estimate);
            found = a;
        }
        if (abs(float(b) - estimate) < best) {
            best = abs(float(b) - estimate);
            found = b;
        }
    }
    return uint(clamp(found - shift, 0, int(n)));
}
/// The canonical rotation (above) of a unit quaternion (x, y, z, w).
public float4 canonicalRotation(float4 q) {
    // q.i = (w, z, -y, -x), q.j = (-z, w, x, -y), q.k = (y, -x, w, -z).
    float4 best = q;
    if (abs(-q.x) > abs(best.w)) {
        best = float4(q.w, q.z, -q.y, -q.x);
    }
    if (abs(-q.y) > abs(best.w)) {
        best = float4(-q.z, q.w, q.x, -q.y);
    }
    if (abs(-q.z) > abs(best.w)) {
        best = float4(q.y, -q.x, q.w, -q.z);
    }
    return best.w < 0.0 ? -best : best;
}
/// The angle between two rotations of an ellipsoid, up to its own half
/// turns (radians).
public float ellipsoidAngle(float4 a, float4 b) {
    // r = conj(a) * b; the four equivalents' w are r.w, -r.x, -r.y, -r.z.
    const float4 r = float4(a.w * b.x - a.x * b.w - a.y * b.z + a.z * b.y,
                            a.w * b.y + a.x * b.z - a.y * b.w - a.z * b.x,
                            a.w * b.z - a.x * b.y + a.y * b.x - a.z * b.w,
                            a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z);
    const float m = max(max(abs(r.x), abs(r.y)), max(abs(r.z), abs(r.w))) / max(length(r), 1e-20);
    return 2.0 * acos(clamp(m, 0.0, 1.0));
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
/// ADAPTIVE PRECISION (`positionTarget` > 0): each block carries its own
/// count of high bits, a byte after the chunk's boxes -- 0 where its box is no
/// wider than the target, else enough that a step of the low part is no
/// wider than the target's: ceil(log2(widest / target)), at most
/// `positionHighBits`. The high parts' storage stays one byte an axis.
public bool adaptive(VideoPlanesParams p) { return p.positionTarget > 0.0; }
public uint bitsBytes(VideoPlanesParams p) {
    return adaptive(p) ? ((p.chunkSplats / p.positionBlock + 3u) & ~3u) : 0u;
}
public uint highsAt(VideoPlanesParams p) { return blockBytes(p) + bitsBytes(p); }
public uint levelsWith(VideoPlanesParams p, uint highBits) { return (1u << (p.bits + highBits)) - 1u; }
public uint bitsFor(VideoPlanesParams p, float widest) {
    if (!(widest > p.positionTarget)) {
        return 0u;
    }
    return min(uint(ceil(log2(widest / p.positionTarget))), p.positionHighBits);
}
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

// --- THE GEOMETRY LAYER (`geometry` 1) ------------------------------------------
//
// Position, scale and rotation beside the video, exact: a gaussian's
// quantised geometry is four words -- position x | y << 16, z | log scale x
// << 16, log scale y | z << 16, and the store's own rotation word
// (packing.slang's smallest three) -- position `geometryPositionBits` an axis
// and log scale `geometryScaleBits` over the record's ranges (attributes 0-2
// and 4-6, one for the whole sequence). A frame sends each gaussian's
// geometry absolute, or as a zigzagged residual from a prediction out of the
// same gaussian's frames before: the frame before (order 1), or its motion
// carried on (order 2: 2 b - c), mod 2^16 a field; the rotation's three 10-bit
// components each mod 2^10 and its index as an exclusive or. 16 byte planes,
// chunk by chunk -- planes 2f, 2f + 1 the low and high byte of field f (px,
// py, pz, lsx, lsy, lsz), planes 12-15 the rotation's bytes -- `chunkSplats`
// bytes a plane, deflated beside the video.

public static const uint kGeometryPlanes = 16;
public static const uint kGeometryKeyBlock = 256;   // a staggered refresh's block

/// Whether gaussian `j` of frame `frame` goes absolute: every one on a key
/// frame (`frame % interval == 0`), or with `stagger` the blocks of 256 whose
/// index matches the frame's phase -- a block is refreshed once an interval.
public bool geometryKey(uint j, uint frame, uint interval, uint stagger) {
    if (interval <= 1) {
        return true;
    }
    if (stagger != 0) {
        return (j / kGeometryKeyBlock) % interval == frame % interval;
    }
    return frame % interval == 0;
}

public uint zigzag16(uint d) {
    const int s = int(d << 16) >> 16;   // the low 16 bits as signed
    return uint((s << 1) ^ (s >> 31)) & 0xFFFFu;
}
public uint unzigzag16(uint z) { return ((z >> 1) ^ (0u - (z & 1u))) & 0xFFFFu; }
public uint zigzag10(uint d) {
    const int s = int(d << 22) >> 22;   // the low 10 bits as signed
    return uint((s << 1) ^ (s >> 31)) & 0x3FFu;
}
public uint unzigzag10(uint z) { return ((z >> 1) ^ (0u - (z & 1u))) & 0x3FFu; }

/// A rotation word's residual against a predicted one: each 10-bit component
/// zigzagged mod 2^10 (at bits 0, 10, 20), the index's exclusive or at 30.
public uint rotationResidual(uint now, uint predicted) {
    uint r = (now ^ predicted) & 3u;
    r <<= 30u;
    for (uint c = 0; c < 3; ++c) {
        const uint shift = 22u - 10u * c;
        r |= zigzag10(((now >> shift) - (predicted >> shift)) & 0x3FFu) << (10u * c);
    }
    return r;
}
public uint rotationFrom(uint residual, uint predicted) {
    uint w = (predicted ^ (residual >> 30u)) & 3u;
    for (uint c = 0; c < 3; ++c) {
        const uint shift = 22u - 10u * c;
        w |= (((predicted >> shift) + unzigzag10((residual >> (10u * c)) & 0x3FFu)) & 0x3FFu) << shift;
    }
    return w;
}
/// The same with components of `bits` (8-10) in the word's top bits: the
/// residuals in steps of those bits, so a coarser rotation costs less.
public uint zigzagBits(uint d, uint n) {
    const int s = int(d << (32u - n)) >> (32u - n);   // the low n bits as signed
    return uint((s << 1) ^ (s >> 31)) & ((1u << n) - 1u);
}
public uint unzigzagBits(uint z, uint n) { return ((z >> 1) ^ (0u - (z & 1u))) & ((1u << n) - 1u); }
public uint rotationResidualBits(uint now, uint predicted, uint bits) {
    const uint n = min(bits, 10u), drop = 10u - n;
    uint r = ((now ^ predicted) & 3u) << 30u;
    for (uint c = 0; c < 3; ++c) {
        const uint shift = 22u - 10u * c + drop;
        r |= zigzagBits((now >> shift) - (predicted >> shift), n) << (10u * c);
    }
    return r;
}
public uint rotationFromBits(uint residual, uint predicted, uint bits) {
    const uint n = min(bits, 10u), drop = 10u - n;
    const uint mask = (1u << n) - 1u;
    uint w = (predicted ^ (residual >> 30u)) & 3u;
    for (uint c = 0; c < 3; ++c) {
        const uint shift = 22u - 10u * c + drop;
        w |= (((predicted >> shift) + unzigzagBits((residual >> (10u * c)) & mask, n)) & mask) << shift;
    }
    return w;
}

/// The prediction of a 10-bit component, or a 16-bit field, by order.
public uint predictField(uint b, uint c, uint order, uint mask) {
    return order >= 2 ? (2u * b - c) & mask : b;
}
public uint predictRotation(uint b, uint c, uint order) {
    if (order < 2) {
        return b;
    }
    uint w = b & 3u;
    for (uint k = 0; k < 3; ++k) {
        const uint shift = 22u - 10u * k;
        w |= predictField((b >> shift) & 0x3FFu, (c >> shift) & 0x3FFu, 2u, 0x3FFu) << shift;
    }
    return w;
}
/// The prediction order for gaussian j at frame k: 0 a key (absolute), 1
/// the frame after its key, else `order`.
public uint geometryOrder(uint j, uint frame, uint interval, uint stagger, uint order) {
    if (geometryKey(j, frame, interval, stagger)) {
        return 0u;
    }
    const uint before = frame == 0 ? 0u : frame - 1u;
    if (frame == 0 || geometryKey(j, before, interval, stagger)) {
        return 1u;
    }
    return min(order, 2u);
}

/// Field f (0-5 the six 16-bit fields, 6 the rotation) of a geometry's words.
public uint geometryField(uint4 g, uint f) {
    if (f == 6) {
        return g.w;
    }
    const uint w = f < 2 ? g.x : (f < 4 ? g.y : g.z);
    return (f & 1u) == 0 ? (w & 0xFFFFu) : (w >> 16u);
}
