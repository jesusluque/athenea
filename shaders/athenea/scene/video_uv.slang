// Copyright (c) 2026 jesus luque.
//
// VIDEO PLANES LAID OUT AS A TEXTURE (athenea video-encode --layout uv): each
// gaussian's samples at the pixel its st points at, a rectangle a mesh, so a
// picture of a plane is the mesh's texture and a gaussian keeps its pixel in
// every frame. The mosaic's own order is unchanged: a plane's tile is 512
// pixels wide, splat i in 64 x 64 blocks of Morton order (video_planes_common's
// `pixelOf`); here the slot of each pixel is filled with the gaussian placed
// there, and a pixel no gaussian took holds a copy of a neighbour's with no
// opacity -- a gaussian that draws nothing and costs the codec little.
//
//   videoUvStats     each mesh's gaussians counted, its st's box and its first index
//   videoUvTarget    each gaussian's pixel: its chart's rectangle, by st (or
//                    by its index in its mesh, row by row, where it has none)
//   videoUvClaim     a round: an unplaced gaussian asks for its pixel moved by
//                    the round's offset (a spiral), the lowest index wins
//   videoUvResolve   ... and takes it
//   videoUvFreeList  the pixels left free, listed in slot order
//   videoUvLeftover  a gaussian no round placed takes the next free pixel
//   videoUvFill      each slot's gaussian: its own, or a neighbour's, flagged
//   videoUvExpand    a frame's gaussians into the slots (positions, shape, sh)
//   videoUvPicture   the base colour of each pixel, eight bits: what the planes
//                    look like
module video_uv;
import athenea.common.packing;

struct UvParams {
    uint  count;      ///< gaussians of the source frame
    uint  slots;      ///< pixels of the atlas: 512 x its height
    uint  meshes;
    uint  round;      ///< claim, resolve: the offset tried
    uint  offsets;    ///< spiral offsets there are
    uint  shWords;
    uint  height;     ///< the atlas's rows
    uint  pad0;
};

/// A mesh's rectangle in the atlas and how its st maps into it.
struct UvChart {
    uint4  rect;   ///< x0, y0, width, height in pixels
    float4 box;    ///< u min, v min, u span, v span
    uint4  info;   ///< the mesh's first index in the cloud, its count, 1 where it has st, 0
};

static const uint kTileWidth = 512;
static const uint kFlag = 0x80000000u;

StructuredBuffer<uint>       sourceMesh;   // the gaussian's mesh rank
StructuredBuffer<float2>     sourceUv;     // its st, NaN without
RWStructuredBuffer<uint>     stats;        // 8 a mesh: count, u min, v min, u max, v max (ordered bits), first
StructuredBuffer<UvChart>    charts;
StructuredBuffer<int2>       spiral;       // offsets nearest first
RWStructuredBuffer<uint>     target;       // x | y << 16
RWStructuredBuffer<uint>     claimed;      // a round's: the highest ~g asked
RWStructuredBuffer<uint>     owner;        // g + 1 placed there, 0 free
RWStructuredBuffer<uint>     placed;       // the slot + 1, 0 unplaced
RWStructuredBuffer<uint>     counts;       // placed at the target, nearby, left over
RWStructuredBuffer<uint>     freeFlag;     // 1 a free slot / an unplaced gaussian
StructuredBuffer<uint>       freeAt;       // their exclusive prefix
RWStructuredBuffer<uint>     freeList;
StructuredBuffer<uint>       leftAt;       // the unplaced gaussians' exclusive prefix
RWStructuredBuffer<uint>     slotSource;   // a slot's gaussian, kFlag where it is a neighbour's copy
ConstantBuffer<UvParams>     uv;

uint orderedBits(float f) {
    const uint b = asuint(f);
    return (b & 0x80000000u) != 0 ? ~b : (b | 0x80000000u);
}

uint spread6(uint v) {
    v &= 0x3Fu;
    v = (v | (v << 4)) & 0x0F0Fu;
    v = (v | (v << 2)) & 0x3333u;
    v = (v | (v << 1)) & 0x5555u;
    return v;
}

/// The slot of an atlas pixel: the inverse of the planes' `pixelOf` in one
/// tile -- bands of 64 rows, blocks of 64 columns, Z order inside.
uint slotOf(uint2 p) {
    return (p.y / 64u) * 32768u + (p.x / 64u) * 4096u + (spread6(p.x % 64u) | (spread6(p.y % 64u) << 1));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvStats(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= uv.count) {
        return;
    }
    const uint m = min(sourceMesh[g], uv.meshes - 1);
    const float2 st = sourceUv[g];
    InterlockedAdd(stats[m * 8], 1u);
    InterlockedMin(stats[m * 8 + 5], g);
    if (all(isfinite(st))) {
        InterlockedMin(stats[m * 8 + 1], orderedBits(st.x));
        InterlockedMin(stats[m * 8 + 2], orderedBits(st.y));
        InterlockedMax(stats[m * 8 + 3], orderedBits(st.x));
        InterlockedMax(stats[m * 8 + 4], orderedBits(st.y));
        InterlockedAdd(stats[m * 8 + 6], 1u);
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvTarget(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= uv.count) {
        return;
    }
    const UvChart c = charts[min(sourceMesh[g], uv.meshes - 1)];
    const float2 st = sourceUv[g];
    uint2 at;
    if (c.info.z != 0 && all(isfinite(st))) {
        const float2 unit = saturate((st - c.box.xy) / max(c.box.zw, float2(1.0e-20)));
        const uint x = min(uint(unit.x * float(c.rect.z)), c.rect.z - 1);
        const uint y = min(uint(unit.y * float(c.rect.w)), c.rect.w - 1);
        at = uint2(c.rect.x + x, c.rect.y + (c.rect.w - 1 - y));   // v up, as a texture is drawn
    } else {
        const uint k = g - min(g, c.info.x);
        at = uint2(c.rect.x + k % c.rect.z, c.rect.y + min(k / c.rect.z, c.rect.w - 1));
    }
    target[g] = at.x | (at.y << 16);
    placed[g] = 0u;
}

/// The pixel a gaussian asks for this round, if it lies in its rectangle.
bool candidate(uint g, out uint slot) {
    const UvChart c = charts[min(sourceMesh[g], uv.meshes - 1)];
    const int2 at = int2(int(target[g] & 0xFFFFu), int(target[g] >> 16)) + spiral[uv.round];
    slot = 0;
    if (at.x < int(c.rect.x) || at.y < int(c.rect.y) || at.x >= int(c.rect.x + c.rect.z) ||
        at.y >= int(c.rect.y + c.rect.w)) {
        return false;
    }
    slot = slotOf(uint2(at));
    return slot < uv.slots;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvClaim(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= uv.count || placed[g] != 0) {
        return;
    }
    uint slot;
    if (candidate(g, slot) && owner[slot] == 0) {
        InterlockedMax(claimed[slot], ~g);   // the lowest index wins, whoever runs first
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvResolve(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= uv.count || placed[g] != 0) {
        return;
    }
    uint slot;
    if (candidate(g, slot) && owner[slot] == 0 && claimed[slot] == ~g) {
        owner[slot] = g + 1;
        placed[g] = slot + 1;
        InterlockedAdd(counts[uv.round == 0 ? 0 : 1], 1u);
    }
}

/// Flags for the prefix sums: the free slots (first `slots`), then the
/// unplaced gaussians (in `placed`'s place, a second dispatch).
[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvFreeFlags(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (uv.round == 0) {
        if (i < uv.slots) {
            freeFlag[i] = owner[i] == 0 ? 1u : 0u;
        }
    } else if (i < uv.count) {
        freeFlag[i] = placed[i] == 0 ? 1u : 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvFreeList(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s < uv.slots && owner[s] == 0) {
        freeList[freeAt[s]] = s;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvLeftover(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= uv.count || placed[g] != 0) {
        return;
    }
    const uint s = freeList[leftAt[g]];
    owner[s] = g + 1;
    placed[g] = s + 1;
    InterlockedAdd(counts[2], 1u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvFill(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= uv.slots) {
        return;
    }
    if (owner[s] != 0) {
        slotSource[s] = owner[s] - 1;
        return;
    }
    // The pixel of this slot, and the nearest one a gaussian holds.
    const uint band = s / 32768u, block = (s % 32768u) / 4096u, z = s % 4096u;
    uint lx = 0, ly = 0;
    for (uint b = 0; b < 6; ++b) {
        lx |= ((z >> (2 * b)) & 1u) << b;
        ly |= ((z >> (2 * b + 1)) & 1u) << b;
    }
    const int2 here = int2(int(block * 64u + lx), int(band * 64u + ly));
    for (uint k = 1; k < uv.offsets; ++k) {
        const int2 at = here + spiral[k];
        if (at.x < 0 || at.y < 0 || at.x >= int(kTileWidth) || at.y >= int(uv.height)) {
            continue;
        }
        const uint o = owner[slotOf(uint2(at))];
        if (o != 0) {
            slotSource[s] = (o - 1) | kFlag;
            return;
        }
    }
    slotSource[s] = kFlag;   // nobody near: the first gaussian, drawing nothing
}

StructuredBuffer<float4>     inPositions;
StructuredBuffer<uint>       inShape;
StructuredBuffer<uint>       inSh;
RWStructuredBuffer<float4>   outPositions;
RWStructuredBuffer<uint>     outShape;
RWStructuredBuffer<uint>     outSh;

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvExpand(uint3 tid: SV_DispatchThreadID) {
    const uint s = tid.x;
    if (s >= uv.slots) {
        return;
    }
    const uint src = slotSource[s];
    const uint g = min(src & ~kFlag, uv.count - 1);
    float4 p = inPositions[g];
    if ((src & kFlag) != 0) {
        p.w = 0.0;
    }
    outPositions[s] = p;
    for (uint w = 0; w < 4; ++w) {
        outShape[s * 4 + w] = inShape[g * 4 + w];
    }
    for (uint w = 0; w < uv.shWords; ++w) {
        outSh[s * uv.shWords + w] = inSh[g * uv.shWords + w];
    }
}

RWStructuredBuffer<uint>     rgba;   // a pixel a word, row 0 at the top

[shader("compute")]
[numthreads(256, 1, 1)]
void videoUvPicture(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= uv.slots) {
        return;
    }
    const uint2 p = uint2(i % kTileWidth, i / kTileWidth);
    const uint s = slotOf(p);
    const uint src = slotSource[s];
    if ((src & kFlag) != 0) {
        rgba[i] = 0xFF000000u;   // a pixel no gaussian took: black
        return;
    }
    const uint g = min(src, uv.count - 1);
    const float3 c = saturate(float3(highHalf(inShape[g * 4 + 2]), lowHalf(inShape[g * 4 + 3]),
                                     highHalf(inShape[g * 4 + 3])));
    const uint3 b = uint3(round(c * 255.0));
    rgba[i] = b.x | (b.y << 8) | (b.z << 16) | 0xFF000000u;
}
