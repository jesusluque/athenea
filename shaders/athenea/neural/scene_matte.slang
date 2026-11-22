// Copyright (c) 2026 jesus luque.
//
// WHICH PRIM A FOUND SCENE'S GAUSSIAN WAS MADE FROM, AND THE MATTE OF IT. A
// gaussian a look makes (scene_render.slang) stands for a cell of the frame,
// and the frame says which prim its middle pixel shows: the first rank of
// the engine's Cryptomatte there, the id that is the hash of the prim's path
// (athenea::core::cryptomatteId). The gaussian keeps it -- a uint beside the
// piece's records, in their order -- and a scene loaded with them is a cloud
// that carries `scene::GpuSplats::crypto`, so the blend that draws its
// picture also says, a pixel, which ids covered it and by how much (the
// rasteriser's own Cryptomatte: `RenderTargets::crypto`).
//
// A matte is then a sum: the coverage of the ranks whose id is one of those
// asked for. What stands in front took its share in the blend; what is
// asked for and stands behind something asked for too is counted once each,
// as the blend counted them.
module scene_matte;

struct MatteParams {
    uint width;
    uint height;
    uint footprint;   // mattePrims: pixels a side of the cell one gaussian is made of
    uint at;          // mattePrims: where the Cryptomatte's first layer starts in `frameIds`, in pixels
    uint count;       // matteGather, matteIndex: the piece's gaussians. matteOf: the ids asked for. matteWords: the indices
    uint base;        // matteGather: the piece's first place in the cut
    uint named;       // matteIndex: the ids in `wanted`
    uint pad1;
};

StructuredBuffer<float4>   frameIds;   // mattePrims: the frame's Cryptomatte, (id, coverage, id, coverage) a pixel
StructuredBuffer<uint>     which;      // mattePrims: a cell: 1 its surface's gaussian, 2 what stands behind it
StructuredBuffer<uint>     dest;       // mattePrims: its first's place among those made. matteGather, matteIndex: a gaussian's place among the drawn
StructuredBuffer<uint>     drawn;      // matteGather, matteIndex: a uint a gaussian: drawn
StructuredBuffer<uint>     primsRead;  // matteGather, matteIndex: the piece's ids. matteWords: an index a gaussian
RWStructuredBuffer<uint>   prims;      // mattePrims: an id a gaussian made. matteGather: the cut's. matteIndex: an index a gaussian handed on. matteWords: two a word
StructuredBuffer<float4>   crypto;     // matteOf: the frame's three layers, plane after plane
StructuredBuffer<uint>     wanted;     // matteOf: the ids asked for, in ascending order. matteIndex: the scene's, the same
RWStructuredBuffer<float4> matte;      // matteOf: the coverage, in every channel
ConstantBuffer<MatteParams> params;

/// The id of the prim a cell's surface is: its middle pixel's first rank, where it covers anything. What
/// stands behind a pane was seen through it, and the frame names the pane there: it is left unnamed.
[shader("compute")]
[numthreads(16, 16, 1)]
void mattePrims(uint3 tid: SV_DispatchThreadID) {
    const uint cellsX = (params.width + params.footprint - 1u) / params.footprint;
    const uint cellsY = (params.height + params.footprint - 1u) / params.footprint;
    if (tid.x >= cellsX || tid.y >= cellsY) {
        return;
    }
    const uint cell = tid.y * cellsX + tid.x;
    const uint kinds = which[cell];
    if (kinds == 0u) {
        return;
    }
    // scene_render.slang's middleOf.
    const uint half = params.footprint / 2u;
    const uint2 mid = uint2(min(tid.x * params.footprint + half, params.width - 1u),
                            min(tid.y * params.footprint + half, params.height - 1u));
    const float4 first = frameIds[params.at + mid.y * params.width + mid.x];
    uint place = dest[cell];
    if ((kinds & 1u) != 0u) {
        prims[place] = first.y > 0.0 ? asuint(first.x) : 0u;
        place += 1u;
    }
    if ((kinds & 2u) != 0u) {
        prims[place] = 0u;
    }
}

/// A piece's ids carried to the cut, as scene_discover.slang's discoverGather carries its gaussians.
[shader("compute")]
[numthreads(64, 1, 1)]
void matteGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || drawn[i] == 0u) {
        return;
    }
    prims[params.base + dest[i]] = primsRead[i];
}

/// Where an id stands among `count` of `wanted`, which are in ascending order; `count` where it is not there.
uint placeOf(uint id, uint count) {
    uint low = 0u;
    uint high = count;
    while (low < high) {
        const uint mid = (low + high) / 2u;
        if (wanted[mid] < id) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return low < count && wanted[low] == id ? low : count;
}

bool asked(uint id) {
    return id != 0u && params.count != 0u && placeOf(id, params.count) < params.count;
}

/// WHAT A VIEWER IS HANDED IN PLACE OF AN ID (neural-scene export): of a piece's gaussians those kept
/// (`drawn`: scene_level.slang's levelFlags, so the same gaussians as the part's file, in its order), each
/// one's place in the scene's list of prims counted from one -- the list in ascending order of id, which is
/// the order the index names them in -- and 0 for a gaussian no prim names.
[shader("compute")]
[numthreads(64, 1, 1)]
void matteIndex(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || drawn[i] == 0u) {
        return;
    }
    const uint id = primsRead[i];
    const uint place = id != 0u ? placeOf(id, params.named) : params.named;
    prims[dest[i]] = place < params.named ? place + 1u : 0u;
}

/// Those indices two a word, the earlier in the low half: a file of sixteen bits a gaussian, as it is written.
[shader("compute")]
[numthreads(64, 1, 1)]
void matteWords(uint3 tid: SV_DispatchThreadID) {
    const uint first = tid.x * 2u;
    if (first >= params.count) {
        return;
    }
    const uint second = first + 1u < params.count ? primsRead[first + 1u] : 0u;
    prims[tid.x] = (primsRead[first] & 0xFFFFu) | ((second & 0xFFFFu) << 16);
}

/// The matte of the ids asked for: what they cover of each pixel.
[shader("compute")]
[numthreads(16, 16, 1)]
void matteOf(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint pixels = params.width * params.height;
    const uint p = tid.y * params.width + tid.x;
    float cover = 0.0;
    for (uint layer = 0u; layer < 3u; ++layer) {
        const float4 ranks = crypto[layer * pixels + p];
        if (ranks.y > 0.0 && asked(asuint(ranks.x))) {
            cover += ranks.y;
        }
        if (ranks.w > 0.0 && asked(asuint(ranks.z))) {
            cover += ranks.w;
        }
    }
    cover = saturate(cover);
    matte[p] = float4(cover, cover, cover, cover);
}
