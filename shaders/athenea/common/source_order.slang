// Copyright (c) 2026 jesus luque.
//
// WHERE ON ITS SOURCE A GAUSSIAN STANDS, AND THE ORDER THAT FOLLOWS FROM IT.
//
// A gaussian `athenea mesh2splat` makes carries its source coordinates
// (`primvars:athenea:splat:source*`): the mesh it came from (its rank among the
// stage's meshes by prim path), the triangle, the cell of that triangle's walk
// and, where the mesh has texture coordinates, the `st` of the point. The
// cloud's order is a function of those and of nothing else, which is what
// lets index i be one point of the surface in every frame of an animation, in
// every rerun, and in a coarser or finer conversion of the same asset:
//
//   key = mesh rank (16 bits) | the st's unit tile (12) | 0 (4) | the Hilbert
//         index of st within that tile, 16 bits an axis (32)
//
// and, for a mesh without st, mesh rank (16) | triangle (24) | the Morton
// index of the cell in the triangle's walk, 12 bits an axis (24). A gaussian
// a reader drops (no finite position, no valid size, no opacity) takes the
// largest key, so it comes last and the ones kept keep their indices.
//
// The key is over a fixed grid of 2^16 a unit of st, not over what a run
// happened to produce: two sets of gaussians on the same mesh, at any
// densities, are ordered by the same curve, so the coarser set's order is the
// finer's restricted to it -- interleaved, never crossed. A neighbour in the
// order is a neighbour in st (the Hilbert curve has no jumps), and so on the
// surface wherever st is continuous: a seam, a mirrored half or charts that
// overlap in st are interleaved where they share texels.
module source_order;

/// The key a gaussian without a source takes (and one a reader drops): last.
public static const uint2 kSourceLast = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);

/// The Hilbert index of a point of a 2^16 x 2^16 grid.
public uint sourceHilbert(uint x, uint y) {
    uint d = 0;
    const uint n = 65536u;
    for (uint s = n / 2; s > 0; s /= 2) {
        const uint rx = (x & s) != 0 ? 1u : 0u;
        const uint ry = (y & s) != 0 ? 1u : 0u;
        d += s * s * ((3u * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            const uint t = x;
            x = y;
            y = t;
        }
    }
    return d;
}

/// Twelve bits of each of two numbers, interleaved: 24 bits.
public uint sourceMorton12(uint x, uint y) {
    uint d = 0;
    for (uint b = 0; b < 12; ++b) {
        d |= ((x >> b) & 1u) << (2 * b);
        d |= ((y >> b) & 1u) << (2 * b + 1);
    }
    return d;
}

/// Whether a source's st is there: a mesh without st writes NaN.
public bool sourceHasUv(float2 uv) {
    return all(isfinite(uv));
}

/// The order key of a gaussian, as (low word, high word) of 64 bits.
/// `cell` is the cell of the triangle's walk, each axis 16 bits (a signed
/// count from the walk's first cell; the walk starts at 0).
public uint2 sourceKey(uint meshRank, float2 uv, uint triangle, uint cell) {
    const uint rank = min(meshRank, 0xFFFEu);
    if (sourceHasUv(uv)) {
        const float2 tile = floor(uv);
        const float2 inside = uv - tile;
        const uint2 q = uint2(min(inside * 65536.0, float2(65535.0)));
        const uint tu = uint(clamp(int(tile.x) + 32, 0, 63));
        const uint tv = uint(clamp(int(tile.y) + 32, 0, 63));
        return uint2(sourceHilbert(q.x, q.y), (rank << 16) | (((tv << 6) | tu) << 4));
    }
    const uint i = cell & 0xFFFu;
    const uint j = (cell >> 16) & 0xFFFu;
    const uint t = min(triangle, 0xFFFFFFu);
    return uint2(((t & 0xFFu) << 24) | sourceMorton12(i, j), (rank << 16) | (t >> 8));
}

uint sourceMix(uint h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

uint sourceFold(uint h, uint v) {
    return sourceMix(h ^ (v + 0x9E3779B9u + (h << 6) + (h >> 2)));
}

/// A gaussian's 64-bit id, (low, high): its mesh's path hashed (64 bits,
/// the host's), the triangle, the cell and its rank among the triangle's
/// gaussians, mixed. The same in every run at the same density; a check on
/// the order, not the order itself.
public uint2 sourceId(uint2 meshHash, uint triangle, uint cell, uint rank) {
    uint lo = meshHash.x;
    uint hi = meshHash.y ^ 0x5BD1E995u;
    for (uint k = 0; k < 3; ++k) {
        const uint v = k == 0 ? triangle : k == 1 ? cell : rank;
        lo = sourceFold(lo, v);
        hi = sourceFold(hi, v ^ 0xA5A5A5A5u);
    }
    return uint2(lo, hi);
}
