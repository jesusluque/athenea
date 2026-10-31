// Copyright (c) 2026 jesus luque.
//
// NanoVDB grids read on the device through PNanoVDB (OpenVDB's own
// portable reader, included as HLSL: its buffer is a StructuredBuffer<uint>
// and its functions are what Slang compiles). Several grids share one
// buffer, each at a 32-byte aligned word offset and each of its own value
// type, read by it; a leaf is addressed by its
// index in the tree's leaf array, which is how the bounds and majorant
// kernels walk a grid without traversing it. PNanoVDB's own types stay
// inside this module; what it exports names grids and leaves by offset.
module nanovdb;

#define PNANOVDB_HLSL
#include "nanovdb/PNanoVDB.h"

/// A grid: its word offset in the buffer, and its value type (NanoVDB's
/// GridType code, the header's): float, double, int32, Vec3f, Vec3d, mask
/// or half. Every read below goes by it.
public struct NvGrid {
    public uint word;
    public uint type;
};

/// A leaf: its byte offset in the buffer, and its grid's value type.
public struct NvLeaf {
    public uint byteOffset;
    public uint type;
};

public static const uint kNvFloat = PNANOVDB_GRID_TYPE_FLOAT;
public static const uint kNvDouble = PNANOVDB_GRID_TYPE_DOUBLE;
public static const uint kNvInt32 = PNANOVDB_GRID_TYPE_INT32;
public static const uint kNvVec3f = PNANOVDB_GRID_TYPE_VEC3F;
public static const uint kNvVec3d = PNANOVDB_GRID_TYPE_VEC3D;
public static const uint kNvMask = PNANOVDB_GRID_TYPE_MASK;
public static const uint kNvHalf = PNANOVDB_GRID_TYPE_HALF;

pnanovdb_grid_handle_t handleOf(NvGrid g) {
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = g.word * 4u;
    return grid;
}

pnanovdb_leaf_handle_t handleOf(NvLeaf l) {
    pnanovdb_leaf_handle_t leaf;
    leaf.address.byte_offset = l.byteOffset;
    return leaf;
}

pnanovdb_root_handle_t rootOf(StructuredBuffer<uint> buf, NvGrid g) {
    return pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, handleOf(g)));
}

/// The grid at `wordOffset`, its type read from its header.
public NvGrid nvGrid(StructuredBuffer<uint> buf, uint wordOffset) {
    NvGrid g;
    g.word = wordOffset;
    pnanovdb_grid_handle_t grid;
    grid.address.byte_offset = wordOffset * 4u;
    g.type = pnanovdb_grid_get_grid_type(buf, grid);
    return g;
}

public uint nvLeafCount(StructuredBuffer<uint> buf, NvGrid g) {
    return pnanovdb_tree_get_node_count_leaf(buf, pnanovdb_grid_get_tree(buf, handleOf(g)));
}

/// A leaf's size in bytes in a grid of `type` (8x8x8 voxels, the values' width).
public uint nvLeafSize(uint type) {
    return PNANOVDB_GRID_TYPE_GET(type, leaf_size);
}

/// Leaf `i` of the tree's leaf array.
public NvLeaf nvLeaf(StructuredBuffer<uint> buf, NvGrid g, uint i) {
    const pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, handleOf(g));
    pnanovdb_address_t at = pnanovdb_address_offset64(tree.address, pnanovdb_tree_get_node_offset_leaf(buf, tree));
    at = pnanovdb_address_offset(at, i * nvLeafSize(g.type));
    NvLeaf leaf;
    leaf.byteOffset = at.byte_offset;
    leaf.type = g.type;
    return leaf;
}

/// A leaf's origin in index space; its voxels span [origin, origin + 8).
public int3 nvLeafOrigin(StructuredBuffer<uint> buf, NvLeaf leaf) {
    const pnanovdb_coord_t c = pnanovdb_leaf_get_bbox_min(buf, handleOf(leaf));
    return int3(c.x, c.y, c.z);
}

/// Whether voxel `n` (0..511, z fastest) of a leaf is active.
public bool nvLeafActive(StructuredBuffer<uint> buf, NvLeaf leaf, uint n) {
    return pnanovdb_leaf_get_value_mask(buf, handleOf(leaf), n);
}

/// A double's bits (low word first) as the nearest float: Metal has no
/// double, so the conversion is integer arithmetic on the device. Zero for
/// a subnormal or an underflow, infinity past the float range, NaN kept.
public float nvDoubleBits(uint lo, uint hi) {
    const uint sign = hi & 0x80000000u;
    const int exponent = int((hi >> 20) & 0x7FFu);
    if (exponent == 0) return asfloat(sign);
    if (exponent == 0x7FF) return asfloat(sign | 0x7F800000u | (((hi & 0xFFFFFu) | lo) != 0u ? 0x400000u : 0u));
    int e = exponent - 1023 + 127;
    // The top 23 of the 52 mantissa bits, rounded at the 24th.
    uint mantissa = ((hi & 0xFFFFFu) << 3) | (lo >> 29);
    mantissa += (lo >> 28) & 1u;
    if (mantissa >= 0x800000u) {
        mantissa = 0u;
        e += 1;
    }
    if (e >= 255) return asfloat(sign | 0x7F800000u);
    if (e <= 0) return asfloat(sign);
    return asfloat(sign | (uint(e) << 23) | mantissa);
}

/// The value at a byte address in a grid of `type`: xyz, and w the number
/// of components (1 a scalar, 3 a vector). A mask has no value stored; its
/// value is its active state, which the callers below read instead.
public float4 nvRead(StructuredBuffer<uint> buf, uint type, uint byteOffset) {
    pnanovdb_address_t at;
    at.byte_offset = byteOffset;
    const uint w = byteOffset / 4u;
    if (type == PNANOVDB_GRID_TYPE_FLOAT) return float4(asfloat(buf[w]), 0.0, 0.0, 1.0);
    if (type == PNANOVDB_GRID_TYPE_DOUBLE) return float4(nvDoubleBits(buf[w], buf[w + 1u]), 0.0, 0.0, 1.0);
    if (type == PNANOVDB_GRID_TYPE_INT32) return float4(float(asint(buf[w])), 0.0, 0.0, 1.0);
    if (type == PNANOVDB_GRID_TYPE_HALF) return float4(pnanovdb_read_half(buf, at), 0.0, 0.0, 1.0);
    if (type == PNANOVDB_GRID_TYPE_VEC3F) return float4(asfloat(buf[w]), asfloat(buf[w + 1u]), asfloat(buf[w + 2u]), 3.0);
    if (type == PNANOVDB_GRID_TYPE_VEC3D) {
        return float4(nvDoubleBits(buf[w], buf[w + 1u]), nvDoubleBits(buf[w + 2u], buf[w + 3u]),
                      nvDoubleBits(buf[w + 4u], buf[w + 5u]), 3.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

/// A value's scalar: itself, or a vector's length (a vector grid named as a
/// density is read as its magnitude).
public float nvScalar(float4 v) {
    return v.w > 1.5 ? length(v.xyz) : v.x;
}

/// Voxel `n`'s value in a leaf, by its grid's type (a mask: 1 where active).
public float4 nvLeafValue(StructuredBuffer<uint> buf, NvLeaf leaf, uint n) {
    if (leaf.type == PNANOVDB_GRID_TYPE_MASK) {
        return float4(nvLeafActive(buf, leaf, n) ? 1.0 : 0.0, 0.0, 0.0, 1.0);
    }
    return nvRead(buf, leaf.type, pnanovdb_leaf_get_table_address(leaf.type, buf, handleOf(leaf), n).byte_offset);
}

/// The word voxel `n`'s value sits at in a float leaf, for a kernel that
/// rewrites it.
public uint nvLeafValueWord(StructuredBuffer<uint> buf, NvLeaf leaf, uint n) {
    return pnanovdb_leaf_get_table_address(PNANOVDB_GRID_TYPE_FLOAT, buf, handleOf(leaf), n).byte_offset / 4u;
}

/// The value at index coordinate `ijk`, through an accessor made for the
/// call: a leaf's voxel, a tile's value or the background, by the grid's type.
public float4 nvValueAt(StructuredBuffer<uint> buf, NvGrid g, int3 ijk) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootOf(buf, g));
    pnanovdb_coord_t c;
    c.x = ijk.x;
    c.y = ijk.y;
    c.z = ijk.z;
    if (g.type == PNANOVDB_GRID_TYPE_MASK) {
        return float4(pnanovdb_readaccessor_is_active(g.type, buf, acc, c) ? 1.0 : 0.0, 0.0, 0.0, 1.0);
    }
    return nvRead(buf, g.type, pnanovdb_readaccessor_get_value_address(g.type, buf, acc, c).byte_offset);
}

/// Whether the voxel at `ijk` is active.
public bool nvActiveAt(StructuredBuffer<uint> buf, NvGrid g, int3 ijk) {
    pnanovdb_readaccessor_t acc;
    pnanovdb_readaccessor_init(acc, rootOf(buf, g));
    pnanovdb_coord_t c;
    c.x = ijk.x;
    c.y = ijk.y;
    c.z = ijk.z;
    return pnanovdb_readaccessor_is_active(g.type, buf, acc, c);
}

