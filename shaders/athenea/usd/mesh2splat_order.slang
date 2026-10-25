// Copyright (c) 2026 jesus luque.
//
// THE CLOUD IN THE ORDER OF ITS SURFACE (athenea mesh2splat): each gaussian's
// key from where on its source it stands (common/source_order.slang), a
// stable radix sort of those keys, and every
// buffer the conversion holds a gaussian long laid out again in that order.
//
//   m2sSourceKey  the key and the 64-bit id of each gaussian, its index as the
//                 sort's value; a gaussian a reader would drop keyed last
//   m2sPermute    `words` words a gaussian, from `from[order[i]]` to `to[i]`
//   m2sSourceKeyOf  the key from the coordinates a file carries (a decimated
//                 cloud's, usd::decimateStage): every gaussian kept
module mesh2splat_order;
import athenea.common.source_order;

struct OrderParams {
    uint count;       ///< gaussians
    uint perRecord;   ///< floats a record (m2sSourceKey)
    uint words;       ///< words a gaussian (m2sPermute)
    uint pad0;
};

StructuredBuffer<float>      records;      // the cloud's records: position, opacity, the three sizes first
StructuredBuffer<uint4>      sources;      // (mesh rank, triangle, cell i | j << 16, rank in its triangle)
StructuredBuffer<float2>     sourceUv;     // st, NaN where the mesh has none
StructuredBuffer<uint2>      meshHashes;   // a mesh rank's path, hashed to 64 bits by the host
RWStructuredBuffer<uint>     keysLo;
RWStructuredBuffer<uint>     keysHi;
RWStructuredBuffer<uint>     values;
RWStructuredBuffer<uint2>    ids;
ConstantBuffer<OrderParams>  order;

/// What a reader keeps (scene/splat_validate.slang, for linear records): a
/// finite position, sizes of which at most one is zero and none past e^60,
/// and an opacity of at least one step of eight bits.
bool m2sKept(uint i) {
    const uint at = i * order.perRecord;
    const float3 p = float3(records[at], records[at + 1], records[at + 2]);
    const float a = records[at + 3];
    const float3 s = float3(records[at + 4], records[at + 5], records[at + 6]);
    uint flats = 0;
    bool fine = all(isfinite(p)) && a >= 1.0 / 255.0;
    for (uint k = 0; k < 3; ++k) {
        if (s[k] == 0.0) {
            ++flats;
        } else if (!(s[k] > 0.0) || !isfinite(s[k]) || abs(log(s[k])) >= 60.0) {
            fine = false;
        }
    }
    return fine && flats <= 1u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sSourceKey(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= order.count) {
        return;
    }
    const uint4 source = sources[i];
    const uint2 key = m2sKept(i) ? sourceKey(source.x, sourceUv[i], source.y, source.z) : kSourceLast;
    keysLo[i] = key.x;
    keysHi[i] = key.y;
    values[i] = i;
    ids[i] = sourceId(meshHashes[source.x], source.y, source.z, source.w);
}

StructuredBuffer<uint>     from;
RWStructuredBuffer<uint>   to;
StructuredBuffer<uint>     sorted;   // the sort's values: the gaussian at each place

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sPermute(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= order.count) {
        return;
    }
    const uint src = sorted[i] * order.words;
    const uint dst = i * order.words;
    for (uint k = 0; k < order.words; ++k) {
        to[dst + k] = from[src + k];
    }
}

StructuredBuffer<int>        fileMesh;       // primvars:athenea:splat:sourceMesh
StructuredBuffer<int>        fileTriangle;   // sourceTriangle
StructuredBuffer<int2>       fileCell;       // sourceCell
StructuredBuffer<float2>     fileUv;         // sourceUv, NaN where the cloud has none

[shader("compute")]
[numthreads(256, 1, 1)]
void m2sSourceKeyOf(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= order.count) {
        return;
    }
    const int2 cell = fileCell[i];
    const uint2 key = sourceKey(uint(max(fileMesh[i], 0)), fileUv[i], uint(max(fileTriangle[i], 0)),
                                (uint(cell.x) & 0xFFFFu) | (uint(cell.y) << 16));
    keysLo[i] = key.x;
    keysHi[i] = key.y;
    values[i] = i;
}
