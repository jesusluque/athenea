// Copyright (c) 2026 jesus luque.
//
// THE WEB'S LEVEL OF DETAIL, what the native kernels do not already do under
// the web's limits. The build and the cut are lod/'s own kernels (lodMorton,
// lodBoundaries, lodGroups, lodLeafMoments, lodMergeMoments, lodFinalize,
// lodCutGroups, lodCutFinest, lodCutSplats); these two are:
//
//   webLodReorder  lodReorder without the normals and the emission a capture
//                  never carries: 7 storage buffers where it has 11. It
//                  writes the splats into the pool's first part, in Morton
//                  order.
//   webLodList     the cut's selection, placed by its prefix sum, as the list
//                  the projection walks (web_list.slang).
module web_lod;
import athenea.lod.lod_common;
import web_list;

StructuredBuffer<uint>     order;
StructuredBuffer<float4>   srcPositions;
StructuredBuffer<uint>     srcShape;
StructuredBuffer<uint>     srcSh;
RWStructuredBuffer<float4> positions;
RWStructuredBuffer<uint>   shape;
RWStructuredBuffer<uint>   sh;
ConstantBuffer<LodParams>  params;   // count, shWords

[shader("compute")]
[numthreads(256, 1, 1)]
void webLodReorder(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const uint i = order[k];
    positions[k] = srcPositions[i];
    for (uint w = 0; w < 4; ++w) {
        shape[k * 4 + w] = srcShape[i * 4 + w];
    }
    for (uint w = 0; w < params.shWords; ++w) {
        sh[k * params.shWords + w] = srcSh[i * params.shWords + w];
    }
}

public struct ListParams {
    public uint count;      // pool entries
    public uint capacity;   // list entries the projection has slots for
    public uint pad0;
    public uint pad1;
};

StructuredBuffer<uint>     selected;
StructuredBuffer<uint>     dest;
RWStructuredBuffer<uint>   listWords;
ConstantBuffer<ListParams> list;

[shader("compute")]
[numthreads(256, 1, 1)]
void webLodList(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= list.count || selected[i] == 0) {
        return;
    }
    const uint at = dest[i];
    if (at < list.capacity) {
        listWords[kListHead + at] = i;
    }
}
