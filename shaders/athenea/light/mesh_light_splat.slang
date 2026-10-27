// Copyright (c) 2026 jesus luque.
//
// THE MESH LIGHTS AS THE SPLATS READ THEM (technique::MeshLights, stage C):
// the splat projection binds no buffer more -- it is at Metal's thirty-one --
// so what a gaussian needs of the mesh lights is copied into two textures
// each time they are refitted:
//
//   meshSplatRects   four texels a cluster, 256 clusters a row: the
//                    rectangle's centre and flags, its half axes (x and y,
//                    each times its half size), and its radiance
//   meshSplatCells   a lightcut of up to K entries for each cell of a grid
//                    over the clouds' bounds, each a cluster and the factor
//                    its entry stands for: 1 for a cluster of its own, a
//                    tree node's power over its most important cluster's for
//                    the rest (the plan's per-chunk cut, by place rather than
//                    by slot: no pass over the gaussians is needed to make it)
module mesh_light_splat;
import athenea.light.lights;
import athenea.light.light_bvh;
import athenea.light.mesh_lights;
import athenea.rt.bvh_common;

StructuredBuffer<float>  meshRegion;
uniform uint             meshRegionWords;
RWTexture2D<float4>      rectsOut;
RWTexture2D<uint2>       cellsOut;      // (cells x K) x cells^2: (cluster, factor bits); kMeshNone past the cut
uniform uint             cellSide;      // cells a side of the grid
uniform uint             cutK;
uniform float4           gridLo;        // xyz
uniform float4           gridHi;

public static const uint kSplatRectsPerRow = 256u;

[shader("compute")]
[numthreads(64, 1, 1)]
void meshSplatRects(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= meshClusterCount(meshRegion, meshRegionWords)) {
        return;
    }
    const MeshCluster m = meshCluster(meshRegion, c);
    const uint2 at = uint2((c % kSplatRectsPerRow) * 4u, c / kSplatRectsPerRow);
    rectsOut[at] = float4(m.centre, asfloat(m.flags));
    rectsOut[at + uint2(1, 0)] = float4(m.axisX * m.halfX, m.power);
    rectsOut[at + uint2(2, 0)] = float4(m.axisY * m.halfY, 0.0);
    rectsOut[at + uint2(3, 0)] = float4(m.radiance, asfloat(m.group));
}

/// How much a node can matter at p whatever way a gaussian there faces.
float importanceAt(LightNode node, float3 p) {
    const float3 centre = 0.5 * (node.lo.xyz + node.hi.xyz);
    const float3 toLight = centre - p;
    const float len = length(toLight);
    return lightImportance(node, p, len > 0.0 ? toLight / len : float3(0.0, 0.0, 1.0));
}

[shader("compute")]
[numthreads(4, 4, 4)]
void meshSplatCells(uint3 tid: SV_DispatchThreadID) {
    if (any(tid >= uint3(cellSide))) {
        return;
    }
    const uint k = min(cutK, 8u);
    const uint2 at = uint2(tid.x * k, tid.y * cellSide + tid.z);
    const uint clusters = meshClusterCount(meshRegion, meshRegionWords);
    uint entries[8];
    float weight[8];
    uint count = 0u;
    const float3 p = lerp(gridLo.xyz, gridHi.xyz, (float3(tid) + 0.5) / float(cellSide));
    const uint base = meshWord(meshRegion, kMeshNodeBase);
    if (clusters > 0u) {
        entries[0] = 0u;
        weight[0] = 1.0e30;
        count = 1u;
        for (uint step = 0u; step < 16u && count < k; ++step) {
            int best = -1;
            float bestWeight = -1.0;
            for (uint e = 0u; e < count; ++e) {
                if (lightNodeAt(meshRegion, base, entries[e]).links.w == 0u && weight[e] > bestWeight) {
                    best = int(e);
                    bestWeight = weight[e];
                }
            }
            if (best < 0) {
                break;
            }
            const LightNode node = lightNodeAt(meshRegion, base, entries[best]);
            const LightNode a = lightNodeAt(meshRegion, base, node.links.x);
            const LightNode b = lightNodeAt(meshRegion, base, node.links.y);
            entries[best] = node.links.x;
            weight[best] = importanceAt(a, p) + 1.0e-6 * a.lo.w;
            entries[count] = node.links.y;
            weight[count] = importanceAt(b, p) + 1.0e-6 * b.lo.w;
            ++count;
        }
    }
    for (uint e = 0u; e < k; ++e) {
        if (e >= count) {
            cellsOut[at + uint2(e, 0)] = uint2(kMeshNone, 0u);
            continue;
        }
        // A node of the tree stands in by its most important cluster here.
        LightNode node = lightNodeAt(meshRegion, base, entries[e]);
        const float power = node.lo.w;
        for (uint step = 0u; step < 64u && node.links.w == 0u; ++step) {
            const LightNode a = lightNodeAt(meshRegion, base, node.links.x);
            const LightNode b = lightNodeAt(meshRegion, base, node.links.y);
            node = importanceAt(a, p) >= importanceAt(b, p) ? a : b;
        }
        const uint c = node.links.x & ~kLeafBit;
        const float factor = node.lo.w > 0.0 ? power / node.lo.w : 0.0;
        cellsOut[at + uint2(e, 0)] = uint2(c, asuint(factor));
    }
}
