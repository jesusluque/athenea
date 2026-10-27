// Copyright (c) 2026 jesus luque.
//
// THE MESH LIGHTS IN CLOSED FORM, PER TILE (technique::MaterialShading):
//
//   meshLightCut      a lightcut of up to K nodes a 16 x 16 tile, from the
//                     mesh lights' tree at the tile's middle surface: the
//                     most important node split while there is room. Every
//                     emitter is under exactly one entry -- a cluster the
//                     shading integrates over its rectangle, or a node it
//                     stands in for by its most important cluster, scaled to
//                     its power and diffuse only.
//   meshShadowFilter  the shading's shadow, a ratio (Heitz, Hill and McGuire
//                     2018): of a few stochastic samples a pixel, what was
//                     visible over what was there unshadowed, each sum
//                     filtered over the neighbourhood by depth and normal
//                     before they are divided, so the closed form keeps its
//                     smoothness and the shadow its edges.
module mesh_light_cut;
import athenea.light.lights;
import athenea.light.light_bvh;
import athenea.light.mesh_lights;
import athenea.rt.bvh_common;
import athenea.common.octahedral;
import athenea.technique.material_surface;

StructuredBuffer<float>     meshRegion;
uniform uint                meshRegionWords;
Texture2D<uint4>            visibility;   // (instance + 1, triangle); row 0 on top
ConstantBuffer<CameraParams> camera;
ConstantBuffer<ViewToWorld>  toWorld;
RWTexture2D<uint>           cutOut;       // (tilesX K) x tilesY: node indices, kMeshNone past the cut
uniform uint                cutK;
uniform uint                tilesX;
uniform uint                tilesY;

static const uint kCutTile = 16u;
static const uint kCutMaxK = 16u;

/// The surface a tile shows: its middle pixel, or the nearest of a few around
/// it that sees something.
bool tileSurface(uint2 tile, out float3 p, out float3 n) {
    p = float3(0.0);
    n = float3(0.0, 0.0, 1.0);
    const int2 offsets[5] = {int2(8, 8), int2(4, 4), int2(12, 4), int2(4, 12), int2(12, 12)};
    for (uint k = 0u; k < 5u; ++k) {
        const uint2 px = tile * kCutTile + uint2(offsets[k]);
        if (px.x >= camera.width || px.y >= camera.height) {
            continue;
        }
        const uint4 seen = visibility.Load(int3(int(px.x), int(camera.height - 1u - px.y), 0));
        if (seen.x == 0u) {
            continue;
        }
        const Surface s = surfaceAt(camera, px.x, px.y, seen);
        p = applyRows(toWorld, s.viewPosition, 1.0);
        float3 normal = s.normal;
        if (dot(s.geometricNormal, s.viewDirection) > 0.0) {
            normal = -normal;
        }
        n = normalize(applyRows(toWorld, normal, 0.0));
        return true;
    }
    return false;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void meshLightCut(uint3 tid: SV_DispatchThreadID) {
    const uint2 tile = tid.xy;
    if (tile.x >= tilesX || tile.y >= tilesY) {
        return;
    }
    const uint k = min(cutK, kCutMaxK);
    uint entries[kCutMaxK];
    float weight[kCutMaxK];
    uint count = 0u;
    float3 p;
    float3 n;
    const uint clusters = meshClusterCount(meshRegion, meshRegionWords);
    if (clusters > 0u && tileSurface(tile, p, n)) {
        const uint base = meshWord(meshRegion, kMeshNodeBase);
        entries[0] = 0u;
        weight[0] = 1.0e30;
        count = 1u;
        // Split the most important node that is not a leaf, while there is room.
        for (uint step = 0u; step < 2u * kCutMaxK; ++step) {
            if (count >= k) {
                break;
            }
            int best = -1;
            float bestWeight = -1.0;
            for (uint e = 0u; e < count; ++e) {
                const LightNode node = lightNodeAt(meshRegion, base, entries[e]);
                if (node.links.w == 0u && weight[e] > bestWeight) {
                    best = int(e);
                    bestWeight = weight[e];
                }
            }
            if (best < 0) {
                break;
            }
            const LightNode node = lightNodeAt(meshRegion, base, entries[best]);
            const uint a = node.links.x;
            const uint b = node.links.y;
            // An importance of zero at the middle still keeps its place, by
            // power: the tile is wider than its middle.
            entries[best] = a;
            const LightNode na = lightNodeAt(meshRegion, base, a);
            weight[best] = lightImportance(na, p, n) + 1.0e-6 * na.lo.w;
            entries[count] = b;
            const LightNode nb = lightNodeAt(meshRegion, base, b);
            weight[count] = lightImportance(nb, p, n) + 1.0e-6 * nb.lo.w;
            ++count;
        }
    }
    for (uint e = 0u; e < k; ++e) {
        cutOut[uint2(tile.x * k + e, tile.y)] = e < count ? entries[e] : kMeshNone;
    }
}

// ---- the shadow's ratio ------------------------------------------------------------

StructuredBuffer<float4>  ratio;        // a pixel: visible, unshadowed, view z, the normal (octahedral, 16 bits each)
RWTexture2D<float>        shadowOut;    // a pixel: the filtered ratio; 1 where nothing was measured
uniform uint              radius;       // the filter's half width, in pixels

float3 ratioNormal(float4 r) {
    const uint word = asuint(r.w);
    const float2 e = float2(float(word & 0xFFFFu), float(word >> 16u)) / 65535.0;
    return octDecode(e);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void meshShadowFilter(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    const float4 centre = ratio[at];
    if (!(centre.z > 0.0)) {
        shadowOut[tid.xy] = 1.0;
        return;
    }
    const float3 n0 = ratioNormal(centre);
    float visible = 0.0;
    float total = 0.0;
    const int r = int(radius);
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            const int2 q = int2(tid.xy) + int2(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= int(camera.width) || q.y >= int(camera.height)) {
                continue;
            }
            const float4 s = ratio[uint(q.y) * camera.width + uint(q.x)];
            if (!(s.z > 0.0)) {
                continue;
            }
            // Joint weights: a neighbour on another surface -- another depth,
            // another facing -- says nothing about this one's shadow.
            const float depthWeight = exp(-abs(s.z - centre.z) / (0.02 * centre.z));
            const float normalWeight = pow(saturate(dot(ratioNormal(s), n0)), 16.0);
            const float spatial = exp(-float(dx * dx + dy * dy) / (2.0 * float(r * r) * 0.5 + 1.0));
            const float w = depthWeight * normalWeight * spatial;
            visible += w * s.x;
            total += w * s.y;
        }
    }
    shadowOut[tid.xy] = total > 0.0 ? saturate(visible / total) : 1.0;
}
