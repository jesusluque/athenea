// Copyright (c) 2026 jesus luque.
//
// THE FOUND SCENE AS WHAT A BOUNCE MEETS.
//
// A found scene's gaussian holds the light that leaves a surface where it stands. A path that bounces off a
// surface and lands where the scene already holds one need not shade it, light it and bounce again to learn
// that light: the scene says it. This keeps the scene's light in cells of the world -- a cube of `side` cells
// an axis over the stage's bounds, a mean a cell -- so a path tracing kernel reads it by a position alone.
//
// Plane 0 is the scene's own light; planes 1.. are its groups of lights' layers, each a gaussian's light under
// that group alone. A cell's count is plane 0's: a gaussian a group does not light is in the cell and sends
// none of that group's light.
//
// Not kept: a gaussian that lets light through (its colour is what stands behind it) and one that mirrors
// (its colour is what its maker's eye saw in it).
module scene_cache;
import athenea.common.packing;
import athenea.common.color;
import athenea.scene.splat_encoding;

struct CacheParams {
    uint  count;        // gaussians (cacheAdd); cells (cacheResolve, cacheClear: words)
    uint  side;         // cells an axis
    uint  plane;        // which plane this cloud's light goes into
    uint  linearCloud;
    float minX; float minY; float minZ;
    float cell;         // a cell's edge
    uint  hasPbr;
    uint  hasLayer;     // the light is `layer`'s (a group's), not the cloud's own
    uint  planes;       // cacheResolve: how many planes
    uint  pad0;
};

StructuredBuffer<float4>   positions;   // xyz, opacity (0 and under: not standing)
StructuredBuffer<uint>     shape;       // four words a gaussian; the colour is read
StructuredBuffer<uint>     pbr;         // metallic, roughness, transmission
StructuredBuffer<float4>   layer;       // a group's light a gaussian (rgb, linear; a: over 0 where it sends any)
RWStructuredBuffer<uint>   grid;        // four words a cell a plane: the light summed in 1/256ths, and (plane 0) how many
RWStructuredBuffer<float4> cells;       // the mean light a cell a plane, and (a) how many gaussians the cell holds
ConstantBuffer<CacheParams> params;

uint cacheHash(uint v) {
    uint s = v * 747796405u + 2891336453u;
    const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cacheAdd(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || positions[i].w <= 0.0) {
        return;
    }
    if (params.hasPbr != 0u) {
        const uint word = pbr[i];
        if (pbrTransmission(word) > 0.02 || (pbrMetallic(word) > 0.5 && pbrRoughness(word) < 0.3)) {
            return;
        }
    }
    const float3 at = (positions[i].xyz - float3(params.minX, params.minY, params.minZ)) / params.cell;
    if (any(at < float3(0.0)) || any(at >= float3(float(params.side)))) {
        return;
    }
    float3 light;
    if (params.hasLayer != 0u) {
        const float4 said = layer[i];
        if (!(said.a > 0.0)) {
            return;
        }
        light = max(said.rgb, float3(0.0));
    } else {
        const uint row = i * 4u;
        const float3 base = float3(highHalf(shape[row + 2u]), lowHalf(shape[row + 3u]), highHalf(shape[row + 3u]));
        light = max(cloudLight(base, params.linearCloud != 0u), float3(0.0));
    }
    const uint3 c = uint3(at);
    const uint slot = (params.plane * params.side * params.side * params.side +
                       (c.z * params.side + c.y) * params.side + c.x) * 4u;
    // In 1/256ths, rounded up or down by a lot of the gaussian's own so that a cell of dark gaussians does
    // not read as none; held to 16 so that a cell of a million gaussians fits its word.
    const float3 scaled = min(light, float3(16.0)) * 256.0;
    for (uint k = 0; k < 3u; ++k) {
        const float lot = float(cacheHash(i * 3u + k + params.plane * 7919u) >> 8) * (1.0 / 16777216.0);
        InterlockedAdd(grid[slot + k], uint(scaled[k] + lot));
    }
    if (params.plane == 0u) {
        InterlockedAdd(grid[slot + 3u], 1u);
    }
}

// The grid's words set to nothing: `count` of them.
[shader("compute")]
[numthreads(256, 1, 1)]
void cacheClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.count) {
        grid[tid.x] = 0u;
    }
}

// The cells' means, a plane after another: the sums over how many gaussians the cell holds (plane 0's count).
[shader("compute")]
[numthreads(256, 1, 1)]
void cacheResolve(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint held = grid[tid.x * 4u + 3u];
    for (uint p = 0; p < params.planes; ++p) {
        const uint slot = (p * params.count + tid.x) * 4u;
        const float3 sum = float3(float(grid[slot]), float(grid[slot + 1u]), float(grid[slot + 2u]));
        cells[p * params.count + tid.x] = held > 0u ? float4(sum / (256.0 * float(held)), float(held)) : float4(0.0);
    }
}
