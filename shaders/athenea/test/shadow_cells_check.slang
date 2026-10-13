// Copyright (c) 2026 jesus luque.
//
// WHICH WAYS OUT ARE OPEN AT 16 x 16 OR 32 x 32 CELLS, AGAINST A ROOF.
//
// The TX transfer (docs/decisions.md, task TX) traces one ray a cell of a
// finer octahedral grid over the whole sphere -- the half below the surface
// as well -- and keeps a bit where the ray left the scene. Over a single-sided
// floor under a square roof every bit has a closed form: above the floor a
// line against the roof's square, and below it nothing at all, since the
// floor is a sheet with empty space beneath. A cell whose centre passes within
// `margin` of the roof's edge, or lies on the horizon, is left out.
module shadow_cells_check;
import athenea.common.octahedral;

struct ShadowCellsCheckParams {
    uint   count;        ///< points
    uint   entries;      ///< float4 a point
    uint   firstPlane;   ///< the first float4 of the bits, after the coverage
    uint   side;         ///< 16 or 32

    float  roofHeight;   ///< above the plane the points stand on
    float  roofHalf;     ///< half the side of the roof, centred over the origin
    float  margin;       ///< how near the roof's edge a cell is not judged
    float  pad0;

    float3 normal;       ///< the plane's, +z
    float  pad1;
};

StructuredBuffer<float4>  baked;
StructuredBuffer<float4>  points;    ///< xyz where each point stands
RWStructuredBuffer<uint>  stats;     ///< 8 wide
ConstantBuffer<ShadowCellsCheckParams> params;

/// stats[0] cells above the floor whose bit is not what the roof says
/// stats[1] cells above the floor judged
/// stats[2] of those, cells set (open)
/// stats[3] cells below the floor that read closed
/// stats[4] cells left out at the roof's edge or on the horizon
/// stats[5] cells below the floor judged
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowCellsCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float3 p = points[k].xyz;
    const uint cells = params.side * params.side;
    for (uint cell = 0; cell < cells; ++cell) {
        const float4 packed = baked[k * params.entries + params.firstPlane + (cell >> 7u)];
        const uint word = asuint(packed[(cell >> 5u) & 3u]);
        const bool open = ((word >> (cell & 31u)) & 1u) != 0u;
        const float3 d = octDecode((float2(float(cell % params.side), float(cell / params.side)) + 0.5) /
                                   float(params.side));
        const float up = dot(d, params.normal);
        if (abs(up) < 0.02) {
            InterlockedAdd(stats[4], 1u);
            continue;
        }
        if (up < 0.0) {
            if (!open) {
                InterlockedAdd(stats[3], 1u);
            }
            InterlockedAdd(stats[5], 1u);
            continue;
        }
        const float t = params.roofHeight / d.z;
        const float2 at = p.xy + d.xy * t;
        const float inside = params.roofHalf - max(abs(at.x), abs(at.y));
        if (abs(inside) < params.margin) {
            InterlockedAdd(stats[4], 1u);
            continue;
        }
        const bool wantOpen = inside < 0.0;
        if (open != wantOpen) {
            InterlockedAdd(stats[0], 1u);
        }
        if (open) {
            InterlockedAdd(stats[2], 1u);
        }
        InterlockedAdd(stats[1], 1u);
    }
}
