// Copyright (c) 2026 jesus luque.
//
// WHICH WAYS OUT ARE OPEN, AGAINST A ROOF WHOSE SHADOW HAS A CLOSED FORM.
//
// A point on a plane under a square roof sees the sky in every direction of
// its hemisphere except the ones that meet the roof, and whether a direction
// meets it is a line against a square. The bake traced one ray a cell of the
// 8 x 8 octahedral grid, at the cell's centre; this walks the same centres
// and says what each bit must be. A cell whose centre passes within `margin`
// of the roof's edge is left out, because a ray there is as likely to graze
// either side of it; a cell below the point's horizon must read clear,
// because nothing was traced there.
module shadow_bits_check;
import athenea.common.octahedral;

struct ShadowBitsCheckParams {
    uint   count;        ///< points
    uint   entries;      ///< float4 a point; the open directions are the last
    float  roofHeight;   ///< above the plane the points stand on
    float  roofHalf;     ///< half the side of the roof, centred over the origin

    float3 normal;       ///< the plane's, +z
    float  margin;       ///< how near the roof's edge a cell is not judged
};

StructuredBuffer<float4>  baked;
StructuredBuffer<float4>  points;    ///< xyz where each point stands
RWStructuredBuffer<uint>  stats;     ///< 8 wide
ConstantBuffer<ShadowBitsCheckParams> params;

/// stats[0] cells whose bit is not what the roof says
/// stats[1] cells judged
/// stats[2] cells set (open), of those judged
/// stats[3] cells below the horizon that read set
/// stats[4] cells left out at the roof's edge or on the horizon
[shader("compute")]
[numthreads(64, 1, 1)]
void shadowBitsCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= params.count) {
        return;
    }
    const float4 packed = baked[k * params.entries + params.entries - 1];
    const uint2 bits = uint2(asuint(packed.x), asuint(packed.y));
    const float3 p = points[k].xyz;
    for (uint cell = 0; cell < 64u; ++cell) {
        const float3 d = octDecode((float2(float(cell % 8u), float(cell / 8u)) + 0.5) / 8.0);
        const bool open = (((cell < 32u ? bits.x : bits.y) >> (cell & 31u)) & 1u) != 0u;
        // Sixteen of the grid's centres lie on the equator itself, where a
        // ray runs along the plane and which side the bake counted it on is
        // rounding: those are the horizon, and are not judged.
        if (abs(dot(d, params.normal)) < 1.0e-4) {
            InterlockedAdd(stats[4], 1u);
            continue;
        }
        if (dot(d, params.normal) < 0.0) {
            if (open) {
                InterlockedAdd(stats[3], 1u);
            }
            continue;
        }
        // Where the line meets the roof's plane, and how far inside the
        // square that is: negative outside.
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
