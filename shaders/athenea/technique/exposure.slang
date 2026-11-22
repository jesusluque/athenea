// Copyright (c) 2026 jesus luque.
//
// The camera's exposure, applied once over the composed frame: everything the
// camera sees -- meshes, splats, points, domes -- scaled by the same 2^stops,
// and nothing that is not a colour (the AOVs) touched. Alpha stays.
module exposure;

struct ExposureParams {
    float scale;   // 2^stops
    uint  pixels;
    uint  pad0; uint pad1;
};

RWStructuredBuffer<float4>     colour;
ConstantBuffer<ExposureParams> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void applyExposure(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.pixels) {
        const float4 c = colour[tid.x];
        colour[tid.x] = float4(c.rgb * params.scale, c.a);
    }
}

/// Planes copied out of a larger buffer, scaled on the way: the path tracer's
/// light group means, which accumulate in its own buffer and must not be
/// scaled where they stand (a converged pixel is not written again, and would
/// take the exposure twice).
StructuredBuffer<float4> source;
uniform uint             sourceBase;   // entries
uniform uint             colourBase;   // entries: where in `colour` the copy starts

[shader("compute")]
[numthreads(256, 1, 1)]
void copyScaled(uint3 tid: SV_DispatchThreadID) {
    if (tid.x < params.pixels) {
        const float4 c = source[sourceBase + tid.x];
        colour[colourBase + tid.x] = float4(c.rgb * params.scale, c.a);
    }
}

/// WHAT NO LIGHT GROUP HOLDS OF A PATH TRACED FRAME, a plane beside the groups': the share of each pixel's
/// colour that the path tracer's own sums say came from no group -- its colour sum less its groups' sums,
/// over its colour sum, a channel -- times the frame's colour as it stands (not yet denoised). The same
/// paths gathered the colour and the groups, so the share is exact; taken before the denoiser, the plane is
/// denoised as the groups' are, and is not the difference of two denoised pictures.
StructuredBuffer<float4> whole;        // the frame's colour, a pixel
uniform uint             groups;       // how many group sums follow the colour's in `source`

[shader("compute")]
[numthreads(256, 1, 1)]
void restOfGroups(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.pixels) {
        return;
    }
    const float3 all = source[tid.x].rgb;
    float3 held = float3(0.0);
    for (uint g = 0; g < groups; ++g) {
        held += max(source[(1u + g) * params.pixels + tid.x].rgb, float3(0.0));
    }
    const float4 c = whole[tid.x];
    float3 share;
    for (int k = 0; k < 3; ++k) {
        share[k] = all[k] > 0.0 ? saturate((all[k] - held[k]) / all[k]) : 1.0;
    }
    colour[colourBase + tid.x] = float4(max(c.rgb, float3(0.0)) * share, c.a);
}

/// A DENOISED FRAME'S GROUPS, MADE TO ADD UP TO IT. The denoiser is run on the colour, on each group's plane
/// and on what no group holds, one at a time, and it is not linear: the denoised planes do not add up to the
/// denoised colour. Each group's plane is scaled, a channel, by what the colour has past the plane of no
/// group over what the groups' planes add up to -- so the frame is again what no group holds and its groups,
/// as it was before the denoiser. Held to four times: a pixel where the groups' planes say next to nothing
/// is not made to carry what the colour says.
[shader("compute")]
[numthreads(256, 1, 1)]
void groupsToWhole(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.pixels) {
        return;
    }
    float3 sum = float3(0.0);
    for (uint g = 0; g < groups; ++g) {
        sum += max(colour[g * params.pixels + tid.x].rgb, float3(0.0));
    }
    const float3 want = max(whole[tid.x].rgb - max(colour[groups * params.pixels + tid.x].rgb, float3(0.0)), float3(0.0));
    float3 by;
    for (int k = 0; k < 3; ++k) {
        by[k] = sum[k] > 1.0e-6 ? clamp(want[k] / sum[k], 0.0, 4.0) : 1.0;
    }
    for (uint g = 0; g < groups; ++g) {
        const float4 c = colour[g * params.pixels + tid.x];
        colour[g * params.pixels + tid.x] = float4(max(c.rgb, float3(0.0)) * by, c.a);
    }
}
