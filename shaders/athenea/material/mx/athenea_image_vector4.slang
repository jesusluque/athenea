// Copyright (c) 2026 jesus luque.
//
// <image> (vector4): the texture table, or the default where the file or its tile is missing.
#include "pbrlib/genglsl/lib/mx_closure_type.glsl"

void athenea_image_vector4(SamplerTexture2D tex_sampler, int layer, float4 defaultval, float2 texcoord, int uaddressmode,
                   int vaddressmode, int filtertype, int framerange, int frameoffset, int frameendaction,
                   float2 uv_scale, float2 uv_offset, out float4 result)
{
    const float2 uv = texcoord * uv_scale + uv_offset;
    bool found = false;
    const float4 value = sampleTexture(gTextures, tex_sampler.id, tex_sampler.sampler, uv,
                                       gAtheneaInputs.uvDx * uv_scale, gAtheneaInputs.uvDy * uv_scale, found);
    result = found ? value : defaultval;
}
