// Copyright (c) 2026 jesus luque.
//
// `athenea mesh2splat --validate`: a conversion measured material by material
// against the stage path traced, on the device (docs/decisions.md, task TX).
//
//   validateMask      which pixels are a material's, out of the mesh frame's
//                     Cryptomatte: the coverage of every rank whose id is one
//                     of the material's meshes', summed; and the mask's sum
//                     and its box, by atomics
//   validateApply     a frame times the mask, into a picture the Measure
//                     effect reads -- so outside the material both frames are
//                     nothing and agree
//   validateTriptych  GT, the mesh rasterised and the cloud, side by side and
//                     cut to the material's box: linear for an EXR, sRGB
//                     bytes for a PNG
//
// Frames are bottom row first, as the engine renders; the PNG's bytes top row
// first, as a PNG keeps them.
module m2s_validate;
import athenea.common.color;

struct ValidateParams {
    uint width;       ///< the frame
    uint height;
    uint idCount;     ///< the material's mesh ids, sorted
    uint layers;      ///< Cryptomatte layers, two ranks each

    uint srcStride;   ///< pixels a row of the frame's picture
    uint dstStride;   ///< pixels a row of the masked picture
    uint x0;          ///< the triptych's box, bottom row first
    uint y0;

    uint cutWidth;
    uint cutHeight;
    uint pad0;
    uint pad1;
};

StructuredBuffer<float4>     matte;     ///< `layers` planes of (id, coverage, id, coverage) a pixel
StructuredBuffer<uint>       ids;
RWStructuredBuffer<float>    mask;      ///< a pixel
RWStructuredBuffer<uint>     counters;  ///< [0] the mask's sum in 1/256, [1..4] its box: x0 y0 x1 y1
StructuredBuffer<float4>     frame;
RWStructuredBuffer<float4>   masked;
StructuredBuffer<float4>     gt;
StructuredBuffer<float4>     meshFrame;
StructuredBuffer<float4>     cloud;
RWStructuredBuffer<float4>   tripLinear;   ///< 3 * cutWidth by cutHeight, bottom row first
RWStructuredBuffer<uint>     tripBytes;    ///< the same, sRGB RGBA8, top row first
ConstantBuffer<ValidateParams> params;

bool isOurs(uint id) {
    uint lo = 0;
    uint hi = params.idCount;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        const uint at = ids[mid];
        if (at == id) {
            return true;
        }
        if (at < id) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void validateMask(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const uint at = tid.y * params.width + tid.x;
    const uint pixels = params.width * params.height;
    float coverage = 0.0;
    for (uint l = 0; l < params.layers; ++l) {
        const float4 ranks = matte[l * pixels + at];
        if (ranks.y > 0.0 && isOurs(asuint(ranks.x))) {
            coverage += ranks.y;
        }
        if (ranks.w > 0.0 && isOurs(asuint(ranks.z))) {
            coverage += ranks.w;
        }
    }
    coverage = saturate(coverage);
    mask[at] = coverage;
    if (coverage > 0.0) {
        InterlockedAdd(counters[0], uint(coverage * 256.0 + 0.5));
        InterlockedMin(counters[1], tid.x);
        InterlockedMin(counters[2], tid.y);
        InterlockedMax(counters[3], tid.x + 1);
        InterlockedMax(counters[4], tid.y + 1);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void validateApply(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.width || tid.y >= params.height) {
        return;
    }
    const float m = mask[tid.y * params.width + tid.x];
    masked[tid.y * params.dstStride + tid.x] = frame[tid.y * params.srcStride + tid.x] * m;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void validateTriptych(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.cutWidth * 3 || tid.y >= params.cutHeight) {
        return;
    }
    const uint panel = tid.x / params.cutWidth;
    const uint sx = min(params.x0 + tid.x % params.cutWidth, params.width - 1);
    const uint sy = min(params.y0 + tid.y, params.height - 1);
    const uint at = sy * params.srcStride + sx;
    const float4 v = panel == 0 ? gt[at] : (panel == 1 ? meshFrame[at] : cloud[at]);
    const uint width = params.cutWidth * 3;
    tripLinear[tid.y * width + tid.x] = v;
    const float3 s = linearToSrgb(saturate(v.rgb));
    const uint4 b = uint4(uint3(s * 255.0 + 0.5), 255u);
    tripBytes[(params.cutHeight - 1 - tid.y) * width + tid.x] = b.x | (b.y << 8) | (b.z << 16) | (b.w << 24);
}
