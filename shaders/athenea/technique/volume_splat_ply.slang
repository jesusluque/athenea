// Copyright (c) 2026 jesus luque.
//
// A MEDIUM WITH ITS LIGHT BAKED, AS A 3DGS PLY (ATHENEA_VOLUME_PLY): each
// gaussian of a medium the frame drew, with the radiance the frame's shading
// baked for it (volume_splat_shade.slang) as its base colour, for a player
// that draws ordinary splats. 62 floats a record, INRIA's PLY order, laid as
// `athenea vdb2splat --cloud` lays a medium's splats (vdb_splats.slang
// vsExportPly), so `video-encode` and `vdb2splat --from-athv` read it the same:
//
//   position, ln deviations and rotation in the cloud's own space (the prim's
//   transform is not applied);
//   f_dc     the radiance, display-referred: exposure x L through Reinhard
//            (x / (1 + x) a channel) and the sRGB curve, as (c - 0.5) / C0;
//   f_rest_0 ln sigma (the extinction's peak, density units), held at the
//            floor -10 or over; f_rest_1 the temperature;
//   opacity  vsExport's for a plain viewer: 1 - exp(-sigma s sqrt(2 pi) d),
//            d the deviations' geometric mean, s the medium's density scale.
//
// One record a record of the file (`declared`): a gaussian validation dropped,
// and a dead slot of a sequence (sigma 0), is a speck at the floor at the
// origin, so a sequence's slots keep their places frame to frame.
module volume_splat_ply;
import athenea.common.packing;

public struct VolumePlyParams {
    public uint  count;      ///< splats of the cloud (mode 1) or records of the file (mode 0)
    public uint  mode;       ///< 0: every record a speck; 1: each splat into its record
    public uint  hasOrigin;  ///< 1: `origin` says which record a splat came from
    public float densityScale;
    public float exposure;   ///< a factor on the radiance before the curve
};

StructuredBuffer<float4>          positions;
StructuredBuffer<uint>            shape;        // 4 words a gaussian, as GpuSplats packs them
StructuredBuffer<uint>            volume;       // two words a gaussian (scene/volume_pack.slang)
StructuredBuffer<uint>            radiance;     // rgb halves, two words a gaussian
StructuredBuffer<uint>            origin;       // the record each splat came from
RWStructuredBuffer<float>         plyOut;       // 62 floats a record
ConstantBuffer<VolumePlyParams>   plyParams;

static const float kC0 = 0.28209479177387814;
static const float kFloorLogSigma = -10.0;

float srgbOf(float x) {
    return x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

void speck(uint o) {
    for (uint k = 0; k < 62; ++k) plyOut[o + k] = 0.0;
    plyOut[o + 9] = kFloorLogSigma;
    plyOut[o + 54] = -20.0;
    for (uint a = 0; a < 3; ++a) plyOut[o + 55 + a] = log(1e-6);
    plyOut[o + 58] = 1.0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeSplatPly(uint3 tid: SV_DispatchThreadID) {
    const VolumePlyParams p = plyParams;
    const uint i = tid.x;
    if (i >= p.count) return;
    if (p.mode == 0) {
        speck(i * 62);
        return;
    }
    const uint record = p.hasOrigin != 0 ? origin[i] : i;
    const uint o = record * 62;
    const uint2 medium = uint2(volume[i * 2], volume[i * 2 + 1]);
    const float sigma = lowHalf(medium.x);
    if (!(sigma > 0.0) || !(log(sigma) > kFloorLogSigma + 0.25)) {
        speck(o);
        return;
    }
    const uint row = i * 4;
    const float3 lnScale = float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2]));
    const float4 q = decodeQuaternion(shape[row]);   // x y z w
    const float3 l = float3(lowHalf(radiance[i * 2]), highHalf(radiance[i * 2]), lowHalf(radiance[i * 2 + 1]));
    const float3 x = max(l * p.exposure, float3(0.0));
    const float3 mapped = x / (float3(1.0) + x);
    const float3 c = float3(srgbOf(mapped.x), srgbOf(mapped.y), srgbOf(mapped.z));
    const float meanStd = exp((lnScale.x + lnScale.y + lnScale.z) / 3.0);
    const float alpha = clamp(1.0 - exp(-sigma * p.densityScale * 2.50662827 * meanStd), 1.0 / 254.0, 0.999);
    const float3 at = positions[i].xyz;
    for (uint k = 0; k < 62; ++k) plyOut[o + k] = 0.0;
    plyOut[o + 0] = at.x; plyOut[o + 1] = at.y; plyOut[o + 2] = at.z;
    plyOut[o + 6] = (c.x - 0.5) / kC0;
    plyOut[o + 7] = (c.y - 0.5) / kC0;
    plyOut[o + 8] = (c.z - 0.5) / kC0;
    plyOut[o + 9] = max(log(sigma), kFloorLogSigma + 0.5);   // f_rest_0
    plyOut[o + 10] = highHalf(medium.x);                     // f_rest_1: the temperature
    plyOut[o + 54] = log(alpha / (1.0 - alpha));
    plyOut[o + 55] = lnScale.x; plyOut[o + 56] = lnScale.y; plyOut[o + 57] = lnScale.z;
    plyOut[o + 58] = q.w; plyOut[o + 59] = q.x; plyOut[o + 60] = q.y; plyOut[o + 61] = q.z;
}
