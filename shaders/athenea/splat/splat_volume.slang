// Copyright (c) 2026 jesus luque.
//
// A MEDIUM'S GAUSSIANS, PROJECTED (proposal 105 §3).
//
// A cloud whose gaussians carry an extinction (scene::GpuSplats::volume) is a
// participating medium, mu_t(x) = densityScale * sum_i sigma_i G_i(x) with G_i
// of peak one, cut at 3.5 deviations as `athenea vdb2splat` fitted it. Its
// gaussians go into the same depth sort and the same tiles as every other
// splat; what differs is the alpha the blend gives one at a pixel, which is
// not an opacity times a 2D gaussian but the exact optical depth of the
// pixel's ray through it:
//
//   tau_i = densityScale sigma_i |r| integral_{t0}^{t1} G_i(t r) dt
//         = densityScale sigma_i |r| exp(-d^2/2) sqrt(pi / 2a) [erf(k (t1 - t*)) - erf(k (t0 - t*))]
//
// for the ray t r, r = ((x - cx)/fx, (y - cy)/fy, 1) in the view (so t is the
// view depth), with u = U r, m = U c (U the upper Cholesky factor of the
// precision in the view, c the centre), a = |u|^2, t* = u.m / a,
// d^2 = |t* u - m|^2 and k = sqrt(a / 2). The segment is the chord the
// 3.5-deviation ellipsoid cuts from the ray, clipped at the camera and at the
// opaque layer under the splats: smoke standing on the ground is integrated
// up to the ground and no further. alpha_i = 1 - exp(-tau_i), and since the
// transmittance is a product the order of the gaussians changes only how
// their radiances are weighed, never how much light gets through.
//
// What the record carries for that (frame.slang's ProjRecord, colour.w =
// kVolumeMark):
//
//   screen   the image ellipse's centre, the centre's view depth, the cut k
//   conic    the image ellipse of the ellipsoid at level k, scaled to power k
//            (so the tile tests are the ones every splat has), and in w the
//            peak extinction densityScale * sigma
//   colour   m, the centre in the factor's frame
//   surfel   U00 U01 U02 U11 as floats
//   slopes   U12 U22 as floats, then the radiance as rgb halves (the blend
//            reads a marked record's `slopes` beside it)
//
// The cut k is where the optical depth falls under kVolumeLeast of the peak
// or the fit's own 3.5 deviations, whichever is first.
module splat_volume;
import athenea.common.packing;
import frame;

StructuredBuffer<float4>      positions;
StructuredBuffer<uint>        shape;
/// Two words a gaussian (scene/volume_pack.slang): sigma and the temperature
/// as halves, the albedo as rgba8.
StructuredBuffer<uint>        volume;
/// What a frame baked for each gaussian, rgb halves in two words
/// (technique::VolumeSplatShading); read where `volumeParams.hasRadiance`.
StructuredBuffer<uint>        volumeRadiance;
RWStructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint4>     slopes;
RWStructuredBuffer<uint>      cryptoIds;
RWStructuredBuffer<uint>      visible;
RWStructuredBuffer<uint>      tileRects;
RWStructuredBuffer<uint>      tilesTouched;
RWStructuredBuffer<uint>      depthKeys;
ConstantBuffer<FrameParams>   params;

public struct VolumeProjectParams {
    public float  densityScale;
    public uint   hasRadiance;
    public float  pad0;
    public float  pad1;
    public float3 albedo;     ///< the medium's, times each gaussian's
    public float  pad2;
    public float3 emission;   ///< what a gaussian with no baked radiance gives off, times 1 - albedo
    public float  pad3;
};
ConstantBuffer<VolumeProjectParams> volumeParams;

/// The optical depth, over the peak's, a gaussian is drawn out to: a few
/// thousand of them overlap in a smoke, so the cut is well under the 1/255 a
/// surface's splat stops at.
public static const float kVolumeLeast = 1.0 / 1024.0;

void cull(uint slot, uint reason) {
    tilesTouched[slot] = 0;
    visible[slot] = 0;
    depthKeys[slot] = kCulledKey | reason;
    tileRects[slot * 4] = 1;
    tileRects[slot * 4 + 1] = 1;
    tileRects[slot * 4 + 2] = 0;
    tileRects[slot * 4 + 3] = 0;
}

float3x3 inverse3(float3x3 n, out float det) {
    const float3 c0 = cross(n[1], n[2]);
    const float3 c1 = cross(n[2], n[0]);
    const float3 c2 = cross(n[0], n[1]);
    det = dot(n[0], c0);
    // The inverse's columns are the cofactor rows over the determinant.
    return transpose(float3x3(c0, c1, c2)) / det;
}

/// THE IMAGE OF THE ELLIPSOID (x - c)^T P (x - c) <= level through a pinhole
/// at the origin: the rays r = (x, y, 1) whose least squared distance is at
/// most `level` are r^T C r <= 0 with C = (c^T P c - level) P - (P c)(P c)^T.
/// An ellipse where the ellipsoid lies wholly in front of the camera; false
/// otherwise (the camera inside it, or the ellipsoid across the plane z = 0),
/// and the caller draws it over the whole frame.
bool imageEllipse(float3x3 precision, float3 c, float level, out float2 centre, out float3 shape2) {
    const float3 pc = mul(precision, c);
    const float3x3 conicMatrix = (dot(c, pc) - level) * precision -
                                 float3x3(pc * pc.x, pc * pc.y, pc * pc.z);
    const float a = conicMatrix[0][0];
    const float b = conicMatrix[0][1];
    const float e = conicMatrix[1][1];
    const float d = conicMatrix[0][2];
    const float f = conicMatrix[1][2];
    const float g = conicMatrix[2][2];
    const float det = a * e - b * b;
    centre = float2(0.0);
    shape2 = float3(0.0);
    if (!(det > 0.0) || !(a > 0.0)) {
        return false;
    }
    const float x0 = (-d * e + f * b) / det;
    const float y0 = (-f * a + d * b) / det;
    const float q0 = g + d * x0 + f * y0;
    if (!(q0 < 0.0)) {
        return false;
    }
    centre = float2(x0, y0);
    shape2 = float3(a, b, e) / -q0;   // q2(dx, dy) <= 1 inside
    return true;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatProjectVolume(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;
    if (params.orthographic != 0) {
        // A medium is drawn through a pinhole only (TileRasterizer says so once).
        cull(slot, kCullDepth);
        return;
    }
    const VolumeProjectParams vp = volumeParams;
    const uint2 medium = uint2(volume[i * 2], volume[i * 2 + 1]);
    const float peak = vp.densityScale * lowHalf(medium.x);
    if (!(peak > 0.0)) {
        cull(slot, kCullFaint);
        return;
    }
    const float3 c = toView(params, positions[i].xyz);
    const uint row = i * 4;
    const float3 scale = max(exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2]))),
                             float3(1.0e-12));
    const float4 q = decodeQuaternion(shape[row]);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float3x3 rot = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    // N maps the unit ball onto the gaussian's deviations in the view; the
    // precision there is N^-T N^-1.
    const float3x3 n = mul(viewLinear(params), float3x3(rot[0] * scale, rot[1] * scale, rot[2] * scale));
    float det;
    const float3x3 ninv = inverse3(n, det);
    if (!(abs(det) > 1.0e-30)) {
        cull(slot, kCullDegenerate);
        return;
    }
    const float3x3 precision = mul(transpose(ninv), ninv);
    // Cholesky, P = L L^T; U = L^T.
    const float l00 = sqrt(max(precision[0][0], 1.0e-30));
    const float l10 = precision[1][0] / l00;
    const float l20 = precision[2][0] / l00;
    const float l11 = sqrt(max(precision[1][1] - l10 * l10, 1.0e-30));
    const float l21 = (precision[2][1] - l20 * l10) / l11;
    const float l22 = sqrt(max(precision[2][2] - l20 * l20 - l21 * l21, 1.0e-30));
    const float3 m = float3(l00 * c.x + l10 * c.y + l20 * c.z, l11 * c.y + l21 * c.z, l22 * c.z);

    // How deep the ellipsoid reaches toward the camera: the view z's
    // deviation is |N's third row|.
    const float zReach = sqrt(kVolumeTruncation) * length(n[2]);
    if (c.z + zReach <= params.nearZ || c.z - zReach > params.farZ) {
        cull(slot, kCullDepth);
        return;
    }

    // The peak optical depth, along the ray through the centre: what says
    // how far out the gaussian is worth drawing.
    const float3 rc = c / max(c.z, 1.0e-6);
    const float3 uc = float3(l00 * rc.x + l10 * rc.y + l20 * rc.z, l11 * rc.y + l21 * rc.z, l22 * rc.z);
    const float tauPeak = peak * length(rc) * sqrt(2.0 * 3.14159265358979 / max(dot(uc, uc), 1.0e-30));
    const float level = min(2.0 * log(max(tauPeak / kVolumeLeast, 1.0001)), kVolumeTruncation);
    if (tauPeak < kVolumeLeast) {
        cull(slot, kCullFaint);
        return;
    }
    // The tiles: the image ellipse a little wider than the cut, so what the
    // float arithmetic of a far gaussian loses is never a pixel the blend
    // would have drawn.
    float2 ellipseCentre;
    float3 ellipse;
    float2 centre;
    float3 conic;
    if (c.z > 0.0 && imageEllipse(precision, c, min(level * 1.1, kVolumeTruncation), ellipseCentre, ellipse)) {
        centre = float2(params.focalX * ellipseCentre.x + params.centreX, params.focalY * ellipseCentre.y + params.centreY);
        conic = level * float3(ellipse.x / (params.focalX * params.focalX), ellipse.y / (params.focalX * params.focalY),
                               ellipse.z / (params.focalY * params.focalY));
    } else {
        // The camera in it, or it across the camera's plane: every tile.
        centre = float2(params.centreX, params.centreY);
        conic = float3(1.0e-9, 0.0, 1.0e-9);
    }
    const float2 extent = footprintExtent(conic, level);
    uint4 rect;
    tileRect(params, centre, extent, rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        cull(slot, kCullOffscreen);
        return;
    }
    uint touched = 0;
    for (uint ty = rect.y; ty <= rect.w; ++ty) {
        for (uint tx = rect.x; tx <= rect.z; ++tx) {
            if (tileTouched(params, centre, conic, level, tx, ty)) {
                touched += 1;
            }
        }
    }
    if (touched == 0) {
        cull(slot, kCullNoTile);
        return;
    }

    float3 radiance;
    if (vp.hasRadiance != 0) {
        const uint2 baked = uint2(volumeRadiance[i * 2], volumeRadiance[i * 2 + 1]);
        radiance = float3(lowHalf(baked.x), highHalf(baked.x), lowHalf(baked.y));
    } else {
        const float3 own = float3(float(medium.y & 255u), float((medium.y >> 8) & 255u),
                                  float((medium.y >> 16) & 255u)) / 255.0;
        radiance = (float3(1.0) - saturate(own * vp.albedo)) * vp.emission;
    }
    radiance = max(radiance, float3(0.0));

    ProjRecord record;
    const float depth = max(c.z, params.nearZ);
    record.screen = float4(centre, depth, level);
    record.conic = float4(conic, peak);
    record.colour = float4(m, kVolumeMark);
    record.surfel = uint4(asuint(l00), asuint(l10), asuint(l20), asuint(l11));
    slopes[slot] = uint4(asuint(l21), asuint(l22), packHalves(radiance.x, radiance.y), packHalves(radiance.z, 0.0));
    proj[slot] = record;
    if (params.cryptoMode != 0) {
        cryptoIds[slot] = 0u;
    }
    tileRects[slot * 4] = rect.x;
    tileRects[slot * 4 + 1] = rect.y;
    tileRects[slot * 4 + 2] = rect.z;
    tileRects[slot * 4 + 3] = rect.w;
    tilesTouched[slot] = touched;
    visible[slot] = 1;
    depthKeys[slot] = bit_cast<uint>(depth) >> 7;
}
