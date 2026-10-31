// Copyright (c) 2026 jesus luque.
//
// A MEDIUM'S GAUSSIANS INTO THE CLOUD SHADOW MAP (proposal 105 §4).
//
// The map holds, a texel, the optical depth along the light's direction
// through it, summed over every caster without an order (splat_shadow_map).
// A surface's gaussian puts in -ln(1 - alpha) of its footprint; a medium's
// gaussian has an optical depth of its own and puts in exactly that, into a
// total and Fourier terms of the medium's own (splat_shadow_read's
// shadowStrideOf says why apart from the surfaces'): along a
// line of direction w through a point p,
//
//   integral sigma G(p + s w) ds = sigma sqrt(2 pi) sqrt(det Sigma) / sqrt(det Sigma_2) exp(-q(p_perp) / 2)
//
// with Sigma_2 the covariance squashed onto the map's plane -- the same 2D
// covariance a surface's gaussian has there -- and q its Mahalanobis form.
// So the footprint is the one every caster has, and only its peak and its
// cut differ. Widened by the third of a texel every caster is widened by,
// its peak is lowered so that what it adds over the plane stays its mass.
// The Fourier terms are added with the same weights, and so a medium is
// shadowed by itself, by the clouds and by the meshes' casters through one
// map, and it shadows them back (volume_splat_shade.slang reads it).
//
// What is left out: the fit's cut at 3.5 deviations, 0.7% of a gaussian's
// mass, which the map keeps and the raster's chord does not.
module splat_shadow_volume;

import athenea.common.packing;
import athenea.light.lights;
import athenea.splat.frame;
import athenea.technique.splat_shadow_read;
import athenea.technique.splat_shadow_map;

/// Two words a gaussian (scene/volume_pack.slang): sigma and the temperature
/// as halves, the albedo as rgba8.
StructuredBuffer<uint> shadowVolume;

[shader("compute")]
[numthreads(256, 1, 1)]
void shadowMapSplatVolume(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= shadowParams.count) {
        return;
    }
    const ShadowMapFrame f = shadowFrameOf<ShadowBufferSource>(shadowParams.light);
    if (f.valid == 0) {
        return;
    }
    const float peak = shadowParams.volumeDensity * shadowParams.density * lowHalf(shadowVolume[i * 2]);
    if (!(peak > 0.0)) {
        return;
    }
    const float3 local = shadowPositions[i].xyz;
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shadowShape[row + 1]), highHalf(shadowShape[row + 1]),
                                    lowHalf(shadowShape[row + 2])));
    const float3 world = toWorld(local);
    const float u = dot(f.rowU.xyz, world) + f.rowU.w;
    const float v = dot(f.rowV.xyz, world) + f.rowV.w;
    const float z = clamp(dot(f.rowZ.xyz, world) + f.rowZ.w, 0.0, 1.0);
    const float4 q = decodeQuaternion(shadowShape[row]);
    const float x = q.x, y = q.y, zq = q.z, w = q.w;
    const float3x3 r = float3x3(
        1.0 - 2.0 * (y * y + zq * zq), 2.0 * (x * y - w * zq), 2.0 * (x * zq + w * y),
        2.0 * (x * y + w * zq), 1.0 - 2.0 * (x * x + zq * zq), 2.0 * (y * zq - w * x),
        2.0 * (x * zq - w * y), 2.0 * (y * zq + w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);
    const ShadowMapParams s = shadowParams;
    const float3x3 objectLinear = float3x3(s.w00, s.w01, s.w02, s.w10, s.w11, s.w12, s.w20, s.w21, s.w22);
    const float3x3 mw = mul(objectLinear, m);
    // The 3D covariance's square root determinant, in the world.
    const float volume3 = abs(determinant(mw));
    const float3 ax = mul(f.rowU.xyz, mw);
    const float3 ay = mul(f.rowV.xyz, mw);
    float a = dot(ax, ax);
    const float b = dot(ax, ay);
    float c = dot(ay, ay);
    const float det0 = a * c - b * b;
    if (!(det0 > 1.0e-20)) {
        return;
    }
    // The line integral's peak, in the map's texels: sqrt(det Sigma_2) there
    // is texels^2 times the world's.
    const float texels2 = f.texelsPerUnit * f.texelsPerUnit;
    float tau = peak * sqrt(2.0 * kShadowPi) * volume3 * texels2 / sqrt(det0);
    a += 0.3;
    c += 0.3;
    const float det = a * c - b * b;
    tau *= sqrt(det0 / det);
    const float3 conic = float3(c / det, -b / det, a / det);
    // Out to where a texel would get under half a unit of the fixed point, or
    // the fit's 3.5 deviations.
    const float cut = min(log(max(tau * 2.0 * kShadowFixedScale, 1.0001)), 6.125);
    const float2 extent = footprintExtent(conic, 2.0 * cut);
    const int loX = max(int(floor(u - extent.x)), 0);
    const int hiX = min(int(ceil(u + extent.x)), int(f.resolution) - 1);
    const int loY = max(int(floor(v - extent.y)), 0);
    const int hiY = min(int(ceil(v + extent.y)), int(f.resolution) - 1);
    if (loX > hiX || loY > hiY) {
        return;
    }
    if (f.medium == 0) {
        return;   // the map was made with no room for a medium
    }
    const uint pairs = (f.medium - 1) / 2;
    const uint stride = shadowStrideOf(f.coefficients, f.medium);
    for (int ty = loY; ty <= hiY; ++ty) {
        for (int tx = loX; tx <= hiX; ++tx) {
            const float dx = (float(tx) + 0.5) - u;
            const float dy = (float(ty) + 0.5) - v;
            const float power = 0.5 * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
            if (power < 0.0 || power > cut) {
                continue;
            }
            const float sigma = tau * exp(-power);
            // The medium's own words, after the surfaces' terms and depths
            // (splat_shadow_read's shadowStrideOf): a medium is no surface,
            // and moves neither the nearest caster nor the farthest.
            const uint at = f.offset + (uint(ty) * f.resolution + uint(tx)) * stride + f.coefficients + 2;
            // Rounded, not truncated: a smoke is thousands of small sums.
            InterlockedAdd(shadowMap[at], uint(sigma * kShadowFixedScale + 0.5));
            for (uint k = 1; k <= pairs; ++k) {
                const float angle = 2.0 * kShadowPi * float(k) * z;
                InterlockedAdd(shadowMap[at + k * 2 - 1], uint(int(round(sigma * cos(angle) * kShadowFixedScale))));
                InterlockedAdd(shadowMap[at + k * 2], uint(int(round(sigma * sin(angle) * kShadowFixedScale))));
            }
        }
    }
}
