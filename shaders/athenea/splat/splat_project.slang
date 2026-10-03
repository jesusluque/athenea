// Copyright (c) 2026 jesus luque.
//
// Per splat: where it lands, its 2D shape, its colour from this eye, its depth
// key, and exactly how many tiles its footprint touches. See frame.slang.
module splat_project;
import athenea.common.packing;
import athenea.common.sh;
import frame;
import athenea.common.edit;
// The core of the light module, which reads no texture: a dome is its colour
// here, since this kernel lives below material and has no table to sample.
// What a relit splat does with those lights is splat_relight's, shared with
// the ray traced route so that the two cannot drift.
import splat_relight;
import athenea.common.color;
import athenea.scene.splat_encoding;
import athenea.common.splat_override;
// A shadow ray against the cloud's own proxies, where the caller built them.

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      sh;
StructuredBuffer<uint>      pbr;      // metallic and roughness a splat, where the cloud carries them
// The shading normal a splat, where the cloud carries one (`params.hasNormals`).
StructuredBuffer<uint>      normals;
// Per-prim material overrides, keyed on the id each gaussian carries:
// `params.overrideCount` of 0 is what says there are none.
StructuredBuffer<float4>    splatOverrides;
StructuredBuffer<LightRecord> lights;
// The Cryptomatte id a splat inherited from the prim it was made from (a
// cloud athenea mesh2splat wrote). Read only where params.cryptoMode == 1.
StructuredBuffer<uint>      cloudCrypto;
RWStructuredBuffer<uint>    cryptoIds;    // one a slot, beside `proj`; written where cryptoMode != 0
RWStructuredBuffer<ProjRecord> proj;
RWStructuredBuffer<uint>    visible;      // 1 when the splat touches a tile
RWStructuredBuffer<uint>    tileRects;    // 4 per splat
RWStructuredBuffer<uint>    tilesTouched;
RWStructuredBuffer<uint>    depthKeys;
ConstantBuffer<FrameParams> params;

// Relighting's shadow, as splat_shadow.slang measured it: one float per
// (splat, light), the transmittance between that splat and that light's
// centre. It is a separate kernel because an inline ray is a feature a device
// may not have -- putting the RayQuery here made this kernel, and with it the
// whole splat rasteriser, unavailable on CUDA (measured: "entrypoint
// splatProject uses features that are not available").
StructuredBuffer<float>     shadowFactors;
// THE SKY, PREPARED FOR A KERNEL THAT CANNOT SAMPLE IT (technique::Environment,
// athenea/light/environment.slang). Bound whether or not the frame has a dome:
// `params.envLights` of 0 is what says there is nothing to read.
StructuredBuffer<uint>      envTexels;
StructuredBuffer<float4>    envSh;
// The sun `env_sun` took out of that sky: two float4 a dome.
StructuredBuffer<float4>     envSun;
StructuredBuffer<uint>      envOfLight;
// How much of an environment reaches each gaussian, where its cloud carries
// it; `params.transferCount` of 0 says there is nothing to read.
StructuredBuffer<uint>      transfer;
// Which ways out of each gaussian are open, two words a gaussian; bound
// whether or not `params.shadowBits` says to read them.
StructuredBuffer<uint>      shadowBits;

// What a skeleton moved this splat over the shutter, in the cloud's own space:
// two words of halves a splat, written by splat_skin.slang. Bound whether or
// not `params.motion` says to read it, as every name a shader declares must
// be.
StructuredBuffer<uint>      motion;

void cull(uint slot) {
    // Every early return zeroes the record first: a slot keeps last frame's
    // splat otherwise (openFXplayer learned this one the hard way).
    tilesTouched[slot] = 0;
    visible[slot] = 0;
    depthKeys[slot] = 0x00FFFFFFu;
    tileRects[slot * 4] = 1;
    tileRects[slot * 4 + 1] = 1;
    tileRects[slot * 4 + 2] = 0;
    tileRects[slot * 4 + 3] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void splatProject(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint slot = params.base + i;

    float4 posOpacity = positions[i];
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]),
                                    lowHalf(shape[row + 2])));
    // The SplatEdit, in the cloud's own space, before anything is projected.
    const EditFate fate = editFate(params.edit, posOpacity.xyz, posOpacity.w, scale);
    if (fate.removed) {
        cull(slot);
        return;
    }
    posOpacity.w = fate.opacity;

    const float3 t = toView(params, posOpacity.xyz);
    if (!(t.z > params.nearZ) || t.z > params.farZ) {
        cull(slot);
        return;
    }

    // 3D covariance in the cloud's space: R S S^T R^T.
    const float4 q = decodeQuaternion(shape[row]);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float3x3 r = float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3x3 m = float3x3(r[0] * scale, r[1] * scale, r[2] * scale);

    // T = J W, then the 2D covariance is (T M)(T M)^T.
    const bool ortho = params.orthographic != 0;
    float3 jx, jy;
    float2 centre;
    if (ortho) {
        jx = float3(params.focalX, 0.0, 0.0);
        jy = float3(0.0, params.focalY, 0.0);
        centre = float2(params.focalX * t.x + params.centreX, params.focalY * t.y + params.centreY);
    } else {
        const float invZ = 1.0 / t.z;
        centre = float2(params.focalX * t.x * invZ + params.centreX,
                        params.focalY * t.y * invZ + params.centreY);
        // The Jacobian is linearised at a point clamped to 1.3x the frustum:
        // outside it the affine approximation explodes and a splat far off
        // screen smears across the whole picture (3DGS does the same).
        const float limLoX = (-0.15 * float(params.width) - params.centreX) / params.focalX;
        const float limHiX = (1.15 * float(params.width) - params.centreX) / params.focalX;
        const float limLoY = (-0.15 * float(params.height) - params.centreY) / params.focalY;
        const float limHiY = (1.15 * float(params.height) - params.centreY) / params.focalY;
        const float tx = clamp(t.x * invZ, limLoX, limHiX) * t.z;
        const float ty = clamp(t.y * invZ, limLoY, limHiY) * t.z;
        jx = float3(params.focalX * invZ, 0.0, -params.focalX * tx * invZ * invZ);
        jy = float3(0.0, params.focalY * invZ, -params.focalY * ty * invZ * invZ);
    }
    const float3x3 wv = viewLinear(params);
    const float3 tx3 = mul(jx, wv);   // row vectors: J rows times W
    const float3 ty3 = mul(jy, wv);
    const float3 ax = mul(tx3, m);
    const float3 ay = mul(ty3, m);
    float a = dot(ax, ax);
    float b = dot(ax, ay);
    float c = dot(ay, ay);

    // THE SHUTTER, AS ONE RANK-ONE TERM.
    //
    // A gaussian smeared along a straight path is that gaussian convolved with
    // the path. For a uniform shutter the path is a segment, whose variance
    // along itself is |d|^2/12, so matching second moments is one addition of
    // (d d^T)/12 to the 2D covariance -- and convolving by a normalised
    // measure keeps the integral, which is what the square root below pays
    // back. One pass, where sub-sampling the mean costs one pass a sample.
    //
    // The displacement is turned by the same `tx3`/`ty3` the covariance was,
    // so the blur and the shape live in one linearisation of the projection
    // and not two; the frustum clamp above comes with them, which is what
    // stops a splat behind the camera from smearing across the frame.
    //
    // What this does not do: a gaussian convolved with a segment is a plateau
    // with soft ends, and the moment-matched gaussian is a peak -- exact in
    // energy, centroid and second moment, up to 38% too peaked once the
    // displacement is far wider than the splat. That is the price of the one
    // pass, and it is why the test measures energy there rather than pixels.
    const float detBefore = a * c - b * b;   // the sharp, still covariance's
    float spreadPaid = 1.0;

    // THE LENS, AS ANOTHER CONVOLUTION.
    //
    // A pinhole draws the gaussian; a lens of radius A draws it convolved with
    // the disk its out-of-focus point spreads into. The ray tracer samples
    // that disk once a path (`lensRadius` in the path tracer's camera); here
    // the disk is a measure to convolve by, and a uniform disk of radius R has
    // variance R^2/4 along either axis. So the same one addition the shutter
    // makes, with the same square root paying its energy back -- one pass, no
    // samples, and a splat in focus is untouched.
    //
    // R comes out of the thin lens: a ray through lens point o meets depth z
    // at x = z*dir + o(1 - z/focus), so the image moves by
    // -focal * o * (1/z - 1/focus). Over the disk of radius A that is a disk
    // of radius focal * A * |1/z - 1/focus| pixels, which is the circle of
    // confusion and nothing else.
    if (params.lensRadius > 0.0 && !ortho) {
        const float focus = max(params.focusDistance, params.nearZ);
        const float spread = params.lensRadius * abs(1.0 / t.z - 1.0 / focus);
        float rx = params.focalX * spread;
        float ry = params.focalY * spread;
        // The same cap the shutter's displacement takes: a splat just off the
        // lens would otherwise cover the frame and cost what that costs.
        const float longest = max(rx, ry);
        if (longest > params.motionMax) {
            rx *= params.motionMax / longest;
            ry *= params.motionMax / longest;
        }
        a += rx * rx * 0.25;
        c += ry * ry * 0.25;
        spreadPaid = sqrt(max(detBefore, 0.0) / max(a * c - b * b, 1e-12));
    }
    if (params.motion != 0) {
        float2 d = float2(0.0, 0.0);
        if ((params.motion & 1u) != 0u) {
            const uint word = i * 2;
            const float3 step = float3(lowHalf(motion[word]), highHalf(motion[word]),
                                       lowHalf(motion[word + 1])) * params.motionScale;
            d += float2(dot(tx3, step), dot(ty3, step));
        }
        if ((params.motion & 2u) != 0u) {
            const float4 p = float4(posOpacity.xyz, 1.0);
            const float3 step = float3(dot(float4(params.s00, params.s01, params.s02, params.s03), p),
                                       dot(float4(params.s10, params.s11, params.s12, params.s13), p),
                                       dot(float4(params.s20, params.s21, params.s22, params.s23), p));
            d += float2(dot(jx, step), dot(jy, step));
        }
        // A splat on a fast joint may not own the screen: the footprint of a
        // blurred gaussian grows nearly with the displacement, and one of them
        // could take every tile there is. Clamping shortens the streak and
        // leaves it energy-correct, since what is paid back is measured after.
        const float longest = max(abs(d.x), abs(d.y));
        if (longest > params.motionMax) {
            d *= params.motionMax / longest;
        }
        a += d.x * d.x * (1.0 / 12.0);
        b += d.x * d.y * (1.0 / 12.0);
        c += d.y * d.y * (1.0 / 12.0);
        spreadPaid = sqrt(max(detBefore, 0.0) / max(a * c - b * b, 1e-12));
    }

    // A third of a pixel of dilation, and the Mip-Splatting 2D filter paying
    // its energy back as opacity (openFXplayer splat.slang; gsplat 'antialiased').
    a += 0.3;
    c += 0.3;
    const float det = a * c - b * b;
    if (!(det > 1e-12)) {
        cull(slot);
        return;
    }
    float alpha = posOpacity.w;
    if (params.antialias != 0) {
        // `detBefore` is the still covariance's, so this one square root pays
        // for both convolutions: the shutter's and the mip filter's.
        alpha *= sqrt(max(detBefore, 0.0) / det);
    } else {
        // Without the filter there is still the shutter and the lens to pay
        // for; a spread splat that kept its peak would gain energy and the
        // wings, or the bokeh, would burn.
        alpha *= spreadPaid;
    }
    if (!(alpha >= 1.0 / 255.0)) {
        cull(slot);
        return;
    }
    const float3 conic = float3(c / det, -b / det, a / det);
    const float cutoff = cutoffPower(alpha);
    const float2 extent = footprintExtent(conic, cutoff);
    uint4 rect;
    tileRect(params, centre, extent, rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        cull(slot);
        return;
    }

    // The exact count, with the same test the emit repeats.
    uint touched = 0;
    for (uint ty = rect.y; ty <= rect.w; ++ty) {
        for (uint tx = rect.x; tx <= rect.z; ++tx) {
            if (tileTouched(params, centre, conic, cutoff, tx, ty)) {
                touched += 1;
            }
        }
    }
    if (touched == 0) {
        cull(slot);
        return;
    }

    // Colour from this eye, in the cloud's space: the harmonics were trained
    // there, and a camera left in world space makes colours swim when the
    // cloud is turned.
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]),
                               highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 dir = normalize(posOpacity.xyz - float3(params.eyeX, params.eyeY, params.eyeZ));
    float3 colour = cloudLight(base + evaluateRest(sh, i * params.shWords, degree, dir), params.linearCloud != 0);
    if (params.relight != 0) {
        // AtheneaSplatLightingAPI: the radiance a capture was baked with replaced
        // by what this scene's lights put on it. What that means is in
        // splat_relight.slang, which the ray traced route reads too.
        // In the world, where the lights are: the cloud's own space is where
        // its harmonics were trained, and a light stands somewhere else.
        const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
        const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
        const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
        const float3 eye = float3(params.eyeWorldX, params.eyeWorldY, params.eyeWorldZ);
        const float3 cloudNormal =
            splatShadingNormal(r, scale, posOpacity.xyz, float3(params.eyeX, params.eyeY, params.eyeZ),
                               params.hasNormals != 0, params.hasNormals != 0 ? normals[i] : 0u);
        SplatSurface surface;
        surface.position = relightToWorld(w0, w1, w2, posOpacity.xyz);
        surface.normal = relightDirectionToWorld(w0, w1, w2, cloudNormal);
        // What the cloud shows in this direction, in linear light already.
        // The harmonics are part of it -- a baked cloud keeps the body's own
        // change with direction in them -- so it is the evaluated colour and
        // not the constant term alone.
        surface.albedo = colour;
        surface.metallic = params.hasPbr != 0 ? pbrMetallic(pbr[i]) : 0.0;
        surface.roughness = params.hasPbr != 0 ? pbrRoughness(pbr[i]) : 1.0;
        surface.transmission = params.hasPbr != 0 ? pbrTransmission(pbr[i]) : 0.0;
        surface.thinWalled = params.hasPbr != 0 && pbrThinWalled(pbr[i]);
        // WHAT A PICK MAY HAVE SAID ABOUT THIS PRIM. The id a gaussian carries
        // is the one the matte writes, so a table keyed on it is a material
        // override addressed by clicking (athenea/common/splat_override.slang).
        if (params.overrideCount != 0) {
            const SplatOverride said =
                splatOverrideFor(splatOverrides, params.overrideCount, cloudCrypto[i]);
            if (said.found) {
                surface.metallic = splatOverrideOf(surface.metallic, said.metallic);
                surface.roughness = splatOverrideOf(surface.roughness, said.roughness);
                surface.transmission = splatOverrideOf(surface.transmission, said.transmission);
                surface.albedo = splatOverrideColour(surface.albedo, said);
            }
        }
        surface.ior = params.ior;
        surface.backFacing =
            splatFacing(r, scale, posOpacity.xyz, float3(params.eyeX, params.eyeY, params.eyeZ)) < 0.0;
        // The rasteriser traces nothing, so its glass bends once and shows
        // the sky alone.
        surface.hasBehind = false;
        surface.behindColour = float3(0.0);
        surface.hasReflection = false;
        surface.reflectionColour = float3(0.0);
        surface.reflectionTraced = false;
        surface.hasExit = false;
        surface.exitDirection = float3(0.0, 0.0, 1.0);
        surface.lit = params.litBody != 0;
        surface.transferCount = params.transferCount;
        surface.transferBase = i * params.transferWords * 2;
        surface.hasShadowBits = params.shadowBits != 0 && params.transferCount >= 9;
        surface.shadowBits = surface.hasShadowBits ? uint2(shadowBits[i * 2], shadowBits[i * 2 + 1]) : uint2(0);
        const uint shadowLights = params.shadowRays != 0 ? params.shadowLights : 0;
        colour = relitSplat(surface, normalize(eye - surface.position), lights,
                            params.lightCount, params.categoriesLo, params.categoriesHi,
                            shadowFactors, slot * params.shadowLights, shadowLights,
                            envTexels, envSh, envOfLight, params.envLights, params.envBaseSide,
                            envSun, transfer,
                            params.transferIndirect != 0);
    }
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }

    ProjRecord record;
    record.screen = float4(centre, t.z, cutoff);
    record.conic = float4(conic, alpha);
    record.colour = float4(colour, 0.0);
    proj[slot] = record;
    if (params.cryptoMode != 0) {
        cryptoIds[slot] = params.cryptoMode == 1 ? cloudCrypto[i] : 0u;
    }
    tileRects[slot * 4] = rect.x;
    tileRects[slot * 4 + 1] = rect.y;
    tileRects[slot * 4 + 2] = rect.z;
    tileRects[slot * 4 + 3] = rect.w;
    tilesTouched[slot] = touched;
    visible[slot] = 1;
    // Positive floats order as their bits do. The top 24 bits -- exponent and
    // 15 bits of mantissa, a relative depth step of 3e-5 -- sort in three
    // radix passes instead of four; splats nearer each other than that keep
    // index order, as exact ties always did.
    depthKeys[slot] = bit_cast<uint>(t.z) >> 7;
}
