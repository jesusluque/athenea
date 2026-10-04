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

/// The opacity a gaussian of a thin sheet of glass is drawn at, its colour
/// scaled to keep alpha x colour (the faint cull, task TX, s62b).
static const float kSheetAlpha = 0.1;
// A shadow ray against the cloud's own proxies, where the caller built them.

StructuredBuffer<float4>    positions;
StructuredBuffer<uint>      shape;
StructuredBuffer<uint>      sh;
StructuredBuffer<uint>      pbr;      // metallic and roughness a splat, where the cloud carries them
// The shading normal a splat, where the cloud carries one (`params.hasNormals`).
StructuredBuffer<uint>      normals;
// Three floats a splat, the shape operator in its first two axes, where
// `params.hasCurvature`; and what the reflection's slope across the footprint
// came to, a slot each, where the record's colour.w says so (kSlopeMark).
StructuredBuffer<float>     curvature;
RWStructuredBuffer<uint4>   slopes;
// A splat each: a TX transfer's view-independent terms -- the body's light
// (rgb), the field's coupling to the sky (rgb), the sun's share, and 1 where
// written -- as halves (splatTransferViewless writes them, splatProject
// reads them where `params.txCache`).
RWStructuredBuffer<uint4>   txCache;
// The radiance a splat gives off, RGB9E5, where the cloud carries it
// (`params.hasEmission`).
StructuredBuffer<uint>      emission;
// What its material layered over the base, three words a splat, where the
// cloud carries it (`params.hasLobes`).
StructuredBuffer<uint>      lobes;
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

void cull(uint slot, uint reason) {
    // Every early return zeroes the record first: a slot keeps last frame's
    // splat otherwise (openFXplayer learned this one the hard way).
    tilesTouched[slot] = 0;
    visible[slot] = 0;
    depthKeys[slot] = kCulledKey | reason;
    tileRects[slot * 4] = 1;
    tileRects[slot * 4 + 1] = 1;
    tileRects[slot * 4 + 2] = 0;
    tileRects[slot * 4 + 3] = 0;
}

/// The face a splat's transfer was measured for: its stored shading normal,
/// or its short axis as the frame turns it, in the cloud's space.
float3 storedFacing(float3x3 r, float3 scale, uint i) {
    if (params.hasNormals != 0) {
        return unpackNormal(normals[i]);
    }
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    return normalize(float3(r[0][shortest], r[1][shortest], r[2][shortest]));
}

/// `projectSplat`'s kTransfer for a SHADOW CATCHER (athenea mesh2splat
/// --shadow-catcher): projected as any splat, drawn black, its opacity what
/// the object took of the light reaching it (catcherOpacity).
static const int kProjectCatcher = 3;

/// HOW MANY OF A CATCHER'S GAUSSIANS A POINT OF IT LIES UNDER, weighed by
/// their falloff: a square lattice of cell d whose gaussians are sigma d
/// wide sums to 2 pi sigma^2 everywhere, and the conversion's sigma is 1. A
/// gaussian of opacity a then leaves about exp(-2 pi a) of what is behind
/// the layer. (2.4, the coverage a converted surface reaches at opacity one,
/// was the first guess: the Corvette's shadow came out 2.2 to 3 times as
/// deep as the path traced frame's in optical depth, which is 2 pi over 2.4.)
static const float kCatcherDepth = 6.2831853;

/// WHAT THE OBJECT TOOK OF THE LIGHT REACHING A CATCHER GAUSSIAN, as the
/// opacity that takes as much out of what is drawn behind it.
///
/// Both sides of the ratio in one quadrature, so that a gaussian nothing
/// shadows reads exactly one under any sky: over the cells of its open
/// directions (the TX bake's rays, one a cell), the sky's radiance from the
/// frame's harmonics times the cosine and the cell's solid angle -- with the
/// object, the cells the bake found open; without it, every cell above the
/// surface -- and the sun `env_sun` took out of the sky, at its cosine, read
/// along its own direction through the same bits on the first side alone.
/// The transfer's projection onto harmonics was the first try, against the
/// sky's own irradiance on the other side: two band limits that do not cancel,
/// and a ground that was not one where nothing stood over it.
///
/// The layer leaves exp(-kCatcherDepth a) of what is behind it, so a =
/// -ln(ratio) / that. Light the object bounces onto the ground beyond what it
/// takes (a ratio over one) is not drawn: a black layer can only take.
float catcherOpacity(uint i, float3x3 r, float3 scale, float opacity) {
    if (params.relight == 0 || params.shadowBits < 8u) {
        return 0.0;
    }
    uint dome = kEnvNone;
    for (uint k = 0; k < params.lightCount && k < params.envLights; ++k) {
        if (lights[k].kind == kLightDome && envOfLight[k] != kEnvNone) {
            dome = envOfLight[k];
            break;
        }
    }
    if (dome == kEnvNone) {
        return 0.0;
    }
    const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
    const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
    const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
    const float3 n = relightNormalToWorld(w0, w1, w2, storedFacing(r, scale, i));
    const uint words = params.shadowBits;
    const uint side = splatCellSide(words);
    const uint base = i * words;
    const uint coefficients = 16u;
    const float3 luma = float3(0.2126, 0.7152, 0.0722);
    // The sky's luminance coefficients, once.
    float sky[16];
    [unroll] for (uint k = 0; k < coefficients; ++k) {
        sky[k] = dot(envSh[dome * kEnvCoefficients + k].rgb, luma);
    }
    float with = 0.0;
    float without = 0.0;
    for (uint cell = 0; cell < side * side; ++cell) {
        const float2 uv = (float2(float(cell % side), float(cell / side)) + 0.5) / float(side);
        const float3 d = octDecode(uv);
        const float cosine = dot(n, d);
        if (!(cosine > 0.0)) {
            continue;
        }
        float radiance = 0.0;
        [unroll] for (uint k = 0; k < coefficients; ++k) {
            radiance += sky[k] * envBasisValue(k, d);
        }
        const float l1 = abs(d.x) + abs(d.y) + abs(d.z);
        const float worth = max(radiance, 0.0) * cosine * (4.0 / float(side * side)) * l1 * l1 * l1;
        without += worth;
        if (((shadowBits[base + (cell >> 5u)] >> (cell & 31u)) & 1u) != 0u) {
            with += worth;
        }
    }
    const float4 disc = envSun[dome * 2];
    if (disc.w > 0.0) {
        const float facing = max(dot(n, disc.xyz), 0.0);
        const float sun = dot(envSun[dome * 2 + 1].rgb, luma) * facing;   // irradiance, as `worth` is
        // The sun's open share: its four cells about its direction, weighed
        // bilinearly, those behind the surface left out (splatCellsOpen).
        const float2 at = octEncode(disc.xyz) * float(side) - 0.5;
        const float2 lo = floor(at);
        const float2 f = at - lo;
        float open = 0.0;
        float weight = 0.0;
        [unroll] for (uint k = 0; k < 4; ++k) {
            const int2 c = clamp(int2(lo) + int2(k & 1, k >> 1), int2(0), int2(int(side) - 1));
            const float wk = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
            if (wk <= 0.0 || dot(octDecode((float2(c) + 0.5) / float(side)), n) <= 0.0) {
                continue;
            }
            const uint index = uint(c.y) * side + uint(c.x);
            open += wk * float((shadowBits[base + (index >> 5u)] >> (index & 31u)) & 1u);
            weight += wk;
        }
        with += sun * (weight > 0.0 ? open / weight : 1.0);
        without += sun;
    }
    const float ratio = saturate(with / max(without, 1.0e-6));
    return opacity * saturate(-log(max(ratio, 1.0e-4)) / kCatcherDepth);
}

/// ONE SPLAT PROJECTED, and shaded where its cloud is relit. `kTransfer` 0
/// compiles the transfer's shading out (splatProjectPlain): a cloud with no
/// transfer never takes it, and on Metal the kernel that holds it drew a
/// relit cloud without one with whole splats gone wrong -- a stretched
/// splat of one colour channel, a dozen a frame, somewhere new each run --
/// where the same kernel with either branch compiled out drew it clean, and
/// the ray traced route, the same shading, was clean too.
void projectSplat<let kTransfer : int>(uint3 tid) {
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
        cull(slot, kCullEdit);
        return;
    }
    posOpacity.w = fate.opacity;

    const float3 t = toView(params, posOpacity.xyz);
    if (!(t.z > params.nearZ) || t.z > params.farZ) {
        cull(slot, kCullDepth);
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
        cull(slot, kCullDegenerate);
        return;
    }
    // `detBefore` is the still covariance's, so this one square root pays
    // for both convolutions: the shutter's and the mip filter's. Without the
    // filter there is still the shutter and the lens to pay for; a spread
    // splat that kept its peak would gain energy and the wings, or the bokeh,
    // would burn.
    const float paid = params.antialias != 0 ? sqrt(max(detBefore, 0.0) / det) : spreadPaid;
    if (kTransfer == kProjectCatcher) {
        posOpacity.w = catcherOpacity(i, r, scale, posOpacity.w);
    }
    // A THIN SHEET OF GLASS IS DRAWN AS A LAYER. Its gaussians cover what the
    // sheet reflects head on -- 0.0055 for a windshield at 1.16 -- and carry
    // that reflection scaled up by 1/F0 (splat_relight's thinWallGain). At
    // that opacity the faint cull, the cut radius and the blend's 1/255 per
    // pixel dropped most of them before anything was shaded: a windshield
    // came out the interior seen through, salted with the few that survived
    // at a hundred and eighty times their colour. So a sheet's opacity is
    // held to `kSheetAlpha` and its colour scaled down by as much, which
    // keeps alpha x colour -- the reflection -- and draws it whole. The
    // price is a sheet that covers a tenth of what is behind it per gaussian,
    // where it should let all of it through (until the blend carries a
    // transmittance of its own); measured on the Corvette's windshield it is
    // the smaller error by far (s62b: relMSE 2.26 to 0.89 under the shop,
    // 0.31 to 0.067 under goegap).
    const bool sheet = kTransfer != kProjectCatcher && params.hasPbr != 0 && pbrThinWalled(pbr[i]) &&
                       pbrTransmission(pbr[i]) > 0.5;
    // A sheet's own share is its coverage as the conversion solved it, without
    // the 1/255 m2sCoverageAlpha adds for the radius it would be cut at: drawn
    // at the floor, nothing of it is cut, and at 0.0018 of coverage that
    // 1/255 had tripled it -- the windshield read 3.9 times the path traced
    // one under the shop's lamps (s65).
    const float alphaOwn = sheet ? paid * max(posOpacity.w - 1.0 / 255.0, 0.0) : paidOpacity(posOpacity.w, paid);
    const float alpha = sheet ? max(alphaOwn, kSheetAlpha) : alphaOwn;
    if (!(alpha >= 1.0 / 255.0)) {
        cull(slot, kCullFaint);
        return;
    }
    const float3 conic = float3(c / det, -b / det, a / det);
    const float cutoff = cutoffPower(alpha);
    const float2 extent = footprintExtent(conic, cutoff);
    uint4 rect;
    tileRect(params, centre, extent, rect);
    if (rect.x > rect.z || rect.y > rect.w) {
        cull(slot, kCullOffscreen);
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
        cull(slot, kCullNoTile);
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
    // Whether `slopes[slot]` holds the reflection's slope (kSlopeMark).
    bool sloped = false;
    if (kTransfer == kProjectCatcher) {
        colour = float3(0.0);   // black: it only takes
    } else if (params.relight != 0) {
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
        surface.normal = relightNormalToWorld(w0, w1, w2, cloudNormal);
        // What the cloud shows in this direction, in linear light already.
        // The harmonics are part of it -- a baked cloud keeps the body's own
        // change with direction in them -- so it is the evaluated colour and
        // not the constant term alone.
        surface.albedo = colour;
        surface.metallic = params.hasPbr != 0 ? pbrMetallic(pbr[i]) : 0.0;
        surface.roughness = params.hasPbr != 0 ? pbrRoughness(pbr[i]) : 1.0;
        surface.transmission = params.hasPbr != 0 ? pbrTransmission(pbr[i]) : 0.0;
        surface.thinWalled = params.hasPbr != 0 && pbrThinWalled(pbr[i]);
        surface.schlickMetal = params.hasPbr != 0 && pbrSchlickMetal(pbr[i]);
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
        surface.exitThrough = float3(1.0);
        surface.hasReflection = false;
        surface.reflectionColour = float3(0.0);
        surface.reflectionTraced = false;
        surface.hasExit = false;
        surface.exitDirection = float3(0.0, 0.0, 1.0);
        surface.lit = params.litBody != 0;
        surface.emission = params.hasEmission != 0 ? unpackRgb9e5(emission[i]) : float3(0.0);
        surface.lobes = splatLobesOf(params.hasLobes != 0, lobes, i);
        // THE INDEX A GAUSSIAN BENDS BY IS ITS OWN where its cloud carries
        // the layers: the specular's index is the material's, where the
        // cloud's single `ior` is whichever glass came first (task TX, step 5;
        // the Corvette's read 1.053 for glass of 1.6). Only where the cloud
        // bends at all, and only on what transmits.
        if (params.hasLobes != 0 && surface.ior > 1.0 && surface.transmission > 0.0) {
            surface.ior = max(surface.lobes.specularIor, 1.0);
        }
        surface.transferCount = params.transferCount;
        surface.transferBase = i * params.transferWords * 2;
        // The frame the gaussian has now -- posed, for a cloud a skeleton
        // carries -- which is what a zonal transfer's lobes turn with.
        splatTransferFrame(surface, transfer, r, w0, w1, w2);
        // `params.shadowBits` is the words a gaussian: two of the first
        // transfer's 64 bits, eight or thirty-two of a TX transfer's cells.
        surface.hasShadowBits = params.shadowBits != 0 && params.shadowBits < 8u && params.transferCount >= 9;
        surface.shadowBits = surface.hasShadowBits ? uint2(shadowBits[i * 2], shadowBits[i * 2 + 1]) : uint2(0);
        surface.cellWords = params.shadowBits >= 8u && params.transferCount >= 9 ? params.shadowBits : 0u;
        surface.cellBase = i * surface.cellWords;
        const uint shadowLights = params.shadowRays != 0 ? params.shadowLights : 0;
        // WHICH WAY THE SURFACE FACES A PIXEL AWAY (SplatSlope): the shape
        // operator, in the splat's first two axes, applied to the step in
        // those axes that one pixel right and one pixel down are -- the
        // projection's own linearisation (`tx3`, `ty3`), inverted on the
        // splat's plane -- and turned with the normal the eye sees.
        SplatSlope slope = noSlope();
        if (kTransfer >= 2 && params.hasCurvature != 0) {
            // The mean curvature, on the face the eye sees (a lens's far face
            // is guessed from it): the shape operator was measured on the
            // mesh's normal.
            const float3 axisN = float3(r[0][2], r[1][2], r[2][2]);
            const float faces = dot(cloudNormal, axisN) >= 0.0 ? 1.0 : -1.0;
            slope.curvature = faces * 0.5 * (curvature[i * 3] + curvature[i * 3 + 2]);
        }
        if (kTransfer >= 2 && params.hasCurvature != 0 && surface.cellWords >= 8u) {
            const float3 u = float3(r[0][0], r[1][0], r[2][0]);
            const float3 v = float3(r[0][1], r[1][1], r[2][1]);
            const float2x2 toPixels = float2x2(dot(tx3, u), dot(tx3, v), dot(ty3, u), dot(ty3, v));
            const float det = toPixels[0][0] * toPixels[1][1] - toPixels[0][1] * toPixels[1][0];
            if (abs(det) > 1.0e-12) {
                // A pixel right, (1, 0), and down, (0, 1), on the splat's plane.
                const float2 stepX = float2(toPixels[1][1], -toPixels[1][0]) / det;
                const float2 stepY = float2(-toPixels[0][1], toPixels[0][0]) / det;
                const float3 shapeUV = float3(curvature[i * 3], curvature[i * 3 + 1], curvature[i * 3 + 2]);
                const float2x2 shapeOf = float2x2(shapeUV.x, shapeUV.y, shapeUV.y, shapeUV.z);
                // The shape operator was measured on the mesh's normal; the
                // eye may be on the other side of the disc.
                const float3 axis = float3(r[0][2], r[1][2], r[2][2]);
                const float side = dot(cloudNormal, axis) >= 0.0 ? 1.0 : -1.0;
                const float2 turnX = mul(shapeOf, stepX) * side;
                const float2 turnY = mul(shapeOf, stepY) * side;
                // A step the plane barely holds (a splat seen edge on) turns
                // the normal at most this far.
                const float most = 0.5;
                const float3 dx = u * turnX.x + v * turnX.y;
                const float3 dy = u * turnY.x + v * turnY.y;
                const float3 hx = dx * min(1.0, most / max(length(dx), 1.0e-9));
                const float3 hy = dy * min(1.0, most / max(length(dy), 1.0e-9));
                slope.on = true;
                slope.normalX = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hx));
                slope.normalY = relightNormalToWorld(w0, w1, w2, normalize(cloudNormal + hy));
            }
        }
        // WHAT THE EYE DOES NOT CHANGE, kept from when the sky and the lights
        // last changed (splatTransferViewless), where the eye sees the face
        // the terms were kept for.
        if (kTransfer >= 2 && params.txCache != 0 && surface.transferCount != kTransferZonalCount &&
            dot(cloudNormal, storedFacing(r, scale, i)) > 0.0) {
            const uint4 kept = txCache[i];
            if (kept.w != 0u) {
                slope.cached = true;
                slope.cachedDome = uint(highHalf(kept.w) + 0.5) - 1u;
                slope.cachedFront = float3(lowHalf(kept.x), highHalf(kept.x), lowHalf(kept.y));
                slope.cachedCoupling = float3(highHalf(kept.y), lowHalf(kept.z), highHalf(kept.z));
                slope.cachedSunShare = lowHalf(kept.w);
            }
        }
        colour = relitSplat<kTransfer>(surface, normalize(eye - surface.position), lights,
                            params.lightCount, params.categoriesLo, params.categoriesHi,
                            shadowFactors, slot * params.shadowLights, shadowLights,
                            envTexels, envSh, envOfLight, params.envLights, params.envBaseSide,
                            envSun, transfer, shadowBits,
                            params.transferIndirect != 0, slope);
        if (kTransfer >= 2 && slope.on) {
            // THE RAMP NEVER GOES BELOW BLACK INSIDE THE FOOTPRINT. A
            // chrome strip turns its normal so fast that a pixel's slope,
            // carried across the gaussian, went far past the highlight (the
            // strips 0.77 against the path traced frame, 0.59 held to the
            // colour). Held per axis to twice the colour it still went down to
            // minus the colour at the footprint's edge, and the blend's
            // clamp at zero drew that as black: a dashed dark line along a
            // crease, a pawn's gold ring under its glass head. So per
            // channel the ramp's largest fall over the ellipse of three
            // sigma -- 3 sqrt(g^T Sigma g) for the slope g and the projected
            // covariance -- is at most the colour, and the shade stays at or
            // above zero wherever the gaussian draws.
            {
                const float3 fall = 3.0 * sqrt(max(a * slope.slopeX * slope.slopeX +
                                                       2.0 * b * slope.slopeX * slope.slopeY +
                                                       c * slope.slopeY * slope.slopeY,
                                                   float3(0.0)));
                const float3 keep = min(float3(1.0), max(colour, float3(0.0)) / max(fall, float3(1.0e-9)));
                slope.slopeX *= keep;
                slope.slopeY *= keep;
            }
            // A sheet's ramp goes with its colour: both are scaled down to
            // the floor's alpha below, and a ramp left at the 1/F0 gain was
            // fifty times its colour -- the windshield's lamp tubes drawn as
            // a bright mirror, whatever its own share (s69), and past what a
            // half holds where a lamp is a hundred times the sky.
            if (sheet) {
                slope.slopeX *= alphaOwn / alpha;
                slope.slopeY *= alphaOwn / alpha;
            }
            sloped = true;
            slopes[slot] = uint4(packHalves(slope.slopeX.x, slope.slopeX.y),
                                 packHalves(slope.slopeX.z, slope.slopeY.x),
                                 packHalves(slope.slopeY.y, slope.slopeY.z), 0u);
        }
    }
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }

    ProjRecord record;
    record.screen = float4(centre, t.z, cutoff);
    if (sheet) {
        colour *= alphaOwn / alpha;   // the sheet's alpha x colour, as its own opacity had it
    }
    record.conic = float4(conic, alpha);
    record.colour = float4(colour, sloped ? kSlopeMark : kTransfer == kProjectCatcher ? kCatcherMark : 0.0);
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

/// A SHADOW CATCHER: projected, black, as opaque as what its object took
/// (catcherOpacity) -- no relighting at all, so a kernel of its own and a
/// small one.
[shader("compute")]
[numthreads(256, 1, 1)]
void splatProjectCatcher(uint3 tid: SV_DispatchThreadID) {
    projectSplat<kProjectCatcher>(tid);
}

/// A cloud that carries a TX transfer (the cells): everything.
[shader("compute")]
[numthreads(256, 1, 1)]
void splatProject(uint3 tid: SV_DispatchThreadID) {
    projectSplat<2>(tid);
}

/// A cloud that carries the first transfer (no cells): its shading, without
/// the TX transfer's (projectSplat says why kernels are split by size).
[shader("compute")]
[numthreads(256, 1, 1)]
void splatProjectFirst(uint3 tid: SV_DispatchThreadID) {
    projectSplat<1>(tid);
}

/// A cloud that carries none: the same with the transfer's shading compiled
/// out (projectSplat says why).
[shader("compute")]
[numthreads(256, 1, 1)]
void splatProjectPlain(uint3 tid: SV_DispatchThreadID) {
    projectSplat<0>(tid);
}

/// A TX TRANSFER'S VIEW-INDEPENDENT TERMS, A SPLAT EACH (transferViewless):
/// the body's light under the frame's first prepared dome, the sun's share
/// and the field's coupling to the sky, for the face the transfer was
/// measured for. Run when the sky, the lights, the cloud or its place
/// change; every frame in between reads them back (splatProject), and what
/// it computes is what the eye changes -- 37 of the TX car's 85 ms of
/// projection were these.
[shader("compute")]
[numthreads(256, 1, 1)]
void splatTransferViewless(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    txCache[i] = uint4(0u);
    // The frame's first dome with a prepared sky: what relitSplat answers a
    // dome with, and the only one kept.
    uint dome = kEnvNone;
    for (uint k = 0; k < params.lightCount && k < params.envLights; ++k) {
        if (lights[k].kind == kLightDome && envOfLight[k] != kEnvNone) {
            dome = envOfLight[k];
            break;
        }
    }
    if (params.transferCount < 9u || params.transferCount == kTransferZonalCount || dome == kEnvNone) {
        return;
    }
    const uint row = i * 4;
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    const float4 q = decodeQuaternion(shape[row]);
    const float3x3 r = splatRotation(q);
    const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
    const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
    const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
    SplatSurface s = {};
    s.normal = relightNormalToWorld(w0, w1, w2, storedFacing(r, scale, i));
    s.transferCount = params.transferCount;
    s.transferBase = i * params.transferWords * 2;
    s.hasShadowBits = params.shadowBits != 0 && params.shadowBits < 8u;
    s.shadowBits = s.hasShadowBits ? uint2(shadowBits[i * 2], shadowBits[i * 2 + 1]) : uint2(0);
    s.cellWords = params.shadowBits >= 8u ? params.shadowBits : 0u;
    s.cellBase = i * s.cellWords;
    s.lit = params.litBody != 0;
    if (s.lit) {
        return;
    }
    float3 front;
    float3 coupling;
    float  sunShare;
    transferViewless<2>(s, transfer, shadowBits, envSh, envSun, dome, params.transferIndirect != 0, front, coupling,
                        sunShare);
    // The slice, one up, in the last half: zero is "not kept".
    txCache[i] = uint4(packHalves(front.x, front.y), packHalves(front.z, coupling.x),
                       packHalves(coupling.y, coupling.z), packHalves(sunShare, float(dome + 1u)));
}
