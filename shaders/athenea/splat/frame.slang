// Copyright (c) 2026 jesus luque.
//
// What every stage of the splat rasteriser agrees on: the frame, the tile
// grid, how a splat's footprint is cut, and how a tile is tested against it.
//
// THE PIPELINE (athenea::render::TileRasterizer drives it)
//
//   project   per splat: EWA covariance, conic, colour, depth key, footprint,
//             and the exact number of tiles the footprint touches
//   sort      splats by depth (32-bit key)                  -- RadixSort
//   gather    tiles-touched in depth order, prefix sum      -- PrefixSum
//   emit      per splat in depth order: one (tile, splat) pair per tile
//   sort      pairs by tile, stably (tile bits only)        -- RadixSort
//   ranges    where each tile's pairs start and end
//   blend     per pixel: its tile's pairs, front to back
//
// Sorting splats by depth once and pairs by tile stably gives every tile its
// splats in depth order, ties in splat order: the same order a full 64-bit
// (tile, depth) sort gives, with a sort of N 32-bit keys and one of P keys of
// ~15 bits, instead of one of P 47-bit keys. Deterministic by construction.
//
// CONVENTIONS (openFXplayer's, measured and documented there)
//   - the kernel's frame: +z forward, x right, y up; screen origin bottom-left;
//     pixel centres at +0.5. The -Z camera of D36 is folded into the view.
//   - square pixels: focalY == focalX unless a window scale says otherwise.
//   - colour is blended in linear light (the working space, linear Rec.709).
//     A capture's colours are sRGB, as its photographs were, and each splat
//     is decoded when its harmonics are evaluated (common/color.slang,
//     `cloudLight`); a cloud that holds light already (`linearCloud`) is
//     taken as it is. sRGB appears only where an image is shown.
module frame;
import athenea.common.edit;

public static const uint kTile = 16;

// WHY A SPLAT WAS CULLED, kept where a culled slot's depth key would be: the
// compaction reads a key only where `visible` is 1, so the word is free, and
// a panel's counters (splat_frame_counters.slang) read the reason from it
// rather than the projection paying a counter of its own.
public static const uint kCulledKey     = 0xFF000000u;   // | the reason
public static const uint kCullEdit      = 1;   // removed by the prim's SplatEdit
public static const uint kCullDepth     = 2;   // nearer than the near plane or past the far one
public static const uint kCullDegenerate = 3;  // a footprint of no area
public static const uint kCullFaint     = 4;   // under 1/255 of opacity, once paid for its spread
public static const uint kCullOffscreen = 5;   // its footprint's tiles lie outside the frame
public static const uint kCullNoTile    = 6;   // in the frame, but no tile meets its ellipse
public static const uint kCullReasons   = 7;   // 0 is a slot no splat projection wrote (points)

public struct FrameParams {
    public uint  width;
    public uint  height;
    public uint  tilesX;
    public uint  tilesY;
    public float focalX;
    public float focalY;
    public float centreX;
    public float centreY;
    public float nearZ;
    public float farZ;
    public uint  orthographic;
    public uint  antialias;
    // Object -> view, rows of a 3x4.
    public float m00; public float m01; public float m02; public float m03;
    public float m10; public float m11; public float m12; public float m13;
    public float m20; public float m21; public float m22; public float m23;
    // The camera in the cloud's own space, for the harmonics' view direction.
    public float eyeX; public float eyeY; public float eyeZ;
    public uint  count;          // splats in this cloud / pairs / ...
    public uint  base;           // first projected slot of this cloud
    public uint  restPerColour;  // harmonics kept: 0, 3, 8, 15
    public uint  shWords;
    public uint  shLimit;        // highest degree to evaluate this frame
    public uint  tileBits;
    public uint  depthMode;      // 0 alpha-weighted mean, 1 first crossing
    public float depthThreshold;
    public float bgR; public float bgG; public float bgB; public float bgA;
    public uint  linearCloud;    // 1: this cloud's colours are linear light; 0: sRGB, decoded per splat
    public uint  hasUnder;       // 1: an opaque layer (raster points) is under the splats
    public uint  relight;        // 1: light these splats rather than show them as baked
    public uint  lightCount;     // lights in `lights`, 0 when nothing is relit
    public uint  categoriesLo;   // this cloud's categories, for a light's link
    public uint  categoriesHi;
    public uint  shadowRays;     // 1: a relit splat is shadowed by the cloud's own proxies
    public uint  shadowLights;   // lights splat_shadow measured a factor for
    public float shadowOffset;   // where its shadow ray starts, in the splat's own sigmas
    public float shadowCut;      // and where the ray gives up
    public uint  hasPbr;         // 1: `pbr` holds this cloud's metallic and roughness
    public uint  hasNormals;     // 1: `normals` holds this cloud's shading normals
    public uint  hasEmission;    // 1: `emission` holds the radiance this cloud's gaussians give off
    public uint  hasLobes;       // 1: `lobes` holds what this cloud's materials layered over the base
    public uint  litBody;        // 1: the colours are light already, and only the polish is added
    /// Lights an environment was prepared for: `envOfLight[k]` is the slice
    /// of light k, or `kEnvNone`. Zero: no sky prepared, and a dome is its
    /// colour as it always was (athenea/light/environment.slang).
    public uint  envLights;
    /// The octahedral side of that sky's level 0, which follows the dome's
    /// own image rather than being fixed: one number both the writer and the
    /// reader of the map must agree on (athenea/light/environment.slang).
    public uint  envBaseSide;
    /// ROWS OF `splatOverrides`, the per-prim material table a pick edits:
    /// 0 says there is none and the cloud's own values stand alone.
    public uint  overrideCount;
    /// A cloud's transfer: words a gaussian, values a gaussian (0, 9 or 36),
    /// and whether the indirect half is added. The toggle is a frame's, so
    /// the same cloud can be drawn with and without it and the difference
    /// measured without baking anything again.
    public uint  transferWords;
    public uint  transferCount;
    public uint  transferIndirect;
    /// The words a gaussian of which ways out are open (`shadowBits`), 0 where
    /// the cloud carries none: 2 for 8 x 8 cells, 8 or 32 for a TX transfer.
    public uint  shadowBits;
    /// THE INDEX A TRANSMITTING GAUSSIAN BENDS BY, 0 where it bends nothing.
    ///
    /// A cloud has no surfaces to refract at: a glass ball is hundreds of
    /// gaussians deep and Snell at every one of them is mush. What this does
    /// instead is bend the direction the transmitted half looks along, once,
    /// at the gaussian's own normal, and read the prepared sky there -- so
    /// the ball shows the room behind it turned, as glass does, rather than
    /// the cosine average of it. What it cannot show is what stands between
    /// the ball and the sky: that wants a ray, not an index.
    public float ior;
    // CRYPTOMATTE. 0: no matte this frame, and `cryptoIds` is not written.
    // 1: this cloud's `cloudCrypto` is the id a slot carries. 2: this cloud
    // has no ids (a capture, a point cloud) and its slots carry 0, which the
    // blend leaves out of the matte.
    public uint  cryptoMode;
    public uint  hasUnderCrypto; // 1: `underCrypto` holds the under layer's id a pixel
    // What the blend adds into `contribution` at each splat's slot, for a
    // cloud being pruned (`athenea decimate --drop-hidden`, `--prune-error`).
    // 0: nothing. 1: the weight it blended with (transmittance times alpha),
    // in 1/4096 of a pixel. 2: the squared error its removal would leave
    // (`finalColour` is the frame drawn whole), in 2^-24 as two words a slot.
    public uint  contributionMode;
    // Relighting happens in the world, where the lights are: the cloud to
    // world rows, and the eye there. The colour from the harmonics is still
    // the cloud's own business, which is why the eye above stays as it is.
    public float w00; public float w01; public float w02; public float w03;
    public float w10; public float w11; public float w12; public float w13;
    public float w20; public float w21; public float w22; public float w23;
    public float eyeWorldX; public float eyeWorldY; public float eyeWorldZ;
    // Where the sky turned into this cloud's frame starts in `environment.sh`
    // (splat/sky_turn.slang), for a transfer the stage turned; 0: the cloud
    // is not turned, and the world's sky is the transfer's.
    public uint  skyCloud;
    // A parameter of the gaussians drawn in place of their light
    // (splat_relight's kSplatView*, athenea:splatView); 0: the light.
    public uint  splatView;
    // UNDER A SHUTTER: HOW FAR THIS SPLAT MOVES DURING IT.
    //
    // Bit 0 of `motion` says the per-splat displacement buffer is read (what a
    // skeleton moved, in the cloud's own space); bit 1 says the rows below are
    // (what the camera and the prim's transform moved, as a view-space step).
    // Zero is the still frame, which must stay bit for bit what it was.
    public uint  motion;
    public float motionScale;    // the shutter over the span the displacement was measured on
    public float motionMax;      // longest screen displacement taken, pixels: a cap, not a look
    // A THIN LENS. The radius of the diaphragm in scene units (0 is a
    // pinhole, and the frame is what it always was) and the depth in focus.
    // A point at view depth z blurs into a disk of focalX * lensRadius *
    // |1/z - 1/focus| pixels; the covariance takes its variance, which is the
    // radius squared over four.
    public float lensRadius;
    public float focusDistance;
    // Object -> view, the DIFFERENCE over the shutter. Applied to a point it
    // is that point's view-space step, so the camera's motion and the prim's
    // are one set of twelve numbers rather than two mechanisms.
    public float s00; public float s01; public float s02; public float s03;
    public float s10; public float s11; public float s12; public float s13;
    public float s20; public float s21; public float s22; public float s23;
    /// 1: `curvature` holds this cloud's shape operators, three floats a
    /// gaussian, and a TX transfer's reflection gets its slope across the
    /// footprint (splat_project); 2: the same, one packed word a gaussian
    /// (common/packing.slang's packCurvature).
    public uint  hasCurvature;
    /// 1: `txCache` holds this cloud's view-independent transfer terms
    /// (splatTransferViewless), each with the dome slice it is for.
    public uint  txCache;
    /// THE LAMPS' LAYERS (GpuSplats::lightLayers): how many groups this
    /// cloud carries, a word each a splat in `lightLayers`, each added
    /// times its weight in `lightWeights` below (splat_project's lampLight).
    /// 0: none, and nothing is read.
    public uint  lightLayers;
    /// WHERE A SHARP RECORD'S POLISH STARTS IN `slopes`, in uint4 entries:
    /// the slot's slope at [slot], its base polish and sun weights at
    /// [polishPlane + slot]. One buffer, not two: the TX projection binds
    /// as many as Metal's thirty-one allow (splat_project).
    public uint  polishPlane;
    /// SURFELS (docs/decisions.md, surfels): 1 draws a gaussian with a flat
    /// axis -- a scale of exactly 0 -- as the disc it is, by the ray through
    /// each pixel (ProjRecord::surfel), 0 as the thin gaussian it projects to.
    public uint  surfels;
    /// 1: THE CLOUD'S PAYLOAD IS READ THROUGH `payloadIndex` (a cut of a
    /// `.athc`, lod::CutSelector): its positions and shapes are the cut's own,
    /// a gaussian each; its material, layers, transfer, open directions,
    /// curvature and ids stay where the levels of detail keep them, and
    /// `payloadIndex[i]` says which. 0: read at `i`.
    public uint  payloadIndexed;
    /// The lamps' weights this frame, a float4 a layer (linear rgb: the
    /// group's level times its radiance and colour), GpuSplats::
    /// kMaxLightGroups of them. Here and not a buffer, for the same count.
    public float4 lightWeights[16];
    /// 1: BLENDED AS A STANDARD VIEWER BLENDS. Each splat's colour is
    /// sRGB-encoded at its projection, the blend sums encoded values over the
    /// encoded layer under it, and the pixel is decoded to linear once at the
    /// end -- what Spark, SuperSplat and the 3DGS rasteriser draw, for
    /// measuring a file written for them (athenea flatten). 0: linear light,
    /// as every other frame.
    public uint  displayBlend;
    public EditParams edit;      // this cloud instance's SplatEdit (common/edit.slang)
    /// THE MESH LIGHTS' GRID (light/mesh_light_splat.slang): cells a side (0:
    /// no mesh light reaches a relit splat), entries a cell, and the box it
    /// is cut over, in the world.
    public uint  meshCellSide;
    public uint  meshCutK;
    public float meshGridLoX; public float meshGridLoY; public float meshGridLoZ;
    public float meshGridHiX; public float meshGridHiY; public float meshGridHiZ;
};

/// What the projection leaves for the blend, one record per splat, so a pair
/// costs the blend one memory access and not three.
/// What a record's colour.w says when the splat's colour has a slope across
/// its footprint in `slopes` (splat_project, splat_blend): neither 0, a plain
/// splat, nor above a half, an opaque point of the layer under the splats.
public static const float kSlopeMark = 0.25;
/// A ProjRecord's colour.w for a shadow catcher's gaussian (splat_project's
/// splatProjectCatcher): black, and drawn over an opaque layer under it
/// even where its centre's depth reads behind that layer -- it lies on it.
public static const float kCatcherMark = 0.125;
/// A ProjRecord's colour.w for a LIGHT CATCHER's gaussian (splat_project's
/// splatProjectLightCatcher): the lamps' light on the ground, drawn over the
/// layer under it as a catcher is, and added -- it takes nothing from what
/// is behind it, so it can only brighten.
public static const float kLightCatcherMark = 0.0625;
/// A record whose sharp lobe the blend draws per pixel (splat_project's
/// SplatSlope sharp lobe): `slopes` holds its mirror and the mirror's turn.
public static const float kSharpMark = 0.375;
/// A record of a MEDIUM's gaussian (splat_volume.slang): its alpha at a pixel
/// is the exact optical depth of the pixel's ray through it, and what it
/// needs for that is in the record and its `slopes` slot (splat_volume says
/// where).
public static const float kVolumeMark = 0.4375;
/// A medium's gaussian stops at 3.5 deviations, as `athenea vdb2splat` fits
/// it: squared, the furthest its least distance to a ray is counted at.
public static const float kVolumeTruncation = 12.25;

/// erf, Abramowitz and Stegun 7.1.26: an absolute error under 1.5e-7.
public float volumeErf(float x) {
    const float s = sign(x);
    const float a = abs(x);
    const float t = 1.0 / (1.0 + 0.3275911 * a);
    const float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
                           0.254829592) * t * exp(-a * a);
    return s * y;
}

public struct ProjRecord {
    public float4 screen;   // centre.xy, view z, cutoff power
    public float4 conic;    // a, b, c, alpha
    public float4 colour;   // rgb (linear), 0
    /// A SURFEL'S DISC ON SCREEN, where conic.w < 0 marks one: its two axes
    /// (scale times direction) in homogeneous screen coordinates -- (x w, y w,
    /// w) of the disc's u and v -- as halves, u.xy, u.w v.x, v.y v.w, 0. With
    /// the centre's own (screen.xy times its depth, and the depth) they are
    /// the 3x3 map from the disc's plane to the screen that the blend inverts
    /// at every pixel (Huang et al. 2024, ray-splat intersection). 0 for a
    /// gaussian.
    public uint4  surfel;
};

public float3 toView(FrameParams f, float3 p) {
    return float3(f.m00 * p.x + f.m01 * p.y + f.m02 * p.z + f.m03,
                  f.m10 * p.x + f.m11 * p.y + f.m12 * p.z + f.m13,
                  f.m20 * p.x + f.m21 * p.y + f.m22 * p.z + f.m23);
}

public float3x3 viewLinear(FrameParams f) {
    return float3x3(f.m00, f.m01, f.m02, f.m10, f.m11, f.m12, f.m20, f.m21, f.m22);
}

/// THE PEAK A SPREAD SPLAT IS DRAWN WITH, so that what it draws is what the
/// still, sharp one draws. Spreading a gaussian (the antialiasing filter, the
/// shutter, the lens) by a determinant ratio `paid^2` keeps its energy when
/// its peak goes to `paid * alpha`. But a splat is drawn only out to where it
/// falls under 1/255 (`cutoffPower`), and what that leaves out is 1/255 of
/// peak times its area -- an area the spread made `1/paid` times larger. So a
/// splat a fraction of a pixel wide lost most of itself: a converted surface
/// meant to cover 0.25 covered 0.18 at a twentieth of the screen, where the
/// ray tracer, which draws the sharp splat, kept 0.25 at every size. Drawn at
/// `paid * alpha + (1 - paid) / 255` the spread splat keeps `alpha - 1/255`
/// of the sharp one's area, exactly what the sharp one keeps, and a splat the
/// spread does not touch (`paid` 1) is unchanged. Under the three-sigma cap a
/// splat keeps 98.9% of itself whatever its peak, and the term is under a
/// 255th either way.
///
/// ONLY WHERE THE SPREAD SPLAT IS DRAWN AT ALL (`paid * alpha >= 1/255`), as
/// before. Paying every splat back would draw the ones the spread had made
/// too faint to be drawn, and those are what a distant cloud is made of: the
/// converted chess pawn at 80 pixels of a 512 x 512 frame went from 8.7 to
/// 29.8 ms (debug build) for 730 thousand gaussians a few hundredths of a
/// pixel wide. Paid back only where drawn it costs 11.0 ms, and that pawn
/// still all but vanishes from the raster (the ray tracer keeps it): such a
/// cloud wants its levels of detail.
public float paidOpacity(float alpha, float paid) {
    const float spread = paid * alpha;
    return spread >= 1.0 / 255.0 ? spread + (1.0 - paid) * (1.0 / 255.0) : spread;
}

/// The squared Mahalanobis radius a splat is cut at: where its own
/// contribution falls below 1/255, and never past three sigma. The blend cuts
/// at the same radius, so no pixel outside a counted tile could have been
/// touched -- which is what keeps tile edges seamless.
public float cutoffPower(float alpha) {
    return min(2.0 * log(max(255.0 * alpha, 1.0001)), 9.0);
}

/// SPARK'S HARDER KERNEL, an opacity D over 1 (sparkjs splatFragment.glsl;
/// docs/decisions.md, kernel hardness): the composite of A = exp((D^2 - 1)/e)
/// coincident gaussians, 1 - (1 - g)^A -- a flat top as wide as sqrt(2 ln A)
/// sigmas and the gaussian's shoulder past it. A merge whose members covered
/// its area D times over draws as solid as they did. D is held to 5, and the
/// cut widened as Spark widens it, by 0.7 (D - 1) sigma.
public static const float kFalloffMost = 5.0;

public float falloffCount(float d) {
    return exp((d * d - 1.0) * 0.36787944);
}

public float falloffAlpha(float d, float g) {
    return d > 1.0 ? 1.0 - pow(max(1.0 - g, 0.0), falloffCount(d)) : d * g;
}

public float falloffCutoff(float d) {
    const float r = 3.0 + 0.7 * (min(d, kFalloffMost) - 1.0);
    return min(2.0 * log(255.0 * falloffCount(d)), r * r);
}

/// Minimum over a box of q(d) = a dx^2 + 2 b dx dy + c dy^2, for d measured
/// from the splat's centre. Exact: q is convex, so with the centre outside the
/// box the minimum lies on an edge, and on an edge it is a one-dimensional
/// quadratic minimised in closed form and clamped.
public float minPowerOverBox(float a, float b, float c, float2 lo, float2 hi) {
    if (lo.x <= 0.0 && hi.x >= 0.0 && lo.y <= 0.0 && hi.y >= 0.0) {
        return 0.0;
    }
    float best = 3.4e38;
    // x fixed at lo.x / hi.x, y free in [lo.y, hi.y]
    for (int e = 0; e < 2; ++e) {
        const float x = e == 0 ? lo.x : hi.x;
        const float y = c > 1e-12 ? clamp(-b * x / c, lo.y, hi.y) : lo.y;
        best = min(best, a * x * x + 2.0 * b * x * y + c * y * y);
    }
    for (int e = 0; e < 2; ++e) {
        const float y = e == 0 ? lo.y : hi.y;
        const float x = a > 1e-12 ? clamp(-b * y / a, lo.x, hi.x) : lo.x;
        best = min(best, a * x * x + 2.0 * b * x * y + c * y * y);
    }
    return best;
}

/// Does tile (tx, ty) hold a pixel centre inside the footprint?
public bool tileTouched(FrameParams f, float2 centre, float3 conic, float cutoff, uint tx, uint ty) {
    const float2 first = float2(float(tx * kTile) + 0.5, float(ty * kTile) + 0.5);
    const float2 last = float2(float(min((tx + 1) * kTile, f.width) - 1) + 0.5,
                               float(min((ty + 1) * kTile, f.height) - 1) + 0.5);
    return minPowerOverBox(conic.x, conic.y, conic.z, first - centre, last - centre) <= cutoff;
}

/// The footprint's axis-aligned extent, half-widths in pixels, from the 2D
/// covariance (the conic's inverse): sqrt(cutoff * sigma_xx), sqrt(cutoff * sigma_yy).
public float2 footprintExtent(float3 conic, float cutoff) {
    const float det = conic.x * conic.z - conic.y * conic.y;
    const float inv = det > 1e-20 ? 1.0 / det : 0.0;
    return sqrt(max(float2(cutoff * conic.z * inv, cutoff * conic.x * inv), 0.0));
}

/// The tile rectangle [lo, hi] (inclusive) of a footprint; lo > hi when it is off screen.
public void tileRect(FrameParams f, float2 centre, float2 extent, out uint4 rect) {
    const float x1 = centre.x - extent.x;
    const float y1 = centre.y - extent.y;
    const float x2 = centre.x + extent.x;
    const float y2 = centre.y + extent.y;
    if (x2 < 0.0 || y2 < 0.0 || x1 >= float(f.width) || y1 >= float(f.height)) {
        rect = uint4(1, 1, 0, 0);
        return;
    }
    rect = uint4(uint(max(x1, 0.0)) / kTile, uint(max(y1, 0.0)) / kTile,
                 uint(min(x2, float(f.width) - 1.0)) / kTile,
                 uint(min(y2, float(f.height) - 1.0)) / kTile);
}
