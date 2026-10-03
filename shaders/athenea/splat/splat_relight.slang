// Copyright (c) 2026 jesus luque.
//
// What this scene's lights put on a splat, for both routes that draw one.
//
// WHY IT LIVES IN A MODULE OF ITS OWN
//
// Relighting used to be written into `splat_project.slang`, which is the tile
// rasteriser's -- so a cloud that asked to be relit was relit when rasterised
// and shown exactly as it was baked when path traced. The flag meant two
// things, and with a cloud converted from a mesh (whose colours are an albedo
// and not radiance anybody captured) the traced frame was the albedo itself:
// a white material came back at 1.0 and a black marble at 0.003, both with no
// light on them at all. This engine's rule everywhere else is that the routes
// agree; here they now share the function rather than a description of it.
//
// WHAT IT IS, AND WHAT IT IS NOT
//
// A splat is not a surface. The normal is its shortest axis -- the direction a
// flattened Gaussian faces -- turned towards the eye, or where the cloud keeps a
// shading normal of its own (`GpuSplats::normals`, a conversion's normal map)
// that one, turned to the same side (`splatShadingNormal`); its albedo is the
// harmonics' constant term, the colour it shows on average. On top of that
// there is now a specular lobe, because without one a dark glossy material is
// black: the chess set's marble is `0.003` of diffuse albedo and everything
// you see of it in a render of the mesh is reflection. The lobe is GGX with
// Smith's height-correlated shadowing and Schlick's Fresnel, the same shape
// the surface lobes use, driven by a metallic and a roughness the splat
// carries where its cloud was converted from a mesh.
//
// WHAT A MATERIAL LAYERS OVER ITS BASE (`SplatSurface::lobes`, proposal 026).
// Where the conversion read them, a gaussian also carries what OpenPBR,
// standard_surface and UsdPreviewSurface put over the base: a weight, colour
// and index for the dielectric reflection, a clear coat, and a sheen. None of
// it is a new lobe -- the coat is the same GGX dielectric the polish is, at
// its own roughness and index, and the sheen is the lobe library's Imageworks
// sheen -- and each is layered the way MaterialX's `layer` node layers: what
// lies on top takes from what reaches the layer below it the share it
// reflects at the eye (`SplatLayers`). A gaussian that carries none of it
// reflects with `plainLobes()`, which is this module as it was before.
//
// One sample at the middle of each light, no second sample, and a shadow only
// where the caller measured one beforehand. That is what a splat gets: it has
// no BSDF to importance sample and no ray of its own in the rasteriser.
module splat_relight;
__exported import athenea.light.lights;
__exported import athenea.light.environment;
import athenea.common.color;
import athenea.common.octahedral;
import athenea.common.packing;
// The DFG fit the surface lobes use: one definition for both routes. It
// declares no binding, which is what lets a module below material import it.
import athenea.material.microfacet;

/// What a splat reflects with.
/// One value of a gaussian's transfer, f16 pairs, `base` counted in halves.
public float transferAt(StructuredBuffer<uint> transfer, uint base, uint k) {
    const uint at = base + k;
    const uint word = transfer[at >> 1];
    return f16tof32((at & 1u) != 0u ? (word >> 16) : (word & 0xFFFFu));
}

public struct SplatSurface {
    public float3 position;
    public float3 normal;      ///< already turned towards the eye
    public float3 albedo;
    /// HOW MUCH OF AN ENVIRONMENT REACHES THIS GAUSSIAN, where its cloud
    /// carries it: `transferBase` is where its values start in the buffer,
    /// counted in halves, and `transferCount` is 0, 9 (the direct half) or 36
    /// (both). With it a cloud is lit by a sky it was never baked under.
    public uint   transferBase;
    public uint   transferCount;
    /// WHICH WAYS OUT ARE OPEN, where the cloud carries them: sixty-four bits
    /// of an 8 x 8 octahedral grid, set where the bake's ray left the scene.
    /// What the sun is shadowed by when they are there (`splatSunOpen`).
    public bool   hasShadowBits;
    public uint2  shadowBits;
    /// A THIN WALL: a glass sheet the conversion made as transparent as the
    /// sheet is (its opacity is what the sheet reflects head on), so the
    /// blend is its transmission and what it shades is its reflection alone.
    public bool   thinWalled;
    public float  metallic;    ///< 0 for a capture, which carries no such thing
    public float  roughness;   ///< 1 is the Lambert-only answer this had before
    /// What the material lets through, and the reason a splat can be
    /// translucent at all. A gaussian cannot refract; what it can do is send
    /// on the light that arrives from **behind** it, which is what glass lit
    /// from the far side does and what a purely reflective lobe never does.
    /// So the body of the material is split: `1 - transmission` of it faces
    /// the light, and `transmission` of it faces away. What stands behind the
    /// splat is seen through its own opacity, which the conversion lowers for
    /// the same reason.
    public float  transmission;
    /// WHETHER THIS GAUSSIAN IS THE FAR FACE OF WHAT IT STANDS ON.
    ///
    /// A normal is turned towards the eye before anything is shaded, which is
    /// right for a body and wrong for glass: a ball is a shell, the eye sees
    /// both of its faces through each other, and each of them handed over the
    /// image behind it -- the same sky twice, and a ball twice as bright as
    /// the mesh beside it. What a solid transmits is what its near face
    /// transmits; the far one is where the ray left, and it was counted
    /// there.
    public bool   backFacing;
    /// WHERE THE GLASS ACTUALLY SENDS IT, when a ray went and looked.
    ///
    /// A ball is a thick lens and one bend is not a lens: what turns the room
    /// over is the second interface. Where the route can trace the cloud
    /// (rt_glass.slang finds the far face), the direction that comes back
    /// from bending in and out again is handed over here, and the single bend
    /// below is what is left for a route that cannot.
    public bool   hasExit;
    public float3 exitDirection;
    /// WHAT THE FAR FACE LETS OUT, where a ray found one: what is not
    /// reflected back into the glass there, and the material's tint a second
    /// time, since a ray through a solid crosses two interfaces and the mesh
    /// tints at each. One where nothing was found.
    public float3 exitThrough;
    /// WHAT THE MIRROR DIRECTION ACTUALLY MEETS, where a ray went and looked.
    ///
    /// The polish reads the prepared sky along the mirror direction, which is
    /// right for a gaussian with nothing in front of it and wrong for one
    /// facing its own asset: a collar under a ball reflects the ball, not the
    /// ceiling. Where the route can trace the cloud, what the mirror ray meets
    /// is handed over here -- the colour of that particle, which the pass
    /// before this one has already shaded.
    public bool   hasReflection;
    public float3 reflectionColour;
    /// WHETHER A RAY WENT AND LOOKED AT ALL.
    ///
    /// Where one did and met nothing, nothing is in the way and the whole sky
    /// reaches the mirror direction: the fit below is not asked. Where none
    /// was cast, the fit is all there is. Leaving the fit in after a ray had
    /// already answered took a tenth off a gold collar under a sky of one
    /// colour, which is a surface that should read exactly what the mesh
    /// beside it reads.
    public bool   reflectionTraced;
    /// WHAT THE RAY MET AFTER LEAVING, where it met anything: the colour of
    /// the cloud's own particle behind the glass, already linear. A ball
    /// shows the room, and a pawn's ball shows its collar; the sky is only
    /// what is left where nothing stands between.
    public bool   hasBehind;
    public float3 behindColour;
    /// THE INDEX THAT HALF BENDS BY, 0 where nothing bends it.
    ///
    /// With one, the transmitted half stops being a cosine average of what
    /// is behind and becomes the sky along the direction Snell gives at this
    /// gaussian's own normal: a ball shows the room turned, which is the one
    /// thing that says glass. It is the entry face alone -- a cloud has no
    /// far face to find without a ray -- so a thick lens is not what this
    /// draws.
    public float  ior;
    /// True where `albedo` is not an albedo at all but light: a cloud whose
    /// body was baked (`athenea mesh2splat`, which asks the path tracer for the
    /// light on every gaussian). What is left for a frame to add is the
    /// polish -- the reflection, which is the part of a surface that one
    /// colour cannot hold and which a bake therefore left out.
    public bool   lit;
    /// WHAT THE SURFACE GIVES OFF BY ITSELF, linear radiance (the cloud's
    /// `emission`, a converted lamp shade or screen; zero where it has none).
    /// Added as it is: no light reaches it and no shadow takes it away, and
    /// it is the same from both sides of the disc. Not where the body is
    /// `lit` -- a radiance bake met the emission at its first vertex and
    /// holds it already, and adding it again would be it twice.
    public float3 emission;
    /// WHAT THE MATERIAL LAYERED OVER ITS BASE: the specular's weight,
    /// colour and index, the coat, the sheen (packing.slang's `SplatLobes`).
    /// `plainLobes()` where the cloud carries none (`splatLobesOf`).
    public SplatLobes lobes;
};

/// A gaussian's layers, or the plain ones where its cloud carries none.
public SplatLobes splatLobesOf(bool has, StructuredBuffer<uint> lobes, uint i) {
    return has ? unpackLobes(uint3(lobes[i * 3], lobes[i * 3 + 1], lobes[i * 3 + 2])) : plainLobes();
}

/// A dielectric's reflectivity head on, at an index: 0.04 at 1.5 exactly,
/// which is the number every cloud before an index was reflected with.
float dielectricF0(float ior) {
    return abs(ior - 1.5) < 1.0e-6 ? 0.04 : iorToF0(max(ior, 1.0));
}

/// What tints the dielectric reflection: the specular's weight times its
/// colour, one for a gaussian that carries neither.
float3 specularTint(SplatSurface s) {
    return s.lobes.specularWeight * s.lobes.specularColour;
}

/// GGX's D times Smith's height-correlated V for a light in `wi`, at
/// `roughness`: the shape every reflection here shares, the coat's as much
/// as the polish's. Zero below the surface.
float ggxShape(float3 n, float3 wo, float3 wi, float roughness) {
    const float3 h = normalize(wo + wi);
    const float nDotV = max(dot(n, wo), 1.0e-4);
    const float nDotL = max(dot(n, wi), 0.0);
    const float nDotH = max(dot(n, h), 0.0);
    if (nDotL <= 0.0) {
        return 0.0;
    }
    // A perfectly smooth splat would be a mirror the size of a texel, which
    // at one sample a light is a spike nothing resolves; the floor is the
    // same one an interactive frame puts on a surface.
    const float alpha = max(roughness * roughness, 1.0e-3);
    const float a2 = alpha * alpha;
    const float d = nDotH * nDotH * (a2 - 1.0) + 1.0;
    const float distribution = a2 / max(3.14159265358979 * d * d, 1.0e-9);
    const float lambdaV = nDotL * sqrt(nDotV * nDotV * (1.0 - a2) + a2);
    const float lambdaL = nDotV * sqrt(nDotL * nDotL * (1.0 - a2) + a2);
    return distribution * 0.5 / max(lambdaV + lambdaL, 1.0e-9);
}

/// GGX, as the surface lobes write it: D, Smith's height-correlated V, and
/// Schlick's F folded into one weight for a light in direction `wi`.
public float3 specularLobe(SplatSurface s, float3 wo, float3 wi) {
    const float shape = ggxShape(s.normal, wo, wi, s.roughness);
    if (shape <= 0.0) {
        return float3(0.0);
    }
    const float vDotH = max(dot(wo, normalize(wo + wi)), 0.0);
    const float p5 = pow(1.0 - vDotH, 5.0);
    // A dielectric reflects 4% head on (at 1.5), tinted by the specular's
    // weight and colour; a metal reflects its own colour, turning to the
    // specular colour at grazing -- except where the metal is already baked,
    // and then there is nothing of it left to add. With the plain lobes both
    // are what they were: `f0 + (1 - f0) (1 - v.h)^5`.
    const float3 tint = specularTint(s);
    const float f0d = dielectricF0(s.lobes.specularIor);
    float3 fresnel;
    if (s.lit) {
        const float3 f0 = f0d * tint * (1.0 - s.metallic);
        fresnel = f0 + (tint - f0) * p5;
    } else {
        const float3 dielectric = f0d * tint + (tint - f0d * tint) * p5;
        const float3 metal = s.albedo + (s.lobes.specularColour - s.albedo) * p5;
        fresnel = lerp(dielectric, metal, s.metallic);
    }
    return shape * fresnel;
}

/// The coat's reflection of a light in `wi`: the same GGX at the coat's
/// roughness, with a dielectric's Schlick at the coat's index, by its weight.
float3 coatLobe(SplatSurface s, float3 wo, float3 wi) {
    if (s.lobes.coatWeight <= 0.0) {
        return float3(0.0);
    }
    const float shape = ggxShape(s.normal, wo, wi, s.lobes.coatRoughness);
    const float vDotH = max(dot(wo, normalize(wo + wi)), 0.0);
    const float f0 = dielectricF0(s.lobes.coatIor);
    return float3(s.lobes.coatWeight * shape * (f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0)));
}

/// The sheen's reflection of a light in `wi`, Imageworks' (Conty and
/// Kulla's) as the lobe library has it: its colour times the BRDF.
float3 sheenLobe(SplatSurface s, float3 wo, float3 wi) {
    if (all(s.lobes.sheenColour <= float3(0.0))) {
        return float3(0.0);
    }
    const float nDotL = dot(s.normal, wi);
    if (nDotL <= 0.0) {
        return float3(0.0);
    }
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float nDotH = max(dot(s.normal, normalize(wo + wi)), 0.0);
    return s.lobes.sheenColour * imageworksSheen(nDotL, nDotV, nDotH, max(s.lobes.sheenRoughness, 0.01));
}

/// What a lobe of this roughness reflects of everything around it, as
/// Lazarov's analytic fit of the split-sum environment term gives it. It is
/// what a dome needs: a dome is not somewhere, so one sample of it is not a
/// highlight but a fraction of a whole sky, and the fraction is this.
/// WHAT A GAUSSIAN SENDS BACK OF AN ENVIRONMENT, AS THE SURFACE DOES IT.
///
/// This was Lazarov's fit, which is cheap and old, while the surface lobes
/// beside it used MaterialX's `ggxDirectionalAlbedo`. Asked the same
/// question over a grid of roughness and `N.V` the two came apart by 76 % --
/// worst where a surface grazes, which is every silhouette in a frame -- so
/// a cloud and the mesh next to it under one sky could not have agreed.
///
/// What is answered here is what `lobes.slang` answers for the two lobes a
/// metallic-roughness material is, mixed by the metalness exactly as the
/// `mix` node in `athenea_usd_preview_surface.mtlx` mixes them:
///
///   a dielectric   `F0` of 0.04 against a grazing one, Fresnel taken
///                  exactly (`fresnelDielectric`), which is `kLobeDielectric`
///   a conductor    its own colour at both ends, which is what a conductor's
///                  Fresnel does and what `kLobeConductor` passes
///
/// and both are scaled by `ggxEnergyCompensation`, the light that leaves a
/// rough surface after bouncing between its microfacets. Without it a rough
/// gaussian is darker than the rough mesh beside it by everything the second
/// bounce carries, which at alpha 0.5 is a tenth and climbing.
public float3 ggxEnvDielectric(float roughness, float nDotV) {
    return ggxEnvDielectricAt(roughness, nDotV, 1.5);
}

/// The same at an index of its own: a specular's `specular_ior`, a coat's.
public float3 ggxEnvDielectricAt(float roughness, float nDotV, float ior) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    const float3 F = float3(fresnelDielectric(nDotV, max(ior, 1.0)));
    return ggxDirectionalAlbedo(nDotV, alpha, float3(dielectricF0(ior)), float3(1.0)) *
           ggxEnergyCompensation(nDotV, alpha, F);
}

float3 ggxEnvConductor(float3 albedo, float roughness, float nDotV) {
    return ggxEnvConductorEdged(albedo, float3(1.0), roughness, nDotV);
}

float3 ggxEnvConductorEdged(float3 albedo, float3 edge, float roughness, float nDotV) {
    const float alpha = max(roughness * roughness, 1.0e-4);
    // A conductor's Fresnel stands for both ends of the fit, and it is taken
    // the way the surface takes it: MaterialX's artistic index out of the
    // colour, then the conductor's own Fresnel. Schlick between the colour
    // and one was a per cent off in the middle of the curve and six per cent
    // off on gold, which is a gold collar six per cent darker than the mesh
    // beside it under a sky of one colour.
    // The edge is white, not the colour: `standard_surface`'s metal takes its
    // grazing tint from `specular_color`, which is white unless an artist says
    // otherwise, and a gold that stays gold at grazing came back a fifth dark
    // in blue against the mesh beside it.
    float3 index;
    float3 extinction;
    mxArtisticIor(saturate(albedo), saturate(edge), index, extinction);
    const float3 F = fresnelConductor(nDotV, index, extinction);
    return ggxDirectionalAlbedo(nDotV, alpha, F, F) * ggxEnergyCompensation(nDotV, alpha, F);
}

/// The two, mixed by the metalness.
public float3 environmentBrdf(float3 albedo, float metallic, float roughness, float nDotV) {
    return lerp(ggxEnvDielectric(roughness, nDotV), ggxEnvConductor(albedo, roughness, nDotV),
                saturate(metallic));
}

/// The two with a gaussian's own layers: the dielectric at the specular's
/// index and tinted by its weight and colour, the metal turning to the
/// specular colour at grazing (standard_surface's `specular_color` on a
/// metal). The plain lobes make it `environmentBrdf`.
public float3 environmentBrdfOf(SplatSurface s, float nDotV) {
    return lerp(ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) * specularTint(s),
                ggxEnvConductorEdged(s.albedo, s.lobes.specularColour, s.roughness, nDotV), saturate(s.metallic));
}

/// WHAT THE LAYERS OVER THE BASE TAKE AND GIVE AT THE EYE.
///
/// MaterialX's `layer` weighs the base by the top's throughput at the view
/// direction, `1 - weight x directional albedo`, and the bake's lobe stack
/// does the same (material_runtime.slang): so a coat of weight one at 1.5
/// takes 4% from everything under it head on and most of it at grazing, and
/// gives it back as its own reflection. The coat is over the sheen and the
/// sheen over the base, as standard_surface stacks them (OpenPBR puts its
/// fuzz over the coat; with both on one gaussian the order is
/// standard_surface's). The sheen's weight is its
/// colour's largest channel -- exact for the white sheen every surface
/// defaults to -- since a gaussian carries the two multiplied.
public struct SplatLayers {
    public float3 coat;       ///< the coat's directional albedo, times its weight
    public float3 sheen;      ///< the sheen's, times its colour
    public float3 under;      ///< what reaches the base: through the coat and the sheen
    public float3 belowCoat;  ///< what reaches the sheen: through the coat
};

public SplatLayers splatLayers(SplatSurface s, float nDotV) {
    SplatLayers l;
    l.coat = s.lobes.coatWeight > 0.0
                 ? s.lobes.coatWeight * ggxEnvDielectricAt(s.lobes.coatRoughness, nDotV, s.lobes.coatIor)
                 : float3(0.0);
    const float sheenAlbedo = any(s.lobes.sheenColour > float3(0.0))
                                  ? imageworksSheenDirectionalAlbedo(nDotV, max(s.lobes.sheenRoughness, 0.01))
                                  : 0.0;
    l.sheen = s.lobes.sheenColour * sheenAlbedo;
    const float sheenWeight = max(max(s.lobes.sheenColour.x, s.lobes.sheenColour.y), s.lobes.sheenColour.z);
    l.belowCoat = saturate(float3(1.0) - l.coat);
    l.under = l.belowCoat * saturate(1.0 - sheenWeight * sheenAlbedo);
    // OPENPBR'S COAT DARKENING. Light that enters a coat and comes back up
    // from the base is partly reflected down again at the coat's underside,
    // and the base absorbs some of it each time: the base comes out darker
    // than the coat's own throughput says, `(1 - K) / (1 - E K)` with
    // `K = 1 - (1 - F0) / n^2` the coat's internal reflection and `E` the
    // base's albedo -- the graph's own estimate (open_pbr_surface.mtlx,
    // `base_darkening`): the base colour, times the specular weight for a
    // metal. Under a lacquer at 1.45 a dark metal keeps 0.47 of itself; the
    // car's paint read 1.5 times the mesh at the centre of its ball without
    // it. A baked body is darkened already (the bake's stack is the graph),
    // so only the polish the frame adds is, with its albedo taken as none.
    if (s.lobes.coatWeight > 0.0 && s.lobes.coatDarkening >= 0.5) {
        const float n = max(s.lobes.coatIor, 1.0);
        const float K = 1.0 - (1.0 - dielectricF0(n)) / (n * n);
        const float3 E = s.lit ? float3(0.0)
                               : saturate(lerp(s.albedo, s.albedo * s.lobes.specularWeight, saturate(s.metallic)));
        l.under *= lerp(float3(1.0), (1.0 - K) / max(float3(1.0) - E * K, float3(1.0e-4)), s.lobes.coatWeight);
    }
    return l;
}

/// A dome, answered whole rather than sampled.
///
/// Every other light is somewhere: one sample at its centre is a direction
/// worth taking, and the lobe evaluated there is the highlight. A dome is
/// everywhere, and a single sample of it gives a splat one spike where a
/// surface would show a broad sheen -- which is exactly what the chess set's
/// polished marble is made of. So it is integrated instead: uniform radiance
/// over the hemisphere is `albedo * L` of diffuse, and `environmentBrdf * L`
/// along the mirror direction of specular, both closed forms.
/// WHAT THE SKY IS, WHERE ONE WAS PREPARED.
///
/// A dome's image goes through the material texture table and this kernel has
/// none, so until `technique::Environment` a cloud under an HDRI was lit by
/// the dome's constant colour: the sky's mean, with none of its direction,
/// while the mesh beside it reflected the sky itself. What the environment
/// prepares is that image in two forms a buffer can hold -- nine harmonics of
/// radiance for the body, and the sky convolved with a GGX lobe for the
/// reflection -- and both already carry the light's own emission.
///
/// Without one (`kEnvNone`), every line below falls back to what it read
/// before: a constant dome's irradiance is `pi * L`, so the body reads `L`
/// either way, and the two are the same picture.
/// THE BODY, FROM A TRANSFER RATHER THAN FROM AN IRRADIANCE.
///
/// A gaussian that carries a transfer carries how much of an environment
/// reaches it -- visibility times the cosine, its own albedo taken as one --
/// so what it reflects under any sky is that dotted with the sky's own
/// harmonics, times its albedo. The direct half is a scalar a coefficient
/// (geometry alone); the indirect one is coloured by what the light bounced
/// off on the way, so it is an rgb triple a coefficient.
float3 transferredBody(SplatSurface s, StructuredBuffer<uint> transfer, StructuredBuffer<float4> envSh,
                       uint dome, bool withIndirect) {
    const uint base = dome * kEnvCoefficients;
    float3 sum = float3(0.0);
    [unroll] for (uint k = 0; k < kEnvCoefficients; ++k) {
        sum += envSh[base + k].rgb * transferAt(transfer, s.transferBase, k);
    }
    if (withIndirect && s.transferCount >= 36) {
        [unroll] for (uint k = 0; k < kEnvCoefficients; ++k) {
            const uint at = s.transferBase + 9 + k * 3;
            const float3 indirect = float3(transferAt(transfer, at, 0), transferAt(transfer, at, 1),
                                           transferAt(transfer, at, 2));
            sum += envSh[base + k].rgb * indirect;
        }
    }
    return sum;
}

/// HOW MUCH OF THE SUN REACHES THIS GAUSSIAN, out of its own transfer.
///
/// The disc was taken out of the harmonics (`env_sun.slang`) and comes back
/// as a direction, so what it needs is a visibility along that direction --
/// and a transfer is exactly a visibility, projected. Evaluated at the sun it
/// gives the cosine-weighted, visibility-weighted answer; the same sum with
/// an open hemisphere's own coefficients gives what it would be with nothing
/// in the way, and the ratio of the two is the share that gets through.
///
/// Degree 2, so it is a soft shadow: a smooth falloff where a hard edge
/// belongs, and a little light where there should be none. It is
/// nevertheless the first shadow a relit cloud has ever had that knows which
/// way the light came from, and it costs no ray and no byte. A per-gaussian
/// bitmask of traced directions is what would make the edge hard, and is the
/// next thing.
///
/// Numerator and denominator share the truncation, which is what makes the
/// ratio one where nothing occludes. `T` is the degree-2 projection of
/// `V cos/pi`; with `V` one that function is the clamped cosine over pi,
/// whose own coefficients are `{1, 2/3, 1/4} Y_k(n)` -- the convolution
/// weights `environment.slang` keeps, over pi. Comparing the truncation
/// against the *exact* cosine instead reads 1.06 at the pole and 5.9 at
/// eighty-nine degrees, which is the truncation showing, not an occlusion.
///
/// Past about eighty-seven degrees the denominator is small enough to be
/// noise and the cosine below has almost closed the term anyway, so the share
/// goes to one rather than to whatever the division says.
public static const float kOverPiSun = 1.0 / 3.14159265358979;

public float splatSunShare(SplatSurface s, StructuredBuffer<uint> transfer, float3 sunDirection) {
    if (s.transferCount < 9) {
        return 1.0;
    }
    float lit = 0.0;
    float open = 0.0;
    const float3 n = s.normal;
    [unroll] for (uint k = 0; k < kEnvCoefficients; ++k) {
        const float basis = envBasisValue(k, sunDirection);
        const float band = k == 0 ? 1.0 : (k < 4 ? (2.0 / 3.0) : 0.25);
        lit += transferAt(transfer, s.transferBase, k) * basis;
        open += band * envBasisValue(k, n) * basis;
    }
    return open > 0.04 ? saturate(lit / open) : 1.0;
}

/// HOW MUCH OF THE SUN GETS THROUGH, out of the bits the bake traced.
///
/// The transfer's own estimate (`splatSunShare`) is degree 2 and can only
/// shadow softly. Where the cloud carries the sixty-four bits, the sun's
/// direction is looked up in them instead: the four cells around it on the
/// octahedral grid, weighed bilinearly, so an edge falls over one cell -- a
/// quarter of a radian -- rather than over a hemisphere. A cell below the
/// gaussian's own horizon was never traced (its bit is clear because nothing
/// was asked, not because something was in the way), so it is left out of
/// the weights rather than counted as shadow. The grid is not wrapped across
/// the octahedral seam; a sun on the seam reads the cells on its own side.
public float splatSunOpen(SplatSurface s, StructuredBuffer<uint> transfer, float3 sunDirection) {
    if (!s.hasShadowBits) {
        return splatSunShare(s, transfer, sunDirection);
    }
    const float2 at = octEncode(sunDirection) * 8.0 - 0.5;
    const float2 lo = floor(at);
    const float2 f = at - lo;
    float open = 0.0;
    float weight = 0.0;
    [unroll] for (uint k = 0; k < 4; ++k) {
        const int2 cell = clamp(int2(lo) + int2(k & 1, k >> 1), int2(0), int2(7));
        const float w = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        const float3 towards = octDecode((float2(cell) + 0.5) / 8.0);
        if (dot(towards, s.normal) <= 0.0 || w <= 0.0) {
            continue;
        }
        const uint index = uint(cell.y) * 8u + uint(cell.x);
        const uint word = index < 32u ? s.shadowBits.x : s.shadowBits.y;
        open += w * float((word >> (index & 31u)) & 1u);
        weight += w;
    }
    // No traced cell near the sun at all: it grazes this gaussian's horizon,
    // where the cosine has already closed the term.
    return weight > 0.0 ? open / weight : 1.0;
}

/// WHERE THE TRANSMITTED HALF LOOKS, WITH AN INDEX.
///
/// Snell at the gaussian's own normal, entering: the normal is already turned
/// towards the eye, so the ratio is `1 / ior`. Past the critical angle there
/// is no direction to look along and the straight one is what is left, which
/// is what this did before an index existed.
public float3 bentThrough(SplatSurface s, float3 wo) {
    const float3 bent = refract(-wo, s.normal, 1.0 / max(s.ior, 1.0));
    return dot(bent, bent) > 1.0e-8 ? normalize(bent) : -wo;
}

/// HOW MUCH OF ITS SKY A GAUSSIAN CAN SEE AT ALL, out of its own transfer.
///
/// The constant term of a transfer is the clamped cosine lobe weighed by
/// visibility: over an open hemisphere it is exactly `A0 * Y0 / pi`, which is
/// 0.28209, so the ratio is the fraction of the sky that reaches this
/// gaussian. It is an ambient occlusion nobody had to bake twice.
public float splatOpenness(SplatSurface s, StructuredBuffer<uint> transfer) {
    if (s.transferCount < 9) {
        return 1.0;
    }
    return saturate(transferAt(transfer, s.transferBase, 0) / 0.28209479177387814);
}

/// AND HOW MUCH OF IT THE REFLECTION CAN SEE.
///
/// The reflection was added with nothing in front of it: a gaussian in a
/// pocket reflected the whole sky, and the measurement said so -- the pawn's
/// marble disc, whose stem stands over it, read a tenth bright against the
/// mesh path traced beside it. What a ray would answer exactly, an occlusion
/// answers cheaply: Lagarde and de Rousiers' fit, which narrows the opening
/// as the lobe sharpens, so a mirror in a pocket goes dark and a rough one
/// hardly moves. Where a cloud carries no transfer, openness is one and this
/// is what it always was.
public float specularOpenness(float nDotV, float openness, float roughness) {
    return saturate(pow(nDotV + openness, exp2(-16.0 * roughness - 1.0)) - 1.0 + openness);
}

/// WHAT A FRAME STILL OWES A GAUSSIAN, once its body is accounted for.
///
/// A cloud that was not baked owes the whole reflection, metal and all. One
/// whose body was baked (`lit`) already carries the metal's, because a bake
/// keeps a conductor's reflection in the body it fits -- what is left is the
/// dielectric polish, and only as much of it as the surface is not metal.
public float3 splatPolish(SplatSurface s, float nDotV) {
    return s.lit ? ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) * specularTint(s) * (1.0 - s.metallic)
                 : environmentBrdfOf(s, nDotV);
}

/// What a thin wall's reflection is scaled by: the sheet reflects `2/(1+R0)`
/// of one face's Fresnel, and its gaussians were given `2R0/(1+R0)` of
/// opacity, so the blend leaves the reflection whole at `1/R0` of a face's.
float thinWallGain(SplatSurface s) {
    if (!s.thinWalled) {
        return 1.0;
    }
    const float r = (max(s.ior, 1.5) - 1.0) / (max(s.ior, 1.5) + 1.0);
    return 1.0 / max(r * r, 1.0e-3);
}

/// How much of the body is not a thin wall's transmission, which is the blend's.
float transmittedWeight(SplatSurface s) {
    return s.thinWalled ? 0.0 : s.transmission;
}

/// HOW SOFT THE ROOM IS THROUGH THE GLASS, as the roughness of a reflection.
///
/// The prepared sky's levels are reflection lobes: a microfacet tilted by
/// `theta` turns a mirror ray by `2 theta`. Through an interface it turns the
/// ray by far less -- `(1 - 1/ior) theta` going in, `(ior - 1) theta` coming
/// out, and the first of those is opened by `ior` again where the ray leaves
/// -- so a ball whose two faces are as rough as its polish shows the room
/// `sqrt(2) (ior - 1) / 2` as wide as it reflects it, a third at 1.5. Read at
/// the polish's own roughness, the pawn's head (0.23) was a haze where the
/// mesh beside it shows the workshop: relMSE over the head 0.081 to 0.048.
/// A level's width goes as the square of the roughness, so the roughness goes
/// as the root of that ratio.
///
/// Only where both faces were found. The single bend a route with no tree
/// makes is not the image a lens forms at all, and sharpening it made the
/// rasterised head worse against the mesh (0.138 to 0.211): there the blur
/// is what hides that it is the wrong picture.
public float transmittedRoughness(float roughness, float ior) {
    const float index = max(ior, 1.0);
    return roughness * sqrt(saturate(0.70710678 * (index - 1.0)));
}

/// WHAT THE TRANSMITTED HALF SEES: what a ray found behind the gaussian, the
/// sky along the direction the glass sends it where the route traced one or
/// an index bends it, and the sky's irradiance from behind -- translucency --
/// where there is no index. The same for a cloud with a transfer and one
/// without: an index is a property of the material, not of the bake, and a
/// skinned glass bird carries no transfer.
///
/// All of it leaves through the far face, so all of it is weighed by what
/// that face lets out (`exitThrough`): what a ray met behind the glass as
/// much as the sky. `behindColour` is the colour the particle it met was
/// shaded with, which knows nothing of the glass in front of it.
float3 transmittedBehind(SplatSurface s, float3 wo, uint dome, StructuredBuffer<uint> envTexels,
                         StructuredBuffer<float4> envSh, uint envBaseSide) {
    const float3 seen =
        s.hasBehind ? s.behindColour
                    : (s.hasExit ? envRadiance(envTexels, dome, s.exitDirection,
                                               transmittedRoughness(s.roughness, s.ior), envBaseSide)
                                 : (s.ior > 1.0
                                        ? envRadiance(envTexels, dome, bentThrough(s, wo), s.roughness, envBaseSide)
                                        : envIrradiance(envSh, dome, -s.normal) * (1.0 / 3.14159265358979)));
    return seen * s.exitThrough;
}

/// How much of it passes: what the polish does not reflect, and nothing on
/// the far face of a solid, whose share the near face already sent on.
float3 transmittedShare(SplatSurface s, float nDotV) {
    return s.backFacing && s.ior > 1.0
               ? float3(0.0)
               : saturate(float3(1.0) - ggxEnvDielectricAt(s.roughness, nDotV, s.lobes.specularIor) *
                                            s.lobes.specularWeight);
}

float3 relitByDome(SplatSurface s, float3 wo, LightRecord light, uint dome,
                   StructuredBuffer<uint> envTexels, StructuredBuffer<float4> envSh, uint envBaseSide,
                   StructuredBuffer<float4> envSun, StructuredBuffer<uint> transfer, bool withIndirect) {
    const float nDotV = max(dot(s.normal, wo), 1.0e-4);
    const float3 mirror = reflect(-wo, s.normal);
    const float3 body = s.lit ? float3(0.0) : s.albedo * (1.0 - s.metallic);
    // The layers over the base: what they take from what reaches it, and
    // what they give back. A baked body was baked under them already (the
    // bake's stack weighs its body by the same throughput), so it is the
    // polish and the albedo alone that pass through them here.
    const SplatLayers layers = splatLayers(s, nDotV);
    const bool coated = s.lobes.coatWeight > 0.0;
    // A dome is on both sides of a splat at once, so the translucent half
    // sees exactly what the facing half sees and the two sum to the whole
    // body: a glass ball under a sky is its own colour, not a dark shell.
    // A cloud that carries a transfer answers the body from it, and answers
    // it for any sky: what it keeps is the geometry, not the light.
    if (dome != kEnvNone && s.transferCount >= 9 && !s.lit) {
        float3 front = transferredBody(s, transfer, envSh, dome, withIndirect);
        // THE DISC THE HARMONICS NO LONGER HOLD. `env_sun` took it out of
        // them and handed over its direction and the irradiance it delivers,
        // so it is added back here as the light it is: an exact cosine, and a
        // share of it that the transfer itself estimates along that
        // direction. A solid angle of zero says this dome had no sun and
        // nothing was taken, and then there is nothing to add.
        //
        // Added to the body only. The prefiltered map kept its own sun, so the
        // polish already has it, and what the polish owes is the shadowed
        // share of it taken back out (below).
        const float4 disc = envSun[dome * 2];
        const float3 irradiance = disc.w > 0.0 ? envSun[dome * 2 + 1].rgb : float3(0.0);
        const float cosine = max(dot(s.normal, disc.xyz), 0.0);
        const float sunShare = disc.w > 0.0 ? splatSunOpen(s, transfer, disc.xyz) : 1.0;
        front += irradiance * (cosine * sunShare * kOverPiSun);
        // The translucent half faces the other way, and a transfer was
        // measured over the hemisphere the normal faces alone: what arrives
        // behind is answered by the sky's harmonics with no occlusion, which
        // is what every side of a cloud was answered with before a transfer
        // existed. Without this a glass ball that carries one is a dark
        // shell, because half its body has nowhere to take its light from.
        const float3 behind = transmittedBehind(s, wo, dome, envTexels, envSh, envBaseSide);
        // WHAT PASSES IS WHAT IS NOT REFLECTED. The reflection is added
        // below whatever the surface does, so a transmitted half at full
        // strength is more light than arrived: a glass ball came back half
        // again as bright as the mesh beside it, with a rim two and a half
        // times too hot. `lobes.slang` writes the same complement for a
        // transmitting lobe (`kScatterTransmit`), so the two agree.
        // The far face of a solid transmits nothing of its own: its share of
        // the light is what the near face already sent on.
        const float3 passes = transmittedShare(s, nDotV);
        const float3 body2 = layers.under * s.albedo * (1.0 - s.metallic) *
                             ((1.0 - s.transmission) * front + transmittedWeight(s) * passes * behind);
        // What the reflection sees: the cloud itself where a ray found it,
        // and the sky narrowed by how much of it this gaussian can see where
        // nothing was traced.
        const float3 seen = s.hasReflection
                                ? s.reflectionColour
                                : envRadiance(envTexels, dome, mirror, s.roughness, envBaseSide);
        const float narrowed =
            s.reflectionTraced ? 1.0
                               : specularOpenness(nDotV, splatOpenness(s, transfer), s.roughness);
        // THE SUN IN THE POLISH. The prefiltered map kept its disc, so what
        // the polish reads carries the sun whether or not anything stands in
        // its way -- on a rough surface a broad lobe of it, and on a metal all
        // of the sun there is, since a metal has no body to shadow. The share
        // that does not get through is taken back out as the analytic lobe of
        // a directional light of that irradiance, which is what the map's own
        // sun is where the surface is rough enough for it to matter. On a
        // mirror the analytic lobe is a spike the map does not hold, so the
        // sum is clamped rather than let go below nothing. A traced
        // reflection replaced the map and has no sun of its own to take back.
        const float3 shadowedSun = s.hasReflection
                                       ? float3(0.0)
                                       : specularLobe(s, wo, disc.xyz) * irradiance * (cosine * (1.0 - sunShare));
        // THE COAT: the same reflection at its own roughness and index, over
        // everything, narrowed by what this gaussian sees of the sky at that
        // roughness, and with its own share of a shadowed sun taken back out.
        float3 coat = float3(0.0);
        if (coated) {
            const float3 coatSeen = s.hasReflection
                                        ? s.reflectionColour
                                        : envRadiance(envTexels, dome, mirror, s.lobes.coatRoughness, envBaseSide);
            const float coatNarrowed =
                s.reflectionTraced ? 1.0
                                   : specularOpenness(nDotV, splatOpenness(s, transfer), s.lobes.coatRoughness);
            const float3 coatSun = s.hasReflection
                                       ? float3(0.0)
                                       : coatLobe(s, wo, disc.xyz) * irradiance * (cosine * (1.0 - sunShare));
            coat = max(layers.coat * coatNarrowed * coatSeen - coatNarrowed * coatSun, float3(0.0));
        }
        // THE SHEEN: a lobe as broad as a body's, so it is answered as one --
        // its albedo at this eye times the light the transfer says reaches
        // the gaussian, shadow and sun included -- under the coat.
        const float3 sheen = layers.belowCoat * layers.sheen * front;
        return body2 +
               layers.under * thinWallGain(s) *
                   max(splatPolish(s, nDotV) * narrowed * seen - narrowed * shadowedSun, float3(0.0)) +
               coat + sheen;
    }
    float3 above, behind, reflected;
    if (dome == kEnvNone) {
        above = lightHit(light, s.position, s.normal).radiance;
        behind = lightHit(light, s.position, -s.normal).radiance;
        reflected = lightHit(light, s.position, mirror).radiance;
    } else {
        // Irradiance over pi is the radiance a Lambert body sends back, which
        // is what the constant dome above hands over directly.
        const float kOverPi = 1.0 / 3.14159265358979;
        above = envIrradiance(envSh, dome, s.normal) * kOverPi;
        behind = envIrradiance(envSh, dome, -s.normal) * kOverPi;
        // The disc `env_sun` took out of the harmonics, given back unshadowed:
        // a cloud with no transfer has nothing to shadow it by.
        const float4 disc = envSun[dome * 2];
        if (disc.w > 0.0) {
            const float3 irradiance = envSun[dome * 2 + 1].rgb;
            const float cosine = dot(s.normal, disc.xyz);
            above += irradiance * (max(cosine, 0.0) * kOverPi);
            behind += irradiance * (max(-cosine, 0.0) * kOverPi);
        }
        reflected = envRadiance(envTexels, dome, mirror, s.roughness, envBaseSide);
    }
    // A prepared sky gives the transmitted half the same answer a transfer
    // cloud gets: what stands behind, bent by the index. The complement is
    // taken only there, where the polish is the prepared sky's too; a
    // constant dome, and translucency with no index, keep what they did.
    if (dome != kEnvNone && (s.hasBehind || s.hasExit || s.ior > 1.0)) {
        behind = transmittedBehind(s, wo, dome, envTexels, envSh, envBaseSide) * transmittedShare(s, nDotV);
    }
    // The coat reflects along the same mirror, read at its own roughness.
    float3 coat = float3(0.0);
    if (coated) {
        const float3 coatSeen = s.hasReflection     ? s.reflectionColour
                                : dome == kEnvNone ? reflected
                                                   : envRadiance(envTexels, dome, mirror, s.lobes.coatRoughness,
                                                                 envBaseSide);
        coat = layers.coat * coatSeen;
    }
    return layers.under * body * ((1.0 - s.transmission) * above + transmittedWeight(s) * behind) +
           layers.under * thinWallGain(s) * splatPolish(s, nDotV) *
               (s.hasReflection ? s.reflectionColour : reflected) +
           coat + layers.belowCoat * layers.sheen * above;
}

/// The light `light` puts on `s`, seen from `wo`, with `through` of it
/// arriving.
///
/// Three terms: the body of the material facing the light, the body of it
/// facing away -- which is the translucent half, and is why a light behind a
/// glass splat lights it instead of leaving it black -- and the reflection,
/// which is on the lit side only.
public float3 relitByOne(SplatSurface s, float3 wo, LightSample ls, float3 through) {
    if (ls.pdf <= 0.0) {
        return float3(0.0);
    }
    const float nDotL = dot(s.normal, ls.wi);
    const float front = max(nDotL, 0.0);
    const float back = -nDotL;   // negative in front of the surface; the wrap below clamps it
    // Nothing to gather where the body is already lit: what was baked is the
    // light on it, and lighting it again would be that light twice.
    const float3 body = s.lit ? float3(0.0) : s.albedo * (1.0 - s.metallic) * (1.0 / 3.14159265358979);
    // Translucency is not transparency. What comes through a translucent
    // material has been scattered on the way, so it does not stop at the
    // terminator the way a reflection does and it does not draw what stands
    // behind: it wraps around the body and leaves softened. `kWrap` is how
    // far round it reaches -- half a hemisphere, which is what reads as jade
    // or wax rather than as a window.
    const float kWrap = 0.5;
    const float wrapped = saturate((back + kWrap) / (1.0 + kWrap));
    // Under the coat and the sheen, which take their share at the eye and
    // add their own lobes on the lit side.
    const SplatLayers layers = splatLayers(s, max(dot(s.normal, wo), 1.0e-4));
    const float3 diffuse = layers.under * body * ((1.0 - s.transmission) * front + transmittedWeight(s) * wrapped);
    const float3 specular = layers.under * specularLobe(s, wo, ls.wi) * front * thinWallGain(s) +
                            (coatLobe(s, wo, ls.wi) + layers.belowCoat * sheenLobe(s, wo, ls.wi)) * front;
    return through * (diffuse + specular) * ls.radiance / ls.pdf;
}

/// Every light this splat is linked to, one sample each at its centre.
///
/// `shadowLights` of 0 means nothing measured a shadow, and each light is
/// taken whole; otherwise `shadowFactors[shadowBase + k]` is what reaches the
/// splat from light k (splat_shadow.slang).
public float3 relitSplat(SplatSurface s, float3 wo, StructuredBuffer<LightRecord> lights, uint lightCount,
                         uint categoriesLo, uint categoriesHi, StructuredBuffer<float> shadowFactors,
                         uint shadowBase, uint shadowLights, StructuredBuffer<uint> envTexels,
                         StructuredBuffer<float4> envSh, StructuredBuffer<uint> envOfLight,
                         uint envLights, uint envBaseSide, StructuredBuffer<float4> envSun,
                         StructuredBuffer<uint> transfer, bool withIndirect) {
    // A baked body starts from the light it already carries -- its emission
    // in it; an albedo starts from what it gives off by itself and gathers.
    float3 lit = s.lit ? s.albedo : s.emission;
    for (uint k = 0; k < lightCount; ++k) {
        const LightRecord light = lights[k];
        // Light linking reaches a cloud as it reaches a mesh: by one bit.
        if (!lightLinked(light.lightCategory, categoriesLo, categoriesHi)) {
            continue;
        }
        const float through = k < shadowLights ? shadowFactors[shadowBase + k] : 1.0;
        if (light.kind == kLightDome) {
            // A dome is everywhere and nowhere: no distance to fade over.
            const uint dome = k < envLights ? envOfLight[k] : kEnvNone;
            lit += shadowTint(light, through, 0.0) *
                   relitByDome(s, wo, light, dome, envTexels, envSh, envBaseSide, envSun, transfer,
                               withIndirect);
            continue;
        }
        const LightSample ls = sampleLight(light, s.position, s.normal, float2(0.5, 0.5));
        if (!ls.valid) {
            continue;
        }
        lit += relitByOne(s, wo, ls, shadowTint(light, through, ls.distance));
    }
    return lit;
}

/// The rotation a packed quaternion stands for, columns in the frame's order.
public float3x3 splatRotation(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

/// The normal a splat stands for: its shortest axis, turned towards the eye.
/// `rotation` is the frame's three columns, `scale` its three sizes.
/// Which side of a gaussian the eye is on, before the normal is turned
/// towards it: positive where the eye sees the face the surface faced.
public float splatFacing(float3x3 rotation, float3 scale, float3 position, float3 eye) {
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    const float3 normal = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    return dot(normal, normalize(eye - position));
}

public float3 splatNormal(float3x3 rotation, float3 scale, float3 position, float3 eye) {
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    float3 normal = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    if (dot(normal, normalize(eye - position)) < 0.0) {
        normal = -normal;
    }
    return normal;
}

/// THE NORMAL A RELIT SPLAT IS SHADED WITH. Where the cloud keeps a shading
/// normal of its own (`stored`, packed, `hasStored`) it is that one: what a
/// mesh's normal map made of the surface, which the disc's short axis -- the
/// face's own normal -- never knew. A splat is two-sided, so the disc's axis
/// is turned towards the eye first (`splatNormal`), and the stored normal is
/// put on the same side of the disc: the eye on the far side of a surface sees
/// its relief from the far side. Everything geometric -- the footprint, where
/// a ray meets the disc, the plane a reflection clears -- keeps the disc's.
public float3 splatShadingNormal(float3x3 rotation, float3 scale, float3 position, float3 eye, bool hasStored,
                                 uint stored) {
    const float3 disc = splatNormal(rotation, scale, position, eye);
    if (!hasStored) {
        return disc;
    }
    const float3 n = unpackNormal(stored);
    return dot(n, disc) < 0.0 ? -n : n;
}

/// A point of the cloud in the world, by the rows a caller passes.
public float3 relightToWorld(float4 row0, float4 row1, float4 row2, float3 p) {
    return float3(dot(row0, float4(p, 1.0)), dot(row1, float4(p, 1.0)), dot(row2, float4(p, 1.0)));
}

/// A direction of the cloud in the world -- a ray, the way out of a glass --
/// by the same rows, made unit. A normal is not a direction under a scale
/// that is not uniform: relightNormalToWorld.
public float3 relightDirectionToWorld(float4 row0, float4 row1, float4 row2, float3 v) {
    const float3 turned = float3(dot(row0.xyz, v), dot(row1.xyz, v), dot(row2.xyz, v));
    return dot(turned, turned) > 1.0e-20 ? normalize(turned) : float3(0.0, 0.0, 1.0);
}

/// A NORMAL of the cloud in the world, by the same rows: not as a direction
/// but by the inverse transpose, which keeps it perpendicular to the surface
/// under any linear map. Up to a scale that is the cofactor matrix, whose
/// columns are `b x c`, `c x a` and `a x b` for the columns `a`, `b`, `c` of
/// the linear part -- what the skinner turns a normal with too. Taken as a
/// direction, a card scaled (2, 1, 1) with its normal tilted 45 degrees in
/// x tilted to 63 instead of 27. A mirror flips the cofactor against the
/// inverse transpose, so the determinant's sign is put back: the normal stays
/// on the side of the surface it was on, the eye's.
public float3 relightNormalToWorld(float4 row0, float4 row1, float4 row2, float3 n) {
    const float3 a = float3(row0.x, row1.x, row2.x);
    const float3 b = float3(row0.y, row1.y, row2.y);
    const float3 c = float3(row0.z, row1.z, row2.z);
    const float3 bc = cross(b, c);
    const float side = dot(a, bc) < 0.0 ? -1.0 : 1.0;
    const float3 turned = side * (n.x * bc + n.y * cross(c, a) + n.z * cross(a, b));
    return dot(turned, turned) > 1.0e-20 ? normalize(turned) : float3(0.0, 0.0, 1.0);
}
