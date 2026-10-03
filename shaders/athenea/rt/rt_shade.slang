// Copyright (c) 2026 jesus luque.
//
// One instance's particle colours for this frame, into the colour buffer at
// `base`, and the opacity its SplatEdit leaves each particle (w). A particle's
// colour depends on the direction from the eye to its centre (the rasteriser's
// rule, and how 3DGS trained it), not on the ray, so it is evaluated once a
// frame here instead of once for every ray it reaches.
//
// A relit cloud is relit here, for the same reason and at the same moment: one
// answer a particle a frame. Until this it was relit only in the tile
// rasteriser, so `AtheneaSplatLightingAPI` meant one thing when a frame was
// rasterised and nothing at all when it was traced -- a converted mesh came
// back as its own albedo, unlit, which for a white material is 1.0 and for
// black marble is 0.003. What relighting is lives in splat_relight.slang, and
// both routes read it from there.
module rt_shade;
import athenea.common.packing;
import athenea.common.sh;
import athenea.scene.splat_encoding;
import athenea.splat.splat_relight;
import rt_glass;
import athenea.common.color;
import athenea.common.splat_override;
import rt_common;

StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     sh;
StructuredBuffer<uint>     pbr;      // metallic and roughness a splat, where the cloud carries them
// The shading normal a splat, where the cloud carries one (`params.hasNormals`).
StructuredBuffer<uint>     normals;
StructuredBuffer<LightRecord> lights;
StructuredBuffer<float>    shadowFactors;   // measured before the frame, where a cloud carries a baked visibility
// The sky prepared for a kernel with no texture table; `params.envLights` of
// 0 says there is nothing to read (athenea/light/environment.slang).
StructuredBuffer<uint>     envTexels;
StructuredBuffer<float4>   envSh;
// The sun `env_sun` took out of that sky: two float4 a dome.
StructuredBuffer<float4>    envSun;
StructuredBuffer<uint>     envOfLight;
StructuredBuffer<uint>     transfer;
// Which ways out of each gaussian are open, two words a gaussian; bound
// whether or not `params.shadowBits` says to read them.
StructuredBuffer<uint>      shadowBits;
// WHICH PRIM EACH GAUSSIAN CAME FROM, and what a frame says that prim is made
// of: the id the matte writes is the key of a table a pick can edit
// (athenea/common/splat_override.slang). `params.overrideCount` of 0: no table.
StructuredBuffer<uint>     cloudCrypto;
StructuredBuffer<float4>   splatOverrides;
// The cloud's own tree, for the one ray a glass gaussian casts: the far face
// of the object it stands on (rt_glass.slang). `params.glassParticles` of 0
// says there is none to walk, and then glass bends once and no more.
StructuredBuffer<float4>   frames;
StructuredBuffer<float4>   bvhBoxes;
StructuredBuffer<uint>     bvhChildren;
StructuredBuffer<uint>     bvhLeaves;
RWStructuredBuffer<float4> colours;
ConstantBuffer<RtParams>   params;   // count, base, shWords, restPerColour, shLimit; v03/v13/v23: eye in cloud space

[shader("compute")]
[numthreads(256, 1, 1)]
void rtShade(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint row = i * 4;
    const float4 p = positions[i];
    const float3 scale = exp(float3(lowHalf(shape[row + 1]), highHalf(shape[row + 1]), lowHalf(shape[row + 2])));
    // The instance's SplatEdit decides the opacity this instance draws the
    // particle with, carried in w: zero is a particle the edit removed.
    const EditFate fate = editFate(params.edit, p.xyz, p.w, scale);
    if (fate.removed) {
        colours[params.base + i] = float4(0.0);
        return;
    }
    const float3 base = float3(highHalf(shape[row + 2]), lowHalf(shape[row + 3]), highHalf(shape[row + 3]));
    const uint degree = min(degreeOf(params.restPerColour), params.shLimit);
    const float3 eye = float3(params.v03, params.v13, params.v23);
    const float3 dir = normalize(p.xyz - eye);
    float3 colour = max(base + evaluateRest(sh, i * params.shWords, degree, dir), 0.0);
    if (params.relight != 0) {
        // In the world, where the lights are: the cloud's own space is where
        // its harmonics were trained, and a light stands somewhere else.
        const float4 w0 = float4(params.w00, params.w01, params.w02, params.w03);
        const float4 w1 = float4(params.w10, params.w11, params.w12, params.w13);
        const float4 w2 = float4(params.w20, params.w21, params.w22, params.w23);
        const float3x3 rotation = splatRotation(decodeQuaternion(shape[row + 0]));
        // The disc's axis for what is geometric -- the plane a reflection must
        // clear -- and the shading normal for what the light does.
        const float3 discNormal = splatNormal(rotation, scale, p.xyz, eye);
        const float3 cloudNormal = splatShadingNormal(rotation, scale, p.xyz, eye, params.hasNormals != 0,
                                                      params.hasNormals != 0 ? normals[i] : 0u);
        const float3 eyeWorld = float3(params.eyeWorldX, params.eyeWorldY, params.eyeWorldZ);
        SplatSurface surface;
        surface.position = relightToWorld(w0, w1, w2, p.xyz);
        surface.normal = relightDirectionToWorld(w0, w1, w2, cloudNormal);
        // What the cloud shows in this direction, decoded: the stored colours
        // are in the space a cloud is blended in and an albedo is light. The
        // harmonics are part of it -- a baked cloud keeps the body's own
        // change with direction in them -- so it is the evaluated colour and
        // not the constant term alone.
        surface.albedo = params.linearise != 0 ? srgbToLinear(colour) : max(colour, 0.0);
        surface.metallic = params.hasPbr != 0 ? pbrMetallic(pbr[i]) : 0.0;
        surface.roughness = params.hasPbr != 0 ? pbrRoughness(pbr[i]) : 1.0;
        surface.transmission = params.hasPbr != 0 ? pbrTransmission(pbr[i]) : 0.0;
        surface.thinWalled = params.hasPbr != 0 && pbrThinWalled(pbr[i]);
        // WHAT A PICK MAY HAVE SAID ABOUT THIS PRIM, over what the file did.
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
        surface.hasExit = false;
        surface.exitDirection = float3(0.0, 0.0, 1.0);
        surface.backFacing = splatFacing(rotation, scale, p.xyz, eye) < 0.0;
        surface.hasBehind = false;
        surface.behindColour = float3(0.0);
        surface.hasReflection = false;
        surface.reflectionColour = float3(0.0);
        surface.reflectionTraced = false;
        // WHAT THE MIRROR DIRECTION MEETS, once everything else is shaded.
        //
        // A gold collar under a glass ball reflects the ball, and a prepared
        // sky cannot know that. One ray a gaussian, in the second pass, so
        // what it meets has a colour already.
        // Not from the far side of an opaque object: a back-facing gaussian
        // has its normal turned to the eye, so its mirror ray points into the
        // object and meets the inside of the far wall, which no one can see.
        // What little of it leaks through the front shell reflected gold at
        // the ball's own interior.
        if (params.glassPass != 0 && params.reflectCloud != 0 && params.glassParticles > 0 &&
            surface.transmission <= 0.0 && !surface.backFacing) {
            const float3 eyeCloudR = float3(params.v03, params.v13, params.v23);
            const float3 woCloud = normalize(eyeCloudR - p.xyz);
            const float3 mirrorCloud = reflect(-woCloud, cloudNormal);
            uint hitParticle = 0;
            const float step = max(length(p.xyz), 1.0) * 1.0e-3;
            // Four footprints above this gaussian's own plane before a hit
            // counts: the cloud's own shell, however far along the ray, never
            // gets that high; a neighbour it can reflect always does.
            const float clearance = 4.0 * max(scale.x, max(scale.y, scale.z));
            const GlassExit met =
                glassNearest(frames, bvhBoxes, bvhChildren, bvhLeaves, params.glassNodeBase,
                             params.glassBase, params.glassParticles, params.glassBase,
                             p.xyz + mirrorCloud * step, mirrorCloud, 0.0, 3.0e38, 2.0,
                             p.xyz, discNormal, clearance, hitParticle);
            surface.reflectionTraced = true;
            if (met.found) {
                const float4 seen = colours[params.base + hitParticle];
                if (seen.w > 0.0) {
                    surface.hasReflection = true;
                    surface.reflectionColour =
                        params.linearise != 0 ? srgbToLinear(seen.xyz) : max(seen.xyz, 0.0);
                }
            }
        }
        // WHERE THE GLASS SENDS THE RAY, when there is a tree to ask.
        //
        // In the cloud's own space, which is where its tree stands: bend
        // going in at this gaussian's normal, walk to the far face, bend
        // coming out at that one's. Past the critical angle there is no way
        // out and the ray turns back in, which is what makes the rim of a
        // ball a mirror.
        // A thin wall bends nothing and sends nothing on: its blend does.
        if (params.glassParticles > 0 && params.ior > 1.0 && surface.transmission > 0.0 && !surface.thinWalled) {
            const float3 eyeCloud = float3(params.v03, params.v13, params.v23);
            const float3 woCloud = normalize(eyeCloud - p.xyz);
            const float3 inside = refract(-woCloud, cloudNormal, 1.0 / params.ior);
            if (dot(inside, inside) > 1.0e-8) {
                const float3 d1 = normalize(inside);
                const float reach = max(length(p.xyz - eyeCloud), 1.0) * 1.0e-4;
                const GlassExit exit = glassExit(frames, bvhBoxes, bvhChildren, bvhLeaves,
                                                 params.glassNodeBase, params.glassBase,
                                                 params.glassParticles, params.glassBase,
                                                 p.xyz + d1 * reach, d1, 0.0, 3.0e38, 0.5);
                if (exit.found) {
                    // The normal a refraction wants is the one on the side
                    // the ray comes from, and coming out that side is the
                    // glass: the far face's own normal points the other way.
                    const float3 back = -exit.normal;
                    const float3 outside = refract(d1, back, params.ior);
                    const float3 d2 =
                        dot(outside, outside) > 1.0e-8 ? normalize(outside) : reflect(d1, back);
                    surface.hasExit = true;
                    surface.exitDirection = normalize(relightDirectionToWorld(w0, w1, w2, d2));
                    // WHAT STANDS BEHIND THE GLASS, when the pass before this
                    // one has already shaded it. A ball shows the room, and
                    // this cloud's room is its own particles: the collar under
                    // a chess pawn's glass, which no sky can stand in for.
                    if (params.glassPass != 0) {
                        uint hitParticle = 0;
                        const float step = max(length(exit.position), 1.0) * 1.0e-4;
                        const GlassExit behind =
                            glassNearest(frames, bvhBoxes, bvhChildren, bvhLeaves, params.glassNodeBase,
                                         params.glassBase, params.glassParticles, params.glassBase,
                                         exit.position + d2 * step, d2, 0.0, 3.0e38, 0.5,
                                         float3(0.0), float3(0.0), -1.0, hitParticle);
                        if (behind.found) {
                            const float4 met = colours[params.base + hitParticle];
                            if (met.w > 0.0) {
                                surface.hasBehind = true;
                                surface.behindColour =
                                    params.linearise != 0 ? srgbToLinear(met.xyz) : max(met.xyz, 0.0);
                            }
                        }
                    }
                }
            }
        }
        surface.lit = params.litBody != 0;
        surface.transferCount = params.transferCount;
        surface.transferBase = i * params.transferWords * 2;
        surface.hasShadowBits = params.shadowBits != 0 && params.transferCount >= 9;
        surface.shadowBits = surface.hasShadowBits ? uint2(shadowBits[i * 2], shadowBits[i * 2 + 1]) : uint2(0);
        // A traced frame has no shadow pass of its own, but a cloud that
        // carries a baked visibility had its factors measured before the
        // frame, and they are read here exactly as the rasteriser reads its.
        colour = relitForBlend(relitSplat(surface, normalize(eyeWorld - surface.position), lights,
                                          params.lightCount, params.categoriesLo, params.categoriesHi,
                                          shadowFactors, (params.shadowBase + i) * params.shadowLights,
                                          params.shadowLights, envTexels, envSh, envOfLight,
                                          params.envLights, params.envBaseSide, envSun, transfer,
                                          params.transferIndirect != 0),
                               params.linearise != 0);
    }
    if (fate.graded) {
        colour = editGrade(params.edit, colour);
    }
    colours[params.base + i] = float4(colour, fate.opacity);
}
