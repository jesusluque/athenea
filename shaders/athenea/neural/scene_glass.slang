// Copyright (c) 2026 jesus luque.
//
// WHAT A GLASS GAUSSIAN SHOWS, FROM WHERE THE EYE STANDS.
//
// A gaussian a look made of solid glass keeps the picture that look saw in
// it, and a picture is one eye's: from anywhere else the glass shows that
// eye's room. What is the glass's own does not depend on the eye -- how much
// it transmits, its index, what a crossing tints, the way its surface faces
// -- and a found scene keeps those a gaussian (scene_render.slang's record).
// What the glass shows is then worked out a view: the eye's ray is reflected
// at the gaussian and bent into it, walked to the far face of the same glass
// and bent out (rt/rt_glass.slang's `glassExit`), and each of the two rays
// reads the first gaussian of the scene it meets that is not glass
// (`glassNearest`). The scene is the room: what stands behind the glass and
// what it mirrors are the gaussians the looks found there.
//
//   seen = F reflected + (1 - F) (1 - F') tint^2 behind
//
// with F the Fresnel share at the near face, F' at the far one, and the tint
// taken once a crossing. A ray that meets nothing keeps the picture the look
// left, which is the best there is to say of a room nobody has looked at.
//
// One ray each way a gaussian: a polished glass. The tree is the one the
// gaussian ray tracer builds over the cloud (GaussianRayTracer::treeOf), and
// the cloud is the whole scene cut for the view, glass and room together.
module scene_glass;
import athenea.common.packing;
import athenea.common.color;
import athenea.scene.splat_encoding;
import athenea.material.microfacet;
import athenea.rt.rt_glass;

struct GlassParams {
    uint  count;        // gaussians in the cloud (glassSeen) or in the piece (glassGather)
    uint  base;         // glassGather: where the piece's begin in the cloud
    uint  nodeBase;     // the tree's first node
    uint  leafBase;     // the cloud's first particle in the tree's frames and leaves
    float eyeX; float eyeY; float eyeZ;
    uint  linearCloud;  // the cloud's colours are light already
};

StructuredBuffer<float4>   positions;
RWStructuredBuffer<uint>   shape;        // four words a gaussian; the colour of a glass one is written
StructuredBuffer<uint>     pbr;          // metallic, roughness, transmission, as GpuSplats keeps them
StructuredBuffer<uint>     normals;      // the shading normal, packed
StructuredBuffer<float4>   matter;       // what one crossing tints (rgb) and the index (w; 0: bends nothing)
StructuredBuffer<float4>   frames;
StructuredBuffer<float4>   bvhBoxes;
StructuredBuffer<uint>     bvhChildren;
StructuredBuffer<uint>     bvhLeaves;
RWStructuredBuffer<uint>   counters;     // glass gaussians, rays behind that met the room, rays mirrored that did
// glassGather
StructuredBuffer<uint>     flags;
StructuredBuffer<uint>     places;
RWStructuredBuffer<uint>   pbrOut;
RWStructuredBuffer<uint>   normalsOut;
RWStructuredBuffer<float4> matterOut;
ConstantBuffer<GlassParams> params;

float3x3 rotationOf(float4 q) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;   // as packing.slang's decodeQuaternion hands it
    return float3x3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
                    2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
                    2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y));
}

/// The light a gaussian of the cloud carries: its base colour as the cloud keeps it.
float3 lightOf(uint particle) {
    const uint row = particle * 4u;
    const float3 base = float3(highHalf(shape[row + 2u]), lowHalf(shape[row + 3u]), highHalf(shape[row + 3u]));
    return cloudLight(base, params.linearCloud != 0u);
}

/// What a cut keeps of a gaussian beside what is drawn: copied to its place among the drawn.
[shader("compute")]
[numthreads(64, 1, 1)]
void glassGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || flags[i] == 0u) {
        return;
    }
    const uint to = params.base + places[i];
    pbrOut[to] = pbr[i];
    normalsOut[to] = normals[i];
    matterOut[to] = matter[i];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void glassSeen(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const float4 made = matter[i];
    const float index = made.w;
    // A sheet bends nothing and is blended as it was made; what is not glass is not asked.
    if (pbrTransmission(pbr[i]) < 0.5 || index <= 1.0 || positions[i].w <= 0.0) {
        return;
    }
    InterlockedAdd(counters[0], 1u);
    const uint row = i * 4u;
    const float3 p = positions[i].xyz;
    const float3 eye = float3(params.eyeX, params.eyeY, params.eyeZ);
    const float3 wo = normalize(eye - p);
    const float3 scale = exp(float3(lowHalf(shape[row + 1u]), highHalf(shape[row + 1u]), lowHalf(shape[row + 2u])));
    const float3x3 rotation = rotationOf(decodeQuaternion(shape[row]));
    // The disc's own axis, turned to the eye, and the surface's normal on that side of it.
    const uint shortest = scale.x <= scale.y ? (scale.x <= scale.z ? 0u : 2u) : (scale.y <= scale.z ? 1u : 2u);
    float3 disc = normalize(float3(rotation[0][shortest], rotation[1][shortest], rotation[2][shortest]));
    if (dot(disc, wo) < 0.0) {
        disc = -disc;
    }
    float3 n = unpackNormal(normals[i]);
    if (!(dot(n, n) > 0.25)) {
        n = disc;
    }
    n = normalize(n);
    if (dot(n, disc) < 0.0) {
        n = -n;
    }
    if (dot(n, wo) < 1.0e-3) {
        n = disc;   // a relief that faces away from the eye bends nothing sensibly: the disc's own
    }
    const float cosine = saturate(dot(n, wo));
    const float mirrored = fresnelDielectric(cosine, index);
    const float3 own = lightOf(i);
    const float widest = max(scale.x, max(scale.y, scale.z));
    const float step = max(length(p - eye), 1.0) * 1.0e-4;

    // WHAT IT MIRRORS: the first gaussian along the mirror direction that is not glass, standing clear of
    // this gaussian's own plane.
    float3 reflected = own;
    {
        const float3 along = reflect(-wo, n);
        uint hit = 0u;
        const GlassExit met = glassNearest(frames, bvhBoxes, bvhChildren, bvhLeaves, params.nodeBase, params.leafBase,
                                           params.count, params.leafBase, p + along * step, along, 0.0, 3.0e38, 0.5,
                                           p, disc, 4.0 * widest, hit);
        if (met.found) {
            reflected = lightOf(hit);
            InterlockedAdd(counters[2], 1u);
        }
    }

    // WHAT STANDS BEHIND IT: bent in at this face, walked to the far face of the glass, bent out there, and
    // on to the first gaussian that is not glass. Past the critical angle at the far face the ray turns back
    // into the glass and is followed as it turned, as the engine's own glass does.
    float3 behind = own;
    float3 through = made.rgb;
    {
        float3 from = p;
        float3 along = -wo;
        const float3 inside = refract(-wo, n, 1.0 / index);
        if (dot(inside, inside) > 1.0e-8) {
            const float3 d1 = normalize(inside);
            const GlassExit exit = glassExit(frames, bvhBoxes, bvhChildren, bvhLeaves, params.nodeBase,
                                             params.leafBase, params.count, params.leafBase, p + d1 * step, d1, 0.0,
                                             3.0e38, 0.5);
            if (exit.found) {
                const float3 back = -exit.normal;
                const float3 outside = refract(d1, back, index);
                const bool leaves = dot(outside, outside) > 1.0e-8;
                along = leaves ? normalize(outside) : reflect(d1, back);
                from = exit.position;
                const float kept =
                    leaves ? 1.0 - fresnelDielectric(saturate(dot(d1, exit.normal)), 1.0 / index) : 1.0;
                through = kept * made.rgb * made.rgb;
            } else {
                along = d1;
            }
        }
        uint hit = 0u;
        const float reach = max(length(from - eye), 1.0) * 1.0e-4;
        const GlassExit met = glassNearest(frames, bvhBoxes, bvhChildren, bvhLeaves, params.nodeBase, params.leafBase,
                                           params.count, params.leafBase, from + along * reach, along, 0.0, 3.0e38,
                                           0.5, float3(0.0), float3(0.0), -1.0, hit);
        if (met.found) {
            behind = lightOf(hit);
            InterlockedAdd(counters[1], 1u);
        } else {
            through = float3(1.0);   // the picture the look left has its crossings in it already
        }
    }
    const float3 seen = mirrored * reflected + (1.0 - mirrored) * through * behind;
    const float3 kept = params.linearCloud != 0u ? seen : linearToSrgb(seen);
    shape[row + 2u] = packHalves(lowHalf(shape[row + 2u]), kept.r);
    shape[row + 3u] = packHalves(kept.g, kept.b);
}
