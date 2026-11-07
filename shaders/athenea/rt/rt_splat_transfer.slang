// Copyright (c) 2026 jesus luque.
//
// A TX TRANSFER BAKED ON A CLOUD THAT HAS NO MESH (`athenea splat-transfer`).
//
// mesh2splat bakes a transfer by path tracing a mesh: the paths start on its
// surface and bounce off its triangles. A cloud that was never a mesh -- the
// surfels splat2mesh places on the level set of a capture's opacity -- has no
// triangle to start from or bounce off, and the path tracer meets a cloud only
// as a shadow (PathTracer.cpp's kSplats). So this bake traces the cloud
// itself, with the shadow query's own tables and structure (rt_shadow.slang:
// each particle at its peak along the ray), and computes what the path
// tracer's transfer mode computes, by the same quadrature over the octahedral
// cells it uses when it keeps cells (PathTracer.cpp, "the direct half by
// quadrature"):
//
//   the direct half: sum over the grid's directions w of T(w) cos / pi times
//     the cell's solid angle (4 / side^2 |w|_1^3) times Y_k(w), T the
//     transmittance through the cloud from the gaussian outwards;
//   the open directions: a bit a cell of 16 x 16 (or 32 x 32), open where the
//     mean transmittance of the rays in it is over a half;
//   the indirect half: the same quadrature of what the rays meet -- the
//     particles in front order, each weighed alpha times what is left in front
//     of it -- each its albedo times its own direct half: one bounce off the
//     cloud, rgb a coefficient;
//   the reflected field: over the hemisphere the gaussian faces, the radiance
//     arriving from what the rays meet under a white sky (the met particle's
//     albedo times its direct constant term times sqrt(4 pi)), projected onto
//     the degree-3 basis with the cell's solid angle, and then -- as
//     mesh2splat's bake does (m2sFieldOverClosed) -- divided by the closed
//     directions' own projection and projected again, so a closed direction
//     reads what stands there.
//
// A SOLID GLASS (`ior` over 1) is baked as the path tracer bakes one in cells
// mode (PathTracer.cpp, `bakeSphereDirection`, "behind, the way in bends"):
// its cells behind it are closed by its own far face, and its field is over
// the whole sphere, a direction behind its surface bent in by the index and
// sent on through the glass -- past its own particles -- carrying what comes
// through under a white sky: the glass's throughput (the Fresnel at both faces
// and its tint twice, as the mesh tints at each crossing) times what the ray
// meets beyond, and the sky (radiance one) where it meets nothing. That is
// what the raster's lens reads along the eye's ray through it (splat_relight,
// "a solid glass is a lens"). The far face's own bend is not there.
//
// The basis is athenea's environment basis (light/environment.slang), in the
// cloud's own frame. Two passes: `splatTransferDirect` over every cloud of the
// stage (a met particle's direct half is what the second pass reads), then
// `splatTransferBounced` over the cloud being baked.
module rt_splat_transfer;
import rt_common;
import rt_shadow;
import athenea.common.octahedral;
import athenea.common.packing;
import athenea.light.environment;
import athenea.usd.transfer_field_fill;

public struct SplatTransferParams {
    public uint  count;       // the cloud's gaussians
    public uint  base;        // its first particle's id in the tables
    public uint  grid;        // rays a side of the octahedral grid (a multiple of cellSide)
    public uint  cellSide;    // 16 or 32; 0 writes no bits
    public uint  hasNormals;  // 1: the cloud's shading normals; 0: each gaussian's thinnest axis
    public uint  linear;      // 1: the colours are linear light already
    public uint  writeBits;   // 1: this cloud is the one baked (its bits are written)
    public uint  hasOrigin;   // 1: `origin` says each gaussian's record in the file (some were dropped)
    public float offset;      // the ray starts this many of the gaussian's widest sigma off its surface
    public float cut;         // a ray stops once less than this is left through it
    public float ior;         // over 1: a solid glass (below)
    public uint  hasCurvature; // 1: `curvatureIn` holds the cloud's shape operators, three floats a gaussian
};

// The cloud's tables, as the shadow query reads them (render::ShadowScene).
StructuredBuffer<float4>   frames;
StructuredBuffer<float4>   colours;
StructuredBuffer<float4>   instanceData;
StructuredBuffer<uint>     instanceIndices;
uniform RaytracingAccelerationStructure scene;

// The cloud being passed over (scene::GpuSplats).
StructuredBuffer<float4>   positions;
StructuredBuffer<uint>     shape;
StructuredBuffer<uint>     normals;
StructuredBuffer<uint>     origin;      // each gaussian's record in the file (scene::GpuSplats::origin)

// Every particle of the stage, by id: its direct half (16) and its albedo.
RWStructuredBuffer<float>  direct;
RWStructuredBuffer<float4> albedo;
// The cloud baked, by its record in the file (a record dropped on loading
// keeps zeros).
RWStructuredBuffer<float>  directOut;   // 16
RWStructuredBuffer<uint>   bits;
RWStructuredBuffer<float>  bounced;     // 48: rgb a coefficient
RWStructuredBuffer<float>  reflected;   // 48
RWStructuredBuffer<float>  lensCurvature; // 3: the shape operator a frame's lens reads (below)
RWStructuredBuffer<uint>   counters;    // [gaussians baked, rays traced, cells open, of them on the normal's side,
                                        //  glass taken as a ball, as a slab]
StructuredBuffer<float>    curvatureIn; // the cloud's shape operators, where `params.hasCurvature`

ConstantBuffer<SplatTransferParams> params;

static const float kPi = 3.14159265358979;
static const float kSqrtFourPi = 3.5449077018110318;
static const uint  kNearest = 8;

uint recordOf(uint i) {
    return params.hasOrigin != 0 ? origin[i] : i;
}

/// The dielectric's Fresnel reflectance, unpolarised, at cos theta in a medium
/// of relative index eta.
float fresnelDielectricOf(float cosI, float eta) {
    const float sinT2 = (1.0 - cosI * cosI) / (eta * eta);
    if (sinT2 >= 1.0) {
        return 1.0;
    }
    const float cosT = sqrt(1.0 - sinT2);
    const float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    const float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5 * (rs * rs + rp * rp);
}

float srgbToLinear(float c) {
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

/// A gaussian's centre, its normal (towards the side its normal says), and
/// its widest sigma.
void gaussianAt(uint i, out float3 p, out float3 n, out float widest) {
    p = positions[i].xyz;
    const uint4 s = uint4(shape[i * 4], shape[i * 4 + 1], shape[i * 4 + 2], shape[i * 4 + 3]);
    const float3 logScale = float3(lowHalf(s.y), highHalf(s.y), lowHalf(s.z));
    widest = exp(max(logScale.x, max(logScale.y, logScale.z)));
    if (params.hasNormals != 0) {
        n = unpackNormal(normals[i]);
    } else {
        // The thinnest axis of its frame: the row of world-to-unit-sphere that
        // stretches most.
        const uint id = params.base + i;
        const float3 r0 = frames[id * 4].xyz, r1 = frames[id * 4 + 1].xyz, r2 = frames[id * 4 + 2].xyz;
        const float l0 = dot(r0, r0), l1 = dot(r1, r1), l2 = dot(r2, r2);
        n = normalize(l0 >= l1 && l0 >= l2 ? r0 : l1 >= l2 ? r1 : r2);
    }
}

/// A grid direction and the solid angle of its cell (the octahedral map's own).
float3 gridDirection(uint gx, uint gy, uint grid, out float solid) {
    const float3 w = octDecode((float2(gx, gy) + 0.5) / float(grid));
    const float l1 = abs(w.x) + abs(w.y) + abs(w.z);
    solid = 4.0 / float(grid * grid) * l1 * l1 * l1;
    return w;
}

/// The ray out of a gaussian along w: from its centre, a little off its
/// surface on w's side, so it neither takes the gaussian itself nor grazes
/// its neighbours on the same surface.
AtheneaRay rayFrom(float3 p, float3 n, float widest, float3 w) {
    const float side = dot(n, w) >= 0.0 ? 1.0 : -1.0;
    AtheneaRay ray;
    ray.Origin = p + n * (side * params.offset * widest);
    ray.Direction = w;
    ray.TMin = 1.0e-4 * max(widest, 1.0e-6);
    ray.TMax = 3.0e38;
    return ray;
}

/// A glass's own particles, which a ray sent through it lets pass.
bool ownGlass(uint id) {
    return params.ior > 1.0 && id >= params.base && id < params.base + params.count;
}

/// The way a direction goes out of a gaussian: behind a glass's surface, bent
/// in by its index (PathTracer.cpp's refractInto); `through` says it then
/// passes the glass's own particles.
float3 wayOut(float3 n, float3 w, out bool through) {
    through = params.ior > 1.0 && dot(n, w) < 0.0;
    if (!through) {
        return w;
    }
    const float cosIn = -dot(n, w);
    const float eta = 1.0 / params.ior;
    const float k = max(1.0 - eta * eta * (1.0 - cosIn * cosIn), 0.0);
    return normalize(eta * w + (eta * cosIn - sqrt(k)) * n);
}

/// One particle met: its alpha at its peak along the ray (rt_shadow's rule).
bool particleMet(RayQuery<RAY_FLAG_NONE> query, AtheneaRay ray, out uint id, out float alpha, out float tPeak) {
    alpha = 0.0;
    tPeak = 0.0;
    id = 0xFFFFFFFFu;
    if (query.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) {
        return false;
    }
    const uint instance = query.CandidateInstanceIndex();
    id = instanceIndices[instance * 2] + query.CandidatePrimitiveIndex() / kIcoTriangles;
    const float4 w0 = instanceData[instance * 3];
    const float4 w1 = instanceData[instance * 3 + 1];
    const float4 w2 = instanceData[instance * 3 + 2];
    const float3x3 toCloud = float3x3(w0.xyz, w1.xyz, w2.xyz);
    const float3 origin = mul(toCloud, ray.Origin) + float3(w0.w, w1.w, w2.w);
    const float3 direction = mul(toCloud, ray.Direction);
    const float opacity = colours[id + instanceIndices[instance * 2 + 1]].w;
    alpha = shadowAlpha(frames[id * 4], frames[id * 4 + 1], frames[id * 4 + 2], opacity, origin, direction, ray.TMin,
                        ray.TMax, 0.0, 3.0e38, tPeak);
    return alpha > 0.0;
}

/// What is left through the cloud along the ray (`through`: past a glass's own particles).
float transmittanceAlong(AtheneaRay ray, bool through) {
    ShadowState state = shadowBegin();
    RayDesc traced;
    traced.Origin = ray.Origin;
    traced.Direction = ray.Direction;
    traced.TMin = ray.TMin;
    traced.TMax = ray.TMax;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, traced);
    while (query.Proceed()) {
        uint id;
        float alpha, tPeak;
        if (particleMet(query, ray, id, alpha, tPeak) && !(through && ownGlass(id))) {
            shadowTake(state, id, query.CandidateInstanceIndex(), alpha);
            if (shadowDone(state, params.cut)) {
                query.Abort();
                break;
            }
        }
    }
    return state.transmittance;
}

/// THE FIRST PASS: a cloud's gaussians, their direct half (by every cloud's
/// id), their albedo, and -- for the cloud baked -- their open directions.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatTransferDirect(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint id = params.base + i;
    float3 p, n;
    float widest;
    gaussianAt(i, p, n, widest);

    // Its albedo: the base colour, in linear light.
    const uint4 s = uint4(shape[i * 4], shape[i * 4 + 1], shape[i * 4 + 2], shape[i * 4 + 3]);
    float3 base = saturate(float3(highHalf(s.z), lowHalf(s.w), highHalf(s.w)));
    if (params.linear == 0) {
        base = float3(srgbToLinear(base.x), srgbToLinear(base.y), srgbToLinear(base.z));
    }
    albedo[id] = float4(base, 1.0);

    float sh[16];
    for (uint k = 0; k < 16; ++k) {
        sh[k] = 0.0;
    }
    // Cell by cell, so its rays decide its bit together.
    const uint cellSide = params.cellSide > 0 ? params.cellSide : params.grid;
    const uint per = params.grid / cellSide;
    uint word = 0;
    uint open = 0;
    uint openFacing = 0;
    uint rays = 0;
    for (uint cy = 0; cy < cellSide; ++cy) {
        for (uint cx = 0; cx < cellSide; ++cx) {
            float through = 0.0;
            for (uint sy = 0; sy < per; ++sy) {
                for (uint sx = 0; sx < per; ++sx) {
                    float solid;
                    const float3 w = gridDirection(cx * per + sx, cy * per + sy, params.grid, solid);
                    // A glass's cells behind it are closed by its own far
                    // face, as the path tracer's are: the field says what is
                    // seen through it.
                    const float t = transmittanceAlong(rayFrom(p, n, widest, w), false);
                    through += t;
                    rays += 1;
                    const float cosine = dot(n, w);
                    if (cosine > 0.0) {
                        const float worth = t * cosine * solid / kPi;
                        for (uint k = 0; k < 16; ++k) {
                            sh[k] += worth * envBasisValue(k, w);
                        }
                    }
                }
            }
            if (params.writeBits != 0 && params.cellSide > 0) {
                const uint index = cy * cellSide + cx;
                if (through > 0.5 * float(per * per)) {
                    word |= 1u << (index & 31u);
                    open += 1;
                    const float3 centre = octDecode((float2(cx, cy) + 0.5) / float(cellSide));
                    openFacing += dot(centre, n) > 0.0 ? 1u : 0u;
                }
                if ((index & 31u) == 31u) {
                    bits[recordOf(i) * (cellSide * cellSide / 32u) + (index >> 5u)] = word;
                    word = 0;
                }
            }
        }
    }
    for (uint k = 0; k < 16; ++k) {
        direct[id * 16 + k] = sh[k];
    }
    InterlockedAdd(counters[1], rays);
    if (params.writeBits != 0) {
        InterlockedAdd(counters[0], 1u);
        InterlockedAdd(counters[2], open);
        InterlockedAdd(counters[3], openFacing);
    }
}

/// THE SECOND PASS: the cloud baked, what its rays meet -- the nearest
/// particles in front order, weighed as a blend weighs them -- as the
/// indirect half and the reflected field.
[shader("compute")]
[numthreads(64, 1, 1)]
void splatTransferBounced(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    float3 p, n;
    float widest;
    gaussianAt(i, p, n, widest);

    float3 indirect[16];
    float3 field[16];
    float closed[16];
    // THE LENS A FRAME DRAWS (splat_relight's lensExit): it takes the far
    // face from a sphere of the gaussian's mean curvature, and that sphere's
    // exit does not depend on its radius -- any curvature over zero turns the
    // room over as a whole ball does, none leaves it straight as a slab. On
    // a faceted glass whose curvature is noise about zero, neighbours
    // flipped between the two and the frame swirled. The far faces the field
    // meets say which the glass is here: each one's normal against the
    // ball's and the slab's, summed over the ways through.
    float ballMiss = 0.0;
    float slabMiss = 0.0;
    float facing = -1.0;
    float thickness = 0.0;
    for (uint k = 0; k < 16; ++k) {
        indirect[k] = float3(0.0);
        field[k] = float3(0.0);
        closed[k] = 0.0;
    }
    for (uint gy = 0; gy < params.grid; ++gy) {
        for (uint gx = 0; gx < params.grid; ++gx) {
            float solid;
            const float3 w = gridDirection(gx, gy, params.grid, solid);
            const float cosine = dot(n, w);
            // A glass's field is over the whole sphere; anything else's over
            // the half it faces.
            if (cosine <= 0.0 && !(params.ior > 1.0)) {
                continue;
            }
            bool through;
            const float3 way = wayOut(n, w, through);
            // The nearest particles the ray meets, by where each peaks.
            float nearT[kNearest];
            float nearAlpha[kNearest];
            uint nearId[kNearest];
            for (uint k = 0; k < kNearest; ++k) {
                nearT[k] = 3.0e38;
                nearAlpha[k] = 0.0;
                nearId[k] = 0xFFFFFFFFu;
            }
            uint held = 0;
            AtheneaRay ray = rayFrom(p, n, widest, way);
            // THE FAR FACE: through a glass, the bent ray goes on to where it
            // meets the glass's own far side -- its nearest own particle of
            // real opacity -- and leaves there, bent out by the index through
            // that particle's normal (or reflected, where total internal
            // reflection holds it in), as the path tracer's path does.
            float farOut = 1.0;
            if (through) {
                float tFar = 3.0e38;
                uint idFar = 0xFFFFFFFFu;
                RayDesc inner;
                inner.Origin = ray.Origin;
                inner.Direction = ray.Direction;
                inner.TMin = ray.TMin;
                inner.TMax = ray.TMax;
                RayQuery<RAY_FLAG_NONE> innerQuery;
                innerQuery.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, inner);
                while (innerQuery.Proceed()) {
                    uint id;
                    float alpha, tPeak;
                    if (particleMet(innerQuery, ray, id, alpha, tPeak) && ownGlass(id) && alpha > 0.3 && tPeak < tFar) {
                        tFar = tPeak;
                        idFar = id;
                    }
                }
                if (idFar != 0xFFFFFFFFu) {
                    float3 farP, farN;
                    float farWidest;
                    gaussianAt(idFar - params.base, farP, farN, farWidest);
                    farN = dot(farN, way) < 0.0 ? -farN : farN;
                    const float3 ballN = normalize(way * (2.0 * max(dot(way, -n), 0.0)) + n);
                    ballMiss += (1.0 - dot(farN, ballN)) * solid;
                    slabMiss += (1.0 - dot(farN, -n)) * solid;
                    // How thick it is head on: along the way nearest the
                    // inward normal, from the gaussian's centre.
                    if (dot(way, -n) > facing) {
                        facing = dot(way, -n);
                        thickness = tFar + params.offset * widest;
                    }
                    const float3 at = ray.Origin + way * tFar;
                    const float cosI = saturate(dot(way, farN));
                    const float eta = params.ior;
                    const float k = 1.0 - eta * eta * (1.0 - cosI * cosI);
                    float3 out;
                    if (k < 0.0) {
                        out = reflect(way, farN);
                    } else {
                        out = normalize(eta * way - (eta * cosI - sqrt(k)) * farN);
                        farOut = 1.0 - fresnelDielectricOf(cosI, 1.0 / eta);
                    }
                    ray.Origin = at + out * (params.offset * farWidest);
                    ray.Direction = out;
                    ray.TMin = 1.0e-4 * max(farWidest, 1.0e-6);
                    ray.TMax = 3.0e38;
                }
            }
            RayDesc traced;
            traced.Origin = ray.Origin;
            traced.Direction = ray.Direction;
            traced.TMin = ray.TMin;
            traced.TMax = ray.TMax;
            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(scene, RAY_FLAG_NONE, 0xFF, traced);
            while (query.Proceed()) {
                uint id;
                float alpha, tPeak;
                if (!particleMet(query, ray, id, alpha, tPeak) || (through && ownGlass(id))) {
                    continue;
                }
                bool seen = false;
                for (uint k = 0; k < held; ++k) {
                    seen = seen || nearId[k] == id;
                }
                if (seen) {
                    continue;
                }
                // Insert in order of t, dropping the farthest when full.
                uint at = held < kNearest ? held : kNearest;
                if (at == kNearest && tPeak >= nearT[kNearest - 1]) {
                    continue;
                }
                at = min(at, kNearest - 1);
                while (at > 0 && nearT[at - 1] > tPeak) {
                    nearT[at] = nearT[at - 1];
                    nearAlpha[at] = nearAlpha[at - 1];
                    nearId[at] = nearId[at - 1];
                    at -= 1;
                }
                nearT[at] = tPeak;
                nearAlpha[at] = alpha;
                nearId[at] = id;
                held = min(held + 1, kNearest);
            }
            // Front to back: each met particle's light, weighed by its alpha
            // times what is left in front of it.
            float left = 1.0;
            float3 met[16];
            for (uint k = 0; k < 16; ++k) {
                met[k] = float3(0.0);
            }
            for (uint j = 0; j < held; ++j) {
                const float weight = nearAlpha[j] * left;
                const float3 a = albedo[nearId[j]].rgb * weight;
                for (uint k = 0; k < 16; ++k) {
                    met[k] += a * direct[nearId[j] * 16 + k];
                }
                left *= 1.0 - nearAlpha[j];
            }
            const float worth = max(cosine, 0.0) * solid / kPi;
            float3 white = met[0] * kSqrtFourPi;
            float closedHere = 1.0 - left;
            if (through) {
                // Through the glass: its throughput times what it meets and
                // the white sky past it; closed by its own far face.
                const float f = fresnelDielectricOf(saturate(-cosine), params.ior);
                const float3 tint = albedo[params.base + i].rgb;
                white = (1.0 - f) * farOut * tint * tint * (white + float3(left));
                closedHere = 1.0;
            }
            for (uint k = 0; k < 16; ++k) {
                const float y = envBasisValue(k, w) * solid;
                indirect[k] += met[k] * worth;
                field[k] += white * y;
                closed[k] += closedHere * y;
            }
        }
    }
    m2sFieldOverClosed(field, closed);
    const uint r = recordOf(i);
    const uint id = params.base + i;
    float3 shapeIn = params.hasCurvature != 0
        ? float3(curvatureIn[i * 3], curvatureIn[i * 3 + 1], curvatureIn[i * 3 + 2]) : float3(0.0);
    if (params.ior > 1.0 && facing > 0.0) {
        // The sign a frame reads it with: measured on the frame's third axis,
        // read on the face the eye sees (splat_project, `faces`).
        const uint4 s = uint4(shape[i * 4], shape[i * 4 + 1], shape[i * 4 + 2], shape[i * 4 + 3]);
        const float3x3 axes = axesOfQuaternion(decodeQuaternion(s.x));
        const float side = dot(n, float3(axes[0][2], axes[1][2], axes[2][2])) >= 0.0 ? 1.0 : -1.0;
        if (ballMiss < slabMiss) {
            // A ball: its own curvature where it has a convex one, else the
            // sphere its head-on thickness is the diameter of.
            if (!(side * 0.5 * (shapeIn.x + shapeIn.z) > 0.0)) {
                const float k = 2.0 / max(thickness, 1.0e-6);
                shapeIn = float3(side * k, 0.0, side * k);
            }
            InterlockedAdd(counters[4], 1u);
        } else {
            shapeIn = float3(0.0);
            InterlockedAdd(counters[5], 1u);
        }
    }
    lensCurvature[r * 3] = shapeIn.x;
    lensCurvature[r * 3 + 1] = shapeIn.y;
    lensCurvature[r * 3 + 2] = shapeIn.z;
    for (uint k = 0; k < 16; ++k) {
        directOut[r * 16 + k] = direct[id * 16 + k];
        bounced[r * 48 + k * 3] = indirect[k].x;
        bounced[r * 48 + k * 3 + 1] = indirect[k].y;
        bounced[r * 48 + k * 3 + 2] = indirect[k].z;
        reflected[r * 48 + k * 3] = field[k].x;
        reflected[r * 48 + k * 3 + 1] = field[k].y;
        reflected[r * 48 + k * 3 + 2] = field[k].z;
    }
}
