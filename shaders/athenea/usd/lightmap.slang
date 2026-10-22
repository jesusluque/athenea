// Copyright (c) 2026 jesus luque.
//
// `athenea lightmap`: a mesh's light, path traced into a texture over its own
// set of texture coordinates, on the device.
//
// lightmapCover: a thread a triangle walks the texels its coordinates cover
// in the atlas and claims each one (the first triangle to reach a texel keeps
// it), leaving the point of the surface the texel's centre stands for, the way
// the surface faces there and how wide a texel is on it.
//
// lightmapFlag and lightmapRays: the claimed texels counted, scanned
// (gpu::PrefixSum) and written as the path tracer's bake rays, three float4 a
// point (StageRenderer::bakePointsOnDevice), with the texel each one is for.
//
// lightmapScatter: what the bake answered, the light the point sends out,
// written back into its texel; a point the bake found nothing under is left
// for the dilation.
//
// lightmapDilate: a texel no triangle claimed takes the mean of its claimed
// neighbours, a ring a pass, so a sample that falls in the margin a packer
// left between two islands reads the island's own light and not black.
module lightmap;

struct CoverParams {
    float4 row0;            // object to world, rows of a 3x4
    float4 row1;
    float4 row2;
    float4 normal0;         // its inverse transpose, the same
    float4 normal1;
    float4 normal2;
    uint   triangles;
    uint   size;            // the atlas's side, texels
    uint   uvInterpolation; // 1 a point's, 2 a face corner's
    uint   hasNormals;      // 1 a point's, 2 a face corner's, 0 none: the triangle's own
    uint   mesh;            // this mesh's number, written into the texels it claims (plus one)
    float  offset;          // how far above the surface a bake ray starts, world units
    uint   pad0;
    uint   pad1;
};

ConstantBuffer<CoverParams>  cover;
StructuredBuffer<float4>     positions;   // the mesh's points
StructuredBuffer<uint>       indices;     // 3 a triangle
StructuredBuffer<uint>       corners;     // 3 a triangle: face-vertex indices
StructuredBuffer<float4>     uvs;         // the lightmap coordinates, a float4 an element
StructuredBuffer<float4>     normals;     // the mesh's normals primvar, where it has one
RWStructuredBuffer<uint>     owner;       // a texel: the mesh that claimed it, plus one; 0 none
RWStructuredBuffer<float4>   texelPoint;  // a texel: the point (xyz) and the bake ray's offset (w)
RWStructuredBuffer<float4>   texelNormal; // a texel: the way it faces (xyz) and a texel's width on it (w)

float3 toWorldPoint(float3 p) {
    const float4 h = float4(p, 1.0);
    return float3(dot(cover.row0, h), dot(cover.row1, h), dot(cover.row2, h));
}

float3 toWorldNormal(float3 n) {
    return normalize(float3(dot(cover.normal0.xyz, n), dot(cover.normal1.xyz, n), dot(cover.normal2.xyz, n)));
}

uint elementOf(uint interpolation, uint t, uint k) {
    return interpolation == 1 ? indices[t * 3 + k] : corners[t * 3 + k];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void lightmapCover(uint3 tid: SV_DispatchThreadID) {
    const uint t = tid.x;
    if (t >= cover.triangles) {
        return;
    }
    const float size = float(cover.size);
    const float2 u0 = uvs[elementOf(cover.uvInterpolation, t, 0)].xy * size;
    const float2 u1 = uvs[elementOf(cover.uvInterpolation, t, 1)].xy * size;
    const float2 u2 = uvs[elementOf(cover.uvInterpolation, t, 2)].xy * size;
    const float3 p0 = toWorldPoint(positions[indices[t * 3]].xyz);
    const float3 p1 = toWorldPoint(positions[indices[t * 3 + 1]].xyz);
    const float3 p2 = toWorldPoint(positions[indices[t * 3 + 2]].xyz);
    const float2 e1 = u1 - u0;
    const float2 e2 = u2 - u0;
    const float area = e1.x * e2.y - e1.y * e2.x;   // twice the triangle's, in texels
    if (abs(area) < 1.0e-12) {
        return;
    }
    const float3 face = cross(p1 - p0, p2 - p0);
    const float worldArea = length(face);            // twice the triangle's, in the world
    if (!(worldArea > 0.0)) {
        return;
    }
    const float width = sqrt(worldArea / abs(area));   // a texel's side on the surface
    const float3 faceNormal = face / worldArea;
    float3 n0 = faceNormal, n1 = faceNormal, n2 = faceNormal;
    if (cover.hasNormals != 0) {
        n0 = toWorldNormal(normals[elementOf(cover.hasNormals, t, 0)].xyz);
        n1 = toWorldNormal(normals[elementOf(cover.hasNormals, t, 1)].xyz);
        n2 = toWorldNormal(normals[elementOf(cover.hasNormals, t, 2)].xyz);
    }
    const int2 lo = int2(floor(min(min(u0, u1), u2)));
    const int2 hi = int2(ceil(max(max(u0, u1), u2)));
    const int2 first = clamp(lo, int2(0), int2(int(cover.size) - 1));
    const int2 last = clamp(hi, int2(0), int2(int(cover.size) - 1));
    // A texel whose centre lies just outside the triangle, by up to half a
    // texel along an edge, is still the triangle's: an island's border texels
    // are then lit by it rather than left to the dilation.
    const float slack = 0.5 / max(max(length(e1), length(e2)), 1.0e-6);
    for (int y = first.y; y <= last.y; ++y) {
        for (int x = first.x; x <= last.x; ++x) {
            const float2 c = float2(float(x) + 0.5, float(y) + 0.5) - u0;
            const float b1 = (c.x * e2.y - c.y * e2.x) / area;
            const float b2 = (e1.x * c.y - e1.y * c.x) / area;
            const float b0 = 1.0 - b1 - b2;
            if (b0 < -slack || b1 < -slack || b2 < -slack) {
                continue;
            }
            const uint texel = uint(y) * cover.size + uint(x);
            uint previous;
            InterlockedCompareExchange(owner[texel], 0u, cover.mesh + 1u, previous);
            if (previous != 0u) {
                continue;
            }
            const float3 w = max(float3(b0, b1, b2), float3(0.0));
            const float3 weights = w / max(w.x + w.y + w.z, 1.0e-6);
            const float3 p = p0 * weights.x + p1 * weights.y + p2 * weights.z;
            float3 n = normalize(n0 * weights.x + n1 * weights.y + n2 * weights.z);
            if (dot(n, faceNormal) < 0.0 && cover.hasNormals == 0) {
                n = -n;
            }
            texelPoint[texel] = float4(p, cover.offset);
            texelNormal[texel] = float4(n, width);
        }
    }
}

struct AtlasParams {
    uint texels;   // size * size
    uint size;
    uint count;    // the points baked
    uint pad0;
};

ConstantBuffer<AtlasParams>  atlas;
RWStructuredBuffer<uint>     flags;     // a texel: 1 where claimed
StructuredBuffer<uint>       starts;    // the scan of `flags`
RWStructuredBuffer<float4>   rays;      // three a point
RWStructuredBuffer<uint>     texelOf;   // a point: its texel
StructuredBuffer<float4>     baked;     // a point: the constant term the bake answered
RWStructuredBuffer<float4>   image;     // a texel: linear rgb, and 1 in alpha where it holds light
StructuredBuffer<float4>     imageIn;   // lightmapDilate's last pass

[shader("compute")]
[numthreads(256, 1, 1)]
void lightmapFlag(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= atlas.texels) {
        return;
    }
    flags[tid.x] = owner[tid.x] != 0u ? 1u : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightmapRays(uint3 tid: SV_DispatchThreadID) {
    const uint texel = tid.x;
    if (texel >= atlas.texels || owner[texel] == 0u) {
        return;
    }
    const uint k = starts[texel];
    const float4 n = texelNormal[texel];
    rays[k * 3] = texelPoint[texel];
    rays[k * 3 + 1] = n;
    rays[k * 3 + 2] = float4(0.0);
    texelOf[k] = texel;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void lightmapScatter(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= atlas.count) {
        return;
    }
    const float4 constant = baked[k];
    if (!(constant.w > 0.0)) {
        return;
    }
    // The constant term as 3DGS keeps it: 0.5 + Y0 c, linear light here.
    const float3 radiance = max(0.5 + 0.28209479177387814 * constant.xyz, float3(0.0));
    image[texelOf[k]] = float4(radiance, 1.0);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void lightmapDilate(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= atlas.size || tid.y >= atlas.size) {
        return;
    }
    const uint at = tid.y * atlas.size + tid.x;
    const float4 here = imageIn[at];
    if (here.w > 0.0) {
        image[at] = here;
        return;
    }
    float3 sum = float3(0.0);
    float count = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int2 q = int2(tid.xy) + int2(dx, dy);
            if (any(q < 0) || q.x >= int(atlas.size) || q.y >= int(atlas.size)) {
                continue;
            }
            const float4 v = imageIn[uint(q.y) * atlas.size + uint(q.x)];
            if (v.w > 0.0) {
                sum += v.rgb;
                count += 1.0;
            }
        }
    }
    image[at] = count > 0.0 ? float4(sum / count, 1.0) : float4(0.0);
}

/// Every texel unclaimed and dark, before the meshes are walked.
[shader("compute")]
[numthreads(256, 1, 1)]
void lightmapClear(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= atlas.texels) {
        return;
    }
    owner[tid.x] = 0u;
    image[tid.x] = float4(0.0);
}
