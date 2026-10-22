// Copyright (c) 2026 jesus luque.
//
// A PLATE OVER THE FINISHED FRAME: a mesh marked `primvars:athenea:plate`
// is not drawn with the others. Its pixels are found here, a ray a pixel
// against its triangles in view space, and where it stands nearer than the
// mesh layer's surface -- or where the mesh layer drew nothing -- its
// material's first texture is laid over what the frame holds: its colour,
// and its alpha as a coverage that mixes, not one a lot cuts. A keyed
// camera image so shows the set through its alpha and stays behind what
// stands in front of it, in one pass and one texture read a pixel, with no
// lobe evaluated.
//
// What it does not see: splats, which keep no depth of their own; a plate
// covers them wherever it stands in front of the meshes' surface.
module plate_composite;
import camera;
import athenea.material.texture_table;

struct PlateParams {
    // Object to view, rows of a 3x4.
    float4 row0;
    float4 row1;
    float4 row2;
    uint   triangles;
    uint   stInterpolation;   // 0 none (the plate's corners are (0,0)), 1 a point's, 2 a face corner's
    uint   texture;           // the texture table's id
    uint   sampler;
    uint   premultiplied;     // 1: the texture's colour is premultiplied by its alpha
    uint   hasDepth;          // 0: no mesh layer this frame; the plate is nearest everywhere
    uint   pad0;
    uint   pad1;
};

RWStructuredBuffer<float4>    colour;
StructuredBuffer<float>       meshDepth;   // view z of the mesh layer's surface, 0 where none
StructuredBuffer<float4>      positions;   // the plate's points
StructuredBuffer<uint>        indices;     // 3 a triangle: points
StructuredBuffer<uint>        corners;     // 3 a triangle: face-vertex indices
StructuredBuffer<float4>      st;          // the plate's st primvar, a float4 an element (GpuPrimvar)
ParameterBlock<TextureTable>  gTextures;
ConstantBuffer<CameraParams>  camera;
ConstantBuffer<PlateParams>   plate;

float3 toView(float3 p) {
    const float4 h = float4(p, 1.0);
    return float3(dot(plate.row0, h), dot(plate.row1, h), dot(plate.row2, h));
}

float2 stAt(uint triangle, uint k) {
    if (plate.stInterpolation == 0) {
        return float2(0.0);
    }
    const uint index = plate.stInterpolation == 1 ? indices[triangle * 3 + k] : corners[triangle * 3 + k];
    return st[index].xy;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void plateComposite(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= camera.width || tid.y >= camera.height) {
        return;
    }
    const uint at = tid.y * camera.width + tid.x;
    float3 origin;
    float3 direction;
    viewRay(camera, float2(tid.xy) + 0.5, origin, direction);
    // The nearest of its triangles along the ray, from either side.
    float nearest = 3.4e38;
    float2 uv = float2(0.0);
    bool hit = false;
    for (uint t = 0; t < plate.triangles; ++t) {
        const float3 a = toView(positions[indices[t * 3]].xyz);
        const float3 b = toView(positions[indices[t * 3 + 1]].xyz);
        const float3 c = toView(positions[indices[t * 3 + 2]].xyz);
        const float3 e1 = b - a;
        const float3 e2 = c - a;
        const float3 p = cross(direction, e2);
        const float det = dot(e1, p);
        if (abs(det) < 1e-20) {
            continue;
        }
        const float inverse = 1.0 / det;
        const float3 s = origin - a;
        const float w1 = dot(s, p) * inverse;
        const float3 q = cross(s, e1);
        const float w2 = dot(direction, q) * inverse;
        const float distance = dot(e2, q) * inverse;
        if (w1 < 0.0 || w2 < 0.0 || w1 + w2 > 1.0 || !(distance > 0.0) || distance >= nearest) {
            continue;
        }
        nearest = distance;
        uv = stAt(t, 0) * (1.0 - w1 - w2) + stAt(t, 1) * w1 + stAt(t, 2) * w2;
        hit = true;
    }
    if (!hit) {
        return;
    }
    // The ray's direction has z 1, so the distance along it is view z, as the
    // mesh layer's depth is. A surface nearer than the plate stays in front.
    const float z = origin.z + direction.z * nearest;
    if (plate.hasDepth != 0) {
        const float surface = meshDepth[at];
        if (surface > 0.0 && surface < z) {
            return;
        }
    }
    bool found;
    const float4 image = sampleTexture(gTextures, plate.texture, plate.sampler, uv, float2(0.0), float2(0.0), found);
    if (!found) {
        return;
    }
    const float alpha = saturate(image.a);
    const float3 over = plate.premultiplied != 0 ? image.rgb : image.rgb * alpha;
    const float4 under = colour[at];
    colour[at] = float4(over + under.rgb * (1.0 - alpha), alpha + under.a * (1.0 - alpha));
}
