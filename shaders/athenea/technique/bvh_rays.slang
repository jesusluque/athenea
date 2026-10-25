// Copyright (c) 2026 jesus luque.
//
// Rays through the scene's compute BVHs (world::BvhScene) from any kernel:
// the path tracer's route on a device without inline ray queries, so that
// a path traced frame or a bake is a compute kernel and not a ray tracing
// pipeline. The trees are visibility_bvh's -- a top level over the instances
// in view space, a bottom level per mesh in object space -- and so is the
// rule for a face: a ray that culls back faces culls a single-sided
// triangle's back as RAY_FLAG_CULL_BACK_FACING_TRIANGLES does on the
// hardware route; one that does not sees both.
//
// The rays are in view space. A ray in the world is taken there with the
// inverse of the frame's view to world; t is unchanged by an affine map of
// the origin and the (unnormalised) direction, so the hit's distance is the
// world ray's.
module bvh_rays;
__exported import athenea.technique.surface;
import athenea.rt.bvh_common;

public struct BvhRayParams {
    public uint instances;   // records in the scene: the top level's leaves
    public uint pad0; public uint pad1; public uint pad2;
};

public StructuredBuffer<float4>       rayTopBoxes;       // 2 per internal node
public StructuredBuffer<uint>         rayTopChildren;
public StructuredBuffer<uint>         rayTopLeaves;      // leaf -> instance
public StructuredBuffer<float4>       rayMeshBoxes;      // every mesh's nodes, at its nodeBase
public StructuredBuffer<uint>         rayMeshChildren;
public StructuredBuffer<uint>         rayMeshLeaves;     // at each mesh's firstTriangle: leaf -> triangle
public ConstantBuffer<BvhRayParams>   rayBvh;

static const uint kRayStack = 64;

/// What a ray met: the instance record's index (0xFFFFFFFF for nothing), the
/// triangle, where along the ray, and the barycentrics of vertices 1 and 2
/// as the hardware reports them.
public struct BvhHit {
    public float  t;
    public uint   instance;
    public uint   triangle;
    public float2 barycentrics;

    public bool found() { return instance != 0xFFFFFFFF; }
};

bool rayBoxHit(float3 lo, float3 hi, float3 origin, float3 inverse, float tMin, float tMax) {
    const float3 t0 = (lo - origin) * inverse;
    const float3 t1 = (hi - origin) * inverse;
    const float enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), max(min(t0.z, t1.z), tMin));
    const float leave = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), min(max(t0.z, t1.z), tMax));
    return enter <= leave;
}

/// Moller-Trumbore, t in (tMin, best.t).
void rayTriangleHit(inout BvhHit best, float3 origin, float3 direction, float tMin, MeshRecord m, uint instance,
                    uint triangle, bool cullBack) {
    const uint base = (m.firstTriangle + triangle) * 3;
    const float3 a = positions[m.firstPoint + indices[base]].xyz;
    const float3 e1 = positions[m.firstPoint + indices[base + 1]].xyz - a;
    const float3 e2 = positions[m.firstPoint + indices[base + 2]].xyz - a;
    const float3 p = cross(direction, e2);
    const float det = dot(e1, p);
    if (abs(det) < 1e-30) {
        return;
    }
    // The back of a single-sided triangle, judged in object space as
    // visibility_bvh judges it.
    if (cullBack && (instances[instance].flags & 1u) == 0 && det < 0.0) {
        return;
    }
    const float inverse = 1.0 / det;
    const float3 s = origin - a;
    const float u = dot(s, p) * inverse;
    if (u < 0.0 || u > 1.0) {
        return;
    }
    const float3 q = cross(s, e1);
    const float v = dot(direction, q) * inverse;
    if (v < 0.0 || u + v > 1.0) {
        return;
    }
    const float t = dot(e2, q) * inverse;
    if (t > tMin && t < best.t) {
        best.t = t;
        best.instance = instance;
        best.triangle = triangle;
        best.barycentrics = float2(u, v);
    }
}

void rayMeshTraverse(inout BvhHit best, float3 viewOrigin, float3 viewDirection, float tMin, uint instance,
                     bool cullBack, bool first) {
    const InstanceRecord r = instances[instance];
    const MeshRecord m = meshes[r.mesh];
    // Into object space: the inverse of view's 3x3 is its normal matrix transposed.
    const float3x3 inverseLinear = transpose(float3x3(r.normal0.xyz, r.normal1.xyz, r.normal2.xyz));
    const float3 origin = mul(inverseLinear, viewOrigin - float3(r.view0.w, r.view1.w, r.view2.w));
    const float3 direction = mul(inverseLinear, viewDirection);
    const float3 inverse = 1.0 / direction;
    if (m.triangles == 1) {
        rayTriangleHit(best, origin, direction, tMin, m, instance, 0, cullBack);
        return;
    }
    uint stack[kRayStack];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        const uint node = m.nodeBase + stack[--depth];
        if (!rayBoxHit(rayMeshBoxes[node * 2].xyz, rayMeshBoxes[node * 2 + 1].xyz, origin, inverse, tMin, best.t)) {
            continue;
        }
        for (uint k = 0; k < 2; ++k) {
            const uint child = rayMeshChildren[node * 2 + k];
            if ((child & kLeafBit) != 0) {
                rayTriangleHit(best, origin, direction, tMin, m, instance,
                               rayMeshLeaves[m.firstTriangle + (child & ~kLeafBit)], cullBack);
                if (first && best.found()) {
                    return;
                }
            } else if (depth < kRayStack) {
                stack[depth++] = child;
            }
        }
    }
}

/// The nearest triangle in (tMin, tMax) along a view space ray, of the
/// instances whose mask shares a bit with `mask`; with `first`, any one
/// (a shadow ray's question). Not inlined: every ray of a kernel calls the
/// one copy, so a kernel of many trace sites stays the size of one.
[noinline]
public BvhHit bvhTrace(float3 origin, float3 direction, float tMin, float tMax, uint mask, bool cullBack, bool first) {
    BvhHit best;
    best.t = tMax;
    best.instance = 0xFFFFFFFF;
    best.triangle = 0;
    best.barycentrics = float2(0.0);
    const uint count = rayBvh.instances;
    if (count == 0) {
        return best;
    }
    if (count == 1) {
        if ((instances[0].mask & mask & 0xFFu) != 0) {
            rayMeshTraverse(best, origin, direction, tMin, 0, cullBack, first);
        }
        return best;
    }
    const float3 inverse = 1.0 / direction;
    uint stack[kRayStack];
    uint depth = 0;
    stack[depth++] = 0;
    while (depth > 0) {
        const uint node = stack[--depth];
        if (!rayBoxHit(rayTopBoxes[node * 2].xyz, rayTopBoxes[node * 2 + 1].xyz, origin, inverse, tMin, best.t)) {
            continue;
        }
        for (uint k = 0; k < 2; ++k) {
            const uint child = rayTopChildren[node * 2 + k];
            if ((child & kLeafBit) != 0) {
                const uint instance = rayTopLeaves[child & ~kLeafBit];
                if ((instances[instance].mask & mask & 0xFFu) != 0) {
                    rayMeshTraverse(best, origin, direction, tMin, instance, cullBack, first);
                    if (first && best.found()) {
                        return best;
                    }
                }
            } else if (depth < kRayStack) {
                stack[depth++] = child;
            }
        }
    }
    return best;
}

/// World to view: the inverse of the frame's view-to-world rows (r0, r1,
/// r2), applied to a point (w 1) or a direction (w 0).
public float3 bvhWorldToView(float4 r0, float4 r1, float4 r2, float3 x, float w) {
    const float3 c0 = float3(r0.x, r1.x, r2.x);
    const float3 c1 = float3(r0.y, r1.y, r2.y);
    const float3 c2 = float3(r0.z, r1.z, r2.z);
    const float3 local = x - float3(r0.w, r1.w, r2.w) * w;
    // The inverse of the 3x3 whose columns are c0, c1, c2: its rows are the
    // columns' pairwise cross products over the determinant.
    const float3 i0 = cross(c1, c2);
    const float3 i1 = cross(c2, c0);
    const float3 i2 = cross(c0, c1);
    const float det = dot(c0, i0);
    return float3(dot(i0, local), dot(i1, local), dot(i2, local)) / det;
}
