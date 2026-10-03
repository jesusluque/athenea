// Copyright (c) 2026 jesus luque.
//
// THE FAR FACE OF A GLASS OBJECT, FOUND WITH A RAY.
//
// A cloud has no surfaces, so bending a ray at every gaussian of a glass ball
// is mush, and bending it once at the face it enters shows the room behind
// without turning it over: a ball is a thick lens, and a lens needs its two
// interfaces. What this finds is the second one -- the farthest gaussian
// along the bent ray that is still glass -- so the shading can bend again,
// from glass to air, and look along the direction a lens actually sends.
//
// It walks the cloud's own LBVH, the compute route's (bvh_common.slang), for
// the instance the gaussian belongs to and no other: what is on the far side
// of a glass object is the same object, and a ray that leaves it has left.
//
// Which gaussians are glass is read from the frame's spare lane, which
// rt_frames.slang fills with the particle's transmission. No table of its
// own, and nothing to keep in step.
module rt_glass;
import rt_common;
import bvh_common;

/// A depth-first stack over a Karras tree: the same bound rt_bvh_render uses.
static const uint kGlassStack = 64;

public struct GlassExit {
    public bool   found;
    public float3 position;   ///< cloud space
    public float3 normal;     ///< cloud space, pointing out of the glass
    public float  t;
};

/// The short axis of a particle, which for a gaussian built from a surface is
/// that surface's normal: the row of `m` that shrinks space most, since `m`
/// maps the ellipsoid to the unit sphere.
float3 particleNormal(float4 f0, float4 f1, float4 f2) {
    const float3 rows[3] = {f0.xyz, f1.xyz, f2.xyz};
    uint at = 0;
    float most = dot(rows[0], rows[0]);
    for (uint k = 1; k < 3; ++k) {
        const float length2 = dot(rows[k], rows[k]);
        if (length2 > most) {
            most = length2;
            at = k;
        }
    }
    return normalize(rows[at]);
}

/// Walks `origin + t direction` (cloud space) and returns the last glass
/// particle it meets within `tMax`. `cut` is the transmission a particle must
/// carry to count as glass at all.
public GlassExit glassExit(StructuredBuffer<float4> frames, StructuredBuffer<float4> bvhBoxes,
                           StructuredBuffer<uint> bvhChildren, StructuredBuffer<uint> bvhLeaves,
                           uint nodeBase, uint leafBase, uint particles, uint particleBase,
                           float3 origin, float3 direction, float tMin, float tMax, float cut) {
    GlassExit out;
    out.found = false;
    out.position = origin;
    out.normal = -direction;
    out.t = 0.0;
    if (particles == 0) {
        return out;
    }
    const float3 inverse = float3(1.0) / direction;
    uint stack[kGlassStack];
    float stackEntry[kGlassStack];
    uint depth = 0;
    stack[depth] = particles == 1 ? kLeafBit : 0u;
    stackEntry[depth] = tMin;
    depth += 1;
    float farthest = tMin;
    while (depth > 0) {
        depth -= 1;
        const uint child = stack[depth];
        if ((child & kLeafBit) != 0) {
            const uint particle = bvhLeaves[leafBase + (child & ~kLeafBit)];
            const uint id = particleBase + particle;
            const float4 f0 = frames[id * 4];
            const float4 f1 = frames[id * 4 + 1];
            const float4 f2 = frames[id * 4 + 2];
            const float4 f3 = frames[id * 4 + 3];
            if (!(f3.z >= cut) || f3.w > 0.5) {
                // Not glass, or a thin wall: a feather card over a glass body
                // is not the body's far face, and the ray does not leave
                // through it.
                continue;
            }
            const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
            const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
            const float3 d = mul(m, direction);
            const float a = dot(d, d);
            if (!(a > 0.0)) {
                continue;
            }
            const float tPeak = -dot(o, d) / a;
            const float3 closest = o + tPeak * d;
            const float power = dot(closest, closest);
            if (power > f3.x) {
                continue;
            }
            // Where it leaves that particle's cut ellipsoid, which is the
            // far side of this one gaussian: a shell of them gives the far
            // side of the object.
            const float leave = tPeak + sqrt((f3.x - power) / a);
            if (leave <= farthest || leave < tMin || leave > tMax) {
                continue;
            }
            farthest = leave;
            out.found = true;
            out.t = leave;
            out.position = origin + direction * leave;
            const float3 n = particleNormal(f0, f1, f2);
            // Out of the glass: away from where the ray came from.
            out.normal = dot(n, direction) > 0.0 ? n : -n;
            continue;
        }
        const uint node = nodeBase + child;
        const uint left = bvhChildren[node * 2];
        const uint right = bvhChildren[node * 2 + 1];
        const float leftEntry =
            (left & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + left, origin, inverse, tMin, tMax);
        const float rightEntry =
            (right & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + right, origin, inverse, tMin, tMax);
        if (depth + 2 > kGlassStack) {
            continue;
        }
        // Nothing is nearer than the farthest hit, so the whole tree is
        // walked: what closes a box here is only the ray's own end.
        if (rightEntry <= tMax) {
            stack[depth] = right;
            stackEntry[depth] = rightEntry;
            depth += 1;
        }
        if (leftEntry <= tMax) {
            stack[depth] = left;
            stackEntry[depth] = leftEntry;
            depth += 1;
        }
    }
    return out;
}

/// THE FIRST THING BEHIND THE GLASS, in the same cloud.
///
/// What a ball shows is mostly not the sky: it is whatever stands behind it,
/// bent. Where that is the cloud's own -- a chess pawn's collar behind its
/// own glass ball -- the ray that left the glass can be walked on until it
/// meets a particle that is not glass, and what that particle was shaded
/// with is what the glass shows. Its colour comes from the pass before this
/// one, which is why glass is shaded twice.
/// A RAY THAT STARTS ON A SURFACE MUST NOT MEET THAT SURFACE.
///
/// A convex cloud is a shell of overlapping gaussians, and a mirror ray that
/// leaves one of them at a grazing angle runs inside that shell for a long
/// chord: every neighbour along it is within the cutoff, and the nearest of
/// them is what the ray "meets". A gold ball then reflects itself all round
/// its rim and loses its blue twice over, and a grey one reads a tenth dark
/// beside the mesh -- which is what the reference sheets measured, and what
/// was chased through the bake before it was found here. A step along the
/// ray cannot fix it: the chord is tens of footprints long.
///
/// The guard is by height, not by distance: a candidate whose closest point
/// rises less than `clearance` above the plane through `basePoint` with normal
/// `baseNormal` is the surface the ray started on, and is skipped. The shell's
/// own gaussians along a tangent ray rise by the sagitta, a few thousandths of
/// a footprint; anything else the cloud can reflect -- a collar's ball, the far
/// wall of a cup -- stands well above that plane. A negative `clearance`
/// keeps every candidate, which is what a glass ray wants.
public GlassExit glassNearest(StructuredBuffer<float4> frames, StructuredBuffer<float4> bvhBoxes,
                              StructuredBuffer<uint> bvhChildren, StructuredBuffer<uint> bvhLeaves,
                              uint nodeBase, uint leafBase, uint particles, uint particleBase,
                              float3 origin, float3 direction, float tMin, float tMax, float cut,
                              float3 basePoint, float3 baseNormal, float clearance,
                              out uint hitParticle) {
    GlassExit out;
    out.found = false;
    out.position = origin;
    out.normal = -direction;
    out.t = tMax;
    hitParticle = 0;
    if (particles == 0) {
        return out;
    }
    const float3 inverse = float3(1.0) / direction;
    uint stack[kGlassStack];
    float stackEntry[kGlassStack];
    uint depth = 0;
    stack[depth] = particles == 1 ? kLeafBit : 0u;
    stackEntry[depth] = tMin;
    depth += 1;
    float nearest = tMax;
    while (depth > 0) {
        depth -= 1;
        const uint child = stack[depth];
        if (stackEntry[depth] >= nearest) {
            continue;
        }
        if ((child & kLeafBit) != 0) {
            const uint particle = bvhLeaves[leafBase + (child & ~kLeafBit)];
            const uint id = particleBase + particle;
            const float4 f0 = frames[id * 4];
            const float4 f1 = frames[id * 4 + 1];
            const float4 f2 = frames[id * 4 + 2];
            const float4 f3 = frames[id * 4 + 3];
            if (f3.z >= cut) {
                continue;   // glass: the ray is leaving through it, not stopped by it
            }
            const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
            const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
            const float3 d = mul(m, direction);
            const float a = dot(d, d);
            if (!(a > 0.0)) {
                continue;
            }
            const float tPeak = -dot(o, d) / a;
            const float3 closest = o + tPeak * d;
            const float power = dot(closest, closest);
            if (power > f3.x || tPeak < tMin || tPeak >= nearest) {
                continue;
            }
            if (clearance >= 0.0 &&
                dot(origin + direction * tPeak - basePoint, baseNormal) < clearance) {
                continue;   // the surface the ray started on, not something it met
            }
            nearest = tPeak;
            out.found = true;
            out.t = tPeak;
            out.position = origin + direction * tPeak;
            hitParticle = particle;
            continue;
        }
        const uint node = nodeBase + child;
        const uint left = bvhChildren[node * 2];
        const uint right = bvhChildren[node * 2 + 1];
        const float leftEntry =
            (left & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + left, origin, inverse, tMin, nearest);
        const float rightEntry =
            (right & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + right, origin, inverse, tMin, nearest);
        if (depth + 2 > kGlassStack) {
            continue;
        }
        const bool leftFirst = leftEntry <= rightEntry;
        const uint farChild = leftFirst ? right : left;
        const float farEntry = leftFirst ? rightEntry : leftEntry;
        const uint nearChild = leftFirst ? left : right;
        const float nearEntry = leftFirst ? leftEntry : rightEntry;
        if (farEntry <= nearest) {
            stack[depth] = farChild;
            stackEntry[depth] = farEntry;
            depth += 1;
        }
        if (nearEntry <= nearest) {
            stack[depth] = nearChild;
            stackEntry[depth] = nearEntry;
            depth += 1;
        }
    }
    return out;
}

/// HOW MUCH OF THE SKY A RAY ACTUALLY REACHES, through the cloud itself.
///
/// The same walk as the two above, asked the cheapest question there is: does
/// anything stand between this gaussian and the sky in that direction. A
/// reflection narrowed by the answer is a reflection that knows the asset's
/// own shape is in front of it -- which a fit derived from the transfer can
/// only approximate, because the transfer is one number for the whole
/// hemisphere and a mirror looks in one direction.
///
/// It stops at the first particle dense enough to matter rather than walking
/// on, so an occluded ray is the cheap case and an open one pays the tree.
/// `cut` is the alpha a particle needs before it counts as something in the
/// way.
public bool cloudOccluded(StructuredBuffer<float4> frames, StructuredBuffer<float4> bvhBoxes,
                          StructuredBuffer<uint> bvhChildren, StructuredBuffer<uint> bvhLeaves,
                          uint nodeBase, uint leafBase, uint particles, uint particleBase,
                          float3 origin, float3 direction, float tMin, float tMax, float cut) {
    if (particles == 0) {
        return false;
    }
    const float3 inverse = float3(1.0) / direction;
    uint stack[kGlassStack];
    uint depth = 0;
    stack[depth] = particles == 1 ? kLeafBit : 0u;
    depth += 1;
    while (depth > 0) {
        depth -= 1;
        const uint child = stack[depth];
        if ((child & kLeafBit) != 0) {
            const uint particle = bvhLeaves[leafBase + (child & ~kLeafBit)];
            const uint id = particleBase + particle;
            const float4 f0 = frames[id * 4];
            const float4 f1 = frames[id * 4 + 1];
            const float4 f2 = frames[id * 4 + 2];
            const float4 f3 = frames[id * 4 + 3];
            const float3x3 m = float3x3(f0.xyz, f1.xyz, f2.xyz);
            const float3 o = mul(m, origin - float3(f0.w, f1.w, f2.w));
            const float3 d = mul(m, direction);
            const float a = dot(d, d);
            if (!(a > 0.0)) {
                continue;
            }
            const float tPeak = -dot(o, d) / a;
            if (tPeak < tMin || tPeak > tMax) {
                continue;
            }
            const float3 closest = o + tPeak * d;
            const float power = dot(closest, closest);
            if (power > f3.x) {
                continue;
            }
            // What the particle is worth where the ray passes through it: the
            // opacity the frame kept for it, fallen off by how far from its
            // centre the ray went.
            if (f3.y * exp(-0.5 * power) >= cut) {
                return true;
            }
            continue;
        }
        const uint node = nodeBase + child;
        const uint left = bvhChildren[node * 2];
        const uint right = bvhChildren[node * 2 + 1];
        const float leftEntry =
            (left & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + left, origin, inverse, tMin, tMax);
        const float rightEntry =
            (right & kLeafBit) != 0 ? tMin : boxEntryAt(bvhBoxes, nodeBase + right, origin, inverse, tMin, tMax);
        if (depth + 2 > kGlassStack) {
            continue;
        }
        if (rightEntry <= tMax) {
            stack[depth] = right;
            depth += 1;
        }
        if (leftEntry <= tMax) {
            stack[depth] = left;
            depth += 1;
        }
    }
    return false;
}

/// Where [tMin, tMax] enters box `node`, or a miss. rt_bvh_render's, with the
/// buffer handed over rather than bound.
public float boxEntryAt(StructuredBuffer<float4> boxes, uint node, float3 origin, float3 inverse, float tMin,
                        float tMax) {
    const float3 lo = boxes[node * 2].xyz;
    const float3 hi = boxes[node * 2 + 1].xyz;
    const float3 a = (lo - origin) * inverse;
    const float3 b = (hi - origin) * inverse;
    const float3 near = min(a, b);
    const float3 far = max(a, b);
    const float enter = max(max(near.x, near.y), max(near.z, tMin));
    const float leave = min(min(far.x, far.y), min(far.z, tMax));
    return enter <= leave ? enter : 3.4e38;
}
