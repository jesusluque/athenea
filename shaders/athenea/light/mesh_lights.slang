// Copyright (c) 2026 jesus luque.
//
// MESH LIGHTS: emitting triangles -- a material's emission, or a mesh with
// UsdLux's MeshLightAPI -- gathered into clusters that light the scene as
// lights do. technique::MeshLights builds them on the device into the mesh
// region at the front of the light words buffer (`iesValues`, which every
// kernel that lights anything binds already), and this module is how a
// kernel reads them: the region's header, a cluster's record, its triangles,
// the tree over the clusters, and the emission a mesh light's surface shows.
//
// A cluster keeps its real triangles, which a sample is drawn from (the path
// tracer, a stochastic estimate), and a rectangle standing in for them -- its
// power kept -- which a closed form integrates (LTC, the splats). The tree
// is the UsdLux lights' (light_bvh.slang): the same nodes, the same
// importance, in the region with clusters for leaves.
//
// The region, in words (floats; counts and offsets as their bits):
//
//   header      kMeshHeaderWords: counts, offsets, the groups' weights and
//               the emitter sets (a MeshLightAPI prim's controls: colour
//               times intensity, sync mode, flags -- 1 two-sided, 4 colour
//               temperature -- exposure, temperature; 8 words a set)
//   records     instances + 1: each instance record's first triangle in the
//               scene's numbering (EmissiveTable's), then the total
//   qualified   triangles: the scene numbers of the emitting triangles, rising
//   slots       triangles: where each of those sits in the sorted order
//   triangles   16 words each, in cluster order (kMeshTriangleWords)
//   clusters    32 words each (kMeshClusterWords)
//   nodes       16 words each: 2 clusters - 1 of them
//
// This module declares no global parameter, so importing it costs a kernel
// no binding.
module mesh_lights;
import athenea.light.lights;
import athenea.light.light_bvh;
import athenea.rt.bvh_common;
import athenea.world.scene_types;

// ---- the header ---------------------------------------------------------------

public static const uint kMeshClusterCount = 0;
public static const uint kMeshTriangleCount = 1;
public static const uint kMeshTreeNodes = 2;
public static const uint kMeshTotalPower = 3;      // float: the clusters' power, summed (the root's)
public static const uint kMeshInstanceCount = 4;
public static const uint kMeshNodeBase = 5;
public static const uint kMeshClusterBase = 6;
public static const uint kMeshTriangleBase = 7;
public static const uint kMeshRecordBase = 8;
public static const uint kMeshQualifiedBase = 9;
public static const uint kMeshSlotBase = 10;
public static const uint kMeshRevision = 11;       // counts every rebuild and refit
public static const uint kMeshOn = 12;             // 1: the clusters are lights (athenea:meshLights)
public static const uint kMeshDepth = 13;          // the tree's depth, which a refit settles in
public static const uint kMeshGroupBase = 16;      // 32 groups, 4 words each: rgb weight
public static const uint kMeshSetBase = 144;       // 256 emitter sets, 8 words each
public static const uint kMeshHeaderWords = 2192;
public static const uint kMeshMaxGroups = 32;
public static const uint kMeshMaxSets = 128;

public static const uint kMeshTriangleWords = 16;
public static const uint kMeshClusterWords = 32;
public static const uint kMeshNone = 0xFFFFFFFFu;

/// An emitter set's sync mode, as UsdLux's `light:materialSyncMode` names it.
public static const uint kMeshSyncTints = 0;        // materialGlowTintsLight
public static const uint kMeshSyncIndependent = 1;  // independent
public static const uint kMeshSyncNoMaterial = 2;   // noMaterialResponse

/// InstanceRecord.flags: bit 2 a MeshLightAPI prim, bits 3 to 7 its
/// emission group (0: none), bits 24 to 30 its emitter set (0: none), bit 31 that it may emit at all.
public static const uint kInstanceMeshLight = 4u;

public uint meshInstanceGroup(uint flags) { return (flags >> 3) & 31u; }
public uint meshInstanceSet(uint flags) { return (flags >> 24) & 127u; }

public uint meshWord(StructuredBuffer<float> w, uint at) { return asuint(w[at]); }

public float3 meshFloat3(StructuredBuffer<float> w, uint at) { return float3(w[at], w[at + 1], w[at + 2]); }

/// The clusters a kernel may draw from: none where the region is empty or
/// the setting is off.
public uint meshClusterCount(StructuredBuffer<float> w, uint words) {
    if (words < kMeshHeaderWords || meshWord(w, kMeshOn) == 0u) {
        return 0u;
    }
    return meshWord(w, kMeshClusterCount);
}

public float meshTotalPower(StructuredBuffer<float> w, uint words) {
    return meshClusterCount(w, words) > 0u ? max(w[kMeshTotalPower], 0.0) : 0.0;
}

/// A light group's weight on what its emitters give off (athenea:lightLevel).
public float3 meshGroupWeight(StructuredBuffer<float> w, uint words, uint group) {
    if (group == 0u || words < kMeshHeaderWords) {
        return float3(1.0);
    }
    return meshFloat3(w, kMeshGroupBase + 4u * min(group, kMeshMaxGroups - 1u));
}

/// WHAT A SURFACE GIVES OFF, given what its material said: the MeshLightAPI
/// prim's own emission by its sync mode, and its group's weight on either.
///
/// A MeshLightAPI prim emits its LightAPI radiance (colour, intensity,
/// exposure, temperature, normalize over its area) from its front faces, or
/// both where the gprim is double sided. `materialGlowTintsLight` multiplies
/// it by the material's emission where the material emits; `independent`
/// adds the two; `noMaterialResponse` drops the material's. One radiance is
/// both what a ray sees and what lights the scene, so every estimator --
/// the path tracer's two strategies, the raster's sample and closed form --
/// agrees on it.
public float3 meshLightEmission(StructuredBuffer<float> w, uint words, uint flags, bool backface,
                                float3 materialEmission) {
    if (words < kMeshHeaderWords) {
        return materialEmission;
    }
    const float3 weight = meshGroupWeight(w, words, meshInstanceGroup(flags));
    const uint set = meshInstanceSet(flags);
    if (set == 0u) {
        return materialEmission * weight;
    }
    const uint at = kMeshSetBase + 8u * set;
    const uint mode = meshWord(w, at + 3);
    const uint setFlags = meshWord(w, at + 4);
    const bool twoSided = (setFlags & 1u) != 0u;
    // The LightAPI's radiance, derived here as lights.slang derives a
    // light's: colour times intensity (the host's product), exposure, and
    // the blackbody of a colour temperature where it is enabled.
    float3 scale = meshFloat3(w, at) * exp2(w[at + 5]);
    if ((setFlags & 4u) != 0u) {
        scale *= blackbody(w[at + 6]);
    }
    const float3 light = backface && !twoSided ? float3(0.0) : scale;
    float3 out;
    if (mode == kMeshSyncIndependent) {
        out = materialEmission + light;
    } else if (mode == kMeshSyncNoMaterial) {
        out = light;
    } else {
        out = any(materialEmission > float3(0.0)) ? light * materialEmission : light;
    }
    return out * weight;
}

// ---- clusters -------------------------------------------------------------------

/// A cluster: its real triangles, and the rectangle that stands in for them.
public struct MeshCluster {
    public float3 centre;       // the rectangle's
    public float  power;        // luminance times area, one face, its group's weight in it
    public float3 axisX;        // unit, in its plane
    public float  halfX;
    public float3 axisY;
    public float  halfY;
    public float3 normal;       // unit: axisX x axisY
    public uint   flags;        // kLightTwoSided
    public float3 radiance;     // power over the rectangle's area, as rgb: what the rectangle gives off
    public float  area;         // the triangles' own, summed
    public uint   first;        // its first triangle's slot
    public uint   count;
    public uint   group;
    public uint   node;         // its leaf in the tree
    public float3 lo;           // the triangles' box
    public float  cosThetaO;    // its normals' cone (cos of the half angle; -1 all round)
    public float3 hi;
    public float  pad;
};

public uint meshClusterAt(StructuredBuffer<float> w, uint c) {
    return meshWord(w, kMeshClusterBase) + c * kMeshClusterWords;
}

public MeshCluster meshCluster(StructuredBuffer<float> w, uint c) {
    const uint at = meshClusterAt(w, c);
    MeshCluster m;
    m.centre = meshFloat3(w, at);
    m.power = w[at + 3];
    m.axisX = meshFloat3(w, at + 4);
    m.halfX = w[at + 7];
    m.axisY = meshFloat3(w, at + 8);
    m.halfY = w[at + 11];
    m.normal = meshFloat3(w, at + 12);
    m.flags = meshWord(w, at + 15);
    m.radiance = meshFloat3(w, at + 16);
    m.area = w[at + 19];
    m.first = meshWord(w, at + 20);
    m.count = meshWord(w, at + 21);
    m.group = meshWord(w, at + 22);
    m.node = meshWord(w, at + 23);
    m.lo = meshFloat3(w, at + 24);
    m.cosThetaO = w[at + 27];
    m.hi = meshFloat3(w, at + 28);
    m.pad = w[at + 31];
    return m;
}

/// A cluster's triangle: its corners in world space, its own power and the
/// power of its cluster up to and with it, the radiance it gives off (without
/// its group's weight) and which instance and triangle it is.
public struct MeshTriangle {
    public float3 a;
    public float  cumulative;
    public float3 b;
    public float  power;
    public float3 c;
    public uint   instance;
    public float3 radiance;
    public uint   triangle;
};

public uint meshTriangleAt(StructuredBuffer<float> w, uint slot) {
    return meshWord(w, kMeshTriangleBase) + slot * kMeshTriangleWords;
}

public MeshTriangle meshTriangle(StructuredBuffer<float> w, uint slot) {
    const uint at = meshTriangleAt(w, slot);
    MeshTriangle t;
    t.a = meshFloat3(w, at);
    t.cumulative = w[at + 3];
    t.b = meshFloat3(w, at + 4);
    t.power = w[at + 7];
    t.c = meshFloat3(w, at + 8);
    t.instance = meshWord(w, at + 11);
    t.radiance = meshFloat3(w, at + 12);
    t.triangle = meshWord(w, at + 15);
    return t;
}

/// A cluster as a LightRecord: the rectangle, shining down its own -Z as a
/// UsdLux rect does (so lights.slang's rect sampling, pdf and hit read it),
/// its radiance the cluster's power over the rectangle's area.
public LightRecord meshClusterRect(MeshCluster m) {
    LightRecord l;
    l.kind = kLightRect;
    l.flags = (m.flags & kLightTwoSided) | kLightShadow;
    l.sizeX = 2.0 * m.halfX;
    l.sizeY = 2.0 * m.halfY;
    l.colour = m.radiance;
    l.exposure = 0.0;
    l.temperature = 6500.0;
    l.coneCos = -1.0;
    l.coneSoftness = 0.0;
    l.texture = 0xFFFFFFFFu;
    l.sampler = 0u;
    l.lightCategory = kLightUnlinked;
    l.shadowCategory = kLightUnlinked;
    l.cumulative = m.power;
    l.ies = 0xFFFFFFFFu;
    l.iesAngleScale = 0.0;
    l.pad0 = m.node;
    l.group = m.group;
    // -Z is the normal, so Z = -normal; Y = Z x X keeps the frame right handed.
    const float3 z = -m.normal;
    const float3 x = m.axisX;
    const float3 y = cross(z, x);
    l.row0 = float4(x.x, y.x, z.x, m.centre.x);
    l.row1 = float4(x.y, y.y, z.y, m.centre.y);
    l.row2 = float4(x.z, y.z, z.z, m.centre.z);
    l.shadowColour = float3(0.0);
    l.shadowDistance = -1.0;
    l.shadowFalloff = -1.0;
    l.shadowFalloffGamma = 1.0;
    l.pad1 = 0u;
    l.pad2 = 0u;
    return l;
}

// ---- sampling a cluster's triangles ----------------------------------------------

public struct MeshLightSample {
    public float3 wi;          // towards the light, unit
    public float  distance;
    public float3 radiance;    // what the triangle gives off, its group's weight in it
    public float  pdf;         // per solid angle, the cluster given
    public float3 position;
    public uint   slot;        // the triangle's
    public float3 normal;      // the triangle's, facing the point
    public float  emitterCosine;
    public bool   valid;
};

/// The cluster's triangle that holds `target` of its power.
public uint meshPickTriangle(StructuredBuffer<float> w, MeshCluster m, float target) {
    uint lo = 0u;
    uint hi = m.count > 0u ? m.count - 1u : 0u;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        if (w[meshTriangleAt(w, m.first + mid) + 3] <= target) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return m.first + lo;
}

/// A point of cluster `c` seen from `p`: a triangle by its power, a point
/// uniform over it. The density is the triangle's share times the area
/// density turned into a solid angle one.
public MeshLightSample sampleMeshCluster(StructuredBuffer<float> w, uint words, uint c, float3 p, float3 u) {
    MeshLightSample s;
    s.valid = false;
    s.pdf = 0.0;
    s.distance = 0.0;
    s.wi = float3(0.0, 0.0, 1.0);
    s.radiance = float3(0.0);
    s.position = float3(0.0);
    s.normal = float3(0.0, 0.0, 1.0);
    s.emitterCosine = 1.0;
    s.slot = kMeshNone;
    const MeshCluster m = meshCluster(w, c);
    if (m.count == 0u || !(m.power > 0.0)) {
        return s;
    }
    const float total = w[meshTriangleAt(w, m.first + m.count - 1u) + 3];
    if (!(total > 0.0)) {
        return s;
    }
    const uint slot = meshPickTriangle(w, m, clamp(u.x, 0.0, 0.999999) * total);
    const MeshTriangle t = meshTriangle(w, slot);
    const float probability = t.power / total;
    if (!(probability > 0.0)) {
        return s;
    }
    const float root = sqrt(u.y);
    const float3 weights = float3(1.0 - root, u.z * root, root * (1.0 - u.z));
    const float3 at = t.a * weights.x + t.b * weights.y + t.c * weights.z;
    const float3 crossed = cross(t.b - t.a, t.c - t.a);
    const float twiceArea = length(crossed);
    const float3 offset = at - p;
    const float distance2 = dot(offset, offset);
    if (!(twiceArea > 0.0) || !(distance2 > 0.0)) {
        return s;
    }
    s.distance = sqrt(distance2);
    s.wi = offset / s.distance;
    float3 n = crossed / twiceArea;
    float cosine = dot(n, -s.wi);
    if ((m.flags & kLightTwoSided) == 0u && cosine <= 0.0) {
        return s;   // the back of a one-sided emitter
    }
    if (cosine < 0.0) {
        n = -n;
        cosine = -cosine;
    }
    if (cosine <= 1.0e-6) {
        return s;
    }
    s.pdf = probability * distance2 / (cosine * 0.5 * twiceArea);
    s.radiance = t.radiance * meshGroupWeight(w, words, m.group);
    s.position = at;
    s.normal = n;
    s.emitterCosine = cosine;
    s.slot = slot;
    s.valid = s.pdf > 0.0 && any(s.radiance > float3(0.0));
    return s;
}

/// The density `sampleMeshCluster` gives the point `at` on the triangle in
/// `slot`, from `p`: what a ray that met that triangle is weighed by.
public float meshTrianglePdf(StructuredBuffer<float> w, uint c, uint slot, float3 p, float3 at) {
    const MeshCluster m = meshCluster(w, c);
    if (m.count == 0u) {
        return 0.0;
    }
    const float total = w[meshTriangleAt(w, m.first + m.count - 1u) + 3];
    const MeshTriangle t = meshTriangle(w, slot);
    if (!(total > 0.0) || !(t.power > 0.0)) {
        return 0.0;
    }
    const float3 crossed = cross(t.b - t.a, t.c - t.a);
    const float twiceArea = length(crossed);
    const float3 offset = at - p;
    const float distance2 = dot(offset, offset);
    if (!(twiceArea > 0.0) || !(distance2 > 0.0)) {
        return 0.0;
    }
    const float cosine = dot(crossed / twiceArea, -offset / sqrt(distance2));
    if ((m.flags & kLightTwoSided) == 0u && cosine <= 0.0) {
        return 0.0;
    }
    if (abs(cosine) <= 1.0e-6) {
        return 0.0;
    }
    return (t.power / total) * distance2 / (abs(cosine) * 0.5 * twiceArea);
}

/// Where the ray from `p` along `wi` first meets one of cluster `c`'s
/// triangles, and the density sampleMeshCluster gives that point: what a
/// chi-square asks of a cluster with no triangle to name.
public float meshClusterPdf(StructuredBuffer<float> w, uint c, float3 p, float3 wi) {
    const MeshCluster m = meshCluster(w, c);
    float nearest = 1.0e30;
    uint hitSlot = kMeshNone;
    for (uint k = 0u; k < m.count; ++k) {
        const MeshTriangle t = meshTriangle(w, m.first + k);
        const float3 e1 = t.b - t.a;
        const float3 e2 = t.c - t.a;
        const float3 q = cross(wi, e2);
        const float det = dot(e1, q);
        if (abs(det) < 1.0e-20) {
            continue;
        }
        const float inverse = 1.0 / det;
        const float3 o = p - t.a;
        const float bu = dot(o, q) * inverse;
        const float3 r = cross(o, e1);
        const float bv = dot(wi, r) * inverse;
        const float tt = dot(e2, r) * inverse;
        if (bu < 0.0 || bv < 0.0 || bu + bv > 1.0 || !(tt > 0.0) || tt >= nearest) {
            continue;
        }
        nearest = tt;
        hitSlot = m.first + k;
    }
    if (hitSlot == kMeshNone) {
        return 0.0;
    }
    return meshTrianglePdf(w, c, hitSlot, p, p + wi * nearest);
}

// ---- the tree, and which triangle a hit was -------------------------------------

/// One cluster for the point p with normal n, by the tree's importance.
public LightChoice chooseMeshCluster(StructuredBuffer<float> w, uint words, float3 p, float3 n, float u) {
    LightChoice choice;
    choice.index = 0u;
    choice.probability = 0.0;
    choice.valid = false;
    const uint clusters = meshClusterCount(w, words);
    if (clusters == 0u) {
        return choice;
    }
    return chooseLightBvh(w, meshWord(w, kMeshNodeBase), meshWord(w, kMeshTreeNodes), p, n, u);
}

/// The probability chooseMeshCluster gives cluster `c` at p, n.
public float meshClusterChoicePdf(StructuredBuffer<float> w, uint words, uint c, float3 p, float3 n) {
    if (meshClusterCount(w, words) == 0u) {
        return 0.0;
    }
    return lightPdfChoiceBvh(w, meshWord(w, kMeshNodeBase), meshCluster(w, c).node, p, n);
}

/// The scene number of instance record `instance`'s triangle `triangle`.
public uint meshSceneTriangle(StructuredBuffer<float> w, uint instance, uint triangle) {
    if (instance >= meshWord(w, kMeshInstanceCount)) {
        return kMeshNone;
    }
    const uint base = meshWord(w, kMeshRecordBase);
    const uint first = meshWord(w, base + instance);
    return first + triangle < meshWord(w, base + instance + 1u) ? first + triangle : kMeshNone;
}

/// Where that triangle sits among the clusters' triangles, or kMeshNone where
/// it does not emit enough to be one: a binary search over the emitting
/// triangles' scene numbers.
public uint meshSlotOf(StructuredBuffer<float> w, uint words, uint instance, uint triangle) {
    if (meshClusterCount(w, words) == 0u) {
        return kMeshNone;
    }
    const uint g = meshSceneTriangle(w, instance, triangle);
    const uint count = meshWord(w, kMeshTriangleCount);
    if (g == kMeshNone || count == 0u) {
        return kMeshNone;
    }
    const uint base = meshWord(w, kMeshQualifiedBase);
    uint lo = 0u;
    uint hi = count;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2u;
        if (meshWord(w, base + mid) < g) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    if (lo >= count || meshWord(w, base + lo) != g) {
        return kMeshNone;
    }
    return meshWord(w, meshWord(w, kMeshSlotBase) + lo);
}

/// The cluster a slot belongs to: kept in the triangle's last word's
/// companion, the cluster index, beside the slot table.
public uint meshClusterOfSlot(StructuredBuffer<float> w, uint slot) {
    return meshWord(w, meshWord(w, kMeshSlotBase) + meshWord(w, kMeshTriangleCount) + slot);
}

/// THE DENSITY BOTH STRATEGIES WEIGH A POINT `at` ON TRIANGLE `triangle` OF
/// INSTANCE `instance` BY, seen from p: the triangle's share of the
/// clusters' power, the area turned to solid angle. 0 where the triangle is
/// not a light.
///
/// Not the tree's own density: MIS needs only that next event estimation and
/// the material's ray weigh a point by the same function, and the samples
/// keep their true density (`chooseMeshCluster`'s probability) as the
/// estimator's divisor. A second ascent of the tree in the path tracer's
/// kernel -- the UsdLux lights' is the first -- put Apple's Metal compiler
/// into an internal error (AGXMetalG17X, Code=3) on a UsdPreviewSurface
/// material; this costs no ascent at all.
public float meshLightPdfAt(StructuredBuffer<float> w, uint words, uint instance, uint triangle, float3 p, float3 n,
                            float3 at) {
    const uint slot = meshSlotOf(w, words, instance, triangle);
    if (slot == kMeshNone) {
        return 0.0;
    }
    const float total = meshTotalPower(w, words);
    const MeshTriangle t = meshTriangle(w, slot);
    if (!(total > 0.0) || !(t.power > 0.0)) {
        return 0.0;
    }
    const float3 crossed = cross(t.b - t.a, t.c - t.a);
    const float twiceArea = length(crossed);
    const float3 offset = at - p;
    const float distance2 = dot(offset, offset);
    const float cosine = twiceArea > 0.0 && distance2 > 0.0 ? abs(dot(crossed / twiceArea, offset / sqrt(distance2))) : 0.0;
    if (!(cosine > 1.0e-6)) {
        return 0.0;
    }
    return (t.power / total) * distance2 / (cosine * 0.5 * twiceArea);
}
