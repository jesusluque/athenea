// Copyright (c) 2026 jesus luque.
//
// The mesh lights' build (technique::MeshLights), in the order it runs:
//
//   meshCandidates   which instance records may emit, and their triangles
//   (probe)          each candidate triangle's radiance: MeshLights' generated
//                    module, which imports the materials
//   meshCompact      the triangles that emit enough, in scene order
//   meshCentroids    their centres, for the bounds the keys are cut in
//   meshKeys         group | sides | normal octant | Morton cell, then sorted
//   meshHeads        where the key changes: a cluster begins
//   meshPlace        each triangle at its slot, each cluster's first slot
//   meshCounts       each cluster's triangle count
//   meshTreeLeaves   the clusters' Morton codes, for the tree (bvh_hierarchy)
//   meshPack         the tree's links; light_bvh_build's lightParents fills
//                    the parents and lightSettle the boxes, cones and power
//   meshTreeDepth    how many settle passes a refit takes
//
// and every frame something moved or dimmed, with no host wait:
//
//   meshTriangleRefit  corners in world space, power with the group's weight
//   meshClusterRefit   the rectangle, the in-cluster shares, the leaf node
//   lightSettle x depth, then meshFinish: the total and the revision
//
// Membership -- which triangle is in which cluster -- is decided only here,
// on a rebuild, so a deforming emitter keeps its clusters and nothing
// flickers. The region's layout is mesh_lights.slang's.
module mesh_light_build;
import athenea.light.lights;
import athenea.light.light_bvh;
import athenea.light.mesh_lights;
import athenea.rt.bvh_common;
import athenea.common.morton;
import athenea.technique.surface;

RWStructuredBuffer<float>  words;            // the light words buffer, mesh region first
RWStructuredBuffer<uint>   candidateCounts;  // per instance record (+1): its triangles if it may emit
StructuredBuffer<uint>     candidateBases;   // their exclusive prefix: the records' first candidate
StructuredBuffer<float4>   probed;           // per candidate: radiance rgb
StructuredBuffer<uint>     probedFlags;      // per candidate: 1 where it emits enough
StructuredBuffer<uint>     probedOffsets;    // their exclusive prefix
RWStructuredBuffer<float4> compactRadiance;  // per emitting triangle, scene order
RWStructuredBuffer<float4> centroids;        // per emitting triangle
StructuredBuffer<float4>   bounds;           // bounds_reduce's: lo, hi
RWStructuredBuffer<uint>   keys;
RWStructuredBuffer<uint>   values;
StructuredBuffer<uint>     sortedKeys;
StructuredBuffer<uint>     sortedValues;
RWStructuredBuffer<uint>   heads;
StructuredBuffer<uint>     headOffsets;
RWStructuredBuffer<float4> leafBoxes;
StructuredBuffer<uint>     children;         // bvh_hierarchy's
StructuredBuffer<uint>     leaves;
RWStructuredBuffer<uint>   depth;            // [the deepest leaf's steps to the root]

struct MeshBuildParams {
    uint records;      // instance records
    uint candidates;   // candidate triangles
    uint triangles;    // emitting triangles
    uint clusters;
    uint nodeBase;     // the tree's first word in `words`
    uint pad0;
    uint pad1;
    uint pad2;
};
ConstantBuffer<MeshBuildParams> build;

static const float3 kBuildLuminance = float3(0.2126, 0.7152, 0.0722);

uint regionWord(uint at) { return asuint(words[at]); }
float3 regionFloat3(uint at) { return float3(words[at], words[at + 1], words[at + 2]); }
void putFloat3(uint at, float3 v) {
    words[at] = v.x;
    words[at + 1] = v.y;
    words[at + 2] = v.z;
}

uint triangleAt(uint slot) { return regionWord(kMeshTriangleBase) + slot * kMeshTriangleWords; }
uint clusterAt(uint c) { return regionWord(kMeshClusterBase) + c * kMeshClusterWords; }
float3 groupWeight(uint group) {
    return group == 0u ? float3(1.0) : regionFloat3(kMeshGroupBase + 4u * min(group, kMeshMaxGroups - 1u));
}

/// Instance record r's triangle in world space, with its frame's positions.
void recordTriangle(uint r, uint triangle, out float3 a, out float3 b, out float3 c) {
    const InstanceRecord record = instances[r];
    const MeshRecord m = meshes[record.mesh];
    const uint t = m.firstTriangle + triangle;
    const uint base = m.firstPoint + record.pointsOffset;
    a = rowsApply(record.world0, record.world1, record.world2, positions[base + indices[t * 3]].xyz, 1.0);
    b = rowsApply(record.world0, record.world1, record.world2, positions[base + indices[t * 3 + 1]].xyz, 1.0);
    c = rowsApply(record.world0, record.world1, record.world2, positions[base + indices[t * 3 + 2]].xyz, 1.0);
}

/// Which record a candidate is, by the bases' binary search.
uint recordOfCandidate(uint g) {
    uint lo = 0u;
    uint hi = build.records;
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if (candidateBases[mid] <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// The same, read from the region's copy of the bases.
uint recordOfSceneTriangle(uint g) {
    const uint base = regionWord(kMeshRecordBase);
    uint lo = 0u;
    uint hi = build.records;
    while (hi - lo > 1u) {
        const uint mid = (lo + hi) / 2u;
        if (regionWord(base + mid) <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Whether an emitter set says two-sided; a material's own emission is.
bool setTwoSided(uint flags) {
    const uint set = meshInstanceSet(flags);
    if (set == 0u) {
        return true;
    }
    return (regionWord(kMeshSetBase + 8u * set + 4u) & 1u) != 0u;
}

/// A record may emit where the host said so (bit 31: its material, a
/// subset's or a MeshLightAPI) and it stands at the shutter's first bucket:
/// under motion blur a record is drawn once a bucket, and each emitter is
/// counted once.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshCandidates(uint3 tid: SV_DispatchThreadID) {
    const uint r = tid.x;
    if (r > build.records) {
        return;
    }
    if (r == build.records) {
        candidateCounts[r] = 0u;
        return;
    }
    const InstanceRecord record = instances[r];
    const bool may = (record.flags & 0x80000000u) != 0u && (record.mask == 0xFFu || (record.mask & 1u) != 0u);
    candidateCounts[r] = may ? meshes[record.mesh].triangles : 0u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshCompact(uint3 tid: SV_DispatchThreadID) {
    const uint g = tid.x;
    if (g >= build.candidates || probedFlags[g] == 0u) {
        return;
    }
    const uint c = probedOffsets[g];
    words[regionWord(kMeshQualifiedBase) + c] = asfloat(g);
    compactRadiance[c] = probed[g];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshCentroids(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= build.triangles) {
        return;
    }
    const uint g = regionWord(regionWord(kMeshQualifiedBase) + c);
    const uint r = recordOfSceneTriangle(g);
    float3 a, b, d;
    recordTriangle(r, g - regionWord(regionWord(kMeshRecordBase) + r), a, b, d);
    centroids[c] = float4((a + b + d) / 3.0, 0.0);
}

/// Two triangles share a cluster when they share a group, sidedness, the
/// octant their normal points into and a cell of a 16^3 grid over the
/// emitters' bounds. A two-sided triangle's normal is turned to the side its
/// largest component is positive, so a sheet's two windings agree.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshKeys(uint3 tid: SV_DispatchThreadID) {
    const uint c = tid.x;
    if (c >= build.triangles) {
        return;
    }
    const uint g = regionWord(regionWord(kMeshQualifiedBase) + c);
    const uint r = recordOfSceneTriangle(g);
    const InstanceRecord record = instances[r];
    float3 a, b, d;
    recordTriangle(r, g - regionWord(regionWord(kMeshRecordBase) + r), a, b, d);
    float3 n = cross(b - a, d - a);
    const bool twoSided = setTwoSided(record.flags);
    if (twoSided) {
        const float3 m = abs(n);
        const float major = m.x >= m.y && m.x >= m.z ? n.x : (m.y >= m.z ? n.y : n.z);
        n = major < 0.0 ? -n : n;
    }
    const uint octant = (n.x >= 0.0 ? 4u : 0u) | (n.y >= 0.0 ? 2u : 0u) | (n.z >= 0.0 ? 1u : 0u);
    const float3 lo = bounds[0].xyz;
    const float3 hi = bounds[1].xyz;
    // Cubic cells, the bounds' longest side cut in 16: a flat row of
    // emitters is not cut finer across than along.
    const float side = max(max(hi.x - lo.x, hi.y - lo.y), max(hi.z - lo.z, 1.0e-20));
    const uint cell = morton30(((a + b + d) / 3.0 - lo) / side) >> 18;   // 12 bits: 16^3
    keys[c] = (meshInstanceGroup(record.flags) << 16) | ((twoSided ? 1u : 0u) << 15) | (octant << 12) | cell;
    values[c] = c;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshHeads(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= build.triangles) {
        return;
    }
    heads[i] = i == 0u || sortedKeys[i] != sortedKeys[i - 1u] ? 1u : 0u;
}

/// Each emitting triangle at its slot: what does not move between rebuilds
/// (which instance and triangle, the radiance), and which cluster it is in.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshPlace(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= build.triangles) {
        return;
    }
    const uint c = sortedValues[i];
    const uint cluster = headOffsets[i] + heads[i] - 1u;
    const uint g = regionWord(regionWord(kMeshQualifiedBase) + c);
    const uint r = recordOfSceneTriangle(g);
    const uint triangle = g - regionWord(regionWord(kMeshRecordBase) + r);
    words[regionWord(kMeshSlotBase) + c] = asfloat(i);
    words[regionWord(kMeshSlotBase) + build.triangles + i] = asfloat(cluster);
    const uint at = triangleAt(i);
    words[at + 11] = asfloat(r);
    putFloat3(at + 12, compactRadiance[c].rgb);
    words[at + 15] = asfloat(triangle);
    if (heads[i] != 0u) {
        const uint record = clusterAt(cluster);
        const uint flags = instances[r].flags;
        words[record + 15] = asfloat(setTwoSided(flags) ? kLightTwoSided : 0u);
        words[record + 20] = asfloat(i);
        words[record + 22] = asfloat(meshInstanceGroup(flags));
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshCounts(uint3 tid: SV_DispatchThreadID) {
    const uint cluster = tid.x;
    if (cluster >= build.clusters) {
        return;
    }
    const uint first = regionWord(clusterAt(cluster) + 20);
    const uint next = cluster + 1u < build.clusters ? regionWord(clusterAt(cluster + 1u) + 20)
                                                     : build.triangles;
    words[clusterAt(cluster) + 21] = asfloat(next - first);
    // Its leaf in the tree: internal nodes first, then a leaf per cluster.
    const uint internalCount = build.clusters > 1u ? build.clusters - 1u : 0u;
    words[clusterAt(cluster) + 23] = asfloat(internalCount + cluster);
}

// ---- refit ------------------------------------------------------------------------

/// A triangle where its instance stands this frame, and its power with its
/// group's weight: luminance times area, one face.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshTriangleRefit(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= build.triangles) {
        return;
    }
    const uint at = triangleAt(i);
    const uint r = regionWord(at + 11);
    float3 a, b, c;
    recordTriangle(r, regionWord(at + 15), a, b, c);
    putFloat3(at, a);
    putFloat3(at + 4, b);
    putFloat3(at + 8, c);
    const float area = 0.5 * length(cross(b - a, c - a));
    const float3 weight = groupWeight(meshInstanceGroup(instances[r].flags));
    words[at + 7] = max(dot(regionFloat3(at + 12) * weight, kBuildLuminance), 0.0) * area;
}

/// A cluster from its triangles: the shares accumulated in order, the
/// rectangle that stands in for them, and its leaf. The rectangle lies in the
/// plane of the area-weighted normal through the area-weighted centre, its
/// long side along the triangles' principal direction in that plane, its
/// extent their corners' -- and it gives off the cluster's power over its own
/// area, so it carries as much light as the triangles do however much larger
/// it is than they.
[shader("compute")]
[numthreads(64, 1, 1)]
void meshClusterRefit(uint3 tid: SV_DispatchThreadID) {
    const uint cluster = tid.x;
    if (cluster >= build.clusters) {
        return;
    }
    const uint record = clusterAt(cluster);
    const uint first = regionWord(record + 20);
    const uint count = regionWord(record + 21);
    const uint flags = regionWord(record + 15);
    const bool twoSided = (flags & kLightTwoSided) != 0u;
    float running = 0.0;
    float area = 0.0;
    float3 normalSum = float3(0.0);
    float3 centreSum = float3(0.0);
    float3 radianceSum = float3(0.0);
    float3 lo = float3(3.4e38);
    float3 hi = float3(-3.4e38);
    const float3 weight = groupWeight(regionWord(record + 22));
    for (uint k = 0u; k < count; ++k) {
        const uint at = triangleAt(first + k);
        const float3 a = regionFloat3(at);
        const float3 b = regionFloat3(at + 4);
        const float3 c = regionFloat3(at + 8);
        running += words[at + 7];
        words[at + 3] = running;
        float3 n = cross(b - a, c - a);
        const float twice = length(n);
        if (twoSided && dot(n, normalSum) < 0.0) {
            n = -n;
        }
        normalSum += n;
        area += 0.5 * twice;
        centreSum += (a + b + c) / 3.0 * (0.5 * twice);
        radianceSum += regionFloat3(at + 12) * weight * (0.5 * twice);
        lo = min(lo, min(a, min(b, c)));
        hi = max(hi, max(a, max(b, c)));
    }
    const float3 centre = area > 0.0 ? centreSum / area : 0.5 * (lo + hi);
    float3 normal = dot(normalSum, normalSum) > 0.0 ? normalize(normalSum) : float3(0.0, 0.0, 1.0);
    // The principal direction in the plane: the corners' covariance, its
    // larger eigenvector, by the 2x2 closed form.
    float3 tx, ty;
    lightFrame(normal, tx, ty);
    float sxx = 0.0, sxy = 0.0, syy = 0.0;
    float cosO = 1.0;
    for (uint k = 0u; k < count; ++k) {
        const uint at = triangleAt(first + k);
        for (uint v = 0u; v < 3u; ++v) {
            const float3 d = regionFloat3(at + 4u * v) - centre;
            const float x = dot(d, tx);
            const float y = dot(d, ty);
            sxx += x * x;
            sxy += x * y;
            syy += y * y;
        }
        const float3 a = regionFloat3(at);
        const float3 n = cross(regionFloat3(at + 4) - a, regionFloat3(at + 8) - a);
        if (dot(n, n) > 0.0) {
            cosO = min(cosO, twoSided ? abs(dot(normalize(n), normal)) : dot(normalize(n), normal));
        }
    }
    const float angle = 0.5 * atan2(2.0 * sxy, sxx - syy);
    const float3 axisX = normalize(tx * cos(angle) + ty * sin(angle));
    const float3 axisY = cross(normal, axisX);
    float2 extentLo = float2(3.4e38);
    float2 extentHi = float2(-3.4e38);
    for (uint k = 0u; k < count; ++k) {
        const uint at = triangleAt(first + k);
        for (uint v = 0u; v < 3u; ++v) {
            const float3 d = regionFloat3(at + 4u * v) - centre;
            const float2 q = float2(dot(d, axisX), dot(d, axisY));
            extentLo = min(extentLo, q);
            extentHi = max(extentHi, q);
        }
    }
    if (count == 0u) {
        extentLo = float2(0.0);
        extentHi = float2(0.0);
    }
    // A cluster whose triangles are all in one plane is that plane; one that
    // curves keeps the plane through its centre. Half sizes held off zero, so
    // a line of light still has a rectangle to integrate.
    const float2 mid = 0.5 * (extentLo + extentHi);
    const float2 half = max(0.5 * (extentHi - extentLo), float2(1.0e-5 * max(length(hi - lo), 1.0e-6)));
    const float3 rectCentre = centre + axisX * mid.x + axisY * mid.y;
    const float rectArea = 4.0 * half.x * half.y;
    putFloat3(record, rectCentre);
    words[record + 3] = running;
    putFloat3(record + 4, axisX);
    words[record + 7] = half.x;
    putFloat3(record + 8, axisY);
    words[record + 11] = half.y;
    putFloat3(record + 12, normal);
    putFloat3(record + 16, rectArea > 0.0 ? radianceSum / rectArea : float3(0.0));
    words[record + 19] = area;
    putFloat3(record + 24, lo);
    // Two-sided: light leaves both ways, a cone all round.
    words[record + 27] = twoSided ? -1.0 : max(cosO - 1.0e-4, -1.0);
    putFloat3(record + 28, hi);
    // Its leaf: the box, the power, the normals' cone, a hemisphere of emission.
    LightNode node;
    node.lo = float4(lo, max(running, 0.0));
    node.hi = float4(hi, twoSided ? -1.0 : max(cosO - 1.0e-4, -1.0));
    node.cone = float4(normal, 0.0);
    node.links = uint4(kLeafBit | cluster, kLeafBit | cluster, regionWord(build.nodeBase + regionWord(record + 23) * 16u + 14u),
                       1u);
    storeLightNode(words, build.nodeBase, regionWord(record + 23), node);
}

/// The frame's total, once the tree has settled, and the revision.
[shader("compute")]
[numthreads(1, 1, 1)]
void meshFinish(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0u) {
        return;
    }
    words[kMeshTotalPower] = build.clusters > 0u ? words[build.nodeBase + 3u] : 0.0;
    words[kMeshRevision] = asfloat(regionWord(kMeshRevision) + 1u);
}

// ---- the tree -------------------------------------------------------------------------

[shader("compute")]
[numthreads(256, 1, 1)]
void meshTreeLeaves(uint3 tid: SV_DispatchThreadID) {
    const uint cluster = tid.x;
    if (cluster >= build.clusters) {
        return;
    }
    const uint record = clusterAt(cluster);
    const float3 lo = bounds[0].xyz;
    const float3 hi = bounds[1].xyz;
    keys[cluster] = morton30((regionFloat3(record) - lo) / max(hi - lo, float3(1.0e-20)));
    values[cluster] = cluster;
}

/// The tree's links from the hierarchy: internal node u at u, the leaf of
/// cluster c at (clusters - 1) + c. Boxes, cones and power are lightSettle's.
[shader("compute")]
[numthreads(256, 1, 1)]
void meshPack(uint3 tid: SV_DispatchThreadID) {
    const uint n = build.clusters;
    const uint u = tid.x;
    const uint internalCount = n > 1u ? n - 1u : 0u;
    if (u >= internalCount + n) {
        return;
    }
    LightNode node;
    node.lo = float4(0.0);
    node.hi = float4(0.0, 0.0, 0.0, 1.0);
    node.cone = float4(0.0, 0.0, 1.0, 0.0);
    if (u < internalCount) {
        uint2 kids;
        for (uint k = 0u; k < 2u; ++k) {
            const uint c = children[u * 2u + k];
            kids[k] = (c & kLeafBit) != 0u ? internalCount + leaves[c & ~kLeafBit] : c;
        }
        node.links = uint4(kids.x, kids.y, kNoNode, 0u);
    } else {
        const uint cluster = u - internalCount;
        node.links = uint4(kLeafBit | cluster, kLeafBit | cluster, kNoNode, 1u);
    }
    storeLightNode(words, build.nodeBase, u, node);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void meshTreeDepth(uint3 tid: SV_DispatchThreadID) {
    const uint cluster = tid.x;
    if (cluster >= build.clusters) {
        return;
    }
    const uint internalCount = build.clusters > 1u ? build.clusters - 1u : 0u;
    uint node = internalCount + cluster;
    uint steps = 0u;
    for (uint k = 0u; k < 64u; ++k) {
        const uint parent = regionWord(build.nodeBase + node * 16u + 14u);
        if (parent == kNoNode) {
            break;
        }
        node = parent;
        ++steps;
    }
    InterlockedMax(depth[0], steps);
}
