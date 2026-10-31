// Copyright (c) 2026 jesus luque.
//
// THE SOLID LAYER'S FIELD (SSF, docs/decisions.md "Solid splats (SSF)").
//
//   F(x) = sum_i w_i k(r_i(x)) - tau,   k(r) = (1 - r^2)^3 for r < 1, else 0
//   r_i^2 = (d.t1 / a)^2 + (d.t2 / b)^2 + (d.n / c)^2,   d = x - mu_i
//
// in the blob's frame (t1, t2, n): n its normal, t1 and t2 the tangent frame
// turned by theta about it from a frame n alone fixes (Duff et al. 2017), a
// and b its tangent half-axes, c its anchor's thickness. The gradient is
// analytic: grad k = -6 (1 - r^2)^2 (p.x / a t1 + p.y / b t2 + p.z / c n),
// p = (d.t1 / a, d.t2 / b, d.n / c). A blob reaches nothing past
// max(a, b, c), its support radius.
//
// NEIGHBOURS FROM THE MORTON CELLS (as lod_morton): blobs sorted by their
// centre's 30-bit code over the layer's bounds widened by the largest
// support radius; the level L is the finest whose cells are at least that
// radius wide, so a point's blobs are in its cell at L and the 26 around it,
// each a run of the sorted codes found by binary search. Nothing but the
// sort: no hash, no lists.
//
// Every kernel that imports this binds the same names (lod::SolidField::bind).
module ssf_field;
import athenea.common.morton;
import athenea.common.packing;

public StructuredBuffer<float4> blobPositions;   ///< xyz, weight
public StructuredBuffer<uint>   blobShape;       ///< 4 words a blob (packSolidBlob)
public StructuredBuffer<uint>   blobNormals;     ///< packNormal a blob
public StructuredBuffer<uint>   anchors;         ///< 4 words an anchor (packAnchor)
public StructuredBuffer<uint>   gridKeys;        ///< sorted Morton codes of the blobs' centres
public StructuredBuffer<uint>   gridOrder;       ///< the blob each sorted code is of
public StructuredBuffer<float>  gridInfo;        ///< kGrid* below

public static const uint kGridLoX = 0;
public static const uint kGridLoY = 1;
public static const uint kGridLoZ = 2;
public static const uint kGridExtent = 3;
public static const uint kGridLevel = 4;    ///< as a float, a whole number 0..10
public static const uint kGridRadius = 5;   ///< the largest support radius
public static const uint kGridCount = 6;    ///< blobs, as a float (exact to 2^24)
public static const uint kGridWords = 8;

/// One blob, ready for the field: its centre, weight and the inverse
/// half-axes along its frame.
public struct FieldBlob {
    public float3 centre;
    public float  weight;
    public float3 t1, t2, n;
    public float3 inverseAxes;   ///< 1/a, 1/b, 1/c
    public float  radius;        ///< max(a, b, c)
};

/// A frame n alone fixes (Duff, Burgess, Christensen et al., "Building an
/// orthonormal basis, revisited", 2017): continuous but at n.z = 0 crossing
/// to negative, where it flips; a blob's frame is turned by theta from it.
public void frameOf(float3 n, out float3 u, out float3 v) {
    const float s = n.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + n.z);
    const float b = n.x * n.y * a;
    u = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    v = float3(b, s + n.y * n.y * a, -n.y);
}

public FieldBlob fieldBlob(uint i) {
    const uint4 shape = uint4(blobShape[i * 4], blobShape[i * 4 + 1], blobShape[i * 4 + 2], blobShape[i * 4 + 3]);
    const SolidBlob s = unpackSolidBlob(blobPositions[i], shape);
    const uint a = s.anchor * 4;
    const SolidAnchor anchor =
        unpackAnchor(uint4(anchors[a], anchors[a + 1], anchors[a + 2], anchors[a + 3]));
    FieldBlob f;
    f.centre = s.position;
    f.weight = s.weight;
    f.n = unpackNormal(blobNormals[i]);
    float3 u, v;
    frameOf(f.n, u, v);
    const float c = cos(s.theta), sn = sin(s.theta);
    f.t1 = c * u + sn * v;
    f.t2 = -sn * u + c * v;
    f.inverseAxes = 1.0 / float3(s.radii, anchor.thickness);
    f.radius = max(max(s.radii.x, s.radii.y), anchor.thickness);
    return f;
}

/// The blob's support radius, for the grid.
public float supportRadius(uint i) {
    const uint4 shape = uint4(blobShape[i * 4], blobShape[i * 4 + 1], blobShape[i * 4 + 2], blobShape[i * 4 + 3]);
    const SolidBlob s = unpackSolidBlob(blobPositions[i], shape);
    return max(max(s.radii.x, s.radii.y), asfloat(anchors[s.anchor * 4]));
}

/// A blob's term at x: w k(r) and its gradient, added to (grad, value).
public void addBlob(FieldBlob f, float3 x, inout float4 field) {
    const float3 d = x - f.centre;
    const float3 p = float3(dot(d, f.t1), dot(d, f.t2), dot(d, f.n)) * f.inverseAxes;
    const float r2 = dot(p, p);
    if (r2 >= 1.0) {
        return;
    }
    const float s = 1.0 - r2;
    const float3 q = p * f.inverseAxes;   // A d in the frame, halved
    const float3 dr2 = q.x * f.t1 + q.y * f.t2 + q.z * f.n;
    field.xyz += f.weight * (-6.0 * s * s) * dr2;
    field.w += f.weight * s * s * s;
}

/// The first sorted code at or past `key`.
public uint gridLowerBound(uint key, uint count) {
    uint lo = 0, hi = count;
    while (lo < hi) {
        const uint mid = (lo + hi) / 2;
        if (gridKeys[mid] < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// (grad F, F) at x through the grid: -tau exactly and no gradient where no
/// blob reaches.
public float4 fieldAt(float3 x, float tau) {
    return fieldFacing(x, tau, float3(0.0), false);
}

/// The same, from the blobs facing the side `facing` faces alone (n . facing
/// > 0): one face's field, which the other face of a thin piece -- its
/// normals the other way -- does not reach into. What solidifying weighs,
/// moves and turns a blob by, so that two faces near each other stay two.
public float4 fieldFacing(float3 x, float tau, float3 facing, bool oneSide) {
    float4 field = float4(0.0);
    const uint count = uint(gridInfo[kGridCount]);
    const uint level = uint(gridInfo[kGridLevel]);
    const float3 lo = float3(gridInfo[kGridLoX], gridInfo[kGridLoY], gridInfo[kGridLoZ]);
    const float cells = float(1u << level);
    const float3 at = floor((x - lo) / gridInfo[kGridExtent] * cells);
    const int side = int(1u << level);
    const uint shift = 3 * (10 - level);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 cell = int3(at) + int3(dx, dy, dz);
                if (any(cell < 0) || any(cell >= side)) {
                    continue;
                }
                const uint code = cellCode(uint3(cell)) << shift;
                const uint end = level == 0 ? count : gridLowerBound(code + (1u << shift), count);
                for (uint j = gridLowerBound(code, count); j < end; ++j) {
                    const FieldBlob f = fieldBlob(gridOrder[j]);
                    if (!oneSide || dot(f.n, facing) > 0.0) {
                        addBlob(f, x, field);
                    }
                }
            }
        }
    }
    field.w -= tau;
    return field;
}

/// How far x is from the edge of any blob's support that reaches near it:
/// the least |r - 1| / max(1/a, 1/b, 1/c), a lower bound of the distance in
/// space, over the blobs of its 27 cells. The field is a polynomial, smooth,
/// wherever this exceeds a stencil's reach; at the edge k is only C2, which
/// is where differences stop being an oracle.
public float supportMargin(float3 x) {
    float margin = 3.4e38;
    const uint count = uint(gridInfo[kGridCount]);
    const uint level = uint(gridInfo[kGridLevel]);
    const float3 lo = float3(gridInfo[kGridLoX], gridInfo[kGridLoY], gridInfo[kGridLoZ]);
    const float3 at = floor((x - lo) / gridInfo[kGridExtent] * float(1u << level));
    const int side = int(1u << level);
    const uint shift = 3 * (10 - level);
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int3 cell = int3(at) + int3(dx, dy, dz);
                if (any(cell < 0) || any(cell >= side)) {
                    continue;
                }
                const uint code = cellCode(uint3(cell)) << shift;
                const uint end = level == 0 ? count : gridLowerBound(code + (1u << shift), count);
                for (uint j = gridLowerBound(code, count); j < end; ++j) {
                    const FieldBlob f = fieldBlob(gridOrder[j]);
                    const float3 d = x - f.centre;
                    const float3 p = float3(dot(d, f.t1), dot(d, f.t2), dot(d, f.n)) * f.inverseAxes;
                    const float most = max(f.inverseAxes.x, max(f.inverseAxes.y, f.inverseAxes.z));
                    margin = min(margin, abs(length(p) - 1.0) / most);
                }
            }
        }
    }
    return margin;
}

/// The same over every blob, no grid: what a test holds the grid to.
public float4 fieldEverywhere(float3 x, float tau, uint count) {
    float4 field = float4(0.0);
    for (uint i = 0; i < count; ++i) {
        addBlob(fieldBlob(i), x, field);
    }
    field.w -= tau;
    return field;
}

/// Taubin's distance F / |grad F|; 0 where there is no gradient.
public float taubinDistance(float4 field) {
    const float g = length(field.xyz);
    return g > 0.0 ? field.w / g : 0.0;
}

/// Polynomial smooth minimum (Quilez): min(a, b) - h^2 k / 4,
/// h = max(k - |a - b|, 0) / k; the plain minimum at k = 0. Symmetric, C1
/// where |a - b| = k. `dA` is its derivative by a.
public float smin(float a, float b, float k, out float dA) {
    if (k <= 0.0) {
        dA = a < b ? 1.0 : (a > b ? 0.0 : 0.5);
        return min(a, b);
    }
    const float h = max(k - abs(a - b), 0.0) / k;
    // d/da of min is 1 below b; of h^2 k / 4 is h/2 times d|a - b|/da.
    const float side = a < b ? 1.0 : (a > b ? 0.0 : 0.5);
    const float dh = a < b ? 0.5 * h : (a > b ? -0.5 * h : 0.0);
    dA = side - dh;
    return min(a, b) - h * h * k * 0.25;
}
