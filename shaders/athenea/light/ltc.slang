// Copyright (c) 2026 jesus luque.
//
// AREA LIGHT IN CLOSED FORM: what a polygon of uniform radiance gives a
// diffuse lobe (its form factor) and a GGX lobe (a linearly transformed
// cosine, Heitz, Dupuy, Hill and Neubelt 2016), with the polygon clipped to
// the horizon by the sphere it is equivalent to (Hill and Heitz 2017) -- no
// clipping code, no branch per vertex.
//
// The LTC table is not the published one: its licence was not a plain one
// to read, so ltc_fit.slang fits the same parametrisation on the device when
// a process first needs it (light::LtcTable). A texel at (roughness,
// sqrt(1 - cos theta)) holds the inverse transform's four free entries.
//
// Everything here works on directions from the shading point: a polygon is
// its corners minus the point, in world space. Nothing is declared global.
module ltc;

static const float kLtcPi = 3.14159265358979;

/// The edge's share of the polygon's vector form factor: theta / (2 pi)
/// times the edge's unit normal, with theta / (2 pi sin(theta)) by Hill's
/// rational fit, which holds near theta = 0 and pi where acos and a division
/// do not. Summed over the edges it is the form factor's vector F: F.z is
/// the cosine's integral over pi, |F| the sin^2 of the equivalent sphere.
public float3 ltcEdge(float3 v1, float3 v2) {
    const float x = dot(v1, v2);
    const float y = abs(x);
    const float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    const float b = 3.4175940 + (4.1616724 + y) * y;
    const float v = a / b;
    const float thetaOverSin = x > 0.0 ? v : 0.5 * rsqrt(max(1.0 - x * x, 1.0e-7)) - v;
    return cross(v1, v2) * thetaOverSin;
}

/// The integral of the cosine about +z over the cap a sphere of angular
/// radius sigma subtends at elevation theta, clipped by the horizon
/// (Snyder 1996, as Frostbite writes it): pi sin^2(sigma) cos(theta) where the
/// cap is whole above it.
public float ltcSphereIntegral(float cosTheta, float sinSigma2) {
    const float s2 = clamp(sinSigma2, 0.0, 1.0);
    if (cosTheta * cosTheta > s2) {
        return kLtcPi * s2 * saturate(cosTheta);
    }
    if (s2 <= 0.0) {
        return 0.0;
    }
    const float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    const float x = sqrt(max(1.0 / s2 - 1.0, 0.0));
    const float y = clamp(-x * cosTheta / max(sinTheta, 1.0e-7), -1.0, 1.0);
    const float sinThetaSqrtY = sinTheta * sqrt(max(1.0 - y * y, 0.0));
    return max((cosTheta * acos(y) - x * sinThetaSqrtY) * s2 + atan(sinThetaSqrtY / max(x, 1.0e-7)), 0.0);
}

/// THE COSINE INTEGRAL OVER A QUAD seen through a transform: the corners
/// (relative to the point) taken by `m` into the cosine's space, normalised,
/// their edges' vector form factor summed, and the horizon clipped by the
/// sphere of the same vector. `twoSided`: a quad seen from behind counts too.
/// Divided by pi it is the fraction of a normalised cosine lobe the quad
/// holds.
public float ltcQuad(float3x3 m, float3 p0, float3 p1, float3 p2, float3 p3, bool twoSided) {
    const float3 l0 = normalize(mul(m, p0));
    const float3 l1 = normalize(mul(m, p1));
    const float3 l2 = normalize(mul(m, p2));
    const float3 l3 = normalize(mul(m, p3));
    float3 f = ltcEdge(l0, l1) + ltcEdge(l1, l2) + ltcEdge(l2, l3) + ltcEdge(l3, l0);
    const float len = length(f);
    if (!(len > 1.0e-9)) {
        return 0.0;
    }
    // The sum points along the winding's normal; the cosine's vector
    // integral points at the quad, whichever way it winds.
    if (dot(f, l0 + l1 + l2 + l3) < 0.0) {
        f = -f;
    }
    // Seen from behind, a one-sided quad gives nothing: the face normal by
    // the corners as authored, against the way back to the point.
    const float3 faceNormal = cross(p1 - p0, p3 - p0);
    if (!twoSided && dot(faceNormal, -p0) < 0.0) {
        return 0.0;
    }
    return ltcSphereIntegral(f.z / len, len);
}

/// A shading frame about `n` with `toEye` in its xz plane: rows that take a
/// world direction into it.
public float3x3 ltcFrame(float3 n, float3 toEye) {
    float3 t1 = toEye - n * dot(toEye, n);
    if (dot(t1, t1) < 1.0e-12) {
        t1 = abs(n.x) < 0.9 ? cross(n, float3(1.0, 0.0, 0.0)) : cross(n, float3(0.0, 1.0, 0.0));
    }
    t1 = normalize(t1);
    const float3 t2 = cross(n, t1);
    return float3x3(t1, t2, n);
}

/// The inverse transform from its table entries, in the shading frame: the
/// fit's four free entries, the rest of the matrix fixed.
public float3x3 ltcInverse(float4 t) {
    return float3x3(float3(t.x, 0.0, t.y), float3(0.0, t.z, 0.0), float3(t.w, 0.0, 1.0));
}

/// The table's coordinates for a GGX alpha and the cosine to the eye.
public float2 ltcCoordinates(float alpha, float cosTheta, uint size) {
    const float r = sqrt(saturate(alpha));
    const float t = sqrt(saturate(1.0 - cosTheta));
    const float scale = float(size - 1u) / float(size);
    const float bias = 0.5 / float(size);
    return float2(r, t) * scale + bias;
}
