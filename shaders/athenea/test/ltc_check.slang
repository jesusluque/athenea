// Copyright (c) 2026 jesus luque.
//
// The LTC table against what it stands for (tests/technique/test_mesh_lights.cpp):
// for a rectangle seen from a point, the GGX lobe (no Fresnel, over its
// albedo) integrated over the rectangle by quadrature, against the table's
// closed form; and a diffuse rectangle's form factor against the same
// quadrature of the cosine. One thread a case.
module ltc_check;
import athenea.light.ltc;

Texture2D<float4>          ltcTable;
RWStructuredBuffer<float4> results;   // a case: closed form, quadrature, diffuse closed, diffuse quadrature
uniform uint               cases;

static const float kCheckPi = 3.14159265358979;

float ggxD(float alpha, float cosH) {
    const float a2 = alpha * alpha;
    const float d = cosH * cosH * (a2 - 1.0) + 1.0;
    return a2 / (kCheckPi * d * d);
}
float lambda(float alpha, float c) {
    const float c2 = c * c;
    return 0.5 * (-1.0 + sqrt(1.0 + alpha * alpha * max(1.0 - c2, 0.0) / max(c2, 1.0e-12)));
}
float brdfCos(float3 v, float3 l, float alpha) {
    if (l.z <= 0.0 || v.z <= 0.0) {
        return 0.0;
    }
    const float3 h = normalize(v + l);
    const float g2 = 1.0 / (1.0 + lambda(alpha, v.z) + lambda(alpha, l.z));
    return ggxD(alpha, h.z) * g2 / (4.0 * v.z);
}

float4 tableAt(float alpha, float cosTheta) {
    const uint size = 64u;
    const float2 uv = ltcCoordinates(alpha, cosTheta, size) * float(size) - 0.5;
    const int2 i0 = clamp(int2(floor(uv)), int2(0), int2(int(size) - 1));
    const int2 i1 = min(i0 + 1, int2(int(size) - 1));
    const float2 f = saturate(uv - float2(i0));
    const float4 a = lerp(ltcTable.Load(int3(i0.x, i0.y, 0)), ltcTable.Load(int3(i1.x, i0.y, 0)), f.x);
    const float4 b = lerp(ltcTable.Load(int3(i0.x, i1.y, 0)), ltcTable.Load(int3(i1.x, i1.y, 0)), f.x);
    return lerp(a, b, f.y);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void ltcCheck(uint3 tid: SV_DispatchThreadID) {
    const uint k = tid.x;
    if (k >= cases) {
        return;
    }
    // Cases: roughness and the eye's angle on a grid, a rectangle above and
    // to the side of the mirror direction.
    const float alpha = lerp(0.08, 0.8, float(k % 4u) / 3.0);
    const float theta = lerp(0.1, 1.2, float(k / 4u) / 3.0);
    const float3 v = float3(sin(theta), 0.0, cos(theta));
    const float3 centre = float3(-1.2 + 0.3 * float(k % 3u), 0.4, 2.0);
    const float3 ax = float3(0.6, 0.0, 0.0);
    const float3 ay = float3(0.0, 0.4, 0.2);
    const float3 p0 = centre - ax - ay;
    const float3 p1 = centre + ax - ay;
    const float3 p2 = centre + ax + ay;
    const float3 p3 = centre - ax + ay;
    // The frame is the identity: n is +z and the eye in the xz plane.
    const float3x3 identity = float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    const float3x3 m = mul(ltcInverse(tableAt(alpha, v.z)), identity);
    const float closed = ltcQuad(m, p0, p1, p2, p3, true) / kCheckPi;
    const float diffuseClosed = ltcQuad(identity, p0, p1, p2, p3, true) / kCheckPi;
    // Quadrature over the rectangle: the lobe over its albedo, the area turned
    // to solid angle.
    const uint n = 256u;
    float sum = 0.0;
    float diffuse = 0.0;
    const float area = 4.0 * length(ax) * length(ay);
    const float3 normal = normalize(cross(ax, ay));
    for (uint j = 0u; j < n; ++j) {
        for (uint i = 0u; i < n; ++i) {
            const float2 u = (float2(i, j) + 0.5) / float(n) * 2.0 - 1.0;
            const float3 q = centre + ax * u.x + ay * u.y;
            const float d2 = dot(q, q);
            const float3 l = q / sqrt(d2);
            const float dw = area / float(n * n) * abs(dot(normal, l)) / d2;
            sum += brdfCos(v, l, alpha) * dw;
            diffuse += max(l.z, 0.0) / kCheckPi * dw;
        }
    }
    // The lobe's albedo, the same quadrature over the hemisphere.
    float albedo = 0.0;
    const uint h = 256u;
    for (uint j = 0u; j < h; ++j) {
        for (uint i = 0u; i < h; ++i) {
            const float ct = (float(j) + 0.5) / float(h);
            const float phi = (float(i) + 0.5) / float(h) * 2.0 * kCheckPi;
            const float st = sqrt(1.0 - ct * ct);
            const float3 l = float3(st * cos(phi), st * sin(phi), ct);
            albedo += brdfCos(v, l, alpha) * (2.0 * kCheckPi / float(h * h));
        }
    }
    results[k] = float4(closed, sum / max(albedo, 1.0e-9), diffuseClosed, diffuse);
}
