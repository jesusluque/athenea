// Copyright (c) 2026 jesus luque.
//
// WHAT THE MESH LIGHTS PUT ON A RELIT GAUSSIAN (stage C): each entry of the
// lightcut of the cell the gaussian stands in (light/mesh_light_splat.slang)
// is a rectangle of uniform radiance, integrated in closed form by the
// gaussian's layers (`relitByRect`): its body by the form factor, the horizon
// clipped by the sphere of the same vector (ltc.slang), its base's and its
// coat's reflections by the LTC fitted to GGX, its sheen as a body. A TX
// transfer shadows each entry by its own bits over the cone of that sphere,
// and the indirect half is read along the sphere's direction, as a UsdLux
// light's is.
//
// Nothing here is a binding: the caller passes the textures.
module splat_mesh_lights;
import athenea.light.lights;
import athenea.light.ltc;
import athenea.light.mesh_lights;
__exported import athenea.splat.splat_relight;

public struct SplatMeshLights {
    public uint   cellSide;   // 0: no mesh light reaches the splats
    public uint   cutK;
    public float3 gridLo;
    public float3 gridHi;
};

/// The LTC table's texel for a GGX alpha and the cosine to the eye, bilinear.
float4 splatLtcAt(Texture2D<float4> table, float alpha, float cosTheta) {
    const uint size = 64u;
    const float2 uv = ltcCoordinates(alpha, cosTheta, size) * float(size) - 0.5;
    const int2 i0 = clamp(int2(floor(uv)), int2(0), int2(int(size) - 1));
    const int2 i1 = min(i0 + 1, int2(int(size) - 1));
    const float2 f = saturate(uv - float2(i0));
    const float4 a = lerp(table.Load(int3(i0.x, i0.y, 0)), table.Load(int3(i1.x, i0.y, 0)), f.x);
    const float4 b = lerp(table.Load(int3(i0.x, i1.y, 0)), table.Load(int3(i1.x, i1.y, 0)), f.x);
    return lerp(a, b, f.y);
}

/// A RECTANGLE OF UNIFORM RADIANCE ON A GAUSSIAN, in closed form, by the
/// layers `relitByOne` weighs a light's sample with: the body by the
/// rectangle's form factor (and through the gaussian by the form factor seen
/// from behind, for what transmits), the base's reflection and the coat's by
/// the LTC fitted to GGX at their roughness times their albedo in reflection,
/// the sheen as a body of its colour. `p0` to `p3` are the corners less the
/// gaussian's position.
float3 relitByRect(SplatSurface s, float3 wo, float3 p0, float3 p1, float3 p2, float3 p3, bool twoSided,
                   Texture2D<float4> ltcTable) {
    const float3 n = s.normal;
    const float nDotV = max(dot(n, wo), 1.0e-4);
    const SplatLayers layers = splatLayers(s, nDotV);
    const float3x3 front = ltcFrame(n, wo);
    const float formFront = ltcQuad(front, p0, p1, p2, p3, twoSided) / 3.14159265358979;
    const float formBack = transmittedWeight(s) > 0.0
                               ? ltcQuad(ltcFrame(-n, wo), p0, p1, p2, p3, twoSided) / 3.14159265358979
                               : 0.0;
    const float3 albedo = s.lit ? float3(0.0) : s.albedo * (1.0 - s.metallic);
    float3 out = layers.under * albedo * ((1.0 - s.transmission) * formFront + transmittedWeight(s) * formBack);
    // The base's reflection: the LTC at its roughness (alpha = r^2), times
    // what it reflects of everything (the split-sum albedo the sky uses).
    const float alpha = clamp(s.roughness * s.roughness, 4.0e-4, 1.0);
    const float3x3 base = mul(ltcInverse(splatLtcAt(ltcTable, alpha, nDotV)), front);
    out += layers.under * thinWallGain(s) * environmentBrdfOf(s, nDotV) * ltcQuad(base, p0, p1, p2, p3, twoSided) /
           3.14159265358979;
    if (s.lobes.coatWeight > 0.0) {
        const float coatAlpha = clamp(s.lobes.coatRoughness * s.lobes.coatRoughness, 4.0e-4, 1.0);
        const float3x3 coat = mul(ltcInverse(splatLtcAt(ltcTable, coatAlpha, nDotV)), front);
        out += layers.coat * ltcQuad(coat, p0, p1, p2, p3, twoSided) / 3.14159265358979;
    }
    out += layers.belowCoat * layers.sheen * formFront;
    return out;
}

public float3 relitByMeshLights<let kTransfer : int>(SplatSurface s, float3 wo, SplatMeshLights grid,
                                                     Texture2D<float4> rects, Texture2D<uint2> cells,
                                                     Texture2D<float4> ltcTable, StructuredBuffer<uint> transfer,
                                                     StructuredBuffer<uint> cellBits, bool withIndirect) {
    float3 lit = float3(0.0);
    // A cloud whose colours are light already keeps them: the lamps it was
    // baked under are in them.
    if (grid.cellSide == 0u || s.lit) {
        return lit;
    }
    const float3 unit = saturate((s.position - grid.gridLo) / max(grid.gridHi - grid.gridLo, float3(1.0e-9)));
    const uint3 cell = min(uint3(unit * float(grid.cellSide)), uint3(grid.cellSide - 1u));
    const uint2 at = uint2(cell.x * grid.cutK, cell.y * grid.cellSide + cell.z);
    for (uint e = 0u; e < grid.cutK; ++e) {
        const uint2 entry = cells.Load(int3(int(at.x + e), int(at.y), 0));
        if (entry.x == kMeshNone) {
            break;
        }
        const float factor = asfloat(entry.y);
        const int2 r = int2(int((entry.x % 256u) * 4u), int(entry.x / 256u));
        const float4 t0 = rects.Load(int3(r, 0));
        const float4 t1 = rects.Load(int3(r + int2(1, 0), 0));
        const float4 t2 = rects.Load(int3(r + int2(2, 0), 0));
        const float4 t3 = rects.Load(int3(r + int2(3, 0), 0));
        const bool twoSided = (asuint(t0.w) & kLightTwoSided) != 0u;
        const float3 q = t0.xyz - s.position;
        if (!twoSided && dot(cross(t1.xyz, t2.xyz), -q) <= 0.0) {
            continue;   // behind a one-sided emitter
        }
        const float3 p0 = q - t1.xyz - t2.xyz;
        const float3 p1 = q + t1.xyz - t2.xyz;
        const float3 p2 = q + t1.xyz + t2.xyz;
        const float3 p3 = q - t1.xyz + t2.xyz;
        // The direction and the size of the sphere of the same form factor:
        // what the transfer's bits are asked along, and the bounce read along.
        const float3 l0 = normalize(p0);
        const float3 l1 = normalize(p1);
        const float3 l2 = normalize(p2);
        const float3 l3 = normalize(p3);
        float3 f = ltcEdge(l0, l1) + ltcEdge(l1, l2) + ltcEdge(l2, l3) + ltcEdge(l3, l0);
        if (dot(f, l0 + l1 + l2 + l3) < 0.0) {
            f = -f;
        }
        const float len = length(f);
        if (!(len > 1.0e-9)) {
            continue;
        }
        const float3 wi = f / len;
        const float sin2 = saturate(len);
        float through = 1.0;
        if (kTransfer >= 2 && s.cellWords >= 8u) {
            const float open = splatConeOpen(s, cellBits, wi, asin(sqrt(sin2)), s.normal);
            through = open >= 0.0 ? open : 1.0;
        }
        const float3 radiance = t3.rgb * factor;
        lit += through * radiance * relitByRect(s, wo, p0, p1, p2, p3, twoSided, ltcTable);
        if (kTransfer >= 2 && withIndirect) {
            LightSample ls;
            ls.wi = wi;
            ls.distance = length(q);
            ls.radiance = radiance * (3.14159265358979 * sin2);
            ls.pdf = 1.0;
            ls.delta = false;
            ls.valid = true;
            ls.emitterCosine = 1.0;
            lit += splatLightBounce(s, wo, ls, transfer, cellBits);
        }
    }
    return lit;
}
