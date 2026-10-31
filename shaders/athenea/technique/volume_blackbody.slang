// Copyright (c) 2026 jesus luque.
//
// A BLACK BODY'S LIGHT, AS A TABLE OVER TEMPERATURE (proposal 103 §2, 105 §3):
// what a medium's gaussian at temperature T kelvin glows with.
//
// Each entry is Planck's law integrated against the CIE 1931 colour-matching
// functions -- Wyman, Sloan and Shirley's multi-lobe fit (JCGT 2013) -- from
// 360 to 830 nm, taken to linear Rec.709 and normalised to luminance one,
// then scaled as Cycles' Principled Volume scales its blackbody: sigma_SB T^4 /
// pi * 1e-6, a radiance a scene of unit lights shows at a fire's few thousand
// kelvin. One thread an entry, `size` entries from 0 to `maxKelvin`; the
// lookup is linear between them (volume_splat_shade.slang).
module volume_blackbody;

struct BlackbodyParams {
    uint  size;
    float maxKelvin;
};
RWStructuredBuffer<float4>       blackbodyOut;
ConstantBuffer<BlackbodyParams>  blackbodyParams;

/// A lobe of the fit: a gaussian with one width left of its peak and another
/// right of it.
float lobe(float x, float mean, float left, float right) {
    const float t = (x - mean) / (x < mean ? left : right);
    return exp(-0.5 * t * t);
}

float3 cie(float nm) {
    const float x = 1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) -
                    0.065 * lobe(nm, 501.1, 20.4, 26.2);
    const float y = 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1);
    const float z = 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8);
    return float3(x, y, z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeBlackbody(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    const BlackbodyParams p = blackbodyParams;
    if (i >= p.size) {
        return;
    }
    const float kelvin = p.maxKelvin * float(i) / float(max(p.size - 1, 1u));
    if (kelvin < 300.0) {
        blackbodyOut[i] = float4(0.0);
        return;
    }
    // Planck's spectral radiance up to a constant, which the normalisation
    // takes away: (lambda / 500)^-5 / (exp(c2 / (lambda T)) - 1), c2 = hc / k
    // in nm K.
    static const float c2 = 1.4387769e7;
    float3 xyz = float3(0.0);
    for (uint s = 0; s <= 94; ++s) {
        const float nm = 360.0 + 5.0 * float(s);
        const float e = c2 / (nm * kelvin);
        if (e > 80.0) {
            continue;
        }
        const float r = nm / 500.0;
        const float planck = 1.0 / (r * r * r * r * r * (exp(e) - 1.0));
        xyz += planck * cie(nm);
    }
    const float3 rgb = max(float3(dot(float3(3.2404542, -1.5371385, -0.4985314), xyz),
                                  dot(float3(-0.9692660, 1.8760108, 0.0415560), xyz),
                                  dot(float3(0.0556434, -0.2040259, 1.0572252), xyz)),
                           float3(0.0));
    const float luminance = dot(float3(0.2126, 0.7152, 0.0722), rgb);
    if (!(luminance > 0.0)) {
        blackbodyOut[i] = float4(0.0);
        return;
    }
    // sigma_SB / pi * 1e-6 T^4, in two steps so T^4 stays in range.
    const float t2 = kelvin * kelvin * 1.0e-6;
    const float scale = 5.670374e-8 / 3.14159265358979 * t2 * t2 * 1.0e6;
    blackbodyOut[i] = float4(rgb / luminance * scale, kelvin);
}
