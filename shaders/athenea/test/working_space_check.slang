// Copyright (c) 2026 jesus luque.
//
// The working-space record the device binds, against the constants the
// kernels held before it: the record's words one by one, and every function
// that reads it against the expression it replaced, written here again with
// its literals -- compared by their bits, since the frame must not move by
// one. A count of what differs, nothing else.
import athenea.common.working_space;
import athenea.common.color;
import athenea.common.edit;
import athenea.light.lights;

RWStructuredBuffer<uint> counts;   // [record words, blackbody, cloudLight, editGrade, luminance, fromXyz, threads]
uniform uint             count;

static const float3 kOldLuminance = float3(0.2126, 0.7152, 0.0722);

/// light/lights.slang's blackbody as it was: XYZ to linear sRGB by its own
/// rows, BT.709's weights.
float3 oldBlackbody(float kelvin) {
    const float t = clamp(kelvin, 1000.0, 15000.0);
    const float t2 = t * t;
    const float u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t2) /
                    (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t2);
    const float v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t2) /
                    (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t2);
    const float denominator = 2.0 * u - 8.0 * v + 4.0;
    const float x = 3.0 * u / denominator;
    const float y = 2.0 * v / denominator;
    const float z = 1.0 - x - y;
    const float3 xyz = float3(x / max(y, 1e-6), 1.0, z / max(y, 1e-6));
    const float3 rgb = float3(dot(float3(3.2404542, -1.5371385, -0.4985314), xyz),
                              dot(float3(-0.9692660, 1.8760108, 0.0415560), xyz),
                              dot(float3(0.0556434, -0.2040259, 1.0572252), xyz));
    const float3 positive = max(rgb, float3(0.0));
    const float luminance = dot(float3(0.2126, 0.7152, 0.0722), positive);
    return luminance > 0.0 ? positive / luminance : float3(1.0);
}

uint differs(float3 a, float3 b) {
    return any(asuint(a) != asuint(b)) ? 1u : 0u;
}

uint differs4(float4 a, float4 b) {
    return any(asuint(a) != asuint(b)) ? 1u : 0u;
}

float unitOf(uint v) {
    return hash01(v);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void workingSpaceCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= count) {
        return;
    }
    if (i == 0) {
        // The record of lin_rec709_scene, word for word.
        uint wrong = 0;
        wrong += differs4(gWorkingSpace.toXyz0, float4(0.4124564, 0.3575761, 0.1804375, 0.0));
        wrong += differs4(gWorkingSpace.toXyz1, float4(0.2126729, 0.7151522, 0.0721750, 0.0));
        wrong += differs4(gWorkingSpace.toXyz2, float4(0.0193339, 0.1191920, 0.9503041, 0.0));
        wrong += differs4(gWorkingSpace.fromXyz0, float4(3.2404542, -1.5371385, -0.4985314, 0.0));
        wrong += differs4(gWorkingSpace.fromXyz1, float4(-0.9692660, 1.8760108, 0.0415560, 0.0));
        wrong += differs4(gWorkingSpace.fromXyz2, float4(0.0556434, -0.2040259, 1.0572252, 0.0));
        wrong += differs4(gWorkingSpace.fromSrgb0, float4(1.0, 0.0, 0.0, 0.0));
        wrong += differs4(gWorkingSpace.fromSrgb1, float4(0.0, 1.0, 0.0, 0.0));
        wrong += differs4(gWorkingSpace.fromSrgb2, float4(0.0, 0.0, 1.0, 0.0));
        wrong += differs4(gWorkingSpace.toSrgb0, float4(1.0, 0.0, 0.0, 0.0));
        wrong += differs4(gWorkingSpace.toSrgb1, float4(0.0, 1.0, 0.0, 0.0));
        wrong += differs4(gWorkingSpace.toSrgb2, float4(0.0, 0.0, 1.0, 0.0));
        wrong += differs4(gWorkingSpace.luminance, float4(0.2126, 0.7152, 0.0722, 0.0));
        wrong += any(gWorkingSpace.flags != uint4(kWorkingSpaceRec709Primaries, 0u, 0u, 0u)) ? 1u : 0u;
        InterlockedAdd(counts[0], wrong);
    }
    // A colour of this thread's, past 1 and below 0 at the ends, and a
    // temperature across the blackbody's whole range and beyond it.
    const float3 c = float3(unitOf(i * 3u), unitOf(i * 3u + 1u), unitOf(i * 3u + 2u)) * 2.4 - 0.2;
    const float kelvin = 500.0 + 15500.0 * float(i) / float(max(count - 1u, 1u));

    InterlockedAdd(counts[1], differs(blackbody(kelvin), oldBlackbody(kelvin)));
    InterlockedAdd(counts[2], differs(cloudLight(c, false), srgbToLinear(c)) +
                              differs(cloudLight(c, true), max(c, float3(0.0))));

    EditParams e = (EditParams)0;
    e.tintR = 1.0 + c.z;
    e.tintG = 0.6;
    e.tintB = 0.3 + c.x;
    e.brightness = 1.5;
    e.saturation = c.y;
    const float3 tinted = c * float3(e.tintR, e.tintG, e.tintB) * e.brightness;
    const float3 oldGrade = max(lerp(float3(dot(tinted, kOldLuminance)), tinted, e.saturation), 0.0);
    InterlockedAdd(counts[3], differs(editGrade(e, c), oldGrade));

    InterlockedAdd(counts[4], asuint(workingLuminance(c)) != asuint(dot(c, kOldLuminance)) ? 1u : 0u);
    const float3 rgb = float3(dot(float3(3.2404542, -1.5371385, -0.4985314), c),
                              dot(float3(-0.9692660, 1.8760108, 0.0415560), c),
                              dot(float3(0.0556434, -0.2040259, 1.0572252), c));
    InterlockedAdd(counts[5], differs(workingFromXyz(c), rgb) + differs(workingFromSrgbLinear(c), c) +
                              differs(workingToSrgbLinear(c), c));
    InterlockedAdd(counts[6], 1u);
}
