// Copyright (c) 2026 jesus luque.
//
// Each light's share of the frame's power, as the prefix accumulated it,
// against the shares written again here from the records: radiance over area
// for an area light, over the scene's cross-section for a dome (pi R^2) and a
// sun (R^2, times its disc's projected solid angle without normalize). A
// count of lights whose share differs, and the worst ratio.
import athenea.light.lights;

struct PowerCheck {
    uint  count;
    float sceneRadius;
    float tolerance;   // relative
    uint  pad0;
};

StructuredBuffer<LightRecord> lights;
ConstantBuffer<PowerCheck>    check;
RWStructuredBuffer<uint>      counts;   // [shares off, lights]
RWStructuredBuffer<float>     worst;    // [largest relative difference]

float expectedPower(LightRecord l) {
    const float luminance = max(0.2126 * l.colour.x + 0.7152 * l.colour.y + 0.0722 * l.colour.z, 0.0) * exp2(l.exposure);
    const float pi = 3.14159265358979;
    // A dome ignores normalize; a sun's normalize divides out its disc's
    // projected solid angle and leaves the scene's cross-section.
    if (l.kind == kLightDome) {
        return max(luminance * pi * check.sceneRadius * check.sceneRadius, 1.0e-6);
    }
    if (l.kind == kLightDistant) {
        const float s = sin(0.5 * l.sizeX);
        const float disc = (l.flags & kLightNormalize) == 0 && l.sizeX > 0.0 ? pi * s * s : 1.0;
        return max(luminance * disc * check.sceneRadius * check.sceneRadius, 1.0e-6);
    }
    if ((l.flags & kLightNormalize) != 0) {
        return max(luminance, 1.0e-6);
    }
    float extent = 1.0;
    if (l.kind == kLightSphere) extent = 4.0 * pi * l.sizeX * l.sizeX;
    if (l.kind == kLightDisk) extent = pi * l.sizeX * l.sizeX;
    if (l.kind == kLightRect) extent = l.sizeX * l.sizeY;
    if (l.kind == kLightCylinder) extent = 2.0 * pi * l.sizeX * l.sizeY;
    return max(luminance * extent, 1.0e-6);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void lightPowerCheck(uint3 tid: SV_DispatchThreadID) {
    if (tid.x != 0) return;
    float total = 0.0;
    for (uint k = 0; k < check.count; ++k) {
        total += expectedPower(lights[k]);
    }
    const float accumulated = lights[check.count - 1].cumulative;
    uint off = 0;
    float largest = 0.0;
    for (uint k = 0; k < check.count; ++k) {
        const float share = (lights[k].cumulative - (k > 0 ? lights[k - 1].cumulative : 0.0)) / accumulated;
        const float expected = expectedPower(lights[k]) / total;
        const float difference = abs(share - expected) / max(expected, 1e-12);
        largest = max(largest, difference);
        if (difference > check.tolerance) off += 1;
    }
    counts[0] = off;
    counts[1] = check.count;
    worst[0] = largest;
}
