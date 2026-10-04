// Copyright (c) 2026 jesus luque.
//
// A TX transfer's reflected field made to say what the closed directions
// show (m2sFieldOverClosed), apart from the kernels that use it so a test
// kernel can import it with no resource of theirs.
module transfer_field_fill;

import athenea.common.sh;
import athenea.common.octahedral;

/// THE FIELD OVER WHAT IS CLOSED (task TX). The bake projected the radiance
/// the closed directions show, zero along every open one and along the half
/// never drawn; sixteen harmonics of that read a closed direction near an
/// open one -- a door's ground at the horizon, with the unsampled half under
/// it -- at a fraction of what stands there, and the paint's closed
/// reflection came back near nothing. The bake projected the closed
/// directions themselves too (`closed`, the planes' alpha), so the ratio of
/// the two read at a direction is the mean radiance of the closed directions
/// about it (a normalised convolution). That ratio, the closed directions'
/// mean where they are too few to say, is what is projected again over a
/// 16 x 16 octahedral grid by its cells' solid angles, and written in place
/// of the field: the same sixteen rgb, read the same way, which now say
/// along a closed direction what stands there.
public void m2sFieldOverClosed(inout float3 field[16], float closed[16]) {
    if (!(closed[0] > 1.0e-3)) {
        return;   // nothing closed, or a bake from before the alpha was kept
    }
    const float3 mean = max(field[0], float3(0.0)) / closed[0];
    float3 again[16];
    for (uint c = 0; c < 16u; ++c) {
        again[c] = float3(0.0);
    }
    const uint side = 16u;
    for (uint cell = 0; cell < side * side; ++cell) {
        const float2 uv = (float2(float(cell % side), float(cell / side)) + 0.5) / float(side);
        const float3 d = octDecode(uv);
        const float l1 = abs(d.x) + abs(d.y) + abs(d.z);
        const float solid = 4.0 / float(side * side) * l1 * l1 * l1;
        float3 radiance = float3(0.0);
        float share = 0.0;
        for (uint c = 0; c < 16u; ++c) {
            const float y = shBasisValue(c, d);
            radiance += field[c] * y;
            share += closed[c] * y;
        }
        const float trust = saturate((share - 0.1) / 0.3);
        const float3 here = min(max(radiance, float3(0.0)) / max(share, 0.1), float3(4.0));
        const float3 value = lerp(mean, here, trust);
        for (uint c = 0; c < 16u; ++c) {
            again[c] += value * (shBasisValue(c, d) * solid);
        }
    }
    for (uint c = 0; c < 16u; ++c) {
        field[c] = again[c];
    }
}

