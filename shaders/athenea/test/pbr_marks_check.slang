// Copyright (c) 2026 jesus luque.
//
// WHAT RIDES ON A GAUSSIAN'S TRANSMISSION COMES BACK OFF IT.
//
// A thin wall, a Schlick metal and the material's own roughness are carried
// from a stage to the pbr word as marks on the transmission (streams.slang),
// which `packPbr` takes back off. A cloud read through USD with all three must
// give back the metallic, roughness and transmission bytes the same cloud
// uploaded directly has, and each mark where it was written.
module pbr_marks_check;
import athenea.scene.splat_encoding;

struct PbrMarksParams {
    uint count;
    uint carriesOwn;   ///< 1: the cloud was written with its material roughness
    uint pad0;
    uint pad1;
};

StructuredBuffer<uint>   carried;   ///< the pbr words of the cloud read back through USD
StructuredBuffer<uint>   plain;     ///< the same cloud's, uploaded directly
StructuredBuffer<float4> expected;  ///< (own roughness or -1, thin, schlick, 0) a gaussian
RWStructuredBuffer<uint> counts;    ///< 8 wide
ConstantBuffer<PbrMarksParams> params;

/// counts[0] metallic, roughness or transmission bytes apart from the plain cloud's
/// counts[1] a thin wall or a Schlick mark not where it was written
/// counts[2] the own roughness's presence not what the cloud carries
/// counts[3] an own roughness more than half a step from what was written
/// counts[4] gaussians checked
[shader("compute")]
[numthreads(64, 1, 1)]
void pbrMarksCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const uint word = carried[i];
    const float4 e = expected[i];
    if ((word & 0xFFFFFFu) != (plain[i] & 0xFFFFFFu)) {
        InterlockedAdd(counts[0], 1u);
    }
    if (pbrThinWalled(word) != (e.y > 0.5) || pbrSchlickMetal(word) != (e.z > 0.5)) {
        InterlockedAdd(counts[1], 1u);
    }
    const float own = pbrMaterialRoughness(word);
    if ((own >= 0.0) != (params.carriesOwn != 0u)) {
        InterlockedAdd(counts[2], 1u);
    }
    if (params.carriesOwn != 0u && abs(own - saturate(e.x)) > 0.5 / 31.0 + 1.0e-4) {
        InterlockedAdd(counts[3], 1u);
    }
    InterlockedAdd(counts[4], 1u);
}
