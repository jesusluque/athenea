// Copyright (c) 2026 jesus luque.
//
// A volume's fields by name, on the device: each probe names a field by its
// hash and a world point, and expects a slot (-1 for a name the volume does
// not have), a value, a role and a vector role. What is read is compared
// with what is expected here; the host reads the counts.
import athenea.volume.fields;

struct FieldsParams {
    uint probes;
    uint volume;
};

ConstantBuffer<FieldsParams> fieldsCheck;
StructuredBuffer<uint>       words;
StructuredBuffer<uint>       probes;   // 12 words a probe: world xyz, name hash, expected xyzw, role, vector role, slot, pad
RWStructuredBuffer<uint>     counts;   // [slot wrong, value wrong, role wrong, probes]

[shader("compute")]
[numthreads(64, 1, 1)]
void volumeFieldsCheck(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= fieldsCheck.probes) return;
    const uint b = i * 12u;
    const float3 world = float3(asfloat(probes[b]), asfloat(probes[b + 1]), asfloat(probes[b + 2]));
    const float4 expected = float4(asfloat(probes[b + 4]), asfloat(probes[b + 5]), asfloat(probes[b + 6]),
                                   asfloat(probes[b + 7]));
    const int slot = volumeFieldSlot(words, fieldsCheck.volume, probes[b + 3]);
    InterlockedAdd(counts[3], 1u);
    if (slot != asint(probes[b + 10])) {
        InterlockedAdd(counts[0], 1u);
        return;
    }
    if (slot < 0) return;
    const float4 v = volumeFieldAt(words, fieldsCheck.volume, uint(slot), world);
    if (any(v != expected)) InterlockedAdd(counts[1], 1u);
    const VolumeField f = volumeField(words, fieldsCheck.volume, uint(slot));
    if (f.role != probes[b + 8] || f.vectorRole != probes[b + 9]) InterlockedAdd(counts[2], 1u);
}
