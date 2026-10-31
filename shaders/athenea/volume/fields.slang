// Copyright (c) 2026 jesus luque.
//
// A volume's fields by name, on the device. Each record carries a table of
// up to eight fields after its 40 words (VolumeSet lays it out): the field's
// typed grid, a hash of the name the Volume gives it (its `field:NAME`
// relationship), its role and vector role, its grid type, and the map from
// the volume's density index space to the field's own -- invariant under
// instancing, so an instance's record needs no field transforms of its
// own. The medium walks the density alone (medium.slang); what reads a
// field reads it here, by name, through nanovdb.slang's typed accessor.
module fields;

import athenea.volume.medium;
import athenea.volume.nanovdb;

public static const uint kVolumeFieldBase = 40;    // the table's first word in a record
public static const uint kVolumeFieldWords = 16;
public static const uint kVolumeMaxFields = 8;

/// What a field is, from its name (VolumeSet's FieldRole).
public static const uint kFieldOther = 0;
public static const uint kFieldDensity = 1;
public static const uint kFieldTemperature = 2;
public static const uint kFieldFlame = 3;
public static const uint kFieldColor = 4;
public static const uint kFieldVelocity = 5;

public struct VolumeField {
    public uint     gridWord;      // the typed grid
    public uint     nameHash;      // FNV-1a of the field's name
    public uint     role;          // kField*
    public uint     vectorRole;    // io::VectorRole: none, point, normal, vector, color
    public uint     gridType;      // NanoVDB's GridType code
    public float3x4 fromDensity;   // density index space to the field's, rows
};

uint recordBase(uint index) {
    return kVolumeHeaderWords + index * kVolumeRecordWords;
}

public uint volumeFieldCount(StructuredBuffer<uint> words, uint index) {
    return words[recordBase(index) + 36];
}

public VolumeField volumeField(StructuredBuffer<uint> words, uint index, uint slot) {
    const uint b = recordBase(index) + kVolumeFieldBase + slot * kVolumeFieldWords;
    VolumeField f;
    f.gridWord = words[b + 0];
    f.nameHash = words[b + 1];
    f.role = words[b + 2] & 0xFFu;
    f.vectorRole = (words[b + 2] >> 8) & 0xFFu;
    f.gridType = words[b + 3];
    f.fromDensity = float3x4(asfloat(words[b + 4]), asfloat(words[b + 5]), asfloat(words[b + 6]), asfloat(words[b + 7]),
                             asfloat(words[b + 8]), asfloat(words[b + 9]), asfloat(words[b + 10]), asfloat(words[b + 11]),
                             asfloat(words[b + 12]), asfloat(words[b + 13]), asfloat(words[b + 14]), asfloat(words[b + 15]));
    return f;
}

/// The slot of the field named by `nameHash` in a record, or -1.
public int volumeFieldSlot(StructuredBuffer<uint> words, uint index, uint nameHash) {
    const uint count = min(volumeFieldCount(words, index), kVolumeMaxFields);
    for (uint k = 0; k < count; ++k) {
        if (words[recordBase(index) + kVolumeFieldBase + k * kVolumeFieldWords + 1] == nameHash) return int(k);
    }
    return -1;
}

/// The slot of the first field of a role, or -1.
public int volumeFieldOfRole(StructuredBuffer<uint> words, uint index, uint role) {
    const uint count = min(volumeFieldCount(words, index), kVolumeMaxFields);
    for (uint k = 0; k < count; ++k) {
        if ((words[recordBase(index) + kVolumeFieldBase + k * kVolumeFieldWords + 2] & 0xFFu) == role) return int(k);
    }
    return -1;
}

/// A field's value at a world point, by its type: xyz and the component
/// count (nanovdb.slang's nvRead), at the nearest voxel (values stand at
/// integer index coordinates, as the medium's).
public float4 volumeFieldAt(StructuredBuffer<uint> words, uint index, uint slot, float3 world) {
    const VolumeRecord r = volumeRecord(words, index);
    const VolumeField f = volumeField(words, index, slot);
    const float3 p = mul(f.fromDensity, float4(mul(r.worldToIndex, float4(world, 1.0)), 1.0));
    return nvValueAt(words, nvGrid(words, f.gridWord), int3(floor(p + 0.5)));
}
