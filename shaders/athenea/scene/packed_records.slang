// Copyright (c) 2026 jesus luque.
//
// A file's own records -- a PLY's vertex element of any property types, a
// .splat's 32 bytes -- widened into the float records validate and decode read
// (io::PackedFields): field k of record i, of the type `fields[k].y` at byte
// `fields[k].x` of the record, becomes float k. Values are converted, not
// decoded: a double is rounded to the nearest float, as a C++ cast rounds it,
// and integers become the floats of their values. One thread a record.
module packed_records;

public struct PackedRecordParams {
    public uint count;    // records in this slice
    public uint first;    // the slice's first record
    public uint stride;   // bytes a record
    public uint fields;   // fields a record, and floats a record written
};

StructuredBuffer<uint>             packed;   // the records' bytes, little-endian, and a word past them
StructuredBuffer<uint2>            fields;   // (byte offset, type) a field
RWStructuredBuffer<float>          records;
ConstantBuffer<PackedRecordParams> params;

static const uint kF32 = 0;
static const uint kF64 = 1;
static const uint kU8 = 2;
static const uint kI8 = 3;
static const uint kU16 = 4;
static const uint kI16 = 5;
static const uint kU32 = 6;

/// The 32 bits at any byte offset.
uint wordAt(uint offset) {
    const uint first = packed[offset >> 2];
    const uint shift = (offset & 3u) * 8u;
    return shift == 0u ? first : (first >> shift) | (packed[(offset >> 2) + 1u] << (32u - shift));
}

/// A double's bits (hi, lo) as the nearest float, ties to even.
float floatOfDouble(uint hi, uint lo) {
    const uint sign = hi & 0x80000000u;
    const int exponent = int((hi >> 20) & 0x7FFu);
    const uint mantissaHi = hi & 0xFFFFFu;
    if (exponent == 0x7FF) {
        const bool nan = mantissaHi != 0u || lo != 0u;
        return asfloat(sign | 0x7F800000u | (nan ? 0x400000u : 0u));
    }
    if (exponent == 0) {
        return asfloat(sign);   // a double this small is a float zero
    }
    int e = exponent - 1023 + 127;
    uint mantissa = (mantissaHi << 3) | (lo >> 29);
    const uint rest = lo & 0x1FFFFFFFu;
    if (e <= 0) {
        // Below the floats' normal range: computed as a float, which gives
        // the subnormal (or zero) the hardware rounds to.
        const float value = (1.0 + float(mantissa) / 8388608.0) * exp2(float(exponent - 1023));
        return sign != 0u ? -value : value;
    }
    if (rest > 0x10000000u || (rest == 0x10000000u && (mantissa & 1u) != 0u)) {
        mantissa += 1u;
        if (mantissa == 0x800000u) {
            mantissa = 0u;
            e += 1;
        }
    }
    if (e >= 255) {
        return asfloat(sign | 0x7F800000u);
    }
    return asfloat(sign | (uint(e) << 23) | mantissa);
}

float fieldValue(uint at, uint type) {
    if (type == kF32) {
        return asfloat(wordAt(at));
    }
    if (type == kF64) {
        return floatOfDouble(wordAt(at + 4u), wordAt(at));
    }
    const uint word = wordAt(at);
    if (type == kU8) {
        return float(word & 0xFFu);
    }
    if (type == kI8) {
        return float(int(word << 24) >> 24);
    }
    if (type == kU16) {
        return float(word & 0xFFFFu);
    }
    if (type == kI16) {
        return float(int(word << 16) >> 16);
    }
    if (type == kU32) {
        return float(word);
    }
    return float(int(word));   // I32
}

[shader("compute")]
[numthreads(256, 1, 1)]
void packedRecords(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint record = (params.first + tid.x) * params.stride;
    const uint at = tid.x * params.fields;
    for (uint k = 0; k < params.fields; ++k) {
        const uint2 field = fields[k];
        records[at + k] = fieldValue(record + field.x, field.y);
    }
}
