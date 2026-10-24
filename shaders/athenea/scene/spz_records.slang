// Copyright (c) 2026 jesus luque.
//
// SPZ's six quantised streams, as the file packed them, into the float records
// every other format's validate and decode read (io::readSpz describes them):
//
//   0..2 position (the 24-bit fixed point, as an integer)   3 alpha (byte)
//   4..6 scale (bytes)   7..9 colour (bytes)
//   10..12 rotation: three bytes, or the smallest-three word as two 16-bit halves at 10 and 11
//   14.. harmonics, rgb per basis (bytes), the first `keep` bases of the file's `shDim`
//
// Nothing is dequantised here: values are widened as they are, and the
// decode's SPZ modes (splat_encoding.slang) do the rest. One thread a splat.
module spz_records;

public struct SpzRecordParams {
    public uint count;          // splats in this slice
    public uint first;          // the slice's first splat
    public uint stride;         // floats a record: 14 + 3 * keep
    public uint keep;           // bases per colour kept
    public uint shDim;          // bases per colour in the file
    public uint smallestThree;  // 1: four rotation bytes a splat
    public uint positions;      // each stream's byte offset in `packed`
    public uint alphas;
    public uint colours;
    public uint scales;
    public uint rotations;
    public uint sh;
};

StructuredBuffer<uint>          packed;
RWStructuredBuffer<float>       records;
ConstantBuffer<SpzRecordParams> params;

uint byteAt(uint offset) {
    return (packed[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void spzRecords(uint3 tid: SV_DispatchThreadID) {
    if (tid.x >= params.count) {
        return;
    }
    const uint i = params.first + tid.x;
    const uint at = tid.x * params.stride;
    for (uint k = 0; k < 3; ++k) {
        // A little-endian 24-bit signed integer: read, not scaled.
        const uint p = params.positions + i * 9 + k * 3;
        int fixed = int(byteAt(p) | (byteAt(p + 1) << 8) | (byteAt(p + 2) << 16));
        if ((fixed & 0x800000) != 0) {
            fixed |= int(0xFF000000u);
        }
        records[at + k] = float(fixed);
    }
    records[at + 3] = float(byteAt(params.alphas + i));
    for (uint k = 0; k < 3; ++k) {
        records[at + 4 + k] = float(byteAt(params.scales + i * 3 + k));
        records[at + 7 + k] = float(byteAt(params.colours + i * 3 + k));
    }
    if (params.smallestThree != 0) {
        const uint r = params.rotations + i * 4;
        records[at + 10] = float(byteAt(r) | (byteAt(r + 1) << 8));
        records[at + 11] = float(byteAt(r + 2) | (byteAt(r + 3) << 8));
        records[at + 12] = 0.0;
    } else {
        const uint r = params.rotations + i * 3;
        records[at + 10] = float(byteAt(r));
        records[at + 11] = float(byteAt(r + 1));
        records[at + 12] = float(byteAt(r + 2));
    }
    records[at + 13] = 0.0;
    // The first `keep` bases: the file's own count says where the next
    // splat starts.
    for (uint k = 0; k < params.keep * 3; ++k) {
        records[at + 14 + k] = float(byteAt(params.sh + i * params.shDim * 3 + k));
    }
}
