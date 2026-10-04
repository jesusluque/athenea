// Copyright (c) 2026 jesus luque.
//
// AN SPZ (v2, v3) INTO RECORDS: the file's attribute streams, as the browser
// decompressed them, read a byte at a time into the float records webDecode
// takes -- the layout io::readSpz writes natively (x y z, opacity, scales,
// colour, rotation, then rgb per basis), each value converted, none decoded:
// webDecode's SPZ modes do that, as splatDecode's do.
//
// The streams sit one after another in `bytes` (positions 9 bytes a point,
// alphas 1, colours 3, scales 3, rotations 3 or 4, harmonics shDim x 3), the
// file's own order, at the byte offsets `spz` gives. One slice of points at a
// time, from `first`.
module web_spz;

public struct SpzParams {
    public uint count;          // points in this slice
    public uint first;          // its first point in the file
    public uint floats;         // floats a record: 14 + keep * 3
    public uint keep;           // rest bases kept a colour (<= 15)
    public uint shDim;          // rest bases the file has a colour
    public uint smallestThree;  // 1: rotations are 4 bytes (v3)
    public uint positionsAt;    // byte offsets of the streams
    public uint alphasAt;
    public uint coloursAt;
    public uint scalesAt;
    public uint rotationsAt;
    public uint shAt;
};

StructuredBuffer<uint>    bytes;
RWStructuredBuffer<float> raw;
ConstantBuffer<SpzParams> spz;

uint byteAt(uint offset) {
    return (bytes[offset >> 2] >> ((offset & 3u) * 8u)) & 0xFFu;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void webSpzRecords(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= spz.count) {
        return;
    }
    const uint p = spz.first + i;
    const uint out = i * spz.floats;
    for (uint k = 0; k < 3; ++k) {
        // A little-endian 24-bit signed integer: parsed, not scaled.
        const uint at = spz.positionsAt + p * 9 + k * 3;
        uint fixed = byteAt(at) | (byteAt(at + 1) << 8) | (byteAt(at + 2) << 16);
        if ((fixed & 0x800000u) != 0) {
            fixed |= 0xFF000000u;
        }
        raw[out + k] = float(int(fixed));
    }
    raw[out + 3] = float(byteAt(spz.alphasAt + p));
    for (uint k = 0; k < 3; ++k) {
        raw[out + 4 + k] = float(byteAt(spz.scalesAt + p * 3 + k));
        raw[out + 7 + k] = float(byteAt(spz.coloursAt + p * 3 + k));
    }
    if (spz.smallestThree != 0) {
        const uint at = spz.rotationsAt + p * 4;
        raw[out + 10] = float(byteAt(at) | (byteAt(at + 1) << 8));
        raw[out + 11] = float(byteAt(at + 2) | (byteAt(at + 3) << 8));
        raw[out + 12] = 0.0;
    } else {
        const uint at = spz.rotationsAt + p * 3;
        raw[out + 10] = float(byteAt(at));
        raw[out + 11] = float(byteAt(at + 1));
        raw[out + 12] = float(byteAt(at + 2));
    }
    raw[out + 13] = 0.0;
    for (uint k = 0; k < spz.keep * 3; ++k) {
        raw[out + 14 + k] = float(byteAt(spz.shAt + p * spz.shDim * 3 + k));
    }
}
