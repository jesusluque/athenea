// Copyright (c) 2026 jesus luque.
//
// AN ASSET'S rANS TABLES, AS THE DECODER LOOKS THEM UP: from the file's
// frequencies (256 u16 a table, summing to 2^tableLog) to a slot each --
// every slot s of 0..2^tableLog the symbol whose run of slots holds it, its
// frequency and where its run starts (athz_common.slang's decode). Once, at
// open; the frequencies are what the file keeps (512 bytes a table where the
// slots would be 2^tableLog words).
module athz_tables;

struct AthzTableParams {
    uint tables;
    uint tableLog;
    uint pad0;
    uint pad1;
};
ConstantBuffer<AthzTableParams> params;

ByteAddressBuffer        frequencies;   ///< tables * 256 u16
RWStructuredBuffer<uint> slots;         ///< tables * 2^tableLog: sym | freq << 8 | cum << 20

uint frequency(uint table, uint s) {
    const uint at = (table * 256u + s) * 2u;
    const uint w = frequencies.Load(at & ~3u);
    return (at & 2u) != 0u ? (w >> 16) : (w & 0xFFFFu);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void athzExpandTables(uint3 tid: SV_DispatchThreadID) {
    const uint size = 1u << params.tableLog;
    const uint table = tid.x / size;
    const uint slot = tid.x % size;
    if (table >= params.tables) {
        return;
    }
    uint cum = 0u;
    for (uint s = 0; s < 256u; ++s) {
        const uint f = frequency(table, s);
        if (slot < cum + f) {
            slots[tid.x] = s | (f << 8) | (cum << 20);
            return;
        }
        cum += f;
    }
    slots[tid.x] = 0u;   // frequencies that do not fill the table: a corrupt asset, refused at open
}
