// Copyright (c) 2026 jesus luque.
//
// WHICH OF A PIECE'S RECORDS ARE HANDED ON, AND AS WHAT. A found scene is
// handed to a viewer in parts, and a viewer shades some of its gaussians by
// what they are made of: a lens bends what stands behind it, a mirror shows
// what stands before it, and the rest carry their light. So a piece's
// records are set apart by kind before they are written, and at a level of
// detail only those the cut draws (scene_discover.slang's discoverCutFlags,
// asked with widths in the scene's own units) are written at all.
//
//   levelFlags    a record is kept where the cut draws it and it is of the kind asked
//   levelGather   the kept records, each at its place among them
module scene_level;

struct LevelParams {
    uint count;
    uint stride;     // floats a record (26, 27 or 30)
    uint kind;       // 0: every kind; 1: neither lens nor mirror; 2: a lens; 3: a mirror
    uint pad0;
};

StructuredBuffer<float>   records;
StructuredBuffer<uint>    drawn;      // the cut's flag a record
RWStructuredBuffer<uint>  kept;       // levelFlags: 1 where the record is handed on as `kind`
StructuredBuffer<uint>    keptRead;
StructuredBuffer<uint>    places;     // its place among the kept
RWStructuredBuffer<float> gathered;
ConstantBuffer<LevelParams> params;

// scene_render.slang's record: metallic 17, roughness 18, transmission 19, index 21, sheet 22.
uint kindOf(uint at) {
    const bool carried = params.stride >= 26u;
    const float metallic = records[at + 17u];
    const float roughness = records[at + 18u];
    const float transmission = records[at + 19u];
    // A lens: glass that is no sheet and has an index to bend by.
    if (carried && transmission >= 0.5 && records[at + 22u] <= 0.5 && records[at + 21u] > 1.0) {
        return 2u;
    }
    // A mirror: a polished metal, which shows nothing of its own.
    if (transmission < 0.5 && metallic >= 0.5 && roughness <= 0.12) {
        return 3u;
    }
    return 1u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void levelFlags(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count) {
        return;
    }
    const bool wanted = params.kind == 0u || kindOf(i * params.stride) == params.kind;
    kept[i] = drawn[i] != 0u && wanted ? 1u : 0u;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void levelGather(uint3 tid: SV_DispatchThreadID) {
    const uint i = tid.x;
    if (i >= params.count || keptRead[i] == 0u) {
        return;
    }
    const uint from = i * params.stride;
    const uint to = places[i] * params.stride;
    for (uint k = 0; k < params.stride; ++k) {
        gathered[to + k] = records[from + k];
    }
}
