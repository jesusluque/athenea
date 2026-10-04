// Copyright (c) 2026 jesus luque.
//
// CORE-RASTER: E1 and E4-E8 of the frame (docs/decisions.md, "The web
// viewer") -- decode, project, counts, the depth sort, emit, the tile sort,
// ranges, blend and present -- whose order lib/engine.js runs. This file is
// its manifest and its switches; every module after it hooks into that frame.

export const manifest = {
  id: "core-raster",
  version: "1",
  minTier: "T1",
  requires: [],
  conflicts: [],
  sections: ["S0"],
  arenas: [
    { name: "B0", access: "read", bytesPerSplat: 32 },        // positions, shape (+ sh: the sh module's)
    { name: "W0", access: "write", bytesPerSlot: 48 },        // proj
    { name: "W1", access: "write", bytesPerSlot: 28 },        // visible, tilesTouched, depthKeys, tileRects
  ],
  hooks: [
    { stage: "E1", slangInterface: null, implementation: "webDecode, webSpzRecords" },
    { stage: "E4", slangInterface: null, implementation: "webProject" },
    { stage: "E5", slangInterface: null, implementation: "radix (chunked, 8 bits)" },
    { stage: "E6", slangInterface: null, implementation: "splatEmit, splatRanges" },
    { stage: "E7", slangInterface: null, implementation: "webBlend" },
    { stage: "E8", slangInterface: null, implementation: "webPresent" },
  ],
  flags: [
    { name: "antialias", type: "u32", default: 1 },
    { name: "linearCloud", type: "u32", default: 1 },
  ],
  ui: [
    { id: "antialias", label: "Mip filter", kind: "toggle", effect: "flag", default: true },
    { id: "linear", label: "Blend in linear light", kind: "toggle", effect: "flag", default: false },
    { id: "exposure", label: "Exposure", kind: "slider", range: [0.25, 4], step: 0.05, effect: "flag", default: 1 },
  ],
  workgroupBytes: 12296,
  kernels: ["webDecode", "webSpzRecords", "boundsChunks", "boundsReduce", "webProject", "prefixChunkTotals",
    "prefixChunkStarts", "prefixLocal", "splatCompact", "radixHistogram", "radixTotals", "radixStarts",
    "radixScatter", "splatGatherCounts", "splatEmit", "splatTilesClear", "splatRanges", "webBlend", "webPresent"],
};

export function create(engine) {
  const values = { antialias: true, linear: false, exposure: 1 };
  return {
    values,
    set(id, value) {
      if (id === "antialias" || id === "linear") values[id] = !!value;
      else if (id === "exposure") values.exposure = Math.max(0.01, Number(value));
      else return false;
      engine.sky.exposure = values.exposure;
      return true;
    },
  };
}
