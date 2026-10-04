// Copyright (c) 2026 jesus luque.
//
// SH: the view-dependent colour, its spherical harmonics evaluated to a
// degree (E4). The degree is a WGSL override of webProject -- a pipeline a
// degree, no data moved -- so lowering it is the cheapest feature to drop
// (072's degradation order puts it after the cut's threshold). Its hook is
// ISplatColour's place in 081; the interface itself comes with the TX colour.

export const manifest = {
  id: "sh",
  version: "1",
  minTier: "T1",
  requires: ["core-raster"],
  conflicts: [],
  sections: ["S0"],
  arenas: [{ name: "B0", access: "read", bytesPerSplat: "2-92 (sh: degree 0-3)" }],
  hooks: [{ stage: "E4", slangInterface: "ISplatColour", implementation: "webProject's evaluateRest, kShDegree" }],
  flags: [{ name: "shLimit", type: "u32", default: 3 }],
  ui: [{ id: "shDegree", label: "Harmonics degree", kind: "slider", range: [0, 3], step: 1, effect: "preset", default: 3 }],
  workgroupBytes: 0,
  kernels: ["webProject"],
};

export function create() {
  const values = { shDegree: 3 };
  return {
    values,
    set(id, value) {
      if (id !== "shDegree") return false;
      values.shDegree = Math.max(0, Math.min(3, Math.round(Number(value))));
      return true;
    },
    prepare(frame) {
      frame.constants.kShDegree = values.shDegree;
    },
    stats(cloud) {
      return { degree: values.shDegree, stored: cloud ? [0, 3, 8, 15].indexOf(cloud.keep) : 0 };
    },
  };
}
