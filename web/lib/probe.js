// Copyright (c) 2026 jesus luque.
//
// W-PROBE: which tier a device is (proposal 072 section 3, 074 section 2), from
// what its adapter says and what platform it is on, and that tier's policy.
// Page logic: no data of the scene is read. A performance probe (074 2.1's
// decisive signal) is the frame-time controller's job once frames are drawn
// (lib/modules/lod.js), not a separate run.

const MB = 1e6;

/** 074 section 2.2: what each tier draws, at what floor, at what resolution. */
export const POLICY = {
  T0: { label: "T0 · no WebGPU", drawn: 0, floorPx: 2, startPx: 2, pixelRatio: 1, targetMs: 33.3 },
  T1: { label: "T1 · WebGPU, default limits", drawn: 1.25e6, floorPx: 1.5, startPx: 2, pixelRatio: 1.5, targetMs: 33.3 },
  T2: { label: "T2 · 32 KiB, 644 MB a binding", drawn: 2.5e6, floorPx: 1, startPx: 1, pixelRatio: 2, targetMs: 16.7 },
  T3: { label: "T3 · 1.25 GB a binding, BC", drawn: 4.5e6, floorPx: 1, startPx: 1, pixelRatio: Infinity, targetMs: 16.7 },
};

export const TIERS = ["T0", "T1", "T2", "T3"];

/** Whether the page runs on a phone, a tablet or a headset (074: they take T1 at most). */
export function handheld(nav = globalThis.navigator) {
  const ua = nav?.userAgent ?? "";
  if (nav?.userAgentData?.mobile) return true;
  return /iPhone|iPad|iPod|Android|Mobile|OculusBrowser|Quest|visionOS/i.test(ua) ||
    (/Macintosh/.test(ua) && (nav?.maxTouchPoints ?? 0) > 1);   // an iPad asking for the desktop site
}

/**
 * The tier of an adapter (null: no WebGPU): T3 when a binding takes 1.25 GB
 * and the adapter has BC textures; T2 at 32 KiB of workgroup memory and
 * 644 MB a binding; T1 otherwise, and always on a handheld.
 */
export function tierOf(adapter, nav = globalThis.navigator) {
  if (!adapter) return "T0";
  const L = adapter.limits;
  const binding = Math.min(L.maxStorageBufferBindingSize, L.maxBufferSize);
  let tier = "T1";
  if (L.maxComputeWorkgroupStorageSize >= 32768 && binding >= 644 * MB) tier = "T2";
  if (tier === "T2" && binding >= 1250 * MB && adapter.features?.has?.("texture-compression-bc")) tier = "T3";
  if (handheld(nav)) tier = "T1";
  return tier;
}

/** What the probe found, for a panel or a report. */
export function describe(adapter, tier) {
  const L = adapter?.limits;
  return {
    tier,
    policy: POLICY[tier],
    adapter: adapter?.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture } : null,
    limits: L ? {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
      maxStorageBuffersPerShaderStage: L.maxStorageBuffersPerShaderStage,
    } : null,
    handheld: handheld(),
  };
}
