// Copyright (c) 2026 jesus luque.
//
// THE SITE'S RENDERER SLOT: athenea's engine (lib/engine.js) as a renderer of
// the athenea-web viewer (site/src/renderers/index.js documents the
// interface). The page's calls map onto the engine's --
//
//   createRenderer({canvas, scene, file, url, onProgress})
//                       Engine.create (tier probed) + engine.load({url, format, transform})
//   setCamera / getCamera            engine.setCamera / getCamera
//   setBackground({color, hdriUrl, showHdri, exposure, blur})
//                                    engine.setSky
//   setDetail(scale)                 lod's threshold: 1 / scale times the tier's
//   setLightState / setFeatures      the engine's, as they are
//   stats()                          { total, rendered, fps, extra (one line), debug }
//   snapshot()                       the canvas, as webp, after a frame
//   dispose()                        the device and every buffer
//
// The orbit is the module's own (lib/orbit.js), on the canvas.

import { Engine } from "./lib/engine.js";
import { Orbit } from "./lib/orbit.js";

export const RENDERER_INFO = { label: "athenea · WebGPU", formats: ["ply", "spz"], needsWebGPU: true };

export async function createRenderer({ canvas, scene: meta = {}, file = {}, url, onProgress }) {
  const engine = await Engine.create({ canvas });
  try {
    await engine.load({ url, format: file.format, onProgress, transform: file.transform ?? meta.transform });
  } catch (e) {
    engine.dispose();
    throw e;
  }
  const orbit = new Orbit(canvas, (c) => engine.setCamera(c));
  const box = meta.bounds ?? engine.bounds();
  if (meta.camera) orbit.set(meta.camera);
  else if (box) orbit.frame(box);

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, engine.policy.pixelRatio);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  let last = { total: engine.cloud.count, drawn: 0, visible: 0, pairs: 0, ms: 0, modules: {} };
  let fps = 0, frames = 0, since = performance.now();
  let running = true, error = null;
  const snapshots = [];
  (async () => {
    while (running) {
      try {
        last = await engine.frame();
      } catch (e) {
        error = e;
        console.error("athenea-webgpu:", e);
        break;
      }
      if (snapshots.length) for (const done of snapshots.splice(0)) canvas.toBlob(done, "image/webp", 0.85);
      frames++;
      const now = performance.now();
      if (now - since >= 500) { fps = (frames * 1000) / (now - since); frames = 0; since = now; }
      await new Promise((r) => requestAnimationFrame(r));
    }
  })();

  const lod = engine.module("lod");
  return {
    name: "athenea",
    backend: "WebGPU",
    bounds: () => box,
    setCamera: (c) => orbit.set(c),
    getCamera: () => engine.getCamera(),
    setBackground({ color, hdriUrl, showHdri = true, exposure = 1 } = {}) {
      // No sky shown before the relit preset: the colour stands behind the splats.
      return engine.setSky({ color: color ?? "#0b0b0f", hdriUrl: showHdri ? hdriUrl ?? null : null, exposure });
    },
    // The LOD multiplier: more detail is a smaller cell, from the tier's start.
    setDetail(scale = 1) {
      if (lod) engine.setFeatures({ lodThreshold: engine.policy.startPx / Math.max(scale, 0.05), lodAuto: scale === 1 });
    },
    hasLod: !!lod,
    hasRoi: false,
    roi: false,
    setRoi() {},
    setLightState: (s) => engine.setLightState(s),
    setFeatures: (f) => engine.setFeatures(f),
    // `extra` is a line the page prints as it is; `debug` the numbers.
    stats: () => ({
      total: last.total, rendered: last.drawn, fps,
      extra: error ? `error: ${error.message}`
        : `${engine.tier} · ${(last.visible ?? 0).toLocaleString("en")} visible · ${(last.pairs ?? 0).toLocaleString("en")} pairs · ${(last.ms ?? 0).toFixed(1)} ms`,
      debug: { ...last, error: error?.message, tier: engine.tier },
    }),
    snapshot: () => new Promise((resolve) => snapshots.push(resolve)),
    dispose() {
      running = false;
      ro.disconnect();
      orbit.dispose();
      engine.dispose();
    },
  };
}
