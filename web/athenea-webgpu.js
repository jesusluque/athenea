// Copyright (c) 2026 jesus luque.
//
// THE RENDERER SLOT: athenea's raster as a renderer of the athenea-web
// viewer (site/src/renderers/index.js documents the interface). A thin
// adapter over W-host (host.js): the page's calls map onto the host's --
//
//   createRenderer({canvas, scene, file, url, onProgress})
//                       AtheneaHost.create + host.load({url, format, transform})
//   setCamera / getCamera            host.setCamera / host.getCamera
//   setBackground({color, hdriUrl, showHdri, exposure, blur})
//                                    host.setSky
//   setDetail(scale)                 host.setFeatures({ shDegree }) (no LOD in T1)
//   setLightState / setFeatures      the host's, as they are
//   stats()                          the last host.frame()'s counts, and fps
//   snapshot()                       the canvas, as webp, after a frame
//   dispose()                        host.dispose: the device and every buffer
//
// The orbit is this module's own (left drag turns, right drag or shift pans,
// the wheel dollies), so the page needs no three.js for it; the page's
// setCamera calls move it as presets do.

import { AtheneaHost } from "./host.js";

export const RENDERER_INFO = { label: "athenea · WebGPU", formats: ["ply"], needsWebGPU: true };

export async function createRenderer({ canvas, scene: meta = {}, file = {}, url, onProgress }) {
  const host = await AtheneaHost.create({ canvas, base: new URL(".", import.meta.url) });
  try {
    await host.load({ url, format: file.format ?? "ply", onProgress, transform: file.transform ?? meta.transform });
  } catch (e) {
    host.dispose();
    throw e;
  }
  const orbit = new Orbit(canvas, host);
  const box = meta.bounds ?? host.bounds();
  if (meta.camera) orbit.set(meta.camera);
  else if (box) orbit.frame(box);

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  };
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  let last = { total: host.cloud.count, visible: 0, pairs: 0 };
  let fps = 0, frames = 0, since = performance.now();
  let running = true, error = null;
  const snapshots = [];
  const loop = async () => {
    while (running) {
      try {
        last = await host.frame();
      } catch (e) {
        error = e;
        console.error("athenea-webgpu:", e);
        break;
      }
      if (snapshots.length) {
        for (const done of snapshots.splice(0)) canvas.toBlob(done, "image/webp", 0.85);
      }
      frames++;
      const now = performance.now();
      if (now - since >= 500) { fps = (frames * 1000) / (now - since); frames = 0; since = now; }
      await new Promise((r) => requestAnimationFrame(r));
    }
  };
  loop();

  return {
    name: "athenea",
    backend: "WebGPU",
    bounds: () => box,
    setCamera: (c) => orbit.set(c),
    getCamera: () => host.getCamera(),
    setBackground({ color, hdriUrl, showHdri = true, exposure = 1 } = {}) {
      // T1 shows no sky: the colour stands behind the splats either way.
      return host.setSky({ color: color ?? "#0b0b0f", hdriUrl: showHdri ? hdriUrl ?? null : null, exposure });
    },
    // No levels of detail in T1: the multiplier lowers the harmonics instead,
    // which is what 072's degradation order drops after the threshold.
    setDetail(scale = 1) {
      host.setFeatures({ shDegree: scale >= 1 ? 3 : scale >= 0.5 ? 1 : 0 });
    },
    hasLod: false,
    hasRoi: false,
    roi: false,
    setRoi() {},
    setLightState: (s) => host.setLightState(s),
    setFeatures: (f) => host.setFeatures(f),
    // `extra` is a line the page prints as it is; `debug` the same as numbers.
    stats: () => ({
      total: last.total, rendered: last.visible, fps,
      extra: error ? `error: ${error.message}`
        : `${(last.pairs ?? 0).toLocaleString("en")} pairs${last.pairsDropped ? ` (${last.pairsDropped.toLocaleString("en")} dropped)` : ""} · ${(last.ms ?? 0).toFixed(1)} ms`,
      debug: { pairs: last.pairs, pairsDropped: last.pairsDropped ?? 0, ms: last.ms ?? 0, error: error?.message },
    }),
    snapshot: () => new Promise((resolve) => snapshots.push(resolve)),
    dispose() {
      running = false;
      ro.disconnect();
      orbit.dispose();
      host.dispose();
    },
  };
}

/** A turntable about a target, Y up. */
class Orbit {
  constructor(canvas, host) {
    this.canvas = canvas;
    this.host = host;
    this.target = [0, 0, 0];
    this.distance = 3;
    this.yaw = 0;
    this.pitch = 0;
    this.fov = 50;
    const down = (e) => {
      this.drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
      canvas.setPointerCapture?.(e.pointerId);
    };
    const move = (e) => {
      if (!this.drag) return;
      const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
      this.drag.x = e.clientX;
      this.drag.y = e.clientY;
      if (this.drag.pan) {
        const s = (2 * this.distance * Math.tan((this.fov * Math.PI) / 360)) / Math.max(canvas.clientHeight, 1);
        const [r, u] = this.axes();
        for (let k = 0; k < 3; ++k) this.target[k] += (-dx * r[k] + dy * u[k]) * s;
      } else {
        this.yaw -= dx * 0.005;
        this.pitch = Math.max(-1.55, Math.min(1.55, this.pitch + dy * 0.005));
      }
      this.apply();
    };
    const up = () => { this.drag = null; };
    const wheel = (e) => {
      e.preventDefault();
      this.distance *= Math.exp(e.deltaY * 0.001);
      this.apply();
    };
    const menu = (e) => e.preventDefault();
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("contextmenu", menu);
    this.dispose = () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("contextmenu", menu);
    };
  }

  eye() {
    const c = Math.cos(this.pitch);
    return [this.target[0] + this.distance * c * Math.sin(this.yaw), this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * c * Math.cos(this.yaw)];
  }

  axes() {
    const e = this.eye();
    const f = [0, 1, 2].map((k) => this.target[k] - e[k]);
    const l = Math.hypot(...f);
    const fw = f.map((v) => v / l);
    const r = [-fw[2], 0, fw[0]];
    const rl = Math.hypot(...r) || 1;
    const right = r.map((v) => v / rl);
    const up = [right[1] * fw[2] - right[2] * fw[1], right[2] * fw[0] - right[0] * fw[2], right[0] * fw[1] - right[1] * fw[0]];
    return [right, up];
  }

  apply() {
    this.host.setCamera({ position: this.eye(), target: this.target.slice(), up: [0, 1, 0], fov: this.fov });
  }

  set({ position, target, fov }) {
    if (fov) this.fov = fov;
    if (target) this.target = target.slice();
    if (position) {
      const d = [0, 1, 2].map((k) => position[k] - this.target[k]);
      this.distance = Math.hypot(...d) || 1;
      this.pitch = Math.asin(Math.max(-1, Math.min(1, d[1] / this.distance)));
      this.yaw = Math.atan2(d[0], d[2]);
    }
    this.apply();
  }

  frame({ min, max }) {
    const centre = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
    const radius = Math.hypot(...[0, 1, 2].map((k) => max[k] - min[k])) / 2 || 1;
    this.target = centre;
    this.distance = radius / Math.sin((this.fov * Math.PI) / 360) * 1.1;
    this.yaw = 0;
    this.pitch = 0.3;
    this.apply();
  }
}
