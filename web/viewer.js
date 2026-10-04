// Copyright (c) 2026 jesus luque.
//
// THE VIEWER PAGE: a canvas, an orbit, a panel made from the modules' ui
// (each module's manifest says what it offers: lib/modules/*.js), the tier
// the probe chose, and the frame's stats. Everything it shows of a cloud the
// engine computed on the GPU; this file only wires the page.
//
// A scene of the site's catalogue is `?s=<id>[&f=<format>]`: its scene.json
// (under `assets`, by default the site's asset host) gives the file, its
// transform, the first camera and the background.

import { Engine, DEFAULT_MODULES } from "./lib/engine.js";
import { Orbit } from "./lib/orbit.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const canvas = $("view");
const status = (text) => { $("status").textContent = text; };
const ASSETS = "https://athenea-assets.lucab.co.uk/scenes/";
/** The formats this page reads, in the order it prefers them for a scene. */
const FORMATS = ["spz", "ply"];

let engine = null, orbit = null, running = false, lastStats = null;

async function start() {
  if (!navigator.gpu) {
    status("This browser has no WebGPU (tier T0): open the scene in the standard viewer.");
    return;
  }
  const modules = params.get("modules")?.split(",").filter(Boolean) ?? DEFAULT_MODULES;
  engine = await Engine.create({ canvas, tier: params.get("tier") ?? undefined, modules });
  orbit = new Orbit(canvas, (c) => engine.setCamera(c));
  $("tier").textContent = `${engine.policy.label} · ${engine.policy.drawn ? `${(engine.policy.drawn / 1e6).toFixed(2)} M drawn` : ""}`;
  new ResizeObserver(resize).observe(canvas);
  resize();
  buildPanel();
  wireFiles();
  fetch(new URL("./build.json", import.meta.url)).then((r) => (r.ok ? r.json() : null)).then((b) => {
    if (b) $("build").textContent = `webgpu ${b.short}${b.dirty ? "+" : ""}`;
  }).catch(() => {});
  const url = params.get("url");
  if (params.get("s")) await openScene(params.get("s"), params.get("f"));
  else if (url) await open({ url: new URL(url, location.href).href, name: url });
}

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, engine?.policy.pixelRatio ?? 2);
  const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
  const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

/** One control a ui entry: toggle, slider or select; a module the tier lacks is shown greyed. */
function buildPanel() {
  const root = $("options");
  root.textContent = "";
  let module = null;
  for (const o of engine.options()) {
    if (o.module !== module) {
      module = o.module;
      const h = document.createElement("h3");
      h.textContent = module;
      root.append(h);
    }
    const label = document.createElement("label");
    if (!o.available) { label.className = "off"; label.title = o.why ?? ""; }
    const name = document.createElement("span");
    name.textContent = o.label;
    label.append(name);
    let input;
    if (o.kind === "toggle") {
      input = Object.assign(document.createElement("input"), { type: "checkbox", checked: !!o.value });
      input.onchange = () => engine.setFeatures({ [o.id]: input.checked });
      input.dataset.toggle = o.id;
    } else if (o.kind === "slider") {
      const shown = document.createElement("span");
      input = Object.assign(document.createElement("input"), { type: "range", min: o.range[0], max: o.range[1], step: o.step ?? "any", value: o.value });
      const show = () => { shown.textContent = Number(input.value).toFixed(o.step >= 1 ? 0 : 2); };
      input.oninput = () => { engine.setFeatures({ [o.id]: Number(input.value) }); show(); };
      show();
      input.show = show;
      label.append(shown);
      input.dataset.id = o.id;
    } else if (o.kind === "select") {
      input = document.createElement("select");
      for (const v of o.options) input.append(new Option(v, v, false, v === o.value));
      input.onchange = () => engine.setFeatures({ [o.id]: input.value });
    }
    if (input) {
      input.disabled = !o.available;
      label.append(input);
    }
    root.append(label);
  }
  $("panel").hidden = false;
}

/** A slider a module moves by itself (the controller's threshold) follows it. */
function syncPanel() {
  for (const o of engine.options()) {
    const input = document.querySelector(`input[data-id="${o.id}"]`);
    const box = document.querySelector(`input[data-toggle="${o.id}"]`);
    if (box && box.checked !== !!o.value) box.checked = !!o.value;
    if (input && document.activeElement !== input && Number(input.value) !== Number(o.value)) {
      input.value = o.value;
      input.show?.();   // shown, not set: setting it would take the controller's hand off
    }
  }
}

function wireFiles() {
  $("open").onchange = (e) => { const f = e.target.files[0]; if (f) open({ blob: f, name: f.name }); };
  addEventListener("dragover", (e) => { e.preventDefault(); document.body.classList.add("drag"); });
  addEventListener("dragleave", () => document.body.classList.remove("drag"));
  addEventListener("drop", (e) => {
    e.preventDefault();
    document.body.classList.remove("drag");
    const f = e.dataTransfer.files[0];
    if (f) open({ blob: f, name: f.name });
  });
}

/**
 * A scene of the catalogue: scenes/<id>/scene.json, its file in `format` (or
 * the first this page reads), that file's transform or the scene's, its first
 * camera, its background.
 */
async function openScene(id, format) {
  const base = new URL(`${encodeURIComponent(id)}/`, new URL(params.get("assets") ?? ASSETS, location.href));
  status(`scene ${id}…`);
  const response = await fetch(new URL("scene.json", base));
  if (!response.ok) { status(`scene ${id}: HTTP ${response.status}`); return; }
  const scene = await response.json();
  const files = scene.files ?? [];
  const pick = format ? files.find((f) => f.format === format)
    : FORMATS.map((f) => files.find((x) => x.format === f)).find(Boolean);
  if (!pick || !FORMATS.includes(pick.format)) {
    status(`scene ${id}: no ${format ?? FORMATS.join(" or ")} file (it has ${files.map((f) => f.format).join(", ")})`);
    return;
  }
  document.title = `${scene.title ?? id} · athenea`;
  if (scene.background?.color) engine.setSky({ color: scene.background.color });
  const camera = scene.cameras?.[0];
  const transform = pick.transform && Object.keys(pick.transform).length ? pick.transform : scene.transform;
  await open({
    url: new URL(pick.url ?? pick.path, base).href, name: `${scene.title ?? id} (${pick.format})`, format: pick.format,
    transform, camera,
  });
}

async function open({ url, blob, name, format, transform: given, camera }) {
  running = false;
  status(`loading ${name}…`);
  const t0 = performance.now();
  try {
    const transform = params.get("transform") ? JSON.parse(params.get("transform")) : given;
    const loaded = await engine.load({
      url, blob, transform, format,
      onProgress: (f, bytes) => status(`loading ${name} ${f == null ? "" : `${Math.round(f * 100)}% `}${(bytes / 1e6).toFixed(1)} MB`),
    });
    status(`${name} · ${loaded.count.toLocaleString("en")} splats · ${((performance.now() - t0) / 1000).toFixed(1)} s to load and build`);
    const vec = (k) => params.get(k)?.split(",").map(Number);
    if (params.get("eye")) orbit.set({ position: vec("eye"), target: vec("target") ?? [0, 0, 0], fov: Number(params.get("fov") ?? 50) });
    else if (camera) orbit.set(camera);
    else orbit.frame(loaded.bounds);
    if (params.get("features")) engine.setFeatures(JSON.parse(params.get("features")));
    buildPanel();
    run();
  } catch (e) {
    status(String(e?.message ?? e));
    console.error(e);
  }
}

async function run() {
  running = true;
  let frames = 0, since = performance.now(), fps = 0, shot = params.get("shot");
  const shotAt = performance.now() + 4000;   // after the controller has had its first seconds
  while (running) {
    try {
      lastStats = await engine.frame();
    } catch (e) {
      status(`frame: ${e.message}`);
      console.error(e);
      running = false;
      break;
    }
    frames++;
    const now = performance.now();
    if (now - since >= 500) {
      fps = (frames * 1000) / (now - since);
      frames = 0;
      since = now;
      showStats(fps);
      syncPanel();
    }
    if (shot && now > shotAt) {
      const name = shot;
      shot = null;
      canvas.toBlob((blob) => {
        const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: `${name}.webp` });
        a.click();
      }, "image/webp", 0.9);
      console.log("athenea-viewer stats", JSON.stringify({ ...lastStats, fps, tier: engine.tier, bounds: engine.bounds() }));
    }
    await new Promise((r) => requestAnimationFrame(r));
  }
}

function showStats(fps) {
  const s = lastStats;
  const f = (n) => (n ?? 0).toLocaleString("en");
  const lines = [
    `${engine.tier} · ${canvas.width}×${canvas.height} · ${fps.toFixed(1)} fps · ${s.ms.toFixed(1)} ms`,
    `${f(s.total)} splats · ${f(s.drawn)} drawn · ${f(s.visible)} visible · ${f(s.pairs)} pairs${s.pairsDropped ? ` (${f(s.pairsDropped)} dropped)` : ""}`,
  ];
  const lod = s.modules?.lod;
  if (lod?.built) lines.push(`lod ${lod.on ? "on" : "off"} · ${lod.threshold.toFixed(2)} px${lod.auto ? " (auto)" : ""} · ${lod.levels} levels, ${f(lod.merged)} merged · slots ${f(lod.slots)}${lod.dropped ? ` · ${f(lod.dropped)} over` : ""}`);
  $("stats").textContent = lines.join("\n");
}

start().catch((e) => { status(String(e?.message ?? e)); console.error(e); });
