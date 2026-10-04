// Copyright (c) 2026 jesus luque.
//
// W-HOST: athenea's splat raster on a page's WebGPU device, by modules
// (docs/decisions.md, "The web viewer"). The page's API:
//
//   const engine = await Engine.create({ canvas, tier, modules })
//   await engine.load({ url | blob, format, onProgress, transform })
//   engine.setCamera({ position, target, up, fov }), engine.getCamera()
//   engine.setSky({ color, hdriUrl, exposure })
//   engine.setLightState(states)               // none in T1: kept, answered false
//   engine.setFeatures({ id: value, ... })     // a module's ui ids; answers per id
//   engine.options()                           // every module's ui, with values
//   const stats = await engine.frame()
//   engine.bounds(), engine.dispose()
//
// The frame is TileRasterizer::render's order (core-raster), with the
// modules' hooks: E2 (lod's cut into a draw list) and E4's constants (sh's
// degree, whether the projection walks the list). The arithmetic is all in
// the kernels; this file sizes buffers, fills uniforms by name and dispatches.
// One readback a frame (the visible and pair counts, and the modules'
// counters) until the sort takes its count from the GPU (decisions.md).

import { Gpu, roundUp, bitsFor } from "./gpu.js";
import { tierOf, POLICY, describe } from "./probe.js";
import { mul, inverseAffine, apply, transformMatrix, worldToView, parseColour, srgbToLinear } from "./math.js";
import { loadPly } from "./loaders/ply.js";
import { loadSpz } from "./loaders/spz.js";
import * as coreRaster from "./modules/core-raster.js";
import * as sh from "./modules/sh.js";
import * as lod from "./modules/lod.js";

const TILE = 16;
const TIERS = ["T0", "T1", "T2", "T3"];

/** Every module this build has, by id. */
export const MODULES = { "core-raster": coreRaster, sh, lod };

/** The modules a page gets unless it says otherwise: 081's T1 preset. */
export const DEFAULT_MODULES = ["core-raster", "sh", "lod"];

/** `ids` with what they require, ordered so a module comes after what it needs. */
export function resolveModules(ids, tier) {
  const out = [], seen = new Set(), unavailable = [];
  const visit = (id, chain = []) => {
    if (seen.has(id)) return;
    const m = MODULES[id];
    if (!m) throw new Error(`no module '${id}'`);
    if (chain.includes(id)) throw new Error(`modules require each other: ${[...chain, id].join(" -> ")}`);
    for (const r of m.manifest.requires) visit(r, [...chain, id]);
    seen.add(id);
    if (TIERS.indexOf(m.manifest.minTier) > TIERS.indexOf(tier)) { unavailable.push(id); return; }
    out.push(m);
  };
  for (const id of ids) visit(id);
  for (const m of out) {
    for (const c of m.manifest.conflicts ?? []) {
      if (out.some((o) => o.manifest.id === c)) throw new Error(`modules '${m.manifest.id}' and '${c}' conflict`);
    }
  }
  return { modules: out, unavailable };
}

export class Engine {
  /**
   * `canvas`: configured here (rgba8unorm, written by webPresent as a storage
   * texture). `base`: the directory of manifest.json (default: the one above
   * this file). `tier`: forced, or probed. `modules`: ids (default
   * DEFAULT_MODULES). `device`: one the page has, or one is requested.
   */
  static async create({ canvas, base, tier, modules = DEFAULT_MODULES, device } = {}) {
    if (typeof navigator === "undefined" || !navigator.gpu) throw new Error("this browser has no WebGPU");
    const engine = new Engine();
    let adapter = null;
    if (!device) {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU: no adapter");
      // The kernels fit the specification's defaults; the binding and buffer
      // sizes are the adapter's own, since they decide how many splats fit.
      const L = adapter.limits;
      device = await adapter.requestDevice({
        requiredLimits: { maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize },
      });
      engine.ownsDevice = true;
    }
    engine.tier = tier && POLICY[tier] && tier !== "T0" ? tier : tierOf(adapter ?? { limits: device.limits, features: device.features });
    if (engine.tier === "T0") engine.tier = "T1";   // WebGPU is here: T0 is a page without it
    engine.policy = POLICY[engine.tier];
    engine.probe = describe(adapter, engine.tier);
    engine.device = device;
    engine.lost = false;
    device.lost.then((info) => { engine.lost = info; });
    engine.gpu = await Gpu.create(device, new URL(base ?? "../", import.meta.url));
    engine.canvas = canvas;
    engine.context = canvas.getContext("webgpu");
    engine.context.configure({
      device, format: "rgba8unorm", alphaMode: "opaque",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    engine.sky = { color: parseColour("#0b0b0f"), hdriUrl: null, exposure: 1 };
    engine.lightStates = {};
    engine.camera = { position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], fov: 50 };
    engine.cloud = null;
    engine.capacity = { slots: 0, pairs: 0, tiles: 0, pixels: 0 };
    engine.readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "frame.readback" });
    const { modules: resolved, unavailable } = resolveModules(modules, engine.tier);
    engine.unavailable = unavailable;
    engine.modules = resolved.map((m) => {
      for (const k of m.manifest.kernels ?? []) {
        if (!engine.gpu.has(k)) throw new Error(`module '${m.manifest.id}': kernel '${k}' is not in the manifest`);
      }
      return { manifest: m.manifest, instance: m.create(engine) };
    });
    return engine;
  }

  module(id) {
    return this.modules.find((m) => m.manifest.id === id)?.instance ?? null;
  }

  // --- the cloud ---------------------------------------------------------------

  /**
   * { url | blob, format: "ply" | "spz" (by the name when left out),
   * onProgress(fraction|null, bytes), transform }. Resolves when the cloud is
   * on the device and every module has built what it keeps.
   */
  async load({ url, blob, format, onProgress, transform } = {}) {
    const name = String(url ?? blob?.name ?? "");
    format = (format ?? name.split(/[?#]/)[0].split(".").pop() ?? "").toLowerCase();
    const loaders = { ply: loadPly, spz: loadSpz };
    if (!loaders[format]) throw new Error(`format '${format}': this page reads ${Object.keys(loaders).join(", ")}`);
    this.disposeCloud();
    const slotsPerSplat = this.module("lod") ? 1.6 : 1;
    const cloud = await loaders[format](this.gpu, url ? { url } : { blob }, { onProgress, slotsPerSplat });
    cloud.objectToWorld = transformMatrix(transform);
    cloud.bounds = await this.boundsOf(cloud);
    for (const m of this.modules) await m.instance.load?.(cloud);
    this.cloud = cloud;
    return { count: cloud.count, restPerColour: cloud.keep, bounds: this.bounds() };
  }

  /** The cloud's box in its own space, reduced on the GPU (boundsChunks, boundsReduce). */
  async boundsOf(cloud) {
    const gpu = this.gpu;
    const chunk = 4096;
    const chunks = Math.ceil(cloud.count / chunk);
    const extents = gpu.create("bounds.extents", chunks * 32);
    const result = gpu.create("bounds.result", 32);
    const params = { count: cloud.count, chunkSize: chunk, chunkCount: chunks };
    gpu.begin();
    const encoder = this.device.createCommandEncoder({ label: "bounds" });
    const pass = encoder.beginComputePass();
    gpu.dispatch(pass, "boundsChunks", chunks, { positions: cloud.positions, extents, params });
    gpu.dispatch(pass, "boundsReduce", 1, { extents, result, params });
    pass.end();
    gpu.flush();
    this.device.queue.submit([encoder.finish()]);
    const v = new Float32Array(await gpu.read(result, 0, 32));
    extents.destroy();
    result.destroy();
    return { min: [v[0], v[1], v[2]], max: [v[4], v[5], v[6]] };
  }

  /** The cloud's box in the world: its own box's corners, through its transform. */
  bounds() {
    if (!this.cloud) return null;
    const { min, max } = this.cloud.bounds;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let c = 0; c < 8; ++c) {
      const p = apply(this.cloud.objectToWorld, [c & 1 ? max[0] : min[0], c & 2 ? max[1] : min[1], c & 4 ? max[2] : min[2]]);
      for (let k = 0; k < 3; ++k) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    }
    return { min: lo, max: hi };
  }

  // --- what the page sets ------------------------------------------------------

  /** World space, Y up; `fov` vertical, in degrees. */
  setCamera({ position, target, up, fov }) {
    if (position) this.camera.position = position.slice();
    if (target) this.camera.target = target.slice();
    if (up) this.camera.up = up.slice();
    if (fov) this.camera.fov = fov;
  }

  getCamera() {
    return { position: this.camera.position.slice(), target: this.camera.target.slice(), fov: this.camera.fov };
  }

  /** The sky. T1 has no relighting: `color` (sRGB) is the background, `exposure` multiplies the picture. */
  setSky({ color, hdriUrl, exposure } = {}) {
    if (color !== undefined) this.sky.color = parseColour(color);
    if (hdriUrl !== undefined) this.sky.hdriUrl = hdriUrl;
    if (exposure !== undefined) this.module("core-raster")?.set("exposure", exposure);
    return { hdri: false };
  }

  /** { [group]: { on, intensity } }: kept; no module of this preset has lights (072's F6). */
  setLightState(states = {}) {
    Object.assign(this.lightStates, states);
    return { applied: false };
  }

  /** { id: value }: each id goes to the module whose ui has it. Answers per id whether it took. */
  setFeatures(features = {}) {
    const out = {};
    for (const [id, value] of Object.entries(features)) {
      const m = this.modules.find((x) => x.manifest.ui.some((u) => u.id === id));
      out[id] = m ? m.instance.set(id, value) !== false : false;
    }
    return out;
  }

  /** Every module's ui entries, with their current values and whether they apply. */
  options() {
    const out = [];
    for (const m of this.modules) {
      for (const u of m.manifest.ui) out.push({ ...u, module: m.manifest.id, value: m.instance.values?.[u.id] ?? u.default, available: true });
    }
    for (const id of this.unavailable) {
      for (const u of MODULES[id].manifest.ui) out.push({ ...u, module: id, value: u.default, available: false, why: `needs ${MODULES[id].manifest.minTier}` });
    }
    return out;
  }

  // --- buffers -----------------------------------------------------------------

  reserveSlots(n) {
    if (n <= this.capacity.slots) return;
    const slots = roundUp(Math.max(n, 1024), 64);   // W1's streams start on 256 bytes
    const g = this.gpu;
    g.storage("proj", slots * 48);
    g.storage("frameWords", slots * 7 * 4);
    for (const s of ["visibleOffsets", "touchedOffsets", "sortedCounts", "offsets", "depthKeys", "depthValues", "depthScratchKeys", "depthScratchValues"]) g.storage(s, slots * 4);
    g.storage("totals", 768);   // three totals, each on its own 256 bytes
    this.capacity.slots = slots;
  }

  arena() {
    const s = this.capacity.slots;
    return { visibleAt: 0, touchedAt: s, keysAt: 2 * s, rectsAt: 3 * s };
  }

  w1(stream, words) {
    return { buffer: this.gpu.buffers.frameWords, offset: this.arena()[stream] * 4, size: Math.max(words, 1) * 4 };
  }

  total(k) {
    return { buffer: this.gpu.buffers.totals, offset: k * 256, size: 4 };
  }

  reservePairs(n) {
    if (n <= this.capacity.pairs) return;
    let size = Math.max(this.capacity.pairs, 4096);
    while (size < n) size = Math.ceil(size * 1.5);
    const bind = Math.floor(this.gpu.maxBinding / 4);
    size = Math.min(size, bind, this.device.limits.maxComputeWorkgroupsPerDimension * 256);
    for (const s of ["pairTiles", "pairSplats", "pairScratchTiles", "pairScratchSplats"]) this.gpu.storage(s, size * 4);
    this.capacity.pairs = size;
  }

  // --- the frame -------------------------------------------------------------

  /** FrameParams for this frame (FrameParams.h's setFrame and setObject). */
  frameParams(width, height) {
    const c = this.camera;
    const toWorld = this.cloud.objectToWorld;
    const m = mul(worldToView(c), toWorld);
    const eyeObject = apply(inverseAffine(toWorld), c.position);
    const distance = Math.hypot(...[0, 1, 2].map((k) => c.target[k] - c.position[k])) || 1;
    const focal = (height * 0.5) / Math.tan((c.fov * Math.PI) / 360);
    const tilesX = Math.ceil(width / TILE), tilesY = Math.ceil(height / TILE);
    const raster = this.module("core-raster")?.values ?? { antialias: true, linear: false };
    // The background in the space the splats are blended in.
    const bg = raster.linear ? this.sky.color.map((v, i) => (i < 3 ? srgbToLinear(v) : v)) : this.sky.color;
    const p = {
      width, height, tilesX, tilesY, focalX: focal, focalY: focal, centreX: width * 0.5, centreY: height * 0.5,
      nearZ: Math.max(distance * 1e-3, 1e-5), farZ: distance * 1e4, orthographic: 0, antialias: raster.antialias ? 1 : 0,
      eyeX: eyeObject[0], eyeY: eyeObject[1], eyeZ: eyeObject[2],
      restPerColour: this.cloud.keep, shWords: this.cloud.shWords, shLimit: 3,
      tileBits: bitsFor(tilesX * tilesY), depthMode: 0, depthThreshold: 0.5,
      bgR: bg[0], bgG: bg[1], bgB: bg[2], bgA: bg[3],
      // linearCloud 1 takes the colours as they are (color.slang's cloudLight):
      // display sRGB, as standard 3DGS blends, unless the blend is linear.
      linearCloud: this.cloud.linear || !raster.linear ? 1 : 0, envBaseSide: 1,
    };
    ["m00", "m01", "m02", "m03", "m10", "m11", "m12", "m13", "m20", "m21", "m22", "m23"].forEach((k, i) => { p[k] = m[i]; });
    return { common: p, eyeObject };
  }

  /**
   * One frame into the canvas at its pixel size. Resolves to { total, drawn,
   * visible, pairs, pairsDropped, ms, counters, modules }.
   */
  async frame() {
    if (this.lost) throw new Error(`WebGPU device lost: ${this.lost.message}`);
    const width = this.canvas.width, height = this.canvas.height;
    const start = performance.now();
    if (!this.cloud || width === 0 || height === 0) return { total: 0, drawn: 0, visible: 0, pairs: 0, pairsDropped: 0, ms: 0, counters: {}, modules: {} };
    const g = this.gpu;
    const n = this.cloud.count;
    const { common, eyeObject } = this.frameParams(width, height);
    const frame = { common, eyeObject, source: this.cloud, slots: n, listWords: null, constants: { kShDegree: 3, kListed: false }, counters: [] };
    const tiles = common.tilesX * common.tilesY;

    // --- batch 1: E2, E4, and the counts ---------------------------------
    g.begin();
    let encoder = this.device.createCommandEncoder({ label: "frame.project" });
    let pass = encoder.beginComputePass();
    for (const m of this.modules) m.instance.prepare?.(frame, pass);
    const slots = frame.slots;
    this.reserveSlots(slots);
    const B = g.buffers;
    g.dispatch(pass, "webProject", slots, {
      positions: frame.source.positions, shape: frame.source.shape, sh: frame.source.sh,
      listWords: frame.listWords ?? g.placeholders[0], proj: B.proj, frameWords: B.frameWords,
      params: { ...common, count: slots, base: 0 }, arena: this.arena(),
    }, frame.constants);
    g.prefix(pass, this.w1("visibleAt", slots), B.visibleOffsets, this.total(0), slots);
    g.prefix(pass, this.w1("touchedAt", slots), B.touchedOffsets, this.total(1), slots);
    pass.end();
    encoder.copyBufferToBuffer(B.totals, 0, this.readback, 0, 4);
    encoder.copyBufferToBuffer(B.totals, 256, this.readback, 4, 4);
    const counted = frame.counters.slice(0, 60);
    counted.forEach((c, k) => encoder.copyBufferToBuffer(c.buffer, c.offset, this.readback, 8 + k * 4, 4));
    g.flush();
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const read = new Uint32Array(this.readback.getMappedRange(0, 8 + counted.length * 4).slice(0));
    this.readback.unmap();
    const visible = Math.min(read[0], slots);   // never more than were projected
    const pairsWanted = read[1];
    const counters = {};
    counted.forEach((c, k) => { counters[c.name] = read[2 + k]; });
    this.reservePairs(Math.max(pairsWanted, 1));
    const pairs = Math.min(pairsWanted, this.capacity.pairs);
    if (tiles > this.capacity.tiles) { g.storage("ranges", tiles * 8); this.capacity.tiles = tiles; }
    if (width * height > this.capacity.pixels) { g.storage("colour", width * height * 16); this.capacity.pixels = width * height; }

    // --- batch 2: E5-E8 ----------------------------------------------------
    g.begin();
    encoder = this.device.createCommandEncoder({ label: "frame.draw" });
    if (visible > 0) {
      pass = encoder.beginComputePass();
      g.dispatch(pass, "splatCompact", slots, {
        visible: this.w1("visibleAt", slots), visibleOffsets: B.visibleOffsets, depthKeys: this.w1("keysAt", slots),
        sortKeys: B.depthKeys, sortOrder: B.depthValues, params: { ...common, count: slots },
      });
      pass.end();
      g.sort(encoder, B.depthKeys, B.depthValues, B.depthScratchKeys, B.depthScratchValues, visible, 24);
    }
    if (pairs > 0) {
      pass = encoder.beginComputePass();
      g.dispatch(pass, "splatGatherCounts", visible, {
        order: B.depthValues, tilesTouched: this.w1("touchedAt", slots), sortedCounts: B.sortedCounts, params: { ...common, count: visible },
      });
      g.prefix(pass, B.sortedCounts, B.offsets, this.total(2), visible);
      g.dispatch(pass, "splatEmit", visible, {
        order: B.depthValues, offsets: B.offsets, tilesTouched: this.w1("touchedAt", slots), tileRects: this.w1("rectsAt", slots * 4),
        proj: B.proj, pairTiles: B.pairTiles, pairSplats: B.pairSplats, params: { ...common, count: visible, base: this.capacity.pairs },
      });
      pass.end();
      g.sort(encoder, B.pairTiles, B.pairSplats, B.pairScratchTiles, B.pairScratchSplats, pairs, bitsFor(tiles));
    }
    pass = encoder.beginComputePass();
    g.dispatch(pass, "splatTilesClear", tiles, { ranges: B.ranges, params: common });
    if (pairs > 0) g.dispatch(pass, "splatRanges", pairs, { pairTiles: B.pairTiles, ranges: B.ranges, params: { ...common, count: pairs } });
    g.dispatch(pass, "webBlend", [width, height], { ranges: B.ranges, pairSplats: B.pairSplats, proj: B.proj, colour: B.colour, params: common });
    const raster = this.module("core-raster")?.values ?? {};
    g.dispatch(pass, "webPresent", [width, height], {
      colour: B.colour, target: this.context.getCurrentTexture().createView(),
      present: { width, height, exposure: this.sky.exposure, encode: raster.linear ? 1 : 0 },
    });
    pass.end();
    g.flush();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const stats = {
      total: n, drawn: counters.drawn ?? n, visible, pairs, pairsDropped: pairsWanted - pairs,
      ms: performance.now() - start, counters, modules: {},
    };
    for (const m of this.modules) m.instance.after?.(stats);
    for (const m of this.modules) if (m.instance.stats) stats.modules[m.manifest.id] = m.instance.stats(this.cloud);
    return stats;
  }

  disposeCloud() {
    for (const m of this.modules) m.instance.dispose?.();
    this.cloud?.destroy?.();
    this.cloud = null;
  }

  dispose() {
    this.disposeCloud();
    this.gpu.destroy();
    this.readback.destroy();
    try { this.context.unconfigure(); } catch { /* already gone */ }
    if (this.ownsDevice) this.device.destroy();
  }
}
