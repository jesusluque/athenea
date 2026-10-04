// Copyright (c) 2026 jesus luque.
//
// W-HOST, PRESET T1 (proposal 072, stage E1): athenea's splat raster on a
// page's WebGPU device, from the kernels scripts/web-kernels.py compiled out
// of the engine's Slang. The API is the contract with the page
// (docs/operations.md, "The web module"):
//
//   const host = await AtheneaHost.create({ canvas, base })
//   await host.load({ url, format: "ply", onProgress, transform })
//   host.setCamera({ position, target, up, fov })
//   host.setSky({ color, hdriUrl, exposure })
//   host.setLightState(states)        // no lights in T1: kept, reported back
//   host.setFeatures({ shDegree, antialias })
//   const stats = await host.frame()  // { total, visible, pairs, ms }
//   host.bounds(), host.dispose()
//
// The order of a frame is TileRasterizer::render's, pass for pass: project;
// prefix sums of what is visible and of the pairs; one readback of the two
// totals; compact, depth sort, gather, prefix, emit, tile sort, ranges,
// blend -- and present, which the native viewer's DisplayTransform does. The
// arithmetic is all in the kernels; this file sizes buffers, fills uniforms
// by the names the manifest gives, and dispatches. Every binding is by name:
// the manifest maps a name to its @binding, read from Slang's reflection.
//
// Until W-host is the engine's C++ in wasm (decisions.md, "WebGPU"), this file
// repeats TileRasterizer's dispatch order, RadixSort's and PrefixSum's; a
// change there is a change here.

import { parsePlyHeader, headerEnd } from "./ply.js";

const TILE = 16;
const DIGITS = 256;

/** PrefixSum::chunkFor and RadixSort::chunkFor (the same rule). */
function chunkFor(count) {
  let chunk = 256;
  while (chunk < 4096 && Math.floor(count / chunk) > 2048) chunk *= 2;
  return chunk;
}

/** FrameParams.h's bitsFor. */
function bitsFor(values) {
  let bits = 1;
  while (bits < 32 && 2 ** bits < values) ++bits;
  return bits;
}

const roundUp = (n, k) => Math.ceil(n / k) * k;

// --- 4x4 affine matrices, row-major [16]: the camera's bookkeeping ---------

function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let r = 0; r < 4; ++r) for (let c = 0; c < 4; ++c) for (let k = 0; k < 4; ++k) o[r * 4 + c] += a[r * 4 + k] * b[k * 4 + c];
  return o;
}
const identity = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function inverseAffine(m) {
  // The 3x3 inverse by cofactors, then the translation.
  const [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]];
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  const s = 1 / det;
  const r = [
    A * s, -(b * i - c * h) * s, (b * f - c * e) * s,
    B * s, (a * i - c * g) * s, -(a * f - c * d) * s,
    C * s, -(a * h - b * g) * s, (a * e - b * d) * s,
  ];
  const t = [m[3], m[7], m[11]];
  return [
    r[0], r[1], r[2], -(r[0] * t[0] + r[1] * t[1] + r[2] * t[2]),
    r[3], r[4], r[5], -(r[3] * t[0] + r[4] * t[1] + r[5] * t[2]),
    r[6], r[7], r[8], -(r[6] * t[0] + r[7] * t[1] + r[8] * t[2]),
    0, 0, 0, 1,
  ];
}
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const normalize = (a) => { const l = Math.hypot(...a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const apply = (m, p) => [0, 1, 2].map((r) => m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3]);

/**
 * A page's transform -- { position, rotation (degrees, three.js's XYZ Euler
 * order), scale } -- as an object-to-world matrix, or a matrix given as 16
 * numbers, row-major.
 */
export function transformMatrix(t) {
  if (!t) return identity();
  if (Array.isArray(t) && t.length === 16) return t.slice();
  const [x, y, z] = (t.rotation || [0, 0, 0]).map((v) => (v * Math.PI) / 180);
  const rx = [1, 0, 0, 0, 0, Math.cos(x), -Math.sin(x), 0, 0, Math.sin(x), Math.cos(x), 0, 0, 0, 0, 1];
  const ry = [Math.cos(y), 0, Math.sin(y), 0, 0, 1, 0, 0, -Math.sin(y), 0, Math.cos(y), 0, 0, 0, 0, 1];
  const rz = [Math.cos(z), -Math.sin(z), 0, 0, Math.sin(z), Math.cos(z), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const s = t.scale ?? 1;
  const p = t.position || [0, 0, 0];
  const m = mul(mul(rx, ry), rz);
  return [m[0] * s, m[1] * s, m[2] * s, p[0], m[4] * s, m[5] * s, m[6] * s, p[1], m[8] * s, m[9] * s, m[10] * s, p[2], 0, 0, 0, 1];
}

// --- the host ----------------------------------------------------------------

export class AtheneaHost {
  /**
   * `canvas`: the page's canvas; the host configures its context (rgba8unorm,
   * written by webPresent as a storage texture). `base`: the URL of the
   * directory holding manifest.json (by default, this module's own).
   * `device`: a device the page already has, or one is requested.
   */
  static async create({ canvas, base, device } = {}) {
    if (!navigator.gpu) throw new Error("this browser has no WebGPU");
    const host = new AtheneaHost();
    host.base = new URL(base ?? ".", import.meta.url);
    const manifest = await (await fetch(new URL("manifest.json", host.base))).json();
    if (manifest.format !== "athenea-webgpu-kernels" || manifest.version !== 1) {
      throw new Error(`manifest: ${manifest.format} version ${manifest.version} is not one this host reads`);
    }
    host.manifest = manifest;
    if (!device) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU: no adapter");
      // The preset's limits are the specification's defaults; the binding and
      // buffer sizes are the adapter's own, since they decide how many splats
      // fit (072 section 3.1), and asking costs nothing.
      const L = adapter.limits;
      device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
          maxBufferSize: L.maxBufferSize,
        },
      });
      host.ownsDevice = true;
    }
    host.device = device;
    host.limits = device.limits;
    host.lost = false;
    device.lost.then((info) => { host.lost = info; });
    const sources = {};
    await Promise.all(Object.entries(manifest.kernels).map(async ([name, k]) => {
      sources[name] = await (await fetch(new URL(k.file, host.base))).text();
    }));
    host.sources = sources;
    host.pipelines = new Map();
    host.canvas = canvas;
    host.context = canvas.getContext("webgpu");
    host.context.configure({
      device, format: "rgba8unorm", alphaMode: "opaque",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    // `linear`: blend in linear light, athenea's own way (frame.slang), and
    // encode on the way out. Off by default in T1: standard 3DGS -- the
    // trainers, Spark, three.js -- blends a capture's colours in the display
    // sRGB they were trained in, and T1 is compared against those viewers.
    host.features = { shDegree: 3, antialias: true, linear: false };
    host.sky = { color: parseColour("#0b0b0f"), hdriUrl: null, exposure: 1 };
    host.lightStates = {};
    host.camera = { position: [0, 0, 3], target: [0, 0, 0], up: [0, 1, 0], fov: 50 };
    host.cloud = null;
    host.buffers = {};
    host.capacity = { splats: 0, pairs: 0, sortChunks: 0, prefixChunks: 0, tiles: 0, pixels: 0 };
    host.placeholders = {
      a: device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: "placeholder.a" }),
      b: device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE, label: "placeholder.b" }),
    };
    host.readback = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "totals.readback" });
    host.uniformRing = null;
    return host;
  }

  // --- kernels, by name ------------------------------------------------------

  pipeline(name, constants = {}) {
    const key = name + JSON.stringify(constants);
    let p = this.pipelines.get(key);
    if (!p) {
      const k = this.manifest.kernels[name];
      if (!k) throw new Error(`no kernel '${name}' in the manifest`);
      const module = this.device.createShaderModule({ code: this.sources[name], label: name });
      // Override constants by their @id, which is what Slang names them by.
      const byId = {};
      for (const [cname, value] of Object.entries(constants)) {
        const o = k.overrides[cname];
        if (!o) throw new Error(`${name}: no override '${cname}'`);
        byId[String(o.id)] = value;
      }
      p = this.device.createComputePipeline({
        label: name, layout: "auto",
        compute: { module, entryPoint: name, constants: byId },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  /** A uniform's bytes from {field: value}, by the manifest's offsets; the rest 0. */
  uniformBytes(kernel, param, values) {
    const spec = this.manifest.kernels[kernel].parameters[param];
    const bytes = new ArrayBuffer(roundUp(spec.size, 16));
    const view = new DataView(bytes);
    for (const [field, value] of Object.entries(values)) {
      const f = spec.fields[field];
      if (!f) continue;   // a field this kernel's struct does not have
      if (f.type === "f32") view.setFloat32(f.offset, value, true);
      else if (f.type === "i32") view.setInt32(f.offset, value, true);
      else view.setUint32(f.offset, value >>> 0, true);
    }
    return bytes;
  }

  /** A slice of this frame's uniform ring, filled. */
  uniform(kernel, param, values) {
    const bytes = this.uniformBytes(kernel, param, values);
    const align = this.limits.minUniformBufferOffsetAlignment;
    const at = roundUp(this.ringUsed, align);
    if (!this.uniformRing || at + bytes.byteLength > this.uniformRing.size) {
      throw new Error("uniform ring overflow");
    }
    this.ringUsed = at + bytes.byteLength;
    new Uint8Array(this.ringData, at, bytes.byteLength).set(new Uint8Array(bytes));
    return { buffer: this.uniformRing, offset: at, size: bytes.byteLength };
  }

  /**
   * One dispatch of `name` over `threads` threads: `bind` maps every
   * parameter the kernel declares to a buffer, a {buffer, offset, size}
   * range, a texture view, or (for a uniform) an object of field values.
   */
  dispatch(pass, name, threads, bind, constants) {
    const k = this.manifest.kernels[name];
    const pipeline = this.pipeline(name, constants);
    const entries = [];
    for (const [param, spec] of Object.entries(k.parameters)) {
      const what = bind[param];
      if (what === undefined) throw new Error(`${name}: '${param}' is not bound`);
      let resource;
      if (spec.kind === "uniform") resource = this.uniform(name, param, what);
      else if (spec.kind === "storage-texture") resource = what;
      else resource = what.buffer ? what : { buffer: what };
      entries.push({ binding: spec.binding, resource });
    }
    for (const param of Object.keys(bind)) {
      if (!k.parameters[param]) throw new Error(`${name}: declares no '${param}'`);
    }
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries, label: name });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    const [wx, wy] = k.workgroupSize;
    const [tx, ty] = Array.isArray(threads) ? threads : [threads, 1];
    const gx = Math.ceil(tx / wx), gy = Math.ceil(ty / wy);
    if (gx > this.limits.maxComputeWorkgroupsPerDimension || gy > this.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error(`${name}: ${gx} x ${gy} groups is past the device's ${this.limits.maxComputeWorkgroupsPerDimension}`);
    }
    if (gx > 0 && gy > 0) pass.dispatchWorkgroups(gx, gy, 1);
  }

  // --- buffers ---------------------------------------------------------------

  storage(name, bytes, extra = 0) {
    const old = this.buffers[name];
    if (old && old.size >= bytes) return old;
    old?.destroy();
    const size = Math.max(16, roundUp(bytes, 16));
    const max = Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize);
    if (size > max) throw new Error(`${name}: ${size} bytes is past the device's binding size ${max}`);
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra, label: name });
    this.buffers[name] = b;
    return b;
  }

  /** Splat slots: B0 is the cloud's; W0 (proj) and W1 (frameWords) are the frame's. */
  reserveSplats(n) {
    if (n <= this.capacity.splats) return;
    const slots = roundUp(Math.max(n, 1024), 64);   // W1's streams start on 256 bytes
    this.storage("proj", slots * 48);
    this.storage("frameWords", slots * 7 * 4);
    this.arena = { visibleAt: 0, touchedAt: slots, keysAt: 2 * slots, rectsAt: 3 * slots };
    this.storage("visibleOffsets", slots * 4);
    this.storage("touchedOffsets", slots * 4);
    this.storage("sortedCounts", slots * 4);
    this.storage("offsets", slots * 4);
    for (const s of ["depthKeys", "depthValues", "depthScratchKeys", "depthScratchValues"]) this.storage(s, slots * 4);
    // Three totals, each on its own 256 bytes (a storage binding's offset alignment).
    this.storage("totals", 768);
    this.capacity.splats = slots;
  }

  /** Total `k` of the frame's three, as a binding. */
  total(k) {
    return { buffer: this.buffers.totals, offset: k * 256, size: 4 };
  }

  /** A range of W1: one of the frame's per-splat streams, under its own name. */
  w1(stream, words) {
    return { buffer: this.buffers.frameWords, offset: this.arena[stream] * 4, size: words * 4 };
  }

  reservePairs(n) {
    if (n <= this.capacity.pairs) return;
    let size = Math.max(this.capacity.pairs, 4096);
    while (size < n) size = Math.ceil(size * 1.5);
    size = Math.min(size, this.maxPairs());
    for (const s of ["pairTiles", "pairSplats", "pairScratchTiles", "pairScratchSplats"]) this.storage(s, size * 4);
    this.capacity.pairs = size;
  }

  /** As many pairs as a binding holds and one dimension of groups reaches. */
  maxPairs() {
    const bind = Math.floor(Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize) / 4);
    return Math.min(bind, this.limits.maxComputeWorkgroupsPerDimension * 256);
  }

  // --- the cloud ---------------------------------------------------------------

  /**
   * Loads a cloud: { url, format: "ply", onProgress(fraction|null, bytes),
   * transform }. The records stream in by fetch and are decoded on the GPU a
   * slice at a time as they arrive. Resolves when every record is on the device.
   */
  async load({ url, format = "ply", onProgress, transform } = {}) {
    if (format !== "ply") throw new Error(`format '${format}': this module reads ply (3DGS, float32)`);
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    const length = Number(response.headers.get("content-length")) || 0;
    const reader = response.body.getReader();
    let pending = new Uint8Array(0);
    let loaded = 0;
    const take = (chunk) => {
      const joined = new Uint8Array(pending.length + chunk.length);
      joined.set(pending);
      joined.set(chunk, pending.length);
      pending = joined;
    };
    let header = null;
    while (!header) {
      const { done, value } = await reader.read();
      if (done) throw new Error(`${url}: ended inside its header`);
      loaded += value.length;
      take(value);
      if (headerEnd(pending) >= 0) header = parsePlyHeader(pending);
    }
    this.cloud?.destroy?.();
    const cloud = this.prepareCloud(header, transform);
    pending = pending.subarray(header.dataStart);
    // Records go up a slice at a time: one raw buffer, reused, as large as a
    // binding allows and no larger than 64 MiB.
    const sliceRecords = Math.max(256, Math.floor(Math.min(64 << 20, this.limits.maxStorageBufferBindingSize) / header.stride));
    const raw = this.device.createBuffer({ size: roundUp(sliceRecords * header.stride, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: "raw.slice" });
    const sliceBytes = sliceRecords * header.stride;
    let staged = new Uint8Array(sliceBytes);
    let stagedBytes = 0;
    let decoded = 0;
    const flush = () => {
      const records = Math.min(Math.floor(stagedBytes / header.stride), header.count - decoded);
      if (records <= 0) return;
      this.device.queue.writeBuffer(raw, 0, staged, 0, records * header.stride);
      this.beginUniforms(1);
      const encoder = this.device.createCommandEncoder({ label: "decode" });
      const pass = encoder.beginComputePass();
      this.dispatch(pass, "webDecode", records, {
        raw, positions: cloud.positions, shape: cloud.shape, sh: cloud.sh,
        params: { ...cloud.decode, count: records, base: decoded, shWords: cloud.shWords, keepPerColour: cloud.keep, recordBase: decoded },
      });
      pass.end();
      this.endUniforms();
      this.device.queue.submit([encoder.finish()]);
      decoded += records;
      const used = records * header.stride;
      staged.copyWithin(0, used, stagedBytes);
      stagedBytes -= used;
    };
    const feed = (bytes) => {
      let at = 0;
      while (at < bytes.length && decoded < header.count) {
        const n = Math.min(bytes.length - at, sliceBytes - stagedBytes);
        staged.set(bytes.subarray(at, at + n), stagedBytes);
        stagedBytes += n;
        at += n;
        if (stagedBytes === sliceBytes) flush();
      }
    };
    feed(pending);
    onProgress?.(length ? loaded / length : null, loaded);
    while (decoded + Math.floor(stagedBytes / header.stride) < header.count) {
      const { done, value } = await reader.read();
      if (done) break;
      loaded += value.length;
      feed(value);
      onProgress?.(length ? loaded / length : null, loaded);
    }
    flush();
    reader.cancel().catch(() => {});
    await this.device.queue.onSubmittedWorkDone();
    raw.destroy();
    if (decoded < header.count) throw new Error(`${url}: ${decoded} of ${header.count} records arrived`);
    cloud.bounds = await this.boundsOf(cloud);
    this.cloud = cloud;
    onProgress?.(1, loaded);
    return { count: cloud.count, restPerColour: cloud.keep, bounds: this.bounds() };
  }

  prepareCloud(header, transform) {
    // The harmonics kept are what the file has, down to what a binding holds:
    // a device's binding size is a limit on splats a degree (072 section 3.1).
    const maxBind = Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize);
    let keep = header.restPerColour;
    const wordsFor = (k) => (k === 0 ? 1 : Math.floor((k * 3 + 1) / 2));
    while (keep > 0 && header.count * wordsFor(keep) * 4 > maxBind) keep = keep === 15 ? 8 : keep === 8 ? 3 : 0;
    // The largest stream a splat has is its projected record, 48 bytes (W0).
    if (roundUp(Math.max(header.count, 1024), 64) * 48 > maxBind) {
      throw new Error(`${header.count} splats: past this device's ${maxBind} bytes a binding (it holds ${Math.floor(maxBind / 48)})`);
    }
    const shWords = wordsFor(keep);
    const usage = GPUBufferUsage.STORAGE;
    const cloud = {
      count: header.count, keep, shWords, decode: header.decode, linear: false,
      objectToWorld: transformMatrix(transform),
      positions: this.device.createBuffer({ size: roundUp(header.count * 16, 16), usage, label: "B0.positions" }),
      shape: this.device.createBuffer({ size: roundUp(header.count * 16, 16), usage, label: "B0.shape" }),
      sh: this.device.createBuffer({ size: roundUp(header.count * shWords * 4, 16), usage, label: "B0.sh" }),
    };
    cloud.destroy = () => { cloud.positions.destroy(); cloud.shape.destroy(); cloud.sh.destroy(); };
    return cloud;
  }

  /** The cloud's box in its own space, reduced on the GPU (boundsChunks, boundsReduce). */
  async boundsOf(cloud) {
    const chunk = 4096;
    const chunks = Math.ceil(cloud.count / chunk);
    const extents = this.device.createBuffer({ size: chunks * 32, usage: GPUBufferUsage.STORAGE, label: "bounds.extents" });
    const result = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, label: "bounds.result" });
    const read = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "bounds.read" });
    const params = { count: cloud.count, chunkSize: chunk, chunkCount: chunks };
    this.beginUniforms(2);
    const encoder = this.device.createCommandEncoder({ label: "bounds" });
    const pass = encoder.beginComputePass();
    this.dispatch(pass, "boundsChunks", chunks, { positions: cloud.positions, extents, params });
    this.dispatch(pass, "boundsReduce", 1, { extents, result, params });
    pass.end();
    encoder.copyBufferToBuffer(result, 0, read, 0, 32);
    this.endUniforms();
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    for (const b of [extents, result, read]) b.destroy();
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

  /**
   * The sky. T1 has no relighting: `color` (sRGB, "#rrggbb" or [r, g, b])
   * is the background and `exposure` multiplies the picture; `hdriUrl` is
   * kept for the stage that lights by it (072's E4) and reported as not shown.
   */
  setSky({ color, hdriUrl, exposure } = {}) {
    if (color !== undefined) this.sky.color = parseColour(color);
    if (hdriUrl !== undefined) this.sky.hdriUrl = hdriUrl;
    if (exposure !== undefined) this.sky.exposure = exposure;
    return { hdri: false };
  }

  /** { [group]: { on, intensity } }: kept; T1 has no lights (072's F6, E8). */
  setLightState(states = {}) {
    Object.assign(this.lightStates, states);
    return { applied: false };
  }

  /** { shDegree: 0..3, antialias: bool, linear: bool }. What a feature this preset lacks is answered with false. */
  setFeatures(features = {}) {
    const known = {};
    for (const [k, v] of Object.entries(features)) {
      if (k === "shDegree") { this.features.shDegree = Math.max(0, Math.min(3, v | 0)); known[k] = true; }
      else if (k === "antialias") { this.features.antialias = !!v; known[k] = true; }
      else if (k === "linear") { this.features.linear = !!v; known[k] = true; }
      else known[k] = false;
    }
    return known;
  }

  beginUniforms(dispatches) {
    const bytes = dispatches * roundUp(1024, this.limits.minUniformBufferOffsetAlignment);
    if (!this.uniformRing || this.uniformRing.size < bytes) {
      this.uniformRing?.destroy();
      this.uniformRing = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "uniforms" });
    }
    this.ringData = new ArrayBuffer(this.uniformRing.size);
    this.ringUsed = 0;
  }

  endUniforms() {
    this.device.queue.writeBuffer(this.uniformRing, 0, this.ringData, 0, roundUp(this.ringUsed, 4));
  }

  // --- the frame -------------------------------------------------------------

  /** The frame's common uniform: setFrame and setObject of FrameParams.h. */
  frameParams(width, height) {
    const c = this.camera;
    const f = normalize(sub(c.target, c.position));
    const r = normalize(cross(f, c.up));
    const u = cross(r, f);
    const e = c.position;
    // The view: +x right, +y up, +z forward (frame.slang), the -Z camera folded.
    const worldToView = [...r, -dot(r, e), ...u, -dot(u, e), ...f, -dot(f, e), 0, 0, 0, 1];
    const toWorld = this.cloud.objectToWorld;
    const m = mul(worldToView, toWorld);
    const eyeObject = apply(inverseAffine(toWorld), e);
    const distance = Math.hypot(...sub(c.target, c.position)) || 1;
    const focal = (height * 0.5) / Math.tan((c.fov * Math.PI) / 360);
    const tilesX = Math.ceil(width / TILE), tilesY = Math.ceil(height / TILE);
    // The background in the space the splats are blended in.
    const [bgR, bgG, bgB, bgA] = this.features.linear ? this.sky.color.map((v, i) => (i < 3 ? srgbToLinear(v) : v)) : this.sky.color;
    const p = {
      width, height, tilesX, tilesY, focalX: focal, focalY: focal, centreX: width * 0.5, centreY: height * 0.5,
      nearZ: Math.max(distance * 1e-3, 1e-5), farZ: distance * 1e4, orthographic: 0,
      antialias: this.features.antialias ? 1 : 0,
      eyeX: eyeObject[0], eyeY: eyeObject[1], eyeZ: eyeObject[2],
      restPerColour: this.cloud.keep, shWords: this.cloud.shWords, shLimit: 3,
      tileBits: bitsFor(tilesX * tilesY), depthMode: 0, depthThreshold: 0.5,
      bgR, bgG, bgB, bgA,
      // linearCloud 1 takes the colours as they are (common/color.slang's
      // cloudLight): in display mode that is what keeps a capture in sRGB.
      linearCloud: this.cloud.linear || !this.features.linear ? 1 : 0, envBaseSide: 1,
    };
    ["m00", "m01", "m02", "m03", "m10", "m11", "m12", "m13", "m20", "m21", "m22", "m23"].forEach((k, i) => { p[k] = m[i]; });
    return p;
  }

  /** PrefixSum::apply. */
  prefix(pass, input, output, total, count) {
    const chunk = chunkFor(count);
    const chunks = count === 0 ? 1 : Math.ceil(count / chunk);
    if (chunks > this.capacity.prefixChunks) {
      this.storage("prefixChunkTotals", chunks * 4);
      this.storage("prefixChunkStarts", chunks * 4);
      this.capacity.prefixChunks = chunks;
    }
    const params = { count, chunkSize: chunk, chunkCount: chunks };
    const chunkTotals = this.buffers.prefixChunkTotals, chunkStarts = this.buffers.prefixChunkStarts;
    this.dispatch(pass, "prefixChunkTotals", chunks, { input, chunkTotals, params });
    this.dispatch(pass, "prefixChunkStarts", 1, { chunkTotals, chunkStarts, total, params });
    this.dispatch(pass, "prefixLocal", chunks, { input, chunkStarts, output, params });
  }

  /** RadixSort::sort's chunked route (the tiled one wants 27648 bytes of group memory). */
  sort(encoder, keys, values, scratchKeys, scratchValues, count, keyBits) {
    if (count <= 1) return;
    const chunk = chunkFor(count);
    const chunks = Math.ceil(count / chunk);
    if (chunks > this.capacity.sortChunks) {
      this.storage("radixHistogram", chunks * DIGITS * 4);
      this.storage("radixChunkStarts", chunks * DIGITS * 4);
      this.capacity.sortChunks = chunks;
    }
    this.storage("radixDigitTotals", DIGITS * 4);
    const histogram = this.buffers.radixHistogram, chunkStarts = this.buffers.radixChunkStarts, digitTotals = this.buffers.radixDigitTotals;
    let src = [keys, values], dst = [scratchKeys, scratchValues];
    const passes = Math.ceil(keyBits / 8);
    for (let p = 0; p < passes; ++p) {
      const params = { count, chunkSize: chunk, chunkCount: chunks, shift: p * 8, wide: 0 };
      const pass = encoder.beginComputePass({ label: `radix ${p}` });
      // Distinct placeholders for the unused hi words: one buffer read-only
      // and read-write in one dispatch is refused (RadixSort::sort).
      this.dispatch(pass, "radixHistogram", chunks, { keysLo: src[0], keysHi: this.placeholders.a, histogram, params });
      this.dispatch(pass, "radixTotals", DIGITS, { histogram, digitTotals, params });
      this.dispatch(pass, "radixStarts", DIGITS, { histogram, digitTotals, chunkStarts, params });
      this.dispatch(pass, "radixScatter", chunks, {
        srcKeysLo: src[0], srcKeysHi: this.placeholders.a, srcValues: src[1],
        dstKeysLo: dst[0], dstKeysHi: this.placeholders.b, dstValues: dst[1], chunkStarts, params,
      });
      pass.end();
      [src, dst] = [dst, src];
    }
    if (passes % 2 === 1) {
      encoder.copyBufferToBuffer(src[0], 0, keys, 0, count * 4);
      encoder.copyBufferToBuffer(src[1], 0, values, 0, count * 4);
    }
  }

  /**
   * One frame into the canvas, at its current size in device pixels (the
   * page sets the canvas's width and height). Resolves to { total, visible,
   * pairs, pairsDropped, ms }.
   */
  async frame() {
    if (this.lost) throw new Error(`WebGPU device lost: ${this.lost.message}`);
    const width = this.canvas.width, height = this.canvas.height;
    const start = performance.now();
    if (!this.cloud || width === 0 || height === 0) return { total: 0, visible: 0, pairs: 0, pairsDropped: 0, ms: 0 };
    const n = this.cloud.count;
    this.reserveSplats(n);
    const common = this.frameParams(width, height);
    const tiles = common.tilesX * common.tilesY;
    const B = this.buffers;

    // --- batch 1: project; how many are visible and how many pairs --------
    this.beginUniforms(8);
    let encoder = this.device.createCommandEncoder({ label: "frame.project" });
    let pass = encoder.beginComputePass();
    this.dispatch(pass, "webProject", n, {
      positions: this.cloud.positions, shape: this.cloud.shape, sh: this.cloud.sh,
      proj: B.proj, frameWords: B.frameWords,
      params: { ...common, count: n, base: 0 }, arena: this.arena,
    }, { kShDegree: this.features.shDegree });
    this.prefix(pass, this.w1("visibleAt", n), B.visibleOffsets, this.total(0), n);
    this.prefix(pass, this.w1("touchedAt", n), B.touchedOffsets, this.total(1), n);
    pass.end();
    encoder.copyBufferToBuffer(B.totals, 0, this.readback, 0, 4);
    encoder.copyBufferToBuffer(B.totals, 256, this.readback, 4, 4);
    this.endUniforms();
    this.device.queue.submit([encoder.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const totals = new Uint32Array(this.readback.getMappedRange(0, 8).slice(0));
    this.readback.unmap();
    const visible = totals[0];
    const pairsWanted = totals[1];
    this.reservePairs(Math.max(pairsWanted, 1));
    const pairs = Math.min(pairsWanted, this.capacity.pairs);
    if (tiles > this.capacity.tiles) { this.storage("ranges", tiles * 8); this.capacity.tiles = tiles; }
    if (width * height > this.capacity.pixels) { this.storage("colour", width * height * 16); this.capacity.pixels = width * height; }

    // --- batch 2: sort the visible, emit, tile sort, ranges, blend, present --
    this.beginUniforms(64);
    encoder = this.device.createCommandEncoder({ label: "frame.draw" });
    if (visible > 0) {
      pass = encoder.beginComputePass();
      this.dispatch(pass, "splatCompact", n, {
        visible: this.w1("visibleAt", n), visibleOffsets: B.visibleOffsets, depthKeys: this.w1("keysAt", n),
        sortKeys: B.depthKeys, sortOrder: B.depthValues, params: { ...common, count: n },
      });
      pass.end();
      this.sort(encoder, B.depthKeys, B.depthValues, B.depthScratchKeys, B.depthScratchValues, visible, 24);
    }
    if (pairs > 0) {
      pass = encoder.beginComputePass();
      this.dispatch(pass, "splatGatherCounts", visible, {
        order: B.depthValues, tilesTouched: this.w1("touchedAt", n), sortedCounts: B.sortedCounts, params: { ...common, count: visible },
      });
      this.prefix(pass, B.sortedCounts, B.offsets, this.total(2), visible);
      this.dispatch(pass, "splatEmit", visible, {
        order: B.depthValues, offsets: B.offsets, tilesTouched: this.w1("touchedAt", n), tileRects: this.w1("rectsAt", n * 4),
        proj: B.proj, pairTiles: B.pairTiles, pairSplats: B.pairSplats, params: { ...common, count: visible, base: this.capacity.pairs },
      });
      pass.end();
      this.sort(encoder, B.pairTiles, B.pairSplats, B.pairScratchTiles, B.pairScratchSplats, pairs, bitsFor(tiles));
    }
    pass = encoder.beginComputePass();
    this.dispatch(pass, "splatTilesClear", tiles, { ranges: B.ranges, params: common });
    if (pairs > 0) {
      this.dispatch(pass, "splatRanges", pairs, { pairTiles: B.pairTiles, ranges: B.ranges, params: { ...common, count: pairs } });
    }
    this.dispatch(pass, "webBlend", [width, height], { ranges: B.ranges, pairSplats: B.pairSplats, proj: B.proj, colour: B.colour, params: common });
    const target = this.context.getCurrentTexture();
    this.dispatch(pass, "webPresent", [width, height], {
      colour: B.colour, target: target.createView(), present: { width, height, exposure: this.sky.exposure, encode: this.features.linear ? 1 : 0 },
    });
    pass.end();
    this.endUniforms();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    return { total: n, visible, pairs, pairsDropped: pairsWanted - pairs, ms: performance.now() - start };
  }

  dispose() {
    this.cloud?.destroy?.();
    this.cloud = null;
    for (const b of Object.values(this.buffers)) b.destroy();
    this.buffers = {};
    this.placeholders.a.destroy();
    this.placeholders.b.destroy();
    this.readback.destroy();
    this.uniformRing?.destroy();
    this.pipelines.clear();
    try { this.context.unconfigure(); } catch { /* already gone */ }
    if (this.ownsDevice) this.device.destroy();
  }
}

/** A page's colour, "#rrggbb" or [r, g, b(, a)] in sRGB 0-1, kept as sRGB. */
function parseColour(c) {
  if (Array.isArray(c)) return [c[0], c[1], c[2], c[3] ?? 1];
  const m = /^#?([0-9a-f]{6})$/i.exec(String(c));
  if (!m) return [0, 0, 0, 1];
  const v = parseInt(m[1], 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255, 1];
}

/** The sRGB transfer, inverted: a uniform's background, bookkeeping for one colour. */
function srgbToLinear(v) {
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
}
