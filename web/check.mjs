// Copyright (c) 2026 jesus luque.
//
// THE ENGINE AGAINST ITS KERNELS, ON THE CPU: lib/engine.js, every module and
// both loaders, run in node over a WebGPU stand-in that does no arithmetic
// and draws nothing, but holds every call to what a browser's validation
// would: each bind group names every binding the kernel's WGSL declares and
// nothing else, with the resource kind its declaration says (uniform,
// read-only or writable storage, a storage texture); storage offsets on 256
// bytes, ranges inside their buffers and under the binding size; no buffer
// both read-only and writable in one dispatch; at most 8 storage buffers a
// dispatch; groups under 65535 a dimension; copies inside their buffers and
// with their usages. Clouds are made up here as PLYs in memory -- the bytes of
// a file, not data anything computes from -- and an SPZ is read from disk
// when one is given; each is loaded, its levels of detail built, and drawn
// with every module's switches turned, at each tier. The counts a readback
// would return are made up too (a level's groups, the visible splats, the
// pairs, the drawn).
//
// What it cannot say: whether a picture is right. That is a GPU's run
// (docs/decisions.md, "WebGPU").
//
// Usage: node check.mjs [directory built by scripts/web-kernels.py] [an .spz]

import { readFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const dir = path.resolve(process.argv[2] ?? here);
let failures = 0;
const fail = (what) => { failures++; console.error("FAIL", what); };

// --- the stand-in ------------------------------------------------------------

const U = { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128, INDIRECT: 256, QUERY_RESOLVE: 512 };
globalThis.GPUBufferUsage = U;
globalThis.GPUTextureUsage = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 };
globalThis.GPUMapMode = { READ: 1, WRITE: 2 };
const limits = {
  maxStorageBufferBindingSize: 128 << 20, maxBufferSize: 256 << 20, minUniformBufferOffsetAlignment: 256,
  minStorageBufferOffsetAlignment: 256, maxComputeWorkgroupsPerDimension: 65535, maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16384,
};

/** The bindings a WGSL module declares: {binding: kind}. */
function declared(wgsl) {
  const out = {};
  for (const m of wgsl.matchAll(/@binding\((\d+)\)\s*@group\(0\)\s*var(<[^>]*>)?\s+\w+\s*:\s*([^;]+);/g)) {
    const access = m[2] ?? "";
    out[m[1]] = access.includes("uniform") ? "uniform" : access.includes("read_write") ? "storage"
      : access.includes("storage") ? "read-only-storage" : m[3].startsWith("texture_storage") ? "storage-texture" : "?";
  }
  return out;
}

const counters = { dispatches: 0, bindGroups: 0, byKernel: {} };
let pretend = { visible: 0, pairs: 0, drawn: 0, level: 0, splats: 0 };

class Buffer {
  constructor({ size, usage, label }) {
    this.size = size; this.usage = usage; this.label = label;
    if (size > limits.maxBufferSize) fail(`${label}: ${size} bytes is past maxBufferSize`);
  }
  destroy() { this.destroyed = true; }
  async mapAsync() {}
  getMappedRange(offset = 0, size = this.size - offset) {
    // What a readback would hold: the counts a frame asks for, made up.
    const a = new Uint32Array(size / 4);
    if (this.label === "frame.readback") { a[0] = pretend.visible; a[1] = pretend.pairs; if (a.length > 2) a[2] = pretend.drawn; }
    if (this.from === "bounds.result") new Float32Array(a.buffer).set([-1, -1, -1, 0, 1, 1, 1, 0]);
    if (this.from === "lod.total") a[0] = Math.min(8 ** ++pretend.level, pretend.splats);   // a level's groups
    return a.buffer;
  }
  unmap() {}
}

function makeDevice() {
  const device = {
    limits,
    lost: new Promise(() => {}),
    createBuffer: (d) => new Buffer(d),
    createShaderModule: ({ code, label }) => ({ code, label }),
    createComputePipeline({ compute, label }) {
      if (compute.entryPoint !== label) fail(`${label}: entry point ${compute.entryPoint}`);
      return { label, decl: declared(compute.module.code), getBindGroupLayout: function () { return { pipeline: this }; } };
    },
    createBindGroup({ layout, entries, label }) {
      counters.bindGroups++;
      const decl = layout.pipeline.decl;
      const seen = new Set();
      let storage = 0;
      const use = new Map();
      for (const { binding, resource } of entries) {
        const kind = decl[binding];
        if (!kind) { fail(`${label}: @binding(${binding}) is not declared`); continue; }
        seen.add(String(binding));
        if (kind === "storage-texture") {
          if (!resource.view) fail(`${label}: @binding(${binding}) wants a texture view`);
          continue;
        }
        const b = resource.buffer;
        if (!(b instanceof Buffer)) { fail(`${label}: @binding(${binding}) is not a buffer`); continue; }
        if (b.destroyed) fail(`${label}: @binding(${binding}) ${b.label} is destroyed`);
        const offset = resource.offset ?? 0;
        const size = resource.size ?? b.size - offset;
        if (offset + size > b.size) fail(`${label}: ${b.label} range ${offset}+${size} past ${b.size}`);
        if (kind === "uniform") {
          if (!(b.usage & U.UNIFORM)) fail(`${label}: ${b.label} has no UNIFORM usage`);
          if (offset % limits.minUniformBufferOffsetAlignment) fail(`${label}: uniform offset ${offset}`);
        } else {
          storage++;
          if (!(b.usage & U.STORAGE)) fail(`${label}: ${b.label} has no STORAGE usage`);
          if (offset % limits.minStorageBufferOffsetAlignment) fail(`${label}: ${b.label} offset ${offset} not on 256`);
          if (size > limits.maxStorageBufferBindingSize) fail(`${label}: ${b.label} binds ${size} bytes`);
          if (size % 4) fail(`${label}: ${b.label} binds ${size} bytes, not whole words`);
          const was = use.get(b);
          if (was && was !== kind) fail(`${label}: ${b.label} bound read-only and writable in one dispatch`);
          if (was === "storage" && kind === "storage") {
            // Two writable views of one buffer are allowed by WebGPU; this
            // host never needs one, so it is flagged.
            fail(`${label}: ${b.label} bound writable twice`);
          }
          use.set(b, kind);
        }
      }
      for (const b of Object.keys(decl)) if (!seen.has(b)) fail(`${label}: @binding(${b}) left unbound`);
      if (storage > limits.maxStorageBuffersPerShaderStage) fail(`${label}: ${storage} storage buffers`);
      return { label };
    },
    createCommandEncoder: () => ({
      beginComputePass: () => {
        let pipeline = null;
        return {
          setPipeline(p) { pipeline = p; },
          setBindGroup() {},
          dispatchWorkgroups(x, y = 1, z = 1) {
            counters.dispatches++;
            counters.byKernel[pipeline.label] = (counters.byKernel[pipeline.label] ?? 0) + 1;
            for (const g of [x, y, z]) if (!(g >= 1 && g <= limits.maxComputeWorkgroupsPerDimension)) fail(`${pipeline.label}: ${x} x ${y} x ${z} groups`);
          },
          end() {},
        };
      },
      copyBufferToBuffer(src, so, dst, d0, size) {
        if (so % 4 || d0 % 4 || size % 4) fail(`copy ${src.label}: offsets not on 4`);
        if (so + size > src.size || d0 + size > dst.size) fail(`copy ${src.label} -> ${dst.label}: out of range`);
        if (!(src.usage & U.COPY_SRC)) fail(`copy from ${src.label}: no COPY_SRC`);
        if (!(dst.usage & U.COPY_DST)) fail(`copy into ${dst.label}: no COPY_DST`);
        dst.from = src.label;
      },
      finish: () => ({}),
    }),
    queue: {
      writeBuffer(b, offset, data, dataOffset = 0, size) {
        const bytes = size ?? (data.byteLength - dataOffset);
        if (offset + bytes > b.size) fail(`writeBuffer ${b.label}: ${offset}+${bytes} past ${b.size}`);
        if (bytes % 4) fail(`writeBuffer ${b.label}: ${bytes} bytes`);
      },
      submit() {},
      onSubmittedWorkDone: async () => {},
    },
    destroy() {},
  };
  return device;
}

const device = makeDevice();
Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu: { requestAdapter: async () => ({ limits, features: new Set(), requestDevice: async () => device }) } } });

// A cloud of `count` splats with `rest` harmonic coefficients, as a file's bytes.
function plyBytes(count, rest) {
  const names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"];
  for (let i = 0; i < rest; ++i) names.push(`f_rest_${i}`);
  names.push("opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3");
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${count}\n` +
    names.map((n) => `property float ${n}`).join("\n") + "\nend_header\n";
  const head = new TextEncoder().encode(header);
  const out = new Uint8Array(head.length + count * names.length * 4);
  out.set(head);
  return out;
}

globalThis.fetch = async (url) => {
  const u = String(url);
  if (u.startsWith("mem:")) {
    const [count, rest] = u.slice(4).replace(/\.ply$/, "").split(":").map(Number);
    const bytes = plyBytes(count, rest);
    let sent = false;
    return {
      ok: true, status: 200, headers: { get: () => String(bytes.length) },
      body: {
        getReader: () => ({
          read: async () => {
            if (sent) return { done: true };
            sent = true;
            return { done: false, value: bytes };
          },
          cancel: async () => {},
        }),
      },
    };
  }
  const file = fileURLToPath(u);
  const data = await readFile(file);
  return { ok: true, json: async () => JSON.parse(data), text: async () => data.toString() };
};

// --- the run -----------------------------------------------------------------

const { Engine } = await import(pathToFileURL(path.join(dir, "lib/engine.js")));
const canvas = {
  width: 1280, height: 720,
  getContext: () => ({ configure() {}, unconfigure() {}, getCurrentTexture: () => ({ createView: () => ({ view: true }) }) }),
};
const draw = async (engine, count) => {
  for (const [visible, pairs, drawn] of [[0, 0, 0], [count >> 1, count * 3, count >> 1], [count, 20000000, 2 * count]]) {
    pretend.visible = visible; pretend.pairs = pairs; pretend.drawn = drawn;
    engine.setCamera({ position: [0, 0.5, 3], target: [0, 0, 0], fov: 50 });
    const said = engine.setFeatures({ shDegree: visible & 3, antialias: true, linear: pairs > count, lod: drawn !== count, lodThreshold: 1.5, exposure: 1.2, nothing: 1 });
    if (said.nothing !== false || said.shDegree !== true || said.lod !== true) fail(`setFeatures: ${JSON.stringify(said)}`);
    const stats = await engine.frame();
    if (stats.visible > visible) fail(`frame: visible ${stats.visible}`);
    if (stats.pairs + stats.pairsDropped !== pairs) fail(`frame: pairs ${stats.pairs} + ${stats.pairsDropped}`);
  }
};
for (const tier of ["T1", "T2", "T3"]) {
  const engine = await Engine.create({ canvas, base: pathToFileURL(dir + "/"), tier });
  if (engine.tier !== tier) fail(`tier ${engine.tier}, asked ${tier}`);
  if (engine.options().length < 7) fail(`options: ${engine.options().length}`);
  // The sizes of the site's test clouds: soar (200 k, SH3), pawn-r10 (730 k,
  // SH0), pawn-hq (1.88 M, SH0), and a small one.
  for (const [count, rest] of [[1000, 45], [200258, 45], [730559, 0], [1875795, 0]]) {
    pretend.level = 0; pretend.splats = count;
    const loaded = await engine.load({ url: `mem:${count}:${rest}.ply`, transform: { rotation: [180, 0, 0], scale: 0.01 } });
    if (loaded.count !== count) fail(`load ${count}: ${JSON.stringify(loaded)}`);
    await draw(engine, count);
  }
  const spz = process.argv[3];
  if (spz) {
    const blob = new Blob([await readFile(spz)]);
    const head = new Uint8Array(await new Response(blob.stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()).subarray(0, 16);
    pretend.level = 0; pretend.splats = new DataView(head.buffer, head.byteOffset).getUint32(8, true);
    const loaded = await engine.load({ blob, format: "spz" });
    if (loaded.count !== pretend.splats) fail(`spz: ${loaded.count} of ${pretend.splats}`);
    await draw(engine, loaded.count);
  }
  engine.dispose();
}
console.log(`${counters.dispatches} dispatches, ${counters.bindGroups} bind groups:`,
  Object.entries(counters.byKernel).map(([k, n]) => `${k} ${n}`).join(", "));
if (failures) {
  console.error(`${failures} failures`);
  process.exit(1);
}
console.log("ok: every dispatch binds what its kernel declares, within the web's default limits");
