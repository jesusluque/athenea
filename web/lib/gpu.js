// Copyright (c) 2026 jesus luque.
//
// THE KERNEL RUNNER: the manifest scripts/web-kernels.py writes, on a page's
// WebGPU device. Pipelines by kernel name and override value, bind groups by
// the parameter names Slang's reflection gave (never by slot), uniforms
// written field by field at the manifest's offsets into one ring a batch,
// buffers held to the device's binding size. No arithmetic on data: that is
// the kernels'.

export const roundUp = (n, k) => Math.ceil(n / k) * k;

/** PrefixSum::chunkFor and RadixSort::chunkFor (the same rule). */
export function chunkFor(count) {
  let chunk = 256;
  while (chunk < 4096 && Math.floor(count / chunk) > 2048) chunk *= 2;
  return chunk;
}

/** FrameParams.h's bitsFor. */
export function bitsFor(values) {
  let bits = 1;
  while (bits < 32 && 2 ** bits < values) ++bits;
  return bits;
}

const DIGITS = 256;
const RING = 512 << 10;   // a batch's uniforms

export class Gpu {
  /** `base`: the URL of the directory holding manifest.json. */
  static async create(device, base) {
    const gpu = new Gpu();
    gpu.device = device;
    gpu.limits = device.limits;
    const manifest = await (await fetch(new URL("manifest.json", base))).json();
    if (manifest.format !== "athenea-webgpu-kernels" || manifest.version !== 1) {
      throw new Error(`manifest: ${manifest.format} version ${manifest.version} is not one this page reads`);
    }
    gpu.manifest = manifest;
    gpu.sources = {};
    await Promise.all(Object.entries(manifest.kernels).map(async ([name, k]) => {
      gpu.sources[name] = await (await fetch(new URL(k.file, base))).text();
    }));
    gpu.pipelines = new Map();
    gpu.modules = new Map();
    gpu.buffers = {};
    gpu.placeholders = [0, 1, 2].map((k) => device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE, label: `placeholder.${k}` }));
    gpu.ring = device.createBuffer({ size: RING, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: "uniforms" });
    gpu.ringData = new ArrayBuffer(RING);
    gpu.ringUsed = 0;
    return gpu;
  }

  has(name) {
    return !!this.manifest.kernels[name];
  }

  pipeline(name, constants = {}) {
    const key = name + JSON.stringify(constants);
    let p = this.pipelines.get(key);
    if (!p) {
      const k = this.manifest.kernels[name];
      if (!k) throw new Error(`no kernel '${name}' in the manifest`);
      let module = this.modules.get(name);
      if (!module) {
        module = this.device.createShaderModule({ code: this.sources[name], label: name });
        this.modules.set(name, module);
      }
      // Override constants by their @id, which is how Slang names them.
      const byId = {};
      for (const [cname, value] of Object.entries(constants)) {
        const o = k.overrides[cname];
        if (!o) throw new Error(`${name}: no override '${cname}'`);
        byId[String(o.id)] = Number(value);
      }
      p = this.device.createComputePipeline({ label: name, layout: "auto", compute: { module, entryPoint: name, constants: byId } });
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

  /** Starts a batch's uniforms: everything dispatched until `flush` shares the ring. */
  begin() {
    this.ringUsed = 0;
  }

  /** Writes the batch's uniforms; call before submitting what was encoded since `begin`. */
  flush() {
    if (this.ringUsed) this.device.queue.writeBuffer(this.ring, 0, this.ringData, 0, roundUp(this.ringUsed, 4));
    this.ringUsed = 0;
  }

  uniform(kernel, param, values) {
    const bytes = this.uniformBytes(kernel, param, values);
    const at = roundUp(this.ringUsed, this.limits.minUniformBufferOffsetAlignment);
    if (at + bytes.byteLength > RING) throw new Error("the batch's uniforms overflow their ring");
    this.ringUsed = at + bytes.byteLength;
    new Uint8Array(this.ringData, at, bytes.byteLength).set(new Uint8Array(bytes));
    return { buffer: this.ring, offset: at, size: bytes.byteLength };
  }

  /**
   * One dispatch of `name` over `threads` (a count, or [x, y]): `bind` maps
   * every parameter the kernel declares to a buffer, a {buffer, offset, size}
   * range, a texture view, or (a uniform) an object of field values.
   */
  dispatch(pass, name, threads, bind, constants) {
    const k = this.manifest.kernels[name];
    if (!k) throw new Error(`no kernel '${name}' in the manifest`);
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
    const most = this.limits.maxComputeWorkgroupsPerDimension;
    if (gx > most || gy > most) throw new Error(`${name}: ${gx} x ${gy} groups is past the device's ${most}`);
    if (gx > 0 && gy > 0) pass.dispatchWorkgroups(gx, gy, 1);
  }

  /** The largest storage binding (and buffer) this device takes, in bytes. */
  get maxBinding() {
    return Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize);
  }

  /** A storage buffer of at least `bytes`, kept under `name` and grown when asked for more. */
  storage(name, bytes, extra = 0) {
    const old = this.buffers[name];
    if (old && old.size >= bytes) return old;
    old?.destroy();
    const size = Math.max(16, roundUp(bytes, 16));
    if (size > this.maxBinding) throw new Error(`${name}: ${size} bytes is past the device's binding size ${this.maxBinding}`);
    const b = this.device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extra, label: name });
    this.buffers[name] = b;
    return b;
  }

  /** A buffer the caller owns. */
  create(label, bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const size = Math.max(16, roundUp(bytes, 16));
    if ((usage & GPUBufferUsage.STORAGE) && size > this.maxBinding) {
      throw new Error(`${label}: ${size} bytes is past the device's binding size ${this.maxBinding}`);
    }
    return this.device.createBuffer({ size, usage, label });
  }

  /** Reads `bytes` from `buffer` at `offset` once what was submitted is done. */
  async read(buffer, offset, bytes) {
    const read = this.device.createBuffer({ size: roundUp(bytes, 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, label: "read" });
    const encoder = this.device.createCommandEncoder({ label: "read" });
    encoder.copyBufferToBuffer(buffer, offset, read, 0, roundUp(bytes, 4));
    this.device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const out = read.getMappedRange().slice(0);
    read.unmap();
    read.destroy();
    return out;
  }

  /** PrefixSum::apply: output[i] = input[0..i), total = the sum (a 4-byte range). */
  prefix(pass, input, output, total, count) {
    const chunk = chunkFor(count);
    const chunks = count === 0 ? 1 : Math.ceil(count / chunk);
    const chunkTotals = this.storage("prefix.chunkTotals", chunks * 4);
    const chunkStarts = this.storage("prefix.chunkStarts", chunks * 4);
    const params = { count, chunkSize: chunk, chunkCount: chunks };
    this.dispatch(pass, "prefixChunkTotals", chunks, { input, chunkTotals, params });
    this.dispatch(pass, "prefixChunkStarts", 1, { chunkTotals, chunkStarts, total, params });
    this.dispatch(pass, "prefixLocal", chunks, { input, chunkStarts, output, params });
  }

  /**
   * RadixSort::sort's chunked route (the tiled one wants 27648 bytes of group
   * memory), on 32-bit keys of `keyBits` bits; the result in `keys`/`values`.
   */
  sort(encoder, keys, values, scratchKeys, scratchValues, count, keyBits) {
    if (count <= 1) return;
    const chunk = chunkFor(count);
    const chunks = Math.ceil(count / chunk);
    const histogram = this.storage("radix.histogram", chunks * DIGITS * 4);
    const chunkStarts = this.storage("radix.chunkStarts", chunks * DIGITS * 4);
    const digitTotals = this.storage("radix.digitTotals", DIGITS * 4);
    let src = [keys, values], dst = [scratchKeys, scratchValues];
    const passes = Math.ceil(keyBits / 8);
    for (let p = 0; p < passes; ++p) {
      const params = { count, chunkSize: chunk, chunkCount: chunks, shift: p * 8, wide: 0 };
      const pass = encoder.beginComputePass({ label: `radix ${p}` });
      // Distinct placeholders for the unused hi words: one buffer read-only
      // and read-write in one dispatch is refused (RadixSort::sort).
      this.dispatch(pass, "radixHistogram", chunks, { keysLo: src[0], keysHi: this.placeholders[0], histogram, params });
      this.dispatch(pass, "radixTotals", DIGITS, { histogram, digitTotals, params });
      this.dispatch(pass, "radixStarts", DIGITS, { histogram, digitTotals, chunkStarts, params });
      this.dispatch(pass, "radixScatter", chunks, {
        srcKeysLo: src[0], srcKeysHi: this.placeholders[0], srcValues: src[1],
        dstKeysLo: dst[0], dstKeysHi: this.placeholders[1], dstValues: dst[1], chunkStarts, params,
      });
      pass.end();
      [src, dst] = [dst, src];
    }
    if (passes % 2 === 1) {
      encoder.copyBufferToBuffer(src[0], 0, keys, 0, count * 4);
      encoder.copyBufferToBuffer(src[1], 0, values, 0, count * 4);
    }
  }

  destroy() {
    for (const b of Object.values(this.buffers)) b.destroy();
    this.buffers = {};
    for (const b of this.placeholders) b.destroy();
    this.ring.destroy();
    this.pipelines.clear();
    this.modules.clear();
  }
}
