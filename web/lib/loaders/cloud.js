// Copyright (c) 2026 jesus luque.
//
// A CLOUD ON THE DEVICE, and what every loader shares: a file's bytes as a
// stream (fetch or a dropped file), the base arena B0 -- positions, shape,
// sh, in common/packing.slang's layout -- sized to the device, and the
// decode of float records into it a slice at a time (webDecode). The loaders
// read headers and move bytes; the GPU decodes.

export const NO_FIELD = 0xffffffff;
const SLICE = 64 << 20;   // records a decode takes at once, in bytes

/** Words of harmonics a splat for `keep` rest bases a colour (GpuClouds.cpp). */
export const shWordsFor = (keep) => (keep === 0 ? 1 : Math.floor((keep * 3 + 1) / 2));

/**
 * `{url}` or `{blob}` as a reader of byte chunks: `read()` as a stream
 * reader's, `loaded` bytes so far, `fraction()` of the whole or null.
 */
export async function byteStream(source) {
  let body, length = 0, name;
  if (source.blob) {
    body = source.blob.stream();
    length = source.blob.size;
    name = source.blob.name ?? "file";
  } else {
    const response = await fetch(source.url);
    if (!response.ok) throw new Error(`${source.url}: HTTP ${response.status}`);
    length = Number(response.headers.get("content-length")) || 0;
    body = response.body;
    name = String(source.url);
  }
  const reader = body.getReader();
  const s = {
    name, length, loaded: 0,
    async read() {
      const r = await reader.read();
      if (!r.done) s.loaded += r.value.length;
      return r;
    },
    fraction: () => (length ? s.loaded / length : null),
    cancel: () => { reader.cancel().catch(() => {}); },
  };
  return s;
}

/** Everything a stream holds, as one array (a compressed file is read whole). */
export async function readAll(stream, onProgress) {
  const parts = [];
  let total = 0;
  for (;;) {
    const { done, value } = await stream.read();
    if (done) break;
    parts.push(value);
    total += value.length;
    onProgress?.(stream.fraction(), stream.loaded);
  }
  const out = new Uint8Array(total);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

/**
 * B0 for `count` splats, the harmonics dropped a degree at a time until the
 * largest stream fits a binding: `slotsPerSplat` says how many entries a
 * splat may become (the level of detail's pool holds the merged levels too).
 */
export function makeCloud(gpu, { count, restPerColour, slotsPerSplat = 1, linear = false }) {
  const maxBind = gpu.maxBinding;
  const entries = Math.ceil(count * slotsPerSplat);
  let keep = Math.min(restPerColour, 15);
  while (keep > 0 && entries * shWordsFor(keep) * 4 > maxBind) keep = keep === 15 ? 8 : keep === 8 ? 3 : 0;
  // The largest stream a splat has after this is its projected record, 48 bytes (W0).
  if (Math.max(count, 1024) * 48 > maxBind) {
    throw new Error(`${count} splats: past this device's ${maxBind} bytes a binding (it holds ${Math.floor(maxBind / 48)})`);
  }
  const shWords = shWordsFor(keep);
  const cloud = {
    count, keep, restPerColour, shWords, linear,
    positions: gpu.create("B0.positions", count * 16),
    shape: gpu.create("B0.shape", count * 16),
    sh: gpu.create("B0.sh", count * shWords * 4),
  };
  cloud.destroy = () => { cloud.positions.destroy(); cloud.shape.destroy(); cloud.sh.destroy(); };
  return cloud;
}

/**
 * Float records into B0, `stride` bytes each, from an async source of byte
 * chunks; webDecode with `decode` (DecodeParams) a slice at a time.
 */
export async function decodeSlices(gpu, cloud, stride, decode, chunks, name = "cloud") {
  const sliceRecords = Math.max(256, Math.floor(Math.min(SLICE, gpu.maxBinding) / stride));
  const sliceBytes = sliceRecords * stride;
  const raw = gpu.create("raw.slice", sliceBytes);
  const staged = new Uint8Array(sliceBytes);
  let stagedBytes = 0;
  let decoded = 0;
  const flush = () => {
    const records = Math.min(Math.floor(stagedBytes / stride), cloud.count - decoded);
    if (records <= 0) return;
    gpu.device.queue.writeBuffer(raw, 0, staged, 0, records * stride);
    decodeRecords(gpu, cloud, raw, records, decoded, decode);
    decoded += records;
    const used = records * stride;
    staged.copyWithin(0, used, stagedBytes);
    stagedBytes -= used;
  };
  for await (const bytes of chunks()) {
    let at = 0;
    while (at < bytes.length && decoded < cloud.count) {
      const n = Math.min(bytes.length - at, sliceBytes - stagedBytes);
      staged.set(bytes.subarray(at, at + n), stagedBytes);
      stagedBytes += n;
      at += n;
      if (stagedBytes === sliceBytes) flush();
    }
    if (decoded + Math.floor(stagedBytes / stride) >= cloud.count) break;
  }
  flush();
  await gpu.device.queue.onSubmittedWorkDone();
  raw.destroy();
  if (decoded < cloud.count) throw new Error(`${name}: ${decoded} of ${cloud.count} records arrived`);
}

/** One slice of records, already in `raw`, decoded into B0 from splat `base`. */
export function decodeRecords(gpu, cloud, raw, records, base, decode) {
  gpu.begin();
  const encoder = gpu.device.createCommandEncoder({ label: "decode" });
  const pass = encoder.beginComputePass();
  gpu.dispatch(pass, "webDecode", records, {
    raw, positions: cloud.positions, shape: cloud.shape, sh: cloud.sh,
    params: { ...decode, count: records, base, recordBase: base },
  });
  pass.end();
  gpu.flush();
  gpu.device.queue.submit([encoder.finish()]);
}
