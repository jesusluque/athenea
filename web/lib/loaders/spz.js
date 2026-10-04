// Copyright (c) 2026 jesus luque.
//
// AN SPZ, v2 and v3: gunzipped by the browser (DecompressionStream, on the
// CPU: decompressing is a reader's job), its 16-byte header read here, and
// its attribute streams uploaded as they are. webSpzRecords turns them into
// the float records io::readSpz writes natively and webDecode decodes them
// with the same SPZ modes splatDecode has (byte opacity, SPZ scale and
// colour, the rotation's first or smallest three, byte harmonics, fixed-point
// positions, right-up-back turned to the engine's right-down-front). v1
// (float16 positions) was never released, as the native reader says; v4 is
// zstd, which no browser decompresses.

import { NO_FIELD, byteStream, readAll, makeCloud, decodeRecords } from "./cloud.js";

const MAGIC = 0x5053474e;   // "NGSP"
const SLICE = 64 << 20;

/** Whether `bytes` (the file as fetched) starts like a gzip stream. */
export const gzipped = (bytes) => bytes[0] === 0x1f && bytes[1] === 0x8b;

/** The decompressed file's header and where each stream starts. */
export function parseSpzHeader(bytes) {
  if (bytes.length < 16) throw new Error("SPZ: shorter than its header");
  const v = new DataView(bytes.buffer, bytes.byteOffset, 16);
  if (v.getUint32(0, true) !== MAGIC) throw new Error("not an SPZ file (no NGSP header)");
  const version = v.getUint32(4, true);
  const count = v.getUint32(8, true);
  const shDegree = v.getUint8(12);
  const fractionalBits = v.getUint8(13);
  const flags = v.getUint8(14);
  if (version === 1) throw new Error("SPZ v1: float16 positions (a format never released)");
  if (version < 1 || version > 3) throw new Error(`SPZ v${version}: this page reads v2 and v3`);
  if (shDegree > 4) throw new Error(`SPZ: harmonic degree ${shDegree}`);
  const shDim = [0, 3, 8, 15, 24][shDegree];
  const smallestThree = version >= 3;
  let at = 16;
  const streams = {};
  for (const [name, size] of [["positions", 9], ["alphas", 1], ["colours", 3], ["scales", 3],
    ["rotations", smallestThree ? 4 : 3], ["sh", shDim * 3]]) {
    streams[name] = at;
    at += count * size;
  }
  if (!(count > 0) || at > bytes.length) throw new Error(`SPZ: ${count} points do not fit its ${bytes.length} bytes`);
  return { version, count, shDim, keep: Math.min(shDim, 15), smallestThree, fractionalBits, flags, streams, end: at };
}

export async function loadSpz(gpu, source, { onProgress, slotsPerSplat = 1 } = {}) {
  const stream = await byteStream(source);
  const compressed = await readAll(stream, onProgress);
  let bytes = compressed;
  if (gzipped(compressed)) {
    const out = new Blob([compressed]).stream().pipeThrough(new DecompressionStream("gzip"));
    bytes = new Uint8Array(await new Response(out).arrayBuffer());
  } else if (new DataView(compressed.buffer, compressed.byteOffset).getUint32(0, true) !== MAGIC) {
    throw new Error(`${stream.name}: not a gzipped SPZ (v4's zstd is not read here)`);
  }
  const h = parseSpzHeader(bytes);
  const cloud = makeCloud(gpu, { count: h.count, restPerColour: h.keep, slotsPerSplat });
  // The streams, word-padded, in one buffer; then the records a slice at a time.
  const upload = gpu.create("spz.bytes", Math.ceil(h.end / 4) * 4);
  const padded = new Uint8Array(Math.ceil(h.end / 4) * 4);
  padded.set(bytes.subarray(0, h.end));
  gpu.device.queue.writeBuffer(upload, 0, padded);
  const floats = 14 + cloud.keep * 3;
  const sliceRecords = Math.max(256, Math.floor(Math.min(SLICE, gpu.maxBinding) / (floats * 4)));
  const raw = gpu.create("raw.slice", sliceRecords * floats * 4);
  const decode = {
    stride: floats, keepPerColour: cloud.keep, shWords: cloud.shWords,
    x: 0, y: 1, z: 2, opacity: 3, scale0: 4, scale1: 5, scale2: 6, dc0: 7, dc1: 8, dc2: 9,
    rotX: 10, rotY: 11, rotZ: 12, rotW: 13, restBase: 14, filePerColour: cloud.keep, restColourOuter: 0,
    opacityMode: 2, scaleMode: 2, colourMode: 3, rotationMode: h.smallestThree ? 3 : 2, restMode: 1, flipYZ: 1,
    positionScale: 1 / 2 ** h.fractionalBits,
    metallic: NO_FIELD, roughness: NO_FIELD, transmission: NO_FIELD, cryptoObject: NO_FIELD,
    transferBase: NO_FIELD, shadowBits: NO_FIELD, normal: NO_FIELD, emission: NO_FIELD, lobes: NO_FIELD,
  };
  for (let first = 0; first < h.count; first += sliceRecords) {
    const count = Math.min(sliceRecords, h.count - first);
    gpu.begin();
    const encoder = gpu.device.createCommandEncoder({ label: "spz" });
    const pass = encoder.beginComputePass();
    gpu.dispatch(pass, "webSpzRecords", count, {
      bytes: upload, raw,
      spz: { count, first, floats, keep: cloud.keep, shDim: h.shDim, smallestThree: h.smallestThree ? 1 : 0,
        positionsAt: h.streams.positions, alphasAt: h.streams.alphas, coloursAt: h.streams.colours,
        scalesAt: h.streams.scales, rotationsAt: h.streams.rotations, shAt: h.streams.sh },
    });
    pass.end();
    gpu.flush();
    gpu.device.queue.submit([encoder.finish()]);
    decodeRecords(gpu, cloud, raw, count, first, decode);
  }
  await gpu.device.queue.onSubmittedWorkDone();
  raw.destroy();
  upload.destroy();
  return cloud;
}
