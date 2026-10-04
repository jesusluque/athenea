// Copyright (c) 2026 jesus luque.
//
// LOD: levels of detail on the page's device (E2), the native LodBuilder and
// CutSelector kernels (lod_common.slang says the method) with the cut
// compacted into a draw list, and 074's controller over its threshold.
//
// Built once after a load: Morton codes, the chunked radix sort, every
// level's groups (a boundary pass, a prefix sum, one count read a level, at
// load only), the pool -- the cloud's splats in Morton order, then each level's
// merged Gaussians, every part on 64 entries so that a part is a binding's
// range -- and the moments from the finest level up into the pool's parts.
//
// Every frame: each level's cut and the splats' into one `selected` over the
// pool, a prefix sum that places them, and webLodList's list of pool indices
// (web_list.slang), which E4 walks over a fixed number of slots: what the cut
// leaves out costs a word of the cut and nothing after it, and no count is
// read to make the list. The count the cut wanted comes back with the
// frame's stats, and the controller moves the threshold on it and on the
// frame time.

const LEVELS = 10;            // lod_common.slang's kLevels: 30-bit codes
const MAX_GROUP_FRACTION = 0.5;
const CHUNK = 1 << 16;        // LodBuildSettings::chunkSplats
const HEAD = 64;              // web_list.slang's kListHead
const ALIGN = 64;             // entries: 256 bytes of a uint, and of a 16-byte row
const align = (n) => Math.ceil(n / ALIGN) * ALIGN;

export const manifest = {
  id: "lod",
  version: "1",
  minTier: "T1",
  requires: ["core-raster"],
  conflicts: [],
  sections: ["S0"],
  arenas: [
    { name: "B0", access: "read", bytesPerSplat: "the pool: the splats and every merged level, ~1.6 x B0" },
    { name: "W2", access: "write", bytesPerSlot: 4, note: "the draw list" },
  ],
  hooks: [{ stage: "E2", slangInterface: null, implementation: "lodCutGroups, lodCutFinest, lodCutSplats, webLodList" }],
  flags: [{ name: "threshold", type: "f32", default: 1 }],
  ui: [
    { id: "lod", label: "Levels of detail", kind: "toggle", effect: "preset", default: true },
    { id: "lodAuto", label: "Hold the frame time", kind: "toggle", effect: "flag", default: true },
    { id: "lodThreshold", label: "Cell size (px)", kind: "slider", range: [0.5, 8], step: 0.25, effect: "flag", default: 1 },
  ],
  workgroupBytes: 0,
  kernels: ["lodMorton", "lodBoundaries", "lodGroups", "webLodReorder", "lodLeafMoments", "lodMergeMoments",
    "lodFinalize", "lodCutGroups", "lodCutFinest", "lodCutSplats", "webLodList"],
};

export function create(engine) {
  const gpu = engine.gpu;
  const policy = engine.policy;
  const values = { lod: true, lodAuto: true, lodThreshold: policy.startPx };
  let built = null;          // what load built
  let lastDrawn = 0;
  const control = { frames: [], changedAt: 0, raisedAt: -1e9, quietSince: 0 };

  /** Every level's groups, the pool, the moments: LodBuilder::build, on the page's device. */
  async function build(cloud) {
    const n = cloud.count;
    const b = cloud.bounds;
    const lo = b.min;
    const extent = Math.max(Math.max(b.max[0] - lo[0], b.max[1] - lo[1], b.max[2] - lo[2]), 1e-6) * 1.0001;
    const space = { boundsLoX: lo[0], boundsLoY: lo[1], boundsLoZ: lo[2], extent };
    const temp = [];
    const make = (label, bytes) => { const x = gpu.create(label, bytes); temp.push(x); return x; };
    const keys = make("lod.keys", n * 4), order = make("lod.order", n * 4);
    const keys2 = make("lod.keys2", n * 4), order2 = make("lod.order2", n * 4);
    const boundary = make("lod.boundary", n * 4), before = make("lod.before", n * 4);
    const total = make("lod.total", 16);

    gpu.begin();
    let encoder = gpu.device.createCommandEncoder({ label: "lod.morton" });
    let pass = encoder.beginComputePass();
    gpu.dispatch(pass, "lodMorton", n, { positions: cloud.positions, keys, values: order, params: { count: n, ...space } });
    pass.end();
    gpu.flush();
    gpu.device.queue.submit([encoder.finish()]);
    encoder = gpu.device.createCommandEncoder({ label: "lod.sort" });
    gpu.begin();
    gpu.sort(encoder, keys, order, keys2, order2, n, 30);
    gpu.flush();
    gpu.device.queue.submit([encoder.finish()]);

    // Each level's groups, coarse to fine, until cells hold under two splats.
    const levels = [];
    for (let r = 1; r <= LEVELS; ++r) {
      gpu.begin();
      encoder = gpu.device.createCommandEncoder({ label: `lod.level ${r}` });
      pass = encoder.beginComputePass();
      gpu.dispatch(pass, "lodBoundaries", n, { keys, boundary, params: { count: n, level: r } });
      gpu.prefix(pass, boundary, before, { buffer: total, offset: 0, size: 4 }, n);
      pass.end();
      gpu.flush();
      gpu.device.queue.submit([encoder.finish()]);
      const groups = new Uint32Array(await gpu.read(total, 0, 4))[0];
      if (groups > MAX_GROUP_FRACTION * n && r > 1) break;
      const level = {
        level: r, groups,
        group: gpu.create("lod.group", n * 4), starts: gpu.create("lod.starts", groups * 4), cells: gpu.create("lod.cells", groups * 4),
      };
      gpu.begin();
      encoder = gpu.device.createCommandEncoder({ label: `lod.groups ${r}` });
      pass = encoder.beginComputePass();
      gpu.dispatch(pass, "lodGroups", n, { keys, boundary, before, group: level.group, starts: level.starts, cells: level.cells, params: { count: n, level: r } });
      pass.end();
      gpu.flush();
      gpu.device.queue.submit([encoder.finish()]);
      levels.push(level);
    }

    // The pool: splats, then the levels, coarsest first, each on 64 entries.
    let at = align(n);
    for (const l of levels) { l.offset = at; at += align(l.groups); }
    const poolCount = at;
    const shWords = cloud.shWords;
    const pool = {
      count: poolCount,
      positions: gpu.create("lod.pool.positions", poolCount * 16),
      shape: gpu.create("lod.pool.shape", poolCount * 16),
      sh: gpu.create("lod.pool.sh", poolCount * shWords * 4),
    };
    gpu.begin();
    encoder = gpu.device.createCommandEncoder({ label: "lod.reorder" });
    pass = encoder.beginComputePass();
    gpu.dispatch(pass, "webLodReorder", n, {
      order, srcPositions: cloud.positions, srcShape: cloud.shape, srcSh: cloud.sh,
      positions: pool.positions, shape: pool.shape, sh: pool.sh, params: { count: n, shWords },
    });
    pass.end();
    gpu.flush();
    gpu.device.queue.submit([encoder.finish()]);

    // Moments from the finest level up; a Gaussian a group, into its part.
    const keep = cloud.keep;
    const stride = 13 + keep * 3;
    const range = (buffer, entries, bytes, count) => ({ buffer, offset: entries * bytes, size: Math.max(count, 1) * bytes });
    let fineMoments = null;
    const ph = gpu.placeholders;
    for (let k = levels.length - 1; k >= 0; --k) {
      const l = levels[k];
      const moments = gpu.create("lod.moments", l.groups * stride * 4);
      const p = { count: n, groups: l.groups, keep, shWords, stride, normals: 0, emission: 0 };
      gpu.begin();
      encoder = gpu.device.createCommandEncoder({ label: `lod.moments ${l.level}` });
      pass = encoder.beginComputePass();
      if (k === levels.length - 1) {
        gpu.dispatch(pass, "lodLeafMoments", l.groups, {
          positions: pool.positions, shape: pool.shape, sh: pool.sh, starts: l.starts,
          normals: ph[0], emission: ph[0], moments, params: p,
        });
      } else {
        const fine = levels[k + 1];
        gpu.dispatch(pass, "lodMergeMoments", l.groups, {
          starts: l.starts, fineGroup: fine.group, fineMoments, moments, params: { ...p, fineGroups: fine.groups },
        });
      }
      gpu.dispatch(pass, "lodFinalize", l.groups, {
        moments,
        positions: range(pool.positions, l.offset, 16, l.groups), shape: range(pool.shape, l.offset, 16, l.groups),
        sh: range(pool.sh, l.offset, shWords * 4, l.groups), normals: ph[1], emission: ph[2], params: p,
      });
      pass.end();
      gpu.flush();
      gpu.device.queue.submit([encoder.finish()]);
      if (fineMoments) temp.push(fineMoments);
      fineMoments = moments;
    }
    temp.push(fineMoments);
    const finest = levels[levels.length - 1];
    for (const l of levels) if (l !== finest) temp.push(l.group);
    // In memory every chunk is on the device: a 1 a chunk, written once.
    const chunks = Math.ceil(n / CHUNK);
    const resident = gpu.create("lod.resident", chunks * 4);
    gpu.device.queue.writeBuffer(resident, 0, new Uint32Array(chunks).fill(1));
    await gpu.device.queue.onSubmittedWorkDone();
    for (const x of temp) x.destroy();
    let merged = 0;
    for (const l of levels) merged += l.groups;
    return { n, space, levels, finest, pool, chunks, resident, merged, selected: null, capacity: 0 };
  }

  /** The draw list's slots: the tier's budget, under what a binding holds. */
  function capacityFor(b) {
    const byBinding = Math.floor(gpu.maxBinding / 48);
    return Math.min(b.pool.count, Math.max(policy.drawn, 1 << 16), byBinding);
  }

  function frameBuffers(b) {
    if (b.selected) return;
    b.capacity = capacityFor(b);
    b.selected = gpu.create("lod.selected", b.pool.count * 4);
    b.dest = gpu.create("lod.dest", b.pool.count * 4);
    b.state = gpu.create("lod.state", b.finest.groups * 4);
    b.listWords = gpu.create("lod.list", (HEAD + b.capacity) * 4);
  }

  /** 074 section 4.1, on the frame times and the drawn count the engine reports. */
  function steer(stats) {
    const now = performance.now();
    const target = policy.targetMs;
    control.frames.push({ at: now, ms: stats.ms });
    while (control.frames.length && now - control.frames[0].at > 3000) control.frames.shift();
    const recent = control.frames.filter((f) => now - f.at <= 500);
    const over = (fs) => (fs.length ? fs.filter((f) => f.ms > 1.5 * target).length / fs.length : 0);
    const sorted = control.frames.map((f) => f.ms).sort((a, b) => a - b);
    const median = sorted.length ? sorted[sorted.length >> 1] : 0;
    const t = values.lodThreshold;
    let next = t;
    if (lastDrawn > built.capacity && now - control.changedAt > 250) next = t * 1.25;
    else if (recent.length >= 5 && over(recent) > 0.1 && now - control.changedAt > 1000) next = t * 1.25;
    else if (control.frames.length >= 30 && control.frames[0].at < now - 2900 && over(control.frames) < 0.02 &&
      median < 0.7 * target && lastDrawn * 1.25 < built.capacity && now - control.changedAt > 1000 &&
      now - control.raisedAt > 10000) next = t / 1.25;
    next = Math.min(8, Math.max(policy.floorPx, next));
    if (next !== t) {
      if (next > t) control.raisedAt = now;
      control.changedAt = now;
      control.frames.length = 0;
      values.lodThreshold = next;
    }
  }

  return {
    values,
    get active() { return !!built && values.lod; },
    async load(cloud) {
      built = null;
      if (cloud.count < 4096) return;   // nothing to merge
      built = await build(cloud);
      // The cloud is the pool's first part from here: Morton order.
      cloud.destroy();
      cloud.positions = built.pool.positions;
      cloud.shape = built.pool.shape;
      cloud.sh = built.pool.sh;
      cloud.destroy = () => {};
    },
    set(id, value) {
      if (id === "lod") values.lod = !!value;
      else if (id === "lodAuto") values.lodAuto = !!value;
      else if (id === "lodThreshold") { values.lodThreshold = Math.max(0, Number(value)); values.lodAuto = false; }
      else return false;
      return true;
    },
    /** E2: the cut into the draw list, before the projection; the frame then walks the list. */
    prepare(frame, pass) {
      if (!built || !values.lod) return;
      const b = built;
      frameBuffers(b);
      const cut = {
        ...b.space, eyeX: frame.eyeObject[0], eyeY: frame.eyeObject[1], eyeZ: frame.eyeObject[2],
        pixelsPerUnit: frame.common.focalX, threshold: values.lodThreshold, orthographic: 0,
        chunkSplats: CHUNK, chunks: b.chunks, splats: b.n,
      };
      b.levels.forEach((l, k) => {
        const finest = l === b.finest;
        const bind = {
          cells: l.cells, selected: { buffer: b.selected, offset: l.offset * 4, size: l.groups * 4 },
          params: { ...cut, count: l.groups, level: l.level, coarsest: k === 0 ? 1 : 0 },
        };
        if (finest) Object.assign(bind, { starts: l.starts, residentChunks: b.resident, state: b.state });
        gpu.dispatch(pass, finest ? "lodCutFinest" : "lodCutGroups", l.groups, bind);
      });
      gpu.dispatch(pass, "lodCutSplats", b.n, {
        groups: b.finest.group, state: b.state, selected: { buffer: b.selected, offset: 0, size: b.n * 4 },
        params: { ...cut, count: b.n, offset: 0 },
      });
      gpu.prefix(pass, b.selected, b.dest, { buffer: b.listWords, offset: 0, size: 4 }, b.pool.count);
      gpu.dispatch(pass, "webLodList", b.pool.count, {
        selected: b.selected, dest: b.dest, listWords: b.listWords, list: { count: b.pool.count, capacity: b.capacity },
      });
      frame.source = b.pool;
      frame.listWords = b.listWords;
      frame.slots = b.capacity;
      frame.constants.kListed = true;
      frame.counters.push({ name: "drawn", buffer: b.listWords, offset: 0 });
    },
    after(stats) {
      if (!built || !values.lod) return;
      lastDrawn = stats.counters.drawn ?? 0;
      if (values.lodAuto) steer(stats);
    },
    stats() {
      if (!built) return { built: false };
      return {
        built: true, on: values.lod, threshold: values.lodThreshold, auto: values.lodAuto,
        drawn: lastDrawn, slots: built.capacity, dropped: Math.max(0, lastDrawn - built.capacity),
        levels: built.levels.length, merged: built.merged, pool: built.pool.count,
      };
    },
    dispose() {
      if (!built) return;
      const b = built;
      for (const l of b.levels) { l.starts.destroy(); l.cells.destroy(); }
      b.finest.group.destroy();
      for (const x of [b.pool.positions, b.pool.shape, b.pool.sh, b.resident, b.selected, b.dest, b.state, b.listWords]) x?.destroy();
      built = null;
    },
  };
}
