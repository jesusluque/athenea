# Decisions

Why the engine is the way it is, with the measurements behind each choice.
Numbers are from an Apple M5 Pro (Metal) unless a section says otherwise.

## Ray tracing Gaussians

`athenea::render::GaussianRayTracer`, kernels in `shaders/athenea/rt/`.

### What is drawn

Each particle is evaluated in 3D at its peak response along the ray, with the
rasteriser's opacity-aware cut (`min(2 ln(255 alpha), 9)`). Particles are
blended front to back ordered by where they peak along the ray. The colour of
a particle is the rasteriser's: its harmonics evaluated for the direction from
the eye to its centre, once per frame (`rt_shade.slang`).

It deliberately differs from the rasteriser in three ways:

- No EWA approximation. The rasteriser projects with the local affine map;
  the ray tracer is exact for any lens.
- No 0.3 px screen-space dilation.
- Order by peak along the ray, not by centre depth.

### Ground truth

`ReferenceRenderer::renderPeaks` is the ray tracer's GPU reference. It
evaluates every particle for every pixel and sorts exactly by peak, with no
BVH, segments or carry. The ray tracer is held to it at p99 of at most one
8-bit sRGB code value, with at most 0.05% of pixels over two
(`tests/render/test_ray_tracing.cpp`).

Against the rasteriser, the tolerance only holds where the two definitions
agree:

| Scene | p99 tolerance |
|---|---|
| Orthographic, sparse, large particles (EWA is exact) | 2 |
| Perspective, sparse, particles of a few pixels | 4 |

Dense scenes differ by tens of code values. That difference is ordering, not
error. As particles grow, the perspective gap grows with them: p99 is 0 at
sizes 0.03–0.05, 2 at 0.1–0.15, and 20 at 0.3–0.5.

### Two routes, one integrator

Both routes offer entries to the same `Integrator`
(`rt_integrate.slang`), which records, evaluates, orders, blends and
segments:

- **Hardware.** Every particle is a stretched icosahedron (3DGRT's proxy) in
  a device BVH, queried with inline `RayQuery`. Metal and Vulkan.
- **ComputeBvh.** A Karras LBVH built on the GPU and traversed in compute.
  The build is Morton codes, radix sort, the hierarchy, then refit until a
  pass changes nothing. Traversal meets each particle's cut ellipsoid
  directly. It runs on every device, CUDA included.

Measured on train_7k (742k splats) at 1080p:

| Route | Frame | Build |
|---|---|---|
| Hardware | 435 ms | 731 ms |
| ComputeBvh | 343 ms | 168 ms |

On Metal, a non-opaque candidate costs a round trip out of hardware
traversal. An opaque closest-hit query over the same 15M proxy triangles
takes 6 ms, and merely enumerating every candidate takes 255 ms. `Auto`
therefore picks ComputeBvh on Metal. On Vulkan it picks Hardware, which has
not been measured. For comparison, the tile rasteriser draws the same frame
in 13 ms.

### What made it fast

In order, all measured on the same frame:

1. **Starting point: a 3DGRT k-buffer, 2.52 s.** It kept 16 hits sorted
   during traversal and restarted traversal from the 16th entry.
2. **Close segments where entries are dropped, 1.94 s.** Commit the farthest
   dropped entry, and stop committing only when an entry lands exactly in
   slot 16.
3. **Unsorted record, evaluate after traversal, 658 ms.** Per candidate, only
   record the distance, primitive and instance. Evaluate and sort after
   traversal ends. The record holds 256 entries; the sweep across sizes was
   64 → 845 ms, 128 → 657, 192 → 558, 256 → 530, 384 → 548 and 512 → 553.
4. **Precomputed data, no change (652 ms).** Per-particle frames and
   per-frame colours are cleaner, but did not move the time.
5. **Leave the record arrays uninitialised: −48 ms.**
6. **Sort an index permutation, not four arrays: −49 ms.**
7. **Counting sort into 128 buckets before insertion sort: −85 ms** (compute
   route). Insertion sort alone did about 2500 moves per ray. The bucket
   sweep was 32 → 362 ms, 64 → 351, 128 → 344 and 256 → 342.

Tried and rejected:

- **Sorted keys beside the indices.** No change.
- **Carrying peak and alpha from the compute route's leaf test into the
  record.** Slower, 344 → 389 ms.
- **Sharing code through struct methods.** This cost 11% until the scalars
  used per candidate moved into a separate small `Cursor` local. With the
  split, the shared version runs at the same speed as the hand-inlined one.

### Segments, order and carry

A traversal records up to 256 entries. When a ray enters more proxies than
that, the record closes at the nearest entry it had to let go. The next
traversal starts a hair before that point (a relative 1e-5) and recognises by
ID what the overlap reports again.

Restarting a hair *after* the boundary lost particles, as measured against
the reference. Starting exactly at it works on Metal, and the overlap keeps
it working on an intersector that does not report one triangle at one
distance twice.

A particle always peaks inside its proxy, so particles peaking beyond the
closing entry are carried into the next segment. Only when more than 64 are
carried are the nearest blended early; that is the only approximation.
Blending in entry order instead drew every proxy's silhouette as a seam.

A camera inside a particle's bound does not see that particle. Only entries
ahead of the ray start count, which is also what stops a restarted segment
from taking a particle twice.

### Bugs found on the way

These were fixed in `cmake/patches/slang-rhi-metal-acceleration-structures.patch`
or in the engine:

- **Freed structures crashed the next build.** Once any acceleration
  structure had been freed, slang-rhi's Metal backend put a nil into the
  device-wide structure array. The next build threw inside
  `NSArray initWithObjects:` and aborted the process. Freed slots now get
  empty placeholder structures.
- **Indexed builds read past their index window.** slang-rhi's Metal backend
  took `max(vertexCount, indexCount) / 3` as the triangle count. A chunk that
  indexed into a larger vertex buffer therefore read past its own indices.
  The count is fixed in the patch, and the engine now gives each chunk
  chunk-local vertex windows as well.
- **Winding.** Facing is decided in object space, so a mirroring instance
  transform must not flip the front face. Metal behaves this way, and the
  Vulkan and DXR specifications say the same.
- **Instances share particle IDs.** De-duplication has to key on the pair of
  particle and instance.
- **Flat particles.** The textbook ray–ellipsoid discriminant `b² − ac`
  cancels to nothing for very flat particles; it lost 176 of 400 thin layers.
  The compute route now computes the peak and the distance to it instead.

### Not done, not verified

- The OptiX pipeline route (CUDA). CUDA uses ComputeBvh instead.
- Vulkan and CUDA runs of either route. These need the Linux host.
- Rays that are not primary rays: shadows and reflections. The integrator
  takes any `RayDesc`, but nothing traces secondary rays yet.

## SPZ

`io::readSpz`, `third_party/spz`, `shaders/athenea/scene/splat_encoding.slang`.

### Who does what

Niantic's reference reader (MIT, vendored as openFXplayer vendors it) only
decompresses. Version 2 and 3 files are gzip; version 4 is zstd.

The CPU arranges the quantised bytes into float records:

- A 24-bit fixed-point position is parsed into its integer.
- A smallest-three quaternion is split into two 16-bit halves, because a
  float holds 16 bits exactly and not 32.

The GPU decode does the rest:

- the fixed-point scale;
- `byte / 16 - 10` log scales;
- the DC term at SPZ's 0.15 scale;
- both quaternion packings;
- `(byte - 128) / 128` harmonics;
- the turn from SPZ's right-up-back to the right-down-front a 3DGS PLY is in:
  y and z of positions and rotations negate, and each harmonic basis takes
  the sign of its parity in y and z.

### How it is checked

Two tests check it:

- **Hand-written files** with exact decoded values: version 3 with degree-1
  harmonics, and version 2.
- **Niantic's packer against the PLY it packed.** A degree-3 cloud goes
  through Niantic's packer (versions 3 and 4) and is rendered against the
  same cloud read as a PLY.

Results of the second test, p99 in 8-bit sRGB code values:

| Harmonics rendered | p99 | Same comparison with bands 2 and 3 signs wrong |
|---|---|---|
| None | 3 | |
| Degree 1 (5-bit) | 4 | |
| Degree 2 (4-bit) | 10 | 136 |
| Degree 3 (4-bit) | 13 | 234 |

The difference that remains is quantisation. Sign errors are ruled out: the
deliberate control is an order of magnitude worse.

Degree-4 files load with the fourth band dropped, because the engine
evaluates up to degree 3.

## SOG

`io::readSog`, `shaders/athenea/scene/sog_decode.slang`, `scene::loadSplatFile`.

### Who does what

The CPU only unpacks:

- opens the zip, stored or deflated, or the directory beside a `meta.json`;
- decodes each WebP with libwebp to its raw RGBA bytes, never premultiplied,
  since these channels are indices;
- parses ranges and codebooks with nlohmann/json.

The GPU does the reconstruction:

- 16-bit positions in their signed log domain;
- version 2 codebooks and version 1 ranges;
- the smallest-three quaternion with its mode byte;
- the higher-harmonics palette.

It writes records in the engine's float encoding, which then take the same
validate and decode as every other format. Loading a SOG sends nothing back
to the CPU. The USD export, which consumes host records, reads them back
(`CloudLoader::records`).

### How it is checked

PlayCanvas's own converter wrote the fixtures in `tests/data/splats`. The
converter reorders splats and clusters harmonics, so the test compares
renders against the source PLY rather than splat by splat:

| Fixture | Render | p99 |
|---|---|---|
| `tiny`, version 2 | no harmonics | 5 (8-bit codebooks) |
| `sh3`, 64-entry palette | no harmonics | 1 |
| `sh3`, 64-entry palette | degree-3 harmonics | 1 |
| `sh3`, palette red and blue exchanged | degree-3 harmonics | 170 |

A first `sh3` of 200 splats had a 128-entry palette. Its k-means loss on
random harmonics gave p99 61, which proves nothing about decoding. The
fixture was cut to one palette entry per splat.

## SplatEdit and the athenea schemas

`shaders/athenea/common/edit.slang`, `render::SplatEdit`, `modules/usd/schemas`.

### One rule, every renderer

A SplatEdit is openFXplayer's, rule for rule: a box or sphere in the cloud's
own space, what happens to the splats inside it (keep, remove, grade), a
grade (tint, brightness, saturation about Rec.709 luma, opacity), and two
filters that apply wherever the volume is (minimum opacity, maximum scale).

The rule is written once and read by four renderers:

- the tile rasteriser;
- the rasteriser's GPU reference;
- the ray tracer, in its per-instance shade pass, carrying the edited opacity
  that the integrator then cuts by;
- the ray tracer's GPU reference.

An edit belongs to an instance, not to a cloud, so two instances of one
cloud can be edited differently.

Checks, p99 in 8-bit sRGB code values:

| Test | Result |
|---|---|
| Rasteriser vs its reference, keep, remove, inverted grade, filters | 0 |
| Ray tracer vs its reference, both routes, two differently edited instances | 0 |

### In USD

`AtheneaSplatEditAPI` is a codeless applied API schema. Its properties are
constant primvars `primvars:athenea:edit:*`, and constant primvars inherit down
the namespace. An edit authored on an Xform therefore stands over every
ParticleField below it, which is openFXplayer's Edit node over its subtree,
with no UsdImaging adapter to write. A Hydra render of such a stage matches
the direct render with the same edit at p99 1.

`AtheneaPointStyleAPI` declares the point primvars the delegate already read
(`athenea:sizeInPixels`, `athenea:edl`, `athenea:surfaceOffset`).

Both schemas are written by hand in usdGenSchema's output form, since this
OpenUSD build has no Python. They are installed beside hdAthenea, so one
`PXR_PLUGINPATH_NAME` finds both.

No `AtheneaCameraWindowAPI`: a UsdGeomCamera already expresses openFXplayer's
window. Translate is the aperture offsets, scale is the apertures, and roll
is the camera's own rotation. The SceneText bridge maps to those.
`AtheneaStreamedAssetAPI` waits for the LOD work.

## Levels of detail

`modules/lod`, `shaders/athenea/lod`. The method is written out in
`lod_common.slang`.

### Built and cut on the device

- **Build.** Splats are sorted by 30-bit Morton code. An octree cell at
  level r is a run of equal top-3r-bit prefixes, found with a boundary pass
  and a prefix sum. Moments add (after Kerbl et al. 2024): the finest merged
  level is summed from the splats, and each coarser level from its children.
  Each group then becomes one Gaussian, with its covariance diagonalised by
  Jacobi sweeps in the shader.
- **Which levels are stored.** Levels from `coarsestLevel` down to the
  deepest level whose cell count is at most half the splat count.
- **Cut, per group, fully parallel.** A group is drawn when its cell projects
  to at most the threshold and its parent's cell does not. A splat is drawn
  when the finest merged level's cell does not. Projected size is edge over
  nearest distance, and a child cell lies inside its parent, so the test is
  monotone down the tree and every place is drawn at exactly one level.
- **What comes back to the CPU.** Only counts: one per level while building,
  and one read per instance per frame, which holds every part's count plus,
  when streaming, each chunk's need.

### Measured (M5 Pro)

**train_7k (742k splats).** Building took 47 ms and made 146k merged
Gaussians over levels 1 to 10. On a far view at 1080p:

| Threshold | Drawn | Cut | Render | Image |
|---|---|---|---|---|
| 0 (off) | 742k | | 12.6 ms | reference |
| 4 px | 727k | 2.3 ms | 11.4 ms | mean abs 1e-4 |
| 8 px | 83k (11%) | 1.3 ms | 3.3 ms | mean abs 3e-3 |

**Random-colour test clouds**, the worst case for merging:

| Threshold | Drawn | p99 |
|---|---|---|
| 2 px | 64% | 9 |
| 4 px | 19.5% | 29 |

**Exactness at threshold 0.** p99 is at most 1. The residue comes from depth
keys that tie and keep index order, which the Morton sort has changed.

**A cell with one splat.** It merges back into that splat, with covariance
equal to 1e-4.

### Chunks, `.athc` and streaming

- **Chunks.** The cloud's own splats are cut into chunks: runs of
  `chunkSplats` of the Morton order (65536 by default), so each chunk is a
  compact piece of space. The merged levels are small and always on the
  device; chunks may or may not be. A chunk on the device sits in a slot of a
  store, and the store's slots need not follow the chunks' order. Built in
  memory, chunk c is slot c and every chunk is there. The cut draws one run of
  consecutive slots per dispatch, which is a single run in that case.
- **The finest merged level decides for its splats.** A finest-level group
  whose cell wants splats draws them when every chunk holding them is on the
  device, and draws its own Gaussian otherwise. Its merged Gaussian is the
  nearest resident ancestor of those splats, so a missing chunk never leaves
  a hole. Each splat stores the index of its finest-level group, which
  replaces the Morton key at frame time, and reads the group's decision. The
  test no longer runs per splat, and every place is still drawn exactly once.
- **What a view wants.** `lodChunkNeeds` gives, per chunk, the largest
  projected edge among the cells that want its splats: 0 for none, otherwise
  in 1/16 px. It comes back in the same single read as the counts.
- **Why chunks follow the Morton order and not group boundaries.** Fixed-size
  chunks give uniform slots, and a store of uniform slots never fragments.
  Groups that straddle a chunk boundary only need to check more than one
  chunk, which is a short loop in the finest-level kernel.
- **`.athc` v1.**
  - Layout: a header in page 0; level and chunk tables; the finest level's
    group starts; each level's positions, shape, SH and cells; each chunk's
    positions, shape, SH and finest group.
  - Every block begins on a 4096-byte page, so a chunk can be mapped and
    faulted in alone. The bytes use the device's own packing, so reading is
    a copy.
  - The writer writes `name.partial` and then renames it, so a failed write
    never looks like a whole file.
  - `athenea convert in.ply out.athc`.
- **`StreamingPool`.**
  - The file is mapped. The levels go to the device when the pool opens, and
    the store starts empty.
  - After each cut the caller passes its needs to `want`. `update` queues the
    missing chunks, most wanted first, only as many as have a place to go.
  - A place is, in order of preference: a free slot; the slot of a chunk not
    wanted now, least recently wanted first; or the slot of a chunk wanted
    less than half as much. The half stops two chunks from swapping every
    frame.
  - Loader threads copy a chunk off the mapping, so the page faults happen on
    those threads and not in the frame. The next `update` uploads it and
    flips its resident flag.
  - `update(true)` waits for the queued loads; an offline render repeats it
    until a frame places nothing. `athenea render --stream-budget N` does exactly
    that, and `athenea bench` streams without waiting.

### Measured: streaming (M5 Pro)

**train_30k (1.05M splats, 16 chunks).**

- `athenea convert` to `.athc` takes 2.8 s in total and writes 150 MB, against
  266 MB for the PLY.
- At 1080p with `--lod 2`, the PLY built in memory, the `.athc` read whole
  and the `.athc` streamed into 4 slots write byte-identical EXRs. The stream
  settles in 2 cuts (48 ms).
- At `--lod 0.25` with 4 of 16 slots, 12 wanted chunks do not fit, and their
  places are drawn merged. Loading by priority instead of chunk order raised
  the visible splats from 155k to 170k.

**Tests, 20k splats in 20 chunks.**

- A store with its slots reversed gives the same cut and the same image,
  max 0.
- Dropping the chunks a view does not want changes nothing, max 0.
- Missing chunks are drawn merged.
- With 8 slots, the frame the pool settles on is p99 0 against the whole
  cloud.
- Turning the camera to the other side evicts 4 chunks and settles back to
  max 0.

### In USD: `AtheneaStreamedAssetAPI`

- **The schema.** It is codeless, like the others, and its properties are
  constant primvars:
  - `athenea:asset`: the `.athc` file.
  - `athenea:lod:threshold`: pixels.
  - `athenea:stream:budget`: splats, where 0 reads the file whole.
- **What it does to the prim.** Authored on a ParticleField, the asset stands
  in for the prim's own arrays, which may be left empty.
- **Where the file is opened.** The engine opens it in `commit`, on the render
  pass's thread. It opens it again only when the path or budget changes; a
  new threshold alone does not reopen it.
- **One cut per frame.** Every asset is cut in a single `CutSelector` call,
  which is why `LodInstance` carries its own threshold: a second call would
  overwrite the clouds the first returned.
- **Waiting for streams.** `athenea:settleStreams` is a render setting. It is
  false by default, so a viewport fills in over the frames that follow.
  `StageRenderer`, which makes images, sets it true and cuts and loads until
  nothing more is placed.
- **The ray-traced technique.** It draws an asset read whole as its whole
  cloud, and does not draw streamed assets. A cut changes every frame, and
  the tracer would rebuild every frame.
- **Checked.** A stage referencing a `.athc`, read whole and streamed into
  8 of 20 slots, renders through Hydra with max 0 against the same cut and
  stream done directly.

### Not done yet
- **The cut still waits twice a frame:** once for its counts, which come back
  in one read (reading them level by level had cost 3.9 against 2.3 ms), and
  once for the gather.
- **LOD with the ray tracer.** A cut that changes every frame would rebuild
  the structures every frame.

## Time: FrameClock and `athenea live`

`modules/sched` (genlock underneath) and `apps/athenea/src/CmdLive.cpp`.

### What it does

- **The clock.** `FrameClock` runs free on this machine's clock, or follows a
  PTP master as a genlock `PtpClock` slave. Frame N of a rate begins at the
  instant ST 2059-1 gives it, computed from the TAI epoch, never summed. Two
  nodes following one master therefore agree on it without talking to each
  other.
- **Timecode.**
  - It is the UTC time of day, counted from the first frame that begins at or
    after midnight.
  - At 30000/1001 and 60000/1001 it is drop-frame: SMPTE 12M's labels ;00–;01
    (;00–;03 at 59.94) are skipped every minute not a multiple of ten. Other
    rates, 24000/1001 included, count non-drop.
  - `framesFromTimecode` inverts the count, and a test checks the round trip.
- **`athenea live stage.usd [--ptp host --port N] --rate R --frames N --at
  HH:MM:SS:FF -o out.####.exr`.**
  1. It waits for a lock.
  2. It renders one warm-up frame, because the first render loads the stage
     and compiles shaders.
  3. It waits for each frame's instant, renders the stage at that frame's USD
     time, and hands the EXR to a writer thread.
  - **Missed frames.** Frames whose instant passes during a render are
    skipped, not drawn late.
  - **What each EXR carries.** `timeCode` (SMPTE 12M BCD, OpenEXR's type),
    `framesPerSecond` (rational), `athenea:taiNs`, `athenea:frameIndex`,
    `athenea:usdTime`, `athenea:wakeLateMs` and `athenea:clock`.
- **`--at`.** It names the timecode at which `--start` plays. Without it,
  each node counts USD time from when it happened to start, so two nodes
  would draw different times for the same instant: correct frames, wrong
  content. With the same `--at`, they draw the same time.

### Waking on time

`std::this_thread::sleep_for` on macOS overran by 7 ms on average and 10 ms at
worst, measured, whatever the thread's QoS.

- **macOS.** `platform::sleepPrecisely` gives the thread a time-constraint
  (real-time) policy only while it sleeps. It must not keep it while it
  renders, because a thread that overruns its computation budget is demoted.
  The overrun drops to 36 µs at worst.
- **Linux.** The thread's timer slack is set to 1 ns instead.
- **The last 100 µs.** `FrameClock::waitFor` spends them yielding, and it
  rereads the clock because a PTP correction may have moved it.

### Measured (M5 Pro, loopback master `genlock-cli master --port 3190`)

- **Tests.** Free run: the latest of 10 wakes came 9 µs after its alignment
  point. Following the master: the clock reads 0.14–0.41 ms off it, the
  software-timestamp error on a loaded machine.
- **Two `athenea live` nodes on `train_7k` at 640x360 and 25 fps.**
  - The nodes started seconds apart and shared one GPU and one `--at`.
  - Both drew frames 44728571775 to 44728571799, from 16:07:14:00, at
    identical USD times.
  - 0 frames were skipped, the latest wake was 12 µs late, and renders took
    17 ms.
  - Writing EXRs inside the loop had cost 6 skipped frames in 15.

### Not done

- **Output.** It is EXR files only; nothing is sent to a video output or
  over the network.
- **Scanout.** Software cannot phase-lock a display (genlock's README says
  why); an SDI card is what would.
- **Linux PTP.** Untested here. Its kernel software timestamps should narrow
  the error.

## Complete USD: toolchain (M0)

This is the first milestone of the plan to render all of USD: geometry,
materials, lights, cameras, animation, curves, volumes and render settings.
The work proceeds in two techniques, an interactive raster and a path tracer,
with MaterialX feeding a Slang generator and `athenea view` as the viewer.

### What changed and why

- **OpenUSD 26.08 with MaterialX 1.39.5 and OpenVDB 10.1 (with NanoVDB)**,
  in `~/tools/usd-26.08-mx` via `scripts/build-usd.sh`.
  - **MaterialX:** 1.39.5 is the first release with a Slang shader generator
    (`MaterialXGenSlang`, on by default). `MATERIALX_SLANG_RHI_SOURCE_DIR`
    stays unset, since MaterialX's own Slang renderer would bring a second
    slang-rhi into the process.
  - **Build fix:** CMake 4 refuses c-blosc's `cmake_minimum_required`, so the
    script exports `CMAKE_POLICY_VERSION_MINIMUM=3.5`.
  - **Switching over:** the new prefix sat beside the old one until the engine
    passed 54/54 against it. Only then did the presets move.
- **Open Image Denoise 2.5.1, built from source by `scripts/build-oidn.sh`.**
  - **Why not the release binaries:** they ship their own `libtbb.12`, a
    second TBB beside USD's under the same soname.
  - **GPU devices only:** Metal here, CUDA on Linux. A denoiser that could
    fall back to the CPU is a CPU fallback.
  - **Sharing the queue:** `technique::Denoiser` opens OIDN on the engine's
    own Metal command queue.
  - **Result:** `athenea info` reports `denoiser OIDN 2.5.1 on Metal` and
    `tbb libraries 1`. The `single_tbb` test holds that count.
- **The aofx SDK is pinned.**
  - `aofx_sdk_manifest` hashes every header in
    `modules/aofx/sdk/include/aofx` against `tests/aofx/sdk_manifest.txt`.
  - The headers differ from openFXplayer's only in the doc comments corrected
    here; `kAbiVersion` is 22 in both.
  - Re-recording the manifest is allowed only when openFXplayer's SDK moved
    the same way.
- **Four places where the CPU did arithmetic on data, now on the device.**
  - **ParticleField and Points arrays:** Sync keeps the `VtValue`s Hydra hands
    it, float or half, with no copy. The commit uploads their bytes, and
    `scene/streams.slang` interleaves them into records. The per-element
    interleave loops and the half-to-float conversions on the host are gone.
    Half attributes draw as their float twins at p99 1.
  - **Hydra render buffers:** `usd/aov_convert.slang` fills them. It converts
    to the buffer's format and turns view z into the host projection's
    [0, 1], one thread per output word. Rows stay bottom first, as Storm and
    hdEmbree lay out Hydra buffers (M2 found the flip this first did). The per-pixel loops in
    `RenderBuffer::WriteColour` and the render pass are gone.
  - **StageRenderer:** it reads the engine's targets directly, so
    `StageImage.depth` is now view z, as `athenea render` writes it.
- **A latent configure bug.**
  - **Symptom:** in a fresh build directory, Catch2 was never fetched.
  - **Cause:** `include(Dependencies)` ran before `include(CTest)` defined
    `BUILD_TESTING`.
  - **Fix:** CTest is now included first.
- **Deferred to M3:** GLFW and Dear ImGui arrive with `athenea view`, their only
  user, rather than as unused dependencies now.

## Complete USD: GPU foundations (M1)

What the next milestones build on, in `modules/gpu`.

- **`Texture` and `Sampler`.**
  - They own their slang-rhi objects and report failures as `Result`.
  - Each subresource is uploaded with a single command and read back only for
    output and tests.
  - Engine images stay buffers, because kernels index them. A texture is for
    what needs one: material images with mips, render targets, depth.
- **`MipGenerator` (`algo/mips.slang`).** slang-rhi has no mip generation.
  - Each texel of a level is the area-weighted mean of the texels above it.
  - An odd edge of 2n + 1 folds into n with weights (n − x, n, x + 1) / (2n + 1).
  - Every source texel therefore gives exactly n/(2n + 1) of itself to the
    level below, so the chain keeps level 0's mean.
  - Measured on 64², 37×23, 1×9 and 128×5: the means agree to 2e-6, and the
    37×23 chain drifts only in the seventh decimal.
- **`RasterKernel`.** A vertex and fragment pipeline bound by name.
  - Draws pull their data from StructuredBuffers, with no vertex buffers or
    input layouts, as the point rasteriser already did.
  - A draw that binds something of its own gets a fresh root object. Draws
    that bind nothing share the pass's (`RasterPass::bind`), added in M2.
  - Checked: two triangles over the left half of clip space cover exactly
    w/2 × h pixels.
- **`RayTracingKernel`.** A pipeline plus its shader table, for OptiX and
  Vulkan RT.
  - On Metal, slang-rhi has no pipelines, only inline RayQuery in compute, so
    this reports `Unsupported` and the engine traces with ComputeKernels there.
- **`ShaderLibrary` extensions.**
  - **Link-time constants:** a module declares
    `extern static const uint kName;` and is linked against a generated
    exports module. Each distinct set of values gets its own program.
  - **Generated modules:** `loadSource` compiles modules that exist only as
    source (materials). Loading an existing name with different source is
    refused.
- **Persistent shader cache (`DiskShaderCache`).**
  - One file per compiled program, holding its key and its data, written to a
    temporary name and renamed into place.
  - Location: `$ATHENEA_SHADER_CACHE`, or `athenea/shaders` under the
    platform's cache directory.
  - The key comes from slang-rhi and includes the linked program's hash, so
    edited shaders miss the cache.
  - The whole suite dropped from 89 s to 26 s on a warm cache.
- **Image comparisons (`render/ReferenceRenderer`).**
  - **`compareHdr`:** per-pixel relative difference in a logarithmic histogram
    (eight bins per octave), and relMSE Kahan-summed per row. It is for
    radiance above 1 and dark noise, where 8-bit code values say nothing.
    Checked: a against 1.1·a gives p99 0.0964 against an exact 1/11.
  - **`countDifferent`:** counts the differing entries of two uint buffers,
    per chunk, then reduces the counts. It is meant for ID AOVs.

## Complete USD: geometry and visibility (M2)

`UsdGeomMesh`, PointInstancer and native instancing, drawn through Hydra
with ids, depth, normals and primvars as render outputs. There are three
routes to visibility, and all of them agree with Storm.

### On the device, in Hydra's order

- **`geom::MeshBuilder`.**
  - Hydra's topology is triangulated in `HdMeshUtil`'s fan order, with holes
    and left-handed orientation.
  - Smooth normals use `Hd_SmoothNormals`' formula: cross products per
    corner, scattered to points through a radix sort.
  - Primvars of every interpolation are expanded per triangle corner.
    Indexed primvars are resolved on the device, and doubles are decoded
    from their two words.
  - Checked:
    - Areas agree with the faces'.
    - A height field's normals come within 1.1e-7 rad of a brute-force sum.
    - A sphere's normals come within 1.1e-6 rad of radial.
- **`world::GpuScene`.**
  - Every mesh sits in shared pools (positions, indices, corners, faces,
    primvar values) with one 64-byte record each.
  - Every drawn copy has a 176-byte `InstanceRecord`: object to view and its
    normal matrix, object to world, look, ids and the double-sided flag.
  - The pools are repacked only when the mesh set changes, which is what
    `generation()` counts.
- **`world::Instancing`.**
  - Each level is composed as Storm composes it:
    `instancer * T * R * S * instanceTransform`, nested
    `parent[i] * level[j]`, and read from float, half or double primvars.
  - A chain is recomposed only when an instancer in it changes.
  - Instanced sets are pooled on the device, and one dispatch writes every
    set's records: a thread binary-searches its set. This used to be a
    dispatch per set, 32 ms of command recording for Kitchen_set_instanced's
    1462 sets; it is now 0.4 ms.

### Three routes, one visibility buffer

Each route writes (instance + 1, triangle) per pixel. Shading and AOVs
rebuild the hit from those two numbers with Möller–Trumbore in view space
(`technique/surface.slang`), so the routes shade alike.

- **`VisibilityRaster`.**
  - Reversed infinite Z, with depth `near / z` in D32Float.
  - One draw per mesh with `instanceCount`, because Metal has no indirect
    draws.
  - Every draw shares one root object. A draw's mesh and instances arrive
    as its start vertex and start instance.
  - On Metal, `vertex_id` and `instance_id` already include those starts.
    Slang's Vulkan and D3D output subtracts them. `Caps::drawIdsIncludeStart`
    records the difference, and `tests/gpu/test_textures.cpp` measures it.
- **`VisibilityTrace`.**
  - A BLAS per mesh over the pools, rebuilt when they are repacked. A TLAS
    per frame.
  - A kernel writes the instance descriptors from the records, in the
    backend's layout: 64 B generic/D3D12/Vulkan, 80 B OptiX, 68 B Metal.
- **`VisibilityBvh`.**
  - A Karras LBVH per mesh and one over the instances, from the splat ray
    tracer's build kernels.
  - It is for devices without ray tracing hardware.
- **Single-sided meshes keep their front only**, as in Storm.
  - Raster: `SV_IsFrontFace`, flipped when the transform mirrors.
  - Hardware rays: cull flags, with double-sided instances opting out.
  - BVH: `det < 0` in object space.
- **Which route.**
  - `athenea:visibility` (`athenea stage --visibility`) selects `automatic`,
    `raster`, `rays` or `bvh`.
  - Automatic takes rays where the device has ray queries, else raster,
    else the BVH. Rays win on this machine at every size measured (below).
- **Layers.** Meshes and points are composited by view z into the opaque
  layer the splat rasteriser draws over.

### Hydra outputs

- **Render outputs.** primId, instanceId and elementId (Int32, cleared to
  −1), Neye and normal (Float32Vec3), `primvars:NAME`, colour and depth.
- **Conversion.** `usd/aov_convert.slang` converts on the device.
- **Row order.** Buffers are bottom row first, which is Storm's and
  hdEmbree's layout. M0 had flipped them; Storm showed it.

### How it is checked

All comparisons are kernels, and the numbers are from the last run.

- **Analytic square.** 8281 pixels covered, with 0 coverage and 0 colour
  mismatches, and depth exact. Through Hydra: 8464 pixels, all exact.
- **Instancing.**
  - Six instances of one mesh against six meshes: identical colour and
    depth bits.
  - Six nested instances against six authored: relMSE 1.2e-9. The residue
    is half-precision rotations.
  - A PointInstancer against authored transforms: relMSE 3.4e-9.
- **The routes against each other.**
  - On a bumpy grid with twelve squares (single-sided, double-sided,
    mirrored), 0 of 23654 interior pixels differ between raster and either
    ray route.
  - Through Hydra, on a scene with instancing, culling and a mirrored mesh,
    the three routes differ in 15 and 6 id words out of 230400. Those words
    lie along a grazing edge.
- **Culling.** A single-sided square shows 6723 pixels from the front and 0
  from the back, mirrored or not.
- **Layers.** Splats and points behind an opaque wall change 0 pixels. In
  front of it they change 24137, on all three routes.
- **Storm as the oracle** (`athenea_storm_oracle_tests`, in its own process).
  - Setup: Kitchen_set at 480×270, compared on primId segmentation,
    coverage, depth and Neye.

    | Route | Coverage | Segmentation | Depth, worst | Neye > 6/255 |
    |---|---|---|---|---|
    | raster | 0 differ | 18 of 57219 | 1.2e-6 | 17 |
    | rays | 4 differ | 41 of 57236 | 6.3e-6 | 17 |
    | bvh | 4 differ | 36 of 57233 | 6.3e-6 | 17 |

  - **Why segmentation.** Storm numbers prims differently from the engine,
    so ids are compared as a segmentation: a pixel whose 3×3 neighbourhood
    is one prim in one image must be one prim in the other.
  - **Why Neye in bytes.** Storm writes Neye into UNorm8, where negative
    components clamp. Ours is compared in that space.
  - **Storm renders single-sampled.**
    - With multisampling, Metal cannot resolve an R32Sint target. The Metal
      validation layer asserts it, and without the layer the id buffers come
      back with their upper 16 bits unwritten.
    - OpenUSD reads `HDX_MSAA_SAMPLE_COUNT` as its libraries load, so ctest
      sets it in the environment and the test refuses to run without it.

### Measured (M5 Pro, release)

- **Method.** `athenea stage --frames 40 --visibility <route>`, median frame
  after the first.
- **What a frame includes.** Hydra sync, visibility, headlight shading, the
  splat pass (empty here) and reading colour and depth back.
- **Camera.** The oracle's: eye (500, −350, 350), focal 20.

| Scene | Size | raster | rays | bvh |
|---|---|---|---|---|
| Kitchen_set (1788 meshes) | 480×270 | 18.0 ms | 4.8 ms | 8.9 ms |
| Kitchen_set | 1920×1080 | 31.7 ms | 19.2 ms | 29.2 ms |
| Kitchen_set_instanced (1462 sets) | 480×270 | 15.2 ms | 5.4 ms | 9.0 ms |
| Kitchen_set_instanced | 1920×1080 | 29.1 ms | 19.2 ms | 28.6 ms |

- **First frame.** 3.2–4.3 s with raster or rays, 5.5–6.5 s with the BVH.
  That is the stage load, mesh builds and shader compiles on a cold cache.
- **Where raster's time goes.** Recording 1800 draws costs the host about
  8 ms, even with one root object: slang-rhi writes render state and looks
  up binding data per draw.

### Not done, not verified

- **Draw count.** Raster pays per draw, and single-instance meshes could
  share draws.
- **The TLAS is rebuilt every frame**, not refit.
- **Other backends.** Vulkan and D3D12 start-location semantics are
  unverified. So are the OptiX descriptor layout on real hardware and
  visibility on CUDA, which has no raster.
- **Storm oracle coverage.** One camera on one stage. Negative Neye
  components are not compared, because Storm clamps them.
- **Arrives with later milestones.** geomSubsets and materials (M4).
  Deformation and motion, which need BLAS refit (M7). Subdivision, curves
  and implicit surfaces (M8).

## Complete USD: athenea view (M3)

`athenea view stage.usd` is a window onto a stage through the engine's Hydra
delegate.

- **Cameras.** A free camera: orbit with the left button, pan with the
  middle button or shift, dolly with the right button or wheel, and F to
  frame. The stage's own cameras can be picked too.
- **Choices in the panels.**
  - Technique: raster or rt.
  - Mesh visibility route.
  - Output: colour, depth, prim, instance and element ids, Neye, normal.
  - View transform, display, exposure and render scale.
- **Stage.** A tree of the stage, and a click to pick the prim under the
  mouse.

### Frames stay on the device

- **Drawing.** `StageRenderer::draw` executes Hydra and reads nothing back.
- **Hydra buffers.** They are converted only when mapped: the render pass
  leaves each one a fill that runs on its first `Map`. A host that shows an
  output on the device never pays for it on the host. `athenea stage` and tests,
  which map, read what they did before.
- **Display.**
  - `StageRenderer::displaySource` hands the frame's colour or an AOV
    (`Engine::aovView`) to `technique::DisplayTransform`.
  - The transform writes the window's surface texture directly: BGRA8Unorm
    with storage usage, so `framebufferOnly` is off.
  - Any render scale; each output pixel shows the source pixel under it.
- **Panels.**
  - Dear ImGui 1.92.9 (`ImGuiBackendFlags_RendererHasTextures`) draws over
    the display through `view::ImGuiRenderer`, on the same device.
  - Each frame its lists go up as two buffers. A draw pulls vertices through
    32-bit indices from its start vertex and binds its texture and scissor.
  - ImGui tessellates on the CPU. That is the one place this viewer does
    arithmetic on the host, and it is chrome, not scene data.
- **Picking.** `StageRenderer::pick` reads one pixel's two id words and
  resolves the rprim to its USD prim through `HdPrimOriginSchema`.
- **Framing.** `GpuScene::worldBounds` folds every instance's world box on
  the device. Clouds add their decoded boxes through their prims'
  transforms.
- **Window.** GLFW 3.4 with no client API; slang-rhi makes the Metal surface.
  `platform::matchLayerToBacking` sets the layer's contents scale through the
  Objective-C runtime, so drawables map one to one on Retina screens.

### Display transform

- **View transforms.** Standard, and AgX in Wrensch's analytic fit of
  Sobotka's: inset, log2 over [−12.47, 4.03] stops, a sixth-order sigmoid,
  outset, then 2.2.
- **Displays.** sRGB, Rec.709 (BT.1886, a pure 2.4 power) and Display P3
  (P3-D65 primaries with sRGB's transfer).
- **Other outputs.** Depth is a log grey from near to far, ids are hashed
  colours (−1 is the background), and vectors are shown as rgb·½+½.
- **Checked** (`tests/technique/test_display.cpp`).
  - A generated ramp from 2⁻¹⁰ to 2⁶, with hues and partial coverage over a
    background, goes through six view, display and exposure combinations.
  - Each output is compared per pixel with the formulas written again in
    another kernel: the P3 matrix derived from chromaticities, the sigmoid as
    powers, exposure as exp.
  - Worst difference 3.5e-6.

### How it is checked

- **Smoke test** (`athenea_view_tests`). A hidden window draws a square stage
  for four frames. A kernel counts the snapshot's lit pixels (49538 at
  480×320).
  - It skips where GLFW cannot initialise or a window cannot open.
  - A hidden window's drawables come at about 100 ms each; a shown window's
    at the display's rate.
- **Picking and bounds** through `StageRenderer` on the primvars stage:
  - Pixels over each mesh pick `/PerFace` and `/PerCorner`; an empty pixel
    picks nothing.
  - The bounds come to (−2, −1.5, −5)–(2, 1.5, −5), the authored points.
- **Snapshot.** `athenea view --frames N --snapshot out.exr` writes the last
  frame as shown, panels included: a float texture read back for output.
  Looked at for Kitchen_set.

### Measured (M5 Pro, release)

- **Setup.** `athenea view --frames 200` in a 1600×900 window with a free
  camera, raster technique and automatic (ray) visibility.
- **Draw.** Kitchen_set 7.08 ms, Kitchen_set_instanced 7.15 ms (medians).
  That covers Hydra and the engine.
- **Frame.** 10.0 ms for both, which is the display's vsync, not the
  engine.

### Not done

- **Display.** ACES 2.0, OCIO and EDR output (RGBA16Float with extended
  range) are not implemented.
- **Picking under instancing.** It names the prototype's prim and the
  instance number, not the instance proxy's path. The viewport does not
  highlight the selection.
- **Stage tree.** It lists prims and marks native instances. It does not
  walk into instance proxies.
- **Time.** The time slider sets the stage time; animation itself is M7.
- **Platforms.** Linux and Windows windows are untested. X11 is wired
  through GLFW's native handle; Wayland is not.

## Complete USD: textures and materials (M4)

A mesh no longer shows its displayColor: it shows the material bound to it,
compiled from MaterialX into Slang and evaluated on the device.

### Textures

- **Reading.** Hio decodes a file's bytes on the CPU and nothing else: the
  bytes are uploaded raw and a kernel decodes them (v up, since Hydra's rows
  run the other way).
- **Mips** are a kernel, since slang-rhi generates none. An sRGB texture is
  decoded, filtered and encoded again, so a mip's mean is the mean of the
  level above it in light, not in code values. Views are made with the sRGB
  format its samplers want (a slang-rhi patch: a full-range view ignored the
  format it was asked for).
- **UDIM** is an indirection table: a tile that is missing leaves the node's
  default, and the graph says so rather than sampling black.
- **The table.** One `ParameterBlock` of 1024 texture slots and its
  samplers, deduplicated. Metal takes it as an argument buffer; a device
  with bindless will take the same interface.
- **Filtering is per target.** A footprint is sampled with its gradients
  where a compute entry point may ask for them, and otherwise from the level
  the wider side of the footprint lands on -- CUDA has no `SampleGrad` in
  compute. The choice is a `__target_switch` in the shader, not a build
  flag. On Metal, where both exist, they pick the same levels (0, 1, 2, 3
  for footprints of 1, 2, 4 and 8 texels) and the same samples.
- **Colour spaces.** MaterialX `srgb_texture` is sRGB and anything else is
  raw; `UsdUVTexture`'s `sourceColorSpace` is auto, raw or sRGB, auto
  meaning sRGB for 8-bit images.
- **Checked** (`tests/material/test_texture_store.cpp`): a decoded texture
  is the file to the last bit (0 of 3404 components differ), a mip chain's
  1x1 mean is the level 0 mean (0.49616 against 0.49804 raw, 0.30570
  against 0.30499 through sRGB), and UDIM tiles resolve or report missing.

### The lobe library

- **Lobes.** Oren-Nayar and its energy-compensated form (EON), Burley,
  translucent, dielectric (reflection, transmission, both), conductor,
  generalized Schlick with an F82 tint, and sheen in both the Imageworks and
  the Zeltner forms. Each has `eval`, `sample` and `pdf`; microfacets sample
  the visible normal distribution, transmission follows Walter, and sheen's
  albedo comes from an LTC fit.
- **The stack.** A material's lobes are built into a `LobeStack` and
  sampled with one-sample MIS, so a graph of any depth costs one sample.
- **Checked** (`tests/material/test_lobes.cpp`), all on the device: a
  chi-squared of sampled directions against the pdf (393.2 on 399 degrees of
  freedom for the diffuse lobes, 379.0 to 461.0 elsewhere), the pdf's
  integral against the fraction of samples drawn, and a white furnace where
  the albedo sampled and the albedo integrated uniformly agree to 3%.

### MaterialX into Slang

- **The generator** derives from MaterialX's own `SlangShaderGenerator`.
  Every node keeps its genglsl or genslang implementation except the ones
  that cannot mean here what they mean in a rasteriser:
  - the **surface** node, which has no light loop: its BSDF graph runs once,
    pushing lobes, and what it weights them by becomes the material's stack;
  - the **BSDF and EDF** nodes, which push lobes instead of responding to a
    light (`shaders/athenea/material/mx/`, declared in
    `athenea_genslang_closures.mtlx`);
  - the **image** nodes, which sample the texture table;
  - **heighttonormal**, which needs a screen derivative (below).
- **A BSDF value is a weight per built lobe**, not a response: `mix`,
  `layer`, `add` and `multiply` combine those weights the way genglsl
  combines responses, so a value used twice is two weightings of one lobe,
  not two lobes.
- **Uniforms are not baked in.** Every input is read from a float blob, and
  the module is named by a hash of its source: materials that differ only in
  values share one compiled module. `MaterialCompiler::parameters` lays a
  material's values, its textures' ids and its primvars' scene slots into
  that blob.
- **Sizes.** UsdPreviewSurface 580 lines and 23 blob words, standard_surface
  821 and 59, OpenPBR 945 and 55, glTF PBR 575 and 36, an unlit texture
  graph 242 and 22 (one texture, one primvar).
- **Checked against MaterialX itself.** The same graphs compile a second
  time in a reference variant whose closures are MaterialX's own genglsl
  responses; a kernel evaluates both for 65536 light directions. Worst
  component difference 3.2e-5 over eleven graphs, from a single
  `oren_nayar_diffuse_bsdf` to standard_surface with metalness, coat and
  sheen.
  - One difference is deliberate and aligned in the test: genglsl's layering
    scales the base by the top's Fresnel at the half vector, the lobes by
    the Fresnel at the view direction, which is what a sampler can carry.
  - A transmission-only scatter leaves the throughput at 1 in genglsl.

### Shading a frame, and who else evaluates a material

- **One generated module** (`technique::MaterialPrograms`) imports every
  compiled material and dispatches on a material row's function. Shading
  imports it to build a lobe stack; the visibility passes import it to ask
  whether a sample is there at all. It is named after the set it dispatches
  to, so a frame that shows the same materials compiles nothing.
- **The row** is the instance's (`InstanceRecord.flags >> 8`), unless the
  triangle is in a GeomSubset that binds one of its own
  (`triangleSubsets`, `subsetRows`).
- **The light** is still the headlight: a unit light from the eye, as
  `HeadlightShading` drew unshaded meshes. Scene lights are M5.

### Cutouts

MaterialX resolves `opacityThreshold` itself, so a UsdPreviewSurface that
has one leaves opacity at 0 or 1; what is left is deciding who evaluates it.
Shading cannot, because a sample cut away has to let what is behind it
through, so visibility does:

- the rasteriser draws with a generated fragment shader that discards;
- the two ray routes carry the ray on past the sample, up to sixteen times;
- only rows flagged as cutouts pay for the evaluation, and a frame with none
  runs the plain passes.

### Bump

`heighttonormal` -- and so `bump`, which is `heighttonormal` into
`normalmap` -- differences the height in screen space. A material here is
evaluated in a compute kernel, which has no `dFdx` (Slang has no such
identifier at all), so the difference comes from the thread's quad.

- **Which threads a quad holds was measured.** On this device the lanes are
  handed out along the group's rows, so four consecutive lanes were four
  pixels of one row and every vertical derivative was wrong -- all 15939
  threads of a test dispatch.
- **So the kernels walk their pixels in quad order** (`atheneaQuadPixel`): each
  quad of four lanes covers a 2x2 block, and then bit 0 of the lane is x and
  bit 1 is y.
- **The limit is the quad.** Where one of its four threads shades something
  else -- a silhouette -- or leaves early, the derivative is of whatever it
  did evaluate. Measured: one pixel of a square's 360-pixel edge ring.
- `normalmap` needed nothing: the tangent frame (dP/du orthonormalised
  against the normal) was already in `MaterialInputs`.

### In Hydra

- The delegate has a material sprim, and asks for the `mtlx` and the
  universal render contexts.
- A `HdMaterialNetwork2` becomes a MaterialX document through `hdMtlx`,
  after two rewrites: the USD shading nodes are renamed to their nodedefs
  (`UsdPreviewSurface` to `ND_UsdPreviewSurface_surfaceshader`, and so on),
  and `UsdPrimvarReader` nodes become `geompropvalue` (varname to geomprop,
  fallback to default, result to out), whose primvar the generator needs as
  a constant.
- Materials compile when the render thread commits, and the primvars a
  material reads are added to the scene's primvar slots.

### How it is checked

- **Through USD** (`tests/usd/test_usd.cpp`): a MaterialX graph textured by
  an image, a UsdPreviewSurface with a UsdUVTexture read through a
  UsdPrimvarReader, a UsdPreviewSurface without specular, and a GeomSubset
  whose material shades its faces and the mesh's the rest. The analytic
  square (coverage, depth and colour) has 0 mismatches.
- **Raster against rays, materials included**: max 0.
- **Cutouts**: a square cut away shows the square behind it exactly as if it
  were alone (8464 pixels, 0 coverage and 0 colour mismatches) in all three
  routes; the same opacity above its threshold is not cut; and a square half
  cut by a texture's alpha is the same image whichever route drew it (max
  0).
- **Bump**: a height linear in u gives MaterialX's normal,
  (-k * scale / 16, 0, 1) normalised, over all 7921 interior pixels of a
  square; and the quad derivatives of a field linear in the pixel are its
  gradient for every thread of a 161 x 99 dispatch.

### Measured (M5 Pro, release)

- **Method.** `athenea view --frames 200 --size 1600x900`, draw median, as M3
  measured its frames: Hydra sync and drawing, no readback.

| Scene | Route | Draw | With the headlight (M3) |
|---|---|---|---|
| Kitchen_set | rays | 9.27 ms | 7.08 ms |
| Kitchen_set_instanced | rays | 9.36 ms | 7.15 ms |
| Kitchen_set | raster | 21.29 ms | -- |

  Raster's distance from rays is the one M2 measured: recording 1788 draws
  costs the host about 8 ms, which materials do not change.

- **What a material costs.** One UsdPreviewSurface (metallic 0.2, clearcoat
  0.5) over a quad filling 1600x900 draws in 29.9 ms. That is the lobe
  stack, not the textures: four lobes built and evaluated per pixel, in
  registers sized for sixteen.
- **What a cutout costs.** The same quad with an `opacityThreshold` that
  cuts nothing draws in 60.7 ms: the material is evaluated twice, once by
  visibility to decide the sample is there and once by shading. Nothing is
  carried between them.
- **First frame** (`athenea stage`, which compiles when the render thread
  commits): 0.33 s for that one material, and 1.74 s when a cutout pass has
  to be generated as well. Compiling is synchronous, and this is what that
  costs.

### Not done, not verified

- **Storm as an oracle for materials is not possible on this Mac.** Storm's
  own MaterialX shaders fail to compile in this build (undeclared `u_env*`
  in the generated MSL, with a lighting state and a dome light present), so
  the MaterialX TestSuite comparison is written but hidden
  (`[.][usd][gpu][oracle][storm-materialx]`).
- **Compiling is synchronous.** A material compiles when the render thread
  commits it, which stalls the first frame that shows it; the plan's
  placeholder and background compile are not done.
- **Transparency is not blended.** An opacity below 1 without a threshold
  weights the sample's colour but does not let what is behind it through:
  that is the path tracer's, M6.
- **Nodes whose genglsl uses a screen derivative do not compile** unless
  they have a genslang implementation here, which only `heighttonormal` has:
  `aastep` and the hextile nodes would fail on `dFdx`.
- **Displacement and volume terminals** are ignored by the mesh route. A
  conversion to gaussians reads displacement: "Displacement is where a cloud
  is cheaper than a mesh", below.
- **Layering** uses the top's throughput at the view direction; directional
  albedo tables are not computed.
- **The lobe stack is not optimised.** Every material carries sixteen build
  lobes through registers whatever it uses, and a cutout evaluates its
  material a second time rather than keeping what visibility already found.
  Both are measured above and both are worth revisiting once lights (M5)
  settle what shading needs to keep.

## Complete USD: lights (M5)

A mesh is lit by what the stage authored: UsdLux lights reach the engine
through Hydra, and shading samples each one where it stands.

### What a light is

- **A record per light**, in world space and in the units USD authored
  (`modules/light`). What a record becomes is derived in the shader, not on
  the host: exposure, the blackbody of a colour temperature (Krystek's fit of
  the Planckian locus, normalised to luminance 1), and the area a `normalize`
  divides by.
- **Five kinds**: distant with an angular diameter, sphere, disk, rectangle
  and dome. A sphere of radius 0 and a sun of angle 0 are delta lights and
  carry no density.
- **Shaping** is the cone and its softness. IES profiles and cylinder lights
  are not read.

### How one is sampled

- **The cone it subtends** for a distant light and a sphere, uniformly in
  solid angle, which is the density a plane's closed-form irradiance is
  written against.
- **Its own surface** for a disk and a rectangle, uniformly in area, turned
  into a solid-angle density by the distance and the cosine at the light.
- **The surface being shaded** for a dome, cosine weighted: the light comes
  from the hemisphere above the surface, and sampling the whole sphere throws
  half the samples below the horizon -- 7.1% of noise against 0.02% on a
  plane under a constant dome, at the same count.
- **`lightPdf`** gives that density for any direction, not just for the
  sample drawn. The path tracer will weigh hits by it (M6); here it is what
  the chi-square compares against.

### Shadows

Where the device traces rays, a light that casts one is occluded by whatever
lies between the point and the sample. The ray leaves along itself as well as
along the normal, so its origin does not depend on a sign, and by a distance
that grows with the scene -- which costs contact: an occluder within that
offset is not seen. The structure is the scene's own, built for whatever
route drew the frame, and read after the visibility pass rather than before,
since the rays route rebuilds it there.

### The dome

A dome carries a lat-long image through the same texture table the materials
sample, mapped around the light's own axes. It is not a layer: it has no
depth, and giving it one would make the background read as covered, so it is
painted where the frame drew nothing, opaque, after everything else.

A dome follows its image's own brightness. The warp descends the mip chain
the texture store already built, choosing among a cell's children by
luminance, and its density needs no walk at all: a lat-long texel covers
2 pi^2 sin(theta) du dv, and a texel's share is its luminance over the
image's total, which the 1x1 level holds as an average -- so

    pdf = luminance / (average * 2 pi^2 * sin(theta))

Nothing is precomputed, and the choice between warping and sampling around
the surface is made by how much the image varies: the 1x1 level gives the
mean and a middle level the spread. A flat sky is better served by the
cosine, which the numbers below say plainly.

It took two bugs to get there, and the chi-square binned in the image itself
-- where the warp works, so no grid artefact could be blamed -- found both:

- **Splitting left from right and then top from bottom off one level** does
  not give the four children their own probabilities: z 61543 over a million
  samples. An explicit choice among four weights fixed it.
- **Splitting an axis that has no resolution left.** With that fixed a square
  image passed at once (z 0.84) while a 64 x 32 one still read z 1164589: a
  lat-long chain reaches one row while it still has columns, and from there a
  cell has two children rather than four, so probability was being handed to
  texels that are not there. Each level now splits only the axes that still
  divide.

What was never wrong, measured rather than assumed: the chain telescopes to
0.7% (a parent against its four children), uv survives a turn through a
direction exactly (0 of 4032), and every sample agrees with `lightPdf` --
that last one passed throughout, which is the lesson: a per-sample check
compares a density with itself and cannot see a sampler drawing the wrong
distribution.

- **Verified**: z 0.84 with a square image and 2.41 with a 2:1 one, a million
  samples each, against the density integrated over the same bins.
- **Against the cosine**: on a plane under a flat sky the warp is 20% out
  where the cosine is 0.08% at the same count, since a uv-uniform sample
  crowds the poles and drops the cosine. Which is why the rule picks by
  variation, and why a dome with a sun in it is the warp's case, not this
  one.

The three instruments that settled this stay: the chi-square bins a dome in
its image, a cone in its own solid angle and an area light on its own
surface; a round trip checks the mapping without statistics; and a pyramid
check measures whether the chain telescopes at all.

### The cylinder, added with M6

UsdLux's `CylinderLight`: the lateral surface of a cylinder along the prim's
x, of `radius` and `length`, emitting outward, one-sided. Sampled uniformly
on that surface (an angle about the axis, a height along it), with the pdf
of any area light, `d^2 / (cos * area)`; `lightPdf` finds where a direction
meets it by the ray's closest approach to the axis rather than the quadratic's
`b^2 - 4ac`, which cancels catastrophically beside a tangent ray -- measured:
7826 of a million samples disagreeing with their own pdf beyond 1e-3 with the
quadratic, 2378 with the closest-approach form, and what remained was the
conditioning of `1/cos` itself, derived in the check and guarded below a
cosine of 0.017, where the light arriving is of order 0.02% of the whole.
Checked four ways:

- **Chi-square in its own support** (angle by height on the surface, cells
  facing away expecting nothing): z 1.30, the pdf integrating to 0.4512
  against 0.4524 drawn, 0 samples disagreeing with `lightPdf`.
- **The sampler alone**, at three points, against dense quadrature of the
  form-factor integral: 0.04% to 0.15% apart.
- **A Lambert plane under it**, against two closed forms that share nothing
  -- 512 one-sided strips under Lambert's edge formula, and the quadrature --
  which agree with each other to 0.2%: 0 of 2209 pixels beyond 3% at 65536
  light samples. Not 4096 like the flat lights: a one-sided curved emitter
  rejects half its samples and varies over the rest, and the first run read
  3103 of 8281 pixels beyond 2% -- which was noise (sigma ~2% at 4096, 0.9%
  at 32768, 0.4% at 131072), not the bias it looked like, and the sampler
  alone is what said so.
- **Through Hydra**, the same.

Found on the way and fixed: shading's second random number was one LCG step
of the first, tying every sample pair to a lattice. It did not bias the
lights that were checked, but it is the path tracer's PCG chain now.

And one the closing plan found by reading: each light's cumulative share of
the frame's power was accumulated in a host loop -- the one piece of CPU
arithmetic on scene data left in the tree. It is a kernel now
(`light_prefix.slang`), the total is the last record's share where
`chooseLight` reads it, nothing comes back to the host, and the table takes
the shader library to make it. Checked by a kernel that writes the power a
second time from the record alone: five lights of mixed kinds, exposures and
`normalize`, 0 shares missing their power.

### IES profiles, added with M6

UsdLux's `ShapingAPI` IES: `shaping:ies:file`, `angleScale`, `normalize`.
`io::readIes` reads an LM-63 file as authored -- the vertical and horizontal
angle lists and the candela table, the multiplier, the photometric type --
and nothing is normalised, resampled or mirrored on the host: each of those
is arithmetic on the data. The frame's profiles are concatenated into one
values buffer with a record each (`IesRecord`), and the shader samples the
table where it samples the light: the emission direction in the light's own
axes, straight down its -Z at theta 0 as UsdLux orients a profile, bilinear
between the authored nodes, with the horizontal range folded by the symmetry
its last angle declares (one angle: rotational; 90: quadrant; 180: bilateral).
It modulates radiance only, never the density, so every pdf and every
chi-square stands as it was.

- **`angleScale`** as UsdLux defines it: positive divides theta, negative
  scales from 180 degrees, zero is none.
- **`normalize`** divides by the profile's power. UsdLux says the intensity is
  "scaled by the overall power of the IES profile ... integrating the
  luminous intensity over all solid angle patches", which leaves the constant
  open; here the power is that integral over 4 pi, so a normalised profile
  has mean intensity one over the sphere and a uniform profile is unchanged.
  The integral is a statistic, so `ies_prepare` computes it on the device at
  commit, over the patches the angle lists define, folded by the symmetry.

Checked three ways:

- **Every node returns its own candela**: 0 of 37 off, worst 2.2e-5 relative.
- **Between nodes, the closed form**: a profile of 1000 cos^4(theta) authored
  at five-degree nodes, sampled at 4096 directions against the formula. The
  error of piecewise-linear interpolation is bounded by h^2/8 max|f''| --
  0.0873^2 / 8 * 4000 = 3.8 -- and the worst read 3.77.
- **Through Hydra**, a cutoff profile (one to 20 degrees, zero from 25) on a
  small sphere light over the plane: inside the cone the frame is the frame
  without the profile, word for word (0 of 2785 pixels differ), and outside
  it is black (0 of 1488 lit). The band between is the profile's own ramp,
  20 to 25, spread by the sphere's angular radius of 1.43 degrees: a first
  check that skipped three degrees about the cutoff read 1626 lit pixels,
  all within 26.3 degrees and all the ramp's, and was wrong, not the light.

Not done: photometric types B and A are read but sampled as C; `TILT=<file>`
is treated as none; the splat relighting samples a light's centre without
its profile.

### Light instancing, added with M6

A light under an instancer is placed as a mesh under one: the delegate
walks the instancer chain above the sprim (`_UpdateInstancer`, the same
loop `Mesh.cpp` runs), and the engine composes it on the device with
`world::Instancing::compose`, unchanged, once per change of any level. The
light module sits below world, so `light::Light` carries the composed rows
raw -- `instanceRows`, a buffer of 3 float4 rows an instance, and
`instanceCount` -- and `LightTable::set` does bookkeeping only: it copies
the prototype's record once per instance and `light_instances.slang`
rewrites each copy's rows as the instance's rows times the prototype's own,
before `light_prefix` accumulates the power over the whole table. A light
whose instancer has not arrived is not drawn, as a mesh in that state is
not.

Checked twice, both exact. In the technique: a rect light that itself
rotates, under an instancer that rotates and takes four elements out of
order, against the four lights authored at `instancer * element * prototype`
-- `lightInstanceCheck` finds 0 of 6 records differing in rows, size or
cumulative power (a product in the wrong order shows, since both factors
rotate). Through Hydra: a `PointInstancer` whose prototype is a sphere light
at three positions over the plane, against the three lights authored one by
one, relMSE 0 -- Hydra delivers instanced lights in this install, which is
the half light linking is missing.

### In Hydra

The delegate takes sphere, disk, rect, distant, dome and cylinder lights as
sprims, and reads a light's IES profile where `ShapingAPI` authors one. A
light's samples per pixel are a render setting, `athenea:lightSamples`, reachable
from `StageRenderer` and from `athenea view --light-samples`: one is what an
interactive frame takes, and a comparison against a closed form asks for
enough that what is left is the light and not the noise.

### How it is checked

- **Closed forms that share no code with the renderer**
  (`lambert_irradiance.slang`): a Lambert plane under a sphere, a disk (as a
  512-gon), a rectangle (Lambert's formula over its edges), a sun and a dome,
  and a point light behind a square occluder whose umbra is exactly what it
  projects. 0 of 8281 pixels beyond 2% in each: worst 0.26% for the sphere,
  1.6% disk, 1.3% rect, 0.02% dome, and exact for the sun and the umbra.
- **A chi-square per light**, a million samples each, binned in the frame
  that matches the light's support -- a cone's own solid angle, an area
  light's own surface -- against `lightPdf` integrated over each bin: sphere
  z -0.23, disk 0.14, rect -0.27, sun -0.29, dome -1.61, every pdf
  integrating to 1.0000.
  - It carries a per-sample pass too: the density a sample reports against
    the density its own direction has. That is what MIS depends on, it needs
    no histogram, and it is what proved the large statistics were the binning
    rather than the sampling (worst disagreement 3e-7).
- **Through Hydra**: a UsdLuxSphereLight over a Lambert plane, 0 of 8281
  pixels beyond 2% with its centre at 0.03200 against the closed form's
  0.03200; and a dome light's image lighting the same plane to 0.08%, its
  background reading 0.6039 against the 0.6038 its PNG decodes to from sRGB.
  Each of those renders the same stage with no light first and checks it
  against the analytic headlight, exact to 1e-5, so the lit comparison is
  about the light and not the material.

### Bugs these found

- **The shading normal never faced the viewer.** `materialInputsAt` computed
  the backface and did not flip, so a front-facing square handed materials a
  normal pointing away. Shading hid it, since the lobes build their frame
  around the view direction; a shadow ray could not, and every ray hit the
  surface it left.
- **Shading traced against a freed structure.** It took the acceleration
  structure's pointer while preparing the frame, and in the rays route the
  visibility pass rebuilds it there -- releasing the one shading still
  pointed at. Two readings were wrong before that one, and measurement killed
  both.

### Measured (M5 Pro, release)

- **Method.** `athenea view --frames 200 --size 1600x900`, draw median.
- **Scene.** Kitchen_set with four lights (a dome, a rectangle and two
  spheres), authored beside it: the asset itself carries no UsdLux prim.

| What | Draw |
|---|---|
| Four lights, one sample, rays | 27.77 ms |
| The same, compute BVH | 35.14 ms |
| The same, raster | 40.94 ms |
| The same without shadows, rays | 26.69 ms |
| No lights at all (the headlight), rays | 10.61 ms |

- **Samples per light**, rays: 1 gives 27.79 ms, 4 gives 77.72 ms, 16 gives
  275.40 ms. Linear in samples times lights, since every light is sampled at
  every pixel: what a light BVH and MIS are for.

### Not done, not verified

- **A light can be chosen instead of visited.** Shading either loops over
  every light at every pixel -- exact, and the default, because at one sample
  it is the quieter of the two -- or draws one light a sample in proportion to
  its power (`athenea:chooseLights`, `athenea view --choose-lights`), dividing the
  density of that choice back out. The choice is what stops a pixel's cost
  growing with the number of lights; what it costs is noise a frame has to
  average away. There is still no light BVH, which is what the choice would
  need to stay cheap at thousands of lights.
  - **Checked by three sphere lights in the same place**, of intensity 1, 2
    and 3: one light of six times the power, analytically, with a
    distribution over them that is not uniform -- which a single light can
    never exercise. Both ways, 0 of 8281 pixels beyond 3%, worst 0.21% for
    the loop and 0.26% for the choice.
  - **Measured** on Kitchen_set with four lights, 1600x900, draw medians: the
    loop takes 28.49, 79.61 and 282.21 ms at 1, 4 and 16 samples per light;
    the choice takes 11.87, 12.18 and 13.22 ms, over a 10.61 ms frame with no
    lights at all. Flat, because the cost of a sample is small beside the
    frame it sits in -- which is also why the loop is affordable at one
    sample and the default.
- **MIS is the path tracer's** (M6, below): the raster's shading samples the
  lights alone.
- **The two dome densities are not combined.** A dome is sampled either by
  its image or around the surface, whichever its variation calls for, and
  never both with MIS weighing between them: that is the path tracer's, M6.
- **Light linking works in the engine and not through USD.** The engine's
  half is exact: an instance carries a 64-bit mask of its categories (the
  record grew to 192 bytes, and a set record spends two spare words to carry
  the same mask through the instances the device writes), a light carries the
  category it lights, and shading skips a light the surface does not carry --
  checked by counters rather than a tolerance, two squares of different
  categories with 6150 pixels each, the linked one wholly lit and the other
  exactly zero, both lit when the light has no collection.
  - **What does not arrive is the scene index's half**, measured in this
    order rather than guessed: `HdsiLightLinkingSceneIndex` is registered
    from a point every host reaches -- a registry function alone never runs,
    since a host that builds the delegate itself never goes through plug's
    discovery -- and it is appended to the chain (traced); it is given ten
    light types and five geometry types, so its defaults are not the
    obstacle; the stage's collection transports correctly, but only in
    *expression mode* (`membershipExpression='/Left'` reaches the light's
    collections data source, where relationship mode sends UsdLux's default
    `~//*.*`); and the mesh carries a `categories` data source while the
    light carries `lightLink` -- both empty, before the stage is synced and
    after, with the filter inserted first in the chain and last. Whatever
    makes that filter mark a prim is not happening here, and its
    implementation is headers only in this install. The USD case is written
    and hidden (`[.][usd][gpu][mesh][lights][linking]`) with that list in it.
  - **Found at M9, with OpenUSD's sources on the machine.** The filter
    builds its collection cache in `_PrimsAdded`, from the added-prim
    notices that pass through it, and nowhere else. `StageRenderer` handed
    the stage to `UsdImagingCreateSceneIndices`, which populates on the spot,
    before this renderer's filters were appended: a filtering scene index
    made after its input populated never hears of the prims already there,
    and the render index then read them through `GetPrim` from an empty
    cache. Nothing on the list above could have seen it, because every
    data source was right. The chain is now built empty, inserted, and given
    the stage after. The USD case is no longer hidden: the square in the
    light's collection lit over its 6150 pixels, the other drawn and 0 lit.
- **Shadow linking is honoured in the trace.** A light with a shadow link
  walks its ray on past whatever does not carry that category, as a cutout
  walks past what its opacity removed, up to sixteen times; a light without
  one keeps the cheap first-hit query. Checked both ways against the closed
  form: with the link naming the occluder's category the umbra is exactly
  where it projects, and with it naming another the plane is lit as if
  nothing were there -- 0 pixels of 7440 away from the closed form either
  way. From USD a shadow link comes through the same filter as a light
  link, which was missing for the reason found at M9 (above). Checked from
  USD too ("a UsdLux light's shadowLink collection decides what casts its
  shadow"): a sphere light and an occluder, both above the frame, over a
  floor. With the collection left whole the occluder's shadow changes 9108
  words of 24576 under raster and 12396 under rt (no bounces); with
  `collection:shadowLink` naming `/Floor` alone the frame is bit for bit
  the frame without the occluder, 0 words, under both.
- **Light instancing arrived with M6**, below. (So did the cylinder and IES
  profiles.)
- **Splats are relit where their prim asks**, and baked everywhere else.
  `AtheneaSplatLightingAPI` (`primvars:athenea:splat:relight`, a constant primvar, so
  it is inherited) turns a cloud over to the scene's lights: the albedo is the
  harmonics' constant term, the normal is the splat's shortest axis turned
  towards the eye, and light linking reaches a cloud by the same bit it
  reaches a mesh.
  - **What it is not**: one sample at each light's centre, no shadow ray, no
    second sample, and a normal a splat never had. It is for a capture that
    has to sit under different light, and wrong wherever the capture's own
    light was the point -- which is why baked is the default.
  - **What made it possible**: the light module is two, a core that reads no
    texture and `lights_image` on top. A dome's image comes through the
    material texture table, and the splat projection lives below material in
    the module order, so before the split the lights were simply out of its
    reach.
  - **Checked** three ways at once: with relighting off the frame is
    identical to the one that never had the feature (max 0), with it on the
    picture changes (max 185 over 9216 pixels), and a light whose collection
    does not include the cloud lights none of it (max 18 against the relit
    frame).
  - **Not measured.** What relighting costs against showing what was baked has
    no number here: it wants a stage with both a cloud and lights, and there
    is no splat asset on this machine to build one from -- the clouds the tests
    use are synthesised in memory and never reach the command line.
- **A relit splat casts a shadow ray now** (`athenea:splatShadows`, off by
  default): one ray a splat against the cloud's own proxies, through
  `rt_shadow.slang`'s product of `1 - alpha`. The rasteriser's projection
  kernel traces it, and the proxies come from a tracer of the engine's own on
  the hardware route (`GaussianRayTracer::prepare`), since the frame's tracer
  may be on the compute route, which has no structure an inline ray can walk.
  - **The ray starts past the splat's own neighbourhood**, `shadowOffset`
    sigmas of the splat itself (3 by default). A captured surface is a crowd
    of overlapping Gaussians and a ray that starts at one peaks inside its
    neighbours within a fraction of their size: without the bias a relit
    capture renders black, every splat shadowed by the splats it is made of.
    Measured, and black is what it looked like.
  - **A frame of splats alone had no lights at all.** The light table was
    built only where there was a mesh layer, so a stage of a relit capture
    under a light showed what it was baked with -- the feature had only ever
    been exercised through the render API, never through USD. A frame with a
    relit cloud and no mesh now builds the table (and takes the cloud's own
    bounds as the scene's reach), and nothing else of what a mesh layer needs.
  - **What it does not fix** is what relighting approximates: the normal is
    still the splat's shortest axis and the albedo the harmonics' constant
    term, so a photogrammetric capture relights streaky whatever the shadow
    does. The train at 960x540 shows the key light's shadow across the whole
    body, and the same streaks as before under it.
- **Contact shadows** closer than the ray's offset are missed, and a cutout
  material still stops a shadow ray where its opacity would have let it
  through.

## Complete USD: the path tracer (M6)

Paths over the same visibility buffer the raster shading reads, so the two
can be told apart by exactly one thing: the bounces -- and, with a lens, by
the tracer's own primary rays. The milestone's five checks are each in this
section: the white furnace (the closed emissive shell, exact to 1.7e-7 over
0 to 6 bounces), the path tracer against the raster shading where the bounce
contributes nothing, the error falling as one over root N (exponent -0.499),
splats alone under `rt` through the ray tracer, and OIDN lowering the error
(3.65e-4 to 7.17e-5). What was folded into it from M5's deferrals -- the
cylinder, IES profiles, light instancing, the power prefix on the device --
is in the lights section.

### What it does

- **The same surface, the same material, the same light.** A hit is rebuilt by
  `material_surface.slang`, its material evaluated into the same lobe stack,
  and its direct light gathered by next event estimation with the light chosen
  by power -- all of it the machinery M5 left behind.
- **MIS, since the usd-wg end to end** (the section after M6's "not done").
  At first next event estimation covered the analytic lights and sampling
  the material covered emissive geometry, disjoint sets with no weight; a
  one-sided power heuristic had been wrong -- see below.
- **The bounce** samples the material (`stackSample`), traces where it points,
  shades what it lands on, and carries that surface's emission and direct
  light back through the path's throughput.
- **Accumulation** is a running mean of each sample's colour times its
  opacity: a call adds its samples to a sum and says how many the frame holds,
  which is what a progressive render needs.
- **The sampler** folds pixel, stream, absolute path index, bounce and
  dimension through a PCG hash one after another. The path index is absolute
  (`accumulated + sample`), so a frame gathered in one pass and in many draws
  the same samples: 256 paths in one pass against 64 in each of four differ by
  relMSE 3.9e-15, accumulation order alone.
- **Where the device does not trace**, there is no bounce to trace: the kernel
  is generated without one and gathers direct light alone.

### Where it runs from

The `rt` technique used to trace splats and return, which left every mesh and
every material out of a traced frame. Now it returns early only where there is
nothing to compose under: a frame of splats alone is still `GaussianRayTracer`
writing the whole image, at the tolerances `test_ray_tracing` already held it
to. With meshes in the frame the surfaces are path traced and the splats
composed over them by the rasteriser, because the tracer takes no `under`
layer -- splats inside the rays is still to be written.

A path traced surface gets an acceleration structure whatever the lights do,
since its bounce is a ray. Shading needs one only where a light casts a shadow,
and the same structure serves both; without that, a traced frame would trace
against nothing and no test would say so.

`athenea:pathSamples` is how many paths a pixel a pass gathers and
`athenea:pathBounces` how many bounces each takes after the first hit, one of each
by default -- what an interactive frame affords.

### Gathering a frame over several passes

`athenea:pathTotal` is the paths a pixel at which a frame is finished; one, the
default, never accumulates, so nothing that worked before behaves differently.
Above one, drawing the same frame again adds its paths to the running mean and
the pass reports itself unconverged until the total is reached, which is what
makes a viewport quieten down while it is left alone.

What counts as "the same frame" is the engine's to decide, since it is the only
place that sees both the camera and the scene: it remembers the camera element
by element (`Mat4` has no comparison of its own), the frame's size, the samples
and bounces, and a revision. The revision is what says the scene itself moved
-- `commit` raises it whenever it uploads anything, and so does every setting
that changes what a path would find, each of those only when the value really
changes, so a host that re-sends the same settings every frame does not reset
the mean. Moving the finish line is the exception: `athenea:pathTotal` leaves what
has been gathered still valid.

The render buffer reports convergence from the engine too. It used to answer
"always converged", which was true while there was no progressive mode and
would now let a host stop asking for the rest of a frame the pass had not
finished.

### How it is checked

- **A white furnace** under an imageless dome, path traced against unweighted
  NEE: p99 relative 0.0000, max 0.0003 at 4096 light samples against 4096
  paths. The one the MIS weight failed.
- **A closed emissive shell** reads its geometric series exactly: 0 of 3072
  pixels beyond 1e-4 at 0, 1, 2, 3 and 6 bounces, worst 1.7e-7. The plan's
  first check, and the only one on more than one bounce.
- **One bounce against the raster's direct light**, in a scene with nothing
  for a bounce to find: p99 1 and max 1, with no pixel beyond 2, over 4096
  accumulated paths. That is the plan's check, and it holds the two
  estimators to each other rather than to a tolerance of their own.
- **The error falls as 1/sqrt(N)**, measured without a reference: pairs of
  independent estimates at 64 to 1024 paths, six pairs a point, the median
  of their mean squared difference, and a least-squares exponent on the
  square root: -0.499, window -0.42 to -0.58 calibrated as above. The
  plan asked for a 64k spp reference; measured, that is ~80 s for one scene
  at the other cases' resolution, and depth of reference is not what verifies
  a law -- the ratio between points is, and a reference only adds a floor.
- **The sampler**, three ways: the same 256 paths in one, four and sixteen
  passes agree to 3.9e-15; no two of 3072 pixels share a sample sequence or a
  sample set, at the seeds the ladder used; and block-averaged errors fall as
  independent errors do (4.42x and 15.94x for 2x2 and 4x4).
- **A frame of splats alone through `rt` is `GaussianRayTracer`'s**, the
  plan's fourth check: the Hydra case that renders a cloud through the
  delegate's `athenea:technique` against the ray tracer called directly has held
  it at p99 1 and max 2 since M0, and the engine's early return is what keeps
  it true.
- **The bounce carries light from a second surface.** The check above proves
  the bounce takes nothing away where there is nothing to find -- which is
  also exactly what an unbound acceleration structure would look like, and
  that test binds none. So a wall stands along a plane's edge, turned to face
  it, and the same frame is held at nought bounces against itself at one, over
  the same seeds: the direct term is identical, so what is left between them is
  the bounce alone. p99 41 and max 73 over 4688 pixels. The control is the
  scene without the wall, where the two come out at max 0 -- identical frames,
  which is what makes the difference the bounce and not the noise. (It read
  p99 41, max 73 while the bounce carried rho pi / cos; p99 10, max 13 now.)
- **The traced technique over a mesh, through Hydra.** A plane under one sphere
  light has nothing for a bounce to find, so the traced frame and the raster
  frame are two estimators of the same direct light: p99 1 and max 1 at 1024
  paths, with the surface drawn exactly where the surface is (8281 pixels
  covered, no coverage mismatch). Only the coverage is read from
  `squareMismatches` there: it compares colour against albedo times the cosine
  to the eye, which is the headlight's answer and not a lit scene's -- reading
  its colour count as a verdict on a light would have been reading the wrong
  oracle, and the sphere light's closed form is what the M5 case checks with
  `athenea/test/lambert_irradiance`.
- **The accumulation, over the whole chain**: settings, render pass, mean.
  Eight passes of four paths hold 4, 8, 12, 16, 20, 24, 28 and 32 and then
  report converged; a camera somewhere else drops back to 4 and unconverged,
  and changing the bounce count cuts 16 back to 4. That second half is what
  says the revision is armed rather than decorative: a revision nothing raised
  would go on averaging over a scene that had changed, and no image would look
  wrong enough to say so.

### What was wrong, and what was not

Two defects, found by reading before any new test was run, and then measured.

- **The MIS weight was one-sided and lost half the dome.** `gatherLight`
  weighed every non-delta light sample by the power heuristic against the
  material's pdf -- but the strategy it was sharing with never covered an
  analytic light: sampling the material collects emission from geometry, a
  table light has none, and a bounce ray that escapes broke without gathering
  the dome it passed through. The estimator was scaled down and nothing paid
  the remainder. For an imageless dome and a Lambert lobe the two densities
  are the same function (`cos/pi`), so the weight was exactly one half.
  Predicted and then measured: a white furnace, path traced against
  MaterialShading's unweighted NEE, read p99 relative **0.5453** before and
  **0.0000** (relMSE 1.3e-11, max 0.0003) after. Every earlier path check had
  passed because it used a sphere of radius 0.4 at distance 2, where the
  weight is 0.9984 -- invisible at p99 1 in eight bits. That is the "two
  estimators wrong in the same way" the bounce test warns about, met in the
  flesh.
- **The accumulation was a product of means.** `mean(colour) * mean(alpha)`
  rather than `mean(colour * alpha)`: identical while every sample is opaque,
  which was every test, and biased the moment opacity varied.

**Two more in the bounce, found by the first check that ever exercised more
than one.** A closed emissive shell -- every point emits E and reflects rho,
seen from inside -- must read E (1 + rho + ... + rho^N) after N bounces, and
with cosine sampling of a Lambert lobe each bounce's weight over pdf is rho
with no variance at all, so the check is exact, not statistical. It read 5 pi
times the series at one bounce. Two causes: `throughput *= weight / pdf`,
where `LobeSample.weight` is by its own contract already `f |cos| / pdf`, so
the bounce carried rho pi / cos instead of rho; and `shadeHit` rebuilt the
bounce's hit by calling `shadeAt`, which re-intersects the *camera's* ray with
the triangle the bounce found -- a point not on the bounce ray at all, with
barycentrics and a normal to match. The hit is now rebuilt from the ray
query's committed barycentrics (`surfaceFromWeights`, which `surfaceAt` now
shares), with the bounce's own direction deciding which side it arrived at.
Shell: 0 of 3072 pixels beyond 1e-4 at 0, 1, 2, 3 and 6 bounces, worst 1.7e-7.
A box was tried first and leaks at its edges -- the tracer's origin offset
`p + (n + wi) * 1e-3 scale` puts a ray leaving a face beside an edge outside
the box, where the next face is a back face and culled: one sample in sixteen
short in 25 pixels, worst exactly rho^2/(1 + rho + rho^2)/16 -- which is the
test's geometry, not the integrator's, and the reason the shell is a sphere.

**And the 1/sqrt(N) anomaly, which those two explain.** The ladder had read
exponents of -0.43 to -0.45 through five different instruments, and one point
had spread 3.4x between draws. Fourteen hypotheses were killed by measurement
first, in this order: the running mean (read: algebraically right); the hash
(read: a full avalanche); overlapping seeds (computed: disjoint); the split
into passes (the invariant `1x256 = 4x64 = 16x16` holds to 3.9e-15); the MIS
weight and the premultiply (fixed: the ladder did not move); pixels sharing a
sample *sequence* (fingerprints sorted: 0 of 3072); pixels sharing a sample
*set* in another order (a commutative fingerprint: 0 of 3072); neighbours'
errors correlated (block variance fell 4.42x at 2x2 and 15.94x at 4x4, against
4 and 16); the sampler (replaced by the PCG chain: -0.435 before and after);
the metric (relMSE's `1/(b^2 + 1e-2)` weight has a heavy tail: a plain mean
square instead); the reference (dropped: two independent estimates at the same
N have `E[(a - b)^2] = 2 Var`, no floor by construction); the mean over pairs
(the median); and the box's own seams. Each of those was worth doing and none
was the cause. The cause was the double division: rho pi / cos has a finite
mean under cosine sampling but an infinite second moment, so the bounce's
contribution had infinite variance and a mean of N of them does not tighten as
1/sqrt(N). The ladder was right to complain and the shell found why. With the
bounce fixed the same ladder fits **-0.499**. The 3.4x spread was four draws
of a heavy-tailed statistic read as a switch; the window is calibrated to the
measured scatter and still rejects no convergence, a floor and a linear law.

### The denoiser, and what it took to hand OIDN a buffer

`technique::Denoiser` denoises now. OIDN's `RT` filter runs on the engine's
own Metal queue, over the engine's own buffers, and nothing crosses to the
host. What that took, in the order it was found:

- **OIDN shares only Metal buffers with hazard tracking**, and slang-rhi makes
  none: its Metal backend forbids `MTLHazardTrackingModeTracked` on every
  resource and orders its own work. So each image goes through a staging
  buffer the platform makes tracked -- `platform::newTrackedMetalBuffer`, one
  Objective-C call in `core/Platform`, beside `matchLayerToBacking`, which is
  the precedent and the rule -- wrapped for slang-rhi with `Buffer::wrap` and
  shared with OIDN once.
- **slang-rhi's Metal `copyBuffer` does nothing with a wrapped buffer on
  either side.** Silently: 4095 of 4096 words untouched both ways, while a
  kernel reads and writes the same buffer exactly (0 of 4096). Measured by a
  test that stays in the tree (`a tracked Metal buffer is read and written by
  kernels, and not by slang-rhi's blit`). The staging copies are therefore a
  kernel, `buffer_copy.slang`, word by word.
- **OIDN writes three of a pixel's four floats.** The alpha in a fresh private
  buffer is whatever was there, and `compareHdr` compares four channels: the
  first working run read relMSE 0.15 against a reference at 3.7e-4 for the
  noisy input, which is what garbage alpha looks like. The output staging is
  seeded from the input, so the alpha that comes back is the input's.
- On CUDA the engine's buffers are shared directly (`oidnNewSharedBuffer` on
  the pointer); no staging, no copies.
- On Vulkan OIDN has no device of its own, so it runs on the same GPU through
  CUDA and imports the staging buffers' memory: they are created
  `BufferUsage::Shared`, slang-rhi exports an opaque file descriptor
  (`getSharedHandle`), and `oidnNewSharedBufferFromFD` takes a duplicate of
  it -- OIDN owns the descriptor it is given, slang-rhi keeps its own. The
  copies in and out are Metal's kernels. `create` refuses where the CUDA
  device reports no `OPAQUE_FD` in `externalMemoryTypes` rather than
  producing a wrong image.

**Checked**: sixteen paths against a 4096-path reference, relMSE 3.65e-4
noisy, **7.17e-5** denoised with the first hit's albedo and normal, 6.27e-5
without. The plan's fifth check. On a flat Lambert plane with one wall the
unguided filter does a little better; that is printed, not asserted, since
nothing says the guides must win on such a scene. The test skips where OIDN is
not built or the device will not open it -- never a CPU fallback.

From the engine, `athenea:denoise` runs it over a path traced frame once the
frame has gathered `athenea:pathTotal` -- every frame when the total is one -- in
place over the mean, after the frame's batch and never inside it, since OIDN
submits work of its own and waits. Checked through Hydra by the gate: with the
setting and the total reached, 8748 of 27648 words of the frame change; with
the setting and the total not reached, none.

Not done here: un-premultiplying the colour before the filter and
re-premultiplying after (the scene's opacity is 1 everywhere a test looks).

### Adaptive sampling

A pixel keeps its luminance's second moment beside its sum, and stops taking
paths once the relative standard error of its mean -- `sqrt((E[l^2] - E[l]^2)
/ N) / mean` -- falls below a target after at least `minSamples` paths. The
decision is a kernel of its own after each pass (`pathDecide`), one thread a
pixel, which also counts the covered and the stopped pixels by atomics; the
trace kernel skips a stopped pixel. The frame is gathered when every covered
pixel has stopped or `athenea:pathTotal` is reached, whichever first
(`athenea:pathAdaptive`, `athenea:pathError`).

What matters is not that it stops but that the estimate is truthful, and the
check was built to separate two questions: whether the *means* are right, and
whether each pixel's *own error estimate* is. Against a 4096-path reference,
with the reference's own moments kept for the true per-sample spread:

- **The means are right.** 10 of 4212 stopped pixels beyond three true
  standard errors at a minimum of 16 paths (0.24%, worst 4.0 sigma) and 15 at
  a minimum of 64 (0.36%, worst 4.5) -- three sigma leaves 0.27% by chance.
- **A pixel's own estimate is optimistic where it has not yet seen what is
  rare.** Under a bright bounce that a pixel meets in one path in a hundred,
  its first N paths may all miss it, and the spread they show is the direct
  light's alone, a hundred times too small. Measured before any remedy: 526 of
  4212 stopped pixels beyond three of their own sigma, worst 133, every one
  of them below the reference. The remedy is the standard one: the variance a
  pixel stops on is the larger of its own and the mean of its 3x3
  neighbours', since a neighbour that did see the event stands in. After it:
  100 beyond three of their own sigma (2.4%), and 14 (0.3%) with an estimate
  more than threefold optimistic against the truth. Raising the minimum to 64
  does not move that much (96 and 8): the residual is pixels whose whole
  neighbourhood missed the event, and no per-pixel statistic can see it. That
  is the method's known weakness, and the test bounds it at what it measures.
- Second moments never fall below the mean squared (0 of 4212), and the
  1/sqrt(N) ladder pins `adaptive = false`, since a sampler built to beat the
  law would break its window from the other side.

Through Hydra, an image with `athenea:pathAdaptive` at 10% and a total of 100000
gathers 16 paths a pixel and reports itself converged.

### Three things the ground did not turn out to be

Measured while surveying, and worth writing down because each one changes what
the rest of M6 has to build:

- **`technique::Denoiser` is a presence check, not a denoiser.** It has
  `create` and `description` and nothing else: OIDN is available, not applied.
  Denoising is to be written, not wired.
- **`ReferenceRenderer` is a reference for splats**, projecting and blending
  clouds and points. The plan's "error against a 64k spp GPU reference falls
  as 1/sqrt(N)" cannot lean on it: the path tracer will have to accumulate its
  own reference.
- **`rt_integrate.slang` is a splat integrator**, with an ordered record per
  ray and overlap windows. It is what "splats in rays" will reuse, and it is
  not a skeleton for a surface path tracer.

### The camera's lens

Exposure, the diaphragm and radial distortion, all from `UsdGeomCamera`
through `HdCamera`. Exposure scales the composed frame by `2^stops` once,
after the domes -- everything the camera sees, and no AOV -- checked exact
through Hydra (the frame with exposure 1 authored is the plain frame doubled
on the device, 0 of 27648 words apart).

The other two are not a parameter away: a ray through the aperture, or a
distorted one, no longer passes through the pixel's centre, so it cannot
ride on the visibility buffer the path tracer shades from. When the
projection carries a lens radius (`focalLength / (2 fStop)`, focal in
`HdCamera`'s scene units) or a `k1`/`k2`, the tracer casts its own primary
ray a sample (`shadeLensSample`): the pixel's ray, its x/y scaled by
`1 + k1 r^2 + k2 r^4` in ndc radius, then bent by a thin lens -- every ray
through the pixel meets the pixel's ray at the depth in focus, and leaves
the lens from a point drawn uniformly on its disc -- traced with the same
query the bounces use and shaded by `shadeHit` from where it met the
triangle. A lens ray that finds nothing is a transparent sample, counted.
The aux carry the first sample's hit. The engine's path state carries the
lens, so changing it restarts the accumulation.

Checked against closed forms that share nothing with the tracer
(`dof_check.slang`):

- **In focus is the pinhole.** A uniformly lit Lambert plane at the focus
  distance, its edge off centre, lens radius 0.2: relMSE 0 against the
  pinhole frame, bit for bit -- every lens ray through a pixel meets that
  pixel's ray there.
- **Out of focus is a circular segment.** The plane twice as far: its edge
  is blurred by the lens disc projected, a uniform disc of
  `lensRadius |z - f| / (z f) focal` pixels (5.73 here), so a pixel at signed
  distance d from the edge reads the fraction of that disc on the lit side,
  `(R^2 acos(-d/R) + d sqrt(R^2 - d^2)) / (pi R^2)`. At 4096 lens rays a
  pixel, 0 of 4235 pixels within three radii of the edge beyond 4% of the
  profile (worst 2.2%; the coverage's standard error is at most 0.8%).
- **Distortion moves the edge to the pixel.** `k1 0.5, k2 -0.2`, the edge
  off centre so the radial term shows: in each of 121 rows the lit pixels
  are exactly those whose distorted ray meets the plane on the lit side --
  0 rows off at all, all 121 moved by the distortion.
- **Through Hydra**: `fStop 8, focusDistance 5` on the square at 5 gives the
  pinhole frame exactly; `focusDistance 2.5` and `lensDistortion:k1 0.3`
  each change it (relMSE 0.44 and 6.8). The frame is 160 wide on purpose:
  at 161 the pixel centres lay on the square's triangle seam, and one lens
  ray in four converging exactly on the seam fell through it.

**A bug the Hydra check found, in the sun.** A distant light with an angle
handed its intensity out as the disc's radiance, so the irradiance it laid
was `intensity * solid angle`: UsdLux's default 0.53 degree sun lit a plane
6.7e-5 of its intensity, 15000 times short, while a sun of angle 0 -- the
one the closed forms had been checking -- was right. Now the disc's radiance
is `intensity / solid angle`, and the closed form takes a cap's vector
irradiance, `pi sin^2(a)` along its axis, over that solid angle: the 0.2
radian sun reads 0 of 8281 pixels beyond 2%, worst 0.01%.

### Not done

- There is no `HdRenderThread`: the pass draws on the thread that executes
  it. `StageRenderer::render` does draw until the path traced frame holds its
  total (checked: a total of 32 at 4 a pass leaves 32 gathered), so an image
  from the CLI is a gathered one; a viewport is the host's to keep asking for.
- **Splats inside the path tracer's rays.** A splats-only stage under `rt`
  goes whole to `GaussianRayTracer` (checked, with `test_ray_tracing`'s
  tolerances); with meshes the splats are composited over the path traced
  surfaces by depth. A bounce ray does not see them: `rt_integrate.slang`
  owns its pixel and assumes a primary ray, and a splat's contribution along
  a secondary ray is an integral through its Gaussian that no route here
  evaluates yet.
- **A ray that only needs transmittance** has its own kernel now,
  `shaders/athenea/rt/rt_shadow.slang`: no k-buffer, no segments, no order.
  Transmittance is a product of `1 - alpha` and a product does not care in
  what order its terms arrive, so every proxy the traversal offers is taken
  as it comes and the ray stops once the product falls under its cut. Three
  closed forms hold it: a ray through a particle's centre peaks at power 0,
  so it lets exactly `1 - opacity` through (0.50000 measured against 0.5);
  four such particles give `(1 - opacity)^4` (0.06250); and with the cut
  raised to 0.3 the ray stops after three of the four, which the counter
  says (1 ray cut, 3 particles taken).
  - **Back faces are not culled here.** A shadow ray is born on a surface,
    and a surface inside a cloud is surrounded by proxies: with culling on,
    a ray starting inside a proxy sees only the exit face and misses the
    particle it stands in -- exactly the particles whose shadow touches the
    geometry. Measured both ways in the same binary: born inside with the
    peak ahead, 0.50000 against the closed form; with culling on (the
    control), 1.00000 and 0 particles taken.
  - **Both faces then arrive**, so a ring of the last 16 particles taken
    collapses them: 5 particles taken and 5 duplicates caught in the stack
    test, where `rt_integrate` does it by comparing with the last particle
    blended after sorting.
- **The integrator answers a query, not a pixel.** `rtTraceSegment(ray,
  windowMin, windowMax)` returns a `SegmentResult` -- radiance premultiplied
  by what it covered, the transmittance left, the depth -- and `writeSegment`
  turns one into a pixel, so the camera's kernel and a secondary ray ask the
  same thing. `segmentOver(near, far)` composes two. The windows are half
  open, `[a, b)`, so a peak exactly on a cut is taken once.
  - **What the check found.** Drawing a frame as `[near, s)` over `[s, far)`
    and comparing it with the one query: the far query first traversed from
    `s`, and a proxy entered before the cut that peaks after it was lost
    entirely -- max 199 of 255 on 445 pixels of a sparse cloud. The traversal
    and the window are two different things: a query walks from where the ray
    starts and takes the peaks in its window. After that, a sparse cloud is
    **bit for bit** (max 0) and a dense one differs by 1 code at worst with 0
    pixels over 2 -- the carry (`kCarry`), which the header already names as
    the one place order can be approximate.
- **Points as spheres**, spiked and not built. slang-rhi's Metal backend
  does build acceleration structures over AABBs
  (`AccelerationStructureBuildInputType::ProceduralPrimitives`, a
  `BoundingBoxGeometryDescriptor` each) and OptiX does too; `Spheres` and
  `LinearSweptSpheres` it refuses on Metal. So a sphere primitive is an AABB
  with a custom intersection under `RayQuery`, on both devices. What it needs
  beyond that is a second primitive type in the visibility buffer and in
  `surface.slang` -- the same thing M8's curves add, which is where it goes.
- The lens under `raster`. Depth of field and distortion are the path
  tracer's (above): the raster route shades the visibility buffer's hit,
  which is the pixel centre's, and a viewport under `raster` draws a pinhole
  whatever the camera authors.
- Lens distortion beyond the radial terms: `lensDistortion:center`, `anaSq`,
  `asym` and `scale` are read by `HdCamera` and not applied.
- The plan's per-milestone `athenea bench` condition is retired, in CLAUDE.md as
  well: `athenea bench` times splat files and never rendered a stage, so the
  condition had been unmet since meshes arrived. Medians of `athenea stage
  --frames` and `athenea view --frames` are what is recorded, where there is
  something to compare against.
- **Emissive geometry is a light** since the motion work (below, "Emitting
  triangles as a light").

### Multiple importance sampling

**The missing piece was the other direction.** `lightHit(l, p, wi)` (and
`lightHitImaged`, with a dome's image) says where a ray from p along wi
meets a light and the radiance it carries: a sphere's near root, a disk's
or a rect's plane within the shape, a cylinder's lateral surface, and for a
dome or a distant light's cone an infinite distance only an escaping ray
reaches -- the shaping cone and the IES profile applied as `sampleLight`
applies them. Checked in the chi-square tests beside `lightPdf`: every
sampled direction of the sphere, disk, rect, sun, dome, cylinder and a dome
with an image is found again by `lightHit` at the sampled distance and with
the sampled radiance, 0 mismatches each.

**The weights.** At a vertex the path leaves by sampling its material, next
event estimation's sample of light k is weighed by the power heuristic
against `stackPdf` in its direction, and the material's sampled ray -- after
it is traced -- gathers every light it meets before the surface it found
(or, escaping, the domes and distant lights), each weighed against the
density next event estimation would have drawn that direction with: the
light's choice probability at p (by power, or `lightPdfChoiceAny` under the
light BVH) times `lightPdfImaged`. A delta light and a delta lobe keep a
weight of one, as does the last vertex, which samples no material.
`athenea:pathMis` (default on; `StageRenderer::setPathMis`) switches it off.

**What is left out, and why.** A light that casts no shadow, or whose shadow
links leave occluders out, keeps its weight of one: the material's ray is
stopped by any surface and would see another visibility. So is every light
in a frame with volumes (a medium would have to dim the material's ray as it
dims the shadow ray), and in a frame of more than 64 lights, since each
bounce tests its direction against every light.

**A defect before it passed:** an escaping ray's distance and a dome's
were the same 1e30, and `t >= reached` dropped every dome. Deep MIS then
converged (8192 against 32768 paths, 2e-5 apart) to an image off by 100% at
the 99th percentile -- visible only by comparing it with a deep frame of
light sampling alone.

**Checked** on a floor of metal (roughness 0.2) under each light, one
bounce, 64x48: deep frames of 8192 paths with and without MIS agree within
their noise, and at 32 paths against the deep frame without, MIS is 4.5
times less error under a 4x2 rect, 57.7 times under an imageless dome and
2.4 times under a sphere of radius 0.8. The glossy dome's deep frame without
MIS is heavy tailed (8192 against 32768 paths, 3.4e-2 apart), so its
sameness is shown on a rough diffuse floor under the same dome, where light
sampling converges: deep frames 2.6e-7 apart, their summed noise about
4e-7, and MIS 1.1 times less error there, as expected where the material's
density and the dome's are the same function.
### Emitting triangles as a light

**The table** (`technique::EmissiveTable`). Each material row's emission is
probed once on the device -- the material evaluated at a neutral point, its
emission's luminance -- and every triangle of the frame's records weighed by
its world area times its row's luminance, accumulated in order into a
distribution. It lives in one float buffer with its counts, the rows'
luminances and each record's first triangle, since the path tracer's kernel
had one binding left. The probe shapes where samples go, not what they
carry: a textured emitter whose probe point is dark is simply left to the
material's rays, and nothing is biased by it. Rebuilt when the scene, its
positions, the materials or the engine's revision change; not in a frame
with volumes, whose kernel does not sample it.

**Sampling it without a second material call.** The emission a sample
brings back must be the material's at the point it lands on, and a second
place in the kernel that evaluates materials is what ran the Metal compiler
out (the usd-wg section). So the vertex loop became steps: a step shades
either a vertex of the path or the point next event estimation chose on an
emitting triangle, through the one call; a vertex that chose such a point
waits a step, takes its emission, and goes on. The trip count is a
uniform's, so the loop cannot be unrolled into copies. The choice between
the lights and the emitting triangles is by power (an area light's L A
against a triangle's area times luminance), and each side's density carries
the other's share.

**The weights.** Next event estimation's sample of a triangle is weighed by
the power heuristic against the material's density in its direction; emission
the material's ray meets is weighed against the density next event estimation
gives that point -- the triangle's tabled power over the total, times its
distance squared over the cosine and the area, where the area cancels. Both
sides take that tabled density, so the weights sum to one; the estimate
divides by the density the sample was actually drawn with. Emission is
two-sided in both. With `athenea:pathMis` off, next event estimation leaves the
emitting triangles to the material's rays, as before -- otherwise both would
count them.

**A defect on the way:** the shadow ray to the chosen point was cut a
relative 1e-4 short, but `pathOccluded` moves its origin up to 2e-3 of the
scale along the ray, so it reached the emitting triangle and every sample
was its own shadow: the frame came out black. It now stops 3e-3 of the scale
short.

**Checked** through Hydra: a 1 x 1 quad whose MaterialX `surface_unlit`
emits 3, above the frame, lighting a floor, against a UsdLux rect light of
the same size and radiance in its place (one bounce, 64x48): deep frames of
8192 paths agree to relMSE 4.1e-7, and at 32 paths the error is 8.7e-5 sampled
as a light against 11.75 by the material's rays alone.

**Not done.** Emitting triangles are not sampled from inside media, nor
under motion at their shutter slice (the table is the frame's). A table of
millions of triangles accumulates in float: a triangle whose power is below
the running total's precision is sampled with a rounded probability.

## Complete USD: animation and movement (M7)

### Deformation in place, and refit instead of rebuild

Until this, a mesh whose points changed was a new `GpuMesh`, and a new mesh
in the set meant `GpuScene::repack` -- every pool reallocated and copied,
every bottom-level structure and every LBVH built again -- once a frame for
anything animated. The plan named a latent bug here (a mesh deformed in
place would have left the structures stale, since they key on
`generation()` alone); the tree never deformed in place, so the bug never
fired, and the cost stood in for it.

Now a mesh carries a **topology key** (`MeshInput::topology`, kept by the
caller: the engine gives a prim a new key when Hydra marks its topology
dirty and keeps it otherwise). `GpuScene::update` takes a mesh set in which
every slot holds the same mesh or one of the same key and layout (counts,
subsets, primvar names, interpolations, components and counts) as a
**deformation**: the new positions and primvar values are copied over the
old in the pools, the record's box is rewritten, `positionsRevision()` and
that mesh's `meshRevision(k)` rise, and `generation()` does not. Anything
else repacks as before.

What is built on the pools follows the revision. `RayTracingScene` builds
its bottom levels `AllowUpdate` and keeps one scratch of the largest update
size; a mesh whose revision moved is refit in place
(`AccelerationStructureBuildMode::Update`, source and destination the same
structure), one submit each since they share the scratch. `BvhScene` keeps
each mesh's build parameters, recomputes its leaves' boxes from the pool's
positions and settles the internal boxes over the same tree (`bvh_refit`,
the passes already written for the build): the tree keeps the shape the
old positions gave it, so its boxes get looser and never wrong, until the
next repack reshapes it. Both take a `refit` flag whose false leaves a
deformed mesh's structure as it was -- there for the check below, not for a
caller.

**Checked** by what a deformation changes, in `test_visibility` on the
bumpy grid built three times under one key: after a deformation the scene's
generation stands and its positions revision is 1; rays and the compute
walker see the triangles the rasteriser sees (0 of the interior pixels
differ, the rasteriser reading the pool directly); `bvh_check.slang` finds
0 of 2047 LBVH nodes whose box misses a child's (a leaf's box being the one
its triangle's positions make now). With the refit skipped on purpose the
same comparisons say so -- 5378 and 1918 pixels differ, 1143 nodes miss a
child -- and the next build with the refit allowed catches up to 0 again.
Through Hydra, a sheet with time-sampled points: at the second time Hydra
hands new points and the same topology, the engine keeps the key, the
generation stands and the revision rises (`StageRenderer::meshGeneration`
and `meshPositionsRevision`), 28042 of 30000 pixels change, and the three
routes agree on the deformed frame to the same 4 edge pixels the flat one
allows.

**Not done here**: the top level is still rebuilt every call (it is small,
and instances move every frame); a mesh with changed topology still
repacks every pool, not only its own.

### Motion blur: the shutter in buckets, and time samples from Hydra

**What a bucket is.** Metal has no acceleration structure with motion in
it, and a ray query cannot be handed a time; so the shutter is cut into
`buckets` slices (1 to 8), and each slice gets what the scene looks like
at its centre. One top-level structure holds every slice at once: a moving
instance appears once per slice, its instance mask one bit (`1 << b`) and
its transform interpolated to the slice's time; a still instance appears
once, answering to every bit. A path draws a time per sample, takes the
slice it falls in, and every ray of that path -- primary, shadow, bounce
-- traces with that slice's mask. Eight bits of mask are why eight is the
most. Between the two shutter samples everything is linear: a transform's
rows (exact for a translation, an approximation for a turn) and a point.

**Where it lives.** `MeshInstance::motion` (`MeshMotion`: the transform at
the shutter's open and close, and meshes built from the points there when
it deforms, under the instance's own topology key). `GpuScene::update`
takes the bucket count; when something moves the records get a copy per
slice after the frame's own (`motion.slang`: `motionRecords` writes them
on the device, view and normal matrices included), `tlasFirst`/`tlasCount`
say which records the structure holds, and the first `instanceCount` stay
what the rasteriser and the compute BVH draw -- the frame at the frame's
time, without blur. A deforming mesh's positions are laid out once per
slice in the pool (`pointsStride` apart, `positionsLerp` between its two
meshes), its bottom level is built once per slice over that slice's
positions, and its records carry `pointsOffset` so the surface is rebuilt
from the slice's positions (`InstanceRecord.mask` and `pointsOffset` took
the record's two pads). `RayTracingScene` keeps one handle entry a slice a
mesh, a still mesh's all the same, and `instance_descs` takes the slice
from the record's mask. The path tracer casts its own primary rays under
motion, as it does under a lens.

**Checked against the staircase the buckets make** (`motion_check.slang`):
a uniformly lit plane whose edge slides 57 pixels along x over the
shutter, so a pixel's coverage is the fraction of slices at whose centre
the edge is past it. At 1024 samples a pixel and eight slices, 0 of 7381
pixels beyond 8% of the staircase (worst 5.2%; the coverage's standard
error is at most 1.6%) -- the same with the plane's points sliding under a
still transform, which exercises the per-slice positions and bottom levels;
the same at two slices; and with no motion under eight slices the frame is
the still frame bit for bit.

**Through Hydra.** The camera's `shutter:open`/`shutter:close` reach the
delegate through `HdAtheneaRenderParam` -- set by the pass from the `HdCamera`
it draws, and by `StageRenderer::aim` from the stage ahead of the first
Sync, since Sync runs before the pass and a shutter learnt there is a
frame late; when the pass finds it changed it marks every rprim's
transform and points dirty. With a shutter open for a while, `HdAtheneaMesh`
samples the transform (`SampleTransform`) and the points (`SamplePrimvar`)
about its open and close as well as at the frame, and the engine builds the
shutter's meshes under the same topology key. What Hydra hands back are the
**authored samples that bracket the shutter, at their own times** -- a
stage with samples at frames 0 and 1 drawn at 0.5 under a shutter of a
quarter frame either way returns the samples at -0.5 and +0.5 -- not the
values at the shutter's ends; so each sample's time travels with it
(`MeshMotion::timeStart`/`timeEnd`, `MeshTransforms`, the points' times)
and the device places each bucket's centre between them (`bucketFactor`).
A first version took the two samples for the shutter's ends and blurred
over the whole frame. `athenea:motionBuckets` (default 4;
`StageRenderer::setMotionBuckets`, `athenea stage --motion-buckets`) is the
slice count. Checked with a square sliding between two frames under a
shutter of half a frame about frame 0.5, path traced in eight slices:
against the same stage with the shutter closed, relMSE 1.15 with the
transform sliding and the same 1.15 with the points sliding, the two
stages being the same motion. (A first version read `shutter:open` as a
float and got nothing: the attribute is a double.)

**Velocities.** `HdsiVelocityMotionResolvingSceneIndex` is registered
ahead of the delegate's chain (phase 0, at the start, before light
linking), so a prim that authors `velocities` and `accelerations` has its
points and instance positions sampled at any shutter time from them; the
delegate's sampling above reads the same whether a stage authored samples
or velocities. Checked: a square with one sample of points and a velocity
of two units a frame, against the square with the two samples that
velocity reaches, both under the same shutter -- relMSE 0, bit for bit.
What the scene index does, measured: it extrapolates from the value the
frame reads, about the frame's time (`p(frame) + v (t - frame) / tcps`),
so the velocity stage authors its sample at the frame drawn; a sample at
frame 0 read at frame 0.5 had been held and then extrapolated about 0.5,
a frame's worth off the samples.

**Instancers move.** Under a shutter `HdAtheneaInstancer` samples each of its
per-instance arrays (`SamplePrimvar` of translations, rotations, scales and
transforms) and its own transform (`SampleInstancerTransform`) about the
shutter, keeping the samples' times, as a mesh does; the engine composes the
chain three times -- at the frame, and at the two samples, each level from
its own samples where it has them and the frame's arrays where it does not
-- and hands the set both motion chains. `GpuScene` puts moving sets first
among the sets, copies their records per shutter slice after the single
instances' copies (`instanceRecordsMotion`: the chain rows interpolated to
the slice's centre, times the prototype, answering to the slice's bit), and
the acceleration structures take those copies instead of the moving sets'
frame records -- the range stays contiguous because the moving sets lead.
Checked: two squares under a PointInstancer whose positions are time
sampled, path traced in eight slices, are bit for bit the same squares
authored as two meshes sliding by their transforms (relMSE 0, max 0), and
differ from the shutter closed (0.92); with the motion chains withheld from
the scene the instancer drew sharp, and the comparison failed.

**The camera moves.** The delegate's camera sprim is `HdAtheneaCamera`, hd's
`HdCamera` that also samples its transform about the shutter; the pass
hands the projection view to world at both samples (the camera's transform,
then the flip to +z -- no inverse) with their times. A moving camera makes
the scene cut the frame into shutter slices even when nothing else moves,
and turns on the path tracer's own primary rays, each sample's camera
interpolated to the centre of the slice its rays answer to -- the time the
moving geometry it meets is drawn at. The raster draws the frame's camera.
Checked: a camera sliding +x over two still squares, path traced in eight
slices, is bit for bit a still camera over the squares sliding -x (relMSE 0,
max 0), and differs from the shutter closed (0.92); with the samples
withheld from the kernel the camera drew sharp and the comparison failed.

**Lights move.** A light samples its transform about the shutter as a mesh
does; the table carries, only when some light moves, 26 floats a record
after the light BVH's nodes in the IES values buffer -- the rows at both
samples and their times, a still record's times equal -- since the path
tracer's kernel binds its 31 buffers already. A moving light makes the
frame's slices as moving geometry does, and each sample places the light it
chose (next event estimation, media, and the material's rays under MIS)
between its samples at the centre of the slice its rays answer to
(`lightFor`); its choice by power, and the light BVH's, stay the frame's.
Checked by the relative scene: a sphere light sliding over a floor with an
occluder, under a still camera, against the light still and the camera,
floor and occluder sliding the other way -- relMSE 4.9e-9 (p99 relative
1.7e-5) -- while against the light standing still the frames part by
2.8e-4, a soft shadow sweeping a part of the floor. With the samples
withheld from the kernel the moving light drew the still frame (3.7e-13).
(The test's first arrangement was wrong, not the renderer: the reversed
stage put the occluder two units from the light at mid frame instead of
one.)

**A shutter that changes after prims synced.** The pass marked every rprim
dirty through the change tracker, and under scene index emulation those
marks do not reach prims the stage's scene index owns (as with binding
purposes): a camera whose shutter was authored open after a first frame drew
the next frame half resampled, relMSE 1.18 from the stage authored so.
`StageRenderer` now dirties every prim's transform and primvars through its
own filtering scene index when the shutter it reads differs from the one
the prims were sampled about -- meshes, instancers, lights and the camera
alike -- and the frame after the edit is bit for bit the authored stage's.

**Instanced lights move.** A light under an instancer composes its chain
at the shutter's samples as a mesh does (`composeChains`, one helper for
both now), and the table writes each copy's 26 floats as the prototype's
rows at the two samples; `lightInstancesMotion` places them by the chain's
rows at the same samples on the device, as `lightInstances` places the
frame's. The times are the chain's where it moves, the light's where only
the prototype does. Checked in the moving light's test: the bulb as a
`PointInstancer` prototype whose one position slides 0 -> 2, against the
bulb's own transform sliding so -- relMSE 0, max 0; with the chain's
samples withheld the instanced frame parted from it by 3.1e-4, the still
light's distance.

**A host driving the plugin gets the same.** The pass's change tracker
marks could never do it -- under scene index emulation they do not reach
prims a scene index owns, and without emulation (UsdImagingGLEngine's
chain) `MarkRprimDirty` is refused outright as "requires emulation". So the
delegate registers a pass-through scene index of its own for this renderer
(`HdAtheneaResampleSceneIndex`, phase 4 at the end), which every chain built
for `athenea` holds, and finds it by walking the inputs of the
terminal scene index the render index hands it. `StageRenderer` uses the
same call. A shutter changed after the prims synced leaves two frames
unconverged, whatever they hold: the one that drew the old samples and the
one that resamples -- otherwise a host that draws until `IsConverged` stops
at the stale frame. Checked in `athenea_host_tests`, an executable that runs
`UsdImagingGLEngine` -- usdview's engine -- on the plugin loaded by name:
the frame after the edit is the fresh engine's on the edited stage (relMSE
0, max 0) and differs from the sharp one by 1.31. That executable links no
`athenea::usd`: with the delegate's classes in the executable as well as in the
plugin, a template instantiated in both (`make_shared` of the render pass)
binds to the executable's copy, and the pass then fails to recognise the
plugin's own render buffers -- measured, as an image of zeros with five AOV
bindings and no outputs.

**Not done.** The raster technique draws the
frame's time, no blur. A turn between the two shutter samples is
interpolated as rows, not as a rotation. Two samples only: a shutter that
spans more than two authored samples takes the outer two. Where a prim's
transform and points both move at different sample times, the transform's
times are taken for both.

### Skinning and blend shapes, on the device

**Where the inputs come from.** In 26.08 usdSkelImaging resolves a
skinned prim through scene indices: `UsdSkelImagingPointsResolvingSceneIndex`
adds two ext computation prims under the mesh -- an aggregator holding what
does not change per frame (`restPoints`, `geomBindXform`, the joint
`influences` as (joint, weight) pairs with `numInfluencesPerComponent` and
`hasConstantInfluences`, `blendShapeOffsets` as (xyz, sub-shape) with a
`blendShapeOffsetRanges` pair a point) and the computation itself holding
the animation's (`skinningXforms` or `skinningDualQuats` with
`skinningScaleXforms`, `blendShapeWeights` a sub-shape, `skelLocalToWorld`,
`primWorldToLocal`) -- and hands the mesh its `points` as that
computation's output. Hydra never runs the computation for us: the
delegate declares the `extComputation` sprim (`HdExtComputation`, as it
comes), `HdAtheneaMesh::Sync` finds the computed `points` primvar, walks the
computation's scene inputs and its aggregator's outputs by name, and hands
the values whole to the engine (`SkinningArrays`), the rest points standing
in for the mesh's own.

**What runs.** `geom::Skinner` uploads those arrays as they are and
`skinning.slang` (`skinPoints`) does what usdSkelImaging's own
`skinning.glslfx` does, so a host that runs the computation itself and this
one read the same: sub-shape offsets summed into the rest point by their
weights; then linear blend skinning -- each influence's transform applied
to the point taken into bind space by `geomBindXform`, weighed -- or dual
quaternion skinning, the influences' dual quaternions blended on the
pivot's hemisphere (the heaviest influence's), normalised, any scale
applied linearly, then the point turned and moved by the blend; then
`primWorldToLocal * skelLocalToWorld`. The rest points are widened to
float4 by the builder's own decode kernel; the only host work is
transposing the matrices to the rows the kernel multiplies with. The
skinned positions go into `MeshBuilder::build` through
`MeshInput::devicePositions` -- the builder copies them instead of decoding
`points`, so normals and bounds are the skinned mesh's -- under the mesh's
topology key, so a frame of animation is a deformation in place and a
refit. The sub-shape weights, including an inbetween's share of a shape's
weight, are resolved by usdSkel on the host before they reach the
computation: a few floats a shape a frame, USD's own code.

**Checked** in `test_skinning` against closed forms a check kernel
evaluates a second time: one joint of weight 1 through non-trivial
`geomBind`, joint, `skelLocalToWorld` and `primWorldToLocal` matrices is
the matrix chain, 0 of 64 points beyond 1e-5, per-point and constant
influences alike; two joints turning about one axis by 20 and 80 degrees
at equal weight blend, under dual quaternions, to the 50 degree turn
exactly (worst 2.4e-7), where linear blending of the same pulls all 64
points off by up to 0.2; three sub-shapes over 40 of 64 points add their
offsets by their weights exactly. Through Hydra: a square bound to the
sliding joint of a two-joint skeleton is the square authored with that
slide as its transform, 0 pixels beyond 2 at rest and slid, by raster and
by rays, its generation standing across the frames; a blend shape with an
inbetween authored at 0.5 draws, at weight 1, as the square authored with
the shape's offsets and, at 0.5, as the square authored with the
inbetween's own offsets -- not half the shape's -- 0 pixels beyond 2 both.

**A bug the check caught in itself.** The first check kernel read a
point's blend shape range past the end of the ranges buffer for the points
without one; alone the stale memory read as zeros, after two other cases
it did not, and one thread looped for billions of steps -- the device
hung, `submit` never returned, and the case only "failed" by the SIGTERM
that ended it. The kernel now takes how many points have a range, as the
skinning kernel always did. A check has to guard what the kernel guards.

**Not done.** The deferred-skinning route (`HD_ENABLE_DEFERRED_SKINNING`,
`hydra:skinningXforms` and the rest as primvars named by
`HdSkinningSettings::GetSkinningInputNames`) is not read: the variable has
to be set before Hydra loads, which only a process's `main` can do through
`platform::setEnvOnce`, and it would be a second reader of the same
Skinner; the ext computation route is the one every host gets. Skinned
normals are recomputed from the skinned points (smooth) rather than
skinned from authored normals (`skinningNormalsComputation` is not read).
A skinned mesh under a shutter blurs by its transform only: the skinning
transforms are read at the frame, not at the shutter's samples.

### The timeline

`athenea view` had a time slider; it now plays. Play advances the time by the
wall clock at the stage's `timeCodesPerSecond` and wraps at the end, the
step buttons move a frame, and dragging the slider stops the play. What
frame N shows is `SetTime`'s business and when it is drawn the clock's --
the two are not mixed, which is also how `athenea live` already worked: its
`sched` clock decides when frame N is drawn, and `--start` what frame N
is. Nothing here is measured beyond the frame times the panel shows.

## Complete USD: the breadth of geometry (M8)

### hdsi's conversions ahead of the delegate

The delegate draws meshes, points and splats; what USD authors beyond
those reaches it as meshes through hdsi's scene indices, registered for
this renderer in phases ahead of light linking: `HdsiImplicitSurfaceSceneIndex`
with every implicit type (sphere, cube, cone, cylinder, capsule, plane)
set to `toMesh`, `HdsiTetMeshConversionSceneIndex` (a TetMesh's surface
faces), `HdsiNurbsApproximatingSceneIndex` (a NurbsPatch as a mesh),
`HdsiPinnedCurveExpandingSceneIndex` (for the curves M8 adds below), and
`HdsiCoordSysPrimSceneIndex`, which turns a coordinate system bound to any
xformable into a `coordSys` prim under it with that prim's transform. The
delegate's own code did not change; the registration and its arguments did.

**Checked** through Hydra: a `UsdGeomSphere` of radius 1.2 against the
analytic sphere in `sphereCheck` (beside `planeCheck`) -- hdsi tessellates
it with ten segments, so a chord sits inside the sphere by up to
`r (1 - cos(pi/10))` = 0.0587, coverage is judged outside that band about
the silhouette and depth where the ray meets the sphere squarely (within
0.8 r of the axis, where a facet's error along the ray is at most the sag
over 0.6): 0 of 26788 pixels wrong, depth within 0.075 of the sphere over
9772 pixels. A one-tetrahedron `TetMesh` against its four faces authored as
a mesh, by depth (hdsi winds the surface its own way and the headlight
shades the side it sees): relMSE 0. A degree-one `NurbsPatch` against its
quad: 0 pixels beyond 2. The first sphere check was wrong itself -- it took
`sqrt(dist^2 - r^2)` for `dist - r` and judged half the disc wrong -- and
Python recounting the same formula on the dumped depth reproduced the count
exactly, which is what told the kernel from the expectation.

### Invisible faces

A face Hydra marks invisible (`HdMeshTopology::GetInvisibleFaces`) stays
in the topology and is not drawn: unlike a hole, whose triangles the
triangulation drops, an invisible face keeps its triangles and their
numbering, so showing it again is a flag and not a rebuild. The builder
marks the faces the way it marks holes (the same kernel over another
list), `subsetTriangles` writes the flag into the top bit of each
triangle's subset word (a subset index never reaches it), the scene pools
it as it pools the subsets, and the one place every route already
evaluates a sample before keeping it -- the cutout passes' `materialCuts`
-- answers yes for a flagged triangle before it looks at the material. The
engine takes the cutout passes whenever a mesh has a hidden face, cutout
materials or not.

**Why a bit and not a buffer.** The first version gave the flag a buffer
of its own, bound wherever materials are looked up, and every path traced
test failed at once: Metal allows a kernel 31 buffers, the path tracer
was at the edge, and the one more put a binding out of range -- the
kernel did not compile, and the suite said so 18 times. The lesson stands
in the docs because it will bite again: the material frame binds a dozen
buffers and the path tracer adds its own, so a new per-triangle or
per-mesh datum rides in a word that exists.

**Checked** in `test_lights` on the bumpy grid with its odd faces
invisible, by the three routes: every pixel where the full grid showed an
even face shows exactly that (instance, triangle) -- 0 of 9945 differ, by
raster, rays and the compute walker alike, so hiding renumbered nothing
and hid nothing it should not -- and none of the 9945 lies on an odd face;
19903 pixels were drawn with every face. A first version of the check
compared every drawn pixel with the full grid's and found 131 differing:
pixels where an odd face had stood in front of an even one, which the
hidden grid rightly shows. What USD does not have is a way to author
invisible faces on a mesh (`UsdGeomSubset` carries no visibility;
`invisibleIds` is for points and curves), so the Hydra side is read and
not exercised by a stage.

### Basis curves, as tubes

`UsdGeomBasisCurves` arrive through `HdAtheneaBasisCurves` (an `HdRprim`, as
points are) as their topology, points and widths, and `geom::CurveBuilder`
lays a tube over every span on the device: `curve_tube.slang` evaluates
the span at `segments + 1` parameters -- the Bezier, uniform B-spline,
Catmull-Rom or linear form of its control points, with the derivative for
the tangent -- and rings `sides` vertices at half the width about each
point in a frame taken from the tangent; the width is constant, a curve's,
a control point's (blended like the point) or a span end's. The rings'
quads are host bookkeeping (a pattern of indices), the positions the
kernel's, and the whole goes through `MeshBuilder::build` as device
positions with smooth normals, so a curve is a `GpuMesh`: every visibility
route draws it, shading, AOVs, cutouts, linking, picking and the path
tracer take it as a mesh, and a curve whose points move is a deformation
and a refit (the same topology key). Periodic curves wrap their spans;
pinned ones are expanded by hdsi ahead of the delegate. A uniform primvar
becomes one value a face, each face taking its curve's.

This is not the plan's design -- a curve primitive of its own in the
visibility buffer, ribbons in raster and swept cones in the BVH -- and the
reason is cost against what it buys: the plan's purpose was that nothing
downstream should change, and a tube as a mesh changes nothing downstream
at all, at the price of a fixed tessellation (a hair far away aliases as
any thin mesh does, and a head of hair is many triangles). A curve
intersector is where the design should go if hair at scale is wanted; the
tube is exact in what it draws.

**Checked** in `test_curves` against the curve: for every basis, periodic
and not, every tube vertex sits at half the width from the point of the
curve it rings, that point evaluated a second time in the check kernel in
the Bernstein, polynomial and Hermite forms (not the builder's), 0 of up
to 504 vertices beyond 1e-5, worst 2.8e-7; the span counts and the
triangle counts are what the rule says. Through Hydra, a straight linear
curve of width 0.4 is a cylinder: `cylinderCheck` finds 0 pixels wrong
beyond the eight-sided tube's facet band and the depth within the sag
over 0.7 of the cylinder's front (0.0169 against 0.0152), by raster, rays
and the compute walker alike.

### The hair lobe (Chiang et al. 2016)

`chiang_hair_bsdf` is one more kind in the lobe library, `kLobeHair`, so
the stack, its one-sample MIS and every consumer are unchanged. The lobe
carries the R, TT and TRT tints in the three colours, the R roughness in
`alpha`, the cuticle angle in `roughness`, the fibre's ior, and the
absorption coefficient and the TT and TRT roughnesses in the struct's pads
and in Schlick's `exponent` (`setHair`, `hairAbsorption` and the two
roughness readers keep that in one place). The Lobe stays 128 bytes on
purpose: the first version grew it by two float4, and the shadowed
shading kernel then drew a point light's umbra wrong on Metal -- 2583 of
7440 pixels off a closed form that had been exact, in a kernel that never
touches the hair -- and packing the fields into the pads made it exact
again. The cause is not explained; the size is what was measured to
matter, so it is held. The MaterialX node's Slang implementation
(`athenea_chiang_hair_bsdf.slang`) fills it and pushes it with the node's
`curve_direction` as the tangent. The model is pbrt-v3's: for lobes p = R,
TT, TRT and a geometric tail, a longitudinal density M_p over theta_i
(d'Eon's, normalised against cos theta d theta), an azimuthal one N_p
over phi (a trimmed logistic about the lobe's centre), and an attenuation
A_p from the Fresnel terms and the absorption through the fibre; f cos
theta_i is their sum, the pdf the same sum with each lobe's share of the
attenuation in place of A_p, and a sample chooses a lobe by that share,
draws theta_i from M_p and phi from N_p. The offset across the fibre comes
from the hit's normal against the outgoing direction, as MaterialX's own
eval takes it: on a tube the normal is the radial one, so the offset
follows from it, and no separate curve intersector is needed. The fibre's
direction is the tube's `tangent` primvar (the curve builder writes one a
vertex, the builder takes it as a device primvar, `material_surface`
prefers it to the texture-derived tangent).

Two things differ from MaterialX's `mx_chiang_hair_bsdf.glsl` on purpose.
The cuticle tilts the *outgoing* angle per lobe (pbrt's way), not the
incoming one (MaterialX's): shifting theta_i changes the measure the
density is normalised against, and a sampler drawing from the shifted
density no longer matches the pdf -- the chi-square said z 126 before the
change and 1.1 after. And the tail lobe's azimuthal density is 1 / (2 pi):
MaterialX writes `1.0 / 2.0 * M_PI`, which is pi / 2.

**Checked** in `test_lobes` as every lobe is: a million samples binned
over the sphere against the pdf integrated over the bins, and the albedo
from the samples' weights against the albedo from eval over uniform
directions. With no absorption and white tints the attenuations sum to
one, so the lobe returns everything: albedo 1.0000 sampled, 0.9944
uniform, z -0.8; absorbing, off-axis at 60 degrees, with and without the
cuticle's tilt: z -1.0 and 1.1, the two albedos 0.307 against 0.306. The
pdf's integral over the bins reaches 0.996 under the R lobe's variance of
0.1 -- the bins' quadrature, allowed 0.01 for hair where the others get
0.002. The model is not reciprocal (the attenuation is the outgoing
side's), as its authors' is not, so no reciprocity is asserted. Two of the
lobe's own bugs the checks caught: a fourth random number taken from the
second's low bits tied theta_i to phi_i (z 4.8), fixed by rescaling what
the lobe choice left over; and dividing the pdf by cos theta_i as well as
f, which put its integral at 1.32.

Through Hydra, a B-spline curve under a chiang material draws lit
(relMSE 4.9 against blank) and unlike the same curve under Lambert
(relMSE 0.13).

### Subdivision surfaces on the device

`geom::Subdivider` refines a mesh `levels` times under Catmull-Clark, Loop
(all-triangle meshes; anything else falls back to Catmull-Clark) or
bilinear rules, and pushes the last level onto the limit surface. The
split of work follows the rule: the host lays out each level's topology
as tables of indices -- which corners make which edges (an edge map over
(min, max) pairs), what each vertex touches (a CSR of its edges and faces),
which edges and vertices are sharp, and the children's corner lists in the
[vertex points][edge points][face points] numbering -- and
`subdivision.slang` places every point: a face's centroid, an edge's point
(the four-point Catmull-Clark rule, Loop's 3/8-1/8, or the midpoint on a
boundary or a sharp edge, blended by a semi-sharp edge's sharpness), and a
vertex's new place (Catmull-Clark's (Q + 2R + (n-3)P)/n, Loop's beta rule,
the crease rule (E1 + 6P + E2)/8 through two sharp edges, blended by
sharpness; a corner, a vertex with three or more sharp edges, or a boundary
vertex that is only its two boundary edges stays -- edgeAndCorner for
points, cornersOnly for a face-varying channel). Creases and corners come
from `UsdGeomMesh` as authored (a sharpness a crease or an edge), a
boundary edge is a crease, and a child edge keeps its parent's sharpness
less one. Face-varying channels run the same kernels over their own
topology: the channel's unique values (its indices) are its vertices, an
edge one face makes is a seam and so a boundary; vertex channels follow
the points, varying ones the bilinear rule, uniform ones map each refined
face to its coarse face, and authored normals are dropped for the refined
surface's own. The limit projection is `subdivLimit`: Halstead, Kass and
DeRose's stencil for Catmull-Clark -- (n^2 P + 4 sum of edge neighbours +
sum of the quads' diagonal vertices) / (n (n + 5)) -- and (1 - n gamma) P
+ gamma sum of neighbours for Loop, (E1 + 4P + E2)/6 along a crease or
boundary, corners fixed. `ref/falcor`'s LoopSubdivide was read for the
Loop rules and never built.

**Checked** in `test_subdivision`, everything exact:

- **Euler's counts**, integer and exact at every level: a cube's 26/24/48
  points, faces and edges after one level, 98/96/192 after two, 386/384/768
  after three; an octahedron under Loop 18/32/48, 66/128/192, 258/512/768.
- **The limit converges and does not drift.** The cube corner's limit from
  the coarse cube alone, by the closed form written a second time in the
  check kernel, is (-0.5, -0.5, -0.5); the kernel's projection of that
  corner's descendant reaches it to 0.000000 at levels 1, 2 and 3 alike.
  The first stencil written took edge midpoints and face centroids for the
  two sums and put the corner at 0.75; the projections then moved with the
  level (0.38, 0.42, 0.43 off), which is what told it from the right one --
  a limit stencil applied at any level must land on one point. Loop's
  octahedron vertex reaches its closed form (0.5, 0, 0) to 0.000000 at
  every level.
- **A crease holds its plane.** A sharpness-10 crease around the cube's
  top keeps every descendant of the top on y = 1 (9 points at level 1, 25
  at level 2, none above) where the smooth cube's highest point is 0.8395.
- **A face-varying square stays a grid**: each of the cube's faces with
  its own unit square of st, refined twice, gives 96 faces whose four st
  corners are all axis-aligned rectangles (a first version of the test
  gave the six faces the same four values, and the six squares became one
  face-varying vertex a corner with no seam).

**Through Hydra.** `HdAtheneaMesh` hands the engine the scheme, the display
style's refine level and `UsdGeomMesh`'s creases and corners; the engine
refines a mesh whose scheme is not `none` at a level above zero (after
the skinning, when there is any; five levels at most) and builds the
refined mesh under the coarse mesh's topology key, so an animated
subdivision surface is still a refit. The level is the display style's,
as usdview's complexity sets it: `StageRenderer::setRefineLevel` and
`athenea stage --refine` set it through `HdsiLegacyDisplayStyleOverrideSceneIndex`
inserted ahead of the renderer's chain; a host's `HdDisplayStyle` reaches
the delegate the ordinary way. Checked with a catmullClark cube at refine
0, 1 and 2: the centre pixel's depth is the flat face's 5.0000 at 0, and
5.168 and 5.164 at 1 and 2 -- the limit surface inside the cube, one
surface at both levels to a facet's sag -- while the coverage falls from
8464 pixels to 2956 and 3196 as the corners pull in.

**Not done.** Boundary interpolation is `edgeAndCorner` and face-varying
interpolation `cornersOnly` whatever the mesh authors (`edgeOnly` and the
other face-varying rules are not read); holes are refined and dropped
after; invisible faces do not survive refinement; a mesh's refined
normals are the refined surface's smooth ones, not limit normals; Storm
was not used as an image oracle here -- the closed forms above are the
checks, and Storm's own OpenSubdiv would have been checked against them
the same way.

### The light BVH

With `athenea:chooseLights` a sample takes one light; it took it by power
alone, which is exact and blind to where the point is. Now the frame's
bounded lights (sphere, disk, rect, cylinder) sit in a BVH
(`light_bvh.slang`), each node a box, an orientation cone (Conty and
Kulla 2018's union of its children's) and the power under it, and a
shading point descends it choosing each child by its importance there --
power over distance squared, within what the cone allows, times the
surface's own cosine with the box's angular uncertainty allowed for --
so a light that cannot reach the point is rarely chosen and one beside it
often. Dome and distant lights are unbounded and never enter the tree:
they sit in a list after it in the same buffer, chosen against the tree
by their share of the power, so a dome never vanishes in silence. The
probability of the choice is the product of the branch fractions, and
`lightPdfChoiceAny` recomputes it for any light by walking its leaf's
parents; where neither child of a node can light the point by its bound
(the parent's looser bound let the descent in) the choice falls back to
power, so no probability is lost.

The build (`light_bvh_build.slang`, `LightTable::buildBvh`) reuses the
compute LBVH's own kernels over the lights' boxes -- Morton codes within
bounds a kernel found, `bvh_hierarchy`, `bvh_refit` settling the boxes --
then packs the hierarchy into nodes, links the parents and settles power
and cones up the tree until nothing changes. The nodes are sixteen floats
each **inside the IES values buffer**, after the profiles, and the shading
kernels read them through `lightNodeBase`: a buffer of their own put the
shading kernel at `buffer(38)` against Metal's thirty-one. And the build's
kernels live in a module of their own because a Slang module's global
parameters join every kernel that imports it -- with the build's buffers
in `light_bvh.slang`, importing the tree to choose from it cost the
shading kernels ten bindings they never used, and the same error came
back with no new buffer in sight.

**Checked** in `test_lights` with forty lights of the four bounded kinds
scattered and turned, a dome and a sun: 0 of 39 internal nodes whose box
or cone misses a child's; at a hundred random points and normals the
choice's probabilities over the 42 lights sum to one (worst 3.6e-7 off);
a million draws at one point all report the probability
`lightPdfChoiceAny` recomputes for the light drawn (0 mismatches), and
their histogram follows those probabilities -- chi-square z 0.62 on 29
degrees of freedom, which is what a per-sample check cannot see and M5's
lesson asked for; the dome's share is 0.0168, the sun's 0.0479. Then the
many-lights closed form, chosen through the tree, holds as it did by
power. A first cone union swapped the cones the wrong way round (13 of
39 cones failed to hold their children) and a first descent returned
nothing where both children's bounds gave zero, losing 2.2% of the
probability (66 of 100 points summed short); both were found by these
checks, not by an image.

**Not done.** Light instancing's copies each take a leaf (a thousand
copies are a thousand leaves); a light's cone ignores an IES profile's
shape; the tree is rebuilt whenever the table is set, which is every
frame the lights change and never when they do not. The table's tree
uniforms are set only where a kernel declares them: the first `bind`
that set them everywhere put a null cursor under two kernels that sample
by power alone (the IES check, the dome's plane) and both crashed the
host, which the suite found and a single test would not have.

### Coordinate systems, resolved by the scene index's prim

A material may name a coordinate system (`UsdShadeCoordSysAPI`: a name on
the prim, bound to an xformable). `HdsiCoordSysPrimSceneIndex` makes a
`coordSys` prim under the target with the target's transform, and the
emulation gives the prim a name of the binding's (`__coordSys_paint`); the
delegate accepts the `coordSys` sprim (`HdCoordSys`, whose Sync is hd's),
and a mesh's Sync reads `GetCoordSysBindings` and each binding's transform
into its `MeshLook` (`CoordSysBinding`: name, to-world), which the engine
keeps beside the mesh and `StageRenderer::coordSysBindings` reads back.
The name is what remains of the prim's past the scene index's prefix --
the prefix is a known constant, not a parsed path -- and the transform is
the prim's, so a target that moves moves its system with no code of ours
computing it.

**Checked** through Hydra: a square with `CoordSysAPI:paint` bound to an
`Xform` translated to (1, 2, 3) reads back one system named `paint` at
exactly that translation.

**Materials read them**, through MaterialX's `transformpoint`,
`transformvector` and `transformnormal`: a space named `world`, `object`
(or `model`) or anything else, which is a system bound to the prim. The
engine hands each binding's transform to world to the mesh as three
constant float4 primvars, `atheneaCoordSys_NAME_0` to `_2` (the rows of a 3x4,
as Hydra gave them; a name with colons is not a MaterialX identifier, which
the first spelling found), and the transform node reads them through the
primvar slots every material already uses -- a per-mesh value with no new
binding in any kernel. `MaterialInputs` carries the object-to-world rows,
composed on the device from the instance's object-to-view rows and view to
world; inverses and inverse transposes are the shader's
(`atheneaTransformBetween`). A system the prim does not have reads as world.

**A defect this found**: genslang's own transform nodes multiplied by
`u_worldMatrix` and its inverse, uniforms the compiler read from the blob
with nothing written there -- identity. Every object/world transform in a
MaterialX graph was a no-op on any mesh away from the origin. The four world
matrices are now built from the shading point's rows, as the float4x4
`mx_matrix_mul` applies.

**Checked** against graphs that reach the same value without a transform
node, frames compared (`compareHdr`), on a square tilted in object space
under a rotation, a non-uniform scale and a translation, with a frame
translated and scaled by 2 bound as `paint`: object point to world against
world position, 1.66e-5 largest relative difference; world to object
against object position, the same; world to `paint` against (P - t) / 2,
the same; the object normal to world against the world normal, bit for bit
-- with a control that the untransformed object normal differs from it
(0.30). That reference graph also showed **hdMtlx typing a USD `vector3f`
value as a string** (a role type MaterialX has no name for), so the value
never reached a `vector3` input and read zero; a string input the nodedef
declares numeric now takes the declared type, and its text parses as that.

**Not done.** A binding whose target moves while the mesh's arrays do not
is read again only when the mesh is rebuilt: the rows travel with the mesh's
primvars. Curves do not take the systems' primvars.

## Complete USD: render settings and outputs (M10)

### Render settings prims, products and vars

A `UsdRenderSettings` prim reaches the delegate as Hydra's `renderSettings`
bprim (`HdAtheneaRenderSettings`, hd's `HdRenderSettings` with a sync counter),
through `HdsiRenderSettingsFilteringSceneIndex` registered with the `athenea`
namespace prefix so `athenea:pathSamples`, `athenea:technique` and the rest arrive
as its namespaced settings, and `HdsiSceneGlobalsSceneIndex`, which is
where the active settings prim is named. `StageRenderer::renderSettings`
makes a prim active, syncs the prims (no tasks, so no frame) and reads the
bprim back: products, vars, purposes, colour space, camera, its settings.
`renderProducts` renders each product at its own resolution from its own
camera with `includedPurposes` as the render tags (default: geometry,
render: render, proxy, guide as themselves) and the `athenea:` settings applied,
and writes its vars as the layers of one OpenEXR (`io::writeExrChannels`:
named channels, half, float or uint, sorted as the format wants; `readExrChannels`
reads any file's channels back). A var names its AOV by `sourceName` (hd's
names; RenderMan's `Ci` and `z` read as colour and depth; `primvar` sources
as `primvars:NAME`; of light path expressions, `C.*<L.'NAME'>` as the light
group NAME). Layers are `NAME.R/G/B/A`, `NAME.x/y/z`, `Z` for a var named
depth, a bare `NAME` for an id plane (uint, so -1 stays 0xFFFFFFFF); 32-bit
floats unless `athenea:exrHalf` is set. `athenea stage --render-settings /Render/X`
renders the products and stops.

**Two things the plumbing needed that the reference host does not say.** A
settings prim's `active` is a dependency the filtering scene index
declares on the scene globals, and dependencies become dirty notices only
through `HdDependencyForwardingSceneIndex`, which the renderer's chain now
ends in; without it the second prim made active synced never and read
inactive. And marking the bprim dirty through the change tracker is a
no-op for a prim the emulation delivered: it dirties the legacy prim scene
index, which does not hold it.

**Checked** through Hydra: a settings prim with two squares (one a guide),
a product of four vars (beauty from `Ci`, depth from `z`, primId, Neye) at
96x64 -- the prim reads back active, synced once, with its purposes and
`athenea:` settings; the product's nine channels, read back from the file and
uploaded, are bit for bit the AOVs rendered one at a time (`countDifferent`
0 words for every layer); a second settings prim that includes guides
covers 3812 pixels against 2916, and reads active once asked, the first
one again after. The rays' render tags are part of the path tracer's frame
key, as the plan asked: a purpose that changes starts the mean again.

### Light groups

A light's `athenea:lightGroup` (or RenderMan's `ri:light:lightGroup`) names its
group; a frame asks for a group as the `lightGroup:NAME` output, and the
engine numbers the groups asked for (1 + index; 0 for a light in none, or
in one nobody asked for) into `LightRecord.group`. Both shading kernels
accumulate each light's direct contribution under its group -- the raster's
per light, the path tracer's at every bounce through the path's throughput
-- and a frame without groups compiles exactly the kernel it always did:
the groups' code is a variant of the generated source, not a branch.
Emission and the background are in no group. At most eight.

**Where the planes live** is a lesson twice over. The raster's go to a
buffer of their own. The path tracer's go into its accumulation buffer after
the colour's plane (sums, then means), because the traced kernel stands at
Metal's limit of 31 buffers: two more put it at `buffer(32)`, and the Metal
compiler in the process did not fail -- it never returned, ten minutes and
counting, while the same source through `xcrun metal` fails in a tenth of
a second with the error. `ATHENEA_SHADER_DUMP=<dir>` now writes every generated
module, which is how that was found. And accumulating the groups in a local
array indexed by the light's group, or writing the buffer inside the bounce
loop, were both tried first; eight registers and a select each is what the
kernel has.

**Checked** through Hydra with a floor under two lights in `key`, one in
`fill` and a var for an empty `rim`: at every pixel the groups summed are
the beauty (worst 1.8e-7 relative raster, 2.4e-7 path traced, over 4292
covered pixels) and the empty group is zero exactly; the same through a
render product whose vars are the light path expressions.

**A Metal problem this found.** The raster technique with any shadow ray
through Hydra wrote rows of garbage in blocks of half a threadgroup,
nondeterministic, on an Apple M5 Pro: clean under Metal shader validation,
clean without the trace, clean with the structure bound but unused -- and
clean once the kernel stopped copying the material's lobe stack into a
local and read it in thread memory where the material left it. The path
tracer, which keeps its stack in a struct it passes on, never showed it.
Live state across the intersector call is the reading that fits; it is
measured, not understood. Every Hydra light test had shadows off, which is
why nothing had caught it; the regression that does compares shadows on
against off over an unoccluded floor, three times, 0 words apart.

### ACES 2.0, on the device, tables and all

`aces2.slang` is the Academy's ACES 2.0 output transform ported function
for function from the reference CTL (aces-core `Lib.Academy.OutputTransform`
and `Lib.Academy.Tonescale`): Hellwig 2022's appearance model to J, M, h;
the tonescale on J; the in-gamut compression of M; the compression towards
the display cube's boundary; and its inverse. What the reference computes
once at init -- the reach of AP1 at every hue, the display gamut's cusp at
every hue with the hue table it samples, the upper hull's gamma -- are
tables of 362 entries built by three kernels (`aces2_prepare.slang`: the
parameters and the hue table on one thread, since the hue table follows
the sorted corner hues in order; then a thread a hue; then the wrap
entries) into one float buffer, `technique::Aces2Tables`, rebuilt when the
peak luminance or the limiting primaries change. The renderer's linear
Rec.709 goes to ACES2065-1 by a Bradford adaptation computed on the device
from the chromaticities. The display transform's third view, limited to
Rec.709 or to P3 as the display is; `athenea view` lists it.

**Checked** in `aces2_check.slang`, at 100 nits limited to Rec.709 and at
1000 nits limited to P3: 256 scene greys over sixteen stops come out
neutral (spread 5e-6), on the tonescale written a second time in the check
from the published constants in its own form (7e-6 relative), rising every
step; 18% grey shows 0.10000 of reference white at 100 nits (the
reference's 10.013 nits less its flare term); inverse then forward over
8192 display colours returns them to 6e-6 of the peak. The gamut
compression approximates the cube's boundary (a smooth cusp, a hull gamma
fitted at five points), so 4.7% of scene colours of any saturation and
exposure land a little outside it at 100 nits (worst 0.11 over), 0.45% at
1000 nits: the reference clamps at the display encoding, and so does the
display kernel; the check bounds the count and the worst.

### Extended range, and what OCIO is not

`DisplayEncoding::LinearP3` leaves linear P3 with 1.0 at the reference
white and the headroom above it, clamped at ACES's peak; `athenea view --edr`
asks the window's Metal layer for extended range content in
`kCGColorSpaceExtendedLinearDisplayP3` (`platform::enableExtendedRange`,
through the Objective-C runtime and CoreGraphics, in Platform where the
rule puts OS calls), takes an `RGBA16Float` surface, and shows ACES 2.0 with
the screen's headroom times 100 nits as its peak
(`platform::extendedRangeHeadroom`, the screen's
`maximumExtendedDynamicRangeColorComponentValue`). Measured here on a
screen reporting a headroom of 1.00: the pipeline runs (draw 12.6 ms
medians at 640x400, snapshot written), which shows the plumbing and not the
range; a screen with headroom is what would.

**OCIO as a compiler.** OpenColorIO 2.5.2 is built by
`scripts/build-ocio.sh` into `~/tools/ocio-2.5.2` on both machines (Ubuntu
ships 2.1, which has no ACES 2.0), its own dependencies linked into it
statically so none meets OpenUSD's. `DisplayTransform::setOcio` takes a
config (OCIO's built-in studio config by default), a display and a view,
from the engine's linear Rec.709 (`lin_rec709_scene`); the processor's GPU
shader is generated as HLSL, which Slang reads as it is, and wrapped in a
generated module named by its hash that imports the display kernel's
pieces -- the pixel under the output pixel, the premultiplied colour over
the background, exposure, and the non-colour modes -- so `ViewTransform::Ocio`
is the same kernel with OCIO's function where the view transform was. Two
rewrites of the text, both about the target and not the maths: 1D LUTs are
asked for as 2D textures (`setAllowTexture1D(false)`), and `Sample` becomes
`SampleLevel(..., 0)`, since a compute kernel has no derivatives and the
tables have one level. Uniforms (dynamic properties) are bound by the names
OCIO gives them; array uniforms are refused when the view compiles.

**The exception, as the plan wrote it.** The LUT values are OCIO's,
computed on the host when the processor is built -- the one piece of
arithmetic on colour this route does not do on the device. They are
uploaded as OCIO lays them out and placed into `RGBA32Float` textures by a
kernel (`athenea_ocio_fill`), so the host does not even re-lay them. Every pixel
is a kernel's. OCIO reports failure by throwing; `setOcio` is where those
exceptions stop and become a `Result`.

**Checked against aces2.slang.** The studio config's "ACES 2.0 - SDR 100
nits (Rec.709)" view on its sRGB display is OCIO's own implementation of the
output transform -- its fixed functions, its hue tables, its Rec.709 to
ACES2065-1 matrix -- and aces2.slang is a port of the reference CTL: two
implementations, each run on the device over the same sixteen stops of hues
and coverages at three exposures. 0 of 16448 pixels differ by more than
1/255 at any exposure; the worst difference is 3.2e-4 at 0 stops, 5.5e-4 at
+2.5 and 2.2e-5 at -3 (the test bounds it at 1e-3). The control, the
config's un-tone-mapped view, differs at every pixel (worst 4.9). The same
on the L4, under Vulkan and under CUDA alike: 0 pixels past 1/255, worst
3.2e-4, 5.5e-4 and 2.2e-5. `athenea view
--ocio-display D --ocio-view V [--ocio-config C]` starts on the OCIO view,
and the panel lists it beside the others (Kitchen_set, 800x450: draw 17.5
ms median, as AgX's 17.3).

**What OCIO does not settle.** `renderingColorSpace` is still read and
reported, not acted on: the engine renders in linear Rec.709 and OCIO is
told so. For a studio config that is not ACES there is no second
implementation to check against, only that the kernel runs what OCIO wrote.

**Light groups and what is not a light.** A dome the camera sees is in its
dome's group: `C.*<L.'NAME'>` matches the camera's ray meeting the light
with `.*` empty. The engine paints it into the group planes as it paints
the colour's background, and the camera's exposure scales the planes as it
scales the colour. For that the path tracer's group means are copied out of
its accumulation each frame into the engine's plane buffer (the raster
writes there itself): scaled in place, a converged pixel that no pass
rewrites would take the exposure twice. Checked with a sky dome in a group
of its own under one stop of exposure: the groups sum to the beauty at all
6144 pixels, sky included, raster and path traced; before, every pixel was
off (relative 1.0, the factor of two). Emission stays in no group, as the
expression says (an emitting surface is `O`, not `L`): a frame with
emissive geometry sums its groups short of the beauty by exactly that.

**Not done.**
`materialBindingPurposes` is applied since the usd-wg end to end (below):
this line used to say the delegate bound `full`, and it bound Hydra's
default, `preview`. Products' `disableMotionBlur` and
`disableDepthOfField` (theirs or their settings prim's) are applied per
product, through the delegate's `athenea:disableMotionBlur` (one shutter slice)
and `athenea:disableDepthOfField` (the camera's fStop ignored), reset when the
products are written: a sliding square under an open shutter and a lens
focused before it, with both switched off, is bit for bit the stage with
neither authored (0 words of 18432); the lens alone drawn, 4464 off; the
blur alone, 7506; a render after the products, the effects again. A render
var of any other
light path expression is refused with a message.

## Complete USD: volumes (M9)

### From .vdb to the device, with no statistic on the host

`io::readVdbGrid` reads a float grid with OpenVDB, voxelises its active
tiles (so the tree the kernels walk is leaves alone) and lays it out with
NanoVDB's converter with statistics and checksum off: what crosses to the
device is the layout and the header's leaf count. The rule's exception is
bounded to a re-layout, as decompressing SPZ is, and nothing about the
values is read on the host. `world::VolumeSet` puts every volume of the
frame in **one buffer of words** -- a header, a 32-word record a volume
(world to index, extinction scale, albedo, phase g, and what the finish
kernel writes: index bounds, leaf 0's word, majorant), the grids, then each
leaf's largest value -- because the traced kernel has one Metal buffer
slot left. `volume_prepare.slang`'s kernels take the bounds from the
leaves' origins, each leaf's maximum over its active voxels and the
grid's majorant.

PNanoVDB compiles as Slang: `athenea/volume/nanovdb.slang` includes
`PNanoVDB.h` as HLSL (its buffer is a `StructuredBuffer<uint>`) and exports
grids and leaves by offset; `slangc` takes it to Metal, CUDA and SPIR-V. The
header travels with the shaders from the USD prefix. NanoVDB 32's
`GridBuilder` still names `std::result_of`, removed in C++20, which libc++
keeps behind `_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS`.

**Checked** on a fixture of two boxes (32^3 voxels at 0.5, 8^3 apart at
2.0): 33280 active voxels counted through the leaf masks, bounds
[0,48)x[0,32)x[0,32), all 65 leaf maxima matching a voxel-by-voxel recount
through the accessor, majorant 2.0.

### The medium: delta and ratio tracking leaf by leaf

`medium.slang` walks a volume's leaves with PNanoVDB's HDDA at the leaf's
size and, inside each, samples free flights (delta tracking) and
transmittance (ratio tracking) under **that leaf's** majorant, so an empty
leaf costs one step and a dense one is sampled at its own rate. The ray is
taken to index space with its parameter kept in world units.
Henyey-Greenstein for the phase, drawn with its own density.

**Checked** through the dense box along six axis directions and one
oblique ray: ratio tracking's transmittance and delta tracking's scatter
fraction both on Beer-Lambert within the standard error the kernel
measures (|z| <= 1.2 over 16384 rays each); no density over the leaf
majorant at 16384 random points in and about the box; the leaf walk
crossing exactly the cells a walk of a tenth of a voxel crosses. That last
one needed the walk to skip cells a rounding sliver long at a boundary.

### In the path tracer

The traced kernel's sample loop now walks vertices that are either a
surface the ray met or a collision in a medium before it. A collision
gathers light through the phase (lights chosen by power) and sends the path
on by the phase; transmittance to a light is ratio-tracked -- **only for a
light that casts shadows**, since a volume is an occluder like any other and
`shadow:enable` off means none dims it. A pixel whose ray finds no surface
still walks the medium along its camera ray. An imageless dome is drawn
uniformly over the sphere at a point in a medium: its cosine sampling about
a normal never draws the half behind, and flipping between halves weighs a
sample near their horizon by one over its cosine. A frame without volumes
compiles exactly the kernel it did (`kVolumes` stubs).

**Checked**, pixel by pixel, with an absorbing box (albedo 0) over the right
half of a sun-lit plane: every pixel sees one radiance per sample, so the
frame with the box over the frame without is a binomial mean whose
expectation is Beer-Lambert along the pixel's own ray and whose deviation
the check derives from the path count. At 1024 paths, 0 of 9801 pixels
beyond five deviations and mean z^2 0.986 with the sun's shadows off
(camera chord only); 0 and 0.992 with them on (camera and light chords);
the 9680 pixels outside the box unchanged. **The volume furnace**: an
albedo-one medium of optical depth at most 0.56 under a dome of radiance 1,
32 bounces -- every walk ends where an escape would have seen the dome, so a
pixel reads 1 whatever the phase: 0.99983 isotropic, 1.00228 at g 0.7, within
0.2 and 0.6 standard errors measured from the pixels' own spread.

The first pass at the Beer-Lambert check read the square of the expected
transmittance: the renderer was right (the sun's light crosses the box
too) and the expectation was missing a chord. And one bug was the host's:
`VolumeSet` read OpenVDB's row-vector map untransposed and composed it on
the wrong side, so a translated volume sat offset by its translation in
voxels. A pure scale -- the first fixture -- cannot show that.

### Through Hydra

`HdAtheneaVolume` (the `volume` rprim) takes the field named `density`, else the
first, its transform, and constant primvars `athenea:densityScale` (default 1),
`athenea:albedo` (0.8) and `athenea:anisotropy` (g, 0); `HdAtheneaVolumeField` (the
`openvdbAsset` bprim) takes the file and grid name. The engine reads each
grid once per file and name, lays the frame's volumes out again whenever a
volume or field changes, and a change raises the path tracer's revision.
**Checked** with the same box authored as a `Volume` and an `OpenVDBAsset`
over a mesh plane under a `DistantLight`: 0 of 9801 pixels beyond five
deviations, mean z^2 1.014, mean ratio 0.1920 against 0.1921, the outside
unchanged.

**A frame of volumes alone** is drawn too. Path tracing runs in the mesh
layer, which the engine took only when a mesh was in the frame, and whose
scene and material programs it made only when a mesh had arrived; a traced
frame with a visible volume now takes the layer with an empty scene, the
visibility finds nothing, and every sample walks its camera ray. Checked as
the furnace through Hydra with no mesh at all -- a `DomeLight` of radiance 1
and an albedo-one `Volume`: 4278 pixels read 1.00044, 0.44 standard errors
from the dome.

### Not done

- The raster technique draws no volume, and the engine says so once.
- Only float grids, one field a volume (density); no temperature, emission
  or colour fields, no `UsdVol` material networks -- the medium's
  parameters are the three primvars.
- Light groups do not collect light scattered in a medium differently from
  a surface's: a collision's light goes to its light's group, as a
  surface's does.
- In a medium lights are chosen by power, not through the light BVH.
- No MIS in media; no spectral tracking for a coloured extinction.
- Verified on Metal; the L4 run of these tests is M11's.

## End to end: Kitchen_set lit, and two defects only a real stage showed

The plan closes with real stages through `athenea stage`. Kitchen_set with the
lights `Kitchen_set_lit.usda` adds (a dome, a window rect, two normalised
spheres), from inside the kitchen, rendered black under both techniques --
while the same view without lights, under the headlight, was right to the
last texture. Two defects, each hidden from the suite by the shape of its
fixtures.

### The geometric normal was reversed in view space

`surface.slang` formed a triangle's geometric normal as the cross product
of its view-space corners. The view is a reflection -- it flips z to look
down +z -- and a cross product under a transform of negative determinant
comes out reversed. So every front face read as a back one, and the
backface test flipped the shading normal. With a computed normal (the
geometric one) the two reversals cancelled; with an **authored** normal,
carried by the normal matrix which has no such sign, the shading normal
ended facing away. Lobes are evaluated with an absolute cosine, so a
sphere, a rect, a sun and the headlight all lit such a surface correctly;
a dome samples its directions about that normal, and lit nothing. Every
lighting fixture computed its normals. The geometric normal is now signed
by the object-to-view determinant, which covers mirrored instances too.

**Checked**: a Lambert plane of albedo 0.18 under an imageless dome of
radiance 1 reads 0.18 at every pixel (worst 1.7e-5 relative) with computed
and with authored normals, raster and rt; without the fix the authored
case is 1.09 off, the computed one passes -- as the cause says. The whole
suite passes with the sign; nothing depended on it.

The dielectric's `inside`, which read the same flipped test, is checked
through a camera on a closed cube (`inside_check.slang`, "a closed mesh is
inside only where it is seen from within"): every covered pixel's
`MaterialInputs` counted, with the camera outside the cube and within it,
the cube plain, mirrored, and authored left-handed. Outside: 7078 pixels, 0
inside; within: all 7081 inside; the shading normal handed to the material
faces the eye at every pixel of the six. Without the sign the plain and
left-handed cubes invert exactly (7078 inside from outside, 0 from within)
and the mirrored one reads right, its reflection cancelling the view's --
which is why no mirrored fixture could have caught it.

### A dome's share of the lights' power ignored the scene's size

The path tracer chooses one light a sample by power. An area light's power
was its radiance over its area in the scene's units; a dome's was its
radiance alone. In a kitchen in centimetres beside a 120 x 160 window light
the dome's share was 2.4e-6 -- unbiased, and dark and blotchy at 256 paths
a pixel, where the raster, which loops over every light, was lit. A dome of
radiance L lights a scene of radius R with pi L pi R^2 and a sun of
irradiance E with E pi R^2; over the pi common to every emitter these are
L pi R^2 and E R^2, set against an area light's L A. The table takes the
scene's radius; the engine reads it from `GpuScene::worldBounds` (a
kernel's) when the mesh set or its points change. **Checked** by a kernel
writing the shares again from the records at scene radii 1 and 400: 0 of 3
off (worst 3e-8). **Not done**: an instance moving without a change of the
mesh set keeps the old radius, which costs sampling efficiency and not
correctness.

After both, Kitchen_set lit renders under both techniques from inside the
kitchen; the path traced frame at 256 paths is clean without the denoiser.

### Measured (release, Apple M5 Pro)

`athenea stage --frames 11` at 1920x1080 from inside the kitchen, median of
the ten frames after the first (Hydra sync, drawing and the readback):

| Stage | raster | rt (1 path, 1 bounce) |
|---|---|---|
| Kitchen_set, no lights (headlight) | 43.1 ms | 333.3 ms |
| Kitchen_set lit (dome, rect, two spheres) | 96.0 ms | 358.9 ms |

The first frame, which loads the stage and compiles, is 4.5 to 5.5 s.
`athenea view --frames 120` on Kitchen_set lit at 1600x900, framing the whole
set: raster 33.3 ms a frame, rt 127.7 ms a frame while it accumulates
(medians).

### A render settings prim over a real stage

A layer over Kitchen_set lit adds a camera, light groups on its lights
(`sky`, `window`, `bulbs`) and a `RenderSettings` prim with one product of
seven vars -- beauty, depth, primId, normal and the three groups as light
path expressions -- traced at 256 paths and 3 bounces. `athenea stage
--render-settings /Render/Final` writes one EXR of 21 channels in 41 s
(release). Read back and checked with the light group kernel: the groups
sum to the beauty wherever a surface was drawn; the 19% of pixels where
they do not are the dome's background seen through the openings, which is
in no group, as M10 records. The `window` group is empty in this layer
because its rect, turned 90 degrees about Y, emits away from the kitchen:
the bench layer's authoring, not the renderer's.

## End to end: usd-wg assets and OpenUSD's own test stages

Nineteen stages, each drawn by `athenea stage` under raster and under rt (64
paths a pass, 128 in all, 2 bounces) at 640x480: from usd-wg/assets the
standard shader ball, McUsd, the chess set, the carbon frame bike, the
elephant with monochord, the spinning pyramids (subdivision, creases), the
MaterialX texture and texture coordinate tests, the Utah teapot and five
USDZ glTF conversions (DamagedHelmet, CesiumMan, RiggedFigure, BrainStem,
AnimatedCube); from OpenUSD's usdImagingGL testenv the basis curves, curves
with vertex colour, LBS skinning, the skinned arm, blend shapes, a VDB smoke
volume and the simple volumes. A stage without a camera is framed by a new
`athenea stage --frame-all`, which is also what a stage with no camera gets: a
small raster frame commits the scene and the engine's bounds of what it drew
place an orbit camera, as `athenea view` opens. Nine defects, every one found by
a real asset and none by a fixture, each now with a case of its own that
fails without its fix where a control was run:

- **Textures authored relative to their layer did not load.** hdMtlx
  writes a file input as its *authored* path, so `./textures/wood.jpg`
  was read relative to the process. The resolved path, which the value
  carries, goes in first ("a texture authored relative to its layer";
  without the fix the square reads black). The teapot, the shader ball, the
  bike and every USDZ lost their textures to this.
- **A float3 primvar reader into a colour input failed the material.** The
  reader is a vector to MaterialX; UsdPreviewSurface's diffuseColor a
  colour. It takes the colour nodedef where what it feeds is one (a tint
  primvar reads 0.699 0.300 0.499; without, the displayColor fallback).
- **USD's types for UsdUVTexture's scale and bias, and a colour into the
  vector normal.** USD authors float4, MaterialX declares color4; a normal
  map's rgb is a colour into a vector. hdMtlx types inputs by what they were
  given, the nodedef stops matching and the whole material fails. Inputs of
  the same float count take the declared type after hdMtlx builds the
  document. McUsd's 21 materials, the bike's, the helmet's and the
  elephant's all failed so. hdMtlx still prints its own mismatch messages
  before the fix-up runs: noise, not failure.
- **UsdUVTexture's wrap `repeat`** is `periodic` in MaterialX's enum, and
  `useMetadata` its default: mapped (CesiumMan and AnimatedCube failed).
- **Material binding purposes.** The delegate never overrode
  `GetMaterialBindingPurpose`, so Hydra's default, `preview`, was resolved,
  and a `material:binding:full` -- the shader ball's walls -- was never
  seen; M10's note said the opposite. The delegate answers `full`, and
  `StageRenderer` resolves a settings prim's `materialBindingPurposes` in a
  filtering scene index of its own (`BindingPurposes.h`), since hdsi's
  resolver fixes its purposes when made and a chain the render index
  observes cannot swap it. Changing the list dirties every prim with
  bindings; `HdChangeTracker::MarkAllRprimsDirty` does not, under scene
  index emulation, reach prims the stage scene index owns (measured: the
  test's square stayed blue). The case: full, preview then all-purpose,
  all-purpose alone, and a settings prim's list through its products.
- **Primvars on the material** (the blend shapes stage's
  `primvars:displayColor` on its Material) reach bound geometry only
  through `HdsiMaterialPrimvarTransferSceneIndex`, which Storm registers for
  itself: now registered for this renderer too, in Storm's phase. The blend
  shapes read green, as the baseline image does.
- **Materials identical but for their name compiled apart.** The generator
  names the surface's variables after the renderable element, which hdMtlx
  names after the material prim, so the shader ball's 17 materials were 13
  modules of 5 sources. The renderable is renamed before generation (17
  materials, 4 modules; "materials that differ in name and values alone
  share one module").
- **The path traced kernel ran the Metal compiler service out.** With the
  shader ball's 13 modules and the chess set's 15, `tracePaths` failed
  ("XPC_ERROR_CONNECTION_INTERRUPTED ... after multiple retries") after 4 to
  5 minutes. The front end is not it: `metal -c` on the translated source
  takes 0.6 s. Timed with a scratch tool that makes the library and the
  pipeline from the same source (chess, 15 materials, 37k lines of MSL):
  as generated, 233.6 s and the service gives up; with only the material
  dispatch marked `noinline`, 5.8 s. The dispatch was reached from five
  call sites -- the camera's hit, a lens sample's, a surface bounce's, a
  medium bounce's -- and each inlined copy carried every material. Slang
  accepts `[noinline]` and emits nothing for Metal, so the kernel was
  restructured instead: finding a hit (`Found`) and evaluating its material
  are apart, and materials are evaluated at one site in the vertex loop,
  the camera's hit still once a pixel. Chess under rt: 31 s for the whole
  command, the shader ball 30 s. The one behaviour that moved: the aux
  planes carry the first *shaded* surface hit, which differs from the old
  first hit only when a medium scatters in front of it on the first path.
  The raster's shading kernel has one call site and never showed it.
- **A stage without lights path traced black** while the raster lit it
  with its headlight. The engine now asks the path tracer for the same
  headlight when the frame has no lights (`PathSettings::headlight`), at the
  first vertex only: an unlit stage is the same image under both (p99 0,
  max 0). It is the engine's to ask and not the kernel's to assume: the
  closed emissive shell, lit by its emission alone, read 1.67 times its
  series when the kernel lit every lightless frame.

**athenea view and athenea live over the same stages** (release, Apple M5 Pro):

- `athenea view --play` starts the timeline as its Play button does, and
  `--frames` now reports how many distinct times the frames drew. The
  skinned arm (raster, 800x600): 240 frames, 240 distinct times, wrapping
  at the end, draw 8.7 ms; CesiumMan path traced: 240 of 240, draw 21.4 ms,
  the snapshot mid-walk. Without `--play`, 1 time over 120 frames. The
  viewer's technique menu said "Ray traced (splats)" for rt, from before rt
  path traced surfaces: "Path traced".
- `athenea live` renders a stage without cameras from the framing camera
  `athenea stage` uses (`StageRenderer::framingCamera`); it refused them. At
  25 fps on this machine's clock: CesiumMan raster at 1280x720, 50 frames,
  render 19.5 ms, none skipped; Kitchen_set raster at 1920x1080, 50 frames,
  render 25 ms, none skipped; the chess set path traced at 1280x720, one
  pass of one path, render 139 ms, so 25 frames written and 75 skipped for
  time -- said, not hidden.

Not defects, and left as they are:

- **McUsd blows out.** Its DistantLight and DomeLight leave intensity
  unauthored ("no intensity often helps the viewer pick a default"), and
  UsdLux's default for a distant light is 50000: with its 1 degree angle
  that is an illuminance of about 12. The renderer follows the schema.
- **The MaterialX texture test's teapot is black**: its `.mtlx` sets
  `fileprefix="./textures/"` and also writes `./textures/` in every value,
  so the path is `./textures/./textures/brass_color.jpg`; its own flattened
  sibling authors `textures/brass_color.jpg`.
- **glslfx materials** (the VDB test's checkerboard, the simple volumes'
  ellipsoids) are Storm's own and have no MaterialX network: displayColor.
- **The chess set's glass pawn heads** are black under the raster's
  headlight, which has no transmission.
- The shader ball under raster is dim beside rt: the box is lit mostly by
  its bounces, which raster does not draw.

## Linux, on the 94 (M11's first half)

### The first table, after the port was reconciled with engine

`engine` at the M6 path tracer, built with GCC 13 on the L4 -- once
`retainedDataSource.h` was patched: GCC rejects the injected-class-name written
with its template arguments in the bool specialisation's constructor
(`HdRetainedTypedSampledDataSource<bool>(const bool&)`), which only a
translation unit including that header meets, and the light linking's retained
data sources do. `scripts/build-usd.sh` now patches it after install so the two
machines build against the same prefix. Then `ctest`: **107 of 119 passed, 12
failed, 38 skipped** (no display, no gpe on this backend, no OptiX). The twelve,
before any of them is looked at: six of the splat ray tracer (the tests that
choose the Hardware route, with no OptiX to give one), three of eight-bit
textures (the port's own section below on what a float4 store becomes), the
lobes against MaterialX's genglsl, and two Hydra cases with splats. That is the
table M11 starts from; the plan's order verifies each new piece on both
devices from here.

The engine built and ran on Linux for the first time: Ubuntu 24.04, an NVIDIA
L4, CUDA as the backend. What follows is what the port needed, what it found
and what is still wrong.

### The toolchain

- **OpenUSD 26.08** with MaterialX 1.39.5 and OpenVDB/NanoVDB, from
  `scripts/build-usd.sh`. `--ignore-homebrew` is a macOS-only option of
  `build_usd.py`, so the script passes it only there.
- **OIDN 2.5.1** with the CUDA device. Its CUDA device wants CUDA 12.8 or
  newer and Ubuntu 24.04 ships 12.0, so `cuda-toolkit-12-8` goes beside it and
  both `scripts/build-oidn.sh` and the top-level CMake pick the newest
  `/usr/local/cuda-*` (`ATHENEA_CUDA_ROOT` overrides). One toolkit for gpe's
  kernels, slang-rhi's CUDA device and OIDN.
- **zlib** is found at the top level now: an imported target belongs to the
  directory that found it, and the tests link `ZLIB::ZLIB` too.
- **`<cstring>`** before slang-rhi's `acceleration-structure-utils.h`, which
  calls `memcpy` without including it -- libc++ carries it in anyway,
  libstdc++ does not.

### The CUDA driver's names were slang-rhi's variables

slang-rhi loads the CUDA driver with `dlopen` and keeps its entry points in
variables named after the driver's own functions (`cuInit`, `cuLaunchKernel`,
...). At global scope those variables are the definitions the rest of the
program binds to, and a strong data definition in an object file beats a
shared library's function whatever the link order -- both orders were tried.

So gpe and OIDN, which call the driver directly, called through slang-rhi's
pointers instead of through libcuda and died on a null one: `athenea info` exited
139 after printing correctly, and the gpu_host tests, every aofx host test and
`single_tbb` crashed in what the backtrace called `cuInit ()` at an address in
the executable's BSS.

`cmake/patches/slang-rhi-cuda-driver-symbols.patch` puts the block in
`namespace rhi::cuda_driver` with a using-declaration after it: the names keep
working inside slang-rhi and collide with nothing outside it. After it,
`cuInit` is undefined in our binaries (resolved from libcuda), `athenea info`
exits 0 and reports `denoiser OIDN 2.5.1 on CUDA` and `tbb libraries 1`, the
aofx tests are green and `single_tbb` passes.

### The radix sort on CUDA: a cursor re-read after it was advanced

For as little as two pairs, one chunk and one pass: the generator writes
`(1766601275, v0) (3252568193, v1)` and the sort leaves
`(1766601275, v1) (0, v0's slot never written)` -- two destinations collided.
What that rules out, each measured rather than argued:

- **The generator** is right: the probe dumps the pairs before the sort.
- **Ordering between passes** is not it: CUDA launches all go on one stream,
  and `ATHENEA_RADIX_SUBMIT_EACH_PASS=1` -- a submit and a wait between every pass
  -- changes nothing, the same 89 assertions fail.
- **The constants** reach each dispatch: four dispatches queued into one batch
  with four different parameter blocks each read back their own `count`,
  `chunkCount` and `shift`.

- **The counting, totalling and cursor stages are right.** One pass over two
  pairs (eight key bits is one pass) leaves a histogram counting both pairs,
  totals that agree, and 197 cursors past zero -- the same 197 as Metal.
  `RadixSort::working` names those buffers so a test can say so.
- **The bindings land where they are named.** Seven buffers, each holding its
  own marker, bound by the names the kernel declares: every name reads its
  own marker on CUDA as on Metal.
- **A scatter-shaped kernel reads what it was given.** The same seven
  bindings, one invocation, two elements, no sorting arithmetic: both elements
  read their own key and value on CUDA as on Metal. Its writes were another
  matter, and they are what gave the answer away: with the cursors zeroed, the
  probe reported taking slots 1 and 2 where Metal took 0 and 1, and the key of
  a pair landed one slot past the value written beside it from the same local.
- **Two wrong turns, recorded so they are not taken again.** Changing which
  buffer the scatter's unused high-word names point at appeared to change a
  low word the sort placed, and the slots the probe reported appeared to be
  off by one. Both readings came from buffers nobody had written -- a buffer a
  test makes is not zeroed -- and both evaporated once the probe wrote its own
  cursors. The sort itself never reads an uninitialised cursor: radix_starts
  writes every entry of every chunk's row. Giving the scatter distinct
  stand-ins made CUDA worse (four of the sort's cases passing fell to one), so
  it binds dummy_ and the histogram as it always did.

What was left was the generated code, and reading it ended the hunt. Slang's
CUDA emitter does not materialise `const uint at = chunkStarts[slot]`; it
keeps the cursor's address and re-reads through it, after the store that
advanced it:

```cuda
uint * _S14 = &chunkStarts_0[slot_0];
*(&chunkStarts_0[slot_0]) = *_S14 + 1U;   // the cursor now holds at + 1
*(&dstKeysLo_0[*_S14]) = _S9;             // re-read: one slot past
uint * _S16 = &dstValues_0[at_0];         // materialised: the right slot
```

One `at` in the source, two indices in the object code -- which is why a key
landed one slot past the value written beside it, and why every stage feeding
the scatter measured correct. The Metal emitter materialises the load, so the
suite here never saw it:

```metal
*((&kernelContext_0)->dstKeysLo_0+at_0) = lo_1;
```

Reduced to the smallest thing that shows it, for whoever takes this upstream
-- a load, a store that advances it, and a second use of the load:

```slang
RWStructuredBuffer<uint> cursor;   // cursor[0] starts at 0
RWStructuredBuffer<uint> out;

[shader("compute")]
[numthreads(1, 1, 1)]
void repro(uint3 tid: SV_DispatchThreadID) {
    const uint at = cursor[0];
    cursor[0] = at + 1;
    out[at] = 7;   // CUDA writes out[1]; Metal writes out[0]
}
```

**The fix is to place the pair and then advance the cursor.** With no store to
the cursor between its load and the uses, there is nothing for an emitter to
re-read, and the walk is one invocation's own, so the order costs nothing and
the sort stays stable. `slangc -target cuda` confirms it on the generated code
before any device runs it, which is the cheapest way to check this class of
bug and worth reaching for earlier next time: four stages were measured right
one at a time when one look at the emitted kernel would have said why.

### What else the port found

- **`SampleGrad` was not available in a compute entry point on the CUDA
  target** (Slang `E36107`). The material system answers that in the shader,
  with a `__target_switch` choosing gradients or an explicit level at the
  footprint's wider side, so the kernels build here now. What is left is not a
  capability error but wrong pixels, and reading the emitted CUDA says why.
  A decoded 8-bit image is an `RGBA8Unorm` texture written through an
  `RWTexture2D<float4>`, which on this target becomes
  `surf2Dwrite<float4>(texel, surf, x * 16, y)`: sixteen raw bytes of float at
  a sixteen-byte stride, into a surface holding four bytes a texel. CUDA
  surface writes do not convert formats; Metal's texture write does. So every
  component is wrong and each row runs four times past its end, which is the
  3386-of-3404 mismatch and the mip means collapsing to zero. No intrinsic is
  missing, so no `__target_switch` reaches it -- it is the store itself that
  means something different on the two targets. The float and half formats go
  through the same kernel untouched, which is why the gpu texture tests, which
  build `RGBA32Float`, pass here while the material ones do not.

  Measured, rather than read off the emitted code: a probe writes
  `float4(1, 0, 64/255, 1)` through an `RWTexture2D<float4>` into one texel of
  an `RGBA8Unorm` texture and reads that texel back, with no decoding,
  sampling or mips in the way. Metal returns the colour written. CUDA returns
  `(0.0, 0.0, 0.502, 0.247)` -- which is `00 00 80 3F`, the four bytes of
  `1.0f`, sitting in the texel as bytes. The store wrote the float4's memory,
  not its colour; the second, third and fourth components landed in the next
  three texels along, which is also why each row runs four times past its end.
- **gpu_host's `OutOfMemory` was never about memory.** A CUDA context is
  current per *thread*, and `CudaDevice::open` makes it current on the thread
  that creates the Context -- not on the worker `Context::run` spawns after
  it. gpe guards its own entry points (`ensureCurrent` before each), so gpe's
  work was fine; the work queued on the GPU thread reaches the same context
  through slang-rhi, so a buffer allocated there was the first driver call on
  a thread holding no context at all. 16 GB free, 4 KB refused, and the
  identical allocation off the thread succeeding. gpe grew a `bindThread()`
  for exactly this -- a caller about to reach the shared context another way
  -- and the GPU thread binds once before it takes work. Metal keeps no such
  per-thread state and takes the default, which does nothing.
- **A buffer nobody wrote is not a buffer of zeros, in the engine too.**
  `meshHoles` marks a hole face with a 1 and leaves every other face alone,
  and nothing else wrote `holeFlags`, so each face was judged by whatever the
  device last left there. Metal returned zeros; CUDA did not, and a stale word
  reads as "this face is a hole", dropping its triangles. The mixed-face mesh
  triangulated to 4 of its 10 triangles, and to none with the handedness
  flipped. The flags are cleared on the device before the topology dispatches.
- **Two failures that look like load, and are not.** The test presets already
  set `execution: { jobs: 1 }`, so the suite is serial as it stands. Run
  entirely alone on an idle box, the free-running clock test (#93) still fails
  in 0.4 seconds, twice running, and the lobe test (#52) still fails in 7.4
  seconds. Neither is contention and neither should be written off as the
  machine: #93 is a timing bound this box does not meet and #52 is a Monte
  Carlo tolerance, and both want measuring on their own terms.
- **Tests skip rather than fail where CUDA cannot answer**: no rasterisation,
  so the raster visibility, mesh and Storm-oracle tests skip with a reason, as
  does `athenea view` with no display.

### Open on this backend

**A float4 stored through an `RWTexture2D` does not arrive as the texture's
format on CUDA.** `athenea_gpu_tests` "a float4 written to an 8-bit texture comes
back as the colour it wrote" fails here, deliberately: it is the smallest
statement of the defect, one texel with no decoding, sampling or mips in the
way, and it names what it means when it fails. It is not skipped, because CUDA
can run it -- it answers wrongly, and a skip would hide that behind a green
suite. Skipping is for what a backend cannot do.

Everything the texture store decodes from an 8-bit image is wrong here until
it is fixed, which is most of what still fails: the PNG and UDIM tests
directly, and the ray tracer and USD tests that shade through a decoded image.
The fix is ours rather than Slang's -- the lowering is faithful, and the two
targets simply mean different things by the store. The obvious route, packing
the texel in the shader and storing it through a uint view of the same
texture, was tried and measured, and it does not hold: **the two backends fail
in opposite places.**

- The `float4` store converts on Metal and writes the float's own bytes on
  CUDA.
- A uint (`R32Uint`) view of the same `RGBA8Unorm` texture aliases correctly
  on CUDA -- written packed, it reads back as the colour through the texture's
  own view -- and does not alias on Metal, where the store lands (read back
  through the uint view, 0 of 256 texels differ) but is invisible through the
  colour view (256 of 256 differ). Metal wants the texture created able to be
  viewed as another format, which slang-rhi does not ask for.

So the packed route trades a CUDA bug for a Metal one, and was reverted after
being measured; Metal is back to its 451 assertions exactly.

**What every kernel that writes a texture does about it.**
`Caps::convertingStores` says whether a float4 store arrives as the texture's
format, false on CUDA, and the probe checks the capability tells the truth
either way -- on CUDA that the texel is *not* the colour. Where it is false
the kernels pack each texel into a buffer instead (`atheneaPackTexel` in
packing.slang: four 8-bit unorms in a word, four halves in two, four floats
in four) and the buffer is copied into the texture with
`copyBufferToTexture`, which asks neither backend to reinterpret anything.
The decode does it for level 0 and the mip generator for every level; the
textures keep the formats they have on Metal. That is the fix the section
above said was left: 8-bit images, their mips and their sRGB views are right
on CUDA now, and the PNG, UDIM and Hydra tests that shade through them pass.

**`RayDesc` is not a type on every target.** The compute BVH route shares its
ray with the hardware route, and `RayDesc` exists only where the target has
ray tracing: on CUDA without OptiX nvrtc answered "identifier RayDesc is
undefined" and the whole route failed to compile, which is why the splat ray
tracer's six cases and two Hydra cases failed here. `rt_integrate.slang`
carries its own `AtheneaRay` (the same four fields) and the hardware route fills
a `RayDesc` from it at the trace. The compute route now runs on CUDA: its
images match the GPU reference (p99 0, max 1 of 47500), and the comparisons
against the rasteriser skip as the device has none.

`athenea_gpu_tests` keeps the probes that establish all of the above, and they
pass on both backends bar the one that names the defect.

### Measured (NVIDIA L4, Ubuntu 24.04, debug)

`ctest --preset linux-x86_64-debug`, with engine's materials merged in:
**91 of 104 pass, 13 fail, 27 of those passes skips** -- from 47 of 83 when the
port first ran, and from 63 of 97 before the sort was fixed. One of the
thirteen is the eight-bit texture probe above, which fails here deliberately
and says why when it does; the other twelve are the texture surface write and
the two measured on their own below. Lights (M5) are merged and build here,
and their tests run: the count grew from 97 to 104 with them.
Passing outright: the prefix sum, textures, mips, the texture table and its
sRGB views, the shader cache and link constants, every loader (PLY, .splat,
SPZ, SOG, points), the lobe library, the display transform, the codeless
schemas, the hdAthenea plugin, timecode and PTP, the aofx host and its SDK
manifest hash, the whole sort, and gpe sharing the device. What is left is the
texture surface write (#46, #48, and the ray tracer and USD tests that shade
through it), the hole flags (#43, fixed after this reading), and the two
measured above.

### The second half's tools: a backend by environment, labels, the remote run

- **`ATHENEA_BACKEND=cuda|vulkan|metal|d3d12`** (an order, comma-separated)
  chooses the backend for a whole run where a caller left it to the
  platform, so one suite runs once a backend at a time; `athenea info` shows
  which was taken.
- **ctest labels.** Every case carries `gpu`; `athenea_view_tests` carries
  `display` and the gpe-backed binaries (`athenea_gpu_host_tests`,
  `athenea_aofx_tests`) `gpe`, so a machine without a window or without gpe
  excludes them by label (`ctest -LE display`) instead of reading their
  skips as passes. A case that skips for a capability the device lacks --
  rasterisation, ray queries, a uint view of an 8-bit texture -- says so in
  its skip message; that is per case, where a label is per binary.
- **`scripts/remote-test.sh [user@host] [preset] [branch]`** checks the
  branch out on the machine, builds the preset, runs ctest and prints the
  table -- passed, failed, and every skip with the reason it gave -- from
  the remote's `LastTest.log`, which it brings back to
  `build/remote/<preset>/`. A skip is not a pass, and the table says which is
  which.
- **A test's counters start from zero on purpose.** `test::uintBuffer` now
  writes zeros: the light BVH's draws test read back 1053144244 draws of a
  million on the L4, the counter having started from what the device last
  left there -- the same lesson the engine's hole flags taught, arriving in
  the tests.

### The three devices, at the end of this pass (commit 2e6e183)

Parity here means the same GPU-computed metric on each machine and each
backend, computed there; no pixels travel between them.

| | Metal (Apple M5 Pro) | Vulkan (NVIDIA L4) | CUDA (NVIDIA L4, OptiX 9, no inline rays) |
|---|---|---|---|
| Cases | 188 | 188 | 188 |
| Passed | 188 | 180 | 96 |
| Failed | 0 | 0 | 1 |
| Skipped, with reason | 0 | 8 | 91 |

**Vulkan's eight skips** are the host's and the hardware's, not the
renderer's: the Metal-only tracked buffer, `athenea view`'s window on a headless
box, the five gpe-backed cases and the aofx host (gpe has no Vulkan backend;
CUDA and Metal cover them), and openFXplayer's own bundles, which are not
built there. Everything else the suite checks -- the path tracer, materials,
lights, motion, skinning, subdivision, curves, volumes, render settings,
Kitchen_set against Storm through EGL, OCIO and the denoiser -- passes on the
L4 exactly as on Metal.

**CUDA's 91 skips** are what a device without rasterisation and without
OptiX cannot answer: 47 "no rasterisation on this device", the rest needing
rasterisation and ray queries together (motion, volumes, the lens, light
linking, the Storm oracle, the host's engine), plus points as discs, the
window, the Metal buffer and openFXplayer's bundles. Each names its
capability in its skip message.

**CUDA's one failure** is `open_pbr_surface` against MaterialX's genglsl
closures, which "Open on this backend" describes. The run before this one
had a second: the layers case composites rasterised passes and said "the
render pass drew nothing" instead of skipping; it names the capability now.
Measured with `ATHENEA_BACKEND=vulkan|cuda scripts/remote-test.sh` at commit
2bf4589, and `ctest --preset macos-arm64-debug` on the Mac.

### Vulkan on the L4

With NVIDIA's Vulkan driver (`libnvidia-gl-580-server`) installed,
`ATHENEA_BACKEND=vulkan` runs the suite on the L4 with rasterisation and ray
queries, the column most of CUDA's skips move to. The first run left four
failures and a handful of skips; each was a real difference between the
backends, fixed where it lives rather than excused.

- **A dispatch that runs too long loses the device.** The lights' chi-square
  consistency check walked every sample in one invocation, which Metal's
  watchdog tolerates and NVIDIA's Vulkan driver answers with
  `VK_ERROR_DEVICE_LOST`. It is a parallel kernel now (`lightConsistent`
  writes a flag and a gap per sample, `lightConsistentReduce` counts them), so
  no invocation is long on any backend.
- **A sphere light's `lightHit` missed at the tangent.** Deciding a hit by the
  sign of the ray-sphere discriminant is at the mercy of FMA: a direction
  `sampleLight` drew inside the cone came back a miss on Vulkan, where the
  compiler contracts differently. The hit is decided by the cone itself --
  `dot(wi, toCentre) >= cosMax`, the same test the pdf uses -- and the
  distance takes a clamped discriminant, so sampling and `lightHit` agree by
  construction.
- **An 8-bit mip level drifted by one.** Writing a float into a `UNORM8`
  texture rounds as the implementation pleases, and Vulkan's differs from
  Metal's at the half. The mip kernel quantises itself for 8-bit formats,
  `(round(saturate(v) * 255) + 0.25) / 255`, so the store has nothing to round.
- **A fixture drew its random numbers in argument order.** GCC evaluates
  function arguments right to left and Clang left to right, so the EWA
  cloud fixture built a different cloud on Linux. Draws go into named
  locals first (`SplatFixtures.h`, `test_lod.cpp`).
- **The free-running clock missed its deadline by the sleep's overrun.**
  `FrameClock` spins the last part of a wait; the margin is learnt now
  (`spinNs_`, from 0.2 ms up to 10 ms as overruns are seen) instead of fixed.
- **Storm needs OpenGL, and a headless box has no window to get it from.**
  `platform::makeHeadlessGlContextCurrent()` opens an EGL display on the GPU
  itself (`EGL_EXT_device_enumeration` through glvnd's dispatch, since the
  prototypes resolve only by `eglGetProcAddress`), a 1x1 pbuffer and a
  GL 4.5 *compatibility* context -- HgiGL issues calls a core profile rejects
  as an invalid enum. The Storm oracle runs before `CreatePlatformDefaultHgi`
  with it, and Kitchen_set against Storm passes on the L4.
- **The denoiser runs on Vulkan through OIDN's CUDA device.** OIDN has no
  Vulkan device, but its CUDA device imports external memory: the staging
  buffers are created `BufferUsage::Shared`, slang-rhi exports their memory
  as an opaque file descriptor, and `oidnNewSharedBufferFromFD` imports a
  duplicate of it (OIDN takes ownership of the one it is given). The copies
  in and out are the Metal staging path's kernels. Measured on the L4, 16
  paths against 4096: relMSE 3.65e-4 noisy, 7.16e-5 denoised with albedo
  and normal, 6.29e-5 without -- the same shape as on Metal -- and the
  `athenea:denoise` render setting changes 8748 of 27648 words at the total and
  none before it.

### The engine as an MCP server

`athenea-mcp` speaks JSON-RPC 2.0 over stdin and stdout, and `modules/mcp` holds
the protocol and the tools. The shape follows openFXplayer's server, which
this project's aofx host already speaks to: the protocol is the only thing
`Server` knows about I/O -- one message in, one reply out -- so the transport
is the app's, and every tool lives in a table whose entries carry their own
schema, which is what keeps the listing and the dispatch from disagreeing
about what exists.

**Why a server rather than a shell.** The device and the stage stay open
between calls. A CLI shelled out to once a frame opens the GPU, compiles the
kernels the frame needs and throws the lot away; here the second render of a
stage costs what a second render should cost. That is the whole reason to
drive the engine from outside.

**A render answers with the image.** The display transform runs on the device
as it does for `athenea view`, the eight-bit result is read back and packed into a
PNG (`io::writePng`, zlib's deflate and a CRC -- a container, not a codec: the
pixels were decided by a kernel), and the PNG goes back as an MCP image block
beside the timing. A model that cannot see what it rendered is guessing.

The tools: `open_stage`, `stage_tree`, `device_info`, `render` (camera of its
own or the stage's, any AOV, either technique, every path tracing and
lighting setting, an EXR beside the image), `pick`, `bounds`,
`render_products`, `convert` (a capture into a stage, on the device the
session already has), `timings` (the median of several frames, as the CLI
measures), `settings`. `.mcp.json` registers it as `athenea`.

**Driven over real stages**, not only the splat captures this engine started
from: the OpenChessSet path traced at 640x360 (64 paths a pixel, one bounce,
ACES 2.0, 25.7 s on the M5 Pro) with its MaterialX materials, its sun and its
dome; Pixar's Kitchen_set path traced from a camera of the caller's own (32
paths, 6.5 s) with its instancing and its textures, where `pick` answers
`/Kitchen_set/Arch_grp/Kitchen_1/Geom/TileFloor/pPlane357` for a pixel of the
floor and `bounds` gives the room's extent. The tools are the engine's, so
what a stage holds -- meshes, materials, lights, curves, volumes, splats --
is what they can show.

**Checked** at the protocol's level (`athenea_mcp_tests`): the handshake answers
in the client's version and in ours where the client asks for one nobody
implements; a notification is answered with silence, since the first message
a client sends is one and answering it is a violation; the listing carries a
schema for every tool; an unknown tool is a result with `isError` while an
unknown *method* is a JSON-RPC error; a message that is not JSON is -32700.
With a GPU: a stage written by the test opens, renders at 96x64, and the
answer carries a PNG (checked by its signature through the base64) and a pick
that names `/Square`.

## A note on WebGPU

Everything a frame does is Slang, and Slang emits WGSL, so the shading,
the materials, the lights and the compute BVH route would port. What would
not, without work: there are no ray queries, so the path tracer would trace
over `BvhScene` alone; buffers bind through bind groups with a small limit
per stage and no bindless textures, which the texture table and the
kernels at Metal's 31-buffer limit would both have to be reshaped for; no
fp64, no OIDN, no OpenUSD in a browser -- so the delegate stays native and
a WebGPU build would take the engine's records, not Hydra's prims.

### Not done

- **open_pbr_surface on CUDA: found, and it was the read-only cache.** Of the
  twelve materials the lobe library is checked against MaterialX's genglsl
  closures with, eleven agreed here to 2e-5 and `open_pbr_surface` did not:
  every component differed, ours 1.339 where genglsl's was 0.054, with the
  same source agreeing on Metal and on Vulkan. It was the same defect as the
  furnace's, written up below: the material read its own parameters through
  `__ldg`. The tell was in the line the test prints -- **six lobes where
  there are four** -- so a stale read had changed the lobe stack itself, not
  just a value. With the prelude fixed it builds four lobes and agrees to
  9.39e-06; with `ATHENEA_CUDA_LDG=1` it builds six again and fails exactly as it
  always did.
- A release build and any timing beyond the sort's own.
- gpe on Vulkan: gpe has no Vulkan backend, so its tests and aofx's host
  run on CUDA and Metal only.
- **Ray tracing on CUDA works now, as a pipeline.** Three things were in the
  way, and none of them was the SDK.
  - **Slang has no `RayQuery` for the CUDA target at all**, not even inside a
    ray generation program ("uses features that are not available in
    '_raygen' stage for 'cuda'"). A ray there is the classic kind:
    `TraceRay` with a payload, a miss and a closest hit. `visibility_trace`
    keeps the inline traversal and the compute entry, `visibility_trace_rays`
    holds the pipeline's three programs, and `visibility_trace_common` what
    both use -- three modules because **a module compiles every entry point
    it holds**: with the closest hit beside the compute kernel, Metal could
    no longer load the module at all and the whole visibility pass went with
    it.
  - **nvrtc wants `<optix.h>` at run time.** Slang's CUDA path compiles
    kernels as the frame asks for them, and a kernel that traces includes the
    OptiX header; Slang looks for an installed SDK (`NVIDIA-OptiX-SDK-*`) and
    the box has none -- only the headers slang-rhi fetches for itself. The
    build now tells the device where those are (`ATHENEA_OPTIX_INCLUDE_DIR`, and
    `ATHENEA_OPTIX_INCLUDE` overrides it) and the device hands nvrtc the include
    path through Slang's downstream arguments.
  - **`RayTracingScene` asked for inline rays** before it would build
    anything. The structure is the same either way.
  - **Checked** by a test that needs no rasteriser, so it runs on every
    device: the ray route's ids against the compute BVH's, over three turned
    squares at three depths. 0 of 43621 pixels differ on Metal, on Vulkan and
    on CUDA, where it reports "ray route: a ray tracing pipeline".
  - **The shadow query is ported too** (`rt_shadow_rays.slang`): the
    candidate loop becomes an **any hit** program that evaluates the
    particle, multiplies `1 - alpha` into the payload and ignores the hit so
    the traversal carries on, and ends the ray with
    `AcceptHitAndEndSearch` once what is left is under the cut -- which is
    what `query.Abort()` is inline. The ring that collapses a proxy offered
    twice lives in the payload, eight entries rather than sixteen, since a
    payload is registers. The closed forms come out identical to the inline
    route on the L4: 0.06250 through four particles of opacity 0.5, 0.50000
    through one, 5 particles taken and 5 duplicates caught, the cut stopping
    the ray after three, and the control (back faces culled) missing the
    particle a ray is born inside. Two things OptiX taught while porting: a
    payload is declared and **checked** (it refused a trace using 32 values
    where 24 were configured -- `payloadBytes` is not advisory), and
    `TraceRay` needs its culling flags passed explicitly, since a template
    parameter is not where they live.
    - The tracer builds its proxies and structures there without being able
      to draw with them: `GaussianRayTracer::prepare` and `shadowScene` work
      where `render` refuses, which is the split a shadow query needs.
  - **A mesh frame renders on CUDA now**, which it never had: the engine
    takes the pipeline route for `MeshVisibility::Rays` (it asked for inline
    rays before), so Kitchen_set lit draws at 480x270 on the L4 through
    OptiX with the lights and materials it has, on a device with no
    rasteriser at all. Before this the only thing CUDA could draw was
    splats, through the compute BVH.
  - **What is still inline only**: the path tracer, the cutout pass (a
    generated kernel) and the splat integrator. The last one needs its
    k-buffer in global memory, since a payload cannot hold 256 entries.
- **What the engine's inline rays mean for CUDA.**
  The box has OptiX after all -- `athenea info` on the L4 reports `optix 90000`,
  the driver's `libnvoptix.so.1` and the headers slang-rhi fetches itself
  (`_deps/optix_8_0-src`, `8_1`, `9_0`) -- and its caps read `ray tracing
  yes (pipeline), no (ray query), yes (AS)`. What is missing is inline
  `RayQuery` in a compute kernel, which OptiX does not offer: its traversal
  lives in ray generation and hit programs, reached through a shader binding
  table. Every ray this engine traces is inline (`visibility_trace.slang`,
  the path tracer, `rt_render.slang`, `rt_shadow.slang`), so on CUDA those
  kernels have no route and their tests skip. Earlier notes here said OptiX
  was absent; that was wrong, and the skips were right for the wrong reason.
  Giving CUDA the rays back means a second route through ray tracing
  pipelines, which is a piece of work nobody has started.

## A ray tracing launch on CUDA reads memory that is no longer current

The path tracer's OptiX route was ported and its shadow and visibility routes
measured exact, and then the closed form of a furnace stopped holding: a
sphere seen from inside, every face emitting `E` and reflecting `rho`, must
read `E (1 + rho + ... + rho^N)` to float precision, because cosine sampling
of a Lambert lobe has no variance at all. On Metal it does. On CUDA the same
frame read the series of a **different** `N` -- and not always the same one,
and not always all of it.

**What it turned out to be, measured.** The test that says it is
`tests/gpu/test_uniforms.cpp`, "a launch reads the uniform it was given, not
the one before it". A kernel of one line writes a number back; the number is
changed before every launch and read three ways -- out of a `ConstantBuffer`
(how every per-frame parameter is bound), out of a `StructuredBuffer` (how
every scene resource is bound), and out of an `RWStructuredBuffer` over the
**same memory** as the read-only one. An atomic tallies the launches that ran.
Sixty-four launches, the value alternating so that reading the one before is
never off by one:

| route | compute, CUDA | ray generation, CUDA |
|---|---|---|
| `ConstantBuffer` | 0 of 64 wrong | **31 of 64 wrong**, all of them the launch before |
| `StructuredBuffer` | 0 of 64 wrong | **32 of 64 wrong** |
| `RWStructuredBuffer`, same memory | 0 of 64 wrong | 0 of 64 wrong |
| launches that ran | 64 | 64 |

The pattern of the failures is `..X.X.X.X...`, every other launch, and the
value a failing launch reads is frozen rather than lagging. Every launch runs.
So: **inside one OptiX launch, a read-only load and a writable load of the
same address return different values**, and the read-only one is serving a
line no launch invalidated. It does not matter who wrote the memory -- the
host through `Buffer::write`, or a compute kernel dispatched immediately
before, both measured, both stale. On Metal every route is exact, and the ray
generation section skips there for want of ray tracing pipelines.

**What was ruled out, each by measurement, not by reading:**

- *Our shader.* The same body is exact on Metal inline and reaches the same
  closed forms. A counter inside the kernel (a bitmask of every `path.bounces`
  a thread saw across its sixteen samples) came back with two bits set: one
  thread, one launch, two different values of one uniform.
- *Ordering of the upload.* `CUDA_LAUNCH_BLOCKING=1`; a `cuStreamSynchronize`
  after the constant pool's `cuMemcpyHtoDAsync`; another immediately before
  `optixLaunch`. A probe printing the writes, the upload and the launch in
  order shows them in the right order, to the right device address, every
  time. None of the three changed the count.
- *The address.* Forcing the parameter block to land somewhere new on every
  launch changed nothing.
- *The pipeline's stack.* `OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW` and
  `OPTIX_EXCEPTION_FLAG_TRACE_DEPTH` with validation on report nothing, and an
  explicit `optixPipelineSetStackSize` made it worse (OptiX's own default is
  documented correct for a call tree of depth one, which is ours).
- *The payload and the binding table.* `maxRayPayloadSize` 64 and 128,
  `maxRecursion` 1 and 2, and a geometry contribution multiplier of 0 instead
  of 2 all behave the same.
- *The optimiser.* `OPTIX_COMPILE_OPTIMIZATION_LEVEL_0` cannot be measured:
  compiling the path tracer's ray generation entry that way is killed for
  memory on a 16 GB box.
- *Memory errors.* `compute-sanitizer --tool memcheck` reports 0 errors -- and
  passes the furnace, which is itself a datum: it serialises the launches.

Driver 580.173.02, OptiX 9.0 (`optix 90000`), NVIDIA L4, slang-rhi pinned at
`e17f6d7`.

**What this means for the CUDA route.** A single launch is sound -- which is
why the mesh visibility comparison (0 of 43621 pixels differing against the
compute BVH) and the shadow query's closed forms were exact, and why
`Kitchen_set` path traced to a plausible image. A *sequence* of launches is
not: whatever a ray generation entry reads through a constant or read-only
buffer may be what the launch before it read. That is every per-frame
parameter and every scene resource, so an animation, an accumulating frame or
a sweep of settings can silently render the frame before. Nothing in an image
says so; only a closed form does, which is how this was found.

**The fix: the CUDA prelude does not read through that cache.** Slang's CUDA
target emits every load of a constant buffer or a read-only buffer as
`__ldg`, the load that goes through the read-only data cache -- visible by
compiling anything with `slangc -target cuda`:

```
uint _S1 = __ldg(&globalParams_0->echo_0->count_0);
uint _S9 = __ldg((&(globalParams_0->echoWords_0)[int(0)]));
```

A writable buffer is not read that way, which is exactly why it was the one
route that stayed current. So `gpu::Device` now creates the Slang global
session itself for CUDA and appends to that target's prelude

```
#undef __ldg
#define __ldg(p) (*(p))
```

after Slang's own declaration of `__ldg`, so the declaration still parses and
every call site that follows it is an ordinary load. It gives up that cache
and nothing else. nvrtc could not be told this instead: Slang keeps **one**
`DownstreamArgs` entry per downstream compiler -- a second is silently dropped,
measured by handing it a define that would have changed every answer and
seeing none change -- and that one entry is already the OptiX include path.

**After it**, on the same box: the echo test reads 0 of 64 wrong on all three
routes in a ray generation entry, where it read 31 and 32; the furnace reads
its series to `0.00e+00` relative at every bounce count, run after run; and
`open_pbr_surface`, which had been the one material of twelve that disagreed
with genglsl on CUDA and nowhere else, agrees to 9.39e-06. **The whole CUDA
suite is 197 of 197**, where it had been 97 passed, 2 failed, 94 skipped
before the ray tracing pipelines and 2 failed after them. Metal is 197 of 197
and Vulkan on the same box is 197 of 197, so for the first time the three
backends are green together.

**What it costs**, medians of `athenea stage --frames` on the L4, Kitchen_set,
twice each way so the pairs can be read against their own spread:

| | without the cache | with it (`ATHENEA_CUDA_LDG=1`) |
|---|---|---|
| raster, 1280x720 | 20.39 ms, 21.44 ms | 22.81 ms, 21.35 ms |
| rt, 640x360, 4 paths, 2 bounces | 55.79 ms, 56.12 ms | 56.51 ms, 56.14 ms |

Nothing outside the spread of a repeat. The loads this gives up are of
parameters and pools that every thread reads alike, which the ordinary caches
hold as well.

**One trap this leaves, and it is closed.** The shader cache is keyed by
slang-rhi, which knows nothing of a prelude this process hands Slang -- so a
cache written before the fix would have been served back into a fixed build,
putting the defect quietly under it. The cache path now carries a generation
(`shaders/gen1`, and `gen1-ldg` under the escape hatch), so a changed prelude
looks elsewhere instead.

## athenea view: the path tracer's own settings in the window

Switching the viewer's Technique from Raster to Path traced was reported not
to work. What was measured, before touching anything:

- **The switch itself works.** A new test ("a renderer told another technique
  between frames draws that technique") draws raster frames and then rt
  frames on one `StageRenderer`, and the other way round, against renderers
  that drew one technique from the start: 0.00e+00 relative both ways, with
  the two techniques 7.07e-01 apart at their worst pixel as the control.
- **A viewport accumulates.** `draw` adds its paths every frame of a still
  camera -- 2 after two frames, then 3, 4, ... 12 -- whatever `pathTotal`
  says; the total only decides when the frame counts as converged (and so
  when a denoise runs).
- **Nothing failed in the window**: a session on Kitchen_set_lit logged 608
  frames and no error.
- **Kitchen_set.usd has no lights.** Path traced, it is lit by the headlight,
  which lights the first hit from the eye as the raster does: the two look
  the same by construction. `Kitchen_set_lit.usda` is the one to look at.

What the window lacked is the path tracer's settings: it left the delegate's
defaults, one path a pixel a frame and one bounce, which lights a room little
more than the raster does. The View panel now shows, under Path traced,
paths per frame, bounces (4 by default in the window), denoise, and how many
paths a pixel the frame holds; `athenea view` takes `--path-samples`,
`--path-bounces`, `--path-total` and `--denoise` as `athenea stage` does.

The report's screenshots then said the rest, on the chess set:

- **The window froze on the switch.** The first frame of a technique compiles
  its kernels for every material on the stage -- the chess set's fifteen --
  on the thread that draws the window. The first time that is long; with the
  shader cache warm, opening the chess set path traced takes 6 s in all. The
  window now puts up a notice over the last frame first ("Preparing Path
  traced: its kernels compile for this stage's materials"), and draws the
  compiling frame after it. The wait is not removed: compiling off the
  drawing thread would mean a second caller on the device.
- **Path traced still looked like the raster.** The chess set authors no
  lights, like Kitchen_set, so both were lit from the eye.
  `StageRenderer::setDefaultLights` puts a sky dome (0.6) and a sun (2.5,
  2 degrees, tilted from overhead) in the stage's **session layer**, at
  `/atheneaDefaultLights`: the file is not touched, and the lights reach the
  engine through Hydra as authored ones do. `hasLights` does not count them.
  The viewer turns them on for a stage with no lights, with a checkbox to
  turn them off, and `--no-default-lights`. A test on a floor and a wall with
  no lights: lit against unlit relMse 0.366 raster and 0.504 rt, and removed
  again the frame is the unlit one to 0.00e+00 on both.
- **A frame took 1190 ms** at 1920x1018 path traced with 4 bounces; Render
  scale is what trades that for interactivity.

## The raster sees lights at infinity along its lobes: glass and polished metal

The chess set's glass pawn heads and polished rims drew black under the
raster. Its surfaces were lit by light sampling alone, and three kinds of
lobe get nothing from that: light **through** the surface (a dome is sampled
over the hemisphere above the normal), a **delta** lobe (smooth glass, a
mirror, which answers no light sample), and a **glossy** lobe too narrow for a
dome's samples to find.

`MaterialShading` now also samples the lobe stack (up to 32 samples a pixel)
and meets the lights at infinity -- domes and distant lights -- along those
samples: through the surface, only the dome, which light sampling never
reaches there; a delta lobe, weight one; a narrow lobe, weighed against light
sampling by the power heuristic. Through the surface the lobe's sample is
followed as it leaves this surface, without the second refraction a solid adds
on its far side, and the raster traces nothing behind it: a glass object in
the raster shows the sky, not the room behind it.

**What the Metal compiler allowed, measured each time.** This kernel already
carries a note: with a copy of the lobe stack live across the shadow ray's
intersector, it writes garbage. Three arrangements hit it again, each caught
by a test that must be bit exact:

| arrangement | caught by | result |
|---|---|---|
| lobe samples drawn after the light loops | a raster frame after a path traced one, against a fresh one | differed in 3 runs of 4, worst 69 times the value |
| drawn before, kept in arrays for shadow rays after | shadows on against off over an unoccluded floor | 681-1625 of 24576 words, different each run |
| the true lobe density (`stackPdf`) asked for inside a loop that traces, before or after its shadow ray | the same | 969-1424 words |

What holds: every lobe sample is drawn, looked up and summed **before the
first shadow ray**, and nothing of the stack is asked for after. That has two
consequences, both deliberate:

- **A lobe's sample traces no shadow ray.** A reflection sees the sky whether
  or not something stands in the way, as an environment map's does.
- **Both strategies are weighed by a proxy density**, not the stack's: a Phong
  lobe about the mirror direction with the stack's own peak density, taken
  once before any shadow ray. MIS weights only have to sum to one wherever
  both strategies can sample, which a density shared by both does. The proxy
  is **zero for a broad stack** (peak density 4 or less, a Phong exponent
  near 24): there light sampling keeps the whole weight and its shadow ray, so
  a diffuse floor's dome shadows are untouched. The narrow lobes the proxy
  hands to the lobe side are where the missing shadow ray shows, as a
  reflection of sky under something that should hide it.

After it: the floor 0 words in 6 runs of 6, the technique switch 0.00e+00 in
4 of 4, and the light groups and shadow link tests, which the corrupted
arrangements had also failed, pass.

Measured, in "the raster sees a dome through glass as the path tracer does":

| material, under a uniform dome | check | result |
|---|---|---|
| `dielectric_bsdf` RT, roughness 0 | every pixel reads the dome (lossless) | worst 1.66e-05, at 1 and 4 samples |
| `standard_surface`, transmission 1 | raster against rt at 4096 paths, 2 and 32 samples | relMse 1.98e-03 then 1.24e-04: **16.0** |
| `conductor_bsdf`, roughness 0.1 | the same | relMse 7.51e-02 then 4.78e-03: **15.7** |

Sixteen times the samples dividing the squared error by sixteen is what an
unbiased estimator does. A bias would have stayed where it was. The first attempt
measured 1 against `standard_surface`, which was wrong: MaterialX layers its
specular reflection over the transmission and attenuates that by one minus
the reflection's albedo, so a pane reads F + (1 - F)^2, and the error sat at
3.8% whatever the samples. That is why the lossless case is a bare dielectric.

Not done: reflections in the raster see only the sky -- not the neighbouring
pieces, and not what should hide the sky -- so the chess set's rims read
darker than under rt.

`ATHENEA_VIEW_SWITCH_AT=N` makes `athenea view` flip its Technique selector at frame
N, as a click would, so a sequence someone reports ("opened in Raster, switched
to Path traced") runs under `--frames` and `--snapshot` instead of being
described. Run that way on the chess set, the switched frame showed the glass
heads as glass: the black heads in the report were the raster's.

## aofx at ABI 25, from aopenfx

The AOFX ABI now lives in its own repository, **github.com/jesusluque/aopenfx**,
and that is where the SDK is copied from -- `sdk/include/aofx`, verbatim, at
`73d8071` -- not openFXplayer. The ask came from `bundles openFXplayer built
load in this host` failing: openFXplayer's bundles were rebuilt against ABI 25,
and this host spoke 23.

What the two bumps are, with comments set aside (every header differed, but
eleven only in their copyright line and in dropping the names of particular
hosts):

- **24.** `EffectDesc::flowsWhen` (a node is a live source while a parameter
  says so), `RenderRequest::projectWidth`, `projectHeight` and `complaint`,
  and the verbs `Gpu::borrow` and `Gpu::importFd`, which default to an invalid
  buffer -- the contract's "use `keep`".
- **25.** `buildTag()` says which ABI of its standard library a translation
  unit was built with and what `std::string`, `std::vector`, `std::function`
  and a pointer measure. No struct moves; the tag's contents do, and the host
  compares tags, so 24 and 25 do not match.

What the host does with 24:

- **`borrow`**, where the device reads host memory in place: the pages wrapped
  as a shared-storage Metal buffer without a copy
  (`platform::newMetalBufferOverPages`: page-aligned, whole pages, refused on
  a discrete GPU), adopted by gpe's pool like any foreign buffer, given back
  by `drop`. The same key over the same pages hands back the same buffer;
  over other pages the old binding goes first. Measured: a kernel's read of a
  borrowed page gave 3, then `0xb0770` after the test wrote that into the
  page, while a `keep` of the same pages still read 3.
  `platform::pageSize`, `mapPages` and `unmapPages` join Platform for it.
- **`importFd`** answers invalid, as the reference host does: importing
  another process's exported device memory is CUDA or Vulkan external memory,
  which gpe does not reach.
- **`projectWidth`/`projectHeight`** from `EffectJob`, which gains the two
  fields; zero there means the output's bounds, since this host renders one
  image and that image is the frame.
- **`complaint`**: when `process` returns false, the effect's own sentence is
  what the error says ("'tv.mediapro.aofx.test.reporter' did not render: the
  reporter was told to fail"), and the host's own complaint from a refused
  load or run only when the effect gave none.
- **`flowsWhen`** changes nothing here: it tells a node graph that a node is a
  live source, and this host renders single jobs.

After it, `bundles openFXplayer built load in this host` loads openFXplayer's
ABI 25 bundles, `aofx_sdk_manifest` is re-recorded against the copied
headers, and the manifest's own message now names aopenfx as the reference.

## Splats shadow meshes in the path tracer

A relit cloud lit a floor it never darkened: the path tracer traces triangles,
and a splat is not one. Its shadow rays now carry the transmittance of every
cloud in the frame -- the same query the splat tracer has always had
(`rt_shadow.slang`: a particle evaluated at its peak, `1 - alpha` multiplied
in, a ring that collapses a proxy offered twice), asked at three places: a
light's next event estimate, the same inside a medium, and an emitting
triangle's.

**The tables are packed into one buffer** (`rt_shadow_packed.slang`,
`technique::SplatShadows`). The query wants four -- frames, colours, each
instance's world-to-cloud rows, its bases -- and a kernel of its own can bind
them; the path tracer cannot, because it sits at Metal's limit of 31 buffers.
So `rtShadowPack` gathers them on the device into one `float4` buffer whose
offsets travel in `PathParams`, and the two uint tables ride in float4 lanes
(`asfloat` in, `asuint` out). Two slots were still one too many, so **the
denoiser's guides became a link-constant variant too**: a frame that asks for
albedo and shading normal compiles a kernel with them, one that does not gets
the buffer back. With both asked for at once the guides win, and the engine
says so once -- a denoised frame needs them, while a cloud that shadows
nothing is a frame too bright in one place.

The engine asks for the guides only when something wants them (`denoise`, or
an albedo/shadingNormal AOV: `AovRequest::aux`), so the ordinary path traced
frame has room for the cloud.

**Measured** (`tests/technique/test_splat_shadows.cpp`):

- **The packed tables answer what the separate ones answer**: 0 of 4 rays
  differ, over a cloud of 20004 particles, with the first ray stopped almost
  entirely (0.0000) so that agreement is not two ones agreeing.
- **A cloud between a point light and a floor darkens it by what it lets
  through**: under an isotropic particle's centre a shadow ray peaks with
  power 0 and takes its opacity whole, so the frame with the cloud over the
  frame without it reads **0.5000** where the particle's opacity is 0.5, and
  **1.0000** at a pixel whose ray passes ten sigmas away. No pixel of 3185
  came out brighter with the cloud than without.

**What the dispatch cost to find.** `ComputeKernel::dispatch` takes threads
and divides by the kernel's group size; the packing passed groups, which were
divided again. A cloud of 741872 particles wants 3.7 million entries packed
and got 14592 -- the frames, and not one colour -- so every shadow ray read
opacity 0 and said the cloud was transparent, while the tests passed: a cloud
of four particles needs nine entries, and one group is 256 threads. The test
now packs twenty thousand particles, where the difference is a black floor
rather than nothing at all.

**Not done.** A cloud still does not shadow itself through this path (that is
the splat tracer's own `--splat-shadows`, unchanged), a bounce ray meets no
splats (only shadow rays do), and on a device that traces in a pipeline
(CUDA) there are no splat shadows at all: the traversal would have to be an
any hit program of its own, and the path tracer already carries two.

## mesh2splat: a mesh becomes a cloud, through the plugin interface

Electronic Arts' [mesh2splat](https://github.com/electronicarts/mesh2splat)
turns a textured mesh into 3D gaussians. It is BSD-3, the algorithm is short
and the result is exactly the primitive this engine already draws, so it was
worth having. What was decided is **where it runs**: it is an AOFX effect
(`plugins/mesh2splat`), not a module of the engine.

**Why a plugin and not a module.** The conversion is a kernel over triangles
with no state, no device to own and no scene to walk -- which is the shape an
AOFX effect has, exactly. Making it one costs a boundary and buys three
things: it runs unchanged in openFXplayer, it is the proof that this SDK
serves a *process* and not only a filter over pictures, and the engine keeps
a conversion out of its own modules, where it would have been the first thing
in `modules/` that is neither a renderer nor a loader.

**What the boundary costs.** An effect is handed pictures. A mesh is not one,
so the triangles travel as a picture of numbers -- six `float4` entries a
triangle, `(position, u)` and `(normal, v)` for each corner, in world space
(`shaders/athenea/usd/mesh_pack.slang`) -- and every map a material names travels
as rows of linear `float4` sampled out of the texture table
(`shaders/athenea/usd/texture_rows.slang`). Neither is a copy: the pictures are
allocated by the AOFX host's own image storage, which on a device with
unified memory is memory a kernel reads, and `Context::renderView` gives a
slang-rhi view of the same bytes, so the packing kernel writes where the
effect will read. What does cross back is the records, once, because
`usd::writeParticleFieldStage` takes a `io::RawSplats` -- and the numbers in
a file are the processor's business by definition.

**The port.** Their pipeline is a geometry shader and a fragment shader; this
is one compute kernel with the fragments walked (`m2sEmit`, one thread a
triangle, over the cells of its projected bounding box). Three things in it
were read out of their source rather than guessed, and getting the first two
wrong is what the first render showed:

- the Jacobian is over their **`orthogonalUvs`** -- the triplanar projection,
  normalised by the model's box -- and **not** over the texture coordinates;
- the scale is `|Ju| * sigma / resolution`, the `/ resolution` being applied
  by their *exporter* (`SceneManager.cpp`: `scaleMultiplier = gaussianStd /
  resolutionTarget`) and not by any shader. Without it every gaussian is the
  size of the whole model, which renders as a heap of slabs;
- the frame is the triangle's **longest edge**, its normal and the cross of
  the two -- not the Jacobian's columns, which is what the sizes are measured
  along. That inconsistency is theirs and it is kept: the gaussian is a disc
  in the triangle's plane however that plane was parametrised.

**What is ours, and not theirs: transmission.** Their fragment shader samples
albedo, normal and metallic-roughness. A `standard_surface` with
`transmission = 1` -- the chess set's glass -- has no channel there at all, so
it would come out as an opaque white gaussian. Here the material's
`transmission` and `transmission_color` are read from the stage and the
gaussian keeps `lerp(1, minOpacity, transmission)` of its opacity and takes
that colour. `minOpacity` is 0.25 rather than zero because `1 - transmission`
is nothing at all for glass, and a gaussian with no opacity is not a
translucent gaussian but an absent one -- `splatExport` drops it. So glass
converts to **a tint, not a lens**: what stands behind it is dimmed rather
than refracted, and that is the honest limit of the primitive.

**Reading a stage without Hydra.** `usd::MeshStage` is new, and it is the
first thing in this repository to walk `UsdShade` itself: every other route
into the engine goes through Hydra, which hands over a network. It reads the
meshes, their world transforms and, from the bound material's surface, the
`standard_surface` or `UsdPreviewSurface` inputs a gaussian can carry --
following connections through node graphs, interface inputs and a normal-map
node to whichever image node finally produces them. A value that is *computed*
rather than authored (a mix, a noise) has no answer without shading a point,
so the default stands.

**Measured** (`tests/aofx/test_mesh2splat.cpp`, the inputs written by a kernel
and the answers counted by one):

- **A unit quad at resolution 16** gives between 256 and 272 gaussians -- one
  a cell whose centre it covers, the slack being the cells the diagonal runs
  through, which both triangles claim. None off the quad's plane, none off
  its extent.
- **Every one is `sigma / resolution` wide** on both axes, `1e-7` on the
  third, and its frame's short axis is the quad's normal: 0 violations of
  each over every gaussian.
- **Colour**: against a checker, a gaussian well inside a cell takes that
  cell's colour, through the texture coordinate its record carries. 0 wrong.
- **Glass**: `transmission = 1` with `minOpacity = 0.25` leaves every gaussian
  at a quarter of its opacity and the transmission colour, exactly.
- **The chess pawn**, end to end: `athenea mesh2splat Pawn.usd` reads the two
  meshes with their bound materials (`M_Pawn_Body_B` with its black marble
  maps, `M_Pawn_Top_B` with `transmission = 1`), writes 729073 gaussians at
  resolution 512 in about two and a half seconds of a debug build, and the
  stage path traces with the viewer's default lights into a pawn whose head is
  green glass.

**Not done.** The PBR channels the conversion can write (shading normal,
metallic, roughness, the texture coordinate) have nowhere to live in a
`ParticleField3DGaussianSplat` yet, so `athenea mesh2splat` asks for the four-entry
record and leaves them out; they arrive when the per-gaussian PBR carrier
does. A mesh whose `GeomSubset`s bind different materials is converted as one
material, the one bound to the mesh. Nothing here is built or checked on
Linux yet, so CUDA is unverified.

## A texture's colour space, which a material network drops

Two renders of the same chess pawn disagreed: the mesh showed pale grey marble
where the cloud converted from it showed the black marble the asset is made of.
The cloud was right.

**What was happening.** `HdMaterialNode2::parameters` is a map of names to
values and nothing else. USD carries a texture's colour space beside the value
-- as `colorSpace` metadata on `inputs:file`, or as a UsdUVTexture's
`sourceColorSpace`, which usdImaging folds into the same place -- and all of it
is gone by the time a delegate reads `GetMaterialResource()`. So hdMtlx wrote a
document whose file inputs said nothing, `MaterialCompiler` read every texture
**raw**, and an 8-bit sRGB base colour was taken for linear: 0.2 became 0.2
where it should have been 0.033. Dark textures came out pale and washed; light
ones came out flat. Every asset with an sRGB base colour was affected, which is
every asset.

**Where it survives**, and where a Hydra 2.0 delegate should have been reading
it: the scene index. `HdMaterialNodeParameterSchema` carries `colorSpace`
beside the value, and `UsdImagingDataSourceAttributeColorSpace` is what fills
it -- including the `sourceColorSpace` consolidation. So `HdAtheneaMaterial::Sync`
reads the network for its values as before and the terminal scene index for
this one thing, and writes it onto the MaterialX input the document ended up
with (`applyColourSpaces`, matching nodes by `HdMtlxCreateNameFromPath`).

**And a second half, in the compiler.** A MaterialX shader port does not carry
the colour space its document input had: MaterialX puts one there for a colour
management system to act on, and this generator registers none -- the decode is
`TextureStore`'s, on the device, where the texture already is. So
`compileDocument` now reads the document itself before generating (every
`filename` input's `getActiveColorSpace()`, which falls back to the document's
own), keyed by the path, and a file input whose port says nothing takes its
answer from there. `auto` is passed through as a colour space of its own, which
is not MaterialX's: it means the file decides, which is what USD's
`sourceColorSpace = auto` says and what `ColourSpace::Auto` already did.

**Measured** (`tests/usd/test_usd.cpp`, "a file's colour space reaches the
decode"): one 8-bit picture read twice through a MaterialX `image` node. With
`colorSpace = "srgb_texture"` every pixel of the square shades to
(0.6039, 0.3184, 0.0332), the texture's (0.8, 0.6, 0.2) through the sRGB curve
(0.6038, 0.3185, 0.0331) -- and every pixel equals its centre exactly, so it is
one decode and not a gradient. With `lin_rec709` it shades to the values as
held, 0 mismatches at the check's own 1e-5.

**A converted cloud is relit.** The other half of the pawns' difference was
not a defect: a cloud out of `athenea mesh2splat` carries an albedo, not radiance
somebody captured, so the export now writes `primvars:athenea:splat:relight = 1`
(`ExportOptions::relight`, `athenea mesh2splat --baked` to turn it off) and the
scene's lights light it. What is left after that is the representation itself:
splat relighting is one diffuse sample per light with a normal a splat never
had, so a dark glossy marble reads darker as gaussians than as a surface with
a specular lobe. That is what the per-gaussian PBR carrier is for.

## A relit splat: both routes, and what it reflects with

Two renders of the converted pawn disagreed again, and this time the cloud was
the wrong one: black where the mesh was polished marble. Two separate defects
and one missing piece.

**The flag meant nothing when a frame was traced.** `AtheneaSplatLightingAPI` was
read in `splat_project.slang`, the tile rasteriser's, and nowhere else. A
cloud drawn by the ray tracer -- which is what a stage of nothing but splats
gets under `athenea:technique = rt` -- showed exactly what it was baked with. For a
capture that is right by luck; for a cloud converted from a mesh, whose colours
are an albedo, it is the albedo itself with no light on it: measured, a white
material came back at **1.0** and the chess set's black marble at **0.003**.
The relight now lives in `shaders/athenea/splat/splat_relight.slang` and both
routes call it, so the two cannot drift again. The traced route needed the
light table as well, which only a mesh layer used to build
(`Engine::prepareSplatLights`).

**It was relit in the wrong space.** The rasteriser sampled world-space lights
with a splat's position in its *cloud's* space. Every stage this had been
tried on had the cloud at the origin, so nothing showed. Both routes now take
the cloud to the world first, by rows the host passes, which is what
`splat_shadow.slang` already did.

**And it was Lambert.** A splat had a diffuse lobe and nothing else, which is
why the marble was black: at `0.003` of albedo, everything you see of that
material in a render of the mesh is **reflection**. So a splat now has a
specular lobe -- GGX, Smith's height-correlated visibility, Schlick's Fresnel,
the shape the surface lobes use -- and carries what it reflects with:
`GpuSplats::pbr`, one word a splat, metallic and roughness a byte each. The
conversion writes them (`primvars:athenea:splat:metallic` and `:roughness` on the
ParticleField prim), `CloudLoader` packs them, and a cloud without them -- a
capture, every file a trainer writes -- relights as it did, with metallic 0
and roughness 1.

**A dome is answered whole rather than sampled.** Every other light is
somewhere: one sample at its centre is a direction worth taking. A dome is
everywhere, and one sample of it gives a splat a spike where a surface shows a
broad sheen. So the dome's contribution is integrated in closed form: uniform
radiance over the hemisphere is `albedo * L` of diffuse and
`environmentBrdf * L` along the mirror direction of specular, with Lazarov's
analytic fit of the split-sum term.

**Measured.**

- `tests/render/test_splat_render.cpp`, "both routes relight a splat, and
  alike": the traced frame changes with the flag (max 178, 9216 pixels past
  2), and the relit pair agrees between the routes exactly as closely as the
  baked pair does (p99 1, max 1 for both) -- so relighting does not widen the
  difference the two renderers always have.
- The pawn, path traced with the viewer's default lights, over a 60 by 60
  patch of its body: the mesh averages **0.085, 0.099, 0.097** and the cloud
  **0.044, 0.046, 0.046**. Before this it was **0.0036** against the mesh's
  floor of 0.043, and black on the screen.

**What is left, and why.** The cloud is about half as bright as the mesh over
that patch. A splat gets one sample a light, no bounce, and a normal it never
had; the mesh gets a path tracer. That is the representation, not a defect,
and it is the reason the conversion carries metallic and roughness at all --
without them the difference was a factor of ten and a black pawn.

**One more thing the conversion got wrong**, found on the way: a material that
names a roughness map and no metallic map had its roughness written into all
four channels of the packed map, so metallic came out equal to roughness --
the chess set's glass, whose only map is a roughness. The defaults are laid
down first now, and each file writes only its own channel.

## What a converted surface is worth, measured against the mesh it came from

The relit pawn was still half the mesh's brightness, so it was measured
properly rather than argued about: a unit quad, one material (base colour 0.5,
roughness 0.3, metallic 0), one distant light of intensity 1 head on, rendered
as a mesh and as the cloud converted from it. The mesh reads **0.5032**. The
cloud read **0.0358**. Four things were wrong, each found by taking the next
number apart.

**The colour was put through the sRGB curve twice.** A cloud is blended in the
space it was trained in and the finished pixel is linearised
(`frame.slang`); relighting produces light, which was then linearised again.
0.5 came out 0.214, and where the curve is steepest a dark material came out
ten times too dark -- that is what made the pawn black. A relit colour now
goes into the blend encoded (`relitForBlend`), and the blend's own linearise
gives back exactly the light that was computed. The same applies the other
way: the **albedo** a relit splat uses is the stored colour decoded, which no
relight had been doing -- a capture's trained colour is sRGB too.

**A conversion's colours are light, and a cloud's are not.** So
`io::SplatEncoding::Colour::LinearLight` says which, and the export encodes
them into the cloud's space on the way in. A baked converted quad now reads
0.3508 against a coverage of 0.7016: 0.5 exactly.

**The traced route never set `linearise`.** The shade pass writes the colours
the rays read and did not know which space it was writing into.

**And the conversion never passed the material's own metallic and
roughness.** mesh2splat's shader defaults to (0.1, 0.5) and reads the rest
from a map; a material that names no map got that plastic, whatever it
authored. They are parameters of the effect now, multiplied into the map where
there is one, as glTF multiplies its factors.

**With all four, the quad reads 0.5009 against the mesh's 0.5032** -- 0.5%,
which is what is left of MIS weighting the surface's own specular sample.

## Translucency: what a gaussian can do about glass

A gaussian cannot refract. What it can do is three things, and with them glass
reads as glass:

- **Let what is behind it through**, which is its opacity: a transmitting
  material keeps `lerp(1, minOpacity, transmission)` of it (0.15 by default),
  so the collar is visible through the pawn's head.
- **Reflect**, which is the specular lobe, and which is most of what a real
  glass surface shows.
- **Send on the light that arrives from behind it.** This is the translucent
  half and it is new: the body of the material is split, `1 - transmission` of
  it facing the light and `transmission` of it facing away, so a light behind
  a glass splat lights it instead of leaving it black. Under a dome, which is
  on both sides at once, the two halves sum to the whole body -- a glass ball
  under a sky is its own colour, not a dark shell. `transmission` travels per
  splat, in the third byte of the packed word beside metallic and roughness.

Killing the diffuse outright was tried first and is wrong: the pawn's head
went grey, because what makes it mint green is exactly the light that goes in
and comes back out.

## The width of a converted gaussian, and what it covers

mesh2splat's `sigma` is 0.65 of a cell. Traced here, that leaves a converted
surface **30% transparent**: the ray meets a gaussian 0.7 of a cell from its
centre and takes what the falloff leaves. Measured on the quad, the coverage
(and the pixel, against the mesh's 0.5032):

| sigma | coverage | pixel |
|---|---|---|
| 0.65 | 0.702 | 0.351 |
| 0.85 | 0.848 | 0.425 |
| 1.0  | 0.911 | 0.456 |
| 1.2  | 0.957 | 0.479 |
| 1.5  | 0.986 | 0.493 |

So `athenea mesh2splat` defaults to **1.0** and `--sigma 0.65` asks for theirs.
It is a departure from their number and it is deliberate: their renderer
composites its own way, and a surface you can see 30% through is not what a
conversion of a solid mesh means. The rasteriser is less affected than the ray
tracer (0.94 against 0.70 at 0.65), which is the two integrations differing,
not the cloud.

## Path traced into gaussians: the bake

A relit cloud is an approximation and says so -- one sample a light, no
bounce, a normal a splat never had. The conversion can do better, because the
scene it converts is the scene the path tracer already knows: `athenea mesh2splat`
now **bakes**, and what it bakes is the path tracer's own answer at every
gaussian.

**How little it took, and why that matters.** A bake is a frame whose camera
is a list of rays. So it is not a second integrator -- the thing this
repository would least want -- but a *variant of the path tracer's kernel*
where the first hit comes from a buffer instead of from a camera
(`kBake`/`foundBaked`), and everything after that first vertex is the frame's
own path: the same lights, the same shadows, the same bounces, the same
materials. The engine reaches it through `Engine::bakePoints`, which is a
render with a bake request in it, so the scene preparation is shared by
construction rather than by copy.

**Which direction is baked.** One colour cannot be view-dependent, so the
bake stores the **cosine-weighted average over the hemisphere the surface
faces** -- the radiance a diffuse surface of the same radiosity would have.
The direction varies per sample, so the mean over the paths is the mean over
the hemisphere.

Two things were measured on the way:

- **Along the normal was wrong.** It was tried first, and under a dome it
  makes every gaussian show the same reflection: the chess set's marble came
  out as polished plastic, 0.21 against the mesh's 0.085, with its texture
  gone.
- **The directions must be stratified, not drawn.** The radiance leaving a
  glossy surface swings by orders of magnitude across the hemisphere, so
  random directions leave one gaussian in the mirror of the sun and its
  neighbour nowhere near it. That was salt and pepper over the whole model,
  and **four times the paths barely touched it** -- 64 against 256 looked the
  same, which is how it was diagnosed as not being noise. A grid with a jitter
  in each cell took it out at 64.
- **A ray starts along the direction it is seen from**, not along the normal:
  one that starts above the point and travels sideways misses its own surface
  at grazing angles, and the gaussians it misses come back black.

**And two defects it turned up in the tracer:**

- `setPrograms` did not know about the new variant, so it returned early and
  the frame ran the kernel compiled without it: the bake's first hits came
  from a camera of one pixel, and **exactly one gaussian of 729073** came back
  with anything in it.
- A bake has no camera, so it can have no headlight. Without that, a stage
  with no lights of its own bakes in a lamp standing whereever the frame's
  one-pixel camera happened to be.

**Where it stands.** The pawn bakes in 25 seconds at 64 paths a gaussian
(debug, 729073 gaussians) and the body's mean lands at 0.099/0.097/0.085
against the mesh's 0.085/0.099/0.098 -- the same light. What it does not
reproduce is the *look* from a camera: a glossy surface reads as uniformly
shiny, because the average over the hemisphere has the specular everywhere
while a view has it in one place. That is the limit of a colour with no
direction in it, and what answers it is spherical harmonics -- which the
clouds already carry to degree 3 and the bake does not write yet.

## What one colour a gaussian cannot do, measured

The baked pawn had the right light in it and still did not look like the mesh:
its marble came out smooth. Three arrangements were tried and measured over
the same 60 by 60 patch of the body, against the mesh's **min 0.024, mean
0.085, max 5.55**:

| what the gaussian carries | min | mean |
|---|---|---|
| the whole material, baked | 0.066 | 0.137 |
| its body baked, the polish added by the frame | 0.071 | 0.110 |
| the material, relit every frame | 0.004 | 0.044 |

The means can be made to match. The **minimum cannot**: every bake floors at
0.066 where the mesh reaches 0.024. That is not a bug to find -- it is what a
colour with no direction in it means. A surface looks dark from the directions
where it reflects nothing bright, and an average over the hemisphere has none
of that: it puts the sky's reflection on every gaussian from every side, which
lifts the darks and, with them, buries a texture whose contrast is smaller
than the lift.

So the split that was built -- the body baked (`bakeBody`: the diffuse, what
the material transmits, and a conductor's reflection, which is all a metal
has) and the polish added at render time from the metallic and roughness the
gaussian carries (`primvars:athenea:splat:litBody`) -- is worth having and is not
enough. It gets the texture's own light right, with its shadows and its
bounces, and leaves the reflection to a lobe that knows where the eye is. What
it does not have is the dome's *visibility* per gaussian, so the added
reflection lifts exactly the places the mesh leaves dark.

**What ends it is spherical harmonics**, which is what a trained cloud carries
and what this engine already reads to degree 3: the bake would run once per
coefficient, projecting the radiance it already samples over the hemisphere
onto the basis, and the renderer would need nothing added at all. Apple's
LiTo calls the same thing a surface light field and learns a latent for it;
the classical version is four to sixteen numbers a gaussian, and the format
has room for them.

## The bake fits harmonics: what a colour could not hold

A gaussian carries one colour and a reflection is a function of direction, so
the bake now fits **spherical harmonics** -- the same ones a trained cloud
carries and this engine already reads to degree 3. Apple's LiTo calls the
thing they stand for a surface light field; the classical version is four to
sixteen numbers a gaussian, and the format has room for them.

**How it is fitted.** The path tracer's bake already looks at every gaussian
from many directions; each sample is now weighed by a basis function where it
looked from (`shBasisValue`, beside `evaluateRest` so the two cannot drift)
and the coefficients come out of one pass, a plane each. Three things had to
be got right, and each was got wrong first:

- **Over the sphere, not the hemisphere.** Harmonics are orthonormal over the
  sphere and over nothing else. Fitted a coefficient at a time over the
  hemisphere a surface faces, each one explains the same light again and their
  sum overshoots: **ten times too bright**, measured. The integral is over the
  sphere with the far half taken as nothing -- which is what a one-sided
  surface sends there -- and drawn from the near half with a measure of 2 pi,
  so no sample is thrown away.
- **One pass, not one a coefficient.** The paths are the same for every
  coefficient and only the weight differs. Tracing them once and weighing them
  sixteen ways took the pawn from **5m43 to 32 seconds** at degree 2.
- **Which surface and from where are two questions.** The ray that finds the
  surface goes straight down the normal, which always meets the point it was
  built from; the direction the sample looks from is set afterwards, on the
  hit. Aiming the ray along that direction instead leaves it travelling beside
  the surface at grazing angles: **55000 gaussians of 729073 found nothing**,
  came back black, and speckled the model in a way no number of paths touched.

**What is baked and what is not.** The body of the material -- its diffuse, what
it transmits, a conductor's reflection -- and not its polish. Baking the polish
was tried: a reflection off a surface of roughness 0.1 is far too sharp for
sixteen coefficients, and the pawn came back **silver, 0.276 where the mesh
reads 0.085**. Adding the polish back at render time was tried too, and the
frame's own reflection has no occlusion in it: **0.135**, and the marble washed
out again. So a cloud with harmonics carries the body and nothing is added; a
cloud baked to a single colour keeps `litBody` and the frame puts the polish
back, which is the best a colour can do.

**Where it lands.** The pawn, 729073 gaussians, 1024 paths each at degree 3:
**6m45 in release**, and over the same patch of its body the cloud reads
**0.095 / 0.093 / 0.081** against the mesh's **0.085 / 0.099 / 0.098**. The
marble is marble again: its texture, its shading and its tone, from a cloud.

**What is still wrong.** The glass head. Nearly all of what it shows is a
mirror reflection of the sky and a refraction of what stands behind, and
neither is a low-frequency function of direction: degree 3 cannot hold the
first, and the second arrives through a lobe that bends every sample somewhere
else. It comes back mottled. What would answer it is either far more
coefficients than a cloud carries or the thing LiTo went after -- a learned
latent instead of a basis -- and neither is this conversion's business today.

## A traced frame of splats alone did not finish

A cloud drawn by the ray tracer -- what a stage of nothing but splats gets
under `rt` -- returned from `Engine::render` as soon as it had drawn, before
the two things every other route ends with: the sky behind the frame
(`paintDomes`) and the camera's exposure (`applyExposure`). So the same cloud
came back over the dome's grey when rasterised and over nothing when traced,
and an exposure the camera asked for was applied to one and not the other.

It was found by looking at two renders side by side and noticing the
backgrounds did not match -- which is the only way a difference like this is
ever found, and the reason the comparison was being made at all. Both routes
now end the same way; the pawn's sky reads 0.600098 in each.

## The bake's ray started a millimetre off the model

The pawn's head has a mint glass ball standing on a ring of gold. Converted
and baked, the ball came back dark and mottled and the ring came back **grey**
-- and the same conversion **without** the bake, the albedo carried and relit
every frame, came back with both right. So the conversion's colours were not
the problem; something the bake did lost them.

What it was, measured over the ring (50 by 5 pixels, both clouds and the mesh
path traced with the same camera and the same default lights):

| | R | G | B | ratio |
|---|---|---|---|---|
| mesh | 0.225 | 0.168 | 0.086 | 1 : 0.75 : 0.38 |
| carried and relit | 0.452 | 0.334 | 0.177 | 1 : 0.74 : 0.39 |
| baked, before | 0.298 | 0.289 | 0.264 | 1 : 0.97 : 0.89 |
| baked, after | 0.406 | 0.325 | 0.191 | 1 : 0.80 : 0.47 |

The baked ring was not dark. It was **the right level with no colour in it**,
which is a surface lit and then multiplied by the wrong albedo -- and the
albedo it had was the marble of the body, a few millimetres below.

**How far off the surface a bake ray starts is a fraction of the model, and of
nothing else.** It was a fraction of the scene's unit instead -- a thousandth,
floored at one -- and the chess pawn is 66 mm tall in a stage whose unit is a
metre. So every ray began **a millimetre** above its gaussian: thicker than
the gold ring, and high enough to start *inside* the glass ball above it. The
ray then came down onto whatever that offset had put it in front of, which for
the ring was the body behind it and for the ball was its own far side. The
floor was there to keep the step from being zero for a model at the origin; it
made the step enormous for every model smaller than a metre. It is now the
bounding box's diagonal times 1e-4 -- 8.8 µm on this pawn.

**And a bake is not a frame, so it has no camera hit to cache.** The tracer
shades the first vertex once a pixel and reuses it for every sample, which is
right when the sample only changes what happens *after* that vertex. A bake's
samples each look at the point from a different direction, so the cached shade
answered all 256 of them with sample zero's eye, and every harmonic above the
constant was noise about zero. `cameraHit` is now false in a bake.

**Where it lands.** The ball reads 0.198/0.304/0.266 against the mesh's
0.222/0.333/0.306, and the ring has its gold back. What is still over is a
factor of 1.8 on the ring, and it is not the bake's: the ring is five pixels
tall and the gaussians that stand on it are wider than it is, so a crop that
catches dark edges on the mesh catches gold on the cloud. The cloud that was
never baked is over by the same factor.

**And the MCP server can ask for the default lights** (`defaultLights`), which
it could not: it drew a stage exactly as the stage stood, so the chess set --
which carries no light of its own -- came back black on black, and no
comparison with what `athenea view` and `athenea stage --default-lights` show was
possible through it.

## The bake fits the harmonics, and no longer projects them

The harmonics were fitted by **projecting** the radiance over the whole sphere
with the half the surface does not face taken as nothing -- which keeps the
basis orthogonal, and is a different question from the one a bake is asking.
Three things came of it, and they are one thing:

- **Degree 0 at a fifth.** The projection gives `Y0 * 2 pi * mean` and reading
  it back gives `kSH0` times that, which is **half** the light the surface
  sends. The pawn's glass ball read 0.048/0.059/0.054 against the mesh's
  0.224/0.335/0.307.
- **A dark rim around every silhouette.** A silhouette is the surface seen
  from the equator of that half -- exactly where the function being projected
  steps from the radiance to nothing, and where a series through a step is
  worth the middle of it. Across the ball, the mesh reads a flat 0.309 and the
  cloud read **0.152 at the edge**, climbing over a hundred pixels to 0.325 at
  the centre.
- **Degree 3 further from the mesh than degree 2**, ringing about it
  (0.224, 0.271, 0.193 across the same ball). Gibbs, not noise: more paths did
  not touch it.

**What settles it** is that a Lambertian surface's radiance does not depend on
direction, so neither can the answer depend on the degree. Over the pawn's
marble body, which is diffuse once the polish is taken out of it:

| | degree 0 | degree 2 | degree 3 |
|---|---|---|---|
| projecting | 0.045 | 0.094 | 0.069 |
| fitting | 0.0749 | 0.0725 | 0.0745 |

Three answers to a question with one, against three that agree to two percent.
`tests/usd/test_usd.cpp` holds that invariant on a plane under a dome.

**The fit, and why it is small enough to solve a gaussian at a time.** It is
the normal equations `G c = b` over the half of the sphere the surface faces,
with `b` what the samples already sum to and `G_kj = integral(Y_k Y_j)` over
that half. Two bands of the **same parity are orthogonal over any half of the
sphere**: `Y(-w) = (-1)^l Y(w)`, so the two halves' integrals are equal and
each is half the sphere's, which is `delta/2` whatever the normal is. So only
the block between the even and the odd bands depends on the direction the
surface faces, and only it is worked out -- six by ten at degree 3. What is
left is `[[I/2, E], [E^T, I/2]]`, whose Schur complement is **six by six**,
symmetric and positive definite, and Cholesky ends it. Degree 0 falls out of
the same arithmetic: no odd bands, so `c = 2 b`, which is the half the
projection was missing.

**The matrix costs no rays.** It depends on the normal and on nothing else --
not the light, not the material, not a path. Estimated from the paths instead,
at 256 of them, the fit came apart: the sampling error on the matrix is the
size of the entries themselves, and the pawn came back with pixels in the
thousands. It is integrated deterministically over 32 elevations by 16
azimuths, which for a product of two basis functions is worth about four
figures, and costs 512 directions of arithmetic against 256 of ray tracing.

**Where the noise is stopped, and why not with a ridge.** Half of a sphere
does not determine sixteen harmonics equally: the combinations that are nearly
nothing on the half the surface faces are what the data cannot see, and solved
exactly the fit puts a few hundred paths` noise into exactly those. On the
pawn a few gaussians reached **65344** -- fp16's ceiling, which is what a
cloud stores them in -- and burned out as white blobs.

A ridge on every diagonal fixes it and charges every gaussian for the few that
need it. At 0.02 the whole fit shrinks by `0.5 / 0.52`, and the test's
Lambertian plane, whose light is 0.4614, came back at **0.4436** -- 3.9% low,
exactly that ratio -- with degree 2 at 9%. A **floor under Cholesky's pivot**
charges nobody: the pivot only goes small where the matrix is near singular,
which is the direction the data could not see, so flooring it bounds what that
direction contributes and leaves every well determined gaussian solved
exactly. At 0.005, against a diagonal of a half, the plane comes back at
**0.4614 exactly at degree 0** and within 1.4% and 2.0% at degrees 2 and 3 --
which is the paths' own noise -- and the pawn's brightest pixel is 8.4 where
the mesh's is 41.

**Where it lands** (1600 by 1600, 512 paths, degree 3, 256 paths a gaussian):

| patch | mesh | carried and relit | projected | fitted |
|---|---|---|---|---|
| glass ball | 0.224/0.335/0.307 | 0.208/0.328/0.299 | 0.198/0.304/0.266 | 0.173/0.265/0.237 |
| gold ring | 0.224/0.167/0.086 | 0.456/0.337/0.179 | 0.408/0.327/0.192 | 0.306/0.239/0.137 |
| marble body | 0.084/0.099/0.097 | 0.064/0.080/0.078 | 0.073/0.094/0.091 | 0.061/0.075/0.074 |

Across the ball's silhouette, where the projection swung from 0.152 to 0.325
against a mesh that is flat at 0.300 to 0.320, the fit reads 0.238 to 0.262 --
**flat**, which is what the dark rim was.

The fit is nearer the mesh than the projection on the ring and further on the
ball and the body, and what is left is **not the harmonics**: the cloud that
was never baked at all sits in the same place as the fitted one on the body
(0.080 against 0.075 against the mesh's 0.099), so the last quarter is the
conversion's own -- how wide a gaussian is against its cell, and what it
therefore covers -- and not how its colour was arrived at.

## The sky was painted where nothing was drawn, not behind what was

Every silhouette in a converted cloud wore a dark fringe. Zoomed to where a
viewer puts it, that fringe is a row of black splinters, which is how it was
noticed.

It is not the cloud and it is not the bake -- the cloud that was never baked
has it too, and so does the rasteriser. Read out of the render buffer, the
pixel just outside the pawn's collar is **0.087/0.071/0.049 with an alpha of
0.199**: a fifth of the body's colour, premultiplied, and none of the four
fifths of sky that belongs behind it.

`dome_background.slang` painted the sky where the depth buffer said nothing
had been drawn, and wrote it **over** the pixel. For a mesh that is right: a
mesh covers a pixel or it does not. A cloud's silhouette is a ramp of partial
coverage five pixels wide, and every one of those pixels has a depth, so none
of them got any sky at all.

The sky is opaque and it is behind everything, so it is composited under, by
the pixel's own coverage, and the pixel is opaque afterwards. The profile
across the collar now runs 0.600, 0.567, 0.554, 0.536, 0.488, 0.446 -- the
sky falling to the body -- where it ran 0.600, 0.087, 0.123, 0.202, 0.332,
0.446. **A fully covered pixel is bit for bit what it was**, which is what a
mesh always saw.

The light groups' planes take the sky under the same coverage, so that they
still sum to the beauty, and they read that coverage out of the beauty's own
alpha -- which the beauty pass sets to one, so the groups are dispatched
first.

**And a gaussian the bake found nothing under is not a black gaussian.** Its
coefficients come back as zeros, and zero is not "no colour": the constant
term is kept shifted to where 3DGS trains it, so a zero there decodes as
`0.5 - 0.5`. A cloud out of mesh2splat is nearly all discs, so one of those
seen edge on at a silhouette is a black splinter of its own -- thirty-nine of
them in 729073 on the pawn. It stands for nothing, so it now draws nothing.

**What was tried and was not it.** The third axis: mesh2splat writes 1e-7
there, a length in the model's own units while the other two sizes are in
cells, so on a pawn 66 mm tall it is a thousandth of a cell -- a disc a ray
meeting it side on is barely stopped by. Widened three hundred times, the dark
minimum at the edge was **just as deep** (0.027 against 0.033), so the razor
was not what made the fringe, and EA's number stands.

## The conversion's order is the mesh's, and a slot with nothing in it is kept

Two changes to the static conversion that nothing animated can do without, and
that are worth having on their own.

**The same mesh now writes the same array.** The slot a gaussian went into came
from `InterlockedAdd(counters[0], 1u, slot)`, and the shader said so in its
header: *"it keeps the atomic append, which is what makes the order of the
output nobody's business"*. Measured rather than assumed: two conversions of
the chess pawn, same options, same stage, wrote `.usdc` files that **differ
from byte 1001**. The same 729073 gaussians, in a different order.

That is fine for one still frame and impossible for a sequence, because a
gaussian is followed from one pose to the next by being the same element of the
array. So the emit is now three passes: a thread counts each triangle's covered
cells, one workgroup settles where each triangle's gaussians start, and a
thread writes each triangle's at that offset. Two conversions now give files
that are **identical byte for byte**.

The scan is one workgroup of 256 in three phases -- each thread adds up a
slice, thread zero runs the 256 slice totals into a running sum, each thread
lays its own running sum down. A serial scan in one thread was the other
option; the pawn has 42892 triangles in one of its two meshes and a character
will have more.

Two things fall out of it. The per-triangle geometry is worked out by one
function, `m2sTriangleOf`, that both the counting pass and the writing pass
call, so they cannot disagree about a single bit. And **a budget too small now
keeps the first splats in the mesh's own order** rather than whichever ones won
a race.

**A slot with nothing in it is still a slot.** The writer skipped a record that
decoded to nothing -- a scale that overflowed, an opacity under a 255th, a
position that was not a number (`splat_export.slang`) -- so the file came out
shorter than the conversion that made it. In a sequence that is worse than
untidy: a triangle that goes degenerate in one pose alone would take its
gaussians out of that frame's array and put every later gaussian out of step,
in that frame and in no other. The slot is kept now and written empty: no
opacity, no size, a position that is at least a number, and zero coefficients.
Nothing draws. The cloud's extent is folded over the splats that are there, not
over where an empty slot happens to stand, and the writer says how many it
wrote empty.

Measured on the pawn: a bake at 16 paths leaves 10 gaussians of 729073 with no
surface under them, and the file now carries 729073 where it carried 729063.

**What checks them**: `tests/aofx/test_mesh2splat.cpp` runs the conversion twice
and counts, in a kernel, the entries that differ bitwise (0 of 1024); and
`tests/usd/test_usd.cpp` writes a cloud with three records that decode to
nothing and reads back the array lengths, which are the conversion's.

## The conversion reads a pose, not a rest

`athenea mesh2splat` had a `--time`, and it reached the bake's ray tracing and
nothing else. The gaussians came from `MeshStage`, which read every attribute
at the stage's **default** time (`UsdGeomXformCache` default-constructed, every
`Get` with no time argument), so at any other instant the rays stood where the
mesh used to be while the scene they traced was somewhere else. On a static
stage nobody could see it; on an animated one it is the whole of the answer.

`MeshStageOptions` carries a `time` now, and the geometry reads and the xform
cache take it. For a stage with no animation in it **nothing changes**: an
attribute with no time samples answers with its default whatever time is asked
for, and the chess pawn converts to a file identical byte for byte to the one
it converted to before.

**A skinned mesh's `points` attribute does not animate**, though, because the
deformation belongs to the skeleton and USD resolves it through UsdSkel. A
conversion that reads `UsdGeomMesh` directly would see the rest pose at every
time. The renderer does not have this problem -- usdSkelImaging hands Hydra an
ext computation and `geom::Skinner` runs it on the device (M7) -- but the
conversion goes round Hydra on purpose: it wants none of a render index, and
the whole material side is built on `MeshStage`'s own narrowing.

So the stage is **posed** before it is read, with `UsdSkelBakeSkinning`, which
writes the posed points as time samples onto the meshes themselves. Two things
make it cheap and safe: it is baked into the **session layer**, so the file on
disk is untouched, and over `GfInterval(time, time)`, so it costs one pose and
not a range. A stage with no `SkelRoot` comes back unchanged. The reads below
then need to know nothing about skinning, and there is no second
implementation of UsdSkel in this repository.

What it leaves out, and what going round Hydra costs: instancing, velocities,
visibility and purposes. A point instancer's copies are not converted, and a
stage whose motion is authored as velocities rather than samples is read at
the sample.

**Measured** (`tests/usd/test_usd.cpp`): a square carried entirely by one joint
that slides (1.2, 0.4, 0) between time 0 and time 1 converts to a mesh whose
bounds -- folded on the device, not on the host -- are the square as authored
at time 0 and exactly that slide away at time 1.

## A gaussian carries the joints its triangle carries

The second half of reading a rigged asset: not the pose, but what moves it.
`MeshStage` now resolves each mesh's skel binding -- `UsdSkelCache` over every
`SkelRoot`, one `UsdSkelSkinningQuery` a skinnable prim -- and hands back, per
mesh, the joints each of its points is held by and how much, its
`geomBindTransform`, its skeleton's path and its skeleton's joint order.

**In the skeleton's order, not the mesh's.** A mesh may name its own subset of
the skeleton's joints with `skel:joints`, and the indices UsdSkel hands back
are then into that subset. USD keeps a mapper for it, and the mapper runs the
other way -- skeleton order to the mesh's -- so it is run over the identity to
learn each mesh index's skeleton index, and the influences are written in the
skeleton's order. One cloud then has one joint order whatever mixture of
meshes it was converted from.

**Posed, or carried, and never both.** `MeshStageOptions::skinned` reads the
**bind** pose and skips `UsdSkelBakeSkinning` entirely, because that is where
the skeleton's transforms expect to find the geometry; posing first would skin
it twice.

**The blend is the effect's.** `mesh_pack.slang` puts each triangle corner's
four heaviest influences in a second picture of exactly the same shape as the
mesh's, so the effect addresses the two alike and needs no second set of
dimensions. The effect then blends the three corners by the barycentric
coordinates of the cell the gaussian stands in, which is what interpolating
the skin means: twelve `(joint, weight)` pairs go in and four come out -- a
joint already there gains the weight, an empty slot takes it, otherwise it
displaces the lightest -- and the four are renormalised, so a gaussian is
carried entirely however its triangle was authored. It is the effect's because
it is the effect that knows where inside the triangle the gaussian stands.

The record grows to **eight entries**: four, six with the PBR channels, eight
with the joints, and nothing in between. `writeInfluences` and the
`Influences` clip are additive, so the bundle's ABI is untouched and
`aofx_sdk_manifest` never moves.

**`--skinned` forces `--no-bake`**, and says so. What the harmonics hold is
this scene's environment and its bounce -- the ground under a paw is in them --
and carrying that up with the leg when it lifts is the mistake of rotating a
lightmap. A cloud a skeleton moves carries its material and is relit every
frame, which is right by construction.

**Measured**: a square bound entirely to one joint converts to 1056 gaussians
that all name that joint with all of the weight; and through the effect, 272
gaussians of a quad whose corners all name joint 3 come out on joint 3, with
nothing in the other three slots and the weights summing to one
(`tests/aofx/test_mesh2splat.cpp`, `tests/usd/test_usd.cpp`).

**Not done here**: nothing yet deforms those gaussians. The cloud carries its
joints and the file does not write them.

## A cloud carried by a skeleton, on the device

`scene::SplatSkinner` is `geom::Skinner`'s counterpart for gaussians, and it
takes the same inputs in the same layout because they come from the same
place: the joints each gaussian is held by, the skeleton's transforms at this
instant, and the three matrices that take a point from the cloud's own space
to the skeleton's and back. One thread a gaussian
(`shaders/athenea/scene/splat_skin.slang`).

**The position is exactly the mesh's arithmetic.** The same linear blend, the
same `geomBindTransform` then `skelLocalToWorld` then `primWorldToLocal`, the
same transposition on the host -- `GfMatrix4f` is row-major with vectors on
the left and the kernel multiplies rows by a column, so the matrix goes over
turned, as a mesh's does.

**The frame is the same chain's linear part.** The two axes a gaussian spreads
along are carried by `worldToPrim . skelToWorld . (sum of w_j X_j) . geomBind`
-- which is the Jacobian of the position map, so a gaussian stretches with its
triangle instead of sliding along beside it. They are squared up again
afterwards, because a joint may shear where a rotation would not, and the
third axis is a disc's and is left alone. The two sizes in the plane come out
as the lengths of the carried axes.

**What is not touched** is everything else a gaussian carries: its opacity,
its colour, its harmonics, its PBR channels, and the word that holds its third
size. A frame therefore costs one kernel over the cloud and no re-decode of
anything -- and, because the deformed cloud is written into buffers that
outlive it, the cloud's identity never changes from one frame to the next.
That is what will let a ray tracer refit rather than rebuild.

`quaternionOfAxes` moved into `common/packing.slang`, beside the quaternion's
own packing, since the LOD merge and the skinner both want it; the LOD build
now calls the one in common rather than its own copy.

**Measured** (`tests/scene/test_loading.cpp`, 4096 gaussians on a helix, each
turned differently and each a different size): a skeleton at rest leaves every
one of them where it was, and a skeleton given a rigid turn of 1.2 radians
about a slanted axis and a slide carries every one of them rigidly -- the
position by the transform, the frame turned with it, the sizes not at all.
Nothing is read back but four counters.

The tolerance is the format's and not the arithmetic's: a frame is a
smallest-three quaternion at ten bits a component, so an axis cannot be pinned
closer than about `sqrt(2)/1023`. At rest the frame written is the frame read,
and re-encoding a value that was already a word's decode lands on that word --
except at the boundary of the rounding, where two gaussians of 4096 did.

## The file carries the rig, not the frames

`AtheneaSplatSkinningAPI` (`modules/usd/schemas/generatedSchema.usda`) is what a
cloud a skeleton moves writes beside its gaussians: `jointIndices` and
`jointWeights`, four a gaussian with `elementSize = 4`; a
`geomBindTransform`; the `skeleton` it was converted against, for provenance;
and `skinningXforms`, one transform a joint in the skeleton's order, **time
sampled**. That last is the only thing about an animated cloud that changes
from one frame to the next, and for sixty joints it is four kilobytes a frame.

The arithmetic, for a plausible character at 150k gaussians over 120 frames:

| what the file carries | size |
|---|---|
| positions, orientations, scales as time samples | **960 MB** (half), 1.2 GB (float) |
| the rig: four joints a gaussian, plus 60 joints a frame | **2.9 MB** |

And the rig is exact at *every* instant of the range, not only at the frames
somebody sampled.

**The transforms are carried on the cloud, not bound to the Skeleton.**
`UsdSkelBindingAPI` on a `ParticleField3DGaussianSplat` is not something
UsdSkel sanctions, and usdSkelImaging makes no ext computation for a prim that
is not `UsdGeomPointBased` -- so nothing would reach a renderer. Carrying the
matrices makes the file answer for itself. **Not done**, and the cost of that
choice: retargeting and clip blending, which a cloud holding baked transforms
cannot do.

`athenea mesh2splat --skinned [--range START:END[:STEP]]` gathers them with
`UsdSkelSkeletonQuery::ComputeSkinningTransforms` at each instant, the stage's
own range by default and a time code a step. Splitting the conversion's
`(joint, weight)` pairs into USD's two arrays is a rearrangement of values the
device computed, as the record unpack already is, and is said so where it
happens.

**Measured**: the rigged square converts to 272 gaussians whose file holds its
two joints at two instants, the second sliding (1.2, 0.4, 0) exactly as the
`SkelAnimation` authored it; and `tests/usd/test_usd.cpp` writes a cloud with
three joints over three time codes and reads every primvar, the element size,
the time samples and the stage's range back off the file.

**Not done here**: nothing reads them yet. `ParticleField::Sync` does not look
for the new primvars and no frame is deformed by them.

## And the engine puts the cloud where the skeleton is

The reading side, which closes it. `ParticleField::Sync` looks for the rig's
primvars beside the ones it already reads, and a time code change dirties the
one of them that is time sampled -- so the frame that follows has the
skeleton's transforms at that instant and nothing else has moved.

`Engine::carryCloud` then does three things. It pairs the file's two arrays
into the `(joint, weight)` layout every skinner here reads, and transposes the
joints' matrices the way `geom::Skinner` does, because USD puts vectors on the
left and the kernel multiplies rows by a column. It keeps the uploaded cloud
as the **bind pose** and writes the posed one into buffers of its own -- a
`GpuSplats` that shares the harmonics and the PBR channels, since the skinner
writes neither -- so what a renderer holds never changes identity between
frames. And it runs `scene::SplatSkinner` over the cloud.

**Two things the measurement turned up**, both of which would have been wrong
in any route that posed a cloud:

- The posed cloud's **extent** is not the bind pose's. A skeleton moves a
  cloud out from under its own box, and everything that culls, frames or sorts
  by it would have been looking in the wrong place. It is folded again on the
  device after the skinning.
- `Engine::bounds()` folded `entry.gpu->bounds` -- the cloud as uploaded --
  rather than the cloud as drawn. It now folds what is drawn.

**Where the spaces meet.** The conversion packs its triangles in world space,
so the gaussians stand there and not in the mesh's own space, while UsdSkel's
bind transform starts from the mesh's. The two are composed once, at
conversion, so what the file carries is the one matrix a renderer needs: the
cloud's own space into the space the joints are measured from. **Not done**: a
stage with several skinned meshes at different transforms keeps only the
first's, and a skeleton with an animated transform of its own is not folded in.

**Measured** (`tests/usd/test_usd.cpp`): a cloud of 512 gaussians whose file
says one joint carries all of them, sliding a unit in x a frame, is drawn two
time codes on with its bounds slid exactly two units in x and not a thousandth
in y or z. And end to end, the rigged square converts and draws from one fixed
camera at two instants with the gaussians where the `SkelAnimation` put them.

**Still not done**: the whole cloud is re-uploaded at every time step before
it is posed, because `ParticleField::Sync` re-reads every array on any
`DirtyPrimvar` and `CloudLoader::upload` has no in-place counterpart. The
skinning adds one cheap pass to that; what it does not do is make the frame
cheap. That is the next change, and it is on the other side of the boundary
from this one.

## What a rigged cloud actually costs

Measured, not estimated. The asset is a tube of 200 rings by 64 segments --
25472 triangles -- with forty joints down its length, two influences a vertex
blended smoothly between neighbours, and a travelling wave of rotations over
25 time codes. It stands in for the Fox until `Fox.glb` has been through a USD
exporter; what it exercises is the same path, and its weights are smooth,
which one influence a vertex would not have been (bound one ring to one joint,
the rings tear apart and the picture shows it).

`athenea mesh2splat --skinned --resolution 512`, on the Mac in release:

| | |
|---|---|
| gaussians | **184 320** |
| joints | 40 |
| the file, at 25 instants | **9.27 MB** |
| the file, at 121 instants | **9.50 MB** |
| the same range as per-frame arrays, half precision | **~713 MB** |

**Five times the frames costs 230 kilobytes.** That is the whole argument for
carrying the rig rather than the frames, and it is why the file is one number
and the sampled alternative is two orders of magnitude bigger. It is also
exact between the instants, where a sampled cloud is whatever the reader
interpolates.

(With two influences a vertex the file is 12.55 MB rather than 9.5: the
blend gives most gaussians four joints where one influence gave them one, and
the joints and weights are 32 bytes a gaussian.)

**A frame costs about 0.9 s** at 640 by 640 path traced, 184320 gaussians --
and almost none of that is the skinning. It is the cloud being re-uploaded and
its BVH rebuilt at every time step, which is the same cost a cloud with
per-frame arrays would pay and is written down above as the next change. The
skinning itself is one kernel over the gaussians.

**Not done**: the Fox. `Fox.glb` is CC0 for the model and CC-BY for the rig
and the glTF conversion (PixelMannen; tomkranis; @AsoboStudio and @scurest),
and getting it into USD needs an exporter this machine does not have -- `guc`
says plainly that "all glTF features with the exception of animation and
skinning are implemented", and Blender is not installed. The run above is what
that run would report.

## The fox

`scripts/fetch-fox.sh` fetches the Khronos glTF sample and turns it into a USD
stage with its rig: model CC0 by PixelMannen, rig and animation CC-BY 4.0 by
tomkranis, glTF conversion CC-BY 4.0 by @AsoboStudio and @scurest. It needs
Blender, which is a dependency of that asset and of nothing else -- `guc`, the
converter this project would otherwise use, says plainly that animation and
skinning are the two glTF features it does not implement.

`athenea mesh2splat --skinned --resolution 512`:

| | |
|---|---|
| triangles | 576 |
| gaussians | **204 517** |
| joints | 24, over 28 instants |
| the cloud | **13.35 MB** |

Rendered pose for pose against the mesh, path traced under the same lights,
the cloud is the fox: rounder at the silhouettes, because a gaussian rounds
off a 576-triangle model's facets, and the same animal in the same pose.

**Three things it turned up**, all of them defects nothing in this repository
had been able to see before, because every asset here until now was hand-built
or Pixar's:

- **One undeclared input lost the whole material.** Blender's USD exporter
  writes `inputs:specular` on a `UsdPreviewSurface`, which the specification
  does not have -- it has `specularColor` -- and MaterialX refuses a node
  whose interface does not match its declaration, so the fox arrived grey:
  *"Could not find a nodedef for node 'Surface'"*. Every asset out of Blender
  did. An input nobody declared is now dropped with a line saying so, beside
  the pass that already repairs mismatched input *types* for the same reason.
- **The written cloud was always Y-up.** The gaussians are in the source
  stage's world space, and `writeParticleFieldStage` said `upAxis = "Y"`
  whatever that stage said -- so a fox exported Z-up, as Blender exports, lay
  on its side. The export carries the source's up axis now.
- **The flat axis is a fraction, not a length.** mesh2splat writes `1e-7`
  there, in the model's own units, while the two sizes across the surface are
  in cells: how thin a gaussian is then depends on how big the model happens
  to be. The chess pawn is 66 mm across and traced correctly; the fox is a
  hundred units long, which makes the same `1e-7` fifteen hundred times more
  extreme, and **the ray tracer saw a ghost where the rasteriser saw a fox** --
  it integrates density along the ray rather than projecting an ellipse.

  It is `min(alongU, alongV) * flatness` now, default 0.1. Measured: the fox
  becomes solid, and the pawn does not move -- its marble body reads
  0.0645/0.0799/0.0780 where it read 0.0644/0.0798/0.0778, and its gold ring
  0.459/0.339/0.180 where it read 0.456/0.337/0.179.

  This was tried once before, as a guess at the dark fringe around every
  silhouette, and **reverted because the measurement did not support it**: the
  fringe was the sky not being composited under partial coverage, and widening
  the axis three hundred times left the dark minimum exactly as deep. The same
  change is right here for a different reason, and this time the measurement
  says so.

## Two assets out of Blender, and the two bugs they found

The BMW 1M of `bmw27` (Mike Pan, CC-BY, `download.blender.org/demo/test/`)
and the Eurasian tree sparrow's flight cycle are the first assets converted
that were not authored for this renderer. Each found one thing wrong.

### A gaussian cannot be bigger than the triangle it stands on

`mesh2splat.slang` takes a gaussian's two sizes across the surface from the
columns of the Jacobian of the triplanar map, `J = V O⁻¹`, and those columns
carry `1 / determinant`. The determinant is guarded against zero at `1e-24`,
which is the guard the projection needs -- but a **sliver seen almost edge on
by its own projection** passes that guard with a determinant of, say, `1e-20`
and the columns come back twenty orders of magnitude too long.

On the BMW, 1859 triangles did: the frame filled with white spikes metres
long radiating from the headlights and the wheel arches. They were there in
the relit cloud and in the baked one, and they were not noise -- a handful of
enormous gaussians.

The bound that is always true is local and needs nothing measured: a gaussian
belongs to a triangle, so it cannot be longer than that triangle's longest
edge.

```slang
const float longestEdge =
    sqrt(max(dot(e1, e1), max(dot(e2, e2), dot(t.c - t.b, t.c - t.b))));
const float alongU = min(length(ju) * params.sigmaX * perCell, longestEdge);
const float alongV = min(length(jv) * params.sigmaY * perCell, longestEdge);
```

It binds only where the projection had already failed: the pawn, the fox and
the car's well-shaped triangles are unchanged, since `length(ju) * sigma /
resolution` is a fraction of a cell there and a cell is far smaller than an
edge.

### A primvar's indices are read at the frame's time

`MeshStage` read texture coordinates at the frame's time and their **indices
at the default time**:

```cpp
primvar.Get(&uvs, at);
primvar.GetIndices(&uvIndices);   // no time
```

Blender writes `primvars:st:indices` as *time samples* when the export carries
animation. An attribute with samples and no default answers nothing, so the
indices came back empty, the face-varying values were then consumed
positionally, and every mesh whose st primvar was indexed sampled one texel:
the sparrow's wings came out flat mauve while its body, whose primvar is not
indexed, was correct. `GetIndices(&uvIndices, at)` is the whole fix.

### What the sparrow needed on Blender's side

Blender's USD exporter has `export_animation = False` by default, and with it
off a `SkelAnimation` is still written -- carrying `blendShapeWeights` and no
joint samples at all. That is what made the first sparrow look static.

With it on the exporter still wrote no joint samples for this rig, whose
action comes from an FBX import and keeps its curves in a slot's channelbag
rather than in `action.fcurves`. The animation is therefore written here, by
sampling the **evaluated pose** -- which is true however the pose is driven --
and solving the per-bone change of basis from the `restTransforms` the
exporter itself wrote, so whatever convention it used is the one reproduced.
The rest pose round-trips to `4.13e-6`; 363 of the 609 joints move across the
cycle.

USD joint names are the Blender bone names with every character USD will not
have in an identifier turned into an underscore, so `Spine.001_Pelvis` is
written `Spine_001_Pelvis`. The map back is by sanitising the *bone* names and
looking the joint up in that, not the other way round.

### The numbers

| | gaussians | joints | instants | file |
|---|---|---|---|---|
| BMW 1M, baked degree 2 | 6 774 631 | -- | -- | resolution 1200 |
| sparrow, skinned | 4 991 908 | 609 | 33 | resolution 1400 |

The BMW's body alone wants 2 917 105 gaussians at resolution 1200 and is
capped at the 2 097 152 a single run may write; the sparrow's feathers want
15 378 695 and take the same cap.

**Relighting a car under a softbox does not work**: an 11.7 × 7.0 area light
over a cloud whose paint is metallic 0.85 at roughness 0.19 blows out, because
a splat has no occlusion against the ones behind it unless `--splat-shadows`
pays for it. The bake is both cheaper and right for a turnaround, whose lights
do not move.

## A feather is a hole in a rectangle

The sparrow's wingtips converted into straight-edged slabs. They are not
slabs: every primary is a **flat card** whose shape lives entirely in a mask,
and the mask is the alpha channel of the map that also carries the feathers'
normals -- an atlas of cut-out silhouettes with their barbs. The material says
so plainly:

```
float inputs:opacity = 1
float inputs:opacity.connect = </root/_materials/feather/Image_Texture_001.outputs:a>
```

`materialOf` read the constant and dropped the connection. The value is
authored *and* connected, so `takeFloat` found `1` and stopped; the texture
`resolve` had already found went nowhere, because `StageMaterial` had no field
for it.

**A cut-out is not a transmission**, and it matters which one it is stored as.
Transmission is a surface you see through; a cut-out is a surface that is not
there. Carried as transmission, a feather becomes a pane of glass shaped like
a rectangle -- which is what the wings were, with the rectangle merely fainter.

Three things were needed.

- **`StageTexture` remembers the channel.** `outputs:a` is not `outputs:rgb`,
  and which one the surface took is part of the connection, not of the file.
  `resolve` reads it off the producing attribute's base name.
- **`StageMaterial::opacityMap`**, separate from `transmission`, fed to the
  plugin as an `Opacity` clip with the channel and a cut (`--opacity-cut`,
  0.5). A mask is not colour, so it is loaded raw.
- **The cut is applied in `m2sCount` as well as in `m2sEmit`.** It has to be
  in both or the slots a triangle is given do not match the slots it fills,
  and putting it in the count is what gives the budget back. `m2sCovered` and
  `m2sUvAt` are shared by the two kernels so they cannot disagree about which
  texel a cell stands on.

Measured on the sparrow, at resolution 1400:

| mesh | wanted before | wanted after |
|---|---|---|
| `sparrow_feather` | 15 378 695 | 5 898 220 |
| `sparrow_wings` | 2 410 709 | 1 375 102 |

62 % of the feathers' budget was being spent on empty rectangle, and
`sparrow_wings` now fits inside the 2 097 152 a single run may write instead
of being capped. The wing reads as separate primaries with ragged tips and
the tail as a fan.

The test feeds the checkerboard the colour test already builds as the cut-out
on its red channel -- red on an even square, zero on an odd one. Half the quad
survives (0.45 to 0.55 of it, the slack being the ring of texels an edge runs
through, which bilinear filtering carries either way), and a kernel counts the
gaussians standing well inside a square the mask cut away: zero.

## The sparrow's animation, and a skinning matrix that has to be true

Blender's USD exporter has `export_animation = False` by default, and with it
off it still writes a `SkelAnimation` -- carrying `blendShapeWeights` and no
joint samples at all. With it on, this rig still gets none: its action comes
from an FBX import and keeps its curves in a slot's channelbag rather than in
`action.fcurves`, which is where the exporter looks. So `scripts/sparrow-anim.py`
writes the animation itself, by sampling the **evaluated pose** -- true
however the pose is driven.

The first version solved a per-bone change of basis from `restTransforms` and
transported the pose through it. The rest round-tripped to `4.13e-6` and the
bird came out with its head, bill, eyes and toes mangled while its body and
wings looked right.

**The round trip proved nothing**: the basis had been *defined* so that the
rest would round-trip. What the check missed is that Blender writes
`restTransforms` from the rest pose and `bindTransforms` from the pose the
mesh was bound in, and on this bird **they are different poses**:

```
rest world chain vs bindTransforms: worst 1.989 at Toe.L.010
```

UsdSkel skins with `W(t) . bindTransform^-1`. A basis solved against the rest
cancels against the rest and not against the bind, so exactly the joints where
the two disagree -- the toes, the head, the bill -- kept a loose rotation of
up to half a turn.

Solving no basis at all is both simpler and right. Ask directly for the world
transform that makes the two skinning matrices equal:

```
W(t) := pose.matrix(t) . matrix_local^-1 . bindTransform
```

Then `W(t) . bindTransform^-1` is `pose.matrix(t) . matrix_local^-1`, which is
Blender's own, exactly, for every joint and every instant, whatever convention
either side keeps its bone frames in. The check now reads back what the file
will hold -- `%.6f` translations and rotations, `half3` scales -- and rebuilds
the chain from it:

```
f1   skinning matrix worst 4.923e-05 at Head.Lowerpeck.stretch.001_end
f31  skinning matrix worst 7.141e-05 at Head.Lowerpeck.stretch.002
f61  skinning matrix worst 4.923e-05
```

From 1.989 to 5e-5, and what is left is the quantisation.

USD joint names are the Blender bone names with every character USD will not
have in an identifier turned into an underscore, so `Spine.001_Pelvis` is
written `Spine_001_Pelvis`. The map back is by sanitising the *bone* names and
looking the joint up in that.

**And the clip that looked like a flap was the bug.** With the basis fixed the
wings barely moved, which looked like a regression. Rendering the clip in
Blender said otherwise: `air_fly_A0` is very nearly a glide there too, and the
dramatic flapping seen before was the broken transform. The bird is shown on
`air_fly_A2` (61 frames), which is a flap. The five FBX files hold 71 clips in
all -- flying, gliding, turns, hops, idles, eating -- and any of them is one
line of the script.

## One cloud, seventy-one clips

The sparrow arrives as five FBX files holding 71 animation clips between them
-- flying, gliding, turns, hops, idles, eating. Converting each would write
313 MB of identical gaussians seventy-one times: 22 GB to say the same four
million splats are doing something else.

A skinned cloud already separates the two. What a clip changes is
`primvars:athenea:splat:skinningXforms` -- 609 matrices a frame -- and that is a
primvar like any other, so a **layer over the one cloud** is the whole clip.
`scripts/sparrow-clips.py` writes three files each:

```
clips/<clip>.usda       the SkelAnimation, over the mesh stage
clips/<clip>_rig.usda   the cloud's skinningXforms
clips/<clip>_gs.usda    the stage that plays it: that rig, the cloud, lights
```

377 MB for all 71, against 22 GB. The five rigs are the same 609 joints, so
one cloud serves every file.

The matrices are what `UsdSkelSkeletonQuery::ComputeSkinningTransforms`
answers -- joint world times inverse bind, in the skeleton's own space --
which is exactly Blender's `pose.matrix . matrix_local^-1`, transposed,
because USD writes a matrix for row vectors. Checked against a conversion of
the same clip, the two agree to `1.4e-05` at every frame.

### And the cloud was playing slow

Getting that check to pass found a real one. The two agreed at frame 1 and
drifted further apart the longer the clip ran, while the matrices themselves
matched to six figures. The written cloud said:

```
endTimeCode = 61
startTimeCode = 1
```

and nothing else. `writeParticleFieldStage` authored the start and end time
codes and **never `SetTimeCodesPerSecond`**. A layer that does not say what a
time code is worth is read at 24, and USD scales every sample it holds by the
root layer's rate over that one -- so a cloud sampled at 30 and composed under
a stage at 30 was stretched by 30/24 and lagged the mesh by a quarter,
cumulatively. `MeshStage::timeCodesPerSecond()` reads the source stage's rate,
`SplatSkinning` carries it, and the export writes it. Before: `0.031` mean
error at frame 16 and `0.049` at frame 61. After: `1.4e-05` and `1.1e-05`.

## A cloud does not shadow itself, and the offset cannot be tuned to fix it

Measured, not assumed. `--splat-shadows` on the traced route changes nothing:
the images come back identical. The flag feeds the **rasteriser's** shadow
pass alone (`splat_shadow.slang`), and `rt_shade.slang` says so in the
declaration it never uses -- *"unused here: a traced splat has no shadow pass
of its own"*. So a relit splat in the path tracer takes every light whole, and
the sparrow's wings do not darken its body.

| route | `--splat-shadows` off vs on |
|---|---|
| `rt` | no difference at all |
| `raster` | 13.7 % of pixels, mean error 0.019 |

And where it does work it is both too expensive and wrong: **49.58 ms to
1810.24 ms**, 36x, and the bird is crushed to near black -- the pixel of
greatest change goes from 6.67 to 0.37.

### Why it crushes, and why no offset saves it

The kernel already knows the hazard and says so: *"without it every splat is
shadowed by the splats it is made of, and a relit capture renders black."* Its
guard is to start the ray at `shadowOffset * own`, three sigmas of **that
splat's** largest scale. Three sigmas is a photogrammetric capture's number,
where splats are fat and sparse. A converted cloud is the opposite: at
resolution 1400 a sparrow splat is 1.6e-4 across and the plumage stacks many
layers inside half a millimetre. Transmittance is a product of `(1 - alpha)`
and `sigma = 1.0` leaves a traced surface 91 % opaque, so three layers of a
splat's own neighbours take it to 0.0007.

Sweeping the offset settles it. Mean image value, against a ceiling of 0.7547
with no shadows at all:

| offset (own sigmas) | mean | of the way to no shadow |
|---|---|---|
| 1 | 0.7172 | 0 % |
| 3 (the default) | 0.7198 | 7 % |
| 12 | 0.7235 | 17 % |
| 50 | 0.7339 | 45 % |
| 200 | 0.7444 | 73 % |
| 400 | 0.7508 | 90 % |
| 800 | 0.7543 | **99 %** |

**There is no plateau anywhere.** It goes from everything black to no shadow
monotonically, and by 800 sigmas -- 0.128 units, over half the bird -- the
shadow is gone entirely. Every step that buys less self-occlusion sells the
same amount of real occlusion. The parameter cannot be tuned to a right
answer.

PTIR-GS reaches the same conclusion with an ablation: a fixed offset *"does
not generalize well across models, since the appropriate offset magnitude
depends on object scale and local Gaussian support thickness"* -- 0.025 leaves
self-occlusion, 0.05 leaks light, and their backface-aware origin
`o + 1(t_peak>0) 1(d.n>0) (t_peak+eps) d` wins on all three of their tasks
(relight 28.94 / 29.52 / **31.55** PSNR). Two independent codebases, R3DG and
IRGS, both ship the same fixed constant of 0.05 world units on a unit-scaled
object.

### What the field does, since none of it was obvious

Surveyed because the measurement said the parameter was the wrong knob.

- **Directional self-shadowing on a deforming cloud is only ever recomputed
  per frame.** Every baked representation -- R3DG's degree-3 scalar visibility
  SH, GS-IR's probe volumes, PRTGaussian's order-9 transfer -- is indexed by
  canonical position and dies when a splat moves relative to its neighbours.
  R3DG says so and re-bakes when it composes objects.
- **The learned shortcut is a dead end, on the authors' own evidence.**
  Animatable & Relightable Gaussians trained a network to predict pose-
  conditioned visibility and abandoned it: *"we do not predict light
  visibility maps during testing because we empirically find it cannot
  accurately generalize to novel poses."* At test time they trace.
- **Two schemes do survive articulation, and both factor by body part.** Lin
  et al. keep 15 part-local visibility MLPs queried in each part's own frame
  and multiplied, which is **directional** and does generalise; DNF-Avatar
  bakes a per-bone canonical SH **ambient occlusion** probe grid and
  multiplies the parts, which is a scalar and runs at 67 FPS with no tracing.
  The rationale both rest on is Sloan's 2005 bargain: *"for a single body
  part, its geometry changes are relatively small among different poses."*
- **Spherical harmonics cannot hold a shadow edge.** RGCA: *"while diffuse
  light transport is a low-pass filter that requires only 2nd or 3rd order SH,
  this is not sufficient to represent shadows"*; PRTGaussian at order 9 still
  *"struggles to reproduce the sharp edges in hard shadows"*; GUS-IR replaced
  GS-IR's SH occlusion with a binary cubemap over leaking.
- **Transmittance is order invariant**, which is why nobody sorts shadow hits;
  ours already accumulates a product. And the closed-form alternative to a
  binary hit is an `erf` line integral of the opacity along the ray,
  normalised by the maximum attainable, which is what RAGA uses and what Deep
  Gaussian Shadow Maps tabulates in an octahedral atlas.

### What this renderer should do, and has not done

Nothing here is built yet. Recorded so the measurements are not lost.

1. **The per-frame rebuild comes first**, because it is half of what a frame
   costs and it makes every other measurement meaningless until it is fixed.
   Measured on the viewer at 1280x720 with 4 269 858 skinned splats: raster
   median 134.96 ms and 31.56 ms once warm, `rt` 268.66 ms and 55.31 ms. And
   the paths are free -- 1, 16 and 64 paths a pixel all cost the same, 102.22
   / 102.18 / 100.77 ms -- so the cost is the skinning and the proxy rebuild,
   not the shading. A **static** 6.77 M cloud costs 604 ms a frame on the same
   route, which can only be the delegate rebuilding it every time.
2. **Per-part directional visibility, baked in each part's canonical frame,
   multiplied over parts.** The parts come free: a skinned cloud already
   carries its joints and weights per gaussian. Lin et al. show the
   factorisation is directional and generalises; DNF-Avatar shows a baked
   per-part table is real time. Nobody has published the combination, and the
   thing to store is an octahedral map rather than harmonics, for the
   frequency reason above. What it gives up is the rigid-part assumption and
   cross-part occlusion as a product of independent terms.
3. **A scalar visibility field for static clouds**, which is the easy case and
   covers the car and captures.
4. **The `erf` line integral in the traced route** as the quality path, which
   wants any-hit shaders and RT cores -- CUDA, not Metal, where the stochastic
   intersection-shader form is the way round the missing any-hit.

## A time change is not a new cloud

`usdVolImaging` flags a `ParticleField` as time varying, so every step of the
timeline reached `HdAtheneaParticleField::Sync` with `DirtyPoints | DirtyPrimvar`
and every array was read again. Two things followed from that, and both were
pure waste on a cloud whose geometry never changes.

**The whole cloud was decoded again, every frame.** `VtArray` is copy-on-write,
so a `Get` at a new time of an attribute that has no time samples hands back
*the same buffer*. Comparing what the arrays point at -- ten addresses and
their sizes, plus the harmonic degree -- says whether anything the decode
depends on actually changed. `skinningXforms` is deliberately not among them:
it is the one array that does change every frame, and no decode depends on it.
On the sparrow that is 609 matrices against 4 269 858 gaussians.

**And the influences were interleaved again, every frame.** `(joint, weight)`
pairs are what every skinner here reads, and building them is a CPU loop over
four values a gaussian and a buffer of eight floats each -- **137 MB** for that
bird -- none of which changes between instants. It is now rebuilt only when the
cloud was uploaded.

Measured on the viewer, 400 frames playing at 1280x720:

| | before | after |
|---|---|---|
| cloud uploads over 400 frames | 400 | **1** |
| raster, median | 134.96 ms | **56.87 ms** |
| raster, warm | 31.56 ms | **13.40 ms** |
| `rt`, median | 268.66 ms | **178.99 ms** |
| `rt`, warm | 55.31 ms | 55.05 ms |

The rasteriser is at 75 fps on 4.27 M skinned gaussians. The traced route
improves in the median and not in its warm frame, so its remaining cost is
elsewhere -- the proxies and the acceleration structure the posed cloud is
traced through, which is the next thing to look at.

## A cloud shadows itself by part: baked once, read every frame, no ray

The previous section measured that a relit cloud in the traced route casts no
shadow ray at all, that the rasteriser's does so at 36x the frame and crushes
the cloud to black, and that no offset fixes it. The survey said the only
baked visibility in the literature that survives articulation factors by body
part (Lin et al. 2024 with a network a part, DNF-Avatar with a scalar AO
grid a part), and that nobody had published the combination: **a directional
table a part**. This is that.

### What it is

A skinned cloud already carries the joints each gaussian is held by. The rig
is partitioned into parts by its largest subtrees (`partitionJoints`: the
biggest subtree anywhere that is not yet a part and has `minJoints` joints
becomes one, until there are enough -- on the sparrow, 12 parts: root, centre,
chest, pelvis, head, and each wing in three segments). For every part a field
is baked **in the pose the cloud was bound in**: a grid of probes over the
cloud's box, and at each probe an octahedral map of directions holding the
transmittance of a ray leaving the probe through that part's gaussians alone
(`splat_visibility.slang`: the part gathered into a cloud of its own, proxies
built over it by the Gaussian ray tracer, one inline ray a texel with the
same `shadowAlpha` product the shadow kernels use). Beside it, one number a
probe: the mean over its directions, for a light that has none.

At render (`splat_visibility_read.slang`) a gaussian asks every part but its
own: its posed position and the light's direction are taken back through the
part's current skinning transform and the bind transform into the space the
field was baked in, the field is read trilinearly over the probes and
bilinearly over the map, and the parts' answers multiply. One factor a light a
gaussian goes into the same `shadowFactors` both routes already read -- so the
traced route, which had no shadow pass of its own, reads them too. A dome is
cut by the ambient term instead. Zero rays a frame.

The fields are the cloud's own: `primvars:athenea:splat:visibilityParts`,
`visibilityTexels`, `visibilityAmbient` and `visibilityPartOf`, written by
`athenea visibility` and read back by `HdAtheneaParticleField`.

### What was found building it

- **A part nothing stands on read as a total shadow.** Its texels were never
  written and held zero, and zero as a transmittance is black from every
  direction: the first sparrow came out black under the sun, from the two
  parts (root, centre) that carry no gaussian. A grid of 0 marks such a part
  and the read skips it.
- **Metal binds 31 buffers, and a module's globals are laid out for every
  entry point it holds.** The read's light tables beside the bake's shadow
  tables put `iesValues` at buffer 31 and the pipeline could not be made,
  though `slangc` was happy. The read is a module of its own.
- **And the bake is a module of its own too, for CUDA.** It traces an inline
  ray, which CUDA has not, and on the L4 the whole module failed to compile
  and took the read down with it (*"unavailable features in entry point"*).
  Split off, the read compiles on every target; without the bake module a
  device says so and reads clouds baked elsewhere -- the fields travel in the
  file, which is the point of baking them.
- **A part does not shadow itself.** Its field, read at one of its own
  gaussians, is the crowd that gaussian stands in -- the same thing that made
  a shadow ray from a splat black, one level up. The read skips the part a
  gaussian is of. What that gives up is the part's own local shadowing --
  feather on feather within one wing segment -- which is the term the survey
  said only a per-frame trace or a per-bone ambient term supplies.
- **Directions cannot be coarse.** With an octahedral map of 8 x 8 a texel is
  some 22 degrees and the penumbra at a wing's height is wider than the wing;
  the test's body-under-a-wing slab could not be resolved at all. At 16 and
  32 the answer is exact outside a band of one probe cell plus the wing's
  height times tan(pitch), and moving the wing's joint aside without baking
  again lights the body: the test's second section, which is the claim that
  matters. Harmonics were never an option here (RGCA, PRTGaussian, GUS-IR).

### What was measured

Sparrow, 4 269 858 gaussians, 609 joints, 12 parts, grid 24, octave 16:

| | |
|---|---|
| bake | 22 s on the M5 Pro, 3 538 944 texels a part |
| the fields in the file | 81 MB (two f16 a word) |
| factors under a half, frame 31, sun and dome | 21.3 % |
| viewer, raster, 400 frames playing | 71.82 ms median, 9.69 ms warm (56.87 / 13.40 without) |
| viewer, `rt`, 400 frames playing | 193.88 ms median, 51.65 ms warm (178.99 / 55.05 without) |

The warm frame does not move: the read is a table lookup a part a light a
gaussian, and it is not what a frame costs.

On the 94's L4, reading the fields the Mac baked (`athenea stage --frames 8`,
1280x720, the same command as the morning's baseline):

| L4, CUDA | without | with the field |
|---|---|---|
| raster | 67.48 ms | 73.04 ms |
| `rt` | 121.48 ms | 128.77 ms |

Five to seven milliseconds a frame for 4.27 M gaussians x 2 lights x 11
parts of lookups, the same 21.3 % of factors under a half as on the Mac, and
the bake module saying plainly that it has no inline ray there. The shadow lands where the mesh's
does -- the far wing, the flank under the raised wing, the tail root -- and
follows the wings from frame to frame.

### What it is not

Weaker than the mesh's path-traced shadow, and for a stated reason: the part's
own local term is left out, the ambient term is a mean over the sphere rather
than a cosine lobe about the receiver's normal, and a part is taken as rigid.
The self-part term is the next thing to add, and the survey says it is either
a per-bone canonical AO grid (DNF-Avatar) or a short trace.

## A map is the value, not a factor on the default

The sparrow's head shone like a marble under the sun. It was not the
texture (with the sun off the spot is gone) and it was not the plumage: it
was the conversion. `materialOf` read a material's roughness as a constant and
its map beside it, and the conversion multiplies the two -- glTF's
convention, where the constant defaults to one. USD's is that a connection
replaces the value, so an input connected to a texture has no constant, and
reading none left `roughness` at the struct's own default of **0.5**: the
bird's roughness map, which runs to one, was halved everywhere. A metallic
map behind the default of 0 would have been erased outright. Connected, the
constant is now one.

And the feathers were a quarter metallic. Blender's FBX importer turned a
specular factor into `Metallic = 0.25`; the source has no metallic map for
them and a feather is not a metal. The stage that carries the bird overrides
it to 0 before the conversion.

## Opacity is coverage, not glass

The sparrow's geometry rendered silver where its photographs show a brown
bird, and the user's diagnosis was the import. Three things were measured
before anything was changed, since the material path has several places to
be wrong in.

**The textures arrive whole.** A quad that fills an orthographic frame at the
texture's own size, the texture connected to a diffuse of roughness one
(specular workflow, specular colour 0, ior 1: the albedo AOV of a black
constant reads 0.00045, of a white one 1.0), and the `albedo` AOV against the
file: the colour map (sRGB) within 0.0013 of the linearised file at every
texel, the roughness and specular maps (raw) within 0.0012, at mip 0 -- the
same against a 2x2 box of the file reads 0.4. A flat normal map is the
identity (mean 1e-5 against the same quad with no map). The `albedo` and
`shadingNormal` AOVs are now render vars a settings prim can ask for; they
are the path tracer's, so a raster product leaves them empty.

**The lobes are right.** A row of spheres per UsdPreviewSurface input --
roughness, metallic, specular workflow, ior, clearcoat, opacity in both modes,
emission, and the bird's own maps -- at `$S/ball/shaderball.py` (a script,
not an asset). Every row reads as it should but one: **a sphere of opacity 0
was a grey ball**. MaterialX's `usd_preview_surface.mtlx` makes an opacity
under one a `dielectric_bsdf` transmission at the surface's ior, so the
sphere was a lens of the sky, and a feather card's soft edge was a glass
edge: the silver. Storm draws the same sphere as nothing.

**So opacity is coverage.** USD says a surface of opacity a is there or it is
not, and a rasteriser blends by a. The engine now:

- compiles UsdPreviewSurface with a nodegraph of its own
  (`shaders/athenea/material/mx/athenea_usd_preview_surface.mtlx`, MaterialX's with
  the transmission mix removed and the surface's opacity output the opacity
  itself, or the threshold's 0 / 1); the reference variant keeps MaterialX's,
  as genglsl does. A material whose `opacity` is connected or under one is
  flagged as a cutout beside one with a threshold (`cutsOut`).
- cuts by lot. The raster route's visibility passes cut a flagged sample
  where its opacity is under a hash of the pixel (`pixelLot`; a frame is one
  image, so no seed), and the shading kernel counts a survivor whole. The
  path tracer's visibility cuts only what is fully clear, and the tracer draws
  a lot a sample at every flagged vertex, camera hit or bounce: a losing
  surface is passed along the ray from the hit, neither a bounce nor a step,
  up to 64 deep (a belly is dozens of cards, most of each clear). A ray that
  escapes this way gathers the lights at infinity along it, since the
  background pass draws only where the visibility pass found nothing. The
  margin off a passed hit is 1e-5 of the scale: at a bounce's 1e-4 the body a
  feather card lies on was skipped. The visibility passes' own cutout walk
  had the same two limits, sixteen steps and a floor of 1e-4, and the hole
  in the sparrow's belly was theirs: with every card cut the body came out
  eaten under the ray routes and whole under the raster's (a fragment
  discard has no step count). Sixty-four steps and a floor of 1e-6 now.
- reads half of each: a red card of opacity 0.5 over a white one, 256 paths,
  centre (1, 0.5, 0.5) within a binomial's three sigma; opacity 0 is cut by
  the visibility pass on both routes; opacity 1 hides the white
  (`athenea_usd_tests [coverage]`, `athenea_material_tests [coverage]`).

The sparrow's feathers, three things of the asset's, not the engine's, each
put in the layer that carries the bird (`SparrowBird.usda`): the feather
material was a quarter metallic and the eyes took their metallic from a
specular map (Blender's FBX import; a sparrow is a dielectric); and the
feather normal map's colour is premultiplied by its alpha (unpremultiplied,
its maximum is exactly 1; the clear corner is black), so every soft edge's
normal came out as (-1, -1, -1) -- the flakes. `UV2_treesparrow_normal_fixed.png`
is the same map composited over a flat normal by its alpha, its alpha kept
for the opacity. The belly's holes were taken for the asset's at first (the body's
material made emissive red showed no red behind them); they were the
sixteen-step limit above.

### Storm beside the engine

`athenea_storm_oracle_tests "[storm-side-by-side]"` draws any stage
(`ATHENEA_ORACLE_STAGE`, `_OUT`, `_CAMERA`, `_SIZE`, `_TIME`, `_TECHNIQUE`,
`_PATHS`) with Storm and the engine into two EXRs, colour only -- a
translucent material has Storm blend, and Metal refuses to blend into an
integer id target. On this Mac Storm then dies in `glGetString` (no GL
context under its UsdPreviewSurface path); on the 94, with the EGL context
`makeHeadlessGlContextCurrent` makes, it should run. Not yet run there.

### What it is not

- A shadow ray does not draw the lot: a soft edge casts a full shadow, as a
  cutout already did. The shadow walk would need the material evaluated at
  the shadow hit, which is a second call site of every material.
- A bounce that passes through and escapes gathers the lights at infinity
  with no MIS weight; the bounce's own escape did that before the pass.
- The raster route dithers: one lot a pixel, no accumulation.
- The `opacityMode` presence / transparent distinction is gone: both are
  coverage here.
- An emissive surface is a light at its whole size whatever its opacity: the
  emissive table does not draw the lot. The test's cards emit and reflect
  nothing for that reason.
- `cutsOut` reads the material's root surface node alone: the document hdMtlx
  hands over carries the libraries, whose implementation graphs wire
  `opacity` up in every surface, and reading those flagged every material.
  Flagged, an opaque MaterialX square lost 32 to 88 pixels of coverage
  against the ray route, a different count each run: the generated cutout
  raster pass is not the plain pass, and that is not looked into here.

## The feathers read their shape by a second set of coordinates

With opacity as coverage and the cutout walk deep enough, the sparrow's
geometry still had hard card edges where the shop's picture has fluff. The
feather and wing meshes carry two UV sets -- `st`, on which the UV1 atlas of
colour, roughness and specular is laid out, and `UVMap_001`, on which the UV2
normal map and its alpha are -- and Blender's export read every texture of
the material through one `st` reader. The alpha cut the cards to the wrong
texels: rectangles with a few feather-shaped holes instead of feathers. With
the two UV2 maps read by `UVMap_001` (in `SparrowBird.usda`, a second
`UsdPrimvarReader_float2` and the two `inputs:st` connections) the bird is
the one in the shop: a fluffy belly, layered coverts, barbed primaries.

The conversion had the same one reader. `mesh2splat` packs one set of
coordinates into its mesh picture -- `(position, u)`, `(normal, v)` a corner
-- and sampled every map by it. Now:

- `MeshStage` follows a texture's `inputs:st` back to its primvar reader's
  `varname` (`StageTexture::uvSet`), and where a map of the material reads by
  a primvar that is not the mesh's first, reads that primvar too (values and
  indices at the conversion's time, as `st`) and carries it as `st2`
  (`StageMesh::uv2` names it). One second set; a third map's would be a third.
- the pack kernel writes it into a picture of the mesh picture's shape,
  `(u2, v2, 0, 0)` then zeros a corner, and the command hands it to the
  effect as the `Texcoord2` clip with `albedoUv2`, `normalUv2`, `mrUv2` and
  `opacityUv2` saying which maps read by it (the SDK's headers do not change:
  a clip and four numbers, both additive). The count and the emit sample the
  cut by the same coordinates, as they must.
- the records keep the first set, as before.

`athenea_aofx_tests [mesh2splat]`: the checker cut read by a second set of
coordinates one cell down keeps the same half of the quad, and it is the
other half -- not one gaussian well inside a kept cell of the first set's
reading, more than a quarter of them inside a cut one.

And the command never asked the texture store for the cut-out map: it was
decoded only while it happened to be the normal map's file (the feathers read
their alpha off it), and with the normal map repaired into a file of its own
the cut silently went -- 15.4 M gaussians wanted for the feathers again, the
number from before there was a cut. Asked for now.

Not done: the Hydra route reads the material as authored (its readers name
their primvars), so nothing there changes; the cloud's own normal
(`--normal-map-turns`) now comes from the right texels but was not
re-measured; the sparrow's cloud and its visibility fields are to be
converted and baked again.

## A mesh past the ceiling is converted in slices, and a barb keeps its alpha

The reconverted sparrow was smooth where the mesh has fluff. Two more things
the cloud dropped, both in the conversion:

- **Half the cards were not there.** A run's output picture is capped at
  2 M gaussians (`kRunCeiling`: 192 MB a picture, and the device pool dies
  past that), and a mesh that wants more "keeps the first of them, in the
  mesh's own order" -- the feathers wanted 6.67 M and got 2.1 M, so the
  cards later in the mesh (the head's and the breast's) were missing from
  the cloud, not thinned. Now the effect says which triangle the budget cut
  into (the slots are a prefix in the mesh's order, so every triangle after
  it is cut too: `InterlockedMin` in the scan, sixth number attached as
  "splats") and takes a `firstTriangle` (a parameter, additive), and the
  command runs the mesh again from there until everything from there fits:
  the same array a single run of the whole would have written, at the same
  ceiling a run ever has open.
- **A barb was a whole gaussian or nothing.** The cut-out map was read as a
  yes or a no at 0.5; a feather's barb is a texel of alpha 0.3. The value now
  goes out as the gaussian's opacity (`params.opacity` times the mask), and
  the cut is what `--opacity-cut` says: 0.5 as it was, or lower where the
  soft edge is wanted (the sparrow: 0.15).

A triangle the budget cuts into is written by no run: the slice that starts
at it writes it whole, and `written` is where it starts -- the first try
wrote its first gaussians in one run and all of them in the next, 4 726
twice over on the feathers.

`athenea_aofx_tests [mesh2splat]`: a run from the second triangle writes the
quad's other half and says every triangle fit; a run of the whole at half its
budget says the budget cut into the first or the second, and what it wrote
plus a run from there is the whole, once each.

With every card in, the sparrow at resolution 1400 is 9.5 M gaussians (the
feathers 7.3 M in four slices, a 731 MB file), and the visibility bake ran
Metal out of memory on the M5 Pro at its largest part. The cloud is converted
at 1100 instead, with `--opacity-cut 0.15` for the soft edges.

The 1400 cloud was tried on the 94 as well: the conversion is 18 s there
(9 360 102 gaussians), the bake does not run on CUDA at all (it traces
inline rays, which that target has not: `ATHENEA_BACKEND=vulkan` does), and
through Vulkan it dies where Metal did -- the largest part, some 3.3 M
gaussians, takes the tracer's proxies to 20 GB of the L4's 23 (a BLAS of
1.17 GB is what it could not allocate), and `--parts 24` leaves that part as
it is. Baking a part in chunks of proxies is what it would take; not done.

## A shadow catcher by composition, and the film on two machines

The user asked for a ground that shows nothing but the bird's shadow. The
engine has no holdout material, and a white plane under a sky is not one (it
is lit, and the sky is not white). Four renders a frame, composited
(`scripts/film/sparrow-shadow-*.sh`, the stages `FilmGsWhite`, `FilmGsGround`,
`FilmGsBird`, `FilmGsMask` beside the assets):

- A: the bird over a white plane, `--splat-shadows`; B: the plane alone; C:
  the bird alone over the sky; M: the bird alone with no dome, whose alpha
  is its coverage (with a dome the background pass writes alpha 1 -- there
  is no other way to the coverage from the command line today).
- `final = C + (1 - M) * sky * (A / B - 1)`: where the plane is, the sky times
  the shadow's ratio; where the bird is, C; where the sky is, A = B = C. The
  plane's pass is traced too (raster, with one light sample, put its dome
  noise into the ratio as specks).

The 410 frames at 1920x1080, 64 paths: the Mac (Metal) 95-100 s a frame,
the 94 (Vulkan: the splat shadows trace inline rays, which CUDA has not, and
Mac asset paths resolved by a symlink) 30-45 s. Split 1-99 / 100-410 as they
went, the 94's frames pulled over as they finished so the Mac skipped them.
The same frame drawn by both differs by a mean of 4.9e-6 and at most 11/255
in one pixel (frame 206): the same kernels and seeds, and the seams are not
there. Two hours and a half in all. `Film_shadow.mp4`, 13.7 s.

Not done: a holdout material in the engine (one render a frame instead of
four); the four passes each reopen a 457 MB cloud, which is most of a frame's
time on the 94.

## The cloud's shadow on a mesh is weaker than the geometry's, and it is not the cloud

The shadow catcher put the cloud's shadow on a white ground, alone, where
there was nothing else to look at -- and it did not look like a bird's. It is
measured here because the film needs it and because the next thing to build
(a shadow in the raster route) has to be checked against something.

The sparrow at frame 1 of the film, over the same plane, the same sun and the
same dome, at 64 paths; the number is the mean transmittance over a 420x140
patch under the bird (1 lets all the light through):

| | mean |
|---|---|
| the mesh it was converted from, path traced | 0.587 |
| the cloud, `--splat-shadows` | 0.762 |
| the cloud, sun only (no dome) | 0.762 against the mesh's 0.587 |

And the shape: the mesh's shadow is one connected ellipse from the feet
backwards; the cloud's is three patches with lit gaps between them.

**What it is not.** Three things were ruled out by measuring, not by reading:

- **Not the soft alpha** the conversion now gives a barb: the previous cloud,
  4.2 M gaussians cut hard at 0.5, reads 0.844 in the same patch against the
  new one's 0.829. Both are equally wrong.
- **Not how flat a gaussian is**: converting the same bird at `--flatness` 0.1
  and 0.6 moves the shadow from 0.50 to 0.45 in its own scene. A disc seen
  edge-on was the obvious suspect and it is worth about a tenth.
- **Not holes in the cloud.** Rendered *from the sun's own direction*, the
  cloud's coverage is 0.1660 and the mesh's 0.1694 -- two per cent apart --
  and both silhouettes are solid. The cloud is opaque along the very
  direction its shadow says it is not.

So the same cloud, along the same direction, is opaque to a camera ray and
translucent and holed to a shadow ray. The defect is in the walk, not in the
bird. `packedShadowTransmittance` (`shaders/athenea/rt/rt_shadow_packed.slang:35`)
derives its particle and opacity the way the camera route does
(`raw / kIcoTriangles`, the same `instanceIndices`), so the remaining suspect
is which proxies the packed TLAS offers a ray at all -- the icosahedral hull
at `rtCutoff` skirts what a camera ray, walking a sorted list, does not miss.
Not confirmed.

Two consequences, both acted on rather than written down and left:

- the film's shadow is composited (the section above) and is therefore the
  cloud's shadow, deficit included;
- the raster route, which is where the bird is actually drawn, has no cloud
  shadow at all, and a transmittance map from the light will be measured
  against the 0.587 above rather than against the ray.

## A splat under a shutter: one rank-one term, not five passes

The bird's wings were frozen in every frame of the film. Mesh motion blur is
the path tracer's (the shutter in buckets, M7), the raster route draws the
frame's instant, and a cloud had no motion at all: `render::SplatInstance`
carried no second transform and `ParticleField` never sampled a primvar over
the shutter.

**What a blurred gaussian is.** The image of a gaussian smeared along a path
is that gaussian convolved with the path. For a uniform shutter the path is a
segment, whose variance along itself is `|d|^2/12`, so matching second moments
makes the blur one addition to the 2D screen covariance:

    S' = S + (d d^T)/12

a rank-one update, in the kernel that already builds the covariance
(`shaders/athenea/splat/splat_project.slang`). Convolving by a normalised measure
keeps the integral, so the peak has to come down by `sqrt(det S / det S')` --
which is the *same square root* Mip-Splatting's filter compensation already
applies there. Computing `detBefore` before both convolutions makes one square
root pay for both. Where the filter is off (`--no-antialias`) the blur still
has to be paid for, which is the `else` branch; without it a blurred splat
keeps its peak, gains light, and the wings burn.

The displacement is turned into pixels by the same `tx3`/`ty3` rows the
covariance was, so the shape and the blur live in one linearisation of the
projection rather than two, and the frustum clamp that keeps a splat behind
the camera from smearing across the frame comes with them.

*Gaussian Splatting on the Move* (ECCV 2024) does the screen-space
approximation with a per-gaussian pixel velocity but blurs by averaging five
sub-samples of the mean; the covariance form is one pass.

**Two sources of `d`, one set of numbers.** `SplatInstance` gained a
`viewStep` -- object to view, the *difference* over the shutter, so the
camera's motion and the prim's are one 3x4 -- and a `motion` buffer, what a
skeleton moved each splat, in the cloud's own space, two words of halves a
splat. They are on the instance and not on `scene::GpuSplats` because they are
this frame's, not the file's.

### Measured

`athenea_render_tests "[motion]"`, against the truth, which is the same rasteriser
run at N instants across the shutter and averaged on the device -- the ground
truth uses none of the machinery under test.

| | |
|---|---|
| step 0.10 (many splat widths) | p99 8, max 24 |
| step 0.03 | p99 2, max 9 |
| energy, step 0.03 | ratio **1.0000** |
| energy, step 0.9 | ratio 0.9030 |
| the two sources of `d`, same translation | p99 0, max 1 |
| shutter of zero against the still frame | max **0**, bit for bit |

The approximation converges as the path shortens against the splat, which is
the assertion the test actually makes; the absolute numbers are where it
starts. At a short step the light is conserved to four figures, which is the
sharp test of the square root.

**Why a long streak reads 0.903 and not 1.000, and why that is not the term's
fault.** A splat is drawn only out to `cutoffPower(alpha)`, a *smaller*
Mahalanobis radius the fainter it is -- `2 ln(255 alpha)`, so the fraction of
the gaussian drawn is `1 - 1/(255 alpha)`. The blur takes a long streak's
opacity down to a few per cent, and at 2% that rule throws away a fifth of the
tails. The blur can only lose light this way, never make it. Raising the cut
would cost every still frame in the engine to buy back energy in a smeared
one; it was not done, and the number is here so nobody reads it as a bug.

Two more things the one pass does not do, stated rather than discovered later:
a gaussian convolved with a segment is a plateau with soft ends and the
moment-matched gaussian is a peak, up to 38% too peaked once the displacement
far exceeds the splat; and a splat pivoting on a joint travels an arc, which a
straight second-moment match straightens. Sub-sampling the mean would not fix
the arc either -- only re-skinning per sub-sample would, which is the cost this
avoids.

### And then the shutter reached it

`ParticleField` now samples `primvars:athenea:splat:skinningXforms` over the
shutter the way `Mesh` samples its points, `carryCloud` uploads both poses
(609 joints twice is 78 kB a frame) and `splat_skin.slang` writes, beside the
pose it already wrote, what the shutter moved each gaussian -- the same blend
again with the second set of matrices, only for the point and not the frame,
which is why the second pose costs about a third of a kernel and not another
one. `athenea stage --shutter OPEN:CLOSE` and `athenea view --shutter` say the shutter
for a camera of the engine's own, which authors none; a stage camera's own is
taken without it.

Two things that had to be got right, and one that was wrong first:

- **Hydra brackets, it does not clip.** `SamplePrimvar` hands back the
  authored samples that *straddle* the shutter, at their own instants -- for
  poses on whole frames and a 180 degree shutter, two samples a frame apart.
  The displacement between them is therefore twice what the shutter saw, and
  the scale is `(close - open) / (t1 - t0)`. Without it the wings blur exactly
  twice as far as they should, which looks plausible, which is worse.
- **The prim works that scale out, not the engine.** `Engine`'s own
  `shutterOpen_`/`shutterClose_` are set by the render pass, which runs
  *after* the commit loop that poses the cloud: measured, the first frame
  posed with a shutter of 0..0 and blurred nothing. `ParticleField::Sync`
  knows the shutter and the two instants at once, so it computes the ratio and
  the arrays carry it.
- **`--shutter` is consulted where the camera's is read**, not set behind that
  line's back: `RenderPass` overwrites the param from the camera every frame,
  so a value pushed in earlier would last exactly until the next one. The
  engine's own camera path resamples about the asked shutter before the first
  Sync, for the same reason the stage camera's does.

`athenea_scene_tests "[skinning]"`: the skinner handed the same pose twice writes
a displacement of zero for all 4096 gaussians; handed a pure slide on every
joint it writes exactly that slide, to what two halves can hold.

The sparrow at frame 24 with `--shutter 0:0.5`: the wings smear and the head
stays sharp, which is the point -- the displacement is the skeleton's, per
gaussian, not the prim's transform.

### What it costs, and the two controls the viewer grew

Measured in `athenea view` with the window hidden (the first measurement was taken
with the window up and was contaminated: an interactive viewer competes with
itself for the device), 1280x800, 120 frames, the 5.9 M gaussian sparrow
skinned every frame:

| | median a frame |
|---|---|
| shutter 0 | **90.0 ms** |
| shutter 0.5 | **100.5 ms** |

11 per cent, and it buys the whole wing. The cost is the second pose (a second
pass over the skinning influences) and the 47 MB displacement buffer; the
projection's arithmetic is a few FMAs on values it already had.

Two controls, because judging blur means moving it:

- **A Shutter slider** under Exposure, 0 to 1 frames. It is pushed to the
  stage when it moves, not every frame, since a shutter change re-dirties
  every prim so it resamples about the new one. It applies to whichever camera
  is looking: `StageRenderer::aim` lets the asked-for shutter beat the USD
  camera's own `shutterOpen`/`shutterClose`, because a control that silently
  loses to the stage is a broken control -- which is exactly how it behaved
  first, reported as the setting doing nothing.
- **An Every frame tick** beside Play. Playing normally advances the timeline
  by the wall clock, so a stage at 30 fps that draws at ten shows one pose in
  three. Ticked, it advances one time code a drawn frame: every pose, in
  order, slower than life. `athenea view --every-frame` is the same switch.

## Seventy-one animations, and not one line about sparrows

The bird knows seventy-one things to do: fly, glide, turn left, hop, eat,
idle, jump down from a branch. Picking one of them in the viewer had to be a
dropdown, and it had to stay *generic* -- an engine that grows a Sparrow menu
is an engine with a Sparrow menu in it forever.

USD already has the mechanism: a **variant set**. A prim carries a named set,
the set carries named variants, and a selection says which one composes. What a
variant means is the asset's business -- an animation, a level of detail, a
shirt -- so the engine's part is only to list what a stage carries, select one,
and say what that changed. Three calls on `StageRenderer`:

- `variantSets()`: every set on the stage, as `{prim, name, variants,
  selected}`, in prim order.
- `setVariantSelection(prim, set, variant)`, and the same spelled the way USD
  spells a selection inside a path, `setVariantSelection("/World{clip=air_fly_A0}")`
  -- which is what `athenea view --variant` and `athenea stage --variant` take, and
  what the MCP `variants` tool takes as `select`.
- `animationRange()`: the interval the stage's samples occupy.

The selection is written to the **session layer**, as the default lights are:
the file on disk is not touched, and a viewer that has been clicked through
forty clips has changed nothing. After it, `ApplyPendingUpdates` is what turns
USD's recomposition into Hydra resyncs, and the next frame draws the new
animation.

`athenea view` shows one combo a set, above Technique, and a set with more than
twelve variants gets a filter box inside the popup -- which is what makes
seventy-one of them usable. Choosing one is applied *after* the combos are
drawn, because it rebuilds the vector they were being drawn from; the cameras
are re-read with them, since another variant may bring another camera.

### Why the timeline stopped asking the stage

A variant set of seventy-one animations, each a different length, has one root
layer, and `startTimeCode`/`endTimeCode` are *layer* metadata: the stage
declares 1..81, the longest of them, for all of them. Playing a hop of
twenty-five frames then froze the bird for fifty-six.

So `animationRange()` answers what a timeline should show: the first and last
time code any authored attribute has a sample at, narrowed to the declared
range where there is one. A stage with nothing animated gets the declared
interval, so nothing else changed; the sparrow's `land_hop_1` gets 1..25 and
`air_fly_A0` 1..33, measured through the MCP's `open_stage`, which now reports
the range and the sets. Only sample *times* are read, never a value -- a
skinned cloud's sample is 609 matrices.

### Packaging: a payload for the cloud, a reference a clip

`scripts/sparrow-clip-variants.py` writes the stage that carries them
(`SparrowClips.usda`, beside the assets -- models do not come into this
repository). Two composition strengths decide its shape, and getting them
wrong is silent:

- the cloud may **not** be sublayered. A sublayer's opinion is *local* to the
  root layer stack, and local beats a variant, so the rest pose would win over
  every clip's samples. It comes in as a **payload** instead.
- each clip is a **reference** inside its variant, and a reference is stronger
  than a payload, so the clip's `skinningXforms.timeSamples` do win over the
  payloaded cloud's default value. Referencing an `over` composes fine.

The whole file is 71 references and a light rig: USD opens the one clip that is
selected, so choosing a variant reads 3 MB, not 213. Measured, at 640x400 with
a window up, the clip stage draws in 167 ms a frame against the single-clip
stage's 259 ms, and picking another clip in the viewer is a resync, not a
reload.

`athenea_usd_tests "[variants]"`: a two-variant stage whose variants move a square
and animate it over two time codes and eight. The listing is exact (prim, set,
names, selection), the two selections draw different frames (relMSE 8.8) and
going back draws the first frame again bit for bit, `animationRange` follows
the selection 0..2 to 0..8, four malformed selections are refused, and the file
on disk never gains a selection.

### What this does not do

The cloud is **re-uploaded every time code**, on this asset, measured: twelve
frames of the timeline decode 5.9 M gaussians twelve times. The guard for it
exists and is load bearing elsewhere (`identityOf`, "A TIME CHANGE IS NOT A NEW
CLOUD"), so something in this stage's arrays does not survive a `Get` at a new
time the way `VtArray`'s copy-on-write should. It is not the variant set -- the
single-clip stage does it too. Worth finding: it is most of a frame.


## A cloud's shadow, measured from the light, with no ray and no sort

The raster route had no cloud shadow at all, and the traced one's
(`--splat-shadows`) lets too much light through -- 0.762 under the sparrow
where the path traced mesh lets 0.587, measured and recorded above. Both facts
have the same fix available: transmittance through a cloud is a **product**
over the gaussians a ray meets, and a product does not care what order its
factors arrive in. So a pass from the light needs no depth sort, no per-pixel
list and no ray at all -- which is also what lets it run on CUDA, where Slang
offers no inline `RayQuery`.

With `sigma_i(x) = -ln(1 - alpha_i(x))`, every gaussian adds its optical depth
into the texels its footprint covers and the texel ends up holding

    a0(x) = sum_i sigma_i(x)        T(x) = exp(-a0(x))

which is **exact** for a receiver behind the whole cloud -- a ground plane
under a bird -- and is the zeroth term of the Fourier expansion (Jansen &
Bavoil, I3D 2010) a receiver inside the cloud needs. One code path:
`coefficients` says how many terms, and one is the total.

`technique::SplatShadowMap` and `shaders/athenea/technique/splat_shadow_map.slang`:

- **One buffer.** A kernel that reads the map may spend only one of Metal's
  thirty-one binding slots on it, so the frames (twenty words a light slot),
  the box the casters occupy and the texels all live in one `uint` buffer, and
  a lookup reads the header out of it with `asfloat`.
- **Fixed point**, because a device need not have float atomics: 4096 to the
  unit of optical depth. The Fourier terms are signed and added as two's
  complement, which an unsigned atomic add does exactly.
- **Nothing on the CPU.** The host hands over the caster's box in its own
  space -- a bound it already holds from the upload -- and eight threads a
  caster put the corners where they land, reducing them with an atomic min and
  max over floats-as-ordered-uints. The frame kernel builds the orthographic
  frame from that box and the light's own axes: a distant light's rays are
  parallel, and a light that stands somewhere gets a frame about the direction
  from it to the box.
- **A dome has no map**, and neither has a light that casts no shadow: the
  slot is marked invalid and every lookup answers 1.

`athenea_technique_tests "[shadowmap]"`, where the closed form is a product of
opacities: four particles of alpha 0.5 stacked along the light's axis let
**0.0627** through where `(1 - alpha)^4` is 0.0625; one alone 0.5001 against
0.5; twice the stack **0.0039**, which is that squared; a point nearer the
light than anything, and one beside the map, both 1.0000; and a light whose
shadow is switched off has no map to read.

### The two receivers, and the step a Fourier series cannot hold

**A mesh** reads the map where it stands. `MaterialShading`'s light loops
multiply by `cloudTransmittance(p, k)` beside the shadow ray they already
trace, and a frame with no cloud casting anything compiles exactly the kernel
it always did -- the snippet is not in the source, so the binding is not in the
kernel. Which mattered: that kernel is **at Metal's limit of thirty-one
buffers**, and one more overflowed the pipeline. Two things were needed:

- the map reaches it as a **texture** (an array layer a coefficient a light,
  the frames in layer zero's first row), since texture slots are plentiful;
- and the read lives in `splat_shadow_read.slang`, a module that **declares no
  parameter of its own**. Importing the pass's module brought the pass's eight
  buffers with it and overflowed again, unused or not.

**A gaussian** reads it at its own depth and writes
`factors[(base + i) * lights + k]` -- the same slot a baked visibility field or
a shadow ray fills, so nothing downstream changes. That is a cloud shadowing
itself and one cloud shadowing another, on a device that cannot trace a ray.
Precedence is explicit: a baked field wins where a cloud carries one, the map
fills every other relit cloud, and the whole buffer is **cleared to 1 first**
-- it never was, and a relit cloud without a field used to read whatever the
last frame left there.

And then the sparrow turned to soot. A Fourier series truncated to two pairs
cannot hold a step, and a splat surface is a step: reconstructed at its own
depth a gaussian reads about half of its own optical depth, so every lit
gaussian came out at `exp(-3)` of itself. The fix is not more terms (a surface
needs dozens) but the one number a series is bad at and an atomic is good at:
each texel also keeps the **depth of the nearest caster in it**, reduced with
an atomic minimum. Nearer than that, by two per cent of the slab, a receiver is
on the lit side and nothing is in the way; behind it, the reconstruction
answers. The terms then do what they are for -- a receiver at an intermediate
depth, which on the bird is the ground plane inside the slab and little else
(1 term against 5: 0.9 % of the pixels differ, at most 0.025, and all of them
on the ground).

**A NaN is not a number, and fast arithmetic does not pretend otherwise.** The
ordered-key reduction started its minima at `0xFFFFFFFF`, which decodes to a
NaN, and Metal compiles with fast arithmetic: `z <= nearest + bias` came back
*true* against that NaN, so an untouched texel read as lit and a corner of the
test cloud stopped casting altogether (0.5001 became 1.0000, and a stack of
eight read as twelve). The sentinels are finite now, 1e30 and -1e30.

### What it costs

`athenea stage --frames 8`, 1280x720, the 5.9 M gaussian sparrow over a ground
plane, one distant light:

| | median a frame |
|---|---|
| no cloud shadow | **109.8 ms** |
| the map, 1 term | **112.9 ms** |
| the map, 5 terms | **116.5 ms** |

Three per cent for a floor that is no longer lit through a bird, six for the
depth-resolved version. The settings are `athenea:cloudShadows` (on by default),
`athenea:cloudShadowResolution` (1024 texels a side, per light) and
`athenea:cloudShadowTerms` (0: one where only meshes receive, five where a relit
cloud does), with `--cloud-shadow-texels` and `--cloud-shadow-terms` on
`athenea stage`.

Against the deficit this set out to fix: under the bird, with the same sun and
the same frame, the cloud's raster shadow now reads **0.0000** of the light
through where the path traced mesh reads 0.0000 and `--splat-shadows` read
0.762 -- the silhouette is the bird's, feather by feather, rather than three
blobs with holes in them.


## Depth of field in the rasteriser: the other convolution

A shutter convolves a gaussian with its path; a lens convolves it with the
disk its out-of-focus point spreads into. Same machinery, same place in
`splat_project.slang`, one addition to the screen covariance and one square
root paying the energy back:

    Sigma' = Sigma + (R^2 / 4) I        R = focal * A * |1/z - 1/focus|

R is the circle of confusion and nothing else. It falls out of the thin lens
the path tracer already samples: a ray through lens point o meets depth z at
`x = z*dir + o(1 - z/focus)`, so the image moves by `-focal * o * (1/z -
1/focus)`, and over a diaphragm of radius A that is a disk of that radius. A
uniform disk has variance R^2/4 along either axis, which is the whole of the
change. One pass; the path tracer's route samples the disk once a path and
needs many.

`athenea_render_tests "[lens]"`, against a truth built the way a path tracer builds
one -- the eye moved over the diaphragm, the picture window sheared so the
focus plane stands still, 64 samples averaged on the device:

| | one pass against 64 samples | not blurring at all |
|---|---|---|
| f/2, 12.7 px of confusion | p99 **17** | p99 113 |
| f/5.6, 4.5 px | p99 **9** | p99 65 |
| f/11, 2.3 px | p99 **9** | p99 33 |

and **what stands at the focus distance is drawn exactly as a pinhole draws
it, max 0, bit for bit** -- which is the property that lets the lens be on by
default without changing any frame that does not ask for it. Energy: 0.9677 of
the sharp frame's at f/11 and 0.8819 at f/2.8, the loss being the footprint's
own cutoff (a splat spread thin has a low peak, and everything under 1/255 of
it is not drawn) -- the same loss the shutter's long steps showed.

What it does not do is the **shape**: a disk is a plateau with a hard rim and a
moment-matched gaussian is a peak, so a bright point out of focus is a soft
blob and not a bokeh circle. That is the price of the one pass, and it is why
the test measures convergence and energy rather than pixels at a wide aperture.

Two things to know when using it:

- **A free camera's diaphragm travels as a render setting** (`athenea:lens`, a
  radius and a focus distance). `athenea view` and `athenea stage --eye` build their
  camera out of two matrices, and a lens does not fit in a matrix; the setting
  beats a stage camera's own, as the shutter's does.
- **The units are UsdGeomCamera's**: `focalLength` is in tenths of a scene
  unit, so a 35 mm lens is 3.5 units and f/8 is a diaphragm 0.22 units across.
  On a stage in metres -- the sparrow is 0.1 across -- that is a lens wider
  than the bird, and the f-numbers that look like photography are f/60 to
  f/120. The path tracer has always read it the same way; the engine does not
  yet consult `metersPerUnit`.

`athenea stage --fstop N --focus D`, `athenea view --fstop --focus` and two sliders in
the panel, and `fStop`/`focus` on the MCP's render tool.

### Where the shadow's density is, and where its colour is not

The map's optical depth is what the cloud's own opacity says -- `-ln(1 - alpha)`
a gaussian a texel -- so there was nothing to turn. `athenea:cloudShadowDensity`
multiplies it as it is accumulated (one multiply in the splat kernel, no read
changed): 1 is the measurement, less lets light through a solid thing, more
darkens a thin one. It is a compositor's knob and it is honest about being one.

**Colour there is none**: transmittance is a scalar a texel, so a cloud stops
light without tinting it. Three channels of optical depth would give a stained
glass or a leaf its coloured shadow, at three times the memory and the atomics
of the total, and a scattering model to say what a splat's albedo does to the
light it does not stop. Not done.

The settings a host offers are the delegate's descriptors, which now name the
four cloud shadow settings and `athenea:splatShadows` beside the path tracer's --
so a render settings prim, usdview's panel, the MCP's `render` and `settings`
tools and `athenea view`'s panel all reach the same switches.


## A binding nobody declared is still a binding: USDZ arrives grey

A Datsun downloaded from Sketchfab as `.usdz` drew as a uniformly grey car
while usdview showed it painted. Everything about the material path was
innocent, and each suspect was ruled out by measurement:

- the MaterialX document was right. `ATHENEA_MTLX_DUMP` (new, the same switch
  `ATHENEA_SHADER_DUMP` is) wrote what the delegate hands MaterialX, and the
  paint's `diffuseColor` was there, `fallback` typed `color4` as the nodedef
  declares, `normal` a `vector3`. The "Input 'fallback' doesn't match
  declaration" chatter comes from hdMtlx while it *builds* the document,
  before `matchDeclaredTypes` has had its say.
- all fourteen materials compiled and took a row (`athenea -v` prints them).
- the textures were not the problem either, though they do live inside the
  package (`file.usdz[0/paint.jpg]`), which Hio reads.

What was wrong is upstream of all of it: **the meshes reached Hydra with no
material binding at all**. UsdShade has required `MaterialBindingAPI` to be
applied since 21.11 and the scene index path enforces it; Sketchfab's usdz
exporter authors `rel material:binding` and applies nothing. usdview still
has the legacy delegate's forgiving path, which is why it looked fine there
and grey here.

The minimal repro is four lines: two identical meshes binding the same
material, one with `prepend apiSchemas = ["MaterialBindingAPI"]` and one
without. The first binds; the second reports `<none>`.

So `StageRenderer::open` sweeps the stage once and applies the schema to every
prim that already authors a `material:binding`, **in the session layer** -- the
file is not touched, exactly as the default lights are not written to it. The
Datsun: 70 prims, and a car with its bronze paint, its glass and its tyres.

Two diagnostics earned their place on the way, and stayed: `ATHENEA_MTLX_DUMP`,
and `athenea -v` printing what material each mesh bound and which row it got.

And one bug of my own, worth writing down because it is the kind that hides:
`std::any_of(prim.GetAuthoredProperties().begin(), prim.GetAuthoredProperties().end(), ...)`
compares iterators into **two different temporaries**. It answered correctly on
a four-prim test file and silently did nothing on the real one.


## Glass that is not there, and textures under another name

The Mustang's windows were open holes in our frames and glass in Cycles', with
the same camera and the same sky (measured: the flat dome reads 0.5992 there
and 0.6001 here, so the light was not the difference). The material said so:

    float inputs:opacity = 0

Blender's Principled has `Transmission Weight = 1` on that material -- real
glass, IOR 0.95 -- and **UsdPreviewSurface has no transmission**, so the
exporter says the only thing it can: opacity 0, which in UsdPreviewSurface
means *this surface is not there*. We drew exactly that, correctly and
uselessly.

The fix is not to reinterpret opacity but to stop losing the transmission:
Blender writes a **MaterialX network** beside the preview surface
(`generate_materialx_network`), the delegate already asks for `mtlx` first
(`GetMaterialRenderContexts`), and MaterialX's `ND_open_pbr_surface` carries
transmission, IOR and the rest. The conversion script asks for both networks
now, and the windows are glass.

### What the texture store learned

With the MaterialX network came its texture paths, and they are **not**
resolved the way UsdPreviewSurface's are: `./textures/paint.jpg`, exactly as
the asset wrote it, with nothing in a Hydra material network to say which
folder that is. Half the Mustang's maps loaded and half did not.

`TextureStore` now tries a path four ways, in this order, and says at debug
level which one answered:

1. **as given** -- an absolute path, a package path (`shot.usdz[0/paint.jpg]`,
   which Hio reads), or one already resolved;
2. **the asset resolver's** answer, for a search-path asset;
3. **anchored at the stage's folder** (`Engine::setAssetSearchPath`, set when
   the stage opens), which is what a relative path means;
4. **by a name with the packaging taken out** -- case, spaces, punctuation and
   extension -- against the files beside the stage, two directories deep.

The fourth is the one that feels like magic and is not: a packager renames
`carpet_Base Color.jpg` to `carpet_BaseColor.jpeg` and writes the network
against one name and the file under the other. Blender's exporter does it in
the very asset this was found on -- the copy is named after the image
datablock and the path after the source file. It runs only when every honest
path has failed.

What is still not right: our glass is noisier than Cycles' at equal samples
(a transmission lobe costs paths), and six of the Mustang's textures are the
ones Sketchfab packs by channel (`carpet_Metallic-carpet_Roughness@channels=B.png`),
which no name can match.


## The spec is the reference, and it is a living one

USD is not a fixed target: it changes release to release, and two of this
day's bugs were the engine having drifted from what the current spec says.
So the rule, written here because it decides arguments: **when a stage looks
wrong, the question is what the specification says, not what another renderer
draws.** Another renderer disagreeing is evidence to chase, never the target.
https://openusd.org/release/spec.html

Both of the day's cases were of that kind:

- **`MaterialBindingAPI`** has been required since 21.11 and the scene index
  path enforces it. The legacy delegate did not, which is why usdview showed a
  car painted and we showed it grey.
- **OpenPBR's weights live in [0, 1]**, and MaterialX hands them straight to
  the lobes: `generalized_schlick_bsdf`'s `weight` for the metal layer *is*
  `specular_weight`. Blender's USD export writes its "Specular IOR Level" of
  0.98 as `2 * 0.98 = 1.96` -- the right conversion for a dielectric's Fresnel
  and the wrong number for a conductor -- so a Mustang's hubcaps reflected
  nearly twice what they have.

  Measured, the same patch of hubcap under the same grey sky: Cycles
  **0.3151**, us before **0.8994**, us after clamping to the spec's range
  **0.3293**. The clamp says so out loud (`materials: ...: specular_weight is
  1.964, which OpenPBR puts in [0, 1]; taken as 1.000`) rather than quietly
  pretending the asset was right. Four of the Mustang's fifty materials are
  out of range that way.

### Where the two renderers still differ, measured

Same camera, same constant dome of 0.6, 160 paths and the denoiser on both
(Cycles was denoising all along, which is most of what looked like our noise):

| patch | Cycles | ours |
|---|---|---|
| hubcap (metal, roughness 0.23) | 0.3151 | **0.3274** |
| glass | 0.1267 | **0.1122** |
| tyre | 0.0727 | **0.0543** |
| paint, blue channel | 0.3888 | **0.5632** |

The metal is within 4 % and the glass within 11 %. The paint is 45 % brighter
here and the tyre 25 % darker, both of them smooth dielectrics, which points at
the dielectric base rather than at anything about that asset: how much the
diffuse is attenuated by the specular's Fresnel, and whether the specular is
energy compensated. That is the next thing to check **against the spec** -- a
furnace on OpenPBR's dielectric base, not a comparison with Cycles.


## Antialiasing: the camera moves inside the pixel between passes

Every ray of this engine goes through the middle of its pixel, and the
visibility buffer holds one hit a pixel, so an edge was a staircase however
many paths were cast at it -- 320 paths a pixel on the Mustang and the roof
line still stepped, with black pixels caught along it.

A path traced frame, though, is a **mean over passes**, and every pass runs the
visibility again. Moving the camera by a fraction of a pixel between passes
turns that mean into an average over the pixel's area, which is what
antialiasing is. It costs nothing: the same passes, the same rays. The offsets
are a Halton sequence in 2 and 3, which fills the pixel evenly at any number of
passes where a random pair leaves clumps, and the **first pass takes no offset
at all**, so a single pass frame is exactly what it always was.

Two things had to be got right:

- **The mean must not restart.** The frame's identity is compared to decide
  whether this pass adds to the last (`PathState`), and it compares the camera
  the caller asked for, not the pass's offset -- otherwise every pass would
  look like a new frame and the mean would never gather.
- **A gathered frame is no longer pixel-exact against another frame.** Four
  tests compared frames or AOV planes pixel by pixel and now differ at edges,
  which is antialiasing doing its job. They ask for `athenea:antialias` off, since
  what they measure is light groups summing, lights restoring and a volume's
  Beer-Lambert, not the shape of an edge.

`athenea_usd_tests "[antialias]"`, on a quad turned so its edges cross the pixel
grid at an angle: one pass leaves **0** pixels of 30 000 holding partial
coverage -- every pixel is the quad or the background -- and 64 passes leave
**148**, spread along the whole perimeter, with the covered area unchanged to
six pixels. The rest of the edge's pixels are covered in nearly every pass or
nearly none, which is what a box filter does to a straight edge.

The setting is `athenea:antialias` (on), `--no-antialias` on `athenea stage`, and a
descriptor the host can show.


## Aligning with the specification, step by step

The plan of 2026-09-20 (`cozy-petting-canyon`): USD 26.08 is the latest release
and the one built here; what follows is the engine brought to what its text
says, one commit a step.

### Step 0: every schema the engine reads is one the registry knows

`AtheneaSplatSkinningAPI` had lived in `generatedSchema.usda` since the skinned
cloud was written and **was never in `plugInfo.json`**: the schema registry
did not know it, `HasAPI` would have said no, and it worked only because
`ParticleField.cpp` reads its primvars by name. Three sets of primvars were in
the same state with no declaration at all -- `athenea:splat:metallic/roughness/
transmission/litBody`, the four `athenea:splat:visibility*` fields that
`athenea visibility` writes, and the volume's `athenea:densityScale/albedo/anisotropy`.

The specification has nothing to say about what these mean, but it says
exactly how an extension is declared: as an applied API schema in a plugin.
So `AtheneaSplatLightingAPI` now declares its other four, `AtheneaSplatVisibilityAPI`
and `AtheneaVolumeAPI` exist, all seven schemas are registered, and the
registration test checks every one with a default each (`[usd][schema]`, 27
assertions). Nothing the engine draws changed; what changed is that a host
asking a stage "what is applied here" gets the truth.


### Step 1: `opacityMode`, and what a window at opacity 0 still does

UsdPreviewSurface 2.6 says what a fractional opacity means, in two modes.
`presence` is what this engine always drew: the whole response scales, the
surface is there for a sample or it is not, by lot. `transparent` -- **the
default** -- says the diffuse goes down in favour of what is behind *while the
specular and the emission stay at full weight*: a window at opacity 0 still
reflects the sky. We drew nothing there, and a Mustang's windows were holes.

The reading lives in the path tracer, not in the material graph, on purpose.
Putting `opacity` into the lobe weights would have changed the raster route
too, and the raster has no way past a surface but the lot: a leaf at opacity
one half would have gone a quarter as bright. So the graph is untouched, the
compiler answers one more question (`MaterialCompiler::transparentOpacity`: a
root UsdPreviewSurface, mode transparent, no threshold, a fractional opacity),
the flag rides in the material record (`kMaterialTransparent`), the traced
visibility pass leaves such a surface in place (`materialCuts` in
`MaterialPrograms.cpp`), and the tracer draws it with a lot whose expectation
is exactly the specification's:

    keep the surface with probability p = max(opacity, 1/20);
    kept:   specular and emission x 1/p, diffuse x opacity/p
    passed: throughput x (1 - opacity) / (1 - p)

which is `specular + emission + opacity * diffuse + (1 - opacity) * behind`,
with p held off zero so a window at opacity 0 still reflects -- five samples
in a hundred at twenty times the weight, noisy and unbiased. The raster keeps
the lot in either mode, and says so in the test.

Measured (`[usd][opacity]`, path traced): at opacity 0, presence against the
back square alone relMSE **1.2e-14** (not there); transparent adds the sun's
reflection, 48.6 M of energy against 9.07 M. And the estimator's linearity: at
opacity one half, transparent less presence is **0.500** of what transparent
added at zero, which is half the specular, as the arithmetic says. The older
coverage test asks for `presence` by name now, and asserts the default beside
it: an emissive red card at opacity 0 over a white one reads (1.94, 1.00, 1.00)
where the specification says (2, 1, 1). 228 of 228 pass.

### Step 2: `ShadowAPI`, and a shadow the specification lets a light colour

UsdLux `ShadowAPI` is where the "colour and density of the shadow" asked for
the day before belong: `shadow:enable`, `shadow:color` (a non-physical
control, and the schema says so), `shadow:distance`, `shadow:falloff` and
`shadow:falloffGamma`. `enable` was already the `kLightShadow` flag; the other
four ride in the light record now (`LightRecord`, in lockstep in
`LightTable.h` and `lights.slang`), and one function reads them:

    shadowTint(light, through, distance) = lerp(1, shadow:color, shadowing)
    shadowing = (1 - through), then past shadow:distance
                x (1 - t^falloffGamma), t = (distance - shadow:distance) / shadow:falloff,
                or 0 where there is no falloff

`through` is what the occluders let by, which is the map's transmittance in
the raster route, the packed shadow query's in the path tracer, and 0 for a
mesh in the way. So the tint is applied where the shadow is *read*, three
channels at the receiver, never in the map: the map stays one number a texel
and a light with a coloured shadow costs nothing more. Every reader was given
it: the raster's two light loops (`MaterialShading.cpp`), the relit cloud
(`splat_relight.slang`: a dome at distance 0, a lamp at its sample's
distance) and the path tracer's two gathers -- the surface's and the
medium's -- where the mesh occlusion and the cloud's transmittance are folded
into one `through` before the tint, so a mesh's shadow is coloured as the
cloud's is. `athenea:cloudShadowDensity` stays the global multiplier it was: a
renderer setting in its own namespace, as the specification has them.

Measured (`[usd][shadowapi]`, a slab of opaque gaussians over a plane, the
light 45 degrees off vertical, the shadow's patch against the same stage
with `shadow:enable = 0`, both routes): plain 0.006 / 0.538; `shadow:color`
(1, 0, 0) reads (0.538, 0.006, 0.006) -- red through whole, green and blue
stopped; a lamp 2.4 units from the plane with `shadow:distance = 2` reads
0.359 / 0.359, no shadow, and without the limit 0.005.

**What the test found on the way.** The path tracer's cloud shadow
(`athenea:splatShadows`) drew the sparrow's shadow as a compact blob that did not
move with the wings. Two causes, both in the tracer:

- `GaussianRayTracer::prepareFrame` counted the instance tables by cloud,
  while on the hardware route the top level and the tables are written a
  *chunk* (2^20 particles). The packed shadow query (`SplatShadows`) trusts
  that count and drops a hit past it, so a cloud of 5.9 million particles in
  six chunks cast the shadow of its first million -- a spatial prefix, since
  the particles are in Morton order. The count is the chunks' now, and the
  packed-shadow test splits its cloud into five chunks of 4096 and sends a
  ray through four particles it appends last: 0.0625 through them, the
  closed form, and 0 of 5 rays differ from the unpacked query.
- The tracer's per-cloud structures were keyed on the positions buffer's
  handle, and the skinner writes a new pose into the same buffers, so the
  proxies built at the first frame were the ones a whole film traced.
  `GpuSplats::revision` is counted up by whatever rewrites the buffers in
  place (the skinner, after each pose), and the key carries it. A rebuild a
  pose is what it costs; a refit is the cheaper thing to write when a traced
  film of a skinned cloud is wanted.

Sparrow over a plane, `shadow:color` (0.9, 0.15, 0.1), times 1, 24 and 40:
the raster map and the traced query now draw the same red bird, with the
traced one a little softer at the feathers (it integrates the particles a
ray meets; the map integrates a texel).

### Step 3: a cloud is bound to its Skeleton the way the specification binds anything

`AtheneaSplatSkinningAPI` duplicated `SkelBindingAPI` under names of its own --
`athenea:splat:jointIndices`, `jointWeights`, `geomBindTransform`, `skeleton` --
on the reasoning that UsdSkel would not skin a ParticleField. It will not:
`UsdSkelImagingPointsResolvingSceneIndex` makes ext computations for meshes,
curves and points and nothing else. But the binding is legal (the API's
`plugInfo` restricts nothing), and what UsdSkel's imaging *does* do for any
prim is resolve the Skeleton: `resolvedSkeleton.skinningTransforms` on the
Skeleton prim, at any instant, out of `skel:animationSource`. So the missing
half is small, and it is ours now:

- `HdAtheneaSkelSplatSceneIndex` (`modules/usd/src/SkelSplat.cpp`, registered
  after the conversions in phase 1): for a `particleField` whose
  `skelBinding` names a Skeleton, it puts the Skeleton's resolved transforms
  on the cloud as the primvar the engine already skins by,
  `athenea:splat:skinningXforms`, remapped by `UsdSkelAnimMapper` where the
  cloud's `skel:joints` name a subset or another order, and sampled as the
  Skeleton's are so the shutter reads the same instants. It keeps the map
  Skeleton -> clouds and forwards a Skeleton's `resolvedSkeleton` dirty to
  them, which is the notice a time change arrives as: the cloud's own
  primvars are not sampled, so nothing else would tell it.
- `HdAtheneaParticleField::Sync` reads `skel:jointIndices`, `skel:jointWeights`
  and `skel:geomBindTransform` where the `athenea:` ones are absent, and the
  engine takes any number of influences a gaussian (`elementSize`), or one
  set for all of them (a constant binding), where it assumed four.
- `athenea mesh2splat --skinned` applies `SkelBindingAPI` and writes the
  influences, the bind transform, `skel:joints` and `skel:skeleton` under
  UsdSkel's names, and keeps `athenea:splat:skinningXforms` as **the cache**:
  the same numbers at the instants the conversion read them, so a stage that
  composes the cloud without its Skeleton (the film's clip layers) still
  moves, and one with it skips the resolving. Where both are present the
  cache wins. The three `athenea:` influence attributes stay in the schema as
  what an older file carries, and are still read.
- Dual quaternion skinning (`skel:skinningMethod`) is not done for a cloud;
  a prim that asks for it is skinned by linear blend and says so once.

Measured (`[usd][skinning][skel]`): the same 512 gaussians exported with
the cache and bound to a one-joint Skeleton whose animation puts the joint
at t=1 where the cache has it at t=2 -- so nothing read from the cache could
pass -- draw, with the cache blocked, the cache's t=2 frame at t=1 **pixel
for pixel (p99 0, max 0)**, and the bounds slide the Skeleton's two units.

And the sparrow, the asset the cache was made for: `Sparrow_gs.usdc` bound
by `SkelBindingAPI` to `/root/Bird/Bird` of `Sparrow.usdc`, the clip
`air_fly_A0` as the Skeleton's `skel:animationSource`, and the cloud's own
cache blocked (`primvars:athenea:splat:skinningXforms = None` -- it carries one,
of the `Action` the mesh was exported with, and the cache wins where it is
present, which the first attempt at this measurement showed by drawing the
`Action` at frame 17). Against the same clip through its rig layer, at frames
17 and 30: the 609 joints the engine receives agree with the rig's (joint 3
reads (0.994, 0.048, 0.099) in both; USD's own
`ComputeSkinningTransforms` against the Blender-written cache differ by
**5e-4 at most** over the 609), and the frames differ by a mean of 8e-6 with
the largest differences on the depth channel at silhouette pixels. The
mesh itself, skinned by UsdSkel's standard path, draws the same pose. So a
stage that composes the cloud with its Skeleton needs no rig layer at all;
the 71 clip rigs stay as the cache they are.

### Step 4: the ParticleField, read against `schema.usda`

Checked, attribute by attribute, against usdVol's schema (26.08):

- `scales` are linear and `opacities` linear, as the schema says ("linear
  scales, in line with scales provided elsewhere in USD"); the USD stream
  uploads them as they are (`streams.slang`: "opacity (linear), scale
  (linear)"), where a PLY's log scale and logit opacity go through the
  decode's other mode. Nothing to change.
- `radiance:sphericalHarmonicsDegree` is `uniform int`, read as one; the
  coefficients are "grouped by particle", DC first then the higher orders,
  which is the order the stream is striped in (`coefficients` a splat).
- The half twins (`positionsh`, `scalesh`, ...) are usdVolImaging's to
  choose: its data source maps whichever of the pair is authored to the
  float name (`UsesFloatOpacities`), so the `either` read of both names is
  belt and braces, and harmless.
- `projectionModeHint` and `sortingModeHint` are, in the schema's words,
  hints a renderer is free to ignore, and Hydra does not carry them
  (usdVolImaging 26.08 leaves them out of the prim's data source, with an
  XXX to that effect). They are read off the stage when it is opened and
  named in the log where a cloud asks for `tangential` (the rasteriser
  projects in perspective) or an order other than `zDepth` (the rasteriser
  sorts by view depth; the traced route meets particles along the ray, which
  is `rayHitDistance`). A test holds the frame to be the same with the hints
  as without, which is what a hint is.

### Step 5: what 26.08 added, and what it found

**`DomeLight_1.poleAxis`.** UsdImaging does not fold the pole's alignment
into the light's transform: it hands it as a light parameter of its own,
`domeOffset` (a matrix, +90 degrees about x for a Z pole), because the
schema says the rotation is the dome's and not its children's. `Light.cpp`
reads it and applies it first, then the prim's transform, as Storm does.

Writing the test for it -- a lat-long white above the horizon and black
below, a plane facing the pole, on a Y-up and a Z-up stage -- found that
**the dome's image was upside down and turned half round**. The rows of a
texture here start at the bottom (`texture_decode.slang`, so that v = 0 is
the bottom of an image as UsdUVTexture has it), and `domeUv` read the pole
at v = 0: the sky's zenith was at the ground. And the longitude was taken
from `atan2(x, -z)`, which puts +z at the image's edge, where the
specification (citing OpenEXR: "latitude 0, longitude 0 points into
positive z; latitude 0, longitude pi/2 into positive x") and Storm's
`ProjectToLatLong` put it in the middle. Every dome test until then used an
image of one colour, which reads the same whichever way it hangs. The
mapping is OpenEXR's now, in `domeUv` and its inverse `domeDirection`, and
the importance sampling follows since it walks the same uv.

Measured (`[usd][poleaxis]`): the plane facing the pole reads its whole
albedo, 0.8, under `scene` on Y-up (0.799), `scene` on Z-up (0.799) and
`Z` on Y-up (0.799); facing +y under `poleAxis = Z` it reads the horizon's
half, 0.400; and the old `DomeLight` on a Z-up stage reads 0.400 facing +z,
which is the specification's own account of why DomeLight_1 exists.

**`HydraRenderPassAPI`.** A `RenderPass` may name the renderer it is for
(`hydra:rendererName`, what usdrecord takes when nobody says `--renderer`).
`StageRenderer::renderSettings` lists the passes whose `renderSource` is
the settings prim, with the renderer each names and whether it is this one
("athenea" or the plugin's id, or unsaid), and says in the log when a
pass is meant for another. `usdHydra` is linked for it.

**The active settings' switches, for a camera of our own.** A settings
prim's `disableMotionBlur` and `disableDepthOfField` held for its products
and for nothing else; a free camera with `athenea:shutter` and `athenea:lens` asked
for blurred whatever the stage said. `aim(const render::Camera&)` now reads
the stage's active settings prim (`renderSettingsPrimPath`) and sets the
two switches from it, so the pass draws one shutter slice and a pinhole.
Measured: the shutter changes 10765 words of the frame; with the flag on
the active settings, 0.

### Step 6: what is the specification's and what is ours, said in one place

`AtheneaSplatLightingAPI` now carries `apiSchemas = ["ParticleFieldRadianceBaseAPI"]`
in its generated schema: it is another definition of a ParticleField's
radiance, and UsdVol gives every radiance definition that base, so applying
ours applies it (the registration test checks the prim definition lists
it). The other six have no standard base to stand on and say so.

The README has the inventory -- "Where the specification ends and this
engine begins" -- in three lists: what is standard and read as written,
what is a standard mechanism used as the specification provides (settings
namespaces, raw and lpe render vars, the session layer, codeless schemas),
and what is this engine's own (the seven schemas, `athenea:lightGroup`,
`.athc`). Falcor is placed there as what it is: the shape a Slang renderer
was read out of, last released in August 2024, not a specification.

### Step 7: the SDK, ready for the next release

`scripts/build-usd.sh` takes `dev` beside a tag, into its own prefix
(`~/tools/usd-dev-mx`), following the branch on each run; its header says
what to match from the release's `VERSIONS.md` (oneTBB 2021.9, OpenVDB
10.1.0, OpenSubdiv 3.6.1, MaterialX 1.39.5, CMake 3.27, clang 16 / gcc
11.5) and which deprecations to read before building. The plan is to build
`dev` on the Linux machine first and keep the Mac on the release until the
engine is green against it.

Deprecations in force, checked against the tree:

- `HdRenderIndex::New` is marked deprecated in 26.08 "in Hydra 2.0,
  applications create scene indices feeding into HdRenderer" -- the future
  entry point, which does not exist yet; ours is the back-end emulation the
  same header keeps (`NewForBackendEmulation` is the same thing with the
  terminal scene index handed in). Kept, with this note.
- `SdfTimeCode` -> `GfTimeCode` (26.08): not used.
- `displayName`, `displayGroup`, `hidden` -> `uiHints` (26.03): the codeless
  schemas author none; the plugin's `displayName` in `plugInfo.json` is the
  renderer plugin's, a different thing.
- The scene delegate path (26.03): this delegate is on the scene index path,
  with its own filtering scene indices.
- `usdc` files older than 0.8.0 warn (`PXR_USDC_EMIT_DEPRECATION_WARNINGS`):
  everything written here is written by 26.08's `UsdStage`.
- Python 3.9/3.10 deprecated: no Python is built.

### Step 8: a volume is shaded by the Material it binds

UsdVol says nothing of how a Volume scatters: that is the Material bound to
it, with a `volume` terminal, as Storm and Karma read it. This engine read
three primvars of its own (`athenea:densityScale`, `athenea:albedo`,
`athenea:anisotropy`, now `AtheneaVolumeAPI`). Both are read now, and the Material
wins:

- `HdAtheneaVolume` reads its material binding; `HdAtheneaMaterial` builds the
  MaterialX document from the `volume` terminal where there is no
  `surface` one (hdMtlx names the material's input after the terminal
  node's type, so nothing in it is special-cased).
- `MaterialCompiler::volumeCoefficients` reads the document: `volume(vdf,
  edf)` with `anisotropic_vdf(absorption, scattering, anisotropy)` or
  `absorption_vdf(absorption)`, and `uniform_edf(color)`. The coefficients
  are per unit of the Volume's `density` field, as every renderer has it.
  An input a graph drives is taken at its declared default and said once:
  a medium has no surface to evaluate a graph at. Layered or mixed VDFs are
  not read.
- The medium walks one extinction and scatters with an albedo, so extinction
  is the mean over the channels of absorption + scattering and the albedo
  the ratio a channel: a coloured extinction rides in the albedo, not in
  the free flight. A limit, written down here.
- Emission is new to the medium (`VolumeRecord.emission`, words 28..30 of
  the record). At a collision, which the tracker draws with sigma_t, the
  path absorbs 1 - albedo and that is where a glowing gas emits: `carried +=
  throughput * L_e * (1 - albedo)`, so an absorbing medium of optical depth
  tau reads L_e (1 - exp(-tau)) and a medium of albedo one glows nothing.
- A Material with only a volume terminal is not compiled as a surface (it
  used to warn "no renderable element").

Measured (`[usd][volume]`): the Beer-Lambert box by Material
(`absorption_vdf` of (1, 1, 1)) against the same by primvars (densityScale
1, albedo 0), p99 of the difference **at most 1** -- the medium's random
walk is not bit-exact between dispatches, a run against itself differs by a
few words; and glowing with `uniform_edf` (1, 0.5, 0.25) over the lit
plane, E - A + L_e A/B over the 9801 pixels through the box reads
**(1.000, 0.500, 0.250)**, which is the closed form to three places.

## A car in gaussians, and the two things it found

A Mustang converted with `athenea mesh2splat` came out with its chrome black --
the bumper, the grille, the wheel rims -- while the mesh it was made from
shows them bright. The cloud had the geometry: from three cameras, of the
pixels the mesh covers, **none** were missing in the cloud (the depth
channel's masks; the 0.17% the other way is the gaussians' soft edge). What
was missing was the light.

**A metal is not always a `conductor_bsdf`.** What a bake keeps of a
material is its body -- the diffuse lobes, what it transmits, and a
conductor's reflection, which is all a metal has -- and what it drops is the
dielectric polish, which `splat_relight` puts back at render time with a
direction in it (`bakeBody`, PathTracer.cpp). MaterialX writes **OpenPBR's**
metal, and Standard Surface's, as a **generalized Schlick** (the F82
parameterisation) rather than as a conductor, and the Mustang's materials
are OpenPBR: `open_pbr_surface` with `base_metalness = 1`. So every metal in
the car was dropped as polish and baked to nothing at all -- not dim,
exactly nothing: all 27016 gaussians of the front bumper stored the constant
term -1.7724538, which is -0.5/kSH0, the one that decodes to black. The same
metal written as UsdPreviewSurface baked correctly, because the graph here
uses `conductor_bsdf` for it, which is what hid this.

What tells a metal's Schlick from a polish's is the reflectivity at normal
incidence: a dielectric's is what its index of refraction gives (0.04 at
1.5, 0.17 at diamond's 2.42, which is as far as any of them reaches), a
conductor's is half the light or more and coloured with it. A Schlick whose
F0 stands above a fifth is kept. Measured (`[usd][bake][metal]`, a new test
of one metal written both ways): UsdPreviewSurface **0.908**, OpenPBR
**0.931**, and a dark dielectric under a bright polish still **0.216**,
which is its body and not its polish. On the car, the bumper went from
100% black gaussians to a mean colour of (0.45, 0.51, 0.61).

**A dome's image was only loaded where there was a mesh.** Lighting an HDR
of a workshop over the cloud -- so that black in a picture means a gaussian
with nothing baked into it, rather than a dim sky -- showed the sky behind
the car as a blank white wall. The light images (a dome's picture, an IES
profile) were requested inside the mesh layer's preparation, which a frame
of splats alone never reaches, so the dome was the white one a light with no
image is. They are `Engine::prepareLightImages` now, called from both places
that build a light table, and a cloud alone under a dome builds one. The
background behind the cloud then reads (0.2545, 0.2641, 0.2080), which is
what the mesh stage reads to four figures.

**And the budget is not the resolution.** The first conversions asked for
`--resolution 1024` against the default budget of 2 M: the car wanted 3.67 M
and the meshes that come late in the stage -- the rear wing, the disc
brakes, the front valance -- got a handful of gaussians each or none, which
is why the car had holes. `--max-splats` is what decides, and the log says
how many did not fit.

## A pixel says which prims it saw: Cryptomatte

There was no matte in this engine at all. The only id a pixel carried was
`primId`, written by meshes alone: a cloud wrote nothing into it, `pick` did
not see one, and a cloud converted from a whole car was one prim
(`/World/Splats`), so a matte per prim would have said "car" and stopped
there. What makes a matte possible for gaussians is that `athenea mesh2splat` is a
conversion, not a capture: every gaussian is born on a triangle of a mesh
whose path is known, and that ancestry is free to keep.

**One layer, `CryptoObject`.** The id is `MurmurHash3_x86_32` of the prim's
path with the specification's fixup (an exponent of all zeros or all ones has
its lowest bit flipped), stored and written already fixed up, so the bits are
always a normal float and survive a float buffer, a float channel and a
manifest's hex. `athenea::core` holds the hash, the fixup, the hex, the layer key
and the manifest in both directions, with the reference's own test vectors
(`athenea_core_tests`). Id 0 is absence: the fixup can never produce it, so a
capture, a point and the background are simply not in the matte rather than
named something they are not.

**The matte is built in the blend, not after it.** The splat blend already
walks a pixel's splats front to back and already computes the one number a
matte needs: `weight = transmittance * alpha` is exactly the coverage that
splat contributed. So `blendPixel` keeps a list of six (id, coverage) pairs in
registers -- find the id, add to it, else take a free slot, else evict the
lightest -- and writes them, ranked, as the three `CryptoObject00..02` layers.
Six ranks is what the format carries; past that the lightest is dropped, which
is what every renderer that writes the format does. The walk without a matte
must not pay for one, so `matte` is a compile-time constant at the call site
and there are four entry points instead of two; `tests/render` measures the
colour bit for bit equal with the matte on and off.

**An id travels inside the record.** The loader compacts on the device (a
record goes to `base + dest[i]`), so a per-gaussian array beside the cloud
would come apart the moment a gaussian is dropped. The id goes into the record
like the PBR fields do -- `streams.slang`'s `cryptoBase`, `SplatEncoding::
cryptoObject`, `splat_decode` writing `GpuSplats::crypto` -- and the frame
reads it through `cryptoIds_`, one word a projected slot, beside `proj_`.

**Meshes are named through a table.** `technique::CryptoShading` reads the
visibility buffer, takes the instance's `primId` and looks it up in
`cryptoOfPrim` -- so a flat prim and an instanced one are answered by the same
kernel with no record of their own. That id plane travels with the mesh layer
(`RenderTargets::cryptoIds`), `layers_nearest` carries it when points are
composited under (a point carries no id, so where a point is nearest the pixel
goes unnamed), and the blend names by it whatever coverage closed the walk.
Because the composite blend runs whenever there is an `under` layer, a
mesh-only frame gets its matte through the same code as a cloud's, with no
splats in it at all.

**What a reader needs.** A render product with a `CryptoObject*` var writes
twelve float channels (`CryptoObject0k.R/.G/.B/.A`, float even when the
product asked for half: an id rounded is another id's name) and the four
string attributes `cryptomatte/<key>/name|hash|conversion|manifest`, the key
being seven hex of the raw hash of "CryptoObject". The manifest is every prim
the frame drew plus whatever each cloud brought in
`primvars:athenea:splat:cryptoManifest`, so a converted cloud names its *source*
prims, not the prim it now is. `athenea view` shows the matte as a reader previews
one (`--aov cryptomatte`: an id a colour, mixed by coverage) and each layer
raw.

**The viewer says who is who.** `athenea view` shows the matte as a reader
previews one, each layer raw, and -- `--isolate <prim>` or the button on a
pick -- one id's matte alone, white on black. A pick reads the nearest rank at
that pixel and names it through the frame's manifest, which is how a cloud is
picked at all: a gaussian writes no `primId`, so before this a click on a
cloud found nothing. Where a cloud stands over a mesh the two disagree on
purpose -- Hydra's pick names the surface behind, the matte names what covers
the pixel -- and the panel shows both, beside the frame's own counts
(gaussians handed over, gaussians a tile kept, pairs the blend walked, mesh
instances, lights), which are numbers the passes already had rather than
anything measured to fill a panel.

**Measured.** MustangHero at 960x540, 934 338 gaussians converted per mesh:
the first rank's coverage averages 0.999 over the frame, the manifest holds
the 106 prims the conversion read, and a frame with the matte on drew in 14.9 ms
against 14.3 ms with it off (`athenea view --frames 60`, medians of three runs,
release: four per cent for six ranks a pixel). The usd test checks a stage
whose cloud names two source prims and whose plane names a third: every pixel
is named, the coverages sum to the pixel's alpha within 1.3e-07, and the ranks
descend.

**What it does not do.** Material and asset mattes (the format admits them as
sibling primvars, and nothing here would change). The traced route for splats
alone (`rt_integrate`) writes no matte. Points carry no id. A captured cloud
has no ancestry and stays out. A mesh edge is one visibility sample, as
`primId` is, so a matte has no sub-pixel coverage at a silhouette; a
`transparent` material counts as coverage 1.

## A cloud sees the sky it stands under

A dome's image goes through the material texture table. The splat projection
lives below `material` in the module order and has no table, so what a cloud
was given of a dome was `lightEmission` -- the light's colour times its
exposure, a constant in every direction. The mesh beside it, meanwhile,
sampled the image itself along its lobe's own samples. Two surfaces under one
sky, lit by two different skies.

The fix is not in the lighting but in the form of the sky. `technique::
Environment` prepares two of them, with kernels that do have the table:

- **Nine harmonics of radiance a dome** (`env_project`), summed over the
  texels of one level of the image's own mip chain, each weighed by the solid
  angle a lat-long texel covers -- `2 pi^2 sin(theta) du dv`, the same measure
  the dome's importance sampling uses, so the two agree by construction. Nine
  is what a diffuse surface needs: the irradiance of any environment is held
  to about a percent by them (Ramamoorthi and Hanrahan 2001).
- **The sky convolved with a GGX lobe** (`env_prefilter`), a mip chain of
  eight levels between a mirror and the whole hemisphere, in an octahedral
  map. The lobe is sampled with `ggxSampleVisibleNormal` from the material
  module, so it is the same distribution the surface lobes use, and a sample
  reads a level of the source's own chain chosen by the solid angle it stands
  for, which is what keeps one texel of a 4k sky from being a firefly nobody
  averages out. How wide the chain starts and how roughness walks it is the
  section below.

Both go in buffers, not textures, and `shaders/athenea/light/environment.slang`
is the layout and the reader: a module that imports nothing a splat kernel
cannot afford. So the rasteriser and the tracer read the same sky through the
same function, as they already did for everything else a cloud is lit by.

**The coefficients are radiance, not irradiance.** Convolving with the cosine
lobe at read time costs three multiplies (the zonal bands are pi, 2pi/3 and
pi/4) and leaves one source for two consumers: a diffuse body wants the
convolution, and the transfer vector a gaussian will carry has the cosine in
it already.

**A dome with no image prepares nothing.** A constant is what the closed form
in `splat_relight` answers exactly, and it still answers it: the fallback is
the code that was there before.

### How it is checked

`athenea_technique_tests "[environment]"`, with the arithmetic in
`shaders/athenea/test/environment_check.slang` and counters crossing back.

- A sky of one colour must project back to `pi * L` of irradiance at **every**
  normal. Measured over 64 directions of a Fibonacci spiral: **6.2841 where
  the closed form is 6.2832, 0.089 %**. A measure wrong at the poles, or
  bands weighed wrongly, reads something else here.
- A sky white above the horizon and black below: the mirror level reads
  **1.000** at the pole it faces and **0.000** at the other, and the roughest
  level reads **0.999** -- not the whole sky's mean but the mean of the
  hemisphere it faces, which is what a GGX lobe of alpha one gathers.

### Measured (M5 Pro, debug)

The chess pawn, relit and not baked, under `autoshop_01_4k.hdr`, against the
same pawn path traced as a mesh under the same dome. Over one patch of its
marble body:

| | R | G | B |
|---|---|---|---|
| the mesh, path traced | 0.043 | 0.057 | 0.054 |
| the cloud, before | 0.070 | 0.083 | 0.081 |
| the cloud, now | 0.044 | 0.059 | 0.056 |

Sixty-five per cent too bright, to three. The reading before was not dark but
*bright*: the dome's mean radiance is brighter than what a body under it
actually receives, and nothing in the old path knew the difference between a
normal facing the workshop's lit ceiling and one facing its floor.

## The prefiltered sky is a mip chain of the dome, not a fixed 256

The map's base was 256 a side for every stage and its six levels were spaced
straight in roughness, `level / 5`. Both of those were blurring things they
had no business blurring, and the second was worse than the first.

**The side.** No prefiltered map can be sharper than its base, and a 256
octahedral texel is a quarter of a degree. That quarter of a degree sat on
every mirror, every smooth reflection and the far face of every glass ball,
and under a 4k HDRI the workshop seen through a glass sphere came back as
haze beside the same sphere path traced. The base now follows the dome's own
image: `Environment::baseSideFor` matches the map's texel to the lat-long's
(a lat-long texel stands for `2 pi^2 / (W H)` steradians at the equator and an
octahedral one for `4 pi / S^2`, so the two meet at `S = sqrt(2 W H / pi)`),
rounded down to a power of two and held between 256 and 2048. A 4k sky takes
2048, a 2k sky 1024, and a dome that is only a colour the floor. It is no
longer a constant, so it reaches the shaders as a uniform (`envBaseSide` in
both frames) and every reader takes it.

**The spacing.** A lookup blends the two levels that bracket it. Straight,
level 1 stood for roughness 0.2: gold at 0.08 came out as *half a mirror
blended with half of a lobe convolved to 0.2*, which is neither of them, and
glass at 0.02 carried a tenth of that same 0.2 as haze. The chain is now
eight levels, each half the side of the one before, and roughness walks it as
its square root -- the levels stand for `(k/7)^2`: 0, 0.020, 0.082, 0.184,
0.327, 0.510, 0.735, 1. That 0.08 now lands on the level convolved to 0.082,
and each level's texels are about as wide as its own lobe. `envRadiance` and
`Environment` write the same curve; the comment in each says so, because a
disagreement is a blur nobody asked for.

**What it costs.** A slice is 45 MB at a 2048 base (4.2 M texels at level 0
and 1.4 M in the rest, two f16 words each), 1.2 MB at the 256 floor, against
0.7 MB before. Four 4k domes at once is 179 MB, which is the price of
`kEnvironmentDomes` being four. The build is one texture tap a texel at level
0 and `16 << (level - 1)` directions, capped at 128, above it: the work halves
going up the chain rather than staying flat, so a 2048 base is about thirty
million samples in all. Measured on the M5 Pro under `autoshop_01_4k`: the
first frame of `athenea stage` goes from about 220 ms to **402 ms**, and the
frames after it are unchanged at **33.5 ms**. It is rebuilt only when a
dome's record changes, so turning a dome in the viewer pays it per drag frame.

### Measured, and where the blur went

The three spheres of `athenea-outputs/work/three.usda` -- gold at roughness 0.08,
a grey dielectric at 0.12, clear glass at 0.02 -- as gaussians against the
same three path traced as meshes, under `autoshop_01_4k`, over the middle 80 %
of each ball:

| | before | now | the mesh |
|---|---|---|---|
| glass, cloud/mesh | 1.21 1.22 1.17 | 1.18 1.18 1.12 | 1 |
| dielectric | 0.93 0.94 0.94 | 0.93 0.94 0.94 | 1 |
| gold | 0.87 0.85 0.83 | 0.84 0.81 0.78 | 1 |

The glass closes and reads visibly sharper; the gold moves *away*, and that is
worth writing down rather than hiding. Under a constant sky the same gold
reads **1.00 0.99 0.97** of the mesh, so its energy is right: what the HDRI
measures is not how much light comes back but from where, and a mean over a
patch of a mirror moves when the reflection sharpens. The reading is a
distribution, not a deficit.

**The sky is no longer what limits the sharpness.** Forcing `envRadiance` to
read level 0 -- the 2048 mirror, no blend at all -- and re-rendering the gold
ball gives an image barely sharper than the chain's own answer and still far
softer than the mesh's. What is left is the cloud, and the section below
measures it.

## A gaussian's footprint is the blur kernel of its reflection

The question the sky did not answer: the gold ball asks for roughness 0.08 and
does not look like 0.08. So it was matched against one. The same three spheres
were converted at three cells and each was held, by RMS over the ball, against
a ladder of the same ball path traced as a mesh with nothing changed but its
`specular_roughness` (0.08, 0.10, 0.12, 0.14, 0.17, 0.20, 0.24, 0.29, 0.34).

| `--resolution` | cell | gaussians | file | frame | reads as roughness |
|---|---|---|---|---|---|
| 256 | 0.0281 | 202 164 | 10 MB | 13.4 ms | 0.34 |
| 512 | 0.0141 | 790 740 | 39 MB | 14.7 ms | 0.20 |
| 1024 | 0.0070 | 3 159 888 | 157 MB | 18.2 ms | 0.14 |

(900 x 340, 64 paths, M5 Pro, release; the frame is the median of a `--frames`
run, so nothing of opening the stage is in it. The spheres are unit radius, so
a cell in world units is also a cell in radians of normal.)

It is linear in the cell, and it adds to the roughness rather than to its
square: `0.34 = 0.08 + 9.25 c`, `0.20 = 0.08 + 8.5 c`, `0.14 = 0.08 + 8.6 c`.

> **A cloud reads as the mesh at `r + 9 c / R`,** where `c` is the conversion's
> cell and `R` the radius of curvature. A mirror at roughness `r` wants a cell
> under `r/9` of that radius.

Which is what a ray through the shell does: it crosses gaussians whose centres
are spread over a couple of cells of surface, their normals differ by that arc,
and the blend averages their mirror directions over it. Nothing downstream can
undo that average, and no sharper sky can either.

**`--sigma` moves it too, and less usefully.** At `--resolution 512`, narrowing
the gaussian against its cell from 0.65 to 0.55 takes the reading from 0.20 to
0.12 and *raises* the mean from 0.90 to 0.94 of the mesh; 0.45 reads 0.12 as
well but drops the mean to 0.89 and the conversion's lattice shows through the
ball as a diagonal moire. Narrower gaussians mean less overlap, which is both
the sharpening and the holes. The cell is the honest knob.

**What this costs is the file, not the frame.** Fifteen times the gaussians is
36 % more time a frame and sixteen times the disk. The ray tracer walks a tree
and hardly notices; what binds is `--max-splats` and what a `.usdc` weighs.

`athenea-outputs/reference/08_what_softens_a_reflection.png` is the sheet, and
`athenea-outputs/work/reference.py` makes it again.

## The sun comes out of the sky and is lit as a light

Nine harmonics hold the irradiance of any environment to about a percent --
of an environment with no sun in it. A half-degree disc a thousand times
brighter than the sky around it is not a degree-2 function: projected, it
smears into a lobe sixty degrees wide. The percent is over the whole sphere;
the error where it matters is total. A pawn under a clear sky gets no
terminator, no highlight and no shadow, only a soft lean towards the bright
side.

So `env_sun` finds the disc, `env_project` leaves it out, and the shading adds
it back as the light it always was: an exact cosine, and a direction a lobe
can be pointed at. It is what the industry calls sun extraction, and it is the
thing a relightable cloud needs before it can cast anything.

### Finding it, and three ways not to

The search is an octahedral grid of 64 a side -- 4096 directions, about 1.8
degrees each, walked twice by one thread a dome. The grid rather than the
image is what keeps the cost off the sky's resolution: a 4k dome and an 8k one
search the same 4096 directions.

What took three tries was deciding what counts.

- **A fixed cone** of seven degrees. Wrong in both directions. A captured sun
  is clipped and bloomed -- `kloppenheim_06_puresky` is still four fifths as
  bright seven degrees off its centre -- so a fixed cone cuts a slice out of
  the middle of it, which is the one thing that must not happen: a ring left
  in the harmonics and a hard terminator where a soft one belongs. And it
  found a source in eleven of thirteen skies, every studio among them.
- **Concentration**, the share of the cone's energy inside three degrees. The
  wrong question. That same real sun puts a fifth of its energy there while a
  small hard softbox puts nearly all of it: the test separates small from
  large, not sun from softbox.
- **The profile, measured.** Rings of two degrees about the brightest cell,
  out to thirty-two, each with the outermost ring's brightness taken off it --
  a low sun sits in a bright horizon, and `sunset_fairway`'s rings never fall
  while the horizon holds them up. The cone ends at the first ring down to a
  tenth of the middle. Nothing wider than ten degrees is taken at all: past
  that a source is something nine coefficients hold well enough, and
  `urban_alley_01` offered a twenty-four degree wall.

Over the thirteen skies in `athenea-outputs/hdris`, with a peak that must also
clear eight times the sky's mean:

| sky | what is found |
|---|---|
| autoshop_01 | a lamp, 4 degrees |
| brown_photostudio_02, golden_gate_hills | 2 degrees |
| dikhololo_night, lythwood_room | 6 degrees |
| kloppenheim_06_puresky, venice_sunset | 4 degrees |
| moonless_golf, studio_small_09, thatch_chapel | 2 degrees |
| urban_alley_01 | 10 degrees, at the cap |
| snowy_park_01 | nothing: an overcast |
| sunset_fairway | nothing: twenty degrees of bright horizon |

Each is said out loud when it is prepared, which is the one thing about a sky
an operator cannot see by looking.

### Both sides must integrate the same way

The disc's irradiance is **not** summed on the search grid. It was, and a grey
ball under `thatch_chapel` went from four per cent bright to fourteen: a
1.8-degree cell sampling a 1024-wide sky reads a half-degree source at its
peak and multiplies it by the whole cell, which is eight times the energy that
source carries. The harmonics, meanwhile, lose exactly the lat-long texels
`env_project` skips.

So the irradiance is summed over those same texels, at the same level, with
the same measure -- `2 pi^2 sin(theta) du dv` -- and what leaves the
coefficients is what arrives as a light, to the last texel. The test checks it
in closed form: **0.1119 against 0.1113**, half a per cent, where the ring sum
read 1.6 times it.

### What it buys, which is nothing yet

Measured, honestly. Three spheres under `thatch_chapel`, the grey one, cloud
over mesh: **1.040 without the extraction, 1.029 with it**. The pawn under the
same sky: **0.935 either way**, identical.

That is the right answer and it is worth saying plainly. The extraction moves
energy from the coefficients into a delta light of the same energy; it does
not add any. What it changes is where that energy points -- an exact cosine
instead of a sixty-degree lobe -- and with no shadow yet the difference sits
below the noise of a patch mean.

**It exists so that the next thing can be built.** A sun smeared across nine
coefficients has no direction to be occluded along. Now it has one, and the
per-gaussian bitmask of traced directions has something to cast. Until then
the sun is attenuated by `splatSunShare`, the transfer's own estimate of
visibility along that direction: numerator and denominator share the degree-2
truncation, so the ratio is exactly one where nothing occludes, and it is a
soft shadow -- the first a relit cloud has had that knows which way the light
came from, at no ray and no byte.

### Not done

- ~~The polish does not see the extracted sun.~~ It does now, as the shadowed
  share taken back out of the map (next section).
- **One sun a dome**, the brightest. Not a decomposition of a sky into several
  area lights: a window in `lythwood_room` and the fluorescents in
  `autoshop_01` are one source each at most.
- **The transmitted half does not receive it.** A sun behind a translucent
  gaussian still comes from the harmonics.

### The sun casts a shadow: sixty-four bits a gaussian

The transfer is degree 2, so the sun it shadows gets a smooth falloff where a
collar has an edge. `kTransfer` now also traces, once per point at its first
sample, one ray a cell of an 8 x 8 octahedral grid over the whole sphere, and
keeps a bit where the ray left the scene: 8 bytes a gaussian, written as
`primvars:athenea:splat:shadowBits` (int[], elementSize 2) and carried like the
Cryptomatte ids, as bits in a float stream, into `GpuSplats::shadowBits`.
`splatSunOpen` looks the sun up in them -- the four cells around it, bilinear,
cells below the gaussian's horizon left out because nothing was asked there --
and falls back to `splatSunShare` where a cloud has none.

**The polish is shadowed too.** The prefiltered map kept its sun, so a rough
reflection and every metal carried an unshadowed sun whatever the bits said;
the pawn in its own metal read the same with and without them. What does not
get through is now taken back out as the analytic GGX lobe of a directional
light of that irradiance, clamped at zero because on a mirror the analytic
lobe is a spike the map does not hold. And a cloud **with no transfer** had
lost the sun's body altogether when `env_project` began cutting it: it gets it
back, unshadowed.

Measured. A roof over a plane has a closed form: of 1508 cells judged over 64
points, **0 wrong** (`the open directions a transfer keeps are the ones a roof
leaves`). The pawn in a white matte material under `golden_gate_hills` (sun 42
degrees up, irradiance 5.9) shows the collar's shadow on the neck and the
stem's on the base where the path traced mesh has them; without the bits there
is none. Frame time 29.5 ms either way. The bake of the pawn at `--resolution
512` is 33.9 s, as before: 64 rays a point against 64 x 3 paths is noise.

- **Under a sky whose light is a window**, not a disc (`thatch_chapel`: 0.0021
  sr), the extracted sun is a small part of the light, and the shadow the mesh
  shows comes from the window, which stays in the nine coefficients.
- **Only 24 of the 64 cells face any one normal**, and 16 lie on the equator of
  a plane facing a pole: the grid covers the sphere so that a normal turned
  towards the eye still finds its cells. An edge therefore falls over a cell,
  about twenty-five degrees, softened by the bilinear read. A hemispherical grid
  in the gaussian's own frame would give it the whole 64.
- **Space.** The bits are baked in the cloud's own space and looked up with the
  world's sun direction, as the transfer already is: right for a cloud whose
  transform is a translation or a scale, turned for one that is rotated.
- **Not through `.athc`, not on a skinned cloud**, as the transfer itself.

### A glass sparrow, and what it found

A test asked for the sparrow all in glass, as the pawn's ball, and in motion
blur. Three things were missing, and the test is what showed each.

- **Saying what a prim is made of from the command line.** `athenea stage
  --splat-override PRIM=M,R,T[,R,G,B[,REPLACE]]` writes the same per-prim rows
  the viewer's Picked panel does, resolved by the cloud's Cryptomatte manifest
  (`*` for every prim), after one small raster frame that asks for the matte --
  the manifest is built only then. A tint multiplied a feather texture and
  left it a feather, so a row's spare component now says the tint **is** the
  colour (`replace`, and the viewer's "The tint is the colour").
- **Glass without a transfer was translucency.** The index, the far face a ray
  found and what stood behind were read only in the branch for a cloud that
  carries a transfer; every other cloud -- every skinned one among them --
  sent on the sky's irradiance from behind, uniform, whatever its `ior`. Both
  branches now share `transmittedBehind` and `transmittedShare`. Translucency
  with no index is unchanged.
- **A cloud's transform never blurred.** `SplatInstance::viewStep` existed,
  and the rasteriser's rank-one term read it, but only the tests ever filled
  it: in the film the wings blurred and a bird crossing the frame at 0.037
  units a frame did not. `ParticleField` now samples its transform over the
  shutter as a mesh does (bracketed, scaled to the shutter as the skinning's
  step is), and `Engine` forms `V_close M_close - V_open M_open` with the
  projection's own camera ends, so the camera's motion blurs a cloud too. A
  cloud moving against the camera moving the other way: relMSE **0**; against
  the shutter closed 0.105 (`a cloud moving under the shutter blurs as the
  camera moving the other way does`).

The sparrow at frame 300 (9.5 M gaussians), 900 x 675, raster: 70 ms still,
164 ms under a 180 degree shutter.

- **The traced route still draws a cloud at the frame's instant.** Its motion
  blur is the shutter in slices, and a cloud is posed once a frame: the glass
  sparrow traced is sharp at any shutter. What the rasteriser smears, the
  tracer would have to pose per slice.
- **The step is centred on the frame's pose**, as the moment match is: a
  shutter of 0 to 0.5 smears about the frame's instant rather than ahead of
  it.
- **A LOD instance carries none.** Only a cloud drawn whole does.

### Thin walls, and a glass bird that is not black

A sparrow in glass, path traced as a mesh, came out almost black under a
bright sky, and its radiance bake with it: the feathers and wings are some
twenty thousand cards, and each was a solid -- bent in, bent out, and a path
through a few dozen of them out of bounces (6, 16 and 32 bounces barely
differed). A card is a sheet. MaterialX puts `thin_walled` on the `surface`
constructor, OpenPBR wires `geometry_thin_walled` to it, and nothing here read
it; `standard_surface` itself only uses it to swap subsurface for
translucency, so the bird's cards are OpenPBR.

- **The compiler** emits `gAtheneaThinWalled` from the constructor before its BSDF
  builds, and `atheneaPushLobe` marks every dielectric it pushes
  (`kFlagThinWalled`); a thin wall has no inside, so the index never inverts.
- **The lobe** reflects with both faces' Fresnel summed, `2R / (1 + R)`, and
  transmits the rest along a delta straight through -- at any roughness, since
  a sheet's two faces share the microfacet and the bend at one undoes the
  other. Smooth, the furnace reads **1.0000**; at alpha 0.25, 0.9963, and the
  reflected part still passes chi-square.
- **The conversion keeps the cut-out** of a MaterialX glass: `opacity`
  (standard_surface) and `geometry_opacity` (OpenPBR) connected to an image
  are coverage, read by the image's first channel and by the UV set its
  `texcoord` names through a `geompropvalue`. The glass sparrow converts to
  9 536 402 gaussians, as the original with its UsdPreviewSurface cut does;
  without the map it would be the 16 M cap.

- ~~A gaussian does not know it is a thin wall.~~ It does now (next section).

### A thin wall is its own transparency

A radiance bake of the glass bird came out opaque and bright: every gaussian
kept, in degree-3 harmonics, what the path tracer saw through it, and the
grass behind a card is far finer than nine or sixteen coefficients. More
paths do not help; it is what the harmonics can hold.

But a thin wall sends what it does not reflect straight on, and that is what
blending a gaussian over what stands behind it already does -- through any
number of cards, on both routes. So:

- **The conversion** gives a thin-walled glass's gaussians the opacity that
  makes a card of them stop what the sheet reflects head on, `2R/(1+R)`
  (0.0769 at 1.5), and no more: `scene::thinWallOpacity`,
  `-ln(1 - F) / (2 pi sigma^2) + 1/255`, for the overlap of a gaussian a cell
  `sigma` cells wide and for the tails a faint splat is not drawn out to.
  Measured before it: each gaussian at the sheet's own 0.077 made a card of
  0.375, and at the overlap alone one of 0.050.
- **The file** says which gaussians those are,
  `primvars:athenea:splat:thinWalled`, which rides on the transmission in a
  record (plus two) and lands in the spare byte of the `pbr` word.
- **The frame** shades a thin wall as its reflection alone -- no transmitted
  term, the blend is that -- scaled by `1/R0`, what the sheet reflects over
  what its gaussians were given.

A card laid out as the conversion lays one out composites to **0.0774**
rasterised and **0.0778** traced, against 0.0769 (`a thin wall's card is as
opaque as the glass sheet reflects, on both routes`), with each gaussian a
few pixels wide; at one pixel the rasteriser's antialiasing filter widens a
splat and the card reads 0.068.

The glass sparrow at t = 30, 4.1 M gaussians with a transfer (238 s): 23 ms
rasterised, 214 ms traced, against 125 s for the mesh path traced. The grass
shows through its wings and feathers on both routes.

- **Seen edge on, the traced route stops more** than the rasteriser: a card
  at ten degrees reads 0.79 against 0.60. Both rise, as a sheet's Fresnel
  does, but they do not agree.
- ~~The solid body's exit ray walks the whole cloud's tree, thin cards
  included.~~ `rt_frames` marks a thin wall in the frame's spare lane and
  `glassExit` passes it by: a feather card over a glass body is not the
  body's far face.
- ~~About 8% of the bird's gaussians get no transfer.~~ They were the ones on
  OpenPBR cards: `MaterialCompiler::cutsOut` read `opacity` and
  `opacityThreshold` and not OpenPBR's `geometry_opacity`, so those cards
  were never cut out, their mask a fractional opacity. Read now; 4 115 839 of
  4 116 403 get one (`an OpenPBR geometry_opacity of zero cuts the surface
  away, on both routes`). The mesh reference changed with it: its cards are
  cut now, and it came out darker and denser than the cloud.
- **A cut-out cast the shadow of its whole card**, since no shadow ray
  asked the material: a feather card, a leaf, cast rectangles, and a dome's
  light through a bird of them was blocked where it should pass. The path
  tracer asks now (next section); the rasteriser does not yet.

### A shadow asks a cut-out whether it is there

The path tracer's kernel evaluates materials at one site on purpose: every
site is a copy of every material once the compiler inlines it, and four had
the chess set's pipeline build for over four minutes against six seconds. So
a shadow ray does not evaluate the material. It asks for the **opacity
alone**: `ClosureVariant::Opacity` compiles a cut-out's graph once more
with a surface node that emits its opacity and nothing else -- the BSDF and
everything only it reads are dead code -- into `gAtheneaOpacity`, leaving the lobe
stack alone; `MaterialPrograms` dispatches them as `evaluateOpacity`. Only
materials that cut are compiled twice, and a frame with none keeps the
first-hit shadow ray it always had (`path.shadowCutouts`).

With cut-outs about, the inline route walks nearest first, as linking
already did, and past a surface a lot says is not there (`shadowPassesThrough`,
the same coverage the path's own rays draw, 64 surfaces at most). The
pipeline route asks in its any hit, and traces shadows `FORCE_NON_OPAQUE` so
that the any hit runs: the instances are built `ForceOpaque`, which had also
kept a light's link from ever being asked there.

A card at `geometry_opacity` 0 over a lit square, shadows on: relMSE **0**
traced against the square alone.

- **The rasteriser does not ask.** Asking from inside its light loop wrote
  rows of black on Metal -- a few per cent of the pixels a shadow ray met a
  card from, different ones each run, gone the moment the call was taken out
  and not cured by `[noinline]`, by ending the ray query first or by leaving
  the camera's footprint out. It is the miscompile the lobe stack's note in
  `MaterialShading` already records. The code is there, switched off.
- ~~A cloud converted with `--skinned` alone drew in a different pose.~~ It
  drew in its bind pose at every time. The rig's influences are a record
  each, and `Engine` checked them against the splats validation *kept*: the
  glass sparrow's faintest feathers fall under 1/255 of opacity (217 887 of
  9 536 402), so its whole rig was dropped with them, in silence. The decode
  now writes, for each splat it keeps, the record it came from
  (`GpuSplats::origin`), and `CloudLoader::keptOnly` packs any per-record
  array to the kept splats on the device -- the influences, and a baked
  visibility's `partOf`, which had the same check; `toRecords` lays a
  baked `partOf` back out a record each for the file, -1 where a record was
  dropped (`a cloud a skeleton carries still moves when validation drops
  some of its gaussians`).

### Levels of detail for a cloud that moves

A `.athc` merges a cloud's cells into coarser ones, sorted by Morton code over
positions that do not move; a skinned cloud's cannot be merged -- a cell that
took wing and body would not know which joint to move with -- and the LOD
path never carried a rig. What a skinned cloud can be is **converted again**
at a coarser cell, each level a cloud with its own rig: a discrete LOD.

- `athenea mesh2splat --lod-levels N` converts at the resolution asked, then half
  of it, and so on, each level to `<name>_lod<n>.usdc`, and writes `-o` as the
  stage that draws them (`usd::writeLodAssembly`): the finest at
  `/World/Splats`, so a rig that names that prim still finds it, each level
  with `athenea:lod:group` and its `athenea:lod:cell`.
- `Engine::lodLevelsFor` draws, of each group, the coarsest level whose cell
  spans no more than `athenea:lod:threshold` pixels (1) at the nearest point of its
  bounds -- posed bounds where a skeleton carries it -- and the finest where
  none does. The others are not drawn, and **not posed** either: a level's
  joints are kept (`deferredPose`) and applied when a view draws it.

The sparrow's clips at 1100, 550 and 275 cells across (5.9 M, 1.5 M and
0.4 M gaussians), each with its per-part visibility baked
(`SparrowClipsLod.usda`), playing at 1600x1000:

| | without | with |
|---|---|---|
| filling the frame | 49.6 ms | 51.5 ms (the finest: its cell is already ~2 px) |
| four times as far | 71.3 ms | **24.7 ms** (level 1), no difference to see |

`a cloud in levels of detail draws the coarsest one whose cell a pixel
holds`: 4096 gaussians drawn near, 256 far.

- **Memory**: every level is resident, and each carries its own per-part
  visibility (81 MB each for the sparrow, the same fields three times).
- **A level switch is a step**, not a blend: at the distance where it happens
  the cells differ by a factor of two and the switch is not seen, but nothing
  cross-fades them.

### Where a rasterised frame of splats went, and what came off it

`ATHENEA_STAGES=1` times each stage of a rasterised frame (each waited for, so
the frame is slower for it). The sparrow's clips, 5 887 323 gaussians at
1600x1000, playing, before and after:

| stage | before | after | what changed |
|---|---|---|---|
| per-part visibility | 12.7 ms | 8.0 ms | a part's posed-to-rest map made once a frame (`visPartFrames`), not two 3x3 inverses a gaussian a part a light |
| depth sort (24 bits) | 11.3 ms | 6.6 ms | the tiled radix route |
| tile sort (8.8 M pairs) | 8.4 ms | 6.0 ms | the same |
| blend | 13-25 ms | 8-13 ms | a tile's records read once into group memory for its 256 pixels |
| project, emit, counts, commit | 13.6 ms | 13.4 ms | -- |
| **frame, playing** | **102 ms** | **49 ms** | with the cloud no longer uploaded every step |
| frame, standing | 59 ms | 42 ms | |

Elsewhere: the HDR pawn rasterised 12.6 -> 10.7 ms, the film's sparrow 76 ->
57 ms; meshes and the traced route do not go through any of it.

**The tiled radix route** (`radix_block.slang`) is the reduce-then-scan form
of the same stable LSD pass: a group of 256 counts a tile of 1024 keys in
group memory, a prefix sum over the counts laid out digit-major says where
each tile's share of each digit starts, and the group sorts its tile by the
digit with eight one-bit splits and writes each pair to its start plus its
rank. Ten million 32-bit pairs: **12.4 ms**, against 39.9 ms for the chunked
passes, which walk a chunk a thread and keep every histogram in device
memory. It is the first thing in the engine that uses group memory; it uses
nothing else a backend might lack (no wave operations, no device atomics),
takes over from 65 536 elements up, and the chunked passes stay for smaller
sorts, for `working()`'s probes and under `ATHENEA_PORTABLE_SORT=1`. The sort
test's sizes and patterns, 24 to 64 bits, pass on it unchanged.

**The render scale does not help this scene**, and now it is clear why: a
frame of six million gaussians is spent on the gaussians -- the pixels cost
the blend only.

### A slower engine, measured: a time step was a new cloud

"The engine is much slower on the Mac since the HDRs" was measured before it
was believed. A release of `6a53e29` (20-09, before any of the prepared sky)
against the current one, same Mac, same commands, 1600x900:

| | before | now |
|---|---|---|
| Kitchen_set lit, raster / rt | 49.5 / 400.7 ms | 50.8 / 400.9 ms |
| the HDR pawn, raster / rt | 11.6 / 36.4 ms | 12.6 / 42.8 ms |
| the same with the camera turning (`ATHENEA_VIEW_ORBIT`) | 11.6 / 35.7 ms | 12.7 / 42.6 ms |
| the film's sparrow, raster | 75.6 ms | 76.3 ms |
| the sparrow's clips, playing | 97.5 ms | 102.4 ms |

No regression, but the last row was worth reading: the same clips stood
still take **59 ms**, and the render scale changes nothing (800x500 is no
faster than 3200x2000). A sample of the playing viewer's main thread put 27%
in Hydra's sync decompressing the cloud's arrays and 16% uploading all of it
again: a skinned cloud's joints have samples, so every step dirties every
primvar, and the crate file keeps integer arrays compressed -- read again,
the same values arrive in a new buffer, and `commit`'s identity took the
cloud for a new one. `HdAtheneaParticleField` now keeps what it read of the
arrays whose data source says they have no samples (`_held`), asked without
reading a value; the cloud keeps its buffers and is not uploaded again.
Playing: **71.7 ms**, the remainder being the pose and the per-part
visibility read a step does need. `a skinned cloud stepping through time is
uploaded once` holds it (three steps, one upload; three before).

- **An edit to one of those arrays in a session layer** is not seen until the
  prim is read anew: what is kept is taken as the file's. Nothing edits them
  that way today.
- **The sky is what did get slower**, and not per frame: since it follows the
  image (2048 a side for a 4k HDRI), turning or changing it rebuilds the whole
  chain, 180 to 400 ms a change, every frame of a drag. Not done yet.

### The sky as a list, in the viewer

`athenea view` gained a **Sky** combo a dome light. It exists because the prepared
sky is the one thing in this engine that is hard to believe without turning it:
a reflection either follows the image or it does not, and swapping the image is
how you find out in a second rather than in a render.

- **Three calls on `StageRenderer`**: `domes()` says which domes there are and
  what each holds, `setDomeTexture` puts an image on one, `setDomeRotation`
  turns it. All three write the session layer, as the variant selection does,
  so the stage on disk is never edited and the test checks that it is not.
- **Both dome schemas.** UsdLux has `DomeLight` and, since 24.11,
  `DomeLight_1`, the one carrying `poleAxis`. They are different prim types
  and `UsdLuxDomeLight` matches only the first, so the first build of this
  showed no sky at all on the stages this project actually uses. The match is
  by prim type name and the texture by its attribute name, as Hydra folds the
  two together.
- **The list needs no option.** `--hdri <dir>` is repeatable, and on top of it
  the folder each dome's own image sits in is offered, which is usually the
  library it came from. A dome that is only a colour brings no folder, and
  then the option is the only way.
- **It costs one frame, not every frame.** The prepared environment is keyed
  on the dome's record, so a swap or a turn rebuilds the harmonics and the
  prefiltered chain on the next frame: about 180 ms under a 4k image. Dragging
  the slider pays that per frame, and that is written down rather than hidden.

## A matte id is a selection, and therefore a handle

Picking a cloud pixel already answered with a Cryptomatte id, because a
gaussian writes no `primId` and the matte is the only name a pixel of splats
has. That id is more than a name: every gaussian carrying it came from one
prim, and therefore stood on one material when `athenea mesh2splat` walked it. So
the id keys a table, and clicking is how a host addresses it.

- **What they carry, counted on the device.** `render::measureSplatId`
  dispatches `splat_id_stats.slang` over a cloud and brings back ten uints:
  the count, and the sum, minimum and maximum of the metallic, roughness and
  transmission bytes. The bounds matter as much as the mean -- a material
  whose roughness came out of a map gives each gaussian its own, and a panel
  showing one number would be claiming they are all the same. The panel says
  `roughness 0.183 (0.080 to 0.310)` when they are not.
- **What the frame says instead.** `render::SplatOverride` is a row keyed on
  that id: metallic, roughness, transmission and a tint. Two float4 a row in a
  buffer both shading kernels read (`splat_override.slang`), looked up by a
  linear walk over the handful of rows a frame holds; `overrideCount` of 0 is
  the whole cost when nobody has touched anything.
- **Negative means "leave it".** An untouched slider sends -1 and the
  gaussian's own value stands, so tinting a prim does not flatten the
  roughness map underneath it. The panel starts each slider at the measured
  mean, which is the value that changes nothing when you nudge it.
- **It is read, never baked.** The cloud's `pbr` and colours stay as the file
  wrote them; a row is an opinion the frame holds over them and clearing it
  puts the file back. Nothing on disk is touched, which is why there is no
  save: it is look-dev, not an edit of the asset.
- **`SplatEdit` was not extended.** It is openFXplayer's, rule for rule, so a
  scene moved between the two renders the same; a selector it does not have
  would break that. The table is its own thing beside it.

The test builds two clouds with an id each, measures one, says otherwise about
it, and checks the other prim's pixels are bit for bit what they were:
`athenea_render_tests "[override]"`.

### Two ways it was unreachable, and what they cost

Shipped, it did nothing, which is worth writing down because neither reason
was in the feature.

- **The matte was only kept when it was being shown.** The viewer asked for
  the three `CryptoObject` layers when the Output combo was on the matte, and
  nowhere else, so a click on the default Colour output came back with a
  `cryptoId` of 0 and the panel never drew. It now asks for them on every
  raster frame: a gaussian writes no `primId`, so the matte is not a view, it
  is the only name a pixel of splats has. The colour is unchanged to the bit
  (`[crypto]` checks that) and the frame goes from **13.7 ms to 15.5 ms** on a
  730 000-gaussian cloud at 1600 x 900. Under `rt` nothing is asked for,
  because the traced splat route writes no matte, and the panel says so with a
  button back to Raster rather than going quiet.
- **The panel was below the fold.** What a pixel is sat at the bottom of the
  View panel, under the timeline and the counters, behind a scrollbar nobody
  drags after clicking. It is its own window now, open from the start, and it
  says what to do when nothing is picked.

### Not done

- **A cloud without a Cryptomatte** cannot be addressed this way: a capture
  from a scanner has no prim to have come from. `mesh2splat` writes the ids,
  and only a converted cloud carries them.
- **Opacity is not in the table.** What a volume grade already does
  (`SplatEdit`) it keeps doing; this is the material, not the presence.
- **The mesh still samples the image** rather than reading the prepared sky.
  It has parallax already, by sampling, and nothing was gained by changing it;
  what it would buy is speed, not capability.
- **A fifth dome** is drawn as its colour: four slices are prepared.
- **The DFG term stays analytic.** `environmentBrdf` (Lazarov) in the splat
  path and `ggxDirectionalAlbedo` (MaterialX's fit) in the surface lobes. A
  tabulated 64 x 64 LUT would have to agree with both or the two paths
  separate, and neither has been measured to be insufficient.
- **A capture is untouched.** The environment reaches a cloud only where the
  cloud asked to be relit, and a capture's colours are radiance already.

## The density is a mesh's own, not the stage's

`--resolution` was cells across the longest side of the box of everything
converted, in world space, and a triangle's cell was that side's range over
the resolution -- or rather the wider of its projection plane's two ranges,
which is mesh2splat's `orthogonalUvs`. Two things followed that a car made
plain. A 16-unit floor in the hero stage made the car's cell 0.031 where the
car alone gets 0.0037 at the same resolution: the floor took 262 144 of the
375 199 gaussians and the bumper 762. And a panel facing forward, whose
projection plane is the car's short side, walked a cell 2.5 times finer
than the hood's, whose plane is the long side.

`--density per-mesh` measures each mesh over its own box: `resolution`
cells across the mesh's longest side, the same side for every triangle of
it (`cellByLongest`, so a mesh has one cell), held in the world between
`--cell-min` and `--cell-max`. Left at 0 those derive from the model's
cell: no mesh coarser than it, none finer than an eighth of it -- a density
ratio of 64 between the finest part and the coarsest, which is what a floor
and a car are apart, and which keeps a bolt from walking a grid sixteen
hundred times finer than the body's. The same number for both is one cell
for the whole stage. A budget-derived floor would be better; it needs a
count-only pass a mesh (a `countOnly` parameter, additive), and is not done.

How it is done: the per-chunk extents `streamBoundsChunks` already writes
for the model's box are folded once more, one thread a mesh, over each
mesh's run of chunks (`bounds_reduce_slices.slang`; `bounds_reduce` has six
other callers and gets no field they never set). The host attaches the
mesh's box in place of the model's, and the kernel clamps the cell in world
units and recomputes `perCell` only where a bound bites, so an unbounded run
keeps `1 / resolution` to the bit and its cell centres where they were.
Three effect parameters, `cellMin`, `cellMax`, `cellByLongest`, additive,
as the ABI asks; the uniform block grows a row (240 to 256 bytes), which is
the plugin's own.

Measured, `MustangHero.usda` (the car and its 16-unit floor) at
`--resolution 512 --no-bake --max-splats 20000000`:

| mode | total | bumper `Object_199` | hood `Kapoot` | floor | reruns |
|---|---|---|---|---|---|
| per-model | 375 199 | 762 (cell 0.0313) | 2 326 | 262 144 | 1 |
| per-mesh, defaults | 934 338 | 3 082 (cell 0.0039) | 20 279 | 262 144 | 5 |
| per-mesh, `--cell-min` = `--cell-max` = 0.03125 | 375 157 | 764 | 2 339 | 262 144 | 1 |

The control row reproduces per-model to 0.01%: what separates the two is
the orientation effect alone (764 against 762, 2 339 against 2 326). With
the defaults every part of the car meets the floor -- its own cell would be
0.0014 -- and comes out eight times finer than under the floor's grid while
the floor keeps exactly its 262 144. The test is two planes twenty times
apart (`tests/data/two_planes.usda`): with one grid the small one gets 4
gaussians, over its own box 144; and the aofx cases pin the clamp on a quad
(a floor of a half gives two by two; a 0.1 quad over its own box gives 272
under the ceiling and 6 under a floor of 0.05, and 4 measured over the unit
box it used to share).

## A gaussian that carries how much sky reaches it

A baked cloud's harmonics held `litBody`: the light that arrived at each
gaussian under the dome that was in the scene when it was baked. Put that
cloud under another sky and the light it carries is the old sky's -- the
shape of it, the colour of it, the window that was behind it. The
infrastructure was warm (a bake traces whatever list of rays it is handed)
and the data was not.

`kTransfer` is a third variant of the path traced kernel beside `kBake` and
`kNoBake`, and where the others gather radiance it gathers **how much of an
environment would arrive**:

```
T_k = (1 / pi) * integral over the hemisphere of V(w) Y_k(w) cos(theta) dw
```

and a frame recombines it with the sky's own coefficients, `<T, L_SH>`, which
is the nine numbers of `technique::Environment` from the section above. The
two halves of the product are the two halves of the section: one prepares a
sky, the other prepares a surface, and neither knows the other until a frame
puts them together.

**The estimator is the sampling that was already there.** A bake's rays are
stratified over the hemisphere; under transfer the first vertex is not the
material's but that direction, taken as a white Lambert, with density `1/2pi`
against an estimator that wants `cos/pi` -- so a sample that escapes is worth
`2 cos(theta) Y_k(w)` and one that does not is worth nothing. There is no fit
and no matrix: the accumulation **is** the projection, because the direction
was drawn from a density the estimator knows. `bakeGram` and `bakeFit` stay
where they were, for `litBody`.

**Two halves in one pass, for nothing.** A path that escapes from the first
vertex is direct transfer, a scalar: it is geometry alone and has no colour.
One that escapes after bouncing carries the colour of what it bounced off,
which is a cloud's static GI, and that one is rgb. The same rays answer both,
so the indirect half costs no ray -- what it costs is the file (see below).
The direct half is always written; `--indirect` (on by default, `--no-indirect`
to drop it) decides whether the other is, and `athenea:splatTransferIndirect`
decides per frame whether it is summed, so the same cloud can be looked at
with and without its bounce light without being baked again.

**Where it travels.** Nine f16 a gaussian for the direct half (20 B), 27 more
for the indirect (72 B in all: 67 MB on a 934 338-gaussian car). In the file
they are `primvars:athenea:splat:transferDirect` (elementSize 9) and
`:transferIndirect` (27) of `AtheneaSplatLightingAPI`; inside they follow the
Cryptomatte id's road, record by record -- `streams.slang`, `splat_decode`,
`GpuClouds`, `PrimData`, `ParticleField`, `Export`. **There is no mode token.**
In this tree the presence of the data is the mode, as `hasPbr` and `hasCrypto`
are, and a token would be a second copy of the truth that nothing checks.

**What the frame does with it.** A relit gaussian with a transfer and without
a baked body reads

```
body   = albedo * (1 - metallic) * <T_dir + toggle * T_ind, L_SH>
polish = environmentBrdf(f0, roughness, NdotV) * prefiltered sky at the mirror
```

so a metal has no body at all and goes entirely to the analytic side with
`f0 = albedo`, which is what a transfer vector cannot hold and a prefiltered
map can. A translucent gaussian keeps the split it always had -- `1 - t` of
the body facing the light and `t` of it facing away -- and the half that
faces away is answered by the sky's harmonics with no occlusion, because a
transfer was measured over the hemisphere the normal faces alone. Without
that half a glass ball with a transfer is a dark shell. Analytic lights are summed as before: a transfer answers for the
environment, not for a spotlight.

### How it is checked

- **The closed form.** A point with nothing above it transfers the clamped
  cosine lobe, whose bands are known exactly: `pi`, `2pi/3`, `pi/4`. Over 64
  points: **0.2820 / 0.3258 / 0.1581 against 0.2821 / 0.3257 / 0.1577**, and
  the same transfer dotted with a constant sky of 0.7 reads **0.6998**. A
  measure, a stratification or a cosine weight wrong in any way reads
  something else here, and no sky can hide it, because a transfer gathers no
  light at all.
- **The corner** (`tests/usd/test_usd.cpp`, `[transfer]`): a floor, a wall
  0.1 away from the first of 32 points, and the transfer recombined with a
  constant sky against a radiance bake of the same points under that sky.
  Three cases, and the middle one is the one that matters:

  | | the transfer | the bake | worst point |
  |---|---|---|---|
  | no wall, 1 bounce | 0.6000 | 0.6000 | 0.0 % |
  | a black wall, 1 bounce | 0.3091 | 0.3039 | 2.5 % |
  | a white wall, 3 bounces | 0.5095 | 0.5325 | 4.3 % |

  The black wall occludes and returns nothing, so the direct half must be all
  of it; the white one returns plenty, and only the indirect half can account
  for the difference.
- **The file** carries it: a cloud converted with `--transfer`, written and
  read back, decodes to the coefficients it was baked with.
- **The basis against itself** (`athenea_scene_tests "[sh]"`): `sh.slang` holds
  the harmonics twice, `evaluateRest` for a renderer and `shBasisValue` for a
  projection, and says the two must agree. A transfer is the second thing to
  project onto them, and nothing checked it until now: one coefficient set to
  one at a time, over 64 directions, read back through the renderer's side.
  960 readings, none off by more than a millionth.
- **The frame is linear in the sky**: the same cloud under a dome of one and
  of two reads **1.004184** and **2.008368**, exactly twice.

### Measured: the pawn under two skies (M5 Pro, release)

The chess pawn converted once with `--transfer` (729 939 gaussians, 512
cells), drawn under `autoshop_01_4k.hdr` and under the same HDRI turned 130
degrees, against the same pawn as a mesh path traced under each -- same
camera, 128 paths, 3 bounces. Patch means, linear, rgb averaged:

| where | sky | the cloud | the mesh |
|---|---|---|---|
| the stem | as it is | 0.0331 | 0.0331 |
| the stem | turned | 0.0318 | 0.0331 |
| the cone | as it is | 0.0630 | 0.0681 |
| the cone | turned | 0.0614 | 0.0641 |
| under the collar | as it is | 0.0723 | 0.0576 |
| under the collar | turned | 0.0642 | 0.0550 |

The first thing the table says is the one the whole branch is for: **the file
did not change between the two rows of each pair**, and it followed the sky.
The second is where it does not: in the pocket under the collar the cloud is a
quarter bright. That is not the transfer -- the same cloud relit with no
transfer at all reads 0.0746 there, and the transfer's own occlusion is what
takes it down to 0.0723 -- it is the analytic reflection, which is added with
no occlusion whatever. Specular occlusion is named in the plan and not
written, and this is the measurement that says what it is worth.

The indirect half is worth **0.15 %** on an object this convex, and 79 MB of
the file (146 MB against 67). It is written by default because it costs no ray
and because a cloud in a room is not a pawn on a table; `--no-indirect` and
`athenea:splatTransferIndirect` are there for when it is not worth its size.

### What the corner found in the bake

The corner's first test failed by a third, and the transfer was not what was
wrong. **A bake encoded every sample to sRGB before fitting it.** The samples
of one point are estimates of the same radiance, so their mean is the
estimate; the mean of their encodings is a different and smaller number,
because the curve is concave. Where every sample agrees -- an open sky, which
is what every bake test had -- the two are identical. Where they disagree,
which is exactly what an occluder does to them, the fit sinks towards the dark
samples: the point beside the wall was fitted at **0.2098 where the light on
it is 0.309**, a third under, and with next event estimation alone at 0.1373.

The fit is now taken in linear light and carried into the cloud's space once,
at the end: the constant term through the curve, the bands that shape it
through its slope there (`srgbSlope`, a first order change of variable, exact
at degree zero). Every contact region of every baked cloud was dark by this,
and it is the same class of error as the one the section above found in the
sky -- a quantity averaged in the wrong space.

### Not done

- **The `.athc` and the LOD lose it**, as they already lose `pbr` and the
  Cryptomatte ids: `packed()` copies positions, shape and sh. A streamed cut
  falls back to the body it has.
- **A skinned cloud is not baked** and so is not transferred either; the
  analytic route (a zonal cosine rotated to the normal, cut by the per-part
  visibility field) is named in the plan and not written.
- **`athenea transfer` for a cloud already converted** needs a normal a gaussian,
  which a file does not keep. It would have to come from the short axis or be
  written at conversion.
- **A cloud still does not light a mesh.** Splats do not emit inside the
  integrator, so the GI that a transfer captures is mesh to cloud only.

## The glass knob that reached nothing

`athenea mesh2splat --glass-opacity` says what a fully transmitting material still
stops, and a window is what it is for. It was sent to the effect as
`minOpacity`, which is the name of an **edit** -- the threshold a frame culls
a splat below (`athenea:edit:minOpacity`) -- and the conversion has no parameter
of that name, so the number went nowhere. Every glass came out as opaque as
its material's opacity, which is why a chess pawn's ball was a mint shell
with a room behind it that never showed.

The effect gains `glassOpacity` in the pad word the uniform block already had
(256 bytes, unchanged), defaulting to **1**: a bundle that does not set it
converts exactly as it did, which is what additive means here. `athenea
mesh2splat` sends its own default of 0.6, so a transmitting material now
keeps 60 % of its opacity, and `--glass-opacity 0.12` is a ball you can see
the workshop through.

What the kernel does is `opacity * mask * lerp(1, glassOpacity,
transmission)`: it is scaled by how much the material transmits, so a
material that transmits nothing is untouched whatever the knob says. The
colour still goes towards the transmission colour as it did. A translucent
material is not a transparent one -- the light comes through scattered and
the body stays -- which is why the default keeps everything and the caller is
the one who says otherwise.

`athenea_aofx_tests` converts the same glass quad twice, at 1 and at a quarter,
and reads the opacity back off the records.

## The index a gaussian bends by

A cloud's transparency was coverage and nothing else: a ray goes through the
gaussian, the colours mix, and it carries on in the direction it had. So a
glass ball let the room behind it be seen and never turned it, which is the
one thing that says glass. What was missing is not a ray -- it is a direction.

`primvars:athenea:splat:ior` gives a cloud an index. Where it is above one, the
transmitted half of a gaussian's body stops being `envIrradiance(-n)/pi`, the
cosine average of everything behind, and becomes `envRadiance` along the
direction Snell gives at that gaussian's own normal, read out of the same
prefiltered sky the reflection reads. It costs one `refract` and one lookup,
no ray, no traversal, and it lives in `splat_relight`, which both routes
share, so the rasteriser and the tracer bend alike.

**What it is not.** It bends at the face the ray enters and at no other: a
cloud has no far face to find without tracing, and a thick lens has two. It
shows the sky, so what stands between the glass and the sky -- the car parked
behind the ball -- is not in it. And it is one index a cloud, not one a
gaussian: `transmission` is per gaussian and is what decides where the index
applies, which is enough for an object whose glass is one glass.

Measured against the polygon: the pawn's ball drawn as gaussians with
`ior = 1.5` displaces what is inside it the way the mesh's does and by less,
because the mesh bends twice and reflects totally at the rim. That gap is the
ray, and it is named in the next section's debts.

`athenea_render_tests "[refract]"` holds the bend itself to Snell over 64
incidences: the sine leaves at `sin(theta) / n`, on the far side, in the plane
it came in, and an index of zero gives back the straight direction that was
there before.

### The prepared sky was too coarse to reflect anything

A cloud's reflection came back matte beside the mesh's, and the reason was the
map: `kEnvBaseSide` was 64, about a degree a texel, where a mesh samples the
sky's own 4k image along its lobe. No prefiltered map can be sharper than its
base, so the cloud's mirror was a blur whatever its roughness said.

At 256 a texel is a quarter of a degree and four domes take 2.8 MB. Measured
on the pawn's gold collar against the mesh path traced (RMS over the patch):
**0.3055 at 256, 0.3012 at 512**, and 512 costs four times the memory for
1.4 %. So 256 is where it stays, and the number is written here so nobody
measures it twice.

What the base does **not** buy is the glint: a prefiltered environment is a
split sum, and a lamp reflected at low roughness is a hot spot the map has
averaged away. The mesh's speckle beside it turned out to be the path
tracer's own noise -- at 1024 paths it is gone -- but the sharp highlight on
the gold is not, and that one is the approximation, not the resolution.

The test that read 0.863 for the roughest level now reads 0.997, and the new
number is the right one: a normal facing a hemisphere of white receives `pi`
of irradiance, so the convolution over it is one. The old reading was the
2 x 2 map that level used to be, blending four texels pointing at four
corners of the sphere.

### One DFG fit, where there were two

How much of an environment a GGX lobe sends back -- the DFG term of the split
sum -- was fitted twice: Lazarov's in `splat_relight`, which a gaussian
reflected with, and MaterialX's `ggxDirectionalAlbedo`, which the surface
lobes reflect with. The plan named the risk; the measurement is worse than the
risk. Over a grid of roughness and `N.V` the two come apart by **76 %**, worst
where a surface grazes (0.0166 against 0.0711 at roughness 0.97, `N.V` 0.03),
and 30 of 256 points are beyond a third. A cloud and the mesh beside it under
one sky could not have agreed at any silhouette.

A gaussian now reflects with the surface's own fit: `environmentBrdf` calls
`ggxDirectionalAlbedo` with `alpha = roughness^2` and `F90 = 1`, which is
exactly the same quantity. The module order allows it -- `microfacet.slang`
declares no binding, and what a module below `material` may not do is import
the texture table, not the arithmetic. On the pawn's collar the change is
worth nothing measurable (RMS against the mesh 0.2995 before, 0.2997 after),
because that patch is nowhere near grazing; what it buys is every rim in every
frame, and one definition where there were two.

### How close the pawn gets, and where it stops

The pawn converted with `--transfer --normal-map-turns --glass-opacity 0.12`,
given `ior = 1.5`, against the same pawn as a mesh path traced at 1024 paths
under the same sky and camera. RMS over each part:

| | RMS against the mesh |
|---|---|
| the base | 0.020 |
| the cone | 0.030 |
| the collar (gold) | 0.059 |
| the ball (glass) | 0.372 |
| the whole frame | 0.095 |

So the marble is within a few per cent of the path traced mesh and the gold
within six, and **the error is the ball**. It is not a setting: a sweep of the
index over 1.2 to 2.4 and of the ball's opacity over 0.10 to 0.18 moves the
ball's RMS between 0.402 and 0.411 -- nothing. What the mesh's ball does is
bend twice and reflect totally at the rim, which turns the room over inside
it; a single bend at the face it enters cannot make that image at any index.
That one is the ray, and it is the third item on the list below.

### The specular, worked until the numbers stopped moving

The pawn's marble disc read a tenth bright against the mesh, and the specular
was where it came from. Four things were done to it, each measured against
the same mesh path traced at 1024 paths (RMS over the patch, under the HDRI):

| | the disc | the cone |
|---|---|---|
| where it started | 0.0588 | 0.0301 |
| the surface's own lobes | 0.0592 | 0.0300 |
| the reflection occluded by the transfer | 0.0412 | 0.0312 |
| and converted at 1024 cells | 0.0411 | 0.0300 |

**The lobes.** A gaussian now answers what `lobes.slang` answers: a dielectric
of `F0` 0.04 with its Fresnel taken exactly, a conductor whose own colour
stands at both ends of the fit, both scaled by `ggxEnergyCompensation` -- the
light that leaves a rough surface after bouncing between its microfacets --
and the two mixed by the metalness, as the `mix` node of
`athenea_usd_preview_surface.mtlx` mixes them. It is worth nothing measurable on
this frame, and it is what makes the next measurement mean something: under a
sky of one colour the cone now reads 0.2575 / 0.2674 / 0.2662 against the
mesh's 0.2585 / 0.2674 / 0.2662, which is a fraction of a per cent. **The
energy is right; what was left was direction.**

**The opening.** The reflection was added with nothing in front of it, so a
gaussian in a pocket reflected the whole sky. It needs no ray: the transfer's
constant term *is* the visible fraction of the sky (`A0 Y0 / pi` = 0.28209
over an open hemisphere), and Lagarde and de Rousiers' fit turns that into
how much of it a lobe of a given sharpness may see. The disc drops 30 %.
`athenea_render_tests "[openness]"` holds the fit to what it may never get wrong:
whole under an open sky, nothing under a closed one, never falling as the
opening grows, never leaving [0, 1].

**A dome with no image is prepared too**, and it was not. A constant sky is
what the closed form answers exactly *for a point that sees all of it*, and a
gaussian carrying a transfer does not: its road to a sky is the harmonics.
Left out, a cloud under a plain dome threw its own visibility away and lit
every pocket as if it were open -- 0.0654 on the disc where the prepared sky
reads 0.0538.

**Density is not shading.** Converted at 1024 cells rather than 512 the disc
improves 14 % under a flat sky and **nothing at all** under the HDRI (0.0412
to 0.0411). What density buys is the crevice between the disc and the stem,
which 512 cells fill with splats from both of its walls; what it does not buy
is a reflection pointing the right way.

What is left on this pawn, in order: the crevice, which is coverage and a
conversion knob; the glint of a lamp at low roughness, which a prefiltered
sky averages away by construction; and the ball, which is the ray.

### A ball is a lens, and a lens has two faces

One bend shows the room behind the glass without turning it over, which is
the one thing a glass ball does. The second interface is where that happens,
and finding it takes a ray -- the first ray a gaussian's shading has ever
cast.

**Where it walks.** The cloud's own LBVH, the compute route's, for the
instance the gaussian belongs to and no other: what is on the far side of a
glass object is that object. Which gaussians are glass travels in the spare
lane of the frame `rt_frames` already writes, so there is no table to keep in
step. The walk takes the farthest glass particle within reach, bends there
from glass to air, and hands the direction to the shading; past the critical
angle it reflects instead, which is what makes the rim of a ball a mirror.

**What passes is what did not reflect.** The reflection is added whatever the
surface does, so a transmitted half at full strength is more light than
arrived: the ball came back half again as bright as the mesh and its rim two
and a half times too hot. `1 - ggxEnvDielectric` is the complement, which is
what `lobes.slang` writes for a transmitting lobe, and
`athenea_render_tests "[glass]"` holds the sum of the two below one.

**A shell shows its image twice.** A normal is turned towards the eye before
anything is shaded, so the far face of a ball pretends to be a near face and
hands over the same sky again. A solid's far face now transmits nothing of its
own. On an opaque shell it changes nothing measurable, because the near face
covers it; on a transparent conversion it is the difference between one ball
and two.

**Glass is shaded twice.** What a ball shows is mostly not the sky but what
stands behind it, and for a cloud that is its own particles: everything is
shaded, then the glass again with the rest already written, so the ray that
leaves can be walked on until it meets something that is not glass and take
its colour.

### What the glass is worth, measured

A glass sphere alone under the HDRI -- 4 608 quads, converted at 700 cells --
against the same sphere as a mesh with a MaterialX `standard_surface` of
transmission 1, path traced at 512 paths. Patch means:

| | the cloud | the mesh |
|---|---|---|
| through the centre | 0.167 | 0.155 |
| halfway out | 0.233 | 0.174 |
| at the rim | 0.159 | 0.183 |

Where the ray goes straight through, the two agree to **7 %**. Halfway out the
cloud is a third bright and at the rim an eighth dark: what a real ball does
there is reflect totally and bounce inside, and one bend in and one out is not
that.

On the chess pawn the ball's RMS goes from 0.380 to 0.397 and every other part
is untouched. **The metric gets worse and the picture gets better**, and the
reason is in the numbers above: a pawn's ball shows its own collar and the
floor, and what a cloud can bend towards is the sky and its own particles. A
first attempt to compare against a `UsdPreviewSurface` with `opacity = 0` said
the opposite; that material is coverage, not refraction, and the mesh it drew
was a see-through shell. The reference has to be a transmitting
`standard_surface`.

**What it does not do.** A mesh behind the glass is not in it. The bright ring
of total internal reflection is not in it. And the walk is the compute route's
alone: on the hardware route there is a top level structure and no walk here,
so its glass bends once, as the rasteriser's does.

## A gaussian reflects the cloud it stands on

The polish read the prepared sky along the mirror direction, which is right
for a gaussian with nothing in front of it and wrong for one facing its own
asset: a collar under a glass ball reflects the ball, not the ceiling. The
ray that finds it is the one `rt_glass.slang` already casts, asked in another
direction -- so the cost is one ray a gaussian a frame, in the second shading
pass, where everything else already has a colour.

`primvars`-free: it is a frame's decision, `athenea:splatReflections` (off by
default), `athenea stage --splat-reflections`, and the MCP's own argument.

**Measured against the pawn's mesh, path traced at 1024 paths** (RMS over the
patch, under the HDRI):

| | before | with the reflection |
|---|---|---|
| the marble disc | 0.0412 | **0.0329** |
| the marble cone | 0.0312 | **0.0291** |

Two things had to be fixed for that to be true:

**A fit that kept narrowing after a ray had already answered.** Where a ray
went and met nothing, nothing is in the way and the whole sky reaches the
mirror direction; leaving Lagarde's fit in as well took a tenth off a gold
collar under a sky of one colour. The fit now applies only where no ray was
cast.

**A metal's Fresnel, taken the way the surface takes it.** Schlick between
the colour and one is not MaterialX's conductor: the index comes from
`mx_artistic_ior` out of the colour, with the **edge white** -- which is what
`standard_surface`'s `specular_color` is unless an artist says otherwise --
and then `fresnelConductor`. A gold that stayed gold at grazing read a fifth
dark in blue.

### What a cloud and the mesh it came from still disagree about

Three spheres -- a gold metal, a glossy dielectric and clear glass -- as
gaussians and as meshes with the same MaterialX materials, under a sky of one
colour so that only energy is in play. Patch means, cloud over mesh:

| | red | green | blue |
|---|---|---|---|
| metal, roughness 0.08 | 1.01 | 0.97 | 0.86 |
| dielectric, roughness 0.12 | 0.92 | 0.93 | 0.94 |
| glass, against the sky behind it | 1.00 | 1.00 | 1.00 |

**Glass is right.** Under a uniform sky a glass ball is invisible -- what it
reflects and what it transmits add to the sky -- and the cloud reads exactly
that. Under an HDRI it reads twice the mesh, and that is not energy but
aim: the mesh's ball inverts the room and shows the dark floor in its upper
half, while one bend in and one out is too weak an inversion and keeps
showing the bright ceiling.

**A metal is a fifth dark in blue** even with the conductor's own Fresnel,
which was put down to the layered specular `standard_surface` puts over its
metal and the cloud does not have. It was not that, and neither was the
dielectric's tenth: both were the ray below, and the table is redone there.

**And the two skies are not the same sky.** A dome of colour 1 and intensity
1 draws its background at 0.9516 in the rasteriser and in the splat tracer,
and at 0.9152 in the mesh path tracer -- 3.8 % apart, and neither of them 1.
Every cloud-against-mesh number in this record carries that bias, and finding
where it comes from is not done: the colour temperature is not applied (the
flag follows `enableColorTemperature`, which these stages leave off), and
both routes scale exactly with intensity, so it is not a transform.

### The ray met the shell it left from

The dielectric's 0.92 above stood in the record for a week as "measured, not
explained", and the bake was the suspect: a surface that reads a tenth dark
under a white sky is a transfer that integrates a tenth short. So the bake
was put in a furnace first. One sphere alone, converted with `--transfer`,
and every gaussian's `T_0` read back off the file: **0.28207** over 684 803
gaussians against the closed form's 0.28209, the median 0.2821, and the
first band's magnitude 0.3259 against 0.3257. The bake is exact. The 52
gaussians reading 0 are the 52 the conversion could not stand a frame on.

The same sphere alone, drawn as gaussians beside the mesh, reads **1.033**
of it -- and the same sphere among its two neighbours read 0.932. Nothing in
the transfer knows about neighbours except the indirect half, and drawing
without it changed nothing. What did change it was `--splat-reflections`:
off, the grey ball among its neighbours reads 1.039, the same as alone.

So every gaussian whose mirror ray had met something was painted red, and
the picture said the rest. On the reference cloud the gold ball was red over
a ring around its whole rim and the grey ball was red *everywhere*; on a
denser conversion of the same spheres, red sat only where the neighbours
truly are, two discs and a thin rim, which is the mesh's own picture. A
convex cloud is a shell of overlapping gaussians, and a mirror ray leaving
one at a grazing angle runs inside that shell for a chord tens of footprints
long, meeting neighbour after neighbour: the nearest of them was what the
gaussian reflected. A grey ball reflected grey where it should have reflected
sky, and lost a tenth; a gold ball reflected gold and lost its blue twice over,
which is the 0.86 above that had been blamed on a specular layer. Why one
conversion and not the other is the footprint against the step the ray
started with, and it does not matter, because a step along the ray can never
fix it: the chord is far longer than any step.

**The guard is by height, not by distance.** `glassNearest` now skips any
candidate that rises less than four footprints above the plane through the
gaussian the ray left, with that gaussian's normal. The shell's own gaussians
along a tangent ray rise by the sagitta, thousandths of a footprint; a collar's
ball, the far wall of a cup, a plate over the object all stand well above.
And a back-facing gaussian of an opaque object casts no mirror ray at all:
its normal is turned to the eye, so the ray pointed into the object and met
the inside of the far wall, and what little leaked through the front shell
reflected gold at the ball's own interior. The glass ray keeps every
candidate, as it should.

The same three spheres, cloud over mesh, with the guard:

| | red | green | blue |
|---|---|---|---|
| metal, roughness 0.08 | 1.00 | 1.00 | 1.00 |
| dielectric, roughness 0.12 | 1.04 | 1.04 | 1.04 |
| glass, against the sky behind it | 1.04 | 1.04 | 1.04 |

The pawn's collar and neck, where the ray earns its keep, still move the
right way: 1.051 of the mesh without the ray, 1.022 with it. The regression
test is a ball of flat gold gaussians alone, drawn with and without the ray:
the two frames are the same to the bit. With a dark plate behind the camera
in the ball's own cloud they differ over the ball's front, which is the
reflection doing its job (`athenea_render_tests "[reflections]"`).

What the dielectric's 1.04 is, and the glass's, is the next thing to
measure, and it is a different thing: three to four per cent *bright*, not a
tenth dark, and the same on a sphere alone.

### A flake worth naming

`a UsdVol volume with an OpenVDB field absorbs as Beer-Lambert says through
Hydra` fails in about one full suite run in three on this machine and passes
every time it is run alone, twice over. It is 9.5 seconds of path tracing
against a tolerance, and what it is sensitive to is the machine being busy,
not the code under it. Written here so the next person does not chase a
regression that is not one; narrowing it is not done.

### What a ray in the raster would buy, in order

The rasteriser can already trace: `technique::SplatShadows` packs a cloud's
tables into one buffer so a raster kernel can query its own proxies, which is
what a Metal limit of 31 buffers forced and what makes the rest possible.
Generalised, the same query answers, most valuable first:

1. **Specular occlusion**, which is the error the pawn measured: the analytic
   reflection is added with nothing to block it, and in the pocket under the
   collar the cloud reads a quarter bright.
2. **Reflections of the scene** rather than of the prepared sky alone.
3. **Geometry seen through glass**, which is what the index cannot do.
4. **Ambient occlusion** for the raster in general.

Refraction is fourth on that list, which is why the index came first: it is
the part that needs no ray at all.

## A stage's lights, switched in the viewer

A sparrow under a dark thatched roof kept a warm highlight on its head and
wing edges whichever sky was chosen. It was the stage's own `DistantLight`
(intensity 4.5): the **Sky** combo changes the dome's image and nothing else,
so a fixed sun stays on under every sky. Rendered with and without it under
the same image, the difference is that rim and nothing more, and it doubles
the brightest pixel (0.54 to 1.25 in red).

`StageRenderer::lights()` lists every `UsdLuxLightAPI` prim, inactive ones
included (`UsdPrimAllPrimsPredicate`), and `setLightOn` deactivates the prim
in the session layer. Deactivation rather than `visibility` or a zero
intensity: it is what every delegate honours the same way -- the sprim is
removed and the engine's `removeLight` runs -- and a dome with intensity 0 would
still be prepared. Switching back clears the session opinion, and writes
`active = true` only where the file itself authored the light off. The test
checks that the frame after off-then-on equals the first one to the last bit,
that a lamp the file left off lights the square when switched on, and that the
file never learns about either.

Not done: a per-light intensity slider, and a sun taken from the chosen sky
standing in for the stage's.

## Displacement is where a cloud is cheaper than a mesh

A mesh cannot follow a relief without being cut into triangles finer than the
relief, which is a tessellator and the memory for what it makes. A gaussian is
a point, so standing it where the relief is costs nothing. `athenea mesh2splat`
now reads a material's height and raises every gaussian by it; the mesh route
still ignores displacement.

**Reading it.** UsdPreviewSurface's `displacement`, through UsdUVTexture's
`scale` and `bias` on the channel connected, and MaterialX's
`ND_displacement_float` named by the material's displacement terminal, times
its `scale`; a constant with no map moves the whole surface. A height is in
the mesh's units, so it is multiplied by the cube root of the transform's
volume. `ND_displacement_vector3` is not read and the log says so.

**Moving is not covering.** A cell of the projection is a square of the flat
surface, and raised it is a patch of the relief as long as the relief is
steep. One gaussian sized for the square leaves the rest bare. Each cell
measures the relief's stretch along its two axes -- the raised surface's
tangents against the flat one's, by central differences half a cell wide --
and is split into `ceil(0.8 * stretch)` gaussians along each (a quarter more
is left alone: the gaussians overlap by more), capped by `--displace-refine`
(8). Each is stood on the relief, turned to the plane of its two raised
tangents and sized for its share. The count and the emit ask the same
question, so the slots agree -- see below for where they did not.

Checked by `athenea_aofx_tests "[displacement]"`: the unit quad under a tent of
height (0 at the edges, a half at the middle, a slope of one), 16 cells
across. 544 gaussians against 264 flat -- every cell off the ridge split in
two, as `sqrt(2)` asks -- and no gaussian off its height (2e-3), off the
slope's normal (0.999), turned otherwise, or sized otherwise than its share
(2 %). A constant height of 0.25 with no map: 272, the flat count, all of it
moved.

**Two defects on the way.**

- *The relief was read at the wrong point.* A sub-cell of a cell whose centre
  is inside a triangle can stand past its edge, and its barycentrics were
  clamped back onto the triangle for the height and the normal while its
  position was not: 8 heights, 20 normals and 16 sizes of 528 wrong, all on
  the diagonal. The triangle's plane, normals and texture coordinates go on
  past its edge; read there, 0 of 544.
- *The count and the emit disagreed about a cell.* They are two kernels. A
  quad's diagonal on a grid that halves it puts cell centres exactly on the
  edge both triangles share, and once the emit changed shape the Metal
  compiler contracted the two differently: the count gave a triangle three
  slots, the emit wrote two, and the third was a gaussian of zero size at the
  origin (`[cell]`'s diluted quad: 2 of 6). The emit now writes exactly the
  slots the count gave -- to the next triangle's start -- stops at the last,
  and writes a slot it has nothing for as a clear gaussian at the triangle's
  middle. Where the relief splits a cell, `ceil` of a stretch is the same
  kind of edge, so this was going to happen anyway.

**A capped cell is thinner, not a spike.** A sphere's texture coordinates
stretch without bound at its poles; sized by the stretch over a capped split,
a gaussian there was metres long, and the cobbled ball below drew white beams
across the frame. A capped share is held to what an uncapped one gets, 1.25
cells: a gap is the lesser wrong, and the log counts the cells.

**The bake.** A raised gaussian is baked from the flat point under it, down
the flat normal -- from where it stands, a ray would begin under the surface
wherever the relief sank it, and find nothing. `bakePoints` takes a facing a
point (three `float4` a point in the kernel now: the caller's two and the
facing, w 1 where there is one), and the hit's shading normal becomes it, so
the material, its normal map included, is lit as the relief turns it and the
hemisphere projected is the facing's. Checked against the tracer itself on a
white Lambertian plane under a sun overhead: points facing 40 degrees over
baked 0.4691, points on a plane really turned by 40 baked 0.4653 (encoded;
0.8 %). A facing of none is the plain bake to the bit, and facing 89 degrees
bakes 0.053 against 0.492.

That took one change. With a bounce, the facing baked 0.4840: a bounce the
tilt opens under the flat surface meets the flat mesh, lit, a hair away --
light the relief, which stands there, would not let through. A bounce from a
raised gaussian's first vertex under the flat surface is now closed.

**Not done.** The relief's shadow on itself: the tracer holds the flat mesh,
so a cobble does not shade its neighbour, and a groove is as lit as a crest
facing the same way. Tracing the relief would need it in the tracer -- the
very tessellation this avoids -- or the cloud itself as the occluder, which
the visibility bake is the start of. And moving the relief after conversion:
the three entries a displaced record carries (the flat point and its height,
the flat normal, the relief's normal) are what a kernel would need to raise
the cloud again at another scale, but the file does not keep them yet, and the
split is decided at conversion for its height.

**A demonstration**, not a test: a cobbled floor and ball
(`~/tools/assets/Displaced/Cobbles.usda`, a height map of rounded cobbles
0.06 tall, 8 to a tile), `--resolution 768`, baked, drawn by raster at
1200x800. 533 029 gaussians flat, 1 505 305 displaced -- 349 701 on the floor,
1 155 604 on the ball, whose cobbles are as tall as they are wide. The
conversion took 26 s flat and 1 min 52 s displaced, the bake nearly all of it.

## Fewer gaussians where the surface is the same

A gaussian a cell is what a converted surface costs wherever it is, and most
of a surface is the same from one cell to the next. `--simplify` walks each
triangle as a tree of blocks (8, 4 and 2 cells a side with `--simplify-levels
3`) and makes a block one gaussian of its size where colour, metallic and
roughness, cut-out, shading normal and relief normal all agree within the
tolerance. A block's gaussian is as wide as the block, as a cell's is as wide
as a cell, so the two overlap their neighbours alike and cover alike.

Three things were found by looking, on the cobbled floor at `--resolution
768` baked at 64 paths, and each changed the rule:

- **Judged on the block alone, the mortar blurred.** A cobble's flat top
  merged right up to the mortar, and the tail of its gaussian -- a block past
  each side at a width of one block -- laid the top's colour over the joint.
  The cells a block past each side must agree too. The floor went from 51 305
  gaussians to 94 856, and the joints came back.
- **At the mesh's border the merged blocks hung past it.** The surround was
  read on the triangle's own plane and coordinates, which go on past its edge
  where the surface does not. The whole reach must lie inside the same
  triangle: 99 413. So it pays on triangles larger than a few blocks, and a
  mesh finer than that -- the cobbled ball's 2304 triangles -- keeps every
  cell.
- **A block's bake was one point's.** A gaussian eight cells wide carried the
  light 64 paths found at its centre, glints of the sun included, now spread
  over 64 cells. The bake takes a width a point (the `w` of its normal) and
  comes down onto a disc of half of it; the conversion gives it when
  simplifying. At 256 paths the floor shows no glint, where the unsimplified
  one at 64 is speckled with them.

The count and the emit share the walk, and the emit writes exactly the slots
the count gave, as it does since the displacement work.

Checked by `athenea_aofx_tests "[simplify]"`: a one-colour unit quad at 64 cells,
4160 gaussians a cell and 2144 simplified, every one on the quad, of that
colour, and one, two, four or eight cells wide (384 of them blocks); under a
checker of four-cell squares, narrower than any block's reach, 4160 and 4160.

| Cobbles, `--resolution 768` | a cell each | `--simplify 0.02` | `--simplify 0.1` |
|---|---|---|---|
| floor, flat | 262 145 | 99 413 | |
| ball, flat | 270 884 | 270 884 | |
| floor, displaced | 349 701 | 309 972 | 272 664 |

The walk costs time on the device, each block reading its surround: the flat
stage without a bake went from 1.3 s to 6.2 s. It is one thread a triangle, so
a floor of two triangles walks 262 144 cells in two threads.

**Not done.** A mesh finer than its blocks gains nothing here: that is what
merging the gaussians themselves is for, after the conversion and on any
cloud, trained captures included -- `athenea decimate`, below. The bake's sample
count is not scaled by the area a gaussian stands for, so a simplified
conversion wants more `--bake-samples` than the same conversion without.

## Decimation: the levels of detail, cut once by what a merge loses

`athenea decimate` takes any cloud -- a conversion's ParticleField, a capture's
`.ply` -- builds its levels of detail and keeps, for good, the coarsest
merge that stands for what is under it and every splat where none does
(`lod::Decimator`, `lod_decimate.slang`). A frame's cut asks how many pixels
a cell spans; this asks what merging it loses, which is a property of the
cloud and not of a view. The error of a merge is the largest of four, each
over its tolerance, and never below its children's, so down any line of the
tree it passes 1 once and every place is kept exactly once.

Every one of the four is where it is because the obvious form failed, on the
cobbled floor (533 007 gaussians, baked at 64 paths) or on a test floor of
96 x 96 discs as wide as their spacing:

- **Colour as a share, not a largest difference.** The largest kept every
  cell holding one glint of the bake's noise: 7 % fewer. A mean over a cell
  two thirds cobble and one third mortar sat within a lenient tolerance of
  both, and at 0.2 the floor went to a brown smear. What tells a glint from an
  edge is how many: the share of the splats a merge reaches that differ by
  more than `--colour-tolerance` must be under `--outliers`.
- **Reach against the splats, not the finer merges.** Those sit near a cell's
  middle however far its splats go; judged against them, a floor of one
  colour went to four blobs, one a quadrant. Against the splats, a merge of
  discs as wide as their spacing is held to about 4 x 4 of them.
- **Flatness as the thickness added.** A merge is never thinner than its
  discs, and discs a tenth as thick as wide merged into nothing on a flat
  floor when the whole thickness was judged.
- **The tail, where nothing may differ.** A merge is written wider than its
  moments (below), and reaches past its splats. A share bounds nothing there:
  an edge that clips the end of the tail is a small share and a line of the
  wrong colour. Past `--reach`, and out to `--reach` + 1, no splat may take
  more than the tolerance of the merge's colour. The search covers as many
  cells as the tail is long: the 26 around a cell were not enough, and a
  checker's squares lost their edges to merges whose tails reached two cells.

**Merged wider than the moments.** The levels merge by moments, which keeps
the second moment: 2 x 2 discs as wide as their spacing merge into one 0.58 of
its spacing, right for a cell seen as a pixel and wrong up close, where the
floor showed its background through a grid of holes (image p99 70 without the
widening, 1 with it). A kept merge is widened in its plane to overlap as its
splats overlapped -- their spread `v`, the spacing of a grid of `n` over it
`d^2 = 12 sqrt(v_a v_b) / (n - 1)`, their own width `c`, and `c / d` of the
span `sqrt(12 v + d^2)` -- held to twice its moments, since splats nearly in
a line make `d` vanish and the merge a streak; its opacity is its splats'
mean.

Checked by `athenea_lod_tests "[decimate]"`, the 96 x 96 floor at a few discs a
pixel (the colours at two depths, so that which is on top does not depend on
the order a merge changes): one colour, 1024 of 9216 at p99 1; a checker of
12-disc squares, 6408 at p99 6; one glint in thirty, 6541 at p99 1. And by
`athenea_usd_tests "[decimate]"`: a cloud unpacked from the device
(`CloudLoader::records`), written, and read back from its ParticleField
(`usd::readParticleFieldRecords`) draws as it did, p99 0 both ways.

On the cobbles, drawn by raster at 1200 x 800 against the undecimated cloud:

| | Kept | Image RMS |
|---|---|---|
| defaults | 82.9 % | 0.0020 |
| `--colour-tolerance 0.1` | 37.9 % | 0.011 |
| baked at 256 paths, defaults | 53.9 % | |
| unbaked, defaults | 57.8 % | |

**Not done.** The widening assumes a merge's splats tile a surface; over a
volume capture it widens by the same rule, unmeasured there. What a cloud
carries beside its gaussians was dropped at first, and is kept since: below.

## A decimation keeps everything a cloud carries

A decimated cloud lost its materials, ids, transfer, bits and rig: it was
written as a new stage of positions, shapes and harmonics. It now comes out
as the stage it came from, with fewer gaussians (`usd::decimateStage`):

- **The stage is copied, not rebuilt.** The root layer is exported -- the
  skeleton and its animation, lights, camera, manifest, variants, constant
  primvars -- its relative asset paths anchored to where it was when the copy
  lands elsewhere (`UsdUtilsModifyAssetPaths`), and on the field only what is
  a gaussian long is replaced. The schema's arrays are the kept gaussians',
  written through `writeParticleFieldStage` to a stage of their own and
  copied from it, so their values are the exporter's kernel's and nothing
  here computes them.
- **What a gaussian carries is found, not listed.** A vertex (or varying)
  primvar, or any other array whose length is a multiple of the gaussians,
  is one (`usd::readGaussianArrays`): a channel added tomorrow is carried
  without being named here. Constant primvars are the field's and stay.
- **Each is merged by what it is** (`lod_attributes.slang`), over the run of
  store splats each kept gaussian stands for -- the decimation now returns
  that run and each store splat's record (the build keeps its Morton order,
  `LodCloud::order`). Floats as a mean weighted as the moments weigh; a rig
  as the weight each joint carries summed and the four heaviest kept; bits a
  bit at a time by half the weight; ints as what the key kept apart.
- **What may not be merged, is not.** Ints -- an id, a part, a sheet -- and a
  rig's dominant joint are folded into a key a record, and a merge whose
  splats' keys differ is refused outright. Metallic, roughness and
  transmission are compared as colour is. The kernels live apart from
  lod_decimate's: one module of thirty-nine buffers put `lodGatherRanges` at
  `buffer(32)`, past the thirty-one a Metal kernel binds.

Checked by `athenea_lod_tests "[decimate]"`: a floor of one colour whose halves
are two prims, two joints and two patterns of bits, with a float that is each
record's x, goes from 4096 gaussians to 256 with every kept gaussian of its
side's id, wholly its side's joint, its side's bits, and its float within
1e-3 of where it stands. And by `athenea_usd_tests "[decimate]"`: a stage with
those arrays, a constant flag, a camera and a relative asset, decimated into
another folder, keeps the flag and the camera, finds its asset, and every
array is as long as the 256 of 2304 gaussians kept.

On real clouds: the cobbles (ids, metallic, roughness, transmission) keep
37.9 % at `--colour-tolerance 0.1`, as without the channels -- floor and ball
never merged into each other -- and draw the same image. The sparrow at its
finest level, 5 887 323 gaussians with ids, material, visibility parts and a
rig, keeps 87.7 % in 28 s; posed at time 20 it draws as the undecimated cloud
at an image RMS of 0.002, wings and all. Its textures are detailed and it is
unbaked, and merges stop at every joint's edge.

## One AOFX host, from aopenfx

`modules/aofx/host` was a port of the compositor's `sdk_host` -- three files
each headed *"Ported from …"* -- and `modules/aofx/sdk` a verbatim copy of the
SDK guarded by a manifest of hashes. Two copies of a host drift; a bundle
loads in two programs only if the host that loads it is the same code with
the same build tag.

Now both are aopenfx's, from `third_party/aopenfx` on its `sparrow` branch:
`sdk/` the headers, `host/` the registry and the runner (`aofx::host`),
with what this engine brings declared in `modules/aofx/host/src/Host.cpp` --
a logger and nothing else. The clip and model verbs answer *"not available in
this host, because it declares no media capability"* from the host's own
code. What stays here is `renderEffect` (what a compositor's engine does for
an AOFX node, without the graph), the bundle rule `aofx_add_bundle` (until
aopenfx's `AofxPlugin.cmake` grows `OUTPUT_DIR` and `LIBRARIES`), and the two
plugins.

The manifest test keeps its job with a different object: it guards the pin
now, not the copy. The headers at the pin are byte for byte what the copy was,
so `tests/aofx/sdk_manifest.txt` did not change. The channel-restore kernel
is the host's, under `aofx.host.channels`; the bundle identifier prefix in
`AofxBundle.cmake` is a cache entry, `AOFX_BUNDLE_ID_PREFIX`.

`athenea aofx list` prints the ABI and the build tag first, so two programs can
be put side by side: same tag, same bundles loaded, and a bundle refused in
one refused in the other with the same sentence.

**What the move to aopenfx's host broke, and what was found by running it.**
Three of `athenea_aofx_tests` crashed in the first render (Invert, the
miscount refusal, the reporter's complaint), all at the same line:
`EffectRunner`'s constructor in `Host.cpp` handed its base
`capabilities()`, meaning this engine's free function -- but inside a class
derived from `aofx::host::EffectRunner` that name finds the base's member
`capabilities() const` first, which reads an `impl_` the base has not built
yet. It compiled, and it was a null dereference on every render. Qualified as
`aofx_host::capabilities()`; nothing in aopenfx was wrong, and no bundle
changes. The registry's constructor has no such member to collide with.

openFXplayer's installed bundles are refused, correctly: its tree pins
aopenfx at ABI 25 and this host speaks 26. aopenfx's rule is one number and a
mismatch is a refusal (`sdk/include/aofx/Version.h`), and its
`HOST_CHANGES.md` for 26 names exactly this check -- *a bundle deliberately
left at 25 is refused* -- because `Gpu` grew two virtuals and a 25 bundle
would call through a vtable two slots short. So the host does not accept 25.
`bundles openFXplayer built load in this host` skips, saying so and listing
every refusal, only when *every* bundle is refused by the ABI gate naming this
host's number -- the same treatment as a build with another toolchain. A
loaded bundle, or any other reason, still fails it; rebuilding openFXplayer on
aopenfx at 26 turns it back into a check.

## Embeddable: the same CMakeLists as a subdirectory of the compositor

The compositor builds this engine inside its tree (`OFXP_LRT_DIR`), so the
build had to work as a subdirectory as well as at the top. What that took,
each a collision found by trying it: the engine's paths through `ATHENEA_ROOT` and
`ATHENEA_BUILD` instead of `CMAKE_SOURCE_DIR`/`CMAKE_BINARY_DIR` (the shader copy
globbed the parent's tree and found nothing); gpe, genlock, aopenfx, spz,
slang-rhi, nlohmann_json, OpenColorIO and Catch2 looked for by target before
being made, so the parent's serve; `ATHENEA_EMBEDDED` keeping the output
directories out of the parent's and `ATHENEA_BUILD_APPS`/`ATHENEA_BUILD_PLUGINS` so
the parent asks for `athenea` by name and gets no second Invert beside its own;
`ATHENEA_USD_ROOT` declared as a cache variable (only the presets set it before);
the engine's own CMake modules included by path, since the parent has a
`cmake/Warnings.cmake` and a `cmake/Dependencies.cmake` too and
`include(Warnings)` found those; `athenea::engine` as the one target an embedder
links. The standalone build is unchanged by all of it: same presets, same
targets, one TBB.

Measured in the compositor's tree: its binary links slang-rhi and one OCIO
(its own, Homebrew's 2.5.2 -- the same version as `~/tools`), its example pass
is identical to the one before the engine was in, and `athenea` built from there
counts one TBB once `TBB_DIR` is not the stale Homebrew one a previous
configure cached: USD's prefix goes first in `CMAKE_PREFIX_PATH` now
(`cmake/Usd.cmake`), and `cmake -UTBB_DIR` is the cure for a directory that
configured before it did. What is still ahead: the compositor's OpenImageIO
brings Homebrew's TBB 12.19, this engine's USD its own 12.12, and the day the
compositor links `athenea::usd` both are in one process -- the fix is a USD built
against the same TBB, `scripts/build-usd.sh` without `--onetbb`, before that
day.

## The crossing as free functions, for a context that is not this one

The compositor this engine is built into has a GPU thread, a pool and an
image storage of its own, and its `Context::create` opened gpe's device
directly. Sharing the device with this engine means it opens `athenea::gpu::Device`
first and gpe adopts that -- the forty lines `gpu_host::Context` had for it,
which it could not call without taking the whole context, a second thread
and a second pool with it. So the three crossings are free functions now,
`gpu_host/Views.h`: `adoptForCompute(gpu::Device&)`, `renderView(device,
compute, …)`, `computeView(compute, buffer)`. `Context` keeps its methods and
calls them; nothing about this engine's own arrangement changes, and
`athenea_gpu_host_tests` covers the functions through the methods as before. One
implementation of the crossing, whichever context surrounds it.

## athenea::engine without athenea::usd, and the TBB an embedder already has

The compositor's first engine draws from its own scene text -- clouds, a
camera, lights -- and needs no stage, so `athenea::engine` no longer links
`athenea::usd`: an embedder that wants stages links it beside. Two reasons, one
of them measured. USD brings its own oneTBB, and the compositor's process
already carries Homebrew's through OpenImageIO; with `athenea::usd` in, both
were in one process (`DYLD_PRINT_LIBRARIES` counted `libtbb.12.19` and
`libtbb.12.12`). And `athenea::io` linked USD's TBB by name: it looks for
`TBB::tbb` before finding its own now, so a parent that found one first --
the compositor finds Homebrew's -- gives it that one. Standalone, nothing
changes: the presets find USD's, and `single_tbb` still counts one.

## A stage on the host's device, and a stage read without one

The compositor this engine is built into loads USD stages now, and it already
drives the GPU: its compute runtime adopts the device this engine opens, and
its renderer node draws clouds on it. A stage has to draw on that device too,
or it is a second device with nothing shared and every frame copied across.
`Engine::create` opened its own (`gpu::Device::create()`, with no way to say
otherwise) because Hydra makes the render delegate and the delegate makes the
engine. `StageRenderer` makes the delegate itself, though, so the device can
travel: `StageRenderer::open(path, device)` → `HdAtheneaRenderDelegate(device)` →
`Engine::create(device, why)`. Null opens one, as before, and the plugin
registry's delegates -- the ones Hydra makes for a `UsdImagingGLEngine` host
-- still open their own.

A tree view of the stage's prims wants the rows, not a renderer: opening a
`StageRenderer` to list children is a device and a Hydra index for a panel.
`outline(path, prim)` and `stageCameras(path)` open the stage alone and keep
it by path and modification time; `children()` and `outline()` share the one
function that makes the rows.

`setPrimVisible` is `setDomeTexture`'s pattern for visibility: a session
opinion, applied through the scene index, and "shown" is the opinion cleared,
not `inherited` written -- a prim the stage made invisible stays invisible
when a host stops hiding it.

A compositor frame must be whole, as `render`'s image is, but it goes on
from the device: `drawImage` is `render` without `readImage`, and `draw` stays
the viewport's (streams filling in over frames). Measured in the compositor on
the chess set: RGBA against `athenea stage` at PSNR 171 dB, the difference being
half-float rounding. `registerPlugins` exists because an embedder's USD has
already started by the time it knows where athenea's schemas are. OpenVDB 10.1
against Homebrew's oneTBB 2023 fails on `TBB_INTERFACE_VERSION` (gone from
`tbb/tbb.h`); force-including `tbb/version.h` is the whole fix.

And `mesh2splat`'s identifier is `rt.sparrow.aofx.mesh2splat` on this branch:
it is the one bundle the compositor builds from this tree, and its catalogue
does not carry the former vendor's name.


### An invisible PointInstancer hides its instances

Hiding a PointInstancer through the session layer, as the compositor's
prim tree does, did nothing: the chess set's pawns stayed. USD 26.08's
scene index keeps an instancer's visibility on the instancer, and the
scene delegate answers `GetVisible(instancerId)`; nothing folds it into the
prototypes, whose own visibility is unchanged, and Storm-era delegates
that deleted prototypes are gone. Every rprim here (mesh, points, curves,
particle field, volume) now also requires each instancer above it to be
visible (`HdAtheneaInstancersVisible`).

The other half is the notice. A change to the instancer dirties the
instancer, not its prototypes, so they would keep their visibility until
something else resynced them. `HdChangeTracker::MarkRprimDirty` on the
subtree was tried: the marks were made and no Sync followed, as the
resample scene index's own comment says of such marks. `setPrimVisible`
now dirties visibility at and beneath the prim through that same scene index
(`DirtyAll(locators, root)`, `HdAtheneaRenderDelegate::DirtyVisibilityBelow`).
Checked in the compositor: hide, draw, show, draw gives the first picture
back exactly, pawns included.

### A shader's texture swapped from outside, and the network USD kept

`setShaderTexture(shader, input, file)` writes a session opinion on a
shader's asset input, or clears it. In the compositor the chessboard kept
its texture after the swap, although the stage and every scene index but one
held the new file. The exception was `HdsiLocatorCachingSceneIndex` for
materials, whose retained `mtlx` network survived. The cause is in USD:
`InvalidateImagingSubprimFromDescendent` dirties
`material/<context>/nodes/<prim name>`, while a node inside a NodeGraph is
keyed `NG_ChessBoard/mtlximage13`. The same edit therefore re-authors the
enclosing material's output connections with their own targets, which
USD's adapter answers by dirtying the whole material. Swap, draw, clear,
draw gives the authored picture back exactly.

### Textures a host fills: the `aofx://` scheme

A material may name a texture no file holds: `aofx://slot1`. `TextureStore`
keeps it as an entry like any other, but never looks for it on disk, reads it
raw, and fills it only when the host calls `updateExternal(name, buffer,
width, height, rowPixels)`. That call takes float4 texels bottom row first,
the layout of an image plane. It makes an RGBA16F texture when the size
changes (and rewrites the records), runs the decode kernel over the host's
buffer (the kernel gained a row pitch and a bottom-first flag), and makes the
mips. It is recorded on the device's queue and not waited for, except where
stores do not convert and a packed buffer has to outlive the batch.
`StageRenderer::updateExternalTexture` is the embedder's door, and
`setShaderTexture(shader, input, "aofx://slot1")` points a material at it.
In the compositor an HD movie on the chessboard costs nothing measurable
over the same frame read from a file.

### mesh2splat converts what the renderer draws

A host baking a stage it shows to gaussians found the cloud disagreeing with
the picture in two ways. `MeshStage::read` walked `Traverse()` and took every
mesh: invisible ones included, so a prim the host had hidden came back in the
bake. And a PointInstancer's prototype was converted once, where it stands in
namespace: the chess set's sixteen pawns became two, at the prototypes'
places. Now a mesh that computes invisible at the read's time code is
skipped, and `MeshStageOptions::hidden` (`--hide`, repeatable) authors
session invisibility first. A prototype's mesh is emitted once per instance
of the innermost instancer that holds it, at (mesh relative to its prototype
root) x (instance transform, prototype root's own transform included) x
(instancer to world). The built mesh is shared between the copies: its
buffers are counted references. Nested instancers are expanded at the
innermost level only.

Not fixed: a light bake of the chess set under the default lights leaves 80
pixels of infinity in a 960x540 frame, on metallic rims and crowns, whose
coefficients overflow half precision. The rest of the bake is right; a clamp
on the baked radiance is the likely fix.

## A bake gives no more light than any path saw, from any side

Measured on the chess pawn under a uniform dome of 1, path traced at 256
samples against the mesh: the cloud baked at degree 0 read a mean of 0.942
(the mesh 0.948), at degree 1 its glass head peaked at 1.7 -- and at degree 2
at **1094**, at degree 3 at **infinity**, body included. Two things, each
enough on its own.

**The bands were taken to the cloud's space to first order.** A cloud keeps
its harmonics in the sRGB space it is blended in, and the bake fits in linear
light; the bands were then scaled by the curve's slope at the mean. That slope
is 12.92 at black, and a surface whose mean is dark and whose light swings
with the direction -- glass, a dark polished metal -- is exactly where it is
steep and where the bands are large.

**The fit knows half the sphere and a frame reads all of it.** The fit is over
the half the surface faces (the projection it replaced put a dark rim on every
silhouette), but a disc is seen edge on and from behind, and there the series
is an extrapolation of the least determined combinations of the bands.

The encoding is now a second fit, done in the kernel (`bakeEncode`): the
linear fit is read back over the same 32 x 16 grid the matrix is integrated
on, held between nothing and the brightest sample the paths returned, encoded
direction by direction, and the far half of the sphere is given the near
half's mirror image across the surface's plane. On the whole sphere the basis
is orthonormal, so the encoded series is a projection with no matrix. Degree 0
is unchanged (the encoding of the mean); above it the series is bounded by
what it fits, up to a projection's overshoot.

| pawn, uniform dome 1, 256 paths | mean | largest | relMSE against the mesh |
|---|---|---|---|
| mesh | 0.948 | 2.55 | -- |
| degree 0 | 0.942 | 1.33 | 0.0135 (as before) |
| degree 2, before | 3.77 | 1094 | 1058 |
| degree 2 | 0.946 | 1.67 | 0.0054 |
| degree 3, before | inf | inf | inf |
| degree 3 | 0.947 | 1.55 | 0.0054 |

The test is the surface that broke it: polished metal under one small sphere
light of radiance 50, baked at degrees 1 to 3 and read over the whole sphere.
Before, 26, 9 and 21 of 64 points decoded beyond the light, the worst to
177 000; now none.

Also found on the pawn: the written ParticleField stage carried no
`metersPerUnit`, so an application honouring units read a cloud converted in
metres at a hundredth of its size (USD's fallback is centimetres); it is
written now, from the source stage. And `athenea` registers its own plugin
directory at start-up: without `PXR_PLUGINPATH_NAME` the schema a converted
cloud applies (`AtheneaSplatCryptomatteAPI`) was an unknown token and was
dropped from the file.

Not done: the bake still fits in linear light and encodes, because a cloud is
still blended in sRGB. Blending every cloud in linear light, with sRGB only
where an image is shown, is the next change, and it removes the encoding.

## A converted glass bends, and its cloud says which schema it carries

Under an HDRI (autoshop_01) the pawn's glass head drew as a milky ball in
every mode a converted cloud has -- relit, transferred, baked -- while the
mesh showed the workshop through it. A transmitting gaussian refracts only
with the cloud's index (`rt_shade`: `ior > 1`), mesh2splat knew the
material's (it sizes a thin wall's opacity from it) and never wrote it. It
writes `primvars:athenea:splat:ior` now, the first glass's when a stage has
two, with a line saying so: a cloud keeps one index.

| pawn, autoshop_01, transfer, 512 paths | relMSE against the mesh | p99 relative | largest |
|---|---|---|---|
| no index | 0.023 | 1.41 | 5.1 |
| with the index | 0.050 | 0.77 | 11.8 |

The relMSE rises because the head now refracts the room and a refraction
seen through 0.73 million gaussians is not the mesh's to the pixel; the p99
halves, and the largest values are the reflections and the refracted lights
that were missing.

The lighting primvars (relight, litBody, ior, the PBR arrays, the transfer
and its shadow bits) were also written without AtheneaSplatLightingAPI
applied, so `relight` came out as a custom attribute nobody declared. The
API is applied whenever any of them is written.

## A material is read in its own words, at its own defaults

An audit of what a mesh renders with against what reaches a converted cloud
found the converter reading every surface with one vocabulary and one set of
defaults:

- **glTF.** MaterialX's `gltf_pbr` calls its inputs `metallic`, `roughness`
  and `ior`; read with standard_surface's names they were not there, and a
  glTF metal arrived a dielectric of roughness 0.5. Its `alpha` is the
  cut-out, and its transmission takes the base colour as its tint.
- **Defaults.** An input nobody authored is worth what its surface says,
  because that is what the mesh renders with: standard_surface 0.8 grey and
  roughness 0.2 (version 1.0.1, the default, whose `base` is 1), OpenPBR
  roughness 0.3, glTF a fully rough metal, UsdPreviewSurface 0.18 at 0.5.
  The converter used white, a dielectric and 0.5 for all of them. A
  connected colour is the map, and its constant is one.
- **The base weight** (`base`, `base_weight`) scales the colour.
- **A packed map's missing channel** is a factor of one. The kernel
  multiplies the map into the material's value, and the picture held
  (roughness 0.5, metallic 0) where no map wrote: a metal with only a
  roughness map became a dielectric, 1 x 0. No test drives it yet: it lives
  in the CLI's map composition, which no test runs.
- **UsdPreviewSurface's opacity** below one was read as transmission, which
  is right, and as a solid's, which is not: the mesh blends the surface over
  what is behind it and bends nothing. It is a thin wall's now --
  `(1 - T) surface + T behind`, untinted -- and only glasses that are not
  thin walls give the cloud its index. Without this, the index the previous
  change exports would have made a translucent plastic refract.

Measured by the audit's claim that would have mattered most: the pawn's
standard_surface was said to lose a quarter of its albedo to a default `base`
of 0.8. It does not -- 0.8 is version 1.0.0's, and an unversioned node gets
1.0.1 -- and the relit pawn's mean already matched the mesh's (0.287 against
0.281).

What a cloud still cannot carry, and the mesh renders: emission (a relit or
transferred cloud of a lamp is dark; a radiance bake keeps it), specular
weight and colour (F0 is 0.04 for every gaussian), coat, sheen, and the
normal map's tangent frame (the conversion builds it from the triangle's
longest edge, not from the UVs). Those need fields a gaussian does not have.

## A normal map is read in the frame its UVs give

mesh2splat read a normal map against the triangle's longest edge -- the
gaussian's own first axis, and nothing the map was painted against. The mesh
reads it against dP/du (`technique/material_surface.slang`), so every facet
of a converted surface turned its relief by its own angle: the pawn's marble
veins were lit from a different side on each triangle. `m2sMapTangent` is
the mesh's formula -- dP/du from the triangle's positions and UVs (the set
the map is read with), orthonormalised against the normal, and the same
frame where the UVs are degenerate -- used wherever the conversion reads a
map: the gaussian's turn with `--normal-map-turns`, and the shading normal a
bake and a transfer orient their hemisphere by.

Test: a quad whose longest edge is its diagonal, u along x, a constant map
leaning to +u. Before, 272 of 272 gaussians turned towards the diagonal (20
degrees off); now none.

Not done: a gaussian keeps no shading normal of its own. Without
`--normal-map-turns` a relit or transferred cloud shades by its disc's axis,
so the relief a map paints is in the bake's light and nowhere else; turning
the disc instead opens the surface where the relief is steep. Storing the
shading normal beside the frame, skinned with it, is the next change.

## A gaussian keeps its shading normal apart from its frame

The change the section above left for next. A converted gaussian now keeps
the normal its normal map gave it -- the `shading` mesh2splat already worked
out for every gaussian -- beside its frame, and a relit or transferred cloud
is lit with it. The disc is not turned: the frame keeps answering everything
geometric (the footprint, where a ray meets it, the plane a reflection must
clear), so the surface does not open where the relief is steep, which is what
`--normal-map-turns` costs and why it stays off by default.

- **The record and the file.** `io::SplatEncoding::normal` names three
  consecutive floats of a record (`kNoField` for every capture). mesh2splat
  writes them always (`record[17..19]`, the harmonics moved to 20), and the
  export writes `primvars:athenea:splat:normal`, `normal3f[]`, vertex,
  declared by `AtheneaSplatLightingAPI` -- normal3f so that a host reads it as
  a normal and a transform turns it as one. It is normalised and turned by the
  export kernel, not on the host. `readParticleFieldRecords` and Hydra
  (`ParticleFieldArrays::normals` -> `SplatStreams::normals` -> the streams
  kernel) read it back.
- **On the device, one word.** `GpuSplats::normals`, octahedral 2 x 16 bits
  (`packNormal` / `unpackNormal` in common/packing.slang, written out there so
  packing imports nothing): a step of a few thousandths of a degree, finer than
  the 10-bit rotation. Encoded by the decode kernel; a record with no
  direction at all gets the frame's shortest axis. Optional as `pbr` is:
  `hasNormals()`, the buffer bound either way and a flag that says.
- **Shading.** `splatShadingNormal` (splat_relight.slang) is the one place:
  the disc's axis turned to the eye as before, and the stored normal put on
  that same side of the disc -- a disc is seen from both faces, and the eye
  behind a surface sees its relief from behind. The rasteriser
  (splat_project) and the traced route (rt_shade) both take it; rt_shade
  keeps the disc's axis for the plane its reflection ray must clear and uses
  the shading normal for the reflection and the refraction directions. Every
  lobe, the dome's irradiance, the transfer's sun share (`splatSunShare`,
  whose open hemisphere was the disc's while the transfer was baked over the
  shading normal's) and the shadow bits' horizon now read the same normal.
- **Skinning.** The skinner turns it by the same blend as the frame, and as a
  normal: `(M a) x (M b)` for two directions in its surface, which is
  `cof(M) n` -- the inverse transpose up to scale, and the same construction
  the disc's own axis already had (`cross(u, v)`), so the two stay on the same
  side of each other through a shear. The posed cloud owns its normals.
- **Levels of detail.** The Morton reorder and the cut carry the word; the
  moments gain three (the sum of `w n`, weighed as a colour is), and each
  merged Gaussian's normal is that sum made unit, or its own shortest axis
  where its children cancel. A decimation merges the file's normal as a
  direction (`AttributeMerge::Normal`): the mean made unit again, where the
  plain mean of three floats came out shorter than one wherever two normals
  disagreed (all 256 kept gaussians of the test floor). `.athc` is version 2:
  the header's former padding is `flags`, bit 0 says every block ends with the
  normals, and a version 1 file -- whose padding was zero -- is read as before.

What it costs: four bytes a gaussian on the device and twelve in the file
(the pawn's relit conversion, 730 559 gaussians, 40.9 -> 49.7 MB), one word
read a relit splat a frame.

Tests, each failing without its half of the change (checked by reverting it):

- `athenea_usd_tests "a cloud's shading normals survive*"`: 3000 gaussians
  with five known normals go through the export and back as records, as
  Hydra's arrays, and through a `.athc` (store and every level, word for
  word); 0 apart on each route. A `.athc` of a cloud without normals, its
  version set back to 1, reads. Without the export or the `.athc` half, the
  normals do not come back.
- `athenea_usd_tests "a relit card with a tilted shading normal*"`: a mesh
  quad whose normals lean 35 degrees and a card of gaussians laid flat that
  carries that normal, under a sun from 40 degrees. Raster and traced, the card
  against the tilted mesh p99 relative 0.020 (the untilted pair: 0.020), the
  card against the flat mesh 0.324. With the shading taking the disc's axis,
  the card against the tilted mesh was 0.229 and against the flat one 0.020:
  it rendered like a quad with no map.
- `athenea_scene_tests "[normals]"`: 4096 discs whose normals lean 0.4 rad off
  their axes, skinned by a turn and by a stretch with shear; 0 of 4096 off the
  inverse transpose, 0 on the other side of their disc. Not turned, 4096 were
  off; turned as a direction, the stretched case was 4096 off.
- `athenea_lod_tests "[normals]"`: a floor whose normals alternate between two
  tilts: every merged normal of every level and of the cut is a unit vector on
  the arc between them (0 off; 1364 of 1364 off with the merge replaced by the
  shortest axis), and a decimation's merged normals are unit.

The pawn under the autoshop HDRI (768 x 768, traced, 512 paths, against the
mesh's render; `--no-bake`): relit relMSE 0.0526 -> 0.0498, transferred
0.0498 -> 0.0470; p99 relative unchanged (0.84 and 0.77), the body's means
within a thousandth. Small, and it is what the pawn has to give: its normal
map leans a median 1.6 degrees (2048 texels), 0.7 degrees by the time a
gaussian samples it at the default `--texture-size 1024` and the cell, so
the stored normals sit a median 0.66 degrees off the discs' axes (p99 3.7).
The grain the mesh shows is mostly its metallic and roughness maps at
pixel scale, which a gaussian a cell averages. Read at 2048 the relit figure
is 0.0505, no better.

Not done: a capture has no normal and keeps its axis. `CloudLoader::records`
(a cloud on the device back into records) does not unpack normals, so a
splat file decimated without a stage keeps none -- a splat file never has
them. The merged levels carry the normal but still no PBR channels.
## A ray that went into a glass meets its far face

The converted pawn's glass head was measured against the mesh path traced
beside it, and came out 0.050 relMSE from it -- worse than with no index at
all. The cloud was not what was wrong. The path tracer culled back faces for
every ray after the first hit (`traceNearestFrom`, both routes), which is what
a single-sided mesh asks of a ray that bounced off it and wrong for one that
went through: inside a solid glass ball the far face is a back face, so the
ray left without bending again, and the ball showed the room bent once -- a
thick lens drawn as one interface. The cloud, which bends at its far face
(`rt_glass`), was being held to a picture no glass makes.

Two measurements said so before anything was changed. The cloud's own single
bend (no far face asked) against the old mesh: relMSE **0.0059**, head 0.050.
And the mesh with its index raised to 2.0 looked as the cloud did at 1.5.

**A crossing sees back faces.** A bounce ray is traced without culling when
it goes through a glass (`throughGlass`): the vertex's material has a
dielectric lobe, and the direction went through the surface it left
(`dot(n, wi) < 0`, with `n` facing the side the path came from) or the
dielectric was met from inside (`kFlagInside`), where a reflection stays in
the glass. A ray that bounced off anything else still culls, as before -- a
back face of a room seen from inside its walls included. The first version
of this asked only whether the surface was met from inside, and the closed
furnace (`athenea_technique_tests "a closed emissive shell*"`), a Lambert
shell seen from inside, counted every bounce free and read its series 33 %
high at one bounce.

**A crossing is not a bounce.** With the far face found, a double-sided ball
-- which was never culled -- showed what the default of one bounce does to a
solid: the ray met the far face with nothing left to leave by, and the ball
drew black. Crossings are free now, up to `kFreeCrossings` (8) a path, as a
renderer keeps its transmission depth apart from its diffuse one. Next event
estimation weighs a light against the material only where the path would go
on in that direction (`pathGoesOn`): out of bounces, that is through a glass
alone. A diffuse transmission (a translucent leaf) is a bounce as it was.

| pawn, autoshop_01, the transferred cloud, 512 paths | whole relMSE | p99 | head relMSE (x 300-470, y 470-590) |
|---|---|---|---|
| against the mesh bent once (before) | 0.0498 | 0.771 | 1.32 |
| against the mesh bent twice (now) | 0.0087 | 0.648 | 0.134 |

The mesh's own frame changes only in the head (relMSE 0.031 between the two,
p99 0.39) and takes 18.4 s where it took 15.2 (debug build, 768 x 768, 512
paths): the paths through the ball are longer. The table "What the glass is
worth" above was measured against the mesh bent once, and its halfway-out and
rim rows say as much about that mesh as about the cloud.

Test: `athenea_usd_tests "a glass ball bends at its far face*"` -- a smooth
glass ball under a four-colour checker sky, single-sided against
double-sided and one bounce against four, all one picture. Before: relMSE 3.6
and 0.18.

### What a glass cloud's far face lets out

Held to the mesh bent twice, the cloud's head was the right picture a fifth
too bright in blue and a few per cent in red and green (head means 0.322
0.348 0.299 against 0.301 0.326 0.258). It bent at both faces and weighed
only the first: what the near face does not reflect, tinted once by the
colour mesh2splat folds the transmission colour into. The mesh's dielectric
lobe reflects its Fresnel share at the far face too, and tints what crosses
it again -- a ball's yellow is the tint squared.

`SplatSurface::exitThrough` carries both now, from `rt_shade` where the far
face is found: `1 - F` at the exit (glass to air, at the angle the ray meets
it) times the colour once more, as much as the gaussian transmits. Past the
critical angle nothing leaves, the turned ray is all the cloud has, and it
keeps its whole weight. Where no far face is found -- the rasteriser, the
hardware route -- it is one.

| pawn head, against the mesh bent twice | relMSE | p99 | means |
|---|---|---|---|
| one face weighed (before) | 0.134 | 2.59 | 0.322 0.348 0.299 |
| and the far face's Fresnel | 0.100 | 2.38 | 0.307 0.332 0.286 |
| and the tint again | 0.081 | 2.00 | 0.307 0.332 0.261 |
| the mesh | | | 0.301 0.326 0.258 |

The whole frame: relMSE 0.0087 to 0.0069, p99 0.648 to 0.595.

The tint taken again is the colour, which is the base colour times the
transmission colour: exact for the pawn (base colour one), and a base colour
too many for a glass whose base colour is not white. A `standard_surface`'s
base colour does not tint its transmission at all; a cloud keeps one colour a
gaussian and cannot say which part of it is which.

Test: `athenea_usd_tests "a glass cloud lets out*"` -- a ball of 60 000
gaussians at the conversion's glass opacity, tint (1, 1, 0.5), under a sky of
one colour, against the mesh ball: green and blue through the middle 0.98 and
0.51 before against the mesh's 0.92 and 0.26, within 3.3 % now.

### The room through a rough glass is sharper than its reflection

With the weights right the head was still a haze where the mesh shows the
workshop. The transmitted half read the prepared sky at the material's own
roughness (0.23 on the pawn, from its map), and the sky's levels are
reflection lobes: a microfacet tilted by `theta` turns a reflected ray by
`2 theta`, and a ray through an interface by `(1 - 1/ior) theta` going in and
`(ior - 1) theta` coming out -- the first opened by `ior` again where the ray
leaves. Through both faces that is `sqrt(2) (ior - 1) theta`, a third of the
reflection's spread at 1.5. `transmittedRoughness` reads the level that wide:
a level's width goes as the roughness squared, so the roughness is scaled by
the root of that ratio, 0.59 at 1.5.

| pawn head, against the mesh bent twice | relMSE | p99 |
|---|---|---|
| the reflection's roughness (before) | 0.081 | 2.00 |
| the factor at 0.8 | 0.061 | 1.83 |
| at 0.7 | 0.054 | 1.68 |
| **at sqrt(0.707 (ior - 1)) = 0.59** | **0.048** | **1.54** |
| at 0.5 | 0.046 | 1.54 |

The test ball (roughness 0.3) is best at 0.59 (0.045 against 0.048 at both 0.5
and 0.7); the pawn would take a little less. The closed form stays.

Only where both faces were found. The single bend of a route with no tree --
the rasteriser, the hardware route -- is not the image a lens forms, and
sharpening it made the rasterised head worse against the mesh (0.138 to
0.211): there the blur is what hides that it is the wrong picture.

**Where the pawn ends up.** Against the mesh bent twice, the transferred cloud
under autoshop_01 at 768 x 768 and 512 paths:

| | whole relMSE | p99 | largest | head relMSE | head p99 |
|---|---|---|---|---|---|
| before these changes (the mesh bent once) | 0.0498 | 0.771 | 83 | 1.32 | 5.66 |
| before, against the mesh bent twice | 0.0087 | 0.648 | 76 | 0.134 | 2.59 |
| now | 0.0057 | 0.545 | 76 | 0.048 | 1.54 |

The body (x 234-534, y 188-448) is 0.0174 throughout: none of this touches
it. The largest relative value is in the opaque parts, not the glass.

Test: `athenea_usd_tests "a rough glass cloud*"` -- a ball of roughness 0.3
under a 16 x 8 checker sky, cloud against mesh: relMSE 0.119 before, 0.045
now.

**Not done.** The rasteriser's glass bends once and shows the sky alone; a
thick lens there needs the far face without a tree. The mesh has a normal map
on the glass (scratches) that the cloud does not carry. The glass's own
opacity was tried: front faces opaque and far faces gone took the head from
0.047 to 0.055, so the conversion's 0.6 stays.

## lucabRTrender's files, migrated

athenea is lucabRTrender renamed (e8ef1eb), and nothing written before the
rename read any more: the stages applied `Lrt*API`, named `primvars:lrt:*`
and `lrt:*` settings, and streamed `.lrtc` files. `athenea migrate`
(`usd::migrate`, `lod::migrateLrtc`) writes copies under the new names.

- **Layer by layer, not composed.** Each layer is opened, its content moved
  into an anonymous layer (the cached original is never edited, so a process
  that opens it afterwards sees the file) and every spec walked, variants
  included: `apiSchemas` list ops (`Lrt` + capital -> `Athenea`), every
  property name whose namespace has an `lrt` component (a rename of the spec,
  so values, metadata, time samples and connections go with it), connection
  and relationship target paths by the same rule (the rule is a function of
  the name, so a target in another layer or prim is renamed without knowing
  where its property was), `propertyOrder`, `customData` and
  `customLayerData` keys, and `hydra:rendererName`'s value. Then `Export`.
- **Asset paths.** `.lrtc` -> `.athc`. A relative path to a file not copied is
  made absolute when the copy lands in another directory, so it resolves; with
  `--recursive`, layers, packages and `.lrtc` files under `--root` are
  migrated to the same place under the output's directory and the paths name
  the copies (absolute paths too: the Sparrow film layers sublayer each other
  by absolute path). A copy may never land on its original.
- **.lrtc.** Its `Lrtc.cpp` (lucabRTrender e51ca4a, never changed after)
  differs from `Athc.cpp` at e8ef1eb in the magic alone, and the last header
  word was padding written as zero -- version 2's `flags`, no normals. The
  header is rewritten (`ATHC`, version 2), the payload copied in slices
  without decoding, and the result parsed before it takes its name.
- **.usdz.** Extracted beside the output, each member layer migrated as itself
  (no anchoring: a package names everything relatively), a `.lrtc` member
  converted and renamed, and the package written in the same order, so the
  first file is still the root layer.

Measured. A stage written by the test with every old name (a schema, nine
edit primvars, one time-sampled, one with doc and customData, a connection,
a relationship, a render setting, a renderer name, layer data) renders after
migration as the stage authored natively: max 0, over2 0; unmigrated it does
not (the edit is not read); migrated twice it reports no rename; in a
package the same. A `.lrtc` made from a written `.athc` (magic and version)
migrates, reads, cuts and draws as the native file (max 0), and streams
through a migrated stage as the native stage does (max 0). The user's
Sparrow assets: `FilmGs.usda` -r (3 files, 10 properties, 4.8 s; 453 MB
crate), `FilmGsGlass.usda` -r (4 files), `Sparrow60.usdz` (12 properties),
`SparrowClips.usda` -r (73 files, 84 properties through the clip variants,
66 s) render skinned and relit; the originals draw the bird in its bind pose,
unlit.

Not done. A `.athc` inside a `.usdz` is migrated but not drawn: the engine
maps a `.athc` as a file, and a package member is not one. Clip
`templateAssetPath`s and asset-path expressions are reported, not
rewritten. A `.usda`'s `#` comments are not kept (USD's parser drops them).
A schema a stage lost when it was written (lucabRTrender's mesh2splat without
its plugins dropped `LrtSplatCryptomatteAPI` as an unknown token, e.g.
`Sparrow_glass_gs.usdc`) is not there to rename; its primvars are.

## A plugin finds its files from itself, not from its host

hdAthenea loaded by another program -- usdview, Houdini, Blender's
`HydraRenderEngine` -- found its shaders from the executable
(`<exe>/../shaders`), which is the host's: inside Blender that is
`Blender.app/Contents/shaders`, which does not exist, and the plugin fell
back to the build tree it was compiled in, which a packaged plugin does not
have. `platform::moduleDir()` names the image the calling code was linked
into (`dladdr` of a function in core, a static library: the executable for a
program, the plugin for the plugin), and `gpu::shaderDirectory` asks it
first: a `shaders` directory holding `athenea/` at that directory or up to
three above it (`<build>/plugin/usd/hdAthenea/../../../shaders`), then the
same from the executable, then the build tree. A program in `bin/` finds
`bin/../shaders` exactly as before; `$ATHENEA_SHADER_DIR` still wins.

hdAthenea's rpath starts with its own directory (`@loader_path`, `$ORIGIN`),
so a package carrying libslang beside the plugin loads that one wherever it
is unpacked; the build tree's rpaths stay after it.

- `athenea_core_tests "[platform]"`: a test binary's module is its
  executable directory.
- `athenea_gpu_tests "[paths]"`: no device; scratch trees for a program in
  `bin/`, the plugin three levels down, a package whose shaders sit beside
  the library and a host with shaders of its own (the module wins), and a
  `shaders` without `athenea/` (not taken).

Not done: Windows (`GetModuleHandleEx` from an address and
`GetModuleFileName`, in the port).

## A material compiler with MaterialX libraries of its own

hdAthenea's materials are generated by MaterialX 1.39.5's Slang generator
from the node definitions and implementations of the libraries the host's
USD loaded (`HdMtlxStdLibraries`). A host with an older MaterialX -- Blender
5.3 ships 1.39.4 -- has no `genslang` implementations, and its
definitions of `dielectric_bsdf` and `generalized_schlick_bsdf` have one
input fewer than the engine's closures implement: every material failed
("no matching implementation for node 'surfacematerial' matching target
'genslang'", then "not enough arguments" and an undefined
`makeClosureData`). `$ATHENEA_MATERIALX_ROOT` names a directory holding
`libraries/` that the compiler loads instead, through the overload that
already took library roots. Only the compiler reads it: putting 1.39.5's
libraries in USD's own search paths would hand them to the host's Storm,
whose generators are 1.39.4's.

A document hdMtlx builds carries the libraries of the host's USD
(`importLibrary` in `HdMtlxCreateMtlxDocumentFromHdNetwork`), and the
compiler used to copy it and import its own libraries after, skipping what
was there: the host's 1.39.4 definitions and implementations stayed, and
their source files were included beside 1.39.5's (`ClosureData` declared
twice). Now a definition, implementation, typedef or definition graph the
document shares with the compiler's libraries (same name, same kind) is
dropped before they are imported, so the compiler's wins -- which also
covers the UsdPreviewSurface graph the code used to remove by name. Only
definitions: a material named `material` is not the typedef of that name.

- `athenea_material_tests`: all pass (513 assertions, 9 cases).
- `athenea_usd_tests "[materials]"`: all pass (281 assertions, 10 cases).

Not done: a test of the override itself. It is exercised by the Blender
spike (the default cube's material compiles and shades); a test would need
a second MaterialX library tree in the build.

## hdAthenea inside Blender: phase 0

Blender 5.3 renders through Hydra (`bpy.types.HydraRenderEngine`, the
delegate named by `bl_delegate_id`) with its own OpenUSD: 26.03, one
monolithic `libusd_ms` in the namespace `pxrBlender_v26_03__pxrReserved__`,
beside oneTBB 2022.3, MaterialX 1.39.4, OpenColorIO 2.5.0 and OIDN 2.5.0, and
no USD headers. A plugin built against our 26.08 would be a second USD in the
process. So the `blender` branch builds hdAthenea against headers that match
Blender's and links Blender's libraries (`macos-arm64-blender`,
`scripts/build-usd-blender.sh`, `cmake/BlenderUsd.cmake`).

What has to match, and why:
- the namespace, or nothing links;
- Python support: Blender's USD has it, and it changes `VtValue`'s type-info
  table and `TfAnyWeakPtr`'s vtable. Python 3.13's headers; Python's
  symbols resolve against Blender's executable at load
  (`-undefined dynamic_lookup`);
- Blender's `usd_ctor.diff`: `ARCH_CONSTRUCTOR` entries go in a section
  `pxbctor`, and Blender's `libusd_ms` runs only those (checked with
  `otool -l`). A plugin built with the stock header puts its
  `TF_REGISTRY_FUNCTION`s -- its renderer plugin's `TfType` -- where nothing
  runs them;
- the rpath: the header build's prefix holds a libtbb and libMaterialX* of
  its own, so hdAthenea's rpath in this preset is its own directory,
  `@executable_path/../Resources/lib` (Blender's) and Slang's, nothing else.

hdAthenea compiled against 26.03 with no change: none of the 26.04-26.08 API
the engine uses was missing. The gaps were MaterialX's. 1.39.4 has no Slang
generator (MaterialXGenSlang is 1.39.5's) and keeps the hardware generator
inside MaterialXGenShader, where 1.39.5 split it into MaterialXGenHw. The
Slang generator and the seven hardware nodes 1.39.4 lacks (lights, surface,
material compound) are compiled from 1.39.5's sources into 1.39.4's
namespace, over shims that give the GenHw names to 1.39.4's classes
(`integrations/blender/materialx`). Three lines of 1.39.5 name what 1.39.4
lacks and are rewritten at configure time: `requiresLighting` is not an
override (the Slang generator is its only caller), `hwAiryFresnelIterations`
is 1.39.5's default (2), and the `$closureDataConstructor` token, which
1.39.5's `mx_closure_type.glsl` returns and 1.39.4's HwShaderGenerator does
not substitute, is registered by the Slang generator itself. The libraries
are 1.39.5's too (`plugin/materialx`, named to the compiler by
`$ATHENEA_MATERIALX_ROOT`, which the add-on sets), since the engine's closures
implement 1.39.5's definitions.

Measured, Blender 5.3.0 alpha (68609be8e23b), M5 Pro, headless
(`Blender -b --factory-startup`, the add-on registered with
`pxr.Plug.Registry().RegisterPlugins`):
- the default scene (cube, point light, camera) renders through F12 and
  `-f 1` at 320x240; the cube shades with its material (the first material
  compile takes the frame to 11 s; 2-5 s once the shader cache is warm);
- after the render the process holds one `libusd_ms`, one `libtbb`, six
  `libMaterialX*`, one `libOpenColorIO` and the two `libOpenImageDenoise*`,
  all from `Blender.app/Contents/Resources/lib` (`_dyld_get_image_name`);
- splats: Blender's PLY import makes a `GAUSSIAN_SPLAT` PointCloud with
  `position`, `radiance:base` (FLOAT4), `scale`, `rotation` (quaternion) and
  `radiance:sh_0..14`. With the **Hydra** export method no rprim reaches the
  delegate (`TF_DEBUG=HD_RPRIM_ADDED`): Blender's scene delegate exports no
  point clouds. With **USD**, `/usd_scene/<obj>/<obj>` arrives as a `Points`
  prim with `primvars:radiance:sh_N` (float3[], varying), `primvars:rotation`
  (quatf[]), `primvars:scale` (float3[]) and constant `widths` 0.02 -- and
  without `radiance:base`, which the writer drops (it writes no FLOAT4
  attribute). The delegate draws it as white points. Blender reads
  `ParticleField3DGaussianSplat` (usd_reader_particlefield.cc) but writes
  none.

Not done:
- the viewport (`view_update`/`view_draw`): untestable headless;
- splats as splats: done, see *Blender's Gaussian splats, drawn as splats*
  (the delegate reads Blender's `Points` as a cloud; the add-on's export
  hook writes `radiance:base`);
- volumes: OpenVDB is off in this build (Blender's `libopenvdb` 13 has no
  headers in the prefix);
- packaging: hdAthenea still loads libslang from `~/tools/slang` and libwebp
  and zstd from Homebrew; a package carries them beside the plugin
  (`@loader_path`). The shaders are found from the plugin
  (`platform::moduleDir`);
- Blender's `main` already builds USD 26.08 (its `versions.cmake`): a later
  5.3 or 5.4 may ship the same USD the engine is built against, and only
  the namespace, Python and the constructor section would still differ.

## Blender's Gaussian splats, drawn as splats

Blender 5.3 keeps a Gaussian-splat cloud as a PointCloud of type
`GAUSSIAN_SPLAT`: `position`, `radiance:base` (FLOAT4), `scale`, `rotation`
(quaternion) and `radiance:sh_0..14`. Its USD export -- the one a Hydra
render with the USD export method runs too -- writes it as a `Points` prim
with those attributes as primvars, except `radiance:base`, which it has no
USD type for (*"Attribute 'radiance:base' (Blender domain 0, type 13) cannot
be converted to USD"*). It writes no ParticleField.

**The conventions, measured.** `sh3.ply` imported headless
(`bpy.ops.wm.ply_import`), every attribute against the file's own values:
position as stored (no axis change, the object's matrix the identity);
`radiance:base.rgb` = `f_dc_*` exactly (the DC coefficient, not a colour);
`radiance:base.a` = sigmoid(`opacity`) (linear opacity); `scale` =
exp(`scale_*`) (linear); `rotation` = `rot_0..3` normalised, w first;
`radiance:sh_k` = (`f_rest_k`, `f_rest_{k+15}`, `f_rest_{k+30}`), one array a
basis function, rgb -- all to 0 or one float32 ulp. These are exactly
ParticleField's: Blender's own reader of one (`usd_reader_particlefield.cc`)
copies `radiance:base = (coefficient 0, opacity)`, `scale`, `rotation` and
`sh_k = coefficient k + 1` across with no arithmetic. So nothing about the
values needs converting; only the layout differs. The colours are what a
trainer gives, sRGB: a cloud from Blender says nothing of its colour space,
and one that says nothing is a capture (`athenea:splat:linear` unset, in the
linear-blending work).

**Where it is done.** Two routes were weighed: an export hook that authors a
ParticleField, or the delegate reading Blender's `Points` as a cloud. Either
needs `radiance:base`, which only Blender's side can supply, so the add-on
(`blender` branch) has a `USDHook` whose `on_export` -- Blender calls it at
the end of `export_to_stage`, which Hydra's USD scene index uses -- copies
the evaluated attribute onto the Points prim as `primvars:radiance:base`
(float4[], vertex), bytes as they are. The rest is the delegate's, so a
`Points` prim out of any Blender export draws as a cloud wherever hdAthenea
runs, not only inside Blender:
- `HdAtheneaPoints` asks for `rotation`, `scale`, `radiance:base` and
  `radiance:sh_N`; a quaternion rotation, a scale and either radiance make it
  a cloud, handed to `Engine::setSplats` under the prim's id (a prim that
  changes kind leaves the old entry);
- `SplatStreams` takes Blender's layout as it is: `radianceBase` (four floats
  a splat) and `shPlanes` (one array a basis function). The CPU uploads the
  planes end to end into one buffer and reads none of them;
  `scene/streams.slang` takes opacity and DC from `radianceBase` (`kBase`)
  and the rest from the planes (`kShPlanes`, `planeLength`), into the same
  records every other layout gives, then the same decode.

Not an AOFX bundle: this is a layout the cloud loader's own stream kernel
reads, one more pair of indices in it, inside the upload every cloud takes.
mesh2splat is a bundle because it makes new data out of a model; this makes
none.

Not measured yet (the GPU was out when this was written): the test
"Blender's Gaussian-splat points draw as the PLY they were imported from"
(`athenea_usd_tests`, fixture `tests/data/splats/sh3_blender.usda`: Blender's
export of `sh3.ply` with the hook) renders that stage against a
ParticleField athenea wrote from the PLY itself, and against the same Points
without `radiance:base` as the control.

Not done:
- half-precision planes of an odd total of halves are refused (a plane would
  start inside a word); Blender writes float32;
- a cloud from Blender does not mark its colour space; that waits for
  `athenea:splat:linear` (the linear-blending branch), where unset is sRGB,
  which is what Blender holds.
