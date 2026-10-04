English · [Español](operations.es.md)

# Running athenea

This document is for the person who runs the engine: what to set in the
environment, what every subcommand takes, how to do the work the engine is
for, what a USD stage can say to it, and what to do when it refuses.

It does not explain how the engine is built inside. Four documents divide
that:

| Document | Answers |
|---|---|
| [`README.md`](../README.md) | why the engine exists, and one invocation of each thing |
| **this one** | how it is run, and how a scene is authored for it |
| [`development.md`](development.md) | how it is made, and how it is changed |
| [`decisions.md`](decisions.md) | why it is so, against what, and what was measured |

This document never links into the source. It links to the README for
building, to the development document for internals, and reaches the decisions
record only through it. Keep it up to date: an option, an `athenea:*` setting, an
environment variable or a printed error that changes, changes this file and
its Spanish translation in the same commit.

**How to read the option tables.** `athenea --help` prints a description for every
option and no default at all, so the tables here carry what it cannot: the
value, the default, the unit, and the constraint that is checked while the
command runs rather than while it parses. A repeatable option is marked
‹repeatable›. A flag has no value; where a flag reads `!--no-x`, the behaviour
is on and the flag turns it off.

## 1. Getting it running

### 1.1 A binary

The README's *Building* section has the prerequisites with their versions and
prefixes, the three dependency scripts, the submodules, and the four CMake
presets. It is not repeated here. What matters afterwards is the environment
the binary runs in.

### 1.2 The runtime environment

| Variable | What it does |
|---|---|
| `ATHENEA_SHADER_DIR` | where the Slang shaders are read from. Without it the engine looks for a `shaders` directory holding `athenea/` beside the image its code was loaded from, or up to three directories above it -- `<exe>/../shaders` for a program, `<build>/shaders` for the Hydra plugin loaded by usdview or Blender -- then beside the executable, then at the directory it was built with. `athenea info` prints the one in use. |
| `PXR_PLUGINPATH_NAME` | points a USD application at `<build>/plugin/usd`, which holds both the Hydra delegate and the codeless schemas. Needed by any host that is not `athenea` itself. `athenea` registers `<its binary>/../plugin/usd` on its own at start-up, so the schemas a converted cloud applies are written whether this is set or not. |
| `ATHENEA_MATERIALX_ROOT` | a directory holding MaterialX's `libraries/`, read by hdAthenea's material compiler in place of the libraries the host's USD loaded. Unset by default: the host's are used. For a host whose MaterialX predates the Slang generator (Blender 5.3 ships 1.39.4: no `genslang` implementations, older node definitions), pointed at 1.39.5's. Read once, when the first material compiles; the host's own renderers keep theirs. |
| `AOFX_PLUGIN_PATH` | extra directories of AOFX bundles, searched before the system path and before `--path`. |
| `ATHENEA_BACKEND` | which device to open, as a comma-separated order: `metal,cuda,vulkan,d3d12`. Unknown words warn and are skipped. |
| `ATHENEA_GPU_BUDGET` | the device memory this run may hold, in MiB. Without it the budget is Metal's recommended working set (`recommendedMaxWorkingSetSize`); on CUDA and Vulkan there is none unless this sets one. The device prints the one in use (`GPU memory budget: N MiB`). An allocation past it fails as `OutOfMemory` before it is made, streaming budgets and splat shadows are sized against what it leaves (§3.3, §9), and it is how a run is held below what other jobs on the same GPU leave. On Apple silicon an allocation must also fit the physical memory the system has free less 1.5 GiB kept for the rest of the machine, whatever the budget says: past it the machine swaps the GPU's memory and stops drawing its windows. |
| `ATHENEA_SHADER_CACHE` | where compiled shaders are cached between runs. The default is a directory under the platform's cache directory. Deleting it costs one slow first frame. |

A binary built without `ATHENEA_BUILD_VIEW` has no `athenea view` subcommand; that is
the expected shape on a render node with no window.

### 1.3 What a missing optional dependency costs

Everything here is found at configure time, and the effect is felt at run time.
`athenea info` reports the ones it can see.

| Missing | What is lost |
|---|---|
| OpenColorIO | the OCIO view transform. AgX and ACES 2.0 still work, and `--ocio-*` is refused. |
| Open Image Denoise | `--denoise` and `athenea:denoise` do nothing; a path traced frame stays as it gathered. |
| libwebp | `.sog` clouds are refused when read. |
| OpenVDB | `.vdb` volume fields are refused when read. |
| OptiX headers (CUDA) | ray tracing by pipeline on CUDA; inline ray queries still work, so most of the engine does. |
| zstd | `.spz` clouds are refused, with `this build reads no .spz`. |

### 1.4 Metal and CUDA

The engine opens one device and keeps it. On macOS that is Metal; on Linux it
is CUDA where a CUDA device is present, else Vulkan. `ATHENEA_BACKEND` overrides
the order.

What differs between them, in practice:

- **Ray tracing.** Metal has inline ray queries and acceleration structures
  but no ray-tracing pipelines, so the hardware route that needs a pipeline is
  unavailable and `athenea info` says so. CUDA has both, given the OptiX headers.
- **The denoiser.** On Metal, Open Image Denoise runs on the engine's own
  queue. On Vulkan it runs through CUDA with imported memory.
- **Half precision.** Reported by `athenea info`; where it is absent the engine
  keeps float buffers and nothing else changes.

### 1.5 Exit codes, and where an error is printed

Every subcommand prints its errors to standard error and exits with `1`,
except when the GPU ran out of memory, which exits with `3` and says what to
ask for less of: the one failure a script can do something about by trying
again later or smaller. A successful run exits with `0`. There are no other
codes: a pipeline should test the exit status and read stderr, not parse
stdout, which carries the report — timings, counts, the path written.

Out of memory is handled before it is reported. A frame that fails with it
gives back what the engine makes again on demand (the rasteriser's grown
buffers, the ray tracers' structures, the denoiser), gives up one thing more,
and is tried again once: splat shadows first, then a level of detail at a time
(a LOD group's next coarser level, a streamed asset's cut at twice the pixels
and half its streaming budget), up to four. Each step is a warning naming
what it gave up. Only a frame that still fails is the command's error.

`-v` (or `--verbose`) before the subcommand turns on debug logging, which goes
to stderr as well.

## 2. The command line

An invocation is `athenea [-v] <subcommand> [options]`. Every subcommand is listed
below with its full option table.

### 2.1 `athenea info` — the device, and what it can do

Its first line is the version, as `athenea --version` prints it: the project's
own, and the tag `vX.Y.Z` the build came from when it came from a release
(`CHANGELOG.md`).

| Option | Value | Default | Notes |
|---|---|---|---|
| `--backend` | `metal` \| `cuda` \| `vulkan` | the platform's preference | one backend, not a list |

Prints the backend and adapter, whether it rasterises, traces rays by pipeline
and by ray query, whether it has timestamps, half and unified memory, the
Slang version, the shader directory in use, the MaterialX version, the
denoiser, whether OpenColorIO is compiled in, and how many TBB libraries are
loaded in the process. That last line is a test of its own: two TBBs in one
process is a defect.

```sh
athenea info
```

### 2.2 `athenea render` — splat files to an EXR

| Option | Value | Default | Notes |
|---|---|---|---|
| `--splats` | paths ‹repeatable› | — | `.ply`, `.splat`, `.spz`, `.sog`, `.athc` |
| `--points` | paths ‹repeatable› | — | `.ply`, `.xyz`, `.txt`, `.pts`, `.csv`, COLMAP `points3D.txt`/`.bin` |
| `--technique` | `raster` \| `rt` \| `rt-hw` \| `rt-bvh` \| `reference` \| `reference-rt` | `raster` | `reference*` are the GPU ground truth, slow by design |
| `--point-route` | `raster` \| `discs` | `raster` | |
| `--point-size` | number | `0.01` | world units, or pixels with the next flag |
| `--point-pixels` | flag | off | read `--point-size` as pixels |
| `--edl` | number | `0` | eye-dome lighting strength, raster route |
| `--surface` | number | `0` | surface-splatting depth slack, raster route |
| `--eye` | 3 numbers | the target plus 1.2 of the scene's extent in z | |
| `--target` | 3 numbers | the centre of every cloud's bounds | |
| `--up` | 3 numbers | `0 1 0` | |
| `--rotate-x` | 1 number | none | degrees, applied to every cloud; COLMAP clouds want `180` |
| `--scale` | 3 numbers | none | applied to every cloud |
| `--focal` | number | `35` | mm, with a 24.576 mm aperture |
| `--size` | `WIDTHxHEIGHT` | `1920x1080` | |
| `--near` | number | `0.01` | scene units |
| `--degree` | 0..3 | `3` | cap on the harmonics evaluated |
| `--lod` | number | `0` | pixels a merged cell may span; 0 draws every splat |
| `--stream-budget` | integer | `0` | `.athc`: splats kept on the device; 0 reads the file whole |
| `--no-antialias` | flag | off | drop the Mip-Splatting 2D filter compensation |
| `-o`, `--output` | path | `out.exr` | half RGBA plus a `Z` channel |

Checked while it runs, not while it parses:

- a `.athc` needs `--lod`; without it the file is refused;
- `--lod` needs `--technique raster` and no points;
- the non-raster techniques refuse points;
- something must be given: `--splats` or `--points`.

```sh
athenea render --splats capture.ply --size 1920x1080 -o shot.exr
```

### 2.3 `athenea bench` — the same frame, timed

Takes every option of `athenea render` except `-o`, and adds:

| Option | Value | Default | Notes |
|---|---|---|---|
| `--repeat` | integer | `20` | frames drawn |
| `--stages` | flag | off | wait after each stage so each is timed separately; slower, and the reason the numbers differ from a real frame |

Writes no image. With `--stages` it reports the project, depth sort, counts,
emit, tile sort and blend times separately.

### 2.4 `athenea convert` — a splat file into USD, or into a `.athc`

| Option | Value | Default | Notes |
|---|---|---|---|
| `input` | path, required | — | `.ply`, `.splat`, `.spz`, `.sog`, `meta.json` |
| `output` | path, required | — | `.usda`, `.usdc`, `.usd`, or `.athc` |
| `--chunk-splats` | integer | `65536` | `.athc`: splats per streamed chunk |
| `--max-group-fraction` | number | `0.5` | `.athc`: groups per splat the finest merged level may hold |
| `--degree` | 0..3 | `3` | harmonics kept |
| `--rotate-x` | number | `0` | degrees; COLMAP clouds want `180` |
| `--no-camera` | flag | off | do not add `/World/Camera` to the stage written |

A `.athc` output refuses a non-zero `--rotate-x`: the container holds the
cloud as it is, and the turn belongs on the prim that references it.

A stage written here says `metersPerUnit = 1`: none of the input formats
records a unit, and a capture's scale is taken to be metres. A cloud in
another unit is scaled where it is referenced, or the stage's
`metersPerUnit` edited.

```sh
athenea convert capture.ply scene.usda
athenea convert capture.ply capture.athc --chunk-splats 131072
```

### 2.4.1 `athenea decimate` — fewer gaussians, kept where they matter

| Option | Value | Default | Notes |
|---|---|---|---|
| `input` | path, required | — | a stage (`.usd`, `.usda`, `.usdc`: its first gaussian ParticleField), or `.ply`, `.splat`, `.spz`, `.sog` |
| `output` | path, required | — | `.usda`, `.usdc`, `.usd` |
| `--prim` | path | the first | the ParticleField to read, where a stage has more than one |
| `--colour-tolerance` | 0..1 | `0.05` | how far a gaussian's colour or opacity may be from the one replacing it before it counts as different. The colour is compared in the cloud's own space: sRGB for a capture, linear light for a cloud that says `primvars:athenea:splat:linear` -- where the same number is a coarser step in the darks and a finer one in the brights |
| `--outliers` | 0..1 | `0.05` | the share of the gaussians a merge stands for that may be different |
| `--flat-tolerance` | 0..1 | `0.08` | where the gaussians are discs, how much thicker than they are a merge may be, against its width |
| `--reach` | 0.1..10 | `1.5` | how many of its standard deviations a gaussian may stand from the merge that replaces it |
| `--degree` | 0..3 | `3` | harmonics kept |
| `--no-camera` | flag | off | do not add `/World/Camera` |

A stage comes out as it went in, with fewer gaussians: its root layer is
copied -- the rig and its animation, the lights, the camera, the Cryptomatte
manifest, the variants, every constant primvar -- relative asset paths are
anchored to where it came from, and every array a gaussian long is merged
over what each kept gaussian stands for. The log names each one:

```
decimate: carries primvars:athenea:splat:cryptoObject (1 a gaussian)
decimate: carries primvars:skel:jointIndices (4 a gaussian)
```

How each merges is what it is: `skel:jointIndices` with `skel:jointWeights`
as a rig (each joint's weight summed, the four heaviest kept); an array whose
name ends in `shadowBits` bit by bit; any other int -- an id, a part, a sheet
-- as what may not be merged across at all; `primvars:athenea:splat:normal`
as a direction (the weighted mean made a unit vector again); any other float
-- metallic, roughness, a transfer, `primvars:athenea:splat:emission` -- as a
mean. Metallic, roughness and transmission are
also compared as colour is. An array sampled in time is merged a sample at a
time. The copy keeps the source's `metersPerUnit` and `upAxis`. A splat
file is written as a new stage, as `athenea convert` writes one, in metres.

```sh
athenea decimate car_gs.usdc car_fewer.usdc
athenea decimate capture.ply capture_fewer.usdc --colour-tolerance 0.1
```

### 2.5 `athenea stage` — a USD stage through the Hydra delegate

| Option | Value | Default | Notes |
|---|---|---|---|
| `stage` | path, required | — | `.usd`, `.usda`, `.usdc` |
| `--camera` | prim path | the stage's first camera | |
| `--time` | number | `0` | USD time code |
| `--size` | `WIDTHxHEIGHT` | `1920x1080` | |
| `--technique` | `raster` \| `rt` | `raster` | the delegate's `athenea:technique` |
| `--visibility` | `automatic` \| `raster` \| `rays` \| `bvh` | `automatic` | how meshes are seen |
| `--path-samples` | integer | `1` | rt: paths a pixel each pass |
| `--path-bounces` | integer | `1` | rt: bounces after the first hit; going through a glass (into it, out of it, inside it) is not one, up to 8 a path |
| `--path-total` | integer | `1` | rt: paths a pixel the image is drawn until it holds |
| `--denoise` | flag | off | rt: denoise once the total is held |
| `--default-lights` | flag | off | a dome and a sun in the session layer, for a stage with none |
| `--shutter` | `OPEN:CLOSE` | unset | in frames; only for a camera made by `--eye` |
| `--variant` | `/World{set=value}` ‹repeatable› | none | a variant selection before the first frame |
| `--motion-buckets` | 1..8 | `4` | rt: shutter slices |
| `--splat-override` | `PRIM=M,R,T[,R,G,B[,REPLACE]]` ‹repeatable› | none | metallic, roughness, transmission and a tint for the gaussians that came from `PRIM` (a path in the cloud's Cryptomatte manifest, `*` for every one); `-1` keeps the gaussian's own; `REPLACE` 1 makes R,G,B the colour rather than a factor on it. Only a cloud converted with ids has a manifest. Fails on a prim none names |
| `--refine` | integer | `0` | subdivision levels; 0 draws the control mesh |
| `--light-samples` | integer | `1` | samples per light per pixel |
| `--no-transfer-indirect` | flag | the half is added | draw a transferred cloud without its bounced half |
| `--splat-reflections` | flag | off | rt: a gaussian reflects the cloud it belongs to rather than only the sky |
| `--splat-shadows` | flag | off | rt: a relit cloud shadows itself, one ray a splat |
| `--no-antialias` | flag | antialias on | |
| `--no-cloud-shadows` | flag | cloud shadows on | |
| `--cloud-shadow-texels` | integer | `1024` | a side, per light |
| `--cloud-shadow-density` | number | `1.0` | multiplier on the cloud's optical depth |
| `--cloud-shadow-terms` | `0`, `1`, `3`, `5`, `7` | `0` | 1 is the total alone, the rest add Fourier pairs; 0 lets what receives decide |
| `--eye` | 3 numbers | none | a camera of its own, instead of one on the stage |
| `--target` | 3 numbers | none | where that camera looks |
| `--up` | 3 numbers | `0 1 0` | |
| `--frame-all` | flag | off, and implied when the stage has no camera | frame everything |
| `--focal` | number | `35` | mm, 24.576 mm aperture |
| `--fstop` | number | `0` | 0 is a pinhole |
| `--focus` | number | `0` | the depth in focus, scene units |
| `--near` | number | `0.1` | |
| `--far` | number | `100000` | |
| `-o`, `--output` | path | `out.exr` | half RGBA plus `Z` |
| `--render-settings` | prim path | unset | render that prim's products instead, and stop |
| `--frames` | integer | `1` | repeat, and report first, median and fastest |

```sh
athenea stage shot.usda --camera /World/Camera -o shot.exr
athenea stage shot.usda --technique rt --path-total 256 --denoise -o shot.exr
athenea stage shot.usda --render-settings /Render/Settings
```

### 2.6 `athenea view` — a window on a stage

Present only in a build with the viewer. The window's controls are in §5.

| Option | Value | Default | Notes |
|---|---|---|---|
| `stage` | path, required | — | |
| `--camera` | prim path | a free camera framing the stage | |
| `--technique` | `raster` \| `rt` | `raster` | |
| `--visibility` | `automatic` \| `raster` \| `rays` \| `bvh` | `automatic` | |
| `--size` | `WIDTHxHEIGHT` | `1600x900` | in points, not pixels |
| `--frames` | integer | `0` | close after this many and print their timings; 0 runs until the window closes |
| `--light-samples` | integer | `1` | 1 is interactive |
| `--choose-lights` | flag | off | one light a sample, chosen by power |
| `--path-samples` | integer | `1` | rt: paths a pixel each frame |
| `--path-bounces` | integer | `4` | going through a glass is not a bounce, up to 8 a path |
| `--path-total` | integer | `64` | where the frame counts as converged, and is denoised |
| `--denoise` | flag | off | |
| `--no-default-lights` | flag | default lights on | a stage with no lights stays unlit |
| `--edr` | flag | off | extended range: a float surface and ACES 2.0 to the screen's peak |
| `--ocio-config` | path or URI | none; `ocio://studio-config-latest` when a display or view is given | |
| `--ocio-display` | name | the config's default | |
| `--ocio-view` | name | the display's default | |
| `--snapshot` | path | none | with `--frames`: the last frame as shown, panels included, to this EXR |
| `--capture` | directory | none | every frame as shown, panels included, one PNG a frame (`frame_00000.png` and up): a recording of the viewer playing |
| `--isolate` | prim path | none | show that prim's matte alone; implies the Cryptomatte output |
| `--aov` | `color`, `depth`, `primId`, `instanceId`, `elementId`, `Neye`, `normal`, `cryptomatte`, `CryptoObject00`..`02` | `color` | which output the window starts on |
| `--fstop` | number | `0` | the free camera's diaphragm; a stage camera brings its own |
| `--focus` | number | `0` | |
| `--variant` | `/World{set=value}` ‹repeatable› | none | |
| `--hdri` | directory ‹repeatable› | the folder each dome's own image sits in | skies the **Sky** combo offers for the stage's dome lights (`.hdr`, `.exr`) |
| `--play` | flag | off | start with the timeline playing |
| `--every-frame` | flag | off | play a time code per drawn frame rather than by the clock |
| `--shutter` | number | `0` | frames the shutter is open; 0.5 is a 180 degree shutter |

```sh
athenea view shot.usda
athenea view car_gs.usdc --aov cryptomatte
athenea view car_gs.usdc --isolate /root/Kapoot/Object_57 --frames 10 --snapshot matte.exr
```

### 2.7 `athenea live` — a stage on a clock

Each frame is drawn for its ST 2059-1 instant, free-running on this machine's
clock or following a PTP master.

| Option | Value | Default | Notes |
|---|---|---|---|
| `stage` | path, required | — | |
| `--camera` | prim path | the first | |
| `--size` | `WIDTHxHEIGHT` | `1920x1080` | |
| `--technique` | `raster` \| `rt` | `raster` | |
| `--rate` | `25`, `50`, `29.97`, `59.94`, `23.976` or `N/D` | `25` | frames per second |
| `--ptp` | host | empty, this machine's clock | follow this PTP master |
| `--port` | integer | PTP's default | 319 needs privileges; any port both ends agree on |
| `--domain` | integer | `0` | PTP domain |
| `--lock-timeout` | seconds | `10` | how long to wait for a lock |
| `--frames` | integer | `25` | frames written |
| `--start` | number | the stage's start | the USD time that plays first |
| `--at` | `HH:MM:SS:FF` | unset | the UTC wall-clock instant the first frame is for |
| `--tai-utc` | seconds | the alignment's default | TAI minus UTC, for timecodes |
| `-o`, `--output` | path | `live.####.exr` | `#`s become the frame number |

Every frame written carries its timecode and rate as EXR attributes, along
with the TAI instant, the frame index, the USD time and how late the wake was.

### 2.8 `athenea mesh2splat` — a model into gaussians

The conversion itself, and the light bake that follows it. What each stage
does is [development.md §6](development.md#6-baking-a-gaussian-in-full); the
recipe is §3.1 below.

| Option | Value | Default | Notes |
|---|---|---|---|
| `stage` | path, required | — | a stage holding meshes |
| `-o`, `--output` | path | `splats.usda` | `.usda`, `.usdc`, `.usd`, or `.athc` with levels of detail: the gaussians and their shading normals only (no metallic/roughness/transmission, Cryptomatte ids, glass index, up axis or unit); `--skinned`, `--transfer` and `--lod-levels` are refused with it |
| `--prim` | prim path | every mesh | only meshes at or under this path |
| `--hide` | prim path, repeatable | none | left out with all beneath it, as if invisible (a session opinion; the file is untouched) |
| `--resolution` | integer | `512` | cells across the longest side of the box the density is measured over |
| `--lod-levels` | integer | `1` | levels of detail: the conversion again at half the resolution each time; `-o` becomes the stage that draws them as one cloud, each level a `<name>_lod<n>.usdc` beside it |
| `--density` | `per-model` \| `per-mesh` | `per-model` | which box that is |
| `--cell-min` | number | `0`, derived | world units; per-mesh, the finest a cell may be |
| `--cell-max` | number | `0`, derived | world units; per-mesh, the coarsest |
| `--max-splats` | integer | `2000000` | the budget, over the whole stage, shared between the meshes in proportion to what each wants |
| `--cell-from-camera` | camera prim path | none | each mesh's cell is what one pixel of that camera covers where the mesh's box is nearest to it (held to the near plane), at `--time`; replaces `--density`. `--cell-min`/`--cell-max` bound it, given; nothing is derived |
| `--camera-pixels` | integer, 1 to 65536 | `1920` | with `--cell-from-camera`: pixels across the camera's horizontal aperture |
| `--sigma` | number | `1.0` | gaussian width in cells; mesh2splat's own is 0.65 |
| `--flatness` | number | `0.1` | the third size as a fraction of the smaller of the other two |
| `--opacity` | number, 0 to 1 | `1.0` | coverage: how much of what stands behind it the converted surface covers, multiplied into the material's own opacity. Every opacity is coverage -- this, the material's constant, a map's value, what a glass keeps -- and each gaussian takes what one of the several over a point needs for it, so 0.5 covers half at any size |
| `--glass-opacity` | number, 0 to 1 | `0.6` | coverage a fully transmitting solid keeps. A thin-walled glass (and a UsdPreviewSurface opacity under one in its default `transparent` mode) covers what the sheet reflects at its index instead |
| `--opacity-cut` | number, 0 to 1 | `0.5` | where a material's opacity is a map with no threshold of its own (UsdPreviewSurface's `opacity`, standard_surface's `opacity`, OpenPBR's `geometry_opacity`, glTF's `alpha` in BLEND): below this no gaussian is written; above it the surface covers what the map reads. A material's own threshold (`opacityThreshold`, glTF's `alpha_cutoff` in MASK) is used instead, and what it keeps is whole |
| `--max-cells` | integer | `262144` | most cells one triangle may walk |
| `--texture-size` | integer | `1024` | a map is read no larger than this; 0 reads it at its own size |
| `--no-textures` | flag | off | ignore the maps; materials keep their constant values |
| `--normal-map-turns` | flag | off | the normal map turns the gaussian, not only its shading. The shading normal is written either way (`primvars:athenea:splat:normal`) |
| `--no-displacement` | flag | off | ignore the materials' displacement: every gaussian stands on the flat mesh |
| `--displace-refine` | integer, 1 to 64 | `8` | where the relief stretches a cell, split it into at most this many gaussians along each of its two axes |
| `--simplify` | number, 0 to 1 | `0` (off) | a block of cells whose colour, metallic, roughness, emission, cut-out and normals move by no more than this -- over the block and a block past each side, all inside one triangle -- becomes one gaussian of its size. Colours and cut-out are 0 to 1; emission is compared in its own units, so a bright one merges less; normals are compared as the length of their difference, about the angle in radians |
| `--simplify-levels` | integer, 1 to 5 | `3` | the largest block `--simplify` may merge is 2^this cells a side |
| `--no-camera` | flag | camera added | |
| `--no-bake` | flag | bake on | carry the material to be relit instead of baking the light in |
| `--bake-samples` | integer | `128` | paths every gaussian takes first; a transfer takes this plus `--bake-extra` |
| `--bake-extra` | integer | `128` | paths a gaussian on average added after the first pass, shared out by sqrt(relative variance / cost) of what the first pass saw; 0 traces none |
| `--bake-pass-samples` | integer, 1 to 4096 | `64` | paths each added pass gives the gaussians it is for; a gaussian gets at most 16 such passes |
| `--bake-filter` | integer, 0 to 8 | `3` | a-trous iterations of the splat bake filter (the `SplatBakeFilter` bundle) over the baked light, and over a TX transfer's bounced halves (its indirect half and reflected field); 0 filters nothing. The first reaches a cell of 1.5 gaussians, each doubles it |
| `--bake-filter-luminance` | number | `4` | the filter's edge: a neighbour whose light differs by this many standard deviations of the gaussian's own noise counts e^-1 as much. Larger smooths more and keeps less of a faint edge |
| `--bake-filter-indirect-only` | flag | off | filter the indirect light alone and leave the direct as traced |
| `--bake-bounces` | integer | `3` | after the first hit |
| `--bake-degree` | 0..3 | `2` | harmonics fitted; 0 is a colour |
| `--transfer` | flag | off | bake how much of a sky reaches each gaussian instead of the light that did |
| `--indirect` / `--no-indirect` | flag | on | with `--transfer`: keep the bounced half as well; a zonal transfer keeps the direct half alone |
| `--transfer-degree` | 2 or 3 | 3 | with `--transfer`: the harmonics' degree, 16 coefficients direct and 48 indirect at 3, the first transfer's 9 and 27 at 2 |
| `--transfer-cells` | 0, 16 or 32 | 16 | with `--transfer`: cells a side of the grid of open directions over the whole sphere (256 or 1024 bits a gaussian); 0 writes the first transfer's 8 x 8 over the half a gaussian faces |
| `--transfer-slice` | integer | `0` | with `--transfer`: gaussians baked at a time. 0 takes as many as an answer of 1.5 GB holds, and at most a million where the bounced halves are filtered; the bake's own batch (524288) is the least a slice takes unless this asks for fewer. The filter sees the neighbours within a slice |
| `--specular-filter` | 0 to 4 | 1 with `--transfer`, else 0 | how much the turn of the surface under a gaussian widens its roughness and its coat's: the corners' normals apart, over the gaussian's width, added to the slopes' variance (Toksvig). 0 keeps the material's roughness; relit and baked conversions keep 0 unless asked |
| `--validate` | directory | — | measure the conversion material by material against the stage path traced, into this directory (below) |
| `--validate-camera` | prim path | `--cell-from-camera`, else the stage's first camera | the frames' camera |
| `--validate-size` | W H | `960 540` | the frames, in pixels |
| `--validate-paths` | integer | `512` | paths a pixel the GT holds |
| `--validate-bounces` | integer | `6` | bounces of the GT's paths |
| `--validate-material` | prim path or name | every material | only this one (repeatable) |
| `--validate-sky` | `white` or an image file | the stage's lights | every frame under another sky -- a constant of radiance one, or that image on the stage's domes -- with its other lights off; the GT is kept as `gt_<sky>.exr` |
| `--transfer-lobes` | 0 to 2 | `0` | with `--transfer`: keep it as this many zonal lobes in each gaussian's own frame (the `SplatTransferZonal` bundle); 0 is two lobes with `--skinned` and nine harmonics in the world otherwise |
| `--skinned` | flag | off | carry the skeleton; forces `--no-bake`, keeps a `--transfer` as zonal lobes |
| `--range` | `START:END[:STEP]` | the stage's own range | time codes a skinned cloud keeps |
| `--default-lights` | flag | off | a dome and a sun for the bake, on a stage with none |
| `--time` | number | `0` | the instant the stage is posed and the bake traces at |
| `--path` | directory ‹repeatable› | none | extra AOFX bundle directories |

`--skinned` and a bake are refused together: a cloud that moves cannot carry
light baked in one pose, so the conversion says so and keeps the material.
`--skinned` and `--transfer` go together: the transfer is kept as zonal lobes
in each gaussian's frame, which turn with the gaussian (below).

A mesh whose GeomSubsets (`materialBind` family) bind materials of their own
is converted a subset at a time, each with its material, and the faces no
subset claims with the mesh's; the log names each subset's prim. Their
gaussians keep the mesh's Cryptomatte id.

The output is written whole or not at all: under `.<name>.partial-<pid>.<ext>`
in the same directory, and renamed to `-o` once it is complete. A conversion
that fails leaves nothing under `-o` -- or the file that was there before, as
it was -- and removes its partial file. With `--lod-levels`, each level and
the stage that draws them are written so.

### 2.9 `athenea visibility` — what a skinned cloud casts, baked by part

| Option | Value | Default | Notes |
|---|---|---|---|
| `stage` | path, required | — | the cloud's stage |
| `--prim` | prim path | `/World/Splats` | the ParticleField |
| `--skeleton-stage` | path, required | — | the source stage, for the joint hierarchy |
| `--skeleton-prim` | prim path, required | — | the Skeleton on it |
| `-o`, `--output` | path | none, edits in place | write a copy instead |
| `--parts` | integer | `12` | how many parts the rig is split into |
| `--min-joints` | integer | `6` | a subtree smaller than this stays with its parent's part |
| `--grid` | integer | `24` | probes along each axis of a part's box |
| `--octave` | integer | `16` | directions along each side of the octahedral map |
| `--cut` | number | `0.001` | transmittance under which a bake ray stops |
| `--time` | number | `0` | the instant the stage is committed at |

### 2.10 `athenea aofx` — the effect plugins

Two sub-subcommands.

`athenea aofx list` prints every bundle found, loaded or refused, and what each
declares: its effects, their inputs and their parameters with defaults.

| Option | Value | Default | Notes |
|---|---|---|---|
| `--path` | directory ‹repeatable› | none | searched after `$AOFX_PLUGIN_PATH` |

`athenea aofx run` runs one effect over EXR files.

| Option | Value | Default | Notes |
|---|---|---|---|
| `effect` | identifier, required | — | for example `tv.mediapro.aofx.invert` |
| `inputs` | paths, required ‹repeatable› | — | in the effect's input order, or `Clip=path` |
| `-o`, `--output` | path | `out.exr` | |
| `--param` | `name=value` or `name=v1,v2,...` ‹repeatable› | the effect's defaults | |
| `--time` | number | `0` | frame time |
| `--path` | directory ‹repeatable› | none | |

```sh
athenea aofx list
athenea aofx run tv.mediapro.aofx.invert shot.exr -o inverted.exr
```

### 2.11 `athenea compare` — what an image holds, measured on the GPU

One image prints the mean and the largest value of each channel. Two print that
for both, then how far the first is from the second, the second taken as the
reference: the relative HDR error (`relMSE`, p99 and largest relative
difference) and the 8-bit sRGB code-value distribution (p99, largest, pixels
over 2). The CPU only reads the files; every number is a kernel's. The kernels
are the Measure effect's (§7.1), run through the AOFX host as a compositor runs
them, so the bundle must be on the search path: one built in this tree always
is.

| Option | Value | Default | Notes |
|---|---|---|---|
| `image` | path, required | — | an EXR |
| `reference` | path | none | an EXR of the same size |
| `--window` | `X0 Y0 X1 Y1` | the whole image | pixels in [X0, X1) × [Y0, Y1), rows counted from the bottom; the means only, the differences are over the whole image. An X1 or Y1 of 0, or past the edge, is the edge |
| `--heatmap` | path | none | also writes the effect's picture, an EXR the size of `image` |
| `--show` | `source`, `difference`, `relative`, `codes` | `codes` | what `--heatmap` draws: the Measure effect's `mode` (§7.1) |
| `--gain` | number ≥ 0 | `1` | what `--heatmap` is multiplied by; the numbers do not depend on it |
| `--path` | directory ‹repeatable› | none | bundle directories searched after `$AOFX_PLUGIN_PATH` |

A mean keeps its sign, which a difference does not: a white furnace that must
return at most 1, or a converted plane that must cover all of its pixels, is a
mean.

```sh
athenea compare cloud.exr mesh.exr
athenea compare furnace.exr --window 192 192 320 320
athenea compare render.exr golden.exr --heatmap where.exr --show relative --gain 4
```

### 2.12 `athenea migrate` — lucabRTrender's files under athenea's names

athenea is lucabRTrender renamed, and a file written before the rename names
nothing this engine reads: its schemas, its primvars, its settings and its
`.lrtc` clouds are ignored. `migrate` writes a copy under the new names; the
input is never written to.

| Option | Value | Default | Notes |
|---|---|---|---|
| `input` | path, required | — | `.usda`, `.usdc`, `.usd`, `.usdz` or `.lrtc` |
| `-o`, `--output` | path, required | — | the same kind of file: a layer as `.usda`, `.usdc` or `.usd` (a `.usd` keeps the input's encoding), a package as `.usdz`, a `.lrtc` as `.athc`; never the input |
| `-r`, `--recursive` | flag | off | also migrates every layer, package and `.lrtc` the file names -- sublayers, references, payloads, value clips, asset-valued attributes -- that lies under `--root`, each to the same place under the output's directory |
| `--root` | directory | the input's directory | what `--recursive` may copy; it must hold the input, and may not be the output's directory |
| `-q`, `--quiet` | flag | off | prints the warnings and the totals only |

What is renamed, layer by layer, without composing the stage (each layer
keeps its own opinions, variants included):

| Before | After |
|---|---|
| `LrtSplatEditAPI`, `LrtSplatLightingAPI`, `LrtSplatSkinningAPI`, `LrtPointStyleAPI`, `LrtStreamedAssetAPI`, `LrtSplatVisibilityAPI`, `LrtSplatCryptomatteAPI`, `LrtVolumeAPI` in `apiSchemas` | `Athenea…API` |
| any property whose name has an `lrt` component: `primvars:lrt:splat:*`, `lrt:*` render settings, `outputs:lrt:*` | the same with `athenea`; value, metadata, time samples and connections kept |
| a connection or relationship target naming such a property | the renamed property |
| `hydra:rendererName` `lrt`, `HdLrtRendererPlugin` | `athenea`, `HdAtheneaRendererPlugin` |
| `customData` and `customLayerData` keys with an `lrt` component | the same with `athenea` |
| an asset path ending `.lrtc` | `.athc` |
| a `.lrtc` (`LRTC`, version 1) | a `.athc` (`ATHC`, version 1, no normals); the payload is copied as it is |

Asset paths. A relative path to a file that is not copied (a texture, a layer
without `--recursive`, one outside `--root`) is made absolute when the output
is in another directory, so it still resolves (`anchored` in the report).
With `--recursive`, a path to a migrated copy names the copy: relative as it
was, or absolute to where the copy is. Inside a `.usdz` every path stays
relative and a `.lrtc` is converted and renamed in the package; the files keep
their order, so the first is still the root layer.

Every rename is printed, one a line (`schema`, `property`, `target`, `value`,
`metadata`, `asset`, `anchored`, `file`, `warning`), and a total. A file
already migrated is written unchanged and reports no rename. A `warning` is
something left as it was: a property whose new name is already authored
beside it (the new one wins), a `.lrtc` the copy names that does not exist
yet (run `migrate` on it), a `.lrtc` in an expression or a clip template.
USD does not keep a `.usda`'s `#` comments.

```sh
athenea migrate old/shot.usda -o new/shot.usda
athenea migrate ~/assets/Sparrow/FilmGs.usda -o ~/migrated/FilmGs.usda --recursive
athenea migrate cloud.lrtc -o cloud.athc
```

## 3. Tasks

### 3.1 A model into a cloud

`athenea mesh2splat` turns the meshes of a stage into one cloud of gaussians and
writes it as a `UsdVolParticleField3DGaussianSplat`. The defaults convert and
bake the light in; the decisions below are the ones worth making by hand.
What is converted is what the renderer draws: an invisible mesh, or one
under an invisible prim, makes no gaussians, and a PointInstancer's
prototype is converted once per instance, where each instance stands.

**How many gaussians, and where.** `--resolution` is cells across the longest
side of a box, and `--density` says which box: `per-model` measures everything
being converted at once, `per-mesh` measures each mesh on its own. Per-model
is what the original algorithm does, and it means a large object in the scene
dilutes a small one — a sixteen-unit ground plane under a two-unit car gives
the car an eighth of the cells it gets when converted alone. Per-mesh gives
every mesh the same number of cells across its own longest side, with the cell
held in world units between `--cell-min` and `--cell-max` so a badge does not
become finer than the eye can use and a floor does not become coarser than the
frame can show. Left at zero, those bounds derive from the model: the
coarsest is the model's own cell, and the finest an eighth of it.

**What a cell buys, besides detail.** It is also the sharpness of every
reflection the cloud carries: a ray crosses gaussians spread over a couple of
cells of surface, their normals differ by that arc, and the blend averages
their mirror directions over it. Measured on a unit sphere of gold asking for
`specular_roughness` 0.08, against the same sphere path traced as a mesh: a
cell of 0.0281 reads as roughness **0.34**, 0.0141 as **0.20**, 0.0070 as
**0.14**. A cloud reads as the mesh at **`r + 9c/R`**, `c` the cell and `R`
the radius of curvature, so a mirror at roughness `r` wants a cell under
`r/9` of that radius. Fifteen times the gaussians cost 36 % more time a frame
— 13.4 ms to 18.2 ms at 900 x 340 and 64 paths — and sixteen times the disk,
10 MB to 157 MB. What binds is `--max-splats` and the file, not the tracer.

The Mustang hero stage at `--resolution 512`, with a sixteen-unit ground plane
in it:

| Density | Total | Bumper | Bonnet | Ground | Meshes run twice |
|---|---|---|---|---|---|
| `per-model` | 375 199 | 762 | 2 326 | 262 144 | 1 |
| `per-mesh` | 934 338 | 3 082 | 20 279 | 262 144 | 5 |

The ground is at its budget in both: it is two triangles and the cell ceiling
decides it, not the density.

**The budget.** `--max-splats` is a ceiling over the whole stage. Every mesh
(every GeomSubset of one) is counted first -- a run of the effect with room
for one gaussian, which counts everything and writes nothing -- and when they
want more than the budget it is shared in proportion: each gets
`budget × wanted / total` (and one at least), and walks a cell
`sqrt(wanted / share)` times coarser so that it wants about that. That is one
density floor for the whole stage: every mesh loses density alike and none is
dropped. The warning says how many were wanted, and each mesh's log line how
much coarser it walked. A mesh that still wants a little more than its share
after the coarsening keeps its first triangles' gaussians, in the mesh's
order. The count costs one short run a mesh.

**From a camera.** `--cell-from-camera` sizes each mesh's cell by the camera
that will look at it (Mesh2GS's rule): `z × (aperture / pixels) / focal`,
`z` the distance from the camera to the nearest point of the mesh's box (its
near clip at least, and zero inside the box), so one cell covers about one
pixel where the mesh is nearest. Every triangle of the mesh walks exactly
that cell. A free camera that is not in the stage is not known to the
conversion, so this is an option and not the default.

**Textures.** Each map travels to the device as float4, sixteen bytes a texel,
so a 4k map is 268 MB and a car with fifteen of them does not fit. The
conversion samples a map once per cell, and at `--resolution 512` the model is
512 cells across, so most of a 4k map is thrown away before it is read.
`--texture-size 1024` is the default ceiling; `0` reads maps at their own size
and is for a close-up of one object.

**Displacement.** A material's height moves its gaussians, which is where a
cloud is cheaper than a mesh: a gaussian is a point, so raising it costs
nothing, where a mesh has to be cut finer than the relief before it can move
at all. The height is read from UsdPreviewSurface's `displacement` (through
UsdUVTexture's `scale` and `bias`, on the channel connected), or from a
MaterialX `displacement` node (`ND_displacement_float`, times its `scale`)
that the material's displacement terminal names; a constant with no map moves
the whole surface. It is in the mesh's own units, so a mesh scaled by two
displaces twice as far. A vector displacement is not read, and the log says
so.

Each gaussian stands on the relief and is turned to it. Where the relief is
steep a cell of the flat surface becomes a longer patch of the relief, and
the cell is split into as many gaussians along each axis as that stretch asks
for, up to `--displace-refine`; a cell that wanted more is left thinner, and
the log counts them:

```
mesh2splat: 312 cells of relief wanted more than 8 gaussians along an axis and were left thinner (--displace-refine)
```

The count is the cost. A cobbled floor and a cobbled ball at `--resolution
768`: 533 029 gaussians flat, 1 505 305 displaced, most of them on the ball,
whose relief is as tall as its cobbles are wide.

The light bake starts from the point of the flat surface under each gaussian
-- the tracer holds the flat mesh -- and shades it as the relief faces, so
the relief is lit as it is turned. What it does not carry is the relief's
shadow on itself: nothing in the tracer stands where the relief does.
`--no-displacement` converts the flat surface. Hydra's own mesh route ignores
displacement.

**What it gives off.** A material's emission is carried, a gaussian at a time,
as the linear radiance its mesh is rendered with: standard_surface's
`emission` times `emission_color`, OpenPBR's `emission_luminance` times
`emission_color` (nits, multiplied in as they stand, which is what its graph
does), glTF's `emissive` times `emissive_strength`, and UsdPreviewSurface's
`emissiveColor`. A map on the colour is the colour and the weight multiplies
it; a map on the weight is read on one channel and the colour multiplies it.
The log line of a mesh that gives off light says so:

```
mesh2splat: /World/Quad uses /World/Looks/Screen (colour 0.50 0.50 0.50, ..., emission 2.000000 2.000000 2.000000 x '.../emission_gradient.png')
```

It is written as `primvars:athenea:splat:emission` (§4.3) only where some
material of the stage emits. Relit (`--no-bake`) and transferred clouds add it
when they are drawn; a baked one holds it in its colours already. What it
does not do is light anything: the mesh's emissive triangles are a light for
the path tracer, the cloud's gaussians are not. An OpenPBR coat over the
emission, which tints and dims it on the mesh, is not carried.

**What it layers over its base.** What a material puts over its colour,
metalness and roughness is carried as constants of the material: the
dielectric reflection's weight and tint (OpenPBR `specular_weight` and
`specular_color`, standard_surface `specular` and `specular_color`, glTF
`specular` and `specular_color`; the tint is also a metal's edge colour) at
the specular's index (`specular_ior`, `specular_IOR`, `ior`); a clear coat
(`coat_weight`, `coat`, UsdPreviewSurface `clearcoat`, glTF `clearcoat`, with
its roughness and index; UsdPreviewSurface's coat is at its own `ior`), with
OpenPBR's `coat_darkening` (1 by default there, none in the other
vocabularies); and a sheen, its colour times its weight (OpenPBR's fuzz,
`fuzz_weight` x `fuzz_color`; standard_surface `sheen` x `sheen_color`; glTF
`sheen_color`) with its roughness. A UsdPreviewSurface in
its specular workflow (`useSpecularWorkflow` 1) is carried as the index whose
reflectivity head on is its `specularColor`'s brightest channel, tinted by the
colour over it, and no metal. A map on the specular's weight or colour, the
coat's weight or roughness, or the sheen's colour, weight or roughness is
sampled at every gaussian, as the base's maps are, in place of the input's
constant -- the first three maps a material has; past those, and on an index
or `coat_darkening`, the constant stands and the log says so. Each surface's own defaults are what is
assumed where nothing is authored (an OpenPBR coat at 1.6, a standard_surface
one 0.1 rough at 1.5). The log line of a mesh whose material layers anything
says what:

```
mesh2splat: /World/Ball layers specular 1.00 x (1.00 1.00 1.00) at 1.500, coat 1.00 rough 0.00 at 1.450, sheen (0.00 0.00 0.00) rough 0.30
```

They are written (the nine primvars of §4.3 from `specularWeight` on) only
where some material of the stage differs from the plain specular -- weight
one, white, an index of 1.5, no coat, no sheen -- and then for every
gaussian. Relit, transferred and baked clouds alike reflect with them when
they are drawn: a bake keeps the body and leaves the reflections to the frame.
The bake tells a metal from a polish by the material's metalness, so a dark
metal under a lacquer (a car's paint, a 0.05 base) bakes as the metal it is
rather than as nothing.

**Fewer gaussians where the surface is the same.** A gaussian a cell is what
the surface costs wherever it is, and most of a surface -- a painted panel, a
wall, a floor -- is the same from one cell to the next. `--simplify` walks each
triangle as a tree of blocks of 8, 4 and 2 cells a side and makes a block one
gaussian of its size where what the maps say is the same across it, within the
tolerance. A block's gaussian is as wide as the block and its tail reaches a
block past each side, so that reach must agree too and must lie inside the
same triangle: an edge in the texture keeps its cells, and a mesh's border does
not grow a fringe. So it pays on large triangles -- floors, walls, panels --
and not on a mesh finer than its blocks.

A floor of cobbles at `--resolution 768`, flat: 262 145 gaussians, 99 413 at
`--simplify 0.02`. The ball beside it, 2304 triangles, is finer than its
blocks and keeps 270 884. The relief is curved everywhere and merges little:
349 701 to 309 972 at 0.02, 272 664 at 0.1. The walk costs conversion time --
1.3 s to 6.2 s without a bake on that stage -- and the bake then has fewer
gaussians to trace. When simplifying, each gaussian's bake is spread over its
footprint rather than taken at its centre; raise `--bake-samples` with it, or
a glint caught by a few paths is spread over a whole block.

**Fewer gaussians afterwards, on any cloud, with all it carries.** `--simplify` only merges within
large triangles. `athenea decimate` works on the cloud itself, whatever made it
-- a conversion, a capture: it builds the levels of detail and keeps, for
good, the coarsest merge that stands for what is under it, and every splat
where none does. A merge stands for its splats when they lie within
`--reach` of it, when they are discs it is no thicker than their surface
allows, and when few of them (`--outliers`) differ from it in colour by more
than `--colour-tolerance`; and a merge's tail, which reaches past its splats,
may not land on anything of another colour at all. A merge is written wider
than its moments make it, so that it overlaps its neighbours as its splats
overlapped theirs.

Measured on the cobbles converted at `--resolution 768` and baked at 64
paths, drawn by raster at 1200 x 800 against the undecimated cloud:

| | Kept | Image RMS |
|---|---|---|
| defaults | 82.9 % | 0.0020 |
| `--colour-tolerance 0.1` | 37.9 % | 0.011 |
| baked at 256 paths, defaults | 53.9 % | |
| the same conversion unbaked, defaults | 57.8 % | |

A baked cloud carries its bake's noise, one gaussian differing from the next
by a few per cent, and the tolerance has to clear it: at 256 paths the
defaults keep 53.9 % where at 64 they keep 82.9 %. It takes seconds:
533 007 gaussians in 5 s.

**The light bake.** By default the conversion path traces the scene's light
into every gaussian and fits it to harmonics, so the cloud stands under the
light it was converted in and needs no lights to be drawn. `--bake-samples`
and `--bake-bounces` are the quality; `--bake-degree` is how much of the
direction the result keeps — 0 is one colour, 2 is where a highlight starts to
look like one. `--no-bake` keeps the material instead, and the cloud is relit
by whatever scene it is put in.

The bake is taken in two halves, the direct light and the indirect, as sums:
`--bake-samples` paths at every gaussian, then `--bake-extra` more on
average where the first pass was noisiest for what its paths cost, then the
light filtered between neighbouring gaussians of one prim by the
`SplatBakeFilter` bundle, which must be on the AOFX search path while
`--bake-filter` is above 0. On the pawn under an HDRI the defaults take 10 %
longer than 256 paths a gaussian and carry half their error against a
4096-path bake (docs/decisions.md, "The bake's grain").

**A cloud that carries the sky instead of the light.** `--transfer` bakes a
transfer vector: for every gaussian, how much of an environment arrives from
each direction, which is geometry and has no colour and no sky in it. A frame
then combines it with the sky the cloud is actually under, so the same file is
right under any environment, where a baked cloud is right under one. The cost
is the file: nine floats a gaussian for the direct half, and twenty-seven more
for the half that arrived after bouncing off the scene, which `--no-indirect`
leaves out. The rays are the same rays, so the second half is free to bake and
`athenea:splatTransferIndirect` turns it off at render time without re-baking.
A transfer and a light bake are exclusive: one is what the light did, the
other is what any light would do.

**A transfer that turns with the gaussian.** Nine harmonics are in the world
and stay there when a skeleton turns the gaussian. `--transfer-lobes 1` or `2`
-- and `--skinned`, where two is the default -- keeps the direct half instead
as one or two zonal lobes a gaussian, each an axis written in the gaussian's
own frame and three coefficients: ten floats a gaussian against nine, with no
indirect half, and the shadow bits laid out over the gaussian's frame. Every
frame turns the axes with whatever frame the gaussian has, so the transfer
follows the wing. A skinned cloud is posed at `--time` before the bake traces
it, since the stage it traces is posed there; what the lobes hold is what that
pose let through around each gaussian, so occlusion by another limb in another
pose is not in them. The fit is the `SplatTransferZonal` bundle, which must be
on the AOFX search path, and the log says its relative error against the nine
harmonics it replaces, `|f - g| / |f|` over the sphere, as a median, a 90th and
a 99th percentile (each the upper edge of a quarter octave):

```
mesh2splat: transfer kept as 2 zonal lobes in each gaussian's frame for <n> gaussians in <t> ms; relative error against the nine harmonics: median <a>, p90 <b>, p99 <c>
```

**A cloud that moves.** `--skinned` builds the gaussians in the bind pose and
gives each one the joints that carry it, so the cloud is deformed at render
time by the Skeleton it is bound to. A bake is refused with it, because light
baked in one pose is wrong in every other. Each gaussian also keeps how its
weights change across it (`jointWeightGradients`, twelve bytes a gaussian),
which is what makes it stretch across a bend as its triangle does; a cloud
converted before those were written still moves, by the blend of its joints
alone, and gains them by being converted again.

```sh
athenea mesh2splat car.usda --density per-mesh --resolution 512 \
    --max-splats 20000000 -o car_gs.usdc
athenea mesh2splat bird.usda --skinned --resolution 1100 -o bird_gs.usdc
athenea visibility bird_gs.usdc --skeleton-stage bird.usda --skeleton-prim /World/Skel
```

The last line is the third bake: what the cloud casts on itself, by part, so a
wing shadows the body at every pose without a ray. It edits the cloud's file
in place unless `-o` names another.

**A conversion measured by the engine.** `--validate DIR` converts the stage
once a material, with every other mesh left a mesh, and measures each against
the stage path traced, over that material's own pixels -- the mask is the
mesh frame's Cryptomatte, so a mesh of two materials (GeomSubsets) counts in
both. The GT is traced once into `DIR/gt.exr` and read back by the next run of
the same size; delete it to trace it again. Every other option is the
conversion's, so `--transfer`, `--no-bake` and the rest are what is measured.
`--hide` leaves its prims out of everything -- the GT, every frame and the
bake -- through `DIR/hidden.usda`, a layer that switches them off; a GT
traced before with other prims hidden is read back all the same, so give
each `--hide` its own directory. What it writes:

| file | what |
|---|---|
| `validate.json` | per material: meshes, gaussians, the share of the frame, relMSE, p99, mean and GT mean over its pixels, and the mesh rasterised's relMSE and mean |
| `<material>.png`, `.exr` | GT, the mesh rasterised and the cloud, side by side, over the material's box |
| `<material>_gs.exr` | the cloud's whole frame |
| `gt.exr`, `mesh.exr` | the stage path traced, and rasterised as meshes |
| `clouds/<material>.usdc`, `.usda` | the cloud, and the stage it was drawn in |

```sh
athenea mesh2splat car.usda --transfer --cell-from-camera /World/Camera \
    --validate car_validate --validate-size 1920 1080 --validate-paths 512
```

### 3.2 Rendering a stage

`athenea stage` draws through the same Hydra delegate a DCC loads, so what it
draws is what a host sees.

**Two routes.** `--technique raster` rasterises the meshes' visibility and the
cloud's splats and composites them by depth; it is the interactive route and
the one a viewer uses. `--technique rt` path traces the surfaces and
composites the cloud over them. A frame of nothing but splats is traced whole
by the gaussian ray tracer.

**Convergence.** `--path-samples` is what each pass gathers and `--path-total`
is where the image is finished; `athenea stage` draws until the total is held.
`--denoise` runs Open Image Denoise once it is.

**A dome lights a cloud with its image.** A cloud that asked to be relit
(`primvars:athenea:splat:relight`) takes the sky's own direction from a dome that
carries one: its body from the sky's irradiance and its reflection from the
sky convolved to its roughness. Nothing is authored for it; the frame prepares
it when the dome changes. Four domes are prepared, and a fifth is drawn as its
colour.

**Lights.** `--light-samples` is samples per light per pixel, and one is an
interactive frame. `--default-lights` puts a dome and a sun in the session
layer for a stage that authors none, which is what makes an unlit asset
visible without editing it.

**A `DistantLight` is in UsdLux's units.** `intensity` (times `2^exposure`
and `color`) is the radiance of the sun's disc, in nits; with
`normalize = 1` it is divided by the disc's projected solid angle,
`pi sin^2(angle / 2)`, and becomes the illuminance on a surface facing the
light, in lux. An `angle` of 0 is a parallel light whose irradiance is the
intensity either way. So a sun authored without `normalize` lays
`intensity * pi sin^2(angle / 2)`: UsdLux's default sun (50000 at 0.53
degrees) lays about 3.4, and one of intensity 3 at the default angle is all
but black -- author `normalize = 1` for a sun whose intensity is what it
lights with. The default sun is normalized. A `DomeLight` ignores
`normalize`, as the schema says.

**Shadows.** Meshes shadow by ray on the traced route. A cloud casts through a
transmittance map at each light, with no ray at all:
`--cloud-shadow-texels`, `--cloud-shadow-density` and `--cloud-shadow-terms`
control it, and `--no-cloud-shadows` turns it off. A `DomeLight` casts too:
in the map's slots the lights leave (eight in all), six maps along its zenith
and a ring forty degrees up, and a mesh's sample of the dome reads the one
nearest its direction. `--splat-shadows` is the
other direction on the traced route: a relit cloud shadowing itself, one ray a
splat.

**A camera, or one made here.** `--camera` takes a camera on the stage, with
its shutter, f-stop and lens distortion. `--eye`/`--target`/`--up` make one
instead, and then `--focal`, `--fstop`, `--focus`, `--near`, `--far` and
`--shutter` describe it.

**Outputs.** `-o` writes one EXR of the colour and a `Z` channel.
`--render-settings` renders a `UsdRenderSettings` prim's products instead:
each product becomes one EXR whose channels are its render vars. A var's
`sourceName` decides what it reads — `Ci` is the colour, `z` the depth, a
`primvar` source reads a primvar, and a light-path expression of the form
`C.*<L.'NAME'>` reads that light group. `CryptoObject00`, `01` and `02` are
the Cryptomatte layers, and a product that holds them also carries the four
standard attributes that name them, including the manifest that turns an id
back into a prim path.

```sh
athenea stage shot.usda --technique rt --path-samples 8 --path-total 512 \
    --denoise --light-samples 4 -o shot.exr
```

### 3.3 A large cloud: levels of detail and streaming

A `.athc` holds a cloud sorted by Morton code, cut into chunks, with merged
levels above them: a group of splats too small to tell apart at this distance
is drawn as the one gaussian that stands for it. `athenea convert` writes the
container; `--chunk-splats` sets how much travels at once and
`--max-group-fraction` how aggressive the finest merged level is.

At render time, `--lod` is the cut: how many pixels a merged cell may span
before the renderer takes the finer level instead. Smaller is finer and
costlier. `--stream-budget` caps how many splats stay on the device, and the
rest are read as the view asks for them; `0` reads the file whole.

On a stage, the same two are `primvars:athenea:lod:threshold` and
`primvars:athenea:stream:budget` on the prim that names the asset with
`primvars:athenea:asset`.

A `.athc` merges cells, which a cloud a skeleton carries cannot: a cell that
took wing and body would not know which to move with. Such a cloud has
levels of detail by being converted again, coarser (`athenea mesh2splat
--lod-levels`): each level is a ParticleField of its own, with its own rig,
and prims of one `primvars:athenea:lod:group` are one cloud. A view draws the
coarsest level whose `primvars:athenea:lod:cell` (in its own units) spans no more
than `primvars:athenea:lod:threshold` pixels where the cloud is nearest -- the
finest where none does -- and only that level is posed.

A cloud that keeps shading normals (a conversion's, `primvars:athenea:splat:normal`)
keeps them in its `.athc`: four bytes more a gaussian, the merged levels'
the weighted mean of what they stand for made unit again. That is version 2
of the format; a version 1 file, which has none, is still read. The same
version keeps whether the colours are linear light (`primvars:athenea:splat:linear`)
in its header's flags (bit 1, beside bit 0 for the normals); a file written
before has it clear and is read as a capture, sRGB. A cloud that
gives off light (`primvars:athenea:splat:emission`) keeps that too, four bytes
more a gaussian (one RGB9E5 word, after the normals where both are there),
the merged levels' the weighted mean of what they stand for; it is bit 2 of
the header's `flags` (bit 0 is the normals), so a file without it reads as
before. A cloud without normals, without emission and in sRGB is still written as
version 1, so a reader of version 1 alone opens it: version 2 is written only
where the `flags` are not zero. A cloud with a material (`pbr`, and its
layers where it has them) keeps it under bit 4, a word an element and three
more for the layers; one with a transfer keeps it under bit 5, its values as
f16 pairs and its open directions after them -- the counts in a header of
thirty-two bytes after the first, and the merged levels' a group's transfer
averaged by opacity (a zonal one, whose axes do not average, a gaussian's),
its bits set where half the weight has them, its material a gaussian's. Bit
3 is kept for proposal 009. A `flags` bit this build does not know (bit 3,
or past bit 5) is refused, file named: a later bit may add a block, and a reader
that skipped it would read every block after it from the wrong place.

What a budget too small looks like: groups whose chunks have not arrived draw
their merged gaussian, so the cloud is there but blunt, and it sharpens as the
chunks land.

The budget is held to the device's (`ATHENEA_GPU_BUDGET`, §1.2): a stream is
opened with at most half of what the device's budget has left, at 132 bytes
a splat, and an asset asked to be read whole whose file would take more than
that half is streamed instead. Either is printed when it happens
(`streaming budget N splats (~M MiB; asked ...)`). `athenea stage` waits for the streams to settle before a still, so a
rendered frame is never half-arrived.

### 3.4 Colour

Everything inside the engine is linear. An EXR written by `athenea render`, `athenea
stage` or `athenea live` is linear premultiplied RGBA, half by default, with `Z`
in view units; `athenea:exrHalf` on a render settings prim chooses half or float
for its products, and a Cryptomatte layer stays float whatever it says,
because an id rounded to half is another id's name.

A display transform is applied only where an image is shown: the viewer, and
the MCP preview. It offers AgX, ACES 2.0, and OpenColorIO where the build has
it — `--ocio-config`, `--ocio-display` and `--ocio-view` compile that config's
display and view into the kernel. `--edr` asks for a float surface and takes
ACES 2.0 up to the screen's own peak, which on a standard display is the same
image as without it.

A texture is read in the colour space its material or light names --
MaterialX's `colorspace`, a UsdUVTexture's `sourceColorSpace`, USD's
`colorSpace` on the file input, a dome's `colorSpace` on
`inputs:texture:file` -- and brought into the working space (linear
Rec.709) on the device. Names are those of OpenColorIO's studio config
(`srgb_texture`, `lin_rec709`, `acescg`, `g22_rec709`, ...), its aliases and
roles, USD's (`lin_ap1_scene`, `srgb_rec709_scene`, ...) and UsdUVTexture's
(`sRGB`, `raw`, `auto`). `raw`, `data`, `Non-Color`, `none`, `identity` and
`Utility - Raw` read the file as data. No name, or `auto`: an 8-bit image is
sRGB unless the file says otherwise, anything else linear. A name nothing
knows is warned of once and the file is read as it says. Without OpenColorIO
in the build only sRGB, linear Rec.709 and data are known.

`athenea view --snapshot` writes the frame **as shown**, display-encoded and with
the panels in it. It is a screenshot, not a render output.

### 3.5 A sequence on a clock

`athenea live` draws each frame for the instant it belongs to, rather than as fast
as it can. `--rate` is the frame rate, including the drop-frame ones. Without
`--ptp` the clock is this machine's; with it, the engine follows a PTP master
(`genlock-cli master` at the other end) on `--port` and `--domain`, and waits
up to `--lock-timeout` for a lock. `--at` names the UTC wall-clock instant the
first frame is for, `--start` the USD time that plays first.

Each EXR carries its timecode, its rate as a rational, the TAI instant, the
frame index, the USD time and how late the wake was, so a frame can be placed
on a timeline by what is inside it.

## 4. Authoring for this engine in USD

### 4.1 Pointing an application at the plugin

```sh
export PXR_PLUGINPATH_NAME=<build>/plugin/usd
```

That directory holds the Hydra delegate `hdAthenea` and the codeless schemas
together, so one variable finds both. A host then offers the renderer under
its own name, and the settings below appear in its renderer-settings panel.

### 4.2 Render settings

These are authored in the `athenea:` namespace on a `UsdRenderSettings` prim, or
set by a host through the delegate. The first group is declared by the
delegate and appears in a settings panel; the second is read where it is
found and does not.

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `athenea:technique` | token | `raster` | `raster` or `rt` |
| `athenea:visibility` | token | `automatic` | how meshes are seen: `automatic`, `raster`, `rays`, `bvh` |
| `athenea:settleStreams` | bool | `false` | wait for streamed assets before drawing; a still sets it |
| `athenea:lightSamples` | int | `1` | samples per light |
| `athenea:chooseLights` | bool | `false` | one light a sample, chosen by power |
| `athenea:pathSamples` | int | `1` | rt: paths a pixel each pass |
| `athenea:pathBounces` | int | `1` | rt: bounces after the first hit; up to 8 crossings of a glass a path are not counted |
| `athenea:pathTotal` | int | `1` | rt: paths a pixel to converge to |
| `athenea:pathAdaptive` | bool | `false` | stop a pixel once its error is low enough |
| `athenea:pathError` | float | `0.02` | the relative standard error it stops at |
| `athenea:pathMis` | bool | `true` | multiple importance sampling |
| `athenea:denoise` | bool | `false` | denoise once gathered |
| `athenea:motionBuckets` | int | `4` | rt: shutter slices, 1 to 8 |
| `athenea:antialias` | bool | `true` | a sub-pixel offset per pass |
| `athenea:splatTransferIndirect` | bool | `true` | a transferred cloud adds its bounced half |
| `athenea:splatReflections` | bool | `false` | rt: a gaussian reflects the cloud it belongs to, one ray each |
| `athenea:splatShadows` | bool | `false` | rt: a relit cloud shadows itself |
| `athenea:cloudShadows` | bool | `true` | a cloud's transmittance map at each light |
| `athenea:cloudShadowResolution` | int | `1024` | texels a side, per light |
| `athenea:cloudShadowTerms` | int | `0` | 1 is the total; 3, 5, 7 add Fourier pairs; 0 lets what receives decide |
| `athenea:cloudShadowDensity` | float | `1.0` | multiplier on the cloud's optical depth |

Read but not declared:

| Setting | Type | Where | Meaning |
|---|---|---|---|
| `athenea:shutter` | double2 | settings prim | open and close, in frames |
| `athenea:lens` | double2 | camera | aperture radius and focus distance |
| `athenea:exrHalf` | bool | settings prim | write products as half rather than float |
| `athenea:disableMotionBlur` | bool | settings or product | for this product |
| `athenea:disableDepthOfField` | bool | settings or product | for this product |
| `athenea:lightGroup` | string | a light prim | the group a light's contribution is gathered under |

### 4.3 The API schemas

Eight codeless API schemas, applied to a prim like any other. Every attribute
is a primvar, so it inherits down the hierarchy.

**`AtheneaSplatLightingAPI`** — whether a cloud is relit by the scene rather than
showing the radiance it carries.

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:splat:relight` | bool | `false` |
| `primvars:athenea:splat:litBody` | bool | `false` |
| `primvars:athenea:splat:linear` | bool | `false` |
| `primvars:athenea:splat:metallic` | float[] | — |
| `primvars:athenea:splat:roughness` | float[] | — |
| `primvars:athenea:splat:transmission` | float[] | — |
| `primvars:athenea:splat:ior` | float | `0` |
| `primvars:athenea:splat:transferDirect` | float[] ‹9 or 16 a gaussian› | — |
| `primvars:athenea:splat:transferIndirect` | float[] ‹27 or 48 a gaussian› | — |
| `primvars:athenea:splat:transferReflected` | float[] ‹48 a gaussian› | — |
| `primvars:athenea:splat:transferZonal` | float[] ‹10 a gaussian› | — |
| `primvars:athenea:splat:shadowBits` | int[] ‹2, 8 or 32 a gaussian› | — |
| `primvars:athenea:splat:thinWalled` | int[] ‹1 a gaussian› | — |
| `primvars:athenea:splat:curvature` | float[] ‹3 a gaussian› | — |
| `primvars:athenea:splat:schlickMetal` | int[] ‹1 a gaussian› | — |
| `primvars:athenea:splat:normal` | normal3f[] ‹1 a gaussian› | — |
| `primvars:athenea:splat:emission` | color3f[] ‹1 a gaussian› | — |
| `primvars:athenea:splat:specularWeight` | float[] ‹1 a gaussian, 0 to 1› | `1` |
| `primvars:athenea:splat:specularColor` | color3f[] ‹1 a gaussian, 0 to 1› | `(1, 1, 1)` |
| `primvars:athenea:splat:specularIor` | float[] ‹1 a gaussian, 1 to 2.99› | `1.5` |
| `primvars:athenea:splat:coatWeight` | float[] ‹1 a gaussian, 0 to 1› | `0` |
| `primvars:athenea:splat:coatRoughness` | float[] ‹1 a gaussian, 0 to 1› | `0` |
| `primvars:athenea:splat:coatIor` | float[] ‹1 a gaussian, 1 to 2.98› | `1.5` |
| `primvars:athenea:splat:sheenColor` | color3f[] ‹1 a gaussian, 0 to 1› | `(0, 0, 0)` |
| `primvars:athenea:splat:sheenRoughness` | float[] ‹1 a gaussian, 0 to 1› | `0` |
| `primvars:athenea:splat:coatDarkening` | float[] ‹1 a gaussian, 0 to 1, on at 0.5› | `0` |

`relight` says the colours are an albedo the scene's lights must light.
`litBody` says they are already the light on the material's body, so what a
frame adds is the reflection — which is what a baked conversion writes.
`ior` is what its transmitting gaussians bend the sky by. At 0 the
transmitted half is the average of what stands behind, which is translucency;
above one it is the sky along the direction Snell gives at each gaussian's own
normal, which is what makes a glass ball show the room turned. The ray traced
route bends twice: it walks the cloud's own tree to the far face of the object
and bends again coming out, which is what makes a ball a lens rather than a
tinted window, and where the ray then meets the cloud's own particles it shows
those. The rasteriser bends once, at the face the ray enters, and shows the
sky. Neither shows a **mesh** through the glass.
The two transfer arrays are what `--transfer` writes instead: how much of any
sky reaches the gaussian, direct and after a bounce, which the frame combines
with the sky that is there. A cloud that has them needs no `litBody`, and
there is no attribute saying so — carrying them is what says it. Sixteen and
forty-eight a gaussian are degree 3 (`--transfer-degree 3`, the default),
dotted with sixteen harmonics of the sky; nine and twenty-seven are the first
transfer's, read as they always were.
`transferZonal` is the direct half as two zonal lobes in each gaussian's own
frame, written in place of `transferDirect` (`--transfer-lobes`, `--skinned`):
for each lobe the octahedral square's (u, v) of its axis in the frame the
gaussian's orientation gives, then its zonal coefficients for bands 0, 1 and 2
(a one-lobe fit writes the second as zeros). Where both are there it is the one
read. A reader that does not know it draws the cloud relit with no transfer,
which is the whole of its versioning: it is a new primvar, not a new meaning of
an old one; a `.athc` carries either kind under its bit 5.
`shadowBits` is written beside them: sixty-four bits a gaussian, one a cell of
an 8 x 8 octahedral grid over the sphere in the cloud's own space -- over the
gaussian's own frame beside `transferZonal` -- set where the bake's ray in that
direction left the scene. It is what shadows the sun a frame takes out of the
sky; it is read only on a cloud that also carries `transferDirect` or
`transferZonal`, and without it the sun is shadowed softly by the transfer.
Eight or thirty-two ints a gaussian are the TX transfer's (`--transfer-cells`
16 or 32): a 16 x 16 or 32 x 32 grid over the whole sphere, the half behind
the surface traced as well. The count is what says which; nothing else does.
With them a frame shadows the sun and any light that is a direction, and
narrows every reflection -- the base's and the coat's -- by the share of its
own lobe that the bits leave open, rather than by one number for the whole
hemisphere.
`transferReflected` is what the closed directions show: the light arriving at
the gaussian by direction after it met the scene, under a white sky of
radiance one, as sixteen rgb harmonics (degree 3). A frame scales it to the
sky it has -- by how much more the indirect half gathers under that sky than
under the white one, the sun's bounce included -- and a reflection shows it
where the bits say the sky does not reach: the body in the chrome, the
ground in the paint. It is written with the cells and the indirect half, and
read only beside them; `athenea:splatTransferIndirect` turns it off with the
indirect half. A conversion fills it over the closed directions: along each
it holds the mean of what the closed directions about it show, and along an
open one the closed directions' mean, so a closed direction beside an open
one is not read diluted. A cloud converted before that holds the projection
as the bake made it, and is read the same way.
A light that is not the sky -- distant, sphere, disk, rect -- lights a cloud
with the cells through the same lobes, shadowed by the bits over the cone
the light subtends from each gaussian (its penumbra), and its bounce reaches
the body and the reflections as the sun's does. Where a frame also measured
a shadow for it (`--splat-shadows`, or the cloud's shadow map), the darker of
the two stands. The bits
say which ways leave the scene, not which reach a lamp: an object beyond a
lamp that stands among things still shadows it.
A transmitting gaussian of a TX transfer keeps the far half too: its field is
baked over the whole sphere, so the rasteriser shows, through a windscreen,
the sky where the bits behind are open and what the field holds -- the
cabin -- where they are not, along the way straight through, as a slab sends
it. Where a cloud carries the layers, a transmitting gaussian bends by its
own index (`specularIor`) wherever the cloud's `ior` says it bends at all;
the cloud's single `ior` is whichever glass the conversion met first.
`curvature` is how the surface turns under the gaussian: its shape operator
in the gaussian's own first two axes (uu, uv, vv), from the mesh's corner
normals. `athenea mesh2splat --transfer` writes it (not with levels of detail
or `--skinned`), and a frame then turns the reflection across each
gaussian's footprint -- the highlight a car's lacquer shows moves across a
gaussian rather than standing flat on it.
`schlickMetal` is nonzero where the gaussian's metal is a Schlick -- OpenPBR's
and glTF's, from its colour head on to its specular colour at grazing --
rather than the conductor of an artistic index that standard_surface's and
UsdPreviewSurface's are; `athenea mesh2splat` writes it from the material's
vocabulary. For a dark metal the two part by twice at sixty degrees.
`thinWalled` is nonzero where the gaussian came from a thin-walled glass
(OpenPBR `geometry_thin_walled`): the conversion made it as transparent as
the sheet (a card of them stops `2R/(1+R)`, 0.077 at index 1.5) and the frame
shades its reflection alone.
`normal` is the shading normal, apart from the gaussian's frame: which way the
surface faced once the mesh's normal map turned it, in the field's own space,
as positions are. A relit gaussian is lit with it -- put on the side of its
disc the eye is on, since a disc is seen from both -- instead of with its
shortest axis, which is the face's own normal and knows nothing of the map;
the frame keeps answering everything geometric (the footprint, where a ray
meets the disc). `athenea mesh2splat` always writes it (twelve bytes a
gaussian in the file, four on the device); a skeleton that carries the cloud
turns it as it turns the frame; a capture has none.
`linear` says the colours (the harmonics, every degree) are linear light,
linear Rec.709 -- the working space every gaussian is blended in -- and are
drawn as they are. Without it they are taken for a capture's: the sRGB every
trainer fits, made linear a gaussian at a time when the harmonics are
evaluated, before the blend. sRGB appears nowhere else until an image is
shown (the view transform, OpenColorIO). `athenea mesh2splat` writes it on
every cloud it makes -- an albedo, a transfer's albedo, a bake -- and a cloud
written again (`athenea decimate`, an export) keeps it. A capture's look
moves a little: where splats overlap, the mean of their light is brighter
than the light of their mean (decisions.md has the measurement).

`emission` is the light each gaussian gives off by itself, linear radiance in
the scene's units, unbounded: what its material's emission was where it stood
(below). A relit gaussian adds it to what it reflects, transferred or not,
unshadowed and the same from both sides of its disc; one whose colours are
`litBody` does not, because the bake that wrote them met the emission and
holds it already. It lights nothing else -- an emissive gaussian is not a
light, and a mesh beside it is not lit by it. `athenea mesh2splat` writes it
only where some material of the stage gives off light (twelve bytes a
gaussian in the file, four on the device as RGB9E5: three 9-bit mantissas
under a shared exponent, up to 65408, each channel to 1/512 of the
brightest); a capture has none.

The nine from `specularWeight` to `coatDarkening` are what the material
layered over its base, in OpenPBR's units: the dielectric reflection's weight,
tint and index (the tint is also a metal's edge colour), a clear coat over
everything -- a GGX dielectric of its own roughness and index -- and a sheen
(its colour times its weight, Imageworks' lobe). Each layer takes from what
lies under it the share it reflects at the eye, as MaterialX's `layer` does, so
a coat makes the body under it darker at grazing and gives that light back as
its own reflection; with `coatDarkening` the base under the coat is darker
still, by what the coat's inside reflects back into it (OpenPBR's
`(1 - K) / (1 - E K)`). A cloud carries them where any is authored and reads a
missing one at the default in the table; one with none of them reflects with
the plain specular, exactly as before they existed. They are clamped into
their ranges, and on the device they are three words a gaussian, a byte a
value (an index in steps of 1/128, the coat's in steps of 1/64 beside its
darkening's bit). `athenea mesh2splat` writes all nine where some material of
the stage layers anything (52 bytes a gaussian in the file, twelve on the
device). The levels of detail and a `.athc` carry them with the material
(bit 4), a merged group taking one gaussian's.
 — the joints that carry a cloud.

| Attribute | Type | Note |
|---|---|---|
| `primvars:athenea:splat:jointIndices` | int[] | four a gaussian |
| `primvars:athenea:splat:jointWeights` | float[] | four a gaussian |
| `primvars:athenea:splat:geomBindTransform` | matrix4d | |
| `primvars:athenea:splat:skinningXforms` | matrix4d[] | one a joint, the only thing that changes over time |
| `primvars:athenea:splat:jointWeightGradients` | half[] | six a gaussian: the first three joints' weight gradients along its two rest axes, per unit of the cloud's space; the fourth's is minus their sum. Optional |
| `primvars:athenea:splat:skeleton` | string | where it came from |

**`AtheneaSplatVisibilityAPI`** — what a skinned cloud casts, baked by part.

| Attribute | Type | Note |
|---|---|---|
| `primvars:athenea:splat:visibilityParts` | float[] | twelve floats a part |
| `primvars:athenea:splat:visibilityTexels` | int[] | two halves a word |
| `primvars:athenea:splat:visibilityAmbient` | int[] | a probe's mean, for domes |
| `primvars:athenea:splat:visibilityPartOf` | int[] | the part each gaussian belongs to |

**`AtheneaSplatCryptomatteAPI`** — which prim each gaussian came from.

| Attribute | Type | Note |
|---|---|---|
| `primvars:athenea:splat:cryptoObject` | int[] | one id a gaussian |
| `primvars:athenea:splat:cryptoManifest` | string | `{"<path>":"<eight hex digits>", ...}` |

**`AtheneaSplatEditAPI`** — a non-destructive edit over the clouds at and below a
prim: keep what is inside a volume, remove it, or grade it.

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:edit:active` | bool | `false` |
| `primvars:athenea:edit:shape` | token | `box`, or `sphere` |
| `primvars:athenea:edit:mode` | token | `grade`, or `keep`, or `remove` |
| `primvars:athenea:edit:centre` | float3 | `(0, 0, 0)`, in the cloud's own space |
| `primvars:athenea:edit:size` | float3 | `(1, 1, 1)`: half extents of the box, or the sphere's radius in x |
| `primvars:athenea:edit:tint` | color3f | `(1, 1, 1)` |
| `primvars:athenea:edit:saturation` | float | `1` |
| `primvars:athenea:edit:brightness` | float | `1` |
| `primvars:athenea:edit:opacity` | float | `1` |
| `primvars:athenea:edit:minOpacity` | float | `0` |
| `primvars:athenea:edit:maxScale` | float | `0` |
| `primvars:athenea:edit:invert` | bool | `false` |

A grade (`tint`, `saturation`, `brightness`) works on each splat's colour in
linear light, as the blend does: a brightness of 2 doubles the light, and a
capture graded here does not match the same numbers applied to its sRGB.

**`AtheneaStreamedAssetAPI`** — a cloud drawn from a `.athc`.

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:asset` | asset | — |
| `primvars:athenea:lod:threshold` | float | `1` |
| `primvars:athenea:stream:budget` | int64 | `0`, read whole |

**Levels of detail of a ParticleField** — the same cloud converted at several
cells (`athenea mesh2splat --lod-levels`).

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:lod:group` | string | —, drawn as it is |
| `primvars:athenea:lod:cell` | float ‹its own units› | — |
| `primvars:athenea:lod:threshold` | float ‹pixels› | `1` |

**`AtheneaPointStyleAPI`** — how a `UsdGeomPoints` is drawn.

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:sizeInPixels` | float | `0` |
| `primvars:athenea:edl` | float | `0` |
| `primvars:athenea:surfaceOffset` | float | `0` |

**Blender's Gaussian splats as `UsdGeomPoints`** — not a schema: the
attributes of a Blender PointCloud of type Gaussian splat, as Blender's USD
export writes them. A `Points` prim carrying a quaternion `rotation` and a
`scale`, and `radiance:base` or `radiance:sh_0`, is drawn as a splat cloud,
not as points; `widths` and the point styles are then ignored. The values are
what a ParticleField holds (Blender's importer of one copies them across
unchanged): the colours are a capture's, sRGB.

| Primvar | Type | Meaning |
|---|---|---|
| `primvars:rotation` | quatf[] or quath[] | the gaussian's orientation |
| `primvars:scale` | float3[] or half3[] | its three standard deviations, linear, in the prim's units |
| `primvars:radiance:base` | float4[] or half4[] | the DC coefficient (rgb) and the opacity (linear, 0 to 1). Blender's own export drops it; the `athenea_hydra` add-on writes it. Without it the cloud is opaque and its DC is 0 (grey), with a warning |
| `primvars:radiance:sh_N` | float3[] or half3[] | coefficient N + 1 (rgb), N from 0; 3, 8 or 15 of them make degree 1, 2 or 3, and a degree's incomplete remainder is not read |

**`AtheneaVolumeAPI`** — how a `UsdVol` Volume scatters, where no Material is
bound to say it.

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:densityScale` | float | `1` |
| `primvars:athenea:albedo` | color3f | `(0.8, 0.8, 0.8)` |
| `primvars:athenea:anisotropy` | float | `0` |

A Material with a `volume` terminal is the standard way to say the same
thing, and where one is bound it wins.

### 4.4 What the engine ignores, and what it refuses

It **ignores** what it has no meaning for: a primvar it does not read, a
render setting outside the `athenea:` namespace, a schema it does not know.
Nothing is reported, because a stage carries what other renderers need.

It **refuses**, with a message, what it is asked for and cannot do: a `.spz`
in a build without zstd, a `.sog` without libwebp, a `.vdb` without OpenVDB,
mesh visibility by rays on a device with no ray tracing, a `.athc` drawn
without `--lod`, a render product with no resolution or no vars.

## 5. The viewer

`athenea view` opens a window on a stage and keeps every frame on the device: the
display transform writes the window's surface and Dear ImGui draws over it.
Nothing but a picked pixel and a snapshot comes back.

### 5.1 Mouse and keyboard

| Input | What it does |
|---|---|
| left drag | orbit |
| shift + left drag, or middle drag | pan |
| right drag, or the wheel | dolly in and out |
| left click without dragging | pick what is under the cursor |
| `F` | frame the whole stage, and go back to the free camera |
| `Escape` | close the window |

### 5.2 The panels

The panels, by role rather than by widget, since they move as the engine
grows.

**View** holds the frame: which camera (the free one, or any on the stage) and
its focal length, f-stop and focus; the technique and, under `rt`, the paths a
frame, the bounces, the denoiser and how many paths a pixel has gathered so
far; cloud shadows and their density; default lights, offered only where the
stage has none; mesh visibility; the **Output** combo; the view transform and
display encoding; exposure; the shutter; the render scale, which draws at a
fraction of the window and costs proportionally; the timeline with play,
pause, step and *every frame*; the variant sets the stage carries, with a
filter when one has more than a dozen variants; the **Lights** block; the
**Sky** block; and, at the
bottom, the device, the frame's timings, what the last frame held, and what
was picked.

**The sun, taken out of the sky.** When a dome is prepared, its brightest
source is looked for: an octahedral search of 4096 directions, then the source's
own radial profile, and a disc is taken only when its peak clears eight times
the sky's mean and its brightness falls back to the sky within ten degrees.
What is found leaves the nine harmonics -- which cannot hold a half-degree disc
at all -- and is lit as a directional light instead, with an exact cosine.
Energy is conserved to half a per cent: both sides sum the same texels.

Every prepared dome says what was found, which is the one thing about a sky
you cannot see by looking:

```
athenea [info] sky 0: a sun at (-0.539, 0.183, 0.822), 0.0247 sr, irradiance
           0.093 0.080 0.061; it is 4 degrees wide
athenea [info] sky 0: no sun (20 degrees of bright sky, which nine coefficients
           hold well enough)
```

An overcast and a wide bright horizon are left alone on purpose: pulling a
core out of a broad source leaves a ring in the harmonics and puts a hard
terminator where a soft one belongs. The sun reaches a cloud's body only --
the prefiltered map keeps its own disc for reflections, so it is counted once.

**Sky** is one combo a dome light, and a *Turn* slider beside it. The combo
lists every `.hdr` and `.exr` found in the folder that dome's own image sits
in, plus whatever `--hdri` added, so a stage whose sky came out of a library
offers the whole library without being told where it is. Choosing one relights
the frame with nothing else on the stage touched; *(the stage's own)* puts
back what the file asked for. Both are written to the session layer, so the
stage on disk is never edited.

Changing either changes the dome's record, which is what the prepared sky is
keyed on: the next frame rebuilds the sky's nine harmonics and its prefiltered
chain, about 180 ms under a 4k image, and the frames after it cost nothing
extra. Dragging *Turn* therefore pays that rebuild per frame.

**Lights** is one checkbox for every light the stage authored, its schema
beside it, the ones the file left inactive included. Unticking one
deactivates the prim in the session layer, so Hydra removes the light as if
it had never been written; ticking it clears that opinion, and asserts
`active = true` only for a light the file itself left off. The stage on disk
is never edited. It is how a sky is judged alone: a stage's own
`DistantLight` stays on whichever image the **Sky** combo puts on the dome,
and leaves a highlight no sky explains. A dome switched off takes its sky,
background included, and leaves the **Sky** block until it is switched back
on; switching a dome back costs the rebuild a new sky costs.

**Stage** is the prim tree, and what is picked is selected in it.

**Picked** is what a pixel turned out to be, and it opens with the window
rather than waiting to be found: the prim and instance Hydra names, the matte
that names a cloud, and what that prim's gaussians are made of.

**Gaussians** is what the splats on screen are and what the frame did with
them, collapsible section by section, numbers with thousands separators. Over
a stage with no clouds it says *no gaussians in this stage* and nothing else.
It is described once in `modules/ui` (`ui::gaussianPanel`) and drawn after the
frame, so its first row is the frame on screen.

Two kinds of number sit in it, and the panel says which frame each belongs
to. What the frame was handed -- the stage's clouds, what the level of detail
kept, what each carries and holds -- is exact for the frame on screen. What
the device counted is read back without waiting, so it belongs to the frame
named on the **Counted** row, which may be a frame or two behind (under the
raster route today it is the same frame, because the rasteriser already
waits for itself at the end).

| Row | What it is | Unit, and when it is shown |
|---|---|---|
| Frame | the engine's count of frames drawn, and the route: *rasterised*, *splats traced*, or *meshes traced, splats rasterised* | always, over a stage with clouds |
| Counted | the frame the device's counts below belong to, and how far behind it is | raster routes |
| In the stage | every cloud's gaussians, drawn or not | gaussians |
| Submitted | handed to the renderer: after the level of detail, the hidden prims and the variant levels not chosen; the share of *In the stage* | gaussians |
| Visible | kept by the projection, the depth sort's size; the share of the counted frame's submitted | gaussians, raster routes |
| Culled, and a row a reason | *Removed by an edit*, *Outside near/far*, *No area*, *Too faint* (under 1/255 once spread over its footprint: what a too-small gaussian becomes), *Off screen* (outside the frustum's sides), *Touch no tile*; a reason is a row only where it culled something | gaussians, raster routes |
| Tile pairs | (tile, gaussian) pairs, the tile sort's size, and the mean a visible gaussian | pairs |
| Most tiles | the most tiles one gaussian touched | tiles |
| Sort sizes | the depth sort's and the tile sort's keys | keys |
| Traced | what the ray tracer drew: it culls nothing to count | gaussians, `rt` alone |
| Time each stage | a switch: each rasteriser stage then waits for the device, and the frame is slower by those waits | off by default |
| Project ... Total | the rasteriser's stages | ms, while *Time each stage* is on |
| Structures, Build, Trace, Total | the ray tracer: whether its structures were rebuilt or kept, its route, and its times | ms, `rt` alone |
| Clouds (memory) | every cloud's arrays on the device, a posed copy and its skeleton included | bytes |
| Levels of detail | the assets cuts are taken from, and the streaming stores | bytes, where there are any |
| one row a cloud | the prim; its gaussians, how many were submitted, and the device's visible and pairs for it; its level (*level 1 of 3 in 'bird'*, or *cut* with its own and merged gaussians); for a streamed `.athc`, chunks on the device, wanted, missing and loading; what it carries (SH degree, linear or capture sRGB, relit, lit body, transfer, skinned, normals, emission, PBR, ids, baked visibility, ior); what it holds | up to 24 clouds; the rest are in the totals |

The counts cost four small dispatches and a copy a frame, and are taken only
while the panel is open: collapsing its window stops them.

**GPU memory** appears at the bottom of the window when the device ran short:
what the engine gave up (splat shadows, a level of detail) and how many levels
coarser than asked it now draws, or, where nothing was left to give, that the
frame was skipped and the next one tries again. The window stays; closing the
panel dismisses the message until the next time.

### 5.2.1 Changing what a picked prim is made of

The rasteriser keeps a Cryptomatte in every frame it draws, whether or not
the matte is what the window is showing, because a gaussian writes no `primId`
and the matte is the only name a pixel of splats has. It costs the colour
nothing and the frame about a tenth: 13.7 ms to 15.5 ms on a 730 000-gaussian
cloud at 1600 x 900. The traced route writes no matte, so under `rt` a cloud
cannot be picked by id at all and the panel says so with a button back to
Raster.

A picked cloud pixel answers with a Cryptomatte id, and every gaussian
carrying that id came from one prim. The panel therefore counts them and says
what they are made of -- `12 400 gaussians carry it`, then the metallic,
roughness and transmission, with the range beside the mean where a map gave
each gaussian its own.

Under **Say otherwise** are the same three as sliders plus a tint. Moving one
puts a row in the frame's table, keyed on that id: every gaussian of that prim
takes it, at once, on both routes. What you did not move is left as the file
wrote it, so a tint does not flatten a roughness map. **Put the file back**
drops that prim's row and **Put every prim back** drops them all.

It is the frame's opinion and nothing else: the cloud on disk is not touched
and there is nothing to save. A cloud with no Cryptomatte -- a capture, which
came from no prim -- has nothing to address and the panel says so.

### 5.3 Outputs, and isolating a matte

The Output combo chooses what the window shows: the colour, the depth, the
prim, instance and element ids, the eye and world normals, the Cryptomatte
previewed with an id a colour, or any of its three layers raw. `--aov` starts
on one of them.

Picking a pixel names it twice over. Hydra's own pick gives the prim behind
the surface; the matte gives what covers the pixel most, which for a cloud is
the only name it has, since a gaussian writes no prim id. Where a cloud stands
over a mesh the two differ on purpose, and the panel shows both.

With a matte picked, **Isolate this matte** shows that id alone, white on
black — the check that the bonnet's id is the bonnet's. `--isolate <prim>`
does the same from the command line, and turns the Cryptomatte output on by
itself.

### 5.4 Snapshots

`--frames N --snapshot out.exr` closes the window after N frames and writes
the last one **as shown**: display-encoded, with the panels in it. It is how
the images in this repository's documentation are made, and it is not a render
output — for that, use `athenea stage`.

## 6. The MCP server

`athenea-mcp` is the engine as a Model Context Protocol server: newline-delimited
JSON-RPC 2.0 over stdin and stdout. It keeps the device and the stage open
between calls, so the second render of a stage costs what a second render
should.

```sh
claude mcp add athenea -- <build>/bin/athenea-mcp
```

Standard output carries protocol only; logging goes to standard error.

| Tool | What it does |
|---|---|
| `open_stage` | open a stage and keep it; answers with its cameras, up axis and time range |
| `stage_tree` | what is under a prim path: name, type, whether it has children |
| `variants` | the variant sets the stage carries, and makes a selection |
| `device_info` | the GPU opened and what it can do |
| `render` | render the open stage; answers with a preview image and timings. Settings are sticky between calls |
| `render_products` | render a `UsdRenderSettings` prim's products to EXR |
| `pick` | what the last frame drew at a pixel |
| `bounds` | where what the last frame drew is, in world space |
| `convert` | a splat capture into a USD stage |
| `timings` | several frames of the open stage, and their median |
| `settings` | what the session holds: the stage, the last frame, what a render can be asked for |

**`render`'s `output` writes two different files.** A name ending in `.exr`
gets the beauty's own numbers, linear. Any other name gets the frame **as it
is shown**: the AOV that `aov` asked for, through the view transform, as a
PNG. That is the only way to keep a Cryptomatte's colours, a normal's or a
depth's, since none of them is a colour and the beauty's numbers are not
theirs — and it is what a sequence of an AOV is shot with, one call a frame
over a stage that stays open.

Each tool's arguments are declared in the protocol and shown by the client, so
they are not repeated here. The shape of a session is: `open_stage`, then
`stage_tree` to find a camera or a prim, then `render`, then `pick` on
something in the picture. `settings` says what the session currently holds.

## 7. AOFX effects

The engine hosts AOFX plugins, and `athenea mesh2splat` is itself one running
through that host. Bundles are searched in this order, and a duplicate found
twice is loaded once:

1. the directories in `$AOFX_PLUGIN_PATH`;
2. the system path — `/Library/AOFX/Plugins` on macOS, `C:/Program
   Files/Common Files/AOFX/Plugins` on Windows, `/usr/AOFX/Plugins`
   elsewhere;
3. the directories given with `--path`;
4. the bundle directory this build compiled in.

`athenea aofx list` prints what was found, what was refused and why, and what each
bundle declares. `athenea aofx run` runs one effect over EXR files, with
`--param name=value` for anything it declares.

### 7.1 Measure — `rt.sparrow.aofx.measure`

A QC node: `Source` against `Reference` (optional), measured on the device
they are on, the numbers attached to the output. It is `athenea compare`'s
implementation, and the same bundle loads in openFXplayer.

| Parameter | Value | Default | Notes |
|---|---|---|---|
| `mode` | choice: `source`, `difference`, `relative`, `codes` | `codes` | the picture out. `source` passes Source through; `difference` is \|S − R\| × gain, alpha 1; `relative` is a ramp (black, blue, cyan, green, yellow, red) of the largest channel's \|S − R\| / max(\|R\|, 0.001), red at 1 / gain; `codes` colours the pixels whose 8-bit difference is over `threshold`, red at 32 / gain, and leaves the rest black. Without a Reference the picture is Source |
| `gain` | number ≥ 0 | `1` | the heatmap only |
| `threshold` | integer 0–255 | `2` | 8-bit code values a pixel may differ by and not be counted as over |
| `window` | four numbers, pixels | `0 0 0 0` | X0 Y0 X1 Y1 in Source's pixels, rows from the bottom; an X1 or Y1 of 0 is the edge. The means and the largest values only |

What it attaches, floats all. A count is two floats, `high × 2^24 + low`, each
exact (high is 0 below 16 777 216):

| Id | Values |
|---|---|
| `source` | mean R G B A, largest R G B A, sum R G B A, the window's pixels (high, low) |
| `reference` | the same for Reference; only when it is wired |
| `hdr` | relMSE, p99 relative difference, largest relative difference, the relative squared error's sum, pixels (high, low) |
| `codes` | 8-bit p99, 8-bit largest, pixels over `threshold` (high, low), `threshold`, pixels (high, low) |

The relative difference is binned in eighths of an octave from 2^-16, so its
p99 and largest are a bin's upper bound (about 9 %). The sums are there for a
host that divides in double, as `athenea compare` does. Two pictures of
different sizes are refused (`measure compares pictures of one size`), and so
is a window with nothing in it (`measure: an empty window`).

## 8. Reference

### 8.1 Environment variables

Those that change how a run behaves are in §1.2. The rest are for looking
inside a run, and belong to whoever is working on the engine rather than
running it; they are listed in [development.md §3](development.md#3-working-in-the-tree).

### 8.2 Formats

**Clouds read.** `.ply` (3DGS), `.splat`, `.spz` (Niantic, needs zstd), `.sog`
or a bare `meta.json` (PlayCanvas, needs libwebp), `.athc` (this engine's
chunked container).

**Points read.** `.ply`, `.xyz`, `.txt`, `.pts`, `.csv`, and COLMAP's
`points3D.txt` and `points3D.bin`.

**Other inputs.** `.vdb` volume fields through OpenVDB; IESNA LM-63
photometric profiles; and material textures through OpenUSD's image plugins,
so whatever that build reads.

**Images written.** OpenEXR, linear premultiplied, bottom row first inside the
engine and written top row first as the format wants. A frame is half by
default with a `Z` channel; a render product's channels follow its vars, and a
Cryptomatte layer is always float. PNG is written only as the MCP preview.

### 8.3 What each command writes

| Command | Writes |
|---|---|
| `athenea info` | a report on standard output |
| `athenea render` | one EXR: half RGBA and `Z` |
| `athenea bench` | nothing; timings on standard output |
| `athenea convert` | a USD stage, or a `.athc` |
| `athenea stage` | one EXR; or, with `--render-settings`, one per product |
| `athenea view` | nothing, or one EXR with `--snapshot` |
| `athenea live` | one EXR a frame, numbered, each with its timecode |
| `athenea mesh2splat` | a USD stage holding the cloud |
| `athenea visibility` | the cloud's file, edited in place or copied |
| `athenea aofx run` | one EXR |
| `athenea migrate` | a copy of the stage, package or cloud; with `--recursive`, of what it names too |

### 8.4 The scripts

`scripts/` holds what builds the dependencies and what reproduces the assets
the README shows.

| Script | What it does |
|---|---|
| `build-usd.sh [version\|dev]` | builds OpenUSD with MaterialX and OpenVDB into its prefix, without Python |
| `build-oidn.sh [version]` | builds Open Image Denoise, GPU devices only |
| `build-ocio.sh [version]` | builds OpenColorIO with its dependencies linked statically |
| `fetch-fox.sh [dir]` | the Khronos Fox, through Blender, for a skinned conversion |
| `sketchfab-to-usd.sh <zip> [name]` | a Sketchfab archive into a USD asset |
| `readme-images.sh [outdir]` | the images in the README, from the sparrow asset |
| `remote-test.sh [user@host] [preset]` | builds and runs the suite on another machine and brings the log back |
| `scripts/film/` | the sparrow film: its frames, its camera and its shadow |
| `bmw-to-usd.py`, `sparrow-*.py` | authoring helpers for those assets |

A script's own header says what it needs and where it puts things.

## 9. When something fails

| What is printed | What it means | What to do |
|---|---|---|
| `no GPU device` (tests skip) | no device could be opened | check `athenea info`; on Linux set `ATHENEA_BACKEND` |
| `colour: no colour space '<name>' in <config>; read as the file says` | a texture names a colour space neither the config nor the studio config knows | correct the name (`athenea info` says whether OpenColorIO is built in); the texture is read as if no colour space were given |
| `no Measure bundle on the AOFX search path` | `athenea compare` found no `rt.sparrow.aofx.measure` | build the `athenea_aofx_measure` target, or give its directory with `--path` |
| `measure: an empty window` | `athenea compare --window` (or the effect's `window`) holds no pixel | give X0 < X1 and Y0 < Y1 inside the image |
| a shader compile error naming a path | the shaders on disk do not match the binary | rebuild, or point `ATHENEA_SHADER_DIR` at this build's `shaders` |
| `this build reads no .spz` | zstd was missing when this binary was built | rebuild with zstd, or convert the capture elsewhere |
| `.sog` refused | libwebp was missing | install it and rebuild |
| `a .athc is drawn through its levels of detail: give --lod` | a container was given to a plain render | add `--lod`, and use `--technique raster` |
| `mesh visibility by rays: the device has no ray tracing` | `--visibility rays` on a device without it | use `automatic`, which picks what the device has |
| a host does not offer the renderer | the plugin was not found | set `PXR_PLUGINPATH_NAME` to `<build>/plugin/usd` |
| `render product '<path>' has no resolution` / `no vars` | the settings prim is incomplete | give the product a resolution and ordered vars |
| a converted cloud is coarser than asked, and `warning: the meshes want N splats` | `--max-splats` was below what the meshes wanted, and every mesh was coarsened alike to share it | raise `--max-splats`, or lower `--resolution` to choose the coarseness yourself |
| `warning: the budget is exhausted` or `the budget ran out before N mesh(es)` | `--max-splats` was smaller than what the meshes wanted | raise `--max-splats`, lower `--resolution`, or with `--density per-mesh` raise `--cell-min` |
| `warning: N cells lay past --max-cells` | a triangle wanted more cells than one triangle may walk; the rest of it is bare | raise `--max-cells`, or lower `--resolution` |
| a converted cloud is black | the bake found no light | give the stage lights, or `--default-lights`, or `--no-bake` |
| a cloud's reflections look flatter than the mesh's | it carries no shading normal (`primvars:athenea:splat:normal`): converted before conversions wrote one | convert it again; `--normal-map-turns` also turns the discs themselves |
| `cells of relief wanted more than N gaussians`, and the relief shows gaps on its steepest slopes | the relief stretched those cells past the split allowed | raise `--displace-refine`; a pole of the texture coordinates stretches without bound and keeps a few whatever the value |
| a cloud's reflections look softer than the mesh's | the conversion's cell is the blur kernel: a cloud reads as the mesh at `r + 9c/R`, where `c` is the cell and `R` the radius of curvature | convert at a finer `--resolution`: a mirror at roughness `r` wants a cell under `r/9` of that radius. It costs the file, not the frame -- fifteen times the gaussians was 36 % more time a frame and sixteen times the disk |
| a glass ball shows the room but does not bend it | the cloud has no index | `athenea mesh2splat` writes the glass material's IOR; for a cloud from elsewhere author `primvars:athenea:splat:ior` (1.5 is glass). A cloud keeps one index: with two glasses of different IOR the first is kept and the conversion says so; a cloud that carries the layers bends each gaussian by its own specular index instead |
| `<file>: not a readable .athc (unknown flag bits N; this reads bits 0 (normals), 1 (linear), 2 (emission), 4 (material) and 5 (transfer))` | the `.athc` was written by a newer engine, with something in its blocks this build does not know where to find | read it with that engine, or update this one |
| a cloud renders blunt and then sharpens | chunks are still arriving | raise `--stream-budget`, or wait; a still settles first |
| `OutOfMemory: ... does not fit in the GPU's memory budget`, exit code 3 | the frame needed more than the device's budget, even after the engine gave back what it could and stepped down | close what else holds the GPU, render smaller, give a streamed asset a smaller budget; `ATHENEA_GPU_BUDGET` raises or lowers the budget |
| `OutOfMemory: ... does not fit in the memory the system has free` | on Apple silicon, the machine's free memory less its 1.5 GiB reserve would not hold the allocation: other processes hold the rest | close what else runs, or run smaller; the reserve is not configurable |
| `the GPU ran out of memory running ...` | a command buffer failed on the device itself (Metal's `Insufficient Memory`), with the budget not yet reached -- other jobs hold the rest | the same; lowering `ATHENEA_GPU_BUDGET` keeps this run below what they leave |
| `trying again with ...; levels of detail N coarser than asked` | the device ran short and the engine stepped down; the frames go on | nothing, or the same as above to get the detail back (a new run starts as asked) |
| `splat shadows skipped: the proxies of N gaussians need ~M MiB` | splat shadows were asked for a cloud whose proxies would take more than half of what the budget has left (about 1.7 KB a gaussian) | a smaller cloud or its levels of detail; asking again (switching `athenea:splatShadows` off and on) tries again |
| the Storm oracle tests fail | `HDX_MSAA_SAMPLE_COUNT` is not 1 | ctest sets it; set it by hand if running the binary directly |

## 10. Glossary

| Term | Español | What it means here |
|---|---|---|
| splat, gaussian | gaussiana | one anisotropic gaussian: a position, three sizes, a rotation, an opacity and its harmonics |
| cloud | nube | a set of gaussians drawn as one prim |
| capture | captura | a cloud trained from photographs; it has no ancestry in a model |
| conversion | conversión | a cloud made from meshes by `athenea mesh2splat` |
| bake | horneado | writing into the cloud what would otherwise be computed each frame: the light, or what a part occludes |
| cut | corte | which level of detail a frame takes, decided by how many pixels a merged cell spans |
| chunk | chunk | the unit a `.athc` streams: a run of gaussians in Morton order |
| group | grupo | the gaussians a merged level replaces with one |
| part | parte | what one joint carries, and the unit a visibility bake is made in |
| matte | matte | a Cryptomatte layer: which prims covered a pixel, and by how much |
| manifest | manifest | the map from an id to the prim path it stands for |
| product, var | producto, var | a `UsdRenderProduct` and its `UsdRenderVar`: a file, and a layer in it |
| instance record | registro de instancia | what the device holds for one drawn instance of a prim |
| lobe | lobe | one term of a material: a diffuse, a specular, a conductor |
