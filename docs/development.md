English · [Español](development.es.md)

# Working on athenea

This document is for the person who changes the engine: how it is shaped, what
rules it holds to and what each one is there to prevent, how to work in the
tree, how to add the things that are added often, how a gaussian is baked, and
how any of it is checked.

| Document | Answers |
|---|---|
| [`README.md`](../README.md) | why the engine exists, and one invocation of each thing |
| [`operations.md`](operations.md) | how it is run, and how a scene is authored for it |
| **this one** | how it is made, and how it is changed |
| [`decisions.md`](decisions.md) | why it is so, against what, and what was measured |

This document links freely into the source and into the decisions record. It
cites a decision by the title of its section, never by a line number, and it
cites a specification that already lives in a header rather than copying it.
The rules themselves are in [`CLAUDE.md`](../CLAUDE.md), which is the
enforceable copy; §2 here says what each one is for. Keep this file up to
date: a module, a rule, a way of adding something or a step of the bake that
changes, changes this file and its Spanish translation in the same commit.

## 1. The shape of the engine

### 1.1 The modules, and the header that specifies each

`modules/<name>` is a static library `athenea::<name>`. They are listed in
`modules/CMakeLists.txt` in dependency order, and each links only the ones
above it. `CLAUDE.md` has the table of what each holds; what follows is the
other half of it — for each module, the header whose comment is that
subsystem's actual specification. When one of them disagrees with this
document, the header is right.

| # | Module | Read this first |
|---|---|---|
| 1 | core | `core/Result.h` (the error convention), `core/Platform.h` (the whole OS surface), `core/Hash.h` |
| 2 | ui | `ui/Controls.h` (what a panel is), `ui/ViewerPanels.h` (the viewer's panels), `ui/GaussianPanel.h` and `ui/GaussianReport.h` (the Gaussians panel, and the record the engine fills for it) — described once, drawn by `athenea view` and the iOS app |
| 3 | sched | `sched/FrameClock.h` — genlock, PTP and ST 2059-1 alignment |
| 4 | image | `image/Image.h` — the host-side image the AOFX host passes about |
| 5 | io | `io/Sog.h`, `io/Exr.h`, `io/Vdb.h` — one per format, each with its own layout |
| 6 | gpu | `gpu/Device.h` (backends, the shader directory, the cache), `gpu/ComputeKernel.h` (binding by name), `gpu/AsyncReadback.h` (numbers from a frame without waiting for it) |
| 7 | gpu_host | `gpu_host/Context.h` — one device, two runtimes on it, one thread that talks to it |
| 8 | colour | `colour/ColourCompiler.h` (OpenColorIO as a compiler of Slang functions and LUTs), `colour/ColourNames.h` (what a colour space's name means) |
| 9 | scene | `scene/GpuClouds.h` — the cloud layout every renderer reads |
| 10 | render | `render/TileRasterizer.h` and `shaders/athenea/splat/frame.slang` (the pipeline), `render/GaussianRayTracer.h` |
| 11 | geom | `geom/Skinner.h`, `geom/Subdivision.h` |
| 12 | material | `material/MaterialCompiler.h` — MaterialX into Slang |
| 13 | light | `light/LightTable.h` — a light on the device |
| 14 | world | `world/GpuScene.h` — the scene as every technique reads it |
| 15 | technique | `technique/PathTracer.h`, `technique/SplatVisibility.h`, `technique/Environment.h`, `technique/DisplayTransform.h`, `technique/MaterialPrograms.h` |
| 16 | lod | `lod/Athc.h` — **the only specification of the `.athc` format**, as a page map; `shaders/athenea/lod/lod_decimate.slang` for what a decimation keeps, `lod_attributes.slang` for what it carries (`usd::decimateStage` is the whole of it) |
| 17 | usd | `usd/MeshStage.h` (reading a stage without Hydra), `src/Engine.h` (the frame), `usd/Migrate.h` (what lucabRTrender's names became, and `athenea migrate`) |
| 18 | mcp | `mcp/Server.h` — the JSON-RPC transport and what a tool is |
| 19 | aofx | `aofx/Features.h`, `aofx/Version.h` — the ABI, copied verbatim from its own repository |
| 20 | view | `view/Viewer.h` — the window's options |

One thing that looks like a violation and is not: `render` (10) links
`aofx::aofx` (19). That target is headers only, an `INTERFACE` library, and
what `render` takes from it is `Mat4` and `Vec3`. The ordering rule is about
compiled libraries.

### 1.2 One GPU, one thread

Hydra's `Sync` runs on whatever thread the host gives it, and may run several
at once. It never touches the device: it takes the lock and hands the engine
CPU-side records. The render pass thread then commits them — uploads, opens
`.athc` files, builds acceleration structures — and draws. So the device has
exactly one caller, and everything about ordering follows from that.

`gpu_host::Context` is where the other runtime fits: gpe adopts the device
slang-rhi opened, so one `MTLDevice` or one CUDA context serves both and
buffers are shared without copies. The crossing itself -- adopting, a
slang-rhi view of a gpe buffer, a gpe handle on a slang-rhi buffer -- is
three free functions in `gpu_host/Views.h`, on the bare device and pool; the
context's methods call them, and so does a program that has a GPU thread and
a pool of its own and wants only the crossing.

### 1.3 A frame, rasterised

1. `Engine::render` takes the projection, the settings and the AOV request.
2. Meshes: `world::GpuScene` holds the pools and the instance records;
   a visibility route writes `(instance + 1, triangle)` per pixel into the
   visibility buffer. Three routes write the same thing —
   `VisibilityRaster`, `VisibilityTrace`, `VisibilityBvh` — and a test holds
   them to it.
3. `technique::MaterialShading` (or `HeadlightShading`) rebuilds the surface
   from that buffer and shades it. Where the device traces rays it is three
   kernels: `drawLobes` evaluates the material and writes its lobe samples'
   directions, `traceShadows` traces every light and lobe sample's shadow ray
   into a bit, and `shadeMaterials` lights the pixel by those bits. A kernel
   that evaluates a material holds no ray query: on Metal the two together
   wrote rows of garbage. `AovShading` rebuilds the same surface for
   the ids, the normals and the primvars; `CryptoShading` for the matte's id
   plane.
4. Points, where there are any, are rasterised into a layer of their own and
   composited with the meshes by view z (`layers_nearest.slang`).
5. That opaque layer is handed to `render::TileRasterizer` as its `under`,
   and the splats are projected, sorted, tiled and blended over it, in linear
   light like the layer under them. The pipeline's stages are named in
   `shaders/athenea/splat/frame.slang`.
   While a panel asks (`RenderSettings::countSplats`), four small kernels
   count what the projection did -- why each culled splat was culled, which
   the projection leaves in the culled slot's depth key, and each cloud's
   share -- and `gpu::AsyncReadback` copies the counts out with the frame's
   own submit and a fence. They are read later, never waited for: a number a
   panel shows must not be a stall the panel caused.
6. Domes are painted behind, exposure is applied, and the frame is done.
   `technique::DisplayTransform` is what turns it into something to look at,
   and only a viewer or a preview calls it.

### 1.4 A frame, path traced

The difference begins at step 3: `technique::PathTracer` integrates from the
same visibility buffer, so the two routes shade the same surface built the
same way, and can be compared pixel for pixel. Splats are still composited by
the rasteriser over what the tracer produced. A frame of splats and nothing
else is traced whole by `render::GaussianRayTracer`, whose two routes
(hardware acceleration structures, compute BVH) share
`shaders/athenea/rt/rt_integrate.slang`.

### 1.5 The shader layer

`shaders/athenea/<area>` mirrors the modules: `common`, `algo`, `scene`, `splat`,
`rt`, `points`, `reference`, `lod`, `geom`, `world`, `material`, `light`,
`technique`, `volume`, `usd`, `view`, and `test` for the check kernels.

- **The cross-module contract is `common/packing.slang`**: how a splat's
  opacity, scale, quaternion and DC colour are packed into four words, the
  optional shading normal into one (`packNormal`), the optional emission
  into one (`packRgb9e5`) and the optional layers over the base -- specular,
  coat, sheen -- into three (`packLobes`). Anything that writes a cloud and
  anything that reads one goes through it. An optional buffer of `GpuSplats`
  (`pbr`, `normals`, `emission`, `lobes`) is bound whether or not the cloud has it --
  the shape in its place -- and a flag in the parameters says which.
- **A splat's index is not its record's.** Validation drops what cannot be
  drawn; `GpuSplats::origin` says which record each kept splat came from.
  Anything else a file keeps a gaussian and a kernel reads by splat goes
  through `CloudLoader::keptOnly`, and back through `toRecords` to be written.
- **Every splat is blended in linear light**, linear Rec.709, as meshes,
  points and lights are; sRGB appears only where an image is shown
  (`DisplayTransform`, OpenColorIO). A cloud says which space its colours are
  in (`GpuSplats::linear`, `primvars:athenea:splat:linear`): a capture's are
  the sRGB it was trained in and are made linear a splat at a time, right
  after the harmonics are evaluated (`common/color.slang`, `cloudLight`), in
  every projection -- the rasteriser's, the ray tracer's shade, the
  references'. Nothing after that decodes or encodes a colour: not the
  blend, not relighting, not a grade, not a bake. Anything that copies a
  cloud copies the flag.
- A file declares `module <basename>;` matching its filename. A sibling is
  imported bare (`import frame;`), another area by its dotted path (`import
  athenea.common.packing;`). Anything another module uses is `public`, buffers
  included.
- C++ names a module by its path with slashes: `ComputeKernel::create(library,
  "athenea/technique/crypto_ids", "cryptoIdsPass")`.
- A kernel that evaluates materials must walk its pixels in quad order
  (`atheneaQuadPixel`), because bump takes its screen derivatives from the
  thread's quad. A kernel that walks them in raster order gets flat bump and
  nothing says so.
- **A kernel of the splat raster stays WGSL.** An atomic is an `Atomic<T>`
  (`.add`, `.max`, `.load`, `.store`), in group memory too, never
  `InterlockedAdd` on a plain value: WGSL has atomics only as a type, and the
  same source gives Metal and CUDA their own. A barrier sits only under
  control flow that is the same for the whole group -- bounds from
  `SV_GroupID` rather than the pixel, a decision from group memory read
  through `workgroupUniformLoad` -- or Tint refuses the kernel.
  `scripts/wgsl-report.py` checks all of them.

## 2. The rules, and what each is there to prevent

The rules are stated in [`CLAUDE.md`](../CLAUDE.md). Here is the failure each
came out of, because a rule whose reason is forgotten is a rule that gets
argued with.

**No CPU arithmetic on scene data.** Not a defect but a design: a CPU path
that exists is a CPU path that gets used, and then two implementations drift
and the slow one becomes the oracle. So `SLANG_RHI_ENABLE_CPU` is off, the
ground truth is a GPU renderer (`render::ReferenceRenderer`), the comparison
is a kernel (`compareImages`), and a test reads back counters, not pixels.
The CPU reads files, parses headers, decompresses and keeps counts.

**Shader parameters are bound by name.** Binding by slot means counting slots,
and counting slots is how openFXplayer's D30 bugs happened: a parameter added
in the middle of a structure moves every one after it, and nothing fails to
compile. `cursor["name"].setBinding(...)` fails loudly instead.

**`Result<T>` and `ATHENEA_TRY`, no exceptions.** A renderer that throws through a
command buffer leaves the device in a state nobody can describe. Where a
dependency throws — OpenUSD does — the exception is caught at that module's
boundary and turned into an `Error`.

**The operating system lives in `core/Platform`.** Every OS call the engine
makes outside its dependencies is behind one header, so the Windows port has
one file to start from rather than a search. A file a command writes for
another step to read is written through `platform::writeAtomically`: under a
partial name beside it, renamed when complete, so a failure leaves no half
file under the name asked for.

**aofx changes only additively.** openFXplayer's bundles must keep loading, so
the SDK headers are copied verbatim from their own repository and frozen by
`aofx_sdk_manifest`, which hashes every header and fails on any edit,
addition or removal. A parameter is added by appending a `ParamDesc`, never by
changing one.

**One Slang, one slang-rhi, one TBB.** Two TBBs in a process is two thread
pools fighting over the same cores, and it is invisible until something is
mysteriously slow. `athenea info` counts them and the `single_tbb` test asserts
the count.

**`docs/decisions.md` is updated with the change.** The record is what makes a
decision arguable a year later: what was decided, against what, what it was
measured at, what is not done.

## 3. Working in the tree

### 3.1 Build and test

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
ctest --preset macos-arm64-debug                     # every test, one at a time
ctest --test-dir build/macos-arm64-debug -R lod      # the ones whose name matches
build/macos-arm64-debug/bin/athenea_lod_tests "chunks*"  # one case, or a [tag]
cmake --build build/macos-arm64-debug --target athenea_render_tests
```

The prerequisites, their versions and prefixes, and the four presets are in
the README's *Building*. Tests run one at a time because they share one GPU:
every test preset sets `jobs: 1`, and running two suites at once is how a
timing becomes meaningless.

**The TX gate.** A change to shading or to the conversion runs the gate
before anything else: `ctest -L tx_gate` (and `tx_conversions_render_like_the_mesh`,
whose glass ball is held to the first transfer too). It converts the pawn of
the OpenChessSet under its workshop sky (`ATHENEA_BENCH_DIR`) and the
Corvette (`ATHENEA_ASSETS_DIR`, labelled `slow`) with `--transfer` and with
the first transfer, through `--validate`, and fails where a material comes
out more than 5% worse in TX (`tests/regress/tx_against_first.cmake`); its
GT|mesh|cloud pictures stay under `build/<preset>/tests/tx_gate`. A machine
without those assets skips it.

### 3.2 A shader-only change

Shaders are compiled at run time, not embedded, so a change to an existing
`.slang` file needs the copy step that any build performs and no C++ rebuild
at all. A **new** shader file needs a CMake re-configure, because the glob
that copies them is `CONFIGURE_DEPENDS`.

To check that a shader compiles without building anything:

```sh
~/tools/slang/bin/slangc shaders/athenea/<dir>/<file>.slang -I shaders \
    -target metal -entry <entry> -stage compute -o /dev/null
```

For WGSL, what a browser compiles, `-target wgsl` does the same for one
kernel, and `scripts/wgsl-report.py --markdown` for every kernel of the splat
raster and the Measure effect at once: whether each compiles, its storage
buffers and workgroup bytes against the web's limits, and Naga's and Tint's
verdicts (Tint through Dawn's null backend, `scripts/wgsl-tint.cpp`). All on
the CPU. Naga alone is not a check: it took a barrier under a branch on group
memory that Tint refuses.

**The web module** (`docs/operations.md` 3.6) is `shaders/athenea/web/` -- the
kernels whose native counterparts are over the web's limits (decode, project,
blend) as the T1 preset -- with the raster's own sort, prefix, compact, emit
and ranges, compiled by `scripts/web-kernels.py`, and `web/athenea-webgpu/`,
the page's JavaScript. `host.js` repeats the order of `TileRasterizer::render`,
`RadixSort::sort` (the chunked route) and `PrefixSum::apply` until W-host is
the engine's C++ in wasm: a change to a dispatch there, or to a kernel the web
route uses (its bindings' names, its uniform), is a change to `host.js` too,
and `node check.mjs` on a rebuilt module says whether they still agree. It runs
on the CPU over a WebGPU stand-in that validates what a browser validates and
computes nothing.

Generated shaders — the materials, the shading and path-tracing kernels — are
written out by `ATHENEA_SHADER_DUMP=<dir>`, and compile with `slangc -I shaders -I
<dir>`, which is how a kernel's compile time is measured outside the process.

### 3.3 Looking inside a run

| Variable | What it does |
|---|---|
| `ATHENEA_SHADER_DUMP` | write every generated Slang module to this directory |
| `ATHENEA_MTLX_DUMP` | write the MaterialX document handed to the generator, one per material |
| `ATHENEA_SHADOW_DEBUG` | log the cloud shadow maps: casters, lights, texels, mean transmittance |
| `ATHENEA_VISIBILITY_DEBUG` | log the per-part visibility factors |
| `ATHENEA_CUDA_LDG` | `=1` leaves CUDA's read-only cache path in place, reproducing a defect the prelude otherwise works around |
| `ATHENEA_OPTIX_INCLUDE` | the OptiX headers handed to nvrtc, overriding the build-time path |
| `ATHENEA_TEST_DUMP` | a directory for the images tests dump; without it they dump nothing |
| `ATHENEA_VIEW_SWITCH_AT` | `=N`: the viewer flips its technique at frame N, as a click would, for reproducible `--frames` runs |
| `ATHENEA_VIEW_ORBIT` | `=R`: the viewer's free camera turns R radians about its target every frame, as a drag would, to time a moving camera with `--frames` |
| `ATHENEA_STAGES` | `=1`: a line a rasterised frame of splats saying where it went -- commit, per-part visibility, project, counts, depth sort, emit, tile sort, blend -- each stage waited for, so the frame is slower for it |
| `ATHENEA_PORTABLE_SORT` | `=1`: every radix sort takes the chunked passes, as before the tiled route existed; to set a backend's tiled route aside |
| `ATHENEA_ORACLE_*` | the Storm side-by-side oracle's inputs; seven of them, documented where the test reads them |
| `HDX_MSAA_SAMPLE_COUNT` | must be `1` for the Storm oracle; ctest sets it, and the test fails with an explanation if it is not |

`ATHENEA_BACKEND` runs the whole suite on one backend: `cuda`, `vulkan`, `metal`,
`d3d12`.

### 3.4 Another machine

`scripts/remote-test.sh [user@host] [preset] [branch]` pushes the branch,
builds and runs the suite there, and brings `LastTest.log` back into
`build/remote/<preset>/`. It parses the skip messages out of the log, because
a skip is not a pass.

## 4. How to add something

### 4.1 A compute kernel

1. Write `shaders/athenea/<area>/<name>.slang`: `module <name>;`, the buffers it
   declares, a `ConstantBuffer<Params>`, and `[shader("compute")]
   [numthreads(...)] void <entry>(uint3 tid: SV_DispatchThreadID)`. The entry
   must not be called `main`; Metal reserves it.
2. `gpu::ComputeKernel::create(library, "athenea/<area>/<name>", "<entry>")`.
3. Dispatch it with **threads**, not groups — the group count comes from the
   reflected `[numthreads]`, so the kernel bounds-checks itself:

```cpp
kernel.dispatch(batch, {count, 1, 1}, [&](rhi::ShaderCursor cursor) {
    cursor["positions"].setBinding(cloud.positions.rhi());
    cursor["params"]["count"].setData(count);
});
ATHENEA_TRY(batch.submit(true));
```

Every name the shader declares must be bound, whether or not the kernel reads
it this time. That is what the placeholder buffers in the rasteriser are for.

### 4.2 A test

Add sources to an existing `athenea_test(...)` in `tests/CMakeLists.txt`, or a new
one. The helper links Catch2, depends on the shader copy, sets
`ATHENEA_SHADER_DIR`, and registers the cases. One label per binary: `gpu` (the
default), `gpe` (needs gpe's device), `display` (needs a window), so a machine
without one excludes it by label rather than reading its skips as passes.

In the test body:

```cpp
ATHENEA_REQUIRE_GPU(gpu);                        // or SKIP("no GPU device")
gpu::Buffer stats = test::uintBuffer(*gpu->device, 8, "check.stats");
gpu::ComputeKernel check = test::kernel(*gpu, "athenea/test/thing_check");
// dispatch, submit, then read back counters — never pixels
std::array<uint32_t, 8> row = test::reduceStats(*gpu, stats, rows);
```

`test::dumpPpm` and the EXR dumps write only when `ATHENEA_TEST_DUMP` names a
directory, and nothing is asserted on them: they are for a human to look at.

The first tag is the area (`[render]`, `[usd]`, `[technique]`), and the rest
narrow it (`[crypto]`, `[mis]`, `[chi2]`).

### 4.3 An AOV

Five places, and a name missing from any of them fails silently:

| Where | What to add |
|---|---|
| `modules/usd/src/Engine.h` | a value in `enum class AovKind` |
| `modules/usd/src/Engine.cpp`, `Engine::aovView` | where it lives on the device: buffer, source kind, stride, offset |
| `modules/usd/src/RenderPass.cpp` | the Hydra token (or prefix) to `AovSource`, and what the frame must compute in `AovRequest` |
| `modules/usd/src/RenderDelegate.cpp`, `GetDefaultAovDescriptor` | the format Hydra allocates and its clear value. Missing here, the buffer is never allocated |
| `modules/usd/src/StageRenderer.cpp`, `displaySource` | the name to a `DisplaySource::Kind`, for `athenea view --aov` and the MCP preview |

Two more places carry the list as prose and will go stale quietly: the MCP
tool's help string in `modules/mcp/src/Tools.cpp`, and the comment on
`StageRenderer::displaySource`.

### 4.4 A codeless schema

Two hand-written files, no generated C++:

- `modules/usd/schemas/generatedSchema.usda` — the class, written in
  `usdGenSchema`'s output form. Every property is a `primvars:athenea:*` so it
  inherits down the hierarchy and the delegate reads it as a primvar.
- `modules/usd/schemas/plugInfo.json` — a `Types` entry with
  `schemaKind: "singleApplyAPI"` and `bases: ["UsdAPISchemaBase"]`.

Both are copied beside `hdAthenea` into `<build>/plugin/usd/atheneaSchemas/resources/`,
so one `PXR_PLUGINPATH_NAME` finds the delegate and the schemas together. The
prim adapter that reads the new primvars goes in `modules/usd/src/`.

### 4.5 A module

Add it to `modules/CMakeLists.txt` **in dependency order** and give it
`athenea_add_module(<name> SOURCES ...)`. Its public headers go in
`modules/<name>/include/athenea/<name>/`, its private ones beside its sources.
Write the header comment that will be its specification: this document's §1.1
table is the index of those, and a module without one is a module nobody can
read.

### 4.6 A CLI subcommand

`apps/athenea/src/Cmd<Name>.cpp` with a `void add<Name>(CLI::App&)`, declared in
`Commands.h` and called from `main.cpp`. Options are registered on the
subcommand and read in its callback; an error is printed to stderr and the
callback throws `CLI::RuntimeError(1)`. Then document it: an option that is
not in [`operations.md`](operations.md) does not exist for whoever runs the
engine.

### 4.7 An aofx plugin

A bundle under `plugins/<name>` built by the SDK's CMake helper. The effect
declares its inputs and parameters and is handed pictures and numbers — it
never sees USD, a scene or a device API, which is exactly why the same binary
runs in openFXplayer. `athenea mesh2splat` is the largest example in the tree, and
§5 of this document is what it does. `plugins/measure` is the smallest that
replaces engine code: `athenea compare` runs it through the host
(`aofx_host::renderEffect`) instead of calling `render::imageStats`,
`compareHdr` and `compareImages`, so the command and a compositor's QC node
are one implementation. Its kernels are those three's, rewritten for binding
by order -- the same sums in the same order, so the digits printed are the
same -- and `athenea_aofx_tests "[measure]"` holds every number it attaches to
theirs. A port of engine kernels into a bundle is held to the originals that
way: the originals stay, as the tests' reference.

## 5. The test suite

Nineteen binaries, one per area, all under `tests/`.

| Binary | Covers |
|---|---|
| `athenea_core_tests` | the hash Cryptomatte names things by, and its manifest |
| `athenea_gpu_tests` | prefix sum, radix sort, textures, uniforms |
| `athenea_scene_tests` | loading and decoding clouds |
| `athenea_render_tests` | the rasteriser against the reference, points, ray tracing, motion, lens, SPZ, SOG |
| `athenea_geom_tests` | meshes, skinning, curves, subdivision |
| `athenea_material_tests` | the texture store, the lobes, the MaterialX compiler |
| `athenea_technique_tests` | visibility, display, materials, lights, splat shadows and visibility |
| `athenea_lod_tests` | building levels, cut selection, `.athc` |
| `athenea_volume_tests` | volumes |
| `athenea_usd_tests` | the delegate end to end |
| `athenea_storm_oracle_tests` | Storm as the geometry oracle; its own process and device, and `HDX_MSAA_SAMPLE_COUNT=1` |
| `athenea_mcp_tests` | the JSON-RPC server and its tools |
| `athenea_host_tests` | the plugin through `UsdImagingGLEngine`; **links no `athenea::usd` on purpose** |
| `athenea_view_tests` | the viewer; label `display` |
| `athenea_gpu_host_tests` | gpe adopting slang-rhi's device; label `gpe` |
| `athenea_aofx_tests` | the SDK, the host, the mesh2splat effect, and the Measure effect against render's comparison; label `gpe` |
| `athenea_coverage_tests` | what a converted surface covers against its mesh (`tests/data/coverage`); needs the `mesh2splat_coverage` fixture, which ctest runs first, and skips without it; label `gpe` |
| `athenea_sched_tests` | the frame clock and PTP |

Plus nine tests that are not Catch2: `aofx_sdk_manifest` (the SDK's hashes),
`single_tbb` (one TBB in the process), the two `mesh2splat_density_*`
that run the real CLI and assert on the line it prints, the four
`compare_cli_*`, which run `athenea compare` on two fixture renders
(`tests/data/compare`) and hold what it prints to the text it printed before
it ran the Measure effect (`tests/aofx/CheckCompare.cmake`), and
`mesh2splat_coverage`, the real CLI converting the planes
`athenea_coverage_tests` measures (a fixture: `athenea_test(... FIXTURES
<name>)` makes every case of a binary require one). And one Catch2 case
ctest runs on its own, `materialx_root`: `athenea_usd_tests
"[materialx_root]"` with `ATHENEA_MATERIALX_ROOT` set, since the engine reads
the variable once a process. The case is hidden (`[.materialx_root]`), so the
discovery that registers every other case leaves it out.

**Why `athenea_host_tests` links nothing.** It drives the plugin the way a host
does, by name. Were it to link `athenea::usd` as well, a template instantiated in
both — the `make_shared` of the render pass — would bind to the executable's
copy, and the pass would stop recognising the plugin's render buffers. The
test exists to catch exactly that.

**The oracle pattern.** A test generates its input with a kernel, checks the
result with a kernel, and reads back a handful of numbers. A check kernel
lives in `shaders/athenea/test/`, takes the buffers it is checking plus a `stats`
buffer, and `InterlockedAdd`s a counter per property it tests — so a failure
says how many pixels were wrong, not merely that something was.

A timing worth keeping goes in `decisions.md` with the hardware and the
preset in its caption, not in a comment.

**Validating a conversion** (`athenea mesh2splat --validate`,
`apps/athenea/src/Mesh2SplatValidate.cpp`). `usd::stageMaterialGroups`
reads the stage's bindings (meshes and their GeomSubsets) on the processor;
the conversion runs once a group with every other mesh in `--hide`, through
the same `runConversion` the command runs once; the frames are
`StageRenderer`'s; the mask, the masking and the side-by-side pictures are
`athenea/usd/m2s_validate.slang`; the numbers are the Measure effect's, over
the mask's box, divided by the mask's sum.

## 6. Baking a gaussian, in full

Three different things are called baking here, and they happen in this order:
the **conversion** of a mesh into gaussians, the **light bake** that fills
those gaussians with the radiance leaving the surface, and the **visibility
bake** that records what a skinned cloud casts on itself. The first two are
`athenea mesh2splat`; the third is `athenea visibility`. What follows is all three,
from the boundary inwards.

### 6.1 The boundary: a host, and an effect that has never heard of USD

The conversion is an AOFX effect. It is handed a picture of triangles, up to
three maps, and a handful of numbers, and it writes gaussian records into the
picture it is given. That is all an AOFX effect can be handed — and it is
exactly why the same binary runs unchanged inside openFXplayer.

So the stage is the host's work, in `apps/athenea/src/CmdMesh2Splat.cpp`: open it,
read its meshes, triangulate them on the device, pack the triangles into a
picture, turn every texture a material names into rows of linear float4, run
the effect once per mesh, and write what comes back as a
`UsdVolParticleField3DGaussianSplat`.

The pictures live in the AOFX host's own image storage, which on a device with
unified memory is the memory a kernel reads. The triangles are written where
the effect will read them and nothing is copied. The records stay on the
device from there to the file: each run's picture is laid out into the
cloud's own buffers by `athenea/usd/mesh2splat_gather` (records, the bake's
rays, the joints), the rays' offset is set from the cloud's box by
`mesh2splat_span`, the bake answers into a device buffer that
`mesh2splat_bake` writes into the records, and the export decodes them where
they are (`usd::DeviceSplatRecords`). What crosses to the processor is counts
and, at the end, the values a USD array holds — which are the processor's
business by definition.

### 6.2 One gaussian's life, in fifteen lines

1. A mesh is triangulated on the device and packed into a picture: six
   `float4` an entry — position with `u`, normal with `v`, three times.
2. Its box is reduced on the device, and the cell size follows from it.
3. One thread takes one triangle. It projects the triangle onto the two axes
   its normal points along least, and walks the cells of that projection its
   bounding box covers.
4. A cell whose centre falls inside the triangle, and that a cut-out map does
   not erase, is one gaussian.
5. A prefix scan over triangles turns those counts into the slot each
   triangle's gaussians will occupy.
6. The emit walks the same cells again. At each one it interpolates the
   position, the normal and both sets of texture coordinates, and samples the
   maps there.
7. The gaussian's two sizes across the surface are one cell wide, scaled by
   `--sigma`; its third is a fraction of them.
8. Its frame is the triangle's longest edge, the surface normal, and their
   cross product; that frame becomes a quaternion.
9. Its opacity is the material's times whatever the cut-out map reads.
10. Its colour is the material's colour times the albedo map, tinted towards
    the transmission colour.
11. Its metallic and roughness are the material's times the map's.
12. Its id is the Cryptomatte hash of the prim it came from.
13. The record is written, and the slot moves on.
14. Afterwards, unless `--no-bake`, a ray is traced from the gaussian along
    its shading normal and the light that comes back is fitted to harmonics.
15. The whole cloud is written to a USD stage, with the primvars and schemas
    that say what it carries.

### 6.3 Reading the stage, without Hydra

`usd::MeshStage` opens the stage directly. Hydra is a renderer's interface to
a scene, and this is not rendering: it wants the meshes as authored, their
materials narrowed to what a gaussian can carry, and nothing composed for a
frame.

`geom::MeshBuilder` triangulates on the device, in `HdMeshUtil`'s order, so a
converted cloud's faces correspond to what Hydra would have drawn.

**A mesh of several materials.** The GeomSubsets of a mesh's `materialBind`
family are read with the material each binds (`StageMesh::subsets`) and
handed to the builder, which says on the device which subset every triangle
is in (`GpuMesh::triangleSubsets`). The conversion then runs a *piece* at a
time: each subset's triangles, listed in the mesh's order by
`athenea/usd/mesh2splat_subset` (a flag, a prefix sum, a scatter) and packed
as a mesh of their own (`mesh_pack`'s `listed`), with that subset's material;
and the triangles no subset claims, with the mesh's. A piece's gaussians keep
the mesh's Cryptomatte id: the matte names prims as Hydra does.

The triangle picture (`shaders/athenea/usd/mesh_pack.slang`) is six `float4` an
entry: for each of the three corners, the position with the first texture
coordinate in `w`, then the normal with the second in `w`. A second UV set,
where the material uses one, travels in a picture of its own.

Textures become rows of linear `float4` through `material::TextureStore`. A
map travels as sixteen bytes a texel where the file holds one, so a 4k map is
268 MB on the device and a car with fifteen of them does not fit at all. The
conversion samples a map once a cell, and at `--resolution 512` the model is
512 cells across, so most of a 4k map is thrown away before it is read:
`--texture-size` caps them at 1024 by default, and `0` reads them as they are.

A material's displacement is read as a height: UsdPreviewSurface's
`displacement` through UsdUVTexture's `scale` and `bias` on the connected
channel (`StageTexture::scale`, `bias`), or a MaterialX `ND_displacement_float`
named by the material's displacement terminal, times its `scale`. The result
is `StageMaterial::displacementMap`, `displacementScale` and
`displacementBias`; `StageMesh::displacementUnit` is the cube root of the
transform's volume, since a height is authored in the mesh's own units.

**The cells and the budget, before anything is converted.** What each piece
will walk is worked out on the device over the boxes the packing folded
(`athenea/usd/mesh2splat_cells`): the model's cell, the bounds derived per
mesh, the cell each piece walks -- from `--cell-from-camera`'s camera where
one is given -- and what the effect is handed for it. Then every piece is
counted (a run with room for one gaussian: the effect counts everything a run
would write), and where the total is over `--max-splats` the budget is shared
in proportion to what each wants and the cells are worked out again with each
piece coarsened by `sqrt(wanted / share)`. The host does the shares' integer
arithmetic -- counts and slots -- and relays the kernel's numbers to the
effect.

### 6.4 The one kernel: count, scan, emit

`plugins/mesh2splat/mesh2splat.slang` is a port of Electronic Arts'
mesh2splat from its OpenGL pipeline into one compute kernel. Theirs is a
geometry shader and a fragment shader; there is no rasteriser here, so the
fragments are walked — the same samples the rasteriser would have produced, at
the same density, keeping the atomic append that makes the output's order
nobody's business.

**The projection.** A triangle is projected onto the two axes its normal
points along least — their triplanar choice — with the position taken relative
to the box and divided by a range. That is what their render target
rasterises, so one cell of it is `range / resolution` of the world.

**The Jacobian.** `J = V (O)^-1`, the map from that projection to space. Its
columns are how far a step of one projection unit moves in the world; a step
of one cell is that over the resolution, and a gaussian is `--sigma` of one
cell wide.

**Which box, and how big a cell.** `range` is where the two densities differ.
Per model it is the wider of the projection plane's two extents, which makes a
cell depend on which way a triangle faces: on a car 0.77 wide, 1.92 long and
0.60 tall, a panel facing forward walked a cell 2.5 times finer than the
bonnet. Per mesh, `cellByLongest` measures every triangle against its own
mesh's longest side, so a mesh has one cell whatever its triangles do. Then
the cell is held in world units between `--cell-min` and `--cell-max`, and
`perCell` is recomputed **only where a bound bites**, so an unbounded run
keeps `1 / resolution` exactly and its cell centres where they were.

`decisions.md: "The density is a mesh's own, not the stage's"` has the
measurements: a sixteen-unit ground plane gave a 1.9-unit car an eighth of the
cells it converts alone with.

**A gaussian cannot be bigger than the triangle it stands on.** The Jacobian's
columns carry `1 / determinant`, and a sliver seen almost edge on by its own
projection has a determinant just above the floor, so the columns run away: on
the bmw27 car 1859 such triangles drew white spikes metres long across the
frame. The two sizes are clamped to the triangle's longest edge, which is the
bound that is always true whatever the projection did.

**Count.** One thread a triangle, over the cells of its bounding box in the
projection grid, testing each cell's centre against the triangle and against
the cut-out map. `--max-cells` caps how many cells one triangle may walk; what
it could not walk is reported, so a ceiling that bites is visible.

**Scan.** A prefix sum over the triangles turns the counts into offsets, in
one workgroup of 256 threads and three phases: each thread sums its slice,
thread zero runs the slice totals into a running sum, and each thread walks
its slice again laying its own running sum down. A serial scan in one thread
was the other option; the pawn has 42 892 triangles and the fox has more.

The scan is also where the budget is applied: slots are a prefix in mesh
order, so a budget too small keeps the first triangles whole and cuts at a
triangle boundary, never in the middle of one. The first triangle it cuts into
is recorded, and the host's next slice starts there.

**Slices.** A mesh whose triangles want more than the budget or the ceiling
allows is converted in slices: the host runs the effect again starting at the
triangle the cut recorded, until everything fits or nothing more can. And when
a run wants more gaussians than the first guess gave it, the host runs that
mesh again with the exact count it asked for — the log says how many meshes
had to be run twice, which is the cheapest indicator that a guess is badly
tuned.

**Emit.** The same walk again, writing the records. A triangle that caught no
cell at all — smaller than a cell, or awkwardly placed — still gets one
gaussian at its centroid: a surface that exists must be drawn, and the count
and the emit must agree about that or the slots a triangle was given do not
match the slots it fills.

They are still two kernels, and the compiler contracts each its own way: a
cell centre exactly on the edge two triangles share can be inside for one and
outside for the other. So the emit writes exactly the slots the count gave it
— from its start to the next triangle's — stops at the last, and writes a slot
it has nothing for as a clear gaussian at the triangle's middle.

**Relief.** Where the material displaces, each cell measures how far the
relief stretches it along each projection axis — the raised surface's tangent
against the flat one's, by central differences half a cell wide — and the count
and the emit both split it into `ceil(0.8 * stretch)` gaussians along each
axis, capped at `--displace-refine`. A cell that wanted more is counted in the
sixth counter. The relief is read at the sub-cell's own point, off the
triangle too: the triangle's plane, normals and texture coordinates go on past
its edge, and clamped back a sub-cell read the height of a point it was not
at.

**Blocks.** With `simplify`, the count and the emit both walk the triangle
through `m2sWalkBlocks`: a tree of blocks aligned to the cell grid, 2^levels
cells a side at the top, taken depth first on a stack of sixteen (so five
levels at most). A block is one gaussian (`m2sBlockIsOne`) when it and a block
past each side lie inside the triangle, with a margin at the corners, and
every cell centre over that reach agrees within the tolerance on colour,
metallic and roughness, cut-out (above the cut everywhere), shading normal and
relief normal, with no cell the relief would split. The block is read before
its surround: it is where a block most often fails. Otherwise its four
children are pushed, and a single cell is what it always was. A triangle the
walk finds nothing in falls through to the one gaussian at its middle.

### 6.5 What a gaussian carries

**Position** is the barycentric blend of the triangle's corners at the cell's
centre.

**The three sizes.** Two are the Jacobian's columns times `--sigma` over the
resolution — the gaussian is as wide as a cell is, measured in the world. The
third is `--flatness` times the smaller of those two, and being a *fraction*
rather than a length is the point: mesh2splat writes `1e-7` there, a number in
the model's own units, so how thin a gaussian is would depend on how big the
model happens to be. The chess pawn is 66 mm across and traced fine; the
Khronos fox is a hundred units long, which makes the same `1e-7` fifteen
hundred times more extreme, and the ray tracer — which integrates density
along the ray rather than projecting an ellipse — saw a ghost where the
rasteriser saw a fox.

**The frame.** The tangent is the triangle's longest edge, the short axis is
the surface normal (or the normal map's, with `--normal-map-turns`), and the
third is square to both. Not the Jacobian's columns: the gaussian is a disc in
the triangle's plane however that plane happened to be parametrised. The frame
becomes a quaternion, and a frame that collapses writes no gaussian.

**Opacity** is `--opacity` times what the cut-out map reads, and the map's
value **is** the opacity rather than a yes or a no. A feather's barb is a
texel of alpha 0.3, and a gaussian of opacity 0.3 blends the way the card did;
cut at a half, every barb was either a whole gaussian or nothing.
`--opacity-cut` is where the map stops meaning *thin* and starts meaning
*absent*.

**Colour** is the material's colour times the albedo map, and then taken
towards the transmission colour by the material's transmission. Their
conversion has no channel for transmission at all, so glass came out as an
opaque white gaussian.

**Metallic and roughness** are the material's values times the map's, blue and
green as glTF packs them. Theirs defaulted to (0.1, 0.5) and ignored the
material, so every conversion came out of the same plastic.

**The normal map**, where there is one, is read in the tangent frame and
becomes the gaussian's shading normal — and, with `--normal-map-turns`, its
short axis as well. The shading normal is written whatever the flag says
(`primvars:athenea:splat:normal`, three floats more a record): a relit cloud is
lit with it and keeps the relief, while the disc stays on the face.

**Transmission** goes into a channel of its own rather than lowering the
opacity. A translucent material is not a transparent one, and lowering the
opacity here would say that it was.

**Emission**, where some material of the stage gives off light: the
material's colour times its weight (`StageMaterial::emission`, read in each
of the four vocabularies), times its map where there is one -- the effect's
`Emission` clip, read as rgb or on one channel (`emissionChannel`) --
written by the effect in a record entry of its own, the last, and by the host
into `record[20..22]` (`primvars:athenea:splat:emission`; the harmonics start
at 23). Relit and transferred clouds add it when drawn; the bake meets it at
its first vertex and keeps it in the colours. Adding a material input a
gaussian carries means the same five places: `StageMaterial` and `materialOf`,
a clip or a parameter of the effect, the kernel's `m2sWrite` (and `m2sLookAt`,
so `--simplify` compares it), the record layout in `convert`/`recordFloats`,
and the encoding field the export and the decode read.

**What the material layers over its base**, where some material of the stage
layers anything (`StageMaterial::layered`): the specular's weight, colour and
index, the coat's weight, roughness and index, the sheen's colour and
roughness, the coat's darkening, constants of the material, sent to the
effect as `writeLobes` and nine parameters -- and the maps on them as clips
`Layer0`..`Layer2`, each with the input it stands for (`layer<k>Target`),
sampled per gaussian by `m2sLayersAt` -- and written in four entries of
their own after everything else; the gather puts them in the record's last
thirteen floats, after the
harmonics (`io::SplatEncoding::lobes`, packing.slang's `SplatLobes` order).
On the device they are `GpuSplats::lobes`, three words a splat
(`packLobes`), and `splat_relight` reads them for both routes
(`splatLobesOf`): the coat and the specular are the same GGX it already had,
the sheen the lobe library's Imageworks, each layered by MaterialX's `layer`
rule (`splatLayers`). The plain lobes (`plainLobes`) are what a cloud without
them reads, and must reflect bit for bit as before (the lobes check).

**The bake's metalness.** The gather also writes a quarter of each gaussian's
metalness into the w of its third ray entry (`1 + m/4` raised, `m/4` flat), so
`w > 0.5` still says raised; `bakeBody` reads it (`bakeMetalness`) and keeps a
Schlick lobe as the metal it is wherever the material is metal at all and was
not written with a conductor, whatever its reflectivity.

**Joints and weights**, with `--skinned`: four of each a gaussian, blended
from the triangle's corners, so the cloud deforms with the skeleton that
carried the mesh; and how those weights change across the gaussian, so it
stretches across a bend as its triangle does (§6.7).

**The Cryptomatte id** is the hash of the source prim's path, inherited by
every gaussian the conversion makes from that prim's triangles. It is what
lets a converted car be named part by part in a matte rather than as one
cloud; `decisions.md: "A pixel says which prims it saw: Cryptomatte"`.

**The relief**, where the material displaces: the gaussian stands at the flat
point plus the normal times the height, its short axis is the relief's normal
(the cross product of the two raised tangents) and its first axis the raised
tangent along u, and its two sizes are the flat cell's times the stretch over
the split — no more than 1.25 cells' worth where the split was capped, since
the pole of a sphere's texture coordinates stretches without bound and sized
by that a gaussian was a spike across the frame. Three entries more a record
carry what the bake and a later move of the relief need: the flat point with
the height, the flat normal, and the relief's normal.

### 6.6 The light bake

Unless `--no-bake`, the conversion is followed by a path trace whose camera is
a list of rays rather than a frame.

**Where a ray starts.** From the gaussian's position, along the shading
normal, offset by `1e-4` of the **model's** diagonal. As a fraction of the
scene's unit instead — a thousandth, floored at one — the chess pawn, 66 mm
tall in a stage whose unit is a metre, began its rays a millimetre off the
surface: thicker than the gold ring under the glass ball and high enough to
start inside the ball, so the ray came down onto the wrong surface and the
ring baked grey, the marble body's colour, where the mesh reads gold
(`decisions.md: "The bake's ray started a millimetre off the model"`).

**Over a footprint.** The `w` of a point's normal, where it is not zero, is
the gaussian's width, and each of its samples comes down onto a point of a
disc half that wide instead of onto the centre. The conversion sets it when
`--simplify` is on, so a block's gaussian carries the block's light.

**What traces them.** `StageRenderer::bakePoints` takes two `float4` a point —
the point with its offset, then its normal — and dispatches the path tracer
over them as though they were pixels. `bakePointsOnDevice` is the same with
the rays already on the device in the kernel's three-`float4` layout and the
answer left there, a point's entries together (`athenea/usd/bake_gather`);
the conversion opens the renderer on its own device and uses that one. It
traces in passes of at most `StageRenderer::kBakeBatch` points (2^19), so
the tracer's planes and sums are sized by the pass and the answer is the only
buffer the size of the cloud; each pass draws its own paths, so a batch
changes an answer's noise, not its mean. It is the same integrator a frame uses,
compiled with its bake constant true: not a second implementation.

**The bake in two halves.** What the conversion calls is
`StageRenderer::bakeSplitOnDevice`: the same tracer with `BakePoints::split`,
which fits nothing and writes sums -- each harmonic against the direct light
(emission at the first vertex, next event estimation from it, and what its
own sample met of a light, MIS on both sides) and against the indirect, then
the brightest sample, the luminance's moments and the steps the paths took
(`athenea/usd/bake_resolve` has the layout). Sums add, so a second pass at the
gaussians that need it is added on (`allotBakePasses`: sqrt(relative
variance / cost) per gaussian, MARS with the gaussian as the cell), and the
fit (`common/bake_fit.slang`, the tracer's own, moved out of it) is made once
over every path a gaussian took, each half on its own. Between the fit and
`combineBake`, which adds the halves and bounds them as the tracer's bake
does, the conversion may hand the halves to the splat bake filter
(`plugins/splatbakefilter`, an AOFX effect: a-trous over a hash grid of the
gaussians, weighed by tangent-plane distance, normal, Cryptomatte id and each
one's noise), packed into pictures and back by `athenea/usd/bake_filter_io`.
`bakePointsOnDevice` stays what it was, the fit in the tracer, for a transfer
and for the tests that ask for it. A TX transfer's answer goes through the
same filter before it is written (`athenea/usd/transfer_filter_io`): the
indirect half's rgb and the reflected field, with a variance handed to the
filter that never stops it, since a transfer keeps no moments.

**A raised gaussian** is baked from the flat point under it, down the flat
normal — a ray from where it stands would start under the surface wherever the
relief sank it — with a third entry, its facing, laid out beside the two.
The hit's shading normal becomes the facing, so its material, normal map
included, is lit as the relief turns it, and the directions projected are the
facing's hemisphere. A bounce from that first vertex under the flat surface is
closed: it would meet the flat mesh, lit, where the relief stands. The
relief's shadow on itself is not baked; nothing in the tracer stands there.

**The directions are stratified, not drawn.** The radiance leaving a glossy
surface swings by orders of magnitude across the hemisphere, so directions
taken at random leave one gaussian in the mirror of the sun and its neighbour
nowhere near it — salt and pepper that more paths barely touch. A grid with a
jitter in each cell covers it evenly, and the same count then answers a
different question.

**What is kept, and what is dropped.** At the first vertex the material's lobe
stack is reduced to its *body*: the diffuse lobes, which carry the texture and
the light that reached it, what the material transmits, and a conductor's
reflection — a metal has no body but that, and dropping it would leave gold
black. What is dropped is the dielectric polish and the sheen, which the
renderer puts back at frame time from the metallic and roughness the gaussian
carries, and puts back *with a direction in it*. A reflection is exactly the
part of a surface that one colour cannot hold: baked in, it is the same from
every direction, and the chess set's marble came back as smooth grey plastic
with its veining gone, the texture buried under three percent of a specular.

A metal is not always a `conductor_bsdf`. MaterialX writes OpenPBR's metal —
and Standard Surface's — as a generalized Schlick, the same reflection under
another node. Dropped as polish, a metal authored that way bakes to nothing:
every one of a Mustang's 27 016 chrome gaussians stored the DC that decodes to
black, against the mesh's 0.19. What tells the two apart is the reflectivity
at normal incidence: a dielectric's is what its index of refraction gives,
0.04 at 1.5 and 0.17 at diamond's 2.42, while a conductor's is half the light
or more. So a Schlick whose F0 stands above a fifth is the metal it stands
for.

**The fit.** The result is projected onto spherical harmonics of the degree
`--bake-degree` asks for, over the half of the sphere the surface faces. The
basis is orthonormal over the whole sphere and over nothing else, so fitted a
coefficient at a time over a hemisphere each one explains the same light again
and their sum overshoots — measured, the pawn came back ten times too bright.
The fit is a small linear solve instead, with a floor under the Cholesky pivot
so that the direction the data could not see is bounded rather than amplified.
`decisions.md: "The bake fits the harmonics, and no longer projects them"` has
the arithmetic and what each attempt measured.

**A gaussian the bake found nothing under** gets its opacity set to zero, not
a colour of zero. The coefficients come back as zeros, and zero is not "no
colour": the constant term is kept shifted to where 3DGS trains it, so a zero
there decodes as black. A cloud out of mesh2splat is nearly all discs, so one
of those seen edge on at a silhouette is a black splinter — which is what the
pawn's gold ring had a fringe of, forty-four of them in 729 073.

**The fit is taken in linear light, and stays there**: a cloud is blended in
linear light, so the bake writes light and the file says so
(`primvars:athenea:splat:linear`). What `bakeEncode` still does is bound the
series: read back over the fitted half of the sphere, clamped between nothing
and the brightest sample the paths returned, mirrored onto the far half and
projected on the whole sphere. While clouds were blended in sRGB the fit was
encoded too — sample by sample once, which is a different quantity (the
curve is concave, so a point beside a wall with half its hemisphere blocked
was fitted at 0.2098 against the 0.309 of light on it), then to the fit as a
whole. Neither encoding exists now.

**A bake that measures the sky instead of the light.** `--transfer` runs the
same rays with a third variant of the kernel, `kTransfer`. The first vertex is
not the material's: the surface is taken as a white Lambert, the direction is
the stratified sample's own, and what a path is worth is `2 cos(theta)` — a
uniform hemisphere of density `1/2pi` against an estimator that wants
`cos/pi`. A path that escapes pays the basis read where it left,
`Y_k(omega)`; a path that does not pays nothing. There is no fit, because the
accumulation is the projection.

Two halves come out of the same rays: a path that escaped from the first
vertex is the direct transfer, which is geometry and has no colour, and one
that escaped after bouncing carries the colour of what it bounced off, which
is the cloud's static global illumination. Nine scalars and twenty-seven
floats a gaussian, f16, written as two primvars. What a frame then does is
`<T, L_SH>` with the nine coefficients of the sky the cloud is actually under
(`technique::Environment`), times the albedo, and a metal has no body at all
and goes entirely to the prefiltered reflection with `f0 = albedo`. That
reflection reads a mip chain of the dome: eight octahedral levels whose base
follows the sky's own resolution -- 2048 a side for a 4k image, 256 for a dome
that is only a colour -- and which roughness walks as its square root, so a
level's texels are about as wide as the lobe it holds.

**The sun.** `technique::Environment` also finds each dome's brightest source
and hands it over as a direction and an irradiance (`env_sun.slang`), leaving
it out of the nine coefficients (`env_project` skips its cone) and adding it
back on the body in `relitByDome`. Both sides sum the same lat-long texels with
the same measure, which is what keeps the swap energy-neutral. Its visibility
is `splatSunOpen`: where the cloud carries `shadowBits` -- sixty-four bits a
gaussian that `kTransfer` traces once, at the first sample, one ray a cell of
an 8 x 8 octahedral grid, written as a plane after the coverage -- the four
cells around the sun weighed bilinearly, leaving out cells below the
gaussian's horizon; otherwise `splatSunShare`, the transfer read along that
direction over the same truncation of an open hemisphere, exactly one where
nothing occludes. The polish keeps the map's own sun and takes the share that
does not get through back out as an analytic GGX lobe, clamped at zero, so a
metal is shadowed too. A cloud with no transfer gets the sun back unshadowed.

**The TX transfer's cells** (docs/decisions.md, task TX). `BakePoints::cellSide`
16 or 32 (`path.transfer` 2 or 3 in the kernel) replaces the 64 rays with one
a cell of a 16 x 16 or 32 x 32 octahedral grid over the whole sphere, the far
half included -- `pathOccluded` starts a ray below the surface from its far
side -- written four words a plane, a plane for every 128 cells, as they are
traced, so nothing the size of the grid sits in registers
(`technique::transferPlanes` says how many planes follow the coefficients).
`m2sTransferInto` writes them as 8 or 32 ints a gaussian, and every reader
tells the layout by that count (`GpuSplats::shadowWords`, the frame's
`shadowBits` parameter). `splat_relight` reads them through `splatCellsOpen`
(four cells bilinear, those behind an axis left out), `splatLobeOpen` (the
lobe's centre and a ring at the angle its roughness spreads) and
`splatOpenToward` (a light's direction).

**The reflected field.** In the same mode a path that escapes after its
first bounce also adds, to sixteen rgb sums, the throughput past its first
vertex (the radiance that arrived along its first direction under a white
sky) times `Y_j` of that first direction and `2 pi`: a projection of the
incoming bounced light by arrival direction. The sums are the bake split's
indirect ones, which a transfer does not use, and leave as sixteen planes
after the cells. A transfer's values a gaussian are one run whose count is
its layout (`athenea/common/transfer_layout.slang`): 9 or 16 direct, three
times that indirect, 48 of field. `splatFieldCoupling` scales the field to a
sky, `splatFieldAlong` reads it narrowed to a lobe, `splatIndirectAlong`
reads the indirect half along a light (the sun's bounce).

**Degree 3.** `technique::Environment` projects sixteen coefficients of each
sky (`kEnvCoefficients`, `kEnvironmentCoefficients`), one thread each as
before, so the first nine are the same sums; the irradiance and the first
transfer read those nine (`kEnvIrradianceCoefficients`). A transfer of
sixteen direct coefficients is dotted with all sixteen, as is its indirect
half of forty-eight.

**Lights that are not the sky.** `relitSplat` gives a light other than a dome,
on a cloud with the cells, the bits' share over the cone it subtends
(`splatConeOpen`, `lightHalfAngle`) as its shadow -- the darker of that and a
measured one -- and adds `splatLightBounce`: the indirect half read along the
light, on the body and the sheen, and the reflected field scaled to it where
the base's and the coat's lobes are closed.

**Glass in the transfer.** At a TX transfer's first vertex the kernel notes
whether the material transmits (a lobe that does not only reflect, read
before `bakeBody` drops the dielectric as polish); if it does, the first
direction is drawn over the whole sphere (`bakeSphereDirection`, density
`1/4pi`, the front's estimator `4 cos`), and a sample drawn behind feeds the
field alone. `relitByDome` reads, for a transmitting gaussian with the field
and nothing traced, the sky where the bits behind are open and the field
where they are closed, along `-wo`. The kernels give a transmitting gaussian
its own index from its layers when the cloud bends at all.

**Saying otherwise about a prim.** The Cryptomatte id a gaussian carries is
also a selection -- everything that came from one prim -- so
`render::SplatOverride` is a row keyed on it: metallic, roughness,
transmission and a tint, in a buffer both shading kernels walk
(`shaders/athenea/common/splat_override.slang`). A negative value leaves what the
gaussian carries, and `replaceColour` makes the tint the colour rather than a
factor on it; `athenea stage --splat-override` writes rows by prim path through
the manifest. `render::measureSplatId` answers the other direction, with a
kernel that counts the gaussians of an id and the bounds of what they hold.
Neither touches the file: the table is the frame's, and clearing it puts the
cloud back.

**A thin wall.** A dielectric under a `surface` whose `thin_walled` is on
(OpenPBR's `geometry_thin_walled`) is a sheet: `kFlagThinWalled`, set by
`atheneaPushLobe` from `gAtheneaThinWalled`, which the compiled surface constructor
sets before its BSDF. It reflects `2R / (1 + R)` and sends the rest straight
through along a delta. A gaussian converted from one (`thinWalled`, bit 24
of `pbr`) is as transparent as the sheet: `scene::thinWallOpacity` sets its
opacity, and the frame shades its reflection alone, scaled by `1/R0`.

**A cloud under a shutter.** Two things move a gaussian over it: a skeleton
(`motion`, from `splat_skin.slang`) and object-to-view (`viewStep`). The
second is `V_close M_close - V_open M_open`: `ParticleField` samples the
prim's transform over the shutter and hands `Engine` the step, and the camera's
ends come from the projection. Both reach the rasteriser's one rank-one term;
the tracer draws a cloud at the frame's instant.

**What the cloud then is.** Baked, its colours are the light on the material's
body and the frame adds the polish: that is `relight` with `litBody`. Not
baked, they are an albedo and the frame lights them whole: `relight` alone.
Either way the cloud is written as relit, because a converted mesh is not a
capture and its colours were never radiance somebody photographed.

### 6.7 Skinning

`--skinned` builds the gaussians in the bind pose and gives each one the four
joints and weights blended from the triangle it stands on. A mesh that nothing
carries still takes its place in the rig: a stage's skinned meshes are rarely
all of them, and the influences have to stay one to one with the gaussians or
the cloud and its rig disagree about who is who. Those gaussians get four
joints of no weight, which the skinner reads as *leave this one where the bind
pose put it*.

The file carries the rig, not the frames: the joints' transforms are one
matrix a joint, sampled over the range `--range` asks for, and everything else
is static. On a bird of 4 269 858 gaussians only that array has time samples,
and it is 609 matrices — which is what makes an animated cloud cost kilobytes
a frame instead of tens of megabytes.

Each gaussian also keeps its weights' gradients: the quotient rule over the
four kept weights, from the triangle's barycentric gradients, along the
gaussian's two rest axes, for the first three joints (the fourth's is minus
their sum) — a ninth record entry, and `jointWeightGradients` in the file.
The skinner carries the frame by the whole Jacobian of the blend with them,
the joints' linear parts and `sum (X_k q) grad w_k`, and takes the posed
in-plane covariance's exact eigenvectors for the two axes rather than
squaring them up, which would drop the shear. A cloud without them is carried
by the joints' linear parts alone.

The skinner turns the shading normal with the frame, by the same Jacobian and
as a normal (`(J a) x (J b)` for two directions `a`, `b` in its surface, which
is the inverse transpose up to scale), so a limb's relief bends with it.

A bake is refused with `--skinned`, because light baked in one pose is wrong in
every other.

**A transfer that turns with the gaussian** (proposal 014 B). A transfer is
not refused: with `--skinned` (or `--transfer-lobes`) it is kept as zonal
lobes in each gaussian's own frame, which a pose turns. The steps, in
`Converter::transfer`:

1. `framesForBake` decodes the records as a frame decodes them
   (`CloudLoader`), so the frame the lobes are written against is the packed
   quaternion a renderer reads. For a skinned cloud it then poses the cloud at
   `--time` with `SplatSkinner` -- the joints' transforms at that instant,
   `MeshStage::skeletonTransforms` -- and moves each bake ray with its
   gaussian (`athenea/usd/transfer_zonal_io`, `zonalPoseRays`: the posed point,
   and the turn from the rest frame to the posed one), because the stage the
   bake traces is posed at `--time` and the cloud was built in the bind pose.
2. `kTransfer` bakes the nine harmonics as for any transfer, the direct half
   alone.
3. `fitZonal` packs them, the bits and each gaussian's frame into a picture
   (`zonalPack`) for the `SplatTransferZonal` bundle
   (`plugins/splattransferzonal`, an AOFX effect), which fits one or two lobes
   by searching the axis that keeps most of the harmonics' energy and
   projecting onto it, refits the two against each other, writes the axes in
   the gaussian's frame, lays the sixty-four bits out over that frame, and
   attaches a histogram of its relative error; `zonalUnpack` writes the answer
   as `transferZonal` (ten floats a gaussian) and the bits.

The frame reads them through `splatTransferFrame` (`splat_relight.slang`),
which both shading kernels call with the gaussian's current rotation and the
instance's rows: each axis goes to the world by the frame and the rows, the
lobes become nine harmonics (`z_l sqrt(4 pi / (2l + 1)) Y_lm(a)`, closed), and
everything after it -- the body's dot with the sky, the sun's share, the
openness -- reads those as it read the stored ones. `splatSunOpen` looks the
sun up in the frame where the bits are the frame's. `transferCount` 10 is what
says a cloud's transfer is zonal (`GpuSplats::isZonal`).

### 6.8 The visibility bake

A skinned cloud cannot carry one baked visibility, because the wing moves and
takes its shadow with it. But one part of a body changes shape very little
between poses, so `athenea visibility` gives each part a field of its own: over a
grid of probes (`--grid` a side), in every direction (an octahedral map
`--octave` a side), how much of a ray leaving that probe the part's gaussians
stop, baked in the pose the cloud was bound in. A ray stops when transmittance
falls under `--cut`.

The parts come from the rig: the joint hierarchy is cut into `--parts`
subtrees, and a subtree with fewer than `--min-joints` joints stays with its
parent's. Each gaussian belongs to the part whose joint carries most of it.

At render time a gaussian asks each part, in that part's own current frame,
and multiplies the answers: a table read a part a light a gaussian, and no ray
at all. What it gives up is that a part is taken as rigid and that parts
occlude independently. `decisions.md: "A cloud shadows itself by part: baked
once, read every frame, no ray"`.

### 6.9 What is written to the file

A `UsdVolParticleField3DGaussianSplat` at `/World/Splats`, with the standard
attributes — positions, orientations, scales, opacities, the harmonic degree
and its coefficients, the extent — and a camera framing it unless
`--no-camera`.

Beside them, the primvars that say what this engine needs and the schemas that
declare them: `AtheneaSplatLightingAPI` (`relight`, `litBody`, `linear` --
every conversion's colours are linear light -- and metallic, roughness,
transmission, the shading normal and the emission a gaussian), `AtheneaSplatSkinningAPI` where the cloud
is skinned, `AtheneaSplatCryptomatteAPI` with an id a gaussian and the manifest
that names them, and `AtheneaSplatVisibilityAPI` once `athenea visibility` has run.
[`operations.md §4.3`](operations.md#43-the-api-schemas) is the attribute
reference.

Every slot is written, whether or not a gaussian survived in it: an empty one
holds an opacity of zero and does not stretch the cloud's extent to wherever
it stands. Keeping the slots means the per-gaussian arrays stay index for
index alike, which is what lets the skinning influences, the ids and the
harmonics all be read by the same index.

**Or a `.athc`.** With `-o x.athc` the records go from the device into a
cloud on the device (`CloudLoader::upload` of a device buffer), into levels
of detail there (`lod::LodBuilder`), and out as the file's bytes
(`lod::writeAthc`). A `.athc` keeps positions, shape, harmonics and the
shading normals; it has no room for the material a relit cloud reflects with,
the Cryptomatte ids, a glass's index, the up axis and unit, a rig or a
transfer -- the last two are refused, the rest said.

### 6.10 How each phase is checked, and what it measured

| Phase | Checked by | What it asserts |
|---|---|---|
| the projection and the cell | `athenea_aofx_tests "[mesh2splat]"` | a unit quad at a known resolution yields a known count, and each gaussian is a known width |
| the bounded cell | the same, `[cell]` cases | a bound that bites changes the count, and one that does not leaves it bit for bit |
| per-mesh density | `ctest -R mesh2splat_density` | the small plane of a two-mesh stage gets tens of gaussians per model and hundreds per mesh |
| GeomSubsets | `athenea_mesh2splat_tests` (after the `mesh2splat_outputs` fixture) | two faces bound red and blue on a green mesh convert to red and blue, as many of each, no green |
| the budget | the same | two equal meshes under a budget half what they want keep about half each |
| the cell from a camera | the same | the card three units from the lens holds over three times the one seven away |
| `.athc` output | the same | the same cards as a `.athc` and as a stage draw alike (p99 at most 1) |
| the bake's rays on the device | `athenea_usd_tests "[mesh2splat]"` | every ray starts `1e-4` of the box's diagonal off; the device bake answers what the host's does; passes answer as one |
| the bake in two halves | `athenea_usd_tests "[split]"` | direct and indirect fitted apart and combined answer what the whole bake does, to 1e-4, and every point beside a lit wall has indirect light |
| the adaptive passes | `athenea_usd_tests "[adaptive]"` | on sums whose variance differs a hundredfold, the noisy half gets 8 to 12 times the quiet half's passes, and the two the budget within 5 % |
| the bake filter | `athenea_aofx_tests "[bakefilter]"` | a noisy step of light on a plane comes back with under a quarter of its error, each side of the step within 0.05 of its own light |
| whole or not at all | `athenea_core_tests "[platform]"` | a failing writer leaves no file and no partial one |
| the cut-out map | `athenea_aofx_tests` | an alpha of 0.3 becomes an opacity of 0.3, not a gaussian or nothing |
| the light bake | `athenea_usd_tests` bake cases | a Lambertian plane comes back at the radiance arithmetic says, and a metal is not black |
| the harmonic fit | the same | the fit reproduces a known directional function within tolerance |
| the ids | `athenea_usd_tests "[crypto]"` | the ids a frame's matte names are exactly the prims the conversion read |
| the visibility bake | `athenea_technique_tests` | a baked factor matches a traced one within tolerance |
| the relief | `athenea_aofx_tests "[displacement]"` | under a tent of height every gaussian stands at the height the map reads, faces the slope and is sized for its share of a cell split in two; a constant height moves the surface and splits nothing |
| the height's reading | `athenea_usd_tests "[displacement]"` | UsdUVTexture's scale and bias on the channel connected, MaterialX's node and its scale, a constant, and a mesh scaled by two |
| the relief's bake | the same | points on a flat plane facing 40 degrees over bake what a plane really turned by 40 bakes, within 3 % |
| the blocks | `athenea_aofx_tests "[simplify]"` | a one-colour quad at 64 cells takes 2144 gaussians against 4160, all on it, of that colour and one, two, four or eight cells wide; under a checker of four-cell squares, not one fewer |

Measured, and recorded in `decisions.md` with the hardware in the caption:

| What | Number |
|---|---|
| the chess pawn, converted | 729 073 gaussians, 25 s at 64 paths (debug) |
| its body against the mesh's | 0.099 / 0.097 / 0.085 against 0.085 / 0.099 / 0.098 |
| baked whole, against body-plus-polish, against relit | 0.137, 0.110, 0.044 — the mesh reads 0.085 |
| 64 paths against 256 | indistinguishable |
| the Mustang, per model against per mesh | 375 199 against 934 338 gaussians |

### 6.11 The defects that shaped it

Each of these is one line here and a section there; the record has the
symptom, the cause and the measurement.

- The bake's programs were compiled blind to the bake variant, and one
  gaussian of 729 073 came back lit.
- A bake has no camera, so it cannot have a headlight: what would be baked in
  is a lamp standing wherever the one-pixel camera happened to be.
- Baked along the normal, the marble came back as polished plastic — the
  reason `bakeBody` exists.
- Directions drawn at random gave salt and pepper that four times the paths
  did not touch; stratified, the same count was clean.
- A ray that started on the surface baked black at grazing angles.
- A ray that started a millimetre off the model baked the pawn's gold ring
  grey.
- The conversion read the rest pose where the stage was posed, so a skinned
  cloud was built from a bird that had flown off its bind pose.
- A slot with nothing in it is kept, so the arrays stay index for index alike.
- A USDZ arrives grey, because nobody declared the binding it expects.
- The count and the emit, two kernels, disagreed on a cell centre exactly on
  a quad's diagonal once the emit changed shape: three slots given, two
  written, and the third a gaussian of zero size at the origin.
- A texture's colour space was dropped by the material network and the albedo
  came back linear where it was sRGB.
- Opacity is coverage, not glass: a cut-out map's value is the gaussian's
  opacity.
- The density is a mesh's own, not the stage's — and it took a floor under a
  car to show it.
- The fit averaged its samples in the encoded space, so every contact region
  of every baked cloud was dark by a third — which only an occluder could
  show, and a transfer's corner did.
- `--glass-opacity` was sent to the conversion under the name of an edit, so
  the number reached nothing and every glass came out opaque.

## 7. Dependencies and toolchain

**Required.** Slang at `SLANG_ROOT` (a fatal error names the release if it is
missing), OpenUSD with MaterialX and its Slang generator at `ATHENEA_USD_ROOT`,
zlib, and a C++20 compiler with Ninja and CMake.

**Fetched at configure time.** slang-rhi (pinned to a commit and patched),
CLI11, nlohmann_json, tinyexr, Catch2 under `BUILD_TESTING`, and GLFW with
Dear ImGui for the viewer.

**Optional, and what is lost without each**: OpenColorIO (its view
transform), Open Image Denoise (denoising), libwebp (`.sog`), OpenVDB
(`.vdb`), zstd (`.spz`), the OptiX headers (CUDA ray-tracing pipelines). All
are found only in their own prefix, and each prints a status line at configure
time saying whether it was found. That line is the diagnostic.

**Dawn, only when asked.** `ATHENEA_WEBGPU=ON` (the `*-webgpu` presets) turns on
slang-rhi's WebGPU backend over Dawn as `scripts/build-dawn.sh` puts it in
`ATHENEA_DAWN_ROOT`: the binaries slang-rhi is written against, at the version
its own CMake names, and slang-rhi's `FetchPackage(dawn)` is pointed there so
the configure step downloads nothing. A missing Dawn is a fatal error naming
the script. `libdawn` is copied into `bin/`, where slang-rhi opens it by name
at run time. `gpu::Backend::WebGPU` is never a platform's preference; it is
opened only when named.

**The six slang-rhi patches** are in `cmake/patches/`, applied at fetch time
by a script that skips any already applied, because FetchContent re-runs the
command on a tree that may already carry them:

| Patch | What it fixes |
|---|---|
| Metal render-target array length | a render target array whose length Metal reported wrongly |
| Metal texture view format | a view created with the wrong format |
| CUDA driver symbol shadowing | a driver symbol shadowed by the runtime's |
| Metal acceleration structures | acceleration structure handling on Metal |
| Metal command buffer errors | a command buffer that failed on the device (out of memory) asserted and aborted the process; now the next `submit` or `waitOnHost` returns `SLANG_E_OUT_OF_MEMORY`, which `gpu::CommandBatch::submit` turns into `OutOfMemory` |
| WebGPU device limits | a WebGPU device asked for every limit its adapter had, and nothing could hold it lower; a `WGPUDeviceExtendedDesc` in the device's chain now lowers four of them, which is how `ATHENEA_WEBGPU_WEB_LIMITS` holds a native run to a browser's |

**Submodules.** `third_party/gpe` tracks branch `lrt-fixes`, `third_party/genlock`
tracks `main`. A change to gpe is committed in the submodule, not here.

## 8. The design record

`docs/decisions.md` is the record: one section per subsystem or per piece of
work, appended at the bottom, never reordered. A section says what was
decided and against what alternative, what checks it, what was measured (with
the hardware and preset), and what is not done. The milestones M0 to M11 are
sections of it too, and later work is folded into the milestone that owns the
subsystem rather than appended as a new one.

What belongs there and not here: a number, a defeat, a rejected alternative, a
measurement. What belongs here and not there: how the thing works now, and how
to change it.

### 8.1 Versions

One version, in the top-level `project(athenea VERSION X.Y.Z)` line; the CLI
takes it from there (`athenea --version`, `athenea info`). A release is a tag
`vX.Y.Z` on `main`, made only after the whole `ctest` passes with the GPU
present -- a test that skips for want of a device is not a pass, so the count
of skipped tests is read before tagging -- with an entry in `CHANGELOG.md`
naming what it brings and the commits. A minor version is a set of merged
work; a patch version fixes what a minor one shipped. The tag and `main` are
pushed to the private repository together.

## 9. Known debts

- **Windows.** The port has not started. `core/Platform.h` is the file it
  starts from, and the `#ifdef`s there name what is missing.
- **The ABI in a comment.** The header comment of the top-level
  `CMakeLists.txt` still says aofx is at ABI 23; the code has been at 25 for
  some time. The rule cannot catch a number written in prose.
- **A module missing from a table.** `image` is a real module in the
  dependency order and is absent from `CLAUDE.md`'s table.
- **The AOV list, three times.** `StageRenderer::displaySource` is the truth;
  the comment above it and the MCP tool's help string repeat it, and both
  already lag behind the Cryptomatte outputs.
- **Cryptomatte's gaps**, written down where they belong: the traced route for
  splats alone writes no matte, points carry no id, and a mesh silhouette has
  no sub-pixel coverage because visibility is one sample.

## 10. Glossary

The same terms as [`operations.md`](operations.md#10-glossary), with the ones
that only matter inside:

| Term | Español | What it means here |
|---|---|---|
| visibility buffer | buffer de visibilidad | `(instance + 1, triangle)` per pixel, what all three mesh routes write |
| lobe stack | pila de lobes | what a material evaluates to at a point: a list of weighted terms |
| check kernel | kernel de comprobación | a GPU oracle in `shaders/athenea/test`, counting what is wrong |
| record | registro | one splat as a file holds it, before decode |
| slot | slot | a projected splat's index within a frame's proj buffer |
| pair | par | a (tile, splat) entry the blend walks |

## Building inside another project

The engine builds as a subdirectory of another CMake project -- the compositor
it was written for builds it inside its own tree -- and the same
`CMakeLists.txt` serves both shapes. What the parent sets, before
`add_subdirectory`:

| variable | meaning | default standalone |
|---|---|---|
| `ATHENEA_EMBEDDED` | the parent provides gpe, genlock, aopenfx and spz; the engine keeps its output directories out of the parent's | `OFF` |
| `ATHENEA_BUILD_APPS` | `athenea` and `athenea-mcp` | `ON` |
| `ATHENEA_BUILD_PLUGINS` | the bundles under `plugins/` | `ON` |
| `ATHENEA_BUILD_VIEW` | `athenea view` (GLFW, Dear ImGui) | `ON` |
| `ATHENEA_SHADER_OUTPUT_DIR` | where the shaders are copied for the run-time compiler | `<build>/shaders` |
| `AOFX_BUNDLE_DIR` | where `aofx_add_bundle` puts a bundle | `<build>/aofx` |
| `AOFX_BUNDLE_ID_PREFIX` | the reverse-DNS prefix in a bundle's `Info.plist` | `rt.sparrow.aofxp.` |

Every target a parent may already have -- `gpe`, `genlock::genlock`,
`aofx::aofx`, `aofx::host`, `slang-rhi`, `nlohmann_json::nlohmann_json`,
`OpenColorIO::OpenColorIO`, `Catch2::Catch2WithMain` -- is looked for by name
before it is made, and `ofxp::spz` stands in for the engine's own copy of spz.
Paths of the engine's own go through `ATHENEA_ROOT` and `ATHENEA_BUILD`, never
`CMAKE_SOURCE_DIR` or `CMAKE_BINARY_DIR`, which are the parent's; and the
engine's own CMake modules are included by path, because a parent may well
have a `cmake/Warnings.cmake` of its own earlier in the module path.
`ATHENEA_USD_ROOT` is a cache variable now, so a build outside the presets can be
told where OpenUSD is. One thing to know about a build directory that has
already configured once: `TBB_DIR` is cached by `find_package`, and a prefix
path that changes afterwards is not consulted -- `cmake -UTBB_DIR` when
`athenea info` counts two.

`athenea::engine` is what an embedder links: the renderer whole, without the
viewer, the MCP server, the host's plugins -- or `athenea::usd`, which an
embedder that reads stages links beside it, with a USD built against the
TBB the rest of its process has. `athenea::io` takes a `TBB::tbb` the parent
found before looking for USD's own, for the same reason: one TBB.

An embedder that already drives the GPU hands its device to a stage:
`usd::StageRenderer::open(path, device)` builds the render delegate and the
engine on that `gpu::Device` instead of opening a second one, and every call
on the renderer then comes from the embedder's one GPU thread. Listing a
stage's prims does not need a renderer at all: `usd::outline(path, prim)` and
`usd::stageCameras(path)` read the stage alone -- no Hydra, no device -- for a
tree view on whatever thread draws it. `setPrimVisible(prim, visible)` hides a
prim with a session opinion and, to show it again, clears that opinion rather
than forcing `inherited`. `scripts/build-usd.sh` with `ATHENEA_USD_TBB=<prefix>`
links a TBB the process already has into the USD prefix instead of building
one (the compositor's OpenImageIO brings Homebrew's); OpenVDB 10.1 is then
built with `tbb/version.h` force-included, which oneTBB 2021.12 and later
need. `drawImage(camera, time, width, height, technique)` is `render` without
the readback -- the streams settled, a path traced frame gathered -- for an
embedder that carries the frame on from `displaySource` itself.
`setShaderTexture(shader, input, file)` points a shader's asset input at
another texture as a session opinion (empty gives the authored one back).
A texture named `aofx://<name>` is filled by the embedder from a device
buffer every frame (`updateExternalTexture`), never read from disk.
`usd::registerPlugins(directory)` does what `PXR_PLUGINPATH_NAME` does, for a
program that cannot set it before USD starts (athenea's schemas, from
`<build>/plugin/usd/atheneaSchemas/resources`).
