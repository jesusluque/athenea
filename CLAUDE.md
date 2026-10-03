# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
cmake --preset macos-arm64-debug && cmake --build --preset macos-arm64-debug
cmake --preset linux-x86_64-debug && cmake --build --preset linux-x86_64-debug   # Linux, CUDA
ctest --preset macos-arm64-debug                          # all tests, one at a time (they share the GPU)
ctest --preset linux-x86_64-debug                        # the same on Linux (docs/decisions.md: what CUDA skips)
ctest --test-dir build/macos-arm64-debug -R lod           # tests matching a name
build/macos-arm64-debug/bin/athenea_lod_tests "chunks*"       # one Catch2 case by name (or a [tag])
cmake --build build/macos-arm64-debug --target athenea_render_tests   # one test binary
```

- **Test binaries** are `athenea_<area>_tests` (core, gpu, scene, render,
  colour, geom, material, technique, lod, volume, usd, mcp, view, gpu_host,
  aofx, sched), declared in `tests/CMakeLists.txt` (`athenea_test`), sources in
  `tests/<area>/`. `athenea_host_tests` (tests/usd/test_host.cpp) drives the
  plugin through `UsdImagingGLEngine` and deliberately links no `athenea::usd`,
  so the delegate's classes exist only in the plugin.
  `athenea_storm_oracle_tests` compares Hydra outputs with Storm's; it needs
  `HDX_MSAA_SAMPLE_COUNT=1` in the environment, which ctest sets.
- **Timings** are medians of `athenea stage --frames` and `athenea view --frames`
  (release preset), recorded in `docs/decisions.md` where there is something
  to compare against. `athenea bench` times splat files only; a per-milestone
  bench requirement was retired for that reason (see the M6 section).
- **Generated shaders** (materials, the shading and path tracing kernels)
  are written by `ATHENEA_SHADER_DUMP=<dir>` as files `slangc -I shaders -I <dir>`
  compiles alone, to time a kernel's compile outside the process.
- **Checking a shader compiles** without a build:
  `~/tools/slang/bin/slangc shaders/athenea/<dir>/<file>.slang -I shaders -target metal -entry <entry> -stage compute -o /dev/null`.
  Shaders are copied to `build/<preset>/shaders` by the build and compiled at
  run time, so a shader-only change needs the copy step (any build) but no
  C++ rebuild.

## Rules this codebase holds to

- **No CPU arithmetic on data.**
  - No CPU reference renderer, no CPU fallback, no CPU test oracle.
  - The CPU reads files, parses headers, decompresses, and does bookkeeping
    (counts, slots, queues).
  - Decoding, sorting, merging, culling and image comparison are compute
    kernels.
  - Tests generate inputs and check results with kernels. They read back only
    counters or `compareImages` metrics (p99, max, over2) against a GPU
    reference (`ReferenceRenderer`).
- **Shader parameters are bound by name** through reflection
  (`cursor["name"].setBinding(...)`, `ComputeKernel::dispatch`), never by
  slot.
- **Errors** are `Result<T>` with `ATHENEA_TRY`, no exceptions. OS calls live only
  in `modules/core/Platform`, which is the Windows port's starting point.
- **`docs/decisions.md`** records each subsystem's design, what was measured
  and what is not done. Update it with the change.
- **The manuals.** `docs/operations.md` is updated when a subcommand, an
  option, an environment variable, an `athenea:*` setting, a schema, a format or a
  printed error changes; `docs/development.md` when a module, a rule, a way of
  adding something or a step of the bake changes. The `.es.md` translation
  changes in the same commit, or neither does: it is a translation, not a
  summary, and the two are kept to the same headings and the same tables. A
  manual does not repeat what `--help` prints; it carries the default, the
  unit and the constraint, which `--help` does not say.
- **Submodules.** `third_party/gpe` is on branch `lrt-fixes` and genlock on
  `main`. gpe changes are committed in the submodule.
- **aofx compatibility is mandatory.** The SDK and the host are aopenfx's
  own (github.com/jesusluque/aopenfx, `third_party/aopenfx` on its `sparrow`
  branch): `sdk/` is the headers a plugin is built against, `host/` is the
  one registry and runner every program that loads bundles shares, and this
  engine declares what it brings (`modules/aofx/host/src/Host.cpp`). They
  change only additively and only following the ABI there, so the
  compositor's bundles load unchanged. `aofx_sdk_manifest` guards the pin --
  it fails on any header change -- and `athenea_aofx_tests` must stay green.
- **One Slang, one slang-rhi, one TBB** in the process. `single_tbb` checks
  the TBB count.
- **Toolchain.** OpenUSD with MaterialX/OpenVDB is built by
  `scripts/build-usd.sh` into `~/tools/usd-26.08-mx`, OIDN (GPU devices
  only) by `scripts/build-oidn.sh`, and OpenColorIO by
  `scripts/build-ocio.sh`.
- **Roadmap.** The plan for complete USD (milestones M0–M11) is summarised in
  `docs/decisions.md`, one section per milestone.

## Architecture

Modules under `modules/<name>` are static libraries `athenea::<name>`. They are
listed in dependency order in `modules/CMakeLists.txt`, and each links only
the ones above it.

| Module | What it holds |
|---|---|
| core | `Result`, logging, Platform |
| ui | the viewer's panels as a description (`Controls.h`), walked by `athenea view` and the iOS app |
| sched | `FrameClock`: genlock PTP and ST 2059-1 alignment, timecode |
| image | `Image`: float32 RGBA premultiplied, bottom-left origin, pluggable storage; what gpe and aofx exchange |
| io | CPU file readers (PLY header, SPZ, SOG zip/WebP), EXR with attributes |
| gpu | slang-rhi device, `ShaderLibrary`, `ComputeKernel`, `CommandBatch`, `Buffer`; `gpu/algo` for PrefixSum and RadixSort |
| gpu_host | gpe adopting slang-rhi's device: one `MTLDevice` or CUDA context, buffers shared without copies; the crossing as free functions (`Views.h`) for a program with a context of its own |
| colour | `ColourCompiler` (OpenColorIO as a compiler: a colour space or a display and view into a generated Slang function `athenea_cs_<hash>` and its LUTs), `ColourNames` (what a colour space's name means) |
| scene | `CloudLoader`: raw records uploaded, decoded on the GPU into `GpuSplats` / `GpuPoints` |
| render | `TileRasterizer`, `GaussianRayTracer`, `PointRasterizer`, `ReferenceRenderer`, `Camera`/`Projection`, `SplatEdit` |
| geom | `MeshBuilder`: Hydra meshes triangulated, smooth-normalled and their primvars expanded on the GPU, in `HdMeshUtil`'s order |
| world | `GpuScene` (vertex/index/primvar pools, instance records), `Instancing` (Hydra instancer chains), `RayTracingScene` (BLAS/TLAS), `BvhScene` (two-level compute LBVH) |
| material | `TextureStore` (decode into the working space, mips, UDIM, the texture table), `MaterialCompiler` (MaterialX graphs into Slang), the lobe library |
| light | UsdLux lights on the device: a record per light, and how a shading point samples one |
| technique | how a frame is drawn: `VisibilityRaster` / `VisibilityTrace` / `VisibilityBvh` (same ids), `HeadlightShading`, `AovShading`, `Denoiser` (OIDN on the engine's own queue: Metal, CUDA, and Vulkan through CUDA with imported memory), `DisplayTransform` (view transforms, and an OpenColorIO view through `colour::ColourCompiler`), `SplatVisibility` (per-part visibility fields: baked once, a product of table reads a frame, no ray) |
| lod | `LodBuilder`, `CutSelector`; `Athc.h` for the `.athc` reader/writer and `StreamingPool` |
| usd | `Engine`, `StageRenderer`, `Export`; the `hdAthenea` plugin; codeless schemas in `modules/usd/schemas` |
| aofx | this engine's side of the AOFX host (aopenfx `host/`, ABI 26): capabilities, `renderEffect`, the bundles under plugins/ |
| mcp | the engine as an MCP server: JSON-RPC 2.0 (`Server`), the tools over a warm stage (`Tools.cpp`); `apps/athenea-mcp` is the stdio transport |
| view | `athenea view`: GLFW `Window`, `ImGuiRenderer` (Dear ImGui on the engine's device), `runViewer` |

How the pieces fit:

- **The GPU cloud layout** is shared by every renderer, the loaders and the
  LOD. `GpuSplats` holds `positions` (float4), `shape` (4 uint per splat,
  packed opacity, scale, quaternion and DC) and `sh` (`shWords` uint per
  splat, f16). The packing is in `shaders/athenea/common/packing.slang`.
- **Instances** (`SplatInstance`: cloud, objectToWorld, edit) are what every
  renderer takes. `CutSelector::select` turns `LodInstance`s into per-frame
  `SplatInstance`s whose clouds it owns.
- **Shaders** mirror the modules under `shaders/athenea/`: common, algo, scene
  (decode), splat, rt, points, reference, lod, geom, world, material, light,
  technique, usd. The ray tracer's two routes
  share `rt/rt_integrate.slang`.
- **Hydra.** `Sync` (any thread) only hands CPU records to `Engine` under a
  lock. The render pass thread commits (uploads, opens `.athc`) and renders,
  so the device has one caller.
- **Meshes.** A visibility buffer holds (instance + 1, triangle) per pixel.
  Shading and AOVs rebuild the hit from it (`technique/surface.slang`), so all
  three visibility routes shade alike. Meshes and points are composited by
  view z and handed to the splat rasteriser as its opaque `under` layer.
  Hydra render buffers are bottom row first, as Storm's.
- **Materials.** A MaterialX graph is compiled into a Slang module named by a
  hash of its source, so materials that differ only in values share one.
  `technique::MaterialPrograms` generates the module that imports them all and
  dispatches on a material row: shading imports it to build a lobe stack, and
  the visibility passes import it to ask whether a sample's `opacityThreshold`
  cuts it away. A kernel that evaluates materials walks its pixels in quad
  order (`atheneaQuadPixel`), since bump takes its screen derivatives from the
  thread's quad.
- **Lights.** A light is sampled where it stands -- the cone a sphere or a
  sun subtends, the surface of a disk or a rectangle, the hemisphere above
  the surface for a dome -- and `lightPdf` gives that density for any
  direction, which is what a chi-square checks and what the path tracer's
  MIS weighs by; `lightHit` is the other direction, a ray meeting a light.
  Shading loops over every light at every pixel: `athenea:lightSamples` says how
  many samples each one gets, and one is what an interactive frame takes.
- **Levels of detail.**
  - Splats are sorted by Morton code, and an octree level's cells are runs of
    that order.
  - The finest merged level decides for its splats. Splats live in chunks
    that may be absent from the device, and a group whose chunks are missing
    draws its merged Gaussian.
- **AOFX plugins** built with the engine's SDK live one per directory under
  `plugins/` (picked up by glob, no registration) and land in
  `build/<preset>/aofx`. The SDK and host themselves are the `third_party/aopenfx`
  submodule.
- **Measuring a render** without writing a test: `athenea compare A.exr [B.exr]`
  (`CmdCompare`) prints `imageStats` means/max and `compareHdr`/`compareImages`
  against a reference, all computed by kernels.
- **The CLI** is `apps/athenea` (CLI11): `CmdRender` (render/bench),
  `CmdStage` (convert/stage), `CmdView`, `CmdAofx`, `CmdLive`, `CmdInfo`,
  `CmdMesh2Splat` (a mesh into gaussians, optionally skinned), `CmdVisibility`
  (bake a skinned cloud's per-part visibility into its file).
- **athenea view** keeps frames on the device: `StageRenderer::draw`, then
  `DisplayTransform` writes the window's surface texture and ImGui draws over
  it. Hydra render buffers are converted only when a host maps them.
