# Changelog

Every version is a tag `vX.Y.Z` on `main`, made only after the whole `ctest`
passes with the GPU present (a test that skips for want of a device is not a
pass). The version lives in one place, the top-level `project()` line, and is
what `athenea --version` and `athenea info` print. A minor version is a set of
merged work; a patch version fixes what a minor one shipped.

## 0.3.0 — 2026-10-13

The relight baker over Gaussian splats, and the engine on three systems:
macOS (Metal), Linux (CUDA) and Windows (CUDA, natively with MSVC). 433 tests.

- **Pieces into main**: the transfer baked per gaussian (TX: a field in each
  gaussian's frame, zonal lobes for skinned clouds), glass that covers fully
  under a dome and opens against Fresnel, thin lenses, and the bake's validate
  stage (`mesh2splat --validate`).
- **Gaussian reduction** (`simplify`): adaptive emission, pruning by what the
  frames show, merges kept as a tree with coverage, pages by error and a
  shared occlusion codebook (`decimate --tree-*`, `--cut-*`,
  `--occlusion-*`).
- **A merge's second moment about its own centre** (`b1d6959`): float32 moments
  about the origin lost the cross variance of thin gaussians, and a tree built
  again from its leaves drops the old tree's bookkeeping (`3a0a6f8`).
- **Windows, natively**: MSVC builds the engine, the tests and the plugins
  (`windows-x64-*` presets, `scripts/*.ps1` for the dependencies); `Platform`
  maps files, finds its module and replaces files atomically; aopenfx loads
  bundles with `LoadLibrary`; CUDA is preferred, with OptiX.
- **CUDA**: a buffer made without data starts at zero; a texture is read
  through the sampler its view was made with (the sky's poles, the
  prefilter's levels); a dome's residual is stored as its format; a bake runs
  on the OptiX pipeline and is refused where there is no ray tracing (WSL2).
- **A release's layout**: AOFX bundles are found in `aofx/` beside the
  executable's folder; `athenea info` counts TBB on Windows.
- **The same frame on all three** (`75f10ec`): on CUDA a material's lobe
  stack was overwritten while it was built, so MaterialX materials shaded
  wrong and bakes came out black there. A frame now reads the same under
  CUDA, Vulkan and Metal (an OpenPBR ground: 0.274467 0.245033 0.215593 on
  each). Raster glass still differs on CUDA, which has no inline ray queries
  to refract with; the path traced frame does not.
- **Bakes on CUDA through OptiX**: the engine built the ray tracing structure
  only for a device with inline ray queries, so the path tracer's OptiX
  pipeline traced against nothing and a bake found no surface. It is built
  for the pipeline too, and a transfer bake (TX) is right on CUDA. A radiance
  bake is not yet: OptiX does not compile its pipeline (an NVIDIA L4, driver
  596.36), and the bake says so.

Tests: macOS 422 of 433, Linux 431 of 433, Windows 407 of 430 before the bake
fix; on Windows with OptiX (an L4) the radiance bakes fail as above. Failing on every system and before this merge, so not caused by it: the
zonal transfer's fit and turn (two tests), and on macOS a refit's single
pixel, many lights as one, and a texture floor's p99. The clock tests fail
where the machine is loaded.

## 0.2.0 — 2026-10-03

The day's tasks merged. 354 tests, all passing on Metal (Apple M5 Pro); the
same two skip on purpose as in 0.1.0.

- **UsdLux units** (`6686626`): a `DistantLight` is radiance, and `normalize`
  divides by π·sin²θ.
- **Every gaussian blends in linear light** (`1155dc6`); a cloud says whether
  it was captured in sRGB, and sRGB is applied only at the display.
- **Emission per gaussian** (`48cc5e3`), stored as RGB9E5 in `.athc` v2.
- **OpenColorIO on the GPU** (`2fadf32`): the `colour` module compiles colour
  spaces into Slang kernels; textures decode their input space on the device.
- **The sparrow as a mesh** (`e947962`): raster shading, shadows and material
  passes that match the path tracer; feathers cut by stochastic coverage.
- **Partial opacity is the real coverage** (`125043e`).
- **mesh2splat on the GPU**: materials per `GeomSubset`, the budget shared
  between meshes, bake bounds and rays on the device, `.athc` written
  directly, atomic writes.
- **A skinned cloud refits its ray structure** instead of rebuilding it each
  frame.
- **Review fixes**: `exitThrough` precedence, cofactor normals,
  `metersPerUnit` in convert and decimate, `.athc` v1 without blocks.
- **Gaussian stats in `athenea view`**: what is drawn this frame, in a panel.
- **Blender**: the generic plugin path, and Blender's native splats arrive as
  splats.
- **The exact skinning Jacobian** (`eb3822b`): a skinned gaussian deforms by
  the whole derivative of its blend, with weight gradients from mesh2splat.
- **The Measure effect** (`83b7ff0`): an AOFX bundle that measures and compares
  images on the GPU; `athenea compare` runs on it with identical output.
- **Out of GPU memory is an error, not an abort** (`0397b8c`): a slang-rhi
  patch reports failed command buffers; allocations are admitted against a
  budget (`ATHENEA_GPU_BUDGET`, else Metal's working set, and never past the
  machine's free memory minus 1.5 GiB); the engine drops splat shadows and
  LOD levels and retries; the viewer stays open; the CLI exits with code 3.
- **`.athc` readers refuse flag bits they do not know** (`2b17427`).

## 0.1.0 — 2026-10-03

The engine that was lucabRTrender, under its own name, with the day's first
fixes. 293 tests, all passing on Metal (Apple M5 Pro); two skip on purpose:
openFXplayer's installed bundles are AOFX ABI 25 and this host speaks 26, and
one ray generation case that needs a pipeline this device route does not use.

- **Athenea** (`e8ef1eb`): lucabRTrender's `engine`, `dev`, `ios` and `sparrow`
  brought together and renamed: `lrt` → `athenea`, `hdLrt` → `hdAthenea`,
  `Lrt*` schemas → `Athenea*`, `lrt:*` → `athenea:*`, `LRT_*` → `ATHENEA_*`,
  `.lrtc` → `.athc`. `athenea compare` measures images on the GPU.
- **A bake gives no more light than any path saw** (`3ed0aa2`): the bands of a
  dark, glossy surface no longer blow up (the pawn's glass head at degree 2
  went from 1094 to 1.67, at degree 3 from infinity to 1.55). The written cloud
  carries `metersPerUnit`; `athenea` registers its own USD plugins.
- **A converted glass bends** (`84a450c`): mesh2splat writes the cloud's index;
  the lighting schema is applied.
- **A material is read in its own words** (`ebb684a`): gltf_pbr, OpenPBR,
  standard_surface and UsdPreviewSurface at their own defaults; a packed map's
  missing channel; a preview opacity as a thin wall.
- **A normal map in the frame its UVs give** (`b5bb5bd`).
- **A shading normal per gaussian** (`2bd3390`), skinned, through LOD,
  decimate and `.athc` v2.
- **aofx** (`c954b10`): an effect runner is given this engine's capabilities
  (three crashes in the Invert bundle).
- **Glass** (`b1b33fa`): a ray inside a glass meets its far face (the mesh's
  path tracer culled it); a glass cloud's far face keeps its Fresnel and tints
  again; the room through rough glass is sharper than its reflection. The
  pawn, transferred, against the mesh: relMSE 0.0498 → 0.0057.
- **`athenea migrate`** (`6ecae0d`): lucabRTrender's files read again.
- **For Blender** (`dc9cf6e`): shaders found beside the loaded plugin
  (`moduleDir`), and the material compiler's own MaterialX libraries.
- **Versions** (this release): `athenea --version`, `athenea info` and this
  file.
