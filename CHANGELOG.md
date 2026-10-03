# Changelog

Every version is a tag `vX.Y.Z` on `main`, made only after the whole `ctest`
passes with the GPU present (a test that skips for want of a device is not a
pass). The version lives in one place, the top-level `project()` line, and is
what `athenea --version` and `athenea info` print. A minor version is a set of
merged work; a patch version fixes what a minor one shipped.

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
