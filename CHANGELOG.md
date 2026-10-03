# Changelog

Every version is a tag `vX.Y.Z` on `main`, made only after the whole `ctest`
passes with the GPU present (a test that skips for want of a device is not a
pass). The version lives in one place, the top-level `project()` line, and is
what `athenea --version` and `athenea info` print. A minor version is a set of
merged work; a patch version fixes what a minor one shipped.

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
