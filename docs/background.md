# Background: what athenea owes, and where the specification ends

These sections were the second half of the README. They are kept here whole: the
attributions, the line between the USD specification and this engine, and the assets
the engine is shown with.

## What it owes to Falcor

[Falcor](https://github.com/NVIDIAGameWorks/Falcor), NVIDIA's real-time
rendering research framework, is one of the two renderers this one was read
out of before it was written. It sits under `ref/falcor` as reading material
and is **never built, never linked and never shipped**: no header of its is
included anywhere here, and nothing of it ends up in a binary. It is a
reference the way a paper is.

What it was read for, concretely:

- **Loop subdivision.** `ref/falcor`'s `LoopSubdivide` was read for the Loop
  stencils the GPU subdivider implements -- and only read; the implementation
  here is a set of compute kernels checked against closed forms, not a port
  (`docs/decisions.md`, M8, "the limit projection is `subdivLimit`").
- **How a Slang renderer is laid out**: one shader source compiled for
  whichever device the machine has, a render graph of techniques over a
  visibility buffer that every route fills alike, and a reference renderer to
  test the fast paths against. That shape is Falcor's, and it is why
  `modules/technique` reads the way it does.

Where this engine goes its own way is written down too: it refuses CPU
arithmetic on scene data (no CPU reference renderer, no fallback, no test
oracle -- the ground truth is a GPU kernel), its entities come from OpenUSD
through a Hydra 2.0 render delegate rather than from a scene format of its
own, its materials are MaterialX graphs compiled into Slang, and it carries
Gaussian splats as a primitive beside triangles rather than as a demo.

`ref/spire-engine` is kept for the same reason and on the same terms.

Falcor's last release is 8.0 (August 2024, Slang 2024.1), and it is read as
a renderer's shape, not as a specification. The specification this engine
answers to is OpenUSD's, below.

## Where the specification ends and this engine begins

The reference for what a stage means is the [USD
specification](https://openusd.org/release/spec.html) as released -- 26.08
today -- and it is re-read each release rather than remembered
(`docs/decisions.md`, "The spec is the reference, and it is a living one").
What follows is the exact inventory of what here is standard, what is a
standard mechanism used the way the specification says a renderer may, and
what is this engine's own.

**Standard, read as written.** `UsdVolParticleField3DGaussianSplat` (linear
scales and opacities, the harmonics striped by particle, the hints read and
named); `UsdPreviewSurface` 2.6 with both `opacityMode`s; MaterialX 1.39
graphs, including OpenPBR with its weights held to the specification's
range; UsdLux lights with `ShadowAPI` (`shadow:color`, `distance`,
`falloff`), `DomeLight_1.poleAxis`, light linking, IES; `SkelBindingAPI` on
a cloud as on a mesh, the Skeleton's animation resolved at render time;
a Volume's medium from the Material it binds (MaterialX `volume`,
`anisotropic_vdf`, `absorption_vdf`, `uniform_edf`, per unit of its
`density` field); UsdRender settings, products, vars and passes
(`HydraRenderPassAPI`);
cameras with shutter, f-stop and lens distortion; variants, payloads,
instancing, value clips.

**Standard mechanisms, used as the specification provides.** Renderer
settings in their own namespace on a RenderSettings prim (`athenea:*`, through
`HdsiRenderSettingsFilteringSceneIndex`); `RenderVar`s of `sourceType =
raw` for AOVs the delegate defines (`albedo`, `shadingNormal`,
`primvars:<n>`, `lightGroup:<N>`, `CryptoObject00..02`) and `lpe` for
light groups; the session
layer for what an asset omits (`MaterialBindingAPI` on prims that bind,
default lights on a stage with none, a variant selection); codeless API
schemas, declared in the plugin as any extension is.

**This engine's own, declared as such.** The eight codeless schemas under
`modules/usd/schemas` -- `AtheneaSplatLightingAPI` (relighting, built on
`ParticleFieldRadianceBaseAPI`), `AtheneaSplatSkinningAPI` (the joints' cache
beside a `SkelBindingAPI` binding), `AtheneaSplatVisibilityAPI` (baked
visibility fields), `AtheneaSplatCryptomatteAPI` (which prim each gaussian came from, so a
converted cloud is named part by part in a matte),
`AtheneaSplatEditAPI`, `AtheneaStreamedAssetAPI` (`.athc`),
`AtheneaPointStyleAPI`, `AtheneaVolumeAPI` (the shorthand a Volume gets when it
binds no Material; a Material's `volume` terminal is the standard way);
`athenea:lightGroup` on a light (USD has no light groups);
and the `.athc` level-of-detail format, which is the engine's and not USD's.

## The assets it is shown with

`~/tools/assets`, fetched rather than checked in -- the models do not go in
the repository, only the pictures above do:

- **OpenChessSet** -- the marble pawn with the glass head, which is what the
  conversion's textures, transmission and path-traced bake were measured on.
- **Kitchen_set** -- Pixar's, for the breadth of a real stage.
- **Fox** -- the Khronos glTF sample, fetched and turned into USD by
  `scripts/fetch-fox.sh`. Model **CC0** by PixelMannen; rig and animation
  **CC-BY 4.0** by tomkranis; glTF conversion **CC-BY 4.0** by @AsoboStudio
  and @scurest.
  <https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox>
- **bmw27** -- Blender's own benchmark scene, a BMW 1M on a seamless
  backdrop, **CC-BY** by Mike Pan. Its materials predate the Principled BSDF,
  so the Cycles node trees are reduced to Principled equivalents before the
  USD export. <https://download.blender.org/demo/test/BMW27_2.blend.zip>
- **Eurasian tree sparrow** -- by Nestaeric, from [Fab](https://www.fab.com/listings/63e05d9a-2421-4c79-98f0-bf577b55403f): a rigged and flying bird, 609 joints, 71 clips,
  whose animation arrives as FBX. It is the asset `--skinned` is shown moving
  on, the one the per-part visibility bake was written for, and the one in the
  pictures above. What its import cost, and the four bugs it found, is in
  `docs/decisions.md`.

Blender is a dependency of those assets and of nothing else: no build, no test
and no part of the engine uses it. `guc`, the glTF to USD converter this
project would otherwise reach for, says plainly that animation and skinning
are the two glTF features it does not implement.

## What it owes to mesh2splat

[mesh2splat](https://github.com/electronicarts/mesh2splat), a mesh-to-gaussian
converter published by Electronic Arts, is the algorithm behind `athenea
mesh2splat` and the `Mesh2Splat` AOFX plugin under `plugins/mesh2splat`. Its
licence is BSD 3-Clause with a fourth clause on EA's marks (Copyright (c) 2025
Electronic Arts Inc.). The plugin's kernel is derived from its conversion,
ported to Slang from its OpenGL pipeline -- a vertex, geometry and fragment
shader -- into one compute kernel:

- a triangle is projected onto the plane its normal points along least, with
  the position taken relative to the model's box and over the wider of that
  plane's two ranges (their `orthogonalUvs`) -- or, with `--density
  per-mesh`, over the mesh's own box and its longest side, so one mesh has one
  cell whichever way its triangles face;
- the Jacobian of that map to space, `J = V (O)^-1`, gives the two sizes:
  `|Ju| * sigma / resolution` and `|Jv| * sigma / resolution`, so a gaussian
  is as wide as one cell of the grid the triangle is drawn on;
- the frame is the triangle's longest edge, its normal, and the third axis
  square to both;
- `--resolution` counts cells across the box the density is measured over:
  the whole converted set by default, where a floor in the stage dilutes the
  car standing on it and a badge gets the hood's cell; or each mesh's own
  with `--density per-mesh`, the cell held between `--cell-min` and
  `--cell-max` in world units (by default the model's cell and an eighth of
  it, so nothing is coarser than before and a bolt does not eat the budget);
- a gaussian is written for every cell the triangle covers, sampling albedo,
  normal and metallic-roughness there, within a budget.

What differs here, with the reasons in `docs/decisions.md`: the flat axis is a
**fraction** of the other two rather than their `1e-7`, which is a length and
so makes how thin a gaussian is depend on how big the model is; the cells are
counted and then written at an offset rather than appended with an atomic, so
that the same mesh gives the same array and a gaussian can be followed from
one pose of an animation to the next; a mesh that wants more gaussians than a
run can hold is converted in slices rather than truncated; a map is read by
the set of texture coordinates its material names; and the value of a cut-out
map becomes the gaussian's opacity rather than a yes or a no, which is what
keeps a feather's barb soft.

The copyright notice, the four conditions and the disclaimer are reproduced
verbatim at the head of `plugins/mesh2splat/mesh2splat.slang`, the file
derived from their shaders, and in `THIRD_PARTY_NOTICES.md`, which is
installed with the binaries and inside every plugin bundle. The port is a
translation from GLSL to Slang with changes: their density comes from a
rasteriser and here the cells are walked. What is not theirs is marked in
that file -- transmission, which their conversion has no channel for, and
which a gaussian answers with a tint rather than a lens.

Electronic Arts' name is used here only to say where the algorithm comes
from; it does not endorse this project, and none of EA's or SEED's marks or
logos are distributed with it.

