# Large clouds: levels of detail, decimation and streaming

This page covers what athenea offers for a cloud too large to draw whole at
every frame: the `.athc` container, with merged levels of detail streamed
chunk by chunk into a fixed budget; levels of detail made by converting a
model again at coarser cells (`athenea mesh2splat --lod-levels`), which is the
route for a cloud a skeleton carries; `athenea decimate`, which keeps fewer
gaussians for good; the GPU memory budget (`ATHENEA_GPU_BUDGET`) that streaming
is held to; and what a frame does when the device runs out of memory.

## Two kinds of levels of detail

| | `.athc` container | `--lod-levels` conversion |
|---|---|---|
| Written by | `athenea convert`, or `athenea mesh2splat -o <name>.athc` | `athenea mesh2splat --lod-levels` |
| What a level is | merged gaussians over a Morton-ordered cloud, each standing for a group of splats | the conversion again at half the resolution each time, a ParticleField of its own |
| What the frame takes | a cut: a merged cell is drawn while it spans no more than the threshold in pixels | the coarsest level whose cell spans no more than the threshold in pixels |
| Streaming | chunks, into `primvars:athenea:stream:budget` | none: each level is a USD file |
| A cloud a skeleton carries | no: a merged cell would not know which joint to move with | yes: each level has its own rig, and only the level drawn is posed |

## A `.athc` from a cloud

A `.athc` holds a cloud sorted by Morton code, cut into chunks, with merged
levels above them: a group of splats too small to tell apart at this distance
is drawn as the one gaussian that stands for it. `athenea convert` writes it
from `.ply`, `.splat`, `.spz`, `.sog` or `meta.json`:

| Option | Value | Default | Notes |
|---|---|---|---|
| `--chunk-splats` | integer | `65536` | splats per streamed chunk: how much travels at once |
| `--max-group-fraction` | number | `0.5` | groups per splat the finest merged level may hold: how aggressive it is |
| `--rotate-x` | number | `0` | refused for a `.athc`: the turn belongs on the prim that references it |

```sh
athenea convert capture.ply capture.athc --chunk-splats 131072
```

`athenea mesh2splat -o <name>.athc` writes a conversion straight into the
container, with the gaussians and their shading normals only (no
metallic/roughness/transmission, Cryptomatte ids, glass index, up axis or
unit); `--skinned`, `--transfer` and `--lod-levels` are refused with it.

The header's `flags` say what else the container keeps: bit 0 the shading
normals (`primvars:athenea:splat:normal`, four bytes a gaussian), bit 1 that
the colours are linear light (`primvars:athenea:splat:linear`), bit 2 the
emission (`primvars:athenea:splat:emission`, one RGB9E5 word, four bytes a
gaussian). Version 2 of the format is written only where the `flags` are not
zero. A `flags` bit this build does not know (any past bit 2) is refused, file
named:

```
<file>: not a readable .athc (unknown flag bits N; this reads bits 0 (normals), 1 (linear) and 2 (emission))
```

## Drawing a `.athc`

`athenea render` takes a `.athc` through `--splats`, with two options for it:

| Option | Value | Default | Notes |
|---|---|---|---|
| `--lod` | number | `0` | the cut: pixels a merged cell may span before the finer level is taken; 0 draws every splat. Smaller is finer and costlier |
| `--stream-budget` | integer | `0` | splats kept on the device; the rest are read as the view asks for them; 0 reads the file whole |

A `.athc` needs `--lod`, and `--lod` needs `--technique raster` and no points.
Without `--lod` the file is refused with
`a .athc is drawn through its levels of detail: give --lod`.

```sh
athenea render --splats capture.athc --lod 1 --stream-budget 4000000 -o shot.exr
```

On a stage the same two are primvars on the prim that names the asset, with
`AtheneaStreamedAssetAPI`:

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:asset` | asset | — |
| `primvars:athenea:lod:threshold` | float | `1` |
| `primvars:athenea:stream:budget` | int64 | `0`, read whole |

```usda
def ParticleField3DGaussianSplat "Cloud" (
    prepend apiSchemas = ["AtheneaStreamedAssetAPI"]
)
{
    asset primvars:athenea:asset = @./capture.athc@ ( interpolation = "constant" )
    float primvars:athenea:lod:threshold = 1 ( interpolation = "constant" )
    int64 primvars:athenea:stream:budget = 4000000 ( interpolation = "constant" )
}
```

A budget too small shows as a blunt cloud: groups whose chunks have not
arrived draw their merged gaussian, and the cloud sharpens as the chunks land.
`athenea stage` waits for the streams to settle before a still, so a rendered
frame is never half-arrived; `athenea:settleStreams` on a render settings prim
is the same wait, and a still sets it.

## Levels of detail by conversion

`athenea mesh2splat --lod-levels N` runs the conversion again at half the
resolution each time. `-o` becomes the stage that draws the levels as one
cloud, and each level is written as a `<name>_lod<n>.usdc` beside it. Each
level and the stage that draws them are written whole or not at all.

```sh
athenea mesh2splat bird.usda --skinned --resolution 1100 --lod-levels 3 -o bird_gs.usdc
```

The levels are tied together by three primvars:

| Attribute | Type | Default |
|---|---|---|
| `primvars:athenea:lod:group` | string | —, drawn as it is |
| `primvars:athenea:lod:cell` | float ‹its own units› | — |
| `primvars:athenea:lod:threshold` | float ‹pixels› | `1` |

Prims of one `lod:group` are one cloud. A view draws the coarsest level whose
`lod:cell` (in its own units) spans no more than `lod:threshold` pixels where
the cloud is nearest — the finest where none does — and only that level is
posed.

## Fewer gaussians for good: `athenea decimate`

`athenea decimate` works on the cloud itself, whatever made it — a conversion,
a capture. It builds the levels of detail and keeps, for good, the coarsest
merge that stands for what is under it, and every splat where none does. A
merge stands for its splats when they lie within `--reach` of it, when they
are discs it is no thicker than their surface allows, and when few of them
(`--outliers`) differ from it in colour by more than `--colour-tolerance`; a
merge's tail may not land on anything of another colour at all.

| Option | Value | Default | Notes |
|---|---|---|---|
| `input` | path, required | — | a stage (its first gaussian ParticleField), or `.ply`, `.splat`, `.spz`, `.sog` |
| `output` | path, required | — | `.usda`, `.usdc`, `.usd` |
| `--prim` | path | the first | the ParticleField to read, where a stage has more than one |
| `--colour-tolerance` | 0..1 | `0.05` | how far a gaussian's colour or opacity may be from the one replacing it, in the cloud's own space |
| `--outliers` | 0..1 | `0.05` | the share of the gaussians a merge stands for that may be different |
| `--flat-tolerance` | 0..1 | `0.08` | where the gaussians are discs, how much thicker than they are a merge may be, against its width |
| `--reach` | 0.1..10 | `1.5` | how many of its standard deviations a gaussian may stand from the merge that replaces it |
| `--degree` | 0..3 | `3` | harmonics kept |

```sh
athenea decimate car_gs.usdc car_fewer.usdc
athenea decimate capture.ply capture_fewer.usdc --colour-tolerance 0.1
```

A stage comes out as it went in, with fewer gaussians: its root layer is
copied — the rig and its animation, the lights, the camera, the Cryptomatte
manifest, the variants — and every array a gaussian long is merged over what
each kept gaussian stands for. A splat file is written as a new stage, in
metres.

Measured on the cobbles converted at `--resolution 768` and baked at 64
paths, drawn by raster at 1200 x 800 against the undecimated cloud:

| | Kept | Image RMS |
|---|---|---|
| defaults | 82.9 % | 0.0020 |
| `--colour-tolerance 0.1` | 37.9 % | 0.011 |
| baked at 256 paths, defaults | 53.9 % | |
| the same conversion unbaked, defaults | 57.8 % | |

A baked cloud carries its bake's noise, and the tolerance has to clear it:
at 256 paths the defaults keep 53.9 % where at 64 they keep 82.9 %. It takes
seconds: 533 007 gaussians in 5 s.

## The GPU memory budget

`ATHENEA_GPU_BUDGET` is the device memory a run may hold, in MiB. Without it
the budget is Metal's recommended working set (`recommendedMaxWorkingSetSize`);
on CUDA and Vulkan there is none unless it sets one. The device prints the one
in use (`GPU memory budget: N MiB`).

```sh
ATHENEA_GPU_BUDGET=8192 athenea stage city.usda -o city.exr
```

Streaming is held to it: a stream is opened with at most half of what the
device's budget has left, at 132 bytes a splat, and an asset asked to be read
whole whose file would take more than that half is streamed instead. Either
is printed when it happens (`streaming budget N splats (~M MiB; asked ...)`).
Splat shadows are sized against it as well: they are skipped for a cloud whose
proxies would take more than half of what the budget has left (about 1.7 KB a
gaussian).

!!! note
    On Apple silicon an allocation must also fit the physical memory the system
    has free less 1.5 GiB kept for the rest of the machine, whatever the budget
    says. The reserve is not configurable.

## When the device runs out of memory

Out of memory is handled before it is reported. A frame that fails with it
gives back what the engine makes again on demand (the rasteriser's grown
buffers, the ray tracers' structures, the denoiser), gives up one thing more,
and is tried again once: splat shadows first, then a level of detail at a
time — a LOD group's next coarser level, a streamed asset's cut at twice the
pixels and half its streaming budget — up to four. Each step is a warning
naming what it gave up. Only a frame that still fails is the command's error,
and that one exits with `3`, where every other error exits with `1`.

| What is printed | What it means |
|---|---|
| `trying again with ...; levels of detail N coarser than asked` | the engine stepped down; the frames go on, and a new run starts as asked |
| `OutOfMemory: ... does not fit in the GPU's memory budget`, exit code 3 | the frame needed more than the budget, even after stepping down |
| `the GPU ran out of memory running ...` | a command buffer failed on the device itself, with the budget not yet reached |

A smaller render, a smaller budget for a streamed asset, or a lower
`ATHENEA_GPU_BUDGET` brings a run back within what the device holds.

## On a stage and in the viewer

`athenea stage` and `athenea view` read the level-of-detail and streaming
primvars from the stage; neither takes a `--lod` of its own. The options that
bear on a large cloud are the image size (`--size`, `1920x1080` for `stage`
and `1600x900` points for `view`), `--splat-shadows` on `stage` (off), the
first thing given up when memory runs short, and `--frames`, which repeats a
frame and reports first, median and fastest on `stage`, or closes `view` after
that many and prints their timings.

```sh
athenea stage city.usda --size 3840x2160 --frames 3 -o city.exr
athenea view city.usda --frames 200
```

In the viewer the **Gaussians** panel gives each cloud's level (*level 1 of 3
in 'bird'*, or *cut* with its own and merged gaussians) and, for a streamed
`.athc`, the chunks on the device, wanted, missing and loading. A **GPU
memory** panel appears when the device ran short, naming what the engine gave
up and how many levels coarser than asked it now draws.

**Reference:**
[§1.2 The runtime environment](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#12-the-runtime-environment) ·
[§1.5 Exit codes](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#15-exit-codes-and-where-an-error-is-printed) ·
[§2.2 athenea render](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#22-athenea-render--splat-files-to-an-exr) ·
[§2.4 athenea convert](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#24-athenea-convert--a-splat-file-into-usd-or-into-a-athc) ·
[§2.4.1 athenea decimate](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#241-athenea-decimate--fewer-gaussians-kept-where-they-matter) ·
[§2.8 athenea mesh2splat](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#28-athenea-mesh2splat--a-model-into-gaussians) ·
[§3.1 A model into a cloud](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#31-a-model-into-a-cloud) ·
[§3.3 A large cloud](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#33-a-large-cloud-levels-of-detail-and-streaming) ·
[§5.2 The panels](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#52-the-panels) ·
[§9 When something fails](https://github.com/jesusluque/athenea/blob/main/docs/operations.md#9-when-something-fails)
