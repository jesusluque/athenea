<p class="kicker">athenea</p>

# A relight baker for 3D Gaussian splatting

athenea converts production USD assets into Gaussian splats and bakes, into every Gaussian, the light transport that relights it at draw time. The result is judged against a path trace of the source mesh under the same camera and the same light. The baker runs on a USD and Hydra renderer in which every step that touches scene data is a GPU compute kernel.

These pages describe the desktop version on the public `main` branch: how it is built, and how a USD asset becomes a relit cloud.

## Live in the browser

athenea's web side runs at [sparkwebgpu.lucab.co.uk](https://sparkwebgpu.lucab.co.uk) (WebGPU: Chrome, Edge or Safari 26+): the web format streamed by pages and relit as you watch.

| | | |
|:---:|:---:|:---:|
| [![The whole Corvette, streamed](assets/web-corvette-5120.jpg)](https://sparkwebgpu.lucab.co.uk/pieces.html?set=car-compact-tx) | [![The relightable pawn](assets/web-pawn.jpg)](https://sparkwebgpu.lucab.co.uk/pawn.html) | [![The animated sparrow](assets/web-sparrow.jpg)](https://sparkwebgpu.lucab.co.uk/sparrow.html) |
| **[The whole Corvette, streamed](https://sparkwebgpu.lucab.co.uk/pieces.html?set=car-compact-tx)**<br>149 M Gaussians in pages by level of detail; switch the HDRI and paint, chrome and glass follow. | **[The relightable pawn](https://sparkwebgpu.lucab.co.uk/pawn.html)**<br>1.4 M Gaussians with a glass head that acts as a lens. | **[The animated sparrow](https://sparkwebgpu.lucab.co.uk/sparrow.html)**<br>1.5 M skinned Gaussians, 12 animations, relit as it moves. |

## Where athenea comes from

athenea comes from NVIDIA's [Falcor](https://github.com/NVIDIAGameWorks/Falcor).
It follows Falcor's design: one Slang shader source compiled for whichever GPU
the machine has, a frame drawn as techniques over a visibility buffer that every
route fills alike, and a reference renderer the fast paths are tested against.
It was written anew on that shape, not built from Falcor's code, with its scene
coming from OpenUSD through a Hydra 2.0 render delegate.

From a splat renderer it has grown into a relight baker: the light transport
path traced into every Gaussian, levels of detail, a compressed web format, and
the Blender and OpenFX integrations. It takes what a stage means from the
[OpenUSD specification](https://openusd.org/release/spec.html), and the
mesh-to-Gaussian conversion from Electronic Arts'
[mesh2splat](https://github.com/electronicarts/mesh2splat).
[Background](https://github.com/jesusluque/athenea/blob/main/docs/background.md) details what it owes to each.

## A render engine for Blender

<figure markdown>
  ![The pawn in Blender 5.3, drawn by athenea under the Venice sunset HDRI](assets/blender-addon-venice.jpg)
  <figcaption>Blender 5.3 with the athenea add-on: the pawn's Gaussians relit by the World's HDRI (Venice sunset, Poly Haven, CC0) in the viewport. Render Engine: Athenea.</figcaption>
</figure>

The add-on makes athenea Blender's render engine: the viewport and F12 draw the Gaussians relit by the World's sky; it converts meshes into relightable Gaussians (mesh2splat with the path tracer's transfer), adds a shadow catcher for the ground, exports to SPZ, PLY, glTF and USD, and compares with Cycles from the panel. For Blender 5.3 on macOS, Linux and Windows: see [Gaussian splats in Blender](tutorials/blender.md).

<p align="center"><img src="assets/pawn-venice.jpg" width="440" alt="F12 of the pawn under the Venice sunset, drawn by athenea in Blender"></p>

## The flow

| Step | Command | What it produces |
|---|---|---|
| 1. Conversion and bake | `athenea mesh2splat` | a `UsdVolParticleField3DGaussianSplat`: one Gaussian per cell, with its material and, with `--transfer`, how much of a sky reaches it from each direction |
| 2. Relight and render | `athenea stage`, `athenea view` | the cloud drawn under the stage's lights, by raster or by path trace, through the same Hydra delegate a DCC loads |
| 3. Reference | `athenea stage --technique rt` | the source meshes path traced from the same camera |
| 4. Judgement | `athenea compare` | the relative HDR error and the 8-bit code-value distribution between the two images, measured on the GPU |

<figure markdown>
  ![The Open Chess Set pawn as relit Gaussians and path traced as meshes](assets/pawn-gs-pt.png)
  <figcaption>The pawn of the ASWF Open Chess Set (CC BY 4.0): left, its Gaussian cloud relit at draw time; right, the meshes path traced. Same camera and light. See the <a href="tutorials/usd-to-cloud/">tutorials</a> and the <a href="credits/">credits</a>.</figcaption>
</figure>

## Example and test material: the pawn

[athenea-example-pawn.zip](https://github.com/jesusluque/athenea/releases/download/v0.3.0/athenea-example-pawn.zip) (411 MB) holds the pawn of the ASWF Open Chess Set
as source meshes and as the relightable cloud athenea baked from them, with a
camera and two skies, ready to render, compare, and open in Blender, Nuke or
Natron:

```sh
athenea stage pawn_gs.usda   --camera /World/Cam -o gs.exr                                 # the cloud
athenea stage pawn_mesh.usda --camera /World/Cam --technique rt --path-total 64 -o pt.exr  # the meshes, path traced
athenea compare gs.exr pt.exr                                                              # relMSE 0.014, PSNR 30.2 dB
athenea stage pawn_gs_venice.usda --camera /World/Cam -o venice.exr                        # relit under another sky
```

## Seen moving

Each clip opens the full video.

| | |
|:---:|:---:|
| [![The path trace's light, in real time](assets/clip-relight.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/relight-reel.mp4) | [![The whole Corvette on an iPhone](assets/clip-iphone.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-iphone.mp4) |
| The path trace's light, in real time | The whole Corvette on an iPhone |
| [![The sky turns, the same file follows](assets/clip-corvette-sky.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-sky-turns.mp4) | [![A skinned sparrow in flight](assets/clip-sparrow.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/sparrow-flight.mp4) |
| The sky turns, the same file follows | A skinned sparrow in flight (by Nestaeric, [Fab](https://www.fab.com/listings/63e05d9a-2421-4c79-98f0-bf577b55403f)) |

## Where to start

- [Desktop installation](install.md): prerequisites, the CMake presets and the runtime environment on macOS (Apple Silicon, Metal) and Linux (x86_64, CUDA).
- [From a USD asset to a baked cloud](tutorials/usd-to-cloud.md): a pawn from the ASWF Open Chess Set, converted with a transfer bake.
- [Rendering and comparing with the path trace](tutorials/render-compare.md): the same pawn drawn as Gaussians and as meshes, and measured.
- [Preparing a USD stage](tutorials/authoring-usd.md): plugin paths, materials, lights, render settings and the schemas the engine reads.
- [Large clouds and levels of detail](tutorials/large-clouds.md): levels, decimation and streaming under a GPU budget.

The full option tables are in [`docs/operations.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md), which these pages follow.
