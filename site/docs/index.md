<p class="kicker">athenea</p>

# A relight baker for 3D Gaussian splatting

athenea converts production USD assets into Gaussian splats and bakes, into every Gaussian, the light transport that relights it at draw time. The result is judged against a path trace of the source mesh under the same camera and the same light. The baker runs on a USD and Hydra renderer in which every step that touches scene data is a GPU compute kernel.

These pages describe the desktop version on the public `main` branch: how it is built, and how a USD asset becomes a relit cloud.

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

## Seen moving

Each clip opens the full video.

| | |
|:---:|:---:|
| [![The path trace's light, in real time](assets/clip-relight.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/relight-reel.mp4) | [![The whole Corvette on an iPhone](assets/clip-iphone.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-iphone.mp4) |
| The path trace's light, in real time | The whole Corvette on an iPhone |
| [![The sky turns, the same file follows](assets/clip-corvette-sky.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/corvette-sky-turns.mp4) | [![A skinned sparrow in flight](assets/clip-sparrow.webp)](https://github.com/jesusluque/athenea/releases/download/docs-2026-10-19/sparrow-flight.mp4) |
| The sky turns, the same file follows | A skinned sparrow in flight |

## Where to start

- [Desktop installation](install.md): prerequisites, the CMake presets and the runtime environment on macOS (Apple Silicon, Metal) and Linux (x86_64, CUDA).
- [From a USD asset to a baked cloud](tutorials/usd-to-cloud.md): a pawn from the ASWF Open Chess Set, converted with a transfer bake.
- [Rendering and comparing with the path trace](tutorials/render-compare.md): the same pawn drawn as Gaussians and as meshes, and measured.
- [Preparing a USD stage](tutorials/authoring-usd.md): plugin paths, materials, lights, render settings and the schemas the engine reads.
- [Large clouds and levels of detail](tutorials/large-clouds.md): levels, decimation and streaming under a GPU budget.

The full option tables are in [`docs/operations.md`](https://github.com/jesusluque/athenea/blob/main/docs/operations.md), which these pages follow.
