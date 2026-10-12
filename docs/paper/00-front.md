<!-- Copyright (c) 2026 jesus luque. -->

::: {#front}

# athenea: Photoreal USD Assets as Relightable Gaussians, from the Mesh to the Phone {.unnumbered .unlisted .paper-title}

<p class="byline">jesus luque</p>

<p class="date">12 October 2026</p>

## Abstract {.unnumbered .unlisted}

We describe athenea, a USD and Hydra renderer in which every step that touches scene data runs as a GPU compute
kernel, and the pipeline built on it that turns production USD meshes into 3D Gaussian splats that keep their light.
A mesh is converted on the device into anisotropic Gaussians, one per cell of a camera-derived lattice, with
coverage-correct opacity, edges held where a surface stops or folds, layered OpenPBR materials, glass classified as
solid or thin wall, and roughness widened by the spread of normals under each Gaussian. The scene's own path tracer
then bakes a transfer into every Gaussian: degree-3 spherical harmonics of direct and indirect light, a reflected
field, and 256 bits of open directions. At draw time these relight the cloud under any sky and any light, with sharp
reflections turned per pixel by a stored shape operator. Rigged assets keep their skeleton, with a Jacobian that
includes the weights' gradients, zonal transfer lobes that turn with a limb and per-part visibility fields. Per-part
merge trees with a monotone, bounded screen error, back-face and occlusion culls, and pages ordered by error give
levels of detail. Clustered PCA, vector-quantised visibility and a 40-byte compact record carry the whole Chevrolet
Corvette C7 to a WebGPU player on an iPhone. Throughout, the reference is a path trace of the source mesh. Doubling the
bake resolution from 640 to 1280 px raised the car's PSNR from 16.55 to 17.88 dB on its crop and from 22.18 to 23.90 dB
on the frame. The level of detail costs about 0.1 dB. On the phone, the car data stayed at 31–44 MB while moving.
A 5120 px close-up level of 74.5 million leaves in 24 trees was built for close views.

**Keywords:** 3D Gaussian splatting; USD; Hydra; precomputed radiance transfer; spherical harmonics; mesh to
Gaussians; level of detail; clustered PCA; WebGPU; mobile rendering; path tracing; GPU-only pipelines.

## Contents {.unnumbered .unlisted}

::: {#contents}

| 1. Introduction
| 2. The athenea engine and renderer
| 3. From USD meshes to Gaussians (mesh2splat)
| 4. Light transport baked into Gaussians (transfer) and relighting
| 5. Animated assets: skeletons, skinning and per-part visibility (the pawn and the sparrow)
| 6. Levels of detail: merge trees, cuts by error, pages and occlusion
| 7. Compression and the WebGPU player on the iPhone
| 8. Evaluation
| 9. Limitations and future work
| 10. References
| Appendix A. Repositories, commands and figures

:::

:::
