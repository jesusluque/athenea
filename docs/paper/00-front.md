<!-- Copyright (c) 2026 jesus luque. -->

::: {#front}

# athenea: A Relight Baker for Gaussian Splatting from Production USD Assets {.unnumbered .unlisted .paper-title}

<p class="byline">jesus luque</p>

<p class="date">19 October 2026</p>

## Abstract {.unnumbered .unlisted}

We describe athenea, a relight baker for 3D Gaussian splatting: it converts production USD meshes and their materials
into Gaussians and bakes into each Gaussian the light transport needed to relight it at draw time, with the reference
always a path trace of the source mesh. A mesh is converted on the device into anisotropic Gaussians, one per cell of
a camera-derived lattice, with coverage-correct opacity, edges held where a surface stops or folds, layered OpenPBR
materials, glass classified as solid or thin wall, and roughness widened by the spread of normals under each Gaussian.
The renderer's own GPU path tracer then bakes a transfer into every Gaussian: degree-3 spherical harmonics of direct
and indirect light, a reflected field, 256 bits of open directions and the material's marks. At draw time these relight
the cloud under any sky and any light, with sharp reflections turned per pixel by a stored shape operator. On a
Chevrolet Corvette C7 of 192 meshes, doubling the bake resolution from 640 to 1280 px raised the relit car's PSNR
against the path trace from 16.55 to 17.88 dB on its crop and from 22.18 to 23.90 dB on the frame, and a 5120 px
close-up level of 74.5 million leaves in 24 trees was baked for close views. The baker rests on a USD and Hydra
renderer in which every step that touches scene data runs as a GPU compute kernel. Rigged assets keep their skeleton,
with a Jacobian that includes the weights' gradients, zonal transfer lobes that turn with a limb and per-part
visibility fields. Per-part merge trees with a monotone, bounded screen error, back-face and occlusion culls, and pages
ordered by error give levels of detail at a cost of about 0.1 dB. A transfer is read in its cloud's own frame, with
the sky turned into it once per instance, so a cloud a scene rotates relights as it does unrotated (relMSE
$9\times10^{-6}$). Every parameter a Gaussian carries, from its albedo and normal to the transfer's sky visibility,
bounce and reflected field, can be drawn through a camera as an image channel of its own. The bake's output is a
master: a USD file holding every parameter at full precision, judged against the path trace, which the engine, an
OpenFX node in a compositor (with depth, world position and Cryptomatte layers and a sky the artist chooses) and a
Blender add-on read as it is. A separate process with its own goal derives from the master a web format for a phone:
clustered PCA of the transfer, vector-quantised visibility and a 40-byte compact record carry the relit car to a WebGPU
player on an iPhone, its loss measured against the master.

**Keywords:** relight baking; 3D Gaussian splatting; precomputed radiance transfer; spherical harmonics; USD; Hydra;
mesh to Gaussians; path tracing; level of detail; clustered PCA; GPU-only pipelines; WebGPU; OpenFX;
compositing; image channels.

## Contents {.unnumbered .unlisted}

::: {#contents}

| 1. Introduction
| 2. The athenea engine and renderer
| 3. From USD meshes to Gaussians (mesh2splat)
| 4. Light transport baked into Gaussians (transfer) and relighting
| 5. Animated assets: skeletons, skinning and per-part visibility (the pawn and the sparrow)
| 6. Levels of detail: merge trees, cuts by error, pages and occlusion
| 7. Application: compression and a WebGPU player on mobile
| 8. Evaluation
| 9. Limitations and future work
| 10. References
| Appendix A. Repositories and commands

:::

:::
